"""Energy + latency probe: LFM vs SmolVLM2 at each Part 2 dial setting.

Owner-requested addition (change log #10). Measures, for the SAME fixed 100
counting questions (CountBenchQA, fixed_sample n=100 seed=7), at every dial
setting from Part 2 (LFM image-token budgets 32/64/128/256; SmolVLM2 longest
edge 512/1024/1536/2048):

  - accuracy (scored later by the standard scoring path),
  - per-item latency (recorded by the shared cell core, as everywhere),
  - MEASURED energy per answer, from the GPU's cumulative energy counter
    (`nvidia-smi -q -d POWER` "Total Energy Consumption", NVML-backed on
    A100) read at inference-window boundaries — not a spec-sheet derivation.

Task is COUNTING (owner directive): grounding is excluded because it is not a
task both models perform (SmolVLM2 main-table grounding = 0%, "no boxes").

Probe discipline mirrors the sweep exactly: ONE container, ONE physical
A100-40GB (strict assert; UUID stamped in every row), both pinned
environments as venvs (SWEEPBOX image), cells strictly serial, fresh_start.
Model-load energy is EXCLUDED from J/item honestly: the cell core flushes one
row per answer, so the driver watches the output file and reads the energy
counter when row 1 appears and when row N appears — the reported window
covers rows 2..N exactly. A 15 s idle-baseline read before each cell gives a
net-of-idle variant. The 10 Hz power trace is kept as provenance.

Queue position: runs ONLY after the referee GPU box finishes (one GPU at a
time, owner standing rule).

Usage (from repo root):
  .venv/bin/modal run harness/run_power_probe.py               # full probe
  .venv/bin/modal run harness/run_power_probe.py --sample-n 3  # path smoke
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import GPU_TYPE, HF_CACHE_PATH, SWEEPBOX_IMAGE, hf_cache_volume
from harness.cell_core import compute_out_path

app = modal.App("edge-vlm-powerprobe")

VOLUMES = {HF_CACHE_PATH: hf_cache_volume}
RESULTS_BASE = str(Path(HF_CACHE_PATH) / "results")
VENV_FOR_MODEL = {"lfm450m": "tf5", "smolvlm2": "tf457"}
DIALS = (("lfm450m", (32, 64, 128, 256)), ("smolvlm2", (512, 1024, 1536, 2048)))


@app.function(image=SWEEPBOX_IMAGE, gpu=GPU_TYPE, timeout=4 * 3600, volumes=VOLUMES,
              cpu=8.0, memory=32768)
def power_probe(cell_specs: list):
    import os
    import subprocess
    import time

    # NVML direct read: the counter is live on Modal's driver (verified:
    # 2 s idle delta = 60.04 W vs 59.69 W instantaneous) but the nvidia-smi
    # CLI omits the field. Sub-ms reads also shrink window-boundary fuzz.
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def energy_j():
            """Cumulative GPU energy (J) from the NVML mJ counter."""
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(_nvml_handle) / 1000.0

        energy_j()
    except Exception:
        def energy_j():
            return None

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    if energy_j() is None:
        raise RuntimeError("ENERGY_COUNTER_ABSENT: this driver does not expose the "
                           "cumulative energy counter; refusing to substitute an estimate.")

    # 10 Hz power trace for the whole probe — provenance, not the measurement
    trace_path = f"{RESULTS_BASE}/probe/power_trace.csv"
    os.makedirs(f"{RESULTS_BASE}/probe", exist_ok=True)
    trace = open(trace_path, "w")
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits", "-lms", "100"], stdout=trace)

    # Preflight (change #12): fix the shared question set to items EVERY cell
    # completes — the pinned LFM processor crashes on forced-budget upscales
    # (sweep finding: 13.9% at b256 on grounding) and SmolVLM2 rejects
    # grayscale inputs; the efficiency curve must compare identical work.
    pool_n, seed, n_target = cell_specs[0]["_pool_n"], cell_specs[0]["_seed"], cell_specs[0]["_n_target"]
    preflight = {}
    for model_key, venv in (("lfm450m", "tf5"), ("smolvlm2", "tf457")):
        budgets = [c["image_token_budget"] for c, _ in cell_specs[1] if c["model"] == model_key]
        pf = subprocess.run(
            [f"/opt/{venv}/bin/python", "/root/harness/probe_preflight.py"],
            input=json.dumps({"model": model_key, "budgets": budgets,
                              "pool_n": pool_n, "seed": seed}),
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "/root"},
        )
        if pf.returncode != 0:
            raise RuntimeError(f"preflight {model_key} failed:\n{pf.stderr[-3000:]}")
        marker = [l for l in pf.stdout.splitlines() if l.startswith("PREFLIGHT_JSON:")][-1]
        preflight[model_key] = json.loads(marker[len("PREFLIGHT_JSON:"):])

    pool_order = preflight["lfm450m"]["pool"]
    eligible = [iid for iid in pool_order
                if all(preflight[m]["pass"][iid] for m in preflight)]
    if len(eligible) < n_target:
        raise RuntimeError(f"preflight found only {len(eligible)}/{n_target} items that "
                           f"complete at every setting (pool {pool_n}) — enlarge the pool")
    item_ids = eligible[:n_target]
    derivation = (f"first {n_target} of fixed_sample(n={pool_n}, seed={seed}) completing "
                  "a 1-token generate at all dial settings for BOTH probe models "
                  "(preflight, change #12)")

    results = {"gpu": gpu, "cells": [], "preflight": {
        "pool_n": pool_n, "seed": seed, "eligible": len(eligible), "item_ids": item_ids,
        "per_model": {m: {"fail_count": sum(1 for v in preflight[m]["pass"].values() if not all(v)),
                          "fail_examples": preflight[m]["fail_examples"]} for m in preflight},
    }}
    try:
        for config, prompt in cell_specs[1]:
            config = dict(config)
            config["sample"] = dict(config["sample"], item_ids=item_ids, derivation=derivation)
            out_path = Path(compute_out_path(config, RESULTS_BASE))
            if out_path.exists():
                out_path.unlink()  # fresh even in sample mode: energy windows never mix runs
            n_target = config["sample"]["n"]

            # idle baseline (GPU quiesced between cells)
            e0, t0 = energy_j(), time.time()
            time.sleep(15)
            e1, t1 = energy_j(), time.time()
            idle_w = (e1 - e0) / (t1 - t0)

            venv = VENV_FOR_MODEL[config["model"]]
            payload = json.dumps({
                "config": dict(config, fresh_start=True, _cpu_reserved=8.0,
                               _memory_reserved_mb=32768),
                "prompt": prompt,
                "results_base": RESULTS_BASE,
            })
            Path("/tmp/cell_payload.json").write_text(payload)
            e_cell_start, t_cell_start = energy_j(), time.time()
            with open("/tmp/cell_payload.json") as stdin_fh:
                proc = subprocess.Popen(
                    [f"/opt/{venv}/bin/python", "/root/harness/sweep_cell_main.py"],
                    stdin=stdin_fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, env={**os.environ, "PYTHONPATH": "/root"},
                )

            # watch the row-by-row flushed output file for window boundaries
            def lines():
                try:
                    return sum(1 for _ in open(out_path))
                except FileNotFoundError:
                    return 0

            e_first = t_first = e_last = t_last = None
            while proc.poll() is None:
                n = lines()
                if n >= 1 and e_first is None:
                    e_first, t_first = energy_j(), time.time()
                if n >= n_target and e_last is None:
                    e_last, t_last = energy_j(), time.time()
                    break
                time.sleep(0.05)
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"probe cell {config['model']} b{config['image_token_budget']} failed "
                    f"(probe restarts from zero, never mixes devices):\n{stderr[-3000:]}"
                )
            if e_last is None:  # subprocess exited before watcher saw row N
                e_last, t_last = energy_j(), time.time()
            e_cell_end, t_cell_end = energy_j(), time.time()

            marker = [l for l in stdout.splitlines() if l.startswith("SUMMARY_JSON:")][-1]
            summary = json.loads(marker[len("SUMMARY_JSON:"):])
            summary["rows_jsonl"] = out_path.read_text()

            # J/item divides by COMPLETED answers inside the window (rows 2..N);
            # error rows consume ~no energy and must not dilute the number.
            rows = [json.loads(l) for l in summary["rows_jsonl"].splitlines()]
            n_window = max(1, sum(1 for r in rows[1:] if not r.get("error")))
            e_window = e_last - e_first
            wall_window = t_last - t_first
            summary["energy"] = {
                "counter": "NVML nvmlDeviceGetTotalEnergyConsumption (nvidia-ml-py 13.610.43, cumulative mJ)",
                "idle_w": round(idle_w, 2),
                "window_rows": [2, n_target],
                "window_j": round(e_window, 2),
                "window_s": round(wall_window, 2),
                "j_per_item_raw": round(e_window / n_window, 3),
                "j_per_item_net_of_idle": round((e_window - idle_w * wall_window) / n_window, 3),
                "whole_cell_j_incl_load": round(e_cell_end - e_cell_start, 2),
                "whole_cell_s_incl_load": round(t_cell_end - t_cell_start, 2),
                "mean_power_w": round(e_window / wall_window, 2) if wall_window else None,
            }
            results["cells"].append(summary)
            hf_cache_volume.commit()
    finally:
        sampler.terminate()
        trace.close()
        hf_cache_volume.commit()
    return results


def _probe_specs(sample_n=100, sample_seed=7, pool_n=300):
    prompt = (REPO_ROOT / "prompts" / "count_vlmevalkit_v1.txt").read_text()
    specs = []
    for model, budgets in DIALS:
        for b in budgets:
            cfg = {"part": "probe", "model": model, "task": "count",
                   "prompt_id": "count_vlmevalkit_v1", "max_new_tokens": 16,
                   "image_token_budget": b, "seed": 0,
                   "require_gpu_substrings": ["A100", "40GB"],
                   "sample": {"n": sample_n, "seed": sample_seed}}
            specs.append((cfg, prompt))
    return [{"_pool_n": pool_n, "_seed": sample_seed, "_n_target": sample_n}, specs]


@app.local_entrypoint()
def main(sample_n: int = 100, sample_seed: int = 7, pool_n: int = 300):
    result = power_probe.remote(_probe_specs(sample_n, sample_seed, pool_n))
    print("PROBE GPU (single die for all cells):", result["gpu"])
    out_dir = REPO_ROOT / "results" / "raw" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    energy_summary = {"gpu": result["gpu"], "cells": []}
    for summary in result["cells"]:
        out = out_dir / f"{summary['cell_id']}.sample{sample_n}s{sample_seed}.jsonl"
        out.write_text(summary.pop("rows_jsonl"))
        energy_summary["cells"].append(
            {k: summary[k] for k in ("cell_id", "gpu", "gpu_uuid", "n_items", "n_errors", "energy")})
        print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, default=str))
        print(f"wrote {out}")
    ej = REPO_ROOT / "results" / "probe_energy.json"
    ej.write_text(json.dumps(energy_summary, indent=1))
    print(f"wrote {ej}")
    pf = REPO_ROOT / "results" / "probe_preflight.json"
    pf.write_text(json.dumps(result["preflight"], indent=1))
    print(f"wrote {pf} (eligible {result['preflight']['eligible']}/{result['preflight']['pool_n']})")
