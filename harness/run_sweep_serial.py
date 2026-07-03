"""The sweep, maximally fixed (change log #5 as-executed, owner directive):

THE WHOLE SWEEP RUNS IN ONE CONTAINER — one physical A100-40GB for every cell
of both curves, one cell at a time, nothing concurrent:

  - the SWEEPBOX image carries BOTH pinned environments as virtualenvs
    (/opt/tf5 = transformers 5.12.1 for LFM; /opt/tf457 = 4.57.1 for SmolVLM),
    version pins identical to the standalone lane images;
  - each cell executes as a subprocess in its matching venv via
    harness/sweep_cell_main.py, running the SAME shared cell core as the bench
    (harness/cell_core.py);
  - order: LFM b32 -> b64 -> b128 -> b256, then SmolVLM 512 -> 1024 -> 1536
    -> 2048, strictly sequential;
  - the GPU UUID is recorded in every row — a single die for the entire sweep;
  - no resume (fresh_start): if the container dies, the sweep restarts from
    zero on a fresh single GPU, so the dataset never mixes devices.

Usage (from repo root):
  .venv/bin/modal run harness/run_sweep_serial.py                # full sweep
  .venv/bin/modal run harness/run_sweep_serial.py --sample-n 3   # smoke of the path
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import GPU_TYPE, HF_CACHE_PATH, SWEEPBOX_IMAGE, hf_cache_volume
from harness.cell_core import compute_out_path

app = modal.App("edge-vlm-sweep-onebox")

VOLUMES = {HF_CACHE_PATH: hf_cache_volume}
RESULTS_BASE = str(Path(HF_CACHE_PATH) / "results")
VENV_FOR_MODEL = {"lfm450m": "tf5", "smolvlm2": "tf457"}


@app.function(image=SWEEPBOX_IMAGE, gpu=GPU_TYPE, timeout=24 * 3600, volumes=VOLUMES,
              cpu=8.0, memory=32768)
def sweep_onebox(cell_specs: list):
    """Run every sweep cell serially in this one container / one GPU."""
    import os
    import subprocess

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    results = {"gpu": gpu, "cells": []}

    for config, prompt in cell_specs:
        venv = VENV_FOR_MODEL[config["model"]]
        payload = json.dumps({
            "config": dict(config, fresh_start=True, _cpu_reserved=8.0, _memory_reserved_mb=32768),
            "prompt": prompt,
            "results_base": RESULTS_BASE,
        })
        proc = subprocess.run(
            [f"/opt/{venv}/bin/python", "/root/harness/sweep_cell_main.py"],
            input=payload, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "/root"},
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"sweep cell {config['model']} b{config.get('image_token_budget')} failed "
                f"(curve restarts from zero, never mixes devices):\n{proc.stderr[-3000:]}"
            )
        marker = [l for l in proc.stdout.splitlines() if l.startswith("SUMMARY_JSON:")][-1]
        summary = json.loads(marker[len("SUMMARY_JSON:"):])
        summary["rows_jsonl"] = Path(compute_out_path(config, RESULTS_BASE)).read_text()
        results["cells"].append(summary)
        hf_cache_volume.commit()
    return results


@app.function(image=SWEEPBOX_IMAGE, timeout=1200)
def export_sweepbox_locks():
    import subprocess

    return {
        venv: subprocess.run(
            [f"/opt/{venv}/bin/pip", "freeze", "--all"], capture_output=True, text=True
        ).stdout
        for venv in ("tf5", "tf457")
    }


# Token-aligned pairs (Gate A common scale): settings where the two models'
# REALIZED token counts overlap, so rows compare like-for-like:
#   ~64 tokens:   LFM T=64   <->  SmolVLM longest_edge=512 (single 64-token tile)
#   ~256-320:     LFM T=256  <->  SmolVLM longest_edge=1024 (<=320 tokens)
ALIGN_PAIRS = [("lfm450m", 64), ("smolvlm2", 512), ("lfm450m", 256), ("smolvlm2", 1024)]


def _smoke_ids_for_budget(budget, n_crash=5, n_ok=3):
    """First N crashed + first N completed item_ids (file order — deterministic)
    from the ORIGINAL crash-discovery run, for the change-#14 functional smoke.
    Completed controls double as a no-op check: fallback-off and fallback-on
    preprocessing must be identical for items that fit, so greedy outputs
    should reproduce the original run's."""
    raw = REPO_ROOT / "results" / "raw" / "sweep" / f"sweep_lfm450m_grounding_b{budget}.jsonl"
    crashed, completed = [], []
    for line in raw.read_text().splitlines():
        r = json.loads(line)
        (crashed if r.get("error") else completed).append(r["item_id"])
    return crashed[:n_crash] + completed[:n_ok]


def _lfm_v2_specs(smoke=False):
    """Corrected LFM curve (change #14): the SAME four budget configs — same
    prompt, seed, decoding — plus the disclosed local pre-upscale fallback, so
    forced budgets never hit the pinned stack's broken upscale path.
    part=sweep_v2 -> results/raw/sweep_v2/; the original crash-discovery run in
    results/raw/sweep/ stays untouched as provenance."""
    import yaml

    specs = []
    for cfg_path in sorted((REPO_ROOT / "configs" / "sweep").glob("lfm450m_*.yaml"),
                           key=lambda p: int(p.stem.rsplit("_b", 1)[1])):
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["part"] = "sweep_v2"
        cfg["upscale_fallback"] = True
        if smoke:
            ids = _smoke_ids_for_budget(cfg["image_token_budget"])
            if not ids:
                continue
            cfg["sample"] = {
                "n": len(ids), "seed": 0, "item_ids": ids,
                "derivation": "first-5 crashed + first-3 completed item_ids from the "
                              "original run at this budget (change #14 smoke)",
            }
        prompt = (REPO_ROOT / "prompts" / f"{cfg['prompt_id']}.txt").read_text()
        specs.append((cfg, prompt))
    return specs


def _load_specs(sample_n=0, sample_seed=7, align=False, smol_count=False):
    import yaml

    if smol_count:
        # Owner decision (change log #7): SmolVLM2 does not support bounding —
        # its detail sweep runs on COUNTING (a task it supports) instead of the
        # cancelled grounding settings 1024/1536/2048.
        prompt = (REPO_ROOT / "prompts" / "count_vlmevalkit_v1.txt").read_text()
        specs = []
        for b in (512, 1024, 1536, 2048):
            cfg = {"part": "sweep", "model": "smolvlm2", "task": "count",
                   "prompt_id": "count_vlmevalkit_v1", "max_new_tokens": 16,
                   "image_token_budget": b, "is_shipped_default": b == 2048,
                   "seed": 0, "require_gpu_substrings": ["A100", "40GB"]}
            if sample_n:
                cfg["sample"] = {"n": sample_n, "seed": sample_seed}
            specs.append((cfg, prompt))
        return specs

    wanted = ALIGN_PAIRS if align else None
    specs = []
    for model in ("lfm450m", "smolvlm2"):  # LFM curve first, then SmolVLM
        for cfg_path in sorted((REPO_ROOT / "configs" / "sweep").glob(f"{model}_*.yaml"),
                               key=lambda p: int(p.stem.rsplit("_b", 1)[1])):
            cfg = yaml.safe_load(cfg_path.read_text())
            if wanted is not None and (model, cfg["image_token_budget"]) not in wanted:
                continue
            if sample_n:
                cfg["sample"] = {"n": sample_n, "seed": sample_seed}
            prompt = (REPO_ROOT / "prompts" / f"{cfg['prompt_id']}.txt").read_text()
            specs.append((cfg, prompt))
    return specs


@app.local_entrypoint()
def main(sample_n: int = 0, sample_seed: int = 7, export_locks: bool = False, align: bool = False,
         smol_count: bool = False, lfm_upscale: bool = False, lfm_upscale_smoke: bool = False):
    if export_locks:
        locks = export_sweepbox_locks.remote()
        for venv, text in locks.items():
            out = REPO_ROOT / "envs" / "locks" / f"sweepbox-{venv}.lock.txt"
            out.write_text(text)
            print(f"wrote {out} ({len(text.splitlines())} packages)")
        return

    if lfm_upscale or lfm_upscale_smoke:
        specs = _lfm_v2_specs(smoke=lfm_upscale_smoke)
    else:
        specs = _load_specs(sample_n, sample_seed, align=align, smol_count=smol_count)
    result = sweep_onebox.remote(specs)
    print("SWEEP GPU (single die for all cells):", result["gpu"])
    for (config, _prompt), summary in zip(specs, result["cells"]):
        out = compute_out_path(config, REPO_ROOT / "results" / "raw")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary.pop("rows_jsonl"))
        print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, default=str))
        print(f"wrote {out}")
