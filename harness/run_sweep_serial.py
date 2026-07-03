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


def _load_specs(sample_n=0, sample_seed=7):
    import yaml

    specs = []
    for model in ("lfm450m", "smolvlm2"):  # LFM curve first, then SmolVLM
        for cfg_path in sorted((REPO_ROOT / "configs" / "sweep").glob(f"{model}_*.yaml"),
                               key=lambda p: int(p.stem.rsplit("_b", 1)[1])):
            cfg = yaml.safe_load(cfg_path.read_text())
            if sample_n:
                cfg["sample"] = {"n": sample_n, "seed": sample_seed}
            prompt = (REPO_ROOT / "prompts" / f"{cfg['prompt_id']}.txt").read_text()
            specs.append((cfg, prompt))
    return specs


@app.local_entrypoint()
def main(sample_n: int = 0, sample_seed: int = 7, export_locks: bool = False):
    if export_locks:
        locks = export_sweepbox_locks.remote()
        for venv, text in locks.items():
            out = REPO_ROOT / "envs" / "locks" / f"sweepbox-{venv}.lock.txt"
            out.write_text(text)
            print(f"wrote {out} ({len(text.splitlines())} packages)")
        return

    result = sweep_onebox.remote(_load_specs(sample_n, sample_seed))
    print("SWEEP GPU (single die for all cells):", result["gpu"])
    for summary in result["cells"]:
        suffix = f".sample{sample_n}s{sample_seed}" if sample_n else ""
        out = REPO_ROOT / "results" / "raw" / "sweep" / f"{summary['cell_id']}{suffix}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary.pop("rows_jsonl"))
        print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, default=str))
        print(f"wrote {out}")
