"""The sweep, maximally fixed (change log #5, owner directive):

Latency is a first-class output of the sweep, so the sweep runs on ONE GPU,
the SAME physical GPU, ONE cell at a time:

  - all 4 LFM cells run strictly serially inside ONE container on ONE physical
    A100-40GB (GPU UUID recorded in every row as proof of a single die);
  - then all 4 SmolVLM cells run the same way in a second single container,
    strictly AFTER the first (never concurrent).

LFM and SmolVLM cannot share a container (transformers 5.12.1 vs 4.57.1 —
incompatible pinned environments), so "same physical GPU" holds within each
model's curve — where latency points are compared — and across models it is
same-SKU-asserted (A100 + 40GB) on two dies, recorded, not hidden.

No resume: if a container dies mid-curve, that model's curve restarts from
zero on a fresh single GPU (fresh_start=True) — a curve never mixes devices.

Usage (from repo root): .venv/bin/modal run harness/run_sweep_serial.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import GPU_TYPE, HF_CACHE_PATH, TF457_IMAGE, TF5_IMAGE, hf_cache_volume
from harness.run import _run_cell, cell_id

app = modal.App("edge-vlm-sweep-serial")

VOLUMES = {HF_CACHE_PATH: hf_cache_volume}
RESOURCES = dict(cpu=8.0, memory=32768)
TIMEOUT = 24 * 3600  # one container hosts a full model curve


def _run_curve(cell_specs):
    """Run a list of (config, prompt_template) cells strictly serially on the
    one GPU this container owns. Returns per-cell summaries incl. rows."""
    import gc
    import subprocess

    import torch

    uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    out = {"gpu": uuid, "cells": []}
    for config, prompt_template in cell_specs:
        config = dict(config, fresh_start=True)  # a curve never resumes across devices
        summary = _run_cell(config, prompt_template)
        out["cells"].append(summary)
        gc.collect()
        torch.cuda.empty_cache()
    return out


@app.function(image=TF5_IMAGE, gpu=GPU_TYPE, timeout=TIMEOUT, volumes=VOLUMES, **RESOURCES)
def sweep_curve_lfm(cell_specs: list):
    return _run_curve(cell_specs)


@app.function(image=TF457_IMAGE, gpu=GPU_TYPE, timeout=TIMEOUT, volumes=VOLUMES, **RESOURCES)
def sweep_curve_smolvlm(cell_specs: list):
    return _run_curve(cell_specs)


def _load_specs(pattern):
    import yaml

    specs = []
    for cfg_path in sorted((REPO_ROOT / "configs" / "sweep").glob(pattern),
                           key=lambda p: int(p.stem.rsplit("_b", 1)[1])):
        cfg = yaml.safe_load(cfg_path.read_text())
        prompt = (REPO_ROOT / "prompts" / f"{cfg['prompt_id']}.txt").read_text()
        specs.append((cfg, prompt))
    return specs


def _persist(curve_result):
    for summary in curve_result["cells"]:
        out = REPO_ROOT / "results" / "raw" / "sweep" / f"{summary['cell_id']}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary.pop("rows_jsonl"))
        print(json.dumps(summary, default=str))
        print(f"wrote {out}")


@app.local_entrypoint()
def main():
    print("=== LFM curve: one container, one GPU, 4 cells serial ===")
    lfm = sweep_curve_lfm.remote(_load_specs("lfm450m_*.yaml"))
    print("LFM curve GPU:", lfm["gpu"])
    _persist(lfm)

    print("=== SmolVLM curve: second container, strictly after ===")
    smol = sweep_curve_smolvlm.remote(_load_specs("smolvlm2_*.yaml"))
    print("SmolVLM curve GPU:", smol["gpu"])
    _persist(smol)
