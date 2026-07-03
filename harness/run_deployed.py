"""Phase 3B: the deployed-form comparison — ONE BOX, both models, all six cells.

Both as-shipped small forms run in a single CPU container on the same physical
machine, strictly serially, through the SAME shared cell loop as every other
row in the project (harness/cell_core.py):

  1. moondream05b_asshipped  x grounding, pope, count   (int8 .mf, moondream 0.0.6)
  2. lfm450m_gguf            x grounding, pope, count   (vendor Q8_0, llama.cpp b9860)

Same pinned items, same prompt files as the corresponding main-table lanes,
same parsers, same scorer. The host CPU model is recorded in the run summary.
Asymmetry disclosure and precision caveats live in METHODOLOGY.md; results are
reported ONLY in the separate deployed-form table.

Usage:
  .venv/bin/modal run harness/run_deployed.py --sample-n 20   # Gate C2 smoke
  .venv/bin/modal run harness/run_deployed.py                 # full runs
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import HF_CACHE_PATH, MD05_IMAGE, hf_cache_volume

app = modal.App("edge-vlm-deployed-onebox")

RESULTS_BASE = str(Path(HF_CACHE_PATH) / "results")
MAX_NEW_TOKENS = {"grounding": 256, "pope": 16, "count": 16}
# prompts mirror the corresponding main-table lanes
PROMPT_ID = {
    "moondream05b_asshipped": {"grounding": "grounding_moondream2_v1",
                               "pope": "pope_vlmevalkit_v1", "count": "count_vlmevalkit_v1"},
    "lfm450m_gguf": {"grounding": "grounding_lfm450m_v1",
                     "pope": "pope_vlmevalkit_v1", "count": "count_vlmevalkit_v1"},
}
ORDER = [("moondream05b_asshipped", t) for t in ("grounding", "pope", "count")] + \
        [("lfm450m_gguf", t) for t in ("grounding", "pope", "count")]


@app.function(image=MD05_IMAGE, timeout=24 * 3600, volumes={HF_CACHE_PATH: hf_cache_volume},
              cpu=8.0, memory=8192)
def deployed_onebox(cells: list, prompts: dict, sample: dict | None):
    import os
    os.environ["OMP_NUM_THREADS"] = "8"  # onnxruntime thread pool
    from harness.cell_core import run_cell_core

    cpu_model = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    # hardware self-test: Modal hides the CPU name, so record measured proxies —
    # memory copy bandwidth + single-core loop time — for cross-box comparability
    import time as _t
    buf = bytearray(200 * 1024 * 1024)
    t0 = _t.monotonic(); b2 = bytes(buf); bw = 200 / (_t.monotonic() - t0)  # MB/s copy
    t0 = _t.monotonic(); x = 0
    for i in range(10_000_000): x += i
    loop_s = _t.monotonic() - t0
    hw_probe = {"mem_copy_mb_s": round(bw), "single_core_loop_s": round(loop_s, 3)}

    out = {"cpu_model": cpu_model, "hw_probe": hw_probe, "cells": []}
    for model, task in cells:
        config = {
            "part": "deployed",
            "model": model,
            "task": task,
            "prompt_id": PROMPT_ID[model][task],
            "max_new_tokens": MAX_NEW_TOKENS[task],
            "image_token_budget": None,
            "seed": 0,
            "require_gpu_substrings": [],  # CPU box — no GPU assert
            "fresh_start": True,
            "_cpu_reserved": 8.0,
            "_memory_reserved_mb": 8192,
        }
        if sample:
            config["sample"] = sample
        summary = run_cell_core(config, prompts[PROMPT_ID[model][task]], RESULTS_BASE,
                                commit_cb=hf_cache_volume.commit)
        summary["cpu_model"] = cpu_model
        out["cells"].append(summary)
    return out


@app.local_entrypoint()
def main(sample_n: int = 0, sample_seed: int = 7, model: str = ""):
    prompts = {pid: (REPO_ROOT / "prompts" / f"{pid}.txt").read_text()
               for m in PROMPT_ID.values() for pid in m.values()}
    sample = {"n": sample_n, "seed": sample_seed} if sample_n else None

    cells = [(m, t) for m, t in ORDER if not model or m == model]
    result = deployed_onebox.remote(cells, prompts, sample)
    print("DEPLOYED BOX CPU:", result["cpu_model"], "| hw self-test:", result["hw_probe"])
    import json as _j
    (REPO_ROOT / "results" / "raw" / "deployed").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "raw" / "deployed" / f"boxinfo_{model or 'all'}.json").write_text(_j.dumps(result["hw_probe"] | {"cpu_model": result["cpu_model"], "cores": 8, "memory_mb": 8192}))
    for summary in result["cells"]:
        suffix = f".sample{sample_n}s{sample_seed}" if sample_n else ""
        out = REPO_ROOT / "results" / "raw" / "deployed" / f"{summary['cell_id']}{suffix}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary.pop("rows_jsonl"))
        print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, default=str))
        print(f"wrote {out}")
