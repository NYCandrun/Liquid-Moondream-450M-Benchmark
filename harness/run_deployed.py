"""Phase 3B lane 1: Moondream 0.5B AS-SHIPPED, in its own runtime.

Runs the identical pinned benchmark items through the moondream==0.0.6
OnnxVL runtime loading the official int8 .mf artifact (both pinned at Gate A;
sha256-verified at image build). CPU execution — the runtime is onnxruntime-
based and this is the deployed form a buyer actually runs. Greedy-only
decoding (the runtime exposes no sampler controls; its sole knob,
max_tokens, is set per task to match the main-table budgets).

Rows carry runtime + quantization in meta and are written ONLY under the
deployed-form tag (moondream05b_asshipped_*); they never enter the bf16 main
table. Skills: detect() for find-object, query() for yes/no and counting —
mirroring the moondream2 main-table lane.

Usage:
  .venv/bin/modal run harness/run_deployed.py --task grounding --sample-n 20  # Gate C2 smoke
  .venv/bin/modal run harness/run_deployed.py --task grounding               # full cell
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import HF_CACHE_PATH, MD05_IMAGE, hf_cache_volume

app = modal.App("edge-vlm-deployed-md05b")

RESULTS_BASE = str(Path(HF_CACHE_PATH) / "results")
MAX_TOKENS = {"grounding": 256, "pope": 16, "count": 16}


@app.function(image=MD05_IMAGE, timeout=12 * 3600, volumes={HF_CACHE_PATH: hf_cache_volume},
              cpu=8.0, memory=32768)
def run_md05b_cell(task: str, sample: dict | None, prompt_template: str):
    import gzip
    import shutil
    import time
    import traceback

    import moondream as md

    from harness import data as hdata

    # the runtime wants the decompressed .mf
    mf = "/tmp/moondream-0_5b-int8.mf"
    with gzip.open("/opt/moondream-0_5b-int8.mf.gz", "rb") as fin, open(mf, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    model = md.vl(model=mf)

    items = hdata.load_items(task)
    sample_meta = None
    if sample:
        items, chosen = hdata.fixed_sample(items, int(sample["n"]), int(sample["seed"]), task)
        sample_meta = {"n": len(items), "seed": sample["seed"], "indices": chosen}

    meta = {
        "model_id": "vikhyatk/moondream2 (0.5B .mf, branch onnx)",
        "revision": "9dddae84d54db4ac56fe37817aeaeb502ed083e2",
        "artifact": "moondream-0_5b-int8.mf.gz",
        "artifact_sha256": "355c534d5f71fff8524dfeb41ba65f7d07948f910f754de4df242d0037d8be4a",
        "runtime": "moondream==0.0.6 (OnnxVL/onnxruntime, CPU)",
        "quantization": "int8",
        "dtype": None,
        "gpu": None,
        "decoding": {"sampler_controls": "none exposed (greedy-only runtime)",
                     "max_tokens": MAX_TOKENS[task]},
        "dataset_revision": hdata.DATASET_PINS[hdata.TASK_TO_DATASET[task]]["revision"],
        "prompt_id": f"deployed_md05b_{task}_v1",
        "cpu_reserved": 8.0,
        "memory_reserved_mb": 32768,
    }

    cid = f"deployed_moondream05b_asshipped_{task}"
    fname = cid + (f".sample{sample['n']}s{sample['seed']}" if sample else "")
    out_path = Path(RESULTS_BASE) / "deployed" / f"{fname}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_new = n_err = 0
    with open(out_path, "w") as fh:
        for item in items:
            prompt = (prompt_template.replace("{expression}", item.get("expression", ""))
                      .replace("{question}", item.get("question", "")))
            row = {
                "part": "deployed", "model": "moondream05b_asshipped", "task": task,
                "item_id": item["item_id"], "image_ref": item["image_ref"],
                "prompt_id": meta["prompt_id"], "prompt": prompt,
                "image_token_budget": None,
                "image_width": item["image_width"], "image_height": item["image_height"],
                "gt": item["gt"], "raw_output": None, "latency_ms": None,
                "error": None, "meta": meta,
            }
            try:
                t0 = time.monotonic()
                if task == "grounding":
                    result = model.detect(item["image"], prompt)
                    raw = json.dumps(result.get("objects", result), default=str)
                else:
                    settings = {"max_tokens": MAX_TOKENS[task]}
                    result = model.query(item["image"], prompt, settings=settings)
                    raw = result.get("answer", str(result))
                row.update(raw_output=raw, latency_ms=round((time.monotonic() - t0) * 1000, 1))
            except Exception:
                row["error"] = traceback.format_exc().strip().splitlines()[-1][:500]
                n_err += 1
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 100 == 0:
                hf_cache_volume.commit()
    hf_cache_volume.commit()
    return {"cell_id": cid, "n_items": len(items), "n_errors": n_err,
            "sample": sample_meta, "rows_jsonl": out_path.read_text()}


@app.local_entrypoint()
def main(task: str, sample_n: int = 0, sample_seed: int = 7):
    # deployed prompts mirror the moondream2 main-table lane: detect() takes the
    # bare expression; query() takes the same pinned task prompts.
    prompt_file = {"grounding": "grounding_moondream2_v1",
                   "pope": "pope_vlmevalkit_v1",
                   "count": "count_vlmevalkit_v1"}[task]
    prompt = (REPO_ROOT / "prompts" / f"{prompt_file}.txt").read_text()
    sample = {"n": sample_n, "seed": sample_seed} if sample_n else None

    result = run_md05b_cell.remote(task, sample, prompt)
    suffix = f".sample{sample_n}s{sample_seed}" if sample_n else ""
    out = REPO_ROOT / "results" / "raw" / "deployed" / f"{result['cell_id']}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.pop("rows_jsonl"))
    print(json.dumps(result, default=str))
    print(f"wrote {out}")
