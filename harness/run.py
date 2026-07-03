"""The config-driven cell runner — ONE harness for both parts of the project.

A cell = one YAML in configs/{bench,sweep}/ specifying (model, task[, budget]).
All inference runs as Modal functions on a single A100-40GB; the local side only
dispatches and collects the raw JSONL.

Guarantees:
  - greedy decoding (do_sample=False baked into adapters), fixed max_new_tokens
  - GPU type asserted in-container (Modal can substitute hardware; we refuse)
  - incremental persistence to a Modal Volume + resume by item_id, so a
    container death never loses completed items
  - per-item: one retry on infrastructure failure only (CUDA/OOM); model
    behavior failures are recorded as findings, never retried
  - raw JSONL row carries full provenance (revisions, versions, device, prompt)

Usage (from repo root):
  .venv/bin/modal run harness/run.py --config configs/bench/lfm450m_grounding.yaml
  .venv/bin/modal run harness/run.py --config ... --sample-n 20 --sample-seed 7   # Phase 2
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import (
    DATASETS_IMAGE,
    GPU_TYPE,
    HF_CACHE_PATH,
    MOONDREAM2_IMAGE,
    TF457_IMAGE,
    TF5_IMAGE,
    hf_cache_volume,
)

app = modal.App("edge-vlm-runner")

TIMEOUT = 12 * 3600
VOLUMES = {HF_CACHE_PATH: hf_cache_volume}
RESULTS_DIR_IN_VOLUME = Path(HF_CACHE_PATH) / "results"

INFRA_ERROR_MARKERS = ("CUDA", "out of memory", "device-side", "NCCL", "cudnn")


def cell_id(config):
    base = f"{config['part']}_{config['model']}_{config['task']}"
    if config.get("image_token_budget") is not None:
        base += f"_b{config['image_token_budget']}"
    return base


# --------------------------------------------------------------------------
# In-container cell execution
# --------------------------------------------------------------------------

def _assert_gpu(required_substrings):
    import torch

    name = torch.cuda.get_device_name(0)
    for sub in required_substrings:
        if sub not in name:
            raise RuntimeError(
                f"GPU_MISMATCH: got '{name}', required substrings {required_substrings}. "
                "Refusing to run a main-table cell on substituted hardware."
            )
    return name


def _run_cell(config, prompt_template):
    import time
    import traceback

    import torch
    import transformers

    from harness import data as hdata
    from harness.adapters import MODEL_PINS, build_adapter

    cid = cell_id(config)
    gpu_name = _assert_gpu(config.get("require_gpu_substrings", ["A100", "40GB"]))
    torch.manual_seed(int(config.get("seed", 0)))

    pin = MODEL_PINS[config["model"]]
    adapter = build_adapter(config["model"], image_token_budget=config.get("image_token_budget"))

    items = hdata.load_items(config["task"])
    sample_meta = None
    if config.get("sample"):
        items, chosen = hdata.fixed_sample(
            items, int(config["sample"]["n"]), int(config["sample"]["seed"]), config["task"]
        )
        sample_meta = {"n": len(items), "seed": config["sample"]["seed"], "indices": chosen}

    meta = {
        "model_id": pin["model_id"],
        "revision": pin["revision"],
        "resolved_sha": pin.get("resolved_sha", pin["revision"]),
        "env": pin["env"],
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": gpu_name,
        "dtype": "bfloat16",
        "loader": pin["adapter"],
        "decoding": {"do_sample": False, "max_new_tokens": int(config["max_new_tokens"])},
        "dataset_revision": hdata.DATASET_PINS[hdata.TASK_TO_DATASET[config["task"]]]["revision"],
        "prompt_id": config["prompt_id"],
        "image_token_budget": config.get("image_token_budget"),
        "runtime": None,
        "quantization": None,
    }

    # incremental persistence + resume
    out_path = RESULTS_DIR_IN_VOLUME / config["part"] / f"{cid}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists() and not config.get("sample"):
        for line in out_path.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["item_id"])
            except Exception:
                pass
    mode = "a" if done_ids else "w"

    n_new = n_err = 0
    with open(out_path, mode) as fh:
        for item in items:
            if item["item_id"] in done_ids:
                continue
            prompt = (
                prompt_template.replace("{expression}", item.get("expression", ""))
                .replace("{question}", item.get("question", ""))
            )
            row = {
                "part": config["part"],
                "model": config["model"],
                "task": config["task"],
                "item_id": item["item_id"],
                "image_ref": item["image_ref"],
                "prompt_id": config["prompt_id"],
                "prompt": prompt,
                "image_token_budget": config.get("image_token_budget"),
                "image_width": item["image_width"],
                "image_height": item["image_height"],
                "gt": item["gt"],
                "raw_output": None,
                "latency_ms": None,
                "error": None,
                "meta": meta,
            }
            for attempt in (1, 2):
                try:
                    gen = adapter.generate(
                        item["image"], prompt, int(config["max_new_tokens"]), task=config["task"]
                    )
                    row.update(raw_output=gen["raw_output"], latency_ms=gen["latency_ms"])
                    row["n_input_tokens"] = gen.get("n_input_tokens")
                    break
                except Exception:
                    tb = traceback.format_exc()
                    infra = any(m in tb for m in INFRA_ERROR_MARKERS)
                    row["error"] = tb.strip().splitlines()[-1][:500]
                    if infra and attempt == 1:
                        torch.cuda.empty_cache()
                        time.sleep(2)
                        continue  # one retry on infrastructure failure only
                    n_err += 1
                    break
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 100 == 0:
                hf_cache_volume.commit()
    hf_cache_volume.commit()

    return {
        "cell_id": cid,
        "gpu": gpu_name,
        "n_items": len(items),
        "n_resumed": len(done_ids),
        "n_new": n_new,
        "n_errors": n_err,
        "sample": sample_meta,
        "rows_jsonl": out_path.read_text(),
    }


@app.function(image=TF5_IMAGE, gpu=GPU_TYPE, timeout=TIMEOUT, volumes=VOLUMES,
              retries=modal.Retries(max_retries=1, initial_delay=15.0))
def run_cell_tf5(config: dict, prompt_template: str):
    return _run_cell(config, prompt_template)


@app.function(image=TF457_IMAGE, gpu=GPU_TYPE, timeout=TIMEOUT, volumes=VOLUMES,
              retries=modal.Retries(max_retries=1, initial_delay=15.0))
def run_cell_tf457(config: dict, prompt_template: str):
    return _run_cell(config, prompt_template)


@app.function(image=MOONDREAM2_IMAGE, gpu=GPU_TYPE, timeout=TIMEOUT, volumes=VOLUMES,
              retries=modal.Retries(max_retries=1, initial_delay=15.0))
def run_cell_moondream(config: dict, prompt_template: str):
    return _run_cell(config, prompt_template)


ENV_TO_FN = {"tf5": run_cell_tf5, "tf457": run_cell_tf457, "moondream2": run_cell_moondream}


# --------------------------------------------------------------------------
# Local entrypoint (control plane): dispatch one cell, collect raw JSONL
# --------------------------------------------------------------------------

@app.local_entrypoint()
def main(config: str, sample_n: int = 0, sample_seed: int = 0):
    import yaml

    from harness.adapters import MODEL_PINS

    cfg = yaml.safe_load(Path(config).read_text())
    if sample_n:
        cfg["sample"] = {"n": sample_n, "seed": sample_seed}

    prompt_path = REPO_ROOT / "prompts" / f"{cfg['prompt_id']}.txt"
    prompt_template = prompt_path.read_text()

    env = MODEL_PINS[cfg["model"]]["env"]
    fn = ENV_TO_FN[env]
    print(f"dispatching cell={cell_id(cfg)} env={env} model={cfg['model']} task={cfg['task']}")
    result = fn.remote(cfg, prompt_template)

    suffix = f".sample{sample_n}s{sample_seed}" if sample_n else ""
    out = REPO_ROOT / "results" / "raw" / cfg["part"] / f"{cell_id(cfg)}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.pop("rows_jsonl"))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {out}")
