"""The cell execution core — modal-free, so it runs BOTH inside Modal
functions (bench path, harness/run.py) and inside plain venv subprocesses
(sweep path, harness/sweep_cell_main.py). One loop for every cell in the
project (comparability directive): extracted verbatim from run.py's
_run_cell at change log #5; behavior identical.
"""

import json
from pathlib import Path

INFRA_ERROR_MARKERS = ("CUDA", "out of memory", "device-side", "NCCL", "cudnn")


def cell_id(config):
    base = f"{config['part']}_{config['model']}_{config['task']}"
    if config.get("image_token_budget") is not None:
        base += f"_b{config['image_token_budget']}"
    return base


def compute_out_path(config, results_base):
    fname = cell_id(config)
    if config.get("sample"):
        fname += f".sample{config['sample']['n']}s{config['sample']['seed']}"
    return Path(results_base) / config["part"] / f"{fname}.jsonl"


def assert_gpu(required_substrings):
    """Empty/None requirement (CPU deployed-form cells) -> no GPU assert."""
    if not required_substrings:
        return None
    import torch

    name = torch.cuda.get_device_name(0)
    for sub in required_substrings:
        if sub not in name:
            raise RuntimeError(
                f"GPU_MISMATCH: got '{name}', required substrings {required_substrings}. "
                "Refusing to run a main-table cell on substituted hardware."
            )
    return name


def run_cell_core(config, prompt_template, results_base, commit_cb=None):
    """Run one cell; returns the summary dict (incl. rows_jsonl)."""
    import subprocess
    import time
    import traceback

    try:  # absent in the deployed-form (CPU, non-transformers) image
        import torch
        import transformers
        _torch_version, _tf_version = torch.__version__, transformers.__version__
    except ImportError:
        torch = None
        _torch_version = _tf_version = None

    from harness import data as hdata
    from harness.adapters import MODEL_PINS, build_adapter

    cid = cell_id(config)
    gpu_name = assert_gpu(config.get("require_gpu_substrings", ["A100", "40GB"]))
    if torch is not None:
        torch.manual_seed(int(config.get("seed", 0)))

    pin = MODEL_PINS[config["model"]]
    adapter = build_adapter(config["model"], image_token_budget=config.get("image_token_budget"))
    if config.get("upscale_fallback"):
        # change #14 (sweep_v2): disclosed local pre-upscale for forced-budget
        # cells whose too-small images crash the pinned stack (change #12).
        adapter.upscale_fallback = True

    items = hdata.load_items(config["task"])
    sample_meta = None
    if config.get("sample"):
        if config["sample"].get("item_ids"):
            # Explicit fixed item set (probe preflight, change #12): the caller
            # provides the exact item_ids; n/seed in the sample dict name the
            # file and record how the set was derived (see "derivation").
            wanted = set(config["sample"]["item_ids"])
            items = [i for i in items if i["item_id"] in wanted]
            sample_meta = {"n": len(items), "seed": config["sample"].get("seed"),
                           "explicit_item_ids": sorted(wanted),
                           "derivation": config["sample"].get("derivation")}
        else:
            items, chosen = hdata.fixed_sample(
                items, int(config["sample"]["n"]), int(config["sample"]["seed"]), config["task"]
            )
            sample_meta = {"n": len(items), "seed": config["sample"]["seed"], "indices": chosen}

    try:
        gpu_uuid = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
    except FileNotFoundError:  # CPU container
        gpu_uuid = None

    meta = {
        "gpu_uuid": gpu_uuid,
        "model_id": pin["model_id"],
        "revision": pin["revision"],
        "resolved_sha": pin.get("resolved_sha", pin["revision"]),
        "env": pin["env"],
        "transformers": _tf_version,
        "torch": _torch_version,
        "gpu": gpu_name,
        "dtype": pin.get("dtype", "bfloat16"),
        "loader": pin["adapter"],
        "decoding": {"do_sample": False, "max_new_tokens": int(config["max_new_tokens"])},
        "dataset_revision": hdata.DATASET_PINS[hdata.TASK_TO_DATASET[config["task"]]]["revision"],
        "prompt_id": config["prompt_id"],
        "image_token_budget": config.get("image_token_budget"),
        "upscale_fallback": bool(config.get("upscale_fallback")),
        "runtime": pin.get("runtime"),
        "quantization": pin.get("quantization"),
        "cpu_reserved": config.get("_cpu_reserved"),
        "memory_reserved_mb": config.get("_memory_reserved_mb"),
    }

    out_path = compute_out_path(config, results_base)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if config.get("fresh_start") and out_path.exists():
        out_path.unlink()  # sweep serial mode: a curve never resumes across devices
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
                    row["pre_upscale_factor"] = gen.get("pre_upscale_factor")
                    break
                except Exception:
                    tb = traceback.format_exc()
                    infra = any(m in tb for m in INFRA_ERROR_MARKERS)
                    row["error"] = tb.strip().splitlines()[-1][:500]
                    if infra and attempt == 1:
                        if torch is not None:
                            torch.cuda.empty_cache()
                        time.sleep(2)
                        continue  # one retry on infrastructure failure only
                    n_err += 1
                    break
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 100 == 0 and commit_cb is not None:
                commit_cb()
    if commit_cb is not None:
        commit_cb()

    return {
        "cell_id": cid,
        "gpu": gpu_name,
        "gpu_uuid": gpu_uuid,
        "n_items": len(items),
        "n_resumed": len(done_ids),
        "n_new": n_new,
        "n_errors": n_err,
        "sample": sample_meta,
        "out_path": str(out_path),
        "rows_jsonl": out_path.read_text(),
    }
