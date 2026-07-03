"""Phase 0 load matrix: load every checkpoint on Modal and run one 2-image inference.

This is a functional check, not a benchmark: fixed tiny synthetic inputs only.
Failures are findings — every step is wrapped, and the function returns a full JSON
record either way (tracebacks included). Records per checkpoint:
  - transformers / torch / CUDA versions, GPU name
  - requested + resolved revision sha
  - remote-code *.py file sha256 hashes (where trust_remote_code is used)
  - dtype actually used, parameter count, load/gen wall-clock, peak GPU memory
  - raw output of the 2-image inference
  - moondream2: all dated revisions found, newest picked, skill API signatures
    (query/detect/point/caption) as the record of exposed sampler controls

Usage (from repo root):
  .venv/bin/modal run harness/load_check.py                 # all six
  .venv/bin/modal run harness/load_check.py --only lfm450m,moondream2
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import (
    GPU_TYPE,
    HF_CACHE_PATH,
    MOONDREAM2_IMAGE,
    TF457_IMAGE,
    TF5_IMAGE,
    hf_cache_volume,
)

app = modal.App("edge-vlm-phase0-load-check")

TIMEOUT = 2700
VOLUMES = {HF_CACHE_PATH: hf_cache_volume}

PROMPT_2IMG = "Describe each of the two images in one short sentence."
PROMPT_1IMG = "Describe this image in one short sentence."


# --------------------------------------------------------------------------
# In-container helpers (imported lazily so this module also imports locally)
# --------------------------------------------------------------------------

def _synthetic_images():
    from PIL import Image, ImageDraw

    img1 = Image.new("RGB", (640, 480), "white")
    d1 = ImageDraw.Draw(img1)
    d1.ellipse([200, 120, 440, 360], fill="red", outline="black", width=4)

    img2 = Image.new("RGB", (640, 480), (200, 200, 200))
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([180, 100, 460, 380], fill="blue", outline="black", width=4)
    return img1, img2


def _env_info():
    import torch
    import transformers

    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version.split()[0],
    }


def _resolved_sha(model_id, revision):
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id, revision=revision).sha


def _remote_code_hashes(model_id, revision):
    import hashlib
    import pathlib

    from huggingface_hub import snapshot_download

    d = snapshot_download(model_id, revision=revision, allow_patterns=["*.py"])
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(pathlib.Path(d).rglob("*.py"))
    }


def _step(record, name, fn):
    """Run one step; append {name, ok, seconds, detail|error} to record['steps']."""
    import time
    import traceback

    t0 = time.monotonic()
    entry = {"name": name}
    try:
        entry["detail"] = fn()
        entry["ok"] = True
    except Exception:
        entry["ok"] = False
        entry["error"] = traceback.format_exc()
    entry["seconds"] = round(time.monotonic() - t0, 2)
    record["steps"].append(entry)
    return entry


def _from_pretrained_bf16(cls, model_id, revision, **kw):
    """Load in bf16, tolerating the 4.x->5.x torch_dtype -> dtype rename."""
    import torch

    try:
        return cls.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, **kw), "dtype=torch.bfloat16"
    except TypeError:
        return (
            cls.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16, **kw),
            "torch_dtype=torch.bfloat16",
        )


def _cast_and_move(inputs, model):
    import torch

    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(model.device)
            if v.is_floating_point():
                v = v.to(model.dtype)
        out[k] = v
    return out


def _check_itt(model_id, revision, trust_remote_code=False):
    """Generic AutoModelForImageTextToText + chat-template two-image check."""
    import time

    import torch

    record = {"model": model_id, "loader": "image-text-to-text", "revision_requested": revision, "steps": []}
    record["env"] = _env_info()
    _step(record, "resolve_sha", lambda: _resolved_sha(model_id, revision))
    if trust_remote_code:
        _step(record, "remote_code_hashes", lambda: _remote_code_hashes(model_id, revision))

    holder = {}

    def load():
        from transformers import AutoModelForImageTextToText, AutoProcessor

        holder["processor"] = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        model, dtype_kwarg = _from_pretrained_bf16(
            AutoModelForImageTextToText,
            model_id,
            revision,
            trust_remote_code=trust_remote_code,
            device_map="cuda",
        )
        holder["model"] = model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        return {"dtype_kwarg": dtype_kwarg, "model_dtype": str(model.dtype), "n_params": n_params}

    if not _step(record, "load", load)["ok"]:
        return record

    def infer():
        img1, img2 = _synthetic_images()
        processor, model = holder["processor"], holder["model"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": PROMPT_2IMG},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = _cast_and_move(dict(inputs), model)
        n_in = inputs["input_ids"].shape[1]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        gen_s = time.monotonic() - t0
        text = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        return {
            "prompt": PROMPT_2IMG,
            "n_input_tokens": int(n_in),
            "output": text,
            "gen_seconds": round(gen_s, 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }

    _step(record, "two_image_inference", infer)
    return record


def _check_internvl(model_id, revision):
    """InternVL3_5 remote-code path: AutoModel + model.chat(), one call per image.

    Falls back per InternVL convention: 448x448, ImageNet normalization, bf16.
    If the remote .chat() API differs, the traceback is the finding.
    """
    import time

    import torch

    record = {"model": model_id, "loader": "internvl-remote-chat", "revision_requested": revision, "steps": []}
    record["env"] = _env_info()
    _step(record, "resolve_sha", lambda: _resolved_sha(model_id, revision))
    _step(record, "remote_code_hashes", lambda: _remote_code_hashes(model_id, revision))

    holder = {}

    def load():
        from transformers import AutoModel, AutoTokenizer

        model, dtype_kwarg = _from_pretrained_bf16(
            AutoModel,
            model_id,
            revision,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        holder["model"] = model.eval().cuda()
        holder["tokenizer"] = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, use_fast=False
        )
        n_params = sum(p.numel() for p in model.parameters())
        return {"dtype_kwarg": dtype_kwarg, "model_dtype": str(model.dtype), "n_params": n_params}

    if not _step(record, "load", load)["ok"]:
        return record

    def infer():
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose(
            [
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        model, tok = holder["model"], holder["tokenizer"]
        outs = []
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        for img in _synthetic_images():
            pv = transform(img).unsqueeze(0).to(torch.bfloat16).cuda()
            with torch.inference_mode():
                resp = model.chat(tok, pv, "<image>\n" + PROMPT_1IMG, dict(max_new_tokens=64, do_sample=False))
            outs.append(resp)
        return {
            "prompt": PROMPT_1IMG,
            "note": "two single-image chat() calls (remote-code API)",
            "outputs": outs,
            "gen_seconds": round(time.monotonic() - t0, 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }

    _step(record, "two_image_inference", infer)
    return record


def _check_moondream(model_id):
    """moondream2: newest dated revision, trust_remote_code, query()+detect().

    Also records the *signatures* of query/detect/point/caption — that is the
    ground truth for which sampler/decoding controls the skills expose.
    """
    import inspect
    import re
    import time

    import torch

    record = {"model": model_id, "loader": "moondream-skills", "steps": []}
    record["env"] = _env_info()

    def pick_revision():
        from huggingface_hub import HfApi

        refs = HfApi().list_repo_refs(model_id)
        names = [r.name for r in list(refs.branches) + list(refs.tags)]
        dated = sorted(n for n in names if re.fullmatch(r"\d{4}-\d{2}-\d{2}", n))
        return {"all_refs": names, "dated": dated, "newest": dated[-1] if dated else None}

    picked = _step(record, "enumerate_dated_revisions", pick_revision)
    if not picked["ok"] or not picked["detail"]["newest"]:
        return record
    revision = picked["detail"]["newest"]
    record["revision_requested"] = revision

    _step(record, "resolve_sha", lambda: _resolved_sha(model_id, revision))
    _step(record, "remote_code_hashes", lambda: _remote_code_hashes(model_id, revision))

    holder = {}

    def load():
        from transformers import AutoModelForCausalLM

        try:
            model, dtype_kwarg = _from_pretrained_bf16(
                AutoModelForCausalLM, model_id, revision, trust_remote_code=True, device_map={"": "cuda"}
            )
        except Exception:
            # Some moondream revisions manage dtype internally; retry without.
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, trust_remote_code=True, device_map={"": "cuda"}
            )
            dtype_kwarg = "none (model default)"
        holder["model"] = model.eval()
        sigs = {}
        for skill in ("query", "detect", "point", "caption"):
            fn = getattr(model, skill, None)
            sigs[skill] = str(inspect.signature(fn)) if fn is not None else "MISSING"
        n_params = sum(p.numel() for p in model.parameters())
        return {
            "dtype_kwarg": dtype_kwarg,
            "model_dtype": str(next(model.parameters()).dtype),
            "n_params": n_params,
            "skill_signatures": sigs,
        }

    if not _step(record, "load", load)["ok"]:
        return record

    def infer():
        img1, img2 = _synthetic_images()
        model = holder["model"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        q1 = model.query(img1, PROMPT_1IMG)
        q2 = model.query(img2, PROMPT_1IMG)
        det = model.detect(img1, "circle")
        return {
            "note": "query() on both images + detect('circle') on image 1",
            "query_outputs": [q1, q2],
            "detect_output": det,
            "gen_seconds": round(time.monotonic() - t0, 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }

    _step(record, "two_image_inference", infer)
    return record


# --------------------------------------------------------------------------
# Modal functions — one per environment image
# --------------------------------------------------------------------------

def _json_safe(obj):
    """Force the record to JSON primitives before it crosses the Modal boundary.

    The local control plane has no torch, so any torch-referencing object in a
    pickled return value (or raised exception) would fail to deserialize there.
    """
    return json.loads(json.dumps(obj, default=str))


def _guarded(fn, fallback_id):
    import traceback

    try:
        rec = fn()
    except Exception:
        rec = {"model": fallback_id, "fatal": traceback.format_exc()}
    return _json_safe(rec)


@app.function(image=TF5_IMAGE, gpu=GPU_TYPE, timeout=TIMEOUT, volumes=VOLUMES)
def check_tf5(model_id: str, revision: str = "main", trust_remote_code: bool = False):
    return _guarded(lambda: _check_itt(model_id, revision, trust_remote_code), model_id)


@app.function(image=TF457_IMAGE, gpu=GPU_TYPE, timeout=TIMEOUT, volumes=VOLUMES)
def check_tf457(model_id: str, revision: str = "main", trust_remote_code: bool = False, loader: str = "itt"):
    if loader == "internvl":
        return _guarded(lambda: _check_internvl(model_id, revision), model_id)
    return _guarded(lambda: _check_itt(model_id, revision, trust_remote_code), model_id)


@app.function(image=MOONDREAM2_IMAGE, gpu=GPU_TYPE, timeout=TIMEOUT, volumes=VOLUMES)
def check_moondream(model_id: str = "vikhyatk/moondream2"):
    return _guarded(lambda: _check_moondream(model_id), model_id)


PLAN = {
    "lfm450m": (check_tf5, {"model_id": "LiquidAI/LFM2.5-VL-450M"}),
    "smolvlm2": (check_tf457, {"model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"}),
    "llava_ov": (check_tf457, {"model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"}),
    "internvl35": (check_tf457, {"model_id": "OpenGVLab/InternVL3_5-1B", "loader": "internvl", "trust_remote_code": True}),
    "qwen25vl": (check_tf457, {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct"}),
    "moondream2": (check_moondream, {}),
}


@app.local_entrypoint()
def main(only: str = ""):
    wanted = [w.strip() for w in only.split(",") if w.strip()] or list(PLAN)
    calls = {}
    for name in wanted:
        fn, kwargs = PLAN[name]
        calls[name] = fn.spawn(**kwargs)
        print(f"spawned {name}")

    results = {}
    for name, call in calls.items():
        try:
            results[name] = call.get()
        except Exception as e:  # infra-level failure — still a finding
            results[name] = {"model": name, "infra_error": repr(e)}
        steps = results[name].get("steps", [])
        status = "OK" if steps and all(s["ok"] for s in steps) else "HAS FAILURES"
        print(f"{name}: {status}")

    out = REPO_ROOT / "results" / "phase0" / "load_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # merge: a partial (--only) rerun must never clobber other models' records
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=2, default=str))
    print(f"wrote {out} (models now recorded: {sorted(existing)})")
