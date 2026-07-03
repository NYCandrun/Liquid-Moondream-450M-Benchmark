"""Phase 0 knob-change check (sweep part): prove the image-token budget is LIVE.

For LFM (true token budget) and SmolVLM2 (resolution/tile dial), load once and run
one inference at two budget settings; the knob is live iff the realized input
token count differs. Uses a large synthetic image (2500x1800) so resizing always
binds (COCO-sized sources would not exercise SmolVLM's upper points).

Usage: .venv/bin/modal run harness/knob_check.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import GPU_TYPE, HF_CACHE_PATH, TF457_IMAGE, TF5_IMAGE, hf_cache_volume

app = modal.App("edge-vlm-phase0-knob-check")
VOLUMES = {HF_CACHE_PATH: hf_cache_volume}


def _big_image():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (2500, 1800), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([400, 300, 1100, 1000], fill="red", outline="black", width=8)
    d.rectangle([1500, 700, 2200, 1500], fill="blue", outline="black", width=8)
    return img


def _probe(model_key, budgets):
    import json as _json
    import traceback

    from harness.adapters import build_adapter

    out = {"model": model_key, "points": []}
    img = _big_image()
    for b in budgets:
        entry = {"budget": b}
        try:
            adapter = build_adapter(model_key, image_token_budget=b)
            gen = adapter.generate(img, "Describe this image in one short sentence.", 32)
            entry.update(
                n_input_tokens=gen["n_input_tokens"],
                latency_ms=gen["latency_ms"],
                output=gen["raw_output"][:120],
            )
            del adapter
        except Exception:
            entry["error"] = traceback.format_exc()
        out["points"].append(entry)
    toks = [p.get("n_input_tokens") for p in out["points"]]
    out["knob_is_live"] = len(set(t for t in toks if t is not None)) > 1
    return _json.loads(_json.dumps(out, default=str))


@app.function(image=TF5_IMAGE, gpu=GPU_TYPE, timeout=1800, volumes=VOLUMES)
def probe_lfm():
    # 32 (card floor) / 64 / 128 / 256 — all pinned min=max=T, splitting off
    return _probe("lfm450m", [32, 64, 128, 256])


@app.function(image=TF457_IMAGE, gpu=GPU_TYPE, timeout=1800, volumes=VOLUMES)
def probe_smolvlm():
    # longest_edge sweep; 2048 is the shipped default
    return _probe("smolvlm2", [512, 1024, 1536, 2048])


@app.local_entrypoint()
def main():
    calls = {"lfm450m": probe_lfm.spawn(), "smolvlm2": probe_smolvlm.spawn()}
    results = {}
    for name, call in calls.items():
        try:
            results[name] = call.get()
        except Exception as e:
            results[name] = {"model": name, "infra_error": repr(e)}
        print(name, "knob_is_live:", results[name].get("knob_is_live"))
    out = REPO_ROOT / "results" / "phase0" / "knob_check.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
