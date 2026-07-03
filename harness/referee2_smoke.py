"""Feasibility smoke for the referee RE-PIN (change #15): prove the two new
images work end-to-end BEFORE rewriting the runner.

  smoke_lfm    — VLMEVAL2_TF5_IMAGE: LFM2.5-VL-450M is registered, loads via the
                 tool's LFM2VL adapter at OUR pinned revision on transformers
                 5.x, and generates. This is the novel, risky path.
  smoke_exist  — VLMEVAL2_TF457_IMAGE: the newer VLMEvalKit framework still
                 imports and runs an existing model (SmolVLM2) on transformers
                 4.57.1 (so the 4 models can reproduce their numbers).

Usage: .venv/bin/modal run harness/referee2_smoke.py
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
    VLMEVAL2_TF5_IMAGE,
    VLMEVAL2_TF457_IMAGE,
    hf_cache_volume,
)

app = modal.App("edge-vlm-referee2-smoke")
VOL = {HF_CACHE_PATH: hf_cache_volume}


def _try_model(model_name):
    out = {"model": model_name}
    try:
        from PIL import Image
        Image.new("RGB", (320, 240), (128, 128, 128)).save("/tmp/t.png")
        from vlmeval.config import supported_VLM
        out["registered"] = model_name in supported_VLM
        assert out["registered"], f"{model_name} not in supported_VLM"
        model = supported_VLM[model_name]()
        msg = [{"type": "image", "value": "/tmp/t.png"},
               {"type": "text", "value": "Is there a cat in the image? Please answer yes or no."}]
        resp = model.generate(msg, dataset="POPE")
        import transformers
        out.update(ok=True, resp=str(resp)[:200], transformers=transformers.__version__)
    except Exception:
        import traceback
        out.update(ok=False, err=traceback.format_exc()[-2500:])
    return out


@app.function(image=VLMEVAL2_TF5_IMAGE, gpu=GPU_TYPE, timeout=1800, volumes=VOL,
              cpu=8.0, memory=32768)
def smoke_lfm():
    return _try_model("LFM2.5-VL-450M")


@app.function(image=VLMEVAL2_TF457_IMAGE, gpu=GPU_TYPE, timeout=1800, volumes=VOL,
              cpu=8.0, memory=32768)
def smoke_exist():
    return _try_model("SmolVLM2-500M")


@app.local_entrypoint()
def main():
    r5 = smoke_lfm.remote()
    print("=== LFM (tf5) ===")
    print(json.dumps(r5, indent=1)[:3000])
    r457 = smoke_exist.remote()
    print("=== SmolVLM2 (tf457, newer VLMEvalKit) ===")
    print(json.dumps(r457, indent=1)[:3000])
    print("SMOKE_OK" if (r5.get("ok") and r457.get("ok")) else "SMOKE_FAIL")
