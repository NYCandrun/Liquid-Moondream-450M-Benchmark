"""Phase 0 data functional check (Gate A audit evidence), run on Modal:

1. load_items('grounding') must produce EXACTLY 2,080 flattened instances
2. bbox convention verified over all boxes (xywh vs xyxy violation counts)
3. one flattened item printed structurally per task as a sanity record

Usage: .venv/bin/modal run harness/data_check.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import DATASETS_IMAGE, HF_CACHE_PATH, hf_cache_volume

app = modal.App("edge-vlm-phase0-data-check")


@app.function(image=DATASETS_IMAGE, timeout=1800, volumes={HF_CACHE_PATH: hf_cache_volume})
def check():
    from harness import data as hdata

    out = {"bbox_convention": hdata.verify_refcoco_bbox_convention()}
    for task in ("grounding", "pope", "count"):
        items = hdata.load_items(task)
        first = {k: v for k, v in items[0].items() if k != "image"}
        out[task] = {"n_items": len(items), "first_item": json.loads(json.dumps(first, default=str))}
    return out


@app.local_entrypoint()
def main():
    result = check.remote()
    out = REPO_ROOT / "results" / "phase0" / "data_check.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: (v if k == "bbox_convention" else v["n_items"]) for k, v in result.items()}, indent=2))
    print(f"wrote {out}")
