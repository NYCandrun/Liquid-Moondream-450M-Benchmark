"""Phase 0 dataset pinning — runs entirely inside Modal (images are never
processed locally). For each benchmark dataset, records:
  - resolved revision sha (pin)
  - split names and exact row counts
  - schema (features) + one structural sample row (no image bytes)
  - evidence for the RefCOCO-M flattening rule (per-list-column object totals)
  - evidence for the CountBenchQA exclusion rule (missing/broken images)
  - POPE split composition (random / popular / adversarial)
  - proof images are retrievable inside Modal (3 images decoded per dataset)

Usage (from repo root):
  .venv/bin/modal run harness/dataset_pin.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import DATASETS_IMAGE, HF_CACHE_PATH, hf_cache_volume

app = modal.App("edge-vlm-phase0-dataset-pin")

VOLUMES = {HF_CACHE_PATH: hf_cache_volume}


def _dataset_sha(repo_id):
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo_id).sha


def _sample_row_structure(row, max_str=200):
    """A JSON-safe structural view of one row (image bytes elided)."""
    def conv(v):
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        if isinstance(v, str):
            return v[:max_str]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items() if k != "bytes"} | (
                {"bytes": f"<{len(v['bytes'])} bytes elided>"} if "bytes" in v and v["bytes"] else {}
            )
        if isinstance(v, list):
            return [conv(x) for x in v[:5]] + ([f"... +{len(v) - 5} more"] if len(v) > 5 else [])
        return f"<{type(v).__name__}>"

    return {k: conv(v) for k, v in row.items()}


def _decode_first_images(ds, image_col, n=3):
    """Decode n images to prove retrievability; return sizes/modes."""
    import io

    from PIL import Image

    out = []
    for i in range(min(n, len(ds))):
        v = ds[i][image_col]
        if isinstance(v, Image.Image):
            img = v
        elif isinstance(v, dict) and v.get("bytes"):
            img = Image.open(io.BytesIO(v["bytes"]))
        elif isinstance(v, dict) and v.get("path"):
            img = Image.open(v["path"])
        else:
            out.append({"index": i, "error": f"unrecognized image field: {type(v).__name__}"})
            continue
        img.load()
        out.append({"index": i, "size": list(img.size), "mode": img.mode})
    return out


@app.function(image=DATASETS_IMAGE, timeout=3600, volumes=VOLUMES)
def pin_refcoco_m():
    import traceback

    import datasets as hfds

    repo = "moondream/refcoco-m"
    rec = {"repo_id": repo}
    try:
        rec["resolved_sha"] = _dataset_sha(repo)
        rec["splits_available"] = list(hfds.get_dataset_split_names(repo))
        ds = hfds.load_dataset(repo, revision=rec["resolved_sha"], split="validation")
        rec["validation_raw_rows"] = len(ds)
        rec["features"] = str(ds.features)

        # keep image column undecoded for speed + structural dump
        img_cols = [k for k, v in ds.features.items() if isinstance(v, hfds.Image)]
        rec["image_columns"] = img_cols
        ds_nodecode = ds
        for c in img_cols:
            ds_nodecode = ds_nodecode.cast_column(c, hfds.Image(decode=False))
        rec["sample_row"] = _sample_row_structure(ds_nodecode[0])

        # flattening evidence: for every list-typed column, total elements across split
        list_cols = [k for k, v in ds.features.items() if isinstance(v, (hfds.Sequence, list))]
        totals = {}
        for c in list_cols:
            col = ds_nodecode[c]  # materializes one column only
            totals[c] = int(sum(len(x) if x is not None else 0 for x in col))
        rec["list_column_totals"] = totals
        rec["flattening_note"] = (
            "Flatten to one (image, referring expression, gt box) instance per annotated "
            "object; candidate flattened counts are the list_column_totals above — the rule "
            "and exact count are fixed from this evidence at Gate A."
        )

        rec["image_retrieval_check"] = _decode_first_images(ds, img_cols[0]) if img_cols else "NO IMAGE COLUMN"
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


@app.function(image=DATASETS_IMAGE, timeout=3600, volumes=VOLUMES)
def pin_pope():
    import traceback
    from collections import Counter

    import datasets as hfds

    repo = "lmms-lab/POPE"
    rec = {"repo_id": repo}
    try:
        rec["resolved_sha"] = _dataset_sha(repo)
        rec["configs"] = list(hfds.get_dataset_config_names(repo))
        per_config = {}
        for cfg in rec["configs"]:
            splits = hfds.get_dataset_split_names(repo, cfg)
            centry = {"splits": {}}
            for sp in splits:
                ds = hfds.load_dataset(repo, cfg, revision=rec["resolved_sha"], split=sp)
                img_cols = [k for k, v in ds.features.items() if isinstance(v, hfds.Image)]
                dsn = ds
                for c in img_cols:
                    dsn = dsn.cast_column(c, hfds.Image(decode=False))
                sentry = {"rows": len(ds), "features": str(ds.features)}
                # split composition: count values of any category-like column
                for cand in ("category", "source", "subset", "split_type"):
                    if cand in ds.features:
                        sentry[f"composition_by_{cand}"] = dict(Counter(dsn[cand]))
                sentry["sample_row"] = _sample_row_structure(dsn[0])
                sentry["image_retrieval_check"] = (
                    _decode_first_images(ds, img_cols[0]) if img_cols else "NO IMAGE COLUMN"
                )
                centry["splits"][sp] = sentry
            per_config[cfg] = centry
        rec["per_config"] = per_config
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


@app.function(image=DATASETS_IMAGE, timeout=3600, volumes=VOLUMES)
def pin_countbenchqa():
    import traceback

    import datasets as hfds

    repo = "vikhyatk/CountBenchQA"
    rec = {"repo_id": repo}
    try:
        rec["resolved_sha"] = _dataset_sha(repo)
        rec["configs"] = list(hfds.get_dataset_config_names(repo))
        per_split = {}
        for sp in hfds.get_dataset_split_names(repo):
            ds = hfds.load_dataset(repo, revision=rec["resolved_sha"], split=sp)
            img_cols = [k for k, v in ds.features.items() if isinstance(v, hfds.Image)]
            dsn = ds
            for c in img_cols:
                dsn = dsn.cast_column(c, hfds.Image(decode=False))
            sentry = {"rows": len(ds), "features": str(ds.features)}
            # exclusion evidence: rows whose image field is missing/empty
            if img_cols:
                col = dsn[img_cols[0]]
                missing = [
                    i for i, v in enumerate(col)
                    if v is None or (isinstance(v, dict) and not v.get("bytes") and not v.get("path"))
                ]
                sentry["missing_image_rows"] = {"count": len(missing), "indices_first_20": missing[:20]}
                sentry["usable_rows_candidate"] = len(ds) - len(missing)
            sentry["sample_row"] = _sample_row_structure(dsn[0])
            sentry["image_retrieval_check"] = (
                _decode_first_images(ds, img_cols[0]) if img_cols else "NO IMAGE COLUMN"
            )
            per_split[sp] = sentry
        rec["per_split"] = per_split
        rec["exclusion_note"] = (
            "Candidate exclusion rule: rows with missing/empty image payload (evidence above). "
            "The exact usable count and final rule are fixed from this evidence at Gate A."
        )
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


@app.local_entrypoint()
def main():
    calls = {
        "refcoco_m": pin_refcoco_m.spawn(),
        "pope": pin_pope.spawn(),
        "countbenchqa": pin_countbenchqa.spawn(),
    }
    results = {}
    for name, call in calls.items():
        try:
            results[name] = call.get()
        except Exception as e:
            results[name] = {"infra_error": repr(e)}
        print(f"{name}: {'ERROR' if 'error' in results[name] or 'infra_error' in results[name] else 'OK'}")

    out = REPO_ROOT / "results" / "phase0" / "dataset_pins.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out}")
