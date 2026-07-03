"""RefCOCO-M grounding referee (change #15, grounding lane).

Runs the re-pinned VLMEvalKit's RefCOCO grounding scorer (Precision@1 at
IoU>=0.5 — IDENTICAL metric to our Acc@0.5) on OUR EXACT RefCOCO-M items, for
all five models. Because our grounding set is RefCOCO-M (moondream/refcoco-m),
NOT VLMEvalKit's built-in RefCOCO, we materialise a custom TSV from
harness.data.load_items("grounding") — the same 2,080 flattened instances with
byte-identical GT boxes — and register it as a RefCOCODataset variant
(DATASET_FILES patch in the images).

IMPORTANT INTERPRETATION (disclosed): this is a DIFFERENT-PROTOCOL cross-check,
not like-for-like. VLMEvalKit sends ONE uniform grounding prompt to every model
(the `question` column) and parses boxes with ONE generic parser (brackets or
first-4-numbers; auto-detects 0-1 / 0-1000 / absolute scale). Our main-table
grounding uses each model's NATIVE prompt + native box parser. So the referee
measures "each model under VLMEvalKit's uniform grounding protocol"; agreement
is meaningful, disagreement localises to prompt/format handling.

Usage:
  .venv/bin/modal run harness/run_vlmevalkit2_grounding.py::build   # build TSVs on volume
  .venv/bin/modal run harness/run_vlmevalkit2_grounding.py::smoke   # 1 model/img on 24 items
  .venv/bin/modal run harness/run_vlmevalkit2_grounding.py::full    # all 5 models, 2080 items
"""

import base64
import glob
import io
import json
import os
import re
import subprocess
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

app = modal.App("edge-vlm-referee2-grounding")
WORKDIR = HF_CACHE_PATH + "/vlmeval2_out"
LMUDATA = HF_CACHE_PATH + "/lmudata"
VOL = {HF_CACHE_PATH: hf_cache_volume}
# Auto-restart on Modal worker PREEMPTION (spot-GPU reclaim, observed even at 3
# concurrent containers). VLMEvalKit resumes from its per-cell xlsx on the
# volume, so a restarted container re-uses completed predictions.
_RETRIES = modal.Retries(max_retries=4, initial_delay=20.0, backoff_coefficient=2.0)

# Uniform grounding prompt (VLMEvalKit-style; one prompt for every model — the
# independent-instrument protocol). The tool's parser reads [x1,y1,x2,y2] or the
# first four numbers and auto-detects the coordinate scale.
GROUND_PROMPT = ("Please provide the bounding box coordinate of the region this "
                 "sentence describes: {expr}")

# Qwen2.5-VL-7B-Instruct dropped from scope 2026-07-03 (runtime bottleneck; edge-VLM
# framing tightened to <=2B params). Excluded from the grounding referee lane.
TF457_MODELS = ["SmolVLM2-500M", "llava-onevision-qwen2-0.5b-ov-hf",
                "Moondream2"]
TF5_MODELS = ["LFM2.5-VL-450M"]
SMOKE_N = 24


def _make_tsv(dataset_name, limit=0):
    import pandas as pd

    from harness import data as hdata

    items = hdata.load_items("grounding")
    if limit:
        items = items[:limit]
    rows = []
    for i, it in enumerate(items):
        buf = io.BytesIO()
        it["image"].convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        x1, y1, x2, y2 = it["gt"]["box_xyxy"]
        rows.append({
            "index": i,
            "item_id": it["item_id"],
            "question": GROUND_PROMPT.format(expr=it["expression"]),
            "image": b64,
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "width": it["image_width"], "height": it["image_height"],
            "split": "RefCOCO-M", "category": it["gt"]["category"],
        })
    df = pd.DataFrame(rows)
    os.makedirs(LMUDATA, exist_ok=True)
    out = f"{LMUDATA}/{dataset_name}.tsv"
    df.to_csv(out, sep="\t", index=False)
    return {"dataset": dataset_name, "n": len(df), "path": out}


@app.function(image=VLMEVAL2_TF457_IMAGE, timeout=3600, volumes=VOL, cpu=8.0, memory=32768)
def build_tsvs():
    full = _make_tsv("RefCOCO-M")
    smoke = _make_tsv("RefCOCO-M-smoke", limit=SMOKE_N)
    hf_cache_volume.commit()
    return {"full": full, "smoke": smoke}


def _run_cells(models, data):
    os.makedirs(WORKDIR, exist_ok=True)
    log = []
    for model in models:
        proc = subprocess.run(
            ["python", "run.py", "--model", model, "--data", data,
             "--work-dir", WORKDIR, "--verbose"],
            cwd="/opt/VLMEvalKit", capture_output=True, text=True,
        )
        combined = proc.stdout + "\n" + proc.stderr
        os.makedirs(f"{WORKDIR}/logs", exist_ok=True)
        with open(f"{WORKDIR}/logs/{model}__{data}.log", "w") as fh:
            fh.write(combined)
        prec = re.search(r"Precision@1[^0-9]*([0-9.]+)", combined)
        log.append({
            "model": model, "data": data, "returncode": proc.returncode,
            "precision_at_1_from_log": float(prec.group(1)) if prec else None,
            "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:],
        })
        hf_cache_volume.commit()
    return log


@app.function(image=VLMEVAL2_TF457_IMAGE, gpu=GPU_TYPE, timeout=12 * 3600,
              volumes=VOL, cpu=8.0, memory=32768, retries=_RETRIES)
def ground_tf457(data, models=None):
    return _run_cells(models or TF457_MODELS, data)


@app.function(image=VLMEVAL2_TF5_IMAGE, gpu=GPU_TYPE, timeout=12 * 3600,
              volumes=VOL, cpu=8.0, memory=32768, retries=_RETRIES)
def ground_tf5(data, models=None):
    return _run_cells(models or TF5_MODELS, data)


@app.function(image=VLMEVAL2_TF457_IMAGE, volumes=VOL, timeout=1200)
def collect():
    results = {}
    for f in glob.glob(f"{WORKDIR}/**/*", recursive=True):
        if f.endswith((".csv", ".json")) and os.path.getsize(f) < 5_000_000:
            results[f.replace(WORKDIR + "/", "")] = open(f, errors="replace").read()
    return results


def _save(log, results, tag):
    dest = REPO_ROOT / "calibration" / "vlmevalkit2"
    dest.mkdir(parents=True, exist_ok=True)
    for rel, content in results.items():
        (dest / rel.replace("/", "__")).write_text(content)
    (dest / f"grounding_run_log{tag}.json").write_text(json.dumps(log, indent=1))
    for e in log:
        print(f"  {e['model']} x {e['data']}: exit {e['returncode']} "
              f"P@1={e['precision_at_1_from_log']}")


@app.local_entrypoint()
def build():
    print(json.dumps(build_tsvs.remote(), indent=1))


@app.local_entrypoint()
def smoke():
    # one model per image on the 24-item TSV — proves the custom dataset loads,
    # each model produces parseable boxes, and a Precision@1 is scored.
    log = ground_tf5.remote("RefCOCO-M-smoke", models=["LFM2.5-VL-450M"])
    log += ground_tf457.remote("RefCOCO-M-smoke", models=["SmolVLM2-500M"])
    _save(log, collect.remote(), ".smoke")
    print("GROUNDING_SMOKE_OK" if all(e["returncode"] == 0 for e in log) else "GROUNDING_SMOKE_FAIL")


@app.local_entrypoint()
def full():
    # Fan out to ONE container per model (5 concurrent: LFM on tf5, the four on
    # tf457) — combined with the referee lane this peaks at 6 GPU containers, a
    # deliberate test of a higher concurrency ceiling. Failure-tolerant: each
    # call is gathered under try/except so a preemption doesn't lose the rest,
    # and collect() salvages whatever committed to the volume.
    calls = [("LFM2.5-VL-450M", ground_tf5.spawn("RefCOCO-M", models=["LFM2.5-VL-450M"]))]
    for m in TF457_MODELS:
        calls.append((m, ground_tf457.spawn("RefCOCO-M", models=[m])))
    log = []
    for name, c in calls:
        try:
            log += list(c.get())
        except Exception as exc:
            print(f"[{name} FAILED] {exc!r} — partial cells remain on the volume")
    _save(log, collect.remote(), "")
    print(f"grounding referee done: {sum(e['returncode'] == 0 for e in log)}/{len(log)} cells")
