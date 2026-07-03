"""Re-pinned VLMEvalKit referee lane (change #15): ONE newer tool version
(19c0e386, which ships liquid.py) across ALL FIVE models — the four originals
plus LFM 450M (LiquidAI's own eval harness per their card). POPE + CountBenchQA
here; RefCOCO-M grounding is added by run_vlmevalkit2_grounding once this lands.

Model -> image routing (one tool commit, per-model transformers to match our
main-harness lanes and the models' requirements):
  LFM2.5-VL-450M              -> VLMEVAL2_TF5_IMAGE  (transformers 5.12.1)
  SmolVLM2 / LLaVA-OV /
  Qwen2.5-VL / Moondream2     -> VLMEVAL2_TF457_IMAGE (transformers 4.57.1)

Drift verified BYTE-clean vs the old pin for POPE/count scoring + all four
existing adapters, so the four reproduce and LFM is added. Functional-before-
full: a cheap cell per image is checked before the rest.

Usage: .venv/bin/modal run harness/run_vlmevalkit2.py
"""

import glob
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

app = modal.App("edge-vlm-referee2")
WORKDIR = HF_CACHE_PATH + "/vlmeval2_out"
VOL = {HF_CACHE_PATH: hf_cache_volume}

TF457_MODELS = ["SmolVLM2-500M", "llava-onevision-qwen2-0.5b-ov-hf",
                "Qwen2.5-VL-7B-Instruct", "Moondream2"]
TF5_MODELS = ["LFM2.5-VL-450M"]
DATASETS = ["POPE", "CountBenchQA"]


def _run_cells(models, datasets):
    """Run each (model, data) through VLMEvalKit's own run.py, serially, in this
    container. Returns a list of per-cell log entries + the artifact contents."""
    os.makedirs(WORKDIR, exist_ok=True)
    log = []
    for model in models:
        for data in datasets:
            proc = subprocess.run(
                ["python", "run.py", "--model", model, "--data", data,
                 "--work-dir", WORKDIR, "--verbose"],
                cwd="/opt/VLMEvalKit", capture_output=True, text=True,
            )
            combined = proc.stdout + "\n" + proc.stderr
            os.makedirs(f"{WORKDIR}/logs", exist_ok=True)
            with open(f"{WORKDIR}/logs/{model}__{data}.log", "w") as fh:
                fh.write(combined)
            acc = re.search(r'"accuracy"\s*:\s*([0-9.]+)', combined)
            log.append({
                "model": model, "data": data, "returncode": proc.returncode,
                "accuracy_from_log": float(acc.group(1)) if acc else None,
                "eval_failed": "combination failed" in combined,
                "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:],
            })
            hf_cache_volume.commit()
    return log


@app.function(image=VLMEVAL2_TF457_IMAGE, gpu=GPU_TYPE, timeout=12 * 3600,
              volumes=VOL, cpu=8.0, memory=32768)
def referee_tf457(datasets=None):
    return _run_cells(TF457_MODELS, datasets or DATASETS)


@app.function(image=VLMEVAL2_TF5_IMAGE, gpu=GPU_TYPE, timeout=12 * 3600,
              volumes=VOL, cpu=8.0, memory=32768)
def referee_tf5(datasets=None):
    return _run_cells(TF5_MODELS, datasets or DATASETS)


def _collect():
    """Read every artifact the tool wrote to the shared volume workdir."""
    results = {}
    for f in glob.glob(f"{WORKDIR}/**/*", recursive=True):
        if f.endswith((".csv", ".json")) and os.path.getsize(f) < 2_000_000:
            results[f.replace(WORKDIR + "/", "")] = open(f, errors="replace").read()
    return results


@app.function(image=VLMEVAL2_TF457_IMAGE, volumes=VOL, timeout=1200)
def collect():
    return _collect()


@app.local_entrypoint()
def main():
    # Two containers (LFM on tf5 + the four on tf457, one model at a time in
    # each) — within the 3-container cap. FAILURE-TOLERANT: each lane's result
    # is gathered under try/except so one lane dying (e.g. a preemption) neither
    # tears down the other nor loses the per-cell artifacts already committed to
    # the volume; collect() then salvages everything present.
    call5 = referee_tf5.spawn()
    call457 = referee_tf457.spawn()
    log = []
    for name, call in (("tf5/LFM", call5), ("tf457/4-models", call457)):
        try:
            log += list(call.get())
        except Exception as exc:
            print(f"[{name} lane FAILED] {exc!r} — partial cells remain on the volume")

    results = collect.remote()
    dest = REPO_ROOT / "calibration" / "vlmevalkit2"
    dest.mkdir(parents=True, exist_ok=True)
    for rel, content in results.items():
        (dest / rel.replace("/", "__")).write_text(content)
    (dest / "run_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {len(results)} artifacts to calibration/vlmevalkit2/")
    for e in log:
        print(f"  {e['model']} x {e['data']}: exit {e['returncode']} "
              f"acc_log={e['accuracy_from_log']} eval_failed={e['eval_failed']}")
