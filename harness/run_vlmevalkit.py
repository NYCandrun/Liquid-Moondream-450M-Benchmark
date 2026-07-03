"""The VLMEvalKit referee lane (calibration appendix).

Runs the four models VLMEvalKit v0.3rc1 supports (SmolVLM2-500M,
llava-onevision-0.5b, Qwen2.5-VL-7B, Moondream2 @ our pinned revision via the
disclosed one-line patch) on POPE + CountBenchQA, using the pinned community
tool EXACTLY as shipped otherwise. Answer judging: the tool's own exact-match
paths (no LLM judge — no API keys provided, so its judge fallback is inert).

Functional-before-full: the first run is the cheapest cell (SmolVLM2 x
CountBenchQA, 491 items); the script verifies the tool produced a score file
before continuing to the remaining cells, strictly serially, one GPU box.

Usage: .venv/bin/modal run harness/run_vlmevalkit.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import GPU_TYPE, HF_CACHE_PATH, VLMEVAL_IMAGE, hf_cache_volume

app = modal.App("edge-vlm-referee")

WORKDIR = HF_CACHE_PATH + "/vlmeval_out"
MODELS = ["SmolVLM2-500M", "llava-onevision-qwen2-0.5b-ov-hf",
          "Qwen2.5-VL-7B-Instruct", "Moondream2"]


@app.function(image=VLMEVAL_IMAGE, gpu=GPU_TYPE, timeout=12 * 3600,
              volumes={HF_CACHE_PATH: hf_cache_volume}, cpu=8.0, memory=32768)
def referee():
    import glob
    import os
    import subprocess

    os.makedirs(WORKDIR, exist_ok=True)
    log = []

    def run_cell(model, data):
        proc = subprocess.run(
            ["python", "run.py", "--model", model, "--data", data,
             "--work-dir", WORKDIR, "--verbose"],
            cwd="/opt/VLMEvalKit", capture_output=True, text=True,
        )
        entry = {"model": model, "data": data, "returncode": proc.returncode,
                 "stderr_tail": proc.stderr[-1500:] if proc.returncode else ""}
        log.append(entry)
        hf_cache_volume.commit()
        return entry

    # functional check first: cheapest cell must produce a score file
    first = run_cell("SmolVLM2-500M", "CountBenchQA")
    score_files = glob.glob(f"{WORKDIR}/**/*acc*.csv", recursive=True) + \
        glob.glob(f"{WORKDIR}/**/*score*", recursive=True) + \
        glob.glob(f"{WORKDIR}/**/*rating*", recursive=True)
    if first["returncode"] != 0 or not score_files:
        return {"fatal": "functional cell failed or produced no score file",
                "log": log, "files": glob.glob(f"{WORKDIR}/**/*", recursive=True)[:50]}

    # remaining cells, strictly serial
    for model in MODELS:
        for data in ("POPE", "CountBenchQA"):
            if model == "SmolVLM2-500M" and data == "CountBenchQA":
                continue
            run_cell(model, data)

    # collect every result artifact the tool wrote
    results = {}
    for f in glob.glob(f"{WORKDIR}/**/*", recursive=True):
        if f.endswith((".csv", ".json")) and os.path.getsize(f) < 2_000_000:
            results[f.replace(WORKDIR + "/", "")] = open(f, errors="replace").read()
    return {"log": log, "results": results}


@app.local_entrypoint()
def main():
    out = referee.remote()
    dest = REPO_ROOT / "calibration" / "vlmevalkit"
    dest.mkdir(parents=True, exist_ok=True)
    if "fatal" in out:
        print("FATAL:", out["fatal"])
        print(json.dumps(out["log"], indent=1)[:3000])
        return
    for rel, content in out["results"].items():
        p = dest / rel.replace("/", "__")
        p.write_text(content)
    (dest / "run_log.json").write_text(json.dumps(out["log"], indent=1))
    print(f"saved {len(out['results'])} referee artifacts to calibration/vlmevalkit/")
    for e in out["log"]:
        print(f"  {e['model']} x {e['data']}: exit {e['returncode']}")
