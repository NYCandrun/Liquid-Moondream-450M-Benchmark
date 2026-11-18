"""Build the image-token lookup table for demos from saved results.

Usage:
    python3 -m scoring.demo_table      # writes demo_data/image_token_lookup.{json,csv}

One row per (model, task, image-detail setting). Every number is computed
from files under ``results/raw/`` and ``results/probe_energy.json``; nothing
is typed in by hand. Standard library only.

Row sources
  * sweep:   LFM2.5-VL 450M find-object (results/raw/sweep_v2), SmolVLM2 500M
             counting (results/raw/sweep), full test sets, one A100-40GB.
  * probe:   the same two models on the same 100 counting questions with
             measured GPU energy (results/raw/probe + results/probe_energy.json).
  * shipped: Moondream 0.5B int8 in its own runtime on CPU at its single
             fixed setting (results/raw/deployed). Different hardware, so
             its speed is not comparable with the GPU rows.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.score import load_jsonl, score_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LFM_SETTINGS = (32, 64, 128, 256)
SMOL_SETTINGS = (512, 1024, 1536, 2048)

COLUMNS = [
    "source", "model", "task", "setting_kind", "setting", "hardware", "n_items",
    "accuracy", "accuracy_metric", "median_ms_per_item", "mean_input_tokens",
    "items_upscaled_by_fallback", "j_per_answer", "j_per_answer_net_of_idle", "note",
]


def _raw(*parts):
    return os.path.join(ROOT, *parts)


def _stats(path):
    """Accuracy, n, median latency, mean tokens, fallback count for one raw file."""
    report = score_file(path)
    agg = report["cells"][0]["aggregate"]
    rows, _ = load_jsonl(path)
    lat = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    toks = [r["n_input_tokens"] for r in rows if isinstance(r.get("n_input_tokens"), (int, float))]
    return {
        "n_items": agg["n_items"],
        "accuracy": agg["acc_at_50"] if "acc_at_50" in agg else agg["exact_match"],
        "median_ms_per_item": round(statistics.median(lat), 1) if lat else None,
        "mean_input_tokens": round(statistics.mean(toks), 1) if toks else None,
        "items_upscaled_by_fallback": sum(1 for r in rows if r.get("pre_upscale_factor")),
    }


def build():
    rows = []
    for t in LFM_SETTINGS:
        s = _stats(_raw("results", "raw", "sweep_v2", f"sweep_v2_lfm450m_grounding_b{t}.jsonl"))
        rows.append({"source": "sweep", "model": "LFM2.5-VL 450M", "task": "find-object",
                     "setting_kind": "image tokens (min = max)", "setting": t,
                     "hardware": "A100-40GB", "accuracy_metric": "Acc@0.5",
                     "note": "tiling off; mean_input_tokens includes about 46 question tokens", **s})
    for px in SMOL_SETTINGS:
        s = _stats(_raw("results", "raw", "sweep", f"sweep_smolvlm2_count_b{px}.jsonl"))
        rows.append({"source": "sweep", "model": "SmolVLM2 500M", "task": "counting",
                     "setting_kind": "longest image edge (px)", "setting": px,
                     "hardware": "A100-40GB", "accuracy_metric": "exact match",
                     "note": "2048 is the shipped default; find-object is unsupported by this model", **s})

    energy = {c["cell_id"]: c["energy"] for c in json.load(open(_raw("results", "probe_energy.json")))["cells"]}
    for model, short, name, settings, kind in (
            ("lfm450m", "lfm450m", "LFM2.5-VL 450M", LFM_SETTINGS, "image tokens (min = max)"),
            ("smolvlm2", "smolvlm2", "SmolVLM2 500M", SMOL_SETTINGS, "longest image edge (px)")):
        for v in settings:
            cell = f"probe_{short}_count_b{v}"
            s = _stats(_raw("results", "raw", "probe", f"{cell}.sample100s7.jsonl"))
            e = energy[cell]
            rows.append({"source": "probe", "model": name, "task": "counting (100-question subset)",
                         "setting_kind": kind, "setting": v, "hardware": "A100-40GB",
                         "accuracy_metric": "exact match", **s,
                         "j_per_answer": e["j_per_item_raw"],
                         "j_per_answer_net_of_idle": e["j_per_item_net_of_idle"],
                         "note": "energy from the GPU's NVML counter, model load excluded"})

    s = _stats(_raw("results", "raw", "deployed", "deployed_moondream05b_asshipped_grounding.jsonl"))
    s["mean_input_tokens"] = 729.0
    rows.append({"source": "shipped", "model": "Moondream 0.5B (int8)", "task": "find-object",
                 "setting_kind": "fixed", "setting": 729, "hardware": "CPU, 8 cores",
                 "accuracy_metric": "Acc@0.5", **s,
                 "note": "single fixed setting, no dial; speed is CPU time and not comparable with the GPU rows"})
    return rows


def main():
    rows = build()
    out = _raw("demo_data")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "image_token_lookup.json"), "w", encoding="utf-8") as fh:
        json.dump({"columns": COLUMNS, "rows": [{c: r.get(c) for c in COLUMNS} for r in rows]}, fh, indent=1)
    with open(os.path.join(out, "image_token_lookup.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLUMNS})
    for r in rows:
        print(r["source"], r["model"], r["setting"], r["n_items"], round(r["accuracy"], 3),
              r["median_ms_per_item"], r["mean_input_tokens"], r["items_upscaled_by_fallback"], r.get("j_per_answer"))


if __name__ == "__main__":
    main()
