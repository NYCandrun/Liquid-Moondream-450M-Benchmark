"""Phase 2 human-readable dumps: one markdown file per sample cell.

Reads sample JSONLs from results/raw/{part}/, applies the per-model parser and
the metric, and writes results/phase2/dumps/<cell>.md with, per item:
image id, prompt, raw model output, parsed value, ground truth, computed score.
GPU-free; pure stdlib + this repo's parsers/ and scoring/.

Usage: python3 scoring/make_dumps.py [glob ...]
       (default: results/raw/bench/*.sample*.jsonl)
"""

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parsers.answers import parse_count, parse_yes_no
from parsers.boxes import (
    parse_internvl,
    parse_lfm,
    parse_llava_ov,
    parse_moondream_detect,
    parse_qwen25vl,
    parse_smolvlm2,
)
from scoring.metrics import grounding_item_scores

BOX_PARSERS = {
    "lfm450m": parse_lfm,
    "lfm450m_gguf": parse_lfm,  # deployed small form speaks the same format
    "qwen25vl": parse_qwen25vl,
    "internvl35": parse_internvl,
    "smolvlm2": parse_smolvlm2,
    "llava_ov": parse_llava_ov,
}
MOONDREAM_MODELS = {"moondream2", "moondream05b_asshipped"}


def parse_moondream_raw(raw, w, h):
    """moondream detect() output is JSON-dumped by the runner; decode then parse."""
    try:
        objects = json.loads(raw)
    except Exception:
        return None
    return parse_moondream_detect(objects, w, h)


def parse_item(row):
    model, task = row["model"], row["task"]
    raw = row.get("raw_output")
    if raw is None:
        return None
    if task == "grounding":
        w, h = row["image_width"], row["image_height"]
        fn = parse_moondream_raw if model in MOONDREAM_MODELS else BOX_PARSERS[model]
        return fn(raw, w, h)
    if task == "pope":
        return parse_yes_no(raw)
    if task == "count":
        return parse_count(raw)
    raise ValueError(task)


def score_item(row, parsed):
    task = row["task"]
    if task == "grounding":
        # headline convention: first predicted box vs gt
        box = parsed[0] if parsed else None
        s = grounding_item_scores(box, row["gt"]["box_xyxy"])
        return s, (s["hit_at_50"])
    if task == "pope":
        gt = row["gt"]["answer_yes"]
        ok = parsed is not None and parsed == gt
        return {"pred_yes": parsed, "gt_yes": gt}, ok
    if task == "count":
        gt = row["gt"]["number"]
        ok = parsed is not None and parsed == gt
        return {"pred": parsed, "gt": gt}, ok
    raise ValueError(task)


def dump_cell(path, outdir):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        return None
    cell = Path(path).stem
    n_parsed = n_ok = 0
    lines = [f"# Phase 2 dump — {cell}", "", f"source: `{path}`  items: {len(rows)}", ""]
    for r in rows:
        parsed = parse_item(r)
        scores, ok = score_item(r, parsed)
        if parsed is not None:
            n_parsed += 1
        if ok:
            n_ok += 1
        lines += [
            "---",
            f"**item** `{r['item_id']}`  image `{r['image_ref']}`  ({r['image_width']}x{r['image_height']})",
            f"**prompt:** {r['prompt']!r}",
            f"**raw output:** {str(r.get('raw_output'))!r}",
            f"**parsed:** `{parsed}`",
            f"**gt:** `{r['gt']}`",
            f"**score:** `{scores}`  {'✅' if ok else '❌'}",
            f"**latency:** {r.get('latency_ms')} ms" + (f"  **error:** {r['error']}" if r.get("error") else ""),
            "",
        ]
    header = (
        f"\n**cell summary: parse {n_parsed}/{len(rows)}  "
        f"correct {n_ok}/{len(rows)}**\n"
    )
    lines.insert(3, header)
    out = Path(outdir) / f"{cell}.md"
    out.write_text("\n".join(lines))
    return {"cell": cell, "n": len(rows), "parsed": n_parsed, "correct": n_ok, "dump": str(out)}


def main():
    patterns = sys.argv[1:] or [str(REPO_ROOT / "results/raw/bench/*.sample*.jsonl")]
    outdir = REPO_ROOT / "results" / "phase2" / "dumps"
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            s = dump_cell(path, outdir)
            if s:
                summaries.append(s)
                print(f"{s['cell']}: parse {s['parsed']}/{s['n']}  correct {s['correct']}/{s['n']}")
    (outdir.parent / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
