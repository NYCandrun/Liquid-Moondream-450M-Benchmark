"""Score raw benchmark JSONL files into ``.scored.json`` reports (CLI).

Usage
-----
    python -m scoring.score results/raw/cell_a.jsonl results/raw/cell_b.jsonl
    python scoring/score.py results/raw/cell_a.jsonl --out results/scored/

For each input JSONL file this tool:
  1. reads the raw items (schema below), skipping and counting unreadable lines;
  2. infers the task per file from the items' ``task`` field
     ("grounding" | "pope" | "count"); files that mix tasks are scored as one
     cell per task, so nothing is silently dropped;
  3. computes per-item score fields via :mod:`scoring.metrics` and the cell
     aggregate (which always includes ``n_items`` and the exact input
     filename);
  4. writes ``<input basename minus .jsonl>.scored.json`` into ``--out``
     (created if missing), or next to the input file when ``--out`` is absent.

Raw JSONL item schema (shared by both parts of the project):
    {part, model, task, item_id, image_ref, prompt_id, prompt,
     image_token_budget, image_width, image_height, raw_output, parsed, gt,
     latency_ms, error, meta}

Pure Python 3.10+ standard library only -- no GPU, no third-party deps -- so
scoring runs anywhere. Deterministic: output key order and cell order are
fixed, and no randomness or wall-clock values are used.

Robustness policy (mirrors the project-wide parser contract): every
normalization helper returns ``None`` on unusable input and never raises.
Items whose ground truth cannot be normalized (a dataset bug, not a model
failure) are recorded per-item with ``"invalid_gt": true``, excluded from the
aggregate, and counted in the aggregate's ``n_invalid_gt``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

# Support both invocation styles: ``python -m scoring.score`` (package
# context) and ``python scoring/score.py`` (no package context -- add the
# repo root to sys.path so ``scoring.metrics`` resolves).
if __package__ in (None, ""):  # pragma: no cover - direct-script entry
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scoring import metrics
else:
    from . import metrics

# Tasks this scorer knows how to score, in deterministic report order.
KNOWN_TASKS = ("count", "grounding", "pope")


# ---------------------------------------------------------------------------
# Normalizers: return None on unusable input, NEVER raise.
# ---------------------------------------------------------------------------

def _gt_field(item: dict, key: str) -> Any:
    """The published schema stores gt as a labeled dict (e.g. {"box_xyxy": ...});
    accept both the dict form and a bare value for backward compatibility."""
    gt = item.get("gt")
    if isinstance(gt, dict):
        return gt.get(key)
    return gt


def _first_box(value: Any) -> Any:
    """Grounding parsers return a LIST of boxes; the headline metric scores the
    first predicted box. Accept both list-of-boxes and a bare box."""
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return value[0]
    return value


def normalize_box(value: Any) -> Optional[list[float]]:
    """Normalize a value to a pixel box ``[x1, y1, x2, y2]`` of floats.

    Accepts any length-4 sequence of finite real numbers (bool excluded).
    Returns ``None`` for anything else. Never raises.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: list[float] = []
    for coord in value:
        # bool is a subclass of int; a box of booleans is not a real box.
        if isinstance(coord, bool) or not isinstance(coord, (int, float)):
            return None
        coord_f = float(coord)
        if coord_f != coord_f or coord_f in (float("inf"), float("-inf")):
            return None  # NaN / infinity are not valid pixel coordinates
        out.append(coord_f)
    return out


def normalize_yes_no(value: Any) -> Optional[bool]:
    """Normalize a value to a yes(True)/no(False) boolean.

    Accepts real booleans and the strings "yes"/"no" (case/whitespace
    insensitive), which is how some parsers serialize their verdict.
    Returns ``None`` for anything else. Never raises.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "yes":
            return True
        if lowered == "no":
            return False
    return None


def normalize_int(value: Any) -> Optional[int]:
    """Normalize a value to a non-negative-friendly integer count.

    Accepts ints (bool excluded), integer-valued floats, and strings of an
    integer (e.g. "7"). Returns ``None`` for anything else. Never raises.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else None
    if isinstance(value, str):
        stripped = value.strip()
        # int() would accept "+7"/"-7"; keep those -- exact-match scoring
        # decides correctness, not the normalizer.
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Per-task scoring: raw items -> (per_item records, aggregate dict)
# ---------------------------------------------------------------------------

def _score_grounding(items: list[dict]) -> tuple[list[dict], dict]:
    """Score grounding items; returns (per_item records, aggregate)."""
    per_item: list[dict] = []
    scorable: list[dict] = []  # metrics dicts fed to the aggregate
    n_invalid_gt = 0

    for item in items:
        gt_box = normalize_box(_gt_field(item, "box_xyxy"))
        record: dict[str, Any] = {"item_id": item.get("item_id")}
        if gt_box is None:
            # Dataset bug: ground truth is not a valid box. Record, exclude.
            n_invalid_gt += 1
            record["invalid_gt"] = True
            per_item.append(record)
            continue

        parsed_box = normalize_box(_first_box(item.get("parsed")))
        scores = metrics.grounding_item_scores(parsed_box, gt_box)
        scorable.append(scores)
        record.update({"parsed": parsed_box, "gt": gt_box, **scores})
        per_item.append(record)

    aggregate = metrics.grounding_aggregate(scorable)
    aggregate["n_invalid_gt"] = n_invalid_gt
    return per_item, aggregate


def _score_pope(items: list[dict]) -> tuple[list[dict], dict]:
    """Score POPE yes/no items; returns (per_item records, aggregate)."""
    per_item: list[dict] = []
    pairs: list[tuple[Optional[bool], bool]] = []
    n_invalid_gt = 0

    for item in items:
        gt_yes = normalize_yes_no(_gt_field(item, "answer_yes"))
        record: dict[str, Any] = {"item_id": item.get("item_id")}
        if gt_yes is None:
            n_invalid_gt += 1
            record["invalid_gt"] = True
            per_item.append(record)
            continue

        pred_yes = normalize_yes_no(item.get("parsed"))
        pairs.append((pred_yes, gt_yes))
        record.update(
            {
                "pred_yes": pred_yes,
                "gt_yes": gt_yes,
                "parsed_ok": pred_yes is not None,
                # Unparsed counts as wrong, per project-wide policy.
                "correct": pred_yes is not None and pred_yes == gt_yes,
            }
        )
        per_item.append(record)

    aggregate = metrics.pope_aggregate(pairs)
    aggregate["n_invalid_gt"] = n_invalid_gt
    return per_item, aggregate


def _score_count(items: list[dict]) -> tuple[list[dict], dict]:
    """Score counting items; returns (per_item records, aggregate)."""
    per_item: list[dict] = []
    pairs: list[tuple[Optional[int], int]] = []
    n_invalid_gt = 0

    for item in items:
        gt_int = normalize_int(_gt_field(item, "number"))
        record: dict[str, Any] = {"item_id": item.get("item_id")}
        if gt_int is None:
            n_invalid_gt += 1
            record["invalid_gt"] = True
            per_item.append(record)
            continue

        pred_int = normalize_int(item.get("parsed"))
        pairs.append((pred_int, gt_int))
        record.update(
            {
                "pred": pred_int,
                "gt": gt_int,
                "parsed_ok": pred_int is not None,
                # Unparsed counts as wrong, per project-wide policy.
                "correct": pred_int is not None and pred_int == gt_int,
            }
        )
        per_item.append(record)

    aggregate = metrics.count_aggregate(pairs)
    aggregate["n_invalid_gt"] = n_invalid_gt
    return per_item, aggregate


_TASK_SCORERS = {
    "grounding": _score_grounding,
    "pope": _score_pope,
    "count": _score_count,
}


# ---------------------------------------------------------------------------
# File-level scoring
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> tuple[list[dict], int]:
    """Read a JSONL file; returns (item dicts, count of unusable lines).

    Blank lines are ignored. Lines that fail to decode as JSON, or decode to
    something other than an object, are counted as bad and skipped -- the
    scorer never raises on malformed input lines.
    """
    items: list[dict] = []
    n_bad = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if isinstance(obj, dict):
                items.append(obj)
            else:
                n_bad += 1
    return items, n_bad


def score_file(input_path: str) -> dict:
    """Score one raw JSONL file into a report dict (one cell per task found).

    The task is inferred per file from the items' ``task`` field. A raw file
    normally holds exactly one cell (one task); if tasks are mixed, each task
    group becomes its own cell so nothing is silently dropped. Items with an
    unknown/missing task are counted in ``n_unknown_task`` and skipped.
    """
    items, n_bad_lines = load_jsonl(input_path)

    # Group items by task, preserving input order within each group.
    by_task: dict[str, list[dict]] = {}
    n_unknown_task = 0
    for item in items:
        task = item.get("task")
        if task in _TASK_SCORERS:
            by_task.setdefault(task, []).append(item)
        else:
            n_unknown_task += 1

    cells = []
    for task in KNOWN_TASKS:  # fixed order => deterministic output
        if task not in by_task:
            continue
        per_item, aggregate = _TASK_SCORERS[task](by_task[task])
        # Contract: every aggregate carries n_items (from metrics) and the
        # exact input filename as given on the command line.
        aggregate["input_file"] = input_path
        cells.append({"task": task, "aggregate": aggregate, "per_item": per_item})

    return {
        "input_file": input_path,
        "n_lines_scored": sum(len(v) for v in by_task.values()),
        "n_bad_lines": n_bad_lines,
        "n_unknown_task": n_unknown_task,
        "cells": cells,
    }


def output_path_for(input_path: str, out_dir: Optional[str]) -> str:
    """Compute the ``.scored.json`` path for an input file.

    ``foo.jsonl -> foo.scored.json`` (a trailing ``.jsonl`` is replaced;
    otherwise ``.scored.json`` is appended). Written into ``out_dir`` when
    given, else next to the input file.
    """
    base = os.path.basename(input_path)
    if base.endswith(".jsonl"):
        base = base[: -len(".jsonl")]
    scored_name = base + ".scored.json"
    directory = out_dir if out_dir is not None else (os.path.dirname(input_path) or ".")
    return os.path.join(directory, scored_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = all files scored)."""
    parser = argparse.ArgumentParser(
        prog="scoring.score",
        description=(
            "Score raw benchmark JSONL files (grounding / pope / count) into "
            ".scored.json reports. Pure stdlib; no GPU."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="RAW_JSONL",
        help="one or more raw JSONL run files to score",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help=(
            "directory for the .scored.json outputs (created if missing); "
            "defaults to each input file's own directory"
        ),
    )
    args = parser.parse_args(argv)

    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)

    exit_code = 0
    for input_path in args.inputs:
        if not os.path.isfile(input_path):
            print(f"error: no such file: {input_path}", file=sys.stderr)
            exit_code = 1
            continue

        report = score_file(input_path)
        out_path = output_path_for(input_path, args.out)
        with open(out_path, "w", encoding="utf-8") as handle:
            # indent for human diffing; key order is insertion order, which
            # is fixed by construction -> byte-stable, deterministic output.
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(out_path)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
