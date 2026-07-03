"""Frozen task metrics for the edge-VLM benchmark (bench + sweep, one harness).

Pure Python 3.10+ standard library only -- no numpy/torch/PIL -- so scoring
runs anywhere. All functions here are deterministic and side-effect free.

These metric definitions are NOT provisional: unlike the model-output parsers,
they encode task-level math that is independent of any model's output format.

Conventions
-----------
* Boxes are pixel-space ``[x1, y1, x2, y2]``. A box with non-positive width or
  height ("degenerate") has zero area and an IoU of 0.0 against anything.
* "Unparsed" means the parser returned ``None`` for an item. Unparsed items
  ALWAYS count against headline accuracy metrics (unparsed = wrong / miss).
* Aggregates over an empty item list never raise: rates computed over all
  items default to 0.0, while statistics over a conditional subset (e.g.
  mean IoU over items where a box was found) are ``None`` when that subset
  is empty, so they cannot be mistaken for a real 0.0 score.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

# Type alias for readability; a box is any length-4 numeric sequence.
Box = Sequence[float]

# IoU threshold for the headline grounding metric (acc@0.5).
HIT_IOU_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def _box_area(box: Box) -> float:
    """Area of a pixel box ``[x1, y1, x2, y2]``; degenerate boxes -> 0.0."""
    width = float(box[2]) - float(box[0])
    height = float(box[3]) - float(box[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two pixel boxes ``[x1, y1, x2, y2]``.

    Degenerate boxes (zero or negative width/height) have zero area, so any
    pairing that involves one returns 0.0 -- including two identical
    degenerate boxes. Boxes that merely touch along an edge or corner have a
    zero-area intersection and therefore an IoU of 0.0.
    """
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_w = min(float(box_a[2]), float(box_b[2])) - max(float(box_a[0]), float(box_b[0]))
    inter_h = min(float(box_a[3]), float(box_b[3])) - max(float(box_a[1]), float(box_b[1]))
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0

    intersection = inter_w * inter_h
    union = area_a + area_b - intersection
    # union > 0 is guaranteed here because both areas are positive.
    return intersection / union


# ---------------------------------------------------------------------------
# Grounding (referring-expression box localization)
# ---------------------------------------------------------------------------

def grounding_item_scores(parsed_box: Optional[Box], gt_box: Box) -> dict:
    """Per-item grounding scores for one (predicted box, ground-truth box) pair.

    Parameters
    ----------
    parsed_box : the parser's box, or ``None`` if the model output could not
        be parsed into a box (unparsed = automatic miss).
    gt_box : the ground-truth pixel box ``[x1, y1, x2, y2]``.

    Returns a dict with:
    * ``iou``           -- IoU with the ground truth (0.0 when unparsed).
    * ``hit_at_50``     -- True iff ``iou >= 0.5`` (the threshold value itself
                           counts as a hit).
    * ``center_in_box`` -- True iff the CENTER of the predicted box lies inside
                           the ground-truth box; a center exactly on the
                           boundary counts as inside. False when unparsed.
    * ``area_ratio``    -- predicted area / ground-truth area, or ``None`` when
                           unparsed (or when the ground-truth box is degenerate,
                           to avoid division by zero).
    * ``miss_kind``     -- ``None`` if hit; ``"not_found"`` if unparsed;
                           ``"found_but_loose"`` if a box was parsed but its
                           IoU is below 0.5.
    """
    if parsed_box is None:
        return {
            "iou": 0.0,
            "hit_at_50": False,
            "center_in_box": False,
            "area_ratio": None,
            "miss_kind": "not_found",
        }

    box_iou = iou(parsed_box, gt_box)
    hit = box_iou >= HIT_IOU_THRESHOLD

    # Center-in-box: closed interval on both axes, so a center that falls
    # exactly on the ground-truth boundary counts as inside.
    center_x = (float(parsed_box[0]) + float(parsed_box[2])) / 2.0
    center_y = (float(parsed_box[1]) + float(parsed_box[3])) / 2.0
    center_in = (
        float(gt_box[0]) <= center_x <= float(gt_box[2])
        and float(gt_box[1]) <= center_y <= float(gt_box[3])
    )

    gt_area = _box_area(gt_box)
    area_ratio = (_box_area(parsed_box) / gt_area) if gt_area > 0.0 else None

    return {
        "iou": box_iou,
        "hit_at_50": hit,
        "center_in_box": center_in,
        "area_ratio": area_ratio,
        "miss_kind": None if hit else "found_but_loose",
    }


def grounding_aggregate(item_scores: Iterable[dict]) -> dict:
    """Aggregate grounding metrics over per-item dicts from grounding_item_scores.

    Headline metric
    ---------------
    * ``acc_at_50`` -- fraction of ALL items (unparsed included) whose IoU is
      >= 0.5. Unparsed items count as misses.

    Extensions (beyond the headline metric; labeled as such per spec)
    -----------------------------------------------------------------
    * ``center_in_box_recall``    -- EXTENSION: fraction of ALL items whose
      predicted-box center falls inside the ground-truth box. A localization
      signal that is lenient about box extent.
    * ``mean_iou_on_found``       -- EXTENSION: mean IoU over items where a box
      was parsed ("found"); ``None`` if no boxes were parsed.
    * ``mean_area_ratio_on_found``-- EXTENSION: mean pred/gt area ratio over
      found items with a defined ratio; ``None`` if there are none. Diagnoses
      systematic over/under-sizing of predicted boxes.
    * ``miss_decomposition``      -- EXTENSION: splits the miss mass over ALL
      items into ``not_found_rate`` (unparsed) and ``found_but_loose_rate``
      (parsed but IoU < 0.5). By construction,
      ``acc_at_50 + not_found_rate + found_but_loose_rate == 1.0``.

    Bookkeeping
    -----------
    * ``parse_rate`` -- fraction of ALL items with a parsed box
      (== 1 - not_found_rate).
    * ``n_items``    -- number of items aggregated.

    Empty input returns n_items=0, all rates 0.0, and conditional means None.
    """
    scores = list(item_scores)
    n = len(scores)
    if n == 0:
        return {
            "n_items": 0,
            "acc_at_50": 0.0,
            "center_in_box_recall": 0.0,
            "mean_iou_on_found": None,
            "mean_area_ratio_on_found": None,
            "miss_decomposition": {"not_found_rate": 0.0, "found_but_loose_rate": 0.0},
            "parse_rate": 0.0,
        }

    hits = sum(1 for s in scores if s["hit_at_50"])
    centers = sum(1 for s in scores if s["center_in_box"])
    not_found = sum(1 for s in scores if s["miss_kind"] == "not_found")
    loose = sum(1 for s in scores if s["miss_kind"] == "found_but_loose")

    found = [s for s in scores if s["miss_kind"] != "not_found"]
    found_ious = [s["iou"] for s in found]
    found_ratios = [s["area_ratio"] for s in found if s["area_ratio"] is not None]

    return {
        "n_items": n,
        "acc_at_50": hits / n,
        "center_in_box_recall": centers / n,
        "mean_iou_on_found": (sum(found_ious) / len(found_ious)) if found_ious else None,
        "mean_area_ratio_on_found": (
            (sum(found_ratios) / len(found_ratios)) if found_ratios else None
        ),
        "miss_decomposition": {
            "not_found_rate": not_found / n,
            "found_but_loose_rate": loose / n,
        },
        "parse_rate": len(found) / n,
    }


# ---------------------------------------------------------------------------
# POPE (yes/no object-hallucination probing)
# ---------------------------------------------------------------------------

def pope_aggregate(items: Iterable[tuple[Optional[bool], bool]]) -> dict:
    """Aggregate POPE metrics over (pred_yes, gt_yes) pairs.

    ``pred_yes`` is ``True``/``False`` for a parsed yes/no answer and ``None``
    for an unparsed answer. Unparsed answers count as WRONG for accuracy, and
    as "did not say yes" for F1 (so a ground-truth-yes item with an unparsed
    answer is a false negative; a ground-truth-no item with an unparsed answer
    contributes nothing to the yes-positive F1, matching its true-negative
    role).

    Returns:
    * ``accuracy``   -- correct / ALL items (unparsed = wrong).
    * ``f1``         -- F1 with "yes" as the positive class; 0.0 when
      precision + recall is 0 (no division by zero).
    * ``precision``, ``recall`` -- EXTENSION: the F1 components, reported for
      diagnosis (yes as positive class, same unparsed handling as f1).
    * ``yes_ratio``  -- share of PARSED answers that are "yes"; ``None`` when
      nothing parsed (so it cannot be mistaken for "all no").
    * ``parse_rate`` -- parsed / ALL items.
    * ``n_items``    -- number of items aggregated.

    Empty input returns n_items=0, accuracy/f1/parse_rate 0.0, yes_ratio None.
    """
    pairs = list(items)
    n = len(pairs)

    true_pos = false_pos = false_neg = 0
    correct = 0
    parsed = 0
    yes_answers = 0

    for pred, gt in pairs:
        gt_yes = bool(gt)
        pred_yes = bool(pred) if pred is not None else None

        if pred_yes is not None:
            parsed += 1
            if pred_yes:
                yes_answers += 1
            if pred_yes == gt_yes:
                correct += 1

        # Confusion counts with "yes" as the positive class. An unparsed
        # answer never claims "yes", so it can only produce a false negative.
        if pred_yes is True and gt_yes:
            true_pos += 1
        elif pred_yes is True and not gt_yes:
            false_pos += 1
        elif pred_yes is not True and gt_yes:
            false_neg += 1

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "n_items": n,
        "accuracy": (correct / n) if n > 0 else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "yes_ratio": (yes_answers / parsed) if parsed > 0 else None,
        "parse_rate": (parsed / n) if n > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def count_aggregate(items: Iterable[tuple[Optional[int], int]]) -> dict:
    """Aggregate counting metrics over (pred_int, gt_int) pairs.

    ``pred_int`` is the parsed integer count or ``None`` when unparsed.
    Unparsed answers count as WRONG.

    Returns:
    * ``exact_match`` -- fraction of ALL items where pred == gt exactly.
    * ``parse_rate``  -- parsed / ALL items.
    * ``n_items``     -- number of items aggregated.

    Empty input returns n_items=0 and 0.0 rates (no division by zero).
    """
    pairs = list(items)
    n = len(pairs)
    if n == 0:
        return {"n_items": 0, "exact_match": 0.0, "parse_rate": 0.0}

    parsed = sum(1 for pred, _ in pairs if pred is not None)
    correct = sum(1 for pred, gt in pairs if pred is not None and int(pred) == int(gt))

    return {
        "n_items": n,
        "exact_match": correct / n,
        "parse_rate": parsed / n,
    }
