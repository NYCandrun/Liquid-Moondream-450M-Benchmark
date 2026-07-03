"""Pinned dataset loaders — run INSIDE Modal containers only (never locally).

Single source of truth for dataset revisions, splits, flattening and item schema,
shared by both parts of the project. All pins were established by
harness/dataset_pin.py on Modal (results/phase0/dataset_pins.json).

Item schema produced (per task):
  grounding: {item_id, image (PIL), image_ref, expression, gt: {box_xyxy}, image_width, image_height}
  pope:      {item_id, image (PIL), image_ref, question, gt: {answer_yes, category}, ...}
  count:     {item_id, image (PIL), image_ref, question, gt: {number}, ...}
"""

DATASET_PINS = {
    # pinned by harness/dataset_pin.py on Modal, 2026-07-02
    "refcoco_m": {
        "repo": "moondream/refcoco-m",
        "revision": "45686f0eb7980ed4b70d3d5480a198775a26b4c1",
        "split": "validation",
        # 1,190 raw image rows -> exactly 2,080 annotated objects in `samples`
    },
    "pope": {
        "repo": "lmms-lab/POPE",
        "revision": "4db1276663dfa5eb8ad16a52d24c31a09e470896",
        "config": "default",
        "split": "test",  # 9,000 = 3,000 adversarial + 3,000 popular + 3,000 random
    },
    "countbenchqa": {
        "repo": "vikhyatk/CountBenchQA",
        "revision": "76d600309e9d6147bd3713b4cd431518ac1206c8",
        "split": "test",  # 491 rows, 0 missing images -> 491 usable
    },
}

# RefCOCO-M flattening policy (Gate A audit item):
#   - one instance per element of the per-image `samples` list (2,080 total)
#   - referring expression = sentences[0] (deterministic; each object carries
#     multiple expressions; policy recorded in METHODOLOGY.md)
#   - bbox arrives in COCO [x, y, w, h] convention -> converted to [x1,y1,x2,y2];
#     verified empirically by verify_refcoco_bbox_convention() below.
SENTENCE_POLICY = "first"

TASK_TO_DATASET = {"grounding": "refcoco_m", "pope": "pope", "count": "countbenchqa"}


def _load(pin):
    import datasets as hfds

    kwargs = dict(revision=pin["revision"], split=pin["split"])
    if "config" in pin:
        return hfds.load_dataset(pin["repo"], pin["config"], **kwargs)
    return hfds.load_dataset(pin["repo"], **kwargs)


def load_items(task):
    """Return the full, flattened, deterministic item list for a task."""
    pin = DATASET_PINS[TASK_TO_DATASET[task]]
    ds = _load(pin)

    if task == "grounding":
        items = []
        for row in ds:
            img = row["image"]
            w, h = img.size
            for obj in row["samples"]:
                x, y, bw, bh = obj["bbox"]  # COCO xywh
                items.append(
                    {
                        "item_id": f"{row['image_id']}:{obj['id']}",
                        "image": img,
                        "image_ref": row["file_name"],
                        "expression": obj["sentences"][0],
                        "all_sentences": list(obj["sentences"]),
                        "gt": {"box_xyxy": [x, y, x + bw, y + bh], "category": obj["category"]},
                        "image_width": w,
                        "image_height": h,
                    }
                )
        return items

    if task == "pope":
        items = []
        for row in ds:
            img = row["image"]
            w, h = img.size
            items.append(
                {
                    "item_id": f"pope:{row['id']}",
                    "image": img,
                    "image_ref": row["image_source"],
                    "question": row["question"],
                    "gt": {"answer_yes": row["answer"].strip().lower() == "yes", "category": row["category"]},
                    "image_width": w,
                    "image_height": h,
                }
            )
        return items

    if task == "count":
        items = []
        for i, row in enumerate(ds):
            img = row["image"]
            w, h = img.size
            items.append(
                {
                    "item_id": f"countbench:{i}",
                    "image": img,
                    "image_ref": f"countbenchqa:test:{i}",
                    "question": row["question"],
                    "gt": {"number": int(row["number"])},
                    "image_width": w,
                    "image_height": h,
                }
            )
        return items

    raise ValueError(f"unknown task: {task}")


def fixed_sample(items, n, seed, task):
    """Deterministic fixed sample for Phase 2 parser verification.

    Stratified where a stratum exists (POPE category; grounding COCO category
    coarsely), selected once with a recorded seed. Returns (subset, indices).
    """
    import random

    rng = random.Random(seed)
    idx = list(range(len(items)))
    if task == "pope":
        by_cat = {}
        for i in idx:
            by_cat.setdefault(items[i]["gt"]["category"], []).append(i)
        chosen = []
        per = max(1, n // len(by_cat))
        for cat in sorted(by_cat):
            chosen.extend(rng.sample(by_cat[cat], min(per, len(by_cat[cat]))))
        while len(chosen) < n:
            extra = rng.choice(idx)
            if extra not in chosen:
                chosen.append(extra)
        chosen = sorted(chosen[:n])
    else:
        chosen = sorted(rng.sample(idx, min(n, len(idx))))
    return [items[i] for i in chosen], chosen


def verify_refcoco_bbox_convention():
    """Empirical check of the xywh assumption over ALL 2,080 boxes.

    Counts violations under each hypothesis:
      xywh valid: x+w <= W+2 and y+h <= H+2, w>0, h>0
      xyxy valid: x2>x1, y2>y1, x2 <= W+2, y2 <= H+2
    A hypothesis with violations is falsified. Returns both counts as evidence.
    """
    pin = DATASET_PINS["refcoco_m"]
    ds = _load(pin)
    xywh_viol = xyxy_viol = total = 0
    for row in ds:
        w, h = row["image"].size
        for obj in row["samples"]:
            a, b, c, d = obj["bbox"]
            total += 1
            if not (c > 0 and d > 0 and a + c <= w + 2 and b + d <= h + 2):
                xywh_viol += 1
            if not (c > a and d > b and c <= w + 2 and d <= h + 2):
                xyxy_viol += 1
    return {"total_boxes": total, "xywh_violations": xywh_viol, "xyxy_violations": xyxy_viol}
