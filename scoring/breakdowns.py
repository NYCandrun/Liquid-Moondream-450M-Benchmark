"""Breakdowns and confidence intervals over the saved Part 1 answers.

Usage:
    python3 -m scoring.breakdowns            # writes results/tables/breakdowns/

Reads only ``results/raw/`` (plus the grounding hit flags in
``results/tables/bench/*.scored.json``). Pure standard library, no GPU,
deterministic (fixed bootstrap seed).

Produces:
  * 95% Wilson intervals for every headline number.
  * Find-object accuracy by target size (box area / image area).
  * POPE accuracy and yes-rate by question category.
  * Counting accuracy by true count, and the share of answers off by one.
  * A paired bootstrap on the LFM vs moondream2 find-object difference.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.answers import parse_count, parse_yes_no

MODELS = ["qwen25vl", "internvl35", "lfm450m", "moondream2", "llava_ov", "smolvlm2"]
NAMES = {
    "qwen25vl": "Qwen2.5-VL 7B", "internvl35": "InternVL3.5 1B",
    "lfm450m": "LFM2.5-VL 450M", "moondream2": "moondream2 2B",
    "llava_ov": "LLaVA-OV 0.5B", "smolvlm2": "SmolVLM2 500M",
}
SIZE_BINS = (("tiny (<5% of image)", 0.0, 0.05),
             ("medium (5-25%)", 0.05, 0.25),
             ("large (>=25%)", 0.25, 10.0))
HIGH_COUNT = 7  # "large count" threshold used in the README


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _cell(k, n):
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "acc": k / n if n else None, "ci95": [lo, hi]}


def grounding(model, root):
    scored = json.load(open(f"{root}/results/tables/bench/bench_{model}_grounding.scored.json"))
    hits = {p["item_id"]: bool(p["hit_at_50"]) for p in scored["cells"][0]["per_item"]}
    raw = {r["item_id"]: r for r in _rows(f"{root}/results/raw/bench/bench_{model}_grounding.jsonl")}
    bins = defaultdict(lambda: [0, 0])
    for iid, hit in hits.items():
        r = raw[iid]
        g = r["gt"]["box_xyxy"]
        frac = (g[2] - g[0]) * (g[3] - g[1]) / (r["image_width"] * r["image_height"])
        for name, lo, hi in SIZE_BINS:
            if lo <= frac < hi:
                bins[name][0] += 1
                bins[name][1] += hit
    return hits, {"overall": _cell(sum(hits.values()), len(hits)),
                  "by_target_size": {n: _cell(bins[n][1], bins[n][0]) for n, _, _ in SIZE_BINS}}


def pope(model, root):
    rows = _rows(f"{root}/results/raw/bench_suffixed/bench_suffixed_{model}_pope.jsonl")
    cats = defaultdict(lambda: [0, 0, 0])
    tot = [0, 0]
    for r in rows:
        pred = parse_yes_no(r["raw_output"])
        ok = pred is not None and pred == r["gt"]["answer_yes"]
        c = cats[r["gt"]["category"]]
        c[0] += 1
        c[1] += ok
        c[2] += pred is True
        tot[0] += 1
        tot[1] += ok
    out = {"overall": _cell(tot[1], tot[0]), "by_category": {}}
    for k in sorted(cats):
        n, ok, yes = cats[k]
        out["by_category"][k] = {**_cell(ok, n), "yes_rate": yes / n}
    return out


def count(model, root):
    rows = _rows(f"{root}/results/raw/bench/bench_{model}_count.jsonl")
    n = ok = off1 = hi_n = hi_ok = 0
    for r in rows:
        g = r["gt"]["number"]
        p = parse_count(r["raw_output"])
        n += 1
        right = p is not None and p == g
        ok += right
        off1 += p is not None and abs(p - g) == 1
        if g >= HIGH_COUNT:
            hi_n += 1
            hi_ok += right
    return {"overall": _cell(ok, n), "off_by_one_share": off1 / n,
            f"true_count_{HIGH_COUNT}_or_more": _cell(hi_ok, hi_n)}


def paired_bootstrap(a: dict, b: dict, reps: int = 5000, seed: int = 0):
    """Paired bootstrap of mean(a) - mean(b) over shared item ids."""
    ids = sorted(set(a) & set(b))
    d = [int(a[i]) - int(b[i]) for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(reps))
    return {"n": len(d), "diff": sum(d) / len(d),
            "ci95": [means[int(0.025 * reps)], means[int(0.975 * reps) - 1]]}


def build(root):
    out = {"models": {}}
    hits = {}
    for m in MODELS:
        hits[m], g = grounding(m, root)
        out["models"][m] = {"name": NAMES[m], "find": g, "yes_no": pope(m, root), "count": count(m, root)}
    out["lfm_vs_moondream2_find_diff"] = paired_bootstrap(hits["lfm450m"], hits["moondream2"])
    return out


def _pct(c):
    return f"{100 * c['acc']:.1f}% ({100 * c['ci95'][0]:.1f}-{100 * c['ci95'][1]:.1f})"


def markdown(res):
    L = ["## Breakdowns and confidence intervals",
         "",
         "Generated by `python3 -m scoring.breakdowns` from the saved answers. Intervals are 95% Wilson",
         "intervals. Yes/No uses the standard-suffix run. Percentages with fewer than 100 examples are",
         "patterns to check, not settled numbers.",
         "",
         "### Headline scores with 95% intervals", "",
         "| Model | Find-object | Yes/No | Count |", "|---|---|---|---|"]
    for m in MODELS:
        r = res["models"][m]
        L.append(f"| {r['name']} | {_pct(r['find']['overall'])} | {_pct(r['yes_no']['overall'])} | {_pct(r['count']['overall'])} |")
    d = res["lfm_vs_moondream2_find_diff"]
    L += ["", f"LFM minus moondream2 on find-object: {100 * d['diff']:+.1f} points, paired bootstrap 95% interval "
              f"{100 * d['ci95'][0]:+.1f} to {100 * d['ci95'][1]:+.1f} over {d['n']} shared examples.",
          "", "### Find-object accuracy by target size", "",
          "| Model | " + " | ".join(f"{n} (n={res['models'][MODELS[0]]['find']['by_target_size'][n]['n']})" for n, _, _ in SIZE_BINS) + " |",
          "|---|---|---|---|"]
    for m in MODELS:
        b = res["models"][m]["find"]["by_target_size"]
        L.append(f"| {res['models'][m]['name']} | " + " | ".join(f"{100 * b[n]['acc']:.1f}%" for n, _, _ in SIZE_BINS) + " |")
    L += ["", "### Yes/No by question category (accuracy / share of answers that were \"yes\")", "",
          "| Model | random | popular | adversarial |", "|---|---|---|---|"]
    for m in MODELS:
        b = res["models"][m]["yes_no"]["by_category"]
        L.append(f"| {res['models'][m]['name']} | " + " | ".join(
            f"{100 * b[k]['acc']:.1f}% / {100 * b[k]['yes_rate']:.0f}%" for k in ("random", "popular", "adversarial")) + " |")
    key = f"true_count_{HIGH_COUNT}_or_more"
    L += ["", f"### Counting by size of the true count", "",
          f"| Model | All | True count {HIGH_COUNT} or more (n={res['models'][MODELS[0]]['count'][key]['n']}) | Answers off by exactly one |",
          "|---|---|---|---|"]
    for m in MODELS:
        c = res["models"][m]["count"]
        L.append(f"| {res['models'][m]['name']} | {100 * c['overall']['acc']:.1f}% | {100 * c[key]['acc']:.1f}% | {100 * c['off_by_one_share']:.1f}% |")
    return "\n".join(L) + "\n"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = build(root)
    out = os.path.join(root, "results", "tables", "breakdowns")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "breakdowns.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(out, "breakdowns.md"), "w").write(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
