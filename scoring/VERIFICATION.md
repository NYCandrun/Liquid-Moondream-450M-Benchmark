# Scoring verification (adversarial pass, 2026-07-02)

Checked: line-by-line read of metrics.py / score.py / __init__.py against spec; acc@0.5 over ALL
items with unparsed = miss and an INCLUSIVE iou >= 0.5 threshold (exact-0.5 case constructed and
confirmed); POPE F1 yes-positive with unparsed = wrong (accuracy) / FN-on-gt-yes only; 300-trial
randomized invariant fuzz (all rates in [0,1]; acc + not_found + found_but_loose == 1; f1/accuracy
hand-recomputed); normalizers fuzzed (None/bytes/bool/NaN/Inf/tuples/objects) — never raise; empty
aggregates return None for conditional stats; CLI smoke-tested end to end in both invocation styles
(--out dir, bad JSONL line, mixed tasks, unknown task, invalid gt, missing file -> exit 1) with
hand-computed aggregates; stdlib-only confirmed. Fixed: nothing (no defects found in scoring/).
Tests: 31/31 pass. Concern for Gate B: `parsed` single-box vs parser list-of-boxes contract (below).
