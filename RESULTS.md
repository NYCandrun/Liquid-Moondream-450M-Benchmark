# Results

Tables only. Every number in this file has a raw per-item answer file behind it
under `results/raw/`. No number here comes from anywhere else.

## Part 1 — Main benchmark

Six models, three tests, full precision, one A100-40GB per test, greedy
decoding. Find-object = RefCOCO-M Acc@0.5 over 2,080 instances; Yes/No = POPE
over 9,000 items; Count = CountBenchQA exact match over 491 items.

| Model | Find-object | Yes/No acc | Yes/No F1 | Count | Parse rate (find / yn / count) | Median ms/item (find / yn / count) |
|---|---|---|---|---|---|---|
| Qwen2.5-VL 7B | **91.7%** | 83.7% | 0.808 | **85.9%** | 100% / 100% / 100% | 1198 / 508 / 108 |
| InternVL3.5 1B | 89.4% | 79.1% | 0.849 | 49.9% | 100% / 96% / 100% | 1356 / 937 / 221 |
| LFM2.5-VL 450M | 76.9% | 84.8% | **0.857** | 71.5% | 100% / 100% / 100% | 709 / 324 / **40** |
| moondream2 2B | 76.4% | 76.7% | 0.803 | 83.5% | 99% / 100% / 100% | **203** / 316 / 100 |
| LLaVA-OV 0.5B | 3.6% | **85.5%** | 0.837 | 73.1% | 100% / 100% / 100% | 578 / 247 / 321 |
| SmolVLM2 500M | 0.0% | 55.0% | 0.801 | 60.9% | 48% / 78% / 100% | 857 / **141** / 138 |

Latency is the median per-item wall time on the same A100 class, measured inside
the shared harness (image already decoded, greedy decoding; excludes model load).
It compares each model's full shipped pipeline — preprocessing included — not
FLOPs. Grounding latency partly tracks answer length and preprocessing (e.g.
InternVL's tiling); moondream2's fast grounding is its compiled `detect` skill.

## Part 2 — Image-detail dial (sweep)

RefCOCO-M find-object only. One physical GPU for the whole sweep (device ID
stamped in every row), one setting at a time, dedicated CPU cores.

### LFM2.5-VL 450M — true image-token budget (min=max=T, tiling off)

| Detail setting (tokens) | Accuracy | Readable answers | Median speed/item | Mean tokens actually used* |
|---|---|---|---|---|
| 32 (card-recommended floor) | 39.5% | 100.0% | 509 ms | 79 |
| 64 (shipped minimum) | 60.3% | 99.9% | 522 ms | 110 |
| 128 | **69.7%** | 99.5% | 532 ms | 172 |
| 256 (shipped cap) | 66.2% | 86.1% | 521 ms | 296 |
| — factory-default pipeline (tiling ON, cap 256): see Part 1 row — | 76.9% | 100% | — | — |

\* includes ~46 question-text tokens on top of the image tokens.

Readings the data supports: the dial genuinely trades accuracy (39.5% → 69.7%);
the best fixed setting is 128, NOT the maximum — at 256 answers get less
parseable (86.1% readable) and accuracy drops; per-item speed on an A100 is
nearly flat across settings (the model is small enough that overhead dominates);
the factory default (adaptive tiling) beats every fixed setting.

### SmolVLM2 500M — resolution dial, measured on COUNTING (change log #7:
grounding is unsupported by this model; its dial is measured on a task it
performs. DIFFERENT task axis from LFM's curve — never overlaid.)

One physical GPU for all rows (single GPU id stamped throughout).

| Detail setting (longest edge px) | Count accuracy | Readable | Median speed/item | Mean tokens used |
|---|---|---|---|---|
| 512 | 58.0% | 99.8% | 101 ms | 106 |
| 1024 | 60.1% | 99.8% | 112 ms | 369 |
| 1536 | 60.7% | 99.8% | 125 ms | 649 |
| 2048 (shipped default) | 60.9% | 99.8% | 142 ms | 1037 |

Reading: the dial moves counting accuracy by only ~3 points end-to-end while
costing ~40% speed — opposite in character to LFM's find-object curve (which
gains ~30 points from the same kind of dial). SmolVLM2's grounding sweep was
cancelled: the model does not support bounding boxes (main-table row 0.0%,
"no boxes" finding; token-matched probe confirmed detail cannot change it).

### Moondream 0.5B — no dial exists

_Single fixed point (729 tokens); runs in the deployed-form lane._

## Aside — token-matched spot check (50 pictures, single GPU die)

Same 50 pictures, both models at settings that use the SAME number of visual
tokens (~64 each in row 1). 50-item sample; headline numbers remain the full
curves above.

| Matched budget | Model | Tokens used (mean) | Accuracy | Readable | Median speed |
|---|---|---|---|---|---|
| ~64 visual tokens | LFM 450M @ 64 | 110 | **56%** | 100% | 514 ms |
| | SmolVLM2 @ 512px | 108 | 0% | 100% | 496 ms |
| ~250–330 | LFM 450M @ 256 | 295 | **62%** | 88% | 524 ms |
| | SmolVLM2 @ 1024px | 374 | 0% | 60% | 720 ms |

## Deployed-form table (as-shipped small forms; int8-class; CPU; SEPARATE — never comparable to the tables above)

Header disclosure: Moondream 0.5B appears in the ONLY form it exists in
(int8 `.mf`, its own runtime — the buyer has no choice). LFM 450M appears in a
vendor-published small variant (GGUF Q8_0, llama.cpp) selected to precision-
match it; LFM's canonical full-precision form is the Part 1 row, and its delta
column below prices the compression. Moondream 0.5B cannot have a delta column
— no full-precision 0.5B exists.

_Safety checks in progress; full runs follow._

| Model (as deployed) | Runtime | Precision | Find-object | Yes/No acc | Count | Δ vs own full-precision |
|---|---|---|---|---|---|---|
| Moondream 0.5B | moondream 0.0.6 | int8 | — | — | — | n/a (no full-precision form exists) |
| LFM 450M (Q8_0) | llama.cpp b9860 | Q8_0 | — | — | — | — |
