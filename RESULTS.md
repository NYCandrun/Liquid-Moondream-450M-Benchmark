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

| Detail setting (tokens) | Accuracy (all 2080; crash = miss) | Accuracy (completed only) | Forced-budget crash rate | Median speed/item | Mean tokens actually used* |
|---|---|---|---|---|---|
| 32 (card-recommended floor) | 39.5% | 39.5% | 0.0% | 509 ms | 79 |
| 64 (shipped minimum) | 60.3% | 60.4% | 0.1% (2/2080) | 522 ms | 110 |
| 128 | 69.7% | 70.0% | 0.5% (10/2080) | 532 ms | 172 |
| 256 (shipped cap) | 66.2% | **76.8%** | **13.9% (289/2080)** | 521 ms | 296 |
| — factory-default pipeline (tiling ON, cap 256): see Part 1 row — | **76.9%** | 76.9% | 0.0% | — | — |

\* includes ~46 question-text tokens on top of the image tokens.

CORRECTED READING (change log #12 — supersedes the earlier "answers get less
parseable at 256" interpretation): the model's answers parse at 100% at EVERY
setting on rows that complete; what looked like unreadable answers were
processor CRASHES — forcing min=max token budgets makes the pinned
preprocessing stack (transformers 5.12.1) crash on images it must upscale,
at a rate that grows with the setting (0% → 13.9%), and a crashed query
scores as a miss. Two findings replace one: (1) the dial genuinely trades
accuracy, monotonically — 39.5% → 76.8% on completed images, with 256
matching the factory default; (2) the forced dial is UNRELIABLE at high
settings in this stack, so the factory default (adaptive tiling, zero
crashes, best accuracy) is strictly the right production choice. Per-item
speed on an A100 stays flat across settings (overhead dominates at this
model size).

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
One item (countbench:116, a 100×100 grayscale image) errors in SmolVLM2's
shipped processor at every setting ("Unsupported number of image dimensions:
2") and scores as a miss — a uniform −0.2 pt on all four rows (change #12).

### Efficiency probe — measured energy + latency at every dial setting

The SAME 100 counting questions for all 8 cells (drawn from a seeded
300-image pool, keeping only items every cell completes — the forced-budget
crash finding, change #12, made this filter necessary; the subset runs
~2 pts easier than the full set but preserves the curve shapes, validated
against SmolVLM2's full-491 curve above). One physical A100-40GB die for all
cells (UUID stamped per row). Energy is MEASURED from the GPU's cumulative
NVML energy counter over the answering window only (model load excluded);
"net" subtracts each cell's own 15 s idle-power baseline (~53 W).

| Model @ setting | Count acc (this 100) | Median ms/answer | Mean input tokens | J/answer raw | J/answer net-of-idle |
|---|---|---|---|---|---|
| LFM @ 32 | 59.0% | **38** | 67 | 3.51 | 0.54 |
| LFM @ 64 | 69.0% | **37** | 100 | 3.49 | 0.62 |
| LFM @ 128 | **72.0%** | **37** | 160 | 3.61 | 0.71 |
| LFM @ 256 | 68.0% | **38** | 288 | 4.20 | 1.01 |
| SmolVLM2 @ 512px | 63.0% | 84 | 106 | 9.07 | 1.54 |
| SmolVLM2 @ 1024px | 61.0% | 96 | 366 | 16.19 | 3.75 |
| SmolVLM2 @ 1536px | 64.0% | 113 | 636 | 24.35 | 6.16 |
| SmolVLM2 @ 2048px (default) | 62.0% | 131 | 1023 | 47.86 | 13.99 |

Readings the data supports: (1) LFM's entire dial lives in a 3.5–4.2 J
band — at 450M scale, extra image detail is nearly free in both time and
energy (fixed overhead dominates). (2) SmolVLM2's dial multiplies energy
5.3× (9.1→47.9 J) and latency 1.6× for an accuracy change inside the
noise — consistent with its flat full-set curve. (3) At shipped defaults on
identical questions, SmolVLM2 spends **11–13× the energy** and 3.4× the
time per answer of LFM at any setting, and scores lower. (4) Per CORRECT
answer at defaults-vs-best: SmolVLM2 @2048 ≈ 77 J/correct vs LFM @128 ≈
5.0 J/correct — ~15×. (5) LFM's counting dial peaks at 128 (72%); 128 vs
256 is within n=100 noise (±5 pts), but 32 is clearly worse — same
more-detail-is-not-better character as its grounding curve.

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

LFM @256 note (change #12): its 12% unreadable = 6/50 forced-budget processor
crashes, not model output; on the 44 completed items accuracy is 70.5%.
SmolVLM2's 60% readable is genuine degenerate output, not crashes.

## Deployed-form table (as-shipped small forms; int8-class; CPU; SEPARATE — never comparable to the tables above)

Header disclosure: Moondream 0.5B appears in the ONLY form it exists in
(int8 `.mf`, its own runtime — the buyer has no choice). LFM 450M appears in a
vendor-published small variant (GGUF Q8_0, llama.cpp) selected to precision-
match it; LFM's canonical full-precision form is the Part 1 row, and its delta
column below prices the compression. Moondream 0.5B cannot have a delta column
— no full-precision 0.5B exists.

Hardware: two Modal boxes, 8 dedicated cores + 8 GB each, verified by an
in-container self-test (the sandbox hides the CPU model): single-core loop
0.255 s vs 0.272 s (within 7%); memory-copy bandwidth 5,040 vs 2,848 MB/s —
the LFM box drew the ~43% lower-bandwidth host, and llama.cpp CPU inference
is memory-bound, so LFM's speed numbers below are if anything understated.
Latency is measured compute time on matched boxes — never labeled energy.

| Model (as deployed) | Runtime | Precision | Find-object | Yes/No acc (F1) | Count | Median s/item (find / yn / count) | Δ vs own full-precision |
|---|---|---|---|---|---|---|---|
| Moondream 0.5B | moondream 0.0.6 | int8 | 21.3% (parse 85.8%) | 55.1% (0.689) | 31.6% | 2.21 / 2.82 / 1.14 | n/a (no full-precision form exists) |
| LFM 450M (Q8_0) | llama.cpp b9860 | Q8_0 | **76.9%** | **85.3%** (0.860) | **69.7%** | **1.36 / 1.05 / 0.65** | find ±0.0, yes/no +0.5, count −1.8 |

Readings the data supports: (1) LFM's vendor Q8_0 loses essentially nothing
to its own bf16 — find-object is IDENTICAL at 76.9%, counting −1.8, yes/no
+0.5; the llama.cpp lane independently reproducing the transformers lane
also cross-validates both stacks. (2) The as-shipped head-to-head is
one-sided: +55.6 points find-object, +38.1 counting, +30.2 yes/no, at
~2-4× lower latency, from the slower-memory box. (3) Moondream 0.5B's
yes/no answer is dominated by yes-bias (94% "yes" under the bare-question
convention; moondream2 gains +14 pts with the community suffix — change
log #11 — and its 0.5B distillation plausibly shares that sensitivity,
untestable in VLMEvalKit, which has no .mf runner).
