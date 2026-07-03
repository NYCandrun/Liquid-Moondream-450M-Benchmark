# Results

Tables only. Every number in this file has a raw per-item answer file behind it
under `results/raw/`. No number here comes from anywhere else.

## Part 1 — Main benchmark

Six models, three tests, full precision, one A100-40GB per test, greedy
decoding. Find-object = RefCOCO-M Acc@0.5 over 2,080 instances; Yes/No = POPE
over 9,000 items; Count = CountBenchQA exact match over 491 items.

| Model | Find-object | Yes/No acc | Yes/No F1 | Count | Parse rate (find / yn / count) | Median ms/item (find / yn / count) |
|---|---|---|---|---|---|---|
| Qwen2.5-VL 7B | **91.7%** | 88.1% | 0.869 | **85.9%** | 100% / 100% / 100% | 1198 / 508 / 108 |
| InternVL3.5 1B | 89.4% | 86.5% | 0.874 | 49.9% | 100% / 100% / 100% | 1356 / 937 / 221 |
| LFM2.5-VL 450M | 76.9% | 87.9% | 0.876 | 71.5% | 100% / 100% / 100% | 709 / 324 / **40** |
| moondream2 2B | 76.4% | **91.2%** | **0.910** | 83.5% | 99% / 100% / 100% | **203** / 316 / 100 |
| LLaVA-OV 0.5B | 3.6% | 87.0% | 0.858 | 73.1% | 100% / 100% / 100% | 578 / 247 / 321 |
| SmolVLM2 500M | 0.0% | 85.1% | 0.855 | 60.9% | 48% / 100% / 100% | 857 / **141** / 138 |

Yes/No uses the STANDARD POPE protocol — question + "Please answer yes or no."
(prompt id `pope_vlmevalkit_v2`) — so the column aligns with the community
convention and published/referee numbers. This supersedes an earlier
bare-question run; see the yes/no-convention note below. Yes/No accuracy, F1
and parse rate are from the suffixed run; yes/no latency stays from the
original 40GB bare run (the two prompts differ by ~5 tokens — immaterial to
per-item wall time, and keeping it preserves a single 40GB latency basis).

Latency is the median per-item wall time on the same A100 class, measured inside
the shared harness (image already decoded, greedy decoding; excludes model load).
It compares each model's full shipped pipeline — preprocessing included — not
FLOPs. Grounding latency partly tracks answer length and preprocessing (e.g.
InternVL's tiling); moondream2's fast grounding is its compiled `detect` skill.

### Yes/No prompt-convention finding (change #11, retained)

The bare question (no instruction) is a real and large error source for small
VLMs. Measured effect of adding the standard "Please answer yes or no."
suffix, same 9,000 POPE items, same greedy decoding, same parser (only the
prompt differs):

| Model | Bare acc | +Suffix acc | Δ | Parse (bare→suffix) |
|---|---|---|---|---|
| SmolVLM2 500M | 55.0 | **85.1** | **+30.1** | 78% → 100% |
| moondream2 2B | 76.7 | 91.2 | +14.5 | 100% → 100% |
| InternVL3.5 1B | 79.1 | 86.5 | +7.4 | 96% → 100% |
| Qwen2.5-VL 7B | 83.7 | 88.1 | +4.4 | 100% → 100% |
| LFM2.5-VL 450M | 84.8 | 87.9 | +3.1 | 100% → 100% |
| LLaVA-OV 0.5B | 85.5 | 87.0 | +1.5 | 100% → 100% |

Reading: the smaller/weaker the model's instruction-following, the more the
missing convention hurt — SmolVLM2 loses **30 points** without it (much of it
unparseable rambling, parse 78%→100%). The suffixed numbers match the
independent VLMEvalKit referee within ~0.5 pt (SmolVLM2 85.1 vs 85.6,
moondream2 91.2 vs 90.7, Qwen 88.1 vs 87.8, LLaVA 87.0 vs 86.5), cross-
validating the convention. Both runs are kept under `results/raw/`
(`bench/` bare, `bench_suffixed/` standard).

## Part 2 — Image-detail dial (sweep)

RefCOCO-M find-object only. One physical GPU for the whole sweep (device ID
stamped in every row), one setting at a time, dedicated CPU cores.

### LFM2.5-VL 450M — true image-token budget (min=max=T, tiling off)

Corrected curve (change log #14): every one of the 2,080 images now completes
at every setting — zero crashes — via the disclosed local pre-upscale
fallback (see METHODOLOGY #14 for the root cause: the pinned stack's
upscale branch over-shoots the token cap and dies inside `generate`; a small
local upscale reroutes the item through the stack's own downscale path).
One physical A100-40GB for all four cells; raws in `results/raw/sweep_v2/`.

| Detail setting (tokens) | Acc@0.5 (all 2,080) | Median speed/item | Mean input tokens* | Images rescued by fallback** |
|---|---|---|---|---|
| 32 (card-recommended floor) | 39.5% | 487 ms | 79 | 0 |
| 64 (shipped minimum) | 60.4% | 504 ms | 110 | 2 |
| 128 | 70.0% | 496 ms | 172 | 10 |
| 256 (shipped cap) | **77.5%** | 504 ms | 296 | 289 |
| — factory-default pipeline (tiling ON, cap 256): see Part 1 row — | **76.9%** | — | — | — |

\* includes ~46 question-text tokens on top of the image tokens.
\** images too small for the forced budget; upscaled locally so the pinned
processor downscales instead of crashing. Rescued items score normally —
b256 rises to 77.5% (from the old completed-only 76.8%), so the ~14% of
small images the original run dropped as crashes were not systematically
harder; forcing a budget above native resolution neither helps nor hurts.

READING: (1) the dial trades accuracy monotonically, 39.5% → 77.5%, and the
forced cap of 256 now lands essentially ON the factory default (77.5 vs
76.9) — the adaptive tiling pipeline and a flat 256-token budget reach the
same place on this task. (2) Per-item speed stays flat across the whole dial
(overhead dominates at 450M). (3) SUPERSEDED HISTORY (change #12): the
original run scored forced-budget crashes as misses, producing a spurious
b256 drop to 66.2% and a dual "completed-only" column; that run is retained
at `results/raw/sweep/` as the crash-discovery record. The reliability
finding stands — the forced dial is unusable in the shipped stack WITHOUT
this workaround (0% → 13.9% crash rate as the budget rises), so the factory
default remains the right production choice; change #14 only recovers the
clean research curve behind the crashes.

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

## Referee lane — same checkpoints re-scored by VLMEvalKit @ 8eb37bb9 (8/8 cells)

The pinned community tool, run by us on Modal (A100), same model revisions,
its own prompts/parsers — an independent instrument over the same weights.
Two disclosed build patches (METHODOLOGY: moondream revision pin; Qwen
attn to sdpa, change #13). POPE acc/F1 = the tool's "acc"/"Overall" columns
(Overall row); Count = the tool's exact-match accuracy.

| Model | POPE acc: ours / referee | POPE F1: ours / referee | Count: ours / referee |
|---|---|---|---|
| Qwen2.5-VL 7B | 83.7 / 87.8 | 0.808 / 0.865 | 85.9 / **89.7** |
| moondream2 2B | 76.7 / 90.7 | 0.803 / 0.904 | 83.5 / 84.0 ✓ |
| LLaVA-OV 0.5B | 85.5 / 86.5 ✓ | 0.837 / 0.852 | **73.1 / 31.6** |
| SmolVLM2 500M | 55.0 / 85.6 | 0.801 / 0.858 | **60.9 / 47.4** |

Readings: (1) counting agrees to 0.5 pt for moondream2 and ~4 pts for Qwen —
the two instruments reproduce each other where conventions align. (2) The
POPE gaps for SmolVLM2 (+30.6) and moondream2 (+14.0) are the change-#11
prompt-convention finding: the tool's POPE data file bakes in the "Please
answer yes or no." suffix; ours asks the bare question. LLaVA is
convention-insensitive (+1.0). (3) The two counting disagreements go the
SAME direction — our harness scores HIGHER than the tool for both LLaVA-OV
(73.1 vs 31.6, +41.5) and SmolVLM2 (60.9 vs 47.4, +13.5); the count prompt
is byte-identical between lanes, so this is an answer-extraction/scoring
difference (our parser credits outputs the tool's marks wrong), not a
prompt effect — per-item diff analysis in Phase 4. (4) Calibration caveat:
vendor-published Qwen numbers used flash-attn; this referee ran sdpa
(change #13).
