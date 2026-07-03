# Methodology

Single spec for both parts. Every pin, prompt citation, finding, and change lands here.

## Part 1 — Main benchmark (6 models × 3 benchmarks)

### Models & pins
_To be filled at Gate A: checkpoint revision hashes, remote-code file hashes,
transformers versions, dated moondream2 revision._

### Datasets & pins
_To be filled at Gate A: revision hashes, exact counts, RefCOCO-M flattening rule,
CountBenchQA exclusion rule, POPE split composition._

### Decoding & environment
bf16, greedy (`do_sample=False`), fixed `max_new_tokens` per task, A100-40GB for all
main-table runs. Exact values recorded per cell in `configs/bench/`.

### Prompts
POPE and CountBenchQA follow pinned VLMEvalKit conventions (exact source file citations
to be recorded here). Grounding uses each model's native box format; moondream2 via
`detect()`.

### Metrics
Grounding: Acc@0.5 (headline); extensions (labeled as extensions): center-in-box recall,
mean IoU on found, area ratio, miss decomposition, parse rate. POPE: accuracy, F1,
yes-ratio. CountBenchQA: exact-match accuracy, parse rate.

### Deployed-form comparison (separate table, never merged)
Moondream 0.5B as-shipped in its own runtime + LFM2.5-VL-450M in its shipped small
(int8/GGUF) form, same items, same scoring. Runtime/artifact/quantization pins to be
filled at Gate A.

**Interpretation guardrail:** numbers in the deployed-form table are int8/shrunk-down and
are not directly comparable to the bf16 main table. Within the deployed-form table, the
Moondream 0.5B and the LFM 450M are precision-matched to each other, so that head-to-head
is fair; it shows how the two same-size models behave when deployed the way a buyer would
run them. The reader weighs it.

### Moondream 0.5B distribution finding
_To be filled at Gate A with URLs._

## Part 2 — Image-token-budget sweep

### Swept variable
The number of image (visual) tokens per image — the encoder/projector speed-quality
dial. Output length (`max_new_tokens`) is fixed and never swept.

### Per-model knob definitions
_To be filled at Gate A: parameter name, range, default, same-axis-as-LFM (yes/no/partial),
exact API call, per model. Models lacking a true image-token knob are handled honestly:
sweep-as-resolution clearly labeled, or single default point — never faked._

### Benchmark & rationale
RefCOCO-M grounding only (same pinned revision and flattening as Part 1): grounding is
where visual detail matters most, so it is where sweeping the image-token budget should
produce a real accuracy curve. POPE and counting are not swept; an optional
LFM lowest/highest-budget POPE flat-check may be reported as a two-row aside.

### Budget points
_To be filled at Gate A: 3–4 points per model from its real range (max, middle, low,
default marked), with a common-scale column if feasible._

### Metrics
Acc@0.5; per cell also mean IoU on found, parse rate, mean wall-clock latency per item
(latency is first-class here).

## Shared harness principle

One config-driven harness, one set of parsers/prompts/scoring, the same Modal Images,
across both parts (fair-comparison directive). A cell config specifies
`(model, task[, image_token_budget])`. The sweep's default-budget LFM grounding cell and
the Part 1 LFM grounding cell run through identical code.

## Environments
_Modal Image definitions in `envs/`, lockfiles exported per env. Plan to be fixed at
Gate A._

## Change log

(empty — nothing changed after freeze; that is the goal)
