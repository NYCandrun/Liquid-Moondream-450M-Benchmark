# Methodology

Single spec for both parts, as executed. Every pin, prompt citation, finding, and
change lands here. Gates are executed as recorded quality audits (project owner
amendment, 2026-07-02): at each gate the full package is audited, failures are
fixed and re-run, and the audit is written here.

## Shared harness principle

One config-driven harness (`harness/run.py`), one set of adapters/parsers/prompts/
scoring, the same Modal Images, across both parts (owner directive: bias toward
fair comparison; never two harnesses). A cell config specifies
`(model, task[, image_token_budget])`. All inference runs on Modal; the local
machine is a control plane only. One A100-40GB per Modal function; parallelism
comes from separate single-GPU containers; models are never sharded.

## Part 1 — Main benchmark (6 models × 3 benchmarks)

### Model pins (Phase 0 load matrix, executed on Modal — results/phase0/load_matrix.json)

| key | checkpoint | revision sha | env (transformers) | actual params | notes |
|---|---|---|---|---|---|
| lfm450m | LiquidAI/LFM2.5-VL-450M | `fc6221ca597f3315e4f82fc2df606783267b34ba` | tf5 (5.12.1) | 448,718,848 | native `lfm2_vl`, no remote code |
| smolvlm2 | HuggingFaceTB/SmolVLM2-500M-Video-Instruct | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | tf457 (4.57.1) | 507,482,304 | processor hard-requires `num2words` (Phase 0 finding) |
| llava_ov | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | `74dd0bf867a4cda7950c17663794267c60cf4b40` | tf457 (4.57.1) | 893,675,552 | nominal "0.5B" is 0.89B actual |
| internvl35 | OpenGVLab/InternVL3_5-1B | `2f71cf52542334823e48a46ffba0e2bc9add3446` | tf457 (4.57.1) | 1,060,897,792 | trust_remote_code; 5 remote .py files hashed |
| qwen25vl | Qwen/Qwen2.5-VL-7B-Instruct | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | tf457 (4.57.1) | 8,292,166,656 | nominal "7B" is 8.29B actual |
| moondream2 | vikhyatk/moondream2 @ tag `2025-06-21` | `9a7d4024050840e001defacec2b00727e89149e6` | moondream2 (4.57.1 + libvips) | 1,927,237,104 | newest dated revision (14 enumerated); 18 remote .py files hashed; skills query/detect/point/caption |

All six loaded in bf16 on A100-40GB and passed a 2-image inference. Remote-code
file sha256 hashes recorded in the load matrix JSON. transformers 4.57.1 and
5.12.1 verified to exist on PyPI and to satisfy every model-card minimum
(LFM ≥5.1; SmolVLM2 added in 4.50.0; llava-ov ≥4.45.0; InternVL3_5 ≥4.52.1;
Qwen2.5-VL added in 4.49.0). torch 2.12.1+cu130 identical across both GPU lanes.

### Dataset pins (executed on Modal — results/phase0/dataset_pins.json, data_check.json)

| dataset | revision sha | split | count | verification |
|---|---|---|---|---|
| moondream/refcoco-m | `45686f0eb7980ed4b70d3d5480a198775a26b4c1` | validation | 1,190 raw rows → **2,080 flattened instances (exact, counted by the production loader on Modal)** | flattening rule below |
| lmms-lab/POPE | `4db1276663dfa5eb8ad16a52d24c31a09e470896` | default/test | **9,000** = 3,000 adversarial + 3,000 popular + 3,000 random | category composition verified at load |
| vikhyatk/CountBenchQA | `76d600309e9d6147bd3713b4cd431518ac1206c8` | test | **491 rows, 0 missing images → 491 usable** | exclusion rule: none needed (dataset ships pre-filtered) |

Image retrievability verified inside Modal for all three (first images decoded,
sizes recorded).

**RefCOCO-M flattening rule (frozen):** one evaluation instance per element of the
per-image `samples` list (one per annotated object; 2,080 total). Referring
expression = `sentences[0]` (each object carries 1–3 expressions; the first is
taken, deterministically — decision recorded at Gate A; the per-(object,sentence)
alternative would change N and was rejected to match the ~2,080-instance design).
`bbox` arrives in COCO `[x, y, w, h]` and is converted to `[x1,y1,x2,y2]`.
Convention verified empirically over all 2,080 boxes: xywh hypothesis 0
violations; xyxy hypothesis falsified with 1,204 violations
(results/phase0/data_check.json).

### Decoding & environment

bf16, greedy (`do_sample=False`) everywhere; `max_new_tokens`: 256 grounding,
16 POPE, 16 count (recorded per cell in `configs/bench/`). Single GPU type:
the runner asserts the CUDA device name contains "A100" and "40GB" and refuses
substituted hardware (Phase 0 found Modal delivered an A100-80GB PCIe for one
request; policy = fail-and-retry, never silently accept). Seeds fixed (0).
moondream2 runs through its native skills, which expose no sampler controls in
their public signatures (recorded from `inspect.signature` in the load matrix);
skill defaults are used and noted in the config table per the brief.

### Prompts (verbatim templates in prompts/, provenance in prompts/PROVENANCE.md)

- POPE: dataset-level VLMEvalKit convention at the pinned tag — the RAW question,
  no suffix. Citation: VLMEvalKit v0.3rc1 @ `8eb37bb9`,
  `vlmeval/dataset/image_yorn.py` L25-32 (the "Please answer yes or no." suffix
  at L31 fires only for AMBER) + `vlmeval/dataset/image_base.py` L186-203.
  Note: VLMEvalKit *model adapters* disagree among themselves (its moondream
  adapter appends "\nAnswer yes or no.", its Qwen alias runs with
  `use_custom_prompt=False` so gets the raw question). We hold the dataset-level
  convention constant across all six models; per-model parse rate is reported.
- CountBenchQA: verbatim VLMEvalKit template, `vlmeval/dataset/image_vqa.py`
  L2346: `{question} Note that: answer with a number directly e.g. 3. Do not
  include any additional text.`
- Grounding: each model's native convention (sources in prompts/PROVENANCE.md):
  LFM card-verbatim JSON detection prompt (normalized 0-1 output); Qwen2.5-VL
  blog convention (`bbox_2d`, absolute pixels); InternVL `<ref>…</ref>` prompt
  (`<box>[[x1,y1,x2,y2]]</box>`, 0-1000); moondream2 via `detect(expression)`;
  SmolVLM2 and LLaVA-OV best-effort (no published grounding convention —
  "no boxes" is an acceptable, expected finding).

### Metrics

Grounding headline: **Acc@0.5** (IoU ≥ 0.5, computed over ALL items; unparsed =
miss). Extensions (labeled as extensions): center-in-box recall, mean IoU on
found, predicted/GT area ratio, miss decomposition (not-found vs
found-but-loose), parse rate. POPE: accuracy, F1 (yes = positive; unparsed =
wrong), yes-ratio. CountBenchQA: exact-match accuracy, parse rate. Scoring is a
separate GPU-free pass (`scoring/score.py`) over raw JSONL; anyone can re-score
from published raw outputs.

### Instrument decision (Gate A audit)

VLMEvalKit at the pinned tag v0.3rc1 (`8eb37bb96bc508bd016000318548854fdfa1bd30`)
**cannot be the primary instrument**: it has no LFM/LiquidAI adapter at all
(repo-wide grep: zero hits), no RefCOCO/RefCOCO-M dataset, and no InternVL3_5
registry entry. (Current main adds LFM2VL and RefCOCO, but still no
LFM2.5-VL-450M alias and no RefCOCO-M.) Decision: **custom shared harness is
primary for all six models; VLMEvalKit v0.3rc1 is the secondary calibration lane**
for the covered cells (SmolVLM2-500M, llava-onevision-0.5b, Qwen2.5-VL-7B,
moondream2 × POPE + CountBenchQA), with the moondream2 revision overridden to
`2025-06-21` (its adapter defaults to 2025-01-09 — recorded caveat).

Coverage matrix at the pinned tag (evidence paths in
results/phase0/discovery_bundle.json):

| model | POPE | RefCOCO-M | CountBenchQA |
|---|---|---|---|
| LFM2.5-VL-450M | no (no adapter) | no (no dataset) | no (no adapter) |
| SmolVLM2-500M | yes | no | yes |
| LLaVA-OV-0.5B | yes | no | yes |
| InternVL3_5-1B | partial (generic class, unregistered) | no | partial |
| Qwen2.5-VL-7B | yes | no | yes |
| moondream2 | yes (rev caveat) | no | yes (its counting prompt is dead code at tag — misspelled key) |

### Deployed-form comparison (separate table, never merged)

**Moondream 0.5B distribution finding (verified 2026-07-02, URLs in
results/phase0/discovery_bundle.json):** no transformers-loadable (safetensors
fp16/bf16) 0.5B checkpoint exists. The official moondream org hosts no 0.5B
repo; the 0.5B ships only as int8/int4 `.mf` archives (MOON v1 container of ONNX
components) on the `onnx` branch of vikhyatk/moondream2.

- Artifact (pinned): `moondream-0_5b-int8.mf.gz`, vikhyatk/moondream2 branch
  `onnx` @ `9dddae84d54db4ac56fe37817aeaeb502ed083e2`, 621,619,051 bytes, LFS
  sha256 `355c534d5f71fff8524dfeb41ba65f7d07948f910f754de4df242d0037d8be4a`
  (int4 twin exists, 442,376,060 bytes; **int8 selected** as the deployed form
  to precision-match the LFM int8 lane).
- Runtime (pinned): PyPI **moondream==0.0.6** — the last release containing the
  OnnxVL `.mf` loader (onnxruntime ≥1.19.2). Verified: current `moondream` 1.3.0
  + kestrel/Photon 0.4.2 cannot load `.mf` or the 0.5B at all, so 0.0.6 is the
  only publicly runnable path and IS the as-shipped runtime. The version pin is
  mandatory.
- Skill API: `md.vl(model=path)` → `.query(image, question, settings)`,
  `.detect(image, obj)`, `.point`, `.caption`. Decoding: greedy-only; the sole
  sampler-adjacent setting is `max_tokens` (recorded; a reason this table is
  separate).
- LFM 450M shrunk-down lane: GGUF Q8_0 via llama.cpp selected as the standard
  shipped path (to be built and pinned in Phase 3B; recorded there).

**Interpretation guardrail:** numbers in the deployed-form table are
int8/shrunk-down and are not directly comparable to the bf16 main table. Within
the deployed-form table, the Moondream 0.5B and the LFM 450M are precision-
matched to each other, so that head-to-head is fair; it shows how the two
same-size models behave when deployed the way a buyer would run them. The reader
weighs it.

### License appendix (all verified from repo/vendor sources, 2026-07-02)

| subject | license | source |
|---|---|---|
| LiquidAI/LFM2.5-VL-450M | LFM Open License v1.0 (`lfm1.0`; commercial use restricted to <$10M-revenue organizations; research exception) | huggingface.co/LiquidAI/LFM2.5-VL-450M/raw/main/LICENSE |
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct | Apache-2.0 | HF API cardData @ 7b375e1b |
| llava-hf/llava-onevision-qwen2-0.5b-ov-hf | Apache-2.0 | HF API cardData @ 74dd0bf8 |
| OpenGVLab/InternVL3_5-1B | Apache-2.0 | HF API cardData @ 2f71cf52 |
| Qwen/Qwen2.5-VL-7B-Instruct | Apache-2.0 | HF API cardData @ cc594898 |
| vikhyatk/moondream2 | Apache-2.0 | HF API cardData @ 6b714b26 |
| moondream/moondream3-preview (landscape note; not used) | BUSL-1.1 (Licensor M87 Labs; Change License Apache-2.0, 2 years) | huggingface.co/moondream/moondream3-preview/raw/main/LICENSE.md |

## Part 2 — Image-token-budget sweep

### Swept variable

The number of image (visual) tokens per image. Output length is fixed
(`max_new_tokens=256` for grounding) and never swept.

### Per-model knob definitions (Gate A knob table; full evidence in results/phase0/discovery_bundle.json)

| model | parameter | axis | range | default | same axis as LFM? | true token budget? |
|---|---|---|---|---|---|---|
| lfm450m | `min_image_tokens`/`max_image_tokens` (+`do_image_splitting`) | direct per-image token budget (resize computed FROM token bounds) | positive int; trained envelope 64–256; card recommends min 32 | max 256 / min 64 / splitting ON | reference axis | **YES** — kwargs verified at transformers v5.12.1 release tag |
| smolvlm2 | `size={"longest_edge": N}` | input resolution → 512px tile grid; 64 tokens/tile fixed | positive int (512 multiples recommended) | 2048 (≤1088 tokens) | **partial — different axis, labeled as such; never merged with LFM's** | NO — tile-count dial |
| moondream 0.5B (own runtime) | none exposed | fixed pooled grid | n/a | 729 tokens/image (fixed; confirmed in both candidate codebases) | no | **NO KNOB — single default point, never a fake sweep** |

Sweep points (configs/sweep/): LFM T ∈ {32, 64, 128, 256} with min=max=T and
splitting OFF (monotone single-variable axis); the shipped default pipeline
(max 256 + splitting ON) is the bench grounding cell and is the marked default
point. T=32 is the card-recommended minimum but below the shipped config default
of 64 — included, flagged as mild extrapolation. SmolVLM2 N ∈ {512, 1024, 1536,
2048}; 2048 is the shipped default (marked). RefCOCO-M/COCO source images are
typically ≤640px, so N ≥ 1024 up-samples: those points are annotated as
compute-only (no added information) in every output table. Realized input-token
counts are logged per item (`n_input_tokens`) — measured, never assumed.
Approximate common token scale (from the knob table): LFM 64 ↔ SmolVLM 512
(=64 tokens); LFM 256 ↔ SmolVLM 1024 (≤320); Moondream 0.5B is a fixed 729-token
anchor comparable only to the ~640–729 row. Knob liveness proven on Modal before
any full run (results/phase0/knob_check.json).

### Benchmark & rationale

RefCOCO-M grounding only, same pin and flattening as Part 1 (shared cell:
the sweep's default point and the Part 1 grounding cell run identical code).
Grounding is where visual detail matters most, so it is where sweeping the
image-token budget should produce a real accuracy curve. Optional LFM POPE
flat-check (lowest/highest budget) deferred; run only if trivially cheap after
the main sweep.

### Metrics

Acc@0.5 headline; per cell also mean IoU on found, parse rate, and mean
wall-clock latency per item (first-class here).

## Environments

Modal Images defined in `envs/images.py`; full `pip freeze` lockfiles in
`envs/locks/` (tf5, tf457, moondream2, datasets). A Phase 3B image (`md05-onnx`:
moondream==0.0.6 + onnxruntime, `.mf` artifact baked in with sha256
verification; CPU execution provider, held identical across both parts) is added
before Phase 3B runs. Floating top-level deps are re-pinned from the lockfiles
before the Phase 3 freeze (Gate C checklist).

## Gate A audit record (2026-07-02, non-blocking per owner amendment)

Audited: coverage matrix ✓, load matrix 6/6 ✓, dataset pins + exact counts ✓,
bbox convention ✓ (empirical), licenses ✓, env plan ✓, Moondream 0.5B runtime
plan ✓, knob table ✓, calibration anchors fetched live ✓. Decisions made and
recorded above: custom-harness-primary (no flip); int8 deployed form;
moondream2 @ 2025-06-21 everywhere incl. VLMEvalKit override; per-object
flattening with sentences[0]; POPE dataset-level bare-question convention;
InternVL custom-code repo (not the -HF twin; load-matrix-proven on 4.57.1);
GPU hard-assert; SmolVLM2 upsampled points kept but annotated; LFM T=32 floor
included (flagged). Known open items carried forward: LFM 81.28 RefCOCO-M
protocol is unverifiable (no VLMEvalKit revision contains RefCOCO-M) — treated
as a soft anchor in the calibration table with this caveat printed; CountBench
73.31 (card) vs our CountBenchQA-491 is a dataset-version caveat in the
calibration table; realized token counts to be measured at run time; parse_yes_no
to be aligned to VLMEvalKit YOrN extraction before Gate B freeze.

## Gate B audit record (2026-07-02, non-blocking per owner amendment)

Fixed stratified 20-item sample per (model, task), seed 7, run on Modal through
the production runner (18 cells; sample indices recorded in each JSONL). Dumps
in results/phase2/dumps/, summary in results/phase2/summary.json. Audit:

- **Parser fix applied pre-freeze:** `parse_yes_no` re-implemented as an exact
  port of the pinned VLMEvalKit v0.3rc1 `YOrN_Extraction`
  (vlmeval/dataset/utils/yorn.py L226-233) including its
  `process_punctuation` normalization (vlmeval/smp/misc.py L32-49) and its
  quirks (e.g. `*` is not stripped, so `**Yes**` is Unknown). VLMEvalKit's
  LLM-judge fallback is not used; Unknown → None → scored wrong; parse rate
  reported per cell. Tests updated to the pinned semantics (167 pass).
- **After alignment:** every cell except the two best-effort grounding cells
  and SmolVLM2 POPE parses 20/20. Verified live effect: moondream2 POPE
  "There is no car in the image…" now parses (16/20 correct).
- **Findings (models, not parsers), verified against raw outputs:**
  - `llava_ov` grounding: well-formed normalized boxes, poorly localized —
    0/20 at IoU≥0.5 with parse 20/20. Runs in full; reported as-is.
  - `smolvlm2` grounding: degenerate/garbage boxes (`117,117,117,117`,
    `[10,10,15,15]`) — parse 11/20, 0/20 hits. Runs in full; reported as-is.
  - `smolvlm2` POPE: 4/20 answers use "does not show" phrasing = Unknown
    under the pinned rule (VLMEvalKit-faithful). Parse rate reported.
- Box parsers verified against live captured outputs for all six models
  (LFM normalized JSON ✓, Qwen bbox_2d absolute ✓ 18/20 hits, InternVL
  `<box>` 0-1000 ✓, moondream detect ✓, LLaVA bare normalized quad ✓,
  SmolVLM2 degenerate-tolerant ✓).
- **Parsers frozen at this commit.**

## Gate C audit record (2026-07-02, non-blocking per owner amendment)

- Functional-before-full: all 18 bench cells passed load + parser checks
  through the production runner (Phase 2 samples above). Sweep cells share the
  identical code path; budget wiring proven live at every sweep point value
  (results/phase0/knob_check.json: LFM 44/74/137/254 input tokens at
  T=32/64/128/256; SmolVLM2 83/349/680/878 at N=512/1024/1536/2048).
- Configs frozen: 18 bench + 8 sweep YAMLs at the freeze commit.
- Environments frozen: every top-level dep re-pinned to locked versions;
  locks re-exported post-pin (datasets image: zero drift; GPU lanes: intended
  additions only — datasets==5.0.0 pulled its transitive set). This lock set
  is the freeze baseline.
- GPU substitution policy in force: hard assert A100+40GB, up to 3 Modal
  retries reschedule onto new hosts (observed working: a Qwen cell refused an
  80GB card, retried, landed 40GB).
- No code edits during Phase 3 runs; failures are findings; one retry per
  item on infra errors only.

## Version-sensitivity finding (empirical, 2026-07-02)

SmolVLM2's transformers pin (4.57.1) is a study-set anchor — the vendor card
pins nothing beyond the 4.50.0 floor and recommends "latest". Probed whether
the pin is load-bearing: the exact Phase 2 fixed sample (n=20, seed 7,
grounding, greedy) run on transformers 5.12.1 produced **different raw outputs
on 8 of 20 items** vs the frozen 4.57.1 run (both versions emit SmolVLM's
characteristic degenerate boxes; the tokens differ). Conclusion: small-VLM
greedy outputs are NOT portable across transformers major versions; the
per-lane version pins are mandatory, and the sweep's dual-venv single
container (change #5) is necessary — not merely conservative — to keep each
model's sweep environment byte-identical to its bench environment.

## Comparability statement

The uniform rule across every cell in both parts: **one runner, one dataset
loader, one scoring pass, one prompt template per task (grounding excepted: each
model's native box format, brief-mandated), one answer-parser implementation,
and every model at its shipped-default preprocessing.** Irreducible variance and
its justification: three adapter code paths (the models expose physically
different APIs — 4 of 6 share the generic path); two transformers lanes (LFM
requires ≥5.1, InternVL remote code is pinned known-good on 4.57.1; torch/CUDA
identical); per-model grounding prompts (trained output conventions — forcing
one format would advantage whichever model matches it).

## Change log

1. **2026-07-02, Phase 3 start (runs launched, 3 affected cells re-run from
   zero).** InternVL3_5 preprocessing corrected from a single 448px tile to the
   model's shipped-default dynamic tiling (max_num=12 + thumbnail, 448px, code
   ported verbatim from pinned VLMEvalKit v0.3rc1
   `vlmeval/vlm/internvl/utils.py` L23-85 with the model-card quickstart
   defaults). Reason: comparability audit under the owner's
   fewest-variable-methodologies directive found the single-tile path violated
   the uniform "shipped-default preprocessing" rule and under-fed InternVL
   relative to the other five models. Affected cells: `bench_internvl35_*`
   (grounding, pope, count) — stopped before completion, volume partials
   deleted, functional sample re-checked with the new preprocessing, re-run
   from zero. No other cell touched. In the same window (not results-affecting):
   sample runs now persist to distinct volume filenames so a sample can never
   clobber a full cell's resume state.

2. **2026-07-02, during Phase 3 (policy split; no completed cell affected).**
   GPU policy split after four systematic A100-80GB substitutions by Modal
   under a 10-GPU workspace concurrency limit (strict-40GB cells repeatedly
   refused substituted hardware and exhausted retries). New policy: **bench
   (accuracy) cells require any A100** (same GA100 silicon / sm_80 kernels;
   greedy accuracy unaffected by memory size; the exact device name is recorded
   in every raw row's `meta.gpu`; Part 1 publishes no latency metric — wall
   clock there is raw-capture only). **Sweep cells remain strict A100-40GB**,
   because per-item latency is a first-class published output of the sweep and
   80GB parts have different memory bandwidth. Cells completed before the split
   all ran on A100-40GB and satisfy both policies — none re-run. Affected: only
   cells that had failed on GPU_MISMATCH (relaunched under the new policy).

3. **2026-07-02, during Phase 3 (sweep relaunched from zero; bench accuracy
   unaffected).** Explicit dedicated resource reservation added to every runner
   container (`cpu=8.0`, `memory=32768` MB, recorded per row in
   `meta.cpu_reserved`/`meta.memory_reserved_mb`), owner directive: host CPU
   contention must not pollute benchmark timing. Execution model is otherwise
   unchanged (one cell per container, one dedicated GPU, items strictly
   sequential). Impact analysis: bench cells publish accuracy only (greedy GPU
   outputs unaffected by CPU allocation) — completed and in-flight bench cells
   stand. The sweep publishes per-item latency as a first-class output, so ALL
   8 sweep cells were stopped, their volume partials deleted, and relaunched
   from zero under the pinned reservation — no mixed-condition rows exist in
   any sweep cell.

4. **2026-07-02, during Phase 3 (root cause of GPU substitution found and
   fixed; no completed cell affected).** Modal's documentation confirms the
   substitutions are deliberate platform behavior: "Modal may automatically
   upgrade a gpu=\"A100\" request to run on an 80 GB A100. This automatic
   upgrade does not change the cost," with a documented no-upgrade pin — the
   "!" suffix — recommended "for instance, benchmarking"
   (https://modal.com/docs/guide/gpu). Empirical finding: the documented "!"
   no-upgrade suffix is REJECTED by modal client 1.5.1 (probed:
   InvalidError for both 'A100-40GB!' and 'A100!'; the suffix exists only in
   newer clients). Fix as executed: sweep strictness is enforced by the
   in-container device assert (A100 + 40GB) with up to 3 reschedule retries,
   PLUS removal of the contention trigger — the sweep runs after the bench
   backlog drains at max 2 concurrent, well under the 10-GPU workspace limit
   (every observed substitution occurred under 15-25 parallel requests;
   low-pressure requests landed true 40GB consistently all session). Bench
   cells keep the upgradable request with the any-A100 assert of change #2.
   Sweep cells had produced no accepted rows (all relaunched from zero at
   change #3); bench cells were already policy-correct.

5. **2026-07-02, during Phase 3, before any sweep row existed (owner
   directive, twice-strengthened).** Sweep execution mode fixed to maximum
   hardware constancy: THE ENTIRE SWEEP RUNS IN ONE CONTAINER on ONE physical
   A100-40GB, one cell at a time, nothing concurrent. The environment-pin
   conflict (LFM needs transformers 5.12.1, SmolVLM 4.57.1) is resolved
   without weakening any pin: the SWEEPBOX image carries both pinned
   environments as virtualenvs (/opt/tf5, /opt/tf457; identical version pins
   to the standalone lane images; per-venv lockfiles committed as
   envs/locks/sweepbox-*.lock.txt), and each cell executes as a subprocess in
   its matching venv, running the SAME shared cell core as the bench
   (harness/cell_core.py — extracted verbatim from run.py in this change;
   bench behavior identical). Order: LFM b32→b256 then SmolVLM 512→2048. The
   GPU UUID is recorded in every row — a single die for the whole sweep. No
   resume (fresh_start): a dead container restarts the sweep from zero on a
   fresh single GPU, so the dataset never mixes devices. The generic per-cell
   dispatch refuses sweep configs. `meta.gpu_uuid` added to all future raw
   rows. No sweep rows existed before this change; the new path passed a
   functional smoke (sample mode) before the full run.
