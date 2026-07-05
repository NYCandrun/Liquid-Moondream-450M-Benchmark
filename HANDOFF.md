# Session Handoff — Edge-VLM Benchmark (resume point 2026-07-04)

This file lets a **fresh Claude Code session (incl. cloud/web) pick up exactly where the
local session left off.** The local session's memory (`~/.claude/…`) and Modal credentials
do **not** travel to a cloud VM, so everything needed is captured here. Read this top-to-bottom
before touching anything.

---

## 0. FIRST: environment preconditions (a cloud VM will NOT have these by default)

All model inference runs on **Modal** (workspace `jacob-4532`); local/cloud is control-plane only.

- **Modal auth is required.** A fresh cloud environment has no `~/.modal.toml`. Set Modal
  credentials as env vars in the cloud environment settings (or run `modal token set`):
  `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` for workspace **jacob-4532**. Verify with
  `modal app list` (idle/empty right now — that is expected; see §2).
- **Python/venv:** local uses `.venv/bin/modal` (Python 3.12, modal 1.5.1). In a cloud VM,
  `pip install modal` and call `modal …` directly (drop the `.venv/bin/` prefix).
- **Shared state lives on the Modal volume `edge-vlm-hf-cache`** (HF cache + all VLMEvalKit
  outputs under `vlmeval2_out/…`, TSVs under `lmudata/…`). Nothing to re-download.
  Always `hf_cache_volume.reload()` inside a function before reading recent commits.
- **Do not run anything on a local GPU** — there isn't one; Modal only.

---

## 1. What this project is

Publication-grade edge-VLM benchmark. Repo: `NYCandrun/Liquid-Moondream-450M-Benchmark` (private).

- **Models (5):** LFM2.5-VL-450M, SmolVLM2-500M, LLaVA-OneVision-0.5B, InternVL3.5-1B, moondream2.
  > **Qwen2.5-VL-7B was DROPPED from scope on 2026-07-03** (runtime bottleneck ~5× slower/item;
  > dropping it tightens the "edge VLM" framing to ≤2B params). **Do NOT re-add Qwen.** Its
  > already-collected raw files (`results/raw/bench/bench_qwen25vl_*`) are kept on disk as
  > **archived provenance** but are excluded from the referee lanes and must be scrubbed from
  > headline tables at Phase 4.
- **Benchmarks (3):** RefCOCO-M grounding (`moondream/refcoco-m` — **RefCOCO-M everywhere, NOT
  standard RefCOCO**), POPE, CountBenchQA.
- **Metric equivalences:** our grounding Acc@0.5 == VLMEvalKit Precision@1 @ IoU≥0.5. Single
  scorer `scoring/metrics.py`, `HIT_IOU_THRESHOLD=0.5`.
- **Hard rules:** every reported number comes from execution (published numbers only in the
  calibration table, with live URLs); bf16, A100-40GB, greedy decoding; one A100 per Modal
  function, never shard a model; failures are findings.

---

## 2. Current state (snapshot at handoff — 2026-07-04)

**Git:** `main` == `origin/main` (this handoff + the `calibration/vlmevalkit2/` referee output
are the pending commit). Recent: `b8f2d47` handoff update → `b8855a5` add handoff → `0167292`
drop Qwen.

**Modal:** **idle — no apps running.** A 3-hour hard GPU guillotine was armed earlier and fired
on schedule (2026-07-03 ~23:24 EDT); it stopped nothing because all launched GPU work had already
completed naturally. Nothing is detached or in flight now.

### ✅ Referee (POPE + CountBenchQA) — DONE, 8/8 cells, scored and collected locally
`calibration/vlmevalkit2/*.csv`. **VLMEvalKit-side numbers (these are the referee cross-check):**

| Model | POPE acc | POPE F1 (Overall) | CountBenchQA acc |
|---|---|---|---|
| LFM2.5-VL-450M | 87.56 | 86.72 | 72.69 |
| SmolVLM2-500M | 85.64 | 85.83 | 47.02 |
| LLaVA-OneVision-0.5B | 86.48 | 85.25 | 31.42 |
| moondream2 | 90.70 | 90.43 | 83.78 |

POPE=5127 items, CountBenchQA=487 items. Qwen excluded; InternVL3.5-1B is main-harness-only (not
in the referee lane by design). The earlier LLaVA-count "stall" was a false GPT-judge alarm — POPE
and CountBenchQA `evaluate()` are rule-based (substring / YOrN), they do **not** hang on a missing
OPENAI key; the real cause was tangled zombie workers from abrupt local-process kills. All 8 cells
finished clean once run uninterrupted.

> **Referee↔ours agreement holds:** LFM POPE 87.56 vs ours 87.9 ✓; SmolVLM2 POPE 85.64 reproduces
> v1 85.6 ✓. **One known disagreement to write up at Phase 4:** LLaVA CountBenchQA — ours ~73 vs
> referee 31.42. Needs a per-item diff (likely an answer-extraction/format difference, not a model
> difference). This is the main open analysis item.

### ✅ Grounding smoke (RefCOCO-M, 24 items) — passed
- LFM2.5-VL-450M: **Precision@1 = 83.33**, avg IoU 0.573 → healthy.
- SmolVLM2-500M: **0.0** — **genuine protocol mismatch, NOT a bug.** Under the uniform VLMEvalKit
  grounding prompt SmolVLM2 emits prose or degenerate `[100,100,100,100]` placeholder boxes
  (raw preds captured in `grounding_run_log.smoke.json`). Disclose as a finding; do not "fix."

### ✅ Grounding full (RefCOCO-M) — completed on the volume, NUMBERS NOT YET READ LOCALLY
The full 4-model `full()` run finished and each model committed `…_RefCOCO-M_acc.csv` to the Modal
volume (`vlmeval2_out/<model>/`). The background app was later stopped by the guillotine sweep
**after** it had already committed, so no data was lost. **These full RefCOCO-M Precision@1 numbers
still need to be read off the volume** (server-side read; the local `calibration/vlmevalkit2/` dir
currently only has the *smoke* acc files). See §3.1.

### ⏳ Aside 50-pic re-run — NOT launched; user is DELIBERATING whether to run it at all
Small illustrative readability-vs-token-budget side-table (LFM only). The original crashed on
small-image cells (a crash produces no data to score/patch), so the minimal fix is to re-run just
the failed subset. **User has not authorized this.** Open decision (see §3.3): run it, or drop the
Aside table entirely (then nothing to re-run). Do NOT launch without an explicit go.

### ⏳ Phase 4 assembly — NOT started (see §3.4).

---

## 3. Resume order (do these IN ORDER — concurrency ceiling = 4 A100 containers)

### 3.1 Read the grounding-full numbers, build the calibration/agreement tables
The referee POPE+count are already collected (§2 table). What's missing is the **full RefCOCO-M
acc**. Read it server-side (reload the volume first), e.g. a small Modal CPU function that opens
`vlmeval2_out/<model>/<model>_RefCOCO-M_acc.csv` for all 4 models (pattern: the scratchpad
`read_scores.py`, but target `_RefCOCO-M_acc.csv`). Then:
- Fold the full RefCOCO-M Precision@1 into the calibration table (with LiquidAI's published LFM
  RefCOCO-M **81.28** anchor + live URL).
- Build the **4-model** referee agreement table (POPE + count numbers in §2 + grounding).

> If in doubt whether grounding-full truly finished for a model, just re-run
> `.venv/bin/modal run harness/run_vlmevalkit2_grounding.py::full` — it resumes/skips completed
> cells from the volume, so it's cheap and idempotent.

### 3.2 (only if 3.1 shows a model's full grounding cell is missing) re-run grounding full
```
.venv/bin/modal run harness/run_vlmevalkit2_grounding.py::full   # 4 models, 4 parallel containers
```
Uniform-prompt cross-check (VLMEvalKit sends one grounding prompt to all models). Disclosed.

### 3.3 ⚠️ Aside 50-pic re-run — LAST and SOLO (hard user directive) — BUT GET A GO FIRST
User is deciding whether to keep the Aside table at all. **If yes**, only after 3.1/3.2 are done
and nothing else is on GPU (must NOT be parallelized — it has a latency column and is 40GB-pinned):
```
.venv/bin/modal run harness/run_sweep_serial.py --lfm-upscale --sample-n 50 --sample-seed 11
```
Then update the RESULTS "Aside" table (LFM @256 row) to reflect the change-#14 upscale fix.
**If no** → drop the Aside table from RESULTS and skip this entirely.

### 3.4 Phase 4 assembly (task #13)
Master `results.csv`/`json`; per-model config table; calibration table (incl. LiquidAI's published
LFM RefCOCO-M **81.28** anchor with live URL); demo_data JSON; **per-item diff for the LLaVA
CountBenchQA disagreement (ours ~73 vs referee 31.42)**; **scrub Qwen from RESULTS.md/METHODOLOGY
headline tables + add the descope as a documented methodology change**; README repro lines;
Gate D audit; commit + push.

---

## 4. User constraints locked this session (do not violate)
- **Qwen-7B dropped** — never re-add to any lane or table.
- **Concurrency ceiling = 4** parallel A100 containers.
- **Clean sequential finish** — do NOT overlap GPU stages.
- **Aside runs last and completely alone — and only after the user confirms they want it.**
- **Guillotine rule (if ever re-armed):** if a hard GPU-kill fires, STOP all work and ASK the user
  before doing anything else.
- Prefer **plain-language** status updates.
- Commit to **`main`** and push (user asks for "current branch"); don't branch off.

## 5. Key files
- `harness/run_vlmevalkit2.py` — referee POPE+count (Qwen removed from `TF457_MODELS`).
- `harness/run_vlmevalkit2_grounding.py` — grounding referee (`build`/`smoke`/`full`).
- `harness/run_sweep_serial.py` — sweep + Aside (`--lfm-upscale --sample-n 50 --sample-seed 11`).
- `envs/images.py` — Modal images/pins (VLMEvalKit `401916c`, LFM rev, tf5/tf457 images).
- `scoring/metrics.py` — single scorer (grounding IoU 0.5, POPE, count).
- `calibration/vlmevalkit2/` — referee score CSVs collected locally (POPE, count, smoke grounding).
- `RESULTS.md`, `METHODOLOGY.md` — deliverables (still list Qwen; scrub at Phase 4).
