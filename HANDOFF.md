# Session Handoff — Edge-VLM Benchmark (resume point 2026-07-03)

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
  `modal app list` — you must see app `ap-F7KT4julH48UKMjAGw4j5m` (see §2).
- **Python/venv:** local uses `.venv/bin/modal` (Python 3.12, modal 1.5.1). In a cloud VM,
  `pip install modal` and call `modal …` directly (drop the `.venv/bin/` prefix).
- **Shared state lives on the Modal volume `edge-vlm-hf-cache`** (HF cache + all VLMEvalKit
  outputs under `vlmeval2_out/…`, TSVs under `lmudata/…`). Nothing to re-download.
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

## 2. Current state (snapshot at handoff)

**Git:** HEAD = `0167292` "Drop Qwen-7B from scope; cache parsed boxes in raw records",
`main` == `origin/main`, tree clean (aside from this HANDOFF.md commit).

**RUNNING remotely (detached, independent of any laptop):**
- Referee app **`ap-F7KT4julH48UKMjAGw4j5m`** — running the POPE+CountBenchQA tail. Commits
  each cell to the volume; has `Retries(max_retries=4)` so preemption auto-resumes from the volume.

> **KNOWN ISSUE (watch LLaVA counting cell):** a prior detached referee (`ap-tpUsFU…`, now
> stopped) stalled ~30 min at 5/8 — the LLaVA `CountBenchQA` cell never completed, and its logs
> showed `vlmeval.api.gpt` OpenAI-API-key errors (VLMEvalKit falling back to a GPT judge for
> answer extraction, with no key in the image). It was likely aggravated by an abrupt local-process
> kill leaving a tangled worker; a clean relaunch (current app) usually clears that. **If LLaVA
> `CountBenchQA` (or moondream2 count) re-stalls on the same GPT-fallback retry loop, fix it before
> continuing:** force rule-based extraction / disable the LLM judge (or set retry=0) in the
> VLMEvalKit CountBenchQA config so scoring stays offline + reproducible — do NOT add a real GPT key
> (would change methodology). LFM & SmolVLM2 count already scored fine offline, so the fix is to make
> the other two use that same rule-based path.

**Referee cells scored on the volume (`vlmeval2_out/<model>/…_score.csv` / `…_acc.csv`):**

| Model | POPE | Count | 
|---|---|---|
| LFM2.5-VL-450M | ✅ | ✅ |
| SmolVLM2-500M | ✅ | ✅ |
| LLaVA-OneVision-0.5B | ✅ | ⏳ pending |
| moondream2 | ⏳ pending | ⏳ pending |

Expected total = **8 cells (4 models × POPE,count)**. Qwen excluded. (InternVL3.5-1B is not in
the referee lane by design — main-harness only.)

**Verified referee↔ours agreement so far (must hold):** LFM POPE 87.6 acc / 0.867 F1, count 72.7
(matches ours 87.9 / 71.5 ✓); SmolVLM2 POPE 85.6 (reproduces v1 85.6 ✓).

**NOT running / needs action:**
- **Grounding smoke** — 0 predictions on the volume; the previous smoke was orphaned. Must be
  re-run (see §3.2). TSVs already built: `lmudata/RefCOCO-M.tsv`, `lmudata/RefCOCO-M-smoke.tsv`.
- **Grounding full**, **Aside 50-pic**, **Phase 4** — all pending (see §3).

**Benign note:** re-running the scorer re-adds a derived `parsed` bbox field to `results/raw/*`
records. Harmless (raw outputs/latency/meta unchanged); commit or ignore.

---

## 3. Resume order (do these IN ORDER — concurrency ceiling = 4 A100 containers)

### 3.1 Collect + score the referee → 4-model agreement table
Wait until the 8 cells are all present (poll the table in §2), then:
```
.venv/bin/modal run harness/run_vlmevalkit2.py        # resumes/skips done cells → runs collect() → writes calibration/vlmevalkit2/
```
Build the **4-model** referee agreement table (Qwen excluded); confirm each model reproduces v1.

### 3.2 Grounding referee (RefCOCO-M) — smoke, then full
```
.venv/bin/modal run harness/run_vlmevalkit2_grounding.py::smoke   # LFM + SmolVLM2 on 24 items; expect GROUNDING_SMOKE_OK
```
If it passes (Precision@1 printed), run full — **4 models, 4 parallel containers** (fits the ceiling):
```
.venv/bin/modal run harness/run_vlmevalkit2_grounding.py::full
```
Disclosed as a uniform-prompt cross-check (VLMEvalKit sends one grounding prompt to all models).

### 3.3 ⚠️ Aside 50-pic re-run — LAST and SOLO (hard user directive)
**Only after 3.1 and 3.2 are fully done and nothing else is on GPU.** Must NOT be parallelized
with anything (it has a latency column and is 40GB-pinned).
```
.venv/bin/modal run harness/run_sweep_serial.py --lfm-upscale --sample-n 50 --sample-seed 11
```
Then update the RESULTS "Aside" table (LFM @256 row) to reflect the change-#14 upscale fix.

### 3.4 Phase 4 assembly (task #13)
Master `results.csv`/`json`; per-model config table; calibration table (incl. LiquidAI's published
LFM RefCOCO-M **81.28** anchor with live URL); demo_data JSON; per-item diff for any referee count
disagreements; **scrub Qwen from RESULTS.md/METHODOLOGY headline tables + add the descope as a
documented methodology change**; README repro lines; Gate D audit; commit + push.

---

## 4. User constraints locked this session (do not violate)
- **Qwen-7B dropped** — never re-add to any lane or table.
- **Concurrency ceiling = 4** parallel A100 containers.
- **Clean sequential finish** — do NOT overlap the referee tail with grounding.
- **Aside runs last and completely alone.**
- Prefer **plain-language** status updates.
- Commit to **`main`** and push (user asks for "current branch"); don't branch off.

## 5. Key files
- `harness/run_vlmevalkit2.py` — referee POPE+count (Qwen removed from `TF457_MODELS`).
- `harness/run_vlmevalkit2_grounding.py` — grounding referee (`build`/`smoke`/`full`).
- `harness/run_sweep_serial.py` — sweep + Aside (`--lfm-upscale --sample-n 50 --sample-seed 11`).
- `envs/images.py` — Modal images/pins (VLMEvalKit `401916c`, LFM rev, tf5/tf457 images).
- `scoring/metrics.py` — single scorer (grounding IoU 0.5, POPE, count).
- `RESULTS.md`, `METHODOLOGY.md` — deliverables (still list Qwen; scrub at Phase 4).
