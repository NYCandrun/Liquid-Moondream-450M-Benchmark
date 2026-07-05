# Handoff — Edge-VLM Benchmark (where things stand, 2026-07-04)

This note lets a brand-new session (on this laptop or in the cloud) pick up exactly where we
left off. Read it top to bottom before touching anything. It's written plainly on purpose — but
the exact commands and numbers are here too, because whoever resumes needs them.

The one-line version: the big model tests are **done and saved**. What's left is reading one last
set of numbers, deciding on one small optional table, and writing up the final results.

---

## 0. Setup you need first (a fresh cloud machine won't have this)

All the actual model-running happens on **Modal** (a cloud GPU service). Your laptop or the cloud
box just gives the orders — it never runs a model itself.

- **You need Modal login keys.** A fresh machine has none. Set these two values (from the Modal
  account `jacob-4532`) as environment variables, or run `modal token set`:
  `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Check it worked by running `modal app list` — right
  now that list is empty, which is normal (nothing is running).
- **Running the commands:** on this laptop, put `.venv/bin/` in front of `modal` (so
  `.venv/bin/modal ...`). On a fresh cloud box, run `pip install modal` and just type `modal ...`.
- **All the saved work lives in Modal's storage** (called the volume `edge-vlm-hf-cache`). Nothing
  needs to be re-downloaded. One gotcha: inside a Modal function, call `hf_cache_volume.reload()`
  before reading files, or you may see an old copy.
- **There is no local GPU.** Everything runs on Modal. Don't try to run a model locally.

---

## 1. What this project is

A high-quality benchmark comparing small "edge" vision-language models (models small enough to run
on a phone or laptop). Code repo: `NYCandrun/Liquid-Moondream-450M-Benchmark` (private).

- **The 5 models being compared:** LFM2.5-VL-450M, SmolVLM2-500M, LLaVA-OneVision-0.5B,
  InternVL3.5-1B, and moondream2.
  > **We dropped a 6th model, Qwen2.5-VL-7B, on 2026-07-03** because it was too slow (about 5×
  > slower than the others) and too big for the "edge" theme. **Do not add it back.** Its old
  > result files are kept on disk as a record, but leave it out of every table. When we write the
  > final report, remove any mention of it from the headline tables.
- **The 3 tests:** RefCOCO-M (pointing at the right object in an image), POPE (does the model
  hallucinate objects that aren't there — yes/no questions), and CountBenchQA (counting objects).
  > Important: we use **RefCOCO-M**, a specific harder version — not the plain "RefCOCO." Use
  > RefCOCO-M everywhere.
- **The ground rule:** every number in the report has to come from us actually running the test.
  The only outside numbers allowed are in a separate "calibration" table where we compare against a
  company's published score, and even then we link to where they published it.

---

## 2. Where things stand right now (2026-07-04)

**Code:** the `main` branch is pushed and up to date. Latest commits are the handoff updates plus
the Qwen-drop change.

**Modal:** **nothing is running.** Earlier we set a timer to force-kill all GPU work after 3 hours
as a safety net; it went off on schedule but there was nothing left to kill — everything had
already finished. So there's no cost running right now and nothing to babysit.

### ✅ The main cross-check (POPE + counting) — FINISHED for all 4 models
The scores are saved locally in the folder `calibration/vlmevalkit2/` and committed to git. These
are the "referee" numbers — a second, independent tool scoring the same models to make sure our own
numbers are trustworthy:

| Model | POPE (accuracy) | POPE (F1) | Counting (accuracy) |
|---|---|---|---|
| LFM2.5-VL-450M | 87.56 | 86.72 | 72.69 |
| SmolVLM2-500M | 85.64 | 85.83 | 47.02 |
| LLaVA-OneVision-0.5B | 86.48 | 85.25 | 31.42 |
| moondream2 | 90.70 | 90.43 | 83.78 |

(POPE has 5,127 questions; counting has 487.) InternVL3.5-1B isn't in this referee check on
purpose — it only goes through our own main test.

> **Good news: the referee agrees with our own numbers** where we've checked (for example LFM's
> POPE 87.56 vs our 87.9, and SmolVLM2 reproduces an earlier run). **One thing to sort out later:**
> for LLaVA's counting test, our own number (~73) is very different from the referee's (31.42).
> That's almost certainly a difference in how the answer text is read, not a real gap in the model.
> Someone should look at a handful of individual questions side by side to explain it. This is the
> main open question for the final write-up.

### ✅ The pointing test (RefCOCO-M) — quick 24-image trial passed
- LFM scored 83.33 (healthy).
- SmolVLM2 scored 0.0 — and that's **real, not a bug.** When asked to point at an object using the
  standard shared instruction, SmolVLM2 writes a sentence or a fake placeholder box instead of a
  real answer. We just report that as a finding. Don't try to "fix" it.

### ✅ The full pointing test — finished and saved, but numbers not read yet
The full RefCOCO-M run for all 4 models completed and each model's result file is saved in Modal's
storage. The safety-timer stopped the process *after* it had already saved everything, so nothing
was lost. **But we haven't pulled those final pointing scores down to read them yet** — the local
folder only has the 24-image trial so far. Reading these is the first job at resume (see §3.1).

### ⏳ The small optional "Aside" re-run — NOT started; YOU (the user) are still deciding
This is a little bonus table showing how LFM does at different quality settings. The original run
crashed on the small-image cases, so a re-run would just redo the parts that crashed. **You hadn't
decided whether this table is worth keeping at all.** Nothing has been launched. Don't launch it
without a clear go-ahead. If you decide to drop the table, there's nothing to re-run.

### ⏳ Final write-up — not started (see §3.4).

---

## 3. What to do next, in order (run at most 4 GPU jobs at once)

### 3.1 Read the full pointing scores, then build the comparison tables
The POPE and counting scores are already saved locally (the table above). The one missing piece is
the **full pointing (RefCOCO-M) scores.** Read them out of Modal's storage with a small function
that opens each model's `..._RefCOCO-M_acc.csv` file (there's an example to copy in the scratchpad,
`read_scores.py` — just point it at the `_RefCOCO-M_acc.csv` files). Then:
- Add the pointing scores to the calibration table (which also includes the company's published
  LFM pointing score of **81.28**, with a link to where they published it).
- Build the final 4-model comparison table.

> If you're unsure whether the full pointing run really finished for some model, just re-run the
> command in §3.2 — it skips whatever's already done, so it's safe and cheap.

### 3.2 (only if a model's full pointing score is missing) re-run the full pointing test
```
.venv/bin/modal run harness/run_vlmevalkit2_grounding.py::full
```
This runs 4 models at once (one GPU each). Note in the report that every model got the exact same
pointing instruction, so it's a fair apples-to-apples check.

### 3.3 The "Aside" re-run — do this LAST, ALONE, and only if you say yes
You're deciding whether to keep the Aside table. **If yes:** run this only after everything else is
done and no other GPU job is going (it has to run by itself because it measures speed):
```
.venv/bin/modal run harness/run_sweep_serial.py --lfm-upscale --sample-n 50 --sample-seed 11
```
Then update the "Aside" table in the results. **If no:** delete that table from the results and
skip this step entirely.

### 3.4 Write up the final results
Build the master results file; the per-model settings table; the calibration table (with the
company's published 81.28 pointing score and its link); the demo data file; the side-by-side look
at the LLaVA counting difference (our ~73 vs referee 31.42); remove Qwen from the headline tables
and add a short note explaining why it was dropped; add the "how to reproduce this" lines to the
README; do the final quality check; then commit and push.

---

## 4. Rules you set that must not be broken
- **Qwen stays out** — never add it back to any table.
- **Never run more than 4 GPU jobs at the same time.**
- **Finish jobs one stage at a time** — don't overlap them.
- **The "Aside" re-run goes last and completely alone — and only after you confirm you want it.**
- **If a safety timer ever force-kills the GPU jobs again: stop everything and ask the user before
  doing anything else.**
- **Keep status updates in plain English.**
- **Commit to `main` and push** (don't create side branches).

## 5. The important files
- `harness/run_vlmevalkit2.py` — runs the POPE + counting referee check.
- `harness/run_vlmevalkit2_grounding.py` — runs the pointing (RefCOCO-M) check (`build`/`smoke`/`full`).
- `harness/run_sweep_serial.py` — runs the sweep and the "Aside" table.
- `envs/images.py` — the exact software versions used on Modal (so results are reproducible).
- `scoring/metrics.py` — the single piece of code that scores everything.
- `calibration/vlmevalkit2/` — the referee scores saved locally (POPE, counting, and the pointing trial).
- `RESULTS.md`, `METHODOLOGY.md` — the final report files (still mention Qwen; clean that up at the write-up step).
