# Consolidated data — everything from every round, in one place

This folder is a **copy** of all the benchmark data collected across every round of the project,
gathered here so it's in one spot. **Nothing was moved or deleted** — every original file is still
in its normal home (`results/…` and `calibration/…`). This is a duplicate for convenience.

Built 2026-07-05. Total: **201 files, ~203 MB.**

Two kinds of data live here, kept in separate folders:
- **`raw/`** — the unscored data: the model's actual answers, one line per question, exactly as it
  came out (includes timing per answer). This is the source material.
- **`scored/`** — the graded results: accuracy/F1/pointing numbers computed from the raw data,
  both by our own scorer and by the independent referee tool.
- **`checks/`** — setup and sanity-check outputs from early rounds (not benchmark scores).

---

## Folder layout

```
consolidated_data/
├── raw/                                  all raw answers, grouped by round
│   ├── bench/            (37 files)       MAIN benchmark — 6 models × 3 tasks
│   ├── bench_suffixed/   (12 files)       POPE re-run with the reworded yes/no prompt
│   ├── deployed/         (14 files)       real-world builds (LFM GGUF, moondream as-shipped)
│   ├── probe/            (16 files)       speed/energy probe (counting, various settings)
│   ├── sweep/            (21 files)       quality-vs-setting sweep (pointing + counting)
│   └── sweep_v2/         ( 8 files)       corrected LFM pointing sweep
├── scored/                               graded numbers
│   ├── ours_main_bench/               (18 files)  our scorer's grades for the main benchmark
│   ├── referee_v2_vlmevalkit2/        (32 files)  independent referee: POPE, counting, pointing-trial
│   ├── referee_v1_vlmevalkit/         ( 9 files)  older referee run (POPE only)
│   └── referee_grounding_full_RefCOCO-M/ (1 file) referee's full pointing scores (read off Modal)
└── checks/
    ├── phase0/  (5 files)                 early setup checks (model load, dataset pins, etc.)
    └── phase2/  (25 files)                parser sanity-check samples
```

Inside each `raw/` round you'll see two versions of most files:
- the **full** run (e.g. `bench_lfm450m_pope.jsonl`) — every question, this is the real data.
- a **`.sample…`** version (e.g. `…sample20s7.jsonl`) — a small fixed sample (20 items, seed 7)
  used for eyeballing/verification. Same format, just a subset.

---

## What each round is (plain English)

### raw/bench — the MAIN benchmark
The headline runs. **6 models** (LFM2.5-VL-450M, SmolVLM2-500M, LLaVA-OneVision-0.5B,
InternVL3.5-1B, moondream2, and Qwen2.5-VL-7B) each answered all **3 tasks**: POPE (hallucination
yes/no), counting, and pointing (RefCOCO-M). These are the only raw files that already have grades
computed locally — see `scored/ours_main_bench/`.
> Note: **Qwen2.5-VL-7B was later dropped** from the project (too slow/too big for "edge"). Its raw
> files are kept here as a record but are left out of the final tables.

### raw/bench_suffixed — POPE re-run with a reworded prompt
POPE only, all 6 models, using a tweaked yes/no prompt so the answers line up cleanly with the
headline yes/no format. **Not graded locally yet** — raw answers only.

### raw/deployed — the "as you'd actually ship it" builds
Two real-world setups: **LFM as a compressed GGUF** file, and **moondream 0.5B exactly as it ships**.
Each did all 3 tasks. The two `boxinfo_*.json` files record the machine/runtime they ran on.
**Not graded locally yet.**

### raw/probe — the speed/energy probe
Counting task only, for **LFM and SmolVLM2**, run at several image-detail settings (the `b32`,
`b64`, … `b2048` in the names = how many image tokens). Used to see how speed and accuracy trade
off. These are all fixed samples (100-item and 3-item). **Not graded locally yet.**

### raw/sweep — the quality-vs-setting sweep
Same idea as the probe but covering **pointing and counting** for LFM and SmolVLM2 across image-detail
settings. Includes 50-image spot-check samples (seed 11). **Not graded locally yet.**

### raw/sweep_v2 — the corrected LFM pointing sweep
A redo of the LFM pointing sweep after a fix (fall back to upscaling small images before running).
**Not graded locally yet.**

### scored/ours_main_bench — our grades for the main benchmark
One `.scored.json` per model×task (18 total). Each has the overall numbers (accuracy, F1, pointing
precision, etc.) plus a per-question breakdown. Computed by our own scorer (`scoring/score.py`).

### scored/referee_v2_vlmevalkit2 — the independent referee (current)
A second, well-known scoring tool (VLMEvalKit) grading the same models, to double-check our numbers.
Covers **POPE, counting, and a 24-image pointing trial** for the 4 kept models (Qwen excluded).
The many extra `…bak_…` CSVs are timestamped backups from re-runs — the plainly-named ones are the
final scores.

### scored/referee_v1_vlmevalkit — the older referee run
An earlier referee pass, **POPE only** (and it still includes Qwen). Kept for history.

### scored/referee_grounding_full_RefCOCO-M — the referee's FULL pointing scores
The full 2,080-image pointing scores from the referee. These lived only on Modal's storage; they
were read off on 2026-07-04 and saved here as a single CSV. Numbers: moondream2 79.04, LFM 70.48,
LLaVA 4.04, SmolVLM2 0.0.

### checks/phase0 and checks/phase2 — setup & sanity checks
Not benchmark scores. `phase0` = early checks that models load, datasets are pinned, and settings
behave. `phase2` = small samples used to confirm the answer-parsers work. Included for completeness;
safe to ignore if you only want scores.

---

## Which rounds still need grading
Everything under `raw/` **except `bench`** has no local grades yet: `bench_suffixed`, `deployed`,
`probe`, `sweep`, `sweep_v2`. They can be graded anytime with the project's offline scorer, e.g.:

```
python -m scoring.score consolidated_data/raw/sweep_v2/sweep_v2_lfm450m_grounding_b256.jsonl --out /tmp/scored
```

(That scorer is pure Python, no GPU, and doesn't change the raw files.)
