# Edge Vision-Language Model Evaluation

This project tests six small AI models that can look at a picture and answer questions about it. They are the kind of model small enough to run on a phone or a laptop rather than in a data center. Every model gets the same three tasks:

- **Find an object.** Draw a box around the thing a sentence describes, such as "the dog on the left." 2,080 examples from the RefCOCO-M dataset.
- **Answer yes or no.** Say whether an object is in the picture, such as "Is there a car in the image?" 9,000 questions from the POPE dataset.
- **Count.** Say how many of something there are. 491 questions from the CountBenchQA dataset.

For each model, the project records how often it is right, how long it takes per picture, and how often its answer comes back in a form that can be scored at all. Every model runs through the same code with the same settings. So when two models score differently, it is because the models differ, not because one was asked or graded differently.

## What we did

1. **Compared six small models on three tasks** (find, yes/no, count), all at full precision on the same kind of GPU, with the same settings and the same grading code.
2. **Checked our grading against an independent tool** (VLMEvalKit) on the models it supports, and fixed the one place where our question wording differed from the community standard.
3. **Turned the image-detail dial** on two models (how many image tokens each picture is broken into) and measured accuracy, speed and electricity at every setting.
4. **Ran the models the way they ship**: Moondream 0.5B (the only form it exists in) against LFM2.5-VL 450M's vendor-compressed version, both on an ordinary processor, in a separate table.

## What it found

**The headline: a 450M-parameter model matches a model four times its size at finding objects, and uses far less compute doing it.**

- **LFM2.5-VL 450M ties moondream2 (1.9B parameters) at finding objects.** 76.9% against 76.4% on 2,080 examples. The gap is well inside the noise: resampling the examples puts the difference between about -1.8 and +2.5 points. LFM does it in about 0.7 s per picture, and it also keeps that 76.9% (exactly) when compressed to its vendor-shipped Q8 form.
- **Image detail is nearly free for LFM, so use all of it.** Going from 32 to 256 image tokens roughly doubles accuracy (39.5% to 77.5%) while time per picture stays at about half a second and energy stays between 3.5 and 4.2 joules per answer. SmolVLM2 shows the opposite: its detail dial multiplies energy 5.3 times for a change in accuracy that is within noise. At its shipped defaults SmolVLM2 uses 11 to 13 times the energy of LFM per answer, and about 15 times per correct answer, on the same 100 counting questions (a small sample, so treat the ratio as approximate).
- **Small things are where small models break.** Sorting the find-object examples by how much of the picture the target fills, LFM finds 89% of large targets and 73% of medium ones, but only 44% of tiny ones (under 5% of the picture; just 46 such examples, so this is a pattern to check, not a settled number). Qwen2.5-VL 7B stays at 80% on tiny targets. moondream2 is the odd one: it does about as well on tiny targets (78%) as on medium ones (74%).
- **Counting gets harder as the number grows, and models fail differently.** On the 223 questions whose true answer is 7 or more, Qwen2.5-VL 7B gets 79% right, moondream2 73%, LFM 49%, SmolVLM2 45% and InternVL3.5 1B 32%. InternVL3.5 1B and SmolVLM2 are also the most often off by exactly one (25% and 23% of answers), which is the signature of miscounting rather than failing to read the question.
- **"Is it there?" questions get harder when the wrong object is plausible.** POPE has an easy, a medium and a hard version. Every model scores best on random objects and worst on the adversarial ones (objects that often appear alongside what is really in the picture). InternVL3.5 1B drops from 92.8% to 81.1% and answers "yes" 62% of the time on the hard set; moondream2 drops least (92.2% to 89.5%).
- **Accuracy and speed are not the same ranking.** LFM is the fastest at counting (40 ms per question) while ranking fourth on accuracy; moondream2 is the fastest at finding objects (203 ms) and the best at yes/no; Qwen2.5-VL 7B is the most accurate at finding and counting but takes about 1.2 s per picture to find an object. No model wins all three columns.
- **How the question is worded can matter more than which model is used.** Asked only "Is there a car in the image?", SmolVLM2 500M was right 55.0% of the time, often because it rambled instead of answering. Adding the standard instruction "Please answer yes or no." raised it to 85.1% on the same 9,000 questions.
- **The scores match an outside check, and where they differ we say why.** The yes/no scores agreed with VLMEvalKit within about half a percentage point (SmolVLM2: 85.1% here, 85.6% there). On counting our grader credited more answers than the tool did for two models (LLaVA-OV 73.1% against 31.6%); the question is byte-identical, so this is an answer-reading difference, and we list it as open rather than assuming we are right.
- **Shrinking LFM cost almost nothing; Moondream 0.5B has no other form.** In the deployed-form table LFM 450M Q8 scores 76.9% / 85.3% / 69.7% (find / yes-no / count) against Moondream 0.5B's 21.3% / 55.1% / 31.6%. That table is not comparable to the main one.
- **A crash was found, explained, and fixed in the open.** Fixing LFM's number of image tokens made the model's own software crash on up to 14% of pictures, which at first made the highest setting look worse than it was. The cause: when a small picture had to be enlarged to reach the set number, the model's resizing code rounded both sides up, went over its own limit, and crashed. Enlarging those pictures slightly beforehand sends them through the code's shrinking step instead, which rounds down and is safe. Every picture then completed. The earlier run that crashed is kept as a record.

The size, per-category and counting breakdowns above were computed from the saved answers in `results/raw/`; they are not yet in the generated tables.

Full tables are in [RESULTS.md](RESULTS.md). How each test was controlled, what the limits are, and every change made along the way are in [METHODOLOGY.md](METHODOLOGY.md).

## How it is built

- **One short settings file per test.** Each combination of model and task (and detail level) is a small YAML file, and one program runs them all.
- **Answering and grading are separate steps.** First, the models answer every question on a rented GPU (the kind of chip AI models run on), and every answer is saved along with exactly which model version, software versions, hardware, and prompt produced it. Grading then reads only those saved answers, so anyone can re-grade them on an ordinary laptop.
- **A tested answer reader for each model.** Models format their answers differently; a box might come back as four numbers or as a sentence. A small reader for each model turns its answers into something that can be graded, each reader has its own tests, and the share of answers it could read is reported next to every score.
- **The same conditions every time.** Models always give their single most likely answer, with no randomness. Every run checks that it is on the same kind of GPU (an Nvidia A100 with 40 GB of memory) and refuses to run otherwise. Model and dataset versions are pinned to an exact fingerprint. A run that is interrupted picks up where it left off without losing any answers.
- **Failures are reported, not hidden.** A wrong or unreadable answer counts as a result. It is never quietly retried or dropped.

## Check it yourself

Re-grade a stored result on any computer with Python. No GPU or extra packages are needed:

```bash
python3 -m scoring.score results/raw/bench_suffixed/bench_suffixed_smolvlm2_pope.jsonl --out /tmp/rescore
# writes a report showing accuracy 0.8508 over 9,000 questions (RESULTS.md: 85.1%)
```

Running a model again needs an account with [Modal](https://modal.com), a service that rents out GPUs by the second:

```bash
modal run harness/run.py --config configs/bench/lfm450m_grounding.yaml
```

## How the pieces connect

```mermaid
flowchart LR
  C[Settings file<br/>which model, which task] --> H[Run the model<br/>rented A100 GPU on Modal]
  H --> R[Saved answers<br/>one line per question]
  R --> S[Grade the answers<br/>no GPU needed]
  S --> T[Results tables]
  R --> V[Independent re-check<br/>VLMEvalKit]
```

## A design decision worth noting

The models are compared in two separate tables that are never combined. The main table runs every model uncompressed, at full quality, on the same GPU, so the comparison is fair. A second table runs two of the models the way they are actually shipped to devices: compressed to a smaller size and running on an ordinary processor. Putting both in one table would look neater, but then a difference in score could come from the model, the hardware, or the compression, with no way to tell which.

---

## Project overview

One project, two parts, one shared harness. All inference runs on **Modal**; the local
machine is a control plane only (no GPU, no model loads, no dataset image processing).

- **Part 1 — Main benchmark**: six small vision-language models × three public benchmarks
  (RefCOCO-M grounding, POPE, CountBenchQA), bf16, greedy decoding, single GPU type
  (A100-40GB), fully pinned. Plus a strictly separate **deployed-form table**:
  Moondream 0.5B as-shipped in its own runtime, and LFM2.5-VL-450M quantized to its
  shipped small form, on the same items — never merged into the bf16 main table.
- **Part 2 — Image-token-budget sweep**: grounding accuracy and latency vs. image-token
  budget for three small VLMs (LFM2.5-VL-450M, SmolVLM2-500M, Moondream 0.5B) on
  RefCOCO-M, producing an accuracy-vs-budget curve and a demo-consumable lookup table.

Both parts share one config-driven harness, one set of parsers/prompts/scoring, and the
same Modal Images, so differences in results come from models and settings — not from
harness divergence.

## Status

Complete. Results for both parts are in [RESULTS.md](RESULTS.md), with a raw per-item answer file behind every number under `results/raw/`. Published numbers appear only in the calibration table, with live source URLs.

## Repo map

```
README.md          # this file: setup, repo map, one-command repro per cell
METHODOLOGY.md     # full spec, pins, findings, change log (Part 1 & Part 2 sections)
RESULTS.md         # tables only, no narrative (Part 1 & Part 2 sections)
configs/
  bench/           # Part 1: one YAML per (model, task) cell
  sweep/           # Part 2: one YAML per (model, budget) cell
envs/              # Modal Image definitions + exported lockfiles per env
harness/           # inference runners (Modal functions) — shared by both parts
prompts/           # verbatim prompt templates, one file per template id
parsers/           # per-model output parsers + unit tests
scoring/           # scoring pass over raw JSONL only, no GPU needed
results/
  raw/{bench,sweep}/      # per-item JSONL, one file per cell
  tables/{bench,sweep}/   # generated CSV/JSON + markdown tables
calibration/       # Part 1: published-vs-ours table, VLMEvalKit lane outputs
demo_data/         # Part 2: demo-consumable lookup table (schema note below, TBD)
```

## Setup (control plane)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install modal pyyaml          # in-container environments are pinned in envs/locks/
modal token set                    # or use an existing ~/.modal.toml
```

## Reproducing a cell

One command per cell, run from the repository root:

```bash
modal run harness/run.py --config configs/bench/lfm450m_grounding.yaml
modal run harness/run.py --config configs/bench/lfm450m_grounding.yaml --sample-n 20 --sample-seed 7   # quick sample
```

Score raw answers without a GPU:

```bash
python3 -m scoring.score results/raw/bench/<cell>.jsonl --out results/tables/bench/
```

## Provenance rules (non-negotiable)

- Every reported number comes from execution in this project.
- Everything pinned by hash: checkpoints, remote code, datasets, VLMEvalKit commit, envs.
- Freeze before scale; any post-freeze change re-runs affected cells and is logged in
  METHODOLOGY.md.
- Failures are findings: documented and reported, never silently patched or dropped.
