"""POPE suffix re-run (owner directive, 2026-07-03): re-score all six models on
POPE with the STANDARD suffixed prompt (`{question} Please answer yes or no.`,
prompt id pope_vlmevalkit_v2) to align the yes/no headline with the documented
benchmark protocol. The v1 bare-question run is preserved untouched (change #11
prompt-convention finding); this run writes to a distinct part so nothing is
overwritten.

ONLY the prompt changes. Same checkpoints/revisions, same greedy decoding, same
answer parser (the exact VLMEvalKit YOrN port), same A100 class — so the delta
is attributable to the single prompt variable.

Topology (owner directive): the six models run across at most FOUR parallel
containers, one model per container at a time (no GPU sharing), refilled as
each finishes — so a container handles ~1-2 models over the run. This reuses
run.py's per-env Modal functions; each `.spawn()` is its own container.

  part = "bench_suffixed"  -> results/raw/bench_suffixed/bench_suffixed_<model>_pope.jsonl

Usage (from repo root; the entrypoint is named to avoid colliding with run.py's
`main` on the shared app, so it must be addressed explicitly):
  .venv/bin/modal run harness/run_pope_suffixed.py::pope_suffixed
  .venv/bin/modal run harness/run_pope_suffixed.py::pope_suffixed --max-concurrent 4
  .venv/bin/modal run harness/run_pope_suffixed.py::pope_suffixed --sample-n 20 --sample-seed 7  # smoke
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact bench app, images, per-env functions, and shared cell core.
from harness.run import ENV_TO_FN, app  # noqa: E402  (app must be imported for `modal run`)
from harness.adapters import MODEL_PINS  # noqa: E402
from harness.cell_core import cell_id  # noqa: E402

PART = "bench_suffixed"
PROMPT_ID = "pope_vlmevalkit_v2"
# Slowest first (InternVL POPE ~0.9 s/item, Qwen ~0.5 s) so the long poles start
# in the first window and short models backfill behind them.
MODELS = ["internvl35", "qwen25vl", "smolvlm2", "llava_ov", "lfm450m", "moondream2"]


def _config(model):
    return {
        "part": PART,
        "model": model,
        "task": "pope",
        "prompt_id": PROMPT_ID,
        "max_new_tokens": 16,
        "image_token_budget": None,
        "seed": 0,
        # match the original POPE cells' assert (["A100"]); yes/no accuracy is
        # GPU-tier invariant under greedy decoding, so an 80GB free-upgrade is
        # acceptable here (this run is for accuracy alignment, not latency).
        "require_gpu_substrings": ["A100"],
    }


@app.local_entrypoint()
def pope_suffixed(max_concurrent: int = 4, sample_n: int = 0, sample_seed: int = 7):
    prompt = (REPO_ROOT / "prompts" / f"{PROMPT_ID}.txt").read_text()
    out_dir = REPO_ROOT / "results" / "raw" / PART
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = list(MODELS)
    running = {}   # FunctionCall -> (model, config)
    done = {}      # model -> summary (minus rows)

    def launch(model):
        cfg = _config(model)
        if sample_n:
            cfg["sample"] = {"n": sample_n, "seed": sample_seed}
        fn = ENV_TO_FN[MODEL_PINS[model]["env"]]
        call = fn.spawn(cfg, prompt)
        running[call] = (model, cfg)
        print(f"[launch] {cell_id(cfg)} env={MODEL_PINS[model]['env']} "
              f"(running={len(running)}, pending={len(pending)})", flush=True)

    while pending or running:
        while pending and len(running) < max_concurrent:
            launch(pending.pop(0))

        finished = None
        for call in list(running):
            try:
                result = call.get(timeout=3)  # poll; raises if not ready
            except TimeoutError:
                continue
            except Exception as exc:  # a container died past its retries
                model, cfg = running.pop(call)
                print(f"[FAILED] {cell_id(cfg)}: {exc!r}", flush=True)
                done[model] = {"cell_id": cell_id(cfg), "error": repr(exc)}
                finished = True
                break
            model, cfg = running.pop(call)
            suffix = f".sample{sample_n}s{sample_seed}" if sample_n else ""
            out = out_dir / f"{cell_id(cfg)}{suffix}.jsonl"
            out.write_text(result.pop("rows_jsonl"))
            done[model] = result
            print(f"[done] {cell_id(cfg)} acc-cell -> wrote {out} "
                  f"(n_new={result.get('n_new')}, n_err={result.get('n_errors')})", flush=True)
            finished = True
            break

        if finished is None and running:
            time.sleep(2)  # nothing ready this pass; back off before re-polling

    print("POPE_SUFFIXED_SUMMARY:" + json.dumps(done, default=str))
    print(f"completed {sum(1 for v in done.values() if 'error' not in v)}/{len(MODELS)} cells")
