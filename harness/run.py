"""The config-driven cell runner — ONE harness for both parts of the project.

A cell = one YAML in configs/{bench,sweep}/ specifying (model, task[, budget]).
All inference runs as Modal functions on a single A100-40GB; the local side only
dispatches and collects the raw JSONL.

Guarantees:
  - greedy decoding (do_sample=False baked into adapters), fixed max_new_tokens
  - GPU type asserted in-container (Modal can substitute hardware; we refuse)
  - incremental persistence to a Modal Volume + resume by item_id, so a
    container death never loses completed items
  - per-item: one retry on infrastructure failure only (CUDA/OOM); model
    behavior failures are recorded as findings, never retried
  - raw JSONL row carries full provenance (revisions, versions, device, prompt)

Usage (from repo root):
  .venv/bin/modal run harness/run.py --config configs/bench/lfm450m_grounding.yaml
  .venv/bin/modal run harness/run.py --config ... --sample-n 20 --sample-seed 7   # Phase 2
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import (
    DATASETS_IMAGE,
    GPU_TYPE,
    HF_CACHE_PATH,
    MOONDREAM2_IMAGE,
    TF457_IMAGE,
    TF5_IMAGE,
    hf_cache_volume,
)

app = modal.App("edge-vlm-runner")

TIMEOUT = 12 * 3600
VOLUMES = {HF_CACHE_PATH: hf_cache_volume}
RESULTS_DIR_IN_VOLUME = Path(HF_CACHE_PATH) / "results"

from harness.cell_core import cell_id, run_cell_core  # one shared loop for both parts


def _run_cell(config, prompt_template):
    config = dict(config, _cpu_reserved=RESOURCES["cpu"], _memory_reserved_mb=RESOURCES["memory"])
    return run_cell_core(
        config, prompt_template, RESULTS_DIR_IN_VOLUME, commit_cb=hf_cache_volume.commit
    )


# cpu/memory: explicit dedicated reservation (change log #3) so host CPU
# contention can never pollute wall-clock latency; one cell per container,
# one dedicated GPU per cell, items strictly sequential within the cell.
RESOURCES = dict(cpu=8.0, memory=32768)

# GPU requests (change log #4). Modal DOCUMENTS automatic free upgrades of
# A100 requests to 80GB, and documents a "!" no-upgrade suffix — but the "!"
# syntax is REJECTED by modal client 1.5.1 (probed empirically:
# InvalidError('A100-40GB! is not a valid GPU type'); same for 'A100!').
# Enforcement therefore rests on (a) the in-container device assert + retries
# and (b) scheduling: sweep cells run at max 2 concurrent after the bench
# backlog drains — every observed substitution occurred under 15-25 parallel
# requests; low-pressure requests have landed true 40GB consistently.
GPU_BENCH = GPU_TYPE         # "A100-40GB" — upgradable, free per Modal docs

_RETRIES = modal.Retries(max_retries=3, initial_delay=30.0, backoff_coefficient=2.0)


@app.function(image=TF5_IMAGE, gpu=GPU_BENCH, timeout=TIMEOUT, volumes=VOLUMES, **RESOURCES,
              retries=_RETRIES)
def run_cell_tf5(config: dict, prompt_template: str):
    return _run_cell(config, prompt_template)


@app.function(image=TF457_IMAGE, gpu=GPU_BENCH, timeout=TIMEOUT, volumes=VOLUMES, **RESOURCES,
              retries=_RETRIES)
def run_cell_tf457(config: dict, prompt_template: str):
    return _run_cell(config, prompt_template)


@app.function(image=MOONDREAM2_IMAGE, gpu=GPU_BENCH, timeout=TIMEOUT, volumes=VOLUMES, **RESOURCES,
              retries=_RETRIES)
def run_cell_moondream(config: dict, prompt_template: str):
    return _run_cell(config, prompt_template)


ENV_TO_FN = {"tf5": run_cell_tf5, "tf457": run_cell_tf457, "moondream2": run_cell_moondream}


# --------------------------------------------------------------------------
# Local entrypoint (control plane): dispatch one cell, collect raw JSONL
# --------------------------------------------------------------------------

@app.local_entrypoint()
def main(config: str, sample_n: int = 0, sample_seed: int = 0):
    import yaml

    from harness.adapters import MODEL_PINS

    cfg = yaml.safe_load(Path(config).read_text())
    if sample_n:
        cfg["sample"] = {"n": sample_n, "seed": sample_seed}

    prompt_path = REPO_ROOT / "prompts" / f"{cfg['prompt_id']}.txt"
    prompt_template = prompt_path.read_text()

    if cfg["part"] == "sweep":
        raise SystemExit(
            "sweep cells run ONLY via harness/run_sweep_serial.py (single container, "
            "single physical GPU per model curve, strictly serial — change log #5)"
        )
    env = MODEL_PINS[cfg["model"]]["env"]
    fn = ENV_TO_FN[env]
    print(f"dispatching cell={cell_id(cfg)} env={env} model={cfg['model']} task={cfg['task']}")
    result = fn.remote(cfg, prompt_template)

    suffix = f".sample{sample_n}s{sample_seed}" if sample_n else ""
    out = REPO_ROOT / "results" / "raw" / cfg["part"] / f"{cell_id(cfg)}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.pop("rows_jsonl"))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {out}")
