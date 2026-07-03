"""Generate per-cell YAML configs from the single pin source (adapters.MODEL_PINS).

Run locally (no GPU): .venv/bin/python harness/gen_configs.py
Regenerating is idempotent; configs freeze at Gate C (any later change = re-run cell).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml

from harness.adapters import MODEL_PINS

TASKS = {
    "grounding": {"max_new_tokens": 256},
    "pope": {"max_new_tokens": 16},
    "count": {"max_new_tokens": 16},
}

# POPE and CountBenchQA prompts follow the pinned VLMEvalKit conventions (one
# shared template each); grounding uses each model's NATIVE box format.
def prompt_id(model, task):
    if task == "grounding":
        return f"grounding_{model}_v1"
    return f"{task}_vlmevalkit_v1"


# Sweep budget points fixed by the Gate A knob table (results/phase0/discovery_bundle.json):
#   lfm450m:  TRUE token budget, min=max=T, splitting off. 256 = shipped max/cap;
#             the shipped DEFAULT pipeline (max 256 + splitting ON) is the bench
#             grounding cell (image_token_budget null) — the marked default point.
#   smolvlm2: size.longest_edge (resolution/tile axis, NOT a token budget —
#             labeled as its own axis). 2048 = shipped default (marked).
#   moondream 0.5B: no knob — single default point, runs in the Phase 3B runtime.
SWEEP_POINTS = {
    "lfm450m": {"budgets": [32, 64, 128, 256], "default_marker": "bench cell (splitting on, max=256)"},
    "smolvlm2": {"budgets": [512, 1024, 1536, 2048], "default_marker": 2048},
}


def main():
    written = []

    outdir = REPO_ROOT / "configs" / "bench"
    outdir.mkdir(parents=True, exist_ok=True)
    for model in MODEL_PINS:
        for task, tcfg in TASKS.items():
            cfg = {
                "part": "bench",
                "model": model,
                "task": task,
                "prompt_id": prompt_id(model, task),
                "max_new_tokens": tcfg["max_new_tokens"],
                "image_token_budget": None,
                "seed": 0,
                "require_gpu_substrings": ["A100", "40GB"],
            }
            p = outdir / f"{model}_{task}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(f"bench/{p.name}")

    outdir = REPO_ROOT / "configs" / "sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    for model, spec in SWEEP_POINTS.items():
        for b in spec["budgets"]:
            cfg = {
                "part": "sweep",
                "model": model,
                "task": "grounding",
                "prompt_id": prompt_id(model, "grounding"),
                "max_new_tokens": 256,
                "image_token_budget": b,
                "is_shipped_default": b == spec["default_marker"],
                "default_note": str(spec["default_marker"]),
                "seed": 0,
                "require_gpu_substrings": ["A100", "40GB"],
            }
            p = outdir / f"{model}_grounding_b{b}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(f"sweep/{p.name}")

    print(f"wrote {len(written)} configs")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
