"""Modal Image definitions — the single source of truth for every environment.

Each image serves BOTH parts of the project (shared-harness directive): the same
image that runs a model in the Part 1 main table runs it in the Part 2 sweep.

Images
------
TF5_IMAGE        transformers 5.x lane   — LiquidAI/LFM2.5-VL-450M
TF457_IMAGE      transformers 4.57.1 lane — SmolVLM2, LLaVA-OneVision, InternVL3_5,
                 Qwen2.5-VL (InternVL3_5 remote code is known to fail on 5.x)
MOONDREAM2_IMAGE 4.57.1 + libvips/pyvips  — vikhyatk/moondream2 (working transformers
                 version is determined empirically in Phase 0; this is the first try)
DATASETS_IMAGE   CPU-only                 — dataset pinning / scoring, no torch

A Phase 3B image for the Moondream 0.5B as-shipped runtime is added after Gate A
pins the runtime package + version.

Pinning policy: top-level pins here are the spec (transformers versions per the
brief's known-good anchors); the full transitive environment is captured by
harness/export_locks.py into envs/locks/<env>.lock.txt and committed. Before the
Phase 3 freeze, floating top-level deps (torch, datasets, ...) are re-pinned here
to exactly what the exported lockfiles recorded.
"""

import modal

PYTHON_VERSION = "3.12"
HF_CACHE_PATH = "/cache/hf"
GPU_TYPE = "A100-40GB"  # single GPU type for all main-table and sweep runs

# Shared HF cache volume so repeated runs don't re-download weights/datasets.
hf_cache_volume = modal.Volume.from_name("edge-vlm-hf-cache", create_if_missing=True)

# All top-level deps pinned to the exported-lock versions (Gate C prep,
# 2026-07-02): a lockfile with floating tops is not frozen. huggingface_hub
# intentionally differs per lane (transformers 5.x needs hub 1.x).
_COMMON = [
    "accelerate==1.14.0",
    "pillow==12.3.0",
    "einops==0.8.2",
    "timm==1.0.27",
    "sentencepiece==0.2.1",
    "protobuf==7.35.1",
    "datasets==5.0.0",  # benchmark items are loaded inside the container (never locally)
]

TF5_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("torch==2.12.1", "torchvision==0.27.1")
    .pip_install("transformers==5.12.1", "huggingface_hub==1.21.0", *_COMMON)
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)

TF457_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("torch==2.12.1", "torchvision==0.27.1")
    .pip_install(
        "transformers==4.57.1", "huggingface_hub==0.36.2", *_COMMON,
        "qwen-vl-utils==0.0.14", "num2words==0.5.14",
    )
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)
# num2words: hard requirement of the SmolVLM processor (found by the Phase 0
# load matrix — its absence fails AutoProcessor.from_pretrained).

MOONDREAM2_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("libvips-dev")
    .pip_install("torch==2.12.1", "torchvision==0.27.1")
    .pip_install(
        "transformers==4.57.1", "huggingface_hub==0.36.2", *_COMMON, "pyvips==3.1.1",
    )
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)

# SWEEPBOX (change log #5 as-executed): ONE container for the WHOLE sweep.
# Both pinned environments live side by side as virtualenvs; each sweep cell
# runs as a subprocess in its matching venv, strictly serially, so every cell
# of both curves executes on the SAME physical GPU. Version pins identical to
# TF5_IMAGE / TF457_IMAGE; lockfiles exported per venv (sweepbox-tf5 /
# sweepbox-tf457).
_TF5_PKGS = ("transformers==5.12.1 huggingface_hub==1.21.0 accelerate==1.14.0 "
             "pillow==12.3.0 einops==0.8.2 timm==1.0.27 sentencepiece==0.2.1 "
             "protobuf==7.35.1 datasets==5.0.0")
_TF457_PKGS = ("transformers==4.57.1 huggingface_hub==0.36.2 accelerate==1.14.0 "
               "pillow==12.3.0 einops==0.8.2 timm==1.0.27 sentencepiece==0.2.1 "
               "protobuf==7.35.1 datasets==5.0.0 num2words==0.5.14")

SWEEPBOX_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .run_commands(
        "python -m venv /opt/tf5",
        "/opt/tf5/bin/pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1",
        f"/opt/tf5/bin/pip install --no-cache-dir {_TF5_PKGS}",
        "python -m venv /opt/tf457",
        "/opt/tf457/bin/pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1",
        f"/opt/tf457/bin/pip install --no-cache-dir {_TF457_PKGS}",
    )
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)

DATASETS_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "datasets==5.0.0", "pillow==12.3.0", "huggingface_hub==1.21.0", "pyarrow==24.0.0",
    )
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)
