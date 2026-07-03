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

_COMMON = [
    "accelerate",
    "pillow",
    "huggingface_hub",
    "einops",
    "timm",
    "sentencepiece",
    "protobuf",
    "datasets",  # benchmark items are loaded inside the container (never locally)
]

TF5_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("torch", "torchvision")
    .pip_install("transformers==5.12.1", *_COMMON)
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)

TF457_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("torch", "torchvision")
    .pip_install("transformers==4.57.1", *_COMMON, "qwen-vl-utils", "num2words")
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)
# num2words: hard requirement of the SmolVLM processor (found by the Phase 0
# load matrix — its absence fails AutoProcessor.from_pretrained).

MOONDREAM2_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("libvips-dev")
    .pip_install("torch", "torchvision")
    .pip_install("transformers==4.57.1", *_COMMON, "pyvips")
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)

DATASETS_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("datasets", "pillow", "huggingface_hub", "pyarrow")
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)
