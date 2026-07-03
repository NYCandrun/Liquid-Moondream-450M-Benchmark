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
    # DRIVER layer only (system python — the model venvs above are untouched):
    # NVML bindings for the energy probe (change #10). The GPU's cumulative
    # energy counter is live on Modal's A100/driver 580.95.05 but the
    # nvidia-smi CLI omits the field, so the probe reads NVML directly.
    .pip_install("nvidia-ml-py==13.610.43")
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)

# Phase 3B: Moondream 0.5B as-shipped runtime (Gate A pins). CPU-only —
# moondream==0.0.6 is the LAST release able to load .mf archives (onnxruntime
# backend); the pinned int8 artifact is baked in with its sha256 verified at
# build time. Deployed-form rows are tagged runtime+quantization and reported
# ONLY in the separate deployed-form table.
MD05_ARTIFACT_URL = ("https://huggingface.co/vikhyatk/moondream2/resolve/"
                     "9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz")
MD05_SHA256 = "355c534d5f71fff8524dfeb41ba65f7d07948f910f754de4df242d0037d8be4a"

# ONE BOX for the whole deployed-form comparison (owner directive): both
# as-shipped models live in a single CPU container so the head-to-head runs on
# the same physical machine, serially. LFM lane: the VENDOR'S OWN shipped GGUF
# (Q8_0 + Q8_0 vision projector, commit-pinned) run by llama.cpp release b9860
# (official CPU binary). Moondream lane: official int8 .mf + moondream==0.0.6.
LFM_GGUF_COMMIT = "1a3f51cdc26f5d97d5d85cf6901fe5ac7da65a9a"
_LFM_GGUF_BASE = f"https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/{LFM_GGUF_COMMIT}"
LLAMA_CPP_RELEASE = "b9860"

MD05_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")  # moondream 0.0.6 supports 3.10-3.12
    .apt_install("curl", "libgomp1")
    # Inside the as-shipped box the RUNTIME'S OWN dependency constraints win
    # (moondream 0.0.6 pins pillow<11 etc. — it ships that way). Only moondream
    # itself is version-pinned; the resolved set is recorded via lock export.
    .pip_install("moondream==0.0.6", "requests")
    .pip_install("datasets", "huggingface_hub")
    .run_commands(
        # Moondream 0.5B artifact, checksum-verified
        f"curl -L -o /opt/moondream-0_5b-int8.mf.gz '{MD05_ARTIFACT_URL}'",
        f"echo '{MD05_SHA256}  /opt/moondream-0_5b-int8.mf.gz' | sha256sum -c -",
        # LFM 450M vendor-shipped GGUF (Q8_0) + vision projector, commit-pinned
        f"curl -L -o /opt/LFM2.5-VL-450M-Q8_0.gguf '{_LFM_GGUF_BASE}/LFM2.5-VL-450M-Q8_0.gguf'",
        f"curl -L -o /opt/mmproj-LFM2.5-VL-450m-Q8_0.gguf '{_LFM_GGUF_BASE}/mmproj-LFM2.5-VL-450m-Q8_0.gguf'",
        # llama.cpp official CPU binaries, pinned release. The tarball's inner
        # layout varies by release — locate the binary instead of assuming, and
        # leave stable signposts: /usr/local/bin/llama-server + /opt/llama-bin
        f"curl -L -o /tmp/llama.tar.gz 'https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}/llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.tar.gz'",
        "mkdir -p /opt/llama && tar -xzf /tmp/llama.tar.gz -C /opt/llama",
        "find /opt/llama -maxdepth 5 | head -40",
        "BIN=$(find /opt/llama -type f -name llama-server | head -1) && test -n \"$BIN\" "
        "&& ln -sf \"$BIN\" /usr/local/bin/llama-server && ln -sfn \"$(dirname $BIN)\" /opt/llama-bin "
        "&& ls -la /opt/llama-bin/ | head -20",
    )
    .env({"HF_HOME": HF_CACHE_PATH})
    .add_local_python_source("envs", "harness")
)

# VLMEvalKit referee lane: the pinned community tool, same checkpoints, so the
# report carries a same-instrument agreement table. One disclosed build-time
# patch: Moondream2 adapter revision -> the study pin (2025-06-21).
VLMEVAL_COMMIT = "8eb37bb96bc508bd016000318548854fdfa1bd30"

VLMEVAL_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libvips-dev", "curl", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.12.1", "torchvision==0.27.1")
    .run_commands(
        "git clone https://github.com/open-compass/VLMEvalKit /opt/VLMEvalKit",
        f"cd /opt/VLMEvalKit && git checkout {VLMEVAL_COMMIT}",
        "cd /opt/VLMEvalKit && pip install -e .",
        # register RefCOCO-M (OUR items) as a RefCOCODataset variant so
        # `--data RefCOCO-M` resolves; the TSV is written to the volume at
        # LMUData by build_refcoco_m_tsv, so no remote download is attempted.
        "echo \"RefCOCODataset.DATASET_FILES['RefCOCO-M'] = 'RefCOCO-M'\" "
        ">> /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        "echo \"RefCOCODataset.DATASET_FILES['RefCOCO-M-smoke'] = 'RefCOCO-M-smoke'\" "
        ">> /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        "grep -n \"RefCOCO-M\" /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        # disclosed one-line patch: evaluate the SAME moondream2 revision we pin
        "sed -i 's/2025-01-09/2025-06-21/g' /opt/VLMEvalKit/vlmeval/vlm/moondream.py",
        "grep -n '2025-06-21' /opt/VLMEvalKit/vlmeval/vlm/moondream.py",
        # disclosed one-line patch #2 (change #13): the tool's Qwen adapter
        # hardcodes flash_attention_2, which the pinned tool does not install;
        # sdpa instead — which also matches OUR Qwen lane's kernels.
        "sed -i \"s/attn_implementation='flash_attention_2'/attn_implementation='sdpa'/\" "
        "/opt/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py",
        "grep -n \"attn_implementation='sdpa'\" /opt/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py",
    )
    .pip_install(
        "transformers==4.57.1", "huggingface_hub==0.36.2", "accelerate==1.14.0",
        "num2words==0.5.14", "qwen-vl-utils==0.0.14", "pyvips==3.1.1",
        "einops==0.8.2", "timm==1.0.27",
    )
    .env({"HF_HOME": HF_CACHE_PATH, "LMUData": HF_CACHE_PATH + "/lmudata"})
    .add_local_python_source("envs", "harness")
)

# --------------------------------------------------------------------------
# Referee lane RE-PIN (change log #15, owner directive 2026-07-03): move the
# WHOLE referee to ONE newer VLMEvalKit commit that ships vlmeval/vlm/liquid.py
# (LFM support) so LFM 450M — LiquidAI's own eval harness per their model card —
# gets a same-instrument agreement number alongside the four existing models.
# The commit also adds RefCOCO grounding support (used for the RefCOCO-M
# grounding referee, on OUR items).
#
# Commit choice: LFM requires transformers >=5.1 (its model card), but the
# closest-to-original commit with LFM support (19c0e386, Nov 2025) is
# transformers-4.x-only — its granite_vision adapter eagerly imports
# AutoModelForVision2Seq, removed in transformers 5.x, so `import vlmeval`
# dies on the tf5 image (proven empirically by referee2_smoke). The chosen
# commit 401916c (Mar 2026) refactored those adapters for transformers 5.x
# while keeping the POPE scorer, CountBenchQA scoring, and the four existing
# adapters materially intact. Because 401916c is >400 commits past the old pin
# (GitHub's compare diff truncates), the four models' reproduction is verified
# EMPIRICALLY against the v1 referee numbers rather than by static diff. LFM
# runs in the tf5 variant; the other four keep transformers 4.57.1 to match
# both their original referee run and our main-harness lanes.
VLMEVAL2_COMMIT = "401916c00a4769884aeca865a77674abe0c62a26"  # 2026-03-25
LFM_REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"  # our pinned LFM 450M

_VLMEVAL2_BASE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libvips-dev", "curl", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.12.1", "torchvision==0.27.1")
)

# tf457 variant: the four existing models (SmolVLM2, LLaVA-OV, Qwen, moondream2).
# Same two disclosed patches as the v1 referee (both target files verified
# unchanged at this commit, so the patches still apply and are still needed).
VLMEVAL2_TF457_IMAGE = (
    _VLMEVAL2_BASE
    .run_commands(
        "git clone https://github.com/open-compass/VLMEvalKit /opt/VLMEvalKit",
        f"cd /opt/VLMEvalKit && git checkout {VLMEVAL2_COMMIT}",
        "cd /opt/VLMEvalKit && pip install -e .",
        # register RefCOCO-M (OUR items) as a RefCOCODataset variant so
        # `--data RefCOCO-M` resolves; the TSV is written to the volume at
        # LMUData by build_refcoco_m_tsv, so no remote download is attempted.
        "echo \"RefCOCODataset.DATASET_FILES['RefCOCO-M'] = 'RefCOCO-M'\" "
        ">> /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        "echo \"RefCOCODataset.DATASET_FILES['RefCOCO-M-smoke'] = 'RefCOCO-M-smoke'\" "
        ">> /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        "grep -n \"RefCOCO-M\" /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        "sed -i 's/2025-01-09/2025-06-21/g' /opt/VLMEvalKit/vlmeval/vlm/moondream.py",
        "grep -n '2025-06-21' /opt/VLMEvalKit/vlmeval/vlm/moondream.py",
        "sed -i \"s/attn_implementation='flash_attention_2'/attn_implementation='sdpa'/\" "
        "/opt/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py",
        "grep -n \"attn_implementation='sdpa'\" /opt/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py",
    )
    .pip_install(
        "transformers==4.57.1", "huggingface_hub==0.36.2", "accelerate==1.14.0",
        "num2words==0.5.14", "qwen-vl-utils==0.0.14", "pyvips==3.1.1",
        "einops==0.8.2", "timm==1.0.27",
        # newer VLMEvalKit eagerly imports these at package load but does NOT
        # declare them in requirements.txt (verified by referee2_deps.py, which
        # installed until `import vlmeval` succeeded): datasets (chartmuseum),
        # scikit-learn (vlrmbench). pyav is only for video funcs we never call.
        "datasets==5.0.0", "scikit-learn==1.7.2", "rouge==1.0.1",
    )
    .env({"HF_HOME": HF_CACHE_PATH, "LMUData": HF_CACHE_PATH + "/lmudata"})
    .add_local_python_source("envs", "harness")
)

# tf5 variant: LFM 450M via the tool's own LFM2VL adapter (liquid.py). Disclosed
# patches, all confined to LFM: (a) register OUR exact model+revision — the
# config ships LFM2-VL / LFM2.5-VL-1.6B but not LFM2.5-VL-450M; (b) pin the
# revision in liquid.py (the adapter calls from_pretrained without one); (c)
# torch_dtype -> dtype for the transformers-5.x kwarg rename. liquid.py already
# uses sdpa (flash-attn commented out upstream), so no attn patch is needed.
VLMEVAL2_TF5_IMAGE = (
    _VLMEVAL2_BASE
    .run_commands(
        "git clone https://github.com/open-compass/VLMEvalKit /opt/VLMEvalKit",
        f"cd /opt/VLMEvalKit && git checkout {VLMEVAL2_COMMIT}",
        "cd /opt/VLMEvalKit && pip install -e .",
        # register RefCOCO-M (OUR items) as a RefCOCODataset variant so
        # `--data RefCOCO-M` resolves; the TSV is written to the volume at
        # LMUData by build_refcoco_m_tsv, so no remote download is attempted.
        "echo \"RefCOCODataset.DATASET_FILES['RefCOCO-M'] = 'RefCOCO-M'\" "
        ">> /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        "echo \"RefCOCODataset.DATASET_FILES['RefCOCO-M-smoke'] = 'RefCOCO-M-smoke'\" "
        ">> /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        "grep -n \"RefCOCO-M\" /opt/VLMEvalKit/vlmeval/dataset/refcoco.py",
        # (a) register LFM2.5-VL-450M into the LFM series dict (this commit
        # references the class as `vlm.LFM2VL`, not bare `LFM2VL`)
        "sed -i '/^lfm2vl_series = {/a\\    \"LFM2.5-VL-450M\": partial(vlm.LFM2VL, "
        "model_path=\"LiquidAI/LFM2.5-VL-450M\"),' /opt/VLMEvalKit/vlmeval/config.py",
        "grep -n 'LFM2.5-VL-450M' /opt/VLMEvalKit/vlmeval/config.py",
        # (b) pin the revision on both from_pretrained calls in liquid.py
        f"sed -i 's|from_pretrained(model_path)|from_pretrained(model_path, revision=\"{LFM_REVISION}\")|' "
        "/opt/VLMEvalKit/vlmeval/vlm/liquid.py",
        f"sed -i 's|^                model_path,$|                model_path, revision=\"{LFM_REVISION}\",|' "
        "/opt/VLMEvalKit/vlmeval/vlm/liquid.py",
        # (c) transformers 5.x kwarg rename
        "sed -i 's|torch_dtype=torch.bfloat16|dtype=torch.bfloat16|' /opt/VLMEvalKit/vlmeval/vlm/liquid.py",
        f"grep -n 'revision=\"{LFM_REVISION}\"' /opt/VLMEvalKit/vlmeval/vlm/liquid.py",
    )
    .pip_install(
        "transformers==5.12.1", "huggingface_hub==1.21.0", "accelerate==1.14.0",
        "num2words==0.5.14", "qwen-vl-utils==0.0.14", "pyvips==3.1.1",
        "einops==0.8.2", "timm==1.0.27",
        # newer VLMEvalKit eager imports not in its requirements (see tf457 note).
        "datasets==5.0.0", "scikit-learn==1.7.2", "rouge==1.0.1",
    )
    .env({"HF_HOME": HF_CACHE_PATH, "LMUData": HF_CACHE_PATH + "/lmudata"})
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
