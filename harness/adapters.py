"""Model adapters — the single inference code path shared by BOTH parts.

Each adapter knows how to (a) load its model at a pinned revision in bf16,
(b) apply an image-token budget where the model exposes one (sweep part), and
(c) run one deterministic (greedy) inference for a task item.

Runs INSIDE Modal containers only. Code paths were proven by harness/load_check.py
(results/phase0/load_matrix.json).

An adapter that is asked to set an image-token budget it has no wiring for
raises immediately — a sweep point must never silently run at default.
"""

import time


def _from_pretrained_bf16(cls, model_id, revision, **kw):
    """bf16 load tolerating the 4.x->5.x torch_dtype -> dtype rename."""
    import torch

    try:
        return cls.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, **kw)
    except TypeError:
        return cls.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16, **kw)


def _cast_and_move(inputs, model):
    import torch

    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(model.device)
            if v.is_floating_point():
                v = v.to(model.dtype)
        out[k] = v
    return out


class ITTAdapter:
    """Generic AutoModelForImageTextToText path: LFM2.5-VL, SmolVLM2, LLaVA-OV, Qwen2.5-VL."""

    def __init__(self, model_id, revision, trust_remote_code=False, image_token_budget=None,
                 budget_wiring=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.revision = revision
        # budget_wiring: model-specific hook mapping a budget value to
        # {"template": kwargs for apply_chat_template, "pretrained": kwargs for
        # AutoProcessor.from_pretrained} (fixed by the Gate A knob table).
        # Without a hook, setting a budget is an error — never a silent default run.
        self.template_kwargs = {}
        pretrained_kwargs = {}
        if image_token_budget is not None:
            if budget_wiring is None:
                raise NotImplementedError(
                    f"{model_id}: image_token_budget={image_token_budget} requested but no "
                    "budget wiring exists for this model"
                )
            wired = budget_wiring(image_token_budget)
            self.template_kwargs = wired.get("template", {})
            pretrained_kwargs = wired.get("pretrained", {})

        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code, **pretrained_kwargs
        )
        self.model = _from_pretrained_bf16(
            AutoModelForImageTextToText, model_id, revision,
            trust_remote_code=trust_remote_code, device_map="cuda",
        ).eval()

    def generate(self, image, prompt, max_new_tokens, task=None):
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            **self.template_kwargs,
        )
        inputs = _cast_and_move(dict(inputs), self.model)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        latency_ms = (time.monotonic() - t0) * 1000
        text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        return {"raw_output": text, "latency_ms": round(latency_ms, 1), "n_input_tokens": int(n_in)}


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Verbatim port: VLMEvalKit v0.3rc1 vlmeval/vlm/internvl/utils.py L34-47."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Verbatim port: VLMEvalKit v0.3rc1 vlmeval/vlm/internvl/utils.py L50-85,
    with the MODEL CARD's shipped defaults (OpenGVLab/InternVL3_5-1B README
    quickstart: max_num=12, use_thumbnail=True, input_size=448)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class InternVLAdapter:
    """InternVL3_5 remote-code path: AutoModel + model.chat() with the model's
    SHIPPED-DEFAULT dynamic tiling (max_num=12 tiles + thumbnail, 448px), per
    its model-card quickstart. Change-logged at Phase 3 start: an earlier
    version used a single 448px tile, which under-fed InternVL relative to the
    uniform rule (every model runs its shipped default preprocessing)."""

    def __init__(self, model_id, revision, image_token_budget=None):
        from transformers import AutoModel, AutoTokenizer

        if image_token_budget is not None:
            raise NotImplementedError(f"{model_id}: no image-token budget wiring (not a sweep model)")
        self.model_id = model_id
        self.revision = revision
        self.model = _from_pretrained_bf16(
            AutoModel, model_id, revision, trust_remote_code=True, low_cpu_mem_usage=True
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, use_fast=False
        )
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        # build_transform, verbatim port: VLMEvalKit v0.3rc1 internvl/utils.py L23-31
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def generate(self, image, prompt, max_new_tokens, task=None):
        import torch

        tiles = _internvl_dynamic_preprocess(
            image.convert("RGB"), min_num=1, max_num=12, image_size=448, use_thumbnail=True
        )
        pv = torch.stack([self.transform(t) for t in tiles]).to(torch.bfloat16).cuda()
        t0 = time.monotonic()
        with torch.inference_mode():
            resp = self.model.chat(
                self.tokenizer, pv, "<image>\n" + prompt, dict(max_new_tokens=max_new_tokens, do_sample=False)
            )
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "raw_output": resp,
            "latency_ms": round(latency_ms, 1),
            "n_input_tokens": None,
            "n_tiles": len(tiles),
        }


class MoondreamAdapter:
    """moondream2 native skills: detect() for grounding, query() for POPE/count.

    Skills expose no sampler controls in their public signatures (recorded in the
    load matrix); defaults are used and noted in the config table per the brief.
    """

    def __init__(self, model_id, revision, image_token_budget=None):
        from transformers import AutoModelForCausalLM

        if image_token_budget is not None:
            raise NotImplementedError(f"{model_id}: no image-token budget wiring")
        self.model_id = model_id
        self.revision = revision
        self.model = _from_pretrained_bf16(
            AutoModelForCausalLM, model_id, revision, trust_remote_code=True, device_map={"": "cuda"}
        ).eval()

    def generate(self, image, prompt, max_new_tokens, task="pope"):
        import json as _json

        t0 = time.monotonic()
        if task == "grounding":
            result = self.model.detect(image, prompt)
            raw = _json.dumps(result.get("objects", result), default=str)
        else:
            result = self.model.query(image, prompt)
            raw = result.get("answer", str(result))
        latency_ms = (time.monotonic() - t0) * 1000
        return {"raw_output": raw, "latency_ms": round(latency_ms, 1), "n_input_tokens": None}


class MD05Adapter:
    """Moondream 0.5B AS-SHIPPED: official int8 .mf in moondream==0.0.6 (CPU).

    The runtime is greedy-only; its sole knob is max_tokens (mapped from the
    task's max_new_tokens). detect() for grounding, query() otherwise —
    mirroring the moondream2 main-table lane.
    """

    def __init__(self, mf_gz="/opt/moondream-0_5b-int8.mf.gz"):
        import gzip
        import os
        import shutil

        import moondream as md

        mf = "/tmp/moondream-0_5b-int8.mf"
        if not os.path.exists(mf):
            with gzip.open(mf_gz, "rb") as fin, open(mf, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        self.model = md.vl(model=mf)

    def generate(self, image, prompt, max_new_tokens, task=None):
        import json as _json

        t0 = time.monotonic()
        if task == "grounding":
            result = self.model.detect(image, prompt)
            raw = _json.dumps(result.get("objects", result), default=str)
        else:
            result = self.model.query(image, prompt, settings={"max_tokens": int(max_new_tokens)})
            raw = result.get("answer", str(result))
        latency_ms = (time.monotonic() - t0) * 1000
        return {"raw_output": raw, "latency_ms": round(latency_ms, 1), "n_input_tokens": None}


class LlamaServerAdapter:
    """LFM 450M vendor-shipped GGUF Q8_0 via llama.cpp llama-server (CPU).

    Starts the pinned llama-server once; each item is one OpenAI-style chat
    request with the image inlined, temperature 0 / top_k 1 (greedy).
    """

    def __init__(self,
                 model_path="/opt/LFM2.5-VL-450M-Q8_0.gguf",
                 mmproj_path="/opt/mmproj-LFM2.5-VL-450m-Q8_0.gguf",
                 server_bin="/usr/local/bin/llama-server",
                 port=8091):
        import os
        import subprocess

        import requests

        self.url = f"http://127.0.0.1:{port}"
        env = {**os.environ,
               "LD_LIBRARY_PATH": "/opt/llama-bin:" + os.environ.get("LD_LIBRARY_PATH", "")}
        self.proc = subprocess.Popen(
            [server_bin, "-m", model_path, "--mmproj", mmproj_path,
             "--host", "127.0.0.1", "--port", str(port), "-c", "8192", "--threads", "8"],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env,
        )
        for _ in range(240):  # wait for model load
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy")

    def generate(self, image, prompt, max_new_tokens, task=None):
        import base64
        import io

        import requests

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0, "top_k": 1, "max_tokens": int(max_new_tokens),
        }
        t0 = time.monotonic()
        r = requests.post(self.url + "/v1/chat/completions", json=payload, timeout=600)
        latency_ms = (time.monotonic() - t0) * 1000
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        return {"raw_output": raw, "latency_ms": round(latency_ms, 1), "n_input_tokens": None}


# ---------------------------------------------------------------------------
# Registry: model key -> constructor. Revisions pinned by the Phase 0 load matrix.
# ---------------------------------------------------------------------------

MODEL_PINS = {
    "lfm450m": {
        "model_id": "LiquidAI/LFM2.5-VL-450M",
        "revision": "fc6221ca597f3315e4f82fc2df606783267b34ba",
        "env": "tf5",
        "adapter": "itt",
    },
    "smolvlm2": {
        "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        "env": "tf457",
        "adapter": "itt",
    },
    "llava_ov": {
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "revision": "74dd0bf867a4cda7950c17663794267c60cf4b40",
        "env": "tf457",
        "adapter": "itt",
    },
    "internvl35": {
        "model_id": "OpenGVLab/InternVL3_5-1B",
        "revision": "2f71cf52542334823e48a46ffba0e2bc9add3446",
        "env": "tf457",
        "adapter": "internvl",
        "trust_remote_code": True,
    },
    "qwen25vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "env": "tf457",
        "adapter": "itt",
    },
    "moondream2": {
        "model_id": "vikhyatk/moondream2",
        "revision": "2025-06-21",
        "resolved_sha": "9a7d4024050840e001defacec2b00727e89149e6",
        "env": "moondream2",
        "adapter": "moondream",
        "trust_remote_code": True,
    },
    # ---- deployed-form lanes (Phase 3B; separate table, never in the bf16 main table)
    "moondream05b_asshipped": {
        "model_id": "vikhyatk/moondream2 (0.5B .mf, branch onnx)",
        "revision": "9dddae84d54db4ac56fe37817aeaeb502ed083e2",
        "env": "md05",
        "adapter": "md05_skills",
        "runtime": "moondream==0.0.6 (OnnxVL/onnxruntime, CPU)",
        "quantization": "int8",
        "dtype": None,
    },
    "lfm450m_gguf": {
        "model_id": "LiquidAI/LFM2.5-VL-450M-GGUF (Q8_0 + Q8_0 mmproj)",
        "revision": "1a3f51cdc26f5d97d5d85cf6901fe5ac7da65a9a",
        "env": "md05",
        "adapter": "llama_server",
        "runtime": "llama.cpp b9860 (CPU)",
        "quantization": "Q8_0",
        "dtype": None,
    },
}

# Sweep budget wiring, fixed by the Gate A knob table (discovery_bundle.json):
#
# lfm450m — TRUE image-token budget. Pin min=max=T with splitting OFF for a
#   monotone single-variable axis. Kwargs verified present at the transformers
#   v5.12.1 release tag (image_processing_lfm2_vl.py L166-225). Budget value T
#   = per-image token count (near-exact; resize rounds to 32px multiples).
#   Shipped default = max 256 / min 64 / splitting ON (the bench cell runs it).
#
# smolvlm2 — DIFFERENT AXIS (resolution/tile dial, tokens quantized in 64/tile
#   steps): budget value N = size.longest_edge pixels ∈ {512,1024,1536,2048};
#   2048 is the shipped default. Applied at from_pretrained per SmolVLM docs.
#
# moondream 0.5B — NO KNOB (fixed 729 tokens); never wired, single point.
BUDGET_WIRING = {
    "lfm450m": lambda t: {
        "template": {"min_image_tokens": int(t), "max_image_tokens": int(t), "do_image_splitting": False}
    },
    "smolvlm2": lambda n: {"pretrained": {"size": {"longest_edge": int(n)}}},
}


def build_adapter(model_key, image_token_budget=None):
    pin = MODEL_PINS[model_key]
    trc = pin.get("trust_remote_code", False)
    if pin["adapter"] == "itt":
        return ITTAdapter(
            pin["model_id"], pin["revision"], trust_remote_code=trc,
            image_token_budget=image_token_budget,
            budget_wiring=BUDGET_WIRING.get(model_key),
        )
    if pin["adapter"] == "internvl":
        return InternVLAdapter(pin["model_id"], pin["revision"], image_token_budget=image_token_budget)
    if pin["adapter"] == "moondream":
        return MoondreamAdapter(pin["model_id"], pin["revision"], image_token_budget=image_token_budget)
    if pin["adapter"] == "md05_skills":
        if image_token_budget is not None:
            raise NotImplementedError("moondream 0.5B runtime exposes no image-token knob")
        return MD05Adapter()
    if pin["adapter"] == "llama_server":
        if image_token_budget is not None:
            raise NotImplementedError("deployed GGUF lane has no image-token budget wiring")
        return LlamaServerAdapter()
    raise ValueError(f"unknown adapter for {model_key}")
