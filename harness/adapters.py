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


class InternVLAdapter:
    """InternVL3_5 remote-code path: AutoModel + model.chat(), 448x448 single tile."""

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

        self.transform = T.Compose(
            [
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def generate(self, image, prompt, max_new_tokens, task=None):
        import torch

        pv = self.transform(image.convert("RGB")).unsqueeze(0).to(torch.bfloat16).cuda()
        t0 = time.monotonic()
        with torch.inference_mode():
            resp = self.model.chat(
                self.tokenizer, pv, "<image>\n" + prompt, dict(max_new_tokens=max_new_tokens, do_sample=False)
            )
        latency_ms = (time.monotonic() - t0) * 1000
        return {"raw_output": resp, "latency_ms": round(latency_ms, 1), "n_input_tokens": None}


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
    raise ValueError(f"unknown adapter for {model_key}")
