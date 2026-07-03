# Prompt template provenance

Placeholders `{expression}` / `{question}` are substituted by simple string
replacement (never str.format), so literal braces in templates are safe.

| template id | source (verbatim where possible) |
|---|---|
| pope_vlmevalkit_v1 | Bare raw question, held constant across all models. ORIGINAL provenance claim (audit of the tool's CODE): VLMEvalKit v0.3rc1 @ 8eb37bb9 adds no POPE suffix — vlmeval/dataset/image_yorn.py L25-32 ("Please answer yes or no." at L31 applies only to AMBER). CORRECTION (2026-07-03, referee lane finding): the tool's distributed POPE.tsv DATA FILE ships with " Please answer yes or no." already baked into every question string, so the tool's EFFECTIVE convention is question+instruction even though its code adds nothing; our lane used the raw HF dataset question (bare). Measured effect on SmolVLM2: 55.0% (bare, ours) vs 85.8% (suffixed, referee). Disposition pending the remaining referee POPE cells (see METHODOLOGY change log). |
| pope_vlmevalkit_v2 | Suffixed convention: `{question} Please answer yes or no.` — adopts the effective standard POPE protocol shipped in VLMEvalKit's POPE.tsv (the " Please answer yes or no." instruction baked into every question, see v1 correction above). Introduced 2026-07-03 (owner directive) to align the yes/no headline with the documented benchmark protocol; the v1 bare-question run is RETAINED as the measured prompt-convention finding (change #11), not discarded. Answer parser unchanged (still the exact VLMEvalKit YOrN port), so this run isolates the single prompt variable. |
| count_vlmevalkit_v1 | VLMEvalKit v0.3rc1 @ 8eb37bb9, vlmeval/dataset/image_vqa.py L2346 (CountBenchQA.build_prompt), verbatim template. |
| grounding_lfm450m_v1 | LFM2.5-VL-450M model card (sha fc6221ca), verbatim detection prompt; output normalized 0-1 JSON [{"label","bbox"}]. https://huggingface.co/LiquidAI/LFM2.5-VL-450M |
| grounding_qwen25vl_v1 | Qwen2.5-VL official blog grounding convention (bbox_2d, absolute pixels): https://qwenlm.github.io/blog/qwen2.5-vl/ |
| grounding_internvl35_v1 | InternVL grounding eval prompt (verbatim), output <box>[[x1,y1,x2,y2]]</box> normalized 0-1000: https://github.com/OpenGVLab/InternVL/issues/58 ; https://huggingface.co/OpenGVLab/InternVL2-4B |
| grounding_moondream2_v1 | moondream2 native skill: detect(image, object) takes the bare expression; no prompt wrapper. Load matrix skill signature evidence. |
| grounding_smolvlm2_v1 | BEST-EFFORT (no native grounding convention published for SmolVLM2). Expected finding: "no boxes". |
| grounding_llava_ov_v1 | BEST-EFFORT (no native grounding convention published for LLaVA-OneVision-HF 0.5B). Expected finding: "no boxes". |
