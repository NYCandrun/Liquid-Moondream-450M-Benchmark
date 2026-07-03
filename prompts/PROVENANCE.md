# Prompt template provenance

Placeholders `{expression}` / `{question}` are substituted by simple string
replacement (never str.format), so literal braces in templates are safe.

| template id | source (verbatim where possible) |
|---|---|
| pope_vlmevalkit_v1 | VLMEvalKit v0.3rc1 @ 8eb37bb9 dataset-level convention: POPE receives the RAW question with no suffix — vlmeval/dataset/image_yorn.py L25-32 (the "Please answer yes or no." suffix at L31 applies only to AMBER) + vlmeval/dataset/image_base.py L186-203. Model-side overrides in VLMEvalKit differ per adapter (recorded in METHODOLOGY); we hold the dataset-level convention constant across all models. |
| count_vlmevalkit_v1 | VLMEvalKit v0.3rc1 @ 8eb37bb9, vlmeval/dataset/image_vqa.py L2346 (CountBenchQA.build_prompt), verbatim template. |
| grounding_lfm450m_v1 | LFM2.5-VL-450M model card (sha fc6221ca), verbatim detection prompt; output normalized 0-1 JSON [{"label","bbox"}]. https://huggingface.co/LiquidAI/LFM2.5-VL-450M |
| grounding_qwen25vl_v1 | Qwen2.5-VL official blog grounding convention (bbox_2d, absolute pixels): https://qwenlm.github.io/blog/qwen2.5-vl/ |
| grounding_internvl35_v1 | InternVL grounding eval prompt (verbatim), output <box>[[x1,y1,x2,y2]]</box> normalized 0-1000: https://github.com/OpenGVLab/InternVL/issues/58 ; https://huggingface.co/OpenGVLab/InternVL2-4B |
| grounding_moondream2_v1 | moondream2 native skill: detect(image, object) takes the bare expression; no prompt wrapper. Load matrix skill signature evidence. |
| grounding_smolvlm2_v1 | BEST-EFFORT (no native grounding convention published for SmolVLM2). Expected finding: "no boxes". |
| grounding_llava_ov_v1 | BEST-EFFORT (no native grounding convention published for LLaVA-OneVision-HF 0.5B). Expected finding: "no boxes". |
