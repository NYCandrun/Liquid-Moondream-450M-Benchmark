# Parsers verification (adversarial pass, 2026-07-02)

Checked: line-by-line read of boxes.py / answers.py against spec; PROVISIONAL labels present on
parse_internvl / parse_smolvlm2 / parse_llava_ov / parse_yes_no (docstrings + module docstring);
never-raise contract fuzzed across ~40 adversarial inputs x 10 dim combos x 6 parsers (bytes,
unicode, NaN/Inf JSON, bool coords, nested junk, negative/inverted boxes, bracket floods, 100k-char
strings) plus moondream structured-input fuzz; clamping/degenerate-drop and None-vs-[] boundary
spot-checked; scaling verified per model (LFM 0-1, Qwen absolute, InternVL /1000, generic heuristic);
fence dedup confirmed; stdlib-only confirmed. FIXED: quadratic blowup in _scan_segment — 20k-char
bracket flood ("[[[[...") took >30 s; added _MAX_DECODE_ATTEMPTS=100 / _MAX_VALUE_CHARS=10000 caps
plus regression test test_bracket_flood_is_fast (now <1 s). Tests: 165/165 pass (164 + 1 new).
