"""Output parsers for the edge-VLM benchmark (shared by both parts).

Two submodules:

- :mod:`parsers.boxes`   -- per-model grounding parsers. Each converts raw
  model output into pixel-space ``[x1, y1, x2, y2]`` boxes given the image
  dimensions.
- :mod:`parsers.answers` -- task-level answer parsers for POPE-style yes/no
  questions and counting questions.

Contract (enforced for every public parser):

* pure Python 3.10+ stdlib only,
* deterministic,
* returns ``None`` on unparseable input and NEVER raises.
"""

from parsers.answers import parse_count, parse_yes_no
from parsers.boxes import (
    parse_internvl,
    parse_lfm,
    parse_llava_ov,
    parse_moondream_detect,
    parse_qwen25vl,
    parse_smolvlm2,
)

__all__ = [
    "parse_lfm",
    "parse_qwen25vl",
    "parse_internvl",
    "parse_moondream_detect",
    "parse_smolvlm2",
    "parse_llava_ov",
    "parse_yes_no",
    "parse_count",
]
