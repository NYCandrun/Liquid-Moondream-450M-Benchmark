"""Per-model bounding-box parsers for the edge-VLM benchmark grounding task.

Every parser in this module converts a raw model output into a list of
pixel-space boxes ``[x1, y1, x2, y2]`` (floats), given the image dimensions
``(image_width, image_height)`` the model saw.

Shared return contract
----------------------
* ``list[list[float]]`` -- one entry per valid box, pixel coordinates,
  clamped to ``[0, image_width] x [0, image_height]``.
* ``[]``   -- the output contained a recognizable *box structure* (a known
  bbox key, an explicit empty list, or coordinate quadruples) but no box
  survived validation (e.g. every box was degenerate after clamping).
* ``None`` -- no parseable box structure could be found at all.

"Structure" is defined precisely as any of:
  1. a decoded JSON dict containing a recognized bbox key (``bbox``,
     ``bbox_2d``, ``box``, ``boxes`` -- per-parser subsets apply), even if
     the value under that key is empty or malformed;
  2. a quadruple of four finite numbers (bare, nested in lists, or under a
     recognized key);
  3. an explicit empty top-level JSON list (the model saying "no boxes").
Arbitrary JSON with none of the above (e.g. ``{"caption": "a dog"}``) does
NOT count as structure and yields ``None``, so downstream scoring can
distinguish "parsed: found nothing" from "unparseable".

Validation applied by every parser (after any scaling to pixels):
* coordinates clamped to image bounds,
* boxes with ``x2 <= x1`` or ``y2 <= y1`` (post-clamp) are dropped,
* non-finite coordinates (NaN/Inf -- which Python's ``json`` module happily
  decodes) are rejected.

Every public parser is wrapped so it NEVER raises: any unexpected internal
error returns ``None``. All parsing is deterministic. Invalid image
dimensions (non-positive, non-numeric, or non-finite) yield ``None`` since
scaling/clamping would be meaningless.

PROVISIONAL parsers
-------------------
``parse_internvl``, ``parse_smolvlm2`` and ``parse_llava_ov`` are
PROVISIONAL: their exact raw output format has not yet been captured from
the real model. In Phase 2 they are re-tested against captured raw outputs
and only freeze after human approval. See each docstring.
"""

from __future__ import annotations

import functools
import json
import math
import re

__all__ = [
    "parse_lfm",
    "parse_qwen25vl",
    "parse_internvl",
    "parse_moondream_detect",
    "parse_smolvlm2",
    "parse_llava_ov",
]

# --------------------------------------------------------------------------
# Recognized bbox key names per model family.
# --------------------------------------------------------------------------
_LFM_KEYS = ("bbox", "box", "boxes")
_QWEN_KEYS = ("bbox_2d", "bbox")
_GENERIC_KEYS = ("bbox", "bbox_2d", "box", "boxes")

# Markdown code fence: ```lang\n ... ``` (language tag optional).
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)

# A signed decimal number (optionally scientific notation).
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# InternVL-style quadruple inside square brackets: [x1, y1, x2, y2].
_SQUARE_QUAD_RE = re.compile(
    r"\[\s*({n})\s*,\s*({n})\s*,\s*({n})\s*,\s*({n})\s*\]".format(n=_NUM)
)

# Generic bracketed quadruple: square brackets OR parentheses.
_BRACKET_QUAD_RE = re.compile(
    r"[\[(]\s*({n})\s*,\s*({n})\s*,\s*({n})\s*,\s*({n})\s*[\])]".format(n=_NUM)
)

# InternVL <box>...</box> tag (tolerates surrounding <ref>...</ref>, which
# is simply ignored because we only extract from the <box> contents).
_BOX_TAG_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL | re.IGNORECASE)

# <box> content that is an explicit empty list: "", "[]", "[[]]".
_EMPTY_BOX_CONTENT_RE = re.compile(r"\s*(?:\[\s*\]|\[\s*\[\s*\]\s*\])?\s*$")

# Embedded-JSON scan caps. Without them the per-position raw_decode scan is
# O(n^2) on degenerate outputs (e.g. a model emitting thousands of unclosed
# brackets: "[[[[[..." took >30 s at 20k chars). Worst-case work is bounded
# by attempts * window ~= 1e6 chars (a few ms). Realistic outputs sit far
# below both caps: a whole valid JSON payload decodes in ONE attempt, and
# attempts only accumulate across separate embedded '['/'{' starts.
_MAX_DECODE_ATTEMPTS = 100  # max raw_decode attempts per segment
_MAX_VALUE_CHARS = 10000    # max chars a single embedded JSON value may span


# --------------------------------------------------------------------------
# Internal helpers.
# --------------------------------------------------------------------------
def _never_raises(fn):
    """Decorator: guarantee the parser returns None instead of raising.

    Careful input validation should make this a no-op; it exists as a hard
    backstop for the module contract ("NEVER raises").
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 -- contract: never raise.
            return None

    return wrapper


def _valid_dims(width, height) -> bool:
    """True iff both image dimensions are finite positive real numbers."""
    for dim in (width, height):
        if isinstance(dim, bool) or not isinstance(dim, (int, float)):
            return False
        if not math.isfinite(dim) or dim <= 0:
            return False
    return True


def _as_quad(value):
    """Return [f, f, f, f] if `value` is a list of exactly 4 finite numbers.

    Booleans are excluded (bool is a subclass of int in Python) and
    NaN/Infinity are rejected. Returns None otherwise.
    """
    if not isinstance(value, list) or len(value) != 4:
        return None
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if not math.isfinite(v):
            return None
        out.append(float(v))
    return out


def _scan_segment(segment: str) -> list:
    """Decode every JSON value found in a text segment.

    First tries the whole (stripped) segment; failing that, scans for
    embedded values starting at each '{' or '[' using raw_decode, skipping
    past each successful decode so nested values are not double-counted.

    The scan is bounded by ``_MAX_DECODE_ATTEMPTS`` decode attempts and
    ``_MAX_VALUE_CHARS`` chars per candidate value, so pathological inputs
    (bracket floods) cannot cause quadratic blowup.
    """
    stripped = segment.strip()
    if not stripped:
        return []
    try:
        return [json.loads(stripped)]
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    values = []
    i = 0
    attempts = 0
    while i < len(segment) and attempts < _MAX_DECODE_ATTEMPTS:
        if segment[i] in "[{":
            attempts += 1
            window = segment[i : i + _MAX_VALUE_CHARS]
            try:
                value, end = decoder.raw_decode(window)
            except ValueError:
                i += 1
                continue
            values.append(value)
            i += end
        else:
            i += 1
    return values


def _json_candidates(text: str) -> list:
    """All JSON values decodable from `text`, fenced blocks first.

    Fenced code blocks are decoded separately and then *removed* from the
    remainder scan so the same value is never extracted twice.
    """
    fenced = _FENCE_RE.findall(text)
    remainder = _FENCE_RE.sub(" ", text)
    values = []
    for segment in fenced + [remainder]:
        values.extend(_scan_segment(segment))
    return values


def _quads_from_value(value, keys, bare_ok):
    """Recursively extract coordinate quadruples from a decoded JSON value.

    Returns ``(quads, structure)`` where `structure` reports whether box
    structure (per the module-docstring definition) was seen.

    `bare_ok` controls whether a bare 4-number list counts as a box here:
    True at the top level and under recognized keys; False when recursing
    into *unrecognized* dict values, so that e.g. {"size": [0, 0, 4, 4]}
    is not misread as a box.
    """
    quad = _as_quad(value)
    if quad is not None:
        return ([quad], True) if bare_ok else ([], False)

    quads = []
    structure = False
    if isinstance(value, dict):
        for k, v in value.items():
            if k in keys:
                # A recognized bbox key is structure even if its value is
                # empty/malformed: the model attempted the box format.
                structure = True
                sub, _ = _quads_from_value(v, keys, bare_ok=True)
                quads.extend(sub)
            else:
                sub, sub_structure = _quads_from_value(v, keys, bare_ok=False)
                quads.extend(sub)
                structure = structure or sub_structure
    elif isinstance(value, list):
        if not value:
            # An explicit empty list where a bare box (list) is acceptable
            # means "no boxes found" -- that is parsed structure.
            structure = bare_ok
        for item in value:
            sub, sub_structure = _quads_from_value(item, keys, bare_ok)
            quads.extend(sub)
            structure = structure or sub_structure
    return quads, structure


def _extract_quads(raw, keys):
    """Extract all quadruples + structure flag from a raw string via JSON."""
    if not isinstance(raw, str) or not raw.strip():
        return [], False
    quads = []
    structure = False
    for value in _json_candidates(raw):
        sub, sub_structure = _quads_from_value(value, keys, bare_ok=True)
        quads.extend(sub)
        structure = structure or sub_structure
    return quads, structure


def _clamp_boxes(boxes, width, height):
    """Clamp pixel boxes to image bounds; drop degenerate boxes.

    A box is degenerate when, after clamping, x2 <= x1 or y2 <= y1 (this
    also removes boxes that lie entirely outside the image).
    """
    out = []
    for x1, y1, x2, y2 in boxes:
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([float(x1), float(y1), float(x2), float(y2)])
    return out


def _to_float(value):
    """Coerce a JSON-ish scalar to a finite float, else None.

    Accepts int/float (not bool) and numeric strings; rejects NaN/Inf.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------
# Public parsers.
# --------------------------------------------------------------------------
@_never_raises
def parse_lfm(raw, image_width, image_height):
    """Parse LFM2.5-VL grounding output (normalized 0-1 JSON coordinates).

    LFM2.5-VL emits coordinates normalized to [0, 1] in JSON. Accepted
    shapes (possibly wrapped in prose and/or markdown code fences):

    * ``{"bbox": [x1, y1, x2, y2]}``            (also "box"/"boxes" keys)
    * ``[{"bbox": [...]}, {"bbox": [...]}]``
    * ``[[x1, y1, x2, y2], ...]``
    * bare ``[x1, y1, x2, y2]``

    Coordinates are scaled to pixels (x * width, y * height) and clamped;
    values outside [0, 1] are trusted as normalized and simply clamp to the
    image after scaling.

    Returns list of pixel boxes, [] (structure but no valid box), or None
    (no parseable structure). Never raises.
    """
    if not _valid_dims(image_width, image_height):
        return None
    quads, structure = _extract_quads(raw, _LFM_KEYS)
    if not structure:
        return None
    boxes = [
        [x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height]
        for x1, y1, x2, y2 in quads
    ]
    return _clamp_boxes(boxes, image_width, image_height)


@_never_raises
def parse_qwen25vl(raw, image_width, image_height):
    """Parse Qwen2.5-VL grounding output (ABSOLUTE pixel coordinates).

    Qwen2.5-VL emits absolute pixel coordinates, typically as JSON objects
    with a ``"bbox_2d"`` key inside a fenced ```json block::

        [{"bbox_2d": [x1, y1, x2, y2], "label": "dog"}, ...]

    Also accepted: a plain "bbox" key, bare arrays ``[x1, y1, x2, y2]`` /
    ``[[...], ...]``, and prose- or fence-wrapped JSON. No coordinate
    scaling is applied (already pixels); boxes are clamped to the image.

    Returns list of pixel boxes, [] (structure but no valid box), or None
    (no parseable structure). Never raises.
    """
    if not _valid_dims(image_width, image_height):
        return None
    quads, structure = _extract_quads(raw, _QWEN_KEYS)
    if not structure:
        return None
    return _clamp_boxes(quads, image_width, image_height)


@_never_raises
def parse_internvl(raw, image_width, image_height):
    """PROVISIONAL -- parse InternVL grounding output (0-1000 normalized).

    PROVISIONAL: the exact output format has NOT been captured from the
    real model yet. This parser encodes the documented InternVL convention
    and will be re-tested against captured raw outputs in Phase 2; it only
    freezes after human approval.

    Expected convention: ``<ref>label</ref><box>[[x1, y1, x2, y2]]</box>``
    with coordinates normalized to 0-1000. Handling:

    * all quadruples inside every ``<box>...</box>`` tag are extracted
      (``<ref>`` prefixes are ignored);
    * if no ``<box>`` tag is present, bracketed quadruples anywhere in the
      text are used as a fallback (still treated as 0-1000);
    * coordinates are scaled by ``value / 1000 * dimension`` and clamped;
    * a ``<box>`` tag containing only ``[]``/``[[]]``/nothing counts as
      parsed structure with zero boxes -> [].

    Returns list of pixel boxes, [] (structure but no valid box), or None
    (no parseable structure). Never raises.
    """
    if not _valid_dims(image_width, image_height):
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None

    tag_contents = _BOX_TAG_RE.findall(raw)
    raw_quads = []
    saw_empty_tag = False
    if tag_contents:
        for content in tag_contents:
            found = _SQUARE_QUAD_RE.findall(content)
            if found:
                raw_quads.extend(found)
            elif _EMPTY_BOX_CONTENT_RE.match(content):
                saw_empty_tag = True
    else:
        # Fallback: bare bracketed quadruples without <box> tags.
        raw_quads = _SQUARE_QUAD_RE.findall(raw)

    if not raw_quads:
        return [] if saw_empty_tag else None

    boxes = []
    for x1, y1, x2, y2 in raw_quads:
        boxes.append(
            [
                float(x1) / 1000.0 * image_width,
                float(y1) / 1000.0 * image_height,
                float(x2) / 1000.0 * image_width,
                float(y2) / 1000.0 * image_height,
            ]
        )
    return _clamp_boxes(boxes, image_width, image_height)


@_never_raises
def parse_moondream_detect(objects, image_width, image_height):
    """Parse the return value of Moondream's ``detect()`` API.

    Input is NOT raw text: it is the structured ``detect()`` return value,
    a list of dicts with normalized [0, 1] float keys ``x_min``, ``y_min``,
    ``x_max``, ``y_max``. For convenience the common wrapper form
    ``{"objects": [...]}`` (returned by some Moondream revisions) is
    unwrapped automatically.

    Handling:

    * an empty list means "no detections" -> [];
    * items that are not dicts or lack any of the four numeric keys are
      skipped; if the list is non-empty but NO item is well-formed, the
      input is considered unparseable -> None;
    * numeric strings are tolerated for the coordinate values;
    * coordinates are scaled to pixels and clamped; degenerate boxes are
      dropped (possibly yielding []).

    Returns list of pixel boxes, [] or None. Never raises.
    """
    if not _valid_dims(image_width, image_height):
        return None
    if isinstance(objects, dict) and isinstance(objects.get("objects"), list):
        objects = objects["objects"]
    if not isinstance(objects, list):
        return None
    if not objects:
        return []

    boxes = []
    well_formed = 0
    for item in objects:
        if not isinstance(item, dict):
            continue
        coords = [_to_float(item.get(k)) for k in ("x_min", "y_min", "x_max", "y_max")]
        if any(c is None for c in coords):
            continue
        well_formed += 1
        boxes.append(
            [
                coords[0] * image_width,
                coords[1] * image_height,
                coords[2] * image_width,
                coords[3] * image_height,
            ]
        )
    if well_formed == 0:
        return None
    return _clamp_boxes(boxes, image_width, image_height)


def _parse_generic(raw, image_width, image_height):
    """Shared best-effort parser for models with uncaptured formats.

    Strategy (documented ambiguity -- Phase 2 captured outputs decide the
    real format; "no boxes" is an acceptable Phase 2 finding):

    1. Try JSON extraction (keys: bbox / bbox_2d / box / boxes; bare
       arrays; prose- and fence-wrapped). If box structure is found, use
       those quadruples (possibly zero -> []).
    2. Otherwise fall back to bracketed number quadruples in the text:
       ``[a, b, c, d]`` or ``(a, b, c, d)``.
    3. Normalized-vs-absolute heuristic, pooled over ALL extracted
       coordinate values:
       - if every value <= 1.0 -> treat as normalized 0-1 and scale by the
         image dimensions (note: ``[0, 0, 1, 1]`` is therefore read as the
         full image, which is ambiguous for a genuine 1-pixel box);
       - otherwise -> treat as ABSOLUTE pixels with NO scaling. This is
         explicitly ambiguous for 0-1000-normalized conventions on images
         larger than 1000 px: such values are <= 1000 but we still treat
         them as absolute pixels, per spec, until Phase 2 captures settle
         the true convention.

    Returns list of pixel boxes, [] or None. Never raises (callers wrap).
    """
    if not _valid_dims(image_width, image_height):
        return None
    quads, structure = _extract_quads(raw, _GENERIC_KEYS)
    if not structure:
        if not isinstance(raw, str):
            return None
        quads = [
            tuple(float(g) for g in m.groups())
            for m in _BRACKET_QUAD_RE.finditer(raw)
        ]
        if not quads:
            return None
    if not quads:
        return []  # Structure parsed (e.g. "[]" or {"bbox": []}) but no boxes.

    values = [v for quad in quads for v in quad]
    if all(v <= 1.0 for v in values):
        boxes = [
            [x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height]
            for x1, y1, x2, y2 in quads
        ]
    else:
        boxes = [list(quad) for quad in quads]
    return _clamp_boxes(boxes, image_width, image_height)


@_never_raises
def parse_smolvlm2(raw, image_width, image_height):
    """PROVISIONAL -- best-effort parser for SmolVLM2 grounding output.

    PROVISIONAL: SmolVLM2's real grounding output format has NOT been
    captured yet. This parser is a documented best-effort (JSON boxes, then
    bracketed quadruples, with an explicit normalized-vs-absolute
    heuristic -- see ``_parse_generic``). In Phase 2 it is re-tested
    against captured raw outputs and only freezes after human approval;
    "SmolVLM2 emits no usable boxes" is an acceptable Phase 2 finding.

    Returns list of pixel boxes, [] or None. Never raises.
    """
    return _parse_generic(raw, image_width, image_height)


@_never_raises
def parse_llava_ov(raw, image_width, image_height):
    """PROVISIONAL -- best-effort parser for LLaVA-OneVision grounding output.

    PROVISIONAL: LLaVA-OneVision's real grounding output format has NOT
    been captured yet. This parser is a documented best-effort (JSON boxes,
    then bracketed quadruples, with an explicit normalized-vs-absolute
    heuristic -- see ``_parse_generic``). In Phase 2 it is re-tested
    against captured raw outputs and only freezes after human approval;
    "LLaVA-OneVision emits no usable boxes" is an acceptable Phase 2
    finding.

    Returns list of pixel boxes, [] or None. Never raises.
    """
    return _parse_generic(raw, image_width, image_height)
