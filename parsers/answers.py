"""Task-level answer parsers for the edge-VLM benchmark (POPE + counting).

Two parsers, shared by every model (raw text in, structured answer out):

* :func:`parse_yes_no` -- POPE yes/no extraction, FROZEN at Gate B as an
  exact port of VLMEvalKit v0.3rc1 ``YOrN_Extraction``
  (vlmeval/dataset/utils/yorn.py L226-233) including its
  ``process_punctuation`` normalization (vlmeval/smp/misc.py L32-49).
  VLMEvalKit's LLM-judge fallback for 'Unknown' is intentionally NOT used;
  Unknown -> None (scored as wrong), and parse rate is reported.
* :func:`parse_count`  -- first integer in the text (digits or the English
  number words zero..twenty).

Contract: pure stdlib, deterministic, returns ``None`` on unparseable
input and NEVER raises.
"""

from __future__ import annotations

import functools
import re

__all__ = ["parse_yes_no", "parse_count"]

# Alphabetic word tokens (used for the leading-token yes/no rule).
_WORD_RE = re.compile(r"[a-z]+")

# Segment delimiters for the standalone yes/no rule: sentence and clause
# punctuation plus newlines.
_SEGMENT_SPLIT_RE = re.compile(r"[.,;:!?\n]")

# Characters stripped from a segment before the exact-match test
# (whitespace, quotes, markdown emphasis, brackets/parens).
_SEGMENT_STRIP_CHARS = " \t\r\"'`*_()[]{}"

# English number words zero..twenty (inclusive), per spec.
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

# Tokens scanned by parse_count, in text order:
#   - digit runs, optionally with comma group separators ("1,234"),
#   - alphabetic words, keeping hyphenated compounds together so that
#     "twenty-one" is seen as ONE (out-of-vocabulary) token rather than
#     being misparsed as "twenty".
_COUNT_TOKEN_RE = re.compile(r"\d[\d,]*|[A-Za-z]+(?:-[A-Za-z]+)*")

# Strictly valid comma-grouped integer, e.g. "1,234" or "12,345,678".
_COMMA_GROUPED_RE = re.compile(r"\d{1,3}(?:,\d{3})+$")


def _never_raises(fn):
    """Decorator: guarantee the parser returns None instead of raising."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 -- contract: never raise.
            return None

    return wrapper


def _process_punctuation(in_text):
    """Verbatim port of VLMEvalKit v0.3rc1 process_punctuation
    (vlmeval/smp/misc.py L32-49, commit 8eb37bb9), including its quirks:
    the replace-with-empty vs replace-with-space branch keys off the ORIGINAL
    text, and periods not adjacent to digits are removed."""
    out_text = in_text
    punct = [
        ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
        ">", "<", "@", "`", ",", "?", "!",
    ]
    comma_strip = re.compile(r"(\d)(,)(\d)")
    period_strip = re.compile(r"(?<!\d)\.(?!\d)")
    for p in punct:
        if (p + " " in in_text or " " + p in in_text) or (
            re.search(comma_strip, in_text) is not None
        ):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = period_strip.sub("", out_text, re.UNICODE)
    return out_text


@_never_raises
def parse_yes_no(raw):
    """FROZEN (Gate B) -- exact port of VLMEvalKit v0.3rc1 YOrN_Extraction.

    Pinned source (commit 8eb37bb96bc508bd016000318548854fdfa1bd30),
    vlmeval/dataset/utils/yorn.py L226-233::

        s = output.lower()
        words = process_punctuation(s).split()
        if 'yes' in words and 'no' not in words: return 'Yes'
        if 'yes' not in words and 'no' in words: return 'No'
        return 'Unknown'

    Mapping: 'Yes' -> True, 'No' -> False, 'Unknown' -> None. VLMEvalKit's
    LLM-judge fallback for Unknown is NOT used (no LLM in the scoring path);
    Unknown items score as wrong and parse rate is reported per cell.

    Consequences of the word-presence rule (all VLMEvalKit-faithful):
    "There is no car in the image" -> False; "The answer is yes" -> True;
    "The image does not show a car" -> None ('not' is not 'no');
    "yes... no... maybe" -> None (ambiguous); "yesterday" -> None.

    Returns True, False, or None. Never raises.
    """
    if not isinstance(raw, str):
        return None
    s = raw.lower()
    words = _process_punctuation(s).split()
    if "yes" in words and "no" not in words:
        return True
    if "yes" not in words and "no" in words:
        return False
    return None


@_never_raises
def parse_count(raw):
    """Extract the first integer from raw model text, else None.

    Scans tokens in text order and returns the first that is:

    * a digit run: "There are 7 birds" -> 7. Comma group separators are
      honored when strictly valid ("1,234" -> 1234); otherwise only the
      leading digits before the comma are used ("7,8" -> 7).
    * an English number word from zero..twenty (case-insensitive):
      "Seven birds." -> 7, "twenty apples" -> 20.

    Details / edge rules:

    * Hyphenated compounds are kept whole, so out-of-vocabulary compounds
      like "twenty-one" are SKIPPED rather than misparsed as 20 (the scan
      continues; if no other number appears the result is None).
    * Number words embedded in longer words never match ("canine" is not
      9; "onesie" is not 1).
    * Decimals contribute their integer prefix ("3.5" -> 3), matching the
      "first integer in the text" spec.
    * No number anywhere -> None. Non-string input -> None.

    Returns int or None. Never raises.
    """
    if not isinstance(raw, str):
        return None

    for match in _COUNT_TOKEN_RE.finditer(raw):
        token = match.group(0)
        if token[0].isdigit():
            if "," in token:
                if _COMMA_GROUPED_RE.match(token):
                    return int(token.replace(",", ""))
                # Not valid grouping (e.g. "7,8"): take the leading run.
                return int(re.match(r"\d+", token).group(0))
            return int(token)
        # Alphabetic token (hyphenated compounds arrive whole).
        value = _NUMBER_WORDS.get(token.lower())
        if value is not None:
            return value

    return None
