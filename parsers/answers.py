"""Task-level answer parsers for the edge-VLM benchmark (POPE + counting).

Two parsers, shared by every model (raw text in, structured answer out):

* :func:`parse_yes_no` -- POPE-style yes/no extraction (PROVISIONAL rule,
  to be aligned to the pinned VLMEvalKit POPE convention at Gate B).
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


@_never_raises
def parse_yes_no(raw):
    """PROVISIONAL -- extract a yes/no answer from raw model text.

    PROVISIONAL: this rule will be aligned to the pinned VLMEvalKit POPE
    convention at Gate B; until then the exact rule is as follows
    (case-insensitive, robust to punctuation and markdown emphasis):

    Rule A (leading token): if the FIRST alphabetic word token of the text
        is exactly "yes" -> True, exactly "no" -> False.
        Examples: "Yes."; "**No**, there is not"; "yes it is".
        Word-boundary strict: "yesterday"/"nothing" do NOT match.

    Rule B (standalone segment): otherwise, split the text on sentence /
        clause punctuation (. , ; : ! ? and newlines) and return the first
        segment that, after stripping quotes/brackets/emphasis, is EXACTLY
        "yes" or "no". Example: "Hmm. No." -> False.

    Rule C: otherwise None. In particular, an embedded token is NOT an
        answer: "There is no dog" -> None (determiner "no"), and
        "The answer is yes" -> None (mid-segment "yes"). This strictness is
        intentional pending the Gate B alignment.

    Returns True, False, or None. Never raises.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip().lower()
    if not text:
        return None

    # Rule A: leading alphabetic token.
    words = _WORD_RE.findall(text)
    if words:
        if words[0] == "yes":
            return True
        if words[0] == "no":
            return False

    # Rule B: a punctuation-delimited segment that IS the answer.
    for segment in _SEGMENT_SPLIT_RE.split(text):
        stripped = segment.strip(_SEGMENT_STRIP_CHARS)
        if stripped == "yes":
            return True
        if stripped == "no":
            return False

    # Rule C: no explicit answer token.
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
