"""Tests for parsers.answers (stdlib unittest, no third-party deps).

Covers parse_yes_no (leading-token rule, standalone-segment rule, embedded
tokens rejected, punctuation/case/markdown robustness) and parse_count
(digits, comma grouping, number words zero..twenty, hyphenated compounds,
embedded-word rejection, first-occurrence ordering).
"""

import unittest

from parsers.answers import parse_count, parse_yes_no


class TestParseYesNo(unittest.TestCase):
    """PROVISIONAL POPE yes/no rule (aligned to VLMEvalKit at Gate B)."""

    # ---- Rule A: leading token -------------------------------------------
    def test_plain_yes(self):
        self.assertIs(parse_yes_no("yes"), True)

    def test_plain_no(self):
        self.assertIs(parse_yes_no("no"), False)

    def test_yes_with_period(self):
        self.assertIs(parse_yes_no("Yes."), True)

    def test_uppercase_with_exclamation(self):
        self.assertIs(parse_yes_no("YES!"), True)

    def test_no_with_trailing_clause(self):
        self.assertIs(parse_yes_no("no, it is not"), False)

    def test_yes_with_trailing_clause(self):
        self.assertIs(parse_yes_no("Yes, there is a dog in the image."), True)

    def test_leading_whitespace(self):
        self.assertIs(parse_yes_no("   yes  "), True)

    def test_markdown_bold(self):
        self.assertIs(parse_yes_no("**Yes**, there is."), True)

    def test_quoted_answer(self):
        self.assertIs(parse_yes_no('"No"'), False)

    def test_leading_newline(self):
        self.assertIs(parse_yes_no("\nno\n"), False)

    def test_yes_it_is(self):
        self.assertIs(parse_yes_no("yes it is"), True)

    # ---- Rule B: standalone segment --------------------------------------
    def test_standalone_after_hedge(self):
        self.assertIs(parse_yes_no("Hmm. No."), False)

    def test_standalone_yes_after_comma(self):
        self.assertIs(parse_yes_no("I would say, yes."), True)

    def test_standalone_on_own_line(self):
        self.assertIs(parse_yes_no("Looking closely at the image:\nyes"), True)

    # ---- Rule C: no explicit answer token -> None -------------------------
    def test_embedded_determiner_no_is_none(self):
        # Spec example: "There is no dog" -> None.
        self.assertIsNone(parse_yes_no("There is no dog"))

    def test_mid_sentence_yes_is_none(self):
        # Strict provisional rule: not leading, not standalone.
        self.assertIsNone(parse_yes_no("The answer is yes"))

    def test_hedged_answer_is_none(self):
        self.assertIsNone(parse_yes_no("Maybe, it is hard to tell."))

    def test_unrelated_text_is_none(self):
        self.assertIsNone(parse_yes_no("A cat sitting on a couch."))

    # ---- Word-boundary strictness ----------------------------------------
    def test_yesterday_is_not_yes(self):
        self.assertIsNone(parse_yes_no("Yesterday I saw a dog."))

    def test_nothing_is_not_no(self):
        self.assertIsNone(parse_yes_no("Nothing is visible."))

    def test_nope_is_none(self):
        # Only the exact tokens count under the provisional rule.
        self.assertIsNone(parse_yes_no("Nope"))

    # ---- Degenerate inputs -----------------------------------------------
    def test_empty_string(self):
        self.assertIsNone(parse_yes_no(""))

    def test_whitespace_only(self):
        self.assertIsNone(parse_yes_no("   \n\t "))

    def test_punctuation_only(self):
        self.assertIsNone(parse_yes_no("?!."))

    def test_non_string_inputs(self):
        self.assertIsNone(parse_yes_no(None))
        self.assertIsNone(parse_yes_no(123))
        self.assertIsNone(parse_yes_no(True))
        self.assertIsNone(parse_yes_no(["yes"]))

    def test_returns_bool_not_truthy(self):
        # Contract: exactly True/False/None (identity, not just truthiness).
        self.assertIs(parse_yes_no("Yes."), True)
        self.assertIs(parse_yes_no("No."), False)


class TestParseCount(unittest.TestCase):
    """First integer in the text: digits or words zero..twenty."""

    # ---- Digits ------------------------------------------------------------
    def test_spec_example(self):
        self.assertEqual(parse_count("There are 7 birds"), 7)

    def test_bare_digit(self):
        self.assertEqual(parse_count("3"), 3)

    def test_digit_with_punctuation(self):
        self.assertEqual(parse_count("12."), 12)

    def test_first_of_several_digits(self):
        self.assertEqual(parse_count("about 3 or 4 birds"), 3)

    def test_multi_digit(self):
        self.assertEqual(parse_count("I count 42 apples"), 42)

    def test_comma_grouped_number(self):
        self.assertEqual(parse_count("There are 1,234 people"), 1234)

    def test_comma_list_not_merged(self):
        # "7,8" is a list, not a grouped number: first integer is 7.
        self.assertEqual(parse_count("maybe 7,8 of them"), 7)

    def test_decimal_takes_integer_prefix(self):
        self.assertEqual(parse_count("roughly 3.5 birds"), 3)

    def test_zero_digit(self):
        self.assertEqual(parse_count("0 dogs"), 0)

    # ---- Number words -------------------------------------------------------
    def test_word_seven(self):
        self.assertEqual(parse_count("There are seven birds."), 7)

    def test_word_capitalized(self):
        self.assertEqual(parse_count("Seven birds."), 7)

    def test_word_all_caps(self):
        self.assertEqual(parse_count("TWELVE"), 12)

    def test_word_zero(self):
        self.assertEqual(parse_count("zero"), 0)

    def test_word_twenty(self):
        self.assertEqual(parse_count("I count twenty apples"), 20)

    def test_all_words_zero_to_twenty(self):
        words = [
            "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen", "twenty",
        ]
        for expected, word in enumerate(words):
            with self.subTest(word=word):
                self.assertEqual(parse_count(f"There are {word} things"), expected)

    def test_word_before_digit_wins(self):
        # First occurrence in text order, regardless of kind.
        self.assertEqual(parse_count("seven or 8 birds"), 7)

    def test_digit_before_word_wins(self):
        self.assertEqual(parse_count("8 or maybe seven"), 8)

    # ---- Hyphenated compounds and embedded words ----------------------------
    def test_twenty_one_not_misparsed_as_twenty(self):
        # Out-of-vocabulary compound is skipped entirely.
        self.assertIsNone(parse_count("twenty-one birds"))

    def test_compound_skipped_but_later_number_found(self):
        self.assertEqual(parse_count("twenty-one, or really just 5"), 5)

    def test_embedded_word_canine_is_not_nine(self):
        self.assertIsNone(parse_count("A canine is visible."))

    def test_embedded_word_onesie_is_not_one(self):
        self.assertIsNone(parse_count("A onesie on the bed."))

    def test_atone_is_not_one(self):
        self.assertIsNone(parse_count("They atone for it."))

    # ---- No number -> None ---------------------------------------------------
    def test_no_number_returns_none(self):
        self.assertIsNone(parse_count("There are no birds in the image."))

    def test_word_none_is_not_a_number(self):
        # "none" is outside the zero..twenty vocabulary (spec).
        self.assertIsNone(parse_count("none"))

    def test_article_a_is_not_one(self):
        self.assertIsNone(parse_count("There is a bird."))

    def test_empty_string(self):
        self.assertIsNone(parse_count(""))

    def test_whitespace_only(self):
        self.assertIsNone(parse_count("  \n "))

    def test_non_string_inputs(self):
        self.assertIsNone(parse_count(None))
        self.assertIsNone(parse_count(7))
        self.assertIsNone(parse_count(7.0))
        self.assertIsNone(parse_count([7]))

    def test_returns_int_type(self):
        self.assertIsInstance(parse_count("seven"), int)
        self.assertIsInstance(parse_count("7"), int)


class TestNeverRaisesContract(unittest.TestCase):
    """Both answer parsers must never raise on adversarial input."""

    ADVERSARIAL = [
        None, "", "   ", 0, -1.5, True, b"yes", object(), {"a": 1}, ["no"],
        "\x00\x01", "?" * 1000, "9" * 500,
    ]

    def test_parse_yes_no_never_raises(self):
        for bad in self.ADVERSARIAL:
            with self.subTest(raw=repr(bad)[:40]):
                self.assertIn(parse_yes_no(bad), (True, False, None))

    def test_parse_count_never_raises(self):
        for bad in self.ADVERSARIAL:
            with self.subTest(raw=repr(bad)[:40]):
                result = parse_count(bad)
                self.assertTrue(result is None or isinstance(result, int))


if __name__ == "__main__":
    unittest.main()
