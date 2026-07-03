"""Tests for parsers.boxes (stdlib unittest, no third-party deps).

Coverage per parser: valid inputs in multiple shapes, prose-wrapped JSON,
markdown code fences, garbage / empty / non-string input -> None, clamping
to image bounds, degenerate-box rejection ([] vs None semantics), and the
normalized-vs-absolute heuristic for the PROVISIONAL generic parsers.
"""

import unittest

from parsers.boxes import (
    parse_internvl,
    parse_lfm,
    parse_llava_ov,
    parse_moondream_detect,
    parse_qwen25vl,
    parse_smolvlm2,
)

# Default test image dimensions.
W, H = 640, 480


def _rounded(boxes, ndigits=4):
    """Round nested box floats for stable equality checks."""
    return [[round(v, ndigits) for v in box] for box in boxes]


class TestParseLFM(unittest.TestCase):
    """LFM2.5-VL: normalized 0-1 JSON coordinates, scaled to pixels."""

    def test_bare_quad(self):
        result = parse_lfm("[0.1, 0.2, 0.5, 0.6]", 100, 200)
        self.assertEqual(_rounded(result), [[10.0, 40.0, 50.0, 120.0]])

    def test_dict_bbox_key(self):
        result = parse_lfm('{"bbox": [0.0, 0.0, 1.0, 1.0]}', W, H)
        self.assertEqual(_rounded(result), [[0.0, 0.0, 640.0, 480.0]])

    def test_dict_box_key_alias(self):
        result = parse_lfm('{"box": [0.25, 0.25, 0.75, 0.75]}', 400, 400)
        self.assertEqual(_rounded(result), [[100.0, 100.0, 300.0, 300.0]])

    def test_list_of_objects(self):
        raw = '[{"bbox": [0.1, 0.1, 0.2, 0.2]}, {"bbox": [0.5, 0.5, 0.9, 0.9]}]'
        result = parse_lfm(raw, 100, 100)
        self.assertEqual(
            _rounded(result),
            [[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 90.0, 90.0]],
        )

    def test_list_of_quads(self):
        raw = "[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]"
        result = parse_lfm(raw, 200, 200)
        self.assertEqual(
            _rounded(result),
            [[0.0, 0.0, 100.0, 100.0], [100.0, 100.0, 200.0, 200.0]],
        )

    def test_prose_wrapped_json(self):
        raw = 'Sure! The dog is at {"bbox": [0.1, 0.1, 0.2, 0.2]} in the image.'
        result = parse_lfm(raw, 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 10.0, 20.0, 20.0]])

    def test_code_fence_with_language(self):
        raw = '```json\n{"bbox": [0.1, 0.2, 0.3, 0.4]}\n```'
        result = parse_lfm(raw, 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 20.0, 30.0, 40.0]])

    def test_code_fence_without_language(self):
        raw = "```\n[[0.1, 0.2, 0.3, 0.4]]\n```"
        result = parse_lfm(raw, 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 20.0, 30.0, 40.0]])

    def test_fenced_json_not_double_counted(self):
        # The fenced block is removed from the remainder scan, so exactly
        # one box comes back, not two copies.
        raw = 'Here you go:\n```json\n{"bbox": [0.1, 0.1, 0.2, 0.2]}\n```\nDone.'
        result = parse_lfm(raw, 100, 100)
        self.assertEqual(len(result), 1)

    def test_nested_objects_wrapper(self):
        raw = '{"objects": [{"bbox": [0.1, 0.1, 0.4, 0.4]}]}'
        result = parse_lfm(raw, 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 10.0, 40.0, 40.0]])

    def test_unrecognized_key_bare_quad_not_a_box(self):
        # A 4-number list under an unrecognized key is not a box.
        self.assertIsNone(parse_lfm('{"size": [0, 0, 1, 1]}', W, H))

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_lfm("I cannot locate any object here.", W, H))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_lfm("", W, H))

    def test_whitespace_only_returns_none(self):
        self.assertIsNone(parse_lfm("   \n\t ", W, H))

    def test_non_string_input_returns_none(self):
        self.assertIsNone(parse_lfm(None, W, H))
        self.assertIsNone(parse_lfm(12345, W, H))
        self.assertIsNone(parse_lfm(["not", "a", "string"], W, H))

    def test_json_without_box_structure_returns_none(self):
        self.assertIsNone(parse_lfm('{"caption": "a dog on grass"}', W, H))

    def test_citation_like_list_is_not_structure(self):
        # "[1]" decodes as JSON but is not box structure.
        self.assertIsNone(parse_lfm("I see nothing relevant [1].", W, H))

    def test_empty_top_level_list_returns_empty(self):
        self.assertEqual(parse_lfm("[]", W, H), [])

    def test_empty_bbox_value_returns_empty(self):
        self.assertEqual(parse_lfm('{"bbox": []}', W, H), [])

    def test_clamping_out_of_range_normalized(self):
        result = parse_lfm("[-0.1, -0.2, 1.2, 0.5]", 100, 100)
        self.assertEqual(_rounded(result), [[0.0, 0.0, 100.0, 50.0]])

    def test_degenerate_box_dropped_to_empty(self):
        # x2 == x1 after scaling -> dropped; structure parsed -> [].
        self.assertEqual(parse_lfm('{"bbox": [0.5, 0.5, 0.5, 0.9]}', W, H), [])

    def test_inverted_box_dropped(self):
        self.assertEqual(parse_lfm("[0.8, 0.8, 0.2, 0.2]", W, H), [])

    def test_mixed_valid_and_degenerate(self):
        raw = "[[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.5, 0.5]]"
        result = parse_lfm(raw, 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 10.0, 20.0, 20.0]])

    def test_nan_coordinates_rejected(self):
        # Python's json module decodes NaN; the quad must be rejected.
        # The recognized "bbox" key still counts as structure -> [].
        self.assertEqual(parse_lfm('{"bbox": [NaN, 0, 1, 1]}', W, H), [])

    def test_boolean_coordinates_rejected(self):
        self.assertIsNone(parse_lfm("[true, false, true, true]", W, H))

    def test_invalid_dimensions_return_none(self):
        self.assertIsNone(parse_lfm("[0.1, 0.2, 0.3, 0.4]", 0, 100))
        self.assertIsNone(parse_lfm("[0.1, 0.2, 0.3, 0.4]", 100, -5))
        self.assertIsNone(parse_lfm("[0.1, 0.2, 0.3, 0.4]", None, 100))


class TestParseQwen25VL(unittest.TestCase):
    """Qwen2.5-VL: absolute pixel coordinates, no scaling."""

    def test_typical_fenced_output(self):
        raw = '```json\n[\n  {"bbox_2d": [10, 20, 300, 400], "label": "dog"}\n]\n```'
        result = parse_qwen25vl(raw, W, H)
        self.assertEqual(_rounded(result), [[10.0, 20.0, 300.0, 400.0]])

    def test_no_scaling_applied(self):
        # Same absolute coords must come back regardless of image size.
        raw = '{"bbox_2d": [10, 20, 30, 40]}'
        small = parse_qwen25vl(raw, 100, 100)
        large = parse_qwen25vl(raw, 4000, 4000)
        self.assertEqual(_rounded(small), _rounded(large))
        self.assertEqual(_rounded(small), [[10.0, 20.0, 30.0, 40.0]])

    def test_bare_array(self):
        result = parse_qwen25vl("[10, 20, 30, 40]", W, H)
        self.assertEqual(_rounded(result), [[10.0, 20.0, 30.0, 40.0]])

    def test_list_of_quads(self):
        raw = "[[0, 0, 100, 100], [200, 200, 300, 300]]"
        result = parse_qwen25vl(raw, W, H)
        self.assertEqual(
            _rounded(result),
            [[0.0, 0.0, 100.0, 100.0], [200.0, 200.0, 300.0, 300.0]],
        )

    def test_multiple_objects_with_labels(self):
        raw = (
            '[{"bbox_2d": [1, 2, 50, 60], "label": "cat"},'
            ' {"bbox_2d": [100, 100, 200, 200], "label": "dog"}]'
        )
        result = parse_qwen25vl(raw, W, H)
        self.assertEqual(len(result), 2)

    def test_bbox_key_alias(self):
        result = parse_qwen25vl('{"bbox": [5, 5, 50, 50]}', W, H)
        self.assertEqual(_rounded(result), [[5.0, 5.0, 50.0, 50.0]])

    def test_prose_wrapped(self):
        raw = 'The object is located at {"bbox_2d": [10, 10, 20, 20]}.'
        result = parse_qwen25vl(raw, W, H)
        self.assertEqual(_rounded(result), [[10.0, 10.0, 20.0, 20.0]])

    def test_clamped_to_image_bounds(self):
        result = parse_qwen25vl("[600, 400, 700, 500]", W, H)
        self.assertEqual(_rounded(result), [[600.0, 400.0, 640.0, 480.0]])

    def test_box_fully_outside_image_dropped(self):
        # Clamps to zero-area at the image edge -> degenerate -> [].
        self.assertEqual(parse_qwen25vl("[700, 500, 800, 600]", W, H), [])

    def test_degenerate_box_dropped(self):
        self.assertEqual(parse_qwen25vl('{"bbox_2d": [50, 50, 50, 90]}', W, H), [])

    def test_empty_list_returns_empty(self):
        self.assertEqual(parse_qwen25vl("[]", W, H), [])

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_qwen25vl("no objects found in this image", W, H))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_qwen25vl("", W, H))

    def test_non_string_returns_none(self):
        self.assertIsNone(parse_qwen25vl(None, W, H))

    def test_invalid_dimensions_return_none(self):
        self.assertIsNone(parse_qwen25vl("[10, 20, 30, 40]", 0, 0))


class TestParseInternVL(unittest.TestCase):
    """PROVISIONAL InternVL: <box>[[...]]</box>, 0-1000 normalized."""

    def test_ref_and_box_tags(self):
        raw = "<ref>the dog</ref><box>[[100, 200, 500, 800]]</box>"
        result = parse_internvl(raw, W, H)
        self.assertEqual(_rounded(result), [[64.0, 96.0, 320.0, 384.0]])

    def test_box_tag_without_ref(self):
        raw = "<box>[[0, 0, 1000, 1000]]</box>"
        result = parse_internvl(raw, W, H)
        self.assertEqual(_rounded(result), [[0.0, 0.0, 640.0, 480.0]])

    def test_multiple_quads_in_one_tag(self):
        raw = "<box>[[0, 0, 500, 500], [500, 500, 1000, 1000]]</box>"
        result = parse_internvl(raw, 1000, 1000)
        self.assertEqual(
            _rounded(result),
            [[0.0, 0.0, 500.0, 500.0], [500.0, 500.0, 1000.0, 1000.0]],
        )

    def test_multiple_box_tags(self):
        raw = (
            "<ref>cat</ref><box>[[10, 10, 200, 200]]</box> and "
            "<ref>dog</ref><box>[[300, 300, 900, 900]]</box>"
        )
        result = parse_internvl(raw, 1000, 1000)
        self.assertEqual(len(result), 2)

    def test_single_bracket_quad_inside_tag(self):
        raw = "<box>[100, 200, 300, 400]</box>"
        result = parse_internvl(raw, 1000, 1000)
        self.assertEqual(_rounded(result), [[100.0, 200.0, 300.0, 400.0]])

    def test_bare_quad_fallback_without_tags(self):
        result = parse_internvl("[[250, 250, 750, 750]]", 1000, 1000)
        self.assertEqual(_rounded(result), [[250.0, 250.0, 750.0, 750.0]])

    def test_scaling_is_per_dimension(self):
        # x scales by width/1000, y by height/1000.
        result = parse_internvl("<box>[[500, 500, 1000, 1000]]</box>", 200, 800)
        self.assertEqual(_rounded(result), [[100.0, 400.0, 200.0, 800.0]])

    def test_clamping_above_1000(self):
        result = parse_internvl("<box>[[900, 900, 1100, 1200]]</box>", 1000, 1000)
        self.assertEqual(_rounded(result), [[900.0, 900.0, 1000.0, 1000.0]])

    def test_degenerate_box_dropped(self):
        self.assertEqual(
            parse_internvl("<box>[[500, 100, 500, 900]]</box>", W, H), []
        )

    def test_empty_box_tag_returns_empty(self):
        self.assertEqual(parse_internvl("<box>[]</box>", W, H), [])
        self.assertEqual(parse_internvl("<box>[[]]</box>", W, H), [])

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_internvl("The image shows a sunny beach.", W, H))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_internvl("", W, H))

    def test_non_string_returns_none(self):
        self.assertIsNone(parse_internvl(None, W, H))
        self.assertIsNone(parse_internvl(42, W, H))

    def test_box_tag_with_garbage_content_returns_none(self):
        self.assertIsNone(parse_internvl("<box>not coordinates</box>", W, H))

    def test_invalid_dimensions_return_none(self):
        self.assertIsNone(parse_internvl("<box>[[0, 0, 500, 500]]</box>", -1, 100))


class TestParseMoondreamDetect(unittest.TestCase):
    """Moondream detect(): structured list of normalized min/max dicts."""

    def test_typical_detection_list(self):
        objects = [
            {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6},
            {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
        ]
        result = parse_moondream_detect(objects, 100, 200)
        self.assertEqual(
            _rounded(result),
            [[10.0, 40.0, 50.0, 120.0], [0.0, 0.0, 100.0, 200.0]],
        )

    def test_dict_wrapper_with_objects_key(self):
        wrapped = {"objects": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}]}
        result = parse_moondream_detect(wrapped, 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 10.0, 20.0, 20.0]])

    def test_empty_list_means_no_detections(self):
        self.assertEqual(parse_moondream_detect([], W, H), [])

    def test_empty_objects_wrapper(self):
        self.assertEqual(parse_moondream_detect({"objects": []}, W, H), [])

    def test_extra_keys_tolerated(self):
        objects = [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.3, "y_max": 0.3, "label": "cat"}]
        result = parse_moondream_detect(objects, 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 10.0, 30.0, 30.0]])

    def test_numeric_strings_tolerated(self):
        objects = [{"x_min": "0.1", "y_min": "0.1", "x_max": "0.2", "y_max": "0.2"}]
        result = parse_moondream_detect(objects, 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 10.0, 20.0, 20.0]])

    def test_non_list_input_returns_none(self):
        self.assertIsNone(parse_moondream_detect("not a list", W, H))
        self.assertIsNone(parse_moondream_detect(None, W, H))
        self.assertIsNone(parse_moondream_detect(3.14, W, H))
        self.assertIsNone(parse_moondream_detect({"x_min": 0.1}, W, H))

    def test_list_of_garbage_returns_none(self):
        self.assertIsNone(parse_moondream_detect(["a", "b"], W, H))
        self.assertIsNone(parse_moondream_detect([{"foo": 1}], W, H))

    def test_missing_key_item_skipped(self):
        objects = [
            {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2},  # y_max missing
            {"x_min": 0.5, "y_min": 0.5, "x_max": 0.9, "y_max": 0.9},
        ]
        result = parse_moondream_detect(objects, 100, 100)
        self.assertEqual(_rounded(result), [[50.0, 50.0, 90.0, 90.0]])

    def test_degenerate_detection_dropped(self):
        objects = [{"x_min": 0.5, "y_min": 0.1, "x_max": 0.5, "y_max": 0.9}]
        self.assertEqual(parse_moondream_detect(objects, W, H), [])

    def test_clamping(self):
        objects = [{"x_min": -0.1, "y_min": 0.0, "x_max": 1.5, "y_max": 0.5}]
        result = parse_moondream_detect(objects, 100, 100)
        self.assertEqual(_rounded(result), [[0.0, 0.0, 100.0, 50.0]])

    def test_boolean_coordinates_rejected(self):
        objects = [{"x_min": True, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}]
        self.assertIsNone(parse_moondream_detect(objects, W, H))

    def test_invalid_dimensions_return_none(self):
        objects = [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}]
        self.assertIsNone(parse_moondream_detect(objects, 0, 100))


class _GenericParserMixin:
    """Shared tests for the PROVISIONAL SmolVLM2 / LLaVA-OneVision parsers."""

    parser = None  # Set by subclasses.

    def test_json_normalized_scaled(self):
        # All values <= 1.0 -> normalized heuristic -> scaled.
        result = self.parser('{"bbox": [0.1, 0.1, 0.5, 0.5]}', 200, 400)
        self.assertEqual(_rounded(result), [[20.0, 40.0, 100.0, 200.0]])

    def test_json_absolute_not_scaled(self):
        # Values > 1.0 -> absolute heuristic -> no scaling.
        result = self.parser('[{"bbox": [15, 20, 200, 300]}]', W, H)
        self.assertEqual(_rounded(result), [[15.0, 20.0, 200.0, 300.0]])

    def test_bbox_2d_key_accepted(self):
        result = self.parser('{"bbox_2d": [10, 10, 100, 100]}', W, H)
        self.assertEqual(_rounded(result), [[10.0, 10.0, 100.0, 100.0]])

    def test_bracketed_quad_in_prose_absolute(self):
        result = self.parser("The dog is at [50, 60, 200, 220].", W, H)
        self.assertEqual(_rounded(result), [[50.0, 60.0, 200.0, 220.0]])

    def test_parenthesized_quad_normalized(self):
        result = self.parser("Coordinates: (0.1, 0.2, 0.3, 0.4)", 100, 100)
        self.assertEqual(_rounded(result), [[10.0, 20.0, 30.0, 40.0]])

    def test_unit_box_treated_as_normalized(self):
        # Documented heuristic boundary: [0, 0, 1, 1] -> full image.
        result = self.parser("[0.0, 0.0, 1.0, 1.0]", W, H)
        self.assertEqual(_rounded(result), [[0.0, 0.0, 640.0, 480.0]])

    def test_thousand_scale_treated_as_absolute_on_large_image(self):
        # Documented ambiguity: values <= 1000 on a larger image are
        # treated as ABSOLUTE pixels (Phase 2 captures decide the truth).
        result = self.parser("[100, 100, 900, 900]", 2000, 2000)
        self.assertEqual(_rounded(result), [[100.0, 100.0, 900.0, 900.0]])

    def test_heuristic_pooled_across_boxes(self):
        # One coordinate > 1.0 flips the WHOLE set to absolute.
        raw = "[[0.5, 0.5, 0.9, 0.9], [2, 2, 50, 50]]"
        result = self.parser(raw, 200, 200)
        # First box is degenerate as absolute pixels (0.5..0.9 px < 1px is
        # kept: 0.9 > 0.5, so it survives as a sub-pixel box) -- verify no
        # scaling happened on the second box.
        self.assertIn([2.0, 2.0, 50.0, 50.0], _rounded(result))

    def test_fenced_json(self):
        raw = '```json\n{"bbox": [0.2, 0.2, 0.4, 0.4]}\n```'
        result = self.parser(raw, 100, 100)
        self.assertEqual(_rounded(result), [[20.0, 20.0, 40.0, 40.0]])

    def test_clamping_absolute(self):
        result = self.parser("[600, 400, 900, 900]", W, H)
        self.assertEqual(_rounded(result), [[600.0, 400.0, 640.0, 480.0]])

    def test_degenerate_returns_empty(self):
        self.assertEqual(self.parser('{"bbox": [0.5, 0.1, 0.5, 0.9]}', W, H), [])

    def test_empty_list_returns_empty(self):
        self.assertEqual(self.parser("[]", W, H), [])

    def test_garbage_returns_none(self):
        self.assertIsNone(self.parser("I don't see any such object.", W, H))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.parser("", W, H))

    def test_non_string_returns_none(self):
        self.assertIsNone(self.parser(None, W, H))
        self.assertIsNone(self.parser({"bbox": [1, 2, 3, 4]}, W, H))

    def test_pair_of_numbers_is_not_a_box(self):
        # Points / pairs must not be misread as boxes.
        self.assertIsNone(self.parser("The point is at [100, 200].", W, H))

    def test_invalid_dimensions_return_none(self):
        self.assertIsNone(self.parser("[10, 10, 20, 20]", 100, 0))


class TestParseSmolVLM2(_GenericParserMixin, unittest.TestCase):
    parser = staticmethod(parse_smolvlm2)


class TestParseLlavaOV(_GenericParserMixin, unittest.TestCase):
    parser = staticmethod(parse_llava_ov)


class TestNeverRaisesContract(unittest.TestCase):
    """All box parsers must return None/list on adversarial input, never raise."""

    ADVERSARIAL = [
        None,
        "",
        "   ",
        0,
        -1.5,
        b"[0.1, 0.2, 0.3, 0.4]",
        object(),
        {"bbox": object()},
        ["[", "{", "]"],
        "{" * 50,
        "[[[[",
        '{"bbox": "not a list"}',
        "<box></ref><box>",
        "\x00\x01\x02",
        "[1e999, 0, 1, 1]",  # Decodes to Infinity -> must be rejected.
    ]

    TEXT_PARSERS = [parse_lfm, parse_qwen25vl, parse_internvl, parse_smolvlm2, parse_llava_ov]

    def test_text_parsers_never_raise(self):
        for parser in self.TEXT_PARSERS:
            for bad in self.ADVERSARIAL:
                with self.subTest(parser=parser.__name__, raw=repr(bad)[:40]):
                    result = parser(bad, W, H)
                    self.assertTrue(result is None or isinstance(result, list))

    def test_moondream_never_raises(self):
        for bad in self.ADVERSARIAL + [[object()], [{"x_min": object()}]]:
            with self.subTest(objects=repr(bad)[:40]):
                result = parse_moondream_detect(bad, W, H)
                self.assertTrue(result is None or isinstance(result, list))

    def test_bracket_flood_is_fast(self):
        """Degenerate repetition (bracket floods) must not cause quadratic
        blowup: bounded by the _scan_segment attempt/window caps."""
        import time

        floods = ["[" * 20000, "{" * 20000, "[1," * 8000, '{"bbox":' * 3000]
        start = time.monotonic()
        for flood in floods:
            for parser in self.TEXT_PARSERS:
                result = parser(flood, W, H)
                self.assertTrue(result is None or isinstance(result, list))
        # Pre-fix this took >60 s; generous bound to avoid CI flakiness.
        self.assertLess(time.monotonic() - start, 10.0)

    def test_bad_dimensions_never_raise(self):
        for w, h in [(0, 0), (-1, 5), (None, None), ("640", "480"), (float("nan"), 100)]:
            for parser in self.TEXT_PARSERS:
                with self.subTest(parser=parser.__name__, dims=(w, h)):
                    self.assertIsNone(parser("[0.1, 0.2, 0.3, 0.4]", w, h))

    def test_output_boxes_are_floats(self):
        result = parse_qwen25vl("[10, 20, 30, 40]", W, H)
        for value in result[0]:
            self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
