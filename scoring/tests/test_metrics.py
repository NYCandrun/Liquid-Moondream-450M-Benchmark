"""Unit tests for scoring.metrics (stdlib unittest only, no pytest).

All expected values are computed by hand and written as exact fractions
(e.g. ``1.0 / 7.0``) so a failure points at real metric drift, not float
formatting. Run from the repo root with:

    python3 -m unittest discover -s scoring/tests -v
"""

from __future__ import annotations

import unittest

from scoring.metrics import (
    count_aggregate,
    grounding_aggregate,
    grounding_item_scores,
    iou,
    pope_aggregate,
)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        """Identical boxes have IoU exactly 1.0."""
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([3.5, 2.0, 7.5, 9.0], [3.5, 2.0, 7.5, 9.0]), 1.0)

    def test_disjoint_boxes(self):
        """Non-overlapping boxes have IoU 0.0."""
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_touching_edges_is_zero(self):
        """Boxes sharing only an edge have zero-area intersection -> 0.0."""
        self.assertEqual(iou([0, 0, 5, 5], [5, 0, 10, 5]), 0.0)

    def test_half_overlap_hand_computed(self):
        """[0,0,10,10] vs [0,0,10,5]: inter=50, union=100 -> 0.5 exactly."""
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 5]), 0.5)

    def test_corner_overlap_hand_computed(self):
        """[0,0,2,2] vs [1,1,3,3]: inter=1, union=4+4-1=7 -> 1/7."""
        self.assertAlmostEqual(iou([0, 0, 2, 2], [1, 1, 3, 3]), 1.0 / 7.0)

    def test_contained_box_hand_computed(self):
        """[2,2,4,4] inside [0,0,10,10]: inter=4, union=100 -> 0.04."""
        self.assertAlmostEqual(iou([2, 2, 4, 4], [0, 0, 10, 10]), 4.0 / 100.0)

    def test_degenerate_zero_area_box(self):
        """Zero-width/height boxes have zero area -> IoU 0.0 in either slot."""
        line = [5, 0, 5, 10]  # zero width
        self.assertEqual(iou(line, [0, 0, 10, 10]), 0.0)
        self.assertEqual(iou([0, 0, 10, 10], line), 0.0)
        point = [3, 3, 3, 3]  # zero width and height
        self.assertEqual(iou(point, point), 0.0)

    def test_inverted_coordinates_are_degenerate(self):
        """x2 < x1 (negative width) is degenerate -> IoU 0.0."""
        self.assertEqual(iou([10, 0, 0, 10], [0, 0, 10, 10]), 0.0)

    def test_symmetry(self):
        """IoU(a, b) == IoU(b, a)."""
        a, b = [0, 0, 6, 4], [3, 1, 9, 8]
        self.assertEqual(iou(a, b), iou(b, a))


# ---------------------------------------------------------------------------
# Grounding: per-item scores
# ---------------------------------------------------------------------------

class TestGroundingItemScores(unittest.TestCase):
    GT = [0, 0, 10, 10]

    def test_perfect_hit(self):
        s = grounding_item_scores([0, 0, 10, 10], self.GT)
        self.assertEqual(s["iou"], 1.0)
        self.assertTrue(s["hit_at_50"])
        self.assertTrue(s["center_in_box"])
        self.assertEqual(s["area_ratio"], 1.0)
        self.assertIsNone(s["miss_kind"])

    def test_not_found_when_unparsed(self):
        """parsed=None -> automatic miss with miss_kind 'not_found'."""
        s = grounding_item_scores(None, self.GT)
        self.assertEqual(s["iou"], 0.0)
        self.assertFalse(s["hit_at_50"])
        self.assertFalse(s["center_in_box"])
        self.assertIsNone(s["area_ratio"])
        self.assertEqual(s["miss_kind"], "not_found")

    def test_found_but_loose(self):
        """A parsed box with IoU < 0.5 is a 'found_but_loose' miss."""
        s = grounding_item_scores([8, 8, 12, 12], self.GT)  # iou = 4/112
        self.assertAlmostEqual(s["iou"], 4.0 / 112.0)
        self.assertFalse(s["hit_at_50"])
        self.assertEqual(s["miss_kind"], "found_but_loose")

    def test_hit_at_exact_threshold(self):
        """IoU exactly 0.5 counts as a hit (>= threshold), miss_kind None."""
        s = grounding_item_scores([0, 0, 10, 5], self.GT)  # iou = 50/100 = 0.5
        self.assertEqual(s["iou"], 0.5)
        self.assertTrue(s["hit_at_50"])
        self.assertIsNone(s["miss_kind"])

    def test_center_on_boundary_counts_inside(self):
        """Predicted center exactly on the gt edge counts as inside."""
        # pred center = (10, 5): on the right edge of gt [0,0,10,10].
        s = grounding_item_scores([8, 4, 12, 6], self.GT)
        self.assertTrue(s["center_in_box"])
        # pred center = (10, 10): exactly on the gt corner.
        s = grounding_item_scores([8, 8, 12, 12], self.GT)
        self.assertTrue(s["center_in_box"])

    def test_center_outside(self):
        # pred center = (10.5, 5): just past the right edge.
        s = grounding_item_scores([8.5, 4, 12.5, 6], self.GT)
        self.assertFalse(s["center_in_box"])

    def test_area_ratio_hand_computed(self):
        """pred area 25 / gt area 100 -> 0.25."""
        s = grounding_item_scores([0, 0, 5, 5], self.GT)
        self.assertAlmostEqual(s["area_ratio"], 0.25)

    def test_area_ratio_none_when_gt_degenerate(self):
        """Degenerate gt (zero area) -> area_ratio None, no ZeroDivisionError."""
        s = grounding_item_scores([0, 0, 5, 5], [3, 3, 3, 3])
        self.assertIsNone(s["area_ratio"])


# ---------------------------------------------------------------------------
# Grounding: aggregate
# ---------------------------------------------------------------------------

class TestGroundingAggregate(unittest.TestCase):
    GT = [0, 0, 10, 10]

    def _mixed_items(self):
        """1 hit + 1 not_found + 1 found_but_loose, hand-computable."""
        return [
            grounding_item_scores([0, 0, 10, 10], self.GT),  # hit, iou 1.0
            grounding_item_scores(None, self.GT),            # not_found
            grounding_item_scores([8, 8, 12, 12], self.GT),  # loose, iou 4/112
        ]

    def test_mixed_aggregate_hand_computed(self):
        agg = grounding_aggregate(self._mixed_items())
        self.assertEqual(agg["n_items"], 3)
        self.assertAlmostEqual(agg["acc_at_50"], 1.0 / 3.0)
        self.assertAlmostEqual(agg["parse_rate"], 2.0 / 3.0)
        # centers: hit box center (5,5) inside; loose box center (10,10) on
        # the corner counts inside; unparsed counts False -> 2/3.
        self.assertAlmostEqual(agg["center_in_box_recall"], 2.0 / 3.0)
        # mean IoU over the 2 found boxes: (1.0 + 4/112) / 2.
        self.assertAlmostEqual(agg["mean_iou_on_found"], (1.0 + 4.0 / 112.0) / 2.0)
        # mean area ratio over found: (100/100 + 16/100) / 2 = 0.58.
        self.assertAlmostEqual(agg["mean_area_ratio_on_found"], 0.58)
        self.assertAlmostEqual(agg["miss_decomposition"]["not_found_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(agg["miss_decomposition"]["found_but_loose_rate"], 1.0 / 3.0)

    def test_miss_decomposition_sums_to_one_with_acc(self):
        """acc_at_50 + not_found_rate + found_but_loose_rate == 1 exactly."""
        agg = grounding_aggregate(self._mixed_items())
        total = (
            agg["acc_at_50"]
            + agg["miss_decomposition"]["not_found_rate"]
            + agg["miss_decomposition"]["found_but_loose_rate"]
        )
        self.assertAlmostEqual(total, 1.0)

    def test_unparsed_counts_against_headline(self):
        """Two perfect hits + two unparsed -> acc@0.5 is 0.5, not 1.0."""
        items = [
            grounding_item_scores([0, 0, 10, 10], self.GT),
            grounding_item_scores([0, 0, 10, 10], self.GT),
            grounding_item_scores(None, self.GT),
            grounding_item_scores(None, self.GT),
        ]
        agg = grounding_aggregate(items)
        self.assertEqual(agg["acc_at_50"], 0.5)
        self.assertEqual(agg["parse_rate"], 0.5)

    def test_all_not_found(self):
        """No parsed boxes: conditional means None, rates well-defined."""
        agg = grounding_aggregate([grounding_item_scores(None, self.GT)] * 3)
        self.assertEqual(agg["acc_at_50"], 0.0)
        self.assertEqual(agg["parse_rate"], 0.0)
        self.assertIsNone(agg["mean_iou_on_found"])
        self.assertIsNone(agg["mean_area_ratio_on_found"])
        self.assertEqual(agg["miss_decomposition"]["not_found_rate"], 1.0)

    def test_empty_input_no_division_by_zero(self):
        """Empty item list must not raise and must be unambiguous."""
        agg = grounding_aggregate([])
        self.assertEqual(agg["n_items"], 0)
        self.assertEqual(agg["acc_at_50"], 0.0)
        self.assertEqual(agg["parse_rate"], 0.0)
        self.assertIsNone(agg["mean_iou_on_found"])
        self.assertIsNone(agg["mean_area_ratio_on_found"])


# ---------------------------------------------------------------------------
# POPE
# ---------------------------------------------------------------------------

class TestPopeAggregate(unittest.TestCase):
    def test_f1_hand_computed_confusion_matrix(self):
        """tp=2, fp=1, fn=1, tn=1 -> P=2/3, R=2/3, F1=2/3; acc=3/5."""
        items = [
            (True, True),    # tp
            (True, True),    # tp
            (True, False),   # fp
            (False, True),   # fn
            (False, False),  # tn
        ]
        agg = pope_aggregate(items)
        self.assertEqual(agg["n_items"], 5)
        self.assertAlmostEqual(agg["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(agg["recall"], 2.0 / 3.0)
        self.assertAlmostEqual(agg["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(agg["accuracy"], 3.0 / 5.0)
        self.assertAlmostEqual(agg["yes_ratio"], 3.0 / 5.0)  # 3 yes of 5 parsed
        self.assertEqual(agg["parse_rate"], 1.0)

    def test_unparsed_counts_as_wrong(self):
        """Unparsed answers reduce accuracy; gt-yes unparsed is a false neg."""
        items = [
            (True, True),   # correct
            (None, True),   # unparsed on gt=yes -> wrong AND a false negative
            (None, False),  # unparsed on gt=no  -> wrong, but NOT a false pos
        ]
        agg = pope_aggregate(items)
        self.assertAlmostEqual(agg["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(agg["parse_rate"], 1.0 / 3.0)
        # tp=1, fp=0, fn=1 -> P=1, R=1/2, F1=2/3.
        self.assertAlmostEqual(agg["precision"], 1.0)
        self.assertAlmostEqual(agg["recall"], 1.0 / 2.0)
        self.assertAlmostEqual(agg["f1"], 2.0 / 3.0)

    def test_yes_ratio_over_parsed_only(self):
        """yes_ratio is the share of PARSED answers that are yes."""
        items = [(True, True), (False, False), (None, True), (None, False)]
        agg = pope_aggregate(items)
        self.assertAlmostEqual(agg["yes_ratio"], 1.0 / 2.0)  # 1 yes of 2 parsed
        self.assertAlmostEqual(agg["parse_rate"], 2.0 / 4.0)

    def test_all_unparsed(self):
        """Nothing parsed: accuracy/f1 0.0, yes_ratio None (undefined)."""
        agg = pope_aggregate([(None, True), (None, False)])
        self.assertEqual(agg["accuracy"], 0.0)
        self.assertEqual(agg["parse_rate"], 0.0)
        self.assertEqual(agg["f1"], 0.0)  # recall 0 on the gt-yes item
        self.assertIsNone(agg["yes_ratio"])

    def test_empty_input_no_division_by_zero(self):
        agg = pope_aggregate([])
        self.assertEqual(agg["n_items"], 0)
        self.assertEqual(agg["accuracy"], 0.0)
        self.assertEqual(agg["f1"], 0.0)
        self.assertEqual(agg["parse_rate"], 0.0)
        self.assertIsNone(agg["yes_ratio"])


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

class TestCountAggregate(unittest.TestCase):
    def test_exact_match_hand_computed(self):
        """Matches: 3==3 and 0==0; misses: 2!=5. -> 2/3."""
        agg = count_aggregate([(3, 3), (2, 5), (0, 0)])
        self.assertEqual(agg["n_items"], 3)
        self.assertAlmostEqual(agg["exact_match"], 2.0 / 3.0)
        self.assertEqual(agg["parse_rate"], 1.0)

    def test_near_miss_is_not_a_match(self):
        """Off-by-one gets no credit: exact match only."""
        agg = count_aggregate([(4, 5)])
        self.assertEqual(agg["exact_match"], 0.0)

    def test_unparsed_counts_as_wrong(self):
        """pred=None is wrong even though it 'could have been' right."""
        agg = count_aggregate([(None, 1), (7, 7)])
        self.assertEqual(agg["exact_match"], 0.5)
        self.assertEqual(agg["parse_rate"], 0.5)

    def test_empty_input_no_division_by_zero(self):
        agg = count_aggregate([])
        self.assertEqual(agg["n_items"], 0)
        self.assertEqual(agg["exact_match"], 0.0)
        self.assertEqual(agg["parse_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
