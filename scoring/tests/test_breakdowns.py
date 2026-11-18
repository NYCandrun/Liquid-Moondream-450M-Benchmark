"""Unit tests for scoring.breakdowns (stdlib unittest)."""

import unittest

from scoring.breakdowns import paired_bootstrap, wilson


class TestWilson(unittest.TestCase):
    def test_midpoint_interval(self):
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_zero_successes_has_nonnegative_lower_bound(self):
        lo, hi = wilson(0, 2080)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_empty(self):
        self.assertEqual(wilson(0, 0), (0.0, 0.0))


class TestPairedBootstrap(unittest.TestCase):
    def test_identical_results_give_zero_difference(self):
        a = {str(i): i % 2 == 0 for i in range(200)}
        r = paired_bootstrap(a, dict(a), reps=200)
        self.assertEqual(r["diff"], 0.0)
        self.assertEqual(r["ci95"], [0.0, 0.0])

    def test_deterministic(self):
        a = {str(i): i % 3 == 0 for i in range(100)}
        b = {str(i): i % 2 == 0 for i in range(100)}
        self.assertEqual(paired_bootstrap(a, b, reps=300), paired_bootstrap(a, b, reps=300))


if __name__ == "__main__":
    unittest.main()
