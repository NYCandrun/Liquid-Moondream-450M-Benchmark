"""Checks that the demo lookup table reproduces the published sweep numbers."""

import unittest

from scoring.demo_table import build


class TestDemoTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build()

    def _row(self, source, model_prefix, setting):
        return next(r for r in self.rows if r["source"] == source
                    and r["model"].startswith(model_prefix) and r["setting"] == setting)

    def test_row_count(self):
        self.assertEqual(len(self.rows), 17)

    def test_lfm_sweep_matches_results_md(self):
        for setting, expected in ((32, 0.395), (64, 0.604), (128, 0.700), (256, 0.775)):
            self.assertAlmostEqual(self._row("sweep", "LFM", setting)["accuracy"], expected, places=3)

    def test_smolvlm_sweep_matches_results_md(self):
        for setting, expected in ((512, 0.580), (1024, 0.601), (1536, 0.607), (2048, 0.609)):
            self.assertAlmostEqual(self._row("sweep", "SmolVLM2", setting)["accuracy"], expected, places=3)

    def test_probe_rows_carry_energy(self):
        r = self._row("probe", "SmolVLM2", 2048)
        self.assertAlmostEqual(r["j_per_answer"], 47.857, places=2)


if __name__ == "__main__":
    unittest.main()
