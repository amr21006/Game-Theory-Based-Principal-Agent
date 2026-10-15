"""Unit tests for the robustness modules (Areas 1 to 4), on reduced case populations."""

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.robustness_extensions.test_area1_coefficient_bounds import run_area1_coefficient_stress
from tests.robustness_extensions.test_area2_unctad_construction_calibration import run_area2_unctad_calibration
from tests.robustness_extensions.test_area3_legal_security_risk import run_area3_security_risk_stress
from tests.robustness_extensions.test_area4_multiparty_structure import run_area4_multiparty_analysis


class TestRobustnessExtensions(unittest.TestCase):
    def test_area1_coefficient_stress(self):
        df = run_area1_coefficient_stress(n_cases=20_000)
        self.assertEqual(df["perturbation"].nunique(), 17)
        # the grid-selected structure attains the highest score under every perturbation
        for _, group in df.groupby("perturbation"):
            self.assertEqual(group.set_index("scenario")["score_j"].idxmax(), "Grid-selected")

    def test_area2_unctad_calibration(self):
        df, shift = run_area2_unctad_calibration(n_cases=20_000, target_win_rate=0.3514)
        self.assertEqual(len(df), 5)
        self.assertLess(shift, -0.16)
        base_win = df.loc[df["scenario"] == "No funding", "mean_success_prob_filed"].iloc[0]
        self.assertAlmostEqual(base_win, 0.3514, delta=0.002)
        # the grid-selected structure scores above conventional terms under the recalibration
        scores = df.set_index("scenario")["score_j"]
        self.assertGreater(scores["Grid-selected"], scores["Conventional"])

    def test_area3_security_risk(self):
        df = run_area3_security_risk_stress(n_cases=20_000)
        self.assertEqual(len(df), 11)
        high_sec = df[df["security_probability_p_sec"] == 0.70]
        self.assertTrue((high_sec["mean_funder_roi_multiple"] > 1.18).all())

    def test_area4_multiparty_structure(self):
        df = run_area4_multiparty_analysis(n_cases=20_000)
        multi = df[df["model_architecture"].str.contains("Multi-Party")].set_index("scenario")
        self.assertGreater(multi.loc["Grid-selected", "settlement_rate_filed"],
                           multi.loc["Conventional", "settlement_rate_filed"])
        self.assertEqual(multi["score_j"].idxmax(), "Grid-selected")


if __name__ == "__main__":
    unittest.main()
