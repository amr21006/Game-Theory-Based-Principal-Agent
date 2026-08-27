"""Unit tests for the robustness extension modules (Areas 1 to 4)."""

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
        df = run_area1_coefficient_stress(n_cases=2_000)
        self.assertGreater(len(df), 0)
        # Check that Grid-selected always has highest score
        for pert, group in df.groupby("perturbation"):
            scores = group.set_index("scenario")["score_j"]
            self.assertEqual(scores.idxmax(), "Grid-selected")

    def test_area2_unctad_calibration(self):
        df, shift = run_area2_unctad_calibration(n_cases=5_000, target_win_rate=0.3514)
        self.assertGreater(len(df), 0)
        self.assertLess(shift, 0.0)
        # Verify win rate among filed in baseline is approximately 35.1%
        base_win = df[df["scenario"] == "No funding"]["mean_success_prob_filed"].iloc[0]
        self.assertAlmostEqual(base_win, 0.3514, delta=0.03)

    def test_area3_security_risk(self):
        df = run_area3_security_risk_stress(n_cases=2_000)
        self.assertGreater(len(df), 0)
        # Check funder ROI stays positive even at high security risk
        high_sec = df[df["security_probability_p_sec"] == 0.70]
        self.assertTrue((high_sec["mean_funder_roi_multiple"] > 1.18).all())

    def test_area4_multiparty_structure(self):
        df = run_area4_multiparty_analysis(n_cases=2_000)
        self.assertGreater(len(df), 0)
        # Verify grid-selected preserves higher settlement than conventional in multi-party
        multi_df = df[df["model_architecture"].str.contains("Multi-Party")].set_index("scenario")
        self.assertGreater(
            multi_df.loc["Grid-selected", "settlement_rate_filed"],
            multi_df.loc["Conventional", "settlement_rate_filed"],
        )


if __name__ == "__main__":
    unittest.main()
