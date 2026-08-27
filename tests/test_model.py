import unittest
from pathlib import Path

import numpy as np

from src.tpf_model import (
    DEFAULT_SCORE_WEIGHTS,
    SimulationConfig,
    balanced_score,
    evaluate_scenario,
    generate_cases,
    load_scenarios,
    score_components,
    summarize_results,
    weighted_score,
)


ROOT = Path(__file__).resolve().parents[1]


class TestTPFModel(unittest.TestCase):
    def setUp(self):
        self.cases = generate_cases(SimulationConfig(n_cases=5000, seed=42))
        self.scenarios = load_scenarios(ROOT / "data" / "raw" / "scenario_definitions.csv")

    def test_generated_cases_are_calibrated_and_bounded(self):
        self.assertEqual(len(self.cases), 5000)
        self.assertTrue(np.isfinite(self.cases["claim_value_musd"]).all())
        self.assertGreater(self.cases["claim_value_musd"].mean(), 100)
        self.assertLess(self.cases["claim_value_musd"].mean(), 160)
        self.assertTrue(((self.cases["merit"] >= 0) & (self.cases["merit"] <= 1)).all())

    def test_baseline_has_no_tpf_outputs(self):
        baseline = self.scenarios.loc[self.scenarios["scenario"] == "No TPF baseline"].iloc[0]
        result = evaluate_scenario(self.cases, baseline)
        self.assertFalse(result["used_tpf"].any())
        self.assertEqual(float(result["funder_profit_musd"].abs().sum()), 0.0)
        summary = summarize_results(result)
        self.assertGreaterEqual(summary["settlement_rate_filed"], 0.0)
        self.assertLessEqual(summary["settlement_rate_filed"], 1.0)

    def test_conventional_tpf_changes_behavior(self):
        baseline = self.scenarios.loc[self.scenarios["scenario"] == "No TPF baseline"].iloc[0]
        conventional = self.scenarios.loc[self.scenarios["scenario"] == "Conventional TPF"].iloc[0]
        baseline_summary = summarize_results(evaluate_scenario(self.cases, baseline))
        conventional_summary = summarize_results(evaluate_scenario(self.cases, conventional))
        self.assertGreater(conventional_summary["filed_rate"], baseline_summary["filed_rate"])
        self.assertGreaterEqual(conventional_summary["mean_funder_roi_multiple"], 1.0)

    def test_no_nan_in_summaries(self):
        for _, scenario in self.scenarios.iterrows():
            result = evaluate_scenario(self.cases, scenario)
            summary = summarize_results(result)
            for value in summary.values:
                self.assertTrue(np.isfinite(value))

    def test_balanced_score_matches_component_weighting(self):
        baseline = self.scenarios.loc[self.scenarios["scenario"] == "No TPF baseline"].iloc[0]
        conventional = self.scenarios.loc[self.scenarios["scenario"] == "Conventional TPF"].iloc[0]
        baseline_summary = summarize_results(evaluate_scenario(self.cases, baseline))
        conventional_summary = summarize_results(evaluate_scenario(self.cases, conventional))
        components = score_components(conventional_summary, baseline_summary)
        self.assertEqual(set(components), set(DEFAULT_SCORE_WEIGHTS))
        self.assertAlmostEqual(
            balanced_score(conventional_summary, baseline_summary),
            weighted_score(components),
        )
        self.assertTrue(all(np.isfinite(value) for value in components.values()))


if __name__ == "__main__":
    unittest.main()
