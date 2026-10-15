"""Area 2: construction-sector recalibration to the UNCTAD investor success rate (supplement Table S33).

The additive calibration constant of Eq. (1) is reset so that the baseline mean success probability
among filed cases equals the 35.14% investor success rate among the 74 construction-sector treaty
decisions. All five structures are then evaluated and scored by Eq. (15), against the 0.33 settlement
benchmark and, for comparison, against the recalibrated baseline settlement rate.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from scipy.optimize import brentq

from tests.robustness_extensions.common import ROOT, Params, run_structures
from src.tpf_model import SimulationConfig, generate_cases, load_scenarios, summarize_results
from src.tpf_param import evaluate


def _baseline_terms() -> dict:
    fixed = load_scenarios(ROOT / "data" / "raw" / "scenario_definitions.csv")
    return fixed[fixed["scenario"] == "No TPF baseline"].iloc[0].to_dict()


def find_calibration_shift(cases: pd.DataFrame, target_win_rate: float = 0.3514) -> float:
    """Additive constant giving the target mean success probability among filed baseline cases."""
    base = _baseline_terms()

    def gap(shift: float) -> float:
        s = summarize_results(evaluate(cases, base, replace(Params(), shift=shift)))
        return float(s["mean_success_probability_filed"]) - target_win_rate

    return float(brentq(gap, -0.80, -0.16, xtol=1e-5))


def run_area2_unctad_calibration(
    n_cases: int = 150_000, seed: int = 1514, target_win_rate: float = 0.3514
) -> tuple[pd.DataFrame, float]:
    cases = generate_cases(SimulationConfig(n_cases=n_cases, seed=seed))
    shift = find_calibration_shift(cases, target_win_rate)
    summaries, scores, scores_own = run_structures(cases, replace(Params(), shift=shift))
    rows = []
    for name, s in summaries.items():
        rows.append({
            "calibration": f"UNCTAD 35.1% (shift={shift:.4f})",
            "scenario": name,
            "filed_rate": s["filed_rate"],
            "mean_success_prob_filed": s["mean_success_probability_filed"],
            "settlement_rate_filed": s["settlement_rate_filed"],
            "constrained_access": s["constrained_access_rate"],
            "mean_claimant_net_musd": s["mean_claimant_net_musd"],
            "mean_funder_roi_multiple": s["mean_funder_roi_multiple"],
            "governance_mh_indicator": s["mean_moral_hazard_index"],
            "mean_respondent_outflow_musd": s["mean_respondent_outflow_musd"],
            "score_j": scores[name],
            "score_j_baseline_benchmark": scores_own[name],
        })
    return pd.DataFrame(rows), shift


REPLICATION_SEEDS = (1514, 2026, 2701, 31415, 424242)


def run_area2_seed_replication(seeds=REPLICATION_SEEDS, n_cases: int = 150_000) -> pd.DataFrame:
    """Recalibration repeated under the five replication seeds; one row per seed."""
    rows = []
    for seed in seeds:
        df, shift = run_area2_unctad_calibration(n_cases=n_cases, seed=seed)
        s = df.set_index("scenario")
        rows.append({
            "seed": seed, "shift": shift,
            **{f"score_j_{k}": v for k, v in s["score_j"].items()},
            "highest_score": s["score_j"].idxmax(),
            "highest_score_baseline_benchmark": s["score_j_baseline_benchmark"].idxmax(),
            "grid_minus_conventional": s.loc["Grid-selected", "score_j"] - s.loc["Conventional", "score_j"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df, shift = run_area2_unctad_calibration()
    print(f"shift = {shift:.4f}")
    print(df.drop(columns=["calibration"]).round(4).to_string(index=False))
