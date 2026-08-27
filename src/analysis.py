"""Agreement-term search and global sensitivity analysis.

Pure analysis routines: every function takes a case population and returns a
DataFrame or a dict. Nothing here writes files, formats tables or draws figures.

    grid_search        evaluates the 5,400-candidate agreement-term grid and
                       returns the ranked candidates and the winning terms
    sobol_indices      first-order and total-order Sobol indices for the
                       balanced objective
    sobol_indices_for  the same decomposition for any scalar outcome, used for
                       the constrained-access decomposition
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from src.tpf_model import balanced_score, evaluate_scenario, summarize_results

# search grid: six independent variables, two dependent, one fixed
AWARD_SHARES = np.round(np.linspace(0.18, 0.38, 6), 3)
MONITORING = np.round(np.linspace(0.25, 0.85, 5), 3)
SETTLEMENT_CONTROL = np.round(np.linspace(0.05, 0.65, 5), 3)
COPAYS = [0.00, 0.05, 0.10, 0.15]
ADVERSE_COVERAGE = [0.70, 0.85, 1.00]
FUNDING_SHARES = [0.85, 0.95, 1.00]
MIN_ROI = 1.18

SEARCH_CASES = 60_000
SENSITIVITY_CASES = 25_000


def candidate_terms(funding_share, award_share, monitoring, settlement_control,
                    copay, adverse_coverage, scenario="Grid candidate"):
    """Assemble one candidate agreement. Staged-funding efficiency and the
    security-for-costs probability are dependent on monitoring and on
    adverse-cost coverage respectively."""
    return {
        "scenario": scenario,
        "tpf_available": 1.0,
        "funding_share": funding_share,
        "funder_award_share": award_share,
        "monitoring_intensity": monitoring,
        "settlement_control": settlement_control,
        "adverse_cost_coverage": adverse_coverage,
        "claimant_copay_share": copay,
        "staged_funding_efficiency": 0.16 + 0.10 * monitoring,
        "security_for_costs_probability": 0.08 + 0.08 * adverse_coverage,
        "min_roi_multiple": MIN_ROI,
    }


def grid_search(cases: pd.DataFrame, baseline_summary: pd.Series,
                settlement_target: float = 0.33) -> tuple[pd.DataFrame, dict]:
    """Evaluate every candidate on a subsample and rank by the balanced score.

    Returns (candidates, best_terms). `candidates` is sorted by descending
    balanced score; `best_terms` is the winning agreement, ready to pass to
    ``evaluate_scenario`` on the full population.
    """
    search_cases = cases.iloc[:SEARCH_CASES].copy()
    rows = []
    for award_share in AWARD_SHARES:
        for monitoring in MONITORING:
            for settlement_control in SETTLEMENT_CONTROL:
                for copay in COPAYS:
                    for adverse_coverage in ADVERSE_COVERAGE:
                        for funding_share in FUNDING_SHARES:
                            terms = candidate_terms(funding_share, award_share, monitoring,
                                                    settlement_control, copay, adverse_coverage)
                            summary = summarize_results(evaluate_scenario(search_cases, terms))
                            rows.append({
                                **{k: v for k, v in terms.items() if k != "scenario"},
                                **summary.to_dict(),
                                "balanced_score": balanced_score(summary, baseline_summary, settlement_target),
                            })

    candidates = pd.DataFrame(rows).sort_values("balanced_score", ascending=False).reset_index(drop=True)
    best = candidates.iloc[0]
    best_terms = candidate_terms(
        float(best["funding_share"]), float(best["funder_award_share"]),
        float(best["monitoring_intensity"]), float(best["settlement_control"]),
        float(best["claimant_copay_share"]), float(best["adverse_cost_coverage"]),
        scenario="Grid-selected",
    )
    return candidates, best_terms


SOBOL_PROBLEM = {
    "num_vars": 6,
    "names": ["funding_share", "funder_award_share", "monitoring_intensity",
              "settlement_control", "claimant_copay_share", "adverse_cost_coverage"],
    "bounds": [[0.80, 1.00], [0.18, 0.40], [0.10, 0.90],
               [0.00, 0.80], [0.00, 0.20], [0.60, 1.00]],
}


def _sobol_terms(row):
    funding_share, award, monitoring, control, copay, adverse = row
    terms = candidate_terms(funding_share, award, monitoring, control, copay, adverse,
                            scenario="Sobol candidate")
    terms["security_for_costs_probability"] = 0.10  # held fixed in the decomposition
    return terms


def _indices_frame(problem, si, output_metric=None):
    out = pd.DataFrame({
        "parameter": problem["names"],
        "first_order_s1": si["S1"],
        "first_order_s1_conf": si["S1_conf"],
        "total_order_st": si["ST"],
        "total_order_st_conf": si["ST_conf"],
    })
    if output_metric:
        out["output_metric"] = output_metric
    return out.sort_values("total_order_st", ascending=False).reset_index(drop=True)


def sobol_indices(cases: pd.DataFrame, baseline_summary: pd.Series,
                  settlement_target: float = 0.33,
                  n_samples: int = 512, seed: int = 2026) -> pd.DataFrame:
    """Sobol indices for the balanced agreement score, over the six searched terms."""
    param_values = sobol_sample.sample(SOBOL_PROBLEM, n_samples, calc_second_order=False, seed=seed)
    sub = cases.iloc[:SENSITIVITY_CASES].copy()
    y = np.empty(len(param_values))
    for i, row in enumerate(param_values):
        summary = summarize_results(evaluate_scenario(sub, _sobol_terms(row)))
        y[i] = balanced_score(summary, baseline_summary, settlement_target)
    si = sobol_analyze.analyze(SOBOL_PROBLEM, y, calc_second_order=False, print_to_console=False)
    return _indices_frame(SOBOL_PROBLEM, si)


# The constrained-access decomposition is a separate design: five variables,
# a much wider funding-share range, and co-payment, staged-funding efficiency
# and the security probability held fixed. It answers how many constrained
# claimants can proceed, not how the governance objective behaves.
ACCESS_PROBLEM = {
    "num_vars": 5,
    "names": ["funding_share", "funder_award_share", "monitoring_intensity",
              "settlement_control", "adverse_cost_coverage"],
    "bounds": [[0.20, 1.00], [0.18, 0.40], [0.10, 0.90], [0.00, 0.80], [0.60, 1.00]],
}


def sobol_indices_access(cases: pd.DataFrame, n_samples: int = 256, seed: int = 2027) -> pd.DataFrame:
    """Sobol indices for the constrained-access rate, under ACCESS_PROBLEM."""
    param_values = sobol_sample.sample(ACCESS_PROBLEM, n_samples, calc_second_order=False, seed=seed)
    sub = cases.iloc[:SENSITIVITY_CASES].copy()
    y = np.empty(len(param_values))
    for i, row in enumerate(param_values):
        funding_share, award, monitoring, control, adverse = row
        terms = {
            "scenario": "Access candidate",
            "tpf_available": 1.0,
            "funding_share": funding_share,
            "funder_award_share": award,
            "monitoring_intensity": monitoring,
            "settlement_control": control,
            "adverse_cost_coverage": adverse,
            "claimant_copay_share": 0.05,
            "staged_funding_efficiency": 0.10,
            "security_for_costs_probability": 0.08,
            "min_roi_multiple": MIN_ROI,
        }
        y[i] = float(summarize_results(evaluate_scenario(sub, terms))["constrained_access_rate"])
    si = sobol_analyze.analyze(ACCESS_PROBLEM, y, calc_second_order=False, print_to_console=False)
    return _indices_frame(ACCESS_PROBLEM, si, output_metric="constrained_access_rate")


def plateau_extent(candidates: pd.DataFrame) -> pd.DataFrame:
    """Share of the top candidate subsets at the monitoring upper bound and the
    settlement-control lower bound, and the score range of each subset."""
    rows = []
    n = len(candidates)
    for label, frac in (("top 1 percent", 0.01), ("top 5 percent", 0.05),
                        ("top 10 percent", 0.10), ("top 100 percent", 1.00)):
        k = max(1, int(round(n * frac)))
        sub = candidates.iloc[:k]
        rows.append({
            "candidate_subset": label,
            "n_candidates": k,
            "score_minimum": sub["balanced_score"].min(),
            "score_maximum": sub["balanced_score"].max(),
            "share_with_monitoring_at_upper_bound": float((sub["monitoring_intensity"] == MONITORING.max()).mean()),
            "share_with_settlement_control_at_lower_bound": float((sub["settlement_control"] == SETTLEMENT_CONTROL.min()).mean()),
            "share_with_copay_above_zero": float((sub["claimant_copay_share"] > 0).mean()),
        })
    return pd.DataFrame(rows)
