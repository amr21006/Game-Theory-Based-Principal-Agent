"""Area 5: checks on the agreement-term search.

(a) Settlement control extended to the conventional level of 0.70: the 1,080 candidates with
    settlement control at 0.70 are evaluated on the same 60,000-case search subsample and baseline,
    and the rank of the primary selection among all 6,480 candidates is reported.
(b) Settlement-alignment component removed: the 5,400 searched candidates are re-ranked by Eq. (15)
    with the weight on the settlement-alignment component set to zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.robustness_extensions.common import ROOT  # noqa: F401  (sets sys.path)
from src.analysis import (
    ADVERSE_COVERAGE, AWARD_SHARES, COPAYS, FUNDING_SHARES, MONITORING, SEARCH_CASES, SETTLEMENT_CONTROL,
    candidate_terms, grid_search,
)
from src.tpf_model import (
    DEFAULT_SCORE_WEIGHTS, SimulationConfig, balanced_score, evaluate_scenario, generate_cases, score_components,
    summarize_results, weighted_score,
)

BASELINE = {
    "scenario": "No TPF baseline", "tpf_available": 0.0, "funding_share": 0.0, "funder_award_share": 0.0,
    "monitoring_intensity": 0.0, "settlement_control": 0.0, "adverse_cost_coverage": 0.0,
    "claimant_copay_share": 0.0, "staged_funding_efficiency": 0.0, "security_for_costs_probability": 0.0,
    "min_roi_multiple": 0.0,
}
PRIMARY = dict(funding_share=0.85, funder_award_share=0.22, monitoring_intensity=0.85, settlement_control=0.05,
               claimant_copay_share=0.10, adverse_cost_coverage=0.70)
TERM_COLUMNS = list(PRIMARY)


def _is_primary(df: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for col, val in PRIMARY.items():
        mask &= np.isclose(df[col].to_numpy(), val)
    return mask


def run_grid_checks(n_cases: int = 150_000, seed: int = 1514) -> dict:
    cases = generate_cases(SimulationConfig(n_cases=n_cases, seed=seed))
    search = cases.iloc[:SEARCH_CASES].copy()
    base = summarize_results(evaluate_scenario(search, BASELINE))
    candidates, _ = grid_search(cases, base)
    primary_score = float(candidates.loc[_is_primary(candidates), "balanced_score"].iloc[0])

    # (a) settlement control at the conventional level of 0.70
    ext = []
    for award in AWARD_SHARES:
        for monitoring in MONITORING:
            for copay in COPAYS:
                for coverage in ADVERSE_COVERAGE:
                    for funding in FUNDING_SHARES:
                        terms = candidate_terms(funding, award, monitoring, 0.70, copay, coverage)
                        s = summarize_results(evaluate_scenario(search, terms))
                        ext.append(balanced_score(s, base, 0.33))
    ext = np.asarray(ext)
    all_scores = np.concatenate([candidates["balanced_score"].to_numpy(), ext])

    # (b) settlement-alignment component removed
    weights = {**DEFAULT_SCORE_WEIGHTS, "settlement_alignment": 0.0}
    rescored = candidates.apply(lambda r: weighted_score(score_components(r, base, 0.33), weights), axis=1)
    order = rescored.sort_values(ascending=False).index
    top = candidates.loc[order[: max(1, round(0.01 * len(candidates)))]]

    return {
        "search_candidates": len(candidates),
        "settlement_control_levels_searched": [float(x) for x in SETTLEMENT_CONTROL],
        "primary_selection_score": primary_score,
        "candidates_at_settlement_control_070": len(ext),
        "max_score_at_settlement_control_070": float(ext.max()),
        "primary_rank_among_6480": int((all_scores > primary_score).sum() + 1),
        "primary_rank_without_settlement_alignment": int((rescored > rescored[_is_primary(candidates)].iloc[0]).sum() + 1),
        "leader_without_settlement_alignment": candidates.loc[order[0], TERM_COLUMNS].to_dict(),
        "top_1pct_share_monitoring_upper_bound": float(np.isclose(top["monitoring_intensity"], MONITORING.max()).mean()),
        "top_1pct_share_settlement_control_lower_bound": float(np.isclose(top["settlement_control"], SETTLEMENT_CONTROL.min()).mean()),
    }


if __name__ == "__main__":
    for k, v in run_grid_checks().items():
        print(f"{k}: {v}")
