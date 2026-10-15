"""Area 3: security-for-costs and adverse-cost coverage stress (supplement Table S34).

The security-for-costs probability is varied from 0.00 to 0.70 at adverse-cost coverage of 0.70,
the grid-selected level, and of 1.00, with the other grid-selected terms held fixed.
"""

from __future__ import annotations

import pandas as pd

from tests.robustness_extensions.common import GRID_SELECTED, Params
from src.tpf_model import SimulationConfig, generate_cases, summarize_results
from src.tpf_param import evaluate

GRID = (
    (0.70, (0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70)),
    (1.00, (0.00, 0.10, 0.35, 0.70)),
)


def run_area3_security_risk_stress(n_cases: int = 150_000, seed: int = 1514) -> pd.DataFrame:
    cases = generate_cases(SimulationConfig(n_cases=n_cases, seed=seed))
    rows = []
    for coverage, probabilities in GRID:
        for p_sec in probabilities:
            terms = {**GRID_SELECTED, "adverse_cost_coverage": coverage, "security_for_costs_probability": p_sec}
            res = evaluate(cases, terms, Params())
            s = summarize_results(res)
            funded = res["used_tpf"].to_numpy()
            rows.append({
                "adverse_cost_coverage": coverage,
                "security_probability_p_sec": p_sec,
                "funder_acceptance_rate": s["fundable_rate"],
                "filed_rate": s["filed_rate"],
                "constrained_access": s["constrained_access_rate"],
                "mean_funder_roi_multiple": s["mean_funder_roi_multiple"],
                "mean_claimant_net_musd": s["mean_claimant_net_musd"],
                "mean_security_cost_musd": float(res.loc[funded, "expected_security_cost_musd"].mean()),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run_area3_security_risk_stress().round(4).to_string(index=False))
