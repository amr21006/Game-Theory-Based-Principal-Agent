"""Area 4: employer and subcontractor dispute architecture (supplement Table S35).

Each case's counterclaim is divided between an employer claim for delay damages and a subcontractor
cross-claim, with a Beta(2, 3) subcontractor share, separate success probabilities, defense costs of
0.10 and 0.08 of the required arbitration cost, adverse-cost exposures of 0.105 and 0.075 of the
claimant's adverse-cost exposure, a 1.2 complexity multiplier, and a 0.94 settlement factor for the
agreement of a second respondent (``Params(multiparty=True)``). The single-respondent architecture is
the primary model.
"""

from __future__ import annotations

import pandas as pd

from tests.robustness_extensions.common import Params, run_structures
from src.tpf_model import SimulationConfig, generate_cases

ARCHITECTURES = (
    ("Single-Respondent Baseline", Params()),
    ("Multi-Party (Employer + Subcontractor)", Params(multiparty=True)),
)


def run_area4_multiparty_analysis(n_cases: int = 150_000, seed: int = 1514) -> pd.DataFrame:
    cases = generate_cases(SimulationConfig(n_cases=n_cases, seed=seed))
    rows = []
    for label, P in ARCHITECTURES:
        summaries, scores, _ = run_structures(cases, P)
        for name, s in summaries.items():
            rows.append({
                "model_architecture": label,
                "scenario": name,
                "filed_rate": s["filed_rate"],
                "settlement_rate_filed": s["settlement_rate_filed"],
                "constrained_access": s["constrained_access_rate"],
                "mean_claimant_net_musd": s["mean_claimant_net_musd"],
                "mean_funder_roi_multiple": s["mean_funder_roi_multiple"],
                "score_j": scores[name],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run_area4_multiparty_analysis().round(4).to_string(index=False))
