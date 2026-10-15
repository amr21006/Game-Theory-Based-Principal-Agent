"""Area 1: behavioral-coefficient perturbations (supplement Table S32).

The eight coefficients that carry the proof-quality and governance effects are each varied by
+/-50%, giving sixteen perturbed parameterizations. Each is evaluated for all five structures and
scored by Eq. (15) against its own no-funding baseline.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from tests.robustness_extensions.common import Params, run_structures
from src.tpf_model import SimulationConfig, generate_cases

COEFFICIENTS = [
    ("Advocacy-gain amplitude", "adv_amp"),
    ("Underfunding penalty", "underfund_pen"),
    ("Monitoring gain", "monitoring_mult"),
    ("Claim-inflation credibility penalty", "inflation_pen"),
    ("Complexity penalty", "complexity_pen"),
    ("Information-asymmetry penalty", "asymmetry_pen"),
    ("Counterclaim cost fraction", "cc_cost_frac"),
    ("Effective-award pass-through", "passthrough"),
]


def perturbations() -> list[tuple[str, Params]]:
    base = Params()
    out = [("Base parameterization", base)]
    for label, field in COEFFICIENTS:
        for mult, tag in ((1.5, "+50%"), (0.5, "-50%")):
            out.append((f"{label} {tag}", replace(base, **{field: getattr(base, field) * mult})))
    return out


def run_area1_coefficient_stress(n_cases: int = 150_000, seed: int = 1514) -> pd.DataFrame:
    """One row per perturbation and structure."""
    cases = generate_cases(SimulationConfig(n_cases=n_cases, seed=seed))
    rows = []
    for label, P in perturbations():
        summaries, scores, _ = run_structures(cases, P)
        for name, s in summaries.items():
            rows.append({
                "perturbation": label,
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
    df = run_area1_coefficient_stress()
    print(df.pivot(index="perturbation", columns="scenario", values="score_j").round(4))
