"""Area 3: Legal & Security-for-Costs Procedural Risk Stress Test Suite.

Quantifies the sensitivity of TPF viability, funder participation, and claimant access
to aggressive tribunal security-for-costs orders (P_sec from 0.0 to 0.70) and varying
adverse-cost allocations (AC from 0.50 to 1.00). Identifies the threshold where security
demands chill legitimate construction arbitration funding.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpf_model import (
    SimulationConfig,
    generate_cases,
    logistic,
)


def run_area3_security_risk_stress(
    n_cases: int = 50_000,
    seed: int = 1514,
) -> pd.DataFrame:
    """Evaluate funder participation, claimant access, and payoffs across security risks and adverse cost rules."""
    config = SimulationConfig(n_cases=n_cases, seed=seed)
    cases = generate_cases(config)

    m = cases["merit"].to_numpy()
    v = cases["claim_value_musd"].to_numpy()
    c = cases["required_cost_musd"].to_numpy()
    l_c = cases["available_liquidity_musd"].to_numpy()
    v_cc = cases["counterclaim_pct"].to_numpy() * v
    p_cc = cases["counterclaim_success_prob"].to_numpy()
    ad = cases["adverse_cost_musd"].to_numpy()
    daab = cases["daab_pct"].to_numpy() * v
    spo = cases["settlement_offer_pct"].to_numpy()
    com = cases["complexity"].to_numpy()
    asym = cases["information_asymmetry"].to_numpy()
    r_av = cases["risk_aversion"].to_numpy()
    reinvest = cases["reinvestment_roi"].to_numpy()

    c_cc = c * 0.15
    ad_cc = ad * 0.5

    # Fixed grid-selected baseline terms
    theta_f = 0.85
    f_share = 0.22
    gamma = 0.85
    sc = 0.05
    cp = 0.10
    eta = 0.245
    min_roi = 1.18

    # Vary P_sec and AC
    p_sec_levels = [0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
    ac_levels = [0.50, 0.70, 0.85, 1.00]

    records = []

    s_eff = np.full_like(m, theta_f + cp)
    inf = 0.18 * (1.0 - gamma) * (1.0 - cp) * (0.70 + 0.60 * sc)
    b_gamma = 0.045 * gamma - 0.018 * (gamma ** 2)
    a_gain = 0.060 * (1.0 - np.exp(-1.35 * s_eff))
    a_pen = 0.090 * np.maximum(0.0, 1.0 - s_eff)
    d_inf = 0.120 * inf
    k_com = 0.035 * com
    i_asym = 0.025 * asym

    p = np.clip(m + a_gain - a_pen + b_gamma - d_inf - k_com - i_asym - 0.16, 0.03, 0.94)
    v_eff = v * (1.0 + 0.25 * inf)
    c_mon = c * (0.025 + 0.045 * gamma)
    c_f_dep = theta_f * c * (1.0 - eta)
    c_f_dep_cc = theta_f * c_cc * (1.0 - eta)

    for ac in ac_levels:
        for p_sec in p_sec_levels:
            c_sec = p_sec * ad * (0.10 + 0.25 * (1.0 - eta))
            c_f = (
                (1.0 - p) * (c_f_dep + c_f_dep_cc + ac * ad)
                + ac * p_cc * ad_cc
                + c_mon
                + c_sec
            )
            u_f_arb = f_share * p * v_eff - c_f
            roi_f = np.where(c_f > 1e-9, (f_share * p * v_eff) / c_f, 0.0)
            funder_accepts = (u_f_arb > 0) & (roi_f >= min_roi)

            preserved_cash = np.maximum(
                0.0,
                np.minimum(l_c - cp * (c + c_cc), theta_f * (c + c_cc)),
            )
            r_liq = preserved_cash * reinvest
            u_c_tpf = (
                (1.0 - f_share) * p * v_eff
                - (1.0 - p) * (cp * c + (1.0 - ac) * ad)
                - p_cc * (v_cc + cp * c_cc + (1.0 - ac) * ad_cc)
                + r_liq
            )
            tpf_worth = u_c_tpf > daab
            file_tpf = funder_accepts & tpf_worth

            constrained = l_c < c
            constrained_access = (file_tpf & constrained).mean()
            filed_rate = file_tpf.mean()
            funder_acceptance_rate = funder_accepts.mean()

            mean_roi_f = roi_f[file_tpf].mean() if file_tpf.sum() > 0 else 0.0
            mean_c_net = u_c_tpf[file_tpf].mean() if file_tpf.sum() > 0 else 0.0
            mean_sec_cost = c_sec[file_tpf].mean() if file_tpf.sum() > 0 else 0.0

            records.append({
                "security_probability_p_sec": p_sec,
                "adverse_cost_coverage_ac": ac,
                "funder_acceptance_rate": funder_acceptance_rate,
                "overall_filed_rate": filed_rate,
                "constrained_access_rate": constrained_access,
                "mean_funder_roi_multiple": mean_roi_f,
                "mean_claimant_net_filed_musd": mean_c_net,
                "mean_security_cost_musd": mean_sec_cost,
                "chilled_market_flag": funder_acceptance_rate < 0.20,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = run_area3_security_risk_stress(n_cases=50_000)
    out_dir = ROOT / "tests" / "robustness_extensions" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "area3_security_risk_stress_results.csv", index=False)
    print(f"Area 3 Security-for-Costs & Legal Risk Stress completed. {len(df)} configurations evaluated.")
    print(df[df["adverse_cost_coverage_ac"] == 0.70][["security_probability_p_sec", "funder_acceptance_rate", "constrained_access_rate", "mean_funder_roi_multiple"]])
