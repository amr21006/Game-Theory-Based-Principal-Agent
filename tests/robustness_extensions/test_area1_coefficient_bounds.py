"""Area 1: Behavioral Coefficient Perturbation & Elicitation Stress Test Suite.

Tests the stability of model findings when behavioral parameters (advocacy gain,
underfunding penalty, monitoring gains, inflation penalties, complexity, asymmetry,
costs, and pass-throughs) are perturbed by +/-25% and +/-50%.
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

STANDARD_SCENARIOS = [
    {
        "scenario": "No funding",
        "tpf_available": 0,
        "funding_share": 0.0,
        "funder_award_share": 0.0,
        "monitoring_intensity": 0.0,
        "settlement_control": 0.0,
        "adverse_cost_coverage": 0.0,
        "claimant_copay_share": 0.0,
        "staged_funding_efficiency": 0.0,
        "security_for_costs_probability": 0.0,
        "min_roi_multiple": 0.0,
    },
    {
        "scenario": "Conventional",
        "tpf_available": 1,
        "funding_share": 1.0,
        "funder_award_share": 0.35,
        "monitoring_intensity": 0.25,
        "settlement_control": 0.70,
        "adverse_cost_coverage": 1.0,
        "claimant_copay_share": 0.0,
        "staged_funding_efficiency": 0.05,
        "security_for_costs_probability": 0.05,
        "min_roi_multiple": 1.35,
    },
    {
        "scenario": "Regulated disclosure",
        "tpf_available": 1,
        "funding_share": 0.90,
        "funder_award_share": 0.26,
        "monitoring_intensity": 0.60,
        "settlement_control": 0.15,
        "adverse_cost_coverage": 0.80,
        "claimant_copay_share": 0.10,
        "staged_funding_efficiency": 0.15,
        "security_for_costs_probability": 0.20,
        "min_roi_multiple": 1.20,
    },
    {
        "scenario": "Portfolio",
        "tpf_available": 1,
        "funding_share": 0.95,
        "funder_award_share": 0.24,
        "monitoring_intensity": 0.50,
        "settlement_control": 0.25,
        "adverse_cost_coverage": 0.90,
        "claimant_copay_share": 0.05,
        "staged_funding_efficiency": 0.22,
        "security_for_costs_probability": 0.08,
        "min_roi_multiple": 1.05,
    },
    {
        "scenario": "Grid-selected",
        "tpf_available": 1,
        "funding_share": 0.85,
        "funder_award_share": 0.22,
        "monitoring_intensity": 0.85,
        "settlement_control": 0.05,
        "adverse_cost_coverage": 0.70,
        "claimant_copay_share": 0.10,
        "staged_funding_efficiency": 0.245,
        "security_for_costs_probability": 0.136,
        "min_roi_multiple": 1.18,
    },
]


def run_area1_coefficient_stress(
    n_cases: int = 50_000,
    seed: int = 1514,
) -> pd.DataFrame:
    """Evaluate scenario metrics under systematic +/-25% and +/-50% coefficient perturbations."""
    config = SimulationConfig(n_cases=n_cases, seed=seed)
    cases = generate_cases(config)

    perturbations = [
        ("Base Parameterization", 1.0, "base"),
        ("Advocacy Gain Amplitude +50%", 1.5, "advocacy_amp_high"),
        ("Advocacy Gain Amplitude -50%", 0.5, "advocacy_amp_low"),
        ("Underfunding Penalty +50%", 1.5, "underfund_pen_high"),
        ("Underfunding Penalty -50%", 0.5, "underfund_pen_low"),
        ("Monitoring Gain +50%", 1.5, "monitoring_gain_high"),
        ("Monitoring Gain -50%", 0.5, "monitoring_gain_low"),
        ("Claim Inflation Penalty +50%", 1.5, "inflation_pen_high"),
        ("Claim Inflation Penalty -50%", 0.5, "inflation_pen_low"),
        ("Complexity Penalty +50%", 1.5, "complexity_pen_high"),
        ("Complexity Penalty -50%", 0.5, "complexity_pen_low"),
        ("Information Asymmetry Penalty +50%", 1.5, "asym_pen_high"),
        ("Information Asymmetry Penalty -50%", 0.5, "asym_pen_low"),
        ("Counterclaim Cost Fraction +50% (0.225)", 1.5, "cc_cost_high"),
        ("Counterclaim Cost Fraction -50% (0.075)", 0.5, "cc_cost_low"),
        ("Effective Award Pass-Through +50% (0.375)", 1.5, "passthrough_high"),
        ("Effective Award Pass-Through -50% (0.125)", 0.5, "passthrough_low"),
    ]

    records = []

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

    for name, mult, tag in perturbations:
        adv_amp = 0.060 * (mult if "advocacy_amp" in tag else 1.0)
        underfund_pen = 0.090 * (mult if "underfund_pen" in tag else 1.0)
        mon_lin = 0.045 * (mult if "monitoring_gain" in tag else 1.0)
        mon_quad = -0.018 * (mult if "monitoring_gain" in tag else 1.0)
        inf_pen = 0.120 * (mult if "inflation_pen" in tag else 1.0)
        com_pen = 0.035 * (mult if "complexity_pen" in tag else 1.0)
        asym_pen = 0.025 * (mult if "asym_pen" in tag else 1.0)
        cc_cost_frac = 0.15 * (mult if "cc_cost" in tag else 1.0)
        pass_through = 0.25 * (mult if "passthrough" in tag else 1.0)

        c_cc = c * cc_cost_frac
        ad_cc = ad * 0.5

        baseline_net = None

        for row in STANDARD_SCENARIOS:
            s_name = row["scenario"]
            tpf_avail = bool(row["tpf_available"])
            theta_f = float(row["funding_share"])
            f_share = float(row["funder_award_share"])
            gamma = float(row["monitoring_intensity"])
            sc = float(row["settlement_control"])
            cp = float(row["claimant_copay_share"])
            ac = float(row["adverse_cost_coverage"])
            eta = float(row["staged_funding_efficiency"])
            p_sec = float(row["security_for_costs_probability"])
            min_roi = float(row["min_roi_multiple"])

            if not tpf_avail:
                s_eff = np.ones_like(m)
                inf = np.zeros_like(m)
                b_gamma = 0.0
            else:
                s_eff = np.full_like(m, theta_f + cp)
                inf = 0.18 * (1.0 - gamma) * (1.0 - cp) * (0.70 + 0.60 * sc)
                b_gamma = mon_lin * gamma + mon_quad * (gamma ** 2)

            a_gain = adv_amp * (1.0 - np.exp(-1.35 * s_eff))
            a_pen = underfund_pen * np.maximum(0.0, 1.0 - s_eff)
            d_inf = inf_pen * inf
            k_com = com_pen * com
            i_asym = asym_pen * asym

            p = np.clip(
                m + a_gain - a_pen + b_gamma - d_inf - k_com - i_asym - 0.16,
                0.03,
                0.94,
            )
            v_eff = v * (1.0 + pass_through * inf)

            u_c_self = (
                p * v
                - (1.0 - p) * (c + ad)
                - p_cc * (v_cc + c_cc + ad_cc)
            )
            can_self = l_c >= (c + c_cc)
            self_worth = u_c_self > daab
            file_self = can_self & self_worth

            c_mon = c * (0.025 + 0.045 * gamma)
            c_sec = p_sec * ad * (0.10 + 0.25 * (1.0 - eta))
            c_f_dep = theta_f * c * (1.0 - eta)
            c_f_dep_cc = theta_f * c_cc * (1.0 - eta)
            c_f = (
                (1.0 - p) * (c_f_dep + c_f_dep_cc + ac * ad)
                + ac * p_cc * ad_cc
                + c_mon
                + c_sec
            )
            u_f_arb = f_share * p * v_eff - c_f
            roi_f = np.where(c_f > 1e-9, (f_share * p * v_eff) / c_f, 0.0)
            funder_accepts = tpf_avail & (u_f_arb > 0) & (roi_f >= min_roi)

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

            file_case = file_tpf if tpf_avail else file_self
            constrained = l_c < c
            constrained_access = (file_tpf & constrained).mean()

            offer = v * spo
            if not tpf_avail:
                r_self = u_c_self + 0.18 * v - 0.05 * r_av * v
                scale_self = 0.090 * v + c
                p_settle = np.where(file_case, logistic((offer - r_self) / scale_self), 0.0)
            else:
                delta_settle = (0.08 + 0.08 * sc + 0.04 * (1.0 - ac)) * v
                r_c_tpf = u_c_tpf + delta_settle - 0.04 * r_av * v
                u_c_settle = (1.0 - f_share) * offer - cp * (c + c_cc) + r_liq
                scale_c = 0.085 * v + c
                p_c_settle = logistic((u_c_settle - r_c_tpf) / scale_c)

                c_f_settle = (0.35 + 0.30 * (1.0 - eta)) * (c_f_dep + c_f_dep_cc) + c_mon
                u_f_settle = f_share * offer - c_f_settle
                scale_f = 0.080 * v + c
                p_f_settle = logistic((u_f_settle - u_f_arb) / scale_f)

                p_settle = np.where(
                    file_case,
                    p_c_settle * ((1.0 - sc) + sc * p_f_settle),
                    0.0,
                )

            filed_mask = file_case
            filed_rate = filed_mask.mean()
            settled_rate_filed = p_settle[filed_mask].mean() if filed_mask.sum() > 0 else 0.0

            if not tpf_avail:
                mh = 0.0
            else:
                mh = (1.0 - gamma) * (1.0 - cp) * (0.55 + 0.45 * sc)

            mean_roi_f = roi_f[filed_mask].mean() if filed_mask.sum() > 0 and tpf_avail else 0.0
            mean_c_net = np.where(file_case, u_c_tpf if tpf_avail else u_c_self, daab).mean()

            if s_name == "No funding":
                baseline_net = mean_c_net

            sa = 1.0 - np.abs(settled_rate_filed - 0.33) / 0.33
            rf_score = min(1.5, mean_roi_f / 1.18) if tpf_avail else 0.0
            mq = 1.0 - mh
            base_ref = baseline_net if (baseline_net is not None and baseline_net > 0) else mean_c_net
            gc = max(0.0, (mean_c_net - base_ref) / base_ref) if base_ref > 0 else 0.0

            score_j = (
                0.30 * constrained_access
                + 0.20 * gc
                + 0.20 * sa
                + 0.15 * rf_score
                + 0.10 * mq
                - 0.05 * 0.10
            )

            records.append({
                "perturbation": name,
                "scenario": s_name,
                "filed_rate": filed_rate,
                "settlement_rate_filed": settled_rate_filed,
                "constrained_access": constrained_access,
                "mean_claimant_net": mean_c_net,
                "funder_roi": mean_roi_f,
                "governance_mh": mh,
                "score_j": score_j,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = run_area1_coefficient_stress(n_cases=50_000)
    out_dir = ROOT / "tests" / "robustness_extensions" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "area1_coefficient_stress_results.csv", index=False)
    print(f"Area 1 stress test completed. {len(df)} records written to CSV.")
