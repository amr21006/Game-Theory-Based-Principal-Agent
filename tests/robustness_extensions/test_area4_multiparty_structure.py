"""Area 4: Multi-Party Construction Dispute & Subcontractor Chain Architecture.

Implements an explicit multi-party dispute formulation:
- Main Contractor (Claimant) vs. Employer (Respondent 1) & Specialist Subcontractor (Respondent 2).
- Dual counterclaims (Employer delay damages + Subcontractor defect/indemnity cross-claims).
- Joint vs. bilateral settlement dynamics and adverse-cost apportionment.
Evaluates the robustness of TPF agreement terms in multi-party construction settings.
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


def run_area4_multiparty_analysis(
    n_cases: int = 50_000,
    seed: int = 1514,
) -> pd.DataFrame:
    """Evaluate single-respondent vs. multi-party (Employer + Subcontractor) dispute architectures."""
    config = SimulationConfig(n_cases=n_cases, seed=seed)
    cases = generate_cases(config)
    rng = np.random.default_rng(seed + 999)

    m = cases["merit"].to_numpy()
    v = cases["claim_value_musd"].to_numpy()
    c = cases["required_cost_musd"].to_numpy()
    l_c = cases["available_liquidity_musd"].to_numpy()
    ad = cases["adverse_cost_musd"].to_numpy()
    daab = cases["daab_pct"].to_numpy() * v
    spo = cases["settlement_offer_pct"].to_numpy()
    com = cases["complexity"].to_numpy()
    asym = cases["information_asymmetry"].to_numpy()
    r_av = cases["risk_aversion"].to_numpy()
    reinvest = cases["reinvestment_roi"].to_numpy()

    v_cc_total = cases["counterclaim_pct"].to_numpy() * v
    sub_share = rng.beta(2.0, 3.0, size=len(cases))
    v_cc_emp = v_cc_total * (1.0 - sub_share)
    v_cc_sub = v_cc_total * sub_share

    p_cc_base = cases["counterclaim_success_prob"].to_numpy()
    p_cc_emp = np.clip(p_cc_base + rng.normal(0, 0.05, len(cases)), 0.01, 0.60)
    p_cc_sub = np.clip(p_cc_base + rng.normal(0, 0.05, len(cases)), 0.01, 0.60)

    c_cc_emp = c * 0.10
    c_cc_sub = c * 0.08
    c_cc_total = c_cc_emp + c_cc_sub

    ad_cc_emp = ad * 0.35
    ad_cc_sub = ad * 0.25
    ad_cc_total = ad_cc_emp + ad_cc_sub

    records = []

    for model_type in ["Single-Respondent Baseline", "Multi-Party (Employer + Subcontractor)"]:
        is_multiparty = "Multi-Party" in model_type

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
                b_gamma = 0.045 * gamma - 0.018 * (gamma ** 2)

            com_mult = 1.20 if is_multiparty else 1.0
            a_gain = 0.060 * (1.0 - np.exp(-1.35 * s_eff))
            a_pen = 0.090 * np.maximum(0.0, 1.0 - s_eff)
            d_inf = 0.120 * inf
            k_com = 0.035 * com_mult * com
            i_asym = 0.025 * asym

            p = np.clip(m + a_gain - a_pen + b_gamma - d_inf - k_com - i_asym - 0.16, 0.03, 0.94)
            v_eff = v * (1.0 + 0.25 * inf)

            curr_c_cc = c_cc_total if is_multiparty else (c * 0.15)
            curr_ad_cc = ad_cc_total if is_multiparty else (ad * 0.5)
            if is_multiparty:
                exp_cc_loss = p_cc_emp * (v_cc_emp + c_cc_emp + ad_cc_emp) + p_cc_sub * (v_cc_sub + c_cc_sub + ad_cc_sub)
                exp_cc_funder_cost = ac * (p_cc_emp * ad_cc_emp + p_cc_sub * ad_cc_sub)
                exp_cc_claimant_loss = (
                    p_cc_emp * (v_cc_emp + cp * c_cc_emp + (1.0 - ac) * ad_cc_emp)
                    + p_cc_sub * (v_cc_sub + cp * c_cc_sub + (1.0 - ac) * ad_cc_sub)
                )
            else:
                exp_cc_loss = p_cc_base * (v_cc_total + curr_c_cc + curr_ad_cc)
                exp_cc_funder_cost = ac * p_cc_base * curr_ad_cc
                exp_cc_claimant_loss = p_cc_base * (v_cc_total + cp * curr_c_cc + (1.0 - ac) * curr_ad_cc)

            # Self-funded
            u_c_self = p * v - (1.0 - p) * (c + ad) - exp_cc_loss
            can_self = l_c >= (c + curr_c_cc)
            self_worth = u_c_self > daab
            file_self = can_self & self_worth

            # Funder
            c_mon = c * (0.025 + 0.045 * gamma)
            c_sec = p_sec * ad * (0.10 + 0.25 * (1.0 - eta))
            c_f_dep = theta_f * c * (1.0 - eta)
            c_f_dep_cc = theta_f * curr_c_cc * (1.0 - eta)
            c_f = (
                (1.0 - p) * (c_f_dep + c_f_dep_cc + ac * ad)
                + exp_cc_funder_cost
                + c_mon
                + c_sec
            )
            u_f_arb = f_share * p * v_eff - c_f
            roi_f = np.where(c_f > 1e-9, (f_share * p * v_eff) / c_f, 0.0)
            funder_accepts = tpf_avail & (u_f_arb > 0) & (roi_f >= min_roi)

            # Funded claimant
            preserved_cash = np.maximum(
                0.0,
                np.minimum(l_c - cp * (c + curr_c_cc), theta_f * (c + curr_c_cc)),
            )
            r_liq = preserved_cash * reinvest
            u_c_tpf = (
                (1.0 - f_share) * p * v_eff
                - (1.0 - p) * (cp * c + (1.0 - ac) * ad)
                - exp_cc_claimant_loss
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
                u_c_settle = (1.0 - f_share) * offer - cp * (c + curr_c_cc) + r_liq
                scale_c = 0.085 * v + c
                p_c_settle = logistic((u_c_settle - r_c_tpf) / scale_c)

                c_f_settle = (0.35 + 0.30 * (1.0 - eta)) * (c_f_dep + c_f_dep_cc) + c_mon
                u_f_settle = f_share * offer - c_f_settle
                scale_f = 0.080 * v + c
                p_f_settle = logistic((u_f_settle - u_f_arb) / scale_f)

                multi_factor = 0.94 if is_multiparty else 1.0
                p_settle = np.where(
                    file_case,
                    multi_factor * p_c_settle * ((1.0 - sc) + sc * p_f_settle),
                    0.0,
                )

            filed_mask = file_case
            filed_rate = filed_mask.mean()
            settled_rate_filed = p_settle[filed_mask].mean() if filed_mask.sum() > 0 else 0.0
            mean_roi_f = roi_f[filed_mask].mean() if filed_mask.sum() > 0 and tpf_avail else 0.0
            mean_c_net = np.where(file_case, u_c_tpf if tpf_avail else u_c_self, daab).mean()

            records.append({
                "model_architecture": model_type,
                "scenario": s_name,
                "filed_rate": filed_rate,
                "settlement_rate_filed": settled_rate_filed,
                "constrained_access": constrained_access,
                "mean_claimant_net_musd": mean_c_net,
                "mean_funder_roi_multiple": mean_roi_f,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = run_area4_multiparty_analysis(n_cases=50_000)
    out_dir = ROOT / "tests" / "robustness_extensions" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "area4_multiparty_analysis_results.csv", index=False)
    print(f"Area 4 Multi-Party Analysis completed. {len(df)} configurations evaluated.")
    print(df[["model_architecture", "scenario", "filed_rate", "settlement_rate_filed", "mean_funder_roi_multiple"]])
