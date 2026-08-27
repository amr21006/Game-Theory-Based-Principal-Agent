"""Area 2: UNCTAD Construction-Sector Recalibration Sensitivity (35.1% Win Rate).

Calibrates the baseline success probability to the observed UNCTAD construction-sector
investor success rate of 35.14% (from 74 tribunal decisions), rather than the aggregate
ICSID 53.0% benchmark. Evaluates scenario outcomes and grid-search stability.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

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


def find_unctad_calibration_shift(cases: pd.DataFrame, target_win_rate: float = 0.3514) -> float:
    """Find the additive success probability shift that yields the target win rate in the baseline."""
    m = cases["merit"].to_numpy()
    v = cases["claim_value_musd"].to_numpy()
    c = cases["required_cost_musd"].to_numpy()
    l_c = cases["available_liquidity_musd"].to_numpy()
    v_cc = cases["counterclaim_pct"].to_numpy() * v
    p_cc = cases["counterclaim_success_prob"].to_numpy()
    ad = cases["adverse_cost_musd"].to_numpy()
    c_cc = c * 0.15
    ad_cc = ad * 0.5
    daab = cases["daab_pct"].to_numpy() * v

    a_gain = 0.060 * (1.0 - np.exp(-1.35))
    k_com = 0.035 * cases["complexity"].to_numpy()
    i_asym = 0.025 * cases["information_asymmetry"].to_numpy()

    def objective(shift: float) -> float:
        p = np.clip(m + a_gain - k_com - i_asym + shift, 0.03, 0.94)
        u_c_self = p * v - (1.0 - p) * (c + ad) - p_cc * (v_cc + c_cc + ad_cc)
        can_self = l_c >= (c + c_cc)
        self_worth = u_c_self > daab
        filed = can_self & self_worth
        if filed.sum() == 0:
            return 1.0
        return float(p[filed].mean() - target_win_rate)

    res = root_scalar(objective, bracket=[-0.60, 0.10], method="brentq")
    return float(res.root)


def run_area2_unctad_calibration(
    n_cases: int = 100_000,
    seed: int = 1514,
    target_win_rate: float = 0.3514,
) -> tuple[pd.DataFrame, float]:
    """Run all 5 scenarios under the UNCTAD construction investor win rate calibration."""
    config = SimulationConfig(n_cases=n_cases, seed=seed)
    cases = generate_cases(config)

    shift = find_unctad_calibration_shift(cases, target_win_rate)

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

    records = []
    base_claimant_net = 0.0

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

        a_gain = 0.060 * (1.0 - np.exp(-1.35 * s_eff))
        a_pen = 0.090 * np.maximum(0.0, 1.0 - s_eff)
        d_inf = 0.120 * inf
        k_com = 0.035 * com
        i_asym = 0.025 * asym

        p = np.clip(
            m + a_gain - a_pen + b_gamma - d_inf - k_com - i_asym + shift,
            0.03,
            0.94,
        )
        v_eff = v * (1.0 + 0.25 * inf)

        # Self-funded
        u_c_self = p * v - (1.0 - p) * (c + ad) - p_cc * (v_cc + c_cc + ad_cc)
        can_self = l_c >= (c + c_cc)
        self_worth = u_c_self > daab
        file_self = can_self & self_worth

        # Funder
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

        # Funded claimant
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
        mean_p_win_filed = p[filed_mask].mean() if filed_mask.sum() > 0 else 0.0
        settled_rate_filed = p_settle[filed_mask].mean() if filed_mask.sum() > 0 else 0.0

        if not tpf_avail:
            mh = 0.0
        else:
            mh = (1.0 - gamma) * (1.0 - cp) * (0.55 + 0.45 * sc)

        mean_roi_f = roi_f[filed_mask].mean() if filed_mask.sum() > 0 and tpf_avail else 0.0
        mean_c_net = np.where(file_case, u_c_tpf if tpf_avail else u_c_self, daab).mean()

        if s_name == "No funding":
            base_claimant_net = mean_c_net

        sa = 1.0 - np.abs(settled_rate_filed - 0.33) / 0.33
        rf_score = min(1.5, mean_roi_f / 1.18) if tpf_avail else 0.0
        mq = 1.0 - mh
        gc = max(0.0, (mean_c_net - base_claimant_net) / base_claimant_net) if base_claimant_net > 0 else 0.0

        # Respondent outflow
        u_resp_arb = p * v_eff + (0.24 if tpf_avail else 0.22) * c - (1.0 - p) * ad - p_cc * v_cc
        u_resp_settle = offer + 0.20 * c
        exp_resp_outflow = np.where(
            file_case,
            p_settle * u_resp_settle + (1.0 - p_settle) * u_resp_arb,
            0.5 * daab,
        ).mean()

        score_j = (
            0.30 * constrained_access
            + 0.20 * gc
            + 0.20 * sa
            + 0.15 * rf_score
            + 0.10 * mq
            - 0.05 * 0.10
        )

        records.append({
            "calibration": f"UNCTAD 35.1% (shift={shift:.4f})",
            "scenario": s_name,
            "filed_rate": filed_rate,
            "mean_success_prob_filed": mean_p_win_filed,
            "settlement_rate_filed": settled_rate_filed,
            "constrained_access": constrained_access,
            "mean_claimant_net_musd": mean_c_net,
            "mean_funder_roi_multiple": mean_roi_f,
            "governance_mh_indicator": mh,
            "mean_respondent_outflow_musd": exp_resp_outflow,
            "score_j": score_j,
        })

    return pd.DataFrame(records), shift


if __name__ == "__main__":
    df, shift = run_area2_unctad_calibration(n_cases=100_000)
    out_dir = ROOT / "tests" / "robustness_extensions" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "area2_unctad_calibration_results.csv", index=False)
    print(f"Area 2 UNCTAD Calibration (Target=35.14%) completed.")
    print(f"Calculated shift: {shift:.4f}")
    print(df[["scenario", "filed_rate", "mean_success_prob_filed", "settlement_rate_filed", "mean_funder_roi_multiple", "score_j"]])
