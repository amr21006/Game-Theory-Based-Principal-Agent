from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


EPS = 1e-9
SUCCESS_CALIBRATION_SHIFT = -0.16
SELF_FUNDED_ESCALATION_FRICTION = 0.18
DEFAULT_SCORE_WEIGHTS = {
    "access": 0.30,
    "claimant_gain": 0.20,
    "settlement_alignment": 0.20,
    "roi": 0.15,
    "moral_quality": 0.10,
    "respondent_penalty": -0.05,
}


@dataclass(frozen=True)
class SimulationConfig:
    n_cases: int = 150_000
    seed: int = 1514
    mean_amount_musd: float = 130.0
    target_settlement_rate: float = 0.33


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def generate_cases(config: SimulationConfig) -> pd.DataFrame:
    """Generate common-random-number construction arbitration cases.

    Values are in USD millions. The amount-in-dispute distribution is
    calibrated to the public ICC 2024 average of USD 130 million while
    preserving a right-skewed construction arbitration profile.
    """

    rng = np.random.default_rng(config.seed)
    n = config.n_cases

    sigma = 1.62
    median = config.mean_amount_musd / np.exp((sigma**2) / 2.0)
    claim_value = rng.lognormal(mean=np.log(median), sigma=sigma, size=n)
    claim_value = np.clip(claim_value, 0.25, 2_500.0)
    # Scaled by theoretical mean of unclipped distribution to avoid sample coupling and preserve CRN



    base_cost_ratio = 0.025 + 0.155 * np.exp(-claim_value / 55.0)
    cost_noise = rng.lognormal(mean=0.0, sigma=0.42, size=n)
    required_cost_ratio = np.clip(base_cost_ratio * cost_noise, 0.012, 0.24)
    required_cost = claim_value * required_cost_ratio

    merit = rng.beta(4.1, 3.7, size=n)
    complexity = rng.beta(2.5, 2.4, size=n)
    information_asymmetry = rng.beta(2.2, 3.2, size=n)
    risk_aversion = rng.beta(2.1, 4.8, size=n)

    liquidity_multiple = rng.lognormal(mean=np.log(0.82), sigma=0.92, size=n)
    available_liquidity = required_cost * liquidity_multiple

    daab_pct = np.clip(rng.beta(3.2, 4.6, size=n), 0.03, 0.92)
    settlement_offer_pct = (
        rng.beta(3.5, 4.2, size=n)
        + 0.18 * (merit - 0.5)
        - 0.05 * information_asymmetry
    )
    settlement_offer_pct = np.clip(settlement_offer_pct, 0.03, 0.95)

    counterclaim_pct = np.clip(rng.beta(1.4, 8.5, size=n) * 0.85, 0.0, 0.65)
    counterclaim_success_prob = np.clip(
        rng.beta(2.0, 8.0, size=n) + 0.12 * (1.0 - merit), 0.01, 0.55
    )
    adverse_cost = required_cost * np.clip(rng.lognormal(np.log(0.85), 0.38, n), 0.25, 2.0)
    reinvest_roi = np.clip(rng.lognormal(np.log(1.10), 0.16, n) - 1.0, 0.0, 0.55)

    return pd.DataFrame(
        {
            "case_id": np.arange(n),
            "claim_value_musd": claim_value,
            "required_cost_ratio": required_cost_ratio,
            "required_cost_musd": required_cost,
            "available_liquidity_musd": available_liquidity,
            "financially_constrained": available_liquidity < required_cost,
            "merit": merit,
            "complexity": complexity,
            "information_asymmetry": information_asymmetry,
            "risk_aversion": risk_aversion,
            "daab_pct": daab_pct,
            "settlement_offer_pct": settlement_offer_pct,
            "counterclaim_pct": counterclaim_pct,
            "counterclaim_success_prob": counterclaim_success_prob,
            "adverse_cost_musd": adverse_cost,
            "reinvestment_roi": reinvest_roi,
            "settlement_draw": rng.random(n),
        }
    )


def success_probability(
    merit: pd.Series,
    spend_ratio: pd.Series | np.ndarray | float,
    monitoring: float,
    inflation: pd.Series | np.ndarray | float,
    complexity: pd.Series,
    information_asymmetry: pd.Series,
) -> np.ndarray:
    advocacy_gain = 0.060 * (1.0 - np.exp(-1.35 * np.asarray(spend_ratio)))
    underfunding_penalty = 0.090 * np.maximum(0.0, 1.0 - np.asarray(spend_ratio))
    monitoring_gain = 0.045 * monitoring - 0.018 * (monitoring**2)
    credibility_penalty = 0.120 * np.asarray(inflation)
    complexity_penalty = 0.035 * np.asarray(complexity)
    asymmetry_penalty = 0.025 * np.asarray(information_asymmetry)
    p = (
        np.asarray(merit)
        + advocacy_gain
        - underfunding_penalty
        + monitoring_gain
        - credibility_penalty
        - complexity_penalty
        - asymmetry_penalty
        + SUCCESS_CALIBRATION_SHIFT
    )
    return np.clip(p, 0.03, 0.94)


def load_scenarios(path: Path) -> pd.DataFrame:
    scenarios = pd.read_csv(path)
    numeric_cols = [
        "tpf_available",
        "funding_share",
        "funder_award_share",
        "monitoring_intensity",
        "settlement_control",
        "adverse_cost_coverage",
        "claimant_copay_share",
        "staged_funding_efficiency",
        "security_for_costs_probability",
        "min_roi_multiple",
    ]
    scenarios[numeric_cols] = scenarios[numeric_cols].astype(float)
    return scenarios


def evaluate_scenario(cases: pd.DataFrame, terms: pd.Series | dict) -> pd.DataFrame:
    """Evaluate claimant, funder, and respondent outcomes for one scenario."""

    if isinstance(terms, pd.Series):
        terms = terms.to_dict()

    name = str(terms["scenario"])
    tpf_available = float(terms["tpf_available"]) > 0.5
    funding_share = float(terms["funding_share"])
    award_share = float(terms["funder_award_share"])
    monitoring = float(terms["monitoring_intensity"])
    settlement_control = float(terms["settlement_control"])
    adverse_coverage = float(terms["adverse_cost_coverage"])
    claimant_copay_share = float(terms["claimant_copay_share"])
    staged_efficiency = float(terms["staged_funding_efficiency"])
    security_prob = float(terms["security_for_costs_probability"])
    min_roi_multiple = float(terms["min_roi_multiple"])

    v = cases["claim_value_musd"].to_numpy()
    c = cases["required_cost_musd"].to_numpy()
    adverse = cases["adverse_cost_musd"].to_numpy()
    pcc = cases["counterclaim_success_prob"].to_numpy()
    counterclaim = cases["counterclaim_pct"].to_numpy() * v
    daab_value = cases["daab_pct"].to_numpy() * v
    settlement_offer = cases["settlement_offer_pct"].to_numpy() * v

    # Counterclaim defense costs and adverse counterclaim costs (English Rule)
    c_cc = 0.15 * c
    adverse_cc = 0.15 * adverse
    c_cc_resp = 0.15 * c
    adverse_cc_claimant = 0.15 * adverse

    p_self = success_probability(
        cases["merit"],
        1.0,
        0.0,
        0.0,
        cases["complexity"],
        cases["information_asymmetry"],
    )
    # English cost-shifting rule applied to self-funded claimant
    self_net = p_self * v - (1.0 - p_self) * (c + adverse) - pcc * (counterclaim + c_cc + adverse_cc)
    can_self_fund = cases["available_liquidity_musd"].to_numpy() >= c
    self_file = can_self_fund & (self_net > daab_value) & (self_net > 0.0)

    if not tpf_available:
        # Public arbitration benchmarks suggest settlement/discontinuance is
        # meaningful but not dominant. The reservation term includes escalation
        # commitment and case-preparation friction, with a smaller concession
        # for risk aversion.
        reservation = self_net + SELF_FUNDED_ESCALATION_FRICTION * v - 0.05 * cases["risk_aversion"].to_numpy() * v
        settlement_prob = logistic((settlement_offer - reservation) / (0.09 * v + c + EPS))
        settlement = self_file & (cases["settlement_draw"].to_numpy() < settlement_prob)
        filed = self_file
        claimant_net = np.where(settlement, settlement_offer, np.where(filed, self_net, daab_value))
        # English cost-shifting respondent outflow in self-funded baseline
        respondent_outflow = np.where(
            settlement,
            settlement_offer + 0.22 * c,
            np.where(
                filed,
                p_self * (v + c) + c - (1.0 - p_self) * adverse - pcc * counterclaim + (1.0 - pcc) * (c_cc_resp + adverse_cc_claimant),
                daab_value
            ),
        )
        out = pd.DataFrame(
            {
                "case_id": cases["case_id"].to_numpy(),
                "scenario": name,
                "filed": filed,
                "used_tpf": np.zeros(len(cases), dtype=bool),
                "settled": settlement,
                "arbitrated": filed & ~settlement,
                "success_probability": np.where(filed, p_self, 0.0),
                "claimant_net_musd": claimant_net,
                "funder_profit_musd": np.zeros(len(cases)),
                "funder_roi_multiple": np.zeros(len(cases)),
                "respondent_outflow_musd": respondent_outflow,
                "moral_hazard_index": np.zeros(len(cases)),
                "fundable": np.zeros(len(cases), dtype=bool),
                "financially_constrained": cases["financially_constrained"].to_numpy(),
                "claim_value_musd": v,
            }
        )
        return out

    claimant_copay = claimant_copay_share * c
    funder_deployed_cost = funding_share * c * (1.0 - staged_efficiency)
    
    claimant_copay_cc = claimant_copay_share * c_cc
    funder_deployed_cc = funding_share * c_cc * (1.0 - staged_efficiency)

    total_spend_ratio = np.clip((funding_share * c + claimant_copay) / (c + EPS), 0.0, 1.35)

    inflation = (
        0.18
        * (1.0 - monitoring)
        * (1.0 - claimant_copay_share)
        * (0.70 + 0.60 * settlement_control)
    )
    effective_award = v * (1.0 + 0.25 * inflation)
    p_tpf = success_probability(
        cases["merit"],
        total_spend_ratio,
        monitoring,
        inflation,
        cases["complexity"],
        cases["information_asymmetry"],
    )

    monitoring_cost = c * (0.025 + 0.045 * monitoring)
    expected_security_cost = security_prob * adverse * (0.10 + 0.25 * (1.0 - staged_efficiency))
    expected_adverse_cost_to_funder = adverse_coverage * (1.0 - p_tpf) * adverse
    expected_adverse_cc_cost_to_funder = adverse_coverage * pcc * adverse_cc
    
    # Funder expected cost under English Rule (only loses deployed costs when claimant loses primary claim)
    funder_expected_cost = (
        (1.0 - p_tpf) * (funder_deployed_cost + funder_deployed_cc + expected_adverse_cost_to_funder)
        + expected_adverse_cc_cost_to_funder
        + monitoring_cost
        + expected_security_cost
    )
    funder_expected_gross = award_share * p_tpf * effective_award
    funder_profit_arbitration = funder_expected_gross - funder_expected_cost
    funder_roi_multiple = funder_expected_gross / np.maximum(funder_expected_cost, EPS)
    fundable = (funder_roi_multiple >= min_roi_multiple) & (funder_profit_arbitration > 0.0)

    # Opportunity cost of capital: saved capital is stochastically reinvested only up to liquidity limits
    available_liquidity = cases["available_liquidity_musd"].to_numpy()
    saved_cash = np.maximum(0.0, np.minimum(available_liquidity - claimant_copay - claimant_copay_cc, funding_share * c + funding_share * c_cc))
    saved_cash_reinvestment = saved_cash * cases["reinvestment_roi"].to_numpy()

    # Claimant utility under English Rule (costs recovered when primary claim succeeds)
    claimant_tpf_arbitration = (
        (1.0 - award_share) * p_tpf * effective_award
        - (1.0 - p_tpf) * (claimant_copay + (1.0 - adverse_coverage) * adverse)
        - pcc * (counterclaim + claimant_copay_cc + (1.0 - adverse_coverage) * adverse_cc)
        + saved_cash_reinvestment
    )
    choose_tpf = fundable & (claimant_tpf_arbitration > daab_value) & (
        (~can_self_fund) | (claimant_tpf_arbitration > self_net)
    )
    choose_self = (~choose_tpf) & self_file
    filed = choose_tpf | choose_self

    tpf_settlement_friction = (0.08 + 0.08 * settlement_control + 0.04 * (1.0 - adverse_coverage)) * v
    claimant_reservation_tpf = (
        claimant_tpf_arbitration
        + tpf_settlement_friction
        - 0.04 * cases["risk_aversion"].to_numpy() * v
    )
    claimant_settlement_value = (1.0 - award_share) * settlement_offer - claimant_copay - claimant_copay_cc + saved_cash_reinvestment
    funder_settlement_cost = (0.35 + 0.30 * (1.0 - staged_efficiency)) * (funder_deployed_cost + funder_deployed_cc) + monitoring_cost
    funder_settlement_value = award_share * settlement_offer - funder_settlement_cost

    claimant_accept_prob_tpf = logistic(
        (claimant_settlement_value - claimant_reservation_tpf) / (0.085 * v + c + EPS)
    )
    funder_accept_prob = logistic(
        (funder_settlement_value - funder_profit_arbitration) / (0.080 * v + c + EPS)
    )
    consent_component = (1.0 - settlement_control) + settlement_control * funder_accept_prob
    settlement_prob_tpf = claimant_accept_prob_tpf * consent_component

    reservation_self = self_net + SELF_FUNDED_ESCALATION_FRICTION * v - 0.05 * cases["risk_aversion"].to_numpy() * v
    settlement_prob_self = logistic((settlement_offer - reservation_self) / (0.09 * v + c + EPS))

    settlement_prob = np.where(choose_tpf, settlement_prob_tpf, settlement_prob_self)
    settlement = filed & (cases["settlement_draw"].to_numpy() < settlement_prob)

    claimant_net_if_tpf = np.where(settlement, claimant_settlement_value, claimant_tpf_arbitration)
    claimant_net_if_self = np.where(settlement, settlement_offer, self_net)
    claimant_net = np.where(choose_tpf, claimant_net_if_tpf, np.where(choose_self, claimant_net_if_self, daab_value))

    funder_profit = np.where(
        choose_tpf,
        np.where(settlement, funder_settlement_value, funder_profit_arbitration),
        0.0,
    )
    
    # English cost-shifting respondent outflow under TPF (security_prob * adverse is a guarantee/friction, not respondent outflow)
    respondent_outflow_tpf = np.where(
        settlement,
        settlement_offer + 0.24 * c,
        p_tpf * (effective_award + c) + c - (1.0 - p_tpf) * adverse - pcc * (counterclaim + adverse_cc) + (1.0 - pcc) * (c_cc_resp + adverse_cc_claimant),
    )
    respondent_outflow_self = np.where(
        settlement, 
        settlement_offer + 0.22 * c, 
        p_self * (v + c) + c - (1.0 - p_self) * adverse - pcc * counterclaim + (1.0 - pcc) * (c_cc_resp + adverse_cc_claimant)
    )
    respondent_outflow = np.where(
        choose_tpf,
        respondent_outflow_tpf,
        np.where(choose_self, respondent_outflow_self, daab_value),
    )

    moral_hazard = np.where(
        choose_tpf,
        np.clip((1.0 - monitoring) * (1.0 - claimant_copay_share) * (0.55 + 0.45 * settlement_control), 0.0, 1.0),
        0.0,
    )

    return pd.DataFrame(
        {
            "case_id": cases["case_id"].to_numpy(),
            "scenario": name,
            "filed": filed,
            "used_tpf": choose_tpf,
            "settled": settlement,
            "arbitrated": filed & ~settlement,
            "success_probability": np.where(choose_tpf, p_tpf, np.where(choose_self, p_self, 0.0)),
            "claimant_net_musd": claimant_net,
            "funder_profit_musd": funder_profit,
            "funder_roi_multiple": np.where(choose_tpf, funder_roi_multiple, 0.0),
            "respondent_outflow_musd": respondent_outflow,
            "moral_hazard_index": moral_hazard,
            "fundable": fundable,
            "financially_constrained": cases["financially_constrained"].to_numpy(),
            "claim_value_musd": v,
        }
    )


def summarize_results(result: pd.DataFrame) -> pd.Series:
    n = len(result)
    filed = result["filed"].to_numpy(dtype=bool)
    used_tpf = result["used_tpf"].to_numpy(dtype=bool)
    constrained = result["financially_constrained"].to_numpy(dtype=bool)
    settled = result["settled"].to_numpy(dtype=bool)

    def safe_mean(mask: np.ndarray, column: str) -> float:
        if mask.sum() == 0:
            return 0.0
        return float(result.loc[mask, column].mean())

    return pd.Series(
        {
            "n_cases": n,
            "filed_rate": float(filed.mean()),
            "tpf_uptake_rate": float(used_tpf.mean()),
            "fundable_rate": float(result["fundable"].mean()),
            "settlement_rate_filed": float(settled[filed].mean()) if filed.any() else 0.0,
            "arbitration_rate_filed": float(result["arbitrated"].to_numpy(dtype=bool)[filed].mean()) if filed.any() else 0.0,
            "constrained_access_rate": float((filed & constrained).sum() / max(constrained.sum(), 1)),
            "mean_success_probability_filed": safe_mean(filed, "success_probability"),
            "mean_claimant_net_musd": float(result["claimant_net_musd"].mean()),
            "mean_funder_profit_musd": safe_mean(used_tpf, "funder_profit_musd"),
            "mean_funder_roi_multiple": safe_mean(used_tpf, "funder_roi_multiple"),
            "mean_respondent_outflow_musd": float(result["respondent_outflow_musd"].mean()),
            "mean_moral_hazard_index": safe_mean(used_tpf, "moral_hazard_index"),
            "mean_claim_value_filed_musd": safe_mean(filed, "claim_value_musd"),
        }
    )


def summarize_all(results: Iterable[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for res in results:
        s = summarize_results(res)
        s["scenario"] = res["scenario"].iloc[0]
        rows.append(s)
    summary = pd.DataFrame(rows)
    cols = ["scenario"] + [c for c in summary.columns if c != "scenario"]
    return summary[cols]


def bootstrap_summary(
    result: pd.DataFrame,
    metrics: list[str],
    n_boot: int = 300,
    seed: int = 9814,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(result)
    
    # Pre-extract all columns as numpy 1D arrays to avoid slow Pandas indexing inside the loop
    filed = result["filed"].to_numpy(dtype=bool)
    used_tpf = result["used_tpf"].to_numpy(dtype=bool)
    constrained = result["financially_constrained"].to_numpy(dtype=bool)
    settled = result["settled"].to_numpy(dtype=bool)
    claimant_net = result["claimant_net_musd"].to_numpy(dtype=float)
    
    # We will accumulate values for each metric
    boot_metrics = {m: [] for m in metrics}
    
    for _ in range(n_boot):
        boot_idx = rng.choice(n, size=n, replace=True)
        
        # Sample numpy arrays directly
        b_filed = filed[boot_idx]
        b_constrained = constrained[boot_idx]
        b_settled = settled[boot_idx]
        
        # Calculate the metrics directly in numpy (very fast!)
        results_summary = {}
        if "filed_rate" in metrics:
            results_summary["filed_rate"] = float(b_filed.mean())
        if "settlement_rate_filed" in metrics:
            results_summary["settlement_rate_filed"] = float(b_settled[b_filed].mean()) if b_filed.any() else 0.0
        if "constrained_access_rate" in metrics:
            results_summary["constrained_access_rate"] = float((b_filed & b_constrained).sum() / max(b_constrained.sum(), 1))
        if "mean_claimant_net_musd" in metrics:
            results_summary["mean_claimant_net_musd"] = float(claimant_net[boot_idx].mean())
            
        for m in metrics:
            boot_metrics[m].append(results_summary[m])
            
    rows = []
    for metric in metrics:
        arr = np.asarray(boot_metrics[metric])
        rows.append(
            {
                "scenario": result["scenario"].iloc[0],
                "metric": metric,
                "mean": float(arr.mean()),
                "ci_2_5": float(np.quantile(arr, 0.025)),
                "ci_97_5": float(np.quantile(arr, 0.975)),
                "mc_se": float(arr.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def score_components(summary_row: pd.Series, baseline_row: pd.Series, settlement_target: float = 0.33) -> dict[str, float]:
    access = np.clip(summary_row["constrained_access_rate"], 0.0, 1.0)
    claimant_gain = (
        summary_row["mean_claimant_net_musd"] - baseline_row["mean_claimant_net_musd"]
    ) / max(abs(baseline_row["mean_claimant_net_musd"]), EPS)
    claimant_gain = np.clip(claimant_gain, -0.5, 1.5)
    settlement_alignment = 1.0 - min(
        abs(summary_row["settlement_rate_filed"] - settlement_target) / settlement_target,
        1.0,
    )
    roi = np.clip(summary_row["mean_funder_roi_multiple"] / 1.35, 0.0, 1.5)
    moral_quality = 1.0 - np.clip(summary_row["mean_moral_hazard_index"], 0.0, 1.0)
    respondent_excess = (
        summary_row["mean_respondent_outflow_musd"] - baseline_row["mean_respondent_outflow_musd"]
    ) / max(abs(baseline_row["mean_respondent_outflow_musd"]), EPS)
    respondent_penalty = np.clip(respondent_excess, 0.0, 1.0)
    return {
        "access": float(access),
        "claimant_gain": float(claimant_gain),
        "settlement_alignment": float(settlement_alignment),
        "roi": float(roi),
        "moral_quality": float(moral_quality),
        "respondent_penalty": float(respondent_penalty),
    }


def weighted_score(
    components: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    if weights is None:
        weights = DEFAULT_SCORE_WEIGHTS
    return float(sum(float(weights[name]) * float(components[name]) for name in DEFAULT_SCORE_WEIGHTS))


def balanced_score(
    summary_row: pd.Series,
    baseline_row: pd.Series,
    settlement_target: float = 0.33,
    weights: Mapping[str, float] | None = None,
) -> float:
    return weighted_score(score_components(summary_row, baseline_row, settlement_target), weights)
