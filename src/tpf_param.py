"""Coefficient-parameterized form of the primary scenario evaluator.

``evaluate`` reproduces ``src.tpf_model.evaluate_scenario`` exactly at the default
``Params()`` (see ``max_difference`` and ``tests/test_param_equivalence.py``), and
exposes the coefficients that the supplementary robustness checks vary:

    shift              additive calibration constant of Eq. (1)
    adv_amp            advocacy-gain amplitude
    underfund_pen      underfunding penalty
    monitoring_mult    multiplier on the monitoring gain B(gamma)
    inflation_pen      claim-inflation credibility penalty
    complexity_pen     complexity penalty
    asymmetry_pen      information-asymmetry penalty
    cc_cost_frac       counterclaim cost fraction of the required arbitration cost
    passthrough        effective-award pass-through, V_eff = V (1 + passthrough * INF)

The optional multi-party extension (``multiparty=True``) divides each case's
counterclaim between an employer claim for delay damages and a subcontractor
cross-claim, with separate success probabilities, defense costs and adverse-cost
exposures, a complexity multiplier, and a settlement factor for the agreement of a
second respondent. With ``multiparty=False`` the single-respondent model is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.tpf_model import EPS, SELF_FUNDED_ESCALATION_FRICTION, evaluate_scenario, logistic


@dataclass(frozen=True)
class Params:
    adv_amp: float = 0.060
    underfund_pen: float = 0.090
    monitoring_mult: float = 1.0
    inflation_pen: float = 0.120
    complexity_pen: float = 0.035
    asymmetry_pen: float = 0.025
    cc_cost_frac: float = 0.15
    passthrough: float = 0.25
    shift: float = -0.16
    self_friction: float = SELF_FUNDED_ESCALATION_FRICTION
    multiparty: bool = False
    mp_complexity_mult: float = 1.20
    mp_settlement_factor: float = 0.94
    mp_cost_emp: float = 0.10
    mp_cost_sub: float = 0.08
    mp_adverse_emp: float = 0.105
    mp_adverse_sub: float = 0.075
    mp_seed: int = 1514 + 999


def success_probability(P: Params, merit, spend_ratio, monitoring, inflation, complexity, asym):
    spend_ratio = np.asarray(spend_ratio)
    advocacy_gain = P.adv_amp * (1.0 - np.exp(-1.35 * spend_ratio))
    underfunding_penalty = P.underfund_pen * np.maximum(0.0, 1.0 - spend_ratio)
    monitoring_gain = P.monitoring_mult * (0.045 * monitoring - 0.018 * (monitoring ** 2))
    credibility_penalty = P.inflation_pen * np.asarray(inflation)
    cmult = P.mp_complexity_mult if P.multiparty else 1.0
    complexity_penalty = P.complexity_pen * cmult * np.asarray(complexity)
    asymmetry_penalty = P.asymmetry_pen * np.asarray(asym)
    p = (np.asarray(merit) + advocacy_gain - underfunding_penalty + monitoring_gain - credibility_penalty
         - complexity_penalty - asymmetry_penalty + P.shift)
    return np.clip(p, 0.03, 0.94)


def _counterclaims(cases: pd.DataFrame, P: Params, c, adverse):
    v = cases["claim_value_musd"].to_numpy()
    pcc = cases["counterclaim_success_prob"].to_numpy()
    counterclaim = cases["counterclaim_pct"].to_numpy() * v
    if not P.multiparty:
        c_cc = P.cc_cost_frac * c
        adverse_cc = 0.15 * adverse
        return [(pcc, counterclaim, c_cc, adverse_cc)], c_cc, P.cc_cost_frac * c, 0.15 * adverse
    rng = np.random.default_rng(P.mp_seed)
    n = len(cases)
    sub_share = rng.beta(2.0, 3.0, size=n)
    p_emp = np.clip(pcc + rng.normal(0, 0.05, n), 0.01, 0.60)
    p_sub = np.clip(pcc + rng.normal(0, 0.05, n), 0.01, 0.60)
    parts = [(p_emp, counterclaim * (1.0 - sub_share), P.mp_cost_emp * c, P.mp_adverse_emp * adverse),
             (p_sub, counterclaim * sub_share, P.mp_cost_sub * c, P.mp_adverse_sub * adverse)]
    c_cc = (P.mp_cost_emp + P.mp_cost_sub) * c
    return parts, c_cc, c_cc, (P.mp_adverse_emp + P.mp_adverse_sub) * adverse


def evaluate(cases: pd.DataFrame, terms, P: Params = Params()) -> pd.DataFrame:
    """Evaluate one scenario; identical to ``evaluate_scenario`` at ``Params()``.

    Adds one column, ``expected_security_cost_musd`` (funded cases only)."""
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
    daab_value = cases["daab_pct"].to_numpy() * v
    settlement_offer = cases["settlement_offer_pct"].to_numpy() * v

    parts, c_cc, c_cc_resp, adverse_cc_claimant = _counterclaims(cases, P, c, adverse)
    settle_factor = P.mp_settlement_factor if P.multiparty else 1.0
    resp_cc_gain = sum(pk * vk for pk, vk, _, _ in parts)
    resp_cc_gain_with_adv = sum(pk * (vk + ak) for pk, vk, _, ak in parts)
    p_any_cc = 1.0 - np.prod([1.0 - pk for pk, _, _, _ in parts], axis=0) if P.multiparty else parts[0][0]

    p_self = success_probability(P, cases["merit"], 1.0, 0.0, 0.0, cases["complexity"], cases["information_asymmetry"])
    self_net = p_self * v - (1.0 - p_self) * (c + adverse) - sum(pk * (vk + ck + ak) for pk, vk, ck, ak in parts)
    can_self_fund = cases["available_liquidity_musd"].to_numpy() >= c
    self_file = can_self_fund & (self_net > daab_value) & (self_net > 0.0)
    risk_aversion = cases["risk_aversion"].to_numpy()
    draw = cases["settlement_draw"].to_numpy()

    if not tpf_available:
        reservation = self_net + P.self_friction * v - 0.05 * risk_aversion * v
        settlement_prob = settle_factor * logistic((settlement_offer - reservation) / (0.09 * v + c + EPS))
        settlement = self_file & (draw < settlement_prob)
        filed = self_file
        claimant_net = np.where(settlement, settlement_offer, np.where(filed, self_net, daab_value))
        respondent_outflow = np.where(
            settlement, settlement_offer + 0.22 * c,
            np.where(filed, p_self * (v + c) + c - (1.0 - p_self) * adverse - resp_cc_gain
                     + (1.0 - p_any_cc) * (c_cc_resp + adverse_cc_claimant), daab_value))
        zeros = np.zeros(len(cases))
        return pd.DataFrame({
            "case_id": cases["case_id"].to_numpy(), "scenario": name, "filed": filed,
            "used_tpf": np.zeros(len(cases), dtype=bool), "settled": settlement, "arbitrated": filed & ~settlement,
            "success_probability": np.where(filed, p_self, 0.0), "claimant_net_musd": claimant_net,
            "funder_profit_musd": zeros, "funder_roi_multiple": zeros, "respondent_outflow_musd": respondent_outflow,
            "moral_hazard_index": zeros, "fundable": np.zeros(len(cases), dtype=bool),
            "financially_constrained": cases["financially_constrained"].to_numpy(), "claim_value_musd": v,
            "expected_security_cost_musd": zeros})

    claimant_copay = claimant_copay_share * c
    funder_deployed_cost = funding_share * c * (1.0 - staged_efficiency)
    claimant_copay_cc = claimant_copay_share * c_cc
    funder_deployed_cc = funding_share * c_cc * (1.0 - staged_efficiency)
    total_spend_ratio = np.clip((funding_share * c + claimant_copay) / (c + EPS), 0.0, 1.35)
    inflation = 0.18 * (1.0 - monitoring) * (1.0 - claimant_copay_share) * (0.70 + 0.60 * settlement_control)
    effective_award = v * (1.0 + P.passthrough * inflation)
    p_tpf = success_probability(P, cases["merit"], total_spend_ratio, monitoring, inflation,
                                cases["complexity"], cases["information_asymmetry"])

    monitoring_cost = c * (0.025 + 0.045 * monitoring)
    expected_security_cost = security_prob * adverse * (0.10 + 0.25 * (1.0 - staged_efficiency))
    expected_adverse_cost_to_funder = adverse_coverage * (1.0 - p_tpf) * adverse
    expected_adverse_cc_cost_to_funder = adverse_coverage * sum(pk * ak for pk, _, _, ak in parts)
    funder_expected_cost = ((1.0 - p_tpf) * (funder_deployed_cost + funder_deployed_cc + expected_adverse_cost_to_funder)
                            + expected_adverse_cc_cost_to_funder + monitoring_cost + expected_security_cost)
    funder_expected_gross = award_share * p_tpf * effective_award
    funder_profit_arbitration = funder_expected_gross - funder_expected_cost
    funder_roi_multiple = funder_expected_gross / np.maximum(funder_expected_cost, EPS)
    fundable = (funder_roi_multiple >= min_roi_multiple) & (funder_profit_arbitration > 0.0)

    available_liquidity = cases["available_liquidity_musd"].to_numpy()
    saved_cash = np.maximum(0.0, np.minimum(available_liquidity - claimant_copay - claimant_copay_cc,
                                            funding_share * c + funding_share * c_cc))
    saved_cash_reinvestment = saved_cash * cases["reinvestment_roi"].to_numpy()
    claimant_cc_loss = sum(pk * (vk + claimant_copay_share * ck + (1.0 - adverse_coverage) * ak)
                           for pk, vk, ck, ak in parts)
    claimant_tpf_arbitration = ((1.0 - award_share) * p_tpf * effective_award
                                - (1.0 - p_tpf) * (claimant_copay + (1.0 - adverse_coverage) * adverse)
                                - claimant_cc_loss + saved_cash_reinvestment)
    choose_tpf = fundable & (claimant_tpf_arbitration > daab_value) & (
        (~can_self_fund) | (claimant_tpf_arbitration > self_net))
    choose_self = (~choose_tpf) & self_file
    filed = choose_tpf | choose_self

    tpf_settlement_friction = (0.08 + 0.08 * settlement_control + 0.04 * (1.0 - adverse_coverage)) * v
    claimant_reservation_tpf = claimant_tpf_arbitration + tpf_settlement_friction - 0.04 * risk_aversion * v
    claimant_settlement_value = ((1.0 - award_share) * settlement_offer - claimant_copay - claimant_copay_cc
                                 + saved_cash_reinvestment)
    funder_settlement_cost = ((0.35 + 0.30 * (1.0 - staged_efficiency)) * (funder_deployed_cost + funder_deployed_cc)
                              + monitoring_cost)
    funder_settlement_value = award_share * settlement_offer - funder_settlement_cost
    claimant_accept_prob_tpf = logistic((claimant_settlement_value - claimant_reservation_tpf) / (0.085 * v + c + EPS))
    funder_accept_prob = logistic((funder_settlement_value - funder_profit_arbitration) / (0.080 * v + c + EPS))
    consent_component = (1.0 - settlement_control) + settlement_control * funder_accept_prob
    settlement_prob_tpf = settle_factor * claimant_accept_prob_tpf * consent_component
    reservation_self = self_net + P.self_friction * v - 0.05 * risk_aversion * v
    settlement_prob_self = settle_factor * logistic((settlement_offer - reservation_self) / (0.09 * v + c + EPS))
    settlement_prob = np.where(choose_tpf, settlement_prob_tpf, settlement_prob_self)
    settlement = filed & (draw < settlement_prob)

    claimant_net = np.where(choose_tpf, np.where(settlement, claimant_settlement_value, claimant_tpf_arbitration),
                            np.where(choose_self, np.where(settlement, settlement_offer, self_net), daab_value))
    funder_profit = np.where(choose_tpf, np.where(settlement, funder_settlement_value, funder_profit_arbitration), 0.0)
    respondent_outflow_tpf = np.where(
        settlement, settlement_offer + 0.24 * c,
        p_tpf * (effective_award + c) + c - (1.0 - p_tpf) * adverse - resp_cc_gain_with_adv
        + (1.0 - p_any_cc) * (c_cc_resp + adverse_cc_claimant))
    respondent_outflow_self = np.where(
        settlement, settlement_offer + 0.22 * c,
        p_self * (v + c) + c - (1.0 - p_self) * adverse - resp_cc_gain + (1.0 - p_any_cc) * (c_cc_resp + adverse_cc_claimant))
    respondent_outflow = np.where(choose_tpf, respondent_outflow_tpf,
                                  np.where(choose_self, respondent_outflow_self, daab_value))
    moral_hazard = np.where(choose_tpf, np.clip((1.0 - monitoring) * (1.0 - claimant_copay_share)
                                                * (0.55 + 0.45 * settlement_control), 0.0, 1.0), 0.0)
    return pd.DataFrame({
        "case_id": cases["case_id"].to_numpy(), "scenario": name, "filed": filed, "used_tpf": choose_tpf,
        "settled": settlement, "arbitrated": filed & ~settlement,
        "success_probability": np.where(choose_tpf, p_tpf, np.where(choose_self, p_self, 0.0)),
        "claimant_net_musd": claimant_net, "funder_profit_musd": funder_profit,
        "funder_roi_multiple": np.where(choose_tpf, funder_roi_multiple, 0.0),
        "respondent_outflow_musd": respondent_outflow, "moral_hazard_index": moral_hazard, "fundable": fundable,
        "financially_constrained": cases["financially_constrained"].to_numpy(), "claim_value_musd": v,
        "expected_security_cost_musd": np.where(choose_tpf, expected_security_cost, 0.0)})


def max_difference(cases: pd.DataFrame, scenarios) -> float:
    """Largest absolute difference, over all outputs and scenarios, between ``evaluate``
    at ``Params()`` and the primary ``evaluate_scenario``; decision flags count as 1 per mismatch."""
    worst = 0.0
    for terms in scenarios:
        a = evaluate_scenario(cases, terms)
        b = evaluate(cases, terms, Params())
        for col in a.columns:
            if col == "scenario":
                continue
            x, y = a[col].to_numpy(), b[col].to_numpy()
            if x.dtype == bool:
                worst = max(worst, float((x != y).sum()))
            else:
                worst = max(worst, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
    return worst
