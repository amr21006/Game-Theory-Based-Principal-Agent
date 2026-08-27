"""Parameterized re-specifications of the TPF agreement model for sensitivity analysis.

This module does not change the primary model in ``src/tpf_model.py``. It exposes a
parameterized evaluator whose defaults reproduce ``tpf_model.evaluate_scenario``
bit-for-bit, so that alternative moral-hazard, claim-inflation, and settlement
specifications can be tested against an unchanged reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .tpf_model import (
    EPS,
    SELF_FUNDED_ESCALATION_FRICTION,
    logistic,
    success_probability,
)


@dataclass(frozen=True)
class ModelSpec:
    """Alternative specification switches.

    All defaults reproduce the primary model.
    """

    label: str = "primary"

    # Moral-hazard index: MH = (1 - gamma)**mh_monitoring_exponent
    #                        * (1 - CP)**mh_copay_exponent
    #                        * (mh_sc_intercept + mh_sc_slope * SC)
    mh_monitoring_exponent: float = 1.0
    mh_copay_exponent: float = 1.0
    mh_sc_intercept: float = 0.55
    mh_sc_slope: float = 0.45

    # Share of claim inflation attributable to the agreement terms. The remainder
    # is drawn from an exogenous per-case propensity independent of the terms.
    inflation_endogenous_share: float = 1.0
    # Share of the reported moral-hazard index attributable to the agreement terms.
    mh_endogenous_share: float = 1.0

    # Settlement specification multipliers.
    settlement_friction_multiplier: float = 1.0
    logistic_scale_multiplier: float = 1.0
    self_funded_friction: float = SELF_FUNDED_ESCALATION_FRICTION


def moral_hazard_index(
    monitoring: float,
    copay: float,
    settlement_control: float,
    spec: ModelSpec,
) -> float:
    """Governance indicator built from the agreement terms."""

    return float(
        np.clip(
            (1.0 - monitoring) ** spec.mh_monitoring_exponent
            * (1.0 - copay) ** spec.mh_copay_exponent
            * (spec.mh_sc_intercept + spec.mh_sc_slope * settlement_control),
            0.0,
            1.0,
        )
    )


def evaluate_scenario_spec(
    cases: pd.DataFrame,
    terms: pd.Series | dict,
    spec: ModelSpec | None = None,
) -> pd.DataFrame:
    """Evaluate one scenario under an alternative specification.

    With ``spec=None`` the result is identical to ``tpf_model.evaluate_scenario``.
    """

    if spec is None:
        spec = ModelSpec()
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
    if "exogenous_inflation_propensity" in cases.columns:
        exogenous_propensity = cases["exogenous_inflation_propensity"].to_numpy()
    else:
        exogenous_propensity = np.zeros(len(cases))

    c_cc = 0.15 * c
    adverse_cc = 0.15 * adverse
    c_cc_resp = 0.15 * c
    adverse_cc_claimant = 0.15 * adverse

    p_self = success_probability(
        cases["merit"], 1.0, 0.0, 0.0, cases["complexity"], cases["information_asymmetry"]
    )
    self_net = (
        p_self * v
        - (1.0 - p_self) * (c + adverse)
        - pcc * (counterclaim + c_cc + adverse_cc)
    )
    can_self_fund = cases["available_liquidity_musd"].to_numpy() >= c
    self_file = can_self_fund & (self_net > daab_value) & (self_net > 0.0)

    risk_aversion = cases["risk_aversion"].to_numpy()
    lsm = spec.logistic_scale_multiplier

    reservation_self = (
        self_net + spec.self_funded_friction * v - 0.05 * risk_aversion * v
    )
    settlement_prob_self = logistic(
        (settlement_offer - reservation_self) / (lsm * (0.09 * v + c) + EPS)
    )

    if not tpf_available:
        settlement = self_file & (cases["settlement_draw"].to_numpy() < settlement_prob_self)
        filed = self_file
        claimant_net = np.where(
            settlement, settlement_offer, np.where(filed, self_net, daab_value)
        )
        respondent_outflow = np.where(
            settlement,
            settlement_offer + 0.22 * c,
            np.where(
                filed,
                p_self * (v + c)
                + c
                - (1.0 - p_self) * adverse
                - pcc * counterclaim
                + (1.0 - pcc) * (c_cc_resp + adverse_cc_claimant),
                daab_value,
            ),
        )
        return pd.DataFrame(
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

    claimant_copay = claimant_copay_share * c
    funder_deployed_cost = funding_share * c * (1.0 - staged_efficiency)
    claimant_copay_cc = claimant_copay_share * c_cc
    funder_deployed_cc = funding_share * c_cc * (1.0 - staged_efficiency)

    total_spend_ratio = np.clip((funding_share * c + claimant_copay) / (c + EPS), 0.0, 1.35)

    agreement_driver = (
        (1.0 - monitoring) * (1.0 - claimant_copay_share) * (0.70 + 0.60 * settlement_control)
    )
    lam = spec.inflation_endogenous_share
    inflation = 0.18 * (lam * agreement_driver + (1.0 - lam) * exogenous_propensity)

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

    available_liquidity = cases["available_liquidity_musd"].to_numpy()
    saved_cash = np.maximum(
        0.0,
        np.minimum(
            available_liquidity - claimant_copay - claimant_copay_cc,
            funding_share * c + funding_share * c_cc,
        ),
    )
    saved_cash_reinvestment = saved_cash * cases["reinvestment_roi"].to_numpy()

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

    tpf_settlement_friction = (
        spec.settlement_friction_multiplier
        * (0.08 + 0.08 * settlement_control + 0.04 * (1.0 - adverse_coverage))
        * v
    )
    claimant_reservation_tpf = (
        claimant_tpf_arbitration + tpf_settlement_friction - 0.04 * risk_aversion * v
    )
    claimant_settlement_value = (
        (1.0 - award_share) * settlement_offer
        - claimant_copay
        - claimant_copay_cc
        + saved_cash_reinvestment
    )
    funder_settlement_cost = (0.35 + 0.30 * (1.0 - staged_efficiency)) * (
        funder_deployed_cost + funder_deployed_cc
    ) + monitoring_cost
    funder_settlement_value = award_share * settlement_offer - funder_settlement_cost

    claimant_accept_prob_tpf = logistic(
        (claimant_settlement_value - claimant_reservation_tpf) / (lsm * (0.085 * v + c) + EPS)
    )
    funder_accept_prob = logistic(
        (funder_settlement_value - funder_profit_arbitration) / (lsm * (0.080 * v + c) + EPS)
    )
    consent_component = (1.0 - settlement_control) + settlement_control * funder_accept_prob
    settlement_prob_tpf = claimant_accept_prob_tpf * consent_component

    settlement_prob = np.where(choose_tpf, settlement_prob_tpf, settlement_prob_self)
    settlement = filed & (cases["settlement_draw"].to_numpy() < settlement_prob)

    claimant_net_if_tpf = np.where(settlement, claimant_settlement_value, claimant_tpf_arbitration)
    claimant_net_if_self = np.where(settlement, settlement_offer, self_net)
    claimant_net = np.where(
        choose_tpf, claimant_net_if_tpf, np.where(choose_self, claimant_net_if_self, daab_value)
    )

    funder_profit = np.where(
        choose_tpf,
        np.where(settlement, funder_settlement_value, funder_profit_arbitration),
        0.0,
    )

    respondent_outflow_tpf = np.where(
        settlement,
        settlement_offer + 0.24 * c,
        p_tpf * (effective_award + c)
        + c
        - (1.0 - p_tpf) * adverse
        - pcc * (counterclaim + adverse_cc)
        + (1.0 - pcc) * (c_cc_resp + adverse_cc_claimant),
    )
    respondent_outflow_self = np.where(
        settlement,
        settlement_offer + 0.22 * c,
        p_self * (v + c)
        + c
        - (1.0 - p_self) * adverse
        - pcc * counterclaim
        + (1.0 - pcc) * (c_cc_resp + adverse_cc_claimant),
    )
    respondent_outflow = np.where(
        choose_tpf,
        respondent_outflow_tpf,
        np.where(choose_self, respondent_outflow_self, daab_value),
    )

    mh_terms = moral_hazard_index(monitoring, claimant_copay_share, settlement_control, spec)
    mh_value = (
        spec.mh_endogenous_share * mh_terms
        + (1.0 - spec.mh_endogenous_share) * exogenous_propensity
    )
    moral_hazard = np.where(choose_tpf, np.clip(mh_value, 0.0, 1.0), 0.0)

    return pd.DataFrame(
        {
            "case_id": cases["case_id"].to_numpy(),
            "scenario": name,
            "filed": filed,
            "used_tpf": choose_tpf,
            "settled": settlement,
            "arbitrated": filed & ~settlement,
            "success_probability": np.where(
                choose_tpf, p_tpf, np.where(choose_self, p_self, 0.0)
            ),
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


ALTERNATIVE_MH_SPECS: list[ModelSpec] = [
    ModelSpec(label="MH-0 primary"),
    ModelSpec(label="MH-1 settlement control down-weighted", mh_sc_intercept=0.85, mh_sc_slope=0.15),
    ModelSpec(label="MH-2 co-payment up-weighted", mh_copay_exponent=2.0),
    ModelSpec(label="MH-3 monitoring with diminishing effect", mh_monitoring_exponent=0.5),
    ModelSpec(
        label="MH-4 inflation and index partly exogenous",
        inflation_endogenous_share=0.5,
        mh_endogenous_share=0.5,
    ),
]


SETTLEMENT_SPECS: list[ModelSpec] = [
    ModelSpec(label="S-0 primary"),
    ModelSpec(label="S-1 friction halved", settlement_friction_multiplier=0.5, self_funded_friction=0.09),
    ModelSpec(label="S-2 friction increased 50 percent", settlement_friction_multiplier=1.5, self_funded_friction=0.27),
    ModelSpec(label="S-3 logistic scale halved", logistic_scale_multiplier=0.5),
    ModelSpec(label="S-4 logistic scale doubled", logistic_scale_multiplier=2.0),
]
