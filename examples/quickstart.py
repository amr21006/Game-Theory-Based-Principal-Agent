"""Quickstart example for the TPF Arbitration Simulation Framework."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.tpf_model import (
    SimulationConfig,
    generate_cases,
    evaluate_scenario,
    summarize_results,
    score_components,
    balanced_score,
)

def main():
    print("======================================================================")
    print("TPF ARBITRATION SIMULATION FRAMEWORK: QUICKSTART DEMO")
    print("======================================================================\n")

    print("1. Generating synthetic construction dispute cases...")
    config = SimulationConfig(n_cases=10_000, seed=42)
    cases = generate_cases(config)
    print(f"   Successfully generated {len(cases):,} dispute records with Common Random Numbers (CRN).")

    print("\n2. Evaluating Baseline Scenario (Self-Funded Only, No TPF)...")
    baseline_terms = {
        "scenario": "Baseline (No TPF)",
        "tpf_available": 0.0,
        "funding_share": 0.0,
        "funder_award_share": 0.0,
        "monitoring_intensity": 0.0,
        "settlement_control": 0.0,
        "claimant_copay_share": 0.0,
        "adverse_cost_coverage": 0.0,
        "staged_funding_efficiency": 1.0,
        "security_for_costs_probability": 0.0,
        "min_roi_multiple": 1.18,
    }
    baseline_outcomes = evaluate_scenario(cases, baseline_terms)
    baseline_summary = summarize_results(baseline_outcomes)
    print(f"   - Baseline Filing Rate: {baseline_summary['filed_rate']:.2%}")
    print(f"   - Baseline Settlement Rate: {baseline_summary['settlement_rate_filed']:.2%}")
    print(f"   - Baseline Constrained Access: {baseline_summary['constrained_access_rate']:.2%}")
    print(f"   - Baseline Mean Claimant Net: USD {baseline_summary['mean_claimant_net_musd']:.2f}M")

    print("\n3. Evaluating Custom Third-Party Funding Agreement...")
    custom_terms = {
        "scenario": "Configured TPF",
        "tpf_available": 1.0,
        "funding_share": 0.85,
        "funder_award_share": 0.22,
        "monitoring_intensity": 0.85,
        "settlement_control": 0.05,
        "claimant_copay_share": 0.10,
        "adverse_cost_coverage": 0.70,
        "staged_funding_efficiency": 0.90,
        "security_for_costs_probability": 0.35,
        "min_roi_multiple": 1.18,
    }
    custom_outcomes = evaluate_scenario(cases, custom_terms)
    custom_summary = summarize_results(custom_outcomes)

    print(f"   - TPF Filing Rate: {custom_summary['filed_rate']:.2%}")
    print(f"   - TPF Settlement Rate: {custom_summary['settlement_rate_filed']:.2%}")
    print(f"   - TPF Constrained Access: {custom_summary['constrained_access_rate']:.2%}")
    print(f"   - TPF Funder ROI Multiple: {custom_summary['mean_funder_roi_multiple']:.2f}x")
    print(f"   - TPF Mean Claimant Net: USD {custom_summary['mean_claimant_net_musd']:.2f}M")
    print(f"   - Governance Index (Moral Hazard): {custom_summary['mean_moral_hazard_index']:.3f}")

    print("\n4. Computing Multi-Criteria Balanced Objective Score (J)...")
    components = score_components(custom_summary, baseline_summary)
    score_j = balanced_score(custom_summary, baseline_summary)
    print(f"   - Access Gain Component (w=0.30): {components['access']:.3f}")
    print(f"   - Claimant Net Gain Component (w=0.20): {components['claimant_gain']:.3f}")
    print(f"   - Settlement Alignment Component (w=0.20): {components['settlement_alignment']:.3f}")
    print(f"   - Funder ROI Component (w=0.15): {components['roi']:.3f}")
    print(f"   - Governance Quality Component (w=0.10): {components['moral_quality']:.3f}")
    print(f"   - Respondent Outflow Penalty (w=-0.05): {components['respondent_penalty']:.3f}")
    print(f"   ==> Overall Balanced Score (J): {score_j:.4f}")

    print("\n======================================================================")
    print("DEMO COMPLETED SUCCESSFULLY.")
    print("======================================================================")

if __name__ == "__main__":
    main()
