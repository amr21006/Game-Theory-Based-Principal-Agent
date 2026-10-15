"""Run the robustness modules on the primary population (150,000 cases, seed 1514).

    Area 1  behavioral-coefficient perturbations            supplement Table S32
    Area 2  construction-sector recalibration (35.1%)       supplement Table S33
    Area 3  security-for-costs and adverse-cost coverage    supplement Table S34
    Area 4  employer and subcontractor dispute architecture supplement Table S35
    Area 5  checks on the agreement-term search             reported in the manuscript

Result tables and a summary generated from them are written to
``tests/robustness_extensions/outputs/`` (not tracked).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.robustness_extensions.grid_checks import run_grid_checks  # noqa: E402
from tests.robustness_extensions.test_area1_coefficient_bounds import run_area1_coefficient_stress  # noqa: E402
from tests.robustness_extensions.test_area2_unctad_construction_calibration import (  # noqa: E402
    run_area2_seed_replication, run_area2_unctad_calibration,
)
from tests.robustness_extensions.test_area3_legal_security_risk import run_area3_security_risk_stress  # noqa: E402
from tests.robustness_extensions.test_area4_multiparty_structure import run_area4_multiparty_analysis  # noqa: E402


def main(include_grid_checks: bool = True) -> None:
    out = ROOT / "tests" / "robustness_extensions" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lines = ["# Robustness modules: results", "", "Primary population: 150,000 cases, seed 1514.", ""]

    a1 = run_area1_coefficient_stress()
    a1.to_csv(out / "area1_coefficient_stress_results.csv", index=False)
    piv = a1.pivot(index="perturbation", columns="scenario", values="score_j")
    grid = a1[(a1.scenario == "Grid-selected") & (a1.perturbation != "Base parameterization")]
    lines += ["## Area 1: coefficient perturbations (Table S32)",
              f"- Grid-selected attains the highest score under {int((piv.idxmax(axis=1) == 'Grid-selected').sum())} "
              f"of {len(piv)} parameterizations.",
              f"- Grid-selected return multiple {grid.mean_funder_roi_multiple.min():.2f} to "
              f"{grid.mean_funder_roi_multiple.max():.2f}; settlement among filed cases "
              f"{100 * grid.settlement_rate_filed.min():.1f}% to {100 * grid.settlement_rate_filed.max():.1f}%.", ""]

    a2, shift = run_area2_unctad_calibration()
    a2.to_csv(out / "area2_unctad_calibration_results.csv", index=False)
    s2 = a2.set_index("scenario")
    lines += ["## Area 2: construction-sector recalibration (Table S33)",
              f"- Calibration constant {shift:.4f}; baseline filed rate {100 * s2.loc['No funding', 'filed_rate']:.2f}%, "
              f"baseline settlement among filed cases {100 * s2.loc['No funding', 'settlement_rate_filed']:.1f}%.",
              f"- Balanced score: grid-selected {s2.loc['Grid-selected', 'score_j']:.3f}, conventional "
              f"{s2.loc['Conventional', 'score_j']:.3f}; highest: {s2['score_j'].idxmax()}.",
              f"- With the benchmark at the recalibrated baseline settlement rate: grid-selected "
              f"{s2.loc['Grid-selected', 'score_j_baseline_benchmark']:.3f}, conventional "
              f"{s2.loc['Conventional', 'score_j_baseline_benchmark']:.3f}."]
    seeds = run_area2_seed_replication()
    seeds.to_csv(out / "area2_seed_replication.csv", index=False)
    lines += [f"- Across seeds {', '.join(str(s) for s in seeds.seed)}: grid-selected minus conventional "
              f"{seeds.grid_minus_conventional.min():.3f} to {seeds.grid_minus_conventional.max():.3f}; grid-selected "
              f"highest under {int((seeds.highest_score == 'Grid-selected').sum())} of {len(seeds)} seeds "
              f"({int((seeds.highest_score_baseline_benchmark == 'Grid-selected').sum())} of {len(seeds)} with the "
              "baseline benchmark).", ""]

    a3 = run_area3_security_risk_stress()
    a3.to_csv(out / "area3_security_risk_stress_results.csv", index=False)
    lines += ["## Area 3: security-for-costs stress (Table S34)",
              f"- Funder return multiple {a3.mean_funder_roi_multiple.min():.2f} to {a3.mean_funder_roi_multiple.max():.2f} "
              "across the matrix (acceptance threshold 1.18).", ""]

    a4 = run_area4_multiparty_analysis()
    a4.to_csv(out / "area4_multiparty_analysis_results.csv", index=False)
    mp = a4[a4.model_architecture.str.contains("Multi-Party")].set_index("scenario")
    lines += ["## Area 4: employer and subcontractor architecture (Table S35)",
              f"- Balanced score: grid-selected {mp.loc['Grid-selected', 'score_j']:.3f}, conventional "
              f"{mp.loc['Conventional', 'score_j']:.3f}; highest: {mp['score_j'].idxmax()}.",
              f"- Settlement among filed cases: grid-selected {100 * mp.loc['Grid-selected', 'settlement_rate_filed']:.1f}%, "
              f"conventional {100 * mp.loc['Conventional', 'settlement_rate_filed']:.1f}%.", ""]

    if include_grid_checks:
        g = run_grid_checks()
        (out / "area5_grid_checks.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
        lines += ["## Area 5: agreement-term search checks",
                  f"- Settlement control at 0.70: {g['candidates_at_settlement_control_070']} candidates, highest score "
                  f"{g['max_score_at_settlement_control_070']:.3f} against {g['primary_selection_score']:.3f} for the "
                  f"primary selection, which ranks {g['primary_rank_among_6480']} of 6,480.",
                  f"- Settlement-alignment component removed: primary selection ranks "
                  f"{g['primary_rank_without_settlement_alignment']} of {g['search_candidates']}; leader "
                  f"{g['leader_without_settlement_alignment']}.", ""]

    lines.append(f"Run time: {time.time() - t0:.0f} s.")
    (out / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(include_grid_checks="--skip-grid" not in sys.argv)
