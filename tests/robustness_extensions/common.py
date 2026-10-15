"""Shared set-up for the robustness modules: the five evaluated structures and their scoring.

Every module evaluates the primary model (``src.tpf_model``) through its parameterized form
``src.tpf_param`` and scores each structure with the balanced objective of Eq. (15) against
the no-funding baseline of the same specification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import candidate_terms  # noqa: E402
from src.tpf_model import balanced_score, load_scenarios, summarize_results  # noqa: E402
from src.tpf_param import Params, evaluate  # noqa: E402

LABELS = {
    "No TPF baseline": "No funding",
    "Conventional TPF": "Conventional",
    "Regulated disclosure TPF": "Regulated disclosure",
    "Portfolio TPF": "Portfolio",
    "Grid-selected": "Grid-selected",
}
GRID_SELECTED = candidate_terms(0.85, 0.22, 0.85, 0.05, 0.10, 0.70, scenario="Grid-selected")
SETTLEMENT_TARGET = 0.33


def scenarios() -> list[dict]:
    """The four fixed scenario definitions plus the grid-selected configuration."""
    fixed = load_scenarios(ROOT / "data" / "raw" / "scenario_definitions.csv")
    return [row.to_dict() for _, row in fixed.iterrows()] + [dict(GRID_SELECTED)]


def run_structures(cases: pd.DataFrame, P: Params = Params()) -> tuple[dict, dict, dict]:
    """Evaluate the five structures under one specification.

    Returns (summaries, scores against the 0.33 benchmark, scores against the specification's own
    baseline settlement rate), each keyed by the short structure label."""
    summaries = {}
    for terms in scenarios():
        summaries[LABELS[terms["scenario"]]] = summarize_results(evaluate(cases, terms, P))
    base = summaries["No funding"]
    own_target = float(base["settlement_rate_filed"])
    scores = {k: balanced_score(s, base, SETTLEMENT_TARGET) for k, s in summaries.items()}
    scores_own = {k: balanced_score(s, base, own_target) for k, s in summaries.items()}
    return summaries, scores, scores_own
