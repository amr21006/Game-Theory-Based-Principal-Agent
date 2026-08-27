# Tripartite TPF Arbitration Model

A tripartite game-theoretic Monte Carlo simulation of third-party funding
(TPF) agreements in construction arbitration, covering the claimant, the
respondent and the funder.

The funding agreement itself is the object of analysis. Seven agreement terms -
funding share, award share, monitoring intensity, settlement control, claimant
co-payment, adverse-cost coverage and staged-funding efficiency - are treated
as decision variables and evaluated over a synthetic population of construction
arbitration cases.

## Scope of this repository

This repository contains the **model only**: the case generator, the scenario
evaluator, the payoff and settlement functions, the objective function, the
parameterized form used for the alternative specifications, and the tests and
robustness modules that exercise them.

It does **not** contain any figure, table or report generation code.

## Install

```bash
git clone <REPOSITORY-URL>
cd tpf-arbitration-model
pip install -r requirements.txt
```

Python 3.11 or later. Dependencies: numpy, pandas, SALib.

## Quick start

```bash
python examples/quickstart.py
```

Generates a synthetic case population under a fixed seed, evaluates a no-funding
baseline and a funded scenario, and prints the summary aggregates and the
objective-score components.

## Using the model

```python
from src.tpf_model import SimulationConfig, generate_cases, evaluate_scenario, summarize_results

cases = generate_cases(SimulationConfig(n_cases=150_000, seed=1514))
terms = {
    "scenario": "Grid-selected",
    "tpf_available": 1.0,
    "funding_share": 0.85,
    "funder_award_share": 0.22,
    "monitoring_intensity": 0.85,
    "settlement_control": 0.05,
    "claimant_copay_share": 0.10,
    "adverse_cost_coverage": 0.70,
    "staged_funding_efficiency": 0.245,
    "security_for_costs_probability": 0.136,
    "min_roi_multiple": 1.18,
}
summary = summarize_results(evaluate_scenario(cases, terms))
```

`src/analysis.py` provides `grid_search`, which evaluates the 5,400-candidate
agreement-term grid and returns the ranked candidates with the winning terms,
and `sobol_indices` and `sobol_indices_access`, which give the first-order and
total-order Sobol decomposition of the balanced objective and of the
constrained-access rate respectively. These are pure functions: they return
DataFrames and write nothing.

`src/tpf_revision.py` provides `ModelSpec` and `evaluate_scenario_spec`, the
parameterized form of the model used for the alternative specifications of the
governance indicator and of the settlement functions. At its default settings it
reproduces the primary model exactly.

## Layout

```
src/tpf_model.py       case generator, scenario evaluator, payoffs,
                       settlement dynamics, objective function, bootstrap
src/tpf_revision.py    parameterized model specification
src/analysis.py        agreement-term grid search, Sobol decomposition,
                       plateau extent
data/raw/              scenario definitions and public calibration targets
examples/quickstart.py minimal end-to-end example
tests/                 model tests and the four robustness modules
```

## Reproducibility

All draws come from a single pseudorandom stream under an explicit seed, and
common random numbers are applied across scenarios so that differences between
scenarios are not sampling variation. The primary run uses `n_cases=150_000`
and `seed=1514`; the replication seeds are 1514, 2026, 2701, 31415 and 424242.

## Tests

```bash
python -m pytest tests -q
```

The robustness modules write their result tables to
`tests/robustness_extensions/outputs/`, which is not tracked.

## Data

`data/raw/scenario_definitions.csv` holds the four fixed scenario definitions;
the grid-selected configuration is derived by the search rather than stored.
`data/raw/public_calibration_targets.csv` holds the public statistics used as
calibration targets, each with its source.

No confidential arbitration files and no funding agreements are used in, or are
required to run, this model.

## License

MIT. See `LICENSE`.
