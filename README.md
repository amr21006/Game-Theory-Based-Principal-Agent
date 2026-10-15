# Tripartite TPF Arbitration Model

A tripartite game-theoretic Monte Carlo simulation of third-party funding
(TPF) agreements in construction arbitration, covering the claimant, the
respondent and the funder.

The funding agreement itself is the object of analysis. Seven agreement terms are
modeled - funding share, award share, monitoring intensity, settlement control,
claimant co-payment, adverse-cost coverage and staged-funding efficiency. The first
six are searched as decision variables; staged-funding efficiency follows from
monitoring. All are evaluated over a synthetic population of construction
arbitration cases.

The supplemental materials of the accompanying paper (Tables S1 to S35) are
deposited at https://doi.org/10.5281/zenodo.20598940.

## Scope of this repository

This repository contains the **model only**: the case generator, the scenario
evaluator, the payoff and settlement functions, the objective function, the
parameterized form used for the alternative specifications, and the tests and
robustness modules that exercise them.

It does **not** contain any figure, table or report generation code.

## Install

```bash
git clone https://github.com/amr21006/Game-Theory-Based-Principal-Agent.git
cd Game-Theory-Based-Principal-Agent
pip install -r requirements.txt
```

Python 3.11 or later. Dependencies: numpy, pandas, SALib (which installs SciPy).

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

`src/tpf_param.py` provides `Params` and `evaluate`, a coefficient-parameterized
form of the primary evaluator used by the robustness modules. It exposes the
calibration constant of Eq. (1), the behavioral coefficients perturbed in Table
S32, and an employer-and-subcontractor counterclaim extension. At `Params()` it
reproduces `evaluate_scenario` exactly; `tests/test_param_equivalence.py` checks this.

## Layout

```
src/tpf_model.py       case generator, scenario evaluator, payoffs,
                       settlement dynamics, objective function, bootstrap
src/tpf_revision.py    parameterized model specification
src/tpf_param.py       coefficient-parameterized evaluator for the robustness modules
src/analysis.py        agreement-term grid search, Sobol decomposition,
                       plateau extent
data/raw/              scenario definitions and public calibration targets
examples/quickstart.py minimal end-to-end example
tests/                 model tests, the equivalence test and the robustness modules
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

The unit tests run the robustness modules on reduced populations.

## Robustness modules

```bash
python tests/robustness_extensions/run_all_extensions.py
```

runs every module on the primary population (150,000 cases, seed 1514) in a few
minutes and writes the result tables and a summary generated from them to
`tests/robustness_extensions/outputs/`, which is not tracked.

| Module | Check | Paper |
|---|---|---|
| `test_area1_coefficient_bounds.py` | eight behavioral coefficients varied by +/-50% | Table S32 |
| `test_area2_unctad_construction_calibration.py` | recalibration to the 35.1% construction-sector investor success rate, with the five replication seeds | Table S33 |
| `test_area3_legal_security_risk.py` | security-for-costs probability and adverse-cost coverage | Table S34 |
| `test_area4_multiparty_structure.py` | employer and subcontractor counterclaim architecture | Table S35 |
| `grid_checks.py` | settlement control extended to 0.70; search re-ranked without the settlement-alignment component | Results |

Every module evaluates the primary model through `src/tpf_param.py` and scores each
structure by the balanced objective of Eq. (15) against the no-funding baseline of
the same specification.

## Data

`data/raw/scenario_definitions.csv` holds the four fixed scenario definitions;
the grid-selected configuration is derived by the search rather than stored.
`data/raw/public_calibration_targets.csv` holds the public statistics used as
calibration targets, each with its source.

No confidential arbitration files and no funding agreements are used in, or are
required to run, this model.

## License

MIT. See `LICENSE`.
