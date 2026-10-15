"""The parameterized evaluator reproduces the primary model exactly at its default settings."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpf_model import SimulationConfig, generate_cases  # noqa: E402
from src.tpf_param import max_difference  # noqa: E402
from tests.robustness_extensions.common import scenarios  # noqa: E402


class TestParamEquivalence(unittest.TestCase):
    def test_default_params_reproduce_primary_model(self):
        cases = generate_cases(SimulationConfig(n_cases=20_000, seed=1514))
        self.assertLess(max_difference(cases, scenarios()), 1e-9)


if __name__ == "__main__":
    unittest.main()
