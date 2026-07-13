"""Regression test for decision-context fields emitted by environment steps."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from environment import UberPricingEnv  # noqa: E402


class EnvironmentDecisionContextTests(unittest.TestCase):
    def test_step_info_describes_pre_transition_market_state(self) -> None:
        env = UberPricingEnv(max_steps=4)
        env.reset(seed=314)
        expected = {
            "hour": env.hour,
            "demand": env.demand,
            "supply": env.supply,
            "weather": ["Clear", "Rain", "Storm"][env.weather],
            "event": ["None", "MajorEvent"][env.event],
        }

        _, _, _, _, info = env.step(np.array([1.1], dtype=np.float32))

        for field, value in expected.items():
            self.assertEqual(info[field], value, f"incorrect decision {field}")
        self.assertNotEqual(info["hour"], env.hour)


if __name__ == "__main__":
    unittest.main()
