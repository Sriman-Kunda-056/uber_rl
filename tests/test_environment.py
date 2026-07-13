"""Behavioral tests for the custom Gymnasium environment."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from environment import MAX_PRICE, MIN_PRICE, UberPricingEnv  # noqa: E402


class UberPricingEnvTests(unittest.TestCase):
    def test_passes_gymnasium_environment_checker(self) -> None:
        env = UberPricingEnv(max_steps=8)
        with warnings.catch_warnings():
            # The checker recommends normalized actions, but a price multiplier's
            # natural [0.8, 3.0] units are an intentional public API choice.
            warnings.filterwarnings(
                "ignore",
                message=".*recommend using a symmetric and normalized space.*",
            )
            check_env(env, skip_render_check=True)

    def test_seeded_fixed_action_rollout_is_reproducible(self) -> None:
        env = UberPricingEnv(max_steps=12)

        first = self._fixed_action_rollout(env, seed=2026, steps=10)
        second = self._fixed_action_rollout(env, seed=2026, steps=10)

        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])

    def test_observations_and_actions_remain_within_declared_bounds(self) -> None:
        env = UberPricingEnv(max_steps=20)
        observation, info = env.reset(seed=17)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertIsInstance(info, dict)

        boundary_actions = (
            np.array([MIN_PRICE], dtype=np.float32),
            np.array([MAX_PRICE], dtype=np.float32),
        )
        for step in range(env.max_steps):
            action = boundary_actions[step % len(boundary_actions)]
            self.assertTrue(env.action_space.contains(action))

            observation, reward, terminated, truncated, info = env.step(action)

            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(np.isfinite(reward))
            self.assertGreaterEqual(info["price"], MIN_PRICE)
            self.assertLessEqual(info["price"], MAX_PRICE)
            if terminated or truncated:
                break

    def test_time_limit_truncates_exactly_at_max_steps(self) -> None:
        env = UberPricingEnv(max_steps=3)
        env.reset(seed=91)
        action = np.array([1.2], dtype=np.float32)

        for step_number in range(1, 4):
            observation, _, terminated, truncated, info = env.step(action)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertFalse(terminated)
            self.assertEqual(truncated, step_number == 3)
            self.assertEqual(info["step"], step_number)

    @staticmethod
    def _fixed_action_rollout(
        env: UberPricingEnv, seed: int, steps: int
    ) -> list[tuple[np.ndarray, float, bool, bool, dict]]:
        observation, _ = env.reset(seed=seed)
        action = np.array([1.35], dtype=np.float32)
        rollout = []
        for _ in range(steps):
            observation, reward, terminated, truncated, info = env.step(action)
            rollout.append(
                (
                    observation.copy(),
                    reward,
                    terminated,
                    truncated,
                    info.copy(),
                )
            )
            if terminated or truncated:
                break
        return rollout


if __name__ == "__main__":
    unittest.main()
