"""Regression and smoke tests for replay and Soft Actor-Critic components."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from agent import Actor, PrioritizedBuffer, SACAgent  # noqa: E402


ACTION_LOW = np.array([0.8], dtype=np.float32)
ACTION_HIGH = np.array([3.0], dtype=np.float32)


class PrioritizedReplayTests(unittest.TestCase):
    def test_new_transition_gets_current_maximum_tree_priority(self) -> None:
        buffer = PrioritizedBuffer(capacity=8, alpha=0.6, eps=1e-6)
        transition = self._transition(0.0)
        buffer.add(*transition)

        first_leaf = buffer.tree.cap - 1
        buffer.update_priorities([first_leaf], [100.0])
        current_maximum = float(buffer.tree.tree[first_leaf])
        self.assertGreater(current_maximum, 1.0)

        new_leaf = buffer.tree.cap  # second data slot in an empty ring buffer
        buffer.add(*self._transition(1.0))

        self.assertAlmostEqual(
            float(buffer.tree.tree[new_leaf]),
            current_maximum,
            places=12,
            msg=(
                "A fresh transition must receive the existing maximum SumTree "
                "priority; applying PER alpha twice lowers that priority."
            ),
        )

    @staticmethod
    def _transition(value: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        observation = np.full(12, value, dtype=np.float32)
        action = np.array([1.0], dtype=np.float32)
        return observation, action, value, observation + 0.1, False


class ActorTests(unittest.TestCase):
    def test_deterministic_act_does_not_advance_torch_rng(self) -> None:
        torch.manual_seed(1234)
        actor = Actor(
            obs_dim=12,
            action_dim=1,
            hidden=16,
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
        )
        observation = torch.zeros((1, 12), dtype=torch.float32)
        state_before = torch.random.get_rng_state().clone()

        first_action = actor.act(observation, deterministic=True)
        state_after = torch.random.get_rng_state()
        second_action = actor.act(observation, deterministic=True)

        self.assertTrue(
            torch.equal(state_before, state_after),
            "Deterministic inference must not draw a discarded policy sample.",
        )
        np.testing.assert_array_equal(first_action, second_action)


class SACAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.agent = SACAgent(
            obs_dim=12,
            action_dim=1,
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
            hidden=16,
            device="cpu",
        )

    def test_selected_actions_respect_price_bounds(self) -> None:
        observation = np.linspace(-1.0, 1.0, 12, dtype=np.float32)

        actions = [self.agent.select_action(observation, deterministic=True)]
        actions.extend(
            self.agent.select_action(observation, deterministic=False)
            for _ in range(16)
        )

        for action in actions:
            self.assertEqual(action.shape, (1,))
            self.assertTrue(np.all(np.isfinite(action)))
            self.assertTrue(np.all(action >= ACTION_LOW))
            self.assertTrue(np.all(action <= ACTION_HIGH))

    def test_single_update_smoke(self) -> None:
        rng = np.random.default_rng(11)
        batch_size = 8
        batch = (
            rng.uniform(-1.0, 1.0, (batch_size, 12)).astype(np.float32),
            rng.uniform(ACTION_LOW, ACTION_HIGH, (batch_size, 1)).astype(np.float32),
            rng.normal(0.0, 0.2, batch_size).astype(np.float32),
            rng.uniform(-1.0, 1.0, (batch_size, 12)).astype(np.float32),
            np.array([0, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32),
        )
        weights = np.linspace(0.5, 1.0, batch_size, dtype=np.float32)

        metrics = self.agent.update(batch, weights)

        self.assertEqual(self.agent.updates, 1)
        self.assertEqual(metrics["td_errors"].shape, (batch_size,))
        self.assertTrue(np.all(np.isfinite(metrics["td_errors"])))
        for key in ("critic_loss", "actor_loss", "alpha_loss", "alpha"):
            self.assertTrue(np.isfinite(metrics[key]), f"{key} must be finite")
        self.assertGreater(metrics["alpha"], 0.0)


if __name__ == "__main__":
    unittest.main()
