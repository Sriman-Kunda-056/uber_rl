"""Tests for machine-readable benchmark summaries and provenance."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from benchmark import build_benchmark, save_benchmark  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    def test_build_and_save_benchmark(self) -> None:
        results = {
            "SAC (RL)": [12.0, 18.0],
            "Fixed 0.8x": [10.0, 10.0],
            "Fixed 1.0x": [15.0, 11.0],
        }
        detail = {
            "SAC (RL)": [
                [self._step(120.0, 0.70, 1.2, 10.0, 0.10)],
                [self._step(180.0, 0.80, 1.4, 14.0, 0.20)],
            ],
            "Fixed 0.8x": [
                [self._step(100.0, 0.90, 0.8, 9.0, 0.05)],
                [self._step(100.0, 0.90, 0.8, 9.0, 0.05)],
            ],
            "Fixed 1.0x": [
                [self._step(100.0, 0.65, 1.0, 8.0, 0.08)],
                [self._step(125.0, 0.75, 1.0, 12.0, 0.12)],
            ],
        }

        with contextlib.nullcontext(PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "_benchmark_test_checkpoint.pt"
            self.addCleanup(checkpoint.unlink, missing_ok=True)
            checkpoint_bytes = b"synthetic checkpoint for provenance\n"
            checkpoint.write_bytes(checkpoint_bytes)

            summary = build_benchmark(
                results,
                detail,
                n_episodes=2,
                seed=500,
                checkpoint_path=str(checkpoint),
                checkpoint_updates=42,
                acceptance_floor=0.60,
            )

            comparison = summary["comparison_to_strongest_fixed_baseline"]
            self.assertEqual(comparison["policy"], "SAC (RL)")
            self.assertEqual(
                comparison["strongest_fixed_baseline"], "Fixed 1.0x"
            )
            paired = comparison["paired_objective_reward_difference"]
            self.assertAlmostEqual(paired["mean"], 2.0)
            self.assertAlmostEqual(paired["standard_deviation"], 50.0**0.5)
            self.assertAlmostEqual(comparison["policy_win_rate"], 0.5)
            self.assertAlmostEqual(
                comparison["objective_reward_percent_difference"],
                100.0 * (15.0 / 13.0 - 1.0),
            )
            self.assertAlmostEqual(
                comparison["gross_revenue_percent_difference"], 100.0 / 3.0
            )
            self.assertAlmostEqual(
                comparison["served_rides_percent_difference"], 20.0
            )
            self.assertAlmostEqual(
                comparison["acceptance_percentage_point_difference"], 5.0
            )

            self._assert_provenance(
                summary, checkpoint, checkpoint_bytes, expected_updates=42
            )

            json.dumps(summary, allow_nan=False)
            output = root / "_benchmark_test_output.json"
            self.addCleanup(output.unlink, missing_ok=True)
            with contextlib.redirect_stdout(io.StringIO()):
                returned_path = save_benchmark(summary, str(output))
            self.assertEqual(returned_path, str(output))
            self.assertTrue(output.read_text(encoding="utf-8").endswith("\n"))
            with output.open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), summary)

    def _assert_provenance(
        self,
        summary: dict,
        checkpoint: Path,
        checkpoint_bytes: bytes,
        *,
        expected_updates: int,
    ) -> None:
        self.assertEqual(summary["schema_version"], 1)
        generated = datetime.fromisoformat(summary["generated_utc"])
        self.assertIsNotNone(generated.tzinfo)

        protocol = summary["protocol"]
        self.assertEqual(protocol["episodes_per_strategy"], 2)
        self.assertEqual(protocol["steps_per_episode"], 1)
        self.assertEqual(protocol["minutes_per_step"], 10)
        self.assertEqual(protocol["seed_start"], 500)
        self.assertEqual(protocol["seed_end"], 501)
        self.assertTrue(protocol["deterministic_policy"])
        self.assertEqual(protocol["acceptance_floor"], 0.60)
        self.assertIn("Common exogenous RNG seeds", protocol["shared_randomness"])

        checkpoint_metadata = summary["checkpoint"]
        self.assertEqual(
            checkpoint_metadata["path"], str(checkpoint).replace(os.sep, "/")
        )
        self.assertEqual(
            checkpoint_metadata["sha256"],
            hashlib.sha256(checkpoint_bytes).hexdigest(),
        )
        self.assertEqual(checkpoint_metadata["updates"], expected_updates)

        self.assertEqual(
            set(summary["software"]), {"python", "gymnasium", "numpy", "torch"}
        )
        self.assertTrue(all(summary["software"].values()))
        self.assertIn("Reward is an engineered objective", summary["interpretation"])

    @staticmethod
    def _step(
        revenue: float,
        acceptance: float,
        price: float,
        served: float,
        churn: float,
    ) -> dict:
        return {
            "revenue": revenue,
            "acceptance": acceptance,
            "price": price,
            "served": served,
            "churn": churn,
        }


if __name__ == "__main__":
    unittest.main()
