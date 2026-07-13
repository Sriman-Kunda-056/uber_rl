'''Machine-readable benchmark summaries for policy comparisons.'''

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np


def _stats(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(arr)) if len(arr) else 0.0
    return {
        'mean': mean,
        'standard_deviation': std,
        'confidence_interval_95': [mean - margin, mean + margin],
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return 'unknown'


def _checkpoint_metadata(path: str | None, updates: int | None) -> dict:
    if not path:
        return {'path': None, 'sha256': None, 'updates': updates}
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'path': path.replace(os.sep, '/'),
        'sha256': digest.hexdigest(),
        'updates': updates,
    }


def build_benchmark(
    results: dict,
    detail: dict,
    *,
    n_episodes: int,
    seed: int,
    checkpoint_path: str | None = None,
    checkpoint_updates: int | None = None,
    acceptance_floor: float = 0.60,
) -> dict:
    '''Summarise paired policy rollouts without hiding outcome trade-offs.'''
    strategies = {}
    episode_metrics = {}

    for label, rewards in results.items():
        episodes = detail[label]
        values = {
            'objective_reward': np.asarray(rewards, dtype=np.float64),
            'simulated_gross_revenue': np.asarray([
                sum(step['revenue'] for step in episode) for episode in episodes
            ]),
            'acceptance_rate': np.asarray([
                np.mean([step['acceptance'] for step in episode])
                for episode in episodes
            ]),
            'average_price_multiplier': np.asarray([
                np.mean([step['price'] for step in episode])
                for episode in episodes
            ]),
            'simulated_rides_served': np.asarray([
                sum(step['served'] for step in episode) for episode in episodes
            ]),
            'mean_churn_memory': np.asarray([
                np.mean([step['churn'] for step in episode])
                for episode in episodes
            ]),
            'steps_at_or_above_acceptance_floor': np.asarray([
                np.mean([
                    step['acceptance'] >= acceptance_floor for step in episode
                ])
                for episode in episodes
            ]),
        }
        episode_metrics[label] = values
        strategies[label] = {name: _stats(metric) for name, metric in values.items()}

    rl_label = next(label for label in results if 'SAC' in label)
    fixed_labels = [label for label in results if label != rl_label]
    strongest_fixed = max(
        fixed_labels,
        key=lambda label: strategies[label]['objective_reward']['mean'],
    )
    rl = episode_metrics[rl_label]
    baseline = episode_metrics[strongest_fixed]
    reward_difference = rl['objective_reward'] - baseline['objective_reward']

    def _relative_percent(metric: str) -> float:
        base_mean = float(np.mean(baseline[metric]))
        return 100.0 * (
            float(np.mean(rl[metric])) / base_mean - 1.0
        )

    comparison = {
        'policy': rl_label,
        'strongest_fixed_baseline': strongest_fixed,
        'paired_objective_reward_difference': _stats(reward_difference),
        'policy_win_rate': float(np.mean(reward_difference > 0.0)),
        'objective_reward_percent_difference': _relative_percent(
            'objective_reward'
        ),
        'gross_revenue_percent_difference': _relative_percent(
            'simulated_gross_revenue'
        ),
        'served_rides_percent_difference': _relative_percent(
            'simulated_rides_served'
        ),
        'acceptance_percentage_point_difference': 100.0 * (
            float(np.mean(rl['acceptance_rate']))
            - float(np.mean(baseline['acceptance_rate']))
        ),
    }

    steps = len(next(iter(detail.values()))[0])
    return {
        'schema_version': 1,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': {
            'environment': 'synthetic single-zone ride-share simulator',
            'episodes_per_strategy': n_episodes,
            'steps_per_episode': steps,
            'minutes_per_step': 10,
            'seed_start': seed,
            'seed_end': seed + n_episodes - 1,
            'deterministic_policy': True,
            'shared_randomness': (
                'Common exogenous RNG seeds; policy actions still change '
                'endogenous demand and supply trajectories.'
            ),
            'acceptance_floor': acceptance_floor,
        },
        'checkpoint': _checkpoint_metadata(
            checkpoint_path, checkpoint_updates
        ),
        'software': {
            'python': platform.python_version(),
            'gymnasium': _package_version('gymnasium'),
            'numpy': _package_version('numpy'),
            'torch': _package_version('torch'),
        },
        'strategies': strategies,
        'comparison_to_strongest_fixed_baseline': comparison,
        'interpretation': (
            'Reward is an engineered objective, not dollars or profit. '
            'Revenue and ride counts are simulated.'
        ),
    }


def save_benchmark(summary: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(f'  [Benchmark] -> {path}')
    return path
