"""
config.py
─────────
Central configuration for the dynamic pricing RL project.
All hyperparameters in one place — change here, affects everything.
"""

from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ── Environment ───────────────────────────────────────────────────────────
    max_steps:      int   = 144        # steps per episode (144 × 10 min = 1 day)
    seed:           int   = 0

    # ── Training ──────────────────────────────────────────────────────────────
    n_episodes:     int   = 4_000      # training episodes
    warmup_steps:   int   = 3_000      # random actions before learning starts
    batch_size:     int   = 256
    buffer_size:    int   = 200_000    # replay buffer capacity
    use_per:        bool  = True       # True = PER, False = uniform buffer
    gradient_steps: int   = 1         # SAC updates per env step

    # ── SAC ───────────────────────────────────────────────────────────────────
    hidden_dim:     int   = 256
    lr:             float = 3e-4
    gamma:          float = 0.99       # long-horizon (churn builds over steps)
    tau:            float = 0.005      # soft target update rate
    reward_scale:   float = 5e-4       # divides reward before storing

    # ── PER ───────────────────────────────────────────────────────────────────
    per_alpha:      float = 0.6        # priority exponent
    per_beta_start: float = 0.4        # IS-weight start (annealed → 1.0)
    per_beta_end:   float = 1.0
    per_eps:        float = 1e-6

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_freq:      int   = 200        # evaluate every N training episodes
    eval_episodes:  int   = 15
    benchmark_episodes: int = 300      # held-out policy comparison
    benchmark_seed: int = 7_777        # first common exogenous seed
    save_freq:      int   = 500        # checkpoint every N episodes

    # ── Logging ───────────────────────────────────────────────────────────────
    log_freq:       int   = 20         # print progress every N episodes
    results_dir:    str   = "results"
    checkpoint_dir: str   = "checkpoints"
    log_file:       str   = "results/training_log.json"

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── Observation / Action (derived — do not change) ────────────────────────
    obs_dim:        int   = 12         # fixed by environment
    action_dim:     int   = 1
    action_low:     float = 0.8
    action_high:    float = 3.0
