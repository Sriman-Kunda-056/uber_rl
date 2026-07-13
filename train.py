"""
train.py
────────
Training loop for the SAC dynamic pricing agent.

Features
────────
  • Warmup phase       — random actions to pre-fill the replay buffer
  • PER                — prioritised sampling with annealed IS weights
  • Auto-alpha         — entropy temperature self-calibrates
  • Periodic eval      — deterministic rollout every eval_freq episodes
  • Best-checkpoint    — saves whenever eval reward improves
  • JSON log           — all metrics written to results/training_log.json
"""

import os
import json
import time
import random
import platform
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np
import torch
from tqdm import tqdm

from environment import UberPricingEnv
from agent import SACAgent, PrioritizedBuffer, ReplayBuffer
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# Beta annealing for PER
# ─────────────────────────────────────────────────────────────────────────────
def _beta(step, warmup, total, b0, b1):
    if step <= warmup:
        return b0
    frac = min((step - warmup) / max(total - warmup, 1), 1.0)
    return b0 + frac * (b1 - b0)


# ─────────────────────────────────────────────────────────────────────────────
# Single evaluation run
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(env, agent, n_eps=15):
    """Deterministic rollout; returns mean reward and mean acceptance."""
    rewards, accepts = [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=5000 + ep * 37)
        ep_r, ep_a = 0.0, []
        while True:
            act = agent.select_action(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            ep_r  += r
            ep_a.append(info.get("acceptance", 0.7))
            if term or trunc: break
        rewards.append(ep_r)
        accepts.append(float(np.mean(ep_a)))
    return float(np.mean(rewards)), float(np.mean(accepts))


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    _seed_everything(cfg.seed)
    os.makedirs(cfg.results_dir,    exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"\n{'═'*55}")
    print(f"  SAC Dynamic Pricing  |  device: {cfg.device}")
    print(f"  episodes={cfg.n_episodes}  warmup={cfg.warmup_steps}")
    print(f"  buffer={'PER' if cfg.use_per else 'Uniform'}  γ={cfg.gamma}")
    print(f"{'═'*55}\n")

    env   = UberPricingEnv(max_steps=cfg.max_steps, seed=cfg.seed)
    agent = SACAgent(
        obs_dim     = cfg.obs_dim,
        action_dim  = cfg.action_dim,
        action_low  = np.array([cfg.action_low],  dtype=np.float32),
        action_high = np.array([cfg.action_high], dtype=np.float32),
        hidden      = cfg.hidden_dim,
        lr          = cfg.lr,
        gamma       = cfg.gamma,
        tau         = cfg.tau,
        device      = cfg.device,
    )

    buffer = (
        PrioritizedBuffer(cfg.buffer_size, alpha=cfg.per_alpha, eps=cfg.per_eps)
        if cfg.use_per
        else ReplayBuffer(cfg.buffer_size)
    )

    log = {
        'metadata': {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'config': asdict(cfg),
            'versions': {
                'python': platform.python_version(),
                'numpy': np.__version__,
                'torch': torch.__version__,
            },
        },
        "episode_reward":  [],
        "episode_accept":  [],
        "critic_loss":     [],
        "actor_loss":      [],
        "alpha":           [],
        "eval_reward":     [],
        "eval_accept":     [],
        "eval_episodes":   [],
    }

    best_eval     = -np.inf
    total_steps   = 0
    t0            = time.time()
    total_env_steps = cfg.n_episodes * cfg.max_steps

    pbar = tqdm(range(1, cfg.n_episodes + 1), desc="Training", unit="ep")
    for episode in pbar:
        obs, _ = env.reset(seed=cfg.seed + episode)
        ep_r    = 0.0
        ep_a    = []
        ep_cl, ep_al, ep_alp = [], [], []

        while True:
            # Action
            if total_steps < cfg.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            # Scale reward before storing
            buffer.add(obs, action, reward * cfg.reward_scale,
                       next_obs, float(term))

            # Learn
            if total_steps >= cfg.warmup_steps and len(buffer) >= cfg.batch_size:
                beta = _beta(total_steps, cfg.warmup_steps,
                             total_env_steps,
                             cfg.per_beta_start, cfg.per_beta_end)
                for _ in range(cfg.gradient_steps):
                    batch, weights, idxs = buffer.sample(cfg.batch_size, beta)
                    losses = agent.update(batch, weights)
                    buffer.update_priorities(idxs, losses["td_errors"])

                ep_cl.append(losses["critic_loss"])
                ep_al.append(losses["actor_loss"])
                ep_alp.append(losses["alpha"])

            obs         = next_obs
            ep_r       += reward
            ep_a.append(info.get("acceptance", 0.7))
            total_steps += 1
            if done: break

        # ── Bookkeeping ───────────────────────────────────────────────────────
        log["episode_reward"].append(ep_r)
        log["episode_accept"].append(float(np.mean(ep_a)))
        if ep_cl:
            log["critic_loss"].append(float(np.mean(ep_cl)))
            log["actor_loss"].append(float(np.mean(ep_al)))
            log["alpha"].append(float(np.mean(ep_alp)))

        # ── Progress ──────────────────────────────────────────────────────────
        if episode % cfg.log_freq == 0:
            recent_r = np.mean(log["episode_reward"][-cfg.log_freq:])
            recent_a = np.mean(log["episode_accept"][-cfg.log_freq:])
            alpha_v  = log["alpha"][-1] if log["alpha"] else 0.0
            pbar.set_postfix({
                "avg_r":  f"{recent_r:,.0f}",
                "accept": f"{recent_a:.2f}",
                "α":      f"{alpha_v:.4f}",
                "steps":  total_steps,
            })

        # ── Evaluation ────────────────────────────────────────────────────────
        if episode % cfg.eval_freq == 0:
            ev_r, ev_a = evaluate(env, agent, cfg.eval_episodes)
            log["eval_reward"].append(ev_r)
            log["eval_accept"].append(ev_a)
            log["eval_episodes"].append(episode)

            elapsed = (time.time() - t0) / 60
            print(f"\n  [Eval ep={episode}]  "
                  f"reward={ev_r:,.0f}  accept={ev_a:.2f}  "
                  f"({elapsed:.1f} min)")

            if ev_r > best_eval:
                best_eval = ev_r
                agent.save(os.path.join(cfg.checkpoint_dir, "best.pt"))

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if episode % cfg.save_freq == 0:
            agent.save(os.path.join(cfg.checkpoint_dir, f"ep{episode:05d}.pt"))

    # ── Final save ────────────────────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    log['metadata']['training_minutes'] = elapsed
    log['metadata']['best_eval_reward'] = (
        None if not np.isfinite(best_eval) else float(best_eval)
    )
    agent.save(os.path.join(cfg.checkpoint_dir, "final.pt"))
    with open(cfg.log_file, "w") as f:
        json.dump(log, f, indent=2)

    elapsed = (time.time() - t0) / 60
    print(f"\n  Training done in {elapsed:.1f} min")
    print(f"  Best eval reward: {best_eval:,.0f}")
    print(f"  Log saved → {cfg.log_file}")

    return agent, env, log
