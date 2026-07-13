"""
main.py
───────
Single entry point for the SAC dynamic pricing pipeline.

Usage
─────
  python main.py                          # full pipeline: train → eval → plot
  python main.py --mode train             # train only
  python main.py --mode train --eps 500   # quick 500-episode test
  python main.py --mode eval              # evaluate best checkpoint
  python main.py --mode plot              # generate plots from saved checkpoint
  python main.py --device cpu             # force CPU
"""

import argparse
import os
import sys
import numpy as np
import torch

from config import Config
from environment import UberPricingEnv
from agent import SACAgent


def _load_agent(cfg: Config, env: UberPricingEnv):
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
    return agent


def _save_benchmark(cfg, ckpt_path, agent, results, detail):
    from benchmark import build_benchmark, save_benchmark

    summary = build_benchmark(
        results,
        detail,
        n_episodes=cfg.benchmark_episodes,
        seed=cfg.benchmark_seed,
        checkpoint_path=ckpt_path,
        checkpoint_updates=agent.updates,
        reference_label=f'Fixed {cfg.benchmark_reference_price:.1f}×',
    )
    path = os.path.join(cfg.results_dir, 'benchmark.json')
    save_benchmark(summary, path)
    return summary


def mode_train(cfg):
    from train import train
    return train(cfg)


def mode_eval(cfg, ckpt_path):
    from evaluate import compare_strategies
    env   = UberPricingEnv(max_steps=cfg.max_steps, seed=cfg.seed)
    agent = _load_agent(cfg, env)
    agent.load(ckpt_path)
    results, detail = compare_strategies(
        env,
        agent,
        n_episodes=cfg.benchmark_episodes,
        seed=cfg.benchmark_seed,
    )
    _save_benchmark(cfg, ckpt_path, agent, results, detail)
    return env, agent, results


def mode_plot(cfg, ckpt_path, log_path=None):
    from evaluate import compare_strategies
    from visualize import generate_all_plots

    log_path = log_path or cfg.log_file
    env      = UberPricingEnv(max_steps=cfg.max_steps, seed=cfg.seed)
    agent    = _load_agent(cfg, env)
    agent.load(ckpt_path)

    results, detail = compare_strategies(
        env,
        agent,
        n_episodes=cfg.benchmark_episodes,
        seed=cfg.benchmark_seed,
    )
    _save_benchmark(cfg, ckpt_path, agent, results, detail)
    generate_all_plots(log_path, env, agent, results, cfg.results_dir)


def main():
    parser = argparse.ArgumentParser(
        description="SAC Dynamic Pricing — train / eval / plot")
    parser.add_argument("--mode",   choices=["train","eval","plot","all"],
                        default="all")
    parser.add_argument("--eps",    type=int, default=None,
                        help="Override n_episodes")
    parser.add_argument("--ckpt",   type=str, default=None,
                        help="Checkpoint path for eval/plot")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Write outputs under results/<name> and checkpoints/<name>")
    parser.add_argument("--benchmark-episodes", type=int, default=None,
                        help="Override held-out episodes per strategy")
    parser.add_argument("--benchmark-seed", type=int, default=None,
                        help="Override the first common benchmark seed")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    if args.eps:    cfg.n_episodes = args.eps
    if args.device: cfg.device     = args.device
    if args.benchmark_episodes:
        cfg.benchmark_episodes = args.benchmark_episodes
    if args.benchmark_seed is not None:
        cfg.benchmark_seed = args.benchmark_seed
    if args.run_name:
        cfg.results_dir = os.path.join("results", args.run_name)
        cfg.checkpoint_dir = os.path.join("checkpoints", args.run_name)
        cfg.log_file = os.path.join(cfg.results_dir, "training_log.json")

    print(f"\n  Device : {cfg.device}")
    if cfg.device == "cuda" and torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    ckpt = args.ckpt or os.path.join(cfg.checkpoint_dir, "best.pt")

    if args.mode == "train":
        mode_train(cfg)

    elif args.mode == "eval":
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found: {ckpt}"); sys.exit(1)
        mode_eval(cfg, ckpt)

    elif args.mode == "plot":
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found: {ckpt}"); sys.exit(1)
        mode_plot(cfg, ckpt)

    elif args.mode == "all":
        agent, env, log = mode_train(cfg)
        best_ckpt = os.path.join(cfg.checkpoint_dir, 'best.pt')
        if os.path.exists(best_ckpt):
            agent.load(best_ckpt)
        else:
            best_ckpt = os.path.join(cfg.checkpoint_dir, 'final.pt')
            agent.load(best_ckpt)

        from evaluate import compare_strategies
        print("\n  Post-training comparison …")
        results, detail = compare_strategies(
            env,
            agent,
            n_episodes=cfg.benchmark_episodes,
            seed=cfg.benchmark_seed,
        )
        _save_benchmark(cfg, best_ckpt, agent, results, detail)

        from visualize import generate_all_plots
        generate_all_plots(cfg.log_file, env, agent, results, cfg.results_dir)
        print("\n  Pipeline complete.")


if __name__ == "__main__":
    main()
