"""
evaluate.py
───────────
Post-training evaluation utilities.

1. evaluate_agent(env, agent, n_episodes)
   → mean reward + mean acceptance for the trained agent

2. compare_strategies(env, agent, prices, n_episodes)
   → paired comparison: SAC vs constant-price baselines using common
     exogenous random seeds for every strategy

3. collect_rollout(env, agent, seed)
   → full step-level data for a single episode (used by visualize.py)

4. demand_scenario_samples(env, agent)
   → table of sample decisions: High/Medium/Low demand, with events
"""

import numpy as np
from tqdm import tqdm
from environment import UberPricingEnv


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_agent(env, agent, n_episodes=15, deterministic=True):
    """Return (mean_reward, mean_acceptance) over n_episodes."""
    rewards, accepts = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=9000 + ep * 41)
        ep_r, ep_a = 0.0, []
        while True:
            act = agent.select_action(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(act)
            ep_r += r
            ep_a.append(info.get("acceptance", 0.7))
            if term or trunc: break
        rewards.append(ep_r)
        accepts.append(float(np.mean(ep_a)))
    return float(np.mean(rewards)), float(np.mean(accepts))


# ─────────────────────────────────────────────────────────────────────────────
def compare_strategies(env, agent, prices=None, n_episodes=300, seed=7777):
    """
    Paired comparison with common exogenous RNG seeds for every strategy.

    Policies still induce different demand and supply trajectories because
    price, churn, served rides, and driver attraction are endogenous.

    Returns
    -------
    results : dict  {label: np.ndarray of per-episode rewards}
    detail  : dict  {label: list of per-episode step dicts}
    """
    if prices is None:
        prices = [0.8, 1.0, 1.2, 1.5, 2.0]

    results, detail = {}, {}

    def _run(action_fn, label, desc):
        ep_rewards, ep_detail = [], []
        for ep in tqdm(range(n_episodes), leave=False, desc=desc):
            obs, _ = env.reset(seed=seed + ep)
            ep_r, steps = 0.0, []
            while True:
                act = action_fn(obs)
                obs, r, term, trunc, info = env.step(act)
                ep_r += r
                steps.append({
                    "price":       info.get("price",      1.0),
                    "acceptance":  info.get("acceptance", 0.7),
                    "revenue":     info.get("revenue",    0.0),
                    "demand":      info.get("demand",     50.0),
                    "supply":      info.get("supply",     40.0),
                    "hour":        info.get("hour",       12.0),
                    "weather":     info.get("weather",    "Clear"),
                    "event":       info.get("event",      "None"),
                    "churn":       info.get("churn",      0.0),
                    "served":      info.get("served",     0.0),
                })
                if term or trunc: break
            ep_rewards.append(ep_r)
            ep_detail.append(steps)
        results[label] = np.array(ep_rewards)
        detail[label]  = ep_detail

    # SAC (RL)
    print("  Evaluating: SAC (RL)")
    _run(lambda obs: agent.select_action(obs, deterministic=True),
         "SAC (RL)", "RL")

    # Constant baselines
    for p in prices:
        label = f"Fixed {p:.1f}×"
        print(f"  Evaluating: {label}")
        _run(lambda obs, p=p: np.array([p], dtype=np.float32), label, label)

    # Summary table
    _print_comparison(results, detail)
    return results, detail


# ─────────────────────────────────────────────────────────────────────────────
def _print_comparison(results, detail):
    rl_mean = float(np.mean(results["SAC (RL)"]))
    print(f"\n  {'Strategy':<18} {'Mean Reward':>12} {'Std':>8} "
          f"{'vs SAC':>8} {'Avg Accept':>11}")
    print("  " + "─" * 60)
    for lbl, arr in results.items():
        mu   = float(np.mean(arr))
        std  = float(np.std(arr))
        pct  = (mu / rl_mean - 1.0) * 100 if lbl != "SAC (RL)" else 0.0
        tag  = f"{pct:+.1f}%" if lbl != "SAC (RL)" else "baseline"
        ep_a = []
        if lbl in detail:
            for ep in detail[lbl]:
                if ep: ep_a.append(np.mean([s["acceptance"] for s in ep]))
        acc = f"{np.mean(ep_a)*100:.1f}%" if ep_a else "—"
        print(f"  {lbl:<18} {mu:>12,.0f} {std:>8,.0f} {tag:>8} {acc:>11}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
def collect_rollout(env, agent, seed=0, deterministic=True):
    """Collect full step-level data for one episode."""
    obs, _ = env.reset(seed=seed)
    records = []
    while True:
        act = agent.select_action(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(act)
        records.append({
            "step":       info.get("step",       len(records) + 1),
            "hour":       info.get("hour",        12.0),
            "weather":    info.get("weather",     "Clear"),
            "event":      info.get("event",       "None"),
            "demand":     info.get("demand",      50.0),
            "supply":     info.get("supply",      40.0),
            "price":      info.get("price",       1.0),
            "acceptance": info.get("acceptance",  0.7),
            "revenue":    info.get("revenue",     0.0),
            "served":     info.get("served",      0.0),
            "idle":       info.get("idle",        0.0),
            "unmet":      info.get("unmet",       0.0),
            "churn":      info.get("churn",       0.0),
            "reward":     r,
        })
        if term or trunc: break
    return records


# ─────────────────────────────────────────────────────────────────────────────
def demand_scenario_samples(env, agent, n_episodes=8, seed=3000):
    """
    Collect a diverse set of step decisions and return a formatted table.

    Useful for presentation: shows what the agent decided at high, medium,
    and low demand, with and without events.
    """
    all_steps = []
    for ep in range(n_episodes):
        recs = collect_rollout(env, agent, seed=seed + ep * 17)
        for r in recs:
            r["episode"] = ep
            ds = r["demand"] / max(r["supply"], 1.0)
            r["ds_ratio"] = ds
            if ds > 1.5:      r["demand_tier"] = "HIGH"
            elif ds < 0.75:   r["demand_tier"] = "LOW"
            else:              r["demand_tier"] = "MEDIUM"
        all_steps.extend(recs)

    # Pick representative samples
    high   = [s for s in all_steps if s["demand_tier"] == "HIGH"]
    medium = [s for s in all_steps if s["demand_tier"] == "MEDIUM"]
    low    = [s for s in all_steps if s["demand_tier"] == "LOW"]
    events = [s for s in all_steps if s["event"] == "MajorEvent"]

    # Sort by ds_ratio descending for high, ascending for low
    high.sort(key=lambda x: -x["ds_ratio"])
    low.sort(key=lambda x: x["ds_ratio"])

    samples = (high[:5] + medium[:4] + low[:4] + events[:3])
    # Deduplicate
    seen, unique = set(), []
    for s in samples:
        key = (s["episode"], s["step"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"\n  {'Hour':>6} {'Period':>8} {'Weather':>7} {'Tier':>7} "
          f"{'D/S':>6} {'Price':>6} {'Accept':>8} {'Revenue':>9}")
    print("  " + "─" * 68)
    for s in unique[:16]:
        h = s["hour"]
        if   6 <= h < 10:  period = "Morning"
        elif 10 <= h < 16: period = "Midday"
        elif 16 <= h < 21: period = "Evening"
        else:               period = "Night"
        ev = "⚡" if s["event"] == "MajorEvent" else ""
        print(f"  {h:>5.1f}h  {period:>8}  {s['weather']:>7}  "
              f"{s['demand_tier']:>5}{ev:<2}  "
              f"{s['ds_ratio']:>5.2f}  {s['price']:>5.2f}×  "
              f"{s['acceptance']*100:>6.1f}%  "
              f"${s['revenue']:>7.0f}")
    print()
    return unique
