"""
environment.py
══════════════
Uber-Style Dynamic Pricing Environment
=======================================

MDP Design
──────────
  State  : 12-dimensional continuous vector (see _get_obs)
  Action : price multiplier in [0.8, 3.0]  (continuous)
  Reward : balanced revenue + aggregate acceptance target (see _compute_reward)

Simulation
──────────
  • Each step = 10 simulated minutes  →  144 steps ≈ 1 full day
  • Demand follows a realistic two-peak circadian curve (morning + evening rush)
  • Weather and special events add stochastic demand spikes
  • Acceptance is a logistic function of price — high prices lose riders
  • Sustained overpricing builds "churn" that suppresses future demand

Key Design Choices (anti reward-hacking)
─────────────────────────────────────────
  1. Acceptance is directly penalised below 60 % — agent cannot "hide"
     low acceptance behind collapsed demand.
  2. Churn memory: running tracker of poor acceptance. Elevated churn
     reduces future demand (market shrinks) AND adds a direct penalty.
  3. Efficiency bonus: reward for *completing* a high fraction of
     willing trips — agent must actually serve riders, not just price high.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_FARE        = 12.0        # $ per ride at 1× multiplier
STEP_MIN         = 10          # simulated minutes per env step
MAX_ZONE_DEMAND  = 120.0
MAX_ZONE_SUPPLY  = 100.0
MIN_PRICE        = 0.8
MAX_PRICE        = 3.0

# Reward weights
W_REVENUE        = 1.0
W_ACCEPT_PENALTY = 4.0        # multiplier for acceptance below floor
W_CHURN          = 0.6
W_IDLE           = 0.10
W_SMOOTH         = 0.35
W_EFFICIENCY     = 0.40

# Acceptance thresholds
ACCEPT_FLOOR     = 0.60       # below this → linear penalty


def _circadian(hour: float) -> float:
    """
    Smooth demand multiplier based on hour of day [0, 24).

    Two Gaussian peaks:
      Morning rush  ~8:00   peak ≈ 1.30
      Evening rush  ~18:30  peak ≈ 1.60
    Minimum at ~3:00        floor ≈ 0.20
    """
    h = hour % 24.0
    morning = 1.10 * np.exp(-((h - 8.0) ** 2) / (2 * 2.0 ** 2))
    evening = 1.40 * np.exp(-((h - 18.5) ** 2) / (2 * 2.5 ** 2))
    night   = 0.20 * np.exp(-((h - 2.0)  ** 2) / (2 * 1.2 ** 2))
    return 0.20 + morning + evening + night


# ─────────────────────────────────────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────────────────────────────────────
class UberPricingEnv(gym.Env):
    """
    Dynamic ride-share pricing environment.

    Observation (12-dim float32):
      [0]  demand_norm          total demand / MAX_ZONE_DEMAND
      [1]  supply_norm          total supply / MAX_ZONE_SUPPLY
      [2]  ds_ratio_norm        (demand/supply) / 3.0  clipped to [0,1]
      [3]  hour_sin             sin(2π·hour/24)   cyclic time encoding
      [4]  hour_cos             cos(2π·hour/24)
      [5]  weather_norm         0=clear, 0.5=rain, 1=storm
      [6]  event_flag           0 or 1 (major event active)
      [7]  current_price_norm   (price − 0.8) / 2.2
      [8]  surge_memory         short-run price fatigue [0,1]
      [9]  churn_memory         long-run acceptance fatigue [0,1]
      [10] acceptance_last      acceptance rate at previous step [0,1]
      [11] demand_trend         (demand_now − demand_prev) / MAX  momentum

    Action (1-dim float32):
      price multiplier in [0.8, 3.0]

    Episode:
      144 steps (1 simulated day, 10 min/step)
      Start hour sampled uniformly from [5, 22]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 144, seed: int = None):
        super().__init__()
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=np.array([MIN_PRICE], dtype=np.float32),
            high=np.array([MAX_PRICE], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Episode state (set in reset)
        self.step_count    = 0
        self.hour          = 8.0
        self.demand        = 60.0
        self.supply        = 50.0
        self.prev_demand   = 60.0
        self.weather       = 0        # 0=clear 1=rain 2=storm
        self.event         = 0        # 0=none  1=major event
        self.price         = 1.0
        self.surge_memory  = 0.0
        self.churn_memory  = 0.0
        self.accept_last   = 0.80

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        # Let Gymnasium initialise np_random so wrappers and the official
        # environment checker observe the expected seeded-reset contract.
        super().reset(seed=seed)
        self.rng = self.np_random
        if seed is not None:
            self.action_space.seed(seed)

        self.step_count   = 0
        self.hour         = float(self.rng.uniform(5.0, 22.0))
        self.weather      = int(self.rng.choice([0, 1, 2], p=[0.70, 0.22, 0.08]))
        self.event        = int(self.rng.choice([0, 1],    p=[0.90, 0.10]))
        self.price        = 1.0
        self.surge_memory = 0.0
        self.churn_memory = 0.0
        self.accept_last  = 0.80

        self.demand, self.supply = self._init_market()
        self.prev_demand = self.demand

        return self._get_obs(), self._get_info()

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action):
        price = float(np.clip(action[0], MIN_PRICE, MAX_PRICE))

        # Outcomes below belong to the market state in which the action was
        # selected. Preserve that context before evolving to the next state so
        # evaluation and plots do not pair a decision with the following step.
        decision_context = {
            'hour':    self.hour,
            'demand':  self.demand,
            'supply':  self.supply,
            'weather': ['Clear', 'Rain', 'Storm'][self.weather],
            'event':   ['None', 'MajorEvent'][self.event],
        }

        # ── Acceptance (logistic + fatigue) ───────────────────────────────────
        x       = price - 1.0
        fatigue = 1.6 * self.surge_memory + 2.2 * self.churn_memory
        logit   = 2.6 * x + fatigue
        accept  = float(np.clip(1.0 / (1.0 + np.exp(logit)), 0.08, 0.98))

        # ── Demand elasticity ─────────────────────────────────────────────────
        elasticity = float(np.clip(1.0 - 0.20 * (price - 1.0), 0.30, 1.20))
        effective_demand = self.demand * elasticity

        # ── Market clearing ───────────────────────────────────────────────────
        willing   = effective_demand * accept
        served    = min(willing, self.supply)
        idle      = max(self.supply - served, 0.0)
        unmet     = max(willing - served, 0.0)

        # ── Revenue ───────────────────────────────────────────────────────────
        revenue = served * BASE_FARE * price

        # ── Reward ───────────────────────────────────────────────────────────
        reward = self._compute_reward(
            price, revenue, accept, served, willing, idle, unmet
        )

        # ── Evolve market ─────────────────────────────────────────────────────
        self.prev_demand = self.demand
        self._evolve(price, served)

        # ── Update state variables ────────────────────────────────────────────
        self.price        = price
        self.accept_last  = accept
        self.surge_memory = float(np.clip(
            0.80 * self.surge_memory + 0.20 * ((price - 1.0) / 2.0), 0.0, 1.0))
        deficit = max(0.0, 0.60 - accept)
        self.churn_memory = float(np.clip(
            0.93 * self.churn_memory + 0.07 * deficit, 0.0, 1.0))

        self.step_count += 1
        # Demand is clipped to at least 3 rides, so market collapse is not a
        # terminal state. Episodes end only at the configured time limit.
        terminated = False
        truncated  = self.step_count >= self.max_steps

        info = self._get_info()
        info.update({
            "price":       price,
            "acceptance":  accept,
            "revenue":     revenue,
            "served":      served,
            "idle":        idle,
            "unmet":       unmet,
            "demand":      self.demand,
            "supply":      self.supply,
            "hour":        self.hour,
            "weather":     ["Clear", "Rain", "Storm"][self.weather],
            "event":       ["None", "MajorEvent"][self.event],
            "churn":       self.churn_memory,
        })
        info.update(decision_context)
        return self._get_obs(), reward, terminated, truncated, info

    # ── Reward ────────────────────────────────────────────────────────────────
    def _compute_reward(self, price, revenue, accept, served, willing, idle, unmet):
        """
        Multi-objective reward balancing revenue and aggregate acceptance.

        Components:
          + revenue                 core income signal
          + efficiency_bonus        fraction of willing trips served
          − acceptance_penalty      for accept < 60% (linear)
          − churn_penalty           running cost of sustained overpricing
          − idle_penalty            wasted driver capacity
          − smoothness_penalty      discourages erratic price jumps

        With the current price bounds and logistic response curve, the maximum
        attainable acceptance is about 62.7%. A higher acceptance bonus would
        therefore be dead code and is intentionally not included.
        """
        # Efficiency: what fraction of willing rides were completed?
        efficiency    = served / max(willing, 1.0)
        eff_bonus     = W_EFFICIENCY * efficiency * revenue

        # Aggregate acceptance target
        if accept < ACCEPT_FLOOR:
            gap          = ACCEPT_FLOOR - accept
            accept_pen   = W_ACCEPT_PENALTY * gap * revenue
        else:
            accept_pen   = 0.0

        # Churn: sustained overpricing cost
        churn_pen  = W_CHURN * self.churn_memory * self.demand * 0.5

        # Idle drivers
        idle_pen   = W_IDLE * idle

        # Price smoothness
        smooth_pen = W_SMOOTH * abs(price - self.price) * self.demand

        reward = (W_REVENUE * revenue
                  + eff_bonus
                  - accept_pen
                  - churn_pen
                  - idle_pen
                  - smooth_pen)
        return float(reward)

    # ── Market initialisation ─────────────────────────────────────────────────
    def _init_market(self):
        circ    = _circadian(self.hour)
        w_mult  = [1.0, 1.30, 1.75][self.weather]
        e_mult  = 1.40 if self.event else 1.0
        base_d  = 50.0 * circ * w_mult * e_mult
        demand  = float(np.clip(
            base_d * self.rng.uniform(0.80, 1.20), 3.0, MAX_ZONE_DEMAND))
        supply  = float(np.clip(
            45.0 * self.rng.uniform(0.75, 1.25), 5.0, MAX_ZONE_SUPPLY))
        return demand, supply

    # ── Market evolution ──────────────────────────────────────────────────────
    def _evolve(self, price: float, served: float):
        # Advance clock
        self.hour = (self.hour + STEP_MIN / 60.0) % 24.0

        # Weather transition (slow Markov chain)
        w_probs = {0: [0.92, 0.06, 0.02],
                   1: [0.15, 0.78, 0.07],
                   2: [0.05, 0.20, 0.75]}[self.weather]
        self.weather = int(self.rng.choice([0, 1, 2], p=w_probs))

        # Event (5% chance per step)
        self.event = int(self.rng.random() < 0.05)

        # Demand: mean-reverts to circadian anchor, suppressed by churn
        circ       = _circadian(self.hour)
        w_mult     = [1.0, 1.30, 1.75][self.weather]
        e_mult     = 1.40 if self.event else 1.0
        anchor     = 50.0 * circ * w_mult * e_mult
        churn_sup  = 1.0 - 0.30 * self.churn_memory   # churn suppresses demand
        noise      = self.rng.normal(0.0, 0.05)
        self.demand = float(np.clip(
            (0.60 * self.demand + 0.40 * anchor * churn_sup - 0.05 * served)
            * (1.0 + noise),
            3.0, MAX_ZONE_DEMAND,
        ))

        # Supply: drivers attracted by high price, repelled by nothing to do
        price_pull  = 1.0 + 0.18 * (price - 1.0)
        supply_noise = self.rng.normal(0.0, 0.04)
        self.supply = float(np.clip(
            (0.78 * self.supply + 0.22 * 45.0 * price_pull) * (1.0 + supply_noise),
            5.0, MAX_ZONE_SUPPLY,
        ))

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        hr_rad = 2.0 * np.pi * self.hour / 24.0
        trend  = (self.demand - self.prev_demand) / MAX_ZONE_DEMAND
        obs = np.array([
            self.demand / MAX_ZONE_DEMAND,
            self.supply / MAX_ZONE_SUPPLY,
            np.clip((self.demand / max(self.supply, 1.0)) / 3.0, 0.0, 1.0),
            np.sin(hr_rad),
            np.cos(hr_rad),
            self.weather / 2.0,
            float(self.event),
            (self.price - MIN_PRICE) / (MAX_PRICE - MIN_PRICE),
            self.surge_memory,
            self.churn_memory,
            self.accept_last,
            np.clip(trend, -0.5, 0.5),
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> dict:
        return {
            "step":         self.step_count,
            "hour":         self.hour,
            "demand":       self.demand,
            "supply":       self.supply,
            "churn_memory": self.churn_memory,
        }

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self, mode="human"):
        h = int(self.hour); m = int((self.hour - h) * 60)
        print(f"[{h:02d}:{m:02d}] step={self.step_count:3d} | "
              f"price={self.price:.2f}x | accept={self.accept_last:.2f} | "
              f"D={self.demand:.0f} S={self.supply:.0f} | "
              f"churn={self.churn_memory:.3f} | "
              f"weather={['Clear','Rain','Storm'][self.weather]}")
