"""
agent.py
════════
Soft Actor-Critic (SAC)


Contents
────────
  ReplayBuffer          — uniform ring buffer (simple, reliable)
  PrioritizedBuffer     — SumTree PER (optional, used by default)
  Actor                 — squashed-Gaussian policy π_θ(a|s)
  TwinCritic            — two Q-networks, always use min(Q1, Q2)
  SACAgent              — ties everything together

Usage
─────
  agent = SACAgent(obs_dim=12, action_dim=1,
                   action_low=np.array([0.8]),
                   action_high=np.array([3.0]))
  action = agent.select_action(obs)
  losses = agent.update(batch, weights)
  agent.save("checkpoints/best.pt")
  agent.load("checkpoints/best.pt")
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# SumTree  (backbone of PER)
# ─────────────────────────────────────────────────────────────────────────────
class _SumTree:
    def __init__(self, capacity: int):
        self.cap  = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.size = 0
        self._ptr = 0

    def _up(self, idx, delta):
        p = (idx - 1) // 2
        self.tree[p] += delta
        if p: self._up(p, delta)

    def add(self, priority, data):
        idx = self._ptr + self.cap - 1
        self.data[self._ptr] = data
        self.update(idx, priority)
        self._ptr = (self._ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def update(self, idx, priority):
        self._up(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        idx = 0
        while True:
            l, r = 2 * idx + 1, 2 * idx + 2
            if l >= len(self.tree): break
            idx = l if s <= self.tree[l] else r
            if idx == r: s -= self.tree[l]
        di = idx - self.cap + 1
        return idx, float(self.tree[idx]), self.data[di]

    @property
    def total(self): return float(self.tree[0])


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffers
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Simple uniform experience replay."""

    def __init__(self, capacity: int):
        self.cap  = capacity
        self.buf  = []
        self._ptr = 0

    def add(self, obs, act, rew, next_obs, done):
        t = (obs, act, rew, next_obs, done)
        if len(self.buf) < self.cap:
            self.buf.append(t)
        else:
            self.buf[self._ptr] = t
        self._ptr = (self._ptr + 1) % self.cap

    def sample(self, batch_size, beta=None):
        idxs = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idxs]
        obs, act, rew, nobs, done = map(np.array, zip(*batch))
        weights = np.ones(batch_size, dtype=np.float32)
        return (obs.astype(np.float32), act.astype(np.float32),
                rew.astype(np.float32), nobs.astype(np.float32),
                done.astype(np.float32)), weights, idxs

    def update_priorities(self, idxs, td_errors): pass   # no-op for uniform
    def __len__(self): return len(self.buf)


class PrioritizedBuffer:
    """
    Prioritised Experience Replay.
    Sample probability ∝ |TD error|^alpha.
    IS weights w_i = (N·P(i))^{-beta}  to correct bias.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        self.tree     = _SumTree(capacity)
        self.alpha    = alpha
        self.eps      = eps
        self._max_pri = 1.0

    def add(self, obs, act, rew, next_obs, done):
        # _max_pri is already in the exponentiated tree-priority domain.
        # Raising it to alpha again would under-prioritise every new sample.
        self.tree.add(self._max_pri, (obs, act, rew, next_obs, done))

    def sample(self, batch_size: int, beta: float = 0.4):
        seg  = self.tree.total / batch_size
        idxs, pris = [], []
        for i in range(batch_size):
            s  = np.random.uniform(seg * i, seg * (i + 1))
            ix, p, _ = self.tree.get(s)
            idxs.append(ix)
            pris.append(max(p, self.eps))

        probs   = np.array(pris) / self.tree.total
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        batch = [self.tree.data[ix - self.tree.cap + 1] for ix in idxs]
        obs, act, rew, nobs, done = map(np.array, zip(*batch))
        return (obs.astype(np.float32), act.astype(np.float32),
                rew.astype(np.float32), nobs.astype(np.float32),
                done.astype(np.float32)), weights.astype(np.float32), idxs

    def update_priorities(self, idxs, td_errors):
        for ix, err in zip(idxs, td_errors):
            p = (abs(float(err)) + self.eps) ** self.alpha
            self.tree.update(ix, p)
            self._max_pri = max(self._max_pri, p)

    def __len__(self): return self.tree.size


# ─────────────────────────────────────────────────────────────────────────────
# Network helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mlp(in_dim, hidden, out_dim, norm=True):
    """3-layer MLP with optional LayerNorm."""
    layers = []
    dims = [in_dim] + [hidden, hidden]
    for i, (d_in, d_out) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if norm: layers.append(nn.LayerNorm(d_out))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


def _ortho(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Actor  — Squashed Gaussian
# ─────────────────────────────────────────────────────────────────────────────
class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, obs_dim, action_dim, hidden, action_low, action_high):
        super().__init__()
        self.trunk     = _mlp(obs_dim, hidden, hidden)
        self.mean_head = nn.Linear(hidden, action_dim)
        self.std_head  = nn.Linear(hidden, action_dim)

        scale = torch.FloatTensor((action_high - action_low) / 2.0)
        bias  = torch.FloatTensor((action_high + action_low) / 2.0)
        self.register_buffer("scale", scale)
        self.register_buffer("bias",  bias)

        self.trunk.apply(_ortho)
        _ortho(self.mean_head, gain=0.01)
        _ortho(self.std_head,  gain=0.01)

    def forward(self, obs):
        x = self.trunk(obs)
        mean    = self.mean_head(x)
        log_std = self.std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z    = dist.rsample()
        y    = torch.tanh(z)
        action = y * self.scale + self.bias

        log_prob = (dist.log_prob(z)
                    - torch.log(self.scale * (1 - y.pow(2)) + 1e-6)
                    ).sum(dim=-1, keepdim=True)
        mean_act = torch.tanh(mean) * self.scale + self.bias
        return action, log_prob, mean_act

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        if deterministic:
            mean, _ = self.forward(obs)
            action = torch.tanh(mean) * self.scale + self.bias
        else:
            action, _, _ = self.sample(obs)
        return action.cpu().numpy().flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Twin Critic  — two independent Q-networks
# ─────────────────────────────────────────────────────────────────────────────
class TwinCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        in_dim = obs_dim + action_dim

        def _q():
            net = _mlp(in_dim, hidden, 1)
            net.apply(_ortho)
            return net

        self.q1 = _q()
        self.q2 = _q()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# SAC Agent
# ─────────────────────────────────────────────────────────────────────────────
class SACAgent:
    """
    Soft Actor-Critic with:
      • Automatic entropy tuning (auto-α)
      • Twin critics (prevent Q-overestimation)
      • Soft target update
      • Gradient clipping (stable with sharp reward cliffs)

    Parameters
    ──────────
    obs_dim     : observation dimension
    action_dim  : action dimension (1 for scalar price)
    action_low  : numpy array of action lower bounds
    action_high : numpy array of action upper bounds
    hidden      : hidden layer width (default 256)
    lr          : learning rate for all networks (default 3e-4)
    gamma       : discount factor (default 0.99)
    tau         : soft update coefficient (default 0.005)
    device      : "cuda" or "cpu"
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        action_low:  np.ndarray,
        action_high: np.ndarray,
        hidden:      int   = 256,
        lr:          float = 3e-4,
        gamma:       float = 0.99,
        tau:         float = 0.005,
        device:      str   = DEVICE,
    ):
        self.device = torch.device(device)
        self.gamma  = gamma
        self.tau    = tau

        self.actor  = Actor(obs_dim, action_dim, hidden, action_low, action_high
                            ).to(self.device)
        self.critic = TwinCritic(obs_dim, action_dim, hidden).to(self.device)
        self.critic_tgt = TwinCritic(obs_dim, action_dim, hidden).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        # Auto entropy temperature α
        self.target_ent = -float(action_dim)
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=self.device)

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_alpha  = torch.optim.Adam([self.log_alpha],         lr=lr)

        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return self.actor.act(obs_t, deterministic=deterministic)

    def update(self, batch, weights: np.ndarray) -> dict:
        """One SAC gradient step. Returns dict of scalar losses + td_errors."""
        obs, act, rew, nobs, done = batch
        B = len(obs)

        obs_t  = torch.FloatTensor(obs ).to(self.device)
        act_t  = torch.FloatTensor(act ).to(self.device)
        rew_t  = torch.FloatTensor(rew ).unsqueeze(1).to(self.device)
        nobs_t = torch.FloatTensor(nobs).to(self.device)
        done_t = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        w_t    = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # ── Critic targets ────────────────────────────────────────────────────
        with torch.no_grad():
            na, nlp, _ = self.actor.sample(nobs_t)
            q1n, q2n   = self.critic_tgt(nobs_t, na)
            y = rew_t + (1.0 - done_t) * self.gamma * (
                torch.min(q1n, q2n) - self.alpha * nlp)

        # ── Critic loss ───────────────────────────────────────────────────────
        q1, q2   = self.critic(obs_t, act_t)
        td1, td2 = q1 - y, q2 - y
        c_loss   = (w_t * (td1.pow(2) + td2.pow(2))).mean()

        self.opt_critic.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        td_errors = ((td1.abs() + td2.abs()) / 2.0).detach().cpu().numpy().flatten()

        # ── Actor loss ────────────────────────────────────────────────────────
        ap, lp, _ = self.actor.sample(obs_t)
        q1p, q2p  = self.critic(obs_t, ap)
        a_loss    = (self.alpha.detach() * lp - torch.min(q1p, q2p)).mean()

        self.opt_actor.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # ── Alpha loss ────────────────────────────────────────────────────────
        al_loss = -(self.log_alpha * (lp + self.target_ent).detach()).mean()
        self.opt_alpha.zero_grad()
        al_loss.backward()
        self.opt_alpha.step()

        # ── Soft target update ────────────────────────────────────────────────
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.updates += 1
        return {
            "critic_loss": c_loss.item(),
            "actor_loss":  a_loss.item(),
            "alpha_loss":  al_loss.item(),
            "alpha":       self.alpha.item(),
            "td_errors":   td_errors,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "critic_tgt":   self.critic_tgt.state_dict(),
            "log_alpha":    self.log_alpha.detach().cpu(),
            "updates":      self.updates,
        }, path)
        print(f"  [SAC] Saved → {path}")

    def load(self, path: str):
        # Checkpoints contain tensors and primitive metadata only. Restrict the
        # loader to that safe subset instead of unpickling arbitrary objects.
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_tgt.load_state_dict(ckpt["critic_tgt"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        self.updates = ckpt.get("updates", 0)
        print(f"  [SAC] Loaded ← {path}")
