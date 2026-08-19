"""
Reinforcement Learning environment (SRS Section 6).

Models prioritization as a sequential decision problem: a responder with
limited capacity observes a bounded WINDOW of pending messages and must
choose exactly one to serve next each timestep, while new messages continue
to arrive stochastically (SRS 6.1). Implemented as a genuine
Gymnasium-compatible environment with reset()/step() semantics, a continuous
observation space and a discrete action space (SRS 6.2) — this is a real RL
formulation, not a rule-based ranking system relabeled as RL.

State (SRS 6.3), per visible slot:
    [one-hot category (8), urgency score, category confidence,
     normalized waiting time, duplicate flag]
plus global features: normalized queue length, normalized episode progress.

Action (SRS 6.4): discrete, one action per visible window slot. Selecting an
index beyond the current (possibly shorter) window is a valid but penalized
no-op.

Reward (SRS 6.5):
    (a) + reward for serving high-urgency messages
    (b) - penalty for serving a duplicate message
    (c) + bonus for serving the globally highest-urgency message in the
          ENTIRE queue (not just the visible window)
    (d) - penalty for letting messages wait, accrued across the whole
          remaining queue every step
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import CATEGORIES

N_CATEGORIES = len(CATEGORIES)


class MessagePrioritizationEnv(gym.Env):
    """A capacity-constrained emergency-message triage environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        window_size: int = 5,
        max_queue_size: int = 40,
        max_episode_steps: int = 60,
        arrival_rate: float = 1.3,
        duplicate_prob: float = 0.08,
        seed: int | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.max_queue_size = max_queue_size
        self.max_episode_steps = max_episode_steps
        self.arrival_rate = arrival_rate
        self.duplicate_prob = duplicate_prob

        # per-slot feature vector: one-hot category + urgency + confidence + wait + dup flag
        self.per_slot_dim = N_CATEGORIES + 4
        obs_dim = self.window_size * self.per_slot_dim + 2  # + queue_len, progress
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.window_size)

        self._rng = np.random.default_rng(seed)
        self.queue: list[dict] = []
        self.t = 0
        self._next_id = 0

        # Episode-level stats, useful for evaluation / UI display.
        self.episode_served = 0
        self.episode_urgency_served: list[float] = []

    # ------------------------------------------------------------------
    def _sample_message(self) -> dict:
        category = int(self._rng.integers(0, N_CATEGORIES))
        urgency = float(np.clip(self._rng.beta(2.0, 2.2), 0.0, 1.0))
        confidence = float(np.clip(self._rng.normal(0.85, 0.08), 0.35, 1.0))
        is_duplicate = bool(self._rng.random() < self.duplicate_prob)
        msg = {
            "id": self._next_id,
            "category": category,
            "urgency": urgency,
            "confidence": confidence,
            "arrival_step": self.t,
            "is_duplicate": is_duplicate,
        }
        self._next_id += 1
        return msg

    def _get_obs(self) -> np.ndarray:
        feats: list[float] = []
        for i in range(self.window_size):
            if i < len(self.queue):
                m = self.queue[i]
                onehot = [0.0] * N_CATEGORIES
                onehot[m["category"]] = 1.0
                wait = min(1.0, (self.t - m["arrival_step"]) / max(1, self.max_episode_steps))
                feats.extend(onehot)
                feats.extend([m["urgency"], m["confidence"], wait, 1.0 if m["is_duplicate"] else 0.0])
            else:
                feats.extend([0.0] * self.per_slot_dim)
        queue_len_norm = min(1.0, len(self.queue) / self.max_queue_size)
        progress = min(1.0, self.t / self.max_episode_steps)
        feats.extend([queue_len_norm, progress])
        return np.asarray(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.queue = []
        self.t = 0
        self._next_id = 0
        self.episode_served = 0
        self.episode_urgency_served = []

        n_init = int(self._rng.integers(3, self.window_size + 3))
        for _ in range(n_init):
            self.queue.append(self._sample_message())

        return self._get_obs(), {}

    def step(self, action: int):
        self.t += 1
        reward = 0.0
        served_msg = None

        valid_action = 0 <= action < len(self.queue) and action < self.window_size
        if valid_action:
            served_msg = self.queue.pop(action)
        else:
            reward -= 0.2  # penalized no-op on an empty/out-of-range slot

        if served_msg is not None:
            # (a) reward serving high-urgency messages
            reward += 2.0 * served_msg["urgency"]

            # (b) penalize serving duplicate messages
            if served_msg["is_duplicate"]:
                reward -= 1.0

            # (c) bonus for serving the globally highest-urgency message
            #     across the ENTIRE queue (not merely the visible window)
            urgencies_before_pop = [served_msg["urgency"]] + [m["urgency"] for m in self.queue]
            if served_msg["urgency"] >= max(urgencies_before_pop) - 1e-9:
                reward += 1.0

            self.episode_served += 1
            self.episode_urgency_served.append(served_msg["urgency"])

        # (d) penalize letting messages wait, accrued across the whole
        #     remaining queue every step
        reward -= 0.02 * len(self.queue)

        # Stochastic arrivals
        n_new = int(self._rng.poisson(self.arrival_rate))
        for _ in range(n_new):
            if len(self.queue) < self.max_queue_size:
                self.queue.append(self._sample_message())

        terminated = False
        truncated = self.t >= self.max_episode_steps
        obs = self._get_obs()
        info = {
            "served": served_msg,
            "queue_length": len(self.queue),
            "episode_served": self.episode_served,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
