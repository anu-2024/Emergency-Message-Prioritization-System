"""
Dashboard RL Environment — lightweight Gymnasium environment for the
Streamlit dashboard's real-time RL demonstration.

The RL agent observes the current priority queue (top WINDOW messages)
and learns to select which message to serve next, aiming to maximize
urgency-weighted service while avoiding duplicates.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

N_CATEGORIES = 8


class DashboardEnv(gym.Env):
    """Streamlit-friendly RL environment for message triage."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        window_size: int = 5,
        max_steps: int = 60,
        seed: int | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.max_steps = max_steps
        self.max_episode_steps = max_steps

        obs_dim = self.window_size * (N_CATEGORIES + 4) + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.window_size)

        self._rng = np.random.default_rng(seed)
        self.queue: list[dict] = []
        self.t = 0

        self.episode_served = 0
        self.episode_rewards: list[float] = []
        self.episode_urgencies: list[float] = []

    def _make_msg(self) -> dict:
        cat = int(self._rng.integers(0, N_CATEGORIES))
        urg = float(np.clip(self._rng.beta(2.0, 2.2), 0.0, 1.0))
        conf = float(np.clip(self._rng.normal(0.85, 0.08), 0.35, 1.0))
        dup = bool(self._rng.random() < 0.08)
        msg = {
            "category": cat, "urgency": urg, "confidence": conf,
            "is_duplicate": dup, "arrival_step": self.t,
        }
        return msg

    def _obs(self) -> np.ndarray:
        feats: list[float] = []
        for i in range(self.window_size):
            if i < len(self.queue):
                m = self.queue[i]
                onehot = [0.0] * N_CATEGORIES
                onehot[m["category"]] = 1.0
                wait = min(1.0, (self.t - m["arrival_step"]) / max(1, self.max_steps))
                feats.extend(onehot)
                feats.extend([m["urgency"], m["confidence"], wait, 1.0 if m["is_duplicate"] else 0.0])
            else:
                feats.extend([0.0] * (N_CATEGORIES + 4))
        feats.extend([min(1.0, len(self.queue) / 40.0), min(1.0, self.t / self.max_steps)])
        return np.asarray(feats, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.queue = []
        self.t = 0
        self.episode_served = 0
        self.episode_rewards = []
        self.episode_urgencies = []
        n = int(self._rng.integers(3, self.window_size + 3))
        for _ in range(n):
            self.queue.append(self._make_msg())
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        reward = 0.0
        served = None

        valid = 0 <= action < len(self.queue) and action < self.window_size
        if valid:
            served = self.queue.pop(action)
        else:
            reward -= 0.2

        if served is not None:
            reward += 2.0 * served["urgency"]
            if served["is_duplicate"]:
                reward -= 1.0
            all_urg = [served["urgency"]] + [m["urgency"] for m in self.queue]
            if served["urgency"] >= max(all_urg) - 1e-9:
                reward += 1.0
            self.episode_served += 1
            self.episode_urgencies.append(served["urgency"])

        reward -= 0.02 * len(self.queue)
        self.episode_rewards.append(reward)

        n_new = int(self._rng.poisson(1.3))
        for _ in range(n_new):
            if len(self.queue) < 40:
                self.queue.append(self._make_msg())

        terminated = False
        truncated = self.t >= self.max_steps
        info = {
            "served_category": served["category"] if served else None,
            "served_urgency": served["urgency"] if served else None,
            "queue_length": len(self.queue),
        }
        return self._obs(), reward, terminated, truncated, info
