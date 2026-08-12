"""
Stage 7 — Reinforcement Learning environment.

Formulation (documented for the SRS §11.1 / viva):

  PROBLEM: A responder has limited attention/capacity. At each timestep,
  a small candidate set of pending messages is visible (the "window").
  The agent must choose ONE message from the window to dispatch/respond to
  next. This is deliberately the SRS §11.2-recommended formulation:
  discrete-action, small candidate set, reproducible on CPU — not
  free-ranking of an unbounded queue.

  STATE (per candidate slot, concatenated into one vector):
    - category_id (one-hot, 8 categories)
    - urgency_score (0-1, from nlp/urgency.py)
    - category_confidence (0-1)
    - waiting_time (normalized: how many steps this message has waited)
    - is_duplicate (0/1)
    - assistance_type_count (normalized count of extracted assistance types)
  Plus 2 global state features: queue_length (normalized), capacity_used.

  ACTION: Discrete(WINDOW_SIZE) — index of the candidate slot to serve
  next. If a slot is empty (fewer than WINDOW_SIZE messages pending),
  selecting it is a no-op with a small penalty (discourages wasting turns).

  REWARD: (SRS §11.3) positive for surfacing genuinely high-urgency
  messages early, negative for delaying high-urgency messages or
  repeatedly picking low-value/duplicate items. Concretely:
    +urgency_score * 2                          for serving a message
    -0.5                                         if the served message is a duplicate
    -0.01 * waiting_time                         accumulated idle penalty per step for EVERY
                                                   message still waiting (encourages clearing
                                                   the queue, not just cherry-picking)
    -0.3                                         if action selects an empty slot (no-op)
    +1.0 bonus                                   if the served message was the single highest
                                                   urgency_score item in the whole queue
                                                   (not just the visible window) -> rewards
                                                   genuinely correct prioritization

  EPISODE: runs for `episode_length` steps or until the queue is empty and
  no new messages arrive within `max_idle_steps`. New messages arrive
  stochastically (Poisson-like) to simulate a live emergency stream.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

CATEGORIES = ["Medical", "Flood/Rescue", "Fire", "Food", "Water",
              "Shelter", "Infrastructure", "Other/Irrelevant"]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
N_CATEGORIES = len(CATEGORIES)

FEATURES_PER_SLOT = N_CATEGORIES + 5  # one-hot category + 5 scalar features
GLOBAL_FEATURES = 2


@dataclass
class QueuedMessage:
    message_id: str
    category: str
    urgency_score: float
    category_confidence: float
    is_duplicate: bool
    assistance_type_count: int
    waiting_time: int = 0


class MessagePrioritizationEnv(gym.Env):
    """Gymnasium environment: agent selects which pending message to serve
    next from a fixed-size visible window, aiming to serve high-urgency
    messages early under limited responder capacity."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, window_size: int = 5, episode_length: int = 100,
                 arrival_rate: float = 0.6, seed: int = None,
                 message_generator=None):
        super().__init__()
        self.window_size = window_size
        self.episode_length = episode_length
        self.arrival_rate = arrival_rate
        self._rng = np.random.default_rng(seed)
        # message_generator: callable() -> QueuedMessage, injectable for
        # testing / for feeding real NLP-analyzed messages instead of
        # synthetic ones. Defaults to a built-in synthetic generator.
        self.message_generator = message_generator or self._default_generator

        obs_dim = self.window_size * FEATURES_PER_SLOT + GLOBAL_FEATURES
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.window_size)

        self.queue: list = []
        self._step_count = 0
        self._msg_counter = 0

    # ------------------------------------------------------------------
    # Message generation (synthetic, for training). Swappable via
    # message_generator for evaluation against real NLP-analyzed messages.
    # ------------------------------------------------------------------
    def _default_generator(self) -> QueuedMessage:
        category = self._rng.choice(CATEGORIES)
        # Correlate urgency loosely with category, mirroring nlp/synthetic_data.py's
        # weighting, so the RL environment's difficulty matches the NLP layer's output.
        base = {"Medical": 0.75, "Flood/Rescue": 0.7, "Fire": 0.8, "Food": 0.45,
                "Water": 0.5, "Shelter": 0.5, "Infrastructure": 0.5,
                "Other/Irrelevant": 0.1}[category]
        urgency_score = float(np.clip(self._rng.normal(base, 0.15), 0.0, 1.0))
        self._msg_counter += 1
        return QueuedMessage(
            message_id=f"SIM-{self._msg_counter}",
            category=category,
            urgency_score=urgency_score,
            category_confidence=float(self._rng.uniform(0.5, 0.99)),
            is_duplicate=bool(self._rng.random() < 0.08),
            assistance_type_count=int(self._rng.integers(0, 3)),
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.queue = []
        self._step_count = 0
        # seed the queue with a few messages so step 0 isn't empty
        for _ in range(self.window_size):
            self.queue.append(self.message_generator())
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        window = self.queue[: self.window_size]
        slots = []
        for i in range(self.window_size):
            if i < len(window):
                m = window[i]
                one_hot = np.zeros(N_CATEGORIES, dtype=np.float32)
                one_hot[CATEGORY_TO_IDX[m.category]] = 1.0
                waiting_norm = min(m.waiting_time / 20.0, 1.0)
                slot = np.concatenate([
                    one_hot,
                    [m.urgency_score, m.category_confidence, waiting_norm,
                     float(m.is_duplicate), min(m.assistance_type_count / 3.0, 1.0)],
                ])
            else:
                slot = np.zeros(FEATURES_PER_SLOT, dtype=np.float32)
            slots.append(slot)

        queue_len_norm = min(len(self.queue) / 30.0, 1.0)
        capacity_used = min(self._step_count / max(self.episode_length, 1), 1.0)
        obs = np.concatenate(slots + [[queue_len_norm, capacity_used]]).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        return {
            "queue_length": len(self.queue),
            "step": self._step_count,
            "max_urgency_in_queue": max((m.urgency_score for m in self.queue), default=0.0),
        }

    def step(self, action: int):
        reward = 0.0
        window = self.queue[: self.window_size]

        if action >= len(window):
            # Agent selected an empty slot -> no-op, small penalty.
            reward -= 0.3
            served = None
        else:
            served = window[action]
            self.queue.remove(served)
            reward += served.urgency_score * 2.0
            if served.is_duplicate:
                reward -= 0.5
            # Bonus for genuinely picking the highest-urgency item in the
            # FULL queue (not just the visible window) -- rewards correct
            # global prioritization, not just window-local greediness.
            max_urgency = max((m.urgency_score for m in ([served] + self.queue)), default=served.urgency_score)
            if served.urgency_score >= max_urgency - 1e-6:
                reward += 1.0

        # Idle penalty accrues on every message still waiting, and their
        # waiting_time increments -- encourages clearing the queue overall.
        for m in self.queue:
            m.waiting_time += 1
            reward -= 0.01 * min(m.waiting_time, 20)

        # Stochastic new arrivals (simulated live emergency stream).
        if self._rng.random() < self.arrival_rate:
            self.queue.append(self.message_generator())

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.episode_length

        obs = self._get_obs()
        info = self._get_info()
        info["served_message_id"] = served.message_id if served else None
        info["served_urgency"] = served.urgency_score if served else None
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step {self._step_count} | Queue length: {len(self.queue)} | "
              f"Top urgencies: {sorted([round(m.urgency_score, 2) for m in self.queue], reverse=True)[:5]}")
