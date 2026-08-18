"""
Transparent, deterministic rule-based prioritization baseline (SRS 6.7).

Combines urgency, category criticality, message age and classification
confidence into a single explainable score. Usable standalone as a fallback
when no trained RL agent is available (SRS 2.1), and as the comparison
baseline for RL evaluation (SRS 6.8).
"""
from __future__ import annotations

import numpy as np

from src.config import CATEGORIES, CATEGORY_CRITICALITY

# Explicit, documented weights — deliberately simple and explainable per the
# academic explainability requirement (SRS 2.4).
_W_URGENCY = 0.50
_W_CRITICALITY = 0.25
_W_AGE = 0.15
_W_CONFIDENCE = 0.10


def rule_based_priority(urgency: float, category: str | int, age_normalized: float,
                         confidence: float) -> float:
    """Compute an explainable priority score in roughly [0, 1].

    Args:
        urgency: urgency score in [0, 1].
        category: category name (str) or category index (int).
        age_normalized: how long the message has waited, normalized to [0, 1].
        confidence: classifier confidence in [0, 1].
    """
    if isinstance(category, int):
        category = CATEGORIES[category]
    criticality = CATEGORY_CRITICALITY.get(category, 0.3)

    score = (
        _W_URGENCY * urgency
        + _W_CRITICALITY * criticality
        + _W_AGE * age_normalized
        + _W_CONFIDENCE * confidence
    )
    return float(np.clip(score, 0.0, 1.0))


class RuleBasedPolicy:
    """Selects the visible-window slot with the highest rule-based score.

    Operates directly on a `MessagePrioritizationEnv`-style queue so it can
    be evaluated head-to-head against the RL policy inside evaluate.py.
    """

    def __call__(self, env) -> int:
        window = env.queue[: env.window_size]
        if not window:
            return 0
        best_idx, best_score = 0, -1.0
        for i, m in enumerate(window):
            age_norm = min(1.0, (env.t - m["arrival_step"]) / max(1, env.max_episode_steps))
            score = rule_based_priority(m["urgency"], m["category"], age_norm, m["confidence"])
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx


class RandomPolicy:
    """Uniformly random valid-slot selection — the lower-bound comparator."""

    def __call__(self, env) -> int:
        n_visible = min(env.window_size, len(env.queue))
        if n_visible == 0:
            return 0
        return int(np.random.randint(0, n_visible))
