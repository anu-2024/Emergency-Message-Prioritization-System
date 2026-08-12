"""
Stage 8b — Rule-based baseline prioritization.

Per SRS Appendix B: a transparent, deterministic fallback that combines
urgency, category criticality, message age and NLP confidence into a
normalized score. Used both as:
  1. A live fallback if the RL model is unavailable (see app/services).
  2. The comparison baseline for Stage 9 evaluation (Rule-based vs RL).

This is DELIBERATELY simple and inspectable -- it is not RL, does not
learn, and is never mislabeled as RL anywhere in this codebase.
"""
from dataclasses import dataclass

# Static, documented category criticality weights (0-1). These reflect the
# rough relative severity of the category itself if urgency were unknown --
# they nudge the score but urgency still dominates.
CATEGORY_CRITICALITY = {
    "Medical": 1.0,
    "Fire": 1.0,
    "Flood/Rescue": 0.9,
    "Water": 0.6,
    "Shelter": 0.6,
    "Infrastructure": 0.5,
    "Food": 0.5,
    "Other/Irrelevant": 0.1,
}

# Weights for combining the four signals into one priority score.
W_URGENCY = 0.5
W_CATEGORY = 0.2
W_AGE = 0.2
W_CONFIDENCE = 0.1

MAX_AGE_FOR_NORMALIZATION = 20  # steps/minutes after which age contributes its max


def rule_based_priority(urgency_score: float, category: str,
                         waiting_time: int, category_confidence: float,
                         is_duplicate: bool = False) -> float:
    """Returns a priority score in [0, 1]. Higher = serve sooner.

    score = 0.5*urgency + 0.2*category_criticality + 0.2*normalized_age + 0.1*confidence
    Duplicates are penalized by halving the final score (still visible,
    never silently dropped -- a human can still review it).
    """
    category_weight = CATEGORY_CRITICALITY.get(category, 0.5)
    age_norm = min(waiting_time / MAX_AGE_FOR_NORMALIZATION, 1.0)

    score = (
        W_URGENCY * urgency_score
        + W_CATEGORY * category_weight
        + W_AGE * age_norm
        + W_CONFIDENCE * category_confidence
    )
    if is_duplicate:
        score *= 0.5
    return round(min(max(score, 0.0), 1.0), 4)


def rank_queue(messages: list) -> list:
    """messages: list of dicts with keys urgency_score, category,
    waiting_time, category_confidence, is_duplicate, message_id.
    Returns the same list sorted by descending rule-based priority, with
    the computed score attached as 'rule_based_priority'."""
    scored = []
    for m in messages:
        score = rule_based_priority(
            urgency_score=m["urgency_score"],
            category=m["category"],
            waiting_time=m.get("waiting_time", 0),
            category_confidence=m.get("category_confidence", 0.5),
            is_duplicate=m.get("is_duplicate", False),
        )
        scored.append({**m, "rule_based_priority": score})
    return sorted(scored, key=lambda x: x["rule_based_priority"], reverse=True)
