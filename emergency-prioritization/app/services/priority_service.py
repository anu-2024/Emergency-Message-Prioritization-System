"""
Priority scoring service.

Wires together rl/baseline.py (always available, always computed) and the
trained RL agent (used when available) to produce the final ranked queue.
Per the SRS's explicit human-in-the-loop requirement, this service NEVER
autonomously dispatches anything -- it only computes a suggested ranking
that a human responder reviews and can override at any time
(see OverrideRequest / responder_action_service.py).
"""
import logging
from typing import Optional

from rl.baseline import rule_based_priority

logger = logging.getLogger(__name__)

_rl_model = None
_rl_load_attempted = False


def _try_load_rl_model():
    """Lazily attempts to load the trained RL agent. If unavailable
    (not trained yet, or SB3/torch not installed), falls back to
    rule-based-only silently but logs a clear warning once."""
    global _rl_model, _rl_load_attempted
    if _rl_load_attempted:
        return _rl_model
    _rl_load_attempted = True
    try:
        from rl.train import load_trained_agent
        _rl_model = load_trained_agent()
        logger.info("Loaded trained RL agent for prioritization.")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "RL agent unavailable (%s). Falling back to rule-based baseline only. "
            "This is expected until you run `python -m rl.train` locally.", e
        )
        _rl_model = None
    return _rl_model


def compute_rule_based_priority(message) -> float:
    return rule_based_priority(
        urgency_score=message.urgency_score or 0.0,
        category=message.category or "Other/Irrelevant",
        waiting_time=_waiting_time_minutes(message),
        category_confidence=message.category_confidence or 0.5,
        is_duplicate=bool(message.is_duplicate),
    )


def _waiting_time_minutes(message) -> int:
    from datetime import datetime, timezone
    if message.received_at is None:
        return 0
    received = message.received_at
    if received.tzinfo is None:
        received = received.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - received
    return int(delta.total_seconds() // 60)


def compute_rl_priority(message, queue_context: Optional[list] = None) -> Optional[float]:
    """Returns an RL-derived priority score in [0,1], or None if the RL
    model isn't available (caller should fall back to rule-based)."""
    model = _try_load_rl_model()
    if model is None:
        return None
    # NOTE: converting a live message + queue context into the exact
    # observation vector the trained model expects (rl/environment.py's
    # _get_obs layout) requires assembling a window the same way the env
    # does. Implemented in app/services/rl_inference.py to keep this file
    # focused on orchestration.
    from app.services.rl_inference import predict_priority_score
    try:
        return predict_priority_score(model, message, queue_context or [])
    except Exception as e:  # noqa: BLE001
        logger.error("RL inference failed, falling back to rule-based: %s", e)
        return None


def compute_final_priority(message, queue_context: Optional[list] = None) -> dict:
    """Always computes rule-based (source of truth fallback). Attempts RL
    on top; if available, RL becomes the *displayed* priority but the
    rule-based score is always stored too so responders can compare them
    (SRS: 'Compare RL decisions with the baseline')."""
    rule_score = compute_rule_based_priority(message)
    rl_score = compute_rl_priority(message, queue_context)

    if message.human_override_priority is not None:
        final_source = "human_override"
        final_score = message.human_override_priority
    elif rl_score is not None:
        final_source = "rl"
        final_score = rl_score
    else:
        final_source = "rule_based"
        final_score = rule_score

    return {
        "rule_based_priority": rule_score,
        "rl_priority": rl_score,
        "final_priority_source": final_source,
        "final_priority_score": final_score,
    }
