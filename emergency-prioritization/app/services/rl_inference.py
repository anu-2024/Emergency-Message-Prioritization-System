"""
Converts live application messages into the observation vector format
MessagePrioritizationEnv._get_obs() produces, so the trained SB3 model can
score them. Kept separate from rl/environment.py (which must stay a clean
Gymnasium env with no app/DB dependencies) and from priority_service.py
(orchestration only).
"""
import numpy as np

from rl.environment import CATEGORY_TO_IDX, N_CATEGORIES, FEATURES_PER_SLOT, GLOBAL_FEATURES
from app.services.priority_service import _waiting_time_minutes


def _message_to_slot_vector(message) -> np.ndarray:
    one_hot = np.zeros(N_CATEGORIES, dtype=np.float32)
    idx = CATEGORY_TO_IDX.get(message.category, None)
    if idx is not None:
        one_hot[idx] = 1.0
    waiting_norm = min(_waiting_time_minutes(message) / 20.0, 1.0)
    assistance_count = 0
    try:
        import json
        assistance_count = len(json.loads(message.assistance_types or "[]"))
    except Exception:  # noqa: BLE001
        pass
    return np.concatenate([
        one_hot,
        [message.urgency_score or 0.0, message.category_confidence or 0.5,
         waiting_norm, float(bool(message.is_duplicate)),
         min(assistance_count / 3.0, 1.0)],
    ]).astype(np.float32)


def build_observation(target_message, queue_context: list, window_size: int = 5) -> np.ndarray:
    """Builds a window with `target_message` in slot 0 and up to
    window_size-1 other pending messages from queue_context, matching the
    training-time observation layout."""
    others = [m for m in queue_context if m.message_id != target_message.message_id]
    window = [target_message] + others[: window_size - 1]

    slots = [_message_to_slot_vector(m) for m in window]
    while len(slots) < window_size:
        slots.append(np.zeros(FEATURES_PER_SLOT, dtype=np.float32))

    queue_len_norm = min(len(queue_context) / 30.0, 1.0)
    capacity_used = 0.5  # unknown at inference time; neutral placeholder
    return np.concatenate(slots + [[queue_len_norm, capacity_used]]).astype(np.float32)


def predict_priority_score(model, target_message, queue_context: list) -> float:
    """Returns a [0,1] priority score derived from the trained DQN's
    Q-values for serving `target_message` first (slot 0) vs not."""
    obs = build_observation(target_message, queue_context)
    # SB3 DQN exposes q_net for Q-value inspection; predict() alone only
    # gives the discrete action. We want a continuous score for ranking,
    # so read the Q-value of "serve slot 0" (this message) and squash it.
    import torch
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        q_values = model.q_net(obs_tensor).squeeze(0).numpy()
    q_for_this_message = float(q_values[0])
    # Squash Q-value into [0,1] via sigmoid for a comparable priority scale.
    score = 1.0 / (1.0 + np.exp(-q_for_this_message))
    return round(float(score), 4)
