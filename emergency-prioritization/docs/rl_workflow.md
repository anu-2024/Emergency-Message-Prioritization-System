# Reinforcement Learning Workflow

## Problem formulation

A responder has limited attention/capacity. At each timestep, a small
window of pending messages (default 5) is visible. The agent chooses
**one** message from that window to serve next. New messages arrive
stochastically, simulating a live emergency stream. This is the
SRS §11.2-recommended formulation: a discrete-action problem over a
bounded candidate set, not free-ranking of an unbounded queue — chosen
specifically because it is reproducible, explainable, and trains fast on
CPU, while still being genuine sequential decision-making under
uncertainty (the core requirement for RL, as opposed to one-shot ranking).

## State (`rl/environment.py: _get_obs`)

Per visible slot (5 slots × 13 features = 65 values), concatenated:
- Category one-hot (8 dims)
- `urgency_score` (0–1, from the trained urgency classifier)
- `category_confidence` (0–1)
- Normalized waiting time (`min(waiting_time/20, 1)`)
- `is_duplicate` flag (0/1)
- Normalized assistance-type count

Plus 2 global features: normalized queue length, normalized episode
progress. Total observation dimension: **67**.

## Action

`Discrete(5)` — index of the window slot to serve. If a slot is empty
(fewer than 5 messages currently pending), selecting it is a no-op
with a penalty (discourages the agent from "wasting" a turn).

## Reward (`rl/environment.py: step`)

```
+ urgency_score * 2.0                    for serving a message
- 0.5                                     if the served message was a duplicate
+ 1.0                                     bonus if served message had the
                                           GLOBAL highest urgency in the
                                           entire queue (not just the window)
- 0.3                                     if the action selected an empty slot
- 0.01 * min(waiting_time, 20)            accrued for EVERY message still
                                           waiting, each step
```

The waiting-time penalty applies to the whole remaining queue, not just
the served message — this is what prevents a degenerate policy that only
ever grabs the single highest-urgency item in the window while letting
everything else rot; it must also clear the queue.

## Why this reward design, and its limits

- The `+1.0` global-max bonus specifically rewards genuinely correct
  prioritization (matching the *true* most-urgent message queue-wide),
  not just being locally greedy within the visible window — this is the
  reward term most directly testing whether the agent learns something a
  rule-based window-local heuristic could not.
- **Limitation, disclosed**: reward weights (2.0, 0.5, 1.0, 0.3, 0.01) are
  hand-tuned, not learned or derived from a formal utility model of real
  triage priorities. A production system would need domain-expert input
  (e.g. actual disaster-response protocols) to calibrate these. This is
  flagged in `docs/limitations.md`.

## Policy and training (`rl/train.py`)

- **Algorithm**: DQN (Deep Q-Network) via Stable-Baselines3,
  `MlpPolicy` with a small `[64, 64]` hidden-layer network — deliberately
  lightweight for CPU training (SRS's no-GPU constraint).
- Hyperparameters: learning rate 1e-3, replay buffer 20,000, batch size
  64, gamma 0.95, epsilon-greedy exploration decaying over the first 30%
  of training.
- `EvalCallback` checkpoints the best model by held-out evaluation reward
  during training.

## Evaluation

`evaluation/compare_policies.py` runs identical seeded episodes (same
message-arrival sequences) under: random policy (lower bound), rule-based
policy (`rl/baseline.py`), and the trained RL agent — reporting mean
episode reward, messages served, and average urgency of served messages.
See `docs/evaluation_methodology.md` for the full protocol and current
results.

## What was and wasn't tested in the build sandbox

The environment's reward-shaping logic was validated with a hand-written
Gymnasium API shim (no internet access to `pip install gymnasium`): a
greedy urgency-maximizing policy scored **64.7** mean reward over 50 steps
versus **26.4** for a random policy — confirming the reward function does
reward correct behavior before any learning algorithm is even involved.
The actual DQN training loop (`rl/train.py`) requires `torch` and
`stable-baselines3`, which could not be installed in that sandbox, and so
has **not** been executed end-to-end. Run it yourself and check the
printed `mean_reward` from `evaluate_policy` before trusting the saved
model for your demo.
