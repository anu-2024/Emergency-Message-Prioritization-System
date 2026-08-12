# Evaluation Methodology

## Protocol

`evaluation/compare_policies.py` runs three policies over **identical
seeded episodes** (same message-arrival sequence, same starting queue) of
`MessagePrioritizationEnv`:

1. **Random** — uniformly samples an action each step. Lower-bound sanity
   check: if a policy can't beat this, something is broken.
2. **Rule-based** — applies `rl/baseline.py`'s transparent scoring
   formula to the visible window each step, picks the highest-scoring
   slot.
3. **RL agent** — the trained DQN's deterministic action.

Using the same seeds for all three means differences in outcome are
attributable to the *policy*, not to random variation in which messages
happened to arrive.

## Metrics reported

- **Mean episode reward** (± std over N episodes) — the primary metric,
  directly reflects the reward function's definition of "good
  prioritization" (see `docs/rl_workflow.md`).
- **Mean messages served per episode** — throughput / queue-clearing
  ability.
- **Mean average urgency of served messages** — a policy that only ever
  serves low-urgency messages would score poorly here even if it clears
  many messages quickly.

## Current results (from this build's sandbox — random vs. rule-based only; RL agent requires local training, see below)

| Policy | Mean Reward | Mean Msgs Served | Mean Avg Served Urgency |
|---|---|---|---|
| Random (lower bound) | 60.75 +/- 7.41 | 61.4 | 0.539 |
| Rule-based baseline | 116.81 +/- 10.87 | 64.1 | 0.540 |
| RL agent (DQN) | not yet evaluated -- train locally | - | - |

Rule-based nearly doubles random's mean reward, which is expected (it's a
deliberately well-designed heuristic) and confirms the reward function
behaves sensibly before layering RL on top.

## Running your own comparison

```bash
python -m rl.train --timesteps 50000
python -m evaluation.compare_policies --episodes 30
cat evaluation/results.md
```

## Interpreting the RL vs. rule-based result -- either outcome is valid to report

- **If RL beats rule-based**: report the margin, and discuss why -- most
  likely because the +1.0 global-max bonus rewards genuinely
  queue-wide-optimal choices that a window-local rule-based heuristic
  structurally cannot see (the rule-based policy only ever scores the 5
  visible messages, never compares against messages outside the window).
- **If RL does not beat rule-based**: this is a legitimate, common, and
  reportable finding for an MCA project -- it does not mean the RL
  implementation is broken. Possible explanations to discuss: insufficient
  training timesteps, reward-shaping needing further tuning, or the task
  genuinely being simple enough that a well-designed heuristic is close to
  optimal. Increasing --timesteps, tuning gamma/learning_rate in
  rl/train.py, or widening window_size are documented next steps in
  docs/future_enhancements.md.

## Threats to validity (disclose in your report)

- Training and evaluation both happen in the same synthetic simulated
  environment (rl/environment.py's _default_generator) -- this is
  standard practice for RL prototyping but means results describe
  performance within this simulation, not a guarantee of real-world
  performance on live message streams with different arrival
  distributions.
- The urgency scores fed into the RL environment during training come
  from the same weak-supervision-trained urgency classifier described in
  docs/nlp_workflow.md -- errors in that classifier propagate into what
  the RL agent is optimizing for.
