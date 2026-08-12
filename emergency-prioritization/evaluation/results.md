# Baseline vs RL Evaluation Report

Same seeded episodes (identical message arrival sequences) were run under each policy so the comparison is apples-to-apples.

| Policy | Mean Reward | Mean Msgs Served | Mean Avg Served Urgency |
|---|---|---|---|
| Random (lower bound) | 60.75 +/- 7.41 | 61.4 | 0.539 |
| Rule-based baseline | 116.81 +/- 10.87 | 64.1 | 0.540 |
| RL agent (DQN) | *not evaluated (model not trained)* | - | - |

**Interpretation guide for the report/viva:** the RL agent should beat the random baseline convincingly and should match or exceed the rule-based baseline. If it does not exceed rule-based, that is a legitimate, reportable finding for an MCA project -- it shows RL does not automatically beat a well-designed heuristic, and motivates further reward-shaping or training-time tuning discussed in the Limitations section.