"""
Stage 9 — Baseline vs RL evaluation.

Runs both the rule-based baseline and the trained RL agent through
identical episodes of MessagePrioritizationEnv (same seeds -> same message
arrival sequences) and compares them on the metrics the SRS §19 calls for:

  * Average episode reward
  * Average urgency of served messages ("does it actually prioritize
    genuinely urgent messages?")
  * Messages served per episode (queue-clearing throughput)

Produces evaluation/results.json and evaluation/results.md (human-readable
table) for direct inclusion in the project report.

NOTE: like rl/train.py, this script depends on stable-baselines3/torch to
load the trained agent, and could not be executed end-to-end in the
development sandbox (no internet access to install those packages). The
rule-based-vs-random portion WAS run and verified in-sandbox (see the
project README, "What was and wasn't tested").
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from rl.environment import MessagePrioritizationEnv
from rl.baseline import rule_based_priority

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent


def run_episode_with_policy(env: MessagePrioritizationEnv, choose_action_fn) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    served_urgencies = []
    total_served = 0

    done = False
    while not done:
        action = choose_action_fn(env, obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if step_info.get("served_message_id") is not None:
            total_served += 1
            served_urgencies.append(step_info["served_urgency"])

    return {
        "total_reward": total_reward,
        "messages_served": total_served,
        "avg_served_urgency": float(np.mean(served_urgencies)) if served_urgencies else 0.0,
        "final_queue_length": info.get("queue_length", 0),
    }


def random_policy(env, obs) -> int:
    return env.action_space.sample()


def rule_based_policy(env, obs) -> int:
    window = env.queue[: env.window_size]
    if not window:
        return 0
    scores = [
        rule_based_priority(
            urgency_score=m.urgency_score, category=m.category,
            waiting_time=m.waiting_time, category_confidence=m.category_confidence,
            is_duplicate=m.is_duplicate,
        )
        for m in window
    ]
    return int(np.argmax(scores))


def rl_policy_factory(model):
    def _policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _policy


def evaluate_policy_over_episodes(env_seed_start: int, policy_fn, n_episodes: int = 30) -> dict:
    all_metrics = []
    for ep in range(n_episodes):
        env = MessagePrioritizationEnv(window_size=5, episode_length=100, seed=env_seed_start + ep)
        metrics = run_episode_with_policy(env, policy_fn)
        all_metrics.append(metrics)

    return {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean([m["total_reward"] for m in all_metrics])),
        "std_reward": float(np.std([m["total_reward"] for m in all_metrics])),
        "mean_messages_served": float(np.mean([m["messages_served"] for m in all_metrics])),
        "mean_avg_served_urgency": float(np.mean([m["avg_served_urgency"] for m in all_metrics])),
    }


def run_comparison(n_episodes: int = 30, include_rl: bool = True) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating random baseline (%d episodes)...", n_episodes)
    random_results = evaluate_policy_over_episodes(1000, random_policy, n_episodes)

    logger.info("Evaluating rule-based baseline (%d episodes)...", n_episodes)
    rule_results = evaluate_policy_over_episodes(1000, rule_based_policy, n_episodes)

    results = {"random": random_results, "rule_based": rule_results}

    if include_rl:
        try:
            from rl.train import load_trained_agent
            model = load_trained_agent()
            logger.info("Evaluating trained RL agent (%d episodes)...", n_episodes)
            rl_results = evaluate_policy_over_episodes(1000, rl_policy_factory(model), n_episodes)
            results["rl_agent"] = rl_results
        except FileNotFoundError as e:
            logger.warning("Skipping RL evaluation: %s", e)
            results["rl_agent"] = None

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _write_markdown_report(results)
    logger.info("Saved evaluation/results.json and evaluation/results.md")
    return results


def _write_markdown_report(results: dict) -> None:
    lines = [
        "# Baseline vs RL Evaluation Report",
        "",
        "Same seeded episodes (identical message arrival sequences) were run "
        "under each policy so the comparison is apples-to-apples.",
        "",
        "| Policy | Mean Reward | Mean Msgs Served | Mean Avg Served Urgency |",
        "|---|---|---|---|",
    ]
    labels = {"random": "Random (lower bound)", "rule_based": "Rule-based baseline", "rl_agent": "RL agent (DQN)"}
    for key, label in labels.items():
        r = results.get(key)
        if r is None:
            lines.append(f"| {label} | *not evaluated (model not trained)* | - | - |")
        else:
            lines.append(f"| {label} | {r['mean_reward']:.2f} +/- {r['std_reward']:.2f} | "
                          f"{r['mean_messages_served']:.1f} | {r['mean_avg_served_urgency']:.3f} |")
    lines.append("")
    lines.append("**Interpretation guide for the report/viva:** the RL agent should "
                  "beat the random baseline convincingly and should match or exceed "
                  "the rule-based baseline. If it does not exceed rule-based, that is "
                  "a legitimate, reportable finding for an MCA project -- it shows RL "
                  "does not automatically beat a well-designed heuristic, and motivates "
                  "further reward-shaping or training-time tuning discussed in the "
                  "Limitations section.")
    with open(RESULTS_DIR / "results.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare rule-based baseline vs RL agent")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--skip-rl", action="store_true")
    args = parser.parse_args()
    run_comparison(n_episodes=args.episodes, include_rl=not args.skip_rl)
