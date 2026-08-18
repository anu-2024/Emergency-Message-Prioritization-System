"""
Reproducible evaluation procedure (SRS 6.8).

Evaluates random, rule-based, and RL policies on identical seeded episodes,
reporting mean episode reward, mean messages served, and mean average
urgency of served messages. Results are persisted (models/eval_results.json)
for display on the dashboard.
"""
from __future__ import annotations

import json
import numpy as np

from src.rl.environment import MessagePrioritizationEnv
from src.rl.baseline import RandomPolicy, RuleBasedPolicy
from src.config import EVAL_RESULTS_PATH


def run_episode(env: MessagePrioritizationEnv, policy_fn=None, sb3_model=None, seed=None):
    """Run a single episode with either a callable `policy_fn(env) -> action`
    (random / rule-based) or a trained Stable-Baselines3 `sb3_model`."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        if sb3_model is not None:
            action, _ = sb3_model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = policy_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    mean_urgency_served = (
        float(np.mean(env.episode_urgency_served)) if env.episode_urgency_served else 0.0
    )
    return {
        "episode_reward": total_reward,
        "messages_served": env.episode_served,
        "mean_urgency_served": mean_urgency_served,
    }


def evaluate_policy(policy_name: str, n_episodes: int = 30, base_seed: int = 1000,
                     policy_fn=None, sb3_model=None, env_kwargs: dict | None = None):
    env_kwargs = env_kwargs or {}
    results = []
    for i in range(n_episodes):
        env = MessagePrioritizationEnv(**env_kwargs)
        res = run_episode(env, policy_fn=policy_fn, sb3_model=sb3_model, seed=base_seed + i)
        results.append(res)

    summary = {
        "policy": policy_name,
        "n_episodes": n_episodes,
        "mean_episode_reward": float(np.mean([r["episode_reward"] for r in results])),
        "std_episode_reward": float(np.std([r["episode_reward"] for r in results])),
        "mean_messages_served": float(np.mean([r["messages_served"] for r in results])),
        "mean_urgency_served": float(np.mean([r["mean_urgency_served"] for r in results])),
    }
    return summary, results


def evaluate_all(sb3_model=None, n_episodes: int = 30, base_seed: int = 1000,
                  env_kwargs: dict | None = None, save: bool = True) -> dict:
    """Evaluate Random, Rule-based, and (if provided) RL policies on
    identical seeded episodes and optionally persist the comparison."""
    summaries = {}

    random_summary, _ = evaluate_policy(
        "Random", n_episodes=n_episodes, base_seed=base_seed,
        policy_fn=RandomPolicy(), env_kwargs=env_kwargs,
    )
    summaries["Random"] = random_summary

    rule_summary, _ = evaluate_policy(
        "Rule-Based", n_episodes=n_episodes, base_seed=base_seed,
        policy_fn=RuleBasedPolicy(), env_kwargs=env_kwargs,
    )
    summaries["Rule-Based"] = rule_summary

    if sb3_model is not None:
        rl_summary, _ = evaluate_policy(
            "RL (DQN)", n_episodes=n_episodes, base_seed=base_seed,
            sb3_model=sb3_model, env_kwargs=env_kwargs,
        )
        summaries["RL (DQN)"] = rl_summary

    if save:
        with open(EVAL_RESULTS_PATH, "w") as f:
            json.dump(summaries, f, indent=2)

    return summaries


def load_eval_results() -> dict | None:
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH) as f:
            return json.load(f)
    return None
