"""
rl_agent.py
===========
Standalone script that trains, saves and evaluates the Reinforcement
Learning agent for the Emergency Message Prioritization System.

This is where the RL agent is trained and used (SRS Section 6). It:

  1. Builds the Gymnasium-compatible `MessagePrioritizationEnv`
     (src/rl/environment.py).
  2. Trains a Deep Q-Network (DQN) policy with Stable-Baselines3, using a
     lightweight MLP network, entirely on CPU (SRS 6.6).
  3. Persists the trained policy to disk (models/dqn_policy.zip) so it can
     be reloaded at inference time without retraining.
  4. Logs the learning curve (mean reward per training chunk) to
     models/training_curve.json so the Streamlit app can visualize how the
     agent learned.
  5. Runs the reproducible evaluation procedure (SRS 6.8): Random vs.
     Rule-Based vs. RL on identical seeded episodes, and saves the
     comparison to models/eval_results.json.

Usage:
    python rl_agent.py                     # default: 40,000 timesteps
    python rl_agent.py --timesteps 100000   # longer training run
    python rl_agent.py --eval-only          # just re-evaluate a saved model

The Streamlit app (app.py) imports `load_or_train_agent` from this module so
the exact same training/loading logic is used interactively in the UI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.rl.environment import MessagePrioritizationEnv
from src.rl.evaluate import evaluate_all
from src.config import DQN_MODEL_PATH, TRAINING_CURVE_PATH, RANDOM_SEED


def make_env(seed: int = RANDOM_SEED):
    """Factory used by Stable-Baselines3 / Gymnasium wrappers."""
    def _init():
        return MessagePrioritizationEnv(seed=seed)
    return _init


def train_dqn(total_timesteps: int = 40_000, chunk_size: int = 4_000,
              seed: int = RANDOM_SEED, progress_callback=None, verbose: int = 1):
    """Train a DQN policy in chunks, recording a learning curve.

    Training is done in chunks (rather than one blocking `.learn()` call) so
    callers — including the Streamlit UI — can observe and plot the reward
    trend as the agent learns, and so long training runs can report
    incremental progress.

    Args:
        progress_callback: optional callable(chunk_idx, n_chunks, mean_reward)
            invoked after every chunk, e.g. to update a Streamlit progress
            bar / live chart.
    Returns:
        (model, learning_curve) where learning_curve is a list of
        {"timesteps": int, "mean_eval_reward": float} dicts.
    """
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env

    env = Monitor(MessagePrioritizationEnv(seed=seed))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.98,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        seed=seed,
    )

    n_chunks = max(1, total_timesteps // chunk_size)
    learning_curve = []
    eval_env = MessagePrioritizationEnv(seed=seed + 999)

    for chunk_idx in range(n_chunks):
        model.learn(total_timesteps=chunk_size, reset_num_timesteps=False, progress_bar=False)

        # Quick evaluation snapshot (few episodes) to trace the learning curve.
        rewards = []
        for ep in range(3):
            obs, _ = eval_env.reset(seed=seed + 999 + chunk_idx * 10 + ep)
            done, total_r = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = eval_env.step(int(action))
                total_r += r
                done = term or trunc
            rewards.append(total_r)
        mean_r = float(np.mean(rewards))
        timesteps_so_far = (chunk_idx + 1) * chunk_size
        learning_curve.append({"timesteps": timesteps_so_far, "mean_eval_reward": mean_r})

        if verbose:
            print(f"[DQN] timesteps={timesteps_so_far}/{total_timesteps} "
                  f"mean_eval_reward={mean_r:.2f}")
        if progress_callback is not None:
            progress_callback(chunk_idx + 1, n_chunks, mean_r)

    return model, learning_curve


def save_agent(model, learning_curve: list[dict]):
    DQN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(DQN_MODEL_PATH).replace(".zip", ""))
    with open(TRAINING_CURVE_PATH, "w") as f:
        json.dump(learning_curve, f, indent=2)


def load_agent():
    """Load a previously trained DQN policy from disk. Returns None if not
    found or if stable-baselines3 is unavailable."""
    try:
        from stable_baselines3 import DQN
    except ImportError:
        return None
    if not DQN_MODEL_PATH.exists():
        return None
    return DQN.load(str(DQN_MODEL_PATH).replace(".zip", ""))


def load_or_train_agent(total_timesteps: int = 20_000, chunk_size: int = 4_000,
                         progress_callback=None, force_retrain: bool = False):
    """Convenience entry point used by the Streamlit app: load a cached
    policy if one exists on disk, otherwise train a fresh (smaller, UI-
    responsive) one and save it for next time."""
    if not force_retrain:
        model = load_agent()
        if model is not None:
            curve = []
            if TRAINING_CURVE_PATH.exists():
                with open(TRAINING_CURVE_PATH) as f:
                    curve = json.load(f)
            return model, curve, True  # True = loaded from cache

    model, curve = train_dqn(
        total_timesteps=total_timesteps, chunk_size=chunk_size,
        progress_callback=progress_callback,
    )
    save_agent(model, curve)
    return model, curve, False  # False = freshly trained


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the RL prioritization agent.")
    parser.add_argument("--timesteps", type=int, default=40_000,
                         help="Total DQN training timesteps.")
    parser.add_argument("--chunk-size", type=int, default=4_000,
                         help="Timesteps per training chunk (for the learning curve).")
    parser.add_argument("--eval-episodes", type=int, default=30,
                         help="Number of seeded episodes per policy during evaluation.")
    parser.add_argument("--eval-only", action="store_true",
                         help="Skip training; just evaluate the model already saved on disk.")
    args = parser.parse_args()

    if args.eval_only:
        model = load_agent()
        if model is None:
            print("No saved model found at", DQN_MODEL_PATH, "- train first.")
            return
    else:
        start = time.time()
        model, curve = train_dqn(total_timesteps=args.timesteps, chunk_size=args.chunk_size)
        save_agent(model, curve)
        print(f"Training complete in {time.time() - start:.1f}s. "
              f"Saved policy to {DQN_MODEL_PATH}")

    print("Running evaluation: Random vs. Rule-Based vs. RL (DQN) ...")
    summaries = evaluate_all(sb3_model=model, n_episodes=args.eval_episodes, save=True)
    for name, s in summaries.items():
        print(f"  {name:12s} | mean_reward={s['mean_episode_reward']:7.2f} "
              f"| mean_served={s['mean_messages_served']:5.1f} "
              f"| mean_urgency_served={s['mean_urgency_served']:.3f}")
    print(f"Saved evaluation results.")


if __name__ == "__main__":
    main()
