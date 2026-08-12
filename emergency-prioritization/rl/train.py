"""
Stage 8 — RL training.

Trains a DQN agent (Stable-Baselines3) on MessagePrioritizationEnv.

NOTE ON VERIFICATION (disclosed honestly): this script's environment
(rl/environment.py) was unit-tested and sanity-checked in the development
sandbox (greedy-urgency policy reward 64.7 vs random policy reward 26.4,
confirming the reward function rewards correct behavior). The actual SB3
training loop below could NOT be executed in that sandbox (no internet
access to install torch/stable-baselines3) and has not been run end-to-end
by Claude. Run it yourself locally and check the printed eval reward before
trusting the saved model — if something errors, the environment contract
(reset/step signatures, observation_space, action_space) matches the
standard Gymnasium API used by SB3 examples, so a fix should be localized
to this file.

Usage:
    python -m rl.train --timesteps 50000
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from rl.environment import MessagePrioritizationEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "dqn_prioritization_agent"  # SB3 appends .zip
LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "rl_training"


def make_env(seed: int = None):
    env = MessagePrioritizationEnv(window_size=5, episode_length=100, arrival_rate=0.6, seed=seed)
    return Monitor(env)


def train(total_timesteps: int = 50_000, seed: int = 42) -> DQN:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_env = make_env(seed=seed)
    eval_env = make_env(seed=seed + 1)

    # Small MLP policy -- deliberately lightweight for CPU training.
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=20_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=seed,
        tensorboard_log=str(LOG_DIR),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ARTIFACT_DIR / "best_model"),
        log_path=str(LOG_DIR),
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    logger.info("Starting DQN training for %d timesteps...", total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    model.save(str(MODEL_PATH))
    logger.info("Saved trained model -> %s.zip", MODEL_PATH)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    logger.info("Final evaluation: mean_reward=%.2f +/- %.2f over 20 episodes", mean_reward, std_reward)

    return model


def load_trained_agent() -> DQN:
    path = MODEL_PATH.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(
            f"No trained RL agent found at {path}. Train first: python -m rl.train"
        )
    return DQN.load(str(MODEL_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL prioritization agent")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, seed=args.seed)
