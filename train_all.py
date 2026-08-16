"""
train_all.py
============
Optional convenience script: trains and saves BOTH the NLP models and the RL
agent in one command, so that when you push to GitHub and deploy on
Streamlit Community Cloud, `app.py` loads pretrained artifacts instantly
instead of training on first visit.

This is not required — `app.py` and `rl_agent.py` will each train on demand
and cache the result if artifacts aren't found — but running this once
locally before your first deploy gives the smoothest first-load experience.

Usage:
    python train_all.py                  # NLP + a modest RL run (~40k steps)
    python train_all.py --rl-timesteps 100000
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")


def train_nlp():
    from src.data import load_dataset
    from src.nlp import CategoryClassifier, UrgencyClassifier

    print("=" * 70)
    print("Training NLP models (category + urgency classifiers)")
    print("=" * 70)
    df = load_dataset(cache=True)
    print(f"Dataset: {len(df)} messages ({df['provenance'].value_counts().to_dict()})")

    cat_clf = CategoryClassifier()
    cat_clf.fit(df["text"], df["category"])
    cat_clf.save()

    urg_clf = UrgencyClassifier()
    urg_clf.fit(df["text"], df["urgency_level"])
    urg_clf.save()

    print("Saved NLP models to models/\n")


def train_rl(total_timesteps: int, chunk_size: int):
    from rl_agent import train_dqn, save_agent
    from src.rl.evaluate import evaluate_all

    print("=" * 70)
    print(f"Training RL agent (DQN, {total_timesteps:,} timesteps)")
    print("=" * 70)
    model, curve = train_dqn(total_timesteps=total_timesteps, chunk_size=chunk_size)
    save_agent(model, curve)
    print("Saved RL policy to models/dqn_policy.zip\n")

    print("Running evaluation: Random vs Rule-Based vs RL (DQN) ...")
    summaries = evaluate_all(sb3_model=model, n_episodes=30, save=True)
    for name, s in summaries.items():
        print(f"  {name:12s} | mean_reward={s['mean_episode_reward']:7.2f} "
              f"| mean_served={s['mean_messages_served']:5.1f} "
              f"| mean_urgency_served={s['mean_urgency_served']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain NLP + RL models before deployment.")
    parser.add_argument("--skip-nlp", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--rl-timesteps", type=int, default=40_000)
    parser.add_argument("--rl-chunk-size", type=int, default=4_000)
    args = parser.parse_args()

    if not args.skip_nlp:
        train_nlp()
    if not args.skip_rl:
        train_rl(args.rl_timesteps, args.rl_chunk_size)

    print("=" * 70)
    print("All done. models/ now contains pretrained artifacts ready to commit:")
    from src.config import MODELS_DIR
    for f in sorted(MODELS_DIR.iterdir()):
        print(" -", f.name)
    print("=" * 70)


if __name__ == "__main__":
    main()
