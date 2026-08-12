"""
Stage 15 — Automated tests: RL environment + rule-based baseline.

NOTE: requires gymnasium to be installed to run (pip install gymnasium).
Run with: pytest tests/test_rl.py -v
"""
import pytest
import numpy as np

gymnasium = pytest.importorskip("gymnasium")

from rl.environment import MessagePrioritizationEnv
from rl.baseline import rule_based_priority, rank_queue


class TestEnvironment:
    def test_reset_returns_valid_observation(self):
        env = MessagePrioritizationEnv(window_size=5, episode_length=20, seed=1)
        obs, info = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_step_returns_valid_tuple(self):
        env = MessagePrioritizationEnv(window_size=5, episode_length=20, seed=1)
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_truncates_at_episode_length(self):
        env = MessagePrioritizationEnv(window_size=5, episode_length=10, seed=1)
        env.reset(seed=1)
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
        assert truncated is True

    def test_empty_slot_action_penalized(self):
        """Selecting a slot beyond the current queue length should incur
        the no-op penalty and not crash."""
        env = MessagePrioritizationEnv(window_size=5, episode_length=5, arrival_rate=0.0, seed=1)
        env.reset(seed=1)
        # Drain the seeded queue first
        for _ in range(env.window_size):
            env.step(0)
        obs, reward, terminated, truncated, info = env.step(0)  # queue now empty
        assert reward <= 0  # no-op penalty, no positive reward possible

    def test_greedy_urgency_policy_beats_random(self):
        """Regression test for the core reward-design sanity check."""
        def run(policy_fn, seed):
            env = MessagePrioritizationEnv(window_size=5, episode_length=50, seed=seed)
            env.reset(seed=seed)
            total = 0.0
            for _ in range(50):
                action = policy_fn(env)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if truncated:
                    break
            return total

        def greedy(env):
            window = env.queue[:env.window_size]
            if not window:
                return 0
            return max(range(len(window)), key=lambda i: window[i].urgency_score)

        def random_policy(env):
            return env.action_space.sample()

        greedy_reward = run(greedy, seed=7)
        random_reward = run(random_policy, seed=7)
        assert greedy_reward > random_reward


class TestRuleBasedBaseline:
    def test_higher_urgency_scores_higher(self):
        low = rule_based_priority(0.2, "Medical", 0, 0.8, False)
        high = rule_based_priority(0.9, "Medical", 0, 0.8, False)
        assert high > low

    def test_duplicate_penalized(self):
        normal = rule_based_priority(0.8, "Medical", 0, 0.8, False)
        dup = rule_based_priority(0.8, "Medical", 0, 0.8, True)
        assert dup < normal

    def test_score_bounded(self):
        score = rule_based_priority(1.0, "Medical", 100, 1.0, False)
        assert 0.0 <= score <= 1.0

    def test_rank_queue_sorts_descending(self):
        messages = [
            {"message_id": "A", "urgency_score": 0.2, "category": "Food", "waiting_time": 0, "category_confidence": 0.5},
            {"message_id": "B", "urgency_score": 0.9, "category": "Medical", "waiting_time": 0, "category_confidence": 0.9},
        ]
        ranked = rank_queue(messages)
        assert ranked[0]["message_id"] == "B"
