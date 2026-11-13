"""The evaluation environment built by thread_train.make_eval_env (used by the EvalCallback).

Training builds a second environment for evaluation: gym.make -> SeedCycler -> Monitor -> DummyVecEnv.
A NameError there (SeedCycler was removed by mistake in Phase 3) only showed up in a real training
run on the university PC, so this path is now tested with the fake AirSim client.

thread_train imports wandb at module level. wandb is not needed for these tests and is not
installed in CI, so a small fake module is used when the real one is missing.
"""
import sys
import types

import gym
import numpy as np
import pytest


@pytest.fixture
def thread_train(monkeypatch):
    try:
        import wandb  # noqa: F401
    except ImportError:
        fake = types.ModuleType("wandb")
        integration = types.ModuleType("wandb.integration")
        sb3 = types.ModuleType("wandb.integration.sb3")
        sb3.WandbCallback = object
        fake.integration, integration.sb3 = integration, sb3
        for name, module in [("wandb", fake), ("wandb.integration", integration), ("wandb.integration.sb3", sb3)]:
            monkeypatch.setitem(sys.modules, name, module)
    from scripts.utils import thread_train as module
    return module


def test_make_eval_env_resets_and_steps_with_dict_observation(thread_train, thesis_cfg, tmp_path):
    thesis_cfg.set("EVAL", "eval_log_dir", str(tmp_path))
    np.random.seed(0)
    venv = thread_train.make_eval_env(thesis_cfg)

    obs = venv.reset()
    assert set(obs) == {"depth", "state"}
    assert obs["depth"].shape == (1, 1, 70, 200) and obs["depth"].dtype == np.uint8
    assert obs["state"].shape == (1, 6) and obs["state"].dtype == np.float32

    obs, reward, done, info = venv.step(np.zeros((1, 2), dtype=np.float32))
    assert set(obs) == {"depth", "state"}
    assert reward.shape == (1,) and done.shape == (1,)
    assert (tmp_path / "eval_monitor.csv").exists()


class _SeedRecordingEnv(gym.Env):
    observation_space = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def __init__(self):
        self.seeds = []

    def seed(self, seed=None):
        self.seeds.append(seed)
        return [seed]

    def reset(self):
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, {}


def test_seed_cycler_uses_the_next_seed_on_every_reset(thread_train):
    base = _SeedRecordingEnv()
    env = thread_train.SeedCycler(base, [101, 202, 303])

    for _ in range(4):
        env.reset()

    assert base.seeds == [101, 202, 303, 101]


def test_eval_callback_path_runs_one_episode(thread_train, thesis_cfg, tmp_path):
    """What the EvalCallback does at eval_freq: evaluate_policy on the eval env with a SAC model whose
    policy is MultiInputPolicy + CNN_GAP_new, as in training. One episode on the fake client."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.evaluation import evaluate_policy

    from scripts.utils.custom_policy_sb3 import CNN_GAP_new

    thesis_cfg.set("EVAL", "eval_log_dir", str(tmp_path))
    np.random.seed(0)
    venv = thread_train.make_eval_env(thesis_cfg)
    model = SAC(
        "MultiInputPolicy", venv, buffer_size=100, learning_starts=10,
        policy_kwargs=dict(features_extractor_class=CNN_GAP_new, net_arch=[16]),
        device="cpu", seed=0,
    )

    rewards, lengths = evaluate_policy(model, venv, n_eval_episodes=1, deterministic=True,
                                       return_episode_rewards=True)

    assert len(rewards) == len(lengths) == 1
    # A time-limit episode has max_episode_steps + 1 = 401 steps: is_done() is checked before step_num
    # is increased. The training-run code does the same (docs/AUDIT.md, "Out of scope"), so it is kept.
    assert 1 <= lengths[0] <= 401
