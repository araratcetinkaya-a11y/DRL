"""Audit question Q1: does the image observation match the declared observation_space?

The first two tests describe what the frozen thesis code does today (characterization tests).
The xfail tests describe what SB3 needs. They are expected to fail until the observation
pipeline is fixed (roadmap Phase 3). Because pytest.ini sets xfail_strict = true, they turn
into real failures as soon as the fix lands, which is the reminder to remove the marker.
"""
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

Q1 = "Q1 (docs/AUDIT.md): get_obs returns (2, H, W) uint8, observation_space is (H, W, 2)"


def test_declared_observation_space(env):
    space = env.observation_space
    assert space.shape == (70, 200, 2)
    assert space.dtype == np.uint8
    assert space.high.max() == 1  # not 255, so SB3 does not treat it as an image


def test_reset_returns_channels_first_uint8(env):
    obs = env.reset()
    assert obs.shape == (2, 70, 200)
    assert obs.dtype == np.uint8


@pytest.mark.xfail(reason=Q1, strict=True)
def test_reset_observation_matches_declared_space(env):
    obs = env.reset()
    assert env.observation_space.contains(obs)


@pytest.mark.xfail(reason=Q1, raises=ValueError, strict=True)
def test_dummy_vec_env_can_reset(thesis_cfg):
    """SB3 wraps the env in DummyVecEnv for training, evaluation and SAC.load(env=...)."""
    import gym

    def make_env():
        environment = gym.make("airsim-env-v0")
        environment.set_config(thesis_cfg)
        return environment

    np.random.seed(0)
    DummyVecEnv([make_env]).reset()
