"""Shared test setup.

The real `airsim`, `keyboard` and `msgpackrpc` packages are replaced by small fakes
before `gym_env` is imported, so every test runs without AirSim, Unreal Engine,
a GPU or a display. This happens even if the real packages are installed.
"""
import sys
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pytest

from tests.fakes import fake_airsim
from tests.fakes.fake_modules import keyboard, msgpackrpc, msgpackrpc_error

sys.modules["airsim"] = fake_airsim
sys.modules["keyboard"] = keyboard
sys.modules["msgpackrpc"] = msgpackrpc
sys.modules["msgpackrpc.error"] = msgpackrpc_error

import gym  # noqa: E402  (must come after the fakes are installed)
import gym_env  # noqa: E402,F401  (registers 'airsim-env-v0')

REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_CONFIG = REPO_ROOT / "configs" / "config_Maze_SimpleMultirotor_2D.ini"


@pytest.fixture
def thesis_cfg():
    """The configuration of the final thesis training run."""
    cfg = ConfigParser()
    assert cfg.read(THESIS_CONFIG), f"config not found: {THESIS_CONFIG}"
    return cfg


@pytest.fixture
def env(thesis_cfg):
    """The thesis environment, connected to the fake AirSim client.

    np.random is seeded because reset() draws the start heading and goal direction from it.
    """
    np.random.seed(0)
    environment = gym.make("airsim-env-v0")
    environment.set_config(thesis_cfg)
    return environment


@pytest.fixture
def client(env):
    """The fake AirSim client used by `env`, to inspect calls or change the scene."""
    return env.unwrapped.dynamic_model.client
