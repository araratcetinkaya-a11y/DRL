"""Smoke tests: the thesis environment can be built, reset and stepped with the fake AirSim client."""
import numpy as np


def test_env_uses_thesis_config(env):
    raw = env.unwrapped
    assert raw.env_name == "SimpleAvoid"
    assert raw.dynamic_name == "Multirotor"
    assert raw.perception_type == "depth"
    assert raw.reward_type == "reward_final"
    assert (raw.screen_height, raw.screen_width) == (70, 200)


def test_action_space_is_2d_normalized(env):
    assert env.action_space.shape == (2,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])


def test_step_returns_gym_019_tuple(env):
    env.reset()
    obs, reward, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert obs is not None
    assert isinstance(float(reward), float)
    assert isinstance(done, (bool, np.bool_))
    for key in ("is_success", "is_crash", "is_not_in_workspace", "step_num", "reward_terms"):
        assert key in info


def test_step_advances_simulation_by_dt_in_lockstep(env, client):
    """One agent step = one simContinueForTime(dt) call with dt = 0.1 s from the config."""
    env.reset()
    calls_before = [c for c in client.calls if c[0] == "simContinueForTime"]

    env.step(np.array([0.0, 0.0], dtype=np.float32))

    new_calls = [c for c in client.calls if c[0] == "simContinueForTime"][len(calls_before):]
    assert len(new_calls) == 1
    assert new_calls[0][1] == (0.1,)


def test_crash_ends_episode(env, client):
    env.reset()
    client.has_collided = True

    _, _, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert done
    assert info["is_crash"]
