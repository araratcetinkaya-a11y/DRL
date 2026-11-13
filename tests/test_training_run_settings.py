"""The environment and config use the settings of the thesis training run (docs/TRAINING_RUN.md).

These values are not covered by the characterization tests (the fake AirSim client does not fly with
the PID gains, and the scripted episodes are shorter than an episode), so they are checked directly.
"""
import pytest


def test_episode_length_and_start_height(env):
    raw = env.unwrapped
    assert raw.max_episode_steps == 400
    assert list(raw.dynamic_model.start_position) == [0, 0, 5]


def test_reward_coefficients(env):
    raw = env.unwrapped
    assert raw.SAFE_DISTANCE == 2
    assert raw.k_obs_penalty == 0.5
    assert raw.k_speed_near_obs == 1
    assert raw.max_safe_speed_near_obs == 4
    assert (raw.k_jerk_fwd, raw.k_jerk_yaw) == (0.025, 0.03)


def test_action_limits(env):
    dynamics = env.unwrapped.dynamic_model
    assert dynamics.dt == pytest.approx(0.1)
    assert dynamics.v_xy_delta_max == pytest.approx(0.4)  # acc_xy_max, applied per step (no * dt)
    assert (dynamics.v_xy_min, dynamics.v_xy_max) == (0.5, 5)
    assert dynamics.yaw_rate_max_deg == 60


def gains_sent(client, name):
    calls = [c for c in client.calls if c[0] == name]
    assert len(calls) == 1, name
    return calls[0][1][0]


def pid(g):
    return (g.kp, g.ki, g.kd)


def test_controller_gains(client):
    velocity = gains_sent(client, "setVelocityControllerGains")
    assert pid(velocity.x_gains) == pid(velocity.y_gains) == (0.6, 0.0, 0.0)
    assert pid(velocity.z_gains) == (2.0, 2.0, 0.0)

    angle = gains_sent(client, "setAngleLevelControllerGains")
    assert pid(angle.roll_gains) == pid(angle.pitch_gains) == (2.5, 0.0, 0.0)
    assert pid(angle.yaw_gains) == (3.5, 0, 0.0)


def test_training_settings_in_config(thesis_cfg):
    drl = thesis_cfg["DRL"]
    assert drl.getint("buffer_size") == 100000
    assert drl.getint("batch_size") == 256
    assert drl.getint("train_freq") == drl.getint("gradient_steps") == 256
    assert drl.getint("learning_starts") == 25000
    assert thesis_cfg.getint("options", "total_timesteps") == 900000
