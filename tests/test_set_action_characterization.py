"""Characterization tests for set_action after moving its math to gym_env/core/action_model.py.

1. Training run: EXPECTED_TRAINING_RUN was recorded by running the code of the thesis training
   run (tag `thesis-final.1`, docs/TRAINING_RUN.md) on the fake AirSim client with the action
   sequence below. With the thesis config (gain 1.0) the refactored code must send exactly the
   same commands, including the altitude (start height 5 m) and the command duration (dt).
2. Equivalence with `thesis-final`: EXPECTED_GAIN_1_1 was recorded from the code of the
   `thesis-final` tag, which multiplies both actions by 1.1 (a post-training evaluation
   experiment). With `action_gain = 1.1` in the config, the refactored code must reproduce it.
3. The training speed and yaw rate sequence, worked out by hand from the config.

Each recorded row is what reached moveByVelocityZAsync:
(vx_world [m/s], vy_world [m/s], forward speed setpoint [m/s], yaw rate [deg/s]).
The sequence covers the jump to the minimum speed on the first step, the 5 m/s limit,
braking back down, and the full yaw rate range.
"""
import math

import numpy as np
import pytest

from tests.conftest import build_env

ACTIONS = [
    (1.0, 0.0), (1.0, 0.5), (0.5, -1.0), (1.0, 1.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
    (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-1.0, -0.25),
    (0.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0),
]

# (vx_world, vy_world, speed setpoint, yaw rate); every command had z = -5 and duration = 0.1
EXPECTED_TRAINING_RUN = [
    (-0.10849095903294602, -0.4880878115750399, 0.5, 0.0),
    (-0.19528372625930285, -0.8785580608350718, 0.9, 30.0),
    (-0.18215501390865435, -1.0848131410099797, 1.1, -60.0),
    (-0.40166045488036184, -1.4452227783235707, 1.5, 60.0),
    (-0.31463138766040294, -1.873768152653601, 1.9, 0.0),
    (-0.38086957453627723, -2.268245658475412, 2.3, 0.0),
    (-0.4471077614121515, -2.6627231642972222, 2.6999999999999997, 0.0),
    (-0.5133459482880258, -3.057200670119033, 3.0999999999999996, 0.0),
    (-0.5795841351639001, -3.451678175940844, 3.4999999999999996, 0.0),
    (-0.6458223220397744, -3.8461556817626543, 3.8999999999999995, 0.0),
    (-0.7120605089156488, -4.240633187584465, 4.3, 0.0),
    (-0.7782986957915231, -4.6351106934062765, 4.7, 0.0),
    (-0.7120605089156488, -4.240633187584465, 4.3, 0.0),
    (-0.6458223220397744, -3.8461556817626548, 3.9, -15.0),
    (-0.7462816333253164, -3.827932042730291, 3.9, 0.0),
    (-0.6697399273432326, -3.435323628091287, 3.5, 0.0),
    (-0.593198221361149, -3.042715213452283, 3.1, 0.0),
    (-0.5166565153790652, -2.6501067988132787, 2.7, 0.0),
    (-0.4401148093969815, -2.2574983841742746, 2.3000000000000003, 0.0),
    (-0.3635731034148978, -1.8648899695352703, 1.9000000000000004, 0.0),
    (-0.2870313974328141, -1.4722815548962662, 1.5000000000000004, 0.0),
]

EXPECTED_GAIN_1_1 = [
    (-0.10849095903294602, -0.4880878115750399, 0.5, 0.0),
    (-0.20396300298193853, -0.917605085761075, 0.9400000000000001, 33.0),
    (-0.18609824594457786, -1.1449748656002678, 1.1600000000000001, -66.0),
    (-0.43650355542813196, -1.5393065471499172, 1.6, 66.0),
    (-0.32727622562667136, -2.0135764877797815, 2.04, 0.0),
    (-0.39786521546771814, -2.447877298869538, 2.48, 0.0),
    (-0.46845420530876486, -2.8821781099592947, 2.92, 0.0),
    (-0.5390431951498117, -3.3164789210490513, 3.36, 0.0),
    (-0.6096321849908584, -3.750779732138808, 3.8, 0.0),
    (-0.6802211748319053, -4.185080543228565, 4.24, 0.0),
    (-0.750810164672952, -4.619381354318322, 4.680000000000001, 0.0),
    (-0.8021476118300769, -4.935236489656327, 5.0, 0.0),
    (-0.7315586219890301, -4.500935678566569, 4.56, 0.0),
    (-0.6609696321479832, -4.066634867476813, 4.119999999999999, -16.5),
    (-0.7777900637887691, -4.045916783211371, 4.119999999999999, 0.0),
    (-0.6947251055200656, -3.6138285830625847, 3.6799999999999993, 0.0),
    (-0.6116601472513622, -3.181740382913797, 3.2399999999999993, 0.0),
    (-0.5285951889826587, -2.74965218276501, 2.7999999999999994, 0.0),
    (-0.44553023071395514, -2.3175639826162224, 2.3599999999999994, 0.0),
    (-0.3624652724452516, -1.8854757824674353, 1.9199999999999995, 0.0),
    (-0.2794003141765481, -1.453387582318648, 1.4799999999999995, 0.0),
]

# Training behavior (gain 1.0), worked out by hand from the config: speed changes by 0.4 m/s
# per unit of action[0] and stays in [0.5, 5]; yaw rate = action[1] * 60 deg/s.
EXPECTED_SPEED_GAIN_1_0 = [0.5, 0.9, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1, 3.5, 3.9, 4.3,
                           4.7, 4.3, 3.9, 3.9, 3.5, 3.1, 2.7, 2.3, 1.9, 1.5]
EXPECTED_YAW_RATE_GAIN_1_0 = [0.0, 30.0, -60.0, 60.0] + [0.0] * 9 + [-15.0] + [0.0] * 7


def run_actions(environment, z_and_duration=None):
    """Apply ACTIONS one by one; return what reached moveByVelocityZAsync at each step.

    If a list is given as z_and_duration, the altitude and duration of each command are appended to it.
    """
    environment.reset()
    dynamics = environment.unwrapped.dynamic_model
    client = dynamics.client
    sent = []
    for step, action in enumerate(ACTIONS):
        n_calls = len(client.calls)
        dynamics.set_action(np.array(action, dtype=np.float32))
        moves = [c for c in client.calls[n_calls:] if c[0] == "moveByVelocityZAsync"]
        assert len(moves) == 1, f"step {step}: expected one velocity command"
        vx, vy, z, duration = moves[0][1]
        if z_and_duration is not None:
            z_and_duration.append((z, duration))
        yaw_rate = moves[0][2]["yaw_mode"].yaw_or_rate
        sent.append((vx, vy, float(dynamics.v_xy_sp), float(yaw_rate)))
    return sent


def test_thesis_config_uses_training_gain(env):
    assert env.unwrapped.dynamic_model.action_gain == 1.0


def test_reproduces_training_run_commands(env):
    z_and_duration = []
    sent = run_actions(env, z_and_duration)

    for step, (actual, expected) in enumerate(zip(sent, EXPECTED_TRAINING_RUN)):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), f"step {step}, action {ACTIONS[step]}"
    assert z_and_duration == [(-5, 0.1)] * len(ACTIONS)


def test_gain_1_1_reproduces_frozen_thesis_code(thesis_cfg):
    thesis_cfg.set("multirotor", "action_gain", "1.1")
    sent = run_actions(build_env(thesis_cfg))

    for step, (actual, expected) in enumerate(zip(sent, EXPECTED_GAIN_1_1)):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), f"step {step}, action {ACTIONS[step]}"


def test_training_gain_speed_and_yaw_rate_sequence(env):
    sent = run_actions(env)

    for step, (vx, vy, v_xy_sp, yaw_rate) in enumerate(sent):
        assert v_xy_sp == pytest.approx(EXPECTED_SPEED_GAIN_1_0[step]), f"step {step}"
        assert yaw_rate == pytest.approx(EXPECTED_YAW_RATE_GAIN_1_0[step]), f"step {step}"
        assert math.hypot(vx, vy) == pytest.approx(v_xy_sp), f"step {step}: world velocity magnitude"
