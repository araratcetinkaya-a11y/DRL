"""Unit tests for the action model (gym_env/core/action_model.py).

Numbers come from the thesis config: acc_xy_max = 0.4 (used as the max speed change per
step), v_xy_min = 0.5 m/s, v_xy_max = 5 m/s, yaw_rate_max_deg = 60 deg/s.
The default action gain is 1.0, as in training.
"""
import math
import subprocess
import sys

import pytest

from gym_env.core.action_model import (
    DEFAULT_ACTION_GAIN,
    body_to_world_velocity,
    next_speed_setpoint,
    yaw_rate_command_deg,
)

DELTA_MAX = 0.4
V_MIN, V_MAX = 0.5, 5.0
YAW_RATE_MAX = 60.0


def speed(prev, action, gain=DEFAULT_ACTION_GAIN):
    return float(next_speed_setpoint(prev, action, DELTA_MAX, V_MIN, V_MAX, gain=gain))


def test_default_gain_is_training_behavior():
    assert DEFAULT_ACTION_GAIN == 1.0


# --------------------------------------------------------------- forward speed
@pytest.mark.parametrize("action, expected", [(1.0, 2.4), (0.5, 2.2), (0.0, 2.0), (-1.0, 1.6)])
def test_speed_changes_proportionally_to_action(action, expected):
    assert speed(2.0, action) == pytest.approx(expected)


def test_speed_change_per_step_is_limited():
    for prev in (1.0, 2.5, 4.0):
        for action in (-1.0, -0.3, 0.0, 0.7, 1.0):
            assert abs(speed(prev, action) - prev) <= DELTA_MAX + 1e-12


def test_speed_never_exceeds_max():
    assert speed(4.9, 1.0) == V_MAX
    assert speed(V_MAX, 1.0) == V_MAX


def test_speed_never_drops_below_min_so_drone_cannot_hover():
    """The non-zero minimum speed blocks the passive 'hovering' strategy (thesis, reward hacking)."""
    assert speed(0.6, -1.0) == V_MIN
    assert speed(V_MIN, -1.0) == V_MIN


def test_first_step_from_standstill_jumps_to_min_speed():
    assert speed(0.0, 0.0) == V_MIN
    assert speed(0.0, 1.0) == V_MIN  # 0 + 0.4 is still below 0.5


# -------------------------------------------------------------------- yaw rate
@pytest.mark.parametrize("action, expected", [(1.0, 60.0), (0.5, 30.0), (0.0, 0.0), (-1.0, -60.0)])
def test_yaw_rate_is_proportional_to_action(action, expected):
    assert yaw_rate_command_deg(action, YAW_RATE_MAX) == pytest.approx(expected)


def test_yaw_rate_stays_within_config_limit_with_default_gain():
    for action in (-1.0, -0.5, 0.0, 0.5, 1.0):
        assert abs(yaw_rate_command_deg(action, YAW_RATE_MAX)) <= YAW_RATE_MAX


# ------------------------------------------- gain parameter (evaluation experiment)
def test_gain_scales_speed_change_and_yaw_rate():
    """The 1.1 evaluation experiment: 0.44 m/s per step and up to 66 deg/s."""
    assert speed(2.0, 1.0, gain=1.1) == pytest.approx(2.44)
    assert yaw_rate_command_deg(1.0, YAW_RATE_MAX, gain=1.1) == pytest.approx(66.0)


# ---------------------------------------------------------- frame conversion
@pytest.mark.parametrize(
    "yaw_deg, expected",
    [(0, (2.0, 0.0)), (90, (0.0, 2.0)), (180, (-2.0, 0.0)), (-90, (0.0, -2.0)), (45, (math.sqrt(2), math.sqrt(2)))],
)
def test_body_to_world_velocity(yaw_deg, expected):
    vx, vy = body_to_world_velocity(2.0, math.radians(yaw_deg))
    assert (vx, vy) == pytest.approx(expected, abs=1e-12)


def test_body_to_world_keeps_speed_magnitude():
    for yaw_deg in range(-180, 181, 15):
        vx, vy = body_to_world_velocity(3.3, math.radians(yaw_deg))
        assert math.hypot(vx, vy) == pytest.approx(3.3)


# ---------------------------------------------------------------------- purity
def test_core_imports_without_hardware_packages():
    """gym_env.core must not need airsim, keyboard, PyQt5 or msgpackrpc (checked in a clean process)."""
    code = (
        "import sys\n"
        "for name in ('airsim', 'keyboard', 'PyQt5', 'msgpackrpc'):\n"
        "    sys.modules[name] = None  # any import of these now raises ImportError\n"
        "import gym_env.core.action_model\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
