"""Unit tests for the reward terms (gym_env/core/reward.py), with the thesis coefficients."""
import subprocess
import sys

import pytest

from gym_env.core.reward import (
    REWARD_CRASH,
    REWARD_OUTSIDE,
    REWARD_REACH,
    distance_progress_reward,
    jerk_penalty,
    obstacle_penalty,
    speed_near_obstacle_penalty,
    terminal_reward,
)

SAFE_DISTANCE = 2.0
K_OBS = 0.5
K_SPEED = 50.0
MAX_SAFE_SPEED = 3.5
GOAL_DISTANCE = 50.0


# ------------------------------------------------------------ distance progress
@pytest.mark.parametrize(
    "d_prev, d_now, expected",
    [(50.0, 49.5, 0.4), (50.0, 50.0, 0.0), (49.5, 50.0, -0.4), (10.0, 0.0, 8.0)],
)
def test_distance_progress_reward(d_prev, d_now, expected):
    assert distance_progress_reward(d_prev, d_now, GOAL_DISTANCE) == pytest.approx(expected)


def test_distance_progress_sums_to_coefficient_over_a_full_approach():
    """Potential-based shaping: the per-step terms telescope, so going from the start to the goal
    adds up to coef = 40 whatever path the drone takes."""
    distances = [50.0, 47.0, 48.5, 30.0, 12.0, 12.5, 0.0]
    total = sum(distance_progress_reward(a, b, GOAL_DISTANCE) for a, b in zip(distances, distances[1:]))
    assert total == pytest.approx(40.0)


def test_distance_coefficient_can_be_changed():
    assert distance_progress_reward(50.0, 49.0, GOAL_DISTANCE, coef=500) == pytest.approx(10.0)


# ------------------------------------------------------------ obstacle penalty
@pytest.mark.parametrize(
    "dist, expected",
    [(5.0, 0.0), (2.0, 0.0), (1.5, 0.125), (1.0, 0.5), (0.0, 2.0)],
)
def test_obstacle_penalty_is_quadratic_inside_safety_bubble(dist, expected):
    assert obstacle_penalty(dist, SAFE_DISTANCE, K_OBS) == pytest.approx(expected)


def test_obstacle_penalty_starts_exactly_at_the_bubble_boundary():
    assert obstacle_penalty(SAFE_DISTANCE, SAFE_DISTANCE, K_OBS) == 0.0
    assert obstacle_penalty(SAFE_DISTANCE - 1e-6, SAFE_DISTANCE, K_OBS) > 0.0


# ------------------------------------------------- speed near obstacle penalty
@pytest.mark.parametrize(
    "dist, v_xy, expected",
    [
        (5.0, 5.0, 0.0),    # outside the bubble: no penalty even at full speed
        (1.5, 3.0, 0.0),    # inside, but slower than the safe speed
        (1.5, 3.5, 0.0),    # exactly at the safe speed
        (1.5, 4.5, 50.0),   # 1 m/s too fast -> 50 * 1^2
        (1.5, 5.0, 112.5),  # 1.5 m/s too fast -> 50 * 1.5^2
    ],
)
def test_speed_near_obstacle_penalty(dist, v_xy, expected):
    result = speed_near_obstacle_penalty(dist, v_xy, SAFE_DISTANCE, MAX_SAFE_SPEED, K_SPEED)
    assert result == pytest.approx(expected)


def test_speed_penalty_starts_exactly_at_the_bubble_boundary():
    """At exactly SAFE_DISTANCE the drone is outside the bubble, so even 5 m/s is not penalized."""
    assert speed_near_obstacle_penalty(SAFE_DISTANCE, 5.0, SAFE_DISTANCE, MAX_SAFE_SPEED, K_SPEED) == 0.0
    assert speed_near_obstacle_penalty(SAFE_DISTANCE - 1e-6, 5.0, SAFE_DISTANCE, MAX_SAFE_SPEED, K_SPEED) > 0.0


# ------------------------------------------------------------------- jerk
@pytest.mark.parametrize(
    "action, last, k, expected",
    [(1.0, 1.0, 0.025, 0.0), (1.0, 0.0, 0.025, 0.025), (1.0, -1.0, 0.025, 0.1), (-1.0, 1.0, 0.03, 0.12)],
)
def test_jerk_penalty(action, last, k, expected):
    assert jerk_penalty(action, last, k) == pytest.approx(expected)


def test_jerk_penalty_is_symmetric():
    assert jerk_penalty(0.3, -0.5, 0.03) == pytest.approx(jerk_penalty(-0.5, 0.3, 0.03))


# --------------------------------------------------------------- terminal
@pytest.mark.parametrize(
    "reached, crashed, outside, expected",
    [
        (True, False, False, REWARD_REACH),
        (False, True, False, REWARD_CRASH),
        (False, False, True, REWARD_OUTSIDE),
        (False, False, False, 0.0),        # time limit: no terminal reward
        (True, True, False, REWARD_REACH),  # reaching the goal has priority
        (False, True, True, REWARD_CRASH),  # a crash has priority over leaving the workspace
    ],
)
def test_terminal_reward(reached, crashed, outside, expected):
    assert terminal_reward(reached, crashed, outside) == expected


def test_terminal_reward_values():
    assert (REWARD_REACH, REWARD_CRASH, REWARD_OUTSIDE) == (30.0, -25.0, -10.0)


# ------------------------------------------------------------------ purity
def test_reward_module_imports_without_hardware_packages():
    code = (
        "import sys\n"
        "for name in ('airsim', 'keyboard', 'PyQt5', 'msgpackrpc'):\n"
        "    sys.modules[name] = None\n"
        "import gym_env.core.reward\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
