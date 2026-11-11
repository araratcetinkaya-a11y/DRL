"""Unit tests for the observation pipeline (gym_env/core/observation.py).

Numbers come from the thesis config: 70x200 input, max_depth_meters = 15, v_xy in [0.5, 5] m/s,
yaw_rate_max_deg = 60.
"""
import math
import subprocess
import sys

import numpy as np
import pytest

from gym_env.core.observation import (
    normalize_angle,
    normalize_distance,
    normalize_last_action,
    normalize_speed,
    normalize_symmetric,
    preprocess_depth,
    stack_depth_and_state,
    wrap_to_pi,
)

H, W = 70, 200
MAX_DEPTH = 15.0
CAMERA_SHAPE = (144, 256)  # (height, width) of the AirSim camera image


def uniform_depth(meters):
    return np.full(CAMERA_SHAPE, meters, dtype=np.float32)


# ----------------------------------------------------------------- depth image
@pytest.mark.parametrize("height, width", [(70, 200), (80, 200), (64, 64), (144, 256)])
def test_depth_is_resized_to_any_network_input_as_float32(height, width):
    """The size comes from the arguments, not from the config: other input sizes work the same way."""
    image, _ = preprocess_depth(uniform_depth(10.0), height, width, MAX_DEPTH)
    assert image.shape == (height, width)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image, 1 - 10 / 15, atol=1e-6)


@pytest.mark.parametrize("height, width", [(70, 200), (80, 200), (64, 64)])
def test_stacked_observation_shape_follows_input_size(height, width):
    depth = np.zeros((height, width), dtype=np.float32)
    obs = stack_depth_and_state(depth, np.ones(6, dtype=np.float32))
    assert obs.shape == (height, width, 2)


@pytest.mark.parametrize(
    "meters, expected",
    [(0.0, 1.0), (3.0, 0.8), (7.5, 0.5), (10.0, 1 - 10 / 15), (15.0, 0.0), (40.0, 0.0)],
)
def test_depth_is_inverted_and_normalized(meters, expected):
    """Touching an obstacle = 1, at max_depth or farther = 0."""
    image, _ = preprocess_depth(uniform_depth(meters), H, W, MAX_DEPTH)
    np.testing.assert_allclose(image, expected, atol=1e-6)


def test_closer_obstacle_gives_larger_value():
    depth = uniform_depth(12.0)
    depth[:, :128] = 4.0  # an obstacle on the left half of the image
    image, _ = preprocess_depth(depth, H, W, MAX_DEPTH)
    assert image[:, :50].mean() > image[:, -50:].mean()


def test_depth_values_stay_in_unit_range():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.0, 60.0, CAMERA_SHAPE).astype(np.float32)
    image, _ = preprocess_depth(depth, H, W, MAX_DEPTH)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_min_distance_is_taken_in_meters_before_clipping():
    """Used for the crash check and the obstacle penalty, so it must not be capped at max_depth."""
    _, min_distance = preprocess_depth(uniform_depth(40.0), H, W, MAX_DEPTH)
    assert min_distance == pytest.approx(40.0)

    depth = uniform_depth(20.0)
    depth[60:80, 100:140] = 2.5
    _, min_distance = preprocess_depth(depth, H, W, MAX_DEPTH)
    assert min_distance == pytest.approx(2.5)


# ---------------------------------------------------------------- stacking
def test_stacked_observation_layout():
    depth = np.full((H, W), 0.25, dtype=np.float32)
    state = np.array([0.9, 0.5, 0.1, 0.5, 0.75, 0.25], dtype=np.float32)

    obs = stack_depth_and_state(depth, state)

    assert obs.shape == (H, W, 2)
    assert obs.dtype == np.float32
    np.testing.assert_array_equal(obs[:, :, 0], depth)
    np.testing.assert_array_equal(obs[0, :6, 1], state)
    assert np.count_nonzero(obs[1:, :, 1]) == 0  # only row 0 of channel 1 carries the state
    assert np.count_nonzero(obs[0, 6:, 1]) == 0


def test_state_outside_unit_range_is_kept_exactly_and_does_not_touch_depth():
    """Regression: the float32 path must keep values like 1.3 or -0.2 and leave the depth unchanged."""
    depth = np.full((H, W), 1 - 10 / 15, dtype=np.float32)
    state = np.array([1.3, 0.5, -0.2, 0.5, 0.5, 0.5], dtype=np.float32)

    obs = stack_depth_and_state(depth, state)

    np.testing.assert_array_equal(obs[0, :6, 1], state)
    np.testing.assert_array_equal(obs[:, :, 0], depth)


# --------------------------------------------------------------------- state
@pytest.mark.parametrize(
    "angle, expected",
    [(1.0, 1.0), (math.pi, math.pi), (3.5, 3.5 - 2 * math.pi), (-3.5, -3.5 + 2 * math.pi), (-math.pi, -math.pi)],
)
def test_wrap_to_pi(angle, expected):
    assert wrap_to_pi(angle) == pytest.approx(expected)


@pytest.mark.parametrize("distance, expected", [(50.0, 1.0), (25.0, 0.5), (0.0, 0.0), (60.0, 1.2)])
def test_normalize_distance_is_not_clipped(distance, expected):
    assert normalize_distance(distance, 50.0) == pytest.approx(expected)


@pytest.mark.parametrize("angle, expected", [(0.0, 0.5), (math.pi, 1.0), (-math.pi, 0.0), (math.pi / 2, 0.75)])
def test_normalize_angle(angle, expected):
    assert normalize_angle(angle) == pytest.approx(expected)


@pytest.mark.parametrize("v_xy, expected", [(0.5, 0.0), (5.0, 1.0), (2.75, 0.5), (0.0, -1 / 9)])
def test_normalize_speed(v_xy, expected):
    """At standstill (0 m/s) the value is -0.111, because v_xy_min is 0.5 m/s."""
    assert normalize_speed(v_xy, 0.5, 5.0) == pytest.approx(expected)


@pytest.mark.parametrize("value, expected", [(0.0, 0.5), (1.0, 1.0), (-1.0, 0.0), (0.5, 0.75)])
def test_normalize_symmetric(value, expected):
    assert normalize_symmetric(value, 1.0) == pytest.approx(expected)


def test_normalize_symmetric_yaw_rate_in_radians():
    yaw_rate_max_rad = math.radians(60)
    assert normalize_symmetric(math.radians(30), yaw_rate_max_rad) == pytest.approx(0.75)


def test_normalize_last_action():
    np.testing.assert_allclose(normalize_last_action(np.array([-1.0, 0.0, 1.0, 0.5])), [0.0, 0.5, 1.0, 0.75])


# -------------------------------------------------------------------- purity
def test_observation_module_imports_without_hardware_packages():
    code = (
        "import sys\n"
        "for name in ('airsim', 'keyboard', 'PyQt5', 'msgpackrpc'):\n"
        "    sys.modules[name] = None\n"
        "import gym_env.core.observation\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
