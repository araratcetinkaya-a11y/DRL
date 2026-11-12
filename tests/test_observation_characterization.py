"""Characterization test: get_obs_image returns the same observation as the thesis code.

EXPECTED was recorded by running the frozen thesis code (tag thesis-final) on the fake AirSim
client before get_obs_image and _get_state_feature were changed to use
gym_env/core/observation.py. The scene is a depth image that goes from 1 m (left) to 30 m (right)
with an obstacle at 2.5 m in the middle. After reset, the drone takes five steps.

For each observation the test compares: the 6 state values, 8 depth pixels, the sum of the depth
channel and the nearest-obstacle distance.
"""
import numpy as np
import pytest

SAMPLE_PIXELS = [(0, 0), (0, 199), (35, 100), (69, 0), (30, 70), (40, 80), (10, 150), (60, 190)]
ACTIONS = [(1.0, 0.5), (1.0, 0.0), (0.5, -1.0), (1.0, 1.0), (-1.0, 0.0)]

DEPTH_SAMPLES = [0.9322718977928162, 0.0, 0.8333333134651184, 0.9322718977928162,
                 0.25295162200927734, 0.8333333134651184, 0.0, 0.0]
DEPTH_SUM = 3651.640828728676
MIN_DISTANCE = 1.0159215927124023

EXPECTED_STATES = [  # after reset, then after each action
    [1.0, 0.33362412452697754, -0.1111111119389534, 0.5, 0.5, 0.5],
    [0.9994987845420837, 0.32515305280685425, 0.0, 0.75, 1.0, 0.75],
    [0.998681366443634, 0.3248975872993469, 0.08888889104127884, 0.5, 1.0, 0.5],
    [0.9976857900619507, 0.3412514328956604, 0.13333334028720856, 0.0, 0.75, 0.0],
    [0.9960616230964661, 0.32418209314346313, 0.2222222238779068, 1.0, 1.0, 1.0],
    [0.9950748682022095, 0.3238677382469177, 0.13333334028720856, 0.5, 0.0, 0.5],
]


def scene_depth_map(height=144, width=256):
    """1 m on the left edge to 30 m on the right edge, obstacle at 2.5 m in the middle."""
    columns = np.arange(width, dtype=np.float32)
    depth = np.tile(1.0 + 29.0 * columns / (width - 1), (height, 1)).astype(np.float32)
    depth[50:90, 100:140] = 2.5
    return depth


def check_observation(raw_env, expected_state, label):
    obs = raw_env.get_obs_image()

    assert obs.shape == (70, 200, 2), label
    assert obs.dtype == np.float32, label
    assert [float(x) for x in obs[0, :6, 1]] == pytest.approx(expected_state, rel=1e-6, abs=1e-7), label
    assert [float(obs[r, c, 0]) for r, c in SAMPLE_PIXELS] == pytest.approx(DEPTH_SAMPLES, abs=1e-6), label
    assert float(obs[:, :, 0].astype(np.float64).sum()) == pytest.approx(DEPTH_SUM, abs=1e-2), label
    assert float(raw_env.min_distance_to_obstacles) == pytest.approx(MIN_DISTANCE, abs=1e-5), label


def test_get_obs_image_matches_thesis_code(env, client):
    client.depth_map = scene_depth_map()
    env.reset()
    raw = env.unwrapped

    check_observation(raw, EXPECTED_STATES[0], "after reset")
    for step, action in enumerate(ACTIONS, start=1):
        env.step(np.array(action, dtype=np.float32))
        check_observation(raw, EXPECTED_STATES[step], f"after step {step}, action {action}")
