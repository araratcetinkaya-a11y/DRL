"""Observation pipeline of the thesis environment: depth preprocessing and state normalization.

This is the float32 path that the thesis models were trained with (`get_obs_image`):
- the depth image is resized, clipped to `max_depth_m` and normalized so that near = 1, far = 0;
- the normalized state is written into row 0 of a second channel;
- the result has shape (height, width, 2) and dtype float32.

Since Phase 3 the environment returns a Dict observation instead (`quantize_depth` and
`build_observation` below): the depth as uint8 (1, H, W) and the state as float32, stored
separately. `preprocess_depth` and `stack_depth_and_state` are kept for the float32 layout.

The state values are not clipped. Most of them stay in [0, 1], but some can leave it, for example
the speed is below 0 while the drone is slower than v_xy_min, and the distance is above 1 when the
drone is farther from the goal than at the start.
"""
import math

import cv2
import numpy as np


# ------------------------------------------------------------------- depth image
def preprocess_depth(depth_m, height, width, max_depth_m):
    """Depth image in meters -> (processed image, distance to the nearest obstacle).

    The processed image has shape (height, width), dtype float32 and values in [0, 1]:
    1 = touching an obstacle, 0 = at max_depth_m or farther. The nearest-obstacle distance is
    taken from the resized image in meters, before clipping.
    """
    resized = cv2.resize(depth_m, (width, height))
    min_distance_m = resized.min()
    normalized = np.clip(resized, 0, max_depth_m) / max_depth_m
    return (1.0 - normalized).astype(np.float32), min_distance_m


def stack_depth_and_state(depth_processed, state):
    """Depth image (H, W) + state vector -> observation (H, W, 2), float32.

    Channel 0 is the depth image. Channel 1 is zero except row 0, which starts with the state.
    """
    state_channel = np.zeros(depth_processed.shape, dtype=np.float32)
    state_channel[0, :len(state)] = state
    return np.stack([depth_processed.astype(np.float32), state_channel], axis=-1)


def quantize_depth(depth_processed):
    """Processed depth image (H, W) in [0, 1] -> uint8 array (1, H, W) in [0, 255], rounded.

    Rounding (not truncation): 0.3333 -> 85, as 0.3333 * 255 = 84.99. The resolution is
    max_depth / 255, about 6 cm with max_depth = 15 m. SB3 divides by 255 again before the CNN.
    """
    scaled = np.rint(np.clip(depth_processed, 0.0, 1.0) * 255.0)
    return scaled.astype(np.uint8)[np.newaxis, :, :]


def build_observation(depth_uint8, state):
    """Dict observation: {"depth": uint8 (1, H, W), "state": float32 (n,)}. The two never mix."""
    return {
        "depth": depth_uint8,
        "state": np.asarray(state, dtype=np.float32),
    }


# ------------------------------------------------------------------------ state
def wrap_to_pi(angle_rad):
    """Bring an angle back to [-pi, pi] with one correction of 2*pi (enough for |angle| < 3*pi)."""
    if angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    elif angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad


def normalize_distance(distance, goal_distance):
    """1 at the start of an episode, 0 at the goal; above 1 if the drone moves away."""
    return distance / goal_distance


def normalize_angle(angle_rad):
    """[-pi, pi] -> [0, 1]; 0.5 means pointing at the goal."""
    return angle_rad / math.pi / 2 + 0.5


def normalize_speed(v_xy, v_xy_min, v_xy_max):
    """[v_xy_min, v_xy_max] -> [0, 1]; negative while the drone is slower than v_xy_min."""
    return (v_xy - v_xy_min) / (v_xy_max - v_xy_min)


def normalize_symmetric(value, value_max):
    """[-value_max, value_max] -> [0, 1]; used for the yaw rate, the vertical speed and height."""
    return value / value_max / 2 + 0.5


def normalize_last_action(last_action):
    """Actions in [-1, 1] -> [0, 1]."""
    return (last_action + 1.0) / 2.0
