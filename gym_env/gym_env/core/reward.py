"""Reward terms of the thesis reward function (`compute_reward_final`, reward_type = reward_final).

Per step:  reward = distance progress - obstacle penalty - speed-near-obstacle penalty
                    - forward jerk penalty - yaw jerk penalty
At the end of an episode one terminal reward is added: reach +30, crash -25, outside -10
(checked in this order; a time-limit end adds nothing).

The coefficients are hard-coded in `AirsimGymEnv.__init__` (thesis values): SAFE_DISTANCE = 2 m,
k_obs_penalty = 0.5, k_speed_near_obs = 50, max_safe_speed_near_obs = 3.5 m/s,
k_jerk_fwd = 0.025, k_jerk_yaw = 0.03; distance coefficient 40 (500 in NH_center).

Note (docs/AUDIT.md, Q3): with the thesis config, crash_distance = SAFE_DISTANCE = 2 m, so the two
obstacle penalties are only non-zero in the step in which the drone crashes.
"""

REWARD_REACH = 30.0
REWARD_CRASH = -25.0
REWARD_OUTSIDE = -10.0


def distance_progress_reward(d_prev, d_now, goal_distance, coef=40.0):
    """Positive when the drone gets closer to the goal; scaled by the start distance."""
    return coef * (d_prev - d_now) / goal_distance


def obstacle_penalty(dist_to_obs, safe_distance, k):
    """Quadratic penalty inside the safety bubble: k * (safe_distance - dist)^2, else 0."""
    if dist_to_obs < safe_distance:
        return k * (safe_distance - dist_to_obs) ** 2
    return 0.0


def speed_near_obstacle_penalty(dist_to_obs, v_xy, safe_distance, max_safe_speed, k):
    """Inside the safety bubble, penalize the speed above max_safe_speed quadratically."""
    if dist_to_obs < safe_distance and v_xy > max_safe_speed:
        return k * (v_xy - max_safe_speed) ** 2
    return 0.0


def jerk_penalty(action, last_action, k):
    """Penalty for a change of one action component between two steps: k * (a - a_last)^2."""
    return k * (action - last_action) ** 2


def terminal_reward(reached, crashed, outside):
    """Reward added once when the episode ends. Reaching the goal has priority over a crash."""
    if reached:
        return REWARD_REACH
    if crashed:
        return REWARD_CRASH
    if outside:
        return REWARD_OUTSIDE
    return 0.0
