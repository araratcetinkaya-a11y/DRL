"""Action model of the thesis multirotor (2D navigation).

The agent outputs two values in [-1, 1]:
- action[0] changes the forward speed setpoint (incremental / delta control);
- action[1] sets the yaw rate directly (rate control).

The forward speed is then turned into a world-frame (NED) velocity command using the
current heading of the drone.

Action gain: the thesis models were trained with gain 1.0, i.e. the actions are used as they
are. After training, an evaluation experiment multiplied both actions by 1.1 ("Offset var");
that version of the code is what the `thesis-final` tag contains. The gain is now a parameter:
the default is the training behavior (1.0), and the experiment can be repeated by setting
`action_gain = 1.1` in the [multirotor] section of the config.
"""
import math

import numpy as np

DEFAULT_ACTION_GAIN = 1.0


def next_speed_setpoint(prev_v_xy, action_fwd, v_xy_delta_max, v_xy_min, v_xy_max, gain=DEFAULT_ACTION_GAIN):
    """Forward speed setpoint [m/s] after one step of delta control.

    The setpoint changes by at most `gain * v_xy_delta_max` per step and always stays inside
    [v_xy_min, v_xy_max]. With the thesis config: at most 0.4 m/s per step, between 0.5 and 5 m/s.
    """
    return np.clip(prev_v_xy + action_fwd * v_xy_delta_max * gain, v_xy_min, v_xy_max)


def yaw_rate_command_deg(action_yaw, yaw_rate_max_deg, gain=DEFAULT_ACTION_GAIN):
    """Yaw rate command [deg/s], proportional to action[1] (no clipping).

    With gain 1.0 it stays within +/-yaw_rate_max_deg (60 deg/s in the thesis config).
    """
    return action_yaw * yaw_rate_max_deg * gain


def body_to_world_velocity(v_xy, yaw_rad):
    """Forward speed along the current heading -> (vx, vy) in the world (NED) frame."""
    return v_xy * math.cos(yaw_rad), v_xy * math.sin(yaw_rad)
