"""Minimal stand-in for the `airsim` package, used only by the tests.

It implements just the parts of the AirSim API that `gym_env` calls, with simple
kinematics instead of a physics engine:
- `moveByVelocityZAsync` stores the commanded world velocity and yaw rate;
- `simContinueForTime(dt)` moves the vehicle with that velocity for `dt` seconds;
- `simGetImages` returns a depth image filled with `client.depth_m` meters.

Tests can change `client.depth_m` or `client.has_collided` to create situations
such as an obstacle in front of the drone or a crash.
"""
import math

import numpy as np


# ---------------------------------------------------------------- basic types
class Vector3r:
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0):
        self.x_val = float(x_val)
        self.y_val = float(y_val)
        self.z_val = float(z_val)


class Quaternionr:
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0):
        self.x_val = float(x_val)
        self.y_val = float(y_val)
        self.z_val = float(z_val)
        self.w_val = float(w_val)


class Pose:
    def __init__(self, position_val=None, orientation_val=None):
        self.position = position_val if position_val is not None else Vector3r()
        self.orientation = orientation_val if orientation_val is not None else Quaternionr()


def to_quaternion(pitch, roll, yaw):
    """Same formula as airsim.utils.to_quaternion (angles in radians)."""
    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)
    q = Quaternionr()
    q.w_val = t0 * t2 * t4 + t1 * t3 * t5
    q.x_val = t0 * t3 * t4 - t1 * t2 * t5
    q.y_val = t0 * t2 * t5 + t1 * t3 * t4
    q.z_val = t1 * t2 * t4 - t0 * t3 * t5
    return q


def to_eularian_angles(q):
    """Same formula as airsim.utils.to_eularian_angles; returns (pitch, roll, yaw)."""
    z, y, x, w = q.z_val, q.y_val, q.x_val, q.w_val
    ysqr = y * y
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)
    t2 = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)
    return (pitch, roll, yaw)


def list_to_2d_float_array(flst, width, height):
    return np.reshape(np.asarray(flst, np.float32), (height, width))


# ------------------------------------------------------------ request objects
class ImageType:
    Scene = 0
    DepthPlanar = 1
    DepthPerspective = 2
    DepthVis = 3


class ImageRequest:
    def __init__(self, camera_name, image_type, pixels_as_float=False, compress=True):
        self.camera_name = camera_name
        self.image_type = image_type
        self.pixels_as_float = pixels_as_float
        self.compress = compress


class DrivetrainType:
    MaxDegreeOfFreedom = 0
    ForwardOnly = 1


class YawMode:
    def __init__(self, is_rate=True, yaw_or_rate=0.0):
        self.is_rate = is_rate
        self.yaw_or_rate = yaw_or_rate


class PIDGains:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd


class VelocityControllerGains:
    def __init__(self, x_gains=None, y_gains=None, z_gains=None):
        self.x_gains, self.y_gains, self.z_gains = x_gains, y_gains, z_gains


class AngleLevelControllerGains:
    def __init__(self, roll_gains=None, pitch_gains=None, yaw_gains=None):
        self.roll_gains, self.pitch_gains, self.yaw_gains = roll_gains, pitch_gains, yaw_gains


# ------------------------------------------------------------ result objects
class _Future:
    """Returned by the *Async calls; `join()` does nothing."""

    def join(self):
        return None


class _ImageResponse:
    def __init__(self, width, height, depth_m):
        self.width = width
        self.height = height
        self.image_data_float = [float(depth_m)] * (width * height)
        self.image_data_uint8 = b""


class _CollisionInfo:
    def __init__(self, has_collided):
        self.has_collided = has_collided


class _Kinematics:
    def __init__(self, linear_velocity, angular_velocity):
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity


class _MultirotorState:
    def __init__(self, kinematics):
        self.kinematics_estimated = kinematics


# --------------------------------------------------------------------- client
class MultirotorClient:
    """Kinematic fake of airsim.MultirotorClient. Every API call is recorded in `calls`."""

    image_width = 256
    image_height = 144

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.depth_m = 10.0
        self.has_collided = False
        self.reset()

    # --- connection and setup
    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def confirmConnection(self):
        self._record("confirmConnection")

    def enableApiControl(self, enabled, vehicle_name=""):
        self._record("enableApiControl", enabled)

    def armDisarm(self, arm, vehicle_name=""):
        self._record("armDisarm", arm)

    def setVelocityControllerGains(self, gains, vehicle_name=""):
        self._record("setVelocityControllerGains")

    def setAngleLevelControllerGains(self, gains, vehicle_name=""):
        self._record("setAngleLevelControllerGains")

    def simConsoleExecute(self, command):
        self._record("simConsoleExecute", command)

    def simPrintLogMessage(self, message, message_param="", severity=0):
        pass

    # --- state
    def reset(self):
        self._position = Vector3r()
        self._yaw = 0.0
        self._velocity = Vector3r()
        self._yaw_rate_rad = 0.0
        self.paused = False
        self.sim_time = 0.0

    def simPause(self, is_paused):
        self.paused = bool(is_paused)

    def simGetVehiclePose(self, vehicle_name=""):
        position = Vector3r(self._position.x_val, self._position.y_val, self._position.z_val)
        return Pose(position, to_quaternion(0.0, 0.0, self._yaw))

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name=""):
        self._position = Vector3r(pose.position.x_val, pose.position.y_val, pose.position.z_val)
        self._yaw = to_eularian_angles(pose.orientation)[2]

    def getMultirotorState(self, vehicle_name=""):
        angular = Vector3r(0.0, 0.0, self._yaw_rate_rad)
        return _MultirotorState(_Kinematics(self._velocity, angular))

    def simGetCollisionInfo(self, vehicle_name=""):
        return _CollisionInfo(self.has_collided)

    def simGetImages(self, requests, vehicle_name=""):
        return [_ImageResponse(self.image_width, self.image_height, self.depth_m) for _ in requests]

    # --- motion
    def moveToZAsync(self, z, velocity, *args, **kwargs):
        self._position.z_val = float(z)
        return _Future()

    def moveByVelocityZAsync(self, vx, vy, z, duration, drivetrain=None, yaw_mode=None, vehicle_name=""):
        self._record("moveByVelocityZAsync", vx, vy, z, duration, yaw_mode=yaw_mode)
        self._velocity = Vector3r(vx, vy, 0.0)
        self._position.z_val = float(z)
        self._yaw_rate_rad = math.radians(yaw_mode.yaw_or_rate) if yaw_mode is not None else 0.0
        return _Future()

    def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain=None, yaw_mode=None, vehicle_name=""):
        self._record("moveByVelocityAsync", vx, vy, vz, duration, yaw_mode=yaw_mode)
        self._velocity = Vector3r(vx, vy, vz)
        self._yaw_rate_rad = math.radians(yaw_mode.yaw_or_rate) if yaw_mode is not None else 0.0
        return _Future()

    def simContinueForTime(self, seconds):
        """Advance the fake simulation: constant velocity and yaw rate for `seconds`."""
        self._record("simContinueForTime", seconds)
        self._position.x_val += self._velocity.x_val * seconds
        self._position.y_val += self._velocity.y_val * seconds
        self._position.z_val += self._velocity.z_val * seconds
        self._yaw += self._yaw_rate_rad * seconds
        self.sim_time += seconds
