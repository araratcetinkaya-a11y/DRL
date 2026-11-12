"""Target behavior of the Dict observation (roadmap Phase 3). Written before the implementation.

Every test here is marked xfail(strict=True): it fails today, because the code still uses the
2-channel observation. PR 3b implements the Dict observation and removes the markers. If a test
starts passing before that, strict xfail turns it into a failure.

Target:
    observation_space = Dict(
        depth = Box(0, 255, (1, H, W), uint8),     # near obstacle = 255, max_depth or farther = 0
        state = Box(-inf, inf, (n,), float32),     # n = dynamic_model.state_feature_length
    )
- the depth is quantized by rounding (not truncation) from the float32 image of preprocess_depth;
- the state is stored exactly as float32 and never touches the depth image;
- SB3 uses MultiInputPolicy and DictReplayBuffer, and divides the uint8 depth by 255 itself;
- CNN_GAP_new reads obs["depth"] and obs["state"]; its layers are unchanged.
"""
import gym
import numpy as np
import pytest
import torch as th
from gym import spaces

P3 = "Phase 3 (PR 3b): Dict observation not implemented yet"
xfail_p3 = pytest.mark.xfail(reason=P3, strict=True)

H, W, N_STATE = 70, 200, 6


def dict_space(height=H, width=W, n_state=N_STATE):
    return spaces.Dict({
        "depth": spaces.Box(0, 255, (1, height, width), np.uint8),
        "state": spaces.Box(-np.inf, np.inf, (n_state,), np.float32),
    })


# ---------------------------------------------------------- pure functions
@xfail_p3
def test_quantize_depth_dtype_shape_and_range():
    from gym_env.core.observation import quantize_depth

    image = np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape(H, W)
    depth = quantize_depth(image)
    assert depth.dtype == np.uint8
    assert depth.shape == (1, H, W)
    assert depth.min() == 0 and depth.max() == 255


@xfail_p3
@pytest.mark.parametrize("value, expected", [(1.0, 255), (0.0, 0), (1 - 10 / 15, 85), (0.5, 128), (0.499, 127)])
def test_quantize_depth_rounds_instead_of_truncating(value, expected):
    """0.3333 * 255 = 84.99 must become 85 (the old uint8 cast gave 84)."""
    from gym_env.core.observation import quantize_depth

    depth = quantize_depth(np.full((H, W), value, dtype=np.float32))
    assert np.all(depth == expected)


@xfail_p3
def test_build_observation_keeps_state_exact_and_separate_from_depth():
    """Regression for Q1: values like 1.3 or -0.2 must neither change nor affect the depth."""
    from gym_env.core.observation import build_observation

    depth = np.full((1, H, W), 85, dtype=np.uint8)
    state = np.array([1.3, 0.5, -0.2, 0.5, 0.5, 0.5], dtype=np.float32)

    obs = build_observation(depth, state)

    assert set(obs) == {"depth", "state"}
    np.testing.assert_array_equal(obs["depth"], depth)
    assert obs["state"].dtype == np.float32
    np.testing.assert_array_equal(obs["state"], state)


# ------------------------------------------------------------- environment
@xfail_p3
def test_env_observation_space_is_dict(env):
    space = env.observation_space
    n_state = env.unwrapped.dynamic_model.state_feature_length

    assert isinstance(space, spaces.Dict)
    assert space["depth"].shape == (1, H, W)
    assert space["depth"].dtype == np.uint8
    assert (space["depth"].low.min(), space["depth"].high.max()) == (0, 255)
    assert space["state"].shape == (n_state,)
    assert space["state"].dtype == np.float32


@xfail_p3
def test_reset_and_step_observations_are_in_declared_space(env, client):
    client.depth_m = 10.0
    obs = env.reset()
    assert env.observation_space.contains(obs)
    assert np.all(obs["depth"] == 85)  # 10 m with max_depth 15 m, rounded

    for action in [(1.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]:
        obs, _, _, _ = env.step(np.array(action, dtype=np.float32))
        assert env.observation_space.contains(obs)


@xfail_p3
def test_state_above_one_does_not_change_depth(env, client):
    """The case that the old uint8 block got wrong (depth collapsed to 0)."""
    client.depth_m = 10.0
    env.reset()
    env.unwrapped.dynamic_model.goal_distance = 10.0  # distance_norm = 5
    obs = env.unwrapped.get_obs()
    assert obs["state"][0] > 1.0
    assert np.all(obs["depth"] == 85)


@xfail_p3
def test_camera_error_fallback_matches_declared_space(env, monkeypatch):
    from gym_env.envs.airsim_env import CameraError

    env.reset()

    def broken_camera():
        raise CameraError("simulated camera failure")

    monkeypatch.setattr(env.unwrapped, "get_depth_image", broken_camera)
    obs, _, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert done and info["camera_error"]
    # Today the fallback is np.zeros((70, 200, 2)): it happens to match the old (wrong) Box space,
    # but not the shape of a normal observation. The target is a Dict like every other observation.
    assert isinstance(obs, dict)
    assert env.observation_space.contains(obs)


@xfail_p3
def test_dummy_vec_env_returns_dict_observations(thesis_cfg):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from tests.conftest import build_env

    obs = DummyVecEnv([lambda: build_env(thesis_cfg)]).reset()
    assert obs["depth"].shape == (1, 1, H, W)
    assert obs["state"].shape == (1, N_STATE)


# --------------------------------------------------------------------- CNN
def test_cnn_module_can_be_imported():
    """Not xfail: catches a missing dependency (the module imports torchvision) in CI."""
    from scripts.utils.custom_policy_sb3 import CNN_GAP_new  # noqa: F401


def cnn_class():
    from scripts.utils.custom_policy_sb3 import CNN_GAP_new

    return CNN_GAP_new


@xfail_p3
def test_cnn_output_is_cnn_features_followed_by_unchanged_state():
    extractor = cnn_class()(dict_space())
    batch = {
        "depth": th.rand(4, 1, H, W),  # SB3 passes the depth already divided by 255
        "state": th.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 4),
    }

    features = extractor(batch)

    assert features.shape == (4, 128 + N_STATE)
    assert extractor.features_dim == 128 + N_STATE
    th.testing.assert_close(features[:, -N_STATE:], batch["state"])


@xfail_p3
def test_cnn_layers_are_unchanged_at_thesis_resolution():
    """Same layer shapes as the thesis model (recorded from the frozen code), so weights could be ported."""
    expected = {
        "conv1.weight": (8, 1, 5, 5), "conv1.bias": (8,),
        "conv2.weight": (8, 8, 5, 5), "conv2.bias": (8,),
        "conv3.weight": (8, 8, 3, 3), "conv3.bias": (8,),
    }
    extractor = cnn_class()(dict_space())
    assert {k: tuple(v.shape) for k, v in extractor.state_dict().items()} == expected


@xfail_p3
@pytest.mark.parametrize("height, width", [(80, 200), (64, 64)])
def test_cnn_works_for_other_input_sizes(height, width):
    extractor = cnn_class()(dict_space(height, width))
    features = extractor({"depth": th.rand(2, 1, height, width), "state": th.zeros(2, N_STATE)})
    assert features.shape == (2, extractor.features_dim)


# ------------------------------------------------------------ SB3 smoke test
class TinyDictEnv(gym.Env):
    """Random observations with the target Dict spaces, so SAC can run without AirSim."""

    observation_space = dict_space()
    action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.t = 0

    def _obs(self):
        return {
            "depth": self.rng.integers(0, 256, (1, H, W), dtype=np.uint8),
            "state": self.rng.uniform(-0.2, 1.3, N_STATE).astype(np.float32),
        }

    def reset(self):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        return self._obs(), float(-np.abs(action).sum()), self.t >= 50, {}


def make_sac(buffer_size=300):
    from stable_baselines3 import SAC

    return SAC(
        "MultiInputPolicy", TinyDictEnv(),
        buffer_size=buffer_size, learning_starts=50, batch_size=16, train_freq=1, gradient_steps=1,
        policy_kwargs=dict(features_extractor_class=cnn_class(), net_arch=[32, 32]),
        device="cpu", seed=0,
    )


@xfail_p3
def test_sac_multi_input_policy_trains_and_stores_depth_as_uint8():
    from stable_baselines3.common.buffers import DictReplayBuffer

    model = make_sac()
    seen = {}

    def record_input(_module, inputs):
        seen["depth_max"] = float(inputs[0]["depth"].max())

    model.actor.features_extractor.register_forward_pre_hook(record_input)
    model.learn(total_timesteps=200)

    buffer = model.replay_buffer
    assert isinstance(buffer, DictReplayBuffer)
    assert buffer.observations["depth"].dtype == np.uint8
    assert buffer.observations["state"].dtype == np.float32
    assert 0.0 < seen["depth_max"] <= 1.0  # SB3 divided by 255 exactly once


@xfail_p3
def test_replay_buffer_memory_per_transition():
    """obs + next_obs: 2 x (70*200 bytes + 6*4 bytes) = 28,048 bytes (old float32 layout: 224,000)."""
    buffer = make_sac(buffer_size=1000).replay_buffer
    bytes_per_transition = sum(
        arr.nbytes for arr in list(buffer.observations.values()) + list(buffer.next_observations.values())
    ) / buffer.buffer_size

    assert bytes_per_transition == 2 * (H * W + N_STATE * 4)
    assert 224_000 / bytes_per_transition > 7.9  # about 8x less than the thesis layout
