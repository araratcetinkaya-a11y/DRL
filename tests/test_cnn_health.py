"""CNN health monitor (scripts/callbacks/cnn_health.py).

The monitor was added after the run "dev-dict-obs-full", in which conv3 of the image network died
(all outputs zero for every image) shortly after learning started and nobody noticed for 360,000 steps.
It must (1) tell a living CNN from a dead one, (2) stop the training only after learning has started
and only after `patience` all-zero checks in a row, and (3) not change the training itself.
"""
import sys
import types
from configparser import ConfigParser

import numpy as np
import pytest
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictReplayBuffer

from scripts.callbacks.cnn_health import (
    LOG_PREFIX,
    CnnHealthCallback,
    DeadCnnDetector,
    cnn_layer_stats,
    make_cnn_health_callback,
    sample_depth_batch,
)

H, W, N_STATE = 70, 200, 6


def dict_space():
    return spaces.Dict({
        "depth": spaces.Box(0, 255, (1, H, W), np.uint8),
        "state": spaces.Box(-np.inf, np.inf, (N_STATE,), np.float32),
    })


def make_extractor(seed=0):
    from scripts.utils.custom_policy_sb3 import CNN_GAP_new

    th.manual_seed(seed)
    return CNN_GAP_new(dict_space())


def kill_conv3(extractor):
    """Make conv3 output a negative value for every input, as in the failed run."""
    with th.no_grad():
        extractor.conv3.weight.zero_()
        extractor.conv3.bias.fill_(-1.0)


def random_depth(batch=16, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(batch, 1, H, W), dtype=np.uint8)


def as_input(depth_uint8):
    return th.as_tensor(depth_uint8).float() / 255.0


def filled_buffer(n_stored, buffer_size=40):
    action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
    buffer = DictReplayBuffer(buffer_size, dict_space(), action_space, device="cpu")
    rng = np.random.default_rng(1)
    for _ in range(n_stored):
        obs = {"depth": rng.integers(1, 256, size=(1, 1, H, W), dtype=np.uint8),
               "state": np.zeros((1, N_STATE), np.float32)}
        buffer.add(obs, obs, np.zeros((1, 2), np.float32), np.zeros(1), np.zeros(1), [{}])
    return buffer


class RecordingLogger:
    def __init__(self):
        self.values = {}

    def record(self, key, value, exclude=None):
        self.values[key] = value


def fake_model(extractor, buffer, learning_starts=100):
    return types.SimpleNamespace(
        actor=types.SimpleNamespace(features_extractor=extractor),
        replay_buffer=buffer,
        learning_starts=learning_starts,
        num_timesteps=0,
        logger=RecordingLogger(),
        get_env=lambda: None,
    )


def run_checks(callback, model, steps):
    """Call the callback once per step at the given timesteps; returns the step at which it asked to stop."""
    for step in steps:
        model.num_timesteps = step
        if callback.on_step() is False:
            return step
    return None


# ------------------------------------------------------------- layer statistics
def test_living_cnn_has_active_units_in_every_layer():
    stats = cnn_layer_stats(make_extractor(), as_input(random_depth()))

    for layer in (1, 2, 3):
        assert 0.0 < stats[f"conv{layer}_active_frac"] <= 1.0
        assert 1 <= stats[f"conv{layer}_alive_channels"] <= 8
    assert stats["features_nonzero_frac"] > 0.0
    assert stats["conv3_preact_max"] > 0.0


def test_dead_conv3_gives_all_zero_features():
    extractor = make_extractor()
    kill_conv3(extractor)

    stats = cnn_layer_stats(extractor, as_input(random_depth()))

    assert stats["conv1_active_frac"] > 0.0 and stats["conv2_active_frac"] > 0.0
    assert stats["conv3_active_frac"] == 0.0
    assert stats["conv3_alive_channels"] == 0
    assert stats["conv3_preact_max"] == pytest.approx(-1.0)
    assert stats["conv3_bias_mean"] == pytest.approx(-1.0)
    assert stats["features_nonzero_frac"] == 0.0
    assert stats["features_mean"] == 0.0


def test_stats_describe_the_features_the_policy_really_gets():
    extractor = make_extractor()
    depth = as_input(random_depth())

    stats = cnn_layer_stats(extractor, depth)
    with th.no_grad():
        features = extractor._cnn(depth)

    assert stats["features_nonzero_frac"] == pytest.approx(float((features > 0).float().mean()))
    assert stats["features_mean"] == pytest.approx(float(features.mean()))


def test_one_living_channel_is_counted():
    extractor = make_extractor()
    kill_conv3(extractor)
    with th.no_grad():
        extractor.conv3.bias[5] = 1.0  # channel 5 now always fires

    stats = cnn_layer_stats(extractor, as_input(random_depth()))

    assert stats["conv3_alive_channels"] == 1
    assert stats["conv3_active_frac"] == pytest.approx(1 / 8)
    assert stats["features_nonzero_frac"] == pytest.approx(1 / 8)


def test_stats_do_not_change_the_network_or_the_random_state():
    extractor = make_extractor()
    weights_before = {k: v.clone() for k, v in extractor.state_dict().items()}
    cached_layer_output = extractor.layer_3_out  # set by _cnn(); read by other tools
    torch_state, numpy_state = th.get_rng_state(), np.random.get_state()

    cnn_layer_stats(extractor, as_input(random_depth()))

    assert all(th.equal(v, weights_before[k]) for k, v in extractor.state_dict().items())
    assert th.equal(th.get_rng_state(), torch_state)
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_state))
    assert extractor.layer_3_out is cached_layer_output  # the monitor does not go through _cnn()


# ------------------------------------------------------------ replay buffer batch
def test_sample_from_empty_buffer_gives_none():
    assert sample_depth_batch(filled_buffer(0), 8, np.random.default_rng(0)) is None


def test_sample_reads_only_the_filled_part():
    buffer = filled_buffer(5)  # stored images have no zero pixel; the unused rows are all zero

    batch = sample_depth_batch(buffer, 64, np.random.default_rng(0))

    assert batch.shape == (5, 1, H, W) and batch.dtype == np.uint8
    assert batch.min() >= 1


def test_sample_from_full_buffer_uses_the_requested_batch_size():
    buffer = filled_buffer(45, buffer_size=40)
    assert buffer.full

    batch = sample_depth_batch(buffer, 16, np.random.default_rng(0))

    assert batch.shape == (16, 1, H, W)
    assert batch.min() >= 1


def test_sample_does_not_use_the_global_numpy_random_state():
    buffer = filled_buffer(10)
    state = np.random.get_state()

    sample_depth_batch(buffer, 8, np.random.default_rng(0))

    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), state))


# ------------------------------------------------------------------- detector
def test_detector_needs_patience_zero_checks_in_a_row():
    detector = DeadCnnDetector(patience=3)

    assert detector.update(0.0, step=500) is False
    assert detector.update(0.0, step=1000) is False
    assert detector.update(0.0, step=1500) is True
    assert detector.dead_since_step == 500


def test_detector_resets_when_a_feature_is_non_zero_again():
    detector = DeadCnnDetector(patience=2)

    assert detector.update(0.0, step=500) is False
    assert detector.update(0.01, step=1000) is False
    assert detector.dead_since_step is None
    assert detector.update(0.0, step=1500) is False
    assert detector.update(0.0, step=2000) is True
    assert detector.dead_since_step == 1500


def test_detector_rejects_patience_below_one():
    with pytest.raises(ValueError):
        DeadCnnDetector(patience=0)


# ------------------------------------------------------------------- callback
def make_callback(extractor, buffer, learning_starts=100, **kwargs):
    kwargs.setdefault("check_freq", 1)
    kwargs.setdefault("verbose", 0)
    callback = CnnHealthCallback(**kwargs)
    model = fake_model(extractor, buffer, learning_starts)
    callback.init_callback(model)
    return callback, model


def test_callback_never_stops_a_living_cnn():
    callback, model = make_callback(make_extractor(), filled_buffer(10), patience=2)

    assert run_checks(callback, model, range(101, 121)) is None
    assert callback.is_dead is False
    assert model.logger.values[LOG_PREFIX + "dead"] == 0.0
    assert model.logger.values[LOG_PREFIX + "features_nonzero_frac"] > 0.0


def test_callback_stops_a_dead_cnn_after_patience_checks():
    extractor = make_extractor()
    kill_conv3(extractor)
    callback, model = make_callback(extractor, filled_buffer(10), patience=3)

    assert run_checks(callback, model, range(101, 121)) == 103
    assert callback.is_dead is True
    assert callback.detector.dead_since_step == 101
    assert model.logger.values[LOG_PREFIX + "dead"] == 1.0
    assert model.logger.values[LOG_PREFIX + "features_nonzero_frac"] == 0.0
    assert model.logger.values[LOG_PREFIX + "conv3_alive_channels"] == 0


def test_callback_does_not_judge_before_learning_starts():
    extractor = make_extractor()
    kill_conv3(extractor)
    callback, model = make_callback(extractor, filled_buffer(10), learning_starts=100, patience=2)

    assert run_checks(callback, model, range(1, 101)) is None  # steps 1..100: still collecting data
    assert callback.is_dead is False
    assert run_checks(callback, model, range(101, 110)) == 102


def test_callback_can_warn_without_stopping(capsys):
    extractor = make_extractor()
    kill_conv3(extractor)
    callback, model = make_callback(extractor, filled_buffer(10), patience=2, stop_when_dead=False, verbose=1)

    assert run_checks(callback, model, range(101, 111)) is None
    assert callback.is_dead is True
    assert capsys.readouterr().out.count("THE IMAGE NETWORK IS DEAD") == 1  # warned once, not every check


def test_callback_checks_only_every_check_freq_steps():
    extractor = make_extractor()
    kill_conv3(extractor)
    callback, model = make_callback(extractor, filled_buffer(10), check_freq=5, patience=2)

    # checks happen on the 5th and 10th call (steps 105 and 110)
    assert run_checks(callback, model, range(101, 131)) == 110


def test_callback_waits_for_data_in_the_buffer():
    callback, model = make_callback(make_extractor(), filled_buffer(0))

    assert run_checks(callback, model, range(101, 104)) is None
    assert callback.last_stats is None
    assert model.logger.values == {}


def test_callback_rejects_check_freq_below_one():
    with pytest.raises(ValueError):
        CnnHealthCallback(check_freq=0)


# --------------------------------------------------------------------- config
def test_thesis_config_gets_the_monitor_with_default_settings(thesis_cfg):
    assert not thesis_cfg.has_section("CNN_HEALTH")

    callback = make_cnn_health_callback(thesis_cfg)

    assert isinstance(callback, CnnHealthCallback)
    assert (callback.check_freq, callback.batch_size, callback.detector.patience) == (500, 256, 3)
    assert callback.stop_when_dead is True


def test_monitor_settings_are_read_from_the_config(thesis_cfg):
    thesis_cfg.read_dict({"CNN_HEALTH": {
        "check_freq": "250", "batch_size": "64", "patience": "5", "stop_when_dead": "false"}})

    callback = make_cnn_health_callback(thesis_cfg)

    assert (callback.check_freq, callback.batch_size, callback.detector.patience) == (250, 64, 5)
    assert callback.stop_when_dead is False


def test_monitor_can_be_switched_off(thesis_cfg):
    thesis_cfg.read_dict({"CNN_HEALTH": {"enabled": "false"}})

    assert make_cnn_health_callback(thesis_cfg) is None


def test_no_monitor_for_a_policy_without_cnn(thesis_cfg):
    thesis_cfg.set("options", "policy_name", "mlp")

    assert make_cnn_health_callback(thesis_cfg) is None


# ----------------------------------------------------------------------- seed
@pytest.fixture
def thread_train(monkeypatch):
    try:
        import wandb  # noqa: F401
    except ImportError:
        fake = types.ModuleType("wandb")
        integration = types.ModuleType("wandb.integration")
        sb3 = types.ModuleType("wandb.integration.sb3")
        sb3.WandbCallback = object
        fake.integration, integration.sb3 = integration, sb3
        for name, module in [("wandb", fake), ("wandb.integration", integration), ("wandb.integration.sb3", sb3)]:
            monkeypatch.setitem(sys.modules, name, module)
    from scripts.utils import thread_train as module
    return module


def test_seed_is_the_thesis_value_when_the_config_has_none(thread_train, thesis_cfg):
    assert not thesis_cfg.has_option("DRL", "seed")
    assert thread_train.get_seed(thesis_cfg) == 0


def test_seed_is_read_from_the_config(thread_train):
    cfg = ConfigParser()
    cfg.read_dict({"DRL": {"seed": "7"}})
    assert thread_train.get_seed(cfg) == 7
