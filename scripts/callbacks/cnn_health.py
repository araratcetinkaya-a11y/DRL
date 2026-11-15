"""CNN health monitor: detects a dead image network during training.

Background: in the run "dev-dict-obs-full" (November 2026) all units of conv3 stopped firing between
step 30,000 and 35,123 ("dead ReLU"). From then on the 128 image features were zero, the CNN received
no gradient and the agent could not see obstacles; this was only noticed after 360,000 steps.

Every `check_freq` steps this callback
  1. takes a batch of depth images from the replay buffer,
  2. runs them through conv1, conv2, conv3 and the pooling layer of the feature extractor,
  3. logs how many units are active in each layer (shown in W&B under "cnn_health/..."),
  4. stops the training if the image features stay all-zero for `patience` checks in a row.

The check does not change the training: it only reads the weights and the replay buffer, uses its own
random generator (so the random numbers of SB3, numpy and torch are not touched) and runs without
gradients.
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback

LOG_PREFIX = "cnn_health/"


def cnn_layer_stats(extractor, depth01):
    """Activity of each CNN layer for a batch of depth images.

    :param extractor: the feature extractor (CNN_GAP_new); needs conv1, conv2, conv3 and pool
    :param depth01: float tensor (B, 1, H, W) with values in [0, 1], as SB3 gives it to the extractor
    :return: dict of floats
        convN_active_frac     share of the outputs of layer N that are above zero (0..1)
        convN_alive_channels  number of channels of layer N with at least one output above zero
        conv3_preact_mean/max value of conv3 before the ReLU (all <= 0 means the layer is dead)
        conv3_bias_mean       mean bias of conv3
        features_nonzero_frac share of the pooled image features that are above zero (0..1)
        features_mean         mean of the pooled image features
    """
    stats = {}
    with th.no_grad():
        x = (depth01 - 0.5) * 2  # same normalization as CNN_GAP_new._cnn
        for index, conv in enumerate((extractor.conv1, extractor.conv2, extractor.conv3), start=1):
            pre_activation = conv(x)
            x = F.relu(pre_activation)
            active = x > 0
            stats[f"conv{index}_active_frac"] = float(active.float().mean())
            stats[f"conv{index}_alive_channels"] = float(active.transpose(0, 1).flatten(1).any(dim=1).sum())
        stats["conv3_preact_mean"] = float(pre_activation.mean())
        stats["conv3_preact_max"] = float(pre_activation.max())
        stats["conv3_bias_mean"] = float(extractor.conv3.bias.mean())
        features = th.flatten(extractor.pool(x), start_dim=1)
        stats["features_nonzero_frac"] = float((features > 0).float().mean())
        stats["features_mean"] = float(features.mean())
    return stats


def sample_depth_batch(replay_buffer, batch_size, rng):
    """Random depth images from a DictReplayBuffer as a uint8 array (B, 1, H, W); None if it is empty.

    Reads the stored array directly with its own random generator `rng`, so the random numbers that
    SB3 uses for its training batches stay the same with and without this monitor.
    """
    filled = replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos
    if filled < 1:
        return None
    indices = rng.integers(0, filled, size=min(batch_size, filled))
    return replay_buffer.observations["depth"][indices, 0]


class DeadCnnDetector:
    """Decides when the CNN counts as dead: `patience` checks in a row without any non-zero feature."""

    def __init__(self, patience=3):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.zero_checks = 0
        self.dead_since_step = None

    def update(self, features_nonzero_frac, step):
        """Returns True once the CNN is dead."""
        if features_nonzero_frac > 0:
            self.zero_checks = 0
            self.dead_since_step = None
            return False
        if self.zero_checks == 0:
            self.dead_since_step = step
        self.zero_checks += 1
        return self.zero_checks >= self.patience


class CnnHealthCallback(BaseCallback):
    """Logs the CNN activity and (optionally) stops the training when the CNN is dead.

    :param check_freq: check every this many environment steps
    :param batch_size: number of depth images per check
    :param patience: number of all-zero checks in a row before the CNN counts as dead
    :param stop_when_dead: stop the training when the CNN is dead (otherwise only warn once)
    :param rng_seed: seed of the monitor's own random generator
    """

    def __init__(self, check_freq=500, batch_size=256, patience=3, stop_when_dead=True, rng_seed=12345, verbose=1):
        super(CnnHealthCallback, self).__init__(verbose)
        if check_freq < 1:
            raise ValueError("check_freq must be at least 1")
        self.check_freq = check_freq
        self.batch_size = batch_size
        self.stop_when_dead = stop_when_dead
        self.detector = DeadCnnDetector(patience)
        self.last_stats = None
        self.is_dead = False
        self._rng = np.random.default_rng(rng_seed)
        self._warned = False

    def _extractor(self):
        return self.model.actor.features_extractor

    def check(self):
        """Run one check on a replay-buffer batch. Returns the stats, or None if the buffer is empty."""
        depth = sample_depth_batch(self.model.replay_buffer, self.batch_size, self._rng)
        if depth is None:
            return None
        extractor = self._extractor()
        device = next(extractor.parameters()).device
        depth01 = th.as_tensor(depth, device=device).float() / 255.0
        return cnn_layer_stats(extractor, depth01)

    def _on_step(self):
        if self.n_calls % self.check_freq != 0:
            return True
        stats = self.check()
        if stats is None:
            return True
        self.last_stats = stats
        for key, value in stats.items():
            self.logger.record(LOG_PREFIX + key, value)

        # Before learning starts the weights do not change, so the verdict only counts afterwards
        learning = self.num_timesteps > getattr(self.model, "learning_starts", 0)
        self.is_dead = learning and self.detector.update(stats["features_nonzero_frac"], self.num_timesteps)
        self.logger.record(LOG_PREFIX + "dead", float(self.is_dead))

        if self.verbose >= 1:
            print("[CNN health] step {:>7d} | active conv1 {:5.1f}%  conv2 {:5.1f}%  conv3 {:5.1f}% | "
                  "non-zero features {:5.1f}% | conv3 pre-activation max {:+.4f}".format(
                      self.num_timesteps, 100 * stats["conv1_active_frac"], 100 * stats["conv2_active_frac"],
                      100 * stats["conv3_active_frac"], 100 * stats["features_nonzero_frac"],
                      stats["conv3_preact_max"]))

        if self.is_dead and not self._warned:
            self._warned = True
            print("\n" + "!" * 78)
            print("[CNN health] THE IMAGE NETWORK IS DEAD: all image features have been zero since step "
                  "{} ({} checks in a row).".format(self.detector.dead_since_step, self.detector.zero_checks))
            print("[CNN health] The agent cannot see obstacles any more and the CNN gets no gradient, "
                  "so it cannot recover.")
            print("[CNN health] " + ("Stopping the training." if self.stop_when_dead
                                      else "stop_when_dead is off: the training continues."))
            print("!" * 78 + "\n")
        if self.is_dead and self.stop_when_dead:
            return False
        return True


def make_cnn_health_callback(cfg):
    """Build the callback from the optional [CNN_HEALTH] section of the config.

    Defaults (also used when the section is missing): enabled, check every 500 steps on 256 images,
    stop after 3 all-zero checks in a row. Returns None when it is disabled or when the policy has
    no CNN (policy_name = mlp).

        [CNN_HEALTH]
        enabled = true
        check_freq = 500
        batch_size = 256
        patience = 3
        stop_when_dead = true
    """
    section = "CNN_HEALTH"

    def option(name, getter, default):
        return getter(section, name) if cfg.has_option(section, name) else default

    if not option("enabled", cfg.getboolean, True):
        return None
    if cfg.get("options", "policy_name") == "mlp":
        return None
    return CnnHealthCallback(
        check_freq=option("check_freq", cfg.getint, 500),
        batch_size=option("batch_size", cfg.getint, 256),
        patience=option("patience", cfg.getint, 3),
        stop_when_dead=option("stop_when_dead", cfg.getboolean, True),
    )
