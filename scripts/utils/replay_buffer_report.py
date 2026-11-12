"""Startup check for training: how much memory the replay buffer uses per observation key."""


def replay_buffer_memory(model):
    """Bytes per transition (obs + next_obs) for every observation key, and their total.

    Works for ReplayBuffer (one array, reported as "obs") and DictReplayBuffer (one array per key).
    The buffer arrays are allocated when the model is created, so this is the real memory use.
    """
    buffer = model.replay_buffer
    observations = buffer.observations if isinstance(buffer.observations, dict) else {"obs": buffer.observations}
    next_observations = getattr(buffer, "next_observations", None)
    if next_observations is not None and not isinstance(next_observations, dict):
        next_observations = {"obs": next_observations}

    per_key = {}
    for key, array in observations.items():
        size = array.nbytes
        if next_observations is not None:
            size += next_observations[key].nbytes
        per_key[key] = size / buffer.buffer_size
    return per_key, sum(per_key.values())


def log_replay_buffer_memory(model):
    """Print dtype, shape and memory of every observation key in the replay buffer."""
    buffer = model.replay_buffer
    observations = buffer.observations if isinstance(buffer.observations, dict) else {"obs": buffer.observations}
    per_key, total = replay_buffer_memory(model)
    for key, bytes_per_transition in per_key.items():
        array = observations[key]
        print(f"[replay buffer] {key}: dtype={array.dtype}, shape={array.shape[2:]}, "
              f"{bytes_per_transition / 1e3:.1f} KB per transition")
    print(f"[replay buffer] observations: {total * buffer.buffer_size / 1e9:.2f} GB for "
          f"{buffer.buffer_size} transitions ({total / 1e3:.1f} KB per transition)")
