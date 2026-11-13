# Training run behind the thesis results

The thesis models come from one SAC training run. This file records which code that run used and
how it was recovered, so that the tag `thesis-final.1` can be checked against it.

## The run

| Item | Value |
|---|---|
| Run folder | `logs/SimpleAvoid/2025_11_25_17_43_Multirotor_CNN_GAP_SAC` |
| Started | 2025-11-25 16:43 UTC |
| Git commit at start | `79739ba` ("Multirotor airsim dynamics updated") |
| Uncommitted changes | yes, saved by W&B as `diff.patch` |
| Python | 3.8.10 |
| GPU | NVIDIA GeForce RTX 4070 (CUDA) |

## How the code was recovered

W&B stores the commit hash and a `diff.patch` with every uncommitted change when a run starts. Python
loads the modules at that moment, so commit + patch is exactly the code that ran.

1. Check out `79739ba`.
2. Apply the W&B `diff.patch` (it was saved with Windows line endings; convert it to LF first).
3. The result differs from `thesis-final` in four training files. `thesis-final.1` restores those
   four files to the recovered version and leaves everything else as in `thesis-final`.

## What `thesis-final` changed after training

These changes were made after the run, while evaluating and experimenting. `thesis-final.1` reverts them.

| File | Training run | `thesis-final` |
|---|---|---|
| `multirotor_airsim.py` | speed change `action[0] * v_xy_delta_max` | `* 1.1` |
| `multirotor_airsim.py` | yaw rate `action[1] * yaw_rate_max_deg` | `* 1.1` |
| `multirotor_airsim.py` | yaw angle controller gain `Kp = 3.5` | `4` |
| `airsim_env.py` | observation space `Box(0, 1, (70, 200, 2), float32)` | `uint8`, plus a uint8/transpose block in `get_obs` |
| `airsim_env.py` | `k_speed_near_obs = 1`, `max_safe_speed_near_obs = 4` | `50`, `3.5` |
| `airsim_env.py` | `distance_reward_coef = 50` | `40` |
| `airsim_env.py` | SimpleAvoid start height 5 m, `max_episode_steps = 400` | 3 m, `1000` |
| `config_Maze_SimpleMultirotor_2D.ini` | `buffer_size = 100000` | `300000` |
| `thread_train.py` | – | a debug print |

The evaluation scripts (`start_evaluate_with_plot.py`, `thread_evaluation.py`) and the physics test
scripts are kept as in `thesis-final`.

## Action model of the run

- `v_xy_delta_max = acc_xy_max = 0.4` m/s per step (no `* dt`), `dt = 0.1` s.
- Forward speed set point: `clip(prev + action[0] * 0.4, 0.5, 5.0)` m/s.
- Yaw rate: `action[1] * 60` deg/s, sent directly (no delta).
- `moveByVelocityZAsync(..., duration = dt)`, then `simContinueForTime(dt)`.

## Training settings of the run

SAC, `CnnPolicy` with the `CNN_GAP` extractor (128 CNN features + 6 state features),
`net_arch = [64, 32, 16]`, tanh, `learning_rate = 3e-4`, `gamma = 0.99`, `buffer_size = 100000`,
`batch_size = 256`, `train_freq = 256`, `gradient_steps = 256`, `learning_starts = 25000`,
`total_timesteps = 900000`, `ent_coef = auto`, `seed = 0`.
