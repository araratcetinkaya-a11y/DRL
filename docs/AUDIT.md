# Code Audit (Phase 0)

Scope: `dev` at 00026f8 (code identical to `thesis-final` / bd69467), Oct 30, 2026.
No code was changed in this phase. Line numbers refer to this commit.

## Summary
1. **Q1 (resolved on `dev` in Phase 3, PR 3b; still present in `thesis-final`):** in the thesis
   code, the image observation does not match the declared `observation_space`. With the SB3 fork used here, this raises an error as soon as the
   environment is wrapped in `DummyVecEnv` (training, evaluation callback, `SAC.load(env=...)`).
   The thesis models were most likely trained before the uint8/transpose block in `get_obs` was
   added. Details and a simulator-free way to confirm it: see "Observation pipeline".
2. **Q2 (confirmed by the author, Nov 9, 2026):** `thesis-final` contains a post-training
   evaluation experiment: both actions are multiplied by 1.1 in `set_action` ("Offset var").
   The models were trained with gain 1.0. Evaluating old models with the `thesis-final` code
   therefore does not reproduce the training conditions. On `dev` the gain is now the config
   parameter `action_gain` (default 1.0); see "Action gain (Q2)" below.
3. **Q3 (report only, Nov 12, 2026):** with the thesis config `crash_distance = 2` and
   `SAFE_DISTANCE = 2` (hard-coded), the obstacle penalty and the speed-near-obstacle penalty can only
   be non-zero in the step in which the drone crashes. In practice they make the crash penalty larger
   for fast crashes: -25 - 1 * (v - 4)^2 - 0.5 * (2 - d)^2 (training-run values). See "Reward (Q3)" below.
4. The final training path uses `MultirotorDynamicsAirsim`, `compute_reward_final` (second
   definition), `CNN_GAP_new` wrapped by `with_permute`, and `SAC` with `CnnPolicy`.
5. `features_dim` is not hard-coded in the extractor; it is `cnn_feature_num + state_feature_length`
   = 128 + 6 = 134. The CNN output (8 x 2 x 8 = 128) matches `cnn_feature_num = 128` only for
   70x200 input. (The roadmap's "132" came from the standalone test scripts.)
6. Every module imports `airsim`, and `airsim_env.py` also imports `keyboard`, `PyQt5` and `torch`
   at import time. The dynamics classes connect to AirSim in `__init__`. Nothing can be imported in
   CI without fakes for these.
7. About 40% of the Python files are legacy or one-off experiment scripts (list below).
8. **Training run recovered (Nov 13, 2026):** the W&B run behind the thesis results stores the git
   commit (79739ba) and a `diff.patch` of all uncommitted changes. Commit + patch is the exact
   training code. `thesis-final` differs from it in more than the 1.1 gain (reward coefficients,
   start height, episode length, yaw gain, buffer size, the uint8 block). The training code is on
   `thesis` as tag `thesis-final.1`, and `dev` uses the training values again. Details:
   `docs/TRAINING_RUN.md`.

## Environment
| Item | Value | Source |
| --- | --- | --- |
| Python | 3.9.13 (setup script); the training run used 3.8.10 (W&B) | `env/requirements-thesis-lock.txt` |
| torch | 1.10.1+cu113 | lock file |
| gym | 0.19.0 (old step API: obs, reward, done, info) | lock file |
| Stable-Baselines3 | fork heleidsn/stable-baselines3 @ 012c977, 1.4.1a0 | submodule |
| airsim | 1.6.0 | lock file |
| Setup | `env/setup_thesis_env.ps1` | repo |

## Active code path (final config `configs/config_Maze_SimpleMultirotor_2D.ini`)
| Setting | Value | Code |
| --- | --- | --- |
| env_name | SimpleAvoid (start [0,0,5], goal 50 m, random heading, max 400 steps; `thesis-final`: [0,0,3], 1000) | `airsim_env.py` `set_config` |
| dynamic_name | Multirotor -> `MultirotorDynamicsAirsim` | `dynamics/multirotor_airsim.py` |
| perception | depth, 70x200, max_depth_meters 15 | `get_obs_image` |
| state | 4 values (distance, relative yaw, v_xy, yaw rate) + 2 (last action) = 6, normalized to ~[0, 1] | `_get_state_feature` (L316) |
| action | 2D, both in [-1, 1]: action[0] changes v_xy by up to 0.4 m/s per step (acc_xy_max), clipped to [0.5, 5] m/s; action[1] sets the yaw rate directly, up to +/-60 deg/s (yaw_rate_max_deg). Training gain 1.0; `thesis-final` code uses 1.1 (Q2) | `set_action` (L174) |
| dt | 0.1 s, lockstep via `simPause` + `simContinueForTime` | `set_action` |
| reward | `reward_final` -> `compute_reward_final` (second definition, L901) | `airsim_env.py` |
| algorithm | SAC, `CnnPolicy`, extractor `CNN_GAP_new` (L412) wrapped by `with_permute` (thread_train L31, L271) | `scripts/utils/thread_train.py` |
| replay buffer | 100000 (`thesis-final`: 300000), batch 256, learning_starts 25000 | config |
| entry points | `scripts/start_train_with_plot.py`, `scripts/start_evaluate_with_plot.py` (PyQt GUI) | |

Reward coefficients are hard-coded in `AirsimGymEnv.__init__` (L47-52). Training run (and `dev`
since the training values were restored): SAFE_DISTANCE 2 m, k_obs_penalty 0.5, k_speed_near_obs 1,
max_safe_speed_near_obs 4 m/s, k_jerk_fwd 0.025, k_jerk_yaw 0.03, distance shaping coefficient 50.
`thesis-final` has k_speed_near_obs 50, max_safe_speed_near_obs 3.5 and coefficient 40 (changed
after training). Terminal rewards in `compute_reward_final`: reach +30, crash -25, outside -10.

## Phase 3: Dict observation (PR 3a tests, PR 3b implementation)
Since PR 3b, `get_obs` returns `get_obs_dict()`:
`{"depth": uint8 (1, 70, 200), "state": float32 (6,)}`, declared as
`Dict(depth=Box(0, 255, (1, H, W), uint8), state=Box(-inf, inf, (n,), float32))`.

| Item | Thesis (`thesis-final`) | `dev` after PR 3b |
| --- | --- | --- |
| Observation | one array, depth + state in a second channel | Dict, depth and state separate |
| Depth | float32 in training (uint8 block added later, truncating) | uint8, rounded (resolution 15 m / 255, about 6 cm) |
| State | float32 in training | float32, never quantized |
| SB3 policy | `CnnPolicy` + `with_permute` | `MultiInputPolicy`, no permute wrapper |
| Replay buffer | `ReplayBuffer`, 224 KB per transition | `DictReplayBuffer`, 28.0 KB per transition (about 8x less) |
| `CNN_GAP_new` | reads channel 0 and row 0 of channel 1 | reads `obs["depth"]` and `obs["state"]`; same layers, same outputs (characterization test) |
| `features_dim` | from the config, not checked | computed from the input size; a wrong `cnn_feature_num` raises ValueError |

Behavior change: only the depth precision (uint8 instead of float32). A short training run on the
university PC is needed to confirm training still works. Models trained before PR 3b cannot be
loaded with this code (different observation space); use a worktree of the `thesis-final.1` tag.

Not updated yet (report only):
- `thread_evaluation.py` saves `obs_eval.npy` as a list of dicts; `result_analysis.py` and
  `inspect_npy.py` still read it as an array with a channel index and will fail on new evaluations.
- Extractors other than `CNN_GAP_new` (`CNN_FC`, `CNN_GAP_BN`, `CNN_MobileNet`, `No_CNN`) do not
  support the Dict observation; `thread_train.py` now raises a clear error for them.
- The Q-map code now uses `policy.obs_to_tensor` and the model's device, but is not covered by a
  test (needs `generate_q_map = True` and a model).

## Observation pipeline (thesis code, before Phase 3)
| Step | Where | Output |
| --- | --- | --- |
| 1. Depth in meters | `get_depth_image` (L616) | float32 (h, w) |
| 2. Resize, clip to 15 m, normalize, invert (near = 1) | `get_obs_image` (L527) | float32 (70, 200) in [0, 1] |
| 3. State into row 0 of a second channel, stack, swap axes | `get_obs_image` | float32 (70, 200, 2) |
| 4. Scale x255 only if `obs.max() <= 1.0`, cast to uint8 | `get_obs` (L511-517) | uint8 |
| 5. Transpose to channels-first | `get_obs` (L522) | uint8 (2, 70, 200) |
| Declared space | L226-229 | `Box(low=0.0, high=1.0, shape=(70, 200, 2), dtype=uint8)` |

Q1 - why this cannot work as committed:
- `DummyVecEnv` allocates its observation buffer with the declared shape (70, 200, 2) and copies
  each observation into it. A (2, 70, 200) array cannot be broadcast into that shape
  (checked with numpy: "could not broadcast input array from shape (2,70,200) into shape (70,200,2)").
  `SAC.load(model, env=env)` and `model.predict` hit the same shape check.
- Because `high=1.0`, SB3's `is_image_space` returns False, so SB3 does not divide by 255.
  The extractor would then receive values in 0-255 while `CNN_GAP_new` expects [0, 1]
  (`depth_img_norm = (depth_img - 0.5) * 2`).
- `with_permute` permutes (N, H, W, C) -> (N, C, H, W). Applied to an already channels-first
  batch, it produces the wrong layout.
- Step 4 is fragile on its own: the clip on the state is commented out (`multirotor_airsim.py`
  L344), so `distance_norm` can exceed 1. Then `obs.max() > 1`, the x255 is skipped, and the cast
  to uint8 turns the depth channel into 0/1 values for that step.

- Step 4 also truncates instead of rounding (0.3333 x 255 = 84.99 -> 84), and negative state values
  (the speed is -0.111 at standstill) wrap around when cast to uint8 (observed: 228).

Most likely explanation: the final training used steps 1-3 only (float32, channels-last, matching
`with_permute`), and the uint8/transpose block was added later. This matches the note that the
final training stored both channels as float32.

Confirmed (Nov 13, 2026): the W&B config of the training run records
`observation_space = Box(0.0, 1.0, (70, 200, 2), float32)` and the recovered training code has no
uint8 block. Earlier evidence (working assumption of Nov 11, 2026):
- with the block, `DummyVecEnv` cannot even reset (test `test_dummy_vec_env_can_reset`), so no
  training could have run with it;
- commit 79739ba (Nov 19, 2025) declares a float32 `observation_space` and `get_obs` returns
  `get_obs_image` unchanged; the block first appears in the bulk commit bd69467 (Aug 12, 2026);
- comments in `get_obs_image` state that the uint8 cast was removed on purpose.
How to confirm once a thesis model is available: SB3 stores the observation space in the model
file. With the thesis venv:
`python -c "import zipfile,json,base64,cloudpickle,sys; z=zipfile.ZipFile(sys.argv[1]); d=json.loads(z.read('data')); s=cloudpickle.loads(base64.b64decode(d['observation_space'][':serialized:'])); print(s.shape, s.dtype)" <model.zip>`
Expected: `(70, 200, 2) float32`.

Since Phase 1 (PR 1d), `get_obs_image` uses `gym_env/core/observation.py`; its output is covered by
a characterization test recorded from the frozen code. `get_obs` (step 4-5) is unchanged and
covered by the Q1 tests.

How to confirm without the simulator (first characterization test in Phase 1): build the env with
a fake AirSim client, call `reset()`, and check `env.observation_space.contains(obs)` and
`obs.shape`. Done: the tag `thesis-final.1` on `thesis` restores the training code, which has no
uint8 block (docs/TRAINING_RUN.md).

## Action gain (Q2)
| Code | Gain | Speed change per step | Max yaw rate |
| --- | --- | --- | --- |
| Training of the thesis models | 1.0 | 0.4 m/s | 60 deg/s |
| `thesis-final` tag (evaluation experiment) | 1.1, hard-coded | 0.44 m/s | 66 deg/s |
| `dev` from Phase 1 (PR 1c) | config `action_gain`, default 1.0 | 0.4 m/s | 60 deg/s |

To evaluate old models under training conditions, use the tag `thesis-final.1` (training code,
gain 1.0). To repeat the evaluation experiment on `dev`, set `action_gain = 1.1` in the
[multirotor] section of the config.

## Reward (Q3)
Active function: second definition of `compute_reward_final` in `airsim_env.py`. Since Phase 1
(PR 1e) its terms come from `gym_env/core/reward.py`; values are covered by a characterization test
recorded from the training-run code (`thesis-final.1`).

| Term | Formula (training-run values) | Active when |
| --- | --- | --- |
| distance progress | 50 * (d_prev - d_now) / 50 | every step |
| obstacle penalty | 0.5 * (2 - d_obs)^2 | d_obs < 2 m |
| speed near obstacle | 1 * (v_xy - 4)^2 | d_obs < 2 m and v_xy > 4 m/s |
| forward jerk | 0.025 * (a0 - a0_prev)^2 | every step |
| yaw jerk | 0.03 * (a1 - a1_prev)^2 | every step |
| terminal | reach +30, crash -25, outside -10 (in this order) | done |

`is_crashed` is true when `min_distance_to_obstacles < crash_distance` (2 m). The bubble condition
`d_obs < SAFE_DISTANCE` (2 m) is the same condition, so both obstacle penalties are only applied in
the crash step (or never, if the crash is detected by `has_collided` at a larger distance). Whether
SAFE_DISTANCE had another value in the thesis experiments should be checked against the thesis text
and the saved run configs. No change was made.

## Places that used `obs` as an array (thesis code; updated in PR 3b unless noted above)
| Place | Line | Assumption |
| --- | --- | --- |
| CameraError fallback in `step` | L376 | `np.zeros(observation_space.shape, float32)`: (70, 200, 2) float, while normal observations are (2, 70, 200) uint8 |
| Q-map code in `step` | L452-459 | swaps axes assuming channels-last; calls `.cuda()` unconditionally |
| `print_train_info_airsim` | L1360 | receives `obs` but only logs scalars |
| `thread_evaluation.run_drl_model` | | stores raw observations in `obs_eval.npy` |
| `result_analysis.py`, `inspect_npy.py` | | read `obs_eval.npy` with a fixed channel index |
| `CNN_GAP_new.forward` | custom_policy_sb3 L467 | state read from `observations[:, 1, 0, :n]` |

## Import-time side effects and hardware dependencies
- Debug prints at import: `multirotor_airsim.py` L6 (`print("afasdfaf")`), `thread_train.py` L27
  (both removed in Phase 1), `start_evaluate_with_plot.py` L2 and L5.
- `airsim_env.py` imports `airsim`, `keyboard`, `msgpackrpc`, `torch`, `cv2`, `PyQt5` (L2-20).
  `AirsimGymEnv` inherits from both `gym.Env` and `QtCore.QThread`.
- Dynamics classes create `airsim.MultirotorClient()` and call `confirmConnection()` in `__init__`.
- `set_config` sends Unreal console commands (L188 onwards) and builds the dynamics model, so the
  env cannot be configured without a simulator.
- `perception = lgmd` references `LGMD`, which is no longer imported (module removed): dead branch
  that would raise NameError.

## Hard-coded paths and values
| Item | Where |
| --- | --- |
| `eval_path` on the university PC; `-model_path` / `-eval_eps` defined but never parsed; 200 episodes | `scripts/start_evaluate_with_plot.py` L16, L32, L42 |
| `cv2.waitKey()` blocks before evaluation | `scripts/utils/thread_evaluation.py` L120 |
| `sys.path.append(r"C:\Users\helei\...")` | `scripts/utils/thread_evaluation.py` L14 |
| Model path of the original author | `scripts/evaluation.py` L15 |
| `BASE_PATH` on the university PC | `result_analysis.py` |
| Reward coefficients | `airsim_env.py` L47-52 (not in config) |

## Dead, legacy or experiment code (report only, no deletion in Phase 0)
- Duplicate method: `compute_reward_final` at L801 is overridden by L901.
- Unused reward variants for the final config: `compute_reward`, `compute_reward_test`,
  `compute_reward_multirotor_new`, `compute_reward_with_action`, `compute_reward_lqr`,
  `compute_reward_fixedwing`, `compute_reward_final_fixedwing`.
- Unused env branches in `set_config`: NH_center, NH_tree, City, City_400, Tree_200, Forest, Trees.
- Legacy scripts that import modules or names that no longer exist: `scripts/train.py`
  (`CustomNoCNN`, ...), `scripts/evaluation.py` (`gym_airsim_multirotor`).
- Unused for the final config: `fixedwing_simple.py`, `multirotor_simple.py`,
  `thread_train_fixedwing.py`, `thread_train_repeat.py`, extractor classes other than `CNN_GAP_new`.
- One-off test and benchmark scripts in the repo root and in `dynamics/`: `fps_test.py`,
  `fps_test_cnn.py`, `pahse_test.py`, `test1.py`, `test_airsim_physics.py`, `physics_test_runner.py`,
  `basic_speed_test.py`, `dynamics/fps_test_with_action.py`, `dynamics/test_action_script.py`,
  `check_model.py`. They need the simulator and are not pytest tests.
  Proposal: move them to `tools/experiments/` in a later phase (they document the thesis
  profiling and physics tests, so keep them).

## Out of scope (report only)
- `get_obs_vector` divides the state by 255 although the state is already in [0, 1].
- The distance term in `compute_reward_final` is a plain distance difference, i.e. PBRS with
  gamma = 1 (SAC uses gamma = 0.99).
- `dynamic_model.last_action` (used in the state) and `env.last_action` (used for the jerk penalty)
  are two separate copies; check that both are updated in the same step.
- A time-limit episode lasts `max_episode_steps + 1` steps (401): `is_done` compares `step_num`
  before it is increased. Same in the training-run code, so it is kept (`tests/test_eval_env.py`).
- Phase 3 (PR 3b) removed `SeedCycler` from `thread_train.py` by mistake together with
  `with_permute`; the evaluation env then failed with a NameError in the first training run on the
  university PC (Nov 13, 2026). Restored, covered by `tests/test_eval_env.py`, and CI now checks all
  code for undefined names (ruff F821).

## Files to change, by phase
| Phase | Files |
| --- | --- |
| 1 - testable code + tests | new: `gym_env/gym_env/core/` (pure functions: depth preprocessing, delta action model, state normalization, reward terms), `tests/`, `tests/conftest.py` (fake AirSim client), `pytest.ini` or `pyproject.toml`, `requirements-dev.txt`. Edit: `airsim_env.py`, `dynamics/multirotor_airsim.py` (call the pure functions, remove import-time prints, lazy hardware imports), `thread_train.py` (remove print). |
| 2 - CI | new: `.github/workflows/ci.yml`; edit: `README.md` (badge, "Running tests") |
| 3 - Dict observation | `airsim_env.py` (observation_space, `get_obs`, `get_obs_image`, CameraError fallback, Q-map), `custom_policy_sb3.py` (`CNN_GAP_new`), `thread_train.py` (MultiInputPolicy, remove `with_permute`), `thread_evaluation.py`, `start_evaluate_with_plot.py` (use the CLI arguments) |
