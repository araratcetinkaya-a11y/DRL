# SAC Pipeline for UAV Navigation in AirSim

[![CI](https://github.com/araratcetinkaya-a11y/DRL/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/araratcetinkaya-a11y/DRL/actions/workflows/ci.yml)

*Master Thesis · 2026*

This project develops a deep reinforcement learning (DRL) model for autonomous UAV path planning and obstacle avoidance. The agent navigates to a goal using only an onboard depth image and its own state, and is trained and evaluated in Microsoft AirSim. The focus is not on changing the Soft Actor-Critic (SAC) algorithm itself, but on building the pipeline around it: observation design, feature extraction, action space, reward function, vehicle control and evaluation.

## What I did

- **Late-fusion CNN:** a lightweight CNN processes the depth image, and numerical states (goal-relative position, motion state, previous command) are fused in before the SAC actor and critic.
- **Delta-based velocity control:** the agent commands velocity changes instead of absolute velocities, which gives smoother and physically consistent flight.
- **PID tuning:** manually tuned the PID gains of the simulated drone's flight controller so that velocity commands are tracked reliably.
- **Reward design:** an adapted reward for delta control, with an obstacle safety zone and a speed penalty near obstacles.
- **Debugging and evaluation:** fixed depth-image processing in the inherited code, added seeded evaluation with best-model saving, and ran ablations on yaw-rate limits, learning starts and entropy settings.

## Results

The final model was trained in one environment and then tested, without any further training, in two environments it had never seen. Each environment was evaluated over 150 episodes.

| Environment | Description | Success rate |
|---|---|---|
| Training env | Static obstacles, random start heading and goal direction | 98.0 % |
| Evaluation env 1 | Unseen layout with new obstacle shapes, textures, lighting and moving obstacles | 94.7 % |
| Evaluation env 2 | Unseen photorealistic neighborhood with houses, trees, porches and traffic signs | 28.0 % |

The policy transfers well to unseen and dynamic scenes. In the photorealistic neighborhood it covers about 70 % of the path on average even in unsuccessful episodes; remaining collisions happen at thin, fine structures such as branches, signs and roof edges, which are hard to resolve at the low depth-image resolution used for training.

## Tech stack

Python · PyTorch · Stable-Baselines3 · OpenAI Gym · Microsoft AirSim / Unreal Engine · Weights & Biases

## Repository structure

| Path | Content |
|---|---|
| `gym_env/` | AirSim Gym environment: observations, reward, termination, vehicle dynamics |
| `scripts/` | Training and evaluation (PyQt GUI), custom CNN policies, evaluation callback |
| `configs/` | Experiment configurations (`.ini`) |
| `airsim_settings/` | AirSim `settings.json` files |
| `tools/` | Environment, FPS and GPU tests |
| `tests/` | Automated tests (pytest) with a fake AirSim client; run in CI on every push and pull request |
| `env/` | Reproducible thesis environment: setup script and pinned requirements |
| `docs/` | Code audit and project notes |

## Running the tests

The tests run without AirSim, Unreal Engine, a GPU or a display: `tests/fakes/` replaces the
AirSim client with a small kinematic fake. They cover the action model, the environment step
loop and known issues of the thesis code (documented in `docs/AUDIT.md`).

```powershell
# Windows, inside the thesis environment (env/setup_thesis_env.ps1)
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

GitHub Actions (`.github/workflows/ci.yml`) builds the environment from scratch on Linux, runs
`ruff` on the new code and the test suite with coverage on every push and pull request to `dev`
and `main`.

Branches: `thesis` (tag `thesis-final`) holds the frozen code that trained the thesis models;
development happens on `dev` through pull requests.

## Acknowledgements

Built on top of [UAV_Navigation_DRL_AirSim](https://github.com/heleidsn/UAV_Navigation_DRL_AirSim) by Lei He.
