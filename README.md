# Reinforcement learning based motion planning for UAV
### This repository includes implementation of End-to-End motion planning software based on Reinforcement Learning approach toward the Master Thesis. This involves: 
- Implementation of several Deep Learning based RL algorithms
- Evaluation of such algorithms
- Hardware implementation and deployment.

## SETUP:
- Ubuntu 20.04
- Unreal Engine version 4.27, it can be found [here](https://docs.unrealengine.com/4.27/en-US/SharingAndReleasing/Linux/BeginnerLinuxDeveloper/SettingUpAnUnrealWorkflow/).
- Python 3.9
- CUDA 11.7
- Airsim v1.6.1 - Linux 
- Conda for virtual environments. (RECOMMENDED)

## Installation:
### Ubuntu 20.04

- To install Pytorch and the other required libraries.
    - `chmod +x install_pytorch.sh`
    - `./install_pytorch.sh`

### Windows 11

- To install Pytorch and the other required libraries.
    - `./install_pytorch.bat`

The code is tested both on Ubuntu and Windows 11 environment

## Usage:
- Create the Unreal environment with AirSim plugin.
- In the `droneGym\envs\droneEnvDynamics.py` assign the goal positions in the `__init__()` and `reset_drone()` method.
- Run the `ppo_train_wandb.py` script to train the agent.
- After training the agent can be evaluated through `eval.py` script. Path to the model might be needed to be changed.

## Weights and Biases integration
- You will have to create an account with Weights and Biases for experiment tracking. 
- In the training script you need to assign you credentials instead of mine in the `wand.init()` method.
