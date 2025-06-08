import gymnasium as gym
import os
import time
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the environment and directories
models_dir = "models/SAC_AIRSIM"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def make_env():
    return Monitor(
        gym.make(
            "droneGym:AirSimEnv-v4",
            ip_address="127.0.0.1",
            image_shape=(144, 400, 2),
            continuous=True
        )
    )

env = DummyVecEnv([make_env])
env = VecTransposeImage(env)

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 750000,
    "env_id": env,
}



run = wandb.init(
    project="Continue-SAC-newfunction",
    entity="ararat-cetinkaya-tuhh",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)
print(f"Observation space shape: {env.observation_space.shape}")

# Continue training flag
continue_training = True

if continue_training:
    # Load the previously trained model
    model_path = f"{models_dir}/best_model.zip"  # Path to the saved model
    model = SAC.load(model_path, env=env)  # Attach the environment
else:
    # Initialize a new model
    model = SAC(
        "CnnPolicy",
        env,
        learning_rate=0.00003863,  # Adjusted learning rate
        verbose=2,
        batch_size=64,  # Adjusted batch size
        train_freq=1,
        tau=0.01776,  # Adjusted tau
        gamma=0.9667,  # Adjusted gamma
        buffer_size=51734,  # Adjusted buffer size
        learning_starts=10000,
        device=device,
        tensorboard_log=f"runs/{wandb.run.id}",
    )

# Evaluation callback
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=10,
    best_model_save_path=f"{models_dir}/best_model",
    log_path=logdir,
    eval_freq=1000,
)

# Train for 216,000 additional timesteps
model.learn(
    total_timesteps=config["total_timesteps"],  # Additional timesteps
    reset_num_timesteps=False,  # Continue from previous timestep count
    tb_log_name="SAC_AIRSIM_Continue",
    progress_bar=True,
    callback=WandbCallback(
        model_save_freq=10000,
        gradient_save_freq=5000,
        model_save_path=models_dir,
        verbose=2,
    ),
)

# Save the fine-tuned model
model.save(f"{models_dir}/fine_tuned_model")

# Evaluate the model after training
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward after continued training: {mean_reward}")
