import gymnasium as gym
import time, os
import wandb
from wandb.integration.sb3 import WandbCallback

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

models_dir = "models/SAC_AIRSIM"
logdir     = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir,     exist_ok=True)

# Continuous‐action version of your AirSim env
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "droneGym:AirSimEnv-v4",
        ip_address="127.0.0.1",
        image_shape=(144, 400, 2),
        continuous=True
    )
)])
env = VecTransposeImage(env)

print("Obs space:", env.observation_space, "Act space:", env.action_space)

# WandB setup
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 110000, 
    "env_id": "AirSimEnv-v4",
}
run = wandb.init(
    project="Üstükapali-yenikod-110k-katsayili reward",
    entity="ararat-cetinkaya-tuhh",
    config=config,
    sync_tensorboard=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load or create
continue_training = False
models_path = os.path.join(models_dir, "model.zip")
if continue_training and os.path.exists(models_path):
    model = SAC.load(models_path, env=env, device=device)
else:
    model = SAC(
        "CnnPolicy",
        env,
        learning_rate=0.00003863,
        buffer_size=51734,
        batch_size=64,
        tau=0.01776,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        gamma          = 0.9667,
        ent_coef="auto",
        verbose=2,
        device=device,
        tensorboard_log=f"runs/{run.id}",
    )

# Optional eval callback
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=10,
    best_model_save_path=models_dir,
    log_path=logdir,
    eval_freq=1000,
)
callbacks = [eval_callback]

# Train!
model.learn(
    total_timesteps=config["total_timesteps"],
    reset_num_timesteps=True,
    tb_log_name="SAC_AIRSIM",
    progress_bar=True,
    callback=WandbCallback(
        model_save_freq=10000, 
        gradient_save_freq=5_000,
        model_save_path=models_dir + f'/{run.id}',
        verbose=2,
    ),
)

# Save final policy
model.save(os.path.join(models_dir, f"{config['total_timesteps']}"))
