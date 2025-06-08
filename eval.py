import gymnasium as gym
import time
import os
import wandb
from wandb.integration.sb3 import WandbCallback

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import  DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

models_dir =  "models/SAC_AIRSIM"
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 125000,
}
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "droneGym:AirSimEnv-v4", 
        ip_address="127.0.0.1", 
        image_shape=(144, 400, 2),
        continuous=True
    )
)])

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 125000,
    "env_id": env,
    "algorithm": "sac"

}

run = wandb.init(
    project="b3mbb074-110k-eval",
    entity="ararat-cetinkaya-tuhh",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

#models_path =  f"{models_dir}/{config['total_timesteps']}"
#models_path = f"{models_dir}/6f4pl56k"
models_path = os.path.join(models_dir, "b3mbb074", "model.zip")
model = SAC.load(models_path)
 
env.model = model

obs = env.reset()
count = 0
while count < 100:
    action = model.predict(obs)
    # print(action[0][0])  # Commented out to avoid printing actions
    obs, reward, done, info = env.step(action[0])
    #print("Episode: {} Reward: {}".format(count, reward))
    if done[0]:  # Correctly check the 'done' flag for the single environment
        obs = env.reset()
        count += 1