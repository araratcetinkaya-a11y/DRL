import gymnasium as gym
import time, os
import wandb
from wandb.integration.sb3 import WandbCallback

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


if __name__ == "__main__":

    base_seed = 42

    def make_env(rank:int):
        '''
        rank: Ortamın indeks numarası, seed veya loglama için kullanılabilir.
        '''
        def _init():

            import time
            time.sleep(rank * 1)
            port = 41451 + rank
            print(f"[env {rank}] initializing, will connect to 127.0.0.1:{port}")
            ip_port = f"127.0.0.1:{port}" # ip_address içinde hem IP hem port veriyoruz
            env = gym.make(
             "droneGym:AirSimEnv-v4",
             ip_address=ip_port,
             image_shape=(144, 400, 2),
             continuous=True,
                )
            print(f"[env {rank}] created")
            env = Monitor(env)
            env.reset(seed=base_seed + rank)
            try:
                env.action_space.seed(base_seed + rank)
                env.observation_space.seed(base_seed + rank)
            except AttributeError:
                pass

            return env
        return _init
     
    

    models_dir = "models/SAC_AIRSIM"
    logdir     = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir,     exist_ok=True)

    n_envs = 2 # Kaç paralel ortam kullanılacağı (örneğin CPU çekirdek sayınıza göre)

    '''
    print("Waiting 10s for all AirSim instances to spin up…")
    time.sleep(10)
    '''

    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecTransposeImage(env)

    print("Obs space:", env.observation_space, "Act space:", env.action_space)

    # WandB setup
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 200000, 
        "env_id": "AirSimEnv-v4",
    }
    run = wandb.init(
        project="200k-lr-1.79e-4-gputest",
        entity="ararat-cetinkaya-tuhh",
        config=config,
        sync_tensorboard=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load or create
    continue_training = False #True  
    models_path = os.path.join(models_dir, "model.zip")
    if continue_training and os.path.exists(models_path):
        model = SAC.load(models_path, env=env, device=device)
    else:
        model = SAC(
            "CnnPolicy",
            env,
            learning_rate=0.000179,
            buffer_size=50000,
            batch_size=256,
            tau=0.005,
            train_freq=4,
            gradient_steps=1,
            learning_starts=8000, #should be chosen according to the batch size but at leaste %5-10 of buffer 
            gamma          = 0.995,
            ent_coef="auto",
            target_entropy=-0.2,
            verbose=0,
            device=device,
            tensorboard_log=f"runs/{run.id}",
        )
   

    '''
    # Optional eval callback
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=10,
        best_model_save_path=models_dir,
        log_path=logdir,
        eval_freq=1000,
    )
    '''
    wandb_callback = WandbCallback(
        model_save_freq=5000,
        gradient_save_freq=2500,
        model_save_path=os.path.join(models_dir, run.id),
        verbose=0
        )
    callbacks = CallbackList([  wandb_callback ])#eval_callback,

    # Train!
    model.learn(
        total_timesteps=config["total_timesteps"],
        reset_num_timesteps=True ,  
        tb_log_name="SAC_AIRSIM",
        progress_bar=True,
        callback = callbacks,
    )

    # Save final policy
    model.save(os.path.join(models_dir, f"{config['total_timesteps']}"))


