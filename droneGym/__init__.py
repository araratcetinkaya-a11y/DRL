from gymnasium import register

# This init file add the environment to the gymnasium
register(
    id='AirSimEnv-v4',
    entry_point='droneGym.envs:AirsimDynamicsEnv',
    max_episode_steps=1000,
)