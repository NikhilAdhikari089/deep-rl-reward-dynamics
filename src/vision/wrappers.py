import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register the Atari environments with Gymnasium.
gym.register_envs(ale_py)

AIRRAID_ENV_ID = "AirRaidNoFrameskip-v4"

def build_airraid_env(use_skip_and_stack: bool, n_envs: int, seed: int = 42):
    """
    Construct a vectorized AirRaid environment under one preprocessing pipeline.
    If use_skip_and_stack is True, applies frame_skip=6 and n_stack=3.
    """
    frame_skip = 6 if use_skip_and_stack else 1
    venv = make_atari_env(
        AIRRAID_ENV_ID,
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs=dict(frame_skip=frame_skip),
    )
    if use_skip_and_stack:
        venv = VecFrameStack(venv, n_stack=3)
    return venv