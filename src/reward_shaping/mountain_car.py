import gymnasium as gym
import numpy as np

class MyMountainCar(gym.Wrapper):
    """
    Custom wrapper to override the native sparse reward of MountainCar-v0.
    Demonstrates the difference between potential-based shaping and reward hacking.
    """
    def __init__(self, env: gym.Env, reward_strategy: str):
        super().__init__(env)
        valid_strategies = ["native", "position", "velocity"]
        if reward_strategy not in valid_strategies:
            raise ValueError(f"reward_strategy must be one of {valid_strategies}")
        self.reward_strategy = reward_strategy

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position, velocity = obs

        # STRATEGY A: Potential-based shaping (Proximity to goal)
        if self.reward_strategy == "position":
            # Normalize position (-1.2 to 0.5) to a positive reward signal
            reward += (position + 1.2) / 1.7
            
        # STRATEGY B: Reward Hacking vulnerability (Raw momentum)
        elif self.reward_strategy == "velocity":
            # Agent learns to rock back and forth endlessly to farm velocity bonuses
            reward += abs(velocity) * 100
            
        # STRATEGY C: Native sparse reward (-1 per step)
        elif self.reward_strategy == "native":
            pass 

        return obs, reward, terminated, truncated, info

def make_shaped_env(strategy: str) -> gym.Env:
    """Helper function to instantiate a shaped MountainCar environment."""
    env = gym.make("MountainCar-v0")
    return MyMountainCar(env, reward_strategy=strategy)