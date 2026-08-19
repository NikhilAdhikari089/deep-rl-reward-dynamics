import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.common.utils import set_global_seeds, DEVICE, GLOBAL_SEED
from src.vision.wrappers import build_airraid_env, AIRRAID_ENV_ID

EVAL_FREQ_AIRRAID = 5000
EVAL_EPISODES_AIRRAID = 20
EVAL_MAX_STEPS = 1000
AIRRAID_N_ENVS = 8
TOTAL_TIMESTEPS_AIRRAID = 300000

class AtariEvalCallback(BaseCallback):
    """Evaluate an image-based policy on a dedicated vectorized eval env."""
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, max_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.timesteps = []
        self.mean_rewards = []
        self._next_eval = eval_freq

    def _evaluate(self) -> float:
        episode_returns = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            steps = 0
            total = 0.0
            while not done and steps < self.max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = self.eval_env.step(action)
                total += float(rewards[0])
                done = bool(dones[0])
                steps += 1
            episode_returns.append(total)
        return float(np.mean(episode_returns))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            mean_reward = self._evaluate()
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(mean_reward)
            if self.verbose:
                print(f"  step {self.num_timesteps:>8d} | mean reward over {self.n_eval_episodes} episodes: {mean_reward:8.2f}")
        return True

def train_airraid_ppo(use_skip_and_stack: bool, label: str):
    """Train a CNN-policy PPO agent under one preprocessing pipeline."""
    set_global_seeds()
    train_env = build_airraid_env(use_skip_and_stack, n_envs=AIRRAID_N_ENVS)
    eval_env = build_airraid_env(use_skip_and_stack, n_envs=1, seed=GLOBAL_SEED + 1)

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=GLOBAL_SEED,
        device=DEVICE,
        verbose=0,
    )

    callback = AtariEvalCallback(eval_env, EVAL_FREQ_AIRRAID, EVAL_EPISODES_AIRRAID, EVAL_MAX_STEPS, verbose=1)

    print(f"\nTraining PPO on AirRaid [{label}] ...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS_AIRRAID, callback=callback)
    train_env.close()
    eval_env.close()
    
    print(f"  Final evaluation reward [{label}]: {callback.mean_rewards[-1]:.2f}")
    return callback.timesteps, callback.mean_rewards

if __name__ == "__main__":
    print("Starting Pipeline A...")
    steps_skip, rewards_skip = train_airraid_ppo(use_skip_and_stack=True, label="skip=6, stack=3")
    
    print("Starting Pipeline B...")
    steps_raw, rewards_raw = train_airraid_ppo(use_skip_and_stack=False, label="no skip, no stack")