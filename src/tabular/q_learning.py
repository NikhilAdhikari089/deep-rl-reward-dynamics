import numpy as np
import gymnasium as gym
from src.common.utils import set_global_seeds

# Fixed Bellman hyperparameters governing the value update
ALPHA = 0.1          
GAMMA = 0.99         
EPSILON_START = 1.0  
EPSILON_MIN = 0.05   

# Measurement protocol
EVAL_INTERVAL = 500      
EVAL_EPISODES = 30       
TOTAL_TRAIN_STEPS = 400000  

def create_q_table(n_states: int, n_actions: int) -> np.ndarray:
    """Return a zero-initialized action-value table of shape (n_states, n_actions)."""
    return np.zeros((n_states, n_actions), dtype=np.float64)

def select_action(q_table: np.ndarray, state: int, epsilon: float, action_space: gym.spaces.Discrete) -> int:
    """Select an action under an epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return action_space.sample()
    return int(np.argmax(q_table[state]))

def update_q_value(q_table: np.ndarray, state: int, action: int, reward: float, next_state: int, alpha: float, gamma: float) -> None:
    """Apply the Q-learning temporal-difference update in place."""
    best_next = np.max(q_table[next_state])
    td_target = reward + gamma * best_next
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error

def evaluate_greedy_policy(q_table: np.ndarray, n_episodes: int, seed_offset: int = 10000) -> float:
    """Return the mean accumulated reward of the greedy policy."""
    # Updated to Taxi-v4 per Gymnasium 1.3.0 requirements
    eval_env = gym.make("Taxi-v4")
    total_reward = 0.0
    for episode in range(n_episodes):
        state, _ = eval_env.reset(seed=seed_offset + episode)
        done = False
        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
    eval_env.close()
    return total_reward / n_episodes

def train_q_learning(epsilon_decay: float, total_steps: int = TOTAL_TRAIN_STEPS, seed: int = 42):
    """Train a tabular Q-learning agent on Taxi-v4 under a given decay rate."""
    set_global_seeds(seed)
    # Updated to Taxi-v4 per Gymnasium 1.3.0 requirements
    env = gym.make("Taxi-v4")
    q_table = create_q_table(env.observation_space.n, env.action_space.n)

    epsilon = EPSILON_START
    episodes_completed = 0
    eval_steps, eval_rewards = [], []

    state, _ = env.reset(seed=seed)
    for step in range(1, total_steps + 1):
        action = select_action(q_table, state, epsilon, env.action_space)
        next_state, reward, terminated, truncated, _ = env.step(action)
        update_q_value(q_table, state, action, reward, next_state, ALPHA, GAMMA)
        state = next_state

        if terminated or truncated:
            state, _ = env.reset()
            episodes_completed += 1
            # Exponential decay of exploration as a function of episodes
            epsilon = max(EPSILON_MIN, EPSILON_START * np.exp(-epsilon_decay * episodes_completed))

        # Periodic decoupled evaluation of the greedy policy
        if step % EVAL_INTERVAL == 0:
            mean_reward = evaluate_greedy_policy(q_table, EVAL_EPISODES)
            eval_steps.append(step)
            eval_rewards.append(mean_reward)

    env.close()
    return eval_steps, eval_rewards, q_table

if __name__ == "__main__":
    print("Training agent A (epsilon decay = 0.002) ...")
    steps_a, rewards_a, q_table_a = train_q_learning(epsilon_decay=0.002)
    print(f"  Final greedy evaluation reward: {rewards_a[-1]:.2f}")

    print("Training agent B (epsilon decay = 0.006) ...")
    steps_b, rewards_b, q_table_b = train_q_learning(epsilon_decay=0.006)
    print(f"  Final greedy evaluation reward: {rewards_b[-1]:.2f}")