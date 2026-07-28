import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# Reward values
CATCH_REWARD = 100.0
STEP_PENALTY = -1.0
SHAPING_SCALE = 1.0

def decode_state(state: int):
    """State is bot_row*1000 + bot_col*100 + cat_row*10 + cat_col."""
    return state // 1000, (state // 100) % 10, (state // 10) % 10, state % 10

def manhattan(state: int) -> int:
    """Distance between bot and cat."""
    bot_r, bot_c, cat_r, cat_c = decode_state(state)
    return abs(bot_r - cat_r) + abs(bot_c - cat_c)

def compute_reward(state: int, next_state: int, caught: bool, gamma: float) -> float:
    """Reward for one step. The env always returns 0, so we build it here.

    Big bonus for catching, small penalty per step to keep chases short, plus
    distance shaping. The shaping uses the gamma*phi(s') - phi(s) form, which
    does not change the optimal policy, so it only speeds up learning.
    """
    if caught:
        return CATCH_REWARD

    shaping = gamma * -manhattan(next_state) - -manhattan(state)
    return STEP_PENALTY + SHAPING_SCALE * shaping


from typing import List, Tuple

# Cats we have specs for, plus the trainer sandbox
KNOWN_CATS = ["batmeow", "mittens", "paotsin", "peekaboo", "squiddyboi", "trainer"]

def greedy_rollout(env, q_table) -> Tuple[bool, int]:
    """Play one episode greedily, no exploration and no rendering. Returns
    whether the cat was caught and how many moves it took.
    """
    max_steps = 60
    state, _ = env.reset()
    for step in range(1, max_steps + 1):
        action = int(np.argmax(q_table[state]))
        state, _, terminated, truncated, _ = env.step(action)
        if terminated:
            return True, step
        if truncated:
            break
    return False, max_steps

def evaluate(cat_name, q_table, trials: int) -> Tuple[float, float]:
    """Run a bunch of greedy games and average them.
    """
    env = make_env(cat_type=cat_name)
    catches = 0
    steps_when_caught: List[int] = []

    for _ in range(trials):
        caught, steps = greedy_rollout(env, q_table)
        if caught:
            catches += 1
            steps_when_caught.append(steps)

    env.close()

    success_rate = catches / trials
    avg_steps = float(np.mean(steps_when_caught)) if steps_when_caught else float("nan")
    return success_rate, avg_steps

def evaluate_all(trials: int = 30):
    """Train on every known cat with the current train_bot config then report
    how often the bot catches each one and how many moves it takes.
    """
    total_rate = 0.0
    for cat in KNOWN_CATS:
        q_table = train_bot(cat_name=cat)
        rate, steps = evaluate(cat, q_table, trials)
        total_rate += rate

        s_str = f"{steps:5.2f}" if not np.isnan(steps) else "  -  "
        print(f"{cat:<11} success={rate*100:5.1f}%   avg_steps={s_str}")

    mean_rate = total_rate / len(KNOWN_CATS)
    print(f"\nmean success rate across {len(KNOWN_CATS)} cats: {mean_rate*100:.1f}%")



#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################

    alpha = 0.1             # learning rate
    gamma = 0.95            # discount factor
    epsilon = 1.0           # chance of taking a random action
    epsilon_min = 0.05
    epsilon_decay = 0.999   # applied after every episode

    # Matches the 60-move limit the bot gets during evaluation.
    max_steps = 60

    n_actions = env.action_space.n


    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ##############################################################################

        state, _ = env.reset()

        for _ in range(max_steps):
            # Explore or exploit
            if random.random() < epsilon:
                action = random.randrange(n_actions)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, _, terminated, truncated, _ = env.step(action)
            reward = compute_reward(state, next_state, terminated, gamma)

            # Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
            # Nothing to bootstrap from once the cat is caught.
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(q_table[next_state])

            q_table[state][action] += alpha * (target - q_table[state][action])
            state = next_state

            if terminated or truncated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

   
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table