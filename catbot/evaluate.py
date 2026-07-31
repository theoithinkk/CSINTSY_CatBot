"""
Using trained bot, play 20 greeday games against all the 5 graded cats and an additional of 
5 unseen TrainerCat behaviors. 

Included in the report: 
    - how long it took to train the bot on each cat 
    - how many of the 20 games were won 
    - the success rate 
    - the average number of moves it took to catch the cat (if there are cats caught) 
    - the worst number of moves it took to catch the cat (if there are cats caught)  
    - timeout notes if the training took longer than 20 seconds or if any of the 20 games timed out 
    
6767676766767 
To Run: python evaluate.py 
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import numpy as np

import cat_env
from cat_env import make_env
from training import train_bot

TRIALS = 20
MAX_MOVES = 60
TIME_CAP = 20.0

GRADED_CATS = ["batmeow", "mittens", "paotsin", "peekaboo", "squiddyboi"]

#added behaviors for testing: see docs abt their descriptions plz
TRAINER_BEHAVIORS = ["pursuer", "threshold", "warper", "evader", "teleporter"] 

def greedy_rollout(env, q_table):
    state, _ = env.reset()

    for step in range(1, MAX_MOVES + 1):
        action = np.argmax(q_table[state])
        state, _, terminated, truncated, _ = env.step(action)

        if terminated:
            return True, step
        if truncated:
            break

    return False, MAX_MOVES


def assess(cat_name, label):
    start = time.perf_counter()
    q_table = train_bot(cat_name=cat_name)
    train_time = time.perf_counter() - start

    env = make_env(cat_type=cat_name)
    results = [greedy_rollout(env, q_table) for _ in range(TRIALS)]
    env.close()

    caught = [moves for success, moves in results if success]

    wins = len(caught)
    success_rate = wins / TRIALS * 100

    avg_moves = f"{np.mean(caught):.1f}" if caught else "-"
    worst_moves = str(max(caught)) if caught else "-"

    notes = []
    if train_time > TIME_CAP:
        notes.append("Training >20s")
    if wins < TRIALS:
        notes.append(f"{TRIALS - wins} test(s) timeout")

    success = f"{success_rate:.1f}%"
    print(f"{label:<20}{train_time:<12.2f}{f'{wins}/{TRIALS}':<10}{success:<12}{avg_moves:<12}{worst_moves:<10}{', '.join(notes)}")


def main():

    print("\nGRADED CATS")
    print("-" * 95)
    print(f"{'Cat':<20}{'Train(s)':<12}{'Wins':<10}{'Success':<12}{'Avg Moves':<12}{'Worst':<10}Notes")
    print("-" * 95)

    for cat in GRADED_CATS:
        assess(cat, cat)

    print("\nTRAINERCAT (UNSEEN BEHAVIORS)")
    print("-" * 95)
    print(f"{'Cat':<20}{'Train(s)':<12}{'Wins':<10}{'Success':<12}{'Avg Moves':<12}{'Worst':<10}Notes")
    print("-" * 95)

    for behavior in TRAINER_BEHAVIORS:
        cat_env.TrainerCat.behavior = behavior
        assess("trainer", f"trainer-{behavior}")

    print("-" * 95)


if __name__ == "__main__":
    main()