"""
Trains the bot on every cat and plays 20 greedy games against each: the five
graded cats, plus five TrainerCat behaviors the bot has never been tuned on.

Prints, and writes to results/:
    - how long training took for each cat
    - how many of the 20 games were won, and the success rate
    - the average and worst number of moves needed (when the cat was caught)
    - notes when training exceeded 20s or when games timed out at 60 moves

Outputs:
    results/summary.csv          one row per cat, the table below
    results/curve_<label>.csv    per-episode training trace, for the learning
                                 curve graphs in the report

To Run: python evaluate.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import csv
import time
import numpy as np

import cat_env
from cat_env import make_env
from training import train_bot, EPISODE_LOG

TRIALS = 20
MAX_MOVES = 60
TIME_CAP = 20.0
RESULTS_DIR = "results"

GRADED_CATS = ["batmeow", "mittens", "paotsin", "peekaboo", "squiddyboi"]

# Custom behaviors used to check the algorithm generalizes past the five cats
# we have specs for. See the TrainerCat docstring in cat_env.py.
TRAINER_BEHAVIORS = ["pursuer", "skittish", "warper", "coward", "blindspot"]


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


def write_curve(label):
    """Dump the per-episode training trace for the report's learning curves."""
    path = os.path.join(RESULTS_DIR, f"curve_{label}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "steps", "caught"])
        writer.writerows((ep, steps, int(caught)) for ep, steps, caught in EPISODE_LOG)


def assess(cat_name, label):
    start = time.perf_counter()
    q_table = train_bot(cat_name=cat_name)
    train_time = time.perf_counter() - start
    write_curve(label)

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

    return [label, f"{train_time:.2f}", wins, TRIALS, f"{success_rate:.1f}", avg_moves, worst_moves, "; ".join(notes)]


def header():
    print("-" * 95)
    print(f"{'Cat':<20}{'Train(s)':<12}{'Wins':<10}{'Success':<12}{'Avg Moves':<12}{'Worst':<10}Notes")
    print("-" * 95)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    print("\nGRADED CATS")
    header()
    for cat in GRADED_CATS:
        rows.append(assess(cat, cat))

    print("\nTRAINERCAT (UNSEEN BEHAVIORS)")
    header()
    for behavior in TRAINER_BEHAVIORS:
        cat_env.TrainerCat.behavior = behavior
        rows.append(assess("trainer", f"trainer-{behavior}"))

    print("-" * 95)

    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cat", "train_seconds", "wins", "trials", "success_pct",
                         "avg_moves", "worst_moves", "notes"])
        writer.writerows(rows)

    print(f"\nWrote {RESULTS_DIR}/summary.csv and per-cat learning curves.")


if __name__ == "__main__":
    main()
