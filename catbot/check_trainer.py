"""
Development tool, not part of the deliverable. Traces a few steps of each
TrainerCat behavior against a naive chase so we can eyeball whether the cat
moves the way its docstring claims.

To run: python check_trainer.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy") # run the catbot game without opening a window 

import cat_env
from cat_env import make_env


def dist(state):

    # Decode the encoded state into bot and cat positions
    br, bc, cr, cc = (
        state // 1000, 
        (state // 100) % 10, 
        (state // 10) % 10, 
        state % 10
    )

    # returns the manhattan distance between bot and cat & both of their positions
    return abs(br - cr) + abs(bc - cc), (br, bc), (cr, cc)


def run(behavior, steps=30):

    #select which cat behavior to test
    cat_env.TrainerCat.behavior = behavior

    # create the environment with the trainer cat
    env = make_env(cat_type="trainer")
    state, _ = env.reset()

    print(f"\n--- {behavior} ---")
    d, bot, cat = dist(state)
    print(f"  start   bot={bot} cat={cat} dist={d}")

    # Run for a fixed number of steps or until the cat is caught
    for i in range(1, steps + 1):
        _, bot, cat = dist(state)
        # naive chase: close the bigger gap first
        if abs(bot[0] - cat[0]) >= abs(bot[1] - cat[1]):
            action = 1 if cat[0] > bot[0] else 0
        else:
            action = 3 if cat[1] > bot[1] else 2

        state, _, terminated, _, _ = env.step(action)
        d, bot, cat = dist(state)
        print(f"  step {i:<2} bot={bot} cat={cat} dist={d}") # print the updated positions and distance after each step
        if terminated:
            print("  -> CAUGHT")
            break
    env.close()


for b in ["still", "pursuer", "skittish", "warper", "coward", "blindspot"]:
    run(b)