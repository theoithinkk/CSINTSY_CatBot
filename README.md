# CatBot

Reinforcement learning bot that catches cats on an 8x8 grid, for CSINTSY MCO3.
The learning algorithm is tabular Q-learning, implemented in
`catbot/training.py`.

## Setup

Python 3.12 is the version targeted by the project spec.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r catbot/requirements.txt
```

All scripts load sprites through relative paths, so every command below must
be run from inside the `catbot/` directory:

```bash
cd catbot
```

## Launch

| Command | What it does |
|---|---|
| `python play.py --cat mittens` | Freeplay, control the bot yourself with arrow keys |
| `python bot.py --cat paotsin` | Train the bot, then watch it play the trained policy |
| `python bot.py --cat paotsin --render 100` | Train, showing a game every 100 episodes |

Swap `--cat` for any of: `batmeow`, `mittens`, `paotsin`, `peekaboo`,
`squiddyboi`, `trainer`. In freeplay, `q` quits.

## Test

```bash
python evaluate.py
```

Trains the bot fresh and plays 20 greedy games against each of the 5 graded
cats plus 5 unseen TrainerCat behaviors (~30-40s total, each cat trains in
under the 20s cap). Prints a results table and writes it to `results/`:

- `results/summary.csv` — one row per cat: training time, wins, success rate,
  average and worst moves
- `results/curve_<cat>.csv` — per-episode training trace
  (`episode,steps,caught`), for the learning-curve graphs in the report

Read the `Notes` column in the printed table before trusting a number — it
flags training over the 20s cap and any game that hit the 60-move limit.
A single 20-trial run is noisy (Peekaboo alone swings 17-20/20 run to run),
so don't judge a change to `training.py` from one run; re-run a few times.

```bash
python check_trainer.py
```

Dev tool, not part of the deliverable. Traces 30 steps of each TrainerCat
behavior against a naive chase, to sanity-check a behavior moves the way its
docstring in `cat_env.py` claims before trusting `evaluate.py`'s numbers for it.

## Where the learning lives

`training.py` holds everything we wrote. The pieces that matter:

- **Reward** — the environment always returns 0, so `compute_reward` builds it:
  +100 for a catch, -1 per step, plus potential-based distance shaping at half
  weight. The `gamma*phi(s') - phi(s)` form provably leaves the optimal policy
  unchanged, so shaping only speeds up learning rather than biasing it.
- **Actions** — 5, not 4. The environment declares `Discrete(4)`, which hides
  "stay in place", but `step()` handles action 4 correctly. Staying still is
  the move that appeases cats which flee whenever you close in.
- **Exploration** — epsilon-greedy, decayed 1.0 to 0.01. The floor is low
  because grading uses the final greedy policy, not average performance.
- **Experience replay** — one extra update per step, sampled from past
  transitions. The project caps episodes at 5000, not Q-table updates, and
  Q-learning is off-policy, so replay extracts more from the same budget.
- **Prior initialization** — the Q-table starts at a weak distance prior
  (scale 0.01) rather than zeros. An all-zero row makes `argmax` return
  action 0, so a state training never visited would send the bot into the top
  wall forever; the prior makes those states fall back to sensible movement.
- **Snapshot selection** — the Q-table is kept every 1000 episodes, and
  training returns whichever snapshot plays best over 12 greedy test games.
  Training occasionally ends on a broken policy (a run catching Peekaboo
  30/30 at episode 3000 finishing at 0/30), and this bounds that risk. It
  spends extra environment steps, but no extra training episodes, and no
  learning happens during the test games.

Training stays under the 20-second cap for every cat (worst observed: ~4.2s).

## Testing against unseen behavior

The five graded cats are the ones we have specs for; the other five are hidden
until submission. To check the algorithm is not overfitted to the known five,
`TrainerCat` in `cat_env.py` implements extra behaviors (`skittish`, `coward`,
`blindspot`, `warper`, `pursuer`) that `evaluate.py` runs as unseen cats.

Each was given a deliberate weakness, because a cat that always takes the
distance-maximizing move is impossible to catch here: the cat moves second,
and a move blocked by the wall is clamped into a free "stay", so it can always
end its turn at least 2 squares away. The graded cats all have such a weakness
(Paotsin flees only 65% of the time; Peekaboo will not teleport away from a
bot on one particular side), so the test cats do too.

## Outputs

`evaluate.py` writes to `catbot/results/`:

- `summary.csv` — one row per cat: training time, wins, success rate, average
  and worst moves
- `curve_<cat>.csv` — per-episode training trace (`episode,steps,caught`) for
  learning-curve graphs
