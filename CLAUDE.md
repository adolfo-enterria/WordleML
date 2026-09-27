# WordleML

A terminal Wordle clone plus a small reinforcement-learning agent that gets better at
Wordle by playing it, watched through a live browser dashboard. Python 3.12, numpy,
colorama, pytest; the dashboard server is stdlib `http.server` and the page loads
Chart.js from cdn.jsdelivr.net. Developed on Windows (PowerShell / Git Bash).

## Commands

```
python dashboard.py                  # live training dashboard (500 games, 10 games/s) -> opens browser
python dashboard.py --games 2000 --speed 0 --seed 3   # speed 0 = max; seed repeats a run
python play.py                       # play Wordle yourself (6 guesses, real rules)
python play.py --watch [--secret nymph]   # watch the trained AI (models/agent.npz)
python evaluate.py [--agent consistent]   # one game per answer word, average guesses
python -m pytest                     # all tests
```

## Layout

- `wordle/`: `game.py` (scoring with Wordle's duplicate-letter rules; `max_guesses=None` = play until solved), `words.py` (word lists + numpy encoding), `display.py` (colored tiles).
- `agents/`: `features.py` (the rules as a "still possible" mask + the facts the agent weighs), `learning_agent.py` (linear softmax policy, REINFORCE), `consistent_agent.py` (no-strategy baseline), `base.py` (`play_game`).
- `analysis/difficulty.py`: word traits (repeated letters, one-letter look-alikes, letter commonness), `analyze()` (every word once with an AI's best guesses, grouped by trait), and `trace_game()` (one word, with words left after each guess). Both use the same fixed tie-break seed, so the lookup and the analysis always agree.
- `live/`:
  - `session.py`: the background training thread. It handles pause/continue/reset, speed, target games and skill checks, and runs the analysis in a background thread. `snapshot()` is the dashboard's data.
  - `server.py`: JSON API plus static files.
  - `static/`: index.html, app.js, style.css.
- `data/answers.txt` (2,315 possible secrets = the agent's vocabulary), `data/allowed_guesses.txt` (extra words humans may type).
- `models/agent.npz`: written by the dashboard when a run finishes; read by `play.py --watch` and `evaluate.py`.

## How the agent works (and why)

- **It knows the rules and only guesses words that could still be the answer.**
  `Featurizer.possible()` is the rule logic. `tests/test_features.py` checks it
  against plain `score_guess` on random boards, so keep that test passing if you
  touch it.
- **No losses:** games run until solved. The reward is −1 per guess, so learning = fewer guesses.
- **Facts, not judgments.** Features describe a word (distinct letters, how common its letters are among the still-possible words) and never say good or bad. Weights start at 0, which means a uniform random possible word, averaging ~4.13 guesses.
- **Facts are standardized** within the current candidate set (`standardize()`), so one learning rate works whether 2,315 or 3 words remain.
- **Stability measures, each added after an observed failure:**
  - A per-turn running baseline.
  - A cap on each game's total update (`MAX_STEP`), because a single lucky game once swung the weights into a degenerate policy.
  - Only non-redundant facts. For example, "untried letters" was always "distinct letters − constant" among possible words, so it was removed.
- **Tuned values** (lr 0.05, 3 features) came from multi-seed experiments:
  - Training play averages ~4.0 → ~3.7 over the first ~300 games, then plateaus.
  - Greedy skill is ~3.61.
  - A single 500-game run is noisy: luck moves a 50-game average by about ±0.15.
- **The skill check** (`TrainingSession.skill_check`) is the dashboard's main measure. It uses the same 200 words (`SKILL_CHECK_SEED`), checks every 5 games for the first 50, then every `target // 20`. It measures two things:
  - **Best guesses:** only the direction of the weights matters, so this drops within ~5–50 games.
  - **Practice-style:** sampling with a fixed seed, via `_NoLearning` so it never updates. This drops gradually as the weights grow.
  - The game-0 value is the blank AI, the same after every reset.

## Conventions

- **Tuning claims need several seeds.** Single runs routinely mislead here, so compare settings across ≥8 seeds before changing defaults.
- **Randomness:** secrets are drawn uniformly at random every game (`TrainingSession.step`), and the agent stores only weights, never words.
- **Threads:**
  - Don't run a pure-Python CPU-heavy thread alongside training. The GIL "convoy effect" made a 0.5 s skill check take 25 s next to a pure-Python baseline job.
  - Numpy-heavy threads, like the analysis, share fairly.
  - The HTTP handlers only read `snapshot()` or do one quick game.
- **Dashboard data:**
  - New stats go in `TrainingSession.snapshot()` (server side) plus a tile in `index.html` / `render()` in `app.js`.
  - Chart styling follows the dataviz palette tokens in `style.css`, with light and dark themes.
- **Output encoding:**
  - Colored terminal output uses colorama ANSI codes. When piping output, strip them with `sed 's/\x1b\[[0-9;]*m//g'`.
  - When editing files from Python on Windows, pass `encoding="utf-8"`. The default is cp1252, and the web files contain –, …, · and −.
- **Number format:** the page formats numbers as en-US (`toLocaleString("en-US")`, Chart.js `locale`) so the charts and tiles match.
