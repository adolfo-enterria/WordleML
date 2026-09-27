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
- `live/`: `session.py` (background training thread; pause/continue/reset/speed; `snapshot()` is the dashboard's data), `server.py` (JSON API + static files), `static/` (index.html, app.js, style.css).
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

## Conventions

- **Tuning claims need several seeds.** Single runs routinely mislead here, so compare settings across ≥8 seeds before changing defaults.
- **Randomness:** secrets are drawn uniformly at random every game (`TrainingSession.play_one_game`), and the agent stores only weights, never words.
- **Dashboard data:**
  - New stats go in `TrainingSession.snapshot()` (server side) plus a tile in `index.html` / `render()` in `app.js`.
  - Chart styling follows the dataviz palette tokens in `style.css`, with light and dark themes.
- **Output encoding:** colored terminal output uses colorama ANSI codes. When piping output, strip them with `sed 's/\x1b\[[0-9;]*m//g'`.
