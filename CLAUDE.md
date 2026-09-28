# WordleML

A terminal Wordle clone, AIs that get better at Wordle by playing it, a live browser
dashboard to watch them learn, and a reproducible benchmark. Python 3.12, numpy,
matplotlib, colorama, pytest. The dashboard server is stdlib `http.server` and the page
loads Chart.js from cdn.jsdelivr.net. Developed on Windows (PowerShell / Git Bash).

## Commands

```
python dashboard.py                  # live training dashboard -> opens browser (default mode "any")
python dashboard.py --mode possible --games 2000 --speed 0 --seed 3
python benchmark.py [--quick]        # reproducible report in reports/<date-time>/ (full ~20 min)
python play.py                       # play Wordle yourself (6 guesses, real rules)
python play.py --watch [--secret nymph]   # watch the trained AI (models/agent.npz)
python evaluate.py [--agent consistent]   # one game per answer word, average guesses
python -m pytest                     # all tests (~2 min)
```

## Layout

- `wordle/`:
  - `game.py`: scoring with Wordle's duplicate-letter rules; `max_guesses=None` means play until solved.
  - `words.py`: word lists and numpy encoding.
  - `patterns.py`: the 12,972 × 2,315 table of every (guess, answer) color pattern. It's cached in `data/patterns.npz`, built on first use in ~10 s. Answers come first in the guess list, so answer i is guess i.
  - `display.py`: colored tiles.
- `agents/`:
  - `__init__.py`: `make_learner(mode)`, `load_learner(path)` and `make_agent(name)`. These are the entry points.
  - `planning_agent.py`: mode **"any"**, the main AI. It learns V(m) (expected further guesses with m words possible, 3 numbers) and plays the guess with the lowest expected cost.
  - `learning_agent.py`: a REINFORCE policy. Mode **"possible"** uses 3 standardized word facts from `features.py`. Mode "any" uses the split facts from `split_features.py`, but only the benchmark uses it, as a baseline.
  - `split_features.py`: how every valid guess splits the still-possible answers (group sizes → facts), plus `candidates()`.
  - `features.py`: the rules as a mask for "possible" mode.
  - `consistent_agent.py`: a random possible word (no strategy).
- `analysis/`:
  - `difficulty.py`: word traits, `analyze()` and `trace_game()`. Fixed tie-break seed, so the lookup and the analysis agree.
  - `strategies.py`: hand-made greedy heuristics, forced opener, fact subsets.
  - `stats.py`: CIs, paired bootstrap, power analysis, OLS.
  - `benchmark_tasks.py`: worker-process tasks.
  - `figures.py`: report charts.
- `live/`:
  - `session.py`: the training thread. It handles controls, skill checks, the final exam and the "what it learned to do" stats. `snapshot()` is the dashboard's data.
  - `server.py`: JSON API.
  - `static/`: the page.
- `models/agent.npz`: saved by the dashboard when a run finishes (planner or policy; `load_learner` picks the right one).

## How the agents work (and why)

- **They know the rules.** They track which answers still match the colors (`candidates()`) and say the answer when only one is left. `tests/test_features.py` and `tests/test_patterns.py` check the rule logic against `score_guess`; keep them passing.
- **No losses:** games run until solved, and the reward is −1 per guess.
- **Planner (any mode):**
  - For every valid guess: expected cost = Σ over possible answers of V(size of the group it lands in) / n, where a correct guess costs 0.
  - V(m) = a + b·log2 m + c·(log2 m)² is fitted by least squares to its own games, forgetting old games at 0.98 per game.
  - A blank planner (V = 0) plays random valid words.
- **Planner safeguards, each added after an observed failure:**
  - V is made non-decreasing (`np.maximum.accumulate`). A fit that bent downward at large m made the AI avoid narrowing things down, and games ran away.
  - Guesses that can't split the possible answers are never options (they get cost inf, or are excluded in `split_features.state`). A greedy policy-gradient agent looped forever replaying one useless probe.
- **Why a planner and not policy gradient for "any" mode:**
  - REINFORCE on the split facts never learned to value the win chance. With 2 words left, thousands of probes tie with the 2 candidates, so candidates are almost never sampled.
  - The result was either probing every time 2 were left (3.67) or never probing (3.64).
  - A learned "probe or go" gate learned to never probe (~3.55).
  - With the useless-guess rule, REINFORCE on split facts reaches 3.494.
  - The value-learning planner reaches 3.438 (the benchmark in `reports/2026-09-28_095629/`).
- **Performance:**
  - Split facts and costs are computed for all 12,972 guesses each turn. When few words are left, a per-answer group-size lookup is used; when many are left, dense per-guess counts.
  - Turn-1 counts are cached, and frozen agents cache costs for big candidate sets.
  - Typical cost: a best-guess game ~10–30 ms; a practice game (planner) ~30–60 ms.

## Conventions

- **Tuning claims need several seeds.** Single runs routinely mislead here, so compare settings across ≥6–8 seeds before changing defaults. Report numbers from `benchmark.py`, which plays every answer once with fixed tie-breaks.
- **Randomness:** secrets are drawn uniformly at random every practice game, and agents store only learned numbers, never words.
- **Ties:**
  - Practicing agents break exact ties at random.
  - Tested agents (`training = False`: skill checks, final exam, `evaluate.py`, benchmark, lookups) break exact ties with a fixed priority (`tie_ranks`), so same agent + same word = same game everywhere.
  - Ties must be exact (`TIE_TOLERANCE` = 1e-9). An old relative tolerance lumped near-equal openers together and made a 2,315-word score swing by 0.03 between runs.
- **Parallel work:**
  - These workloads are memory-bandwidth heavy. 11–12 parallel processes ran 10–20× slower each, and 3–6 is the sweet spot.
  - Set `OPENBLAS_NUM_THREADS=1` in worker processes (done in `benchmark.py` / `benchmark_tasks.py`).
  - Killing xargs children just starts the next queued job, so stop `xargs` itself first.
- **Threads:**
  - Never run a pure-Python CPU-heavy thread alongside training. The GIL convoy effect made a 0.5 s skill check take 25 s.
  - Numpy-heavy threads share fairly.
- **Dashboard data:**
  - New stats go in `TrainingSession.snapshot()` plus a tile or card in `index.html` / `render()` in `app.js`.
  - `knowledge()` on each agent describes what it has learned (weights or the V curve).
- **Honesty in reports:**
  - The 3.4201 optimum is over all 2,315 answers with any valid guess allowed.
  - Never compare it with a subset score (the dashboard's 100-word skill check, or `--quick`).
- **Output encoding:**
  - Colored terminal output uses ANSI codes; strip them with `sed 's/\x1b\[[0-9;]*m//g'` when piping.
  - When editing files from Python on Windows, pass `encoding="utf-8"` (the web files contain –, …, ·, −).
  - Long patch scripts are safer written to a file than piped through a bash heredoc.
- **Number format:** the page formats numbers as en-US.
