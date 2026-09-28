# WordleML

A terminal Wordle clone, AIs that get better at Wordle by playing it, a live
dashboard to watch them learn, and a benchmark that produces citable numbers.

## Quick start

```
pip install -r requirements.txt
python dashboard.py                    # watch the AI train live in your browser
python benchmark.py                    # reproducible report in reports/<date-time>/ (~20 min)
python play.py                         # play Wordle yourself
python play.py --watch                 # watch the trained AI play, with its top choices
python evaluate.py                     # score the trained AI on every answer word
pytest                                 # run the tests
```

## The AIs

Both know the rules. They keep track of which of the 2,315 answers still match
the colors, and say the answer once only one is left. Games have no guess limit:
they continue until the word is found. Each practice game uses a random secret.

**"Any valid word" (the main AI, a planner).**
- It may guess any of the 12,972 valid words, including *probe* words that can't be the answer but test several letters at once.
- It can work out how any guess would split the words that are still possible.
- What it learns from its own games is one curve, *V(m)*: how many more guesses it usually needs when *m* words are possible. That's 3 learned numbers.
- It plays the guess with the lowest expected cost. Winning right away costs 0; otherwise the cost is the average *V* of the leftover groups.
- A blank AI (V = 0) thinks every guess is equally good, so it plays random words.
- Nobody tells it when to probe or when to go for the win. Both come out of the sum, which means your strategies emerge on their own:
  - A weak opener is followed by fresh common letters.
  - _IGHT with many options → play a word that tests several first letters.
  - With 2 words left → go for the win.

**"Only words that could be the answer".**
- It can never probe.
- It learns 3 weights on word facts with REINFORCE (policy gradient).
- It plateaus around 3.6 guesses, because probing is exactly what it's missing.

## The dashboard

`python dashboard.py` opens http://127.0.0.1:8765. It has:

- **Controls:** Pause, Continue, Reset (a blank AI at game 0), a "Guesses allowed" mode selector, "Games to train", and speed.
- **Skill check:** every 10 games at first, then every ~25, it plays the same 100 words with its best guesses, so luck can't move the curve.
- **Final exam:** when the run ends, it plays all 2,315 words. That's the number to compare with the proven optimum of 3.4201.
- **What it has learned:** the V(m) curve, or the weights in "possible" mode.
- **What it learned to do:** how often it probes, by how many words were left; the latest probe it made; and whether it follows a weak opener with fresh letters.
- **Openers it tried:** with averages, plus how many games trial and error would need to tell openers apart.
- **Word difficulty:** look up any word (`/?word=foyer`), or analyze all 2,315.

## Benchmark and results

`python benchmark.py` plays every one of the 2,315 answers once with every strategy, using
best guesses and fixed tie-breaks, so no score depends on luck. Learners are trained from
scratch over several seeds, and results are the mean ± 95% CI across seeds. It writes
`report.md`, the charts, CSVs and `config.json` (seeds, settings, file hashes) to
`reports/<date-time>/`. `--quick` does a fast smoke test on a word subset; don't cite it.

Results from the full run in `reports/2026-09-28_095629/` (all 2,315 answers; 10 seeds for
learners unless noted):

| Strategy | Average guesses | Worst |
|---|---|---|
| No strategy: random word that could be the answer (5 seeds) | 4.104 ± 0.017 | 9 |
| Blank planner: random valid words | 5.263 | 10 |
| Policy gradient, only possible words | 3.609 ± 0.013 | 8 |
| Policy gradient, any word (5 seeds) | 3.495 ± 0.001 | 5 |
| Hand-made: most information (max entropy) | 3.463 | 6 |
| **Learned planner (500 practice games)** | **3.438 ± 0.001** | 6 |
| Proven optimum (literature) | 3.4201 | 5 |

- **Near-optimal:** the learned planner is 0.52% above the proven optimum. It gets within 0.05 guesses of its final skill after one practice game.
- **Probing emerges:** with 2 words left it never probes; with 6–20 left it probes 56% of the time, and with 21+ left, 96%. For NIGHT it plays REAST → LOGIN, a word that can't win, and that narrows the 35 candidates to 1.
- **Openers:** it picks REAST on its own. Scored exactly, with its own follow-up play, CRATE (3.433) and SALET (3.434) would be slightly better. Telling a 0.05-guess difference apart by trial and error would take ~2,700 games per opener, which is why openers are scored exactly.
- **What makes words hard (for this AI):** how much the opener narrows them down (+0.18 guesses per halving left), rare letters, and big one-letter look-alike families (CATCH/HATCH/MATCH…); repeated letters matter little for this AI. Together these explain ~26% of the difference between words.
- **Published results:** exact optimum 3.4201 with SALET (Bertsimas & Paskov 2022; Selby 2022); hard mode ≈ 3.5076; deep RL ≈ 3.9 after hundreds of thousands of games (Ho); rollout RL near-optimal (Bhambri et al., arXiv:2211.10298). The benchmark report lists them with links.

## Layout

```
wordle/     game rules + scoring, word lists, the precomputed pattern table, terminal colors
agents/     planner (planning_agent.py), policy-gradient learner (learning_agent.py), facts, baselines
analysis/   word difficulty, hand-made strategies, statistics, benchmark tasks, report charts
live/       dashboard: training session, web server, page
dashboard.py, benchmark.py, play.py, evaluate.py
data/       answers.txt (possible secrets), allowed_guesses.txt, patterns.npz (built on first use)
```
