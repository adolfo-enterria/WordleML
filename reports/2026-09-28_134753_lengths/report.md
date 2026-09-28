# How word length changes the game

Generated 2026-09-28 13:57 in 9 minutes.
Regenerate with `python benchmark.py --length-study`.

## Setup

- Word lists: common English words of each length, built by `python -m wordle.wordlists` with one recipe (dictionary words among the most frequent English words, simple plurals dropped; the cutoff is set so that 5 letters gives 2,315 answers, as many as the official list, though not the same words). Allowed guesses: every dictionary word of that length, up to 15,000.
- Every answer played once, no guess limit, best guesses with fixed tie-breaks.
- Learned planner: 5 independent training runs of 300 games per length (± = 95% CI across runs; ± 0.000: the runs' averages agree to 3 decimals). Look-ahead: the median run, weighing its top 10 guesses exactly each turn. No strategy: 5 random seeds. Max-entropy has no randomness, so it's run once.
- These lists are not the official Wordle lists, so the 3.4201 optimum does not apply to any row here.

## Results

| Letters | Answers | Allowed guesses | No strategy (random possible word) | Hand-made max-entropy | Learned planner | Planner + look-ahead | Bits per guess |
|---|---|---|---|---|---|---|---|
| 3 | 669 | 972 | 6.585 ± 0.068 (worst 15) | 5.220 (worst 9) | 5.187 ± 0.050 (worst 9) | 5.073 (worst 9) | 1.85 |
| 4 | 1,759 | 3,903 | 5.152 ± 0.039 (worst 13) | 4.118 (worst 6) | 4.113 ± 0.000 (worst 7) | 4.074 (worst 6) | 2.65 |
| 5 | 2,315 | 8,636 | 4.123 ± 0.022 (worst 10) | 3.503 (worst 6) | 3.452 ± 0.002 (worst 7) | 3.441 (worst 6) | 3.25 |
| 6 | 3,087 | 15,000 | 3.707 ± 0.020 (worst 11) | 3.136 (worst 5) | 3.132 ± 0.007 (worst 6) | 3.122 (worst 5) | 3.71 |
| 7 | 3,152 | 15,000 | 3.358 ± 0.027 (worst 11) | 2.916 (worst 5) | 2.899 ± 0.000 (worst 6) | 2.898 (worst 5) | 4.01 |
| 8 | 2,768 | 15,000 | 3.015 ± 0.020 (worst 8) | 2.630 (worst 4) | 2.630 ± 0.001 (worst 5) | 2.630 (worst 4) | 4.35 |

![length study](length_study.png)

Bits per guess = log2(number of possible answers) / average guesses with look-ahead: how many halvings of the list each guess is worth on average. It doesn't depend on how many words a length happens to have, so it's the fairer way to compare lengths.
