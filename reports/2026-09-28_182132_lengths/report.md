# How word length changes the game

Generated 2026-09-28 18:49 in 28 minutes.
Regenerate with `python benchmark.py --length-study`.

## Setup

- Word lists, built by `python -m wordle.wordlists` with one recipe for every length:
  - Answers (secrets): ENABLE dictionary words that are among the most frequent English words, simple plurals dropped. The frequency cutoff is set so that 5 letters gives 2,315 answers, as many as the official list, though not the same words.
  - Allowed guesses: every dictionary word of that length (ENABLE + SCOWL size 80 + the official Wordle list at 5 letters).
- Every answer played once, no guess limit, best guesses with fixed tie-breaks.
- Learned planner: 5 independent training runs of 300 games per length (± = 95% CI across runs; ± 0.000: the runs' averages agree to 3 decimals). Look-ahead and search: the median run; look-ahead weighs its top 10 guesses exactly each turn, search works out its top 20 guesses at every position to the end of every game. No strategy: 5 random seeds. Max-entropy has no randomness, so it's run once.
- These lists have no published optimum (the 3.4212 optimum is for the official Wordle list only). The search totals are the best results known for them: upper bounds on the optimum.

## Results

| Letters | Answers | Allowed guesses | No strategy (random possible word) | Hand-made max-entropy | Learned planner | Planner + look-ahead | Planner + search | Search total (time) | Bits per guess |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 669 | 1,346 | 6.585 ± 0.068 (worst 15) | 5.001 (worst 8) | 4.989 ± 0.001 (worst 8) | 4.900 (worst 8) | 4.873 (worst 8) | 3,260 (5.3 min) | 1.93 |
| 4 | 1,759 | 4,545 | 5.152 ± 0.039 (worst 13) | 4.111 (worst 6) | 4.109 ± 0.002 (worst 8) | 4.060 (worst 7) | 4.053 (worst 7) | 7,130 (1.8 min) | 2.66 |
| 5 | 2,315 | 13,314 | 4.123 ± 0.022 (worst 10) | 3.492 (worst 6) | 3.454 ± 0.024 (worst 7) | 3.435 (worst 6) | 3.435 (worst 6) | 7,952 (1.5 min) | 3.25 |
| 6 | 3,087 | 16,712 | 3.707 ± 0.020 (worst 11) | 3.136 (worst 5) | 3.124 ± 0.000 (worst 6) | 3.121 (worst 5) | 3.121 (worst 6) | 9,635 (2.3 min) | 3.71 |
| 7 | 3,152 | 25,276 | 3.358 ± 0.027 (worst 11) | 2.881 (worst 4) | 2.881 ± 0.001 (worst 5) | 2.875 (worst 5) | 2.875 (worst 5) | 9,062 (1.8 min) | 4.04 |
| 8 | 2,768 | 31,425 | 3.015 ± 0.020 (worst 8) | 2.629 (worst 4) | 2.629 ± 0.000 (worst 5) | 2.629 (worst 4) | 2.629 (worst 5) | 7,277 (0.4 min) | 4.35 |
| 9 | 2,207 | 29,644 | 2.750 ± 0.013 (worst 5) | 2.399 (worst 3) | 2.399 ± 0.000 (worst 4) | 2.399 (worst 3) | 2.399 (worst 4) | 5,294 (0.1 min) | 4.63 |
| 10 | 1,456 | 25,627 | 2.544 ± 0.022 (worst 5) | 2.260 (worst 3) | 2.260 ± 0.000 (worst 4) | 2.260 (worst 3) | 2.260 (worst 4) | 3,290 (0.0 min) | 4.65 |

![length study](length_study.png)

Bits per guess = log2(number of possible answers) / average guesses with search: how many halvings of the list each guess is worth on average. It doesn't depend on how many words a length happens to have, so it's the fairer way to compare lengths.
