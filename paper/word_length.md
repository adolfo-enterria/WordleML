# How Word Length Shapes Wordle: Proven Floors, Few-Game Learning, and Cross-Length Transfer of a Value-Learning Planner

*Authors: [to be added]*  ·  *Draft, October 2026*

---

## Abstract

We study Wordle generalized to words of 3 to 10 letters and ask three questions: what is the fewest number of
guesses any strategy can average at each length, how many practice games a learning agent needs to approach it,
and what happens when an agent trained on one length plays another. We build common-word lists for every length
with a single recipe and evaluate every strategy exactly, by playing each possible answer once with
deterministic tie-breaks. The agent is a planner that learns a three-parameter value curve $V(m)$, the expected
number of further guesses when $m$ words are still possible, and chooses the guess with the lowest expected
cost. Optimal play needs at most 4.873 guesses on average at 3 letters and exactly 2.2596 at 10. Using a new
branch-and-bound procedure that considers every valid guess, we prove the optimum for 8, 9 and 10 letters
(2.6290, 2.3987, 2.2596); for 3 to 7 letters we bracket it between proven lower bounds and the best plans found,
which are stable across search widths and training seeds. A control with the same number of answers at every
length shows that the decline is due to length, not list size. The planner reaches within 1% of its long-run
performance after 1 to 3 practice games at every length, and plays perfectly at 8 to 10 letters after a median of
1 to 3 games; below 8 letters it levels off 0.2% to 2.2% above the best known plan, a gap that more training does
not close but test-time search does. A planner trained on 4-letter words plays 7-letter words as well as one
trained on 7 letters (2.8804 vs 2.8808 guesses, identical guesses in 98% of games), while planners trained on
8 to 10 letters are 7% to 8% worse on 3-letter words. The asymmetry follows from the range of $m$ each length
exposes during practice.

---

## 1. Introduction

In Wordle a player has to identify a hidden word by guessing words of the same length. After every guess each
letter is colored green (right letter, right place), yellow (in the word, elsewhere) or grey (not in the word, with
standard rules for repeated letters). The official game uses five-letter words. For its original list of 2,315
possible answers and 12,972 allowed guesses, the optimal strategy is known exactly: 7,920 guesses in total over
all answers, 3.4212 on average, opening with SALET (Selby, 2022; Bertsimas & Paskov, 2022).

Word length changes the game in two opposite ways. Longer words produce more possible color patterns
($3^L$ for $L$ letters), so each guess can carry more information; but the dictionary of valid words, the
number of plausible answers, and the structure of their letters all change with length too. We ask:

1. **The floor.** As a function of the number of letters, what is the lowest average number of guesses that
   any strategy can achieve?
2. **Learning speed.** How many practice games does a learning agent need to reach that floor, or its own
   best level, and does this depend on length?
3. **Transfer.** What happens when an agent trained on one length, for instance 4 letters, plays another,
   for instance 7?

Our contributions are:

- A reproducible benchmark of common-word lists for 3 to 10 letters, built with one recipe, and an exact
  evaluation protocol (every answer once, no randomness at test time).
- A lower-bound and proof procedure that considers every valid guess at every position, with a bound based on
  the largest number of color groups any guess can make, which never grows on subsets. It proves the
  optimum for 8, 9 and 10 letters in seconds and gives anytime lower bounds elsewhere.
- Measurements of the floor for every length, with a control at equal list size.
- Learning curves for a value-learning planner at every length, scored exactly at 29 checkpoints over 1,000
  games and 8 seeds, plus the effect of test-time search on barely trained planners.
- A full 8 x 8 cross-length transfer matrix and an explanation of its asymmetry.

## 2. Related work

**Exact Wordle.** Selby (2022) proved by exhaustive search that 7,920 total guesses (3.4212 on average) is
optimal for the original 2,315 answers with any of the 12,972 valid guesses allowed, and 8,122 (3.5084) in hard
mode; Bertsimas & Paskov (2022) gave an exact dynamic-programming solution. The frequently quoted 3.4201 refers to
the later 2,309-answer list and is a different game. These results concern five letters only.

**Mastermind.** Wordle belongs to the family of Mastermind-like code-breaking games; Knuth (1977) showed that
Mastermind can always be solved in five guesses. Our worst-case figures (Section 5.1) are the worst game of the
best average plan, not minimax optima.

**Learning to play.** Bhambri, Bhattacharjee & Bertsekas (2022) used rollout, a one-step policy-improvement
method on top of a base heuristic, and obtained near-optimal Wordle play. Deep reinforcement learning has
reached about 3.9 guesses after hundreds of thousands of games (Ho). The planner studied here is
closer to approximate dynamic programming: it knows the rules, learns only a value curve over the number of
remaining candidates, and uses one-step look-ahead.

## 3. Setup

### 3.1 The game and the objective

For a list of possible answers $A$ ($n = |A|$) and a list of allowed guesses $G \supseteq A$, a strategy is a
decision tree. For a set $S \subseteq A$ of answers still consistent with the feedback, let $T(S)$ be the total
number of guesses needed to solve every answer in $S$, counting the guess about to be played:

$$
T(\{x\}) = 1, \qquad T(S) = |S| + \min_{g \in G} \sum_{C \in \mathcal{P}_g(S),\, C \neq \{g\}} T(C),
$$

where $\mathcal{P}_g(S)$ partitions $S$ by the color pattern that guess $g$ produces, and the all-green group
(the win) costs nothing more. The score of a strategy is its total $T(A)$ over every answer played once; the
average is $T(A)/n$, an exact fraction, so we always report the integer total alongside it. There is no guess
limit: games run until solved.

### 3.2 Word lists

For each length $L \in \{3,\dots,10\}$ we use one recipe. **Answers** are words of the ENABLE dictionary that are
among the $N$ most frequent English words (FrequencyWords, 2018), with simple plurals removed. $N$ is chosen
once ($N \approx 32{,}000$), so that 5 letters gives 2,315 answers like the official list; the same $N$ is used for every length, so the
lists are natural: each length gets as many common words as it really has. **Allowed guesses** are every
dictionary word of that length (ENABLE, SCOWL size 80, and at 5 letters the official Wordle list). The official
Wordle lists are included as a reference, because their optimum is known.

**Table 1.** The lists, the number of possible color patterns, and the most color groups (besides the win) that
any single guess makes on the full list.

| Letters | Answers | Allowed guesses | Color patterns $3^L$ | Most groups one guess makes |
|---|---|---|---|---|
| 3 | 669 | 1,346 | 27 | 21 |
| 4 | 1,759 | 4,545 | 81 | 60 |
| 5 | 2,315 | 13,314 | 243 | 148 |
| 6 | 3,087 | 16,712 | 729 | 359 |
| 7 | 3,152 | 25,276 | 2,187 | 704 |
| 8 | 2,768 | 31,425 | 6,561 | 1,082 |
| 9 | 2,207 | 29,644 | 19,683 | 1,331 |
| 10 | 1,456 | 25,627 | 59,049 | 1,080 |
| Official Wordle (5) | 2,315 | 12,972 | 243 | 149 |

The lists differ in size (669 to 3,152 answers), so length and list size are confounded; Section 4.5 controls
for this.

### 3.3 The planner

The agent knows the rules: after each guess it keeps only the answers consistent with all feedback, and it
says the answer when only one is left. What it learns is a value curve

$$
V(m) = a + b \log_2 m + c\,(\log_2 m)^2,
$$

the expected number of further guesses when $m$ answers are still possible. For every valid guess $g$ it
computes the expected cost

$$
\mathrm{cost}(g) = \frac{1}{|S|} \sum_{x \in S} V\!\left(|C_g(x)|\right) \;-\; \frac{\mathbb{1}[g \in S]}{|S|}\,V(1),
$$

where $C_g(x)$ is the group answer $x$ falls into, and the correction removes the cost of the case where $g$ is
the answer. It plays the cheapest guess. Two safeguards were added earlier in this project after observed failures: $V$ is
made non-decreasing in $m$ (a running maximum), and guesses that cannot split the remaining answers are never
considered unless they could be the answer.

After each practice game, every non-winning guess contributes one data point (words left after it, guesses it
still took), and $(a, b, c)$ is refit by least squares with exponential forgetting (each game discounts earlier
experience by 0.98). A blank planner ($V \equiv 0$) values all useful guesses equally and plays random valid
words. During practice, secrets are drawn uniformly at random and exact ties are broken at random. When it is
tested, ties are broken by a fixed priority, so the same planner always plays the same game.

### 3.4 Exact evaluation

A frozen planner is deterministic, so its total over all answers is a property of its decision tree. Instead of
playing each answer separately we walk the tree once, memoizing positions; this gives the same total as playing
every game (checked: 8,075 both ways in a test at 5 letters) at about a third of the cost, which allows scoring at
29 checkpoints per training run.

### 3.5 Upper bounds: search guided by the planner

The search of Section 3.1's recursion is exact within a candidate set: at each position it considers the planner's
$w$ favourite guesses (width $w$) and computes, for each, the exact total to the end of every game, with branch
and bound, budgets passed down, and memoization of exact values and proven lower bounds. Its total is the total of
a real strategy, hence an upper bound on the optimum. On the official list it reaches 7,920, the proven optimum.

### 3.6 Lower bounds and proofs over every guess

To bound the optimum from below we consider every valid guess, not the planner's candidates.

**Even-split bound with $k$ groups.** Let $f_k(n)$ be the total of an idealized tree in which every guess is the
answer to one word and splits the other $n-1$ as evenly as possible into at most $k$ groups:
$f_k(1)=1$, $f_k(n) = n + \sum_i f_k(q_i)$ with the $q_i$ an even split of $n-1$ into $\min(k, n-1)$ parts. When
$f_k$ is convex (checked numerically), even splits are the cheapest, and $T(S) \geq f_k(|S|)$ whenever no guess can
make more than $k$ groups besides the win on any subset of $S$.

**Lemma.** Let $k(S)$ be the largest number of non-winning color groups any valid guess makes on $S$. Then
$S' \subseteq S$ implies $k(S') \le k(S)$.

*Proof.* For a guess $g$, the patterns it produces on $S'$ are a subset of those on $S$. If $g \in S'$, both counts
lose the all-green pattern. If $g \in S \setminus S'$, the all-green pattern is among $S$'s patterns but not
$S'$'s. If $g \notin S$, neither count contains it. In each case the count on $S'$ is at most that on $S$. $\square$

So $f_{k(S)}$ bounds every position below $S$. At 3 letters $k$ is 21 at the root and much smaller deeper in the
tree, where the generic bound would assume $3^3 - 1 - 3 = 23$ groups everywhere.

**Bounds used, cheapest first.** For a set $S$: (i) $f_{k}(|S|)$; (ii) if some answer in $S$ separates all others,
$T(S) = 2|S| - 1$ exactly, otherwise $T(S) \ge 2|S|$; (iii) a one-level bound computed for every guess at once:
$|S| + \sum_{C} f_{k(S)}(|C|)$ over $g$'s non-winning groups, whose minimum over $g$ bounds $T(S)$;
(iv) branch and bound over every valid guess in increasing order of the one-level bound, where each candidate's
groups first receive bounds (i)-(iii) and only then deeper search.

**Anytime root.** Every opener keeps a proven lower bound in a priority queue. The weakest is always refined next,
with a budget just above the second weakest. The minimum over the queue is a valid lower bound on the optimum at
every moment, so a time limit still yields a proof. Started with the best known total as the budget, the
procedure either finds a better strategy or proves none exists. The search is checked against brute force on
small sets (exact equality for 6 word families), against a definitional computation of the one-level bound on 4
lists, and on the official list, where its bound must stay at or below the proven 7,920 (it reached 7,078 in 20
minutes).

### 3.7 Experiments

| Experiment | What is measured |
|---|---|
| Training | 8 planners per list trained from scratch for 1,000 practice games; exact totals at 29 checkpoints (0, 1, 2, ..., 1,000 games). |
| Floor | Search at width 20 guided by the median, best and worst seed (by final total), and at width 50 guided by the median seed (from 4 letters on; at 3 letters width 50 did not finish in over an hour). Then the proof of Section 3.6 with the best total as budget, up to 20 minutes per list. |
| Search after few games | Width-20 search guided by the median seed's planner after 1, 3, 10 and 1,000 games. |
| Transfer | Every final planner (8 lists x 8 seeds) scored exactly on every list; a game-by-game comparison of the median 4- and 7-letter planners on 7-letter words. |
| Equal size | 669 answers at every length (the 3-letter list; 3 random subsets at 4 to 10 letters, all guesses still allowed), each searched at width 20 and proved for up to 3 minutes. |
| Structure | The largest number of groups per guess, and how the best and the planner's openers split the list. |
| Timing | Milliseconds per practice game of a trained planner, 100 games, one process, idle machine. |

Uncertainties are 95% confidence intervals across seeds (t distribution). All scores are exact totals over the
lists, so there is no sampling error over answers.

## 4. Results

### 4.1 The floor

**Table 2.** The fewest guesses possible. "Planner alone": the 8 planners after 1,000 games (mean ± 95% CI).
"Best plan found": the lowest total of any search. "Proven lower bound": from Section 3.6. Bits per guess =
$\log_2 n$ divided by the best average.

| Letters | Planner alone | Best plan found (total, opener) | Proven lower bound (total) | Optimum | Worst game of best plan | Bits per guess |
|---|---|---|---|---|---|---|
| 3 | 4.9890 ± 0.0006 | 4.8729 (3,260, TAE) | 4.3034 (2,879) | in [4.303, 4.873] | 8 | 1.93 |
| 4 | 4.1092 ± 0.0017 | 4.0534 (7,130, TALE) | 3.7533 (6,602) | in [3.753, 4.053] | 7 | 2.66 |
| 5 | 3.4471 ± 0.0047 | 3.4350 (7,952, TRACE) | 3.0428 (7,044) | in [3.043, 3.435] | 6 | 3.25 |
| 6 | 3.1283 ± 0.0049 | 3.1208 (9,634, CARNET) | 2.9044 (8,966) | in [2.904, 3.121] | 6 | 3.71 |
| 7 | 2.8808 ± 0.0005 | 2.8750 (9,062, CANTIER) | 2.8242 (8,902) | in [2.824, 2.875] | 5 | 4.04 |
| 8 | 2.6291 ± 0.0001 | 2.6290 (7,277, CERATINS) | 2.6290 (7,277) | **2.6290, proven** | 4 | 4.35 |
| 9 | 2.3987 ± 0.0000 | 2.3987 (5,294, CARTELISE) | 2.3987 (5,294) | **2.3987, proven** | 4 | 4.63 |
| 10 | 2.2596 ± 0.0000 | 2.2596 (3,290, CENTRALISE) | 2.2596 (3,290) | **2.2596, proven** | 3 | 4.65 |
| Official (5) | 3.4371 ± 0.0015 | 3.4212 (7,920, SALET) | 3.0575 (7,078) | 3.4212 (Selby, 2022) | 5 | 3.27 |

![The fewest guesses possible, by word length](../reports/2026-10-03_limits/floor.png)

*Figure 1. Best plan found (filled markers: proven optimum; shaded bars: proven bracket), the planner alone, and
the best plans with 669 answers at every length.*

The floor falls steeply and then flattens: 4.87 guesses at 3 letters, 3.44 at 5, 2.63 at 8 and 2.26 at 10. At
8 to 10 letters the optimum is proven, and the proof was quick (12, 8 and 14 seconds) because the one-level bound
already eliminated all but 6, 1 and 1 openers. At 3 to 7 letters, the proof ran out of its 20 minutes and the
lower bounds remain loose, from 1.8% (7 letters) to 12% (3 letters) below the best plan. Three facts suggest
that the best plans are at or very near the optimum there as well:

- **Agreement.** Every search, at widths 20 and 50 and guided by three differently trained planners, found the
  same total, except at 6 letters where totals differed by one guess (9,634 vs 9,635).
- **Calibration.** On the official list, where the optimum is known, the same search reached it exactly
  (7,920), at both widths and from all three seeds.
- **Wider search.** Increasing the width from 20 to 50 never improved a total except the one guess at 6 letters.

The worst game of the best plan shrinks from 8 guesses at 3 letters to 3 at 10. At 10 letters 74% of answers are
solved in two guesses and none needs more than three; at 3 letters only 11% are solved within three.

### 4.2 Learning speed

**Table 3.** Average guesses of the planners (mean over 8 seeds) at selected checkpoints, the long-run level
(every checkpoint from 200 games on), the first checkpoint at which the mean curve is within 1% and 0.5% of that
level, the long-run gap to the best plan, and the time per practice game.

| Letters | 0 games (random) | 1 game | 10 games | 100 games | Long run | Within 1% | Within 0.5% | Gap to best plan | ms per game |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 9.076 | 5.054 | 4.968 | 4.983 | 4.9807 ± 0.0075 | 2 | 2 | +2.21% | 3 |
| 4 | 8.136 | 4.232 | 4.113 | 4.109 | 4.1095 ± 0.0009 | 2 | 2 | +1.38% | 11 |
| 5 | 5.381 | 3.604 | 3.474 | 3.461 | 3.4490 ± 0.0036 | 2 | 25 | +0.41% | 26 |
| 6 | 4.638 | 3.152 | 3.145 | 3.127 | 3.1265 ± 0.0009 | 1 | 13 | +0.18% | 46 |
| 7 | 3.916 | 2.906 | 2.881 | 2.881 | 2.8806 ± 0.0003 | 1 | 2 | +0.19% | 66 |
| 8 | 3.736 | 2.693 | 2.629 | 2.629 | 2.6291 ± 0.0000 | 2 | 2 | +0.00% | 105 |
| 9 | 3.524 | 2.478 | 2.400 | 2.399 | 2.3987 ± 0.0000 | 2 | 2 | +0.00% | 132 |
| 10 | 2.893 | 2.291 | 2.260 | 2.260 | 2.2596 ± 0.0000 | 3 | 3 | +0.00% | 112 |
| Official (5) | 5.263 | 3.449 | 3.441 | 3.437 | 3.4371 ± 0.0005 | 1 | 1 | +0.47% | 25 |

![Learning curves](../reports/2026-10-03_limits/learning_curves.png)

*Figure 2. Total guesses relative to the best plan known, per training seed (thin) and averaged (thick), against
practice games on a logarithmic axis.*

The planner reaches within 1% of its long-run level after 1 to 3 practice games at every length, and within 0.5%
after 2 to 25. The number of games hardly depends on length. Where the optimum is proven, the planner alone plays
perfectly after a median of 2 (8 letters), 1 (9) and 3 (10) games; at 9 and 10 letters every checkpoint from 200
games on is perfect, and at 8 letters 64% are, the rest one guess above (7,278). Below 8 letters the planner levels
off above the best plan, by 2.21% at 3 letters, 1.38% at 4, 0.41% at 5 and 0.18% to 0.19% at 6 and 7, and more
training changes this little: between 100 and 1,000 games the mean moves by at most 0.41% (5 letters) and by 0.13%
or less at every other length.

Cost per game, on the other hand, grows with length, from 3 ms at 3 letters to 105 to 132 ms at 8 to 10 letters,
because every decision evaluates every valid guess against the remaining answers.

**Table 4.** Width-20 search guided by the median planner after few games. Totals over every answer; bold = best
total known.

| Letters | After 1 game | After 3 games | After 10 games | After 1,000 games |
|---|---|---|---|---|
| 3 | 3,269 | **3,260** | **3,260** | **3,260** |
| 4 | 7,132 | 7,132 | 7,134 | **7,130** |
| 5 | 8,392 | 7,953 | **7,952** | **7,952** |
| 6 | 9,635 | 9,636 | 9,635 | 9,635 |
| 7 | **9,062** | **9,062** | **9,062** | **9,062** |
| 8 | **7,277** | **7,277** | **7,277** | **7,277** |
| 9 | **5,294** | **5,294** | **5,294** | **5,294** |
| 10 | **3,290** | **3,290** | **3,290** | **3,290** |
| Official (5) | **7,920** | **7,920** | **7,920** | **7,920** |

With search on top, a few games of training suffice. After 3 games every length is within 2 guesses (in total)
of the best plan known. The one large miss is 5 letters after a single game, where that seed's planner still
averaged 4.27 guesses on its own and guided the search poorly.

### 4.3 Transfer across lengths

![Transfer matrix](../reports/2026-10-03_limits/transfer.png)

*Figure 3. Change in total guesses when planners trained on one length (rows) play another (columns), relative
to planners trained on the column's length (mean of 8 seeds each). The full matrix is in Appendix A.*

**4 letters to 7 letters.** All 8 planners trained on 4-letter words score 9,079 on the 3,152 seven-letter
answers (2.8804 on average); the 8 trained on 7 letters score 9,076 to 9,081 (2.8808 on average). Game by game,
the median 4-letter and 7-letter planners play identical guesses on 3,082 of 3,152 answers (97.8%); the 4-letter
one needs one more guess on 9 answers and one fewer on 10. Both open with SALTIER, and search guided by either
reaches 9,062.

**Short to long transfers; long to short does not.** Planners trained on 3 to 7 letters lose at most 0.19% on
longer lists, except on 5 letters (+1.15% from 3, +0.86% from 4). Planners trained on 8 to 10 letters lose 6.9% to
7.8% on 3-letter words, about 2% on 4 letters and 0.8% to 1.4% on 5 and 6. Untrained planners are 28% to 98% worse
than trained ones, so any training at all captures most of the gain.

**The native planner is not always best.** Planners trained on 4 and 5 letters play 3-letter words 0.43% and 0.48%
better than 3-letter planners do.

![Learned value curves](../reports/2026-10-03_limits/value_curves.png)

*Figure 4. The value curve $V(m)$ learned at each length (median of 8 planners), drawn up to the list's number of
answers.*

### 4.4 What one guess can tell

**Table 5.** How the openers split each list: number of groups (the win included), words left on average over
the answers, and the largest group.

| Letters | Best plan's opener | Planner's opener (median seed) |
|---|---|---|
| 3 | TAE: 22 groups, 139.3 left, largest 256 | TAE: 22, 139.3, 256 |
| 4 | TALE: 57, 133.6, 352 | TARE: 60, 146.2, 389 |
| 5 | TRACE: 149, 76.8, 242 | TRACE: 149, 76.8, 242 |
| 6 | CARNET: 359, 36.6, 136 | CARNET: 359, 36.6, 136 |
| 7 | CANTIER: 684, 16.1, 115 | SALTIER: 669, 13.1, 79 |
| 8 | CERATINS: 1,082, 6.3, 35 | CERATINS: 1,082, 6.3, 35 |
| 9 | CARTELISE: 1,331, 2.8, 14 | CARTELISE: 1,331, 2.8, 14 |
| 10 | CENTRALISE: 1,080, 1.9, 12 | CENTRALISE: 1,080, 1.9, 12 |
| Official (5) | SALET: 148, 71.3, 221 | REAST: 147, 71.8, 227 |

A single 3-letter guess leaves 139 words on average; a single 10-letter guess leaves fewer than 2. At 7 letters
the optimal plan's opener, CANTIER, narrows the list less on average than SALTIER (16.1 vs 13.1 words left): the
best first guess is not simply the most informative one.

### 4.5 Equal list size

**Table 6.** 669 answers at every length (mean over 3 random subsets; the 3-letter list has exactly 669).

| Letters | Planner alone | Best plan found | Proven lower bound | Optimum | Bits per guess |
|---|---|---|---|---|---|
| 3 | 4.9895 | 4.8729 | 4.2362 | bracketed | 1.93 |
| 4 | 3.6178 | 3.5800 | 3.3159 | bracketed | 2.62 |
| 5 | 3.0409 | 3.0274 | 2.8700 | bracketed | 3.10 |
| 6 | 2.7419 | 2.7369 | 2.7369 | proven, every subset | 3.43 |
| 7 | 2.4943 | 2.4943 | 2.4943 | proven, every subset | 3.76 |
| 8 | 2.3363 | 2.3348 | 2.3348 | proven, every subset | 4.02 |
| 9 | 2.1958 | 2.1953 | 2.1953 | proven, every subset | 4.28 |
| 10 | 2.1390 | 2.1390 | 2.1390 | proven, every subset | 4.39 |

At equal size the floor declines monotonically with length, from 4.87 to 2.14, and is proven optimal from 6
letters on. (The 3-letter lower bound here, 4.2362, comes from a 3-minute proof; the 20-minute proof of Table 2,
on the same list, reached 4.3034.) In 11 of the 15 subsets from 6 letters on, the planner alone already plays
optimally.

## 5. Discussion

### 5.1 Why longer words need fewer guesses

The number of color patterns grows as $3^L$, but the number of groups a real guess can make grows more slowly and
saturates (Table 1: 21 groups at 3 letters, 704 at 7, 1,331 at 9 and 1,080 at 10), because words of a language
share letters and positions. Information per guess, measured as $\log_2 n$ over the optimal average, rises from
1.93 bits at 3 letters to 4.65 at 10 and flattens from 9 letters (Table 2); at equal list size it rises from 1.93
to 4.39 (Table 6). Since a strategy needs at least one guess to tell the answers apart and one to say it, no list
can average fewer than $2 - 1/n$ guesses, the case of an opener that is itself an answer and separates all the
others. At 10 letters the optimum is
2.26: the best opener separates most but not all answers (the largest remaining group has 12 words), so 26% of the
answers need a third guess.

### 5.2 Why the planner learns in a few games

The planner does not have to discover the rules, the consequences of a guess, or the space of possible answers;
all of that is built in. It learns only three numbers. For choosing guesses what matters about $V$ is its shape,
not its scale. The decision compares guesses by the expected value of $V$ over the group sizes they produce, and
nearly any increasing, concave curve in $\log m$ makes the planner prefer guesses that leave fewer words. A single
game with a blank planner provides several data points (words left, guesses still needed) that already define
such a curve. By contrast, in an earlier benchmark in this repository on the official 5-letter list
(`reports/2026-09-28_180842/`), two policy-gradient (REINFORCE) agents needed hundreds of practice games and, after
1,000, averaged 3.4949 and 3.6084 guesses, against 3.4380 for the planner.

### 5.3 Why the planner levels off above the floor

The long-run gap at 3 to 7 letters (0.2% to 2.2%) barely moves with more games, so it is not mainly a training
problem. Two limitations cause it: the planner plans one guess ahead, and $V$ depends only on the number of words
left, not on which words they are. Moreover, training fits $V$ to the planner's own games, which estimates how
well it plays, not which $V$ would make it play best. The transfer results show this directly: planners trained
on 4 and 5 letters play 3-letter words better than 3-letter planners. Test-time search removes both limitations at
once and reaches the best known totals after a few games of training (Table 4).

### 5.4 Why transfer is asymmetric

A planner's knowledge is a curve over $m$, the number of words still possible, and it only learns $V$ for the
values of $m$ it encounters. After its first guess, a 3- or 4-letter planner faces sets of up to 256 or 389 words,
so its $V$ is fitted over almost the whole range that matters on any length. After a first guess at 8, 9 or 10
letters at most 35, 14 or 12 words remain (Table 5), so those planners never see large $m$. Their fitted quadratic
peaks within that range, and the monotonicity safeguard then keeps $V$ flat beyond it. In Figure 4 the 8- to
10-letter curves stop rising within the first 10 to 15 words; the median 10-letter planner's curve is flat at
1.74 from $m = 5$ on (Appendix B). On 3-letter words, where a guess routinely leaves dozens or hundreds of words,
such a planner cannot distinguish a guess that leaves 20 words from one that leaves 200. In the other direction,
a 4-letter curve covers every $m$ that occurs on 7-letter words (at most 79 after the opener both planners play).
Its values are higher than the 7-letter curve's ($V(10) = 2.41$ vs $2.05$), but both rise steadily over that
range, and what matters for choosing a guess is how $V$ ranks different splits, not its level: the two planners
make the same choices in 98% of games.

A practical consequence is that a planner meant to play every length should be trained on short words, or on a
mixture, rather than on long ones.

## 6. Limitations

- **Unproven optima at 3 to 7 letters.** The lower bounds there come from 20-minute runs of a Python
  implementation and are loose. Our evidence that the best plans are optimal (agreement across widths and seeds,
  exact calibration on the official list) is strong but not a proof.
- **Search candidates.** Upper bounds come from searches restricted to the planner's favourite guesses. Width 50
  could not be run at 3 letters within an hour.
- **Lists.** The results depend on one construction of word lists (frequency cutoff, dictionaries, plural rule).
  The equal-size control uses three random subsets of one size.
- **Objective.** We minimize the average over equally likely answers with no guess limit and normal rules. A
  6-guess limit, hard mode, non-uniform answer priors, or minimax objectives could give different floors. Our
  worst-case figures are those of the best average plan, not minimax optima.
- **Learning curves.** The long-run level depends on the forgetting factor (0.98), and checkpoint scores of
  individual seeds fluctuate as $V$ keeps adapting.
- **Timing.** Per-game times are machine-dependent (Section 7). The training runs themselves shared the machine
  (and it slept during them), so timings were measured separately.

## 7. Reproducibility

The study runs with `python benchmark.py --limits-study` and writes every result to
`reports/2026-10-03_limits/`: phase results as JSON, CSV tables, the charts above, and `report.md`, which is
rebuilt from the JSON files with `python -m analysis.limits_report reports/2026-10-03_limits`. Each task's result
is saved as it finishes, so an interrupted run resumes where it stopped. Training is seeded; the planner, search
and proofs are deterministic. The bound and proof code is in `analysis/limits.py` and is tested against brute
force (`tests/test_limits.py`); the 116 tests of the repository pass. The experiments ran on one Windows 11 machine
with 12 logical cores (Python 3.12.10, NumPy 2.4.2), with 4 to 6 worker processes, for roughly three hours of
computation. Any saved model can be evaluated on any length, for example
`python evaluate.py --length 7 --model models/agent_common4.npz`.

## 8. Conclusion

Word length dominates the difficulty of Wordle: optimal play needs 4.87 guesses at 3 letters (best plan found)
and exactly 2.26 at 10, and the trend persists when the number of answers is held fixed. The optimum is proven for
8 to 10 letters. A planner that knows the rules and learns a three-number value curve gets within 1% of its best
after 1 to 3 practice games at any length, plays perfectly at 8 to 10 letters, and reaches the best known plans
everywhere when combined with search. Because it learns about numbers of remaining words rather than about words,
it transfers across lengths. It transfers well from short to long words, which expose the whole range of those
numbers, and poorly from long to short. Proving the optimum at 3 to 7 letters is the main open problem left by
this study.

## References

- Bertsimas, D., & Paskov, A. (2022). An exact and interpretable solution to Wordle.
- Bhambri, S., Bhattacharjee, A., & Bertsekas, D. (2022). Reinforcement learning methods for Wordle: a
  POMDP/adaptive control approach. arXiv:2211.10298.
- Ho, A. Wordle solver using deep reinforcement learning. https://andrewkho.github.io/wordle-solver/
- Knuth, D. E. (1977). The computer as Master Mind. *Journal of Recreational Mathematics*, 9(1), 1-6.
- Selby, A. (2022). The best strategies for Wordle.
  https://sonorouschocolate.com/notes/index.php/The_best_strategies_for_Wordle
- Word lists: ENABLE (public domain); SCOWL, size 80 (K. Atkinson, https://wordlist.aspell.net/);
  FrequencyWords, English 2018 (H. Dave, https://github.com/hermitdave/FrequencyWords, CC BY-SA 4.0).
- Summary of optimal Wordle results: L. Poirrier, https://www.poirrier.ca/notes/wordle-optimal/

## Appendix A. Full transfer matrix

Average guesses when planners trained on the row's length play the column's list (mean of 8 seeds), and the
change against planners trained on the column's length.

| Trained on \ tested on | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| 3 | 4.9890 (native) | 4.1106 (+0.04%) | 3.4868 (+1.15%) | 3.1341 (+0.19%) | 2.8804 (-0.02%) | 2.6290 (0.00%) | 2.3987 (0.00%) | 2.2596 (0.00%) |
| 4 | 4.9677 (-0.43%) | 4.1092 (native) | 3.4766 (+0.86%) | 3.1339 (+0.18%) | 2.8804 (-0.02%) | 2.6290 (0.00%) | 2.3987 (0.00%) | 2.2596 (0.00%) |
| 5 | 4.9651 (-0.48%) | 4.1137 (+0.11%) | 3.4471 (native) | 3.1267 (-0.05%) | 2.8807 (0.00%) | 2.6292 (0.00%) | 2.3987 (0.00%) | 2.2596 (0.00%) |
| 6 | 5.0148 (+0.52%) | 4.1216 (+0.30%) | 3.4509 (+0.11%) | 3.1283 (native) | 2.8809 (0.00%) | 2.6292 (+0.01%) | 2.3987 (0.00%) | 2.2596 (0.00%) |
| 7 | 5.0744 (+1.71%) | 4.1303 (+0.52%) | 3.4477 (+0.02%) | 3.1243 (-0.13%) | 2.8808 (native) | 2.6293 (+0.01%) | 2.3987 (0.00%) | 2.2596 (0.00%) |
| 8 | 5.3326 (+6.89%) | 4.1906 (+1.98%) | 3.4922 (+1.31%) | 3.1557 (+0.88%) | 2.8790 (-0.06%) | 2.6291 (native) | 2.3987 (0.00%) | 2.2596 (0.00%) |
| 9 | 5.3791 (+7.82%) | 4.1992 (+2.19%) | 3.4951 (+1.39%) | 3.1564 (+0.90%) | 2.8794 (-0.05%) | 2.6290 (0.00%) | 2.3987 (native) | 2.2596 (0.00%) |
| 10 | 5.3500 (+7.24%) | 4.1938 (+2.06%) | 3.4901 (+1.25%) | 3.1524 (+0.77%) | 2.8796 (-0.04%) | 2.6291 (0.00%) | 2.3987 (0.00%) | 2.2596 (native) |
| Untrained | 9.0762 | 8.1364 | 5.3806 | 4.6378 | 3.9156 | 3.7363 | 3.5242 | 2.8929 |

## Appendix B. Learned value curves

Coefficients of $V(m) = a + b\log_2 m + c(\log_2 m)^2$ for the median planner of each length after 1,000 games,
and the resulting (non-decreasing) $V$ at selected $m$.

| Letters | $a$ | $b$ | $c$ | $V(2)$ | $V(10)$ | $V(100)$ | $V(1000)$ |
|---|---|---|---|---|---|---|---|
| 3 | 1.035 | 0.571 | -0.019 | 1.59 | 2.73 | 4.00 | 4.87* |
| 4 | 1.018 | 0.499 | -0.024 | 1.49 | 2.41 | 3.28 | 3.62 |
| 5 | 1.053 | 0.415 | -0.025 | 1.44 | 2.15 | 2.69 | 2.69 |
| 6 | 1.025 | 0.374 | -0.024 | 1.37 | 2.00 | 2.45 | 2.45 |
| 7 | 1.025 | 0.423 | -0.035 | 1.41 | 2.05 | 2.31 | 2.31 |
| 8 | 1.019 | 0.601 | -0.090 | 1.53 | 2.02 | 2.02 | 2.02 |
| 9 | 0.991 | 0.492 | -0.053 | 1.43 | 2.04 | 2.12 | 2.12 |
| 10 | 1.001 | 0.856 | -0.233 | 1.63 | 1.74 | 1.74 | 1.74 |
| Official (5) | 1.030 | 0.388 | -0.024 | 1.39 | 2.05 | 2.54 | 2.54 |

\* Beyond the 3-letter list's 669 answers; shown for comparison only.

All curves start near $V(1) \approx 1$ (one word left: one more guess), as they must. They differ in how fast they
rise and where they flatten: the longer the words, the sooner the curve stops growing, matching the range of $m$
each length produces in practice (Table 5).

## Appendix C. Proof statistics

| Letters | Openers with one-level bound below the best total | Ruled out in 20 min | Rounds at the root | Time |
|---|---|---|---|---|
| 3 | 1,346 | 0 | 34,677 | 20 min (stopped) |
| 4 | 4,545 | 0 | 5,968 | 20 min (stopped) |
| 5 | 12,865 | 388 | 810 | 20 min (stopped) |
| 6 | 15,900 | 192 | 206 | 20 min (stopped) |
| 7 | 6,407 | 233 | 235 | 20 min (stopped) |
| 8 | 6 | 6 | 7 | 12 s (proven) |
| 9 | 1 | 1 | 1 | 8 s (proven) |
| 10 | 1 | 1 | 1 | 14 s (proven) |
| Official (5) | 12,434 | 913 | 1,192 | 20 min (stopped) |

At 3 and 4 letters the trees are deep (worst games of 8 and 7 guesses) and every opener stays alive for a long
time. At 5 to 7 letters there are thousands of openers to rule out, each with hundreds of groups to bound. Both
point to the same next step for a full proof: a faster implementation and much longer runs.
