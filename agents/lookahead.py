"""Look-ahead ("rollout"): think further ahead when it counts, using what the planner learned.

The planner picks the guess whose leftover groups look cheapest according to its
learned curve V(m), which only knows how MANY words are left, not which ones.

The look-ahead agent takes the planner's `width` favourite guesses and, for each
one, works out EXACTLY how many more guesses the planner would then need: it
walks the planner's whole decision tree over every answer that is still possible
(no sampling, no luck). It plays the guess with the lowest exact total.

The planner's own choice is always among the candidates, so this can never do
worse than the planner (the classic "rollout" policy-improvement step; see
Bhambri, Bhattacharjee & Bertsekas, arXiv:2211.10298). It learns nothing new;
it just spends more thinking time at test time. Results are cached per set of
possible words, so the opener is worked out once and later turns reuse the work.
"""
import numpy as np

from agents.base import Agent
from agents.features import TIE_TOLERANCE, cheapest


class LookaheadAgent(Agent):
    name = "lookahead"

    def __init__(self, base, width=10):
        self.base = base.frozen_copy() if base.training else base
        self.width = width
        self.featurizer = base.featurizer
        self.pool = base.pool
        self.training = False
        self._all_green = self.featurizer.n_patterns - 1
        self._base_value = {}   # possible words -> exact expected guesses for the planner to solve
        self._choice = {}       # possible words -> the look-ahead's guess

    # --- the planner's own play, evaluated exactly ------------------------------

    def _base_move(self, possible):
        cost = self.base.expected_cost(possible)
        best = np.flatnonzero(cost <= cost.min() + TIE_TOLERANCE)
        return best[np.argmin(self.featurizer.tie_rank[best])]

    def base_value(self, possible):
        """Expected number of guesses the planner needs from here (each possible answer equally likely)."""
        key = possible.tobytes()
        value = self._base_value.get(key)
        if value is None:
            value = 1.0 if len(possible) == 1 else 1.0 + self.after(self._base_move(possible), possible)
            self._base_value[key] = value
        return value

    def after(self, guess, possible):
        """Expected further guesses after playing `guess`, if the planner plays on (a win costs 0)."""
        codes = self.featurizer.by_answer[possible, guess]
        order = np.argsort(codes, kind="stable")
        codes, members = codes[order], possible[order]
        edges = np.flatnonzero(np.diff(codes)) + 1
        total = 0.0
        for code, group in zip(codes[np.r_[0, edges]], np.split(members, edges)):
            if code != self._all_green:  # the all-green group is the win: nothing more to do
                total += len(group) * self.base_value(np.sort(group))
        return total / len(possible)

    # --- choosing -----------------------------------------------------------------

    def choose(self, history):
        possible = self.featurizer.candidates(history)
        if len(possible) == 1:
            return self.pool[possible[0]]
        key = possible.tobytes()
        if key not in self._choice:
            candidates = cheapest(self.base.expected_cost(possible), self.width, self.featurizer.tie_rank)
            totals = np.array([self.after(g, possible) for g in candidates])
            best = candidates[totals <= totals.min() + TIE_TOLERANCE]
            self._choice[key] = best[np.argmin(self.featurizer.tie_rank[best])]
        return self.pool[self._choice[key]]

    def top_choices(self, history, k=5):
        """The candidates it weighed, with their exact expected total guesses (this one included)."""
        possible = self.featurizer.candidates(history)
        if len(possible) == 1:
            return [(self.pool[possible[0]], 1.0)]
        candidates = cheapest(self.base.expected_cost(possible), self.width, self.featurizer.tie_rank)
        scored = sorted((1 + self.after(g, possible), self.featurizer.tie_rank[g], g) for g in candidates)
        return [(self.pool[g], total) for total, _, g in scored[:k]]

    def expected_total(self, history=()):
        """Exact expected guesses from this position for the look-ahead itself (all answers equally likely)."""
        history = list(history)
        possible = self.featurizer.candidates(history)
        if len(possible) == 1:
            return 1.0
        guess = self.choose(history)
        g = self.featurizer.index[guess]
        codes = self.featurizer.by_answer[possible, g]
        total = 0.0
        for code in np.unique(codes):
            if code == self._all_green:
                continue
            group = possible[codes == code]
            feedback = tuple((int(code) // 3 ** i) % 3 for i in range(self.featurizer.length))
            total += len(group) * self.expected_total(history + [(guess, feedback)])
        return 1.0 + total / len(possible)
