"""How few guesses is possible at all? Proven lower bounds, and proofs of the optimum where they finish.

The search in agents/search.py only weighs the planner's favourite guesses at each
position, so its total is an upper bound on the optimum (the best total any strategy
can reach). This module looks at EVERY valid guess, so its bounds hold for any
strategy whatsoever:

  T(S) = total guesses to solve every answer in S, as in agents/search.py
  f_k(n) = best_case_totals(n, k): a lower bound for any n answers if no guess can make
         more than k color groups besides the win (every split as even as k allows)

k(S), the most groups any valid guess makes on S (besides the win), never grows on a
subset of S: a guess can't tell apart words in a subset that it can't tell apart in S.
So f_k(S) bounds every position below S, and for short words, where few patterns are
possible, it is far stronger than assuming all 3^L - 1 - L patterns can occur.

Bounds for a set of possible answers S, cheapest first:
  even split   f_k(|S|).
  splitter     a possible answer whose colors tell all the others apart makes
               T(S) = 2|S| - 1 exactly; if there is none, T(S) >= 2|S|.
  one level    for every valid guess at once (numpy): |S| + the sum of f_k(S)(|G|) over
               its real color groups G (not the win). The smallest is a lower bound on T(S).
  exact        branch and bound over every valid guess, cheapest one-level bound first.
               Before searching under a guess, each of its groups gets the cheap bounds
               above; only if they don't already rule the guess out does it search deeper.
               Budgets are passed down, and exact values and proven lower bounds are
               remembered (like agents/search.py, but no guess is left out).

At the root (`prove`), every opener keeps its own proven lower bound, and the opener
with the weakest one is always worked on next, with a budget just above the next
weakest. The smallest of those bounds is a lower bound on the optimum at any moment,
so stopping early (a time limit) still leaves a valid bound. Started with the best
known total, it either finds something better or proves that nothing is.

Every number here is a proof about the word list, not an estimate: a lower bound above
a total some strategy actually reaches would be a bug.
"""
import heapq
import time

import numpy as np

from agents.search import best_case_totals

NEVER = np.iinfo(np.int64).max // 4   # a guess that can't split the words and can't win: never worth playing


class OutOfTime(Exception):
    pass


class EvenSplit:
    """f_k for k = 1 .. 3^L - 1 - L color groups besides the win (agents/search.py: best_case_totals), cached."""

    def __init__(self, featurizer):
        self.n_max = featurizer.n_answers
        self.most = featurizer.n_patterns - 1 - featurizer.length
        self._cache = {}

    def __call__(self, k=None):
        k = self.most if k is None else int(min(max(k, 1), self.most))
        if k not in self._cache:   # both are lower bounds; the max guards best_case_totals' convexity fallback
            self._cache[k] = (best_case_totals(self.n_max, k) if k == self.most
                              else np.maximum(best_case_totals(self.n_max, k), self()))
        return self._cache[k]


def one_level(featurizer, possible, even):
    """(bounds, k). bounds: for every valid guess, n + sum of f_k(|G|) over its color groups G (not the win),
    a lower bound on the total if that guess is played now (NEVER for guesses that can't split the words and
    can't be the answer). k: the most groups besides the win any guess makes here."""
    n = len(possible)
    pool = len(featurizer.pool)
    could_win = np.zeros(pool, np.int64)
    could_win[possible] = 1
    if n == featurizer.n_answers:      # turn 1: exact integer sums from the cached group sizes
        sizes, starts = featurizer.first_turn_groups()
        k = int((np.diff(np.r_[starts, len(sizes)]) - could_win).max())
        f = even(min(k, n - 1))
        out = np.add.reduceat(f[sizes], starts).astype(np.int64)
        useless = np.maximum.reduceat(sizes, starts) == n
    elif n * n > 2 * featurizer.n_patterns:
        # Many words, few color patterns (short words): count every guess's groups pattern by pattern.
        width = featurizer.n_patterns
        out, useless, k = np.empty(pool, np.int64), np.empty(pool, bool), 1
        for start, end, counts, _ in featurizer._blocks(possible):
            per = counts.reshape(end - start, width)
            k = max(k, int(((per > 0).sum(axis=1) - could_win[start:end]).max()))
            useless[start:end] = per.max(axis=1) == n
        f = even(min(k, n - 1))
        for start, end, counts, _ in featurizer._blocks(possible):
            out[start:end] = f[counts].reshape(end - start, width).sum(axis=1)
    else:
        # One pass: group counts (for k) and the sums with the loosest f. A second pass with f_k only if
        # no guess separates every word here (k < n - 1); otherwise f_k and f agree on every group size.
        f = even()
        share = f / np.maximum(np.arange(len(f)), 1)     # each answer carries f(size) / size of its group
        out, useless, k = np.empty(pool), np.empty(pool, bool), 1
        for start, end, sizes in featurizer.group_size_blocks(possible):
            groups = np.rint((1.0 / sizes).sum(axis=0)).astype(np.int64) - could_win[start:end]
            k = max(k, int(groups.max()))
            out[start:end] = share[sizes].sum(axis=0)
            useless[start:end] = sizes.min(axis=0) == n
        if k < n - 1:
            f = even(k)
            share = f / np.maximum(np.arange(len(f)), 1)
            for start, end, sizes in featurizer.group_size_blocks(possible):
                out[start:end] = share[sizes].sum(axis=0)
        out = np.rint(out).astype(np.int64)
    out += n
    out[possible] -= 1                 # the guess itself is one of the answers: that group is the win (f(1) = 1)
    useless[possible] = False
    out[useless] = NEVER
    return out, k


class ExactSearch:
    """Branch and bound over every valid guess. solve(S, limit) returns T(S) if it is below `limit`,
    otherwise a proven lower bound >= limit (the same contract as agents/search.py)."""

    def __init__(self, featurizer):
        self.featurizer = featurizer
        self.even = EvenSplit(featurizer)
        self._all_green = featurizer.n_patterns - 1
        self._exact = {}      # possible answers -> (T, best guess)
        self._lower = {}      # possible answers -> proven lower bound
        self._last = (None, None)   # the last one-level array computed, so solve() needn't redo it
        self.nodes = 0
        self.deadline = None

    @staticmethod
    def _key(possible):
        return possible.astype(np.int16).tobytes()

    def _groups(self, possible, guess):
        """`guess`'s color groups of the possible answers (not the win), largest first."""
        codes = self.featurizer.by_answer[possible, guess]
        order = np.argsort(codes, kind="stable")
        codes, members = codes[order], possible[order]
        edges = np.flatnonzero(np.diff(codes)) + 1
        groups = [g for code, g in zip(codes[np.r_[0, edges]], np.split(members, edges)) if code != self._all_green]
        groups.sort(key=len, reverse=True)
        return groups

    def _splits_all(self, possible):
        """Does some possible answer tell all the others apart? (Then T = 2n - 1, the best there is.)"""
        n = len(possible)
        if n > self.featurizer.n_patterns:
            return False
        codes = np.sort(self.featurizer.by_answer[np.ix_(possible, possible)], axis=0)
        return bool(((np.diff(codes.astype(np.int32), axis=0) != 0).sum(axis=0) == n - 1).any())

    def bound(self, possible):
        """The best cheap lower bound on T(possible) (exact when it can be): memo, splitter, one level."""
        n = len(possible)
        if n <= 2:
            return 2 * n - 1
        key = self._key(possible)
        hit = self._exact.get(key)
        if hit is not None:
            return hit[0]
        low = self._lower.get(key)
        if low is None:
            if self._splits_all(possible):
                self._exact[key] = (2 * n - 1, None)
                return 2 * n - 1
            bounds, k = one_level(self.featurizer, possible, self.even)
            self._last = (key, bounds)
            low = max(int(self.even(k)[n]), 2 * n, int(bounds.min()))
            self._lower[key] = low
        return low

    def _try(self, possible, guess, best):
        """Total for playing `guess` here if it beats `best`, else a lower bound >= best."""
        groups = self._groups(possible, guess)
        lows = [self.bound(group) for group in groups]
        total = len(possible) + sum(lows)
        for group, low in zip(groups, lows):           # largest first
            if total >= best:
                break
            if len(group) > 2:
                total += self.solve(group, best - (total - low)) - low
        return total

    def solve(self, possible, limit):
        n = len(possible)
        floor = self.bound(possible)
        key = self._key(possible)
        if n <= 2 or key in self._exact or floor >= limit:
            return floor
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise OutOfTime
        self.nodes += 1
        bounds = self._last[1] if self._last[0] == key else one_level(self.featurizer, possible, self.even)[0]
        best, best_guess = limit, None
        for g in np.argsort(bounds, kind="stable"):      # ties: answers first (they can win)
            if bounds[g] >= best:
                break
            total = self._try(possible, g, best)
            if total < best:
                best, best_guess = total, int(g)
                if best <= floor:
                    break
        if best_guess is None:
            self._lower[key] = max(floor, int(limit))
            return self._lower[key]
        self._exact[key] = (int(best), best_guess)
        return int(best)

    def prove(self, best_known, seconds=None, progress=None, every=30):
        """Can any strategy beat `best_known` (a total some strategy reaches)? See the module docstring.

        Returns lower and upper bounds on the optimum; proven when they meet. `improved` names an
        opener that beat best_known (the upper bound is then its total)."""
        started = time.perf_counter()
        self.deadline = None if seconds is None else started + seconds
        everything = np.arange(self.featurizer.n_answers)
        n = len(everything)
        one, _ = one_level(self.featurizer, everything, self.even)
        best, improved = int(best_known), None
        heap = [(int(one[g]), int(g)) for g in np.flatnonzero(one < best)]
        alive = len(heap)
        heapq.heapify(heap)
        step = max(1, n // 400)
        rounds, shown = 0, started
        try:
            while heap and heap[0][0] < best:
                if self.deadline is not None and time.perf_counter() > self.deadline:
                    break           # (the cheap first pass over the openers never reaches solve's own check)
                low, g = heapq.heappop(heap)
                second = heap[0][0] if heap else best
                budget = min(best, max(second, low) + step)
                total = self._try(everything, g, budget)
                rounds += 1
                if total < budget:              # exact: this opener's best total
                    if total < best:
                        best, improved = total, self.featurizer.pool[g]
                else:
                    heapq.heappush(heap, (max(low, total), g))
                if progress and time.perf_counter() - shown > every:
                    shown = time.perf_counter()
                    progress(min(best, heap[0][0]) if heap else best, best, sum(1 for b, _ in heap if b < best))
        except OutOfTime:
            heapq.heappush(heap, (low, g))
        self.deadline = None
        lower = min(best, heap[0][0]) if heap else best
        return {"lower": int(lower), "upper": int(best), "proven": lower == best, "improved": improved,
                "openers_alive": alive, "openers_left": sum(1 for b, _ in heap if b < best),
                "one_level": int(one.min()), "rounds": rounds, "nodes": self.nodes,
                "seconds": round(time.perf_counter() - started, 1)}
