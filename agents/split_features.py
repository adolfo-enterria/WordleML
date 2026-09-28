"""Facts for an AI that may guess ANY valid word, including "probe" words that can't win.

The AI knows the rules: after each guess it keeps only the answers that match
the colors. For each of the 12,972 valid words it can then work out how that
word would split the answers that are still possible (which colors each
answer would produce), the way you might think "if I play FLAMB, the F, L
and M would each tell me something". From that it gets a few plain facts:

  win_chance   chance this guess is the answer (0 for a probe word)
  avg_left     how many words would still be possible afterwards, on average
  worst_left   ...in the worst case
  patterns     how many different color patterns it could produce

avg_left, worst_left and patterns are in "bits" (log2), so halving the words
left is always one step, whether it's 2,000 -> 1,000 or 4 -> 2. The facts
never say what's good; the AI learns how much each one matters, including when
testing letters beats going for the win.
"""
import threading
from collections import OrderedDict

import numpy as np

from agents.features import tie_ranks
from wordle.patterns import N_PATTERNS, encode_pattern, pattern_table

SPLIT_FEATURE_NAMES = ["win_chance", "avg_left", "worst_left", "patterns"]
CACHE_MIN_WORDS = 20   # remember the facts for big candidate sets (they recur after a fixed opener)
CACHE_SIZE = 256


class SplitFeaturizer:
    names = SPLIT_FEATURE_NAMES
    standardized = False

    def __init__(self):
        self.pool, answers, table = pattern_table()
        self.by_answer = np.ascontiguousarray(table.T)  # answers x guesses: fast row lookups
        self.index = {w: i for i, w in enumerate(self.pool)}
        self.n_answers = len(answers)
        self.all_options = np.arange(len(self.pool))
        self.tie_rank = tie_ranks(len(self.pool))
        self._offsets = (np.arange(len(self.pool), dtype=np.int32) * N_PATTERNS)[None, :]
        self._cache, self._cache_lock = OrderedDict(), threading.Lock()
        self._first_turn = self._facts(np.arange(self.n_answers))  # turn 1 never changes

    def candidates(self, history):
        """Indices of the answers that still match every color seen so far."""
        candidates = np.arange(self.n_answers)
        for guess, feedback in history:
            candidates = candidates[self.by_answer[candidates, self.index[guess]] == encode_pattern(feedback)]
        return candidates

    def state(self, history):
        """(options, facts, possible): the words worth guessing, their facts, and the still-possible answers.

        A word that would give the same colors for every possible answer can't
        tell the AI anything, so unless it could be the answer itself it isn't
        an option. That way every guess makes progress (and no game can loop)."""
        possible = self.candidates(history)
        facts = self._cached_facts(possible)
        options = np.flatnonzero((facts[:, 3] > 1e-9) | (facts[:, 0] > 0))  # log2(patterns) > 0
        return options, facts[options], possible

    def _cached_facts(self, possible):
        if len(possible) == self.n_answers:
            return self._first_turn
        if len(possible) < CACHE_MIN_WORDS:
            return self._facts(possible)
        key = possible.tobytes()
        with self._cache_lock:
            facts = self._cache.get(key)
            if facts is not None:
                self._cache.move_to_end(key)
        if facts is None:
            facts = self._facts(possible)
            with self._cache_lock:
                self._cache[key] = facts
                if len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
        return facts

    def split(self, guess, possible):
        """How one guess splits the possible answers: {pattern: [answer indices]}."""
        groups = {}
        for answer, code in zip(possible, self.by_answer[possible, self.index[guess]]):
            groups.setdefault(int(code), []).append(int(answer))
        return groups

    def split_stats(self, possible):
        """How every valid guess would split the possible answers into color groups.

        For each possible answer we look up the size of the group it would land
        in, for every guess at once. From that, per guess:
          sum_sq     sum of group size^2 (so sum_sq / n = average words left)
          worst      largest group
          patterns   number of groups (summing 1/size counts each group once)
          bits_left  expected log2(words left); log2(n) - bits_left is the
                     information (entropy) the guess gives, in bits
        """
        n = len(possible)
        rows = self.by_answer[possible]                                # n x guesses
        stats = {key: np.empty(len(self.pool)) for key in ("sum_sq", "worst", "patterns", "bits_left")}
        width = max(256, 4_000_000 // n)  # guesses per block, so big candidate sets stay small in memory
        for start in range(0, len(self.pool), width):
            block = rows[:, start:start + width].astype(np.int32)
            cells = block + self._offsets[:, :block.shape[1]]
            counts = np.bincount(cells.ravel(), minlength=block.shape[1] * N_PATTERNS)
            group_size = counts.astype(np.int32)[cells]
            end = start + block.shape[1]
            stats["sum_sq"][start:end] = group_size.sum(axis=0)
            stats["worst"][start:end] = group_size.max(axis=0)
            stats["patterns"][start:end] = (1.0 / group_size).sum(axis=0)
            stats["bits_left"][start:end] = np.log2(group_size).mean(axis=0)
        return stats

    def _facts(self, possible):
        """The 4 facts for every valid guess, given the still-possible answers."""
        n = len(possible)
        stats = self.split_stats(possible)
        facts = np.empty((len(self.pool), len(self.names)))
        facts[:, 0] = 0.0
        facts[possible, 0] = 1.0 / n  # answers come first in the pool, so answer i is option i
        facts[:, 1] = np.log2(stats["sum_sq"] / n)
        facts[:, 2] = np.log2(stats["worst"])
        facts[:, 3] = np.log2(stats["patterns"])
        return facts
