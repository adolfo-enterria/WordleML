"""Facts for an AI that may guess ANY valid word, including "probe" words that can't win.

The AI knows the rules: after each guess it keeps only the answers that match
the colors. For each valid word it can then work out how that word would split
the answers that are still possible (which colors each answer would produce),
the way you might think "if I play FLAMB, the F, L and M would each tell me
something". From that it gets a few plain facts:

  win_chance   chance this guess is the answer (0 for a probe word)
  avg_left     how many words would still be possible afterwards, on average
  worst_left   ...in the worst case
  patterns     how many different color patterns it could produce

avg_left, worst_left and patterns are in "bits" (log2), so halving the words
left is always one step, whether it's 2,000 -> 1,000 or 4 -> 2. The facts
never say what's good; the AI learns how much each one matters, including when
testing letters beats going for the win.

Works for any word set (any word length): see wordle/words.py.
"""
import threading
from collections import OrderedDict

import numpy as np

from agents.features import tie_ranks
from wordle.patterns import encode_pattern, n_patterns, pattern_table
from wordle.words import DEFAULT_WORD_SET, load_word_set

SPLIT_FEATURE_NAMES = ["win_chance", "avg_left", "worst_left", "patterns"]
CACHE_MIN_WORDS = 20   # remember the facts for big candidate sets (they recur after a fixed opener)
CACHE_SIZE = 256
BLOCK_CELLS = 4_000_000  # how much work to do at once; bounds memory for long words


class SplitFeaturizer:
    names = SPLIT_FEATURE_NAMES
    standardized = False

    def __init__(self, word_set=DEFAULT_WORD_SET):
        self.word_set = load_word_set(word_set)
        self.length = self.word_set.length
        self.n_patterns = n_patterns(self.length)
        self.pool, answers, table = pattern_table(word_set)
        self.by_answer = np.ascontiguousarray(table.T)  # answers x guesses: fast row lookups
        self.index = {w: i for i, w in enumerate(self.pool)}
        self.n_answers = len(answers)
        self.all_options = np.arange(len(self.pool))
        self.tie_rank = tie_ranks(len(self.pool))
        self._cache, self._cache_lock = OrderedDict(), threading.Lock()
        self._first_groups = None
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

    # --- the core computation ------------------------------------------------

    def _blocks(self, possible):
        """For blocks of guesses: (start, end, counts, cells).

        counts[j * P + p] is how many possible answers would give pattern p for
        guess start+j; cells[i, j] is where answer i lands, so counts[cells] is
        the size of each answer's group. Working in blocks keeps memory bounded
        even for 8-letter words (3**8 = 6,561 patterns)."""
        n, P = len(possible), self.n_patterns
        rows = self.by_answer[possible]                                  # n x guesses
        width = max(16, min(BLOCK_CELLS // max(n, 1), BLOCK_CELLS // P))
        offsets = (np.arange(width, dtype=np.int32) * P)[None, :]
        for start in range(0, len(self.pool), width):
            block = rows[:, start:start + width].astype(np.int32)
            cells = block + offsets[:, :block.shape[1]]
            counts = np.bincount(cells.ravel(), minlength=block.shape[1] * P)
            yield start, start + block.shape[1], counts, cells

    def group_size_blocks(self, possible):
        """For blocks of guesses: (start, end, sizes) with sizes[i, j] = size of the
        color group answer possible[i] lands in if guess start+j is played."""
        n = len(possible)
        if n * n <= 2 * self.n_patterns:
            # Few words left: comparing them pairwise (n^2 per guess) is cheaper than
            # counting over all 3^L patterns, which costs the same for 2 words as for 2,000.
            rows = self.by_answer[possible]
            width = max(16, BLOCK_CELLS // max(n * n, 1))
            for start in range(0, len(self.pool), width):
                block = rows[:, start:start + width]
                yield start, start + block.shape[1], (block[:, None, :] == block[None, :, :]).sum(axis=1,
                                                                                                  dtype=np.int32)
            return
        for start, end, counts, cells in self._blocks(possible):
            yield start, end, counts.astype(np.int32)[cells]

    def first_turn_groups(self):
        """Every guess's group sizes over ALL answers, stored compactly (it never changes):
        (sizes, starts) where guess g's groups are sizes[starts[g]:starts[g+1]]."""
        if self._first_groups is None:
            sizes, lengths = [], []
            for start, end, counts, _ in self._blocks(np.arange(self.n_answers)):
                per_guess = counts.reshape(end - start, self.n_patterns)
                nonzero = per_guess > 0
                sizes.append(per_guess[nonzero].astype(np.int32))
                lengths.append(nonzero.sum(axis=1))
            lengths = np.concatenate(lengths)
            self._first_groups = (np.concatenate(sizes), np.concatenate([[0], np.cumsum(lengths)[:-1]]))
        return self._first_groups

    def split_stats(self, possible):
        """How every valid guess would split the possible answers into color groups:
          sum_sq     sum of group size^2 (so sum_sq / n = average words left)
          worst      largest group
          patterns   number of groups (summing 1/size counts each group once)
          bits_left  expected log2(words left); log2(n) - bits_left is the
                     information (entropy) the guess gives, in bits
        """
        stats = {key: np.empty(len(self.pool)) for key in ("sum_sq", "worst", "patterns", "bits_left")}
        for start, end, sizes in self.group_size_blocks(possible):
            stats["sum_sq"][start:end] = sizes.sum(axis=0)
            stats["worst"][start:end] = sizes.max(axis=0)
            stats["patterns"][start:end] = (1.0 / sizes).sum(axis=0)
            stats["bits_left"][start:end] = np.log2(sizes).mean(axis=0)
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
