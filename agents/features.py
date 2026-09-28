"""What the agent sees each turn.

The agent knows the rules. From the colored feedback it works out which words
could still be the answer (green = right letter, right spot; yellow = in the
word, but not in that spot; grey = not in the word) and only guesses those.

Which possible word to guess is what it has to learn. For each one it computes
a few plain facts, e.g. "has 5 different letters" or "its letters are common
among the words still possible". The facts never say whether that's good or
bad: the agent starts with all weights at zero and learns which facts matter.
"""
import numpy as np

from wordle.game import GREEN, GREY
from wordle.words import encode

FEATURE_NAMES = [
    "distinct_letters",    # how many different letters the word has
    "possible_letters",    # how common its letters are among the words still possible
    "possible_positions",  # ...and how common they are in those exact spots
]
N_FEATURES = len(FEATURE_NAMES)

TIE_TOLERANCE = 1e-9  # scores closer than this are a real tie (not just "almost the same")


def tie_ranks(n, seed=0):
    """A fixed, random-looking priority for every word, used to break exact ties
    when an agent is being tested rather than practicing. Same agent + same
    word = same game, everywhere (dashboard, evaluate.py, benchmark, lookups)."""
    return np.random.default_rng(seed).permutation(n)


def cheapest(cost, k, tie_rank):
    """The k options with the lowest cost, cheapest first, skipping useless ones (cost inf).

    Exact ties go by the fixed tie priority, and the first one is always the
    option a tested planner would play itself (lowest cost, ties within
    TIE_TOLERANCE by priority). Look-ahead and search both pick their
    candidates here, so they always weigh the planner's own choice."""
    finite = np.flatnonzero(np.isfinite(cost))
    k = min(k, len(finite))
    if k == 0:
        return np.array([], dtype=np.int64)
    threshold = np.partition(cost[finite], k - 1)[k - 1] + TIE_TOLERANCE
    near = finite[cost[finite] <= threshold]
    best = near[cost[near] <= cost[near].min() + TIE_TOLERANCE]
    first = best[np.argmin(tie_rank[best])]
    rest = near[np.lexsort((tie_rank[near], cost[near]))]
    return np.r_[first, rest[rest != first]][:k].astype(np.int64)


def standardize(facts):
    """Rescale each column to mean 0, spread 1. Columns where all words tie become 0.

    This way one weight means the same on turn 1 with 2,315 options as on
    turn 4 with 5."""
    spread = facts.std(axis=0)
    safe = np.where(spread > 0, spread, 1.0)
    return np.where(spread > 0, (facts - facts.mean(axis=0)) / safe, 0.0)


class Knowledge:
    """Everything the feedback so far tells us, in array form."""

    def __init__(self, history, length=5):
        self.green = np.full(length, -1)              # letter index known at each spot, or -1
        self.wrong_spot = np.zeros((length, 26), bool)  # letter known NOT to be at this spot
        self.min_count = np.zeros(26, int)            # the answer has at least this many
        self.max_count = np.full(26, length)          # ...and at most this many

        for guess, feedback in history:
            letters = [ord(c) - ord("a") for c in guess]
            marked = np.zeros(26, int)  # green+yellow tiles per letter in this guess
            for pos, (letter, state) in enumerate(zip(letters, feedback)):
                if state == GREEN:
                    self.green[pos] = letter
                else:
                    self.wrong_spot[pos, letter] = True
                if state != GREY:
                    marked[letter] += 1
            self.min_count = np.maximum(self.min_count, marked)
            for letter, state in zip(letters, feedback):
                if state == GREY:  # a grey tile caps the count at what was marked
                    self.max_count[letter] = min(self.max_count[letter], marked[letter])


class Featurizer:
    """Facts for an AI that only guesses words that could still be the answer."""
    names = FEATURE_NAMES
    standardized = True

    def __init__(self, words):
        self.words = list(words)
        self.pool = self.words  # the words it may guess
        self.letters, self.counts = encode(self.words)
        self.length = self.letters.shape[1]
        self.has_letter = self.counts > 0
        self.distinct = self.has_letter.sum(axis=1)
        self.tie_rank = tie_ranks(len(self.words))
        self._first_turn = self._compute([])  # turn 1 never changes, so cache it

    def possible(self, history):
        """Boolean mask over the word list: which words could still be the answer.

        This is the rules of Wordle: a word is possible if it keeps every green,
        includes every yellow (somewhere else) and avoids every grey letter.
        """
        return self._possible(Knowledge(history, self.length))

    def _possible(self, k):
        # Only look at the letters the feedback actually says something about (fast).
        known = np.flatnonzero(k.green >= 0)
        present = np.flatnonzero(k.min_count > 0)
        capped = np.flatnonzero(k.max_count < self.length)

        greens_moved = (self.letters[:, known] != k.green[known]).sum(axis=1)
        wrong_spot = k.wrong_spot.copy()
        wrong_spot[:, np.flatnonzero(k.min_count == 0)] = False  # absent letters count as grey
        yellow_same_spot = wrong_spot[np.arange(self.length), self.letters].sum(axis=1)
        left_out = np.maximum(k.min_count[present] - self.counts[:, present], 0).sum(axis=1)
        grey = np.maximum(self.counts[:, capped] - k.max_count[capped], 0).sum(axis=1)
        return (greens_moved + yellow_same_spot + left_out + grey) == 0

    def candidates(self, history):
        """Indices of the words that still match every color seen so far."""
        return np.flatnonzero(self.possible(history))

    def state(self, history):
        """(options, facts, possible): here the options are exactly the possible words."""
        candidates, facts = self.features(history)
        return candidates, standardize(facts), candidates

    def features(self, history):
        """Return (candidates, facts): indices of the still-possible words, and
        a len(candidates) x N_FEATURES matrix with one row of facts per word."""
        if not history:
            return self._first_turn
        return self._compute(history)

    def _commonness(self, subset):
        """How common each word's letters are among `subset`, overall and by spot."""
        letter_share = self.has_letter[subset].mean(axis=0)
        position_share = np.stack([np.bincount(self.letters[subset, p], minlength=26) / len(subset)
                                   for p in range(self.length)])
        letters_score = self.has_letter @ letter_share
        positions_score = position_share[np.arange(self.length), self.letters].sum(axis=1)
        return letters_score, positions_score

    def _compute(self, history):
        k = Knowledge(history, self.length)
        candidates = np.flatnonzero(self._possible(k))
        possible_letters, possible_positions = self._commonness(candidates)
        facts = np.column_stack([self.distinct, possible_letters, possible_positions]).astype(float)
        return candidates, facts[candidates]
