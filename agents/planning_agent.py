"""The planning agent: learns how costly each situation is, then plans one guess ahead.

What it knows (the rules): which answers are still possible, and for any valid
word, how that guess would split them into color groups. Works for any word set
(any word length, see wordle/words.py).

What it learns, by playing: V(m), how many more guesses it usually still needs
when m words are possible. It fits a smooth curve to its own games,
    V(m) = a + b*log2(m) + c*log2(m)^2          (3 learned numbers)
by least squares, gradually forgetting old games (FORGET) so the curve follows
how well it plays now rather than how badly it played at the start.

How it chooses: for every valid word it adds up, over the possible answers,
the V of the group that answer would land in; if the guess IS the answer, that
case costs nothing more. Divided by the number of possible answers, that's the
expected number of further guesses. It plays the cheapest word.
Nobody tells it when to "probe" (play a word that can't win to test letters)
or when to go for the win; both fall out of this sum once V is learned.

Blank AI: V = 0 everywhere, so every guess looks equally good: random valid words.
"""
import numpy as np

from agents.base import Agent
from agents.features import TIE_TOLERANCE
from agents.split_features import SplitFeaturizer
from wordle.words import DEFAULT_WORD_SET

FORGET = 0.98        # each game, older experience counts 2% less (memory of ~50 games)
CACHE_ABOVE = 20     # a frozen agent remembers costs for big candidate sets (they recur)
CURVE_POINTS = [1, 2, 3, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000]


def curve_basis(m):
    lg = np.log2(np.maximum(m, 1))
    return np.stack([np.ones_like(lg), lg, lg ** 2], axis=-1)


class PlanningAgent(Agent):
    name = "planner"
    mode = "any"

    def __init__(self, words=None, seed=None, featurizer=None, forget=FORGET, curve="curved",
                 word_set=DEFAULT_WORD_SET):
        self.featurizer = featurizer or SplitFeaturizer(word_set)
        self.word_set = self.featurizer.word_set.name
        self.pool = self.featurizer.pool
        self.n_answers = self.featurizer.n_answers
        self.forget = forget
        self.curve = curve  # "curved" (3 numbers) or "line" (2 numbers: c fixed at 0)
        self.rng = np.random.default_rng(seed)
        self.training = False
        self.coef = np.zeros(3)             # blank: every situation costs 0
        self.xtx, self.xty = np.zeros((3, 3)), np.zeros(3)
        self._cost_cache = {}               # only used while not learning (beliefs fixed)
        self._basis = curve_basis(np.arange(self.n_answers + 1))

    # --- what it believes ---------------------------------------------------

    def value_table(self):
        """V(m) for m = 0..number of answers (V(0) = 0: a group with no words costs nothing).

        The fitted curve is made never to go down as m grows: more words left
        can't be cheaper. (With little experience, or far beyond the sizes it
        has seen, a curved fit can bend the wrong way; the agent would then
        avoid narrowing things down and games would drag on.)"""
        v = np.maximum.accumulate(self._basis @ self.coef)
        v[0] = 0.0
        return v

    def value_curve(self):
        v = self.value_table()
        return [(m, float(v[m])) for m in CURVE_POINTS if m <= self.n_answers]

    def knowledge(self):
        return {"kind": "value", "items": [[m, v] for m, v in self.value_curve() if m <= 1000]}

    # --- choosing -----------------------------------------------------------

    def expected_cost(self, possible):
        """For every valid word: expected guesses still needed AFTER playing it."""
        n = len(possible)
        key = possible.tobytes() if not self.training and n > CACHE_ABOVE else None
        if key is not None and key in self._cost_cache:
            return self._cost_cache[key]
        cost = self._expected_cost(possible)
        if key is not None:
            self._cost_cache[key] = cost
        return cost

    def _expected_cost(self, possible):
        n = len(possible)
        v = self.value_table()
        if n == self.n_answers:  # turn 1: the groups never change, only V does
            sizes, starts = self.featurizer.first_turn_groups()
            cost = np.add.reduceat(sizes * v[sizes], starts)   # sum over groups of size * V(size)
            useless = np.maximum.reduceat(sizes, starts) == n
        else:
            cost = np.empty(len(self.pool))
            useless = np.empty(len(self.pool), bool)
            for start, end, sizes in self.featurizer.group_size_blocks(possible):
                cost[start:end] = v[sizes].sum(axis=0)         # same sum, answer by answer
                useless[start:end] = sizes.min(axis=0) == n
        cost = cost.astype(float)
        cost[possible] -= v[1]  # if the guess is the answer, that case needs 0 more guesses
        # A word giving the same colors for every possible answer can't tell it anything:
        # never an option (unless it could be the answer). Every guess then makes progress.
        useless[possible] = False
        cost[useless] = np.inf
        return cost / n

    def choose(self, history):
        possible = self.featurizer.candidates(history)
        if len(possible) == 1:  # the rules leave one word: say it
            return self.pool[possible[0]]
        cost = self.expected_cost(possible)
        best = np.flatnonzero(cost <= cost.min() + TIE_TOLERANCE)
        if self.training:  # practicing: exact ties are broken at random
            return self.pool[self.rng.choice(best)]
        # being tested: exact ties go by the fixed tie priority, so the same word is always played the same way
        return self.pool[best[np.argmin(self.featurizer.tie_rank[best])]]

    def top_choices(self, history, k=5):
        """The k cheapest words right now, with their expected total guesses (this one included)."""
        possible = self.featurizer.candidates(history)
        if len(possible) == 1:
            return [(self.pool[possible[0]], 1.0)]
        cost = self.expected_cost(possible)
        top = np.argsort(cost, kind="stable")[:k]
        return [(self.pool[j], 1 + float(cost[j])) for j in top]

    # --- learning -----------------------------------------------------------

    def end_game(self, game):
        """After each guess that didn't win: how many words were left, and how many guesses it still took."""
        if not self.training:
            return
        total = game.guesses_used
        self.xtx *= self.forget
        self.xty *= self.forget
        for turn in range(total - 1):
            left = len(self.featurizer.candidates(game.history[:turn + 1]))
            x = curve_basis(np.array(float(left)))
            self.xtx += np.outer(x, x)
            self.xty += x * (total - turn - 1)
        if np.trace(self.xtx) > 0:
            k = 3 if self.curve == "curved" else 2
            coef = np.zeros(3)
            coef[:k] = np.linalg.solve(self.xtx[:k, :k] + 1e-6 * np.eye(k), self.xty[:k])
            self.coef = coef
            self._cost_cache.clear()

    # --- copies and files ---------------------------------------------------

    def state(self):
        return {"coef": self.coef.copy(), "xtx": self.xtx.copy(), "xty": self.xty.copy()}

    def frozen_copy(self, seed=0):
        """Same beliefs, doesn't learn, fixed tie-breaks: for skill checks and lookups."""
        copy = PlanningAgent(seed=seed, featurizer=self.featurizer, forget=self.forget, curve=self.curve)
        copy.coef = self.coef.copy()
        return copy

    def save(self, path):
        np.savez(path, coef=self.coef, xtx=self.xtx, xty=self.xty, mode="any", kind="planner",
                 word_set=self.word_set)

    @classmethod
    def load(cls, path, words=None, seed=None, featurizer=None):
        data = np.load(path)
        word_set = str(data["word_set"]) if "word_set" in data else DEFAULT_WORD_SET
        agent = cls(seed=seed, featurizer=featurizer, word_set=word_set)
        agent.coef, agent.xtx, agent.xty = data["coef"], data["xtx"], data["xty"]
        return agent
