"""Search: the best play it can find within a thinking budget.

T(S) is how many guesses it takes, in total, to solve every answer in S (the
answers still possible), counting the guess about to be played:

    T({x}) = 1                                        just say it
    T(S)   = |S| + min over guesses g of  sum of T(G) over g's color groups G,
                                          leaving out the all-green group (that's the win)

Average guesses = T(all answers) / number of answers. No sampling: the same
position always gets the same answer, and playing the plan out game by game
gives exactly T.

The thinking budget is which guesses it considers at each position: the
learned planner's `width` favourite guesses (by its own expected cost). So the
planner decides where to look, and the search works out exactly how good each
of those really is, all the way to the end of every game. Within those
candidates the result is the true best. The first guess is special only in
the order openers are tried: the planner's top `root_width` openers are ranked
by what the planner itself would score after each one (the look-ahead's exact
walk), so a strong opener is found first and the rest are cut off sooner.

Three things keep it fast without making it approximate:
  * Lower bound: at best one answer is guessed right away and the rest split as
    evenly as the colors allow (3^L - 1 - L possible groups besides the win), and
    so on down: `best_case_totals`. For small S that's 2|S| - 1, and a possible
    answer that splits S into single words reaches it, so the search stops there.
    For short words (27 color groups at 3 letters) the bound is much higher.
  * Branch and bound: groups are solved largest first, and a guess is dropped
    as soon as its running total reaches the best found so far. The remaining
    budget is passed down, so a sub-search that can't beat it stops early and
    reports a lower bound instead.
  * Memory: every solved position is remembered (and proven lower bounds for
    positions that couldn't beat a budget), shared across openers.

The published optimum for the official lists (every valid guess allowed) is a
proven minimum, so no search can go below it; reaching it is perfect play.
"""
import json
import time

import numpy as np

from agents.base import Agent, play_game
from agents.features import cheapest
from agents.lookahead import LookaheadAgent
from wordle.game import score_guess

WIDTH = 20        # guesses considered at each position
ROOT_WIDTH = 10   # openers ranked by playing the planner out after each, to try the best first


def best_case_totals(n_max, groups):
    """f[n]: a lower bound on the total guesses for n possible answers, if every guess split the
    words left as evenly as `groups` color groups (besides the win) allow.

    One guess leaves the other n - 1 answers in at most `groups` groups, so
    f[n] = n + the best even split of n - 1 into min(groups, n - 1) parts. An even split is the
    cheapest one when f is convex, which is checked (otherwise the plain 2n - 1 bound is used)."""
    f = np.zeros(n_max + 1, dtype=np.int64)
    for n in range(1, n_max + 1):
        parts = min(groups, n - 1)
        if parts == 0:
            f[n] = n
            continue
        q, r = divmod(n - 1, parts)
        f[n] = n + r * f[q + 1] + (parts - r) * f[q]
    if np.any(np.diff(f, 2) < 0):
        f = np.maximum(2 * np.arange(n_max + 1) - 1, 0)
    return f


class SearchAgent(Agent):
    name = "search"

    def __init__(self, base, width=WIDTH, root_width=ROOT_WIDTH, progress=None):
        self.base = base.frozen_copy() if base.training else base
        self.featurizer = base.featurizer
        self.pool = base.pool
        self.width, self.root_width = width, root_width
        self.training = False
        self.progress = progress          # called as progress(openers done, openers, best total so far)
        self._all_green = self.featurizer.n_patterns - 1
        # color groups a guess can make besides the win: all patterns but all-green and the L impossible
        # "one yellow, the rest green" ones
        self._bound = best_case_totals(self.featurizer.n_answers,
                                       self.featurizer.n_patterns - 1 - self.featurizer.length)
        self._exact = {}                  # possible answers -> (best total, its guess)
        self._lower = {}                  # possible answers -> proven lower bound on the total
        self._cands = {}                  # possible answers -> the guesses worth considering there
        self.openers = []                 # [word, total if the planner played on after it, search total or None if cut off]
        self.nodes = 0                    # positions searched (a measure of thinking done)
        self.seconds = 0.0

    # --- the search -----------------------------------------------------------------

    def solve(self, possible, limit=np.inf):
        """Best total within the candidates if it is below `limit`; otherwise a lower bound >= limit."""
        n = len(possible)
        if n <= 2:
            return 2 * n - 1              # 1 word: say it; 2 words: guess one, then the other if needed
        key = possible.tobytes()
        hit = self._exact.get(key)
        if hit is not None:
            return hit[0]
        floor = max(self._lower.get(key, 0), int(self._bound[n]))
        if floor >= limit:
            return floor
        started = time.perf_counter() if n == self.featurizer.n_answers else None
        self.nodes += 1
        perfect = self._perfect_member(possible)
        if perfect is not None:
            self._exact[key] = (2 * n - 1, perfect)
            return 2 * n - 1
        floor = max(floor, 2 * n)         # no possible answer splits them all apart: 2n is the best left
        if floor >= limit:
            self._lower[key] = floor
            return floor
        root = n == self.featurizer.n_answers
        candidates = self._root_candidates(possible) if root else self._candidates(possible)
        best, best_guess = limit, None
        for done, g in enumerate(candidates, 1):
            groups = self._groups(possible, g)
            total = n + int(sum(self._bound[len(group)] for group in groups))  # every group at its lower bound
            if total < best:
                for group in groups:                                 # largest first
                    low = int(self._bound[len(group)])
                    total += self.solve(group, best - (total - low)) - low
                    if total >= best:
                        break
            if root:
                self.openers[done - 1][2] = int(total) if total < best else None
            if total < best:
                best, best_guess = total, g
            if root and self.progress:
                self.progress(done, len(candidates), int(best) if best_guess is not None else None)
            if best == floor:
                break
        if started is not None:
            self.seconds += time.perf_counter() - started
        if best_guess is None:            # nothing beat the limit: that's now a proven lower bound
            self._lower[key] = max(floor, limit)
            return self._lower[key]
        self._exact[key] = (int(best), int(best_guess))
        return int(best)

    def _groups(self, possible, guess):
        """The color groups `guess` splits the possible answers into (not the win), largest first."""
        codes = self.featurizer.by_answer[possible, guess]
        order = np.argsort(codes, kind="stable")        # stable: each group stays sorted
        codes, members = codes[order], possible[order]
        edges = np.flatnonzero(np.diff(codes)) + 1
        groups = [group for code, group in zip(codes[np.r_[0, edges]], np.split(members, edges))
                  if code != self._all_green]
        groups.sort(key=len, reverse=True)
        return groups

    def _perfect_member(self, possible):
        """A possible answer whose colors tell every other possible answer apart (the best any guess can do)."""
        n = len(possible)
        if n > self.featurizer.n_patterns:
            return None
        codes = np.sort(self.featurizer.by_answer[np.ix_(possible, possible)], axis=0)  # column j: guess possible[j]
        distinct = 1 + (np.diff(codes.astype(np.int32), axis=0) != 0).sum(axis=0)
        perfect = possible[distinct == n]
        return int(perfect[np.argmin(self.featurizer.tie_rank[perfect])]) if len(perfect) else None

    def _top(self, cost, k):
        return cheapest(cost, k, self.featurizer.tie_rank)

    def _candidates(self, possible):
        """The planner's `width` favourite guesses here (remembered: a position that couldn't beat one
        budget may be searched again with a bigger one)."""
        key = possible.tobytes()
        if key not in self._cands:
            # the planner's uncached costs: keeping every guess's cost for every position wouldn't fit in memory
            self._cands[key] = self._top(self.base._expected_cost(possible), self.width)
        return self._cands[key]

    def _root_candidates(self, possible):
        """The planner's `width` favourite openers; its top `root_width` go first, best first by what
        the planner would score after each (worked out exactly), the rest follow in its own order."""
        n = len(possible)
        pool = self._top(self.base.expected_cost(possible), max(self.width, self.root_width))
        rollout = LookaheadAgent(self.base, width=1)
        head = pool[:self.root_width]
        totals = np.array([round(n * (1 + rollout.after(g, possible))) for g in head])
        order = np.r_[head[np.lexsort((self.featurizer.tie_rank[head], totals))], pool[self.root_width:]]
        by_word = dict(zip(head.tolist(), totals.tolist()))
        self.openers = [[self.pool[g], int(by_word[g]) if g in by_word else None, None] for g in order]
        return order

    # --- playing ---------------------------------------------------------------------

    def choose(self, history):
        possible = self.featurizer.candidates(history)
        if len(possible) <= 2:
            return self.pool[possible[np.argmin(self.featurizer.tie_rank[possible])]]
        self.solve(possible)
        return self.pool[self._exact[possible.tobytes()][1]]

    def total(self):
        """Total guesses over all answers (each played once) with this plan."""
        return self.solve(np.arange(self.featurizer.n_answers))

    def top_choices(self, history, k=5):
        possible = self.featurizer.candidates(history)
        n = len(possible)
        if n == 1:
            return [(self.pool[possible[0]], 1.0)]
        return [(self.choose(history), self.solve(possible) / n)]

    # --- the plan as a table ------------------------------------------------------------

    def strategy(self):
        """Every answer's guesses under this plan: the whole decision tree."""
        answers = self.pool[:self.featurizer.n_answers]
        return {secret: [g for g, _ in play_game(self, secret).history] for secret in answers}

    def save(self, path, **extra):
        paths = self.strategy()
        total = sum(len(p) for p in paths.values())
        data = {"word_set": self.featurizer.word_set.name, "width": self.width, "root_width": self.root_width,
                "total": total, "average": total / len(paths),
                "seconds": round(self.seconds, 1), "nodes": self.nodes, "openers": self.openers,
                **extra, "paths": paths}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data


class StrategyAgent(Agent):
    """Plays a saved plan (see SearchAgent.save) instantly: position -> guess."""
    name = "search"

    def __init__(self, paths, info=None):
        self.info = info or {}
        self.table = {}     # position -> guess
        self.left = {}      # position -> [guesses still needed, summed over the answers that get there; answers]
        for secret, guesses in paths.items():
            history = []
            for turn, guess in enumerate(guesses):
                key = tuple(history)
                if self.table.setdefault(key, guess) != guess:
                    raise ValueError(f"inconsistent plan at {key}")
                tally = self.left.setdefault(key, [0, 0])
                tally[0] += len(guesses) - turn
                tally[1] += 1
                history.append((guess, score_guess(guess, secret)))
        self.training = False

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data.pop("paths"), data)

    def choose(self, history):
        key = tuple((guess, tuple(feedback)) for guess, feedback in history)
        if key not in self.table:
            raise KeyError("this position isn't in the saved plan (was it made for another word list?)")
        return self.table[key]

    def top_choices(self, history, k=5):
        """Its guess here, with the expected total guesses from here (this one included)."""
        key = tuple((guess, tuple(feedback)) for guess, feedback in history)
        total, answers = self.left[key]
        return [(self.table[key], total / answers)]
