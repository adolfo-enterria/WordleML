"""Reference strategies and wrappers used by the benchmark.

  GreedySplitAgent   hand-made baseline: always plays the guess that splits the
                     possible answers best by a fixed rule (no learning).
                     "entropy": most information (fewest expected bits left);
                     "avg_left": fewest words left on average.
                     Ties go to guesses that could be the answer.
  ForcedOpener       any agent, but with a fixed first guess.
  FactSubsetAgent    a learning agent that only sees some of its facts (for ablations).
"""
import numpy as np

from agents.base import Agent
from agents.learning_agent import LearningAgent, softmax


class GreedySplitAgent(Agent):
    def __init__(self, featurizer, criterion="entropy"):
        self.featurizer = featurizer
        self.pool = featurizer.pool
        self.criterion = criterion
        self.name = f"greedy-{criterion}"
        self._choice = {}  # a fixed rule: the same possible words always get the same guess

    def choose(self, history):
        possible = self.featurizer.candidates(history)
        if len(possible) == 1:
            return self.pool[possible[0]]
        key = possible.tobytes()
        if key not in self._choice:
            self._choice[key] = self._best(possible)
        return self._choice[key]

    def _best(self, possible):
        stats = self.featurizer.split_stats(possible)
        cost = stats["bits_left"] if self.criterion == "entropy" else stats["sum_sq"]
        could_win = np.zeros(len(self.pool), bool)
        could_win[possible] = True  # answers come first in the pool, so answer i is option i
        best = np.flatnonzero(cost <= cost.min() + 1e-9)
        winners = best[could_win[best]]
        return self.pool[(winners if len(winners) else best)[0]]


class ForcedOpener(Agent):
    """Wraps an agent so that its first guess is always `opener`."""

    def __init__(self, agent, opener):
        self.agent, self.opener = agent, opener
        self.name = f"{agent.name}+{opener}"

    def new_game(self):
        self.agent.new_game()

    def choose(self, history):
        return self.opener if not history else self.agent.choose(history)

    def end_game(self, game):
        self.agent.end_game(game)


class FactSubsetAgent(LearningAgent):
    """A learning agent that only sees the facts listed in `keep` (by name)."""

    def __init__(self, words, keep, **kwargs):
        super().__init__(words, **kwargs)
        self.columns = [self.feature_names.index(name) for name in keep]
        self.feature_names = list(keep)
        self.weights = np.zeros(len(keep))

    def policy(self, history):
        options, facts, possible = self.featurizer.state(history)
        facts = facts[:, self.columns]
        return options, facts, softmax(facts @ self.weights), possible
