"""The learning agent: a linear softmax policy trained by REINFORCE.

Two modes, chosen when the agent is created:
  "any"      (default) may guess any of the 12,972 valid words, including probe
             words that can't be the answer. Sees the split facts in
             split_features.py (chance to win now, words left afterwards, ...).
  "possible" only guesses words that could still be the answer. Sees the
             word facts in features.py (standardized per turn).
Either way it knows the rules: it tracks which answers are still possible,
and when only one is left it simply says it.

How it picks a word:
  score = weights . facts for every option, then
  training: sample with probability softmax(scores), which lets it explore;
  playing for real: take the highest score.
  With all weights at zero (a blank AI) every option is equally likely.

How it learns (REINFORCE):
  Every guess costs -1 and a game only ends when the word is found, so the
  "return" from a turn is minus the number of guesses it still took. After each
  game, every choice gets nudged by how much better than usual that went:
      weights += lr * (return - baseline) * grad log p(choice)
  The baseline is the running average return for that turn number. Each
  game's total nudge is capped (MAX_STEP) so one lucky game can't swing things.
"""
import numpy as np

from agents.base import Agent
from agents.features import TIE_TOLERANCE, Featurizer
from agents.split_features import SplitFeaturizer
from wordle.words import DEFAULT_WORD_SET, load_word_set

GUESS_COST = -1.0   # reward for every guess made
BASELINE_RATE = 0.05
MAX_STEP = 1.0      # largest total weight change allowed from a single game
MODES = ("any", "possible")
DEFAULT_LEARNING_RATE = {"any": 0.05, "possible": 0.05}


def softmax(scores):
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def make_featurizer(mode, words=None, word_set=DEFAULT_WORD_SET):
    if mode == "any":
        return SplitFeaturizer(word_set)
    if mode == "possible":
        return Featurizer(words if words is not None else load_word_set(word_set).answers)
    raise ValueError(f"mode must be one of {MODES}")


class LearningAgent(Agent):
    name = "learning"

    def __init__(self, words, learning_rate=None, seed=None, featurizer=None, mode="any",
                 word_set=DEFAULT_WORD_SET):
        self.mode = mode
        self.word_set = word_set
        # The featurizer is read-only after it's built, so copies of an agent can share one.
        self.featurizer = featurizer or make_featurizer(mode, words, word_set)
        self.pool = self.featurizer.pool  # the words it may guess
        self.feature_names = list(self.featurizer.names)
        self.weights = np.zeros(len(self.feature_names))  # all zero: no idea what's good yet
        self.baseline = []  # expected return from each turn on; filled in as it plays
        self.learning_rate = learning_rate or DEFAULT_LEARNING_RATE[mode]
        self.training = False
        self.rng = np.random.default_rng(seed)
        self.new_game()

    def new_game(self):
        self.trajectory = []  # (turn, grad log p) for every real decision this game

    def policy(self, history):
        """Return (options, facts, probabilities, possible answers)."""
        options, facts, possible = self.featurizer.state(history)
        return options, facts, softmax(facts @ self.weights), possible

    def choose(self, history):
        options, facts, probs, possible = self.policy(history)
        if len(possible) == 1:  # the rules leave one word: say it (no decision to learn from)
            return self.pool[possible[0]]
        if self.training:
            j = self.rng.choice(len(probs), p=probs)
            # Gradient of log softmax for a linear model: chosen facts - average facts.
            self.trajectory.append((len(history), facts[j] - probs @ facts))
        else:  # being tested: the top choice; exact ties go by the fixed tie priority
            best = np.flatnonzero(probs >= probs.max() * (1 - TIE_TOLERANCE))
            j = best[np.argmin(self.featurizer.tie_rank[options[best]])]
        return self.pool[options[j]]

    def top_choices(self, history, k=5):
        """The k words the agent likes most right now, with their probabilities."""
        options, _, probs, possible = self.policy(history)
        if len(possible) == 1:
            return [(self.pool[possible[0]], 1.0)]
        top = np.argsort(-probs)[:k]
        return [(self.pool[options[j]], probs[j]) for j in top]

    def end_game(self, game):
        if not self.training:
            return
        n = game.guesses_used
        returns = GUESS_COST * (n - np.arange(n))  # e.g. solved in 4: [-4, -3, -2, -1]

        step = np.zeros_like(self.weights)
        for turn, grad in self.trajectory:
            while len(self.baseline) <= turn:  # first game this long: nothing to compare to yet
                self.baseline.append(returns[len(self.baseline)])
            advantage = returns[turn] - self.baseline[turn]
            step += self.learning_rate * advantage * grad
            self.baseline[turn] += BASELINE_RATE * advantage

        size = np.linalg.norm(step)
        if size > MAX_STEP:
            step *= MAX_STEP / size
        self.weights += step

    def frozen_copy(self, seed=0):
        """Same weights, doesn't learn, fixed tie-breaks: for skill checks and lookups."""
        copy = type(self)(None, seed=seed, featurizer=self.featurizer, mode=self.mode, word_set=self.word_set)
        copy.weights = self.weights.copy()
        return copy

    def knowledge(self):
        return {"kind": "weights", "items": [[n, float(w)] for n, w in zip(self.feature_names, self.weights)]}

    def save(self, path):
        np.savez(path, weights=self.weights, baseline=np.array(self.baseline),
                 feature_names=np.array(self.feature_names), mode=self.mode, kind="policy",
                 word_set=self.word_set)

    @classmethod
    def load(cls, path, words=None, seed=None, featurizer=None):
        data = np.load(path)
        mode = str(data["mode"]) if "mode" in data else "possible"
        word_set = str(data["word_set"]) if "word_set" in data else DEFAULT_WORD_SET
        agent = cls(words, seed=seed, mode=mode, word_set=word_set, featurizer=featurizer)
        if list(data["feature_names"]) != agent.feature_names:
            raise ValueError(f"{path} was trained with different features; retrain it.")
        agent.weights = data["weights"]
        agent.baseline = list(data["baseline"])
        return agent
