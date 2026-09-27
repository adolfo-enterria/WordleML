"""The learning agent: a linear softmax policy trained by REINFORCE.

How it picks a word:
  1. Work out which words could still be the answer (it knows the rules), and
     compute the facts for each of them (see features.py).
  2. Standardize each fact across the possible words (0 = average, +1 = one
     spread above average), so one weight means the same on turn 1 with 2,315
     options as on turn 4 with 5. Score each word: score = weights . facts.
  3. Training: sample a word with probability softmax(scores), which lets it
     explore. Playing for real: take the highest-scoring word.
  With all weights at zero (untrained) that's a random possible word.

How it learns (REINFORCE):
  Every guess costs -1 and a game only ends when the word is found, so the
  "return" from a turn is minus the number of guesses it still took. After each
  game, every choice gets nudged by how much better than usual that went:
      weights += lr * (return - baseline) * grad log p(choice)
  The baseline is the running average return for that turn number, so a game
  only teaches something when it went better or worse than expected. Each
  game's total nudge is capped (MAX_STEP) so one lucky game can't swing things.
"""
import numpy as np

from agents.base import Agent
from agents.features import FEATURE_NAMES, N_FEATURES, Featurizer

GUESS_COST = -1.0   # reward for every guess made
BASELINE_RATE = 0.05
MAX_STEP = 1.0      # largest total weight change allowed from a single game


def standardize(facts):
    """Rescale each column to mean 0, spread 1. Columns where all words tie become 0."""
    spread = facts.std(axis=0)
    safe = np.where(spread > 0, spread, 1.0)
    return np.where(spread > 0, (facts - facts.mean(axis=0)) / safe, 0.0)


def softmax(scores):
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


class LearningAgent(Agent):
    name = "learning"

    def __init__(self, words, learning_rate=0.05, seed=None, featurizer=None):
        self.words = list(words)
        # The featurizer is read-only after it's built, so copies of an agent can share one.
        self.featurizer = featurizer or Featurizer(self.words)
        self.weights = np.zeros(N_FEATURES)  # all zero: no idea what's good yet
        self.baseline = []  # expected return from each turn on; filled in as it plays
        self.learning_rate = learning_rate
        self.training = False
        self.rng = np.random.default_rng(seed)
        self.new_game()

    def new_game(self):
        self.trajectory = []  # one grad-log-prob vector per guess this game

    def policy(self, history):
        """Return (candidates, facts, probabilities) over the still-possible words."""
        candidates, facts = self.featurizer.features(history)
        facts = standardize(facts)
        return candidates, facts, softmax(facts @ self.weights)

    def choose(self, history):
        candidates, facts, probs = self.policy(history)
        if self.training:
            j = self.rng.choice(len(probs), p=probs)
            # Gradient of log softmax for a linear model: chosen facts - average facts.
            self.trajectory.append(facts[j] - probs @ facts)
        else:
            best = np.flatnonzero(probs == probs.max())
            j = self.rng.choice(best)
        return self.words[candidates[j]]

    def top_choices(self, history, k=5):
        """The k words the agent likes most right now, with their probabilities."""
        candidates, _, probs = self.policy(history)
        top = np.argsort(-probs)[:k]
        return [(self.words[candidates[j]], probs[j]) for j in top]

    def end_game(self, game):
        if not self.training:
            return
        n = game.guesses_used
        returns = GUESS_COST * (n - np.arange(n))  # e.g. solved in 4: [-4, -3, -2, -1]

        step = np.zeros_like(self.weights)
        for turn, (grad, ret) in enumerate(zip(self.trajectory, returns)):
            if turn == len(self.baseline):  # first game this long: nothing to compare to yet
                self.baseline.append(ret)
            advantage = ret - self.baseline[turn]
            step += self.learning_rate * advantage * grad
            self.baseline[turn] += BASELINE_RATE * advantage

        size = np.linalg.norm(step)
        if size > MAX_STEP:
            step *= MAX_STEP / size
        self.weights += step

    def save(self, path):
        np.savez(path, weights=self.weights, baseline=np.array(self.baseline),
                 feature_names=np.array(FEATURE_NAMES))

    @classmethod
    def load(cls, path, words, seed=None):
        data = np.load(path)
        if list(data["feature_names"]) != FEATURE_NAMES:
            raise ValueError(f"{path} was trained with different features; retrain it.")
        agent = cls(words, seed=seed)
        agent.weights = data["weights"]
        agent.baseline = list(data["baseline"])
        return agent
