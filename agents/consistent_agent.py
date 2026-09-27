"""Baseline: knows the rules and guesses a random word that could still be the
answer, with no strategy for choosing between them. This is exactly how the
learning agent plays before it has learned anything (all weights zero), so it
shows what the learning is worth. It filters with score_guess directly, as an
independent check on the learning agent's own rule logic.
"""
import numpy as np

from agents.base import Agent
from wordle.game import score_guess


class ConsistentAgent(Agent):
    name = "consistent"

    def __init__(self, words, seed=None):
        self.words = list(words)
        self.rng = np.random.default_rng(seed)
        self.new_game()

    def new_game(self):
        self.candidates = self.words
        self.seen = 0  # how many history entries have been used to filter

    def choose(self, history):
        for guess, feedback in history[self.seen:]:
            self.candidates = [w for w in self.candidates if score_guess(guess, w) == feedback]
        self.seen = len(history)
        return self.candidates[self.rng.integers(len(self.candidates))]
