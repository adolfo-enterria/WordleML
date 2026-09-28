"""The Wordle game itself: secret word, guesses, and colored feedback."""
from collections import Counter

GREY, YELLOW, GREEN = 0, 1, 2
WORD_LENGTH = 5   # the default; any length works (the secret word sets it)
MAX_GUESSES = 6


def score_guess(guess, secret):
    """Return the feedback for a guess as a tuple of GREY/YELLOW/GREEN per letter.

    Duplicate letters follow the real Wordle rules: greens are assigned first,
    then yellows, and a letter is never marked more times than it appears in
    the secret. E.g. secret ABBEY, guess BABES -> yellow yellow green green grey.
    """
    feedback = [GREY] * len(guess)
    unmatched = Counter()

    # Pass 1: exact matches are green; remember the secret's leftover letters.
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            feedback[i] = GREEN
        else:
            unmatched[s] += 1

    # Pass 2: right letter, wrong spot is yellow, but only while leftovers remain.
    for i, g in enumerate(guess):
        if feedback[i] != GREEN and unmatched[g] > 0:
            feedback[i] = YELLOW
            unmatched[g] -= 1

    return tuple(feedback)


class WordleGame:
    """One game of Wordle.

    valid_guesses: optional set of accepted words. Agents pick from the word
    list so they skip validation; human players get checked against it.
    max_guesses: 6 like real Wordle, or None to keep going until it's solved.
    """

    def __init__(self, secret, valid_guesses=None, max_guesses=MAX_GUESSES):
        self.secret = secret.lower()
        self.length = len(self.secret)
        self.valid_guesses = valid_guesses
        self.max_guesses = max_guesses
        self.history = []  # list of (guess, feedback) pairs

    def guess(self, word):
        word = word.lower()
        if self.over:
            raise ValueError("The game is already over.")
        if len(word) != self.length or not word.isalpha():
            raise ValueError(f"Guesses must be {self.length} letters.")
        if self.valid_guesses is not None and word not in self.valid_guesses:
            raise ValueError(f"'{word.upper()}' is not in the word list.")
        feedback = score_guess(word, self.secret)
        self.history.append((word, feedback))
        return feedback

    @property
    def guesses_used(self):
        return len(self.history)

    @property
    def won(self):
        return bool(self.history) and self.history[-1][0] == self.secret

    @property
    def over(self):
        if self.won:
            return True
        return self.max_guesses is not None and self.guesses_used >= self.max_guesses
