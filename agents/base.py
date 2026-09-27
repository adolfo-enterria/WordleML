"""The interface every Wordle-playing agent follows, plus a helper to run one game."""
from wordle.game import WordleGame


class Agent:
    name = "base"

    def new_game(self):
        """Called before each game. Reset any per-game memory here."""

    def choose(self, history):
        """Return the next guess given the list of (guess, feedback) so far."""
        raise NotImplementedError

    def end_game(self, game):
        """Called after each game. Learning agents update themselves here."""


def play_game(agent, secret, max_guesses=None):
    """Play one game. By default there's no guess limit: it goes until solved."""
    game = WordleGame(secret, max_guesses=max_guesses)
    agent.new_game()
    while not game.over:
        game.guess(agent.choose(game.history))
    agent.end_game(game)
    return game
