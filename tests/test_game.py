import pytest

from wordle.game import GREEN as G
from wordle.game import GREY as X
from wordle.game import YELLOW as Y
from wordle.game import WordleGame, score_guess


@pytest.mark.parametrize(
    "guess, secret, expected",
    [
        ("crane", "crane", (G, G, G, G, G)),
        ("ghost", "crane", (X, X, X, X, X)),
        ("nacre", "crane", (Y, Y, Y, Y, G)),
        # Duplicate letters in the guess: only as many marks as the secret has.
        ("babes", "abbey", (Y, Y, G, G, X)),
        ("kebab", "abbey", (X, Y, G, Y, Y)),
        ("eerie", "crane", (X, X, Y, X, G)),
        ("speed", "abide", (X, X, Y, X, Y)),
        ("floor", "robot", (X, X, Y, G, Y)),
        # Duplicate letters in the secret.
        ("abide", "speed", (X, X, X, Y, Y)),
        ("eerie", "keeps", (Y, G, X, X, X)),
    ],
)
def test_score_guess(guess, secret, expected):
    assert score_guess(guess, secret) == expected


def test_win_ends_game():
    game = WordleGame("crane")
    game.guess("slate")
    assert not game.over
    game.guess("crane")
    assert game.won and game.over and game.guesses_used == 2


def test_six_misses_lose():
    game = WordleGame("crane")
    for _ in range(6):
        game.guess("ghost")
    assert game.over and not game.won
    with pytest.raises(ValueError):
        game.guess("crane")


def test_no_limit_keeps_going_until_solved():
    game = WordleGame("crane", max_guesses=None)
    for _ in range(10):
        game.guess("ghost")
    assert not game.over
    game.guess("crane")
    assert game.won and game.over and game.guesses_used == 11


def test_invalid_guesses_rejected():
    game = WordleGame("crane", valid_guesses={"crane", "slate"})
    for bad in ["cran", "cranes", "cr4ne", "zzzzz"]:
        with pytest.raises(ValueError):
            game.guess(bad)
    assert game.guesses_used == 0
