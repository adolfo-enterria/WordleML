"""Colored terminal rendering of Wordle boards."""
from colorama import Back, Fore, Style, just_fix_windows_console

from wordle.game import GREEN, GREY, MAX_GUESSES, YELLOW

just_fix_windows_console()

_TILE_STYLE = {
    GREEN: Back.GREEN + Fore.BLACK,
    YELLOW: Back.YELLOW + Fore.BLACK,
    GREY: Back.LIGHTBLACK_EX + Fore.WHITE,
}
_KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def tile(letter, state):
    return f"{_TILE_STYLE[state]}{Style.BRIGHT} {letter.upper()} {Style.RESET_ALL}"


def render_row(guess, feedback):
    return " ".join(tile(letter, state) for letter, state in zip(guess, feedback))


def render_board(history, max_guesses=MAX_GUESSES, length=5):
    lines = [render_row(guess, feedback) for guess, feedback in history]
    empty = " ".join(f"{Style.DIM}[ ]{Style.RESET_ALL}" for _ in range(length))
    lines += [empty] * (max_guesses - len(history))
    return "\n".join(lines)


def letter_states(history):
    """Best state seen for each guessed letter (green beats yellow beats grey)."""
    states = {}
    for guess, feedback in history:
        for letter, state in zip(guess, feedback):
            states[letter] = max(states.get(letter, GREY), state)
    return states


def render_keyboard(history):
    states = letter_states(history)
    lines = []
    for indent, row in enumerate(_KEYBOARD_ROWS):
        keys = []
        for letter in row:
            if letter in states:
                keys.append(f"{_TILE_STYLE[states[letter]]}{letter.upper()}{Style.RESET_ALL}")
            else:
                keys.append(letter.upper())
        lines.append(" " * indent + " ".join(keys))
    return "\n".join(lines)
