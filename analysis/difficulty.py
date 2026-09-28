"""What makes a word hard? Play every answer word with an AI and compare.

For each word we record how many guesses the AI needed (playing its best guess,
no exploring) and a few traits of the word:
  - repeated_letters: e.g. DADDY, STALL. One yellow E doesn't reveal a second E,
    and an AI that likes 5 different letters rarely guesses doubles.
  - look_alikes: other answers that differ in exactly one letter (CATCH: BATCH,
    HATCH, LATCH, MATCH, PATCH, WATCH). Only guessing possible words, the AI can
    rule out roughly one of them per guess.
  - left_after_opener: words still possible after the AI's first guess, i.e.
    how little its opening word told it about this secret.
  - letter_commonness: how common the word's letters are across the list.
Then we average guesses per trait group, so the numbers say which traits matter.
"""
from collections import Counter

import numpy as np

from agents.base import play_game
from wordle.words import encode

LOOK_ALIKE_GROUPS = [(0, 0, "none"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 99, "6 or more")]
OPENER_GROUPS = [(0, 10, "10 or fewer"), (11, 50, "11-50"), (51, 150, "51-150"), (151, 9999, "over 150")]


def word_traits(words):
    """Per-word traits that don't depend on any AI."""
    letters, counts = encode(words)
    look_alikes = {w: [] for w in words}
    for pos in range(len(words[0])):
        groups = {}
        for w in words:
            groups.setdefault(w[:pos] + "_" + w[pos + 1:], []).append(w)
        for family in groups.values():
            for w in family:
                look_alikes[w] += [x for x in family if x != w]
    share = (counts > 0).mean(axis=0)  # share of words containing each letter
    commonness = [float(share[np.flatnonzero(c)].mean()) for c in counts]
    return {
        w: {
            "repeated_letters": bool((counts[i] > 1).any()),
            "look_alikes": sorted(look_alikes[w]),
            "letter_commonness": commonness[i],
        }
        for i, w in enumerate(words)
    }


TIE_SEED = 0  # when words score exactly the same, the AI picks at random: always the same way here


def best_game(agent, secret):
    """One game with the AI's best guesses and fixed tie-breaks: same AI + same word = same game."""
    agent.training = False
    agent.rng = np.random.default_rng(TIE_SEED)
    return play_game(agent, secret)


def trace_game(agent, secret):
    """Play one game with the AI's best guesses. For each guess: the colors, how many
    words were still possible afterwards, and whether it was a probe (a word that
    couldn't have been the answer, played to test letters)."""
    game = best_game(agent, secret)
    pool_words = agent.pool
    steps = []
    for turn, (guess, feedback) in enumerate(game.history):
        before = {pool_words[i] for i in agent.featurizer.candidates(game.history[:turn])}
        left = len(agent.featurizer.candidates(game.history[:turn + 1]))
        steps.append({"guess": guess, "feedback": list(feedback), "words_left": left,
                      "words_before": len(before), "probe": guess not in before})
    return steps


def analyze(agent, words, traits=None, progress=None, keep_rows=False):
    """Play every word once with the AI and summarize which traits make words hard.
    keep_rows=True also returns every word's row (the benchmark's regression needs them)."""
    traits = traits or word_traits(words)
    rows = []
    for i, secret in enumerate(words):
        game = best_game(agent, secret)
        opener = game.history[0][0]
        left = len(agent.featurizer.candidates(game.history[:1])) if game.guesses_used > 1 else 0
        rows.append({"word": secret, "guesses": game.guesses_used, "opener": opener,
                     "left_after_opener": left, **traits[secret],
                     "look_alike_count": len(traits[secret]["look_alikes"])})
        if progress and i % 100 == 0:
            progress(i / len(words))

    guesses = np.array([r["guesses"] for r in rows])

    def group(mask_label_pairs):
        return [{"label": label, "words": int(mask.sum()),
                 "avg_guesses": float(guesses[mask].mean()) if mask.any() else None}
                for mask, label in mask_label_pairs]

    repeated = np.array([r["repeated_letters"] for r in rows])
    look = np.array([r["look_alike_count"] for r in rows])
    left = np.array([r["left_after_opener"] for r in rows])
    common = np.array([r["letter_commonness"] for r in rows])
    cuts = np.quantile(common, [0.25, 0.5, 0.75])
    quartile = np.digitize(common, cuts)
    opener = Counter(r["opener"] for r in rows).most_common(1)[0][0]

    groups = {
        "repeated_letters": group([(~repeated, "no repeated letters"), (repeated, "has a repeated letter")]),
        "look_alikes": group([((look >= lo) & (look <= hi), label) for lo, hi, label in LOOK_ALIKE_GROUPS]),
        "left_after_opener": group([((left >= lo) & (left <= hi) & (guesses > 1), label)
                                    for lo, hi, label in OPENER_GROUPS]),
        "letter_commonness": group([(quartile == 3, "commonest letters (top 25%)"),
                                    (quartile == 2, "common"), (quartile == 1, "less common"),
                                    (quartile == 0, "rarest letters (bottom 25%)")]),
    }
    hardest = sorted(rows, key=lambda r: (-r["guesses"], r["word"]))[:15]
    extra = {"rows": rows} if keep_rows else {}
    return extra | {
        "words": len(rows),
        "avg_guesses": float(guesses.mean()),
        "opener": opener,
        "groups": groups,
        "conclusions": conclusions(groups, opener),
        "hardest": hardest,
    }


def conclusions(groups, opener):
    """One takeaway per trait: the hardest vs the easiest group, biggest gap first."""
    titles = {
        "repeated_letters": "Repeated letters",
        "left_after_opener": f"Words still possible after the opener ({opener.upper()})",
        "look_alikes": "One-letter look-alikes",
        "letter_commonness": "How common the letters are",
    }
    lines = []
    for name, title in titles.items():
        values = [g for g in groups[name] if g["avg_guesses"] is not None]
        easy = min(values, key=lambda g: g["avg_guesses"])
        hard = max(values, key=lambda g: g["avg_guesses"])
        lines.append({"trait": name, "title": title, "gap": hard["avg_guesses"] - easy["avg_guesses"],
                      "hard_label": hard["label"], "hard": hard["avg_guesses"],
                      "easy_label": easy["label"], "easy": easy["avg_guesses"]})
    return sorted(lines, key=lambda line: -line["gap"])
