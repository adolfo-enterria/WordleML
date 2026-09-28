"""Build the "common words" lists for every word length, the same way for each length.

    python -m wordle.wordlists          # writes data/lists/common{L}_answers.txt and common{L}_guesses.txt

Sources (in data/sources/, see data/sources/README.md for attribution):
  enable1.txt   ENABLE word list (public domain): which strings are real words.
  en_50k.txt    FrequencyWords, 50,000 most frequent English words (CC BY-SA 4.0).

Recipe for length L:
  answers  dictionary words of length L that are among the N most frequent English
           words, minus simple plurals (CATS when CAT is a word, BOXES/BOX, CITIES/CITY).
  guesses  the answers first, then the other dictionary words of length L, most
           frequent first, up to MAX_GUESSES words in total.
N is calibrated once: the smallest cutoff that gives as many 5-letter answers as the
official Wordle list (2,315). The same N is then used for every length, so the lists
are "natural": each length gets however many common words it really has.
"""
from pathlib import Path

from wordle.words import DATA_DIR

SOURCES = DATA_DIR / "sources"
LISTS = DATA_DIR / "lists"
LENGTHS = range(3, 9)
CALIBRATE_TO = 2315      # official Wordle answer count
MAX_GUESSES = 15_000     # keeps the pattern tables for 7-8 letter words small enough


def is_simple_plural(word, dictionary):
    if not word.endswith("s") or word.endswith("ss"):
        return False
    return (word[:-1] in dictionary
            or (word.endswith("es") and word[:-2] in dictionary)
            or (word.endswith("ies") and word[:-3] + "y" in dictionary))


def load_sources():
    dictionary = {w.strip() for w in (SOURCES / "enable1.txt").read_text().split() if w.strip()}
    ranked = [line.split()[0] for line in (SOURCES / "en_50k.txt").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    return dictionary, ranked


def build(dictionary, ranked):
    """{length: (answers, guesses)} plus the frequency cutoff that was used."""
    usable = [w for w in ranked
              if w.isascii() and w.isalpha() and w in dictionary and not is_simple_plural(w, dictionary)]
    rank = {w: i for i, w in enumerate(ranked)}
    fives, cutoff = 0, len(ranked)
    for word in usable:  # smallest cutoff giving CALIBRATE_TO five-letter answers
        fives += len(word) == 5
        if fives == CALIBRATE_TO:
            cutoff = rank[word] + 1
            break
    common = [w for w in usable if rank[w] < cutoff]
    lists = {}
    for length in LENGTHS:
        answers = sorted(w for w in common if len(w) == length)
        chosen = set(answers)
        others = sorted((w for w in dictionary if len(w) == length and w not in chosen),
                        key=lambda w: (rank.get(w, len(ranked)), w))
        lists[length] = (answers, answers + others[:max(0, MAX_GUESSES - len(answers))])
    return lists, cutoff


def main():
    dictionary, ranked = load_sources()
    lists, cutoff = build(dictionary, ranked)
    LISTS.mkdir(parents=True, exist_ok=True)
    print(f"Frequency cutoff: the {cutoff:,} most frequent words (calibrated to {CALIBRATE_TO:,} 5-letter answers)")
    for length, (answers, guesses) in lists.items():
        (LISTS / f"common{length}_answers.txt").write_text("\n".join(answers) + "\n")
        (LISTS / f"common{length}_guesses.txt").write_text("\n".join(guesses) + "\n")
        print(f"  {length} letters: {len(answers):>5,} answers, {len(guesses):>6,} allowed guesses")


if __name__ == "__main__":
    main()
