"""Build the "common words" lists for every word length, the same way for each length.

    python -m wordle.wordlists          # writes data/lists/common{L}_answers.txt and common{L}_guesses.txt

Sources (in data/sources/, see data/sources/README.md for attribution and licenses):
  enable1.txt   ENABLE word list (public domain): the classic word-game dictionary.
  scowl80.txt   SCOWL / English Speller Database, size 80 (MIT-like license): "all the
                strange and unusual words people like to use in word games such as Scrabble".
  en_50k.txt    FrequencyWords, 50,000 most frequent English words (CC BY-SA 4.0).
  plus the official Wordle lists (data/answers.txt, data/allowed_guesses.txt) at 5 letters.

Recipe for length L:
  answers  ENABLE words of length L that are among the N most frequent English words,
           minus simple plurals (CATS when CAT is a word, BOXES/BOX, CITIES/CITY).
  guesses  every dictionary word of length L: ENABLE + SCOWL-80 (lowercase a-z only: no
           names, abbreviations with capitals or apostrophes) + the official Wordle list at
           5 letters. The answers come first, then the rest, most frequent first.
N is calibrated once: the smallest cutoff that gives as many 5-letter answers as the
official Wordle list (2,315). The same N is then used for every length, so the lists
are "natural": each length gets however many common words it really has.
"""
import re

from wordle.words import ALLOWED_GUESSES_PATH, ANSWERS_PATH, DATA_DIR

SOURCES = DATA_DIR / "sources"
LISTS = DATA_DIR / "lists"
LENGTHS = range(3, 11)
CALIBRATE_TO = 2315      # official Wordle answer count


def is_simple_plural(word, dictionary):
    if not word.endswith("s") or word.endswith("ss"):
        return False
    return (word[:-1] in dictionary
            or (word.endswith("es") and word[:-2] in dictionary)
            or (word.endswith("ies") and word[:-3] + "y" in dictionary))


def read_scowl(path):
    """The words of a SCOWL list: everything after the '---' line that ends its license header."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {w for w in lines[lines.index("---") + 1:] if re.fullmatch(r"[a-z]+", w)}


def load_sources():
    """(answer dictionary, guess dictionary, words ranked by frequency)."""
    enable = {w.strip() for w in (SOURCES / "enable1.txt").read_text().split() if w.strip()}
    wordle = set(ANSWERS_PATH.read_text().split()) | set(ALLOWED_GUESSES_PATH.read_text().split())
    everything = enable | read_scowl(SOURCES / "scowl80.txt") | wordle
    ranked = [line.split()[0] for line in (SOURCES / "en_50k.txt").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    return enable, everything, ranked


def build(dictionary, ranked, everything=None):
    """{length: (answers, guesses)} plus the frequency cutoff that was used."""
    everything = dictionary if everything is None else everything
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
        others = sorted((w for w in everything if len(w) == length and w not in chosen),
                        key=lambda w: (rank.get(w, len(ranked)), w))
        lists[length] = (answers, answers + others)
    return lists, cutoff


def main():
    enable, everything, ranked = load_sources()
    lists, cutoff = build(enable, ranked, everything)
    LISTS.mkdir(parents=True, exist_ok=True)
    print(f"Frequency cutoff: the {cutoff:,} most frequent words (calibrated to {CALIBRATE_TO:,} 5-letter answers)")
    for length, (answers, guesses) in lists.items():
        (LISTS / f"common{length}_answers.txt").write_text("\n".join(answers) + "\n")
        (LISTS / f"common{length}_guesses.txt").write_text("\n".join(guesses) + "\n")
        print(f"  {length:>2} letters: {len(answers):>5,} answers, {len(guesses):>6,} allowed guesses")


if __name__ == "__main__":
    main()
