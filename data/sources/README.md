# Word-list sources

Used by `python -m wordle.wordlists` to build `data/lists/common{3..10}_*.txt`.

- `enable1.txt`: ENABLE word list (Enhanced North American Benchmark Lexicon), public domain.
  Downloaded from https://github.com/dolph/dictionary (`enable1.txt`). The possible answers come from it.
- `scowl80.txt`: SCOWL / English Speller Database (ESDB), size 80 ("huge": all the unusual words people
  use in word games such as Scrabble), US + GB + CA + AU spellings, generated with the project's own
  list builder at https://app.aspell.net/create (max_size=80, max_variant=2, diacritics stripped).
  Copyright 2000-2026 by Kevin Atkinson; the full copyright and permission notice is kept at the top of
  the file, as its license requires. Only lowercase a-z entries are used (no names, no apostrophes).
  https://wordlist.aspell.net/
- `en_50k.txt`: the 50,000 most frequent English words with counts, from Hermit Dave's
  FrequencyWords (OpenSubtitles 2018), licensed CC BY-SA 4.0.
  https://github.com/hermitdave/FrequencyWords (`content/2018/en/en_50k.txt`).

Allowed guesses for each length are every word in ENABLE + SCOWL-80 (+ the official Wordle list at
5 letters). The official Wordle lists (`data/answers.txt`, `data/allowed_guesses.txt`) are also used on
their own for the `wordle5` word set.

Not used: wordsrated.com's word finder. Its site blocks automated downloads, shows its lists 500
words at a time, and merges the official Scrabble dictionaries (Collins, NWL), which are copyrighted
and not published for download. There is no single "official English dictionary"; ENABLE + SCOWL-80
is the most complete freely licensed equivalent (70-100% of wordsrated's counts per length).
