# Word-list sources

Used by `python -m wordle.wordlists` to build `data/lists/common{3..8}_*.txt`.

- `enable1.txt`: ENABLE word list (Enhanced North American Benchmark Lexicon), public domain.
  Downloaded from https://github.com/dolph/dictionary (`enable1.txt`).
- `en_50k.txt`: the 50,000 most frequent English words with counts, from Hermit Dave's
  FrequencyWords (OpenSubtitles 2018), licensed CC BY-SA 4.0.
  https://github.com/hermitdave/FrequencyWords (`content/2018/en/en_50k.txt`).

The official Wordle lists (`data/answers.txt`, `data/allowed_guesses.txt`) are separate and are
used for the `wordle5` word set.
