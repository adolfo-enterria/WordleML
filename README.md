# WordleML

A terminal Wordle clone, plus a small AI that gets better at Wordle by playing it,
with a live dashboard where you watch it learn.

## Quick start

```
pip install -r requirements.txt
python dashboard.py                    # watch the AI train live in your browser
python play.py                         # play Wordle yourself
python play.py --watch                 # watch the trained AI play, showing its favorite words
python evaluate.py                     # score the trained AI on every answer word
pytest                                 # run the tests
```

## The dashboard

`python dashboard.py` opens a local page (http://127.0.0.1:8765) with:

- **Controls:**
  - Pause, Continue, and Reset. Reset goes back to a blank AI, with all 3 weights at zero, at game 0.
  - A "Games to train" slider (100–5,000) and a speed slider.
- **The skill check** (main chart). Every 5 games at first, then every ~25, training pauses and the AI plays the same 200 words. The luck of which word comes up can't move these lines, only learning can. It's measured two ways:
  - **Best guesses:** what it knows.
  - **Playing like in practice**, still exploring: how confidently it uses what it knows.
- **What it has learned:** its 3 weights, all zero after a Reset.
- **Why it's already good at game 0:** knowing the rules cuts the 2,315 words to ~130, ~9 and ~2 after each guess, even when the guesses are random.
- **Practice games:** the random-word games it learns from, one dot per game.
- **What makes a word hard?**
  - Look up any word to see the AI play it, with how many words were still possible after each guess.
  - Analyze all 2,315 words to see which traits make words hard.
  - Link straight to a word with `http://127.0.0.1:8765/?word=foyer`.

When a run reaches its target, the trained AI is saved to `models/agent.npz`.
Options: `--games 2000`, `--speed 0` (max), `--seed 3` (repeat a run).

## How the AI works

- **It knows the rules.** After each guess it works out which words could still be the answer (green = right letter, right spot; yellow = in the word, wrong spot; grey = not in the word) and only guesses those.
- **It never loses.** It keeps guessing until it finds the word.
- **Each game's secret is picked at random** from all 2,315 answer words, so it gets no clue before its first guess.
- **What it learns:** *which* possible word to guess. It looks at 3 facts about each one (how many different letters it has, how common its letters are among the words still possible, and how common they are in those spots) and weighs them. The weights start at zero, which means "pick any possible word at random".
- **How it learns:** REINFORCE, a basic reinforcement-learning method. After every game it nudges its weights toward the choices that led to fewer guesses than usual.

A typical 500-game run, on the 200 skill-check words:

| | Best guesses | Playing like in practice |
|---|---|---|
| Game 0 (blank AI = random possible word) | 4.05 | 4.17 |
| After 50 games | ~3.7 | ~3.85 |
| After 500 games | ~3.62 | ~3.67 |

Its best play improves within the first few dozen games. That's because only the
*direction* of its 3 weights matters for its best guess ("prefer common letters").
More training then makes it more confident, so it explores less.

## What makes a word hard

From analyzing all 2,315 words with a trained AI (one game each):

- **The opener's split:** when the opening word leaves few candidates (10 or fewer), the word averages ~2.95 guesses; with over 150 left, ~3.9.
- **Repeated letters:** words like DADDY or STALL average ~3.9–4.1 guesses vs ~3.45 for the rest. One yellow E doesn't reveal a second E, and the AI prefers guesses with 5 different letters.
- **Look-alike families:** CATCH, HATCH, MATCH… (6 or more one-letter look-alikes average ~3.8–4.0). When only possible words are allowed, you can rule out about one of them per guess. FOYER is a two-letter version of this: after _O_ER it has to try COWER, HOVER, POKER…
- **Rare letters** make a word slightly harder.

Together, these traits explain only part of the difference between words. The rest
depends on the exact colors the AI happens to get along the way.

## Layout

```
wordle/     the game: rules and duplicate-letter scoring, word lists, terminal colors
agents/     the rules + facts (features.py), the learner (learning_agent.py), no-strategy baseline
analysis/   word difficulty: traits of each word and what makes words hard (difficulty.py)
live/       the dashboard: training session (session.py), web server (server.py), page (static/)
dashboard.py, play.py, evaluate.py    the commands above
data/       answers.txt (possible secrets) + allowed_guesses.txt (extra words you may type)
```
