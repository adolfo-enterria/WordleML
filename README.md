# WordleML

A terminal Wordle clone, plus a small AI that gets better at Wordle by playing it,
with a live dashboard where you watch it learn.

## Quick start

```
pip install -r requirements.txt
python dashboard.py                    # watch the AI train live in your browser (500 games)
python play.py                         # play Wordle yourself
python play.py --watch                 # watch the trained AI play, showing its favorite words
python evaluate.py                     # score the trained AI on every answer word
pytest                                 # run the tests
```

## The dashboard

`python dashboard.py` opens a local page (http://127.0.0.1:8765) showing:

- **One dot per game.** The x-axis is the game number and the y-axis is how many guesses that game took.
- **A line for the average of the last 50 games**, plus a reference line for "no strategy".
- **Controls:** Pause, Continue, Reset (forget everything and start from game 1) and a speed slider.
- **Stat tiles**, plus a table of every game.

When it reaches the target (500 games by default), the trained AI is saved to
`models/agent.npz`. Options: `--games 2000`, `--speed 0` (max), `--seed 3` (repeat a run).

## How the AI works

- **It knows the rules.** After each guess it works out which words could still be the answer (green = right letter, right spot; yellow = in the word, wrong spot; grey = not in the word) and only guesses those.
- **It never loses.** It keeps guessing until it finds the word.
- **Each game's secret is picked at random** from all 2,315 answer words, so it gets no clue before its first guess.
- **What it learns:** *which* possible word to guess. It looks at 3 facts about each one (how many different letters it has, how common its letters are among the words still possible, and how common they are in those spots) and weighs them. The weights start at zero, which means "pick any possible word at random".
- **How it learns:** REINFORCE, a basic reinforcement-learning method. After every game it nudges its weights toward the choices that led to fewer guesses than usual.

| Player (one game per answer word) | Average guesses |
|---|---|
| No strategy: a random word that's still possible (= the untrained AI) | 4.13 |
| The AI after 500 training games, playing its best guesses | 3.61 |

The dashboard plots the **training** games, where the AI still tries things out
and each secret is random, so single games bounce between 2 and 8 guesses. The
average line is where the learning shows: averaged over many runs, it goes from
about 4.0 down to about 3.7 in the first ~300 games and then levels off.

## Layout

```
wordle/     the game: rules and duplicate-letter scoring, word lists, terminal colors
agents/     the rules + facts (features.py), the learner (learning_agent.py), no-strategy baseline
live/       the dashboard: training session (session.py), web server (server.py), page (static/)
dashboard.py, play.py, evaluate.py    the commands above
data/       answers.txt (possible secrets) + allowed_guesses.txt (extra words you may type)
```
