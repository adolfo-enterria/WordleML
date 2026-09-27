import json
import threading
import urllib.request

import pytest

from live.server import make_server
from live.session import TrainingSession
from wordle.words import load_words

WORDS = load_words()


@pytest.fixture
def session(tmp_path):
    return TrainingSession(WORDS, target_games=5, seed=0, model_path=tmp_path / "agent.npz")


def test_plays_until_target_then_saves(session):
    while session.play_one_game():
        pass
    state = session.snapshot()
    assert state["status"] == "finished"
    assert state["games_played"] == 5
    assert all(guesses >= 1 and secret in WORDS for _, secret, guesses in state["results"])
    assert session.model_path.exists()


def test_pause_continue(session):
    session.pause()
    assert not session.play_one_game()
    session.resume()
    assert session.play_one_game()
    assert session.snapshot()["games_played"] == 1


def test_reset_starts_over_untrained(session):
    for _ in range(3):
        session.play_one_game()
    old_run = session.snapshot()["run_id"]
    session.reset()
    state = session.snapshot()
    assert state["games_played"] == 0 and state["run_id"] == old_run + 1
    assert not session.agent.weights.any()


def test_snapshot_only_sends_new_games(session):
    for _ in range(3):
        session.play_one_game()
    state = session.snapshot(since=2)
    assert [game for game, _, _ in state["results"]] == [3]
    assert state["stats"]["first_count"] == 3


def test_server_round_trip(session):
    server = make_server(session, port=0)  # port 0: any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as page:
            assert b"WordleML" in page.read()
        post = urllib.request.Request(base + "/api/pause", data=b"{}", method="POST")
        with urllib.request.urlopen(post) as reply:
            assert json.load(reply)["status"] == "paused"
        speed = urllib.request.Request(base + "/api/speed", data=b'{"games_per_second": 25}',
                                       method="POST")
        with urllib.request.urlopen(speed) as reply:
            assert json.load(reply)["games_per_second"] == 25
        with urllib.request.urlopen(base + "/api/state?since=0") as reply:
            assert json.load(reply)["status"] == "paused"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/../agents/features.py")
    finally:
        server.shutdown()
