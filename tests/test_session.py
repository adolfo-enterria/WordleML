import json
import threading
import time
import urllib.request

import pytest

from analysis.difficulty import word_traits
from live.server import make_server
from live.session import TrainingSession
from wordle.words import load_words

WORDS = load_words()


@pytest.fixture
def session(tmp_path):
    return TrainingSession(WORDS, target_games=20, seed=0, model_path=tmp_path / "agent.npz")


def run_until_idle(session):
    while session.step():
        pass


def test_skill_checks_at_start_every_interval_and_end(session):
    run_until_idle(session)
    state = session.snapshot()
    assert state["status"] == "finished"
    assert state["games_played"] == 20
    assert [check[0] for check in state["skill_checks"]] == [0, 5, 10, 15, 20]  # every 5 early on
    assert all(1 <= best <= 10 and 1 <= practice <= 10 for _, best, practice in state["skill_checks"])
    assert session.model_path.exists()


def test_untrained_skill_is_the_same_after_every_reset(session):
    session.step()  # game-0 check
    first = session.snapshot()["skill_checks"][0]
    for _ in range(5):
        session.step()
    session.reset()
    session.step()
    assert session.snapshot()["skill_checks"] == [first]


def test_reset_starts_over_untrained(session):
    for _ in range(4):
        session.step()
    old_run = session.snapshot()["run_id"]
    session.reset()
    state = session.snapshot()
    assert state["games_played"] == 0 and state["skill_checks"] == []
    assert state["run_id"] == old_run + 1
    assert all(w == 0 for w in state["weights"].values())


def test_pause_continue(session):
    session.pause()
    assert session.step() is None
    session.resume()
    assert session.step() == "check"
    assert session.step() == "game"


def test_changing_the_target(session):
    run_until_idle(session)
    session.set_target(40)
    assert session.snapshot()["status"] == "paused"  # more to do: waiting for Continue
    session.resume()
    run_until_idle(session)
    state = session.snapshot()
    assert state["games_played"] == 40 and state["status"] == "finished"
    assert state["skill_checks"][-1][0] == 40
    assert session.agent.weights.any()  # skill checks don't stop the real agent learning


def test_snapshot_only_sends_new_games(session):
    for _ in range(4):  # a check, then 3 games
        session.step()
    state = session.snapshot(since=2)
    assert [game for game, _, _ in state["results"]] == [3]


def test_word_traits():
    traits = word_traits(WORDS)
    assert set(traits["catch"]["look_alikes"]) >= {"batch", "hatch", "latch", "match", "patch", "watch"}
    assert traits["daddy"]["repeated_letters"] and not traits["crane"]["repeated_letters"]


def test_word_report(session):
    report = session.word_report("Foyer ")
    assert report["word"] == "foyer"
    assert report["steps"][-1]["guess"] == "foyer" and report["steps"][-1]["words_left"] == 1
    lefts = [s["words_left"] for s in report["steps"]]
    assert lefts == sorted(lefts, reverse=True)
    assert "error" in session.word_report("fayot")


def test_analysis_runs_in_background(session):
    session.start_analysis()
    deadline = time.time() + 60
    while session.analysis_result()["state"] != "done" and time.time() < deadline:
        time.sleep(0.2)
    result = session.analysis_result()["result"]
    assert result["words"] == len(WORDS)
    assert len(result["conclusions"]) == 4 and len(result["hardest"]) == 15
    hardest = result["hardest"][0]  # the lookup must tell the same story as the analysis
    assert len(session.word_report(hardest["word"])["steps"]) == hardest["guesses"]


def test_server_round_trip(session):
    server = make_server(session, port=0)  # port 0: any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body=b"{}"):
        with urllib.request.urlopen(urllib.request.Request(base + path, data=body, method="POST")) as r:
            return json.load(r)

    try:
        with urllib.request.urlopen(base + "/") as page:
            assert b"WordleML" in page.read()
        assert post("/api/pause")["status"] == "paused"
        assert post("/api/speed", b'{"games_per_second": 25}')["games_per_second"] == 25
        assert post("/api/target", b'{"games": 1000}')["target_games"] == 1000
        with urllib.request.urlopen(base + "/api/word?w=crane") as reply:
            assert json.load(reply)["steps"][-1]["guess"] == "crane"
        with pytest.raises(urllib.error.HTTPError):
            post("/api/target", b'{"oops": 1}')
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/../agents/features.py")
    finally:
        server.shutdown()
