"""Saving answers and writing the hand-in."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import habtrain as ht


@pytest.mark.parametrize("value", [..., None, "", "  ...  ", {}, {"a": 1, "b": ...}, []])
def test_blank_answers_are_not_filled(value):
    assert not ht.record.filled(value)


@pytest.mark.parametrize("value", [0, False, "no", {"a": 1, "b": 2}, [0], np.array([]), pd.DataFrame()])
def test_real_answers_are_filled(value):
    assert ht.record.filled(value)


def test_save_skips_a_blank_answer_and_keeps_a_real_one(capsys):
    ht.record.save("k", ..., "the wobble")
    assert "Not saved yet: fill in the wobble" in capsys.readouterr().out
    ht.record.save("k", 3.14159, "the wobble")
    assert "Saved: the wobble = 3.142" in capsys.readouterr().out
    assert ht.record.table(["k", "missing"]).splitlines()[2:] == [
        "| the wobble | 3.142 |",
        "| missing | (not answered) |",
    ]


def test_a_reveal_waits_for_its_prediction(capsys):
    assert not ht.record.revealed("guess")
    ht.record.predict("guess", ...)
    assert not ht.record.revealed("guess")
    ht.record.predict("guess", 5)
    assert ht.record.revealed("guess")
    assert "Prediction saved at" in capsys.readouterr().out


def test_fill_uses_the_stand_in_until_you_answer(capsys):
    assert ht.record.fill(..., "TOI-1 b", "MY_PLANET") == "TOI-1 b"
    assert "MY_PLANET is still empty, so the cells below use TOI-1 b" in capsys.readouterr().out
    assert ht.record.fill(..., [1, 2], "rows") == [1, 2]
    assert "use a stand-in" in capsys.readouterr().out
    assert ht.record.fill("mine", "stand-in", "x") == "mine"


@pytest.fixture
def table():
    return pd.DataFrame({"planet": ["a", "b", "c"]}, index=[10, 20, 30])


def test_pick_by_number_counts_from_one(table, capsys):
    assert ht.record.pick(2, table, "planet", "planet").planet == "b"
    assert "Your planet: number 2, b." in capsys.readouterr().out


@pytest.mark.parametrize("choice", [0, 4, 1.5, "2"])
def test_pick_out_of_range_uses_number_one(choice, table, capsys):
    assert ht.record.pick(choice, table, "planet", "planet").planet == "a"
    assert "Pick a whole number from 1 to 3" in capsys.readouterr().out


def test_pick_left_blank_picks_at_random_and_says_which(table, capsys):
    row = ht.record.pick(..., table, "planet", "planet")
    assert row.planet in {"a", "b", "c"}
    assert "picked number" in capsys.readouterr().out


def test_numbered_counts_rows_from_one(table):
    t = ht.record.numbered(table)
    assert t.index.tolist() == [1, 2, 3]
    assert t.index.name == "#"


def test_show_says_when_there_is_nothing_yet(capsys):
    ht.record.show(...)
    assert "Nothing to show yet" in capsys.readouterr().out


def test_pkg_commit_names_a_commit_or_the_version():
    commit = ht.record.pkg_commit()
    assert commit == f"v{ht.__version__}" or len(commit) == ht.record.SHORT_SHA


def test_handin_writes_answers_figures_and_ticks_the_part(capsys):
    ht.record.save("k", 9.5, "the wobble")
    ht.record.predict("guess", 4)
    ht.record.revealed("guess")
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    ht.record.handin("planet-weighed", ["k", "mass"], figures={"orbit": fig}, numbers={"k": 9.5})
    out = Path(ht.portal.state.handin_dir) / "planet-weighed"
    record = json.loads((out / "results.json").read_text())
    assert record["answers"]["k"]["value"] == 9.5
    assert record["missing"] == ["mass"]
    assert record["predictions"]["guess"]["made_before_reveal"] is True
    assert (out / "results.md").read_text().startswith("| Question | Your answer |")
    assert sorted(p.name for p in out.iterdir()) == ["orbit.png", "results.json", "results.md"]
    printed = capsys.readouterr().out
    assert "1 question(s) still unanswered" in printed
    assert "results.md (the table above), results.json, orbit.png" in printed
    log = json.loads(ht.portal._log_path().read_text())
    assert [entry.get("part") or entry.get("staged") for entry in log] == [
        "planet-weighed",
        "planet-weighed results.json",
    ]


def test_a_prediction_made_after_its_reveal_is_marked():
    ht.record.predict("late", 1)
    ht.record._reveals["late"] = "2000-01-01T00:00:00"
    assert ht.record._record("p", [])["predictions"]["late"]["made_before_reveal"] is False
