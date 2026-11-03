"""The Rakiura bridge: offline it keeps a local log, live it hands the same calls to the helper."""

import json
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest

import habtrain as ht


@pytest.fixture
def helper(monkeypatch):
    """A stand-in for the Rakiura helper that records every call it gets."""
    calls = []
    module = types.ModuleType("rakiura")
    module._session = types.SimpleNamespace(quiet=False)
    module.started = True

    def record(name, result=True):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return call

    module.start = lambda **kwargs: calls.append(("start", (), kwargs)) or module.started
    module.done = record("done")
    module.stage_text = record("stage_text")
    module.submit = record("submit")
    module.me = lambda show: {"member": {"name": "Ada"}}
    module._colab_secret = lambda: None
    module._credentials_path = lambda: "/nowhere/credentials"
    module.calls = calls
    monkeypatch.setitem(sys.modules, "rakiura", module)
    return module


def test_offline_connect_says_the_work_stays_here(capsys, tmp_path):
    assert ht.portal.connect("notebooks/01.ipynb", handin_dir=str(tmp_path / "out")) == "offline"
    assert "Your work stays on this computer" in capsys.readouterr().out
    assert (tmp_path / "out").is_dir()
    assert ht.portal.state.notebook == "notebooks/01.ipynb"


def test_offline_done_and_stage_file_write_the_local_log():
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    assert ht.portal.done("part-1", figures=[fig], numbers={"k": np.float64(2.5), "note": None})
    assert ht.portal.stage_file("hello", "notes.txt")
    log = json.loads((ht.portal._log_path()).read_text())
    assert log[0]["part"] == "part-1"
    assert log[0]["figures"] == 1
    assert log[0]["numbers"] == {"k": 2.5, "note": "blank"}
    assert log[1]["staged"] == "notes.txt"
    assert log[1]["bytes"] == 5
    assert ht.portal.state.last_part == "part-1"


@pytest.mark.parametrize(
    ("part", "numbers", "figures", "says"),
    [
        ("part 1", None, None, "a part name is letters"),
        ("part-1", {"table": [1, 2]}, None, "never a table or an array"),
        ("part-1", {str(i): i for i in range(21)}, None, "up to 20"),
        ("part-1", None, ["not a figure"], "that figure is empty"),
    ],
)
def test_done_refuses_what_the_helper_would_refuse(part, numbers, figures, says, capsys):
    assert not ht.portal.done(part, figures=figures, numbers=numbers)
    assert says in capsys.readouterr().out


def test_long_text_numbers_are_cut_to_200_characters():
    assert ht.portal._clean_numbers({"a": "x" * 300}) == {"a": "x" * 200}


def test_submit_offline_sends_nothing(capsys):
    assert not ht.portal.submit("please")
    assert "nothing was sent" in capsys.readouterr().out


def test_saved_words_name_the_part_without_its_id():
    assert ht.portal.saved_words("planet-picked") == "Part 1 saved"
    assert ht.portal.saved_words("cycle-found") == "Hand-in saved"
    assert ht.portal.saved_words("something-else") == "Progress saved"


def test_steps_come_from_the_file_and_the_environment_overrides_the_step(monkeypatch, tmp_path):
    steps = tmp_path / "steps.json"
    steps.write_text(json.dumps({"notebooks": {"01.ipynb": {"project": "p1", "step": "s1"}}}))
    monkeypatch.setenv("HABTRAIN_STEPS", str(steps))
    assert ht.portal.steps("notebooks/01.ipynb") == {"project": "p1", "step": "s1"}
    assert ht.portal.steps()["notebooks"]["01.ipynb"]["step"] == "s1"
    monkeypatch.setenv("RAKIURA_STEP", "s9")
    assert ht.portal.steps("01.ipynb")["step"] == "s9"
    assert ht.portal.steps("02.ipynb") == {"step": "s9"}


def test_the_repository_steps_file_names_every_notebook():
    conf = ht.portal.steps()
    assert set(conf["notebooks"]) >= {"01_find_your_test_planet.ipynb", "tess_02_radius_and_mass.ipynb"}


@pytest.mark.usefixtures("helper")
def test_rakiura_live_0_stays_offline_even_with_the_helper():
    assert ht.portal.connect("01.ipynb", sign_in=True) == "offline"


def test_live_with_a_key_connects_without_a_click_and_greets_the_member(helper, monkeypatch, capsys):
    monkeypatch.delenv("RAKIURA_LIVE")
    monkeypatch.setenv("RAKIURA_KEY", "test-key")
    assert ht.portal.connect("notebooks/01.ipynb") == "live"
    assert "Connected to Rakiura as Ada." in capsys.readouterr().out
    assert helper.calls[0] == (
        "start",
        (),
        {"notebook": "notebooks/01.ipynb", "repo": ht.portal.DEFAULT_REPO, "quiet": True},
    )
    assert helper._session.quiet is False  # the helper may speak again


def test_live_calls_go_to_the_helper(helper, monkeypatch, tmp_path):
    steps = tmp_path / "steps.json"
    steps.write_text(json.dumps({"notebooks": {"01.ipynb": {"step": "s1"}}}))
    monkeypatch.setenv("HABTRAIN_STEPS", str(steps))
    monkeypatch.setenv("RAKIURA_LIVE", "1")
    assert ht.portal.connect("notebooks/01.ipynb") == "live"
    assert ht.portal.done("part-1", numbers={"k": 1})
    assert ht.portal.stage_file("text", "a.json", mime="application/json")
    assert ht.portal.submit("done")
    names = [name for name, _, _ in helper.calls]
    assert names == ["start", "done", "stage_text", "submit"]
    assert helper.calls[2][2] == {"step": "s1", "mime": "application/json"}
    assert not ht.portal._log_path().exists()  # nothing written locally when live


def test_live_without_a_step_leaves_attaching_to_the_member(helper, monkeypatch):
    monkeypatch.setenv("RAKIURA_LIVE", "1")
    monkeypatch.setenv("HABTRAIN_STEPS", "/nowhere.json")
    ht.portal.connect("elsewhere.ipynb")
    assert not ht.portal.stage_file("text", "a.json")
    assert [name for name, _, _ in helper.calls] == ["start"]


def test_a_failed_sign_in_stays_offline_and_says_what_to_do(helper, monkeypatch, capsys):
    monkeypatch.delenv("RAKIURA_LIVE")
    helper.started = False
    assert ht.portal.go_live("01.ipynb") == "offline"
    assert "Run this cell again" in capsys.readouterr().out


def test_without_a_key_it_does_not_try_to_sign_in(helper, monkeypatch, capsys):
    monkeypatch.delenv("RAKIURA_LIVE")
    assert ht.portal.connect("01.ipynb") == "offline"
    assert helper.calls == []
    assert "Run the Connect cell below" in capsys.readouterr().out


def test_a_helper_that_raises_never_stops_the_notebook(helper, monkeypatch):
    monkeypatch.setenv("RAKIURA_LIVE", "1")

    def broken(**kwargs):
        raise RuntimeError

    helper.start = broken
    assert ht.portal.connect("01.ipynb") == "offline"


def test_a_package_named_rakiura_without_the_helper_calls_is_ignored(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "rakiura", types.ModuleType("rakiura"))
    monkeypatch.delenv("RAKIURA_LIVE")
    assert ht.portal._import_helper() is None
    ht.portal.connect("01.ipynb", sign_in=True)
    assert "helper is not installed" in capsys.readouterr().out
