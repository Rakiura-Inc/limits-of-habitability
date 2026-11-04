"""Choosing by looking: the menu and sliders set the choice, and resolve() falls back to a number."""

import ipywidgets
import pandas as pd
import pytest

import habtrain as ht


@pytest.fixture
def planets():
    return pd.DataFrame({"planet": ["TOI-1 b", "TOI-2 b", "TOI-3 b"]}, index=[7, 8, 9])


@pytest.fixture
def shown(monkeypatch):
    """Use real widgets, but catch what would be displayed instead of drawing it."""
    displayed = []
    monkeypatch.setattr(ht.choose, "_widgets", lambda: (ipywidgets, lambda *w: displayed.extend(w), lambda **kw: None))
    return displayed


def test_resolve_prefers_the_choice_made_in_the_control(planets, capsys):
    chooser = ht.choose.Chooser(["TOI-1 b", "TOI-2 b", "TOI-3 b"])
    chooser.set("TOI-3 b")
    assert ht.choose.resolve(chooser, 1, planets, "planet", "planet") == "TOI-3 b"
    assert "number 3, TOI-3 b. If you come back in a new session, put 3 in MY_PICK" in capsys.readouterr().out


def test_resolve_marks_a_choice_that_is_not_in_the_table(planets, capsys):
    chooser = ht.choose.Chooser([])
    chooser.set("TOI-9 b")
    ht.choose.resolve(chooser, ..., planets, "planet", "planet")
    assert "number ?, TOI-9 b" in capsys.readouterr().out


def test_resolve_uses_the_typed_number_when_nothing_is_chosen(planets, capsys):
    assert ht.choose.resolve(ht.choose.Chooser([]), 2, planets, "planet", "planet") == "TOI-2 b"
    assert ht.choose.resolve(None, "3", planets, "planet", "planet") == "TOI-3 b"
    assert "Your planet: number 3, TOI-3 b." in capsys.readouterr().out


@pytest.mark.parametrize("typed", [..., None, 0, 4])
def test_resolve_picks_at_random_when_nothing_usable_was_given(typed, planets, capsys):
    assert ht.choose.resolve(None, typed, planets, "planet", "planet") in {"TOI-1 b", "TOI-2 b", "TOI-3 b"}
    assert "Nothing chosen yet, so the notebook picked number" in capsys.readouterr().out


def test_menu_without_widgets_asks_for_a_number(monkeypatch, planets, capsys):
    monkeypatch.setattr(ht.choose, "_widgets", lambda: (None, None, None))
    chooser = ht.choose.menu(planets, "planet", lambda n, r: r.planet, lambda t, mark: None)
    assert chooser.value is None
    assert "Type your choice's number" in capsys.readouterr().out


def test_menu_choice_is_kept_and_marked_on_the_map(planets, shown):
    marked = []
    chooser = ht.choose.planet(
        planets.assign(P_days=1.0, sy_tmag=9.0, velocities=10, telescopes="HIRES 10"),
        lambda t, mark: marked.append(mark),
    )
    dropdown = shown[0]
    assert dropdown.options[0] == ht.choose.RANDOM
    assert dropdown.options[2].startswith("2. TOI-2 b: 1.0-day orbit")
    dropdown.value = dropdown.options[2]
    assert chooser.value == "TOI-2 b"
    assert marked == ["TOI-2 b"]
    dropdown.value = ht.choose.RANDOM
    assert chooser.value in chooser.names


def test_star_sliders_list_the_nearest_stars_and_keep_the_pick(shown):
    summary = pd.DataFrame({
        "Name": ["HD 1", "HD 2", "HD 3"],
        "BV": [0.6, 0.75, 0.9],
        "logRhk": [-4.9, -4.9, -4.5],
        "variation": [1.0, 2.0, 3.0],
        "Nobs": [50.0, 60.0, 70.0],
    })
    drawn = []

    def star_map(s, mark, title):
        drawn.append(mark)
        return ht.plot.star_map(s, mark=mark, title=title)

    chooser = ht.choose.star(summary, star_map, lambda s, bv, rhk: ht.data.nearest_stars(s, bv, rhk, n=2))
    colour, activity, dropdown, _ = shown
    assert (colour.value, activity.value) == (ht.choose.START_COLOUR, ht.choose.START_ACTIVITY)
    assert dropdown.options[1].startswith("HD 2: colour 0.75")
    assert drawn[-1] == {"HD 2": "HD 2", "HD 1": "HD 1"}
    dropdown.value = dropdown.options[2]
    assert chooser.value == "HD 1"
    colour.value = 0.9
    activity.value = -4.5
    assert dropdown.options[1].startswith("HD 3")
    assert dropdown.value is None  # moving the crosshair clears the menu
