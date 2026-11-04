"""Choosing by looking: the menus set the choice, and resolve() falls back to a number."""

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

    def display(*items):
        displayed.extend(items)

    def clear_output(**kwargs):
        pass

    monkeypatch.setattr(ht.choose, "_widgets", lambda: (ipywidgets, display, clear_output))
    return displayed


def resolve(choice, typed, planets):
    return ht.choose.resolve(choice, typed, table=planets, name_column="planet", what="planet")


def test_resolve_prefers_the_choice_made_in_the_menu(planets, capsys):
    assert resolve(ht.choose.Choice("TOI-3 b"), 1, planets) == "TOI-3 b"
    out = capsys.readouterr().out
    assert "Your planet: TOI-3 b." in out
    assert "If you come back in a new session, put 3 in MY_PICK." in out


def test_resolve_gives_no_number_for_a_choice_from_another_table(planets, capsys):
    assert resolve(ht.choose.Choice("TOI-9 b"), ..., planets) == "TOI-9 b"
    assert "MY_PICK" not in capsys.readouterr().out


def test_resolve_uses_the_typed_number_when_nothing_is_chosen(planets, capsys):
    assert resolve(ht.choose.Choice(), 2, planets) == "TOI-2 b"
    assert resolve(ht.choose.Choice(), "3", planets) == "TOI-3 b"
    assert "Your planet: number 3, TOI-3 b." in capsys.readouterr().out


@pytest.mark.parametrize("typed", [..., None, 0, 4])
def test_resolve_picks_the_same_planet_at_random_when_nothing_usable_was_given(typed, planets, capsys):
    picked = resolve(ht.choose.Choice(), typed, planets)
    assert picked == ht.choose.random_pick(["TOI-1 b", "TOI-2 b", "TOI-3 b"])
    assert "Nothing chosen yet, so the notebook picked number" in capsys.readouterr().out


def test_menu_without_widgets_asks_for_a_number(monkeypatch, planets, capsys):
    monkeypatch.setattr(ht.choose, "_widgets", lambda: (None, None, None))
    choice = ht.choose.planet_menu(planets.assign(P_days=1.0, sy_tmag=9.0, velocities=10, telescopes="HIRES 10"))
    assert choice.name is None
    assert "Type your choice's number" in capsys.readouterr().out


def test_planet_menu_keeps_the_choice_and_marks_it_on_the_map(planets, shown, monkeypatch):
    marked = []
    monkeypatch.setattr(ht.plot, "planet_map", lambda table, mark: marked.append(mark))
    choice = ht.choose.planet_menu(planets.assign(P_days=1.0, sy_tmag=9.0, velocities=10, telescopes="HIRES 10"))
    dropdown = shown[0]
    assert dropdown.options[0] == ht.choose.RANDOM
    assert dropdown.options[2].startswith("2. TOI-2 b: 1.0-day orbit")
    dropdown.value = dropdown.options[2]
    assert choice.name == "TOI-2 b"
    assert marked == ["TOI-2 b"]
    dropdown.value = ht.choose.RANDOM
    assert choice.name in {"TOI-1 b", "TOI-2 b", "TOI-3 b"}


def test_catalog_menu_says_which_numbers_are_measured(shown):
    candidates = pd.DataFrame({
        "Planet": ["A b", "B c"],
        "Rad": [0.9, 1.5],
        "f_Rad": [0, 1],
        "f_Mass": [1, 0],
        "Teff": [2566.0, 3400.0],
        "S": [1000.0, 500.0],
        "l_Mass": ["", ""],
    })
    ht.choose.catalog_menu(candidates)
    dropdown = shown[0]
    assert dropdown.options[1] == "1. A b: 0.9 Earth radii (measured), mass estimated, star 2566 K"
    assert dropdown.options[2] == "2. B c: 1.5 Earth radii (estimated), mass measured, star 3400 K"


def test_star_sliders_list_the_nearest_stars_and_keep_the_pick(shown, monkeypatch):
    summary = pd.DataFrame({
        "Name": ["HD 1", "HD 2", "HD 3", "HD 4", "HD 5", "HD 6"],
        "BV": [0.6, 0.75, 0.9, 1.2, 1.3, 1.4],
        "logRhk": [-4.9, -4.9, -4.5, -4.0, -4.0, -4.0],
        "variation": [1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        "Nobs": [50.0, 60.0, 70.0, 50.0, 50.0, 50.0],
    })
    choice = ht.choose.star_menu(summary)
    colour, activity, dropdown, _ = shown
    assert (colour.value, activity.value) == (ht.choose.START_COLOUR, ht.choose.START_ACTIVITY)
    assert dropdown.options[1].startswith("HD 2: colour 0.75")
    dropdown.value = dropdown.options[2]
    assert choice.name == "HD 1"
    colour.value = 1.3
    activity.value = -4.0
    assert dropdown.options[1].startswith("HD 5")
    assert dropdown.value is None  # moving the crosshair clears the menu
