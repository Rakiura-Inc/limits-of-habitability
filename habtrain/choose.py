"""Choose a planet or a star by where it sits on a map, not by typing its name.

Each menu sits under a map, and choosing redraws the map with your choice marked. The first option
lets the notebook pick one for you. If the menus do not appear (they need ipywidgets, which Colab
has), the notebook uses the number you type in MY_PICK instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from . import plot
from .data import nearest_stars

RANDOM = "Pick one for me"
FULL_LABEL = {"description_width": "initial"}  # show a control's whole label, not a cut-off one

# Fixed seeds, so a pick made for the member is the same every run and their numbers match the
# worked answers. The bootcamp and the TESS project use different ones, so a member who leaves both
# menus alone still gets two different planets: with planets.csv as of 2026-11-04, TOI-1173 b in
# notebooks 3 and 4, and TOI-1601 b in tess_01 and tess_02.
RANDOM_PICK_SEED = 1
BOOTCAMP_PICK_SEED = 38
TESS_PICK_SEED = 23

# Where the star map's crosshair starts: colour (B-V) and activity (log R'HK).
START_COLOUR = 0.75
START_ACTIVITY = -4.9


@dataclass
class Choice:
    """What was chosen in a menu."""

    name: str | None = None  # set when the member picks from the menu
    pick_seed: int = RANDOM_PICK_SEED  # what "Pick one for me" uses, in the menu and in resolve()


def _widgets():
    """(ipywidgets, display, clear_output), or three Nones when ipywidgets is not installed."""
    try:
        import ipywidgets
        from IPython.display import clear_output, display
    except ImportError:
        return None, None, None
    return ipywidgets, display, clear_output


def random_pick(names: list[str], seed: int = RANDOM_PICK_SEED) -> str:
    """One of `names`, picked with `seed`."""
    rng = np.random.default_rng(seed)
    return names[int(rng.integers(len(names)))]


def _dropdown(widgets, options: list[str], label: str):
    return widgets.Dropdown(
        options=options, value=None, description=label, layout=widgets.Layout(width="95%"), style=FULL_LABEL
    )


def _menu(table, name_column: str, *, descriptions: list[str], draw_map, label: str, pick_seed: int) -> Choice:
    """A menu of the rows of `table`, one description each, that redraws the map with the choice marked.

    `draw_map(table, chosen_name)` draws the map.
    """
    names = list(table[name_column])
    choice = Choice(pick_seed=pick_seed)
    widgets, display, clear_output = _widgets()
    if widgets is None:
        print("The menu needs ipywidgets. Type your choice's number from the map in MY_PICK below instead.")
        return choice
    dropdown = _dropdown(widgets, [RANDOM, *descriptions], label)
    output = widgets.Output()

    def on_change(change):
        if change["new"] is None:
            return
        if change["new"] == RANDOM:
            choice.name = random_pick(names, choice.pick_seed)
        else:
            choice.name = names[dropdown.index - 1]  # option 0 is the random pick
        with output:
            clear_output(wait=True)
            if change["new"] == RANDOM:
                print(f"The notebook picked {choice.name} for you.")
            draw_map(table, choice.name)
            plt.show()

    dropdown.observe(on_change, names="value")
    display(dropdown, output)
    return choice


def describe_catalog_planet(number: int, row) -> str:
    """One menu line for a habitable zone catalog planet, saying which numbers are measured."""
    radius_source = "estimated" if row.f_Rad else "measured"
    mass_source = "estimated" if row.f_Mass else "measured"
    return f"{number}. {row.Planet}: {row.Rad} Earth radii ({radius_source}), mass {mass_source}, star {row.Teff:.0f} K"


def _draw_catalog_choice(table, chosen_name: str):
    return plot.numbered_catalog(table, title="Your choice, circled", highlight=chosen_name)


def catalog_menu(candidates) -> Choice:
    """A menu of the catalog planets in `candidates`, numbered as on the map, that circles your choice."""
    table = candidates.reset_index(drop=True)
    descriptions = []
    for number, row in enumerate(table.itertuples(), start=1):
        descriptions.append(describe_catalog_planet(number, row))
    return _menu(
        table,
        "Planet",
        descriptions=descriptions,
        draw_map=_draw_catalog_choice,
        label="Your planet:",
        pick_seed=RANDOM_PICK_SEED,
    )


def describe_survey_planet(number: int, row) -> str:
    """One menu line for a TESS-Keck Survey planet."""
    return (
        f"{number}. {row.planet}: {row.P_days:.1f}-day orbit, TESS magnitude {row.sy_tmag:.1f} (smaller is brighter), "
        f"{row.velocities} velocities ({row.telescopes})"
    )


def planet_menu(planets, pick_seed: int = RANDOM_PICK_SEED) -> Choice:
    """A menu of the survey planets, numbered as on the planet map, that circles your choice.

    `pick_seed` sets which planet "Pick one for me" gives (BOOTCAMP_PICK_SEED or TESS_PICK_SEED).
    """
    table = planets.reset_index(drop=True)
    descriptions = []
    for number, row in enumerate(table.itertuples(), start=1):
        descriptions.append(describe_survey_planet(number, row))
    return _menu(
        table,
        "planet",
        descriptions=descriptions,
        draw_map=plot.planet_map,
        label="Your planet:",
        pick_seed=pick_seed,
    )


def describe_star(row) -> str:
    """One menu line for a star on the activity map."""
    return (
        f"{row.Name}: colour {row.BV:.2f}, activity {row.logRhk:.2f}, varies {row.variation:.1f}%, "
        f"{int(row.Nobs)} observations"
    )


def _slider(widgets, summary, *, column: str, start: float, label: str):
    return widgets.FloatSlider(
        value=start,
        min=float(summary[column].min()),
        max=float(summary[column].max()),
        step=0.01,
        description=label,
        continuous_update=False,
        style=FULL_LABEL,
    )


def star_menu(summary) -> Choice:
    """Two sliders that move a crosshair over the star map, and a menu of the five closest stars."""
    table = summary.reset_index(drop=True)
    choice = Choice()
    widgets, display, clear_output = _widgets()
    if widgets is None:
        print("The sliders need ipywidgets. Use ht.data.nearest_stars(...) and type a number in MY_PICK below instead.")
        return choice
    colour = _slider(widgets, table, column="BV", start=START_COLOUR, label="Colour (B-V)")
    activity = _slider(widgets, table, column="logRhk", start=START_ACTIVITY, label="Activity (log R'HK)")
    dropdown = _dropdown(widgets, [], "Your star:")
    output = widgets.Output()
    closest = None  # the five stars closest to the crosshair, set by redraw() before the menu shows

    def redraw(*_):
        nonlocal closest
        closest = nearest_stars(table, colour.value, activity.value).reset_index(drop=True)
        dropdown.options = [RANDOM] + [describe_star(row) for row in closest.itertuples()]
        dropdown.value = None
        with output:
            clear_output(wait=True)
            circled = {name: name for name in closest.Name}
            fig = plot.star_map(table, mark=circled, title="The five closest stars are circled")
            fig.axes[0].axvline(colour.value, color="#333", lw=0.8, ls=":")
            fig.axes[0].axhline(activity.value, color="#333", lw=0.8, ls=":")
            plt.show()

    def on_pick(change):
        if change["new"] is None:
            return
        if change["new"] == RANDOM:
            choice.name = random_pick(list(table.Name))
            with output:
                print(f"The notebook picked {choice.name} for you.")
        else:
            choice.name = closest.Name.iloc[dropdown.index - 1]  # option 0 is the random pick

    colour.observe(redraw, names="value")
    activity.observe(redraw, names="value")
    dropdown.observe(on_pick, names="value")
    display(colour, activity, dropdown, output)
    redraw()
    return choice


def _typed_number(typed) -> int | None:
    """The whole number the member typed in MY_PICK, or None if they left it as ... or blank."""
    if typed is ... or typed is None:
        return None
    return int(typed)


def resolve(choice: Choice, typed, *, table, name_column: str, what: str) -> str:
    """The name to use: what you chose in the menu, else the number you typed, else one picked for you."""
    names = list(table.reset_index(drop=True)[name_column])
    if choice.name is not None:
        print(f"Your {what}: {choice.name}.")
        if choice.name in names:
            number = names.index(choice.name) + 1
            print(f"If you come back in a new session, put {number} in MY_PICK.")
        return choice.name

    number = _typed_number(typed)
    if number is not None:
        if 1 <= number <= len(names):
            print(f"Your {what}: number {number}, {names[number - 1]}.")
            return names[number - 1]
        print(f"{typed} is not a number from 1 to {len(names)}.")

    name = random_pick(names, choice.pick_seed)
    number = names.index(name) + 1
    print(
        f"Nothing chosen yet, so the notebook picked number {number}, {name}, for you. "
        f"Put {number} in MY_PICK to keep it, or choose above and run this cell again."
    )
    return name
