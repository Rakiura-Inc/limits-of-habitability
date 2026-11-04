"""Choose a planet or a star by looking at where it sits, not by typing its name.

Each chooser shows a live control under a map. Choosing redraws the map with your choice marked.
The first option always lets the notebook pick one at random. If the controls do not appear (they
need ipywidgets, which Colab has), the notebook falls back to a number you type.
"""

from __future__ import annotations

import numpy as np

RANDOM = "Pick one for me at random"
FULL_LABEL = {"description_width": "initial"}  # show a control's whole label, not a cut-off one

# Where the star map's crosshair starts: colour (B-V) and activity (log R'HK).
START_COLOUR = 0.75
START_ACTIVITY = -4.9


class Chooser:
    """Holds what was chosen. `value` is the chosen name, or None if nothing is chosen yet."""

    def __init__(self, names: list[str]):
        self.names = names
        self.value = None

    def set(self, name: str | None):
        self.value = name


def _widgets():
    """(ipywidgets, display, clear_output), or three Nones when ipywidgets is not installed."""
    try:
        import ipywidgets as w
        from IPython.display import clear_output, display
    except ImportError:
        return None, None, None
    return w, display, clear_output


def _random(names):
    return names[int(np.random.default_rng().integers(len(names)))]


def _number_of(names: list, name) -> int | str:
    """Where `name` sits in `names`, counting from 1, or "?" when it is not there."""
    return names.index(name) + 1 if name in names else "?"


def _dropdown(w, options: list, label: str):
    return w.Dropdown(options=options, value=None, description=label, layout=w.Layout(width="95%"), style=FULL_LABEL)


def menu(table, name_column: str, describe, map_fn, what: str = "Your choice:"):
    """A menu of the rows of `table` that marks your choice on a map.

    Each row is described by describe(number, row); the map is drawn by map_fn(table, mark=name).
    """
    import matplotlib.pyplot as plt

    t = table.reset_index(drop=True)
    names = list(t[name_column])
    chooser = Chooser(names)
    w, display, clear_output = _widgets()
    if w is None:
        print("The menu needs ipywidgets. Type your choice's number from the map in MY_PICK below instead.")
        return chooser
    dropdown = _dropdown(w, [RANDOM] + [describe(i + 1, r) for i, r in t.iterrows()], what)
    out = w.Output()

    def on_change(change):
        if change["new"] is None:
            return
        at_random = change["new"] == RANDOM
        name = _random(names) if at_random else names[dropdown.index - 1]
        chooser.set(name)
        with out:
            clear_output(wait=True)
            if at_random:
                print(f"The notebook picked {name} at random.")
            map_fn(t, mark=name)
            plt.show()

    dropdown.observe(on_change, names="value")
    display(dropdown, out)
    return chooser


def planet(planets, map_fn):
    """A menu of the survey planets, each described, that circles your choice on the planet map."""

    def describe(n, r):
        return (
            f"{n}. {r.planet}: {r.P_days:.1f}-day orbit, TESS magnitude {r.sy_tmag:.1f} (smaller is brighter), "
            f"{r.velocities} velocities ({r.telescopes})"
        )

    return menu(planets, "planet", describe, map_fn, "Your planet:")


def _describe_star(r) -> str:
    return (
        f"{r.Name}: colour {r.BV:.2f}, activity {r.logRhk:.2f}, varies {r.variation:.1f}%, {int(r.Nobs)} observations"
    )


def star(summary, map_fn, nearest_fn):
    """Two sliders that move a crosshair over the star map, and a menu of the five closest stars."""
    import matplotlib.pyplot as plt

    s = summary.reset_index(drop=True)
    names = list(s.Name)
    chooser = Chooser(names)
    w, display, clear_output = _widgets()
    if w is None:
        print("The sliders need ipywidgets. Use ht.data.nearest_stars(...) and type a number in MY_PICK below instead.")
        return chooser

    def slider(start, column, label):
        return w.FloatSlider(
            value=start,
            min=float(s[column].min()),
            max=float(s[column].max()),
            step=0.01,
            description=label,
            continuous_update=False,
            style=FULL_LABEL,
        )

    colour = slider(START_COLOUR, "BV", "Colour (B-V)")
    activity = slider(START_ACTIVITY, "logRhk", "Activity (log R'HK)")
    dropdown = _dropdown(w, [], "Your star:")
    out = w.Output()
    nearest = None  # the stars closest to the crosshair, once it has been drawn

    def redraw(*_):
        nonlocal nearest
        nearest = nearest_fn(s, colour.value, activity.value).reset_index(drop=True)
        dropdown.options = [RANDOM] + [_describe_star(r) for _, r in nearest.iterrows()]
        dropdown.value = None
        with out:
            clear_output(wait=True)
            fig = map_fn(s, mark={n: n for n in nearest.Name}, title="The five closest stars are circled")
            fig.axes[0].axvline(colour.value, color="#333", lw=0.8, ls=":")
            fig.axes[0].axhline(activity.value, color="#333", lw=0.8, ls=":")
            plt.show()

    def on_pick(change):
        if change["new"] is None or nearest is None:
            return
        if change["new"] == RANDOM:
            name = _random(names)
            with out:
                print(f"The notebook picked {name} at random.")
        else:
            name = nearest.Name.iloc[dropdown.index - 1]
        chooser.set(name)

    colour.observe(redraw, names="value")
    activity.observe(redraw, names="value")
    dropdown.observe(on_pick, names="value")
    display(colour, activity, dropdown, out)
    redraw()
    return chooser


def resolve(chooser: Chooser, typed, table, name_column: str, what: str) -> str:
    """The name to use: what you chose in the control, else the number you typed, else a random pick."""
    names = list(table.reset_index(drop=True)[name_column])
    if chooser is not None and chooser.value is not None:
        n = _number_of(names, chooser.value)
        print(f"Your {what}: number {n}, {chooser.value}. If you come back in a new session, put {n} in MY_PICK.")
        return chooser.value
    if typed is not ... and typed is not None:
        i = int(typed)
        if 1 <= i <= len(names):
            print(f"Your {what}: number {i}, {names[i - 1]}.")
            return names[i - 1]
        print(f"{typed} is not a number from 1 to {len(names)}.")
    name = _random(names)
    n = _number_of(names, name)
    print(
        f"Nothing chosen yet, so the notebook picked number {n}, {name}, at random. "
        f"Put {n} in MY_PICK to keep it, or choose above and run this cell again."
    )
    return name
