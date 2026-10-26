"""Choose a planet or a star by looking at where it sits, not by typing its name.

Each chooser shows a live control under a map. Choosing redraws the map with your choice marked.
The first option always lets the notebook pick one at random. If the controls do not appear (they
need ipywidgets, which Colab has), the notebook falls back to a number you type.
"""
from __future__ import annotations

import numpy as np

RANDOM = "Pick one for me at random"


class Chooser:
    """Holds what was chosen. `value` is the chosen name, or None if nothing is chosen yet."""

    def __init__(self, names: list[str]):
        self.names = names
        self.value = None

    def set(self, name: str | None):
        self.value = name


def _widgets():
    try:
        import ipywidgets as w
        from IPython.display import display, clear_output
        return w, display, clear_output
    except Exception:
        return None, None, None


def _random(names):
    return names[int(np.random.default_rng().integers(len(names)))]


def menu(table, name_column: str, describe, map_fn, what: str = "Your choice:"):
    """A menu of the rows of `table`, each described by describe(number, row), that marks your
    choice on the map drawn by map_fn(table, mark=name)."""
    import matplotlib.pyplot as plt
    t = table.reset_index(drop=True)
    names = list(t[name_column])
    ch = Chooser(names)
    w, display, clear_output = _widgets()
    if w is None:
        print("The menu needs ipywidgets. Type your choice's number from the map in MY_PICK below instead.")
        return ch
    dd = w.Dropdown(options=[RANDOM] + [describe(i + 1, r) for i, r in t.iterrows()], value=None,
                    description=what, layout=w.Layout(width="95%"), style={"description_width": "initial"})
    out = w.Output()

    def on_change(change):
        if change["new"] is None:
            return
        name = _random(names) if change["new"] == RANDOM else names[dd.index - 1]
        ch.set(name)
        with out:
            clear_output(wait=True)
            if change["new"] == RANDOM:
                print(f"The notebook picked {name} at random.")
            map_fn(t, mark=name)
            plt.show()

    dd.observe(on_change, names="value")
    display(dd, out)
    return ch


def planet(planets, map_fn):
    """A menu of the survey planets, each described, that circles your choice on the planet map."""
    def describe(n, r):
        return (f"{n}. {r.planet}: {r.P_days:.1f}-day orbit, TESS magnitude {r.sy_tmag:.1f} (smaller is brighter), "
                f"{r.velocities} velocities ({r.telescopes})")
    return menu(planets, "planet", describe, map_fn, "Your planet:")


def star(summary, map_fn, nearest_fn):
    """Two sliders that move a crosshair over the star map, and a menu of the five closest stars."""
    import matplotlib.pyplot as plt
    s = summary.reset_index(drop=True)
    ch = Chooser(list(s.Name))
    w, display, clear_output = _widgets()
    if w is None:
        print("The sliders need ipywidgets. Use ht.data.nearest_stars(...) and type a number in MY_PICK below instead.")
        return ch
    colour = w.FloatSlider(value=0.75, min=float(s.BV.min()), max=float(s.BV.max()), step=0.01,
                           description="Colour (B-V)", continuous_update=False, style={"description_width": "initial"})
    activity = w.FloatSlider(value=-4.9, min=float(s.logRhk.min()), max=float(s.logRhk.max()), step=0.01,
                             description="Activity (log R'HK)", continuous_update=False,
                             style={"description_width": "initial"})
    menu = w.Dropdown(options=[], value=None, description="Your star:", layout=w.Layout(width="95%"),
                      style={"description_width": "initial"})
    out = w.Output()
    near = {"t": None}

    def redraw(*_):
        near["t"] = nearest_fn(s, colour.value, activity.value).reset_index(drop=True)
        opts = [RANDOM] + [f"{r.Name}: colour {r.BV:.2f}, activity {r.logRhk:.2f}, varies {r.variation:.1f}%, "
                           f"{int(r.Nobs)} observations" for _, r in near["t"].iterrows()]
        menu.options = opts
        menu.value = None
        with out:
            clear_output(wait=True)
            fig = map_fn(s, mark={n: n for n in near["t"].Name}, title="The five closest stars are circled")
            fig.axes[0].axvline(colour.value, color="#333", lw=0.8, ls=":")
            fig.axes[0].axhline(activity.value, color="#333", lw=0.8, ls=":")
            plt.show()

    def on_pick(change):
        if change["new"] is None or near["t"] is None:
            return
        if change["new"] == RANDOM:
            name = _random(list(s.Name))
            with out:
                print(f"The notebook picked {name} at random.")
        else:
            name = near["t"].Name.iloc[menu.index - 1]
        ch.set(name)

    colour.observe(redraw, names="value")
    activity.observe(redraw, names="value")
    menu.observe(on_pick, names="value")
    display(colour, activity, menu, out)
    redraw()
    return ch


def resolve(chooser: Chooser, typed, table, name_column: str, what: str) -> str:
    """The name to use: what you chose in the control, else the number you typed, else a random pick."""
    t = table.reset_index(drop=True)
    if chooser is not None and chooser.value is not None:
        n = list(t[name_column]).index(chooser.value) + 1 if chooser.value in list(t[name_column]) else "?"
        print(f"Your {what}: number {n}, {chooser.value}. If you come back in a new session, put {n} in MY_PICK.")
        return chooser.value
    if typed is not ... and typed is not None:
        i = int(typed)
        if 1 <= i <= len(t):
            name = t[name_column].iloc[i - 1]
            print(f"Your {what}: number {i}, {name}.")
            return name
        print(f"{typed} is not a number from 1 to {len(t)}.")
    name = _random(list(t[name_column]))
    n = list(t[name_column]).index(name) + 1
    print(f"Nothing chosen yet, so the notebook picked number {n}, {name}, at random. "
          f"Put {n} in MY_PICK to keep it, or choose above and run this cell again.")
    return name
