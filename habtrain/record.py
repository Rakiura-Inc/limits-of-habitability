"""Record answers as you go, and gather them into the hand-in.

Each answer is saved the moment its cell runs, with the time. A prediction saved before its
reveal is marked as made first. `handin(part, ...)` writes your answers, figures and the commit
you ran into handin/<part>/, and ticks that part on Rakiura when you are connected.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess

from . import __version__, portal

_answers: dict[str, dict] = {}
_predictions: dict[str, dict] = {}
_reveals: dict[str, str] = {}


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def filled(value) -> bool:
    """False for an answer left as ..., None or an empty string. A dict counts only when every
    entry is filled; a table or array you built always counts."""
    if hasattr(value, "shape"):
        return True
    if value is ... or value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in ("", "...")
    if isinstance(value, dict):
        return bool(value) and all(filled(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def _show(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_show(v)}" for k, v in value.items())
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def save(name: str, value, what: str) -> None:
    """Save one answer. `what` says in plain words what it is; it labels the hand-in table."""
    if not filled(value):
        print(f"Not saved yet: fill in {what} above (replace the ...), then run this cell again.")
        return
    _answers[name] = {"what": what, "value": value, "at": _now()}
    print(f"Saved: {what} = {_show(value)}")


def predict(question: str, value) -> None:
    """Save a prediction with the time, before its reveal."""
    if not filled(value):
        print("Make your prediction above first (replace the ...), then run this cell again.")
        return
    _predictions[question] = {"value": value, "at": _now()}
    print(f"Prediction saved at {_predictions[question]['at']}. Now run the next cell.")


def revealed(question: str) -> bool:
    """True when the reveal may run. Otherwise prints one line and returns False."""
    if question not in _predictions:
        print("Make your prediction in the cell above first.")
        return False
    _reveals.setdefault(question, _now())
    return True


def fill(value, default, name: str):
    """`value` if you filled it in; otherwise `default`, with one line saying so."""
    if filled(value):
        return value
    print(f"{name} is still empty, so the cells below use {default if isinstance(default, str) else 'a stand-in'} "
          "to show how they work. Fill it in above and run again to use your own.")
    return default


def numbered(df, columns: list | None = None):
    """Show a table with its rows numbered from 1, so you can pick one by its number."""
    t = (df[columns] if columns else df).reset_index(drop=True)
    t.index = range(1, len(t) + 1)
    t.index.name = "#"
    try:
        from IPython.display import display
        display(t)
    except Exception:
        print(t.to_string())
    return t


def pick(choice, table, name_column: str, what: str):
    """The row you chose by its number (counting from 1), or a random row if you left the choice blank.

    Returns that row. A random pick is printed, so you can write its number in to keep it next time.
    """
    import numpy as np
    t = table.reset_index(drop=True)
    if not filled(choice):
        i = int(np.random.default_rng().integers(len(t)))
        print(f"You left {what} blank, so the notebook picked number {i + 1} at random: {t.loc[i, name_column]}. "
              f"Write {i + 1} in above to keep it if you run this again.")
        return t.loc[i]
    if not isinstance(choice, (int, np.integer)) or not 1 <= int(choice) <= len(t):
        print(f"Pick a whole number from 1 to {len(t)}. Using number 1 for now.")
        choice = 1
    row = t.loc[int(choice) - 1]
    print(f"Your {what}: number {int(choice)}, {row[name_column]}.")
    return row


def show(df, columns: list | None = None):
    """Display a table you filtered, or one line if it is still empty."""
    if not hasattr(df, "shape"):
        print("Nothing to show yet: fill in the ... above and run the cell again.")
        return
    try:
        from IPython.display import display
        display(df[columns] if columns else df)
    except Exception:
        print((df[columns] if columns else df).to_string())


def pkg_commit() -> str:
    """The commit this package was installed from (pip's direct_url.json), or the checkout's HEAD."""
    try:
        from importlib.metadata import distribution
        info = json.loads(distribution("habtrain").read_text("direct_url.json") or "{}")
        sha = (info.get("vcs_info") or {}).get("commit_id")
        if sha:
            return sha[:12]
    except Exception:
        pass
    try:
        here = os.path.dirname(__file__)
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()[:12]
    except Exception:
        pass
    return f"v{__version__}"


def table(names: list[str]) -> str:
    """Your saved answers for `names`, as a two-column table (markdown)."""
    lines = ["| Question | Your answer |", "|---|---|"]
    for n in names:
        a = _answers.get(n)
        lines.append(f"| {a['what'] if a else n} | {_show(a['value']) if a else '(not answered)'} |")
    return "\n".join(lines)


def handin(part: str, names: list[str], figures: dict | None = None, numbers: dict | None = None) -> None:
    """Write the hand-in for one part and tick it on Rakiura when connected.

    names    the answers that belong to this part, in the order they should appear
    figures  {"file-name": figure} saved as PNG beside the answers
    numbers  up to 20 named single values sent with the tick
    """
    out = os.path.join(portal.state.handin_dir, part)
    os.makedirs(out, exist_ok=True)
    figs = []
    for fname, fig in (figures or {}).items():
        path = os.path.join(out, f"{fname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        figs.append(fig)
    record = {
        "part": part, "saved_at": _now(), "package_commit": pkg_commit(),
        "answers": {n: _answers[n] for n in names if n in _answers},
        "missing": [n for n in names if n not in _answers],
        "predictions": {q: dict(p, made_before_reveal=(q not in _reveals or p["at"] <= _reveals[q]))
                        for q, p in _predictions.items()},
    }
    with open(os.path.join(out, "answers.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    md = table(names)
    with open(os.path.join(out, "answers.md"), "w") as fh:
        fh.write(md + "\n")
    try:
        from IPython.display import Markdown, display
        display(Markdown(md))
    except Exception:
        print(md)
    if record["missing"]:
        print(f"{len(record['missing'])} question(s) still unanswered. You can hand in anyway, or answer them and run this cell again.")
    portal.done(part, figures=figs, numbers=numbers)
    portal.stage_file(json.dumps(record, indent=1, default=str), f"{part} answers.json")
    files = ", ".join(["answers.md (the table above)", "answers.json"] + [f"{f}.png" for f in (figures or {})])
    # Say exactly what to hand in and where the files are, connected or not.
    print(f"Saved in the folder {os.path.relpath(out)}/ : {files}.\n"
          "To hand in: open this step in Rakiura, press Hand in, and attach these files or paste the table. "
          "In Colab the folder is in the Files panel on the left; right-click a file to download it.\n"
          "If you connected this notebook to Rakiura, your plots are already on the step. Attach answers.md or paste the table, then press Hand in.")
