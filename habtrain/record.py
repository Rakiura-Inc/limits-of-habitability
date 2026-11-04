"""Record answers as you go, and gather them into the hand-in.

Each answer is saved the moment its cell runs, with the time. A prediction saved before its
reveal is marked as made first. `handin(part, ...)` writes your answers, figures and the commit
you ran into handin/<part>/, and ticks that part on Rakiura when you are connected.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import numpy as np

from . import portal

_answers: dict[str, dict] = {}
_predictions: dict[str, dict] = {}
_reveals: dict[str, str] = {}

SHORT_SHA = 12  # characters of a commit id to keep


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _display(obj) -> None:
    """Show a table or markdown in the notebook, or print it as text outside one."""
    try:
        from IPython.display import display
    except ImportError:
        print(obj.to_string() if hasattr(obj, "to_string") else obj)
        return
    display(obj)


def filled(value) -> bool:
    """False for an answer left as ..., None or an empty string.

    A dict counts only when every entry is filled; a table or array you built always counts.
    """
    if hasattr(value, "shape"):
        return True
    if value is ... or value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "..."}
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
    stand_in = default if isinstance(default, str) else "a stand-in"
    print(
        f"{name} is still empty, so the cells below use {stand_in} to show how they work. "
        "Fill it in above and run again to use your own."
    )
    return default


def numbered(df, columns: list | None = None):
    """Show a table with its rows numbered from 1, so you can pick one by its number."""
    t = (df[columns] if columns else df).reset_index(drop=True)
    t.index = range(1, len(t) + 1)
    t.index.name = "#"
    _display(t)
    return t


def pick(choice, table, name_column: str, what: str):
    """The row you chose by its number (counting from 1), or a random row if you left the choice blank.

    Returns that row. A random pick is printed, so you can write its number in to keep it next time.
    """
    t = table.reset_index(drop=True)
    if not filled(choice):
        i = int(np.random.default_rng().integers(len(t)))
        print(
            f"You left {what} blank, so the notebook picked number {i + 1} at random: {t.loc[i, name_column]}. "
            f"Write {i + 1} in above to keep it if you run this again."
        )
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
    _display(df[columns] if columns else df)


def _installed_commit() -> str | None:
    """The commit pip installed this package from (its direct_url.json), if it was installed from git."""
    try:
        info = json.loads(distribution("habtrain").read_text("direct_url.json") or "{}")
    except (PackageNotFoundError, ValueError):
        return None
    sha = (info.get("vcs_info") or {}).get("commit_id")
    return sha[:SHORT_SHA] if sha else None


def _checkout_commit() -> str | None:
    """The HEAD commit of the git checkout this package sits in, if it sits in one."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH, like the member's own shell
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()[:SHORT_SHA] if out.returncode == 0 else None


def pkg_commit() -> str:
    """The commit this package was installed from (pip's direct_url.json), or the checkout's HEAD."""
    from . import __version__

    return _installed_commit() or _checkout_commit() or f"v{__version__}"


def table(names: list[str]) -> str:
    """Your saved answers for `names`, as a two-column table (markdown)."""
    lines = ["| Question | Your answer |", "|---|---|"]
    for n in names:
        a = _answers.get(n)
        lines.append(f"| {a['what'] if a else n} | {_show(a['value']) if a else '(not answered)'} |")
    return "\n".join(lines)


def _made_before_reveal(question: str, prediction: dict) -> bool:
    return question not in _reveals or prediction["at"] <= _reveals[question]


def _record(part: str, names: list[str]) -> dict:
    """Everything one part's hand-in holds: the answers, what is missing, and every prediction."""
    return {
        "part": part,
        "saved_at": _now(),
        "package_commit": pkg_commit(),
        "answers": {n: _answers[n] for n in names if n in _answers},
        "missing": [n for n in names if n not in _answers],
        "predictions": {q: dict(p, made_before_reveal=_made_before_reveal(q, p)) for q, p in _predictions.items()},
    }


HANDIN_HOWTO = (
    "To hand in: open this step in Rakiura, press Hand in, and attach these files or paste the table. "
    "In Colab the folder is in the Files panel on the left; right-click a file to download it.\n"
    "If you connected this notebook to Rakiura, your plots are already on the step. "
    "Attach results.md or paste the table, then press Hand in."
)


def handin(part: str, names: list[str], figures: dict | None = None, numbers: dict | None = None) -> None:
    """Write the hand-in for one part and tick it on Rakiura when connected.

    names    the answers that belong to this part, in the order they should appear
    figures  {"file-name": figure} saved as PNG beside the answers
    numbers  up to 20 named single values sent with the tick
    """
    out = Path(portal.state.handin_dir) / part
    out.mkdir(parents=True, exist_ok=True)
    figures = figures or {}
    for fname, fig in figures.items():
        fig.savefig(out / f"{fname}.png", dpi=150, bbox_inches="tight")
    record = _record(part, names)
    (out / "results.json").write_text(json.dumps(record, indent=2, default=str))
    md = table(names)
    (out / "results.md").write_text(md + "\n")
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print(md)
    else:
        display(Markdown(md))
    if record["missing"]:
        print(
            f"{len(record['missing'])} question(s) still unanswered. "
            "You can hand in anyway, or answer them and run this cell again."
        )
    portal.done(part, figures=list(figures.values()), numbers=numbers)
    portal.stage_file(json.dumps(record, indent=1, default=str), f"{part} results.json")
    files = ", ".join(["results.md (the table above)", "results.json"] + [f"{f}.png" for f in figures])
    # Say exactly what to hand in and where the files are, connected or not.
    print(f"Saved in the folder {os.path.relpath(out)}/ : {files}.\n{HANDIN_HOWTO}")
