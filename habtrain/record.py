"""Record answers as you go, and gather them into the hand-in.

Each answer is saved the moment its cell runs, with the time. A prediction saved before its
reveal is marked as made first. A cell that draws a figure for the hand-in keeps it with
`keep_figure`. `handin(part, ...)` writes your answers, the kept figures and the commit you ran
into handin/<part>/, and ticks that part on Rakiura when you are connected.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from . import portal

_answers: dict[str, dict] = {}
_predictions: dict[str, dict] = {}
_reveals: dict[str, str] = {}
_figures: dict[str, dict] = {}  # for each hand-in part, its figures by file name

SHORT_SHA = 12  # characters of a commit id to keep


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _display(obj) -> None:
    """Show a table or markdown in the notebook, or print it as text outside one."""
    try:
        from IPython.display import display
    except ImportError:
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)
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
    if columns:
        df = df[columns]
    numbered_table = df.reset_index(drop=True)
    numbered_table.index = range(1, len(numbered_table) + 1)
    numbered_table.index.name = "#"
    _display(numbered_table)
    return numbered_table


def answer(name: str, missing: str):
    """The answer saved as `name` in an earlier part. If there is none, stops with the message `missing`."""
    if name not in _answers:
        raise RuntimeError(missing)
    return _answers[name]["value"]


def keep_figure(part: str, name: str, figure) -> None:
    """Keep a figure for one part's hand-in, where it is saved as <name>.png."""
    _figures.setdefault(part, {})[name] = figure


def show(df, columns: list | None = None):
    """Display a table you filtered, or one line if it is still empty."""
    if not hasattr(df, "shape"):
        print("Nothing to show yet: fill in the ... above and run the cell again.")
        return
    if columns:
        df = df[columns]
    _display(df)


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
    for name in names:
        saved = _answers.get(name)
        if saved:
            lines.append(f"| {saved['what']} | {_show(saved['value'])} |")
        else:
            lines.append(f"| {name} | (not answered) |")
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


def handin(part: str, names: list[str], numbers: dict | None = None) -> None:
    """Write the hand-in for one part and tick it on Rakiura when connected.

    names    the answers that belong to this part, in the order they should appear
    numbers  up to 20 named single values sent with the tick

    Every figure kept for this part with keep_figure() is saved as PNG beside the answers.
    """
    out = Path(portal.state.handin_dir) / part
    out.mkdir(parents=True, exist_ok=True)
    kept = _figures.get(part, {})
    for name, figure in kept.items():
        figure.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
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
    portal.done(part, figures=list(kept.values()), numbers=numbers)
    portal.stage_file(json.dumps(record, indent=1, default=str), f"{part} results.json")
    files = ", ".join(["results.md (the table above)", "results.json"] + [f"{name}.png" for name in kept])
    # Say exactly what to hand in and where the files are, connected or not.
    print(f"Saved in the folder {os.path.relpath(out)}/ : {files}.\n{HANDIN_HOWTO}")
