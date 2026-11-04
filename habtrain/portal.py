"""Bridge to the Rakiura notebook helper (`rakiura`), with a local offline mode.

The helper is installed by each notebook's setup cell:

    %pip install -q "rakiura @ git+https://github.com/Rakiura-Inc/rakiura-py@main"

When it is installed and the member is signed in, the calls below go to the portal exactly as
the helper documents:

    rakiura.start(notebook=..., repo=...)   opens a run (notebook path, repository, commit, where)
    rakiura.done(part, figures, numbers)    ticks the checklist item the PI linked to that part
    rakiura.stage_text(text, title, step)   adds a small file (your answers as JSON) to a step
    rakiura.submit(note=...)                sends the step for review, only when the member asks

Which step a part counts toward is set in Rakiura (Settings > Notebook links), keyed by the
notebook path and the part name; the notebooks carry no Rakiura ids. Staging a file needs a step
id, which is configurable (see `steps()` and `notebooks/rakiura_steps.json`).

Without the helper, or when it is not signed in, offline mode keeps the same calls in
`handin/rakiura_progress.json` with the helper's own validation, and nothing leaves the computer.

Modes (environment variable RAKIURA_LIVE):
    unset  live if the helper is installed and a key is available without a click
           (RAKIURA_KEY, Colab's Secrets panel, or a key saved by an earlier sign-in); else offline.
           connect(..., sign_in=True), the notebooks' Connect cell, also signs in with one click.
    "1"    live; signs in with one click if needed (prints a link)
    "0"    offline, always (even for sign_in=True)
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REPO = "Rakiura-Inc/limits-of-habitability"
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
MAX_NUMBERS = 20
MAX_TEXT = 200  # characters kept of a text value sent with a tick
_HELPER_CALLS = ("start", "done", "stage_text", "submit")
STEPS_FILE = "rakiura_steps.json"

NOT_CONNECTED = "Not connected to Rakiura. Your work stays on this computer."

# What a member reads when a part is ticked. The part name itself stays internal (the portal keys on it).
_SAVED = {"picked": "Part 1 saved", "weighed": "Part 2 saved", "found": "Hand-in saved", "fitted": "Hand-in saved"}


def _absolute(path: str) -> str:
    """The path made absolute without following links, so it reads as the member typed it."""
    return os.path.abspath(path)  # noqa: PTH100 - Path.resolve() would follow links


def _default_handin_dir() -> str:
    return _absolute(os.environ.get("HABTRAIN_HANDIN", "handin"))


@dataclass
class _State:
    """This notebook session's connection: the helper when live, and the offline log."""

    helper: Any = None  # the real `rakiura` module when live
    mode: str = "offline"
    notebook: str | None = None
    handin_dir: str = field(default_factory=_default_handin_dir)
    log: list = field(default_factory=list)
    last_part: str | None = None


state = _State()


def _import_helper():
    """The Rakiura helper when it is installed, else None (any other package named rakiura is ignored)."""
    try:
        import rakiura
    except Exception:  # noqa: BLE001 - a broken or missing helper only means offline mode
        return None
    return rakiura if all(callable(getattr(rakiura, name, None)) for name in _HELPER_CALLS) else None


def _key_without_click(helper) -> bool:
    """True when the helper can sign in with no click: a key in the environment, Colab's Secrets, or saved."""
    if os.environ.get("RAKIURA_KEY"):
        return True
    with contextlib.suppress(Exception):  # the helper's private calls may change; then skip that check
        if helper._colab_secret():
            return True
    try:
        path = helper._credentials_path()
    except Exception:  # noqa: BLE001 - fall back to where the helper keeps the key by default
        path = Path.home() / ".config" / "rakiura" / "credentials"
    return Path(path).exists()


def steps(notebook: str | None = None) -> dict:
    """Step ids for staging files, from $HABTRAIN_STEPS or notebooks/rakiura_steps.json.

    Returns {"project": ..., "step": ...} for one notebook (by file name), or the whole file.
    $RAKIURA_STEP overrides the step for the current notebook.
    """
    candidates = [
        os.environ.get("HABTRAIN_STEPS"),
        STEPS_FILE,
        str(Path("notebooks") / STEPS_FILE),
        str(Path(__file__).parent / ".." / "notebooks" / STEPS_FILE),
    ]
    conf: dict = {}
    for candidate in filter(None, candidates):
        if Path(candidate).exists():
            conf = json.loads(Path(candidate).read_text())
            break
    if notebook is None:
        return conf
    entry = dict(conf.get("notebooks", {}).get(Path(notebook).name, {}))
    if os.environ.get("RAKIURA_STEP"):
        entry["step"] = os.environ["RAKIURA_STEP"]
    return entry


def connect(notebook: str, repo: str = DEFAULT_REPO, handin_dir: str | None = None, sign_in: bool = False) -> str:
    """Start this notebook's session. Returns "live" or "offline". Never raises.

    By default it connects only when that needs no click (see the module notes), so the setup cell
    never waits. ``sign_in=True`` (the Connect cell) also runs the helper's sign-in: it prints a
    link, the member checks the code and clicks Allow. RAKIURA_LIVE=0 keeps both offline.
    """
    try:
        return _connect(notebook, repo, handin_dir, sign_in)
    except Exception:  # noqa: BLE001 - never stop a notebook over the connection
        state.helper, state.mode = None, "offline"
        print(NOT_CONNECTED)
        return "offline"


def _start_live(helper, notebook: str, repo: str, loud: bool) -> bool:
    """Open a live run with the helper. Quiet unless a sign-in link may be needed."""
    try:
        if helper.start(notebook=notebook, repo=repo, quiet=not loud):
            state.helper, state.mode = helper, "live"
            name = _member_name(helper)
            print(f"Connected to Rakiura as {name}." if name else "Connected to Rakiura.")
            return True
    except Exception:  # noqa: BLE001 - a failed start leaves the notebook offline
        return False
    finally:
        _speak(helper)
    return False


def _connect(notebook, repo, handin_dir, sign_in):
    state.notebook = notebook
    if handin_dir:
        state.handin_dir = _absolute(handin_dir)
    Path(state.handin_dir).mkdir(parents=True, exist_ok=True)
    flag = os.environ.get("RAKIURA_LIVE")
    helper = None if flag == "0" else _import_helper()
    loud = sign_in or flag == "1"
    if helper is not None and (loud or _key_without_click(helper)) and _start_live(helper, notebook, repo, loud):
        return "live"
    state.helper, state.mode = None, "offline"
    if sign_in and helper is None and flag != "0":
        print(NOT_CONNECTED + " The Rakiura helper is not installed: run the setup cell above, then this cell again.")
    elif sign_in:
        print(NOT_CONNECTED + " Run this cell again to try once more, or attach your files on the step page.")
    else:
        print(NOT_CONNECTED + " Run the Connect cell below to tick steps on Rakiura.")
    return "offline"


def _speak(helper) -> None:
    """Let the helper print its own lines again (ticks, staged files) after a quiet start."""
    with contextlib.suppress(Exception):  # an older helper has no session to unmute
        helper._session.quiet = False


def _member_name(helper) -> str | None:
    """The signed-in member's name, asked quietly; None if it cannot be had. Never the key."""
    try:
        helper._session.quiet = True
        info = helper.me(show=False) or {}
        name = (info.get("member") or {}).get("name")
        return name if isinstance(name, str) and name.strip() else None
    except Exception:  # noqa: BLE001 - the greeting is optional
        return None
    finally:
        _speak(helper)


def go_live(notebook: str | None = None, repo: str = DEFAULT_REPO) -> str:
    """Sign in now (one click) and switch this session to live. Same as the Connect cell."""
    return connect(notebook or state.notebook or "notebook.ipynb", repo, sign_in=True)


def _log_path() -> Path:
    return Path(state.handin_dir) / "rakiura_progress.json"


def _clean_numbers(numbers: dict | None) -> dict | None:
    """The numbers as plain named values, or None when one is not a single value or there are too many."""
    clean = {}
    for name, value in (numbers or {}).items():
        plain = value.item() if hasattr(value, "item") else value
        if plain is None:
            plain = "blank"
        if not isinstance(plain, (bool, int, float, str)):
            return None
        clean[str(name)] = plain[:MAX_TEXT] if isinstance(plain, str) else plain
    return clean if len(clean) <= MAX_NUMBERS else None


def _offline_log(entry: dict) -> None:
    state.log.append({**entry, "at": dt.datetime.now().isoformat(timespec="seconds")})
    Path(state.handin_dir).mkdir(parents=True, exist_ok=True)
    _log_path().write_text(json.dumps(state.log, indent=2))


def done(part: str, figures: list | None = None, numbers: dict | None = None) -> bool:
    """Tick a part. Live: rakiura.done(...). Offline: one line in the local log."""
    if not isinstance(part, str) or not _PART.match(part):
        print('rakiura: a part name is letters, digits and . _ : - (for example "part-2").')
        return False
    clean = _clean_numbers(numbers)
    if clean is None:
        print(f"rakiura: numbers takes up to {MAX_NUMBERS} named single values, never a table or an array.")
        return False
    figures = list(figures or [])
    if any(not getattr(f, "axes", None) for f in figures):
        print("rakiura: that figure is empty; pass the figure you drew on (fig, ax = plt.subplots()).")
        return False
    state.last_part = part
    if state.mode == "live":
        return bool(state.helper.done(part, figures=figures or None, numbers=clean or None))
    _offline_log({"part": part, "figures": len(figures), "numbers": clean})
    return True


def saved_words(part: str) -> str:
    """What a member reads when `part` is ticked, from the last word of the part name."""
    return _SAVED.get(part.rsplit("-", 1)[-1], "Progress saved")


def stage_file(text: str, title: str, mime: str = "text/plain") -> bool:
    """Add a small text file (up to 50 KB) to this notebook's step, when a step id is configured."""
    step = steps(state.notebook or "").get("step")
    if state.mode == "live" and step:
        return bool(state.helper.stage_text(text, title, step=step, mime=mime))
    if state.mode == "live":
        return False  # no step to attach to: the member attaches the files on the step page
    _offline_log({"staged": title, "bytes": len(text.encode()), "step": step})
    return True


def submit(note: str | None = None) -> bool:
    """Send the step for review. Only ever called by the member, explicitly."""
    if state.mode == "live":
        return bool(state.helper.submit(note=note))
    print("Offline mode: nothing was sent. When you are connected to Rakiura, this sends your step for review.")
    return False
