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

import datetime as _dt
import json
import os
import re

DEFAULT_REPO = "Rakiura-Inc/limits-of-habitability"
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
MAX_NUMBERS = 20


class _State:
    helper = None          # the real `rakiura` module when live
    mode = "offline"
    notebook: str | None = None
    handin_dir = os.path.abspath(os.environ.get("HABTRAIN_HANDIN", "handin"))
    log: list = []
    last_part: str | None = None


state = _State()


_HELPER_CALLS = ("start", "done", "stage_text", "submit")


def _import_helper():
    """The Rakiura helper when it is installed, else None (any other package named rakiura is ignored)."""
    try:
        import rakiura  # type: ignore
    except Exception:
        return None
    return rakiura if all(callable(getattr(rakiura, name, None)) for name in _HELPER_CALLS) else None


def _key_without_click(helper) -> bool:
    if os.environ.get("RAKIURA_KEY"):
        return True
    try:
        if helper._colab_secret():
            return True
    except Exception:
        pass
    try:
        path = helper._credentials_path()
    except Exception:
        path = os.path.join(os.path.expanduser("~"), ".config", "rakiura", "credentials")
    return os.path.exists(path)


def steps(notebook: str | None = None) -> dict:
    """Step ids for staging files, from $HABTRAIN_STEPS or notebooks/rakiura_steps.json.

    Returns {"project": ..., "step": ...} for one notebook (by file name), or the whole file.
    $RAKIURA_STEP overrides the step for the current notebook.
    """
    cands = [os.environ.get("HABTRAIN_STEPS"), "rakiura_steps.json",
             os.path.join("notebooks", "rakiura_steps.json"),
             os.path.join(os.path.dirname(__file__), "..", "notebooks", "rakiura_steps.json")]
    conf: dict = {}
    for c in filter(None, cands):
        if os.path.exists(c):
            with open(c) as fh:
                conf = json.load(fh)
            break
    if notebook is None:
        return conf
    entry = dict(conf.get("notebooks", {}).get(os.path.basename(notebook), {}))
    if os.environ.get("RAKIURA_STEP"):
        entry["step"] = os.environ["RAKIURA_STEP"]
    return entry


NOT_CONNECTED = "Not connected to Rakiura. Your work stays on this computer."


def connect(notebook: str, repo: str = DEFAULT_REPO, handin_dir: str | None = None,
            sign_in: bool = False) -> str:
    """Start this notebook's session. Returns "live" or "offline". Never raises.

    By default it connects only when that needs no click (see the module notes), so the setup cell
    never waits. ``sign_in=True`` (the Connect cell) also runs the helper's sign-in: it prints a
    link, the member checks the code and clicks Allow. RAKIURA_LIVE=0 keeps both offline.
    """
    try:
        return _connect(notebook, repo, handin_dir, sign_in)
    except Exception:  # never stop a notebook over the connection
        state.helper, state.mode = None, "offline"
        print(NOT_CONNECTED)
        return "offline"


def _connect(notebook, repo, handin_dir, sign_in):
    state.notebook = notebook
    if handin_dir:
        state.handin_dir = os.path.abspath(handin_dir)
    os.makedirs(state.handin_dir, exist_ok=True)
    flag = os.environ.get("RAKIURA_LIVE")
    helper = None if flag == "0" else _import_helper()
    if helper is not None and (sign_in or flag == "1" or _key_without_click(helper)):
        try:
            # Quiet unless a sign-in link may be needed: the member reads one plain line below.
            if helper.start(notebook=notebook, repo=repo, quiet=not (sign_in or flag == "1")):
                state.helper, state.mode = helper, "live"
                name = _member_name(helper)
                print(f"Connected to Rakiura as {name}." if name else "Connected to Rakiura.")
                return "live"
        except Exception:
            pass
        finally:
            _speak(helper)
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
    try:
        helper._session.quiet = False
    except Exception:
        pass


def _member_name(helper) -> str | None:
    """The signed-in member's name, asked quietly; None if it cannot be had. Never the key."""
    try:
        helper._session.quiet = True
        info = helper.me(show=False) or {}
        name = (info.get("member") or {}).get("name")
        return name if isinstance(name, str) and name.strip() else None
    except Exception:
        return None
    finally:
        _speak(helper)


def go_live(notebook: str | None = None, repo: str = DEFAULT_REPO) -> str:
    """Sign in now (one click) and switch this session to live. Same as the Connect cell."""
    return connect(notebook or state.notebook or "notebook.ipynb", repo, sign_in=True)


def _log_path() -> str:
    return os.path.join(state.handin_dir, "rakiura_progress.json")


def _clean_numbers(numbers: dict | None):
    clean = {}
    for name, value in (numbers or {}).items():
        value = value.item() if hasattr(value, "item") else value
        if value is None:
            value = "blank"
        if not isinstance(value, (bool, int, float, str)):
            return None
        clean[str(name)] = value[:200] if isinstance(value, str) else value
    return clean if len(clean) <= MAX_NUMBERS else None


def _offline_log(entry: dict) -> None:
    state.log.append({**entry, "at": _dt.datetime.now().isoformat(timespec="seconds")})
    os.makedirs(state.handin_dir, exist_ok=True)
    with open(_log_path(), "w") as fh:
        json.dump(state.log, fh, indent=2)


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


# What a member reads when a part is ticked. The part name itself stays internal (the portal keys on it).
_SAVED = {"picked": "Part 1 saved", "weighed": "Part 2 saved", "found": "Hand-in saved",
          "fitted": "Hand-in saved"}


def saved_words(part: str) -> str:
    return _SAVED.get(part.rsplit("-", 1)[-1], "Progress saved")


def stage_file(text: str, title: str, mime: str = "text/plain") -> bool:
    """Add a small text file (up to 50 KB) to this notebook's step, when a step id is configured."""
    step = steps(state.notebook or "").get("step")
    if state.mode == "live" and step:
        return bool(state.helper.stage_text(text, title, step=step, mime=mime))
    if state.mode == "live":
        return False            # no step to attach to: the member attaches the files on the step page
    _offline_log({"staged": title, "bytes": len(text.encode()), "step": step})
    return True


def submit(note: str | None = None) -> bool:
    """Send the step for review. Only ever called by the member, explicitly."""
    if state.mode == "live":
        return bool(state.helper.submit(note=note))
    print("Offline mode: nothing was sent. When you are connected to Rakiura, this sends your step for review.")
    return False
