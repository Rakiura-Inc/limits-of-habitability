"""Shared fixtures: no network, no plots on screen, and a clean notebook session for every test."""

import io
import urllib.request
import warnings

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pytest

with warnings.catch_warnings():  # lightkurve warns about an optional add-on habtrain never uses
    warnings.simplefilter("ignore", UserWarning)
    import habtrain as ht


class FakeResponse(io.BytesIO):
    """What urllib.request.urlopen gives back, holding `body`."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def place(fields: dict, width: int = 0) -> str:
    """One fixed-width line: fields maps (start, end) byte ranges to the text right-aligned there."""
    line = [" "] * max([width] + [end for _, end in fields])
    for (start, end), text in fields.items():
        line[start:end] = list(str(text).rjust(end - start))
    return "".join(line).rstrip()


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch, tmp_path):
    """Every test starts offline, with no saved answers and its own hand-in folder."""
    monkeypatch.setenv("RAKIURA_LIVE", "0")
    for name in ("RAKIURA_KEY", "RAKIURA_STEP", "HABTRAIN_STEPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ht.portal, "state", ht.portal._State(handin_dir=str(tmp_path / "handin")))
    stores = (ht.record._answers, ht.record._predictions, ht.record._reveals, ht.record._figures)
    for store in stores:
        store.clear()
    yield
    for store in stores:
        store.clear()
    plt.close("all")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any download attempt fails, as it would with no connection. Tests that need a reply replace this."""

    def refuse(*args, **kwargs):
        msg = "network disabled in tests"
        raise OSError(msg)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(urllib.request, "urlretrieve", refuse)


@pytest.fixture
def archive(monkeypatch):
    """Answer downloads from a dict {url fragment: bytes}; records every URL asked for in `.urls`."""

    class Archive(dict):
        urls: list

    replies = Archive()
    replies.urls = []

    def urlopen(url, data=None, timeout=None):
        replies.urls.append(url)
        for fragment, body in replies.items():
            if fragment in url:
                return FakeResponse(body)
        msg = f"no reply for {url}"
        raise OSError(msg)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return replies


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """An empty data cache; write files into it with cache.put(archive_path, text)."""
    folder = tmp_path / "data"
    monkeypatch.setattr(ht.data, "CACHE", str(folder))

    class Cache:
        path = folder

        @staticmethod
        def put(archive_path: str, text: str) -> None:
            folder.mkdir(exist_ok=True)
            (folder / archive_path.replace("/", "_").removesuffix(".gz")).write_text(text, encoding="latin-1")

    return Cache
