"""Put every archive file the notebooks read into notebooks/data/, and check each against its record.

CI runs this before the notebooks, so a file that changed fails the job here, with its name, instead
of quietly changing a notebook's numbers. A file already in notebooks/data/ (from CI's cache, or an
earlier run) is checked the same way as a new download.

    uv run python scripts/fetch_archive_files.py
"""

import hashlib
import sys
from pathlib import Path

import habtrain as ht

NOTEBOOK_DATA = Path(__file__).parent.parent / "notebooks" / "data"


def main() -> int:
    ht.data.CACHE = str(NOTEBOOK_DATA)  # where the notebooks look, since they run from notebooks/
    changed = []
    for path, expected in ht.data.SHA256.items():
        actual = hashlib.sha256(ht.data.fetch(path)).hexdigest()
        if actual != expected:
            changed.append(f"{path}: sha256 {actual}, recorded {expected}")
    if changed:
        print("These archive files differ from habtrain/archive_files.csv:", *changed, sep="\n  ")
        return 1
    print(f"All {len(ht.data.SHA256)} archive files match habtrain/archive_files.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
