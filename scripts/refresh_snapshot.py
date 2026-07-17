"""Refresh the bundled dashboard snapshot from your live local data.

The hosted dashboard reads a read-only copy of the corpus committed to the repo
under ``sample_data/`` (records, vectors, trends). Run this after acquiring /
embedding / recomputing trends locally to update what the deployed app shows:

    python scripts/refresh_snapshot.py
    git add sample_data && git commit -m "refresh snapshot" && git push

It copies from your resolved MAGNETOR_DATA_ROOT (default %LOCALAPPDATA%\\Magnetor
\\data) into ./sample_data. Nothing here contains secrets — only public paper
metadata and vectors.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from magnetor.config import DATA_ROOT

_DEST = Path(__file__).resolve().parent.parent / "sample_data"


def main() -> int:
    src = Path(DATA_ROOT)
    if not src.exists():
        print(f"error: data root not found: {src}", file=sys.stderr)
        return 1
    if _DEST.exists():
        shutil.rmtree(_DEST)
    shutil.copytree(src, _DEST)
    n = sum(1 for _ in _DEST.rglob("*") if _.is_file())
    print(f"snapshot refreshed: {src} -> {_DEST} ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
