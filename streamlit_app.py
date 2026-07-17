"""Streamlit Community Cloud entry point for the Magnetor dashboard.

Streamlit Cloud runs this file. It bootstraps configuration from ``st.secrets``
into environment variables *before* importing ``magnetor`` — ``magnetor.config``
resolves ``MAGNETOR_DATA_ROOT`` at import time, so the order matters — then
points the data root at the read-only snapshot bundled in the repo and launches
the dashboard.

Locally this file also works: with no secrets configured, the search is open and
the data root falls back to the bundled snapshot.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parent

# Promote Streamlit secrets to env vars before magnetor is imported.
for _key in (
    "MAGNETOR_VOYAGE_API_KEY",
    "MAGNETOR_S2_API_KEY",
    "MAGNETOR_SEARCH_PASSWORD",
    "MAGNETOR_DATA_ROOT",
):
    with contextlib.suppress(Exception):  # no secrets file locally -> fine
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])

# Default to the bundled snapshot shipped in the repo (hosted mode has no
# %LOCALAPPDATA%). An explicit MAGNETOR_DATA_ROOT still wins.
os.environ.setdefault("MAGNETOR_DATA_ROOT", str(_ROOT / "sample_data"))

sys.path.insert(0, str(_ROOT / "src"))

from magnetor.dashboard import main  # noqa: E402

# Streamlit re-executes this entry script on every interaction, so main() runs
# each rerun. (dashboard.py guards its own main() call to __name__=="__main__",
# so importing it above does not render anything by itself.)
main()
