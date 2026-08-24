"""Public entry point for the naming experiment — and nothing else.

Deployed so a remote reader can be sent a link. It is a *separate* app from the
dashboard on purpose, because the dashboard cannot safely be shared: it can spawn
harvest processes on the host machine and it holds an embedding API key. Neither
is needed to read titles and write a sentence, so neither is here.

**What this app can reach.** One bundle file — cluster membership and paper
titles. No corpus, no vector index, no graph artifacts, no ``.env``, no API key,
no ability to start a process. If the bundle is absent it says so and stops.

**Where answers go.** A local file, which on a hosted platform is temporary and
disappears when the app restarts. So the reader is given their own answers to
download and send back, and the operator merges them with
``magnetor name import``. That is safe because the log is append-only:
merging is concatenation, and :func:`magnetor.naming.import_lines` drops anything
already present so a reader who sends twice does not count twice.

**There is no sign-in, and readers name themselves.** Anyone with the link can
write. That is tolerable only because nothing here is authoritative: the operator
chooses which returned file to import, so an unexpected name is ignored rather
than defended against. Do not add anything to this app that would make that
untrue.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent

# ``requirements.txt`` deliberately does not pip-install this project, so the
# package has to be found on disk — the same bootstrap ``streamlit_app.py`` does.
# Without it a hosted deploy fails at import with ModuleNotFoundError, which is
# not a failure any local run reproduces: the dev venv has an editable install.
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# Pin storage beside this file *before* magnetor is imported, so the default
# per-user location is never consulted. On a hosted platform that default would
# resolve to whatever home directory the container happens to have.
os.environ.setdefault("MAGNETOR_DATA_ROOT", str(_ROOT / ".naming_data"))

import streamlit as st  # noqa: E402

from magnetor.naming import (  # noqa: E402
    export_lines,
    import_lines,
    read_bundle,
)
from magnetor.naming_ui import (  # noqa: E402
    INTRO,
    SUBTITLE,
    judge_step,
    result_step,
    write_step,
)

#: Shipped beside this file so the deployment is one repository and no secrets.
BUNDLE = _ROOT / "naming_bundle.json"


def main() -> None:
    st.set_page_config(page_title="Name the groups", layout="centered")
    st.title("Name the groups")
    st.markdown(INTRO)
    st.caption(SUBTITLE)

    if not BUNDLE.exists():
        st.error(
            "No groups have been prepared for this app yet. The operator needs to "
            "run `magnetor name bundle` and deploy the file it writes."
        )
        return
    naming_set, titles, seed = read_bundle(BUNDLE)
    if seed:
        # Hosted storage is per-container and temporary, so the other readers'
        # names cannot have survived here. Re-seeding on every load is safe:
        # import_lines drops records already present.
        import_lines(naming_set.set_id, seed)

    who = st.text_input(
        "Your name",
        help="Only used to keep your answers apart from the other reader's.",
    ).strip()
    if not who:
        st.warning("Type your name to start. Use the same name each time you return.")
        return

    step = st.radio(
        "Step",
        ("1 · Write the names", "2 · Judge the names", "3 · Result"),
        horizontal=True,
    )
    st.divider()
    if step.startswith("1"):
        write_step(naming_set, titles, who)
    elif step.startswith("2"):
        judge_step(naming_set, titles, who)
    else:
        result_step(naming_set)

    _send_back(naming_set.set_id, who)


def _send_back(set_id: str, who: str) -> None:
    """Let the reader carry their own work home.

    Hosted storage is temporary, so this is not a convenience — without it a
    restart loses the afternoon someone gave you.
    """
    lines = export_lines(set_id, who)
    if not lines:
        return
    st.divider()
    st.caption(
        "When you have finished, download this and send it back. Your answers are "
        "not stored permanently here."
    )
    st.download_button(
        "Download my answers",
        data=lines,
        file_name=f"naming-{set_id}-{who}.jsonl",
        mime="application/x-ndjson",
    )


# Streamlit executes the module top to bottom; there is no __main__ guard because
# `streamlit run` imports rather than executes as a script.
main()
