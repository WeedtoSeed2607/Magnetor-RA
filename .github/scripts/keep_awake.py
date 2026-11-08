"""Visit the hosted app so Streamlit Community Cloud does not put it to sleep.

A scheduled ``curl`` does not work here, and fails quietly, which is worse than
failing. A plain GET returns Streamlit's static HTML shell with HTTP 200 while
the Python process never starts: the server-side session only begins when the
browser's JavaScript opens a WebSocket to ``/_stcore/stream``. So the check has
to be a real browser, and the proof has to be content the Python app rendered -
not a status code.

A sleeping app serves a page with a wake button, which only a browser can press.

**The app is not in the top-level document.** Community Cloud renders it in an
iframe at ``/~/+/``; the outer page is a wrapper holding Cloud's own toolbar and
a status-page embed, and its body text is empty. Measured on this deployment:

    [0] .../                      body_text_len=0     <- wrapper
    [1] statuspage.io/embed/frame body_text_len=133
    [2] .../~/+/                  body_text_len=2890  <- the app

So every check below searches across frames rather than the top-level document.
Searching all frames, instead of hard-coding ``/~/+/``, keeps this working if
Cloud changes that path or drops the wrapper entirely.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import time

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Frame, Page, sync_playwright

APP_URL = os.environ.get(
    "APP_URL", "https://magnetor-ra-ldljxqct3v7amar5hgtpss.streamlit.app/"
)

#: Text the Python app renders on its default page. Waiting for this - rather
#: than for the page to load - is what distinguishes a live session from the
#: static shell that a sleeping app also serves.
PROOF_OF_LIFE = "Topic-Trend Banner"

WAKE_BUTTON = re.compile("get this app back up", re.IGNORECASE)

#: Waking a sleeping app reinstalls dependencies, which is slow and variable.
WAKE_TIMEOUT_S = 300.0
LOAD_TIMEOUT_MS = 90_000
POLL_S = 3.0

_CONTAINS = "needle => (document.body ? document.body.innerText : '').includes(needle)"


def _frame_showing(page: Page, needle: str) -> Frame | None:
    """The frame whose rendered text contains ``needle``, if any."""
    for frame in page.frames:
        try:
            if frame.evaluate(_CONTAINS, needle):
                return frame
        except PlaywrightError:
            # A frame can detach mid-poll; that is not a failure to report.
            continue
    return None


def _press_wake_button(page: Page) -> bool:
    for frame in page.frames:
        try:
            button = frame.get_by_role("button", name=WAKE_BUTTON)
            if button.count():
                button.first.click()
                return True
        except PlaywrightError:
            continue
    return False


def main() -> int:
    started = time.monotonic()
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)

            if _press_wake_button(page):
                print("app was asleep - pressed the wake button", flush=True)
            else:
                print("no wake button - app was already up", flush=True)

            deadline = time.monotonic() + WAKE_TIMEOUT_S
            while time.monotonic() < deadline:
                frame = _frame_showing(page, PROOF_OF_LIFE)
                if frame is not None:
                    elapsed = time.monotonic() - started
                    print(
                        f"awake: {PROOF_OF_LIFE!r} rendered in {frame.url} "
                        f"after {elapsed:.1f}s",
                        flush=True,
                    )
                    return 0
                # The wake screen can appear after the first load.
                _press_wake_button(page)
                page.wait_for_timeout(int(POLL_S * 1000))

            # Fail loudly. A silent keep-alive that stopped working is
            # indistinguishable from one that never ran, and the first person to
            # find out would otherwise be whoever was sent the link.
            print(
                f"FAILED: no frame rendered {PROOF_OF_LIFE!r} within "
                f"{WAKE_TIMEOUT_S:.0f}s. Frames seen: "
                f"{[f.url for f in page.frames]}",
                file=sys.stderr,
            )
            page.screenshot(path="keep-awake-failure.png", full_page=True)
            return 1
        except PlaywrightError as failure:
            print(f"FAILED to reach the app: {failure}", file=sys.stderr)
            with contextlib.suppress(PlaywrightError):
                page.screenshot(path="keep-awake-failure.png", full_page=True)
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
