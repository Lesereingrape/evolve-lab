"""Record what environment a published artifact was measured on.

The search here is pure Python: seeded ``random.Random`` draws and IEEE-754 double
arithmetic, with no tensor library and no threaded reduction behind it. That makes the
traces reproducible across machines in a way a torch study's are not, but the artifact
still records the interpreter and platform it was produced on so a reader can tell
which run produced which numbers.
"""

from __future__ import annotations

import platform
import sys


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": "cpu (stdlib float arithmetic; no BLAS or thread-count reduction)",
    }
