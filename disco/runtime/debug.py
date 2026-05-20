"""Toggleable debug logging for the runtime.

Enable with ``DISCO_DEBUG=1`` in the environment. Channels (comma-separated
in ``DISCO_DEBUG_CHANNELS``, default all) let you focus on one component:
``diar``, ``turn``, ``enrich``.
"""

import os
import sys
import time

_ENABLED = os.environ.get("DISCO_DEBUG") == "1"
_CHANNELS = set(
    (os.environ.get("DISCO_DEBUG_CHANNELS") or "diar,turn,enrich,tw").split(",")
)


def enabled(channel: str) -> bool:
    return _ENABLED and channel in _CHANNELS


def log(channel: str, *args) -> None:
    if not enabled(channel):
        return
    ts = time.strftime("%H:%M:%S")
    msg = " ".join(str(a) for a in args)
    print(f"[{ts} {channel}] {msg}", file=sys.stderr, flush=True)
