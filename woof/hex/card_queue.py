"""The host card queue's environment: which cards a queue granted to this command, and the queue's hooks.

A host card queue (a scheduler that leases whole or shared cards to commands) tells a granted command:

``CARD_QUEUE_CARDS``      the physical indices it granted, comma or space separated, beside
                          ``CUDA_VISIBLE_DEVICES`` (the same cards, in the same order)
``CARD_QUEUE_RELEASE``    a command, ``$CARD_QUEUE_RELEASE CARD``, that hands one granted card back early
``CARD_QUEUE_MODE``       ``pool`` when the card is shared and this command holds a share of it
``CARD_QUEUE_VRAM_MIB``   that share, in MiB
``CARD_QUEUE_SHARE``      the share's label, recorded in receipts
``CARD_QUEUE_GPU_DONE``   a file the leased command appends its pid to once it will not touch the card again
``CARD_QUEUE_LEASE_PID``  the queue's own pid (only the lease's own command may declare its GPU work done)

Outside such a queue none is set and hex runs on what ``CUDA_VISIBLE_DEVICES`` shows.

A site whose queue exports the same names under another prefix names that prefix once, in
``WOOF_CARD_QUEUE_PREFIX`` or in ``$XDG_CONFIG_HOME/gpuwm-hex/card-queue.json`` (``~/.config`` when unset) as
``{"env_prefix": "..."}``. Both spellings are then read, the standard one first, and an environment hex hands a
child process sets both. ``WOOF_CARD_QUEUE_PREFIX`` set to an empty value means the standard names only.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping, MutableMapping

PREFIX = "CARD_QUEUE_"
CARDS = "CARDS"
RELEASE = "RELEASE"
MODE = "MODE"
VRAM_MIB = "VRAM_MIB"
SHARE = "SHARE"
GPU_DONE = "GPU_DONE"
LEASE_PID = "LEASE_PID"
SUFFIXES = (CARDS, RELEASE, MODE, VRAM_MIB, SHARE, GPU_DONE, LEASE_PID)
PREFIX_ENV = "WOOF_CARD_QUEUE_PREFIX"
_VALID_PREFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@lru_cache(maxsize=8)
def _configured(path: str, stamp: float) -> str:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8")).get("env_prefix", "")
    except (OSError, ValueError, AttributeError):
        return ""
    return value if isinstance(value, str) else ""


def _site_prefix(environ: Mapping[str, str]) -> str:
    # Set (even empty) in the environment, it is the answer: empty means the standard names only.
    value = environ.get(PREFIX_ENV)
    if value is None:
        root = environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
        path = os.path.join(root, "woof hex", "card-queue.json")
        try:
            stamp = os.stat(path).st_mtime
        except OSError:
            return ""
        value = _configured(path, stamp)
    return value if value != PREFIX and _VALID_PREFIX.match(value or "") else ""


def names(suffix: str, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Every variable name that carries ``suffix``: the standard one first, then the site's, if it names one."""

    if suffix not in SUFFIXES:
        raise KeyError(f"no card queue variable {suffix!r}; known: {', '.join(SUFFIXES)}")
    environ = os.environ if environ is None else environ
    site = _site_prefix(environ)
    return (PREFIX + suffix,) + ((site + suffix,) if site else ())


def name(suffix: str) -> str:
    """The standard variable name for ``suffix``."""

    return names(suffix, {})[0]


def get(suffix: str, default: str = "", environ: Mapping[str, str] | None = None) -> str:
    """The value of the first of ``names(suffix)`` that is set, else ``default``."""

    environ = os.environ if environ is None else environ
    for key in names(suffix, environ):
        if key in environ:
            return environ[key]
    return default


def granted(environ: Mapping[str, str] | None = None) -> list[str]:
    """The granted physical card indices, in order (empty outside a queue)."""

    return get(CARDS, "", environ).replace(",", " ").split()


def put(env: MutableMapping[str, str], suffix: str, value: str) -> MutableMapping[str, str]:
    """Set ``suffix`` in ``env`` under every name it is read by; returns ``env``."""

    for key in names(suffix, env):
        env[key] = value
    return env


def drop(env: MutableMapping[str, str], suffix: str) -> MutableMapping[str, str]:
    """Remove ``suffix`` from ``env`` under every name it is read by; returns ``env``."""

    for key in names(suffix, env):
        env.pop(key, None)
    return env


def all_names(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Every card queue variable name, standard and site, for every suffix."""

    return tuple(key for suffix in SUFFIXES for key in names(suffix, environ))


__all__ = ["CARDS", "GPU_DONE", "LEASE_PID", "MODE", "PREFIX", "PREFIX_ENV", "RELEASE", "SHARE", "SUFFIXES",
           "VRAM_MIB", "all_names", "drop", "get", "granted", "name", "names", "put"]
