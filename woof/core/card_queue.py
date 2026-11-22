"""The host card queue's environment: which cards a queue granted to this command.

A host card queue (a scheduler that leases whole or shared cards to commands)
tells a granted command:

``CARD_QUEUE_CARDS``   the physical indices it granted, comma or space
                       separated, beside ``CUDA_VISIBLE_DEVICES`` (the same
                       cards, in the same order)
``CARD_QUEUE_DIR``     the folder holding the queue's per-card owner ledgers
                       (``OWNER-<card>.json``, schema ``<queue>.owner.v1``)

Outside such a queue neither is set and WOOF-ArWen runs on what
``CUDA_VISIBLE_DEVICES`` shows.

A site whose queue exports the same names under another prefix names that
prefix once, in ``WOOF_CARD_QUEUE_PREFIX`` or in
``$XDG_CONFIG_HOME/woof/card-queue.json`` (``~/.config`` when unset; the
hex engine's ``woof hex/card-queue.json`` is read too, so one site file
serves both engines) as ``{"env_prefix": "..."}``.  The standard name is read
first.  ``WOOF_CARD_QUEUE_PREFIX`` set to an empty value means the standard
names only.  These are the same names and the same site setting as hex.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping

PREFIX = "CARD_QUEUE_"
PREFIX_ENV = "WOOF_CARD_QUEUE_PREFIX"
_VALID_PREFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _site_prefix(environ: Mapping[str, str]) -> str:
    value = environ.get(PREFIX_ENV)
    if value is None:
        root = environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config")
        for folder in ("woof", "woof hex"):
            try:
                loaded = json.loads(Path(root, folder, "card-queue.json")
                                    .read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            candidate = loaded.get("env_prefix") if isinstance(loaded, dict) else None
            if isinstance(candidate, str):
                value = candidate
                break
    value = (value or "").strip()
    if not value or value == PREFIX or not _VALID_PREFIX.match(value):
        return ""
    return value


def queue_value(suffix: str, environ: Mapping[str, str] | None = None) -> str | None:
    """The queue's ``suffix`` entry (``CARDS``, ``DIR``, ...), or None outside a queue."""
    env = os.environ if environ is None else environ
    standard = env.get(PREFIX + suffix)
    if standard is not None:
        return standard
    site = _site_prefix(env)
    return env.get(site + suffix) if site else None


def granted_cards(environ: Mapping[str, str] | None = None) -> list[str]:
    """The physical card indices the queue granted, in CUDA order ([] outside a queue)."""
    raw = queue_value("CARDS", environ) or ""
    return [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
