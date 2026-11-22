# SPDX-License-Identifier: Apache-2.0
"""The forecast route a recording runs (data/record-routes.v1.json).

hex picks some kernel forms by environment (the regional column solve in one kernel or four, the edge kernels hoisted
or not, the member-memory policy). The forms of one choice give the same bits, but the window has derivative rows for
one set of kernels only. ``derivative`` (the default) records on that set; ``forecast`` records the installed
forecast's own defaults. The route and the words it set are named in the plan's record-receipt.json.
THE BREAKAGE THIS PREVENTS: the door handed the recorder four of these words from its row table and the forecast picked
the rest itself, so a 12 km plan recorded through --init ran the four-kernel column solve, which has no derivative
row, and a recording made with the recorder directly ran whatever the caller's environment said (2026-11-18).
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import env

ROUTES_FILE = env.DATA / "record-routes.v1.json"


@lru_cache(maxsize=1)
def table() -> dict:
    return json.loads(ROUTES_FILE.read_text())


def names() -> tuple[str, ...]:
    return tuple(table()["routes"])


def default() -> str:
    return table()["default"]


def words(name: str) -> dict[str, str]:
    """The environment words route NAME sets in every forecast of the recording (the route's own, over the caller's)."""
    routes = table()["routes"]
    if name not in routes:
        env.defect(f"--route {name}: the routes are {', '.join(routes)}")
    return {k: str(v) for k, v in routes[name]["env"].items()}


def record(name: str) -> dict:
    """The receipt's line for route NAME."""
    return {"name": name, "what": table()["routes"][name]["what"], "env": words(name)}
