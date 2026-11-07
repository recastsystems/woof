"""Remove only the three named additions from historical TOML controls.

Authored imports used automatic HAILCAST -1. Native namelist imports now
retain WRF's omitted-selector default 0. Each control accepts exactly one
of those lines; every other byte still has to match its historical hash.
"""
from __future__ import annotations

CAMPAIGN_IMPORT_LINES = (
    "ra_sw_eclipse = 1\n",
    "haildt = 60.0\n",
)
HAILCAST_IMPORT_LINES = ("hailcast_opt = -1\n", "hailcast_opt = 0\n")


def without_campaign_import_lines(text: str) -> str:
    for line in CAMPAIGN_IMPORT_LINES:
        assert text.count(line) == 1, line
        text = text.replace(line, "", 1)
    assert sum(text.count(line) for line in HAILCAST_IMPORT_LINES) == 1
    for line in HAILCAST_IMPORT_LINES:
        if line in text:
            text = text.replace(line, "", 1)
    return text
