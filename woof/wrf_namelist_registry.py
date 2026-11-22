"""WRF's namelist vocabulary, read from the packaged tables.

``woof/data/wrf_namelist/registry_v<release>.json`` is extracted from
WRF's own Registry by ``tools/wrf_namelist_registry.py`` (one row per
``rconfig ... namelist,<group>`` line the ARW build compiles).  Two tables
ship: WRF v4.8.0, WOOF-ArWen's referee, and WRF v4.7.1, the previous pin
(:mod:`woof.wrf_reference`).  Every function takes the ``reference`` to
read; omitted, it is the one the running namelist import chose
(:func:`active_reference`), else :data:`WRF_REFERENCE_DEFAULT`.  Consumers:

* the importer's unmapped-key refusal names the group WRF declares a
  misplaced key in (``gwd_opt`` has been a ``&dynamics`` key since WRF
  v4.0), so the refusal has one obvious repair;
* the namelist contract (:mod:`woof.namelist_contract`) checks every key
  WRF can read against the importer, so none reaches a user unclassified.
"""
from __future__ import annotations

import contextvars
import json
from functools import lru_cache
from pathlib import Path

from woof.wrf_reference import (WRF_REFERENCE_DEFAULT, WRF_REFERENCES,
                                 check_wrf_reference)

_TABLE_DIR = Path(__file__).resolve().parent / "data" / "wrf_namelist"

#: The packaged table of each reference.
REGISTRY_TABLES = {reference: _TABLE_DIR / f"registry_v{reference}.json"
                   for reference in WRF_REFERENCES}
#: The default reference's table (kept for callers that name the path).
REGISTRY_TABLE = REGISTRY_TABLES[WRF_REFERENCE_DEFAULT]

#: The reference the namelist import in progress reads (set by
#: woof.namelist_import for the duration of one translation).
_ACTIVE_REFERENCE: contextvars.ContextVar = contextvars.ContextVar(
    "wrf_namelist_reference", default=None)


def active_reference() -> str:
    """The reference in force: the running import's, else the default."""
    return _ACTIVE_REFERENCE.get() or WRF_REFERENCE_DEFAULT


def using_reference(reference: str):
    """Context token: read ``reference`` until :func:`reset_reference`."""
    return _ACTIVE_REFERENCE.set(check_wrf_reference(reference))


def reset_reference(token) -> None:
    _ACTIVE_REFERENCE.reset(token)


@lru_cache(maxsize=None)
def _table(reference: str) -> dict:
    return json.loads(REGISTRY_TABLES[reference].read_text(encoding="utf-8"))


def wrf_namelist_table(reference: str | None = None) -> dict:
    """The packaged table, parsed once per reference."""
    return _table(check_wrf_reference(reference or active_reference()))


@lru_cache(maxsize=None)
def _keys(reference: str) -> dict[tuple[str, str], dict]:
    return {(row["section"], row["key"]): row
            for row in _table(reference)["keys"]}


def wrf_namelist_keys(reference: str | None = None
                      ) -> dict[tuple[str, str], dict]:
    """``{(group, key): row}`` for every WRF namelist key."""
    return _keys(check_wrf_reference(reference or active_reference()))


def wrf_version(reference: str | None = None) -> str:
    return str(wrf_namelist_table(reference)["wrf"]["tag"])


def groups_declaring(key: str, reference: str | None = None
                     ) -> tuple[str, ...]:
    """The namelist groups WRF declares ``key`` in (usually one)."""
    return tuple(sorted(section for section, name
                        in wrf_namelist_keys(reference) if name == key))


def placement_note(section: str, key: str,
                   reference: str | None = None) -> str | None:
    """What WRF says about ``key`` in ``&section``, when it disagrees.

    ``None`` when WRF declares the key in that group (the importer's own
    reason is the whole answer), otherwise one sentence naming where WRF
    reads it, or that WRF reads it nowhere.
    """
    if (section, key) in wrf_namelist_keys(reference):
        return None
    groups = groups_declaring(key, reference)
    version = wrf_version(reference)
    if not groups:
        return f"{key} is not a WRF {version} namelist key"
    where = " and ".join(f"&{group}" for group in groups)
    return (f"WRF {version} declares {key} in {where}, not in &{section}, "
            "and wrf.exe fails its own namelist read on it there")


# WRF 4.6.1 Registry.EM_COMMON:1940-1946 and module_physics_init.F:1197.
# Domain scalars use one-element device arrays in the restart field inventory.
TMN_UPDATE_STATE = {
    name: {"dimensions": dims, "dtype": dtype, "restart": True,
           "source": "WRF-4.6.1/Registry/Registry.EM_COMMON:1940-1946"}
    for name, dims, dtype in (
        ("tmn", ("y", "x"), "float32"),
        ("tlag", ("lagday", "y", "x"), "float32"),
        ("tyr", ("y", "x"), "float32"),
        ("tyra", ("y", "x"), "float32"),
        ("tdly", ("y", "x"), "float32"),
        ("nday", (), "float32"), ("nyear", (), "int32"))
}
