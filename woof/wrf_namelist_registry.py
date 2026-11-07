"""WRF v4.7.1's namelist vocabulary, read from the packaged table.

``woof/data/wrf_namelist/registry_v4.7.1.json`` is extracted from WRF's
own Registry by ``tools/wrf_namelist_registry.py`` (one row per ``rconfig
... namelist,<group>`` line the ARW build compiles).  Two consumers:

* the importer's unmapped-key refusal names the group WRF declares a
  misplaced key in (``gwd_opt`` has been a ``&dynamics`` key since WRF
  v4.0), so the refusal has one obvious repair;
* the namelist contract (:mod:`woof.namelist_contract`) checks every key
  WRF can read against the importer, so none reaches a user unclassified.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REGISTRY_TABLE = (Path(__file__).resolve().parent / "data" / "wrf_namelist"
                  / "registry_v4.7.1.json")


@lru_cache(maxsize=1)
def wrf_namelist_table() -> dict:
    """The packaged table, parsed once."""
    return json.loads(REGISTRY_TABLE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def wrf_namelist_keys() -> dict[tuple[str, str], dict]:
    """``{(group, key): row}`` for every WRF namelist key."""
    return {(row["section"], row["key"]): row
            for row in wrf_namelist_table()["keys"]}


def wrf_version() -> str:
    return str(wrf_namelist_table()["wrf"]["tag"])


def groups_declaring(key: str) -> tuple[str, ...]:
    """The namelist groups WRF declares ``key`` in (usually one)."""
    return tuple(sorted(section for section, name in wrf_namelist_keys()
                        if name == key))


def placement_note(section: str, key: str) -> str | None:
    """What WRF says about ``key`` in ``&section``, when it disagrees.

    ``None`` when WRF declares the key in that group (the importer's own
    reason is the whole answer), otherwise one sentence naming where WRF
    reads it, or that WRF reads it nowhere.
    """
    if (section, key) in wrf_namelist_keys():
        return None
    groups = groups_declaring(key)
    version = wrf_version()
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
