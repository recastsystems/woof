"""`woof hex` resolves its MPAS-physics routes on the host, with no GPU.

THE BREAKAGE: in recast-woof 1.1.1 `woof hex physics list`, `woof hex physics
resolve` and every MPAS-physics suite on `woof hex forecast` (the hex default)
stopped with "ModuleNotFoundError: No module named 'hexcore'".  The v8.4.2
route's default driver module, the RRFS route's driver and native seams in
data/profiles/mpas-physics-suites.v1.json, and a few other strings handed to
the import system still named the hex model's standalone package.

The list and resolve verbs import each route's driver exactly as the forecast
door does, so running them is the plan path of a forecast.  Every module the
suites table names is imported here too, and no string the package hands to
the import system may name the old package.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest

import woof

TABLE = "data/profiles/mpas-physics-suites.v1.json"
#: Keys of the suites table whose values are imported ("module" or "module:attribute").
IMPORT_KEYS = ("driver_module", "native_core_module", "module", "streams_parser", "native_registry_prepare",
               "native_layout_prepare", "native_table_selector", "native_table_stream_selector",
               "native_initialization_namelist")
PACKAGE = Path(woof.__file__).resolve().parent


def _table() -> dict:
    return json.loads(files("woof.hex").joinpath(TABLE).read_text(encoding="utf-8"))


def _named_modules() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    def walk(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                # (1.1.4) a value naming the package is an import whatever its key (hex 0.4.10 moved the route's
                # seams into rows called backend, glue, writer and so on, some of which also hold plain labels)
                if isinstance(value, str) and (key in IMPORT_KEYS or key.endswith("_module")
                                               or value.startswith(("woof.hex.", "hexcore."))):
                    found.append((f"{where}/{key}", value))
                walk(value, f"{where}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{where}/{index}")

    walk(_table(), "")
    return found


def _woof(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, WOOF_NO_LOCAL_GPU="1", GPUWM_NO_LOCAL_GPU="1", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "woof", *args], capture_output=True, text=True, env=env,
                          timeout=900)


def test_the_suites_table_names_the_route_modules() -> None:
    # Guards the parametrization below against passing on an empty set.
    keys = {where.rsplit("/", 1)[1] for where, _ in _named_modules()}
    assert {"driver_module", "native_core_module", "module"} <= keys, sorted(keys)


def _withheld() -> set[tuple[str, str, str]]:
    """(row, slot, scheme) whose port the public distribution withholds for its licence (woof.hex.licence_gate)."""
    gates = json.loads(files("woof.hex").joinpath("data/profiles/licence-gates.v1.json").read_text(encoding="utf-8"))
    held = {(item["row"], item["slot"], item["scheme"]) for item in gates["selections"]}
    # hex 0.4.10's 44c0849 row names the same withheld RRFS Noah-MP port as the rrfs row
    return held | {("mpasphys_rrfs_44c0849", "land_surface", "sf_noahmp")}


@pytest.mark.parametrize("where,value", _named_modules(), ids=lambda value: str(value))
def test_every_module_the_suites_table_names_imports(where: str, value: str) -> None:
    module, _, attribute = value.partition(":")
    parts = where.strip("/").split("/")
    if not module.startswith(("woof.", "hexcore.")):
        # A slot's CUDA entry names its module inside woof.hex.mpasphys (device_pools_v842, driver_v842).
        module = f"woof.hex.mpasphys.{module}"
    assert module.startswith("woof.hex."), f"{where} = {value!r} is not a module of this package"
    if len(parts) > 4 and parts[1] == "slots" and parts[3] == "cuda_entries" and \
            (parts[0], parts[2], parts[4]) in _withheld() and importlib.util.find_spec(module) is None:
        # Held from the public distribution for its licence; the licence gate refuses the selection by name.
        from woof.hex import licence_gate
        assert not licence_gate.port_present(module.removeprefix("woof.hex.mpasphys."))
        return
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name in ("cupy", "cupyx"):
            pytest.skip(f"{module} needs the CUDA extra (recast-woof[gpu-cu12] / [gpu-cu13]) to import")
        raise
    if attribute:
        assert callable(getattr(loaded, attribute, None)), f"{where} = {value!r}: {module} has no {attribute}"


def _mpas_routes() -> list[dict]:
    done = _woof("hex", "physics", "list")
    assert done.returncode == 0, done.stderr[-3000:]
    return [suite for suite in json.loads(done.stdout)["suites"] if suite.get("seam") == "mpas_suite"]


def test_physics_list_runs_every_mpas_route() -> None:
    routes = _mpas_routes()
    assert {suite["backend_row"] for suite in routes} >= {"mpasphys_v842", "mpasphys_rrfs"}, routes
    for suite in routes:
        assert suite["runs"], suite
        assert suite["driver_module"].startswith("woof.hex."), suite


def test_physics_resolve_runs_one_suite_of_each_mpas_route() -> None:
    first: dict[str, str] = {}
    for suite in _mpas_routes():
        first.setdefault(suite["backend_row"], suite["name"])
    for backend_row, name in sorted(first.items()):
        done = _woof("hex", "physics", "resolve", "--physics-suite", name)
        assert done.returncode == 0, (name, done.stderr[-3000:])
        resolved = json.loads(done.stdout)
        assert resolved["resolved"] and resolved["backend_row"] == backend_row, resolved
        importlib.import_module(resolved["driver_module"])


#: Calls whose first argument names a module or package for the import system.
RESOLVERS = {"import_module", "__import__", "find_spec", "spec_from_file_location", "files"}
#: (file, string): why it stays.  doctor asks whether the separate gpuwm-hex
#: distribution is installed; WOOF's own hex model is always present.
ALLOWED = {("doctor.py", "hexcore")}


def _old(node) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and (node.value == "hexcore" or node.value.startswith(("hexcore.", "hexcore:"))))


def test_no_string_the_import_system_resolves_names_the_old_package() -> None:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            hits = []
            if name in RESOLVERS and node.args:
                first = node.args[0]
                hits.append(first.left if isinstance(first, ast.BinOp) else first)
            if name in ("get", "getattr"):
                args = node.args[1:] if name == "getattr" else node.args
                if len(args) >= 2 and isinstance(args[0], ast.Constant) and str(args[0].value).endswith("module"):
                    hits.append(args[1])
            found += [f"{rel}:{hit.lineno}: {hit.value!r}" for hit in hits
                      if _old(hit) and (rel, hit.value) not in ALLOWED]
    assert not found, found
