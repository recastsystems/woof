"""The KIM drag's orographic statistics (MPAS-Model PR #1408) for any hex mesh: ``<mesh>.gwd-kim2026.nc``.

THE BREAKAGE THIS PREVENTS: the KIM drag (``config_gwdo_scheme='bl_kim_gwdo'``, MPAS PR #1409; selected by
name with ``gwdo_version=kim2026``, never the default) blocks the flow below the maximum sub-grid orography ``elvmax`` and
gives no drag at all where ``elvmax <= 50 m`` (bl_gwdo.F90:416).  No init a hex door writes carries
``elvmax`` (MPAS-A v8.4.2's init_atmosphere has none), so without this file every cell is no-drag and the
run looks like a drag run while carrying none.  The forecast refuses a KIM run with no ``elvmax``
(kim_gwdo_2026.gwdo_from_MPAS); this module makes the field so it never has to.

The file is ``rw_mpas_static --gwd-only --gwd-recipe kim2026`` on the run's mesh: MPAS PR #1408's whole
recipe (elvmax plus its revised var2d, con, oa1-4 and ol1-4, cell scaling 2.0), proven 0 of 28,182 words
different from native init_atmosphere v8.4.2 + PR #1408 on x1.2562.  The forecast reads
it after the init (hydration takes the last source that carries a field), so the KIM drag sees PR #1408's
statistics the way MPAS with both PRs would; a run on any other drag never reads it.

It is a product of the mesh and the 30-arc-second terrain and land use alone, so it is built once per mesh
(multi-core Rust), kept in the hex cache under the mesh's topology digest with a receipt, and reused; a file
``<mesh>.gwd-kim2026.nc`` beside the mesh file with a matching receipt is used too.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

SCHEMA = "gpuwm-hex.kim-gwd-static/v1"
RECIPE = "kim2026"
FILE_SUFFIX = ".gwd-kim2026.nc"
FIELDS = ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4", "elvmax")


class KimGwdStaticError(RuntimeError):
    """The statistics could not be built; the message names what is missing."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_digest(mesh_file: str | Path) -> str:
    """The mesh's topology digest (latCell, lonCell, nEdgesOnCell, cellsOnCell): the same for a mesh's grid,
    static and init, so a door that builds on the grid and a forecast that names only the init share one file."""
    try:
        import numpy as np
        from woof.netcdf_bridge import open_dataset
        dataset = open_dataset(Path(mesh_file))
        digest = hashlib.sha256(b"mpas-topology-v1")
        for name, dtype in (("latCell", "<f8"), ("lonCell", "<f8"), ("nEdgesOnCell", "<i4"), ("cellsOnCell", "<i4")):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(np.asarray(dataset.variables[name][:], dtype=dtype)).tobytes())
        return digest.hexdigest()
    except Exception:  # a file the bridge cannot read is keyed by its bytes instead
        return _sha256(Path(mesh_file))


def default_path(mesh_file: str | Path) -> Path:
    mesh_file = Path(mesh_file)
    stem = mesh_file.name.split(".nc")[0]
    return mesh_file.parent / f"{stem}{FILE_SUFFIX}"


def _cache_path(mesh_file: Path, digest: str) -> Path:
    root = Path(os.environ.get("WOOF_HEX_CACHE_ROOT") or Path.home() / ".cache" / "woof hex")
    return root / "kim-gwd" / f"mesh-{digest[:24]}{FILE_SUFFIX}"


def _matches(out: Path, digest: str) -> bool:
    try:
        receipt = json.loads(out.with_name(out.name + ".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (out.is_file() and receipt.get("mesh_sha256") == digest
            and receipt.get("output_sha256") == _sha256(out) and receipt.get("recipe") == RECIPE)


def _has_recipe(exe: Path) -> bool:
    """Whether this rw_mpas_static takes ``--gwd-recipe``."""

    probe = subprocess.run([str(exe), "--help"], capture_output=True, text=True)
    return "--gwd-recipe" in probe.stdout + probe.stderr


def build(mesh_file: str | Path, out: str | Path, geog: str | Path | None = None, *,
          log=None) -> dict[str, Any]:
    """``rw_mpas_static --gwd-only --gwd-recipe kim2026`` on ``mesh_file``; writes ``out`` and its receipt."""
    from . import engines
    mesh_file, out = Path(mesh_file), Path(out)
    say = log or (lambda text: print(text, file=sys.stderr, flush=True))
    try:
        exe = engines.resolve(engines.STATIC)
    except engines.EngineRefusal as error:
        raise KimGwdStaticError(str(error)) from None
    if not _has_recipe(exe):
        # The wheel bundles its own rw_mpas_static (engine-bridges.json, for binary64 statics) and the ladder
        # reads it first; it predates the recipe.  The engine's own builder (gpuwm 3.0 carries
        # --gwd-recipe kim2026) writes only the ten GWD fields here, so it is used for this file alone.
        # THE BREAKAGE THIS PREVENTS (hex 0.5.0 on gpuwm 3.0.0, 2026-11-10): the opt-in KIM drag stopped
        # before the forecast on every install although the engine it declares carries the builder.
        exe = next((Path(c) for c in engines.gpuwm_candidates(engines.STATIC)
                    if Path(c).is_file() and _has_recipe(Path(c))), exe)
    if not _has_recipe(exe):
        raise KimGwdStaticError(
            f"{exe} has no --gwd-recipe: it predates MPAS PR #1408's statistics (woof port/omax), so it cannot "
            "write elvmax and the KIM drag would give no drag anywhere. Use a woof engine whose rw_mpas_static "
            "carries --gwd-recipe kim2026 (or set WOOF_HEX_RW_MPAS_STATIC), or run the v8.4.2 drag with "
            "--physics gwdo_version=v842")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.unlink(missing_ok=True)
    argv = [str(exe), "--grid", str(mesh_file), "--out", str(partial), "--gwd-only", "--gwd-recipe", RECIPE,
            "--receipt", str(partial) + ".receipt.json"]
    if geog is not None:
        argv += ["--geog", str(geog)]
    started = time.monotonic()
    result = subprocess.run(argv, capture_output=True, text=True)
    seconds = round(time.monotonic() - started, 1)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise KimGwdStaticError(f"rw_mpas_static --gwd-recipe {RECIPE} exited {result.returncode}:\n"
                                f"{engines.refusal_text(result.stderr, result.stdout)}")
    native = json.loads(Path(str(partial) + ".receipt.json").read_text(encoding="utf-8"))
    Path(str(partial) + ".receipt.json").unlink(missing_ok=True)
    os.replace(partial, out)
    record = {"schema": SCHEMA, "recipe": RECIPE, "mesh": str(mesh_file), "mesh_sha256": mesh_digest(mesh_file),
              "output": str(out), "output_sha256": _sha256(out), "fields": list(FIELDS),
              "builder": str(exe), "builder_sha256": _sha256(exe), "argv": argv[1:], "seconds": seconds,
              "parallelism": "rw_mpas_static rayon pool (all cores)", "builder_receipt": native}
    out.with_name(out.name + ".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    say(f"kim2026 GWD statistics: built {out} in {seconds} s")
    return record


def ensure(mesh_file: str | Path, geog: str | Path | None = None, *, log=None) -> Path:
    """The PR #1408 statistics for ``mesh_file``: reused when a receipt names the same mesh, else built."""
    mesh_file = Path(mesh_file).resolve()
    if not mesh_file.is_file():
        raise KimGwdStaticError(f"the mesh file {mesh_file} the KIM statistics are built on does not exist")
    digest = mesh_digest(mesh_file)
    beside = default_path(mesh_file)
    cached = _cache_path(mesh_file, digest)
    for out in (cached, beside):
        if _matches(out, digest):
            return out
    if geog is None:
        from .mesh_point import PointPlanRefusal, resolve_geog
        try:
            geog = resolve_geog(None)
        except PointPlanRefusal:
            geog = None  # rw_mpas_static's own ladder names what it is missing
    target = cached
    build(mesh_file, target, geog, log=log)
    return target


def mesh_file_of(arguments: Any) -> Path | None:
    """The file carrying the run's mesh topology: --grid, else --static, else --init."""
    for name in ("grid", "static", "init"):
        value = getattr(arguments, name, None)
        if value is not None and Path(value).is_file():
            return Path(value)
    return None


def forecast_args_select_kim(argv: Sequence[str]) -> bool:
    """Does a planned forecast argv run the KIM drag (MPAS suite route, correct mode, suite YSU drag)?"""
    items = [str(a).split("=", 1)[1] if str(a).startswith("--forecast-arg=") else str(a) for a in argv]
    from .physics_default import ensure as _default_physics
    items = _default_physics(items)  # a forecast that names no physics runs the default MPAS suite
    physics, suite, mode, backend = [], None, "correct", None
    i = 0
    while i < len(items):
        flag, eq, value = items[i].partition("=")
        if flag in ("--physics", "--physics-suite", "--physics-mode", "--physics-backend"):
            if not eq:
                value = items[i + 1] if i + 1 < len(items) else ""
                i += 1
            if flag == "--physics":
                physics.append(value)
            elif flag == "--physics-suite":
                suite = value
            elif flag == "--physics-mode":
                mode = value
            else:
                backend = value
        i += 1
    named = {k.strip().casefold(): v.strip().strip("\"'") for k, _, v in (p.partition("=") for p in physics)}
    if named.get("config_gwdo_scheme") == "bl_kim_gwdo":
        return True
    if named.get("config_gwdo_scheme") not in (None, "suite"):
        return False
    if suite is None and backend != "mpasphys":
        return False
    from .mpasphys.gwdo_version import default_version, normalise, LATEST
    version = normalise(named["gwdo_version"]) if "gwdo_version" in named else default_version(mode)
    if version != LATEST:
        return False
    from .physics_suites import load_table
    suites = (load_table().get("mpas_suites") or {}).get("suites", {})
    return (suites.get(suite or "mesoscale_reference") or {}).get("config_gwdo_scheme") == "bl_ysu_gwdo"


def prepare_for_forecast(forecast_argv: Sequence[str], mesh_file: str | Path, geog: str | Path | None = None,
                         *, log=None) -> Path | None:
    """At a door's prepare stage: build the KIM statistics the planned forecast will read, before any card."""
    if not forecast_args_select_kim(forecast_argv):
        return None
    return ensure(mesh_file, geog, log=log)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m woof.hex.kim_gwd_static", description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh", type=Path, required=True, help="grid, static or init carrying the mesh")
    parser.add_argument("--geog", type=Path)
    args = parser.parse_args(argv)
    print(ensure(args.mesh, args.geog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
