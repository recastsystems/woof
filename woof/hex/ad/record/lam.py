# SPDX-License-Identifier: Apache-2.0
"""The limited-area case of a recording: the mesh kind as data, not a second recorder.

A global mesh needs a grid, an init and (for the GSL drag) the orographic statistics. A limited-area mesh also needs
what the regional door prepared for it: the static file, the lateral-boundary files, the start time, the mesh-rows file
that registers the cull's row, and the contract deck of its rings. ``read_case`` takes either

  * the regional door's work directory (``woof hex regional ... --stage mesh..lbc``: ``regional-plan.json`` and the
    mesh stage's ``*.point-generate.json``), or
  * a case file of ``LAM_*`` words (``LAM_MESH_ROW LAM_GRID LAM_STATIC LAM_INIT LAM_LBC_DIR LAM_START LAM_DT LAM_ORO
    LAM_ROWS LAM_CONTRACT [LAM_HISTORY_MINUTES]``), the form the recorder chains on the rented cards read,

and returns one dict. The recorder adds the forecast words and the environment from it; nothing else changes.
THE BREAKAGE THIS PREVENTS: ``woof.hex.ad.record`` ran the forecast with ``--mesh --grid --init`` only, so a limited-area
mesh had no boundaries, no static file and no contract, and no 3 km storm plan could be recorded from an install; the
only route was a separate chain on rented cards. 2026-11-13.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

#: words a case file must set (LAM_ORO, LAM_DT and LAM_HISTORY_MINUTES are optional)
CASE_WORDS = ("LAM_MESH_ROW", "LAM_GRID", "LAM_STATIC", "LAM_INIT", "LAM_LBC_DIR", "LAM_START", "LAM_ROWS",
              "LAM_CONTRACT")


class CaseError(ValueError):
    pass


#: case-file words that name files or folders; a relative one is read against the case file's own folder
PATH_WORDS = ("LAM_GRID", "LAM_STATIC", "LAM_INIT", "LAM_LBC_DIR", "LAM_ORO", "LAM_ROWS", "LAM_CONTRACT")


def _env_file(path: Path) -> dict:
    words = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.replace("export ", "").strip()
        parts = shlex.split(value, comments=True)
        words[key] = parts[0] if parts else ""
    missing = [k for k in CASE_WORDS if not words.get(k)]
    if missing:
        raise CaseError(f"case file {path} does not set {', '.join(missing)}")
    # A case that travels (a download unpacked anywhere) names its files relative to itself.
    for key in PATH_WORDS:
        if words.get(key) and not Path(words[key]).is_absolute():
            words[key] = str((path.parent / words[key]).resolve())
    return {"source": str(path), "form": "case-file", "row": words["LAM_MESH_ROW"], "grid": words["LAM_GRID"],
            "static": words["LAM_STATIC"], "init": words["LAM_INIT"], "lbc_dir": words["LAM_LBC_DIR"],
            "start_time": words["LAM_START"], "rows": words["LAM_ROWS"], "contract": words["LAM_CONTRACT"],
            "oro": words.get("LAM_ORO") or None, "dt": float(words["LAM_DT"]) if words.get("LAM_DT") else None,
            "history_minutes": float(words.get("LAM_HISTORY_MINUTES") or 15)}


def _work_dir(root: Path) -> dict:
    value = json.loads((root / "regional-plan.json").read_text())
    paths = value.get("paths") or {}
    mesh = Path(paths.get("mesh") or root / "mesh")
    found = sorted(mesh.glob("*.point-generate.json"))
    if len(found) != 1:
        raise CaseError(f"{mesh} holds {len(found)} point-generate receipts (the regional door's mesh stage writes one)")
    cull = json.loads(found[0].read_text())["cull"]
    lbc = Path(cull["lbc_dir"])
    if not sorted(lbc.glob("lbc.*.nc")) and sorted((root / "lbc").glob("lbc.*.nc")):
        lbc = root / "lbc"   # a work dir whose boundary stage wrote beside the mesh
    oro = mesh / f"{cull['row']}.ugwp_oro_data.nc"
    return {"source": str(root), "form": "regional-work-dir", "row": cull["row"], "grid": cull["grid"],
            "static": cull["static"], "init": paths.get("init") or str(root / "init" / "init.nc"),
            "lbc_dir": str(lbc), "start_time": value["start_time"], "rows": str(mesh / "mesh-rows.json"),
            "contract": paths.get("contract") or str(root / "contract"), "oro": str(oro) if oro.is_file() else None,
            "dt": None, "history_minutes": float(value.get("history_every_minutes") or 15)}


def read_case(spec: str | Path) -> dict:
    """The limited-area case named by SPEC (a regional work dir, a file inside one, or a LAM_* case file)."""
    path = Path(spec).expanduser()
    if path.is_file() and path.name != "regional-plan.json" and path.suffix != ".nc":
        case = _env_file(path)
    else:
        root = path if path.is_dir() else path.parent
        for candidate in (root, *root.parents):
            if (candidate / "regional-plan.json").is_file():
                case = _work_dir(candidate)
                break
        else:
            raise CaseError(f"{spec}: neither a LAM_* case file nor inside a regional door work directory "
                            "(no regional-plan.json here or above)")
    for key in ("grid", "static", "init", "rows"):
        if not Path(case[key]).is_file():
            raise CaseError(f"limited-area case {case['source']}: its {key} file {case[key]} is missing")
    if not sorted(Path(case["lbc_dir"]).glob("lbc.*.nc")):
        raise CaseError(f"limited-area case {case['source']}: no lbc.*.nc in {case['lbc_dir']}")
    if not Path(case["contract"]).is_dir():
        raise CaseError(f"limited-area case {case['source']}: its contract deck folder {case['contract']} is missing "
                        "(woof hex regional ... --stage contract makes it)")
    return case


def is_limited_area(grid: str | Path) -> bool:
    """A limited-area grid marks its boundary zone in bdyMaskCell (1..7); a global grid has none or all zeros."""
    try:
        import netCDF4  # noqa: F401
        from netCDF4 import Dataset
    except ImportError:  # pragma: no cover - the [ad] install carries netCDF4
        return False
    import numpy as np
    with Dataset(str(grid)) as ds:
        return "bdyMaskCell" in ds.variables and bool(np.any(np.asarray(ds.variables["bdyMaskCell"][:]) > 0))


def forecast_words(case: dict) -> list[str]:
    """The forecast flags a limited-area recording adds (the regional door's forecast stage, the step named)."""
    return ["--static", str(case["static"]), "--lbc-dir", str(case["lbc_dir"]), "--start-time", case["start_time"],
            "--history-every-minutes", f"{case['history_minutes']:g}"]


#: the mesh-row fields that name files (woof.hex.mesh_rows.MeshRow); grid and static are verified by bytes and sha256
ROW_PATHS = ("grid", "static", "generator_receipt", "static_receipt", "cull_receipt")


def localize_rows(case: dict, work: Path) -> dict:
    """CASE with a mesh-rows file whose file names resolve on this machine.

    A rows file names its grid and static by path, and the forecast verifies those files by bytes and sha256. A rows
    file that travelled (the storm example's download) names them relative to its own folder; this writes a copy with
    each relative name made absolute into WORK and points the case at it. A rows file whose names are all absolute is
    used as it is. THE BREAKAGE THIS PREVENTS: the rows file of a packed case named the folders of the machine that
    made it, and on any other machine the forecast refused "names a grid at ... that is not a file"."""
    rows_path = Path(case["rows"])
    doc = json.loads(rows_path.read_text())
    moved = 0
    for row in doc.get("rows", []):
        for key in ROW_PATHS:
            value = row.get(key)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                row[key] = str((rows_path.parent / value).resolve())
                moved += 1
    if not moved:
        return case
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / "mesh-rows.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    return dict(case, rows=str(out), rows_from=str(rows_path))


def forecast_env(case: dict) -> dict:
    """The cull's registered row and its contract deck, as the regional door hands them to the forecast."""
    return {"WOOF_HEX_MESH_ROWS": str(case["rows"]), "WOOF_HEX_REGIONAL_CONTRACT_DIR": str(case["contract"])}


#: the recorder keeps an upload's bytes when it is at most this small. A limited-area forecast uploads the garbage-column
#: scrub descriptor tables (cuda_solve_region_v841.py, 5 int64 a row, 800 bytes and more), which the sealer rebuilds
#: from live allocations only when their bytes are kept (rule solve_region_scrub_descriptors). THE BREAKAGE THIS
#: PREVENTS: at the global bound (512) those tables were not retained and the seal refused them (a 3 km storm recording, 2026-11-07).
RETAIN_SMALL_UPLOADS = {"global": 512, "limited-area": 65536}
