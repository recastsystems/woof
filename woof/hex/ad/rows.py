# SPDX-License-Identifier: Apache-2.0
"""Operator rows on demand: any mesh, any card class.

The door table (``data/ad-doors.v1.json``) ships the rows that were proven, as a seed. A door asked for another mesh
(``--mesh NAME|PATH``) or run on another card class makes the row it needs and keeps it in the row cache, so the
table is a cache, not a whitelist. A row has two parts that are made separately and cached separately:

* the mesh part, per suite and mesh: the mesh files (grid, GSL-drag orography), cells, edges and levels read from the
  files, the timestep from hex's own rule (6 s per km of ``nominalMinDc``, dividing the radiation interval), the window
  memory need from the suite's measured cost per cell, and the derivative kernel set re-pinned to the mesh's derived
  topology tables (made from the recorded plan by the kernel-set re-pin step);
* the card part, per image set and ``sm_XY``: the physics derivative images (L2 scheme bundle, parameter bundle, TEMPO
  images) for the card present, built from their recipes by the image builder and assembled beside the shipped
  bundle (same rows, same tiles, images for this card).

    WOOF_HEX_AD_ROWS            the row cache (default ``<hex cache>/ad/rows``)
    WOOF_HEX_AD_MESH_PATH       colon list of directories holding ``<mesh>/<mesh>.grid.nc`` or ``<mesh>.grid.nc``
    WOOF_HEX_AD_IMAGE_BUILDER   command building one image set for one arch; ``{arch}``, ``{out}`` and
                                 ``{build_set}`` are filled in
    WOOF_HEX_AD_KSET_REPIN      command re-pinning a kernel set to a plan's mesh; ``{kset}``, ``{plan}``, ``{out}``

A step that cannot be made (no image builder for this card, no re-pin for this mesh) is a named refusal; nothing
falls back to the CPU or to another mesh's tables.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

ROWS_ENV = "WOOF_HEX_AD_ROWS"
MESH_PATH_ENV = "WOOF_HEX_AD_MESH_PATH"
IMAGE_BUILDER_ENV = "WOOF_HEX_AD_IMAGE_BUILDER"
KSET_REPIN_ENV = "WOOF_HEX_AD_KSET_REPIN"
ROW_SCHEMA = "gpuwm-hex.ad-row/v1"
IMAGES_SCHEMA = "gpuwm-hex.ad-row-images/v1"


class AdRowError(RuntimeError):
    """A row for this mesh or card cannot be made, with the reason named."""


# ------------------------------------------------------------------------------------------------------ cache
def rows_cache() -> Path:
    declared = os.environ.get(ROWS_ENV)
    if declared:
        return Path(declared).expanduser().absolute()
    from .. import assets as _assets
    return _assets.cache_root() / "ad" / "rows"


def arch_of(compute_capability: str) -> str:
    """'8.9' -> 'sm_89', '12.0' -> 'sm_120'."""
    major, _, minor = str(compute_capability).strip().partition(".")
    if not major.isdigit() or not (minor or "0").isdigit():
        raise AdRowError(f"compute capability {compute_capability!r} is not MAJOR.MINOR")
    return f"sm_{int(major)}{int(minor or 0)}"


#: NVIDIA card families by compute architecture, with cards of each, for refusals in plain words.
#: THE BREAKAGE THIS PREVENTS: on an H100 the README's command said only "its derivative images are built for sm_120
#: and the row names no image recipe for sm_90": compiler words that name neither the cards that run the row nor what
#: to do (two 0.6.0 builds).
CARD_FAMILIES = {
    "sm_70": ("Volta", "V100"),
    "sm_72": ("Volta", "Jetson AGX Xavier"),
    "sm_75": ("Turing", "GeForce RTX 20 series, T4"),
    "sm_80": ("Ampere", "A100, A30"),
    "sm_86": ("Ampere", "GeForce RTX 30 series, A10, A40, RTX A6000"),
    "sm_87": ("Ampere", "Jetson Orin"),
    "sm_89": ("Ada Lovelace", "GeForce RTX 40 series, L4, L40S, RTX 6000 Ada"),
    "sm_90": ("Hopper", "H100, H200, GH200"),
    "sm_100": ("Blackwell data-centre", "B200, GB200"),
    "sm_103": ("Blackwell Ultra data-centre", "B300, GB300"),
    "sm_110": ("Blackwell", "Jetson Thor"),
    "sm_120": ("Blackwell", "GeForce RTX 50 series, RTX PRO Blackwell"),
    "sm_121": ("Blackwell", "DGX Spark"),
}

#: The cards of a family the rows were measured on, with their memory in GB and the MiB the driver reports for the
#: whole card, to name one that holds a row. THE BREAKAGE THIS PREVENTS: a card was offered when its GB x 1000 covered
#: the need, so a row that needs 96,600 MiB (the 3 km default route's measured peak) named no card at all, though a
#: 96 GB card has 97,887 MiB.
MEASURED_CARDS = {"sm_120": (("GeForce RTX 5090", 32, 32607), ("RTX PRO 6000 Blackwell", 96, 97887))}


def capability_of(arch: str) -> str:
    """'sm_120' -> '12.0', 'sm_90' -> '9.0'."""
    digits = str(arch).removeprefix("sm_").rstrip("af")
    return f"{int(digits[:-1])}.{digits[-1]}" if len(digits) > 1 and digits.isdigit() else str(arch)


def family_words(arch: str) -> str:
    """'sm_90' -> 'a Hopper card (compute capability 9.0, as H100, H200, GH200)'."""
    family, cards = CARD_FAMILIES.get(arch, (None, None))
    if family is None:
        return f"a card of compute capability {capability_of(arch)}"
    return f"a{'n' if family[0] in 'AEIOU' else ''} {family} card (compute capability {capability_of(arch)}, as {cards})"


def runs_on_words(name: str, shipped: list[str], need_mib: int | None, arch: str, card: str | None = None) -> str:
    """The plain words of a card the row has no images for: which cards run it, which this one is, what to do."""
    families = [f"{CARD_FAMILIES[a][0]} cards of compute capability {capability_of(a)} ({CARD_FAMILIES[a][1]})"
                if a in CARD_FAMILIES else f"cards of compute capability {capability_of(a)}" for a in shipped]
    this = f"card {card}" if card is not None else "this card"
    need = f" with at least {int(need_mib):,} MiB free" if need_mib else ""
    fits = [f"{label} ({gb} GB)" for a in shipped for label, gb, mib in MEASURED_CARDS.get(a, ())
            if not need_mib or mib >= int(need_mib)]
    example = f", for example {' or '.join(fits)}" if fits else ""
    return (f"row {name} runs on NVIDIA {' or '.join(families)} only, and {this} is {family_words(arch)}. To run it, "
            f"use a compute capability {' or '.join(capability_of(a) for a in shipped)} card{need}{example}")


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def cached_rows() -> dict[str, dict]:
    """Every row the cache holds (row name -> row), for the door's --row lookup."""
    out: dict[str, dict] = {}
    root = rows_cache()
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*/row.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if doc.get("schema") == ROW_SCHEMA and isinstance(doc.get("row"), dict):
            out[doc["name"]] = doc["row"]
    return out


# ------------------------------------------------------------------------------------------------------- mesh
_GRID_SUFFIXES = (".grid.nc", ".static.nc", ".init.nc")


def _mesh_name(path: Path) -> str:
    name = path.name
    for suffix in _GRID_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def mesh_search_dirs() -> list[Path]:
    dirs = [Path(p).expanduser() for p in os.environ.get(MESH_PATH_ENV, "").split(os.pathsep) if p]
    try:
        from ..global_door import cache_root as global_cache
        dirs.append(global_cache() / "meshes")
    except Exception:   # noqa: BLE001 - the global door is optional here
        pass
    return dirs


def find_mesh(spec: str, grid: Path | None = None, oro: Path | None = None) -> dict:
    """``--mesh`` (a mesh name, a grid file or a directory holding one) -> {mesh, grid, oro}."""
    path = Path(spec).expanduser()
    if grid is not None:
        grid = Path(grid).expanduser()
    elif path.is_file():
        grid = path
    elif path.is_dir():
        found = sorted(path.glob("*.grid.nc"))
        if len(found) != 1:
            raise AdRowError(f"--mesh {spec}: the directory holds {len(found)} *.grid.nc files; name one")
        grid = found[0]
    else:
        tried = []
        for root in mesh_search_dirs():
            for cand in (root / spec / f"{spec}.grid.nc", root / f"{spec}.grid.nc"):
                tried.append(str(cand))
                if cand.is_file():
                    grid = cand
                    break
            if grid is not None:
                break
        if grid is None:
            raise AdRowError(f"--mesh {spec}: no grid file found (looked in {', '.join(tried) or 'no directory'}); "
                             f"pass --mesh PATH/{spec}.grid.nc, or set {MESH_PATH_ENV}")
    if not grid.is_file():
        raise AdRowError(f"mesh grid {grid} is not a file")
    name = _mesh_name(grid) if (path.is_file() or path.is_dir() or not spec) else spec
    if oro is None:
        cand = grid.parent / f"{name}.ugwp_oro_data.nc"
        oro = cand if cand.is_file() else None
    return {"mesh": name, "grid": grid.absolute(), "oro": Path(oro).absolute() if oro else None}


EARTH_RADIUS_M = 6371229.0   # MPAS's sphere radius; pristine NCAR grid files are on the unit sphere


def _dims(ds) -> dict[str, int]:
    out = {}
    for k, v in ds.dimensions.items():
        size = v if isinstance(v, int) else getattr(v, "size", None)
        out[k] = int(size if size is not None else len(v))
    return out


def _attrs(ds) -> dict:
    try:
        return {k: ds.getncattr(k) for k in ds.ncattrs()}
    except Exception:   # noqa: BLE001 - attributes are optional here
        return {}


def _nominal_dc(ds) -> float | None:
    """nominalMinDc when the file names it, else the smallest dcEdge, scaled from a unit sphere to the Earth."""
    attrs = _attrs(ds)
    if attrs.get("nominalMinDc") is not None:
        return float(attrs["nominalMinDc"])
    if "dcEdge" not in ds.variables:
        return None
    import numpy as np
    dc = float(np.min(np.asarray(ds.variables["dcEdge"][:], np.float64)))
    radius = float(attrs.get("sphere_radius", EARTH_RADIUS_M if dc > 10.0 else 1.0))
    return dc * EARTH_RADIUS_M / radius


def mesh_facts(grid: Path, init: Path | None = None) -> dict:
    """cells, edges, levels and the nominal spacing of a mesh, read from its files (the init's levels win)."""
    from woof.netcdf_bridge import open_dataset
    facts: dict = {}
    with open_dataset(grid) as ds:
        dims = _dims(ds)
        facts["cells"], facts["edges"], facts["levels"] = dims.get("nCells"), dims.get("nEdges"), dims.get("nVertLevels")
        facts["nominal_dc_m"] = _nominal_dc(ds)
        # a limited-area mesh marks its boundary zone in bdyMaskCell (1..7); a global mesh has none or all zeros
        if "bdyMaskCell" in ds.variables:
            import numpy as np
            facts["limited_area"] = bool(np.any(np.asarray(ds.variables["bdyMaskCell"][:]) > 0))
    if init is not None:
        with open_dataset(init) as ds:
            dims = _dims(ds)
            if dims.get("nCells") not in (None, facts["cells"]):
                raise AdRowError(f"init {init} has {dims['nCells']} cells; mesh {grid} has {facts['cells']}")
            facts["levels"] = dims.get("nVertLevels", facts["levels"])
            if _attrs(ds).get("nominalMinDc") is not None:
                facts["nominal_dc_m"] = float(_attrs(ds)["nominalMinDc"])
    if not facts.get("cells") or not facts.get("edges"):
        raise AdRowError(f"{grid} names no nCells/nEdges dimension; it is not an MPAS mesh file")
    return facts


def default_dt(nominal_dc_m: float | None, radiation_s: float) -> float:
    if not nominal_dc_m:
        raise AdRowError("the mesh names no nominalMinDc or dcEdge; pass --dt")
    from ..mpas_defaults import default_dt_seconds
    return float(default_dt_seconds(nominal_dc_m, radiation_seconds=radiation_s)["dt_seconds"])


def need_mib(suite: dict, cells: int, levels: int) -> int:
    """Window memory need from the suite's measured cost: fixed arenas plus a per-column-level cost, and the rule's
    stated margin (a factor on the per-column-level part, and margin_mib added on top)."""
    mem = suite if "fixed_mib" in suite else suite["memory"]   # a memory rule, or a suite that carries one
    per = float(mem["mib_per_column_level"]) * cells * max(levels, 1)
    return int(round(float(mem["fixed_mib"]) + per * float(mem.get("margin", 1.0)) + float(mem.get("margin_mib", 0))))


def window_class(suite: dict, cells: int, kind: str = "global", card_mib: int | None = None,
                 levels: int = 59) -> dict | None:
    """The suite's window class for a mesh of this size and kind: the sizes, environment and images of one measured
    recipe (table data). Classes are matched by kind first ("global", or "limited-area": a different kernel set and
    deck, never chosen by size alone), then by size. With ``card_mib`` (the whole memory of the card the window will
    run on) a class whose memory need is above that card is passed over for the next class of the same kind that
    holds the mesh and fits the card (a class with card words sizes itself to the card it is admitted on); with none
    that fits, the first class that holds the mesh is the answer and the card admission refuses it by its need.
    THE BREAKAGE THIS PREVENTS: the 135,318-cell Warn-on-Forecast-sized 3 km box and every limited-area mesh on a
    card smaller than 93,000 MiB were refused, though the runner's out-of-core budget runs them (2026-11-13).
    A mesh above the largest class of its kind is refused with the table's reason."""
    every = suite.get("window_classes")
    if not every:
        return None
    classes = [c for c in every if c.get("mesh_kind", "global") == kind]
    if not classes:
        raise AdRowError(f"a {kind} mesh has no window class in this suite: "
                         + suite.get("window_class_missing_limited_area", "no such window recipe is in the table"))
    holds = [c for c in classes if cells <= int(c["max_cells"])]
    if holds:
        if card_mib is not None:
            for entry in holds:
                if entry.get("memory") is None or need_mib(entry["memory"], cells, levels) <= int(card_mib):
                    return entry
        return holds[0]
    above = suite.get("window_class_above")
    if kind != "global":
        above = suite.get(f"window_class_above_{kind.replace('-', '_')}", above)
    raise AdRowError(f"a mesh of {cells:,} cells is above this suite's largest {kind} window class "
                     f"({max(int(c['max_cells']) for c in classes):,} cells): "
                     + (above or "no larger window recipe is in the table"))


def _same_class(suite: dict, cached: dict, card_mib: int | None) -> bool:
    """Whether a cached row was made for the class this mesh takes on this card (a card of another size can take
    another class of the same kind; the cached row then answers with the other class's words)."""
    was = (cached.get("generated") or {}).get("window_class")
    if not was or not cached.get("cells"):
        return True
    try:
        now = window_class(suite, int(cached["cells"]), was.get("mesh_kind", "global"), card_mib,
                           int(cached.get("levels") or 59))
    except AdRowError:
        return False
    return (now is not None and now.get("from_recipe") == was.get("from_recipe")
            and int(now["max_cells"]) == int(was["max_cells"]))


def _word(spec: dict, value: float) -> str:
    return str(max(int(spec.get("min", 1)), int(value)))


def fill_mesh_words(argv: list[str], card_words: dict, cells: int, levels: int) -> list[str]:
    """A class's words that follow the mesh (``"from": "mesh"``), filled when the row is made: bytes per column level
    x cells x levels, in units, plus ``add`` (rounded up), never below ``min``."""
    import math
    fill = {token: _word(spec, math.ceil(float(spec["bytes_per_column_level"]) * cells * max(levels, 1)
                                         / float(spec["unit_bytes"])) + int(spec.get("add", 0)))
            for token, spec in card_words.items() if spec.get("from") == "mesh"}
    return [fill.get(a, a) for a in argv]


def _units(spec: dict, count: int) -> str:
    unit = int(float(spec["unit_mib"]))
    return f"{count} GiB" if unit == 1024 else f"{count} x {unit:,} MiB"


def fill_card_words(argv: list[str], card_words: dict, card_free_mib: int | None,
                    host_available_mib: int | None, *, card: str | None = None,
                    label: str | None = None) -> tuple[list[str], dict]:
    """A class's words that follow the card it is admitted on (``"from": "card_free_mib"``: the free card memory
    less ``less_mib``) and the host (``"from": "host_available_mib"``: ``fraction`` of the available host memory),
    filled at each admission, in units. Returns the words and what each was filled with. A card or host that gives
    less than the word's ``min`` is refused: the word is never rounded up to a budget the card or host does not have.
    THE BREAKAGE THIS PREVENTS: a row with too small a memory need (--need-mib, or a cached row edited by hand) was
    admitted on a card with 4,000 MiB free and handed the runner --vram-budget-gb 8 (door-row check, 2026-11-13)."""
    import math
    who = f"class {label}" if label else "this row's window class"
    filled: dict = {}
    for token, spec in card_words.items():
        source = spec.get("from")
        least = int(spec.get("min", 1))
        if source == "card_free_mib" and card_free_mib is not None:
            value = (int(card_free_mib) - int(spec.get("less_mib", 0))) / float(spec["unit_mib"])
            if value < least:
                need = least * int(float(spec["unit_mib"])) + int(spec.get("less_mib", 0))
                raise AdRowError(f"card {card if card is not None else '(the admitted card)'} has "
                                 f"{int(card_free_mib):,} MiB free; {who} needs at least {need:,} MiB free for its "
                                 f"smallest card budget ({_units(spec, least)}, word {token}, plus "
                                 f"{int(spec.get('less_mib', 0)):,} MiB the budget does not count); nothing was run")
            filled[token] = str(int(value))
        elif source == "host_available_mib" and host_available_mib is not None:
            fraction = float(spec.get("fraction", 1.0))
            value = int(host_available_mib) * fraction / float(spec["unit_mib"])
            if value < least:
                need = math.ceil(least * float(spec["unit_mib"]) / fraction)
                raise AdRowError(f"this host has {int(host_available_mib):,} MiB available; {who} needs at least "
                                 f"{need:,} MiB available for its smallest host budget ({_units(spec, least)}, word "
                                 f"{token}, {fraction:g} of what is available); nothing was run")
            filled[token] = str(int(value))
        elif source in ("card_free_mib", "host_available_mib"):
            raise AdRowError(f"the window's word {token} follows {source}, which this machine does not report")
    return [filled.get(a, a) for a in argv], filled


def class_floor_mib(table: dict, row: dict) -> int | None:
    """The free card memory the table's class needs for a row made on demand (its ``generated`` suite and window
    class, looked up in the table, not in the row), or None for a shipped row or one whose class the table no longer
    holds. A row's own ``need_free_mib`` is a copy of this when the row is made."""
    made = row.get("generated") or {}
    suite = (table.get("suites") or {}).get(made.get("suite") or "")
    if not suite or not row.get("cells"):
        return None
    was = made.get("window_class")
    memory = suite.get("memory")
    if was:
        hit = [c for c in suite.get("window_classes") or [] if c.get("from_recipe") == was.get("from_recipe")
               and int(c.get("max_cells") or -1) == int(was.get("max_cells") or -2)]
        if not hit:
            return None
        memory = hit[0].get("memory") or memory
    if not memory:
        return None
    cells = int((row.get("record_ranks") or {}).get("cells_per_rank") or row["cells"])   # one rank's card
    return need_mib(memory, cells, int(row.get("levels") or 59))


def with_class_floor(table: dict, row: dict) -> dict:
    """The row with ``need_free_mib`` the larger of its own value and its class's floor (``class_floor_mib``), so a
    row loaded from the cache never asks less of the card than its class does. THE BREAKAGE THIS PREVENTS: a cached
    w900 row edited to need_free_mib 1000 was admitted on a card with 4,000 MiB free (door-row check, 2026-11-13)."""
    floor = class_floor_mib(table, row)
    if floor is None or int(row.get("need_free_mib") or 0) >= floor:
        return row
    return {**row, "need_free_mib": int(floor)}


# -------------------------------------------------------------------------------------------- mesh rows
#: the halo copies a rank holds beside its own cells, as a share of them (a 4-way cut of a 362,280-cell mesh holds
#: about 4 % more; 10 % covers 2 to 8 ranks of a compact region)
RANK_HALO_MARGIN = 1.10


def rank_cells(cells: int, ranks: int = 1) -> int:
    """The cells one rank's card holds: the mesh for one rank, else cells / ranks with its halo margin."""
    ranks = max(int(ranks or 1), 1)
    if ranks == 1:
        return int(cells)
    return int(-(-int(cells) * RANK_HALO_MARGIN // ranks))


def row_name(suite: str, mesh: str) -> str:
    return f"{suite}-{mesh}"


def make_row(table: dict, suite_name: str, mesh: dict, *, init: Path | None = None, dt: float | None = None,
             own_case: bool = False, door: str | None = None, card_mib: int | None = None,
             ranks: int = 1) -> tuple[str, dict]:
    """The row for (suite, mesh): from the row cache, else made from the suite's template row and the mesh files.
    ``card_mib`` is the whole memory of the card the window will run on (None: not known); it picks among the
    classes that hold the mesh (``window_class``), and a cached row made for another class is made again.

    A shipped row of the same name (hex 0.5.0 shipped a pinned x1.2562 row; 0.6.0 ships none) answers when the user brings no case of their own (its
    example plan is then the only plan there is) or asks a door only it runs (param-grad). When the user brings their
    own case (--init or --plan) and the template follows a published set the shipped row does not, the row is made
    from the template like any other mesh's. THE BREAKAGE THIS PREVENTS: --mesh x1.2562 --init USER_INIT ran the
    derivative of hex 0.5.0's pinned row (its own kernel set and bundle) while every other mesh ran the published set's,
    so the 480 km door answered with an older derivative than the 240 km one (2026-11-10).
    """
    suites = table.get("suites") or {}
    if suite_name not in suites:
        raise AdRowError(f"--suite {suite_name!r}: the table makes rows for {', '.join(sorted(suites)) or 'no suite'}")
    suite = suites[suite_name]
    name = row_name(suite_name, mesh["mesh"])
    template = table["rows"][suite["template_row"]]
    shipped = table["rows"].get(name)
    # a row the door made earlier comes back through the cache below (and is made again once its template or suite
    # rule changed), never as a shipped row: the door puts the cached rows into table["rows"] before it calls this.
    # THE BREAKAGE THIS PREVENTS: the 30 km made row cached at 22:01Z on a 96 GB card kept kernel set storm-h after the made
    # 3 km class took storm-bs5-lean2-astra2-dynE5-sp74-ddr (about twice as fast a reverse), because the cached row
    # answered here as a shipped one and the template check below never ran (an adjoint-test, 19 Nov 2026)
    if shipped is not None and dt is None and not shipped.get("generated"):
        newer = bool(template.get("latest_id")) and not shipped.get("latest_id") and not shipped.get("generated")
        if not (own_case and newer and (door is None or door in template.get("doors", ()))):
            return name, shipped
    stamp = hashlib.sha256(json.dumps([template, suite], sort_keys=True).encode()).hexdigest()
    cached = cached_rows().get(name)
    # a cached row is reused only while its template and suite rule are the ones it was made from
    if (cached is not None and (cached.get("generated") or {}).get("template_sha256") == stamp
            and (dt is None or float(cached["dt_s"]) == float(dt)) and _same_class(suite, cached, card_mib)):
        return name, with_class_floor(table, cached)   # never less than the class asks of the card
    facts = mesh_facts(mesh["grid"], init)
    levels = facts.get("levels") or template["levels"]
    step = float(dt) if dt is not None else default_dt(facts["nominal_dc_m"], float(suite["radiation_s"]))
    if mesh["oro"] is None and suite.get("needs_oro", True):
        raise AdRowError(f"mesh {mesh['mesh']}: no {mesh['mesh']}.ugwp_oro_data.nc beside {mesh['grid']} (the GSL "
                         "drag orography); pass --oro")
    kind = "limited-area" if facts.get("limited_area") else "global"
    # A plan split over N ranks (--record-ranks N) runs one rank a card: each card holds its own cells and their halo,
    # so the class is the one that holds a rank (cells / N, and RANK_HALO_MARGIN for the halo copies), not the whole
    # mesh. THE BREAKAGE THIS PREVENTS: 12 km CONUS (362,280 cells) was refused above the largest class (200,000)
    # although four ranks of it (about 100,000 cells each) sit inside the one-card class with spill.
    class_cells = rank_cells(facts["cells"], ranks)
    cls = window_class(suite, class_cells, kind, card_mib, levels)
    memory = (cls or {}).get("memory") or suite["memory"]
    row = copy.deepcopy(template)
    row.pop("example_plan", None)
    row.pop("template_only", None)
    row["what"] = (f"{suite_name} suite on {mesh['mesh']} ({facts['cells']:,} cells, {levels} levels), dt "
                   f"{step:g} s, made on demand from row {suite['template_row']}: " + template["what"].split(":", 1)[-1].strip())
    row.update({"mesh": mesh["mesh"], "cells": facts["cells"], "edges": facts["edges"], "levels": levels,
                "dt_s": step, "need_free_mib": need_mib(memory, class_cells, levels)})
    if ranks > 1:
        row["record_ranks"] = {"ranks": int(ranks), "cells_per_rank": class_cells,
                               "rule": f"cells / ranks x {RANK_HALO_MARGIN:g} (halo copies)"}
    assets = dict(row["assets"])
    if cls and cls.get("whole"):   # a whole class (limited area): its recipe's files, flags, variables replace the template's
        row["window_args"] = list(cls["window_args"])
        row["window_env"] = dict(cls["window_env"])
        assets = {**{k: v for k, v in assets.items() if k in ("params", "bundle_params")}, **cls["assets"]}
        row["asset_args"], row["asset_env"] = dict(cls["asset_args"]), dict(cls["asset_env"])
        row["mesh_kind"] = kind
        # the template's words ("radiation is held ...") do not describe a whole class: it runs its own recipe
        row["what"] = (f"{suite_name} suite on {mesh['mesh']} ({facts['cells']:,} cells, {levels} levels), dt "
                       f"{step:g} s, made on demand as the {kind} class from the published recipe "
                       f"{cls.get('from_recipe')}: " + (cls.get("what") or "(the recipe names no description)"))
        # a class that sizes itself to the mesh and the card: the mesh's words now, the card's at each admission
        if cls.get("card_words"):
            row["card_words"] = copy.deepcopy(cls["card_words"])
            row["window_args"] = fill_mesh_words(row["window_args"], cls["card_words"], facts["cells"], levels)
        if cls.get("cards_max"):   # one card, or more when the user names them (--cards A,B): physics on the others
            row["cards_max"] = int(cls["cards_max"])
            row["shard_env"] = dict(cls.get("shard_env") or {})
            row["need_free_mib_shard"] = int(cls.get("need_free_mib_shard") or 0)
    elif cls:   # the class adds its sizes and environment to the template's, and its own images
        row["window_args"] = list(row["window_args"]) + list(cls.get("window_args", []))
        row["window_env"] = {**row.get("window_env", {}), **cls.get("window_env", {})}
        assets.update(cls.get("assets", {}))
        row["asset_env"] = {**row.get("asset_env", {}), **cls.get("asset_env", {})}
        if cls.get("what"):   # the class swaps images the template's words describe: its own words replace them
            row["what"] = (f"{suite_name} suite on {mesh['mesh']} ({facts['cells']:,} cells, {levels} levels), dt "
                           f"{step:g} s, made on demand from row {suite['template_row']} with the window class of "
                           f"recipe {cls.get('from_recipe')}: " + cls["what"])
        if int(cls.get("cards") or 1) > 1:   # a multi-card class: the door takes that many cards (--cards)
            row["cards"] = int(cls["cards"])
            row["need_free_mib_shard"] = int(cls.get("need_free_mib_shard") or 0)
    # the class's measured host memory (the door refuses before any download on a host with less available)
    for key in ("need_host_mib", "need_host_mib_by_door"):
        if (cls or {}).get(key):
            row[key] = int(cls[key]) if key == "need_host_mib" else {k: int(v) for k, v in cls[key].items()}
        else:
            row.pop(key, None)
    # the class's left-out memory for the strict self dot (a list of scratch ranges, under the asset root): the
    # adjoint-test door uses it on its self-dot window only, and proves it on its own plan with a NaN-poison window
    if (cls or {}).get("self_dot_dead"):
        row["self_dot_dead"] = dict(cls["self_dot_dead"])
    else:
        row.pop("self_dot_dead", None)
    if (cls or {}).get("boundary_layer_default") and row.get("boundary_layer"):
        # the class names the route its windows run by default (the limited-area classes: guarded)
        row["boundary_layer"] = {**row["boundary_layer"], "default": cls["boundary_layer_default"]}
    assets["grid"] = str(mesh["grid"])
    if mesh["oro"] is not None:
        assets["oro"] = str(mesh["oro"])
    row["assets"] = assets
    # the template's kernel set pins the template mesh's derived topology tables; it is re-pinned once a plan exists
    row["kset_mesh"] = template.get("kset_mesh") or template["mesh"]
    row["generated"] = {"from_row": suite["template_row"], "suite": suite_name, "template_sha256": stamp, "nominal_dc_m": facts["nominal_dc_m"],
                        "dt_rule": "given (--dt)" if dt is not None else
                        f"woof.hex.mpas_defaults.default_dt_seconds (radiation {suite['radiation_s']:g} s)",
                        "need_rule": memory,
                        "window_class": ({"max_cells": cls["max_cells"], "from_recipe": cls.get("from_recipe"),
                                          "mesh_kind": cls.get("mesh_kind", "global")}
                                         if cls else None), "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_json(rows_cache() / name / "row.json", {"schema": ROW_SCHEMA, "name": name, "row": row})
    return name, row


def ensure_kset(table: dict, name: str, row: dict, plan: Path, asset_root: Path) -> dict:
    """A kernel set pinned to this row's mesh: the template's when the meshes agree, else the re-pinned one."""
    if row.get("kset_mesh", row["mesh"]) in (row["mesh"], "any"):
        return {}   # pinned to this mesh, or the runner builds the topology tables from the run's own mesh
    out = rows_cache() / name / "kset"
    if (out / "kernels.json").is_file() and (out / ".complete.json").is_file():
        return {"kernels": out / "kernels.json", "cubins": out / "cubins"}
    command = os.environ.get(KSET_REPIN_ENV) or (table.get("suites", {}).get(row.get("suite", ""), {}) or {}).get("kset_repin")
    if not command:
        raise AdRowError(f"row {name}: the kernel set pins the {row['kset_mesh']} mesh's derived topology tables and "
                         f"no re-pin step is installed for {row['mesh']} (set {KSET_REPIN_ENV} to the re-pin command, "
                         "'{kset}' '{plan}' '{out}' filled in)")
    base = Path(row["assets"]["kernels"])
    base = base if base.is_absolute() else asset_root / base
    _run(command, {"kset": str(base.parent), "plan": str(plan), "out": str(out)}, out.parent / "kset-repin.log",
         f"kernel-set re-pin for {row['mesh']}")
    if not (out / "kernels.json").is_file():
        raise AdRowError(f"kernel-set re-pin for {row['mesh']} exited 0 but wrote no {out / 'kernels.json'}")
    _write_json(out / ".complete.json", {"plan": str(plan), "base": str(base.parent)})
    return {"kernels": out / "kernels.json", "cubins": out / "cubins"}


def _run(command: str, fill: dict, log: Path, what: str) -> None:
    argv = [part.format(**fill) for part in shlex.split(command)]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as stream:
        rc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
        raise AdRowError(f"{what} failed (rc {rc}): {tail[0][:300]} (log {log})")


# -------------------------------------------------------------------------------------------- card images
def cubin_arch(image: bytes) -> str | None:
    """sm_XY of a CUDA ELF image (None for PTX, a fatbin or anything else): e_machine 190 (EM_CUDA); the SM is in
    e_flags bits 8-15 under the current ABI (OSABI 0x41) and bits 0-7 under the old one."""
    import struct
    if image[:4] != b"ELF" or len(image) < 52 or struct.unpack_from("<H", image, 18)[0] != 190:
        return None
    flags = struct.unpack_from("<I", image, 48)[0]
    sm = (flags >> 8) & 0xFF if image[7] == 0x41 else flags & 0xFF
    return f"sm_{sm}" if sm else None


_PLAN_DOCS: dict[tuple, dict] = {}


def plan_doc(plan: Path) -> dict:
    """plan.json of a plan package, parsed once per process for the file as it is now (path, size, change time).
    Read only: no caller writes into it. THE COST THIS REMOVES: the 3 km storm example's plan.json takes about 3 s to
    read and parse, and one sensitivity command parsed it four times before its runner started."""
    path = Path(plan) / "plan.json"
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _PLAN_DOCS:
        _PLAN_DOCS.clear()   # one plan a command: an older plan's document is not kept in memory
        _PLAN_DOCS[key] = json.loads(path.read_text(encoding="utf-8"))
    return _PLAN_DOCS[key]


def forget_plan_docs() -> None:
    """Let the parsed plan go (a door calls this when its runner starts: the 3 km storm example's plan.json is 708 MB
    of text, and the door does not hold it through a window that needs the host's memory)."""
    _PLAN_DOCS.clear()


def plan_arch(plan: Path) -> str | None:
    """The arch the plan's recorded forward module images were built for (they replay only on that card class)."""
    import gzip
    try:
        doc = plan_doc(plan)
    except (OSError, ValueError):
        return None
    for module in doc.get("modules", [])[:8]:
        path = plan / module.get("path", "")
        if not path.is_file():
            continue
        data = gzip.open(path).read(64) if module.get("compression") == "gzip" else path.read_bytes()[:64]
        arch = cubin_arch(data)
        if arch:
            return arch
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble_bundle(src: Path, built: Path, arch: str, out: Path, *, shipped_arch: str = "sm_120",
                    drop: tuple[str, ...] = (), shared: Path | None = None) -> dict:
    """A scheme bundle for ``arch``: the shipped bundle's rows, tiles and tables with every image replaced by the one
    built for ``arch`` (same relative path with the arch in its name, or the same path when the name carries none).

    An image with no build for ``arch`` is refused by name, except fat binaries that already hold ``arch`` code and the
    image keys in ``drop`` (removed from the bundle: the runner then has no module for them, and the row refuses those
    launches by name)."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, symlinks=False)
    table = json.loads((out / "schemes.json").read_text(encoding="utf-8"))
    images = dict(table["images"])
    done, kept, dropped = {}, {}, []
    for key, rel in sorted(images.items()):
        if key in drop:
            dropped.append(key)
            images.pop(key)
            continue
        new_rel = rel.replace(shipped_arch, arch)
        for cand in [built / new_rel, built / rel] + ([shared / new_rel, shared / rel] if shared else []):
            if cand.is_file():
                (out / new_rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cand, out / new_rel)
                if new_rel != rel and (out / rel).exists():
                    (out / rel).unlink()
                images[key] = new_rel
                done[key] = {"path": new_rel, "sha256": _sha256(out / new_rel), "from": str(cand)}
                break
        else:
            if rel.endswith((".fatbin", ".bin")):
                kept[key] = rel   # fat binaries carry every arch they were built for; tables are not code
                continue
            raise AdRowError(f"no {arch} build of bundle image {key} ({rel}) in {built}")
    table["images"] = images
    table.setdefault("arch_images", {})[arch] = {"built": str(built), "dropped": dropped}
    (out / "schemes.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and not p.name.startswith("SHA256SUMS"))
    (out / "SHA256SUMS").write_text("".join(f"{_sha256(p)}  ./{p.relative_to(out).as_posix()}\n" for p in files))
    return {"images": done, "kept_fatbins": kept, "dropped": dropped}


def refuse_arch_without_images(name: str, row: dict, arch: str, needed: set[str] | None = None,
                               card: str | None = None, need_mib: int | None = None) -> None:
    """The refusals of ``ensure_arch_images`` that need no asset root: the row names no image recipe for this card
    class, or it names one and there are no built images, no cached ones and no image builder.

    THE BREAKAGE THIS PREVENTS: on a card of compute capability 8.9 a fresh install fetched the row and the example
    plan (942 MB) and only then refused, on the example plan's card class; with a plan recorded there it would have
    refused after the recording, because the wheel ships no image builder (2026-11-11). A door calls this before it
    resolves the asset root."""
    shipped = row.get("image_archs") or [arch_of(cc) for cc in row.get("compute_capability", [])]
    if not shipped or arch in shipped:
        return
    spec = row.get("arch_images")
    words = runs_on_words(name, shipped, need_mib or row.get("need_free_mib"), arch, card)
    if not spec:
        raise AdRowError(f"{words}. (Its derivative images are compiled for {', '.join(shipped)}, and the row names no "
                         f"image recipe for {arch}.) Nothing was downloaded")
    image_set = spec["image_set"]
    keys = [k for k in list(spec.get("bundles", [])) + list(spec.get("files") or {}) if needed is None or k in needed]
    root = rows_cache().parent / "images" / image_set / arch
    receipt = root / "images.json"
    if receipt.is_file():
        try:
            if all(k in json.loads(receipt.read_text(encoding="utf-8"))["assets"] for k in keys):
                return
        except (ValueError, KeyError, TypeError):
            pass
    build_root = rows_cache().parent / "images" / spec.get("build_set", image_set) / arch
    if (build_root / "built" / ".complete.json").is_file():
        return
    if not (os.environ.get(IMAGE_BUILDER_ENV) or spec.get("builder")):
        raise AdRowError(f"{words}, or build this card's images: there are no {arch} images of image set {image_set} "
                         f"in {root} and no image builder ({IMAGE_BUILDER_ENV}, '{{arch}}' '{{out}}' filled in). "
                         "Nothing was downloaded")


def ensure_arch_images(table: dict, name: str, row: dict, arch: str, asset_root: Path,
                       needed: set[str] | None = None) -> dict[str, Path]:
    """Asset overrides (key -> path) carrying this card's images; {} when the row's shipped images are for it."""
    shipped = row.get("image_archs") or [arch_of(cc) for cc in row.get("compute_capability", [])]
    if not shipped or arch in shipped:
        return {}
    refuse_arch_without_images(name, row, arch, needed)
    spec = row["arch_images"]
    image_set = spec["image_set"]
    keys = [k for k in list(spec.get("bundles", [])) + list(spec.get("files") or {}) if needed is None or k in needed]
    root = rows_cache().parent / "images" / image_set / arch
    receipt = root / "images.json"
    # the compiled images are shared by every image set built from the same recipes (build_set); the assembled
    # bundles belong to the image set (their tables differ, e.g. a mesh-generic schemes.json)
    build_root = rows_cache().parent / "images" / spec.get("build_set", image_set) / arch
    if receipt.is_file():
        doc = json.loads(receipt.read_text(encoding="utf-8"))
        if all(k in doc["assets"] for k in keys):
            return {k: Path(doc["assets"][k]) for k in keys}
    built = build_root / "built"
    if not (built / ".complete.json").is_file():
        command = os.environ.get(IMAGE_BUILDER_ENV) or spec.get("builder")
        if not command:
            raise AdRowError(f"row {name}: no {arch} images of image set {image_set} in {root} and no image builder "
                             f"({IMAGE_BUILDER_ENV}, '{{arch}}' '{{out}}' filled in)")
        _run(command, {"arch": arch, "out": str(built), "build_set": spec.get("build_set", image_set),
                       "image_set": image_set}, build_root / "build.log", f"{arch} image build of {image_set}")
        _write_json(built / ".complete.json", {"arch": arch, "command": command})
    base = asset_root
    assets: dict[str, str] = {}
    report: dict = {}
    drop = tuple(spec.get("drop", ()))
    if receipt.is_file():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        assets.update(old.get("assets", {}))
        report.update(old.get("bundles", {}))
    for key in spec.get("bundles", []):
        if key not in keys or key in assets:
            continue
        src = Path(row["assets"][key])
        src = src if src.is_absolute() else base / src
        out = root / key
        # a bundle's own images (its P3 build, say) sit under built/<key>/ and win over the shared ones in built/
        own = built / key
        if key in spec.get("own_images", ()) and not own.is_dir():
            raise AdRowError(f"bundle {key} carries its own images (not the shared set) and the {arch} build has none "
                             f"({own}); its door refuses on this card until the image builder makes them")
        report[key] = assemble_bundle(src.resolve(), own if own.is_dir() else built, arch, out, drop=drop,
                                      shared=built if own.is_dir() else None)
        assets[key] = str(out)
    for key, rel in (spec.get("files") or {}).items():
        if key not in keys:
            continue
        cand = built / rel.replace("{arch}", arch)
        if not cand.is_file():
            raise AdRowError(f"no {arch} build of {key} ({cand})")
        assets[key] = str(cand)
    _write_json(receipt, {"schema": IMAGES_SCHEMA, "image_set": image_set, "arch": arch, "assets": assets,
                          "bundles": report})
    return {k: Path(assets[k]) for k in keys}
