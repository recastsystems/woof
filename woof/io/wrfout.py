"""Minimal WRF-style "wrfout" NetCDF writer.

``WrfoutWriter`` lays out WRF-ARW dimensions/staggering and stamps each
variable with the WRF attribute set (``FieldType``/``MemoryOrder``/
``description``/``units``/``stagger``) so standard wrfout tooling can read
the files.  ``state_frame`` builds the standard frame dict from a
``DomainState`` (winds, T = theta - 300, PH/MU perturbations, the
terrain-consistent PHB/MUB/HGT base fields, and QVAPOR/QCLOUD/QRAIN when
the state carries moisture).  Active Morrison, precipitation, snow, Noah,
and vertical-coordinate history state is included under its WRF Registry
name.  ``wrf_time_str`` formats the ``Times`` record.  Only ``state_frame``
touches the GPU (deferred cupy import) -- the writer itself is
CPU-importable.
"""
from __future__ import annotations

RADAR_INLOOP_VERSION = 2
from collections.abc import Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
import fnmatch
import json
import os
from fractions import Fraction
from pathlib import Path
import queue
import stat
import sys
import threading
import time
import traceback

import numpy as np
from woof.io.netcdf_serialization import netCDF4  # first use only (D-10)

from woof import perf_timing, progress_log, render_layout
from woof.wrf_exact import ENABLED as _WRF_EXACT, DIAGNOSTICS_ENABLED
from woof.config import NO_LAND_SURFACE_SOIL_LAYERS, soil_layer_count
from woof.io.classic_tape import (ClassicDim, ClassicTape, ClassicVariable,
                                   classic_attr_value)
from woof.io.wrf_output_schema import (
    HISTORY_FIELDS_BY_NETCDF_NAME, PHYSICS_SELECTOR_GLOBALS,
    RADIATION_CLDFRA_ATTRIBUTES,
    REGISTRY_VAR_META, SCHEME_OUTPUT_FIELDS, WRF_FIELD_TYPE_INTEGER,
    WRF_FIELD_TYPE_REAL,
)
from woof.io.sfire_schema import (SFIRE_FINE_FIELDS, SFIRE_CLASS_FIELDS,
                                   SFIRE_REGISTRY_FIELDS, SFIRE_PARTICLE_FIELDS)
from woof.supervisor import (_fsync_directory, fsync_file, quarantine_file,
                              replace_file_with_retry, unique_temp_path,
                              writing_progress)

from woof.io.history_layout import (
    CHEM_VAR_META as _CHEM_VAR_META,
    CORE_DIRECT_STATE_FIELDS, MOISTURE_STATE_FIELDS,
    _Z_STAGGERED_MASS_FIELDS,
    live_state_history_fields as _live_state_history_fields,
)

from woof.io.netcdf_serialization import NETCDF4_IO_LOCK
from woof.io.wrf_input_history_schema import (WRF_INPUT_HISTORY_ROWS,
    WRF_INITIALIZED_HISTORY_ROWS, WRF_GNU_RANDOM_SEED_WORDS)

_COMPLETION_ATTR = "GPUWM_WRITE_COMPLETE"
# netCDF4 releases the GIL around HDF5 calls, while the shipped HDF5 library
# is not thread-safe.  Domain D2H streams and staging remain independent; only
# each worker's create/write/close/reopen/publish netCDF session is serialized.
# The Rust engine needs no such serialization of its own. The ordinary
# validation path and explicit Python writer still use netCDF4, so the lock stays.
#
# THE LOCK IS NO LONGER THIS MODULE'S.  Guarding the writer against itself
# left it unguarded against every other netCDF4 READER in the process, and
# one of those killed real runs: see woof.io.netcdf_serialization for the
# measurement.  The name is kept so this module reads as it did.
_NETCDF4_IO_LOCK = NETCDF4_IO_LOCK
_ASYNC_WRITER_POLL_SECONDS = 0.05

#: Which library writes the product tape.  ``rust`` (the DEFAULT) is the
#: dependency-free classic writer at ``tools/rustwx/crates/netcdf-writer``
#: behind the :mod:`woof.io.nc_writer_bridge` ctypes seam, emitting the
#: CDF-2 (``NETCDF3_64BIT_OFFSET``) container.  ``python`` is the
#: netCDF4/HDF5 writer this module used through 2.4, emitting
#: ``NETCDF4_CLASSIC`` -- kept reachable as an EXPLICIT WORKAROUND only,
#: never selected silently.  Their equivalence on real frames is the
#: dual-write verification in ``tests/test_wrfout_dual_write.py`` and
#: ``tools/wrfout_dual_write.py``.
WRFOUT_WRITER_ENV = "WOOF_WRFOUT_WRITER"

_WRFOUT_ENGINES = ("rust", "python")


def resolve_wrfout_engine(engine: str | None = None) -> str:
    """The writer engine to use: explicit argument, else env, else rust."""
    value = engine if engine is not None else os.environ.get(
        WRFOUT_WRITER_ENV, "rust")
    value = str(value).strip().lower()
    if value not in _WRFOUT_ENGINES:
        raise ValueError(
            f"unknown wrfout writer engine {value!r} "
            f"(from {WRFOUT_WRITER_ENV!r} or the engine argument); "
            f"expected one of {_WRFOUT_ENGINES}.  'rust' is the default "
            "classic writer; 'python' is the netCDF4 workaround.")
    return value


#: How history frames are stored.  ``deflate`` (the DEFAULT, the project owner
#: 2026-10-28: "Lossless compression is the default") writes each frame as
#: a NetCDF-4 file whose chunks are byte-shuffled and deflated: every value
#: is the same word after decompression, and the file is about a third of
#: the classic one (4.3x smaller on a measured 1 km quick-look frame).  The
#: chunks are compressed on a pool of worker threads off the step loop
#: (woof.io.nc_writer_bridge.Netcdf4Writer), never by HDF5 on one thread.
#: ``none`` writes the classic CDF-2 frame as before, for readers that need
#: raw classic files (``--history-compression none``).
HISTORY_COMPRESSION_ENV = "GPUWM_HISTORY_COMPRESSION"
#: zlib level of compressed frames (1-9).  4 is where the measured frames
#: stop gaining (zstd 1/3/9 and zlib 4 within 10 % on every field).
HISTORY_COMPRESSION_LEVEL_ENV = "GPUWM_HISTORY_COMPRESSION_LEVEL"
#: Worker threads that compress frames (0 = every core, the default; the
#: workers run at a lower priority than the forecast).
HISTORY_COMPRESSION_THREADS_ENV = "GPUWM_HISTORY_COMPRESSION_THREADS"

_HISTORY_COMPRESSIONS = ("deflate", "none")


def resolve_history_compression(value: str | None = None) -> str:
    """The history storage: explicit argument, else env, else deflate."""
    chosen = value if value is not None else os.environ.get(
        HISTORY_COMPRESSION_ENV, "deflate")
    chosen = str(chosen).strip().lower() or "deflate"
    if chosen not in _HISTORY_COMPRESSIONS:
        raise ValueError(
            f"unknown history compression {chosen!r} (from "
            f"{HISTORY_COMPRESSION_ENV!r} or --history-compression); "
            f"expected one of {_HISTORY_COMPRESSIONS}: 'deflate' (default) "
            "writes lossless compressed NetCDF-4 frames, 'none' classic "
            "CDF-2 frames")
    return chosen


def _history_netcdf4_options() -> dict:
    level = int(os.environ.get(HISTORY_COMPRESSION_LEVEL_ENV, "4") or 4)
    threads = int(os.environ.get(HISTORY_COMPRESSION_THREADS_ENV, "0") or 0)
    return {"deflate": max(1, min(level, 9)), "shuffle": True,
            "threads": max(0, threads)}


#: The classic-container spelling of an attribute value, and the
#: netCDF4-shaped facade that puts this tape on the Rust seam.  Both live
#: in :mod:`woof.io.classic_tape` because the wrfinput/wrfbdy export
#: writes through the same seam: two copies of this facade is how two
#: products come to disagree about what the classic writer does.
_classic_attr_value = classic_attr_value
_ClassicDim = ClassicDim
_ClassicTapeVariable = ClassicVariable

#: What a caller of THIS tape should do instead when the header is
#: already on disk.
_FROZEN_REMEDY = (
    "Declare it before the first frame (or pass field_schema), or use "
    f"the netCDF4 workaround {WRFOUT_WRITER_ENV}=python.")


class _ClassicTape(ClassicTape):
    """The history tape's :class:`~woof.io.classic_tape.ClassicTape`.

    It differs from a bare one in exactly two declared ways: it stamps
    ``GPUWM_WRITE_COMPLETE`` into the header (last, where the netCDF4
    path's close-time ``setncattr`` also lands it), and its
    header-frozen refusal names this writer's own escapes.
    """

    def __init__(self, path, *, container: str = "cdf2",
                 netcdf4_options: dict | None = None):
        super().__init__(path, container=container,
                         completion_attr=_COMPLETION_ATTR,
                         frozen_remedy=_FROZEN_REMEDY,
                         netcdf4_options=netcdf4_options)


def _stringify_and_clear_exception_tracebacks(exc: BaseException) -> str:
    """Return diagnostics without retaining any exception-chain frames."""
    rendered = "".join(traceback.format_exception(
        type(exc), exc, exc.__traceback__))
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        current.__traceback__ = None
    return rendered


#: WRF Registry metadata ``name -> (description, units)``, read from
#: the schema module that owns it.  The table moved there so
#: ``woof.io.history_selection`` can read the NAMES at
#: config-resolution time without importing this writer -- and the
#: supervisor, netCDF4 and runtime behind it.  Kept bound here under
#: the name the writer has always used.
_VAR_META = REGISTRY_VAR_META
#: Description and units of the chem outputs a frame has carried, filled by
#: :func:`_live_state_history_fields` from the chem table (the rows'
#: long_name/units, the diagnostics' description/units).  Empty in every
#: chem-off process, so no default variable's attributes can move.

#: Staggered dimension -> WRF ``stagger`` attribute value.  WRF appends
#: ``_stag`` to a dimension's dataset name exactly when the field is staggered
#: on that axis (``tools/gen_wrf_io.c:138,187,213``), which is why the snow
#: and snow+soil axes below are ``*_stag`` and not the bare dimspec names:
#: every Noah-MP field on them is declared ``Z``-staggered.
_STAGGER = {"west_east_stag": "X", "south_north_stag": "Y",
            "bottom_top_stag": "Z", "soil_layers_stag": "Z",
            "snow_layers_stag": "Z", "snso_layers_stag": "Z", "seed_dim_stag": "Z"}

#: Noah-MP snow-stack history fields, keyed by name for the same reason
#: TSLB/SMOIS/SH2O are: their leading extent is a scheme's own vertical axis,
#: which collides with ``bottom_top`` for a shallow model column.  The
#: dimension names are WRF's (``Registry/registry.dimspec:53,55``: ``snly`` ->
#: ``snow_layers``, ``snsl`` -> ``snso_layers``, each ``_stag``-suffixed by
#: the declared ``Z`` staggering), so a reader that knows WRF knows these.
#: The names are the external ones from the output schema, never the Registry
#: symbols -- ``TSNOXY`` is not a name any WRF ever wrote.
_SNOW_LAYER_FIELDS = frozenset(
    SCHEME_OUTPUT_FIELDS[key].netcdf_name
    for key in ("tsnoxy", "snicexy", "snliqxy"))
_SNSO_LAYER_FIELDS = frozenset({SCHEME_OUTPUT_FIELDS["zsnsoxy"].netcdf_name})

#: MYNN's three ``ikj`` carriers that WRF declares ``Z``-staggered, so WRF
#: writes them on ``bottom_top_stag`` with one more level than the mass
#: column.  woof computes exactly the ``nz`` mass levels the MYNN solver
#: fills; WRF's own array is dimensioned ``kms:kme`` and its PBL driver never
#: writes the extra interface entry either, leaving the Registry cold value
#: zero there.  The writer lifts these onto WRF's axis and pads with that same
#: zero, so a reader that knows WRF's EL_PBL shape receives WRF's EL_PBL
#: shape -- rather than an ``nz``-level array under a name whose declared
#: schema says ``nz + 1``.


#: Every history field whose leading extent is the SOIL axis.  TSLB, SMOIS and
#: SH2O are generic; SMFR3D and KEEPFR3DFLAG are RUC's own Registry package
#: line (``Registry.EM_COMMON:3147``) and are written only under
#: ``sf_surface_physics=3``.  They belong here for exactly the reason the snow
#: sets above exist: ``_dims_for``'s shape table is keyed on ``(nz, ny, nx)``,
#: so a soil array is unroutable by shape alone and a nine-level soil field
#: over a twenty-level column raises ``KeyError`` rather than picking a wrong
#: axis.  That is what it did before these two names were added.
_SOIL_LAYER_FIELDS = frozenset({"TSLB", "SMOIS", "SH2O", "SMCREL",
                                "SMFR3D", "KEEPFR3DFLAG", "TVGL_URB", "SMG_URB"})

#: The land-use table identity stamped on every history file.
#:
#: ``ISOILWATER`` is the water category of the SOIL table, and it is 14 for
#: every MODIS/Noah run woof makes -- but it was the one member of the group
#: this writer left out, while stock WRF writes it in every wrfout
#: (``share/output_wrf.F:973``, an explicit ``wrf_put_dom_ti_integer`` rather
#: than a Registry ``h`` flag, which is why it is easy to miss).  Its absence
#: was not inert: ``woof.offline_child.read_child_surface_state`` reads the
#: four attributes below as REQUIRED evidence precisely so category semantics
#: are never assumed, and then quietly assumed 14 for this one because woof's
#: own files never carried it.  Writing it makes the file say what the reader
#: had to guess.
#:
#: ``NUM_LAND_CAT`` is the same argument one attribute later.  Stock WRF
#: writes it in every history file (``share/output_wrf.F``, from
#: ``grid%num_land_cat``, which real.exe takes from the met_em global that
#: geogrid derived as ``category_max - category_min + 1`` of the dominant
#: land-use field), and the pinned v4.6.1 reference file in this repository
#: carries ``NUM_LAND_CAT = 21`` beside exactly the ``MMINLU``/``ISWATER``/
#: ``ISLAKE``/``ISICE``/``ISURBAN`` values below.  Without it a consumer
#: handed ``MMINLU = "MODIFIED_IGBP_MODIS_NOAH"`` and ``LU_INDEX`` has to
#: ASSUME 21 categories to size a table -- the guess this group exists to
#: retire.  The value here is the legacy MODIS/Noah identity the rest of the
#: group already states; a run with a resolved land-use dataset overrides it
#: from that dataset's own index (``GeogSelection.landuse_global_attrs``),
#: which is what keeps it evidence rather than a constant.
from woof.static.geog import DEFAULT_LANDUSE_ATTRS as _DEFAULT_LANDUSE_ATTRS


def _wrf_patch_extent_attrs(nx: int, ny: int, nz: int) -> dict[str, int]:
    """WRF's ``*_PATCH_*`` extent group for one undecomposed patch.

    Stock WRF writes twelve of these into every history file, one triple of
    (start, unstaggered end, staggered end) per axis, describing the slab of
    the domain THIS file holds.  A serial writer holds all of it, so start is
    1 on every axis (WRF's indices are 1-based), the unstaggered end is the
    mass-point count and the staggered end is one past it -- the same
    relationship ``WEST-EAST_GRID_DIMENSION = nx + 1`` already states, said
    the way the stock tool chain reads it.
    """
    return {
        "WEST-EAST_PATCH_START_UNSTAG": 1,
        "WEST-EAST_PATCH_END_UNSTAG": nx,
        "WEST-EAST_PATCH_START_STAG": 1,
        "WEST-EAST_PATCH_END_STAG": nx + 1,
        "SOUTH-NORTH_PATCH_START_UNSTAG": 1,
        "SOUTH-NORTH_PATCH_END_UNSTAG": ny,
        "SOUTH-NORTH_PATCH_START_STAG": 1,
        "SOUTH-NORTH_PATCH_END_STAG": ny + 1,
        "BOTTOM-TOP_PATCH_START_UNSTAG": 1,
        "BOTTOM-TOP_PATCH_END_UNSTAG": nz,
        "BOTTOM-TOP_PATCH_START_STAG": 1,
        "BOTTOM-TOP_PATCH_END_STAG": nz + 1,
    }


def _producer_version() -> str:
    """The release that wrote this file -- the one that EXECUTED.

    ``GPUWM_VERSION`` is the only version this product stamps onto an
    output artifact, so it is the number a reader quotes back months
    later.  It used to be ``woof.__version__``, which is a claim made
    by distribution METADATA rather than by the code doing the writing:
    ``importlib.metadata`` is asked for the version of the distribution
    NAMED woof, and on a box with a stale editable install that answers
    for a tree which is not the one running.  A file stamped that way
    names a release it was not written by, and nothing downstream can
    tell.

    :func:`woof.provenance_gate.executing_version` prefers the running
    code's own declaration and falls back through the metadata to the
    accurate ``0+unknown``, so the attribute describes the bytes that
    wrote the file.  On a plain wheel install the two are identical by
    construction -- pip wrote the code and the metadata together -- so
    nothing about an ordinary install's output moves.

    Imported at call time rather than at module import so the writer
    stays importable from a source tree that was never installed, and
    fully defensive: a history write must not fail over a label.
    """
    try:
        from woof.provenance_gate import executing_version

        return str(executing_version())
    except Exception:                                   # noqa: BLE001
        from woof import __version__

        return str(__version__)


def wrfout_filename(valid_time, domain_id: int = 1) -> str:
    """Colon-free, second-complete WRF history filename.

    A sub-second instant is refused rather than formatted.  The name carries
    whole seconds and the publisher replaces an existing file, so three legal
    quarter-second frames used to collapse onto one name and silently
    overwrite each other -- the second and third frames of that run simply
    ceased to exist, with no exception anywhere.  Losing output is not an
    acceptable answer to an unsupported cadence; refusing it is.
    """
    if (isinstance(domain_id, bool) or not isinstance(domain_id, int)
            or domain_id < 1):
        raise ValueError(
            f"domain_id must be a positive integer, got {domain_id!r}")
    if getattr(valid_time, "microsecond", 0):
        raise ValueError(
            f"wrfout valid time {valid_time!r} is not on a whole second; "
            "history filenames and the Times record carry whole seconds "
            "only, so distinct sub-second instants would alias onto one "
            "file and the later frame would replace the earlier one")
    return valid_time.strftime(
        f"wrfout_d{domain_id:02d}_%Y-%m-%d_%H_%M_%S")


#: Where a quick-look frame lands: its own folder beside the history tape.
#: The file keeps the wrfout name, because the renderer recognises a WRF
#: history file by it ("unreadable wrfout (not a raw WRF or post-processed
#: NetCDF file this build recognises)" for any other name); the folder is
#: what tells a subset frame from a full one.
QUICKLOOK_DIRECTORY = "quicklook"


def quicklook_path(base_dir, valid_time, domain_id: int):
    """``<base_dir>/quicklook/wrfout_dNN_<time>``: one quick-look frame."""
    return Path(base_dir) / QUICKLOOK_DIRECTORY / wrfout_filename(valid_time, domain_id)


def is_quicklook_frame(path) -> bool:
    """Whether a history path is a quick-look subset frame (by its folder)."""
    return Path(path).parent.name == QUICKLOOK_DIRECTORY


def quicklook_selection(node, ticks: int):
    """The quick-look subset when this frame is a quick-look one, else None.

    A domain with ``quicklook = {...}`` (DomainConfig.full_history_interval_s
    set) rings its history alarm at the quick-look cadence; a frame is FULL
    when its time since the domain's first history instant is a whole
    number of full intervals, and a quick-look frame otherwise.
    """
    cfg = getattr(node, "cfg", None)
    full = getattr(cfg, "full_history_interval_s", None)
    selection = getattr(cfg, "quicklook_output", None)
    if full is None or selection is None:
        return None
    spec = node.clock.spec
    tick_den = int(node.clock.tick_den)
    elapsed = Fraction(int(ticks) - int(spec.start_ticks)
                       - int(spec.history_begin_ticks), tick_den)
    return None if (elapsed / Fraction(str(full))).denominator == 1 else selection


def wrf_physics_selector_attrs(run) -> dict[str, np.int32]:
    """WRF's physics-selector globals, from the resolved woof configuration.

    Stock WRF stamps every physics selector into every history file, and
    that is what lets a reader tell an *absent* scheme from an *inactive*
    one.  Without them a wrfout whose ``RAINSH`` is all zeros is ambiguous:
    a shallow-cumulus scheme may have run and produced nothing, or none may
    exist.  ``SHCU_PHYSICS=0`` is the difference.

    Radiation goes through ``radiation_scheme_ids`` rather than reading the
    two config fields, because woof's ``-1/-1`` is a legacy sentinel
    meaning "use the aggregate ``ra_physics``", not a WRF scheme id.  The
    resolver is the repository's own authority for what actually ran, and
    writing the raw sentinel would put a number in the file that no WRF
    selector has.
    """
    from woof.config import radiation_scheme_ids

    lw, sw = radiation_scheme_ids(run)
    attrs: dict[str, np.int32] = {}
    for selector in PHYSICS_SELECTOR_GLOBALS:
        if selector.source == "config":
            value = getattr(run, selector.run_config_field)
        elif selector.source == "radiation_lw":
            value = lw
        elif selector.source == "radiation_sw":
            value = sw
        elif selector.source == "unimplemented":
            # woof has no such scheme to select, so WRF's "off" value is
            # the resolved truth about this run, not a placeholder.
            value = 0
        else:
            raise ValueError(
                f"unknown selector source {selector.source!r} for "
                f"{selector.name} ({selector.registry})")
        attrs[selector.name] = np.int32(value)
    return attrs


def wrf_global_attrs(
        grid, start_time, *, landuse_attrs=None, grid_id=None,
        parent_id=None, i_parent_start=None, j_parent_start=None,
        parent_grid_ratio=None, dt=None, hybrid_opt=None, etac=None,
        run=None, simulation_start_time=None,
        ) -> dict[str, object]:
    """WRF projection/pole/land-use globals derived from case inputs.

    ``grid`` carries the per-domain center and mother-domain center,
    plus its projection identity (``wrf_map_proj``/``map_proj_char``
    per the WRF convention 1=lambert, 2=polar stereographic,
    3=mercator; grids without the attributes -- legacy callers -- keep
    the Lambert identity).  ``landuse_attrs`` comes from the selected
    WPS_GEOG land-use index; omission preserves the established
    MODIS/Noah legacy identity.  Domain topology and vertical identity
    are optional for generic/idealized files, but the real-case runtime
    supplies the complete groups together.  POLE_LAT/POLE_LON stay
    90/0: WPS writes those constants for every non-lat-lon projection.

    ``start_time`` is THIS file's own start and becomes ``START_DATE``;
    ``simulation_start_time`` is the whole simulation's and becomes
    ``SIMULATION_START_DATE``.  They are two different facts in WRF, and
    they differ exactly when a nest declares a delayed start.
    ``share/output_wrf.F:343-348`` reads ``start_*`` at ``grid%id`` for
    ``START_DATE``, while ``:361-376`` reads ``simulation_start_*`` at
    namelist index **1** for ``SIMULATION_START_DATE`` on every history
    file, and ``share/set_timekeeping.F:379-388`` sets those index-1
    values once, ``IF ( grid%id .EQ. head_grid%id )``.  So WRF's
    ``SIMULATION_START_DATE`` is the head grid's start, identical on every
    domain of the run.  Writing the domain's own value there labelled a
    d02 frame F+03 where the d01 frame of the same instant was F+09,
    because ``rw_wrfbatch`` reads this attribute first as the run origin
    and measures every product's lead from it
    (``woof/io/surface_wrfout.py`` ``ORIGIN_ATTRS``).

    ``simulation_start_time=None`` means "this file starts the
    simulation", which every single-domain and idealized caller is, and
    reproduces the previous single-date behaviour byte for byte.
    """
    # Every projection global below is NC_FLOAT, not NC_DOUBLE.  A Python
    # float would enter netCDF4 as a double, and stock WRF writes all of these
    # single-precision (verified on the group's v4.6.1 wrfout: DX, DY,
    # TRUELAT1/2, STAND_LON, CEN_LAT/LON, MOAD_CEN_LAT, POLE_LAT/LON and DT
    # are every one of them float32).  Readers tolerate the widening, but a
    # file that claims WRF's schema should not differ from WRF's schema.
    attrs = {
        "MAP_PROJ": int(getattr(grid, "wrf_map_proj", 1)),
        "MAP_PROJ_CHAR": str(getattr(grid, "map_proj_char",
                                     "Lambert Conformal")),
        "TRUELAT1": np.float32(grid.truelat1),
        "TRUELAT2": np.float32(grid.truelat2),
        "STAND_LON": np.float32(grid.stand_lon),
        "MOAD_CEN_LAT": np.float32(
            getattr(grid, "moad_cen_lat", grid.ref_lat)),
        "CEN_LAT": np.float32(getattr(grid, "cen_lat", grid.ref_lat)),
        "CEN_LON": np.float32(getattr(grid, "cen_lon", grid.ref_lon)),
        "POLE_LAT": np.float32(90.0), "POLE_LON": np.float32(0.0),
        "SIMULATION_START_DATE": (
            start_time if simulation_start_time is None
            else simulation_start_time).strftime("%Y-%m-%d_%H:%M:%S"),
        "START_DATE": start_time.strftime("%Y-%m-%d_%H:%M:%S"),
        # The same instant as ``START_DATE``, in the numeric spelling
        # ARWpost and the older NCL post-processing chain read to place a
        # file in time.  WRF derives all three in ``set_timekeeping`` from
        # the domain's own start (``nl_set_gmt``/``nl_set_julyr``/
        # ``nl_set_julday``), which is the date this writer already stamps
        # into ``START_DATE``, so the numeric group and the string group
        # cannot disagree.  GMT is NC_FLOAT and the two day/year counts are
        # NC_INT on the v4.6.1 reference file.
        "GMT": np.float32(start_time.hour + start_time.minute / 60.0
                          + start_time.second / 3600.0),
        "JULYR": np.int32(start_time.year),
        "JULDAY": np.int32(start_time.timetuple().tm_yday),
        "GRIDTYPE": "C", **_DEFAULT_LANDUSE_ATTRS,
    }
    if landuse_attrs is not None:
        attrs.update(dict(landuse_attrs))
    if (run is not None and getattr(run, "sf_urban_physics", 0) > 0
            and attrs["MMINLU"] == "MODIFIED_IGBP_MODIS_NOAH"):
        from woof.core.landuse import load_landuse_table
        attrs["NUM_LAND_CAT"] = load_landuse_table().lucats
    domain_values = {
        "GRID_ID": grid_id,
        "PARENT_ID": parent_id,
        "I_PARENT_START": i_parent_start,
        "J_PARENT_START": j_parent_start,
        "PARENT_GRID_RATIO": parent_grid_ratio,
    }
    supplied_domain = [value is not None for value in domain_values.values()]
    if any(supplied_domain) and not all(supplied_domain):
        missing = [name for name, value in domain_values.items()
                   if value is None]
        raise ValueError(
            "WRF domain topology metadata must be supplied together; "
            f"missing {missing}")
    if all(supplied_domain):
        attrs.update({name: np.int32(value)
                      for name, value in domain_values.items()})
    if dt is not None:
        if not np.isfinite(dt) or float(dt) <= 0.0:
            raise ValueError(f"WRF output DT must be finite and > 0, got {dt}")
        attrs["DT"] = np.float32(dt)
    if (hybrid_opt is None) != (etac is None):
        raise ValueError(
            "WRF vertical metadata hybrid_opt and etac must be supplied "
            "together")
    if hybrid_opt is not None:
        attrs["HYBRID_OPT"] = np.int32(hybrid_opt)
        attrs["ETAC"] = np.float32(etac)
    if run is not None:
        attrs.update(wrf_physics_selector_attrs(run))
        attrs["IFIRE"] = np.int32(getattr(run, "ifire", 0))
        if int(getattr(run, "ifire", 0)) == 2:
            attrs.update(IFIRE=np.int32(2), SR_X=np.int32(run.sr_x),
                         SR_Y=np.int32(run.sr_y))
    return attrs


#: WRF's global date format, the one ``START_DATE`` and
#: ``SIMULATION_START_DATE`` are written in
#: (``share/output_wrf.F:352-376`` of the v4.6.1 source: the
#: ``'(I4.4,"-",I2.2,"-",I2.2,"_",I2.2,":",I2.2,":",I2.2)'`` FORMAT).
#: Provenance dates in a wrfout are written the same way, so a reader
#: parsing dates out of this file needs one parser, not two.  The JSON
#: receipts keep ISO-8601 with the trailing ``Z``; they are documents, not
#: wrfout files, and their convention is unchanged.
_WRF_GLOBAL_DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"

#: The receipts' date format, which is what the preparation proof's
#: initial-condition block is written in.
_RECEIPT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Version tag stamped into every provenance-carrying wrfout, so a
#: consumer can key on a contract rather than on the presence of a name.
WRFOUT_INITIAL_CONDITION_SCHEMA = "gpuwm-wrfout-initial-condition-v1"

#: The complete set of global attributes that record WHAT the initial
#: condition was, as opposed to WHEN the model clock started.
#:
#: DOCUMENTED DIVERGENCE FROM WRF.  WRF v4.6.1 has no convention for
#: this.  ``share/output_wrf.F`` writes exactly two date globals --
#: ``START_DATE`` (this file's own start) and ``SIMULATION_START_DATE``
#: (the simulation's, held across restarts) -- and both are model-clock
#: times; the only provenance-shaped global it writes at all is
#: ``FLAG_RESTART`` on a restart file (output_wrf.F:379-381).  Nothing
#: upstream carries it either: metgrid's ``met_em`` globals are geometry,
#: land-use identity and ``FLAG_*`` presence bits, with no statement of
#: which cycle or which lead the fields came from (verified against the
#: reference bundle's own ``met_em.d01`` and ``wrfout_d01`` files).  So
#: WRF's convention is FOLLOWED where it exists -- ``START_DATE`` and
#: ``SIMULATION_START_DATE`` keep their WRF meaning untouched (which for a
#: delayed-start nest means two DIFFERENT dates; see
#: :func:`wrf_global_attrs`), and these
#: names are SCREAMING_SNAKE NC_CHAR/NC_INT globals like WRF's own -- and
#: the gap is filled in the ``GPUWM_`` namespace this writer already uses
#: for ``GPUWM_VERSION``/``GPUWM_FEEDBACK``.
#:
#: The names are the receipt's field names, uppercased and prefixed, so
#: the wrfout and ``report.json`` are readable against each other without
#: a mapping table.
INITIAL_CONDITION_GLOBAL_ATTRS = (
    "GPUWM_INITIAL_CONDITION_SCHEMA",
    "GPUWM_INITIAL_CONDITION_KIND",
    "GPUWM_INITIAL_CONDITION_SOURCE",
    "GPUWM_INITIAL_CONDITION_CYCLE",
    "GPUWM_INITIAL_FORECAST_LEAD_HOURS",
    "GPUWM_INITIAL_CONDITION_GENERATING_PROCESS_ID",
    "GPUWM_INITIAL_CONDITION_MODEL_START_DATE",
    "GPUWM_INITIAL_CONDITION_STATEMENT",
)


def _provenance_date(value, field: str) -> datetime:
    """One receipt date, parsed rather than transcribed."""

    if not isinstance(value, str):
        raise ValueError(
            f"initial-condition provenance {field} is not a date string")
    try:
        return datetime.strptime(value, _RECEIPT_DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"initial-condition provenance {field} {value!r} is not a "
            "receipt timestamp") from exc


def initial_condition_global_attrs(
        provenance, *, source: str | None = None) -> dict[str, object]:
    """wrfout globals recording what the initial state WAS.

    ``provenance`` is the preparation receipt's ``initial_condition``
    block (schema ``gpuwm-gfs-initial-condition-provenance-v1``).
    ``None`` returns ``{}``: a route whose preparation publishes no such
    receipt writes no provenance attribute at all, and a reader must
    treat the absence as UNKNOWN.  Absence is deliberately not spelled
    "analysis" -- that is the exact relabelling this contract exists to
    prevent, and pre-1.4.1 files are indistinguishable from it.

    Every field published here is READ and cross-checked, never copied:
    the kind, the generating-process id and the lead must agree with each
    other, and cycle + lead must compose to the declared model start.  A
    block that fails any of those is refused rather than transcribed, so
    a wrfout cannot carry a statement its own fields contradict.
    """

    if provenance is None:
        return {}
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "initial-condition provenance is malformed")

    lead = provenance.get("initial_forecast_lead_hours")
    if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
        raise ValueError(
            "initial-condition provenance declares an unreadable initial "
            "forecast lead")
    analysis = lead == 0
    kind = provenance.get("initial_condition_kind")
    process = provenance.get("forecast_generating_process_id")
    if (kind != ("analysis" if analysis else "forecast")
            or process != (81 if analysis else 96)):
        raise ValueError(
            f"initial-condition provenance labels forecast lead f{lead:03d} "
            f"as {kind!r} (process {process!r}); a forecast lead is not an "
            "analysis")

    cycle = _provenance_date(provenance.get("cycle"), "cycle")
    start = _provenance_date(
        provenance.get("model_start_time"), "model start time")
    if cycle + timedelta(hours=lead) != start:
        raise ValueError(
            f"initial-condition provenance does not compose: cycle "
            f"{cycle:%Y-%m-%d %H:%M:%S} + f{lead:03d} is not the declared "
            f"model start {start:%Y-%m-%d %H:%M:%S}")
    statement = provenance.get("statement")
    if not isinstance(statement, str) or not statement:
        raise ValueError(
            "initial-condition provenance carries no statement")

    return {
        "GPUWM_INITIAL_CONDITION_SCHEMA": WRFOUT_INITIAL_CONDITION_SCHEMA,
        "GPUWM_INITIAL_CONDITION_KIND": str(kind),
        # The driving model, in the spelling its own front door uses.
        # Unknown only for a caller that declared no source; the cycle is
        # not self-describing without it.
        "GPUWM_INITIAL_CONDITION_SOURCE": (
            "unknown" if source is None else str(source).upper()),
        "GPUWM_INITIAL_CONDITION_CYCLE": cycle.strftime(
            _WRF_GLOBAL_DATE_FORMAT),
        "GPUWM_INITIAL_FORECAST_LEAD_HOURS": np.int32(lead),
        "GPUWM_INITIAL_CONDITION_GENERATING_PROCESS_ID": np.int32(process),
        "GPUWM_INITIAL_CONDITION_MODEL_START_DATE": start.strftime(
            _WRF_GLOBAL_DATE_FORMAT),
        "GPUWM_INITIAL_CONDITION_STATEMENT": statement,
    }


#: The idealized cases' synthetic epoch.  Held here rather than formatted
#: inline because the month and year are no longer literals in the output:
#: they roll over.
_IDEALIZED_EPOCH = datetime(1, 1, 1)


def wrf_time_str(t_s: float) -> str:
    """WRF ``Times`` string for ``t_s`` seconds after the idealized epoch.

    Calendar arithmetic, not field arithmetic.  The previous form advanced
    only the day field and hard-coded the year and month, so a thirty-one
    day integration emitted ``0001-01-32`` and a year-long one emitted
    ``0001-01-366`` -- syntactically invalid ``Times`` strings that no WRF
    reader can parse, produced silently by a helper five idealized
    verification cases use.

    The epoch stays year 1: it is the frozen idealized-case contract, and
    moving it would move every fixture that depends on it.  Callers should
    know that year 1 is outside NumPy's ``datetime64[ns]`` range, so
    wrf-python -- which converts the parsed ``Times`` sequence to
    nanoseconds -- wraps such a date to somewhere in the 1700s.  That is a
    limitation of the synthetic epoch, not of the string: the string is now
    a real date either way, and the real-case path (which is what carries a
    wrf-python compatibility promise) has always used real dates.
    """
    seconds = int(round(t_s))
    if seconds < 0:
        raise ValueError(
            f"idealized Times seconds must be non-negative, got {t_s!r}")
    try:
        valid = _IDEALIZED_EPOCH + timedelta(seconds=seconds)
    except OverflowError as exc:
        raise ValueError(
            f"idealized Times seconds {t_s!r} overflows the calendar past "
            "year 9999") from exc
    # The year is formatted explicitly, not with ``%Y``.  glibc's and
    # MSVCRT's strftime both emit ``%Y`` UNPADDED, so year 1 came out as
    # ``1-01-01_00:00:00`` -- 16 characters, not 19.  ``Times`` is a
    # fixed-width ``(Time, DateStrLen=19)`` char array -- WrfoutWriter's
    # own ``createDimension("DateStrLen", 19)`` and
    # ``createVariable("Times", "S1", ("Time", "DateStrLen"))`` -- and
    # ``write_frame`` null-pads a short record to 19 with
    # ``encoded.ljust(19, b"\x00")``, so the file
    # carried ``1-01-01_00:00:00\0\0\0``: a reader taking WRF's fixed
    # YYYY-MM-DD_HH:MM:SS fields off that gets year ``1-01``, month ``01``
    # from what is really the day, and a NUL inside the seconds field.
    # Every other conversion here (%m %d %H %M %S) is zero-padded to two
    # digits by both C libraries, so only the year needs spelling out.
    return f"{valid.year:04d}-" + valid.strftime("%m-%d_%H:%M:%S")


def _driver_refreshes_psfc(state) -> bool:
    """Is ``state.physics.fields["psfc"]`` a computed surface pressure?

    A physics driver allocates ``psfc`` unconditionally but refreshes it
    from ``p_interface[0]`` only inside the surface/PBL cadence block
    (``woof/core/physics.py``, guarded by ``self.surface_enabled``).  A
    microphysics-only or dycore-plus-radiation composition therefore
    carries the allocation seed -- 100000 Pa -- for the whole forecast,
    and publishing it wrote a 150 hPa fabrication into wrfout from the
    most ordinary idealized run there is.  Asking whether the surface is
    on, rather than whether a driver exists, hands those compositions to
    the diagnostic extrapolation that already sits one branch below and
    is the same answer WRF's ``phy_prep`` computes.

    A driver-shaped object with no ``surface_enabled`` attribute reads as
    "no refresh", which routes to the computed branch: the safe side.
    """
    physics = getattr(state, "physics", None)
    return physics is not None and bool(
        getattr(physics, "surface_enabled", False))


def _state_device_context(state):
    """Select the state card without requiring CUDA for host-only templates."""
    device = getattr(getattr(state, "mup", None), "device", None)
    return device if getattr(device, "id", None) is not None else nullcontext()


def _domain_output_device(state):
    """A host shell keeps its run's card even after its arrays are freed."""
    streamed = getattr(state, "_streamed_domain", None)
    run = getattr(streamed, "_run", None)
    devices = getattr(run, "devices", ())
    if devices:
        return int(devices[0])
    if getattr(run, "_device_id", None) is not None:
        return int(run._device_id)
    return getattr(getattr(getattr(state, "mup", None), "device", None), "id", None)


def state_frame(
        state, *, include_diagnostic_pressure: bool = False
) -> dict[str, np.ndarray]:
    """Standard wrfout frame from a ``DomainState``, host float32 arrays.

    Winds ``U/V/W``, ``T`` (theta - 300, the WRF perturbation convention),
    the ``PH``/``MU`` perturbations, and the terrain-consistent base
    fields: per-column 3-D ``PHB`` (a flat base state's 1-D column is
    broadcast so the on-disk layout is identical with and without
    terrain), ``MUB``, and the terrain height ``HGT``.  States carrying
    moisture add ``QVAPOR``/``QCLOUD``/``QRAIN`` and any live Morrison
    mass/number moments.  P_TOP/ZNU/ZNW plus attached precipitation, snow,
    and Noah history arrays are also carried.  The frozen idealized-case
    contract omits diagnostic pressure fields; callers that require WRF
    tooling compatibility can opt in to ``P``/``PB``/``PSFC`` with
    ``include_diagnostic_pressure=True`` (the state's pressure must be
    diagnosed -- run ``update_diagnostics`` first on a fresh init).
    """
    import cupy as cp  # deferred: the writer itself stays CPU-importable

    with _state_device_context(state):
        if getattr(state, "_moving_frame_velocity", None):
            from woof.moving_frame import initialize_moving_frame
            initialize_moving_frame(state)
        ny, nx = state.mup.shape
        phb = cp.asnumpy(state.phb)
        if phb.ndim == 1:                     # flat base state: broadcast column
            phb = np.ascontiguousarray(
                np.broadcast_to(phb[:, None, None], (phb.size, ny, nx)))
        start_t = _wrf_start_history_t(state)
        if start_t is not None and not isinstance(start_t, np.ndarray):
            start_t = cp.asnumpy(start_t)
        fields = {
            "T": (start_t if start_t is not None
                  else cp.asnumpy(state.thp) if _WRF_EXACT
                  else cp.asnumpy(state.total_theta()) - np.float32(300.0)),
            **{name: cp.asnumpy(getattr(state, attribute))
               for name, attribute in CORE_DIRECT_STATE_FIELDS.items()},
            "PHB": phb,
            "MUB": cp.asnumpy(state.mub2d),
            "HGT": cp.asnumpy(state.ht),
        }
        # WOOF advances dry perturbation theta, the USE_THETA_M=0 record.
        fields["THM"] = (fields["T"] if start_t is None
                         else cp.asnumpy(state.thp))
        if include_diagnostic_pressure:
            pb = state.pb
            pb3 = pb if pb.ndim == 3 else pb[:, None, None]
            fields["P"] = cp.asnumpy(state.p_perturbation if DIAGNOSTICS_ENABLED
                                     else state.p - pb3)
            # Broadcast on the host: a state prepared on the CPU carries numpy
            # arrays, and cp.broadcast_to refuses those where cp.asnumpy does
            # not.  Same bytes for a device state.
            fields["PB"] = np.ascontiguousarray(
                np.broadcast_to(cp.asnumpy(pb3), tuple(state.p.shape)))
            if _driver_refreshes_psfc(state):
                fields["PSFC"] = cp.asnumpy(state.physics.fields["psfc"])
            elif getattr(state, "p_top", None) is not None:
                # Without a physics driver, diagnose PSFC the way WRF's
                # phy_prep extrapolates the full (moist) pressure to the
                # surface in z (module_big_step_utilities_em.F:5566-5578).
                # The previous dry form (total_mu + p_top) understates a
                # moist column's PSFC by the column water weight.
                from woof.core import constants as c
                phb3 = (state.phb[:, None, None]
                        if state.phb.ndim == 1 else state.phb)
                z_if = (phb3 + state.php) / np.float32(c.G)
                z_mid = 0.5 * (z_if[:-1] + z_if[1:])
                w1 = (z_if[0] - z_mid[1]) / (z_mid[0] - z_mid[1])
                fields["PSFC"] = cp.asnumpy(
                    w1 * state.p[0] + (1.0 - w1) * state.p[1])
        if state.qv is not None:
            fields.update({name: cp.asnumpy(getattr(state, attribute))
                           for name, attribute in MOISTURE_STATE_FIELDS.items()})
        for name, array in _live_state_history_fields(state).items():
            if isinstance(array, np.ndarray):
                fields[name] = np.array(array, copy=True, order="C")
            else:
                fields[name] = cp.asnumpy(array)
        if getattr(state, "physics", None) is not None:
            fields.update({name: cp.asnumpy(array)
                           for name, array in state.physics.output_fields().items()})
        if getattr(state, "_moving_frame_velocity", None):
            from woof.moving_frame import ground_relative_components
            fields["U"], fields["V"] = ground_relative_components(state, fields["U"], fields["V"])
        return fields


def _validation_reader(path):
    """Keep Windows Unicode filenames off the narrow NetCDF-C open path.

    The native reader inventories metadata and decodes only Times below; it
    never copies a whole tape or reads weather arrays for publication checks.
    Resolve the bridge before quarantine's input-error handler, since a missing
    or incompatible reader says nothing about whether a frame is complete.
    Other filenames keep the existing independent netCDF4 validation path.
    """
    if os.name == "nt" and not os.path.abspath(os.fspath(path)).isascii():
        from woof.netcdf_bridge import open_dataset, resolve_netcdf_bin
        return partial(open_dataset, executable=resolve_netcdf_bin()), True
    return netCDF4.Dataset, False


def validate_wrfout_file(path, *, inventory, shapes, times):
    """Reopen and prove inventory/shapes/Times/completion before publish."""
    path = Path(path)
    reader, _native = _validation_reader(path)
    with reader(path) as ds:
        if int(getattr(ds, _COMPLETION_ATTR, 0)) != 1:
            raise ValueError(f"wrfout {path} has no completion attribute")
        actual = set(ds.variables)
        if actual != set(inventory):
            raise ValueError(
                f"wrfout {path} inventory mismatch: expected "
                f"{sorted(inventory)}, got {sorted(actual)}")
        for name, expected in shapes.items():
            actual_shape = tuple(ds.variables[name].shape)
            if actual_shape != tuple(expected):
                raise ValueError(
                    f"wrfout {path} variable {name} shape {actual_shape} "
                    f"!= completed shape {tuple(expected)}")
        raw_times = ds.variables["Times"][:]
        actual_times = tuple(
            row.tobytes().decode("ascii").rstrip("\x00")
            for row in raw_times)
        if actual_times != tuple(times):
            raise ValueError(
                f"wrfout {path} Times mismatch: expected {tuple(times)}, "
                f"got {actual_times}")


#: Directory names under a wrfout tree whose contents are ABOUT frames.
#:
#: :func:`woof.progress_log.write_frame_marker` publishes
#: ``<outdir>/ready/<frame>.json``, a readiness receipt that
#: deliberately repeats the history basename, so ``wrfout*`` matches it
#: and ``wrfout*.tmp*`` matches its temporary while that is in flight.
#: Reading one as a frame costs both halves at once.  The sweep below
#: would file a valid receipt under ``ready/.quarantine`` as an
#: "incomplete-wrfout", which is false twice over (the receipt is
#: complete, and so is the frame it names) and which destroys the one
#: positive signal ``write_frame_marker`` documents a consumer may
#: trust: a marker that exists names a frame that is complete and
#: readable, and a consumer reads absence as "not yet".  A render feed
#: on this walk would hand the renderer JSON, which is the "NetCDF:
#: Unknown file format" exit 2 that ``woof.go_cli.WRFOUT_GLOB``
#: records itself as existing to prevent.
NON_FRAME_DIRNAMES = (progress_log.FRAME_MARKER_DIRNAME,)

#: Suffixes a history frame never carries.  ``woof.runplan``'s frame
#: count spells this exclusion its own way for the same tree
#: (``path.suffix != ".json"``, under the comment "Readiness JSON
#: receipts repeat the history basename under ready/"), so it is spelled
#: here beside the directory rule rather than instead of it: the two
#: catch the receipt whatever directory a future writer puts it in, and
#: a receipt in flight whatever suffix its temporary carries.
NON_FRAME_SUFFIXES = (".json",)


def is_wrfout_frame(relative) -> bool:
    """Does ``relative`` name a history frame, or something ABOUT one?

    ``relative`` is a path relative to a wrfout root, and only the shape
    of the name is read, never the bytes.  This is the whole definition
    of "frame" for :func:`iter_wrfout_files`, in one place and settled
    before anything opens a file, because the alternative is to infer
    it, and the inference is wrong: a walk that hands every ``wrfout*``
    match to a reader and calls whatever fails to open "incomplete"
    quarantines a valid readiness receipt under that label.  Discovery
    says what IS a frame; opening it decides only whether that frame is
    finished.

    What is excluded, and the file that would otherwise be misread:

    * Anything under a DOT-PREFIXED directory.
      :func:`woof.supervisor.quarantine_file` files a swept frame at
      ``<parent>/.quarantine/wrfout_....incomplete-wrfout.<stamp>.
      <pid>``, a name that still begins with ``wrfout``, so without this
      the sweep re-sweeps its own quarantine on every run and nests
      ``.quarantine`` inside ``.quarantine`` without bound.  Dot-
      prefixed FILES are kept, which is why only the parents are read:
      an orphan temporary is ``.wrfout_....tmp.<pid>.<n>`` and is
      exactly what the sweep is looking for.
    * Anything under a directory named in :data:`NON_FRAME_DIRNAMES`.
    * Any name whose suffix is in :data:`NON_FRAME_SUFFIXES`.

    The dot-directory rule is the one
    :func:`woof.render_layout.iter_rendered` makes for the PNG tree,
    deliberately spelled the same way.  The receipt rule has no
    counterpart there because nothing writes a receipt beside a picture
    under the picture's own name.
    """

    relative = Path(relative)
    parents = relative.parts[:-1]
    if any(part.startswith(".") for part in parents):
        return False
    if any(part in NON_FRAME_DIRNAMES for part in parents):
        return False
    return relative.suffix not in NON_FRAME_SUFFIXES


def iter_wrfout_files(directory, pattern: str = "wrfout*", *,
                      include_temporaries: bool = True) -> list[Path]:
    """Every history file under ``directory`` matching ``pattern``.

    THE reader for a wrfout directory, and the reason there is only one:
    this module WRITES ``d05/episode-002/wrfout_...`` when a domain
    declares a lifecycle (see :meth:`PerDomainWrfoutWriters.submit`,
    which takes the segment from ``render_layout.episode_segment``), and
    a one-level ``glob`` cannot see a single frame of it.  A flat reader
    therefore reports "no frames" for a run that published a full tape:
    the sweep below leaves every incomplete episode frame in place, and
    a render feed on the same spelling hands the renderer nothing.

    ``pattern`` is a filename glob; :func:`is_wrfout_frame`
    says which of them are frames, from the name alone and before
    anything is opened.  A match is dropped when it sits under a
    dot-prefixed directory (this walk's own ``.quarantine``, which would
    otherwise be re-swept and re-nested on every run) and when it is a
    receipt that repeats a frame's basename rather than a frame: the
    readiness markers at ``ready/<frame>.json``.  That second rule is
    what makes this walk safe to feed a RENDERER as well as the sweep,
    which is the other half of why there is only one of it.

    The walk goes through :func:`woof.render_layout.fs_path` and the
    results come back in the CALLER's spelling, for the same reason the
    picture walk does: a root's own length says nothing about its
    descendants', and a frame that cannot be enumerated is a frame that
    cannot be swept.

    The explicitly selected root may itself be a junction. Descendant
    symlinks and Windows reparse points are never followed: an episode
    link to another run must not let this sweep quarantine that run's files.

    Sorted by path, so two directories walk in the same order.
    Render consumers set ``include_temporaries=False``; the recovery sweep
    keeps the default because finding orphan temporaries is its job.
    """

    directory = Path(directory)
    walk_root = Path(render_layout.fs_path(directory, descend=True))
    if not walk_root.is_dir():
        return []
    def redirected(info):
        return (stat.S_ISLNK(info.st_mode)
                or bool(getattr(info, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))

    found, pending = [], [walk_root]
    while pending:
        folder = pending.pop()
        try:
            if folder != walk_root and redirected(folder.lstat()):
                continue
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if redirected(info):
                        continue
                    path = Path(entry.path)
                    if stat.S_ISDIR(info.st_mode):
                        if not entry.name.startswith(".") and entry.name not in NON_FRAME_DIRNAMES:
                            pending.append(path)
                    elif stat.S_ISREG(info.st_mode) and fnmatch.fnmatch(entry.name, pattern):
                        if not include_temporaries and ".tmp" in entry.name:
                            continue
                        relative = path.relative_to(walk_root)
                        if is_wrfout_frame(relative):
                            found.append(directory / relative)
        except (FileNotFoundError, NotADirectoryError):
            continue
    return sorted(found, key=lambda path: path.as_posix())


def quarantine_orphan_wrfouts(directory):
    """Move orphan temporaries and incomplete final files out of sight.

    Discovery is :func:`iter_wrfout_files` for all three patterns, so an
    episode-foldered run is swept exactly as a flat one is.

    "Cannot be opened, therefore incomplete" is read ONLY of files
    :func:`is_wrfout_frame` has already named as frames.  Everything
    else in the tree -- the readiness receipts under ``ready/``, this
    sweep's own ``.quarantine`` -- is never a candidate, so it is never
    moved and never labelled.
    """
    directory = Path(directory)
    if not directory.exists():
        return tuple()
    moved = []
    temporaries = set(iter_wrfout_files(directory, ".wrfout*.tmp*"))
    temporaries.update(iter_wrfout_files(directory, "wrfout*.tmp*"))
    for path in sorted(temporaries):
        target = quarantine_file(path, reason="orphan-wrfout-tmp")
        if target is not None:
            moved.append(target)
    for path in iter_wrfout_files(directory):
        if ".tmp" in path.name:
            continue
        from woof.netcdf_bridge import NetcdfInputError
        reader, native = _validation_reader(path)
        try:
            with reader(path) as ds:
                try:
                    complete = int(getattr(ds, _COMPLETION_ATTR, 0)) == 1
                except (ValueError, TypeError, OverflowError):
                    complete = False
        except NetcdfInputError:
            complete = False
        except Exception:
            # Crashes, timeouts and malformed reader replies are not evidence
            # that this frame is corrupt. Leave it untouched and report them.
            if native:
                raise
            complete = False
        if not complete:
            target = quarantine_file(path, reason="incomplete-wrfout")
            if target is not None:
                moved.append(target)
    return tuple(moved)


#: The ledger of the history frames a run wrote, one file name per line,
#: beside the frames (``HISTORY_LEDGER`` in ``rw-wrfbatch/src/snowfall.rs``
#: reads it).  THE BREAKAGE: a host removed earlier history frames to save
#: disk (an 8x5090 host, 2026-10-21), and the renderer could not tell a removed frame
#: from a cadence that never wrote one, so a Kuchera snowfall total either
#: had to be drawn across the hole with one end temperature for every
#: interval it spans, or refused for every run whose cadence legitimately
#: changes.  A name in the ledger that is gone from the folder is a frame
#: this run wrote and someone removed.
HISTORY_LEDGER = ".history-written"


def record_history_written(path) -> None:
    """Add a published history frame's name to its folder's ledger.

    Only ``wrfout_d<NN>_<valid time>`` frames are recorded.  One short
    appended line per frame, so concurrent domain writers interleave whole
    lines.  Never raises: the ledger tells the renderer what was removed,
    and a frame whose line could not be written is only one the renderer
    will not call missing.
    """

    path = Path(path)
    name = path.name
    if not (name.startswith("wrfout_d") and len(name) == 30
            and name[10] == "_"):
        return
    with suppress(OSError):
        descriptor = os.open(path.parent / HISTORY_LEDGER,
                             os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(descriptor, (name + "\n").encode("ascii"))
        finally:
            os.close(descriptor)


class WrfoutWriter:
    """The product-tape writer.

    ``engine`` selects which library puts the bytes on disk (see
    :data:`WRFOUT_WRITER_ENV`).  The DEFAULT is the Rust classic writer;
    ``engine="python"`` (or ``WOOF_WRFOUT_WRITER=python``) is the
    netCDF4/HDF5 writer kept as an explicit, documented WORKAROUND.  The
    schema, the attribute inventory, the validation and the atomic
    publication protocol are identical on both engines; the container
    differs (CDF-2 classic vs HDF5 ``NETCDF4_CLASSIC``), which every
    reader in the estate handles (netCDF4, wrf-rust, ``rw_wrfbatch``,
    ``rw_netcdf`` -- proved by the dual-write verification).
    """

    def __init__(self, path, *, nx, ny, nz, dx, dy, title="woof",
                 global_attrs=None, field_schema=None, soil_layers=None,
                 engine=None, compression=None, _retain_identity_handle=False):
        self.engine = resolve_wrfout_engine(engine)
        # Compressed frames are the Rust engine's container; the netCDF4
        # workaround engine keeps writing what it always wrote.
        self.compression = (resolve_history_compression(compression)
                            if self.engine == "rust" else "none")
        self.publication_revision = None
        self._publication_file = None
        self._retain_identity_handle = _retain_identity_handle
        if self.engine == "rust":
            from woof.io import nc_writer_bridge

            reason = nc_writer_bridge.unavailable_reason()
            if reason is not None:
                raise RuntimeError(
                    "the default wrfout engine is the Rust NetCDF writer "
                    "(the netcdf-writer cdylib behind woof.io."
                    "nc_writer_bridge) and it is not loadable here, so "
                    f"the product tape cannot be written:\n  {reason}\n"
                    "A silent fallback would change the tape's container "
                    "with build state and quietly un-test the default, so "
                    "it is refused instead.  Remedies: build the library "
                    "from a checkout (cd tools/rustwx; cargo build "
                    "--release --locked --offline -p netcdf-writer; "
                    "cd ../..), or run `woof fetch-bridges` on an "
                    f"installed wheel; or set {WRFOUT_WRITER_ENV}=python "
                    "to select the netCDF4 writer as an explicit "
                    "workaround.")
        self.nx, self.ny, self.nz = nx, ny, nz
        # WRF's soil axis length is the land-surface scheme's own geometry
        # (Noah/Noah-MP 4, RUC 6 or 9), not a universal constant, so any
        # caller with a RunConfig must pass soil_layer_count(cfg).  ``None``
        # means the caller declared no land-surface scheme -- the idealized
        # cases, which write no soil field on this axis -- and takes the
        # schema's no-LSM length rather than a literal, so this signature
        # carries no soil-geometry constant of its own.  It used to default to
        # 4, and three real callers (woof/runtime.py, offline_child_run.py,
        # offline_child_smoke.py) were silently taking it.
        self.soil_layers = (NO_LAND_SURFACE_SOIL_LAYERS
                            if soil_layers is None else int(soil_layers))
        # Pinned to ABSOLUTE once, here: the publish-time self-validation
        # reopens the temp file with netCDF4, and the netCDF-C build in
        # use refuses a CLASSIC file named by a drive-relative path
        # (`\tmp\...` -> `NetCDF: Unknown file format`) that it opens
        # fine absolutely (MEASURED on the same bytes).  Resolving at
        # construction also makes the fsync/rename sequence immune to a
        # later cwd change, on both engines.
        self._final_path = Path(os.path.abspath(path))
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = unique_temp_path(self._final_path, hidden=True)
        if self.engine == "rust" and self.compression == "deflate":
            from woof.io.classic_tape import NETCDF4_DEFLATE
            self.ds = _ClassicTape(
                self._temp_path, container=NETCDF4_DEFLATE,
                netcdf4_options=_history_netcdf4_options())
        elif self.engine == "rust":
            self.ds = _ClassicTape(self._temp_path)
        else:
            self.ds = netCDF4.Dataset(
                self._temp_path, "w", format="NETCDF4_CLASSIC")
        ds = self.ds
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("west_east", nx)
        ds.createDimension("south_north", ny)
        ds.createDimension("bottom_top", nz)
        ds.createDimension("west_east_stag", nx + 1)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("bottom_top_stag", nz + 1)
        # The land-surface scheme's soil layers live on their own fully
        # dimensioned staggered vertical axis, not bottom_top.
        ds.createDimension("soil_layers_stag", self.soil_layers)
        ds.setncatts({
            # NC_FLOAT, not the NC_DOUBLE a Python float would become: stock
            # WRF writes DX/DY single-precision like every other projection
            # global (see wrf_global_attrs).
            "TITLE": title,
            # TITLE is the caller's configured output title, so it is not a
            # provenance seal: a wrfout separated from its logs could not say
            # which build wrote it.  GPUWM_VERSION can, and it reads the
            # installed distribution's metadata rather than a constant.
            "GPUWM_VERSION": _producer_version(),
            "DX": np.float32(dx), "DY": np.float32(dy),
            "WEST-EAST_GRID_DIMENSION": nx + 1,
            "SOUTH-NORTH_GRID_DIMENSION": ny + 1,
            "BOTTOM-TOP_GRID_DIMENSION": nz + 1,
            # The patch this file holds.  WRF's own netCDF I/O layer and the
            # ndown-class tools read this group -- not the *_GRID_DIMENSION
            # group -- for the extent of the data actually in the file, and
            # the sibling wrfinput export already writes the END half of it.
            # woof writes one undecomposed patch per domain, so the patch IS
            # the domain: 1-based start, unstaggered end at the mass count and
            # staggered end one past it, exactly the values on the pinned
            # v4.6.1 reference file (``woof/wrf_direct_v461_contract.json``).
            **_wrf_patch_extent_attrs(nx, ny, nz),
            "MAP_PROJ": 0,
        })
        if global_attrs:
            ds.setncatts(dict(global_attrs))
        # The physics parameter set this process runs, when it runs one
        # (woof.physics_params).  A default run writes nothing here, so
        # its file is the file it always was.
        from woof.physics_params import wrfout_global_attrs
        parameter_set = wrfout_global_attrs()
        if parameter_set:
            ds.setncatts(parameter_set)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        self._n = 0
        self._times = []
        self._closed = False
        self._declared_fields: frozenset[str] | None = None
        if field_schema is not None:
            schema = dict(field_schema)
            if hasattr(ds, "START_DATE") and hasattr(ds, "DT"):
                schema.setdefault("XTIME", np.asarray(0.0, dtype=np.float32))
                schema.setdefault(
                    "ITIMESTEP", np.asarray(0, dtype=np.int32))
            for name, value in schema.items():
                shape = self._wrf_shape(name, getattr(value, "shape", value))
                self._create_variable(name, self._dims_for(name, shape))
            self._declared_fields = frozenset(schema)

    def _ensure_dimension(self, name, size):
        """Create a dimension the first time a field needs it.

        Created lazily rather than in ``__init__`` so a file from a run
        without that scheme keeps byte-identical headers; the frozen wrfout
        contract is a gate, and silently adding a dimension to every file
        would move it.
        """
        existing = self.ds.dimensions.get(name)
        if existing is None:
            self.ds.createDimension(name, int(size))
            return
        if len(existing) != int(size):
            raise ValueError(
                f"wrfout dimension {name} is {len(existing)}, not {size}")

    def _wrf_shape(self, name, shape):
        """On-disk shape for ``name``: WRF's, which is not always woof's."""
        shape = tuple(shape)
        if name in _Z_STAGGERED_MASS_FIELDS and shape == (
                self.nz, self.ny, self.nx):
            return (self.nz + 1, self.ny, self.nx)
        return shape

    @staticmethod
    def _check_integer_field_dtype(name, array):
        """A field WRF declares integer must arrive as one.

        netCDF4 casts on assignment, so handing a float array to an ``i4``
        variable truncates every value in silence -- which would restore
        the exact defect the schema exists to end, one layer further in.
        Only the integer direction is checked: a real field legitimately
        accepts any float width.
        """
        schema = HISTORY_FIELDS_BY_NETCDF_NAME.get(name)
        if schema is None or schema.dtype != "i4":
            return
        if array.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"wrfout variable {name} is declared integer by WRF v4.6.1 "
                f"({schema.registry}) but the frame supplied "
                f"{array.dtype}; writing it would truncate every value")

    def _wrf_array(self, name, array):
        """``array`` on its WRF axis, zero-padding the unfilled top level."""
        self._check_integer_field_dtype(name, array)
        return wrf_axis_array(name, array, nx=self.nx, ny=self.ny, nz=self.nz)

    def _dims_for(self, name, shape):
        nz, ny, nx = self.nz, self.ny, self.nx
        if name in ("TURBINE_POWER", "TURBINE_HUB_WIND", "TURBINE_FARM_INDEX", "TURBINE_HUB_HEIGHT", "FARM_POWER", "FARM_HUB_WIND"):
            if len(shape) != 1 or shape[0] < 1:
                raise ValueError(f"wind-farm history {name} requires a nonempty one-dimensional index")
            axis = "wind_turbine" if name.startswith("TURBINE_") else "wind_farm"
            self._ensure_dimension(axis, shape[0])
            return ("Time", axis)
        row = WRF_INPUT_HISTORY_ROWS.get(name, WRF_INITIALIZED_HISTORY_ROWS.get(name))
        if row is not None:
            dims = tuple(row["history_dimensions"])
            extents = {"scalar": 1, "soil_layers_stag": self.soil_layers,
                       "bottom_top": nz, "bottom_top_stag": nz + 1,
                       "south_north": ny, "south_north_stag": ny + 1,
                       "west_east": nx, "west_east_stag": nx + 1,
                       "seed_dim_stag": WRF_GNU_RANDOM_SEED_WORDS}
            expected = tuple(extents[d] for d in dims)
            if tuple(shape) != expected:
                raise ValueError(f"native history {name} has shape {shape}, expected {expected}")
            for dim in dims:
                self._ensure_dimension(dim, extents[dim])
            return ("Time", *dims)
        if name in SFIRE_PARTICLE_FIELDS:
            if len(shape)!=1 or shape[0]<1:
                raise ValueError(f"firebrand history field {name} must contain its native particle allocation")
            self._ensure_dimension("fs_maxsize",int(shape[0]))
            return ("Time","fs_maxsize")
        if name in SFIRE_FINE_FIELDS:
            if len(shape) != 2:
                raise ValueError(f"fine fire history field {name} must have two horizontal dimensions")
            fy, fx = map(int, shape)
            if fx < nx or fy < ny or fx % nx or fy % ny:
                raise ValueError(f"fine fire history field {name} must refine every atmospheric mass cell")
            self._ensure_dimension("west_east_subgrid", fx)
            self._ensure_dimension("south_north_subgrid", fy)
            self.ds.setncattr("SR_X", np.int32(fx // nx))
            self.ds.setncattr("SR_Y", np.int32(fy // ny))
            return ("Time", "south_north_subgrid", "west_east_subgrid")
        if name in SFIRE_CLASS_FIELDS or name == "FMEP":
            if len(shape) != 3 or tuple(shape[1:]) != (ny, nx) or shape[0] < 1:
                raise ValueError(f"fire moisture history field {name} must be a class stack over the atmospheric mass grid")
            axis = "fuel_moisture_extended_paramete" if name == "FMEP" else "fuel_moisture_classes_stag"
            self._ensure_dimension(axis, shape[0])
            return ("Time", axis, "south_north", "west_east")
        if name == "LFN_TIME":
            if tuple(shape) != (1,):
                raise ValueError("fire level-function history time must contain one ignition time")
            self._ensure_dimension("i_lfn_history", 1)
            return ("Time", "i_lfn_history")
        if name in _SNOW_LAYER_FIELDS or name in _SNSO_LAYER_FIELDS:
            snow_layers = int(tuple(shape)[0]) if name in _SNOW_LAYER_FIELDS \
                else int(tuple(shape)[0]) - self.soil_layers
            expected = ((snow_layers, ny, nx) if name in _SNOW_LAYER_FIELDS
                        else (snow_layers + self.soil_layers, ny, nx))
            if tuple(shape) != expected or snow_layers < 1:
                raise ValueError(
                    f"Noah-MP snow history field {name} has shape "
                    f"{tuple(shape)}, which is not a snow stack over "
                    f"{(ny, nx)}")
            if name in _SNOW_LAYER_FIELDS:
                self._ensure_dimension("snow_layers_stag", snow_layers)
                axis = "snow_layers_stag"
            else:
                self._ensure_dimension(
                    "snso_layers_stag", snow_layers + self.soil_layers)
                axis = "snso_layers_stag"
            return ("Time", axis, "south_north", "west_east")
        if name in ("ZS", "DZS") and tuple(shape) == (self.soil_layers,):
            # WRF's soil-layer depth and thickness rows (Time, soil_layers_stag).
            return ("Time", "soil_layers_stag")
        if name in ("OM_TMP", "OM_S", "OM_U", "OM_V", "OM_DEPTH", "OM_DENSITY", "OM_TINI", "OM_SINI"):
            if len(shape) != 3 or tuple(shape[1:]) != (ny, nx) or shape[0] < 2:
                raise ValueError(f"ocean history {name} needs a level stack over the mass grid")
            self._ensure_dimension("ocean_levels", shape[0])
            return ("Time", "ocean_levels", "south_north", "west_east")
        if name in _SOIL_LAYER_FIELDS:
            expected = (self.soil_layers, ny, nx)
            if tuple(shape) != expected:
                raise ValueError(
                    f"WRF soil history field {name} must have shape "
                    f"{expected}, got {tuple(shape)}")
            return ("Time", "soil_layers_stag", "south_north", "west_east")
        table = {
            (): (),
            (nz,): ("bottom_top",),
            (nz + 1,): ("bottom_top_stag",),
            (nz, ny, nx): ("bottom_top", "south_north", "west_east"),
            (nz, ny, nx + 1): ("bottom_top", "south_north", "west_east_stag"),
            (nz, ny + 1, nx): ("bottom_top", "south_north_stag", "west_east"),
            (nz + 1, ny, nx): ("bottom_top_stag", "south_north", "west_east"),
            (ny, nx): ("south_north", "west_east"),
            (ny, nx + 1): ("south_north", "west_east_stag"),
            (ny + 1, nx): ("south_north_stag", "west_east"),
        }
        return ("Time",) + table[tuple(shape)]

    def _create_variable(self, name, dims):
        """Variable with WRF type, memory order, Registry, and grid attrs.

        A scheme field's type, ``FieldType``, ``description``, ``units`` and
        staggering all come from its transcribed WRF v4.6.1 Registry row
        rather than from the writer's own assumptions, and the row's stagger
        is checked against the one the dimension table implies.  Those two
        can only disagree if the dimension table routed the field onto the
        wrong axis, which is a schema lie a reader cannot detect -- so it is
        refused here instead of published.
        """
        schema = HISTORY_FIELDS_BY_NETCDF_NAME.get(name)
        if schema is not None:
            dtype, field_type = schema.dtype, schema.field_type
            desc, units = schema.description, schema.units
        else:
            dtype = wrf_field_dtype(name)
            field_type = (WRF_FIELD_TYPE_INTEGER if name == "ITIMESTEP"
                          else WRF_FIELD_TYPE_REAL)
            desc, units = _VAR_META.get(
                name, _CHEM_VAR_META.get(name, ("", "")))
        stagger = next((s for d, s in _STAGGER.items() if d in dims), "")
        # Native fire Registry z applies to refined and class grids, and
        # even its 2-D flux diagnostics retain Z without a vertical axis.
        if name in SFIRE_REGISTRY_FIELDS:
            stagger = SFIRE_REGISTRY_FIELDS[name][1]
        if schema is not None and schema.stagger != stagger:
            raise ValueError(
                f"wrfout variable {name} lands on dimensions {dims}, whose "
                f"stagger is {stagger!r}, but WRF v4.6.1 declares it "
                f"{schema.stagger!r} ({schema.registry})")
        var = self.ds.createVariable(name, dtype, dims)
        fire_coordinates = str(getattr(self.ds, "FIRE_COORDINATE_MODE", "geographic"))
        if name in ("FXLAT", "FXLONG"):
            if fire_coordinates not in ("metric", "geographic"):
                raise ValueError("fire history coordinates must declare metric or geographic mode so readers can locate the fine grid")
            if fire_coordinates == "metric":
                desc = "y coordinate of midpoints of fire cells" if name == "FXLAT" else "x coordinate of midpoints of fire cells"
                units = "m"
                var.standard_name = "projection_y_coordinate" if name == "FXLAT" else "projection_x_coordinate"
        if name == "XTIME":
            origin = str(getattr(self.ds, "START_DATE", "")).replace(
                "_", " ", 1)
            desc = units = f"minutes since {origin}"
        var.setncattr("FieldType", np.int32(field_type))
        if dims == ("Time",):
            var.MemoryOrder = "0  "
        elif len(dims) == 2:
            var.MemoryOrder = "Z  "
        else:
            var.MemoryOrder = "XYZ" if len(dims) == 4 else "XY "
        var.description = desc
        var.units = units
        var.stagger = stagger
        if name == "CLDFRA":
            # Radiation's cadence is coarser than history's: say what a
            # between-calls frame and the initial frame carry.
            for key, value in RADIATION_CLDFRA_ATTRIBUTES.items():
                var.setncattr(key, value)
        spatial = any(d in dims for d in (
            "west_east", "west_east_stag", "south_north",
            "south_north_stag", "west_east_subgrid", "south_north_subgrid"))
        if spatial:
            time_link = (" XTIME" if hasattr(self.ds, "START_DATE")
                         and hasattr(self.ds, "DT") else "")
            if name in SFIRE_FINE_FIELDS:
                var.coordinates = "FXLONG FXLAT" + time_link
            elif name in ("XLAT_U", "XLONG_U"):
                var.coordinates = "XLONG_U XLAT_U"
            elif name in ("XLAT_V", "XLONG_V"):
                var.coordinates = "XLONG_V XLAT_V"
            elif name in ("XLAT", "XLONG"):
                var.coordinates = "XLONG XLAT"
            elif "west_east_stag" in dims:
                var.coordinates = "XLONG_U XLAT_U" + time_link
            elif "south_north_stag" in dims:
                var.coordinates = "XLONG_V XLAT_V" + time_link
            else:
                var.coordinates = "XLONG XLAT" + time_link
        return var

    def _time_coordinate_fields(self, time_str: str) -> dict[str, np.ndarray]:
        """WRF XTIME/ITIMESTEP values derived from bound start time and DT."""
        if not hasattr(self.ds, "START_DATE") or not hasattr(self.ds, "DT"):
            return {}
        try:
            valid = datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%S")
            start = datetime.strptime(
                str(self.ds.START_DATE), "%Y-%m-%d_%H:%M:%S")
        except ValueError as exc:
            raise ValueError(
                "WRF time coordinates require START_DATE and frame time in "
                "YYYY-MM-DD_HH:MM:SS format") from exc
        elapsed_seconds = (valid - start).total_seconds()
        if elapsed_seconds < 0.0:
            raise ValueError(
                f"wrfout valid time {time_str} precedes START_DATE "
                f"{self.ds.START_DATE}")
        dt = float(self.ds.DT)
        return {
            "XTIME": np.asarray(elapsed_seconds / 60.0, dtype=np.float32),
            "ITIMESTEP": np.asarray(
                int(round(elapsed_seconds / dt)), dtype=np.int32),
        }

    def write_frame(self, time_str: str, fields: dict[str, np.ndarray]):
        with perf_timing.stage("io.wrfout.write_frame") as timed:
            return self._write_frame(time_str, fields, timed)

    def _write_frame(self, time_str: str, fields: dict[str, np.ndarray],
                     timed):
        t = self._n
        # netCDF4 1.7.x stringtochar mangles S-dtype input under numpy 2,
        # so build the 19-char record directly (null-pad).  A longer value
        # is refused rather than truncated: the only way to exceed 19 is a
        # sub-second instant, and silently dropping the fraction is how two
        # distinct frames came to describe the same second.
        encoded = time_str.encode("ascii")
        if len(encoded) > 19:
            raise ValueError(
                f"wrfout Times record {time_str!r} exceeds the 19-character "
                "WRF format; sub-second history instants are not supported "
                "and must not be truncated into an aliased one")
        buf = encoded.ljust(19, b"\x00")
        frame = dict(fields)
        for name, value in self._time_coordinate_fields(time_str).items():
            frame.setdefault(name, value)
        if (self._declared_fields is not None
                and set(frame) != set(self._declared_fields)):
            missing = sorted(set(self._declared_fields) - set(frame))
            extra = sorted(set(frame) - set(self._declared_fields))
            raise ValueError(
                "wrfout frame does not match the creation-time field "
                f"schema: missing={missing}, extra={extra}")
        # Declare every variable of this frame BEFORE its first data byte
        # lands.  On the Rust engine the classic header freezes at the
        # first data write, so a variable created mid-frame would be a
        # named refusal; on netCDF4 this only moves creation ahead of the
        # Times record write, which leaves variable-creation order -- the
        # thing HDF5's name heap is laid out by -- exactly as it was.
        arrays: dict[str, np.ndarray] = {}
        for name, arr in frame.items():
            arr = self._wrf_array(name, np.asarray(arr))
            if name not in self.ds.variables:
                self._create_variable(name, self._dims_for(name, arr.shape))
            arrays[name] = arr
        self.ds.variables["Times"][t] = np.frombuffer(buf, dtype="S1")
        written = 0
        for name, arr in arrays.items():
            self.ds.variables[name][t] = arr
            written += int(arr.nbytes)
        timed.count(fields=len(frame), bytes_written=written)
        self._times.append(buf.rstrip(b"\x00").decode("ascii"))
        self._n += 1

    def _abandon_ds(self):
        """Release the dataset without publishing: abort a classic tape
        (numrecs stays 0 on the partial file), plain-close a netCDF4 one
        (no completion attribute, so the sweep quarantines it)."""
        if isinstance(self.ds, _ClassicTape):
            self.ds.abort()
        else:
            self.ds.close()

    def close(self):
        if self._closed:
            return
        inventory = tuple(self.ds.variables)
        shapes = {name: tuple(variable.shape)
                  for name, variable in self.ds.variables.items()}
        times = tuple(self._times)
        try:
            if self._n == 0:
                # A tape that never received a frame must not be published.
                # Every publication step below succeeds on an empty file:
                # the classic tape freezes its header (which is where
                # GPUWM_WRITE_COMPLETE is written), finish() finds no
                # unwritten region because there are no records,
                # validate_wrfout_file passes with times == () and every
                # shape carrying Time = 0, and the temp file is renamed onto
                # its final name.  The completion attribute is exactly what
                # quarantine_orphan_wrfouts trusts, so the sweep would then
                # leave a complete-looking, empty wrfout in place forever.
                # Raising HERE, inside the try, routes it through the same
                # abandon-and-quarantine path every other publication failure
                # takes.  A caller that means to discard the file calls
                # abort(), which is already the documented way to do that.
                raise ValueError(
                    f"wrfout {self._final_path.name} was closed with no "
                    "frames; a history tape carrying zero Times records "
                    "must not be published (call abort() to discard it)")
            if isinstance(self.ds, _ClassicTape):
                # The completion attribute is already in the header (the
                # tape writes it at freeze, last); close() refuses any
                # tape with an unwritten region, then patches numrecs,
                # flushes and fsyncs.
                self.ds.close()
            else:
                self.ds.setncattr(_COMPLETION_ATTR, np.int32(1))
                self.ds.close()
            self._closed = True
            fsync_file(self._temp_path)
            from woof.output_identity import PublicationFile
            self._publication_file = PublicationFile(self._temp_path)
            validate_wrfout_file(
                self._temp_path, inventory=inventory, shapes=shapes,
                times=times)
            self._publication_file.check_validated()
            if getattr(self, "_retain_identity_handle", False):
                # Finish identity outside the NetCDF lock, after releasing
                # host staging and before exposing the final history name.
                self.publication_revision = self._publication_file.written
                return
            self._publish_output()
            self.release_identity_handle()
        except BaseException:
            self.release_identity_handle(preserve=True)
            if not self._closed:
                # A half-closed netCDF handle can fail repeatedly. Preserve
                # the original publication error and still reach quarantine.
                with suppress(BaseException):
                    self._abandon_ds()
                self._closed = True
            if self._temp_path.exists():
                with suppress(BaseException):
                    quarantine_file(
                        self._temp_path,
                        reason="failed-wrfout-publication")
            raise

    def _publish_output(self, *, allow_missing=False):
        # Sharing violations receive the same capped 0.50 s retry as the
        # heartbeat, but a durable wrfout publication remains fail-loud.
        replace_file_with_retry(self._temp_path, self._final_path)
        # Sync the name before emitting a ready marker. A durable payload
        # alone could otherwise recover under its hidden temporary name.
        _fsync_directory(self._final_path.parent)
        self.publication_revision = self._publication_file.published(
            self._final_path, allow_missing=allow_missing)
        record_history_written(self._final_path)

    def release_identity_handle(self, *, preserve=False):
        retained, self._publication_file = self._publication_file, None
        if retained is None:
            return
        try:
            if preserve:
                # A replacement may have unlinked the actual written file.
                # Save its retained descriptor only on this failure path;
                # never move or overwrite the replacement at the address.
                info = os.fstat(retained.handle.fileno())
                named = False
                for path in (self._temp_path, self._final_path):
                    with suppress(OSError):
                        addressed = path.stat()
                        named = named or ((addressed.st_dev, addressed.st_ino)
                                          == (info.st_dev, info.st_ino))
                if not named:
                    recovery = unique_temp_path(self._final_path, hidden=True)
                    retained.handle.seek(0)
                    with recovery.open("xb") as destination:
                        for block in iter(lambda: retained.handle.read(1 << 20), b""):
                            destination.write(block)
                        destination.flush()
                        os.fsync(destination.fileno())
                    quarantine_file(recovery, reason="replaced-wrfout-publication")
        finally:
            retained.close()

    def complete_output_identity(self, *, cancel_event=None):
        from woof.output_identity import CompletedFileRecord, completed_file_record
        try:
            proof = completed_file_record(
                self._temp_path, published=self.publication_revision,
                cancel_event=cancel_event, handle=self._publication_file.handle)
            self._publication_file.check_validated()
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Output publication was cancelled before completion.")
            self._publish_output(allow_missing=True)
            return CompletedFileRecord(
                str(self._final_path.resolve()), proof.size, proof.sha256,
                self.publication_revision, str(self._final_path))
        except BaseException:
            self.release_identity_handle(preserve=True)
            if self._temp_path.exists():
                with suppress(BaseException):
                    quarantine_file(self._temp_path, reason="failed-wrfout-identity")
            raise
        finally:
            self.release_identity_handle()

    def abort(self):
        """Close without completion and quarantine a failed write."""
        if self._closed:
            return
        with suppress(BaseException):
            self._abandon_ds()
        self._closed = True
        if self._temp_path.exists():
            with suppress(BaseException):
                quarantine_file(self._temp_path, reason="aborted-wrfout")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


#: Device memory a history frame's snapshots leave free (bytes).
_HISTORY_SNAPSHOT_HEADROOM = 1 << 30


def _history_snapshot_budget() -> int:
    """Bytes of device snapshot one history frame may take.

    Only memory the driver still holds free counts: the memory pool's cached
    blocks are what the following steps allocate their transients from, and
    the snapshots stay alive while those steps run.
    """
    import cupy as cp

    free, _total = cp.cuda.runtime.memGetInfo()
    return max(0, int(free) - _HISTORY_SNAPSHOT_HEADROOM)

def wrf_field_dtype(name):
    """The stored history type, shared by file and in-memory history."""
    schema = HISTORY_FIELDS_BY_NETCDF_NAME.get(name)
    return schema.dtype if schema is not None else ("i4" if name == "ITIMESTEP" else "f4")


def wrf_axis_array(name, array, *, nx: int, ny: int, nz: int):
    """One frame field as the writer stores it: on WRF's vertical axis, the
    unfilled top level of a z-staggered mass field zero-padded.  The writer's
    own transform (:meth:`WrfoutWriter._wrf_array`), shared with an in-memory
    frame consumer so its words are the file's words."""
    WrfoutWriter._check_integer_field_dtype(name, array)
    array = np.asarray(array, dtype=wrf_field_dtype(name))
    if (name not in _Z_STAGGERED_MASS_FIELDS
            or array.shape != (nz, ny, nx)):
        return array
    lifted = np.zeros((nz + 1, ny, nx), dtype=array.dtype)
    lifted[:nz] = array
    return lifted


def _wrf_start_history_t(state):
    """Strict mode: the T a domain's frame writes when it is not t_2, or ``None``.

    WRF's history T is th_phy_m_t0, which only the input sets before the
    first step; THM is t_2.  A WRF-file nest's adjust_tempqv and a parent's
    initial feedback move t_2 but not th_phy_m_t0, so until the domain's
    first step (no carried strict muts yet) T is the file's word
    (woof.ingest.wrfinput_nest_initial.finish_strict_nest_start).
    """
    if not _WRF_EXACT:
        return None
    from woof.state_serialization_contract import WRF_MUTS_SLOT
    start_t = getattr(state, "_wrf_start_history_t", None)
    if start_t is not None and state.existing_scratch(WRF_MUTS_SLOT) is None:
        return start_t
    # After the first step: th_phy_m_t0 where it parts from t_2, a
    # parent's (woof.core.nest_round_trip.th_phy_m_t0_after_step).
    return getattr(state, "_wrf_th_phy_m_t0", None)


def _device_state_frame(state, *, include_diagnostic_pressure: bool = True):
    """Build the standard frame as live device arrays for side-stream D2H."""
    import cupy as cp

    with _state_device_context(state):
        if getattr(state, "_moving_frame_velocity", None):
            from woof.moving_frame import initialize_moving_frame
            initialize_moving_frame(state)
        ny, nx = state.mup.shape
        phb = state.phb
        if phb.ndim == 1:
            phb = cp.broadcast_to(phb[:, None, None], (phb.size, ny, nx))
        start_t = _wrf_start_history_t(state)
        fields = {
            "T": (cp.asarray(start_t) if start_t is not None
                  else state.thp if _WRF_EXACT
                  else state.total_theta() - cp.float32(300.0)),
            **{name: getattr(state, attribute)
               for name, attribute in CORE_DIRECT_STATE_FIELDS.items()},
            "PHB": phb, "MUB": state.mub2d, "HGT": state.ht,
        }
        fields["THM"] = fields["T"] if start_t is None else state.thp
        if include_diagnostic_pressure:
            pb = state.pb
            pb3 = pb if pb.ndim == 3 else pb[:, None, None]
            fields["P"] = (state.p_perturbation if DIAGNOSTICS_ENABLED
                           else state.p - pb3)
            fields["PB"] = cp.broadcast_to(pb3, state.p.shape)
            if _driver_refreshes_psfc(state):
                fields["PSFC"] = state.physics.fields["psfc"]
            elif getattr(state, "p_top", None) is not None:
                from woof.core import constants as c
                phb3 = (state.phb[:, None, None]
                        if state.phb.ndim == 1 else state.phb)
                z_if = (phb3 + state.php) / cp.float32(c.G)
                z_mid = 0.5 * (z_if[:-1] + z_if[1:])
                w1 = (z_if[0] - z_mid[1]) / (z_mid[0] - z_mid[1])
                fields["PSFC"] = w1 * state.p[0] + (1.0 - w1) * state.p[1]
        if state.qv is not None:
            fields.update({name: getattr(state, attribute)
                           for name, attribute in MOISTURE_STATE_FIELDS.items()})
        fields.update(_live_state_history_fields(state))
        if getattr(state, "physics", None) is not None:
            fields.update(state.physics.output_fields())
        if _WRF_EXACT and getattr(state, "_wrf_frame_round_trip_due", False):
            from woof.core.nest_round_trip import round_tripped_frame_fields
            fields = round_tripped_frame_fields(state, fields)
        if getattr(state, "_moving_frame_velocity", None):
            from woof.moving_frame import ground_relative_components
            fields["U"], fields["V"] = ground_relative_components(state, fields["U"], fields["V"])
        return fields


@dataclass
class _AsyncFrame:
    path: Path
    time_str: str
    fields: dict[str, np.ndarray]
    event: object
    device_refs: tuple[object, ...]
    pinned_refs: tuple[object, ...]
    admitted: bool = False
    #: The submitting datetime, carried so a landing observer receives
    #: the valid time as a value rather than re-parsing ``time_str``
    #: (or the filename) on the writer thread.
    valid_time: object = None
    #: Per-frame global-attribute override.  Carried on the ticket so a
    #: submit-time snapshot (carrier provenance, which can change at a
    #: resume or a mid-run forcing) lands on exactly the file whose frame
    #: it describes -- without draining the async queue the way a
    #: whole-writer ``update_global_attrs`` swap must.  ``None`` means
    #: "use the writer's standing set".
    global_attrs: object = None
    #: The frame's field bytes, counted once at admission while the
    #: fields still exist, so the writer can say how much it has left to
    #: write after the worker has released them.
    nbytes: int = 0
    #: A frame whose members are still on their way to the host (the
    #: ranked road's ``tilestream.output.DeferredStoreFrame``), with the
    #: selection and the extra rows it is composed with; the WORKER
    #: assembles it, after its download lands.  ``None`` for every other
    #: frame, whose ``fields`` are complete at admission.
    deferred: object = None
    #: Required producer identity handoff. Unlike telemetry, failure prevents
    #: publication being treated as a safely consumable ensemble history.
    completed_observer: object = None
    #: perf_counter at admission, for the frame's timing record.
    admitted_at: float = 0.0
    #: Seconds the producer waited for the byte budget before admission.
    admission_wait_s: float = 0.0


def _compose_host_frame(frame_items, refl, extra) -> dict:
    """A host frame's fields in THE order: frame rows, REFL_10CM, extras.

    One function for the admitted frame and the deferred one, because the
    order is the file: HDF5 lays its name heap out in variable-creation
    order, and the same numbers in another order hash differently.
    ``extra`` rows never replace a frame row (prognostic and state-derived
    fields win, notably a child's blended HGT over the static HGT_M).
    """
    fields = dict(frame_items)
    if refl is not None:
        fields["REFL_10CM"] = refl
    for name, value in extra.items():
        if name not in fields:
            fields[name] = value
    return fields


def _frame_nbytes(fields) -> int:
    return sum(int(getattr(value, "nbytes", 0) or 0)
               for value in (fields or {}).values())


def _contiguous_preserving_shape(value, xp=np):
    """Contiguous storage on the original axis, including a zero-D scalar.

    NumPy and CuPy's ascontiguousarray promote zero-D values to (1,).
    History scalar dimensions must survive staging just like vector axes.
    """
    array = xp.asarray(value)
    return xp.ascontiguousarray(array).reshape(array.shape)


class _AsyncTicketQueue(queue.Queue):
    """Bounded queue that records ticket admission under its mutex."""

    def release_completed_device_refs(self, stream) -> None:
        # A slow file write can leave the next ticket queued long after its
        # download finishes. Reuse that snapshot storage before allocating
        # another capture, without waiting for unfinished transfers. Hold the
        # queue mutex so the worker cannot take a ticket during this handoff.
        with self.mutex, stream:
            for ticket in self.queue:
                if (ticket is not None and ticket.device_refs
                        and ticket.event.done):
                    ticket.device_refs = ()

    def _put(self, item) -> None:
        if item is None:
            super()._put(item)
            return

        # Queue.put holds the queue mutex while calling _put.  Mark first so
        # there is no state in which a worker can take this ticket while its
        # producer still considers it unadmitted.  If insertion unwinds before
        # appending, restore the marker while the same mutex is still held.
        try:
            item.admitted = True
            super()._put(item)
        except BaseException:
            if not any(queued is item for queued in self.queue):
                item.admitted = False
            raise


def _with_remedy(error, cause):
    """``error`` carrying ``cause``'s remedy, which a failed event shows."""
    remedy = getattr(cause, "remedy", None)
    if isinstance(remedy, str) and remedy:
        error.remedy = remedy
    return error


class AsyncDomainWrfoutWriter:
    """One domain's side-stream D2H staging and dedicated writer thread."""

    #: Which domain this writer is, and who to tell when one of its
    #: frames becomes durable.  CLASS attributes, not only instance
    #: ones: this writer is also stood up field-by-field through
    #: ``object.__new__`` by CPU-only harnesses that cannot allocate a
    #: CuPy stream (tests/test_wrfout.py builds exactly that shell), and
    #: an optional feature that only exists on the ``__init__`` path
    #: would turn every such construction into an AttributeError on the
    #: worker thread.  Declaring the default here is also where a reader
    #: looks to learn the feature is optional.
    grid_id = None
    landing_observer = None
    #: This domain's ``[output]`` history-variable selection
    #: (:class:`woof.io.history_selection.HistorySelection`), or ``None``
    #: -- which means the same thing ``HistorySelection.FULL`` does:
    #: write every variable the run produces and stamp no attribute.
    #: Declared here for the same reason the two above are: the test
    #: suite builds CPU-only shells of this class with
    #: ``object.__new__``, and an optional feature that exists only on
    #: the ``__init__`` path turns every such construction into an
    #: AttributeError on the worker thread.
    history_selection = None
    _identity_pending = False
    #: Field bytes of every admitted frame not yet written, and of the
    #: written frame whose output identity is still being hashed.  Class
    #: defaults for the same ``object.__new__`` shells as above.
    _pending_bytes = 0
    _identity_bytes = 0
    _device = None
    #: Frames this writer has made durable, ever.  Monotonic (a history
    #: rewind edits ``paths`` but never this), so a checkpoint writer on
    #: another thread can wait for "every frame queued before me" without
    #: draining frames queued after it (:meth:`durability_target`).
    _landed_total = 0
    #: One timing record per written frame, keyed by its path: where the
    #: frame's wall went on the writer thread (queue, D2H wait, host
    #: assembly, NetCDF write, identity).  Class default for shells.
    _frame_timings: dict = {}

    #: An in-memory consumer of this domain's frames (a resident member
    #: session's :class:`woof.io.frame_sink.FrameSink`), called on the
    #: writer thread with the composed host fields the file is written from,
    #: before the write.  ``write_files = False`` makes the frame land in the
    #: sink only (a member whose history files nothing reads).  Class
    #: defaults: every existing writer has neither and writes exactly as it
    #: did.
    frame_sink = None
    write_files = True

    @staticmethod
    def _new_ticket_queue() -> queue.Queue:
        # Unbounded by count: admission is bounded by BYTES instead
        # (_admit, _pending_budget_bytes).  A one-ticket queue blocked the
        # step loop whenever two frames were waiting behind the one being
        # written: on V61 (8x RTX 5090, HRRR v4 recipe, 15-minute history,
        # 2026-10-21) an 18.7 GB frame took about 40 s to land while the
        # model made the next 15 minutes in about 15 s, so steps stalled
        # 10-30 s at every write and the stall grew over the run.
        return _AsyncTicketQueue(maxsize=0)

    @staticmethod
    def _pending_budget_bytes() -> int:
        """Host bytes of admitted, unwritten frames before the producer waits.

        A quarter of physical memory, at most 64 GiB
        (WOOF_HISTORY_QUEUE_BYTES overrides).  One frame is always admitted
        whatever its size, so a frame larger than the budget still runs, as
        it did with the one-ticket queue.
        """
        override = os.environ.get("WOOF_HISTORY_QUEUE_BYTES")
        if override:
            return max(0, int(float(override)))
        try:
            physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, ValueError, OSError):
            physical = 16 * 1024 ** 3
        return int(min(64 * 1024 ** 3, physical // 4))

    def __init__(self, *, nx, ny, nz, dx, dy, title, global_attrs,
                 abort_event=None, soil_layers=None, grid_id=None,
                 landing_observer=None, history_selection=None, device=None):
        import cupy as cp

        #: Which domain this writer is, and who to tell when one of its
        #: frames becomes durable.  Both optional: every existing caller
        #: constructs this writer without them and is unaffected.  The
        #: observer is invoked ON THE WRITER THREAD, after the file has
        #: been fsynced, self-validated and renamed onto its final name
        #: -- so it is told about a file that exists, never one that is
        #: merely queued.
        self.grid_id = None if grid_id is None else int(grid_id)
        self.landing_observer = landing_observer
        self.history_selection = history_selection
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        # Same contract as WrfoutWriter: the only production caller
        # (open_domain_writer below) resolves this from the domain's cfg.
        self.soil_layers = (NO_LAND_SURFACE_SOIL_LAYERS
                            if soil_layers is None else int(soil_layers))
        self.dx, self.dy = float(dx), float(dy)
        self.title = title
        self.global_attrs = dict(global_attrs)
        self._device = cp.cuda.Device(device)
        with self._device:
            self.stream = cp.cuda.Stream(non_blocking=True)
        self._queue = self._new_ticket_queue()
        self._condition = threading.Condition()
        self._pending = 0
        self._failure: BaseException | None = None
        self._failure_traceback: str | None = None
        self._closed = False
        self._abort_event = (threading.Event() if abort_event is None
                             else abort_event)
        self.paths: list[Path] = []
        self._completed_records = []
        self._frame_timings = {}
        self._thread = threading.Thread(
            target=self._worker, name=f"gpuwm-wrfout-{id(self):x}",
            daemon=True)
        self._thread.start()

    def _device_context(self):
        # CPU-only writer shells have no CUDA owner. Real writers always do.
        return self._device if self._device is not None else nullcontext()

    @property
    def pending(self) -> int:
        with self._condition:
            return self._pending + int(self._identity_pending)

    @property
    def pending_work_bytes(self) -> int:
        """Bytes this writer still moves before :meth:`drain` returns.

        A frame not yet written is written and then read back once for its
        output identity; a written frame whose identity is in progress is
        read once.  The run's finalization heartbeat declares this so the
        supervisor can bound the drain by its size.
        """
        with self._condition:
            return 2 * self._pending_bytes + self._identity_bytes

    @property
    def completed_records(self):
        with self._condition:
            return tuple(getattr(self, "_completed_records", ()))

    def _raise_failure(self) -> None:
        if self._failure is not None:
            # This wrapper is what a run's failed event and its map page
            # show, so it carries the cause's words and remedy; a bare
            # "writer failed" left the reader with nothing to act on.
            failure = self._failure
            raise _with_remedy(RuntimeError(
                "per-domain wrfout writer failed: "
                f"{str(failure) or type(failure).__name__}"),
                failure) from failure

    def _peer_failure(self):
        """The FIRST failure recorded on the shared abort event, if any.

        The abort event is shared across every domain's writer; the
        exception itself lives on the writer that hit it.  Before this
        attribute existed, a run whose d01 writer failed reported only
        d02's bare 'writing was aborted' -- the root cause was recorded
        on an object nothing consulted and appeared in no log, capsule,
        or traceback (measured: a 6 h moving-nest run whose only
        diagnostic was that one sentence).
        """
        return getattr(self._abort_event, "first_failure", None)

    def _check_admission_liveness(self) -> None:
        """Reject work that no live, healthy worker can consume."""
        self._raise_failure()
        if not self._thread.is_alive():
            # Recheck after observing thread death so a just-published worker
            # exception remains the cause reported to the producer.
            self._raise_failure()
            raise RuntimeError("per-domain wrfout worker stopped unexpectedly")
        if self._abort_event.is_set():
            self._raise_failure()
            peer = self._peer_failure()
            if peer is not None:
                raise _with_remedy(RuntimeError(
                    "per-domain wrfout writing was aborted by another "
                    f"domain's writer: {peer[1]}"), peer[0]) from peer[0]
            raise RuntimeError("per-domain wrfout writing was aborted")

    def _admit(self, ticket: _AsyncFrame) -> None:
        """Admit one staged frame with bounded, liveness-aware waits."""
        ticket.admitted_at = time.perf_counter()
        ticket.nbytes = (int(ticket.deferred["frame"].nbytes)
                         if ticket.deferred is not None
                         else _frame_nbytes(ticket.fields))
        budget = self._pending_budget_bytes()
        waited = 0.0
        with self._condition:
            # Wait only while frames are already pending AND this one would
            # take the backlog past the byte budget: the step loop is never
            # blocked by an ordinary write cadence, only by a backlog that
            # host memory cannot hold.
            while (self._pending > 0
                   and self._pending_bytes + ticket.nbytes > budget):
                self._check_admission_liveness()
                started = time.perf_counter()
                self._condition.wait(timeout=_ASYNC_WRITER_POLL_SECONDS)
                waited += time.perf_counter() - started
            self._pending += 1
            self._pending_bytes += ticket.nbytes
        ticket.admission_wait_s = round(waited, 6)
        try:
            while True:
                self._check_admission_liveness()
                try:
                    self._queue.put(
                        ticket, timeout=_ASYNC_WRITER_POLL_SECONDS)
                except queue.Full:
                    continue
                return
        finally:
            # Queue.put may unwind for cancellation, KeyboardInterrupt, or a
            # queue implementation failure.  Admission truth is written by
            # _AsyncTicketQueue._put under the queue mutex, not inferred from
            # whether this producer observed Queue.put return.
            if not ticket.admitted:
                with self._condition:
                    self._pending -= 1
                    self._pending_bytes -= ticket.nbytes
                    self._condition.notify_all()

    def _frame_lead_seconds(self, valid_time, global_attrs=None):
        """Seconds from the run's START_DATE to ``valid_time``, or None."""
        start = (global_attrs or self.global_attrs or {}).get("START_DATE")
        if start is None or valid_time is None:
            return None
        try:
            begin = datetime.strptime(str(start), "%Y-%m-%d_%H:%M:%S")
            if isinstance(valid_time, datetime):
                valid = valid_time.replace(tzinfo=None)
            else:
                valid = datetime.strptime(str(valid_time)[:19].replace("T", "_"),
                                          "%Y-%m-%d_%H:%M:%S")
        except (TypeError, ValueError):
            return None
        return (valid - begin).total_seconds()

    def _history_plan(self, produced, selection=None, lead_seconds=None) -> tuple[frozenset[str], dict]:
        """(names to write, global attributes recording the selection).

        Resolved ONCE per frame, before a single byte is staged, because
        that is what makes the selection worth having: a dropped 3-D
        volume must never cross the PCIe bus, and filtering at write
        time would copy every one of them host-ward first.  On a 400^2 x
        50 domain that is the majority of what the user is trying to
        save.

        A ``None`` selection -- and ``preset = "full"`` with no drop
        list -- keeps everything and stamps nothing, so a default run's
        frame and its header are exactly what they were before this
        surface existed.
        """
        # A per-frame selection (a quick-look frame's subset) overrides
        # the writer's standing one for that frame only; the standing one
        # takes the between-hours subset at a sub-hourly lead.
        if selection is None:
            selection = self.history_selection
            if selection is not None:
                selection = selection.for_lead(lead_seconds)
        if selection is None or selection.writes_everything:
            return frozenset(produced), {}
        return frozenset(selection.select(produced)), \
            selection.wrfout_attrs(produced)

    def _frame_attrs(self, global_attrs, history_attrs) -> dict | None:
        """The ticket's global-attribute override, or ``None`` for none.

        The history stamp has to reach the FILE, so it is folded onto
        whichever attribute set this frame was going to carry: the
        per-frame override when the caller supplied one (carrier
        provenance), the writer's standing set otherwise.
        """
        if not history_attrs:
            return None if global_attrs is None else dict(global_attrs)
        base = self.global_attrs if global_attrs is None else global_attrs
        return {**base, **history_attrs}

    def submit(self, path, valid_time, state, *, extra_fields=None,
               refl_field=None, frame=None, global_attrs=None, completed_observer=None,
               history_selection=None) -> None:
        """Queue a nonblocking, stream-ordered snapshot of one domain.

        ``frame`` is a COMPLETE host frame, already assembled, and it is the
        seam an out-of-core domain publishes through.  A streamed domain has
        no resident ``DomainState`` to read: ``woof.core.streaming.attach``
        copies the carriers into a pinned host store and the sweep writes
        the STORE, so ``state`` still exists, still has the right shapes,
        and still holds the values it held at t = 0 forever.  Passing it
        here is the shipped defect this parameter closes -- a full run's
        worth of frames that are correct in inventory, in Times and in
        global attributes, and frozen at the initial condition
        (``tilestream.test_history.negative_frame_from_state`` measures how
        much of a frame that freezes and requires it to differ).

        With ``frame`` given, ``state`` is not read at all -- pass ``None``
        -- and there is no D2H: the fields are already host arrays, so the
        side-stream staging this class exists for becomes a no-op and the
        writer thread is fed directly.  ``tilestream.output.StoreFrame``
        builds exactly such a frame off the pinned store, in the device
        frame's own field order (which is essential: HDF5 lays its name
        heap out in variable-creation order, and the same numbers in a
        different order give a file that hashes differently while every
        variable in it compares equal).

        THE FRAME'S ARRAYS ARE BORROWED, NOT COPIED, and that is the caller's
        problem to get right, exactly as it is for ``state``.  With
        ``StoreFrame(overlap=False)`` the carriers are zero-copy views of the
        pinned store -- which is what makes an out-of-core frame 2.2x-2.4x
        cheaper than the resident path -- so the caller must ``drain()``
        before it sweeps again, or use ``overlap=True`` and pay one host
        memcpy for a snapshot the writer thread may hold.  Submitting views
        and sweeping on writes a frame the scatter is mutating underneath;
        it is a property, not a race, and
        ``tilestream.test_history.negative_async_no_drain`` asserts it.

        ``global_attrs`` is a per-frame global-attribute override (carrier
        provenance above all); it rides the ticket so the snapshot taken at
        submit time lands on exactly the file that carries this frame.
        """
        import cupy as cp

        # A caller can finish another rank with a different card current.
        # Stream entry does not select its card in CuPy.
        with self._device_context():
            if self._closed:
                raise RuntimeError("cannot submit to a closed wrfout writer")
            self._raise_failure()
            producer = cp.cuda.get_current_stream()
            if state is not None and getattr(state, "_moving_frame_velocity", None):
                from woof.moving_frame import moving_frame_attrs
                base = self.global_attrs if global_attrs is None else global_attrs
                global_attrs = {**base, **moving_frame_attrs(state)}
            if state is not None:
                fire_attrs = fire_history_attrs(getattr(state, "physics", None))
                if fire_attrs:
                    base = self.global_attrs if global_attrs is None else global_attrs
                    global_attrs = {**base, **fire_attrs}
            try:
                if frame is not None:
                    if state is not None:
                        raise ValueError(
                            "submit() was given both a prepared host frame and a "
                            "device state; they are two different domains' worth of "
                            "numbers and there is no rule for which wins.  A "
                            "streamed domain passes state=None.")
                    self._admit_host_frame(path, valid_time, frame,
                                           extra_fields=extra_fields,
                                           refl_field=refl_field, producer=producer,
                                           global_attrs=global_attrs,
                                           completed_observer=completed_observer,
                                           history_selection=history_selection)
                    return
                self._queue.release_completed_device_refs(self.stream)
                device_fields = _device_state_frame(
                    state, include_diagnostic_pressure=True)
                if refl_field is not None:
                    device_fields["REFL_10CM"] = refl_field
                # The [output] selection, resolved BEFORE the staging loop below
                # so a dropped field costs no D2H at all -- see _history_plan.
                produced = list(device_fields)
                produced.extend(name for name in (extra_fields or ())
                                if name not in device_fields)
                keep, history_attrs = self._history_plan(
                    produced, history_selection,
                    self._frame_lead_seconds(valid_time, global_attrs))
                ready = cp.cuda.Event()
                ready.record(producer)
                self.stream.wait_event(ready)

                host_fields: dict[str, np.ndarray] = {}
                device_refs: list[object] = []
                pinned_refs: list[object] = []
                with self.stream:
                    # Freeze live state in device memory before starting D2H.
                    # The next step needs only this GPU copy to finish; waiting
                    # for the PCIe download made history a step-stream barrier.
                    # Share exact aliases (notably T/THM) in both staging tiers.
                    snapshots = {}
                    aliases = {}
                    # The device snapshots live until their D2H finishes, while
                    # the next steps already run, so they may only take memory
                    # the device has never handed to the pool: the pool's
                    # cached blocks are the steps' transient workspace.  A
                    # field past that budget is copied straight to pinned
                    # host memory on the producer stream instead (the step
                    # waits for that one PCIe copy).  Breakage this prevents:
                    # full-CONUS HRRR resident on one 96 GB card died at its
                    # first history frame, "Out of memory allocating
                    # 68,585,472 bytes (allocated so far: 99,013,528,576
                    # bytes)" at this snapshot loop, with every step fitting.
                    snapshot_budget = _history_snapshot_budget()
                    direct = 0
                    for name, value in device_fields.items():
                        if name not in keep:
                            continue
                        key = id(value)
                        if key not in aliases:
                            if isinstance(value, np.ndarray):
                                aliases[key] = np.array(value, copy=True, order="C", subok=False)
                            elif int(value.nbytes) <= snapshot_budget:
                                snapshot_budget -= int(value.nbytes)
                                aliases[key] = cp.array(value, copy=True, order="C")
                            else:
                                source = (value if value.flags.c_contiguous
                                          else cp.ascontiguousarray(value))
                                memory = cp.cuda.alloc_pinned_memory(int(source.nbytes))
                                host = np.frombuffer(
                                    memory, dtype=source.dtype,
                                    count=source.size).reshape(source.shape)
                                source.get(out=host, stream=producer, blocking=True)
                                pinned_refs.append(memory)
                                aliases[key] = host
                                direct += 1
                        snapshots[name] = aliases[key]
                    if direct and not getattr(self, "_direct_staging_noted", False):
                        self._direct_staging_noted = True
                        print(f"note: history: {direct} field(s) of this frame "
                              "copied straight to host (the device has no room "
                              "beside the steps' workspace for their snapshots)",
                              file=sys.stderr, flush=True)
                    captured = cp.cuda.Event()
                    captured.record(self.stream)
                producer.wait_event(captured)

                with self.stream:
                    staged = {}
                    for name, value in device_fields.items():
                        if name not in keep:
                            continue
                        key = id(value)
                        if key in staged:
                            host_fields[name] = staged[key]
                            continue
                        value = snapshots[name]
                        if isinstance(value, np.ndarray):
                            # Already on host: staging it through the device would
                            # be a pure bounce (measured ~1.8 GiB of cached device
                            # staging across the initial frames).  np.array preserves
                            # a scalar P_TOP's 0-D shape; np.ascontiguousarray would
                            # silently promote it to (1,) and give it a vertical dim.
                            host_fields[name] = staged[key] = value
                            continue
                        array = value
                        memory = cp.cuda.alloc_pinned_memory(int(array.nbytes))
                        host = np.frombuffer(memory, dtype=array.dtype,
                                             count=array.size).reshape(array.shape)
                        array.get(out=host, stream=self.stream, blocking=False)
                        host_fields[name] = staged[key] = host
                        device_refs.append(array)
                        pinned_refs.append(memory)
                    for name, value in (extra_fields or {}).items():
                        # Prognostic/state-derived fields win (notably child HGT,
                        # which is blended while static HGT_M remains unblended).
                        if name not in host_fields and name in keep:
                            host_fields[name] = _contiguous_preserving_shape(value)
                    done = cp.cuda.Event()
                    done.record(self.stream)
                # Only the worker waits for D2H. Device snapshots remain owned
                # by the ticket until that event completes, independently of
                # the live state which the next step can already mutate.
                ticket = _AsyncFrame(
                    path=Path(path),
                    time_str=valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
                    fields=host_fields, event=done,
                    completed_observer=completed_observer,
                    device_refs=tuple(device_refs),
                    pinned_refs=tuple(pinned_refs),
                    valid_time=valid_time,
                    global_attrs=self._frame_attrs(global_attrs, history_attrs))
                self._admit(ticket)
            finally:
                # CuPy stream contexts restore their context stack, which can
                # differ from a producer selected with Stream.use(). Keep the
                # caller on the stream that waits for this frame's capture.
                producer.use()


    def _admit_host_frame(self, path, valid_time, frame, *, extra_fields,
                          refl_field, producer, global_attrs=None, completed_observer=None,
                          history_selection=None) -> None:
        """Publish a frame that is already on the host.

        The whole of ``submit``'s device machinery collapses here and it is
        worth saying which parts and why, because each one was doing a job
        for the resident path that an out-of-core frame has already done:

        * no ``_device_state_frame`` -- the caller assembled the frame;
        * no pinned staging and no ``array.get`` -- the arrays are host
          arrays, and staging them THROUGH the device would be the pure
          bounce the resident path already refuses for its own host-side
          fields (measured ~1.8 GiB of cached device staging);
        * no ``device_refs`` to release and no ``pinned_refs`` to hold --
          the writer thread reads the caller's arrays directly.

        The event is still recorded on the producing stream and still
        waited on by the worker.  It fences nothing of this frame's, but a
        streamed run's REFL_10CM is the exception that makes it necessary
        rather than ceremonial: ``refl_field`` may be a live device array
        (a resident domain's) or a host one (a streamed domain's scattered
        store slot), and the device case needs the same ordering guarantee
        every other device field gets.
        """
        import cupy as cp

        deferred = frame if getattr(frame, "deferred", False) else None
        frame = {} if deferred is not None else dict(frame)
        # Same [output] selection, same place in the order: before any
        # staging.  A streamed domain's carriers are already on the host,
        # but REFL_10CM may still be a live device array, and a dropped
        # name must not be copied, borrowed or declared either way.
        produced = (list(deferred.names) if deferred is not None
                    else list(frame))
        if refl_field is not None and "REFL_10CM" not in produced:
            produced.append("REFL_10CM")
        produced.extend(name for name in (extra_fields or ())
                        if name not in produced)
        keep, history_attrs = self._history_plan(
            produced, history_selection,
            self._frame_lead_seconds(valid_time, global_attrs))
        host_fields: dict[str, np.ndarray] = {}
        device_refs: list[object] = []
        pinned_refs: list[object] = []
        ready = cp.cuda.Event()
        ready.record(producer)
        self.stream.wait_event(ready)
        with self.stream:
            staged = [(name, self._host_or_staged(value, device_refs,
                                                  pinned_refs))
                      for name, value in frame.items() if name in keep]
            refl = None
            if refl_field is not None and "REFL_10CM" in keep:
                refl = self._host_or_staged(
                    refl_field, device_refs, pinned_refs)
            extra = {name: _contiguous_preserving_shape(value)
                     for name, value in (extra_fields or {}).items()
                     if name in keep}
            if deferred is None:
                host_fields = _compose_host_frame(staged, refl, extra)
            done = cp.cuda.Event()
            done.record(self.stream)
        producer.wait_event(done)
        self._admit(_AsyncFrame(
            path=Path(path),
            time_str=valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
            fields=host_fields, event=done,
            completed_observer=completed_observer,
            device_refs=tuple(device_refs),
            pinned_refs=tuple(pinned_refs),
            valid_time=valid_time,
            global_attrs=self._frame_attrs(global_attrs, history_attrs),
            deferred=(None if deferred is None else
                      {"frame": deferred, "keep": keep, "refl": refl,
                       "extra": extra})))

    def _host_or_staged(self, value, device_refs, pinned_refs) -> np.ndarray:
        """A host array for one field, staging it off the device if needed."""
        import cupy as cp

        if isinstance(value, np.ndarray):
            # Borrowed, NOT copied -- see submit's docstring.  Copying here
            # would silently make every out-of-core frame cost a full extra
            # host pass and would disarm the drain discipline by making the
            # unsafe call safe, which is the wrong direction for a defect
            # that is otherwise invisible.
            return value
        array = _contiguous_preserving_shape(value, cp)
        memory = cp.cuda.alloc_pinned_memory(int(array.nbytes))
        host = np.frombuffer(memory, dtype=array.dtype,
                             count=array.size).reshape(array.shape)
        array.get(out=host, stream=self.stream, blocking=False)
        device_refs.append(array)
        pinned_refs.append(memory)
        return host

    def _worker(self) -> None:
        # CUDA current devices are thread-local. A new writer thread starts
        # on card 0 even when its stream and buffers belong to another card.
        with self._device_context():
            self._worker_on_device()

    def _worker_on_device(self) -> None:
        if not hasattr(self, "_completed_records"):
            self._completed_records = []
        while True:
            ticket = self._queue.get()
            if ticket is None:
                self._queue.task_done()
                return
            staging_consumed = False
            landed = False
            writer = None
            ticket_bytes = int(getattr(ticket, "nbytes", 0) or 0)
            dequeued = time.perf_counter()
            timing = {"time": getattr(ticket, "time_str", None),
                      "bytes": ticket_bytes,
                      # Seconds the stepping thread waited to admit this
                      # frame (0 unless the byte budget was full).
                      "admission_wait_s": float(
                          getattr(ticket, "admission_wait_s", 0.0) or 0.0),
                      "queued_s": round(dequeued - float(
                          getattr(ticket, "admitted_at", 0.0) or dequeued), 6)}
            try:
                ticket.event.synchronize()
                # D2H is complete.  Drop device ownership on the side stream
                # that consumed the arrays, while retaining pinned backing
                # until NetCDF has finished reading the host field views.
                with self.stream:
                    ticket.device_refs = ()
                if ticket.deferred is not None:
                    # Assembled HERE, off the stepping thread: waits for the
                    # frame's own download, then the same host derivation
                    # and the same field order as an admitted frame.
                    held, ticket.deferred = ticket.deferred, None
                    ticket.fields = _compose_host_frame(
                        ((name, value) for name, value
                         in held["frame"].materialize().items()
                         if name in held["keep"]),
                        held["refl"], held["extra"])
                    waited = getattr(held["frame"], "wait_seconds", None)
                    if waited is not None:
                        timing["copy_wait_s"] = round(float(waited), 6)
                assembled = time.perf_counter()
                timing["copy_and_assemble_s"] = round(assembled - dequeued, 6)

                sink = self.frame_sink
                if sink is not None:
                    # The fields the file is written from, on this thread,
                    # before the write: a sink that fails fails the frame.
                    sink(grid_id=self.grid_id, valid_time=ticket.valid_time,
                         time_str=ticket.time_str, fields=ticket.fields,
                         nx=self.nx, ny=self.ny, nz=self.nz,
                         global_attrs=(ticket.global_attrs
                                       if ticket.global_attrs is not None
                                       else self.global_attrs))
                if not self.write_files:
                    ticket.fields = {}
                    ticket.pinned_refs = ()
                    # This frame is durable in the sink. It has no file
                    # publication, but the common completion path still
                    # records its timing after releasing staging ownership.
                    timing_path = str(ticket.path)
                    timing["sink_only"] = True
                    timing["durable_after_s"] = round(time.perf_counter() - float(
                        getattr(ticket, "admitted_at", 0.0) or dequeued), 6)
                    landed = True
                    continue
                with _NETCDF4_IO_LOCK:
                    if self._abort_event.is_set():
                        continue
                    try:
                        with WrfoutWriter(
                                ticket.path, nx=self.nx, ny=self.ny, nz=self.nz,
                                dx=self.dx, dy=self.dy, title=self.title,
                                global_attrs=(
                                    ticket.global_attrs
                                    if ticket.global_attrs is not None
                                    else self.global_attrs),
                                soil_layers=self.soil_layers,
                                field_schema=ticket.fields,
                                _retain_identity_handle=True) as writer:
                            writer.write_frame(ticket.time_str, ticket.fields)
                            # write_frame synchronously consumes every host
                            # view. Native finish/fsync, validation and hashing
                            # no longer need the borrowed staging arrays: a
                            # streamed next step may reuse them immediately.
                            ticket.fields = {}
                            ticket.pinned_refs = ()
                            with self._condition:
                                self._identity_pending = True
                                self._identity_bytes = ticket_bytes
                                self._pending -= 1
                                self._pending_bytes -= ticket_bytes
                                staging_consumed = True
                                self._condition.notify_all()
                    except BaseException:
                        # Set the experiment-wide abort while still holding
                        # the netCDF lock.  A later domain cannot slip into a
                        # file session between this failure and publication
                        # cancellation.
                        self._abort_event.set()
                        raise
                written = time.perf_counter()
                timing["write_s"] = round(written - assembled, 6)
                published = getattr(writer, "publication_revision", None)
                if published is not None:
                    proof = writer.complete_output_identity(cancel_event=self._abort_event)
                    if ticket.completed_observer is not None:
                        ticket.completed_observer(proof, ticket.valid_time)
                    with self._condition:
                        self._completed_records.append(proof)
                self.paths.append(ticket.path)
                landed = True
                done_at = time.perf_counter()
                timing["identity_s"] = round(done_at - written, 6)
                timing["durable_after_s"] = round(done_at - float(
                    getattr(ticket, "admitted_at", 0.0) or dequeued), 6)
                timing_path = str(ticket.path)
                # The file is durable HERE and nowhere earlier: the
                # WrfoutWriter context above has exited, so its close()
                # completed the fsync, the self-validation and the
                # rename onto the final name, and its output identity is
                # ready. A failure at any of those
                # went to the handler below instead.  An observer that
                # raises must not take the writer thread down with it --
                # the run's outputs matter more than its telemetry.
                if self.landing_observer is not None:
                    try:
                        self.landing_observer(
                            domain=self.grid_id, valid_time=ticket.valid_time,
                            path=ticket.path)
                    except Exception:  # noqa: BLE001 - telemetry never fails a run
                        pass
            except BaseException as exc:
                failure_traceback = _stringify_and_clear_exception_tracebacks(
                    exc)
                with self._condition:
                    if self._failure is None:
                        self._failure = exc
                        self._failure_traceback = failure_traceback
                # Publish the ROOT CAUSE on the shared event before (or
                # atomically with) setting it, so every OTHER domain's
                # liveness refusal can name it instead of the bare
                # "writing was aborted" (see _peer_failure).
                if getattr(self._abort_event, "first_failure", None) is None:
                    self._abort_event.first_failure = (
                        exc, f"{type(exc).__name__}: {exc}")
                self._abort_event.set()
            finally:
                if writer is not None:
                    release = getattr(writer, "release_identity_handle", None)
                    if release is not None:
                        release()
                ticket.pinned_refs = ()
                self._queue.task_done()
                # Clear before publishing pending == 0 and before the next
                # blocking queue.get(), so an idle worker cannot retain the
                # completed frame or any of its arrays.
                ticket = None
                with self._condition:
                    if landed:
                        self._landed_total += 1
                    if staging_consumed:
                        self._identity_pending = False
                        self._identity_bytes = 0
                    else:
                        self._pending -= 1
                        self._pending_bytes -= ticket_bytes
                    self._condition.notify_all()
                if landed:
                    # Telemetry, after the counts above are published: read
                    # once the writer is drained (frame_timings).
                    self._frame_timings[timing_path] = timing

    def update_global_attrs(self, global_attrs) -> None:
        """Swap the per-file global attributes for frames submitted later.

        A relocated domain's placement, centre and corner attributes
        change at the move; frames produced before it must keep the
        attributes of the placement that produced them, so this drains
        the queue first and then swaps.  The worker reads
        ``self.global_attrs`` per file, so the swap is complete for every
        subsequent submit.
        """
        self.drain()
        self.global_attrs = dict(global_attrs)

    def durability_target(self) -> int:
        """The landed count at which every frame queued so far is durable."""
        with self._condition:
            return (self._landed_total + self._pending
                    + int(self._identity_pending))

    def wait_landed(self, target: int) -> None:
        """Block until :meth:`durability_target`'s frames are durable.

        Safe from any thread.  Raises the writer's failure, or names the
        breakage when a frame queued before ``target`` was taken settled
        without becoming durable (an abort, or a refused admission), so a
        waiting checkpoint can never be published over a missing frame.
        """
        while True:
            with self._condition:
                if self._landed_total >= target:
                    return
                settled = not self._pending and not self._identity_pending
                if (self._failure is None and self._thread.is_alive()
                        and not settled):
                    self._condition.wait(timeout=_ASYNC_WRITER_POLL_SECONDS)
                    continue
            self._raise_failure()
            raise RuntimeError(
                "a history frame queued before this checkpoint never became "
                "durable, so the checkpoint is not published: a restart "
                "from it would resume past a frame that does not exist")

    def drain(self) -> None:
        """Wait for durable files and their completed output identities."""
        self._drain(include_identity=True)

    def drain_staging(self) -> None:
        """Wait until native writes have released borrowed host arrays."""
        self._drain(include_identity=False)

    def _drain(self, *, include_identity) -> None:
        worker_stopped = False
        with self._condition:
            while self._pending or (include_identity and self._identity_pending):
                if not self._thread.is_alive():
                    worker_stopped = True
                    break
                self._condition.wait(timeout=_ASYNC_WRITER_POLL_SECONDS)
        if worker_stopped:
            self._raise_failure()
            raise RuntimeError(
                "per-domain wrfout worker stopped with pending tickets")
        self._raise_failure()

    def close(self) -> None:
        if self._closed:
            self._raise_failure()
            return
        saved: BaseException | None = None
        try:
            self.drain()
        except BaseException as exc:
            saved = exc
        finally:
            # Sentinel and join belong in finally: no live daemon may outlive
            # a failed close or keep publishing after unwind.  A dead worker
            # cannot consume a sentinel, and its queue may still be full.
            self._closed = True
            while self._thread.is_alive():
                try:
                    self._queue.put(
                        None, timeout=_ASYNC_WRITER_POLL_SECONDS)
                except queue.Full:
                    continue
                break
            self._thread.join()
        if saved is not None:
            raise saved
        self._raise_failure()


def carrier_provenance_attrs(physics) -> dict:
    """wrfout globals for the surface-radiation carrier contract.

    The contract (woof/core/radiation_carriers.py) promises that a run's
    carrier provenance -- who wrote GLW/SWDOWN/GSW/COSZEN and when --
    appears in the output metadata, so a reader of a wrfout can ask "did
    this file integrate a sky nobody computed" without the run directory.
    Built from :meth:`CarrierContract.report` at submit time, so the
    stamped provenance is the provenance of the frames in that file, not
    of the driver at construction.

    Fire coordinate identity accompanies these per-frame attributes when
    the driver owns a coupled fire grid. A state with neither carrier nor
    fire model returns an empty mapping.
    ``..._LAST_UPDATE`` is the producer's model second, ``-1.0`` for a
    source that is constant by declaration and has no age.
    """
    carriers = getattr(physics, "carriers", None)
    chemistry = chem_history_attrs(getattr(physics, "state", None))
    if carriers is None:
        return {**fire_history_attrs(physics), **chemistry}
    return {**_carrier_attrs(str(carriers.policy), carriers.report()),
            **fire_history_attrs(physics), **chemistry}


def chem_history_attrs(state, *, store=None):
    """Species-sized float64 mass vectors at this frame's model time."""
    if getattr(state, "chem", None) is None:
        return {}
    import json
    from woof.core.chem_driver import ledger_report
    return {"GPUWM_CHEM_LEDGER": json.dumps(ledger_report(state, store=store), sort_keys=True)}


def fire_history_attrs(physics) -> dict:
    """Fine-grid coordinate identity carried by this frame's fire model."""
    fire = getattr(physics, "fire", None)
    return {} if fire is None else dict(fire.output_attributes())


def streamed_carrier_provenance_attrs(streamed) -> dict:
    """The same globals, taken from a STREAMED domain's live ledger.

    :func:`carrier_provenance_attrs` reads the contract off a
    ``PhysicsDriver``, and under ``[tiles] store = "host"`` the driver the
    route holds is the snapshot the store was filled from -- it never ran
    radiation, so it reports ``unwritten`` for carriers whose values are in
    the file and correct.  ``StreamedDomain.carrier_provenance`` returns the
    ledger the sweep advances; this renders it in exactly the same keys, so
    a streamed wrfout and a resident one of the same configuration carry
    IDENTICAL provenance attributes rather than merely plausible ones.

    Returns ``{}`` when the streamed domain carries no contract, which sends
    the caller back to the state -- the same answer a pre-physics write
    gets, and not the same thing as a contract that says ``unwritten``.
    """
    provenance = None
    template = getattr(streamed, "template_state", None)
    chemistry = (chem_history_attrs(template, store=streamed._planning_store())
                 if getattr(template, "chem", None) is not None else {})
    reader = getattr(streamed, "carrier_provenance", None)
    if reader is not None:
        provenance = reader()
    if not provenance:
        return chemistry
    policy = provenance.get("policy")
    if policy is None:
        return chemistry
    return {**_carrier_attrs(str(policy), provenance["records"]), **chemistry}


def _carrier_attrs(policy: str, rows) -> dict:
    """Render one carrier ledger as wrfout globals.

    Shared by the resident and streamed readers so the two cannot drift
    into writing the same provenance under different keys or precisions --
    which is what the frame-parity gate compares attribute by attribute.
    """
    attrs = {"GPUWM_SURFACE_RADIATION_POLICY": str(policy)}
    for name, row in rows.items():
        key = str(name).upper()
        attrs[f"GPUWM_CARRIER_{key}_SOURCE"] = str(row["source"])
        last = row["last_update_model_time"]
        attrs[f"GPUWM_CARRIER_{key}_LAST_UPDATE"] = np.float64(
            -1.0 if last is None else last)
    return attrs


class PerDomainWrfoutWriters:
    """One asynchronous writer/side stream per domain in an experiment."""

    _simulated_radar = None
    _output_observer = None
    #: ``progress_callback.frame_submitted``, when it has one: told the
    #: moment each frame is handed over, so the landing it later reports is
    #: timed from the frame's own valid time and not from whatever step the
    #: model has reached by then (:meth:`woof.progress_log.StepLog.frame_submitted`).
    _frame_submitted = None

    #: The tree-wide ``[output]`` history selection this writer set was
    #: built with (``ExperimentConfig.output``), or ``None`` for the FULL
    #: default.  A class attribute so the verification cases and the
    #: idealized runners, which construct this object without one, are
    #: unaffected.
    history_selection = None

    #: The run's progress object, told before and after each write this
    #: set makes between two model steps (:meth:`attach_write_progress`).
    #: ``None`` publishes nothing, which is every caller that attaches none.
    write_progress = None

    def __init__(self, model, output_dir, *, start_time, title,
                 initial_condition=None, source=None,
                 progress_callback=None, history_selection=None,
                 episodes_by_grid_id=None, simulated_radar=None,
                 radar_output_dir=None):
        """``initial_condition`` is the preparation receipt's provenance
        block, stamped onto every domain's frames so the durable artifact
        states what its initial state was and not only when it began.

        ``progress_callback`` is the run's existing progress object.  If
        it carries an ``output_committed`` attribute (the optional hook
        :func:`woof.runtime._output_committed` discovers by name), each
        per-domain writer is given it and calls it as its frames become
        durable; anything else is ignored, so every existing caller is
        unaffected by passing one or by not passing one.

        A caller whose progress object does not exist yet at
        construction time uses :meth:`attach_progress_callback` instead;
        both spellings run the same one implementation.

        ``episodes_by_grid_id`` is the RESUME seed: a domain restored in
        the middle of its second lifecycle episode must write
        ``d0N/episode-002/`` from its FIRST frame, not from the next
        spawn boundary, or one episode's history lands in two places.
        The default -- empty -- gives every domain episode 0, which is
        the flat historical pathname, so a run that is not a lifecycle
        resume is byte-inert under this parameter.  Values come from
        :func:`woof.core.nest_lifecycle.output_episode`, the same
        function ``add_domain``'s caller uses at a live spawn boundary,
        so a resumed episode and a freshly born one are numbered by one
        rule.
        """
        from woof.runtime import _global_wrf_attrs, _metadata_frame

        self.model = model
        self.history_selection = history_selection
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = start_time
        self.title = title
        # Retained for refresh_domain: a relocated domain's later frames are
        # re-stamped with the same provenance the original writer carried.
        self._initial_condition = initial_condition
        self._source = source
        prepared = model._prepared_by_grid_id
        self._writers = {}
        self._metadata_by_grid_id = {}
        self._episode_by_grid_id = {}
        self._archived_paths = []
        self._archived_records = []
        #: Every final pathname THIS writer set has published, which is
        #: what the duplicate-valid-time guard in submit() is scoped to.
        self._published_paths = set()
        #: Submitted frames in order: (grid_id, path, stepping-thread s).
        self._frame_log = []
        self._captured_paths = []
        self._abort_event = threading.Event()
        self._output_observer = None
        self._simulated_radar = None
        resumed_episodes = {int(gid): int(episode) for gid, episode
                            in dict(episodes_by_grid_id or {}).items()}
        for node in model.walk_parent_first():
            case = prepared[node.cfg.grid_id]
            configured_start = getattr(node.cfg, "start_time", None)
            domain_start_time = (
                start_time if configured_start is None
                else configured_start)
            metadata = _metadata_frame(node.grid, case.static_fields)
            file_grid = getattr(case, "wrf_file_grid", None)
            self._metadata_by_grid_id[node.cfg.grid_id] = (
                {**metadata, **file_grid} if file_grid else metadata)
            self._episode_by_grid_id[node.cfg.grid_id] = resumed_episodes.get(
                int(node.cfg.grid_id), 0)
            self._writers[node.cfg.grid_id] = AsyncDomainWrfoutWriter(
                nx=node.cfg.run.nx, ny=node.cfg.run.ny, nz=node.cfg.run.nz,
                dx=node.cfg.run.dx, dy=node.cfg.run.dy, title=title,
                soil_layers=soil_layer_count(node.cfg.run),
                global_attrs=_global_wrf_attrs(
                    node.grid, domain_start_time,
                    getattr(case, "geog_selection", None),
                    domain=node.cfg, coord=case.initial_result.coord,
                    configured_dt=float(node.clock.spec.dt_fp32),
                    feedback=getattr(model, "_feedback_provenance", None),
                    initial_condition=initial_condition, source=source,
                    simulation_start_time=start_time,
                    static_fields=getattr(case, "static_fields", None)),
                abort_event=self._abort_event,
                grid_id=node.cfg.grid_id,
                history_selection=self._selection_for(node.cfg),
                device=_domain_output_device(node.state))
        self.last_durable_wrfout = None
        if simulated_radar is not None and simulated_radar.enabled:
            from woof.simulated_radar import LiveSimulatedRadar
            self._simulated_radar = LiveSimulatedRadar(
                simulated_radar, radar_output_dir or self.output_dir)
            for writer in self._writers.values():
                writer.landing_observer = self._notify_output
        if progress_callback is not None:
            self.attach_progress_callback(progress_callback)
            self.attach_write_progress(progress_callback)

    def _selection_for(self, domain_cfg):
        """One domain's ``[output]`` selection: its own, else the tree's.

        Per domain and not per tree because that is where the bytes
        actually are: a 1 km child at the same history cadence as its
        12 km parent writes an order of magnitude more of them, and
        "keep the parent whole, trim the child" has to be sayable.
        """
        from woof.io.history_selection import resolve

        return resolve(self.history_selection,
                       getattr(domain_cfg, "output", None))

    def attach_progress_callback(self, progress_callback) -> None:
        """Bind the output-landing hook onto every per-domain writer.

        The constructor keyword cannot serve every caller.  Both
        prepared runners build their progress closure OVER this object
        -- it reports ``writers.paths`` -- so the closure genuinely
        cannot exist before the thing it reads.  Reordering does not
        break a real cycle; late binding does, and this is it.

        Refused once any domain has submitted a frame.  Attaching after
        the first output has landed would skip it, and an event stream
        silently missing its first frame is worse than one that refused
        to start: the consumer has no way to notice.
        """

        busy = sorted(grid_id for grid_id, writer in self._writers.items()
                      if writer.paths or writer.pending)
        if busy:
            raise RuntimeError(
                "cannot attach an output observer after domain(s) "
                f"{busy} have already submitted frames; the observer "
                "would silently miss them.  Attach before the first "
                "history period, or pass progress_callback to the "
                "constructor.")
        self._output_observer = getattr(progress_callback, "output_committed", None)
        self._frame_submitted = getattr(progress_callback, "frame_submitted", None)
        observer = (self._notify_output if getattr(self, "_simulated_radar", None) is not None
                    else self._output_observer)
        for writer in self._writers.values():
            writer.landing_observer = observer

    def _notify_output(self, **event):
        # Requested radar output is checked by drain/close on the model
        # thread, since the writer treats landing callbacks as telemetry.
        if self._simulated_radar is not None:
            from woof.simulated_radar import inloop_enabled
            private = (inloop_enabled()
                       and getattr(self.model.node(int(event["domain"])).cfg, "quicklook_output", None) is not None)
            if not private:
                self._simulated_radar.output_committed(**event)
        if self._output_observer is not None:
            self._output_observer(**event)

    def attach_write_progress(self, progress_callback) -> None:
        """Tell ``progress_callback`` about each write between two steps.

        A history frame is submitted, and on a streamed domain written,
        after one model step's heartbeat and before the next, and a
        restart drains every queued frame there too.  A supervisor that
        hears nothing in that stretch times it as one model step: a
        1132x906x55 streamed forecast's last 5.24 GB frame took 81 s and,
        with the stop-tick checkpoint after it, went past the 120 s step
        bound, so a run with every step done was stopped.  With a progress
        object attached, each write publishes a ``writing:`` record that
        declares its bytes (:func:`woof.supervisor.writing_progress`).
        This is its own method because the runners attach their landing
        hooks through a fan-out that does not carry the progress object.
        """
        self.write_progress = progress_callback

    def _write_beat(self, phase: str, work_bytes: int | None):
        return writing_progress(self.write_progress, phase,
                                work_bytes=work_bytes)

    @property
    def pending(self) -> int:
        return sum(writer.pending for writer in self._writers.values())

    @property
    def pending_work_bytes(self) -> int:
        """Every domain's :attr:`AsyncDomainWrfoutWriter.pending_work_bytes`."""
        return sum(int(getattr(writer, "pending_work_bytes", 0) or 0)
                   for writer in self._writers.values())

    @property
    def paths(self) -> tuple[Path, ...]:
        ret = list(self._archived_paths)
        for gid in sorted(self._writers):
            ret.extend(self._writers[gid].paths)
        return tuple(ret)

    @property
    def captured_paths(self) -> tuple[Path, ...]:
        """Logical history cadence captured without claiming member files."""
        return tuple(getattr(self, "_captured_paths", ()))

    def frame_timings(self) -> list[dict]:
        """One record per submitted history frame, in submission order.

        ``submit_s`` is the stepping thread's wall inside :meth:`submit`
        (the frame's whole cost to the model); the rest is the writer
        thread's (:class:`AsyncDomainWrfoutWriter`): ``queued_s`` behind
        earlier frames, ``copy_and_assemble_s`` (the D2H wait,
        ``copy_wait_s`` of it on the ranked road, plus host assembly),
        ``write_s`` (NetCDF write, fsync, validation, rename),
        ``identity_s`` and ``durable_after_s`` from admission.
        """
        rows = []
        for grid_id, path, submit_s in getattr(self, "_frame_log", ()):
            writer = self._writers.get(grid_id)
            timing = dict(getattr(writer, "_frame_timings", {}).get(path, {}))
            rows.append({"grid_id": grid_id, "path": Path(path).name,
                         "submit_s": submit_s, **timing})
        return rows

    @property
    def completed_records(self):
        ret = list(getattr(self, "_archived_records", ()))
        for gid in sorted(self._writers):
            ret.extend(getattr(self._writers[gid], "completed_records", ()))
        return tuple(ret)

    #: Set by :meth:`attach_frame_sink`; class defaults leave every writer as it was.
    _frame_sink = None
    _write_files = True

    def attach_frame_sink(self, sink, *, write_files: bool = True) -> None:
        """Give every domain's frames to ``sink`` too (see
        :attr:`AsyncDomainWrfoutWriter.frame_sink`); ``write_files=False``
        keeps them in the sink only.  A domain added later gets the same."""
        self._frame_sink = sink
        self._write_files = bool(write_files)
        for writer in self._writers.values():
            writer.frame_sink = sink
            writer.write_files = bool(write_files)

    def add_domain(self, grid_id: int, *, grid, static_fields, episode: int = 0) -> None:
        """Mint a writer for a domain that appeared AFTER construction.

        A spawn-triggered nest does not exist when the writer set is
        built -- it is dormant, and its birth is a leg boundary in the
        middle of the run.  This is the same per-domain construction the
        constructor performs, for exactly one late arrival; refusing a
        second call for the same grid keeps it from silently discarding
        a writer that already holds queued frames.
        """
        from woof.runtime import _global_wrf_attrs, _metadata_frame

        grid_id = int(grid_id)
        if grid_id in self._writers:
            raise ValueError(
                f"d{grid_id:02d} already has a wrfout writer; a spawned "
                "domain is added once, and re-adding would drop frames "
                "the existing writer has queued (use refresh_domain to "
                "re-stamp metadata after a relocation)")
        node = self.model.node(grid_id)
        case = self.model._prepared_by_grid_id[grid_id]
        configured_start = getattr(node.cfg, "start_time", None)
        domain_start_time = (self.start_time if configured_start is None
                             else configured_start)
        self._metadata_by_grid_id[grid_id] = _metadata_frame(
            grid, static_fields)
        self._episode_by_grid_id[grid_id] = int(episode)
        self._writers[grid_id] = AsyncDomainWrfoutWriter(
            nx=node.cfg.run.nx, ny=node.cfg.run.ny, nz=node.cfg.run.nz,
            dx=node.cfg.run.dx, dy=node.cfg.run.dy, title=self.title,
            soil_layers=soil_layer_count(node.cfg.run),
            global_attrs=_global_wrf_attrs(
                grid, domain_start_time,
                getattr(case, "geog_selection", None),
                domain=node.cfg, coord=case.initial_result.coord,
                feedback=getattr(self.model, "_feedback_provenance", None),
                initial_condition=self._initial_condition,
                source=self._source,
                simulation_start_time=self.start_time,
                static_fields=static_fields),
            abort_event=self._abort_event,
            grid_id=grid_id,
            # Only [simulated_radar] listens for a late domain's frames.
            # Without it a spawned nest reports nothing, exactly as 2.8.3
            # wrote it: no output_committed events for those frames.
            landing_observer=(self._notify_output if self._simulated_radar is not None
                              else None),
            history_selection=self._selection_for(node.cfg),
            device=_domain_output_device(node.state))
        if self._frame_sink is not None:
            self._writers[grid_id].frame_sink = self._frame_sink
            self._writers[grid_id].write_files = self._write_files

    def remove_domain(self, grid_id: int) -> None:
        """Drain and close one retired episode without losing its paths."""
        grid_id = int(grid_id)
        writer = self._writers.pop(grid_id, None)
        if writer is None:
            return
        writer.drain()
        writer.close()
        self._archived_paths.extend(writer.paths)
        if not hasattr(self, "_archived_records"):
            self._archived_records = []
        self._archived_records.extend(getattr(writer, "completed_records", ()))
        if writer.paths:
            self.last_durable_wrfout = writer.paths[-1]
        self._metadata_by_grid_id.pop(grid_id, None)
        self._episode_by_grid_id.pop(grid_id, None)

    def refresh_domain(self, grid_id: int, *, grid, static_fields) -> None:
        """Re-derive one domain's frame metadata after a relocation.

        The XLAT/XLONG/MAPFAC/HGT metadata frame and the placement-carrying
        global attributes were computed at construction and would otherwise
        describe the footprint the domain no longer covers.  The route's
        relocation preparer calls this with the rebuilt child's grid and
        footprint-rebuilt statics; the writer drains first, so frames from
        the outgoing placement keep the coordinates that produced them.
        """
        grid_id = int(grid_id)
        node = self.model.node(grid_id)
        case = self.model._prepared_by_grid_id[grid_id]
        from woof.runtime import _global_wrf_attrs, _metadata_frame

        configured_start = getattr(node.cfg, "start_time", None)
        domain_start_time = (self.start_time if configured_start is None
                             else configured_start)
        metadata = _metadata_frame(grid, static_fields)
        file_grid = getattr(case, "wrf_file_grid", None)
        self._metadata_by_grid_id[grid_id] = (
            {**metadata, **file_grid} if file_grid else metadata)
        self._writers[grid_id].update_global_attrs(_global_wrf_attrs(
            grid, domain_start_time,
            getattr(case, "geog_selection", None),
            domain=node.cfg, coord=case.initial_result.coord,
            configured_dt=float(node.clock.spec.dt_fp32),
            feedback=getattr(self.model, "_feedback_provenance", None),
            initial_condition=self._initial_condition,
            source=self._source,
            simulation_start_time=self.start_time,
            static_fields=static_fields))

    def submit(self, node, ticks: int, *, refl_field=None) -> None:
        """One domain's history frame, from wherever that domain's truth is.

        For a resident domain that is ``node.state``, as it always was.  For
        a STREAMED one it is not: ``woof.core.streaming.attach`` copies the
        carriers into a pinned host store and the sweep advances the STORE,
        leaving ``node.state`` holding the initial condition for the rest of
        the run.  Submitting it produced a full run of frames with the
        correct inventory, the correct Times and the correct global
        attributes, every value frozen at t = 0 -- which is not a crash, not
        a warning, and not obviously wrong in ncview
        (``tilestream.test_history.negative_frame_from_state`` measures how
        much of a frame that freezes and requires it to differ).  So the
        source is asked of the domain rather than assumed, through the
        marker ``streaming.StreamedDomain`` leaves on the state it took
        over.
        """
        submitted = time.perf_counter()
        try:
            return self._submit(node, ticks, refl_field=refl_field,
                                submitted=submitted)
        finally:
            log = getattr(self, "_frame_log", None)
            if log is not None and log and log[-1][2] is None:
                log[-1][2] = round(time.perf_counter() - submitted, 6)

    def _capture_radar(self, node, valid_time, refl_field, *, quicklook=False):
        radar = getattr(self, "_simulated_radar", None)
        if radar is not None and getattr(node.cfg, "quicklook_output", None) is not None:
            from woof.simulated_radar import inloop_enabled, RADAR_SCRATCH_FRAMES, RADAR_SNAPSHOT_FIELDS
            if inloop_enabled():
                writer = self._writers[node.cfg.grid_id]
                # Snapshot staging and queue backpressure are output work, not a stalled model step.
                work_bytes = (RADAR_SCRATCH_FRAMES * 4 * (writer.nx + 1) * (writer.ny + 1)
                              * (len(RADAR_SNAPSHOT_FIELDS) * writer.nz + 15))
                with self._write_beat(f"radar-snapshot-d{int(node.cfg.grid_id):02d}", work_bytes):
                    radar.capture(node, valid_time, writer,
                                  self._metadata_by_grid_id[node.cfg.grid_id], refl_field=refl_field, quicklook=quicklook)

    def _submit(self, node, ticks: int, *, refl_field=None,
                submitted=None) -> None:
        if getattr(self, "_simulated_radar", None) is not None:
            self._simulated_radar.check()
        seconds = ticks / node.clock.tick_den
        valid_time = self.start_time + timedelta(seconds=seconds)
        self._capture_radar(node, valid_time, refl_field,
                            quicklook=quicklook_selection(node, ticks) is not None)
        episode = int(self._episode_by_grid_id.get(node.cfg.grid_id, 0))
        base_dir = self.output_dir
        # THE spelling, from `woof.render_layout`, because the delivered
        # PNG tree carries the same segment for the same frames: a
        # reader who found `d05/episode-002` here has to find that
        # episode's pictures under a folder spelled the same way, and
        # two independent format strings are two spellings waiting to
        # drift apart.
        segment = render_layout.episode_segment(episode)
        if segment is not None:
            base_dir = (self.output_dir / f"d{int(node.cfg.grid_id):02d}"
                        / segment)
            base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / wrfout_filename(valid_time, node.cfg.grid_id)
        # THE QUICK-LOOK STREAM (DomainConfig.full_history_interval_s): a
        # frame off the full lattice carries only the quick-look subset
        # and is written beside the tape, never as a wrfout_d* file.
        quicklook = quicklook_selection(node, ticks)
        static_attrs = {}
        if quicklook is not None:
            path = quicklook_path(base_dir, valid_time, node.cfg.grid_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            quicklook, static_attrs = self._quicklook_static(
                node, quicklook, moving=segment is not None)
        # SAME-RUN duplicates only.  The breakage this names is a lifecycle
        # or restart boundary publishing one valid time twice inside a
        # single run: the second frame's atomic replace destroys the first,
        # and neither the run nor the reader can tell.  A file left by a
        # PREVIOUS run is not that -- re-running into an existing output
        # directory has always overwritten, and refusing it here would
        # strand every repeat run instead of preventing a defect.
        if path in self._published_paths:
            raise RuntimeError(
                f"refusing to replace history frame {path}, which THIS run "
                f"already published for d{int(node.cfg.grid_id):02d}: a "
                "lifecycle/restart boundary produced a duplicate valid time, "
                "and the atomic replace would silently destroy the earlier "
                "frame. (A frame left by a PREVIOUS run at this path is "
                "replaced as it always has been.)")
        self._published_paths.add(path)
        if getattr(self, "_frame_log", None) is not None:
            self._frame_log.append([int(node.cfg.grid_id), str(path), None])
        if self._frame_submitted is not None:
            try:
                self._frame_submitted(domain=int(node.cfg.grid_id),
                                      valid_time=valid_time)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass
        from woof.ensemble.runtime_context import current_capture
        capture = current_capture()
        if capture is not None:
            metadata = self._metadata_by_grid_id[node.cfg.grid_id]
            with self._write_beat(f"ensemble-history-d{int(node.cfg.grid_id):02d}",
                                  capture.work_bytes(metadata)):
                capture.submit(
                    state=node.state,
                    streamed=getattr(node.state, "_streamed_domain", None),
                    metadata=metadata,
                    refl_field=refl_field, valid_time=valid_time,
                    grid_id=node.cfg.grid_id, episode=episode, clock=node.clock)
            if not hasattr(self, "_captured_paths"):
                self._captured_paths = []
            self._captured_paths.append(path)
            if not capture.keep_member_files:
                return
        writer = self._writers[node.cfg.grid_id]
        committed = (None if capture is None else capture.history_committer(
            grid_id=node.cfg.grid_id, episode=episode))
        # CARRIER PROVENANCE, snapshotted per frame.  Each valid time is
        # its own file, and the snapshot rides the ticket rather than the
        # writer's standing attribute set, so the provenance the driver
        # holds NOW lands on exactly the file that carries this frame --
        # a resumed or mid-run-forced carrier is labelled in the file it
        # affects -- without draining the async queue the way an
        # update_global_attrs swap must.
        #
        # ASKED OF THE DOMAIN, exactly as the frame below is.  Under
        # [tiles] store = "host" the driver on node.state is the snapshot
        # the store was filled from: radiation ran on the tile buffers, so
        # that contract still says every carrier is `unwritten` while the
        # frame beside it carries the sky those producers computed.  The
        # streamed reader falls back to the state's contract when the
        # domain carries none, so a resident run is unchanged.
        streamed = getattr(node.state, "_streamed_domain", None)
        carrier_attrs = {}
        if streamed is not None:
            carrier_attrs = streamed_carrier_provenance_attrs(streamed)
        if not carrier_attrs:
            carrier_attrs = carrier_provenance_attrs(
                getattr(node.state, "physics", None))
        frame_attrs = (None if not carrier_attrs
                       else {**writer.global_attrs, **carrier_attrs})
        if static_attrs:
            frame_attrs = {**(writer.global_attrs if frame_attrs is None
                              else frame_attrs), **static_attrs}
        phase = f"history-d{int(node.cfg.grid_id):02d}"
        if streamed is not None:
            # A streamed domain's numbers live in the pinned host store;
            # the provenance snapshot applies to its frames the same way.
            # Its frame is written HERE, between two model steps: the
            # frame is assembled off the store, then this thread waits for
            # the frames queued ahead of it and for its own write.  The
            # first record covers the assembly and the second, once the
            # frame's size is known, declares the bytes still to move:
            # everything queued plus this frame, written and read back.
            with self._write_beat(phase, self.pending_work_bytes):
                frame = streamed.history_fields()
                deferred = bool(getattr(frame, "deferred", False))
                frame_bytes = (int(frame.nbytes) if deferred
                               else _frame_nbytes(frame))
                with self._write_beat(
                        phase, self.pending_work_bytes + 2 * frame_bytes):
                    writer.submit(
                        path, valid_time, None, frame=frame,
                        extra_fields=self._metadata_by_grid_id[
                            node.cfg.grid_id],
                        refl_field=refl_field,
                        global_attrs=frame_attrs,
                        completed_observer=committed,
                        history_selection=quicklook)
                    # The next sweep may reuse these pinned views once the
                    # native write releases them. Identity hashing can
                    # continue on the worker; final drain/close still wait
                    # for the complete record.
                    consume_staging = getattr(writer, "drain_staging",
                                              writer.drain)
                    if deferred:
                        # The ranked road: no sweep writes the store, so the
                        # model steps on while the writer downloads and
                        # writes; the next write INTO the store waits here
                        # first (tilestream.ranks.RankedRun.add_store_guard).
                        streamed.add_store_guard(consume_staging)
                    else:
                        consume_staging()
            return
        # A resident frame is staged device to host and queued; this thread
        # waits only for room in the queue, behind the frames already in it.
        with self._write_beat(phase, self.pending_work_bytes):
            writer.submit(
                path, valid_time, node.state,
                extra_fields=self._metadata_by_grid_id[node.cfg.grid_id],
                refl_field=refl_field,
                global_attrs=frame_attrs,
                completed_observer=committed,
                history_selection=quicklook)

    def _quicklook_static(self, node, selection, *, moving: bool):
        """(selection, global attributes) for one quick-look frame.

        A fixed domain's quick-look frame leaves PHB out and names the
        domain's first full frame (``../wrfout_dNN_<first history time>``)
        in :data:`~woof.io.history_selection.STATIC_FRAME_ATTR`, where a
        reader finds it.  A domain that moves (a follow, or relocation
        episodes) keeps PHB in every frame and names nothing.
        """
        from woof.io.history_selection import (QuickLookSelection,
                                                STATIC_FRAME_ATTR)

        if not isinstance(selection, QuickLookSelection):
            return selection, {}
        if moving or getattr(node.cfg, "follow", None) is not None:
            return selection.on_moving_domain(), {}
        spec = node.clock.spec
        first = self.start_time + timedelta(seconds=float(Fraction(
            int(spec.start_ticks) + int(spec.history_begin_ticks),
            int(node.clock.tick_den))))
        name = wrfout_filename(first, node.cfg.grid_id)
        return selection, {STATIC_FRAME_ATTR: f"../{name}"}

    def durability_barrier(self):
        """A callable that waits, on any thread, for every frame queued now.

        Taken on the stepping thread at a checkpoint and run on the
        checkpoint writer's, so the checkpoint is still published only
        after every history frame up to its valid time is durable -- the
        order :meth:`drain` gave it -- without the model waiting for the
        write.  Frames queued after the barrier are not waited for.
        ``None`` where the order cannot be expressed that way (simulated
        radar volumes are drained only through :meth:`drain`, and a writer
        shell without the counters); the caller then drains as before.
        """
        if getattr(self, "_simulated_radar", None) is not None:
            return None
        targets = []
        for gid in sorted(self._writers):
            writer = self._writers[gid]
            if not hasattr(writer, "wait_landed"):
                return None
            targets.append((writer, writer.durability_target()))

        def wait() -> None:
            for writer, target in targets:
                writer.wait_landed(target)
                if writer.paths:
                    self.last_durable_wrfout = writer.paths[-1]

        return wait

    def drain(self, *, before_domain=None) -> None:
        """Wait for every domain's durable files and output identities.

        ``before_domain(grid_id, work_bytes)``, when given, is called before
        each domain is waited on with :attr:`pending_work_bytes` for the
        whole set: the domains share one NetCDF lock, so the first wait can
        cover all of them.

        Without ``before_domain`` this is a drain between two model steps
        (a restart waits here for every queued frame before it writes its
        checkpoint), and each domain's wait is a ``writing:`` record on the
        attached write progress, sized the same way.
        """
        for gid in sorted(self._writers):
            if before_domain is not None:
                before_domain(gid, self.pending_work_bytes)
                self._writers[gid].drain()
            else:
                with self._write_beat(f"drain-history-writers-d{int(gid):02d}",
                                      self.pending_work_bytes):
                    self._writers[gid].drain()
            if self._writers[gid].paths:
                self.last_durable_wrfout = self._writers[gid].paths[-1]
        if getattr(self, "_simulated_radar", None) is not None:
            self._simulated_radar.drain()

    def rewind_to_checkpoint(self, valid_time, *, marker_directory=None,
                             before_delete=None):
        """Discard only this writer's unchanged frames after a retry seam.

        Drains every asynchronous publication before inspecting output paths.
        Every victim must still match its writer-completion proof. The whole
        deletion plan is checked before removing a frame or its ready marker.
        Earlier frames and unrelated files are preserved. Return path/size
        receipts so failed-leg simulation data need not be retained.
        """
        from woof.output_identity import file_record
        from woof.restart_render import _frame_valid

        if getattr(self, "_simulated_radar", None) is not None:
            raise ValueError("automatic history rewind cannot roll back "
                             "simulated-radar volumes")
        self.drain()
        root = self.output_dir.resolve()
        proofs = {Path(proof.address): proof for proof in self.completed_records}
        victims = set()
        removals = []
        for path in sorted(self._published_paths):
            instant = _frame_valid(path)
            if instant is None:
                raise ValueError(f"history rewind cannot date owned frame {path}")
            if instant <= valid_time:
                continue
            resolved = path.resolve(strict=True)
            if path.is_symlink() or not resolved.is_relative_to(root):
                raise ValueError(f"history rewind refuses a linked or outside "
                                 f"frame {path}")
            proof = proofs.get(path.absolute())
            if proof is None:
                raise ValueError(f"history rewind lacks writer-completion "
                                 f"proof for {path}")
            record = file_record(path, completed=proof)
            removals.append((path, record))
            victims.add(path)
            if marker_directory is not None:
                marker_root = Path(marker_directory)
                marker = marker_root / f"{path.name}.json"
                if marker.exists() or marker.is_symlink():
                    if (marker.is_symlink() or not marker.resolve().is_relative_to(
                            marker_root.resolve())):
                        raise ValueError(f"history rewind refuses linked marker {marker}")
                    data = json.loads(marker.read_text(encoding="utf-8"))
                    if (data.get("schema") != "gpuwm.frame-ready/v1"
                            or data.get("path") != str(resolved)):
                        raise ValueError(f"history rewind marker names another "
                                         f"publication: {marker}")
                    removals.append((marker, {"path": str(marker.resolve()),
                                              "bytes": marker.stat().st_size}))
        if before_delete is not None:
            before_delete([record for _path, record in removals])
        receipts = []
        for path, record in removals:
            path.unlink()
            receipts.append(record)
        absolute_victims = {str(path.absolute()) for path in victims}
        for writer in self._writers.values():
            with writer._condition:
                writer.paths[:] = [path for path in writer.paths
                                   if path not in victims]
                writer._completed_records[:] = [proof for proof in
                    writer._completed_records
                    if proof.address not in absolute_victims]
        self._archived_paths[:] = [path for path in self._archived_paths
                                   if path not in victims]
        self._archived_records[:] = [proof for proof in self._archived_records
                                    if proof.address not in absolute_victims]
        self._captured_paths[:] = [path for path in self._captured_paths
                                   if path not in victims]
        self._published_paths.difference_update(victims)
        retained = self.paths
        self.last_durable_wrfout = (None if not retained else
                                   max(retained, key=_frame_valid))
        return receipts

    def close(self) -> None:
        saved: BaseException | None = None
        for gid in sorted(self._writers):
            writer = self._writers[gid]
            try:
                writer.close()
            except BaseException as exc:
                if saved is None:
                    saved = exc
            if writer.paths:
                self.last_durable_wrfout = writer.paths[-1]
        if getattr(self, "_simulated_radar", None) is not None:
            try:
                self._simulated_radar.close()
            except BaseException as exc:
                if saved is None:
                    saved = exc
        if saved is not None:
            raise saved

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            # A stopping forecast does not wait for queued radar scans: they
            # are dropped, the running rw_simradar is terminated, and close()
            # names every history left without a volume.
            if getattr(self, "_simulated_radar", None) is not None:
                with suppress(BaseException):
                    self._simulated_radar.cancel()
            # D2H already in flight must complete before process teardown;
            # publication failures remain quarantined by WrfoutWriter.
            self._abort_event.set()
            with suppress(BaseException):
                self.close()
        return False
