"""Resolve a WRF real.exe directory using the shared namelist translator."""

from __future__ import annotations

from woof.config import ConfigurationRefused

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from woof.ingest.wrfinput import (
    WrfinputMetadata, check_boundary_coverage,
    check_grid_agreement, read_wrfinput_metadata,
    wrfbdy_coverage,
)

#: The projection ids WPS and WRF share, spelled the way
#: ``namelist.wps`` spells them.  ``MAP_PROJ`` is an integer in the
#: wrfinput and a word in the WPS namelist, and this is the only place
#: the two spellings meet.
_MAP_PROJ_NAMES = {
    1: "lambert", 2: "polar", 3: "mercator", 6: "lat-lon",
}

#: Global attributes the synthesized WPS namelist is built from.  Named
#: so a wrfinput missing one is refused by ATTRIBUTE rather than by a
#: KeyError six frames down.
_REQUIRED_LAYOUT_ATTRIBUTES = (
    "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON",
    "DX", "DY", "GRID_ID", "PARENT_ID", "I_PARENT_START", "J_PARENT_START",
    "PARENT_GRID_RATIO",
)


@dataclass(frozen=True)
class WrfinputRun:
    """One ``real.exe`` run directory, resolved but not yet restored."""

    directory: Path
    namelist_input: Path
    wrfinput_paths: Mapping[int, Path]
    wrfbdy_path: Path
    metadata: Mapping[int, WrfinputMetadata]
    coverage: object
    experiment: object
    toml_text: str
    substitution_report: object
    wps_namelist_text: str
    #: A WRF warm start: each domain's ``wrfrst`` file, the root restart's
    #: clock (:class:`woof.ingest.wrfrst.WrfRestartClock`) and the declared
    #: changes the run admits.  ``None`` on a real.exe cold start, whose
    #: ``wrfinput_paths`` are then the initial condition.
    restart_paths: Mapping[int, Path] | None = None
    restart_clock: object | None = None
    restart_notes: tuple[str, ...] = ()

    @property
    def grid_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.restart_paths or self.wrfinput_paths))


def discover_run_directory(directory: str | Path, *, restart: bool = False
                           ) -> tuple[Path, Path, dict[int, Path], Path]:
    """Locate ``namelist.input``, every ``wrfinput_d0N`` and ``wrfbdy_d01``.

    Named refusals, one per missing piece, because a reader who pointed
    the door at the wrong directory is owed the list of what a right one
    contains.  ``restart`` is a warm start: its state comes from the
    ``wrfrst`` files, so a ``wrfinput_d01`` is not required.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigurationRefused(
            f"--wrfinput {directory} is not a directory; point it at the "
            "WRF run directory holding wrfinput_d0N, wrfbdy_d01 and "
            "namelist.input")
    namelist = directory / "namelist.input"
    inputs = {}
    for path in sorted(directory.glob("wrfinput_d??")):
        suffix = path.name.removeprefix("wrfinput_d")
        if suffix.isdigit():
            inputs[int(suffix)] = path
    bdy = directory / "wrfbdy_d01"
    absent = []
    if not namelist.is_file():
        absent.append(
            "namelist.input (the run settings and the physics; WOOF reads "
            "the same file wrf.exe would)")
    if not inputs and not restart:
        absent.append(
            "wrfinput_d01 (real.exe's initial condition; wrfinput_d02.. "
            "join automatically when present)")
    if not bdy.is_file():
        absent.append(
            "wrfbdy_d01 (real.exe's lateral boundary values and tendencies; "
            "it is what caps the run length)")
    if absent:
        raise ConfigurationRefused(
            f"{directory} is not a complete real.exe handoff.  Missing:\n"
            + "\n".join(f"  - {item}" for item in absent))
    if 1 not in inputs and not restart:
        raise ValueError(
            f"{directory} has {sorted(inputs)} but no wrfinput_d01; the "
            "root domain is the one wrfbdy_d01 forces and cannot be "
            "omitted")
    return directory, namelist, inputs, bdy


def namelist_restart_request(namelist: Path):
    """``(restart, start, end)`` as wrf.exe reads them from ``namelist``.

    ``restart`` is ``&time_control restart``; ``start`` the root domain's
    start instant; ``end`` its end -- ``start + run_*`` when the run length
    is positive (WRF never reads end_* then), else ``end_*``.
    """
    from woof.fortran_namelist import parse_namelist

    control = parse_namelist(namelist).get("time_control", {})

    def first(key, default):
        values = control.get(key)
        return values[0] if values else default

    restart = bool(first("restart", False))
    start = datetime(int(first("start_year", 1993)), int(first("start_month", 3)),
                     int(first("start_day", 13)), int(first("start_hour", 0)),
                     int(first("start_minute", 0)), int(first("start_second", 0)))
    run = sum(float(first(key, 0)) * scale for key, scale in (
        ("run_days", 86400), ("run_hours", 3600), ("run_minutes", 60),
        ("run_seconds", 1)))
    if run > 0:
        end = start + timedelta(seconds=run)
    else:
        end = datetime(int(first("end_year", 1993)), int(first("end_month", 3)),
                       int(first("end_day", 14)), int(first("end_hour", 0)),
                       int(first("end_minute", 0)), int(first("end_second", 0)))
    return restart, start, end


#: &time_control keys a warm start rewrites or consumes itself: the window
#: is the producing simulation's start to the namelist's end, and the
#: restart switches are what this door honours by restarting.
_RESTART_WINDOW_KEYS = (
    "run_days", "run_hours", "run_minutes", "run_seconds",
    "start_year", "start_month", "start_day", "start_hour", "start_minute",
    "start_second", "end_year", "end_month", "end_day", "end_hour",
    "end_minute", "end_second", "restart", "override_restart_timers",
    "write_hist_at_0h_rst",
)


def restart_window_namelist(text: str, *, start: datetime, end: datetime,
                            max_dom: int) -> str:
    """``namelist.input`` text with WRF's restart window made explicit.

    wrf.exe restarts on the producing simulation's timeline: its boundary
    clock starts at the simulation start the restart file records, and the
    run ends at the namelist's end.  The translated experiment therefore
    starts at that simulation start and runs to that end; the model clock
    opens at the restart instant (woof.ingest.wrfrst.apply_warm_start_clock).
    """
    import re

    pattern = re.compile(
        r"(?im)^[ \t]*(" + "|".join(_RESTART_WINDOW_KEYS) + r")[ \t]*=[^\n]*\n")
    text = pattern.sub("", text)

    def row(key, value):
        return f" {key} = " + ", ".join([str(value)] * max_dom) + ","

    lines = [row(f"start_{part}", getattr(start, part)) for part in
             ("year", "month", "day", "hour", "minute", "second")]
    lines += [row(f"end_{part}", getattr(end, part)) for part in
              ("year", "month", "day", "hour", "minute", "second")]
    text, count = re.subn(r"(?im)^([ \t]*&time_control)\b",
                          lambda m: m[0] + "\n" + "\n".join(lines), text,
                          count=1)
    if count != 1:
        raise ValueError("namelist.input has no &time_control group")
    return text


def refuse_latlon_files(metadata: Mapping[int, object]) -> None:
    """Refuse a WRF-input or met_em handoff on a lat-lon grid, by name.

    Asked first, from the files' own ``MAP_PROJ``, before anything is
    read as a grid: the reconstituted namelist would carry the files'
    metre DX and no pole, and every check after it would be reading a
    lat-lon grid (WRF 6) as a conformal one.
    """
    from woof.static.projection import (
        implemented_projections, is_latlon_map_proj, latlon_refusal)

    if _MAP_PROJ_NAMES[6] in implemented_projections():
        return
    for grid_id in sorted(metadata):
        item = metadata[grid_id]
        value = item.global_attributes.get("MAP_PROJ")
        if value is not None and is_latlon_map_proj(value):
            raise ConfigurationRefused(
                f"{item.path} declares MAP_PROJ={int(float(value))}: "
                f"{latlon_refusal()}")


def synthesize_wps_namelist(metadata: Mapping[int, WrfinputMetadata]) -> str:
    """Write the ``&share``/``&geogrid`` facts out of the wrfinput files.

    This is the whole difference between this door and
    ``woof import-namelist``.  A ``real.exe`` handoff has no
    ``namelist.wps`` -- WPS ran days ago on someone else's machine -- but
    every fact that namelist carried was stamped into the wrfinput's
    global attributes by ``real.exe`` itself.  Reconstituting it here
    means ONE translator serves both doors, and it means the projection
    the run uses is the projection the FILES were written on, not a
    second declaration that could disagree with them.
    """
    ordered = [metadata[grid_id] for grid_id in sorted(metadata)]
    root = ordered[0]
    for item in ordered:
        missing = sorted(
            name for name in _REQUIRED_LAYOUT_ATTRIBUTES
            if name not in item.global_attributes)
        if missing:
            raise ConfigurationRefused(
                f"{item.path} is missing the global attribute(s) {missing} "
                "that carry the domain layout; real.exe stamps every one "
                "of them, so this file was not written by real.exe or was "
                "rewritten by a tool that dropped them")
    refuse_latlon_files(metadata)
    map_proj = int(float(root.global_attributes["MAP_PROJ"]))
    if map_proj not in _MAP_PROJ_NAMES:
        raise ConfigurationRefused(
            f"{root.path} declares MAP_PROJ={map_proj}, which is not one "
            f"of WPS's {sorted(_MAP_PROJ_NAMES)} "
            f"({', '.join(_MAP_PROJ_NAMES[k] for k in sorted(_MAP_PROJ_NAMES))})")

    def column(attribute):
        # The Rust NetCDF bridge promotes every numeric attribute to f64,
        # so PARENT_ID arrives as 0.0; the namelist needs the integer.
        return ", ".join(
            str(int(float(item.global_attributes[attribute])))
            for item in ordered)

    def parent_start_column(attribute):
        """The head grid's parent start, in WPS's spelling (1).

        real.exe stamps whatever ``&domains i/j_parent_start`` said, and
        for the head grid that coordinate places nothing (it has no
        parent), so templates carry 1 or 0.  WPS always writes 1 for d01
        and woof's WPS reader enforces that.  Named breakage: NCAR's v4.4
        CONUS benchmark wrfinput carries I/J_PARENT_START = 0 and this door
        refused it as a malformed root topology, so the benchmark folder
        could only be run after hand-editing its wrfinput.  Children carry
        their real offsets through unchanged.
        """
        values = []
        for index, item in enumerate(ordered):
            value = int(float(item.global_attributes[attribute]))
            values.append(1 if index == 0 and value == 0 else value)
        return ", ".join(str(value) for value in values)

    def parent_id_column():
        """WPS spells the root's parent as ITSELF; WRF spells it 0.

        ``&geogrid parent_id`` is 1 for domain 1 (WPS's own convention,
        which woof's WPS reader enforces), while ``real.exe`` stamps
        ``PARENT_ID = 0`` into the root wrfinput.  Same fact, two
        spellings; this is the one place they meet.  Children carry their
        real parent id through unchanged.
        """
        values = []
        for item in ordered:
            value = int(float(item.global_attributes["PARENT_ID"]))
            values.append(1 if value == 0 else value)
        return ", ".join(str(value) for value in values)

    start = root.start_date or ""
    lines = [
        "&share",
        " wrf_core = 'ARW',",
        f" max_dom = {len(ordered)},",
        " start_date = " + ", ".join(f"'{start}'" for _ in ordered) + ",",
        "/",
        "",
        "&geogrid",
        f" parent_id         = {parent_id_column()},",
        f" parent_grid_ratio = {column('PARENT_GRID_RATIO')},",
        f" i_parent_start    = {parent_start_column('I_PARENT_START')},",
        f" j_parent_start    = {parent_start_column('J_PARENT_START')},",
        " e_we              = " + ", ".join(
            str(item.nx + 1) for item in ordered) + ",",
        " e_sn              = " + ", ".join(
            str(item.ny + 1) for item in ordered) + ",",
        f" dx = {float(root.global_attributes['DX']):.10g},",
        f" dy = {float(root.global_attributes['DY']):.10g},",
        f" map_proj = '{_MAP_PROJ_NAMES[map_proj]}',",
        f" ref_lat   = {float(root.global_attributes['CEN_LAT']):.10g},",
        f" ref_lon   = {float(root.global_attributes['CEN_LON']):.10g},",
        f" truelat1  = {float(root.global_attributes['TRUELAT1']):.10g},",
        f" truelat2  = {float(root.global_attributes['TRUELAT2']):.10g},",
        f" stand_lon = {float(root.global_attributes['STAND_LON']):.10g},",
        "/",
        "",
    ]
    return "\n".join(lines)


def check_projection_agreement(experiment, metadata: Mapping[
        int, WrfinputMetadata], *, tolerance_deg: float = 5.0e-3) -> dict:
    """Prove the reconstituted projection lands on the file's own grid.

    The layout above is reconstituted from ``CEN_LAT``/``CEN_LON``, which
    is WPS's ``ref_lat``/``ref_lon`` only when the reference point sat at
    the domain centre -- the default, but not a law.  Rather than trust
    it, this evaluates WOOF's projection on every mass point and
    compares it against the ``XLAT``/``XLONG`` arrays ``real.exe`` wrote
    into the same file.  An absent check here would be scored as
    agreement; this is the measurement.

    Returns the per-domain maximum absolute latitude/longitude
    disagreement in degrees.
    """
    from woof import netcdf_bridge
    from woof.static.projection import grids_from_projection_config

    grids = grids_from_projection_config(experiment)
    worst = {}
    problems = []
    for domain, grid in zip(experiment.domains, grids):
        item = metadata.get(domain.grid_id)
        if item is None:
            continue
        with netcdf_bridge.open_dataset(item.path) as dataset:
            missing = [name for name in ("XLAT", "XLONG") if name not in dataset.variables]
            if missing:
                raise ConfigurationRefused(f"{item.path}: missing coordinate evidence {missing}")
            xlat = np.asarray(dataset.variables["XLAT"][...],
                              dtype=np.float64)
            xlon = np.asarray(dataset.variables["XLONG"][...],
                              dtype=np.float64)
        xlat = xlat.reshape(xlat.shape[-2:])
        xlon = xlon.reshape(xlon.shape[-2:])
        lat, lon = grid.latlon_mass()
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.shape != xlat.shape:
            problems.append(
                f"  d{domain.grid_id:02d}: WOOF's grid is {lat.shape}, "
                f"{item.path.name} XLAT is {xlat.shape}")
            continue
        dlat = float(np.abs(lat - xlat).max())
        dlon = float(np.abs(
            ((lon - xlon + 180.0) % 360.0) - 180.0).max())
        worst[domain.grid_id] = (dlat, dlon)
        if not np.isfinite([dlat, dlon]).all() or max(dlat, dlon) > tolerance_deg:
            problems.append(
                f"  d{domain.grid_id:02d}: max |dlat| = {dlat:.6g} deg, "
                f"max |dlon| = {dlon:.6g} deg")
    if problems:
        raise ConfigurationRefused(
            "the projection reconstituted from the wrfinput global "
            "attributes does not reproduce the grid real.exe wrote into "
            "the same file:\n" + "\n".join(problems)
            + f"\n(tolerance {tolerance_deg} deg).  The layout is taken "
            "from CEN_LAT/CEN_LON, which is WPS's ref_lat/ref_lon only "
            "when the reference point sat at the domain centre; this "
            "handoff placed it elsewhere.  Use `woof import-namelist` "
            "with the original namelist.wps instead.")
    return worst


#: The declared divergences a reachable WRF door admits, as
#: ``(key, wrf_value, gpuwm_value)`` rows.  Each names a route ArWen HAS
#: and integrates, differing from the literal namelist request in a way
#: the importer states and ``announce_wrf_substitutions`` prints at the
#: terminal.  A row here is the ONLY way a substitution survives this
#: gate, and adding one is table work rather than another branch.
#:
#: * ``use_theta_m`` 1 -> 0: the moist theta_m prognostic is implemented
#:   nowhere in the engine, and the initial and boundary state is
#:   recovered exactly on every import route.
#: * ``fine_input_stream`` 2 -> 0: WRF's delayed-nest-start stream.  The
#:   child still starts at its declared start time; only the provenance
#:   of its masked surface state differs
#:   (:func:`woof.namelist_import.fine_input_stream_decision`, the one
#:   function this door, the importer and the RW-WPS support report all
#:   read the answer from).
#: * ``fractional_seaice`` 1 -> 0 under a non-RUC land surface: the
#:   fractional ice/water flux blend is ported in the RUC seam only.  Until
#:   2026-10-12 this door accepted it with no word (the importer booked it
#:   as "fixed"); it is now announced, with the only cells it changes
#:   (sea-ice fraction strictly between 0.02 and 1).  Admitted rather than
#:   refused because most WRF namelists set it and most domains have no
#:   such cell; the Noah fractional-ice port is the fix that retires it.
ADMITTED_DECLARED_DIVERGENCES = frozenset({
    ("use_theta_m", 1, 0),
    ("fine_input_stream", 2, 0),
    ("fractional_seaice", 1, 0),
})


def producing_wrf_line(metadata: Mapping[int, WrfinputMetadata]) -> str | None:
    """The WRF line ("3" or "4") the files' own ``TITLE`` names, or None.

    real.exe stamps its release into every wrfinput (share/output_wrf.F:
    ``TITLE`` = "OUTPUT FROM REAL_EM V<release> PREPROCESSOR"), and the
    namelist that produced the files resolved its omitted keys against
    THAT release's Registry.  An omitted ``&dynamics/use_theta_m`` is the
    one key whose default moved between the lines (V3.9
    Registry.EM_COMMON:2633 says 0; V4 says 1), so the door reads the line
    here rather than assuming V4: an operational HRRR v4 run directory
    (WRFV3.9) whose namelist omits the key resolves to the dry theta its
    files declare, with no flag.  Files that disagree, or that carry no
    readable release, return None, and the importer keeps its V4 reading.
    """
    line, _title = producing_wrf_line_and_title(metadata)
    return line


def producing_wrf_line_and_title(
        metadata: Mapping[int, WrfinputMetadata]) -> tuple[str | None, str]:
    """:func:`producing_wrf_line` with the ``TITLE`` that named it.

    The import report records what chose the WRF line; on this door that
    is the files' own ``TITLE``, and the receipt quotes it.  The title is
    the lowest domain's (the files agree on the line or there is none).
    """
    import re

    lines = set()
    for item in metadata.values():
        title = str(item.global_attributes.get("TITLE", ""))
        found = re.search(r"\bV(\d+)\.\d", title)
        lines.add(found.group(1) if found else None)
    if len(lines) != 1:
        return None, ""
    (line,) = lines
    if line not in ("3", "4"):
        return None, ""
    return line, str(
        metadata[min(metadata)].global_attributes.get("TITLE", "")).strip()


def producing_wrf_release(metadata: Mapping[int, WrfinputMetadata]) -> str | None:
    """The V4 release ("4.6.1") every file's ``TITLE`` names, or None.

    real.exe writes " OUTPUT FROM REAL_EM V4.6.1 PREPROCESSOR" and the
    hierarchy export " ... V4.6.1-COMPAT DIRECT EXPORTER".  The door hands
    the release to the importer, which pins the scheme generations that
    release runs (namelist_import.wrf_release_generations), so a WRF 4.6.1
    handoff runs WRF 4.6.1's YSU and sun rather than the engine's newest.
    Files that disagree, a V3 title, or no readable release give None.
    """
    import re

    releases = set()
    for item in metadata.values():
        title = str(item.global_attributes.get("TITLE", ""))
        found = re.search(r"\bV(4)\.(\d+)(?:\.(\d+))?", title)
        releases.add(".".join(part for part in found.groups() if part is not None)
                     if found else None)
    if len(releases) != 1:
        return None
    return releases.pop()


def require_preserved_wrf_selectors(report):
    """A reachable WRF door must not replace an explicit physics package.

    Only a row of :data:`ADMITTED_DECLARED_DIVERGENCES` is an admitted
    declared divergence, and it must carry the reason announced at the
    terminal by ``announce_wrf_substitutions``. A reason explains a
    change; it does not authorize replacing any other requested physics
    selector.
    """
    for item in report.substitutions:
        if (item.key == item.gpuwm_key
                and (item.key, item.wrf_value, item.gpuwm_value)
                in ADMITTED_DECLARED_DIVERGENCES
                and item.reason):
            continue
        if item.wrf_value != item.gpuwm_value:
            raise ConfigurationRefused(
                f"the requested {item.wrf_name} ({item.key}={item.wrf_value}) has no native implementation; "
                f"translation would replace it with {item.gpuwm_name} ({item.gpuwm_key}={item.gpuwm_value}). "
                "Select a supported physics package explicitly in the producing namelist")


def resolve_wrfinput_run(directory: str | Path, *, name: str | None = None,
                         acknowledgements: Sequence[str] = (),
                         rrtmg_variant: str | None = None,
                         thompson_version: str | None = None,
                         urban_ucm_generation: str | None = None,
                         urban_controls: dict | None = None,
                         check_projection: bool = True,
                         restart_from: str | Path | None = None) -> WrfinputRun:
    """Read, translate and CHECK one real.exe handoff.  No field is decoded.

    Every refusal this door owns fires here, on CPU, before any state
    exists: unsupported physics packages by name, a grid the namelist and
    the files disagree about, a projection that does not reproduce the
    file's own XLAT/XLONG, and a run longer than wrfbdy's coverage.

    A WARM START continues a WRF run from its restart file: when
    ``restart_from`` names one (any file name -- NCAR ships its benchmark
    restarts suffixed ``.gnu``/``.ifort``), or when ``namelist.input``
    says ``restart = .true.`` and the directory holds wrf.exe's own
    ``wrfrst_d0N_<start>``.  The domain layout, physics identity and
    vertical grid are then read from the restart files; the run is the
    producing simulation's, from the start the file records to the
    namelist's end, and the model clock opens at the restart instant.
    """
    import tomllib

    from woof.experiment import build_experiment
    from woof.ingest.wrfinput_identity import read_wrfinput_identity, check_wrfinput_identity
    from woof.config import soil_layer_count

    probe_directory = Path(directory)
    namelist_restart = False
    if (probe_directory / "namelist.input").is_file():
        namelist_restart, namelist_start, namelist_end = namelist_restart_request(
            probe_directory / "namelist.input")
    warm = restart_from is not None or namelist_restart
    directory, namelist, inputs, bdy = discover_run_directory(
        directory, restart=warm)
    restart_paths = restart_clock = None
    restart_notes: tuple[str, ...] = ()
    if warm:
        from woof.ingest.wrfrst import (discover_restart_files,
                                         read_wrfrst_clock)

        restart_paths = discover_restart_files(
            directory, explicit=restart_from,
            restart_time=None if restart_from is not None else namelist_start)
        clocks = {gid: read_wrfrst_clock(path)
                  for gid, path in restart_paths.items()}
        restart_clock = clocks[min(clocks)]
        if min(clocks) != 1:
            raise ValueError(
                f"{restart_paths[min(clocks)]} restarts d{min(clocks):02d}; a "
                "warm start needs the root domain's wrfrst_d01, whose "
                "boundary clock the run continues")
        instants = {clock.valid_time for clock in clocks.values()}
        if len(instants) != 1:
            raise ValueError(
                "the restart files are at different instants: "
                + ", ".join(f"{path.name}" for path in restart_paths.values()))
        if namelist_end <= restart_clock.valid_time:
            raise ValueError(
                f"{namelist} ends at {namelist_end:%Y-%m-%d_%H:%M:%S}, at or "
                "before the restart instant "
                f"{restart_clock.valid_time:%Y-%m-%d_%H:%M:%S}; there is "
                "nothing to integrate.  Set the namelist's end (or run_*) "
                "past the restart instant")
        state_files = restart_paths
    else:
        state_files = inputs
    # Scheme support is established HERE, from each file's own
    # MP_PHYSICS/SF_SURFACE_PHYSICS, before the namelist is even parsed:
    # an unported package is the user's shortest path to an answer and
    # must not be buried under a namelist translation refusal.
    metadata = {grid_id: read_wrfinput_metadata(path)
                for grid_id, path in sorted(state_files.items())}
    for grid_id, item in metadata.items():
        if item.grid_id != grid_id:
            raise ConfigurationRefused(
                f"{item.path} declares GRID_ID={item.grid_id} but is named "
                f"for domain {grid_id}; the file and its name disagree "
                "about which domain it initializes")
    refuse_latlon_files(metadata)
    identities = {grid_id: read_wrfinput_identity(path, restart=warm)
                  for grid_id, path in state_files.items()}
    landuse_identity = None
    for item in metadata.values():
        identity = {key: item.global_attributes.get(key) for key in ('MMINLU', 'NUM_LAND_CAT')}
        if landuse_identity is not None and identity != landuse_identity:
            raise ConfigurationRefused('WRF input files disagree on the producing run land-use identity')
        landuse_identity = identity
    wps_text = synthesize_wps_namelist(metadata)
    line, title = producing_wrf_line_and_title(metadata)
    if line == "4":
        line = producing_wrf_release(metadata) or line
    window = None
    if warm:
        window = (restart_clock.simulation_start, namelist_end)
    toml_text, report = _import_with_synthesized_wps(
        wps_text, namelist, inherited_eta=identities[min(identities)].eta_levels,
        restart_window=window, name=name, rrtmg_variant=rrtmg_variant,
        urban_ucm_generation=urban_ucm_generation, urban_controls=urban_controls,
        acknowledgements=tuple(acknowledgements), landuse_identity=landuse_identity,
        wrfinput_qv_domains=tuple(
            grid_id for grid_id, item in metadata.items()
            if getattr(item, "has_qv", False)),
        wrfinput_radar_tendency_domains=tuple(
            grid_id for grid_id, item in metadata.items()
            if getattr(item, "has_radar_tendency", False)),
        wrf_boundary_use_theta_m=int(metadata[min(metadata)].global_attributes['USE_THETA_M']),
        **({} if line is None else {
            "wrf_version": line,
            "wrf_version_source": f"the files' own TITLE ({title!r})"}))
    require_preserved_wrf_selectors(report)
    # ``import_namelists`` has already validated this text through
    # ``build_experiment``; building it again is how the door gets the
    # ExperimentConfig object rather than a second validation.
    from woof.thompson_generations import matching_runtime_titles, select_classic_generation
    runtime_titles = () if warm else matching_runtime_titles(directory, metadata)
    generation, generation_receipt = select_classic_generation(
        metadata, runtime_titles=runtime_titles, override=thompson_version)
    if generation is not None:
        import json
        # Insert into the existing shared table, so the serialized config,
        # the domain RunConfigs and restart identity all carry one choice.
        selection = (f'thompson_version = {json.dumps(generation)}\n'
                     + (f'thompson_table_dir = {json.dumps(str(directory.resolve()))}\n'
                        if generation == "wrf_44" else ""))
        toml_text = toml_text.replace("[shared]\n", "[shared]\n" + selection, 1)
        toml_text = "# Thompson generation: " + json.dumps(generation_receipt) + "\n" + toml_text
    experiment = build_experiment(
        tomllib.loads(toml_text), source=f"{directory}:wrfinput")
    kind = "wrfrst" if warm else "wrfinput"
    if len(experiment.domains) != len(metadata):
        raise ConfigurationRefused(
            f"{namelist} declares {len(experiment.domains)} domain(s) but "
            f"{directory} carries {kind} files for {sorted(metadata)}; "
            f"WRF writes one {kind} per domain, so one of the two "
            "is from a different experiment")
    for domain in experiment.domains:
        item = metadata.get(domain.grid_id)
        if item is None:
            raise ConfigurationRefused(
                f"{namelist} declares domain {domain.grid_id} but "
                f"{directory} has no {kind}_d{domain.grid_id:02d}")
        check_grid_agreement(item, domain.run, source=str(namelist))
        check_wrfinput_identity(identities[domain.grid_id], domain=domain,
                               vertical=experiment.vertical,
                               start_time=(restart_clock.valid_time if warm else
                                           domain.start_time or experiment.start_time),
                               soil_layers=soil_layer_count(domain.run))
        if warm:
            from woof.ingest.wrfrst import (check_restart_physics,
                                             file_physics_selectors)
            restart_notes += tuple(check_restart_physics(
                file_physics_selectors(item.global_attributes), domain.run,
                path=item.path))
    if _namelist_uses_base_params(namelist):
        check_namelist_base_state(namelist, metadata)
    if check_projection:
        check_projection_agreement(experiment, metadata)
    coverage = wrfbdy_coverage(bdy)
    check_boundary_coverage(coverage, experiment.run_seconds,
                            source=str(namelist))
    _check_start_time(experiment, metadata, coverage, source=str(namelist))
    if warm and restart_clock.elapsed_seconds >= float(experiment.run_seconds):
        raise ValueError(
            f"the restart instant lies {restart_clock.elapsed_seconds:g} s into "
            f"the simulation, at or past this run's {experiment.run_seconds:g} s")
    return WrfinputRun(
        directory=directory, namelist_input=namelist,
        wrfinput_paths=MappingProxyType(dict(sorted(inputs.items()))),
        wrfbdy_path=bdy, metadata=MappingProxyType(metadata),
        coverage=coverage, experiment=experiment, toml_text=toml_text,
        substitution_report=report, wps_namelist_text=wps_text,
        restart_paths=(None if restart_paths is None
                       else MappingProxyType(dict(sorted(restart_paths.items())))),
        restart_clock=restart_clock, restart_notes=restart_notes)


#: WRF's &dynamics base-state constants and their Registry defaults
#: (Registry.EM_COMMON:2905-2910), keyed by the wrfinput/wrfrst variable
#: that carries the same constant.
_NAMELIST_BASE_STATE = (
    ("P00", "base_pres", 100000.0), ("T00", "base_temp", 290.0),
    ("TLP", "base_lapse", 50.0), ("TISO", "iso_temp", 200.0),
    ("TLP_STRAT", "base_lapse_strat", -11.0),
    ("P_STRAT", "base_pres_strat", 0.0),
)


def _namelist_uses_base_params(namelist: Path) -> bool:
    from woof.fortran_namelist import parse_namelist

    values = parse_namelist(namelist).get("dynamics", {}).get(
        "use_baseparam_fr_nml", [False])
    return bool(values and values[0])


def check_namelist_base_state(namelist: Path, metadata) -> None:
    """Refuse files whose base-state constants differ from the namelist's.

    wrf.exe takes them from &dynamics base_* when ``use_baseparam_fr_nml``
    is set (start_em.F:516-523) and otherwise from the file; woof's own
    initialization reads them from the namelist.  The two readings agree
    exactly when the file carries the namelist's constants, which is what
    this compares, variable by variable.  On a warm start the file's base
    state is integrated as written, so the comparison holds it to the same
    constants the run is configured with.
    """
    from woof import netcdf_bridge
    from woof.fortran_namelist import parse_namelist

    dynamics = parse_namelist(namelist).get("dynamics", {})
    problems = []
    for item in metadata.values():
        with netcdf_bridge.open_dataset(item.path) as dataset:
            for variable, key, default in _NAMELIST_BASE_STATE:
                if variable not in dataset.variables:
                    continue
                found = float(np.asarray(dataset.variables[variable][...]).reshape(-1)[0])
                wanted = float((dynamics.get(key) or [default])[0])
                if found != wanted:
                    problems.append(f"  {item.path.name} {variable} = {found:g}, "
                                    f"&dynamics {key} = {wanted:g}")
    if problems:
        raise ValueError(
            "the WRF files' base-state constants differ from the namelist's:\n"
            + "\n".join(problems)
            + "\nwrf.exe and woof would rebuild different base states; use "
            "the namelist that produced the files")



def _import_with_synthesized_wps(wps_text: str, namelist: Path, *, inherited_eta,
                                 restart_window=None, **kwargs):
    """Run the shared translator against the reconstituted WPS namelist.

    ``import_namelists`` takes PATHS and it is the one translator both
    doors use, so the reconstituted text becomes a file for the length of
    the call.  It goes in a TEMPORARY directory, never beside the run: a
    WRF run directory is the user's evidence of what they ran, and a door
    whose job is to READ one must not leave a file in it.  The refusals
    that quote the layout quote ``run.wps_namelist_text``, which is
    carried on the result, so nothing is lost by not writing it there.

    ``restart_window`` (a warm start's ``(simulation start, end)``) writes
    WRF's restart timeline into the effective namelist
    (:func:`restart_window_namelist`).
    """
    import re
    import tempfile

    from woof.namelist_import import import_namelists, parse_namelist

    with tempfile.TemporaryDirectory(prefix="gpuwm-wrfinput-") as scratch:
        wps_path = Path(scratch) / "namelist.wps"
        wps_path.write_text(wps_text, encoding="utf-8")
        effective_input = namelist
        text = namelist.read_text(encoding="utf-8")
        edited = False
        parsed = parse_namelist(namelist)
        if not parsed.get("domains", {}).get("eta_levels"):
            # real.exe has already generated this coordinate. Bind exactly
            # those values when the producing namelist used its generator.
            # Explicit user eta values are never replaced and are compared.
            eta = ", ".join(repr(float(value)) for value in inherited_eta)
            text, count = re.subn(r"(?im)^([ \t]*&domains)\b", lambda m: m[0] + "\n eta_levels = " + eta + ",", text, count=1)
            if count != 1:
                raise ValueError(f"{namelist}: cannot locate &domains to bind file eta levels")
            edited = True
        if restart_window is not None:
            max_dom = int((parsed.get("domains", {}).get("max_dom") or [1])[0])
            text = restart_window_namelist(
                text, start=restart_window[0], end=restart_window[1],
                max_dom=max_dom)
            edited = True
        if edited:
            effective_input = Path(scratch) / "namelist.input"
            effective_input.write_text(text, encoding="utf-8")
        # Both metric mixing values are implemented for real-data input.
        # The file door checks nonzero idealized base profiles separately;
        # ordinary importer normalization must not replace an authored value.
        return import_namelists(wps_path, effective_input,
                                preserve_wrf_mixing=True, **kwargs)


def _check_start_time(experiment, metadata, coverage, *, source: str) -> None:
    """Refuse a namelist whose start instant is not the files' own."""
    problems = []
    for grid_id in sorted(metadata):
        declared = metadata[grid_id].start_date
        if not declared:
            continue
        try:
            stamped = datetime.strptime(declared, "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            problems.append(
                f"  wrfinput_d{grid_id:02d} START_DATE {declared!r} is not "
                "a WRF timestamp")
            continue
        if stamped != experiment.start_time:
            problems.append(
                f"  wrfinput_d{grid_id:02d} was written for {stamped}, "
                f"{source} starts at {experiment.start_time}")
    if coverage.times and coverage.start != experiment.start_time:
        problems.append(
            f"  wrfbdy_d01's first boundary record is {coverage.start}, "
            f"{source} starts at {experiment.start_time}")
    if problems:
        raise ConfigurationRefused(
            "the initial instant does not agree across the handoff:\n"
            + "\n".join(problems)
            + "\nreal.exe stamped these files for one instant; running "
            "them from another silently shifts the whole forecast.")
