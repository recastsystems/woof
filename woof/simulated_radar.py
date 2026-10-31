"""Simulated radar: the ``woof simulated-radar`` door and live history orchestration.

The native Rust bridge reads history, samples beams, writes radar formats and
renders PPI images. This module passes paths and options only.
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tempfile
import time
from types import SimpleNamespace

# WOOF's in-loop path uses the normal GPU history selection/staging code, never
# a persisted quick-look frame. Three slots include the scan's bracketing pair.
RADAR_INLOOP_VERSION = 2
RADAR_SCRATCH_FRAMES = 3
RADAR_QUICKLOOK_FIELDS = ("REFL_10CM", "T", "PH", "PHB", "U", "V", "W")
RADAR_SNAPSHOT_FIELDS = (
    "REFL_10CM", "T", "PH", "PHB", "U", "V", "W", "P", "PB", "QVAPOR", "QRAIN",
    "QCLOUD", "QICE", "QSNOW", "QGRAUP", "QHAIL", "QNCLOUD", "QNRAIN", "QNICE",
    "QNSNOW", "QNGRAUP", "QNHAIL", "QIR", "QIB", "QZI", "QICE2", "QNICE2", "QIR2", "QIB2",
    "QVOLI", "QAOLI", "QVOLI2", "QAOLI2", "QICE3", "QNICE3", "QVOLI3", "QAOLI3",
    "D_ICE", "RHO_ICE", "PHI_ICE", "V_ICE", "D_ICE2", "RHO_ICE2", "PHI_ICE2", "V_ICE2",
    "D_ICE3", "RHO_ICE3", "PHI_ICE3", "V_ICE3", "TKE_PBL", "TKE", "QKE",
)


def inloop_enabled():
    return os.environ.get("WOOF_RADAR_INLOOP") == "1"

from woof.simulated_radar_config import (
    FIELDS, FORMATS, MANIFEST_SCHEMA, OFF, REFUSAL_LABEL, REQUEST_SCHEMA,
    SCAN_STRATEGIES, RadarSite, SimulatedRadarOptions, declared_key_rows,
    door_refusals, load_options, require_admitted, scene_shapes,
    validate_history_selection,
)


def quicklook_scan(config: dict) -> dict:
    """The scan of a quick-look-only radar snapshot: VCP 212's three lowest elevations (``low_tilts``) when the run asks
    for the default VCP 212 frozen at its history time, else the request unchanged.

    The drawn PPIs (0.5 and 0.9 deg reflectivity and velocity) are byte-identical either way (14 of 14 images on a
    lab-box A/B, frame wrfout_d02_2026-10-29_01_35_00, 3 sites), and the scan costs about a fifth: 3 cuts against 17.
    Full-history snapshots keep the whole requested volume. THE BREAKAGE THIS PREVENTS (WOOF, 2026-10-30): radar every
    5 min scanned 17 cuts per site per snapshot, so on a 3 km run the radar, not the forecast, set the run's length and
    price (run_e190e1a9ec413168: 3,130 s of forecast stage for a 131 s forecast; 783 c of an 846 c quote).
    """
    if config.get("scan_strategy", "vcp212") != "vcp212" or config.get("timing", "history") != "history":
        return {}
    return {"scan_strategy": "low_tilts", "elevations_deg": list(SCAN_STRATEGIES["low_tilts"])}


@dataclass(frozen=True)
class RadarVolumeResult:
    """One committed site-volume; paths are absolute and times are UTC."""

    paths: dict[str, Path]
    volume_time: datetime
    site: RadarSite
    scan_strategy: str
    metadata: dict


@dataclass(frozen=True)
class RadarResult:
    """Products of this call, excluding older volumes in the output directory.

    ``volumes`` has one item per input time. ``paths`` groups their absolute
    paths by format. The native manifest retains scientific and timing provenance.
    """

    paths: dict[str, tuple[Path, ...]]
    volumes: tuple[RadarVolumeResult, ...]
    site: RadarSite
    scan_strategy: str
    manifest_path: Path
    resources: dict
    input_transport: str

    @property
    def volume_time(self) -> datetime:
        """The time of a single-volume result; use ``volumes`` for a series."""
        if len(self.volumes) != 1:
            raise ValueError("this result has multiple volume times; use result.volumes")
        return self.volumes[0].volume_time


def _utc_time(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("_", "T").replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("valid_time must be a datetime or ISO UTC timestamp")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    if value.microsecond:
        raise ValueError("radar history needs whole-second valid_time; truncation would alias distinct volumes")
    return value


def _snapshot_history(state, cfg, grid, valid_time, directory):
    """Use the history owner's layout, precision and scheme diagnostic once.

    Native-atmosphere columns require rectilinear geographic axes, moist-air
    fractions and earth-relative mass winds. A projected WOOF C-grid cannot
    use that contract without rotating/re-staggering its winds and changing
    words. Its temporary history preserves the native grid instead.
    """
    import cupy as cp
    from woof.core.refl import analysis_refl_10cm
    from woof.io.wrfout import (
        WrfoutWriter, _device_state_frame, _state_device_context, wrf_global_attrs, wrfout_filename,
    )

    if grid is None:
        raise ValueError("in-memory radar needs grid= with georeferenced lat/lon and wind rotation; "
                         "a RunConfig alone does not locate the atmosphere")
    from woof.simulated_radar_capabilities import input_contracts
    contract = input_contracts()["wrf"]
    # Thermodynamics/hydrometeors and number moments are needed by the same
    # scheme-aware polar operator the history door uses, even with REFL_10CM.
    keep = set(contract["required_variables"]) | set(contract["optional_variables"])
    keep.update(name for alternative in contract["reflectivity_alternatives"] for name in alternative)
    with _state_device_context(state):
        fields = _device_state_frame(state)
        fields = {name: cp.asnumpy(value) for name, value in fields.items() if name in keep}
        driver = getattr(state, "physics", None)
        refl = getattr(driver, "refl_10cm", None)
        if refl is None:
            refl = analysis_refl_10cm(state, cfg)
        fields["REFL_10CM"] = cp.asnumpy(refl)
    lat, lon = grid.latlon_mass()
    sina, cosa = grid.rotation_m()
    fields.update(XLAT=lat, XLONG=lon, SINALPHA=sina, COSALPHA=cosa)
    attrs = wrf_global_attrs(grid, valid_time, run=cfg)
    path = Path(directory) / wrfout_filename(valid_time, getattr(cfg, "grid_id", 1))
    with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                      global_attrs=attrs, compression="none") as writer:
        writer.write_frame(valid_time.strftime("%Y-%m-%d_%H:%M:%S"), fields)
    return path


def simulate(source, *, site, outdir, scan_strategy="vcp212", formats=("level2",),
             fields=None, cfg=None, grid=None, valid_time=None, timing="history",
             elevations_deg=None, range_km=230.0, gate_spacing_m=250.0,
             azimuth_step_deg=1.0, volume_duration_s=300.0,
             color_tables="standard", band=None, wavelength_mm=None) -> RadarResult:
    """Simulate radar through the same native engine as ``woof simulated-radar``.

    ``source`` is a history path, a sequence of history paths, a DomainState
    with ``cfg=``/``grid=``, or an object with ``state``, ``cfg`` and ``grid``.
    Configurations containing ``run`` are unwrapped. An object exposing a
    ``native_columns`` path uses the existing native-atmosphere adapter.
    An in-memory state requires ``valid_time=`` (or ``source.valid_time``).
    History timestamps remain authoritative and cannot be overridden.

    ``site`` is a table ID or :class:`RadarSite`; antenna height is MSL.
    ``fields=None`` uses the door's automatic moments. Explicit unavailable
    moments, unsupported bands and missing binaries raise named refusals.
    No forecast is stepped. Call between steps, on the state's owning thread.
    See ``docs/simulated-radar-python.md`` for contracts, cost and examples.
    """
    from contextlib import ExitStack
    from woof import rustwx

    if not isinstance(site, (str, RadarSite)):
        raise TypeError("site must be a site ID or RadarSite")
    table = dict(enabled=True, sites=[site.to_mapping() if isinstance(site, RadarSite) else site],
                 scan_strategy=scan_strategy, formats=list(formats),
                 fields="auto" if fields is None else fields, timing=timing,
                 range_km=range_km, gate_spacing_m=gate_spacing_m,
                 azimuth_step_deg=azimuth_step_deg, volume_duration_s=volume_duration_s,
                 color_tables=color_tables, band=band, wavelength_mm=wavelength_mm)
    if elevations_deg is not None:
        table["elevations_deg"] = elevations_deg
    options = SimulatedRadarOptions.from_mapping(table, source="simulate()")
    outdir = Path(outdir).resolve()
    path_input = isinstance(source, (str, os.PathLike, list, tuple))
    columns = None if path_input else getattr(source, "native_columns", None)
    state = None if path_input or columns is not None else getattr(source, "state", source)
    config = cfg if cfg is not None else getattr(source, "cfg", None)
    config = getattr(config, "run", config)
    shape = []
    if state is not None:
        if config is None:
            raise ValueError("in-memory radar needs cfg=RunConfig or source.cfg")
        shape = [SimpleNamespace(run=config)]
    # The forecast door's binary and memory admission remains authoritative.
    require_admitted(SimpleNamespace(simulated_radar=options, domains=shape))
    with ExitStack() as stack:
        if path_input:
            if valid_time is not None:
                raise ValueError("history Times is authoritative; valid_time cannot override a history file")
            paths = [source] if isinstance(source, (str, os.PathLike)) else list(source)
            if not paths:
                raise ValueError("simulate needs at least one history path")
            paths = list(dict.fromkeys(Path(path).resolve() for path in paths))
            transport = "history"
        else:
            temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="simulated-radar-columns-"))
            if columns is not None:
                if valid_time is not None:
                    raise ValueError("native columns valid_time is authoritative; do not override it")
                paths = [rustwx.canonical_radar_scene(Path(columns), outdir=Path(temporary))]
                transport = "native_columns"
            else:
                instant = valid_time if valid_time is not None else getattr(source, "valid_time", None)
                instant = _utc_time(instant)
                paths = [_snapshot_history(state, config, grid if grid is not None else
                                            getattr(source, "grid", None), instant, temporary)]
                transport = "projected_staggered_history"
        for path in paths:
            if not path.is_file():
                raise ValueError(f"history input does not exist: {path}")
        resources = rustwx.estimate_simulated_radar(paths, outdir=outdir, config=options.to_mapping())
        if not resources.get("memory_admitted_now"):
            raise rustwx.SimulatedRadarRefusal(str(resources.get("memory_refusal")))
        events = []
        manifest = rustwx.simulate_radar(paths, outdir=outdir, config=options.to_mapping(), events=events)
        selected = {(event["domain"], event["valid_time"]) for event in events
                    if event.get("event") in ("volume_committed", "already_complete")}
        chosen = options.sites[0]
        site_id = chosen if isinstance(chosen, str) else chosen.id
        volumes = []
        for row in manifest["volumes"]:
            if row["site"]["id"] != site_id or (row["domain"], row["valid_time"]) not in selected:
                continue
            resolved = row["site"]
            resolved = RadarSite(resolved["id"], resolved["latitude_deg"],
                                 resolved["longitude_deg"], resolved["antenna_height_msl_m"],
                                 band=resolved.get("band", "S"),
                                 beamwidth_deg=getattr(chosen, "beamwidth_deg", 0.95))
            paths_by_format = {item["format"]: outdir / item["path"] for item in row["files"]}
            volumes.append(RadarVolumeResult(paths_by_format, _utc_time(row["valid_time"]),
                                            resolved, options.scan_strategy, row))
        if not volumes:
            raise RuntimeError("native radar manifest contains no volume for the requested site and inputs")
        paths_by_format = {name: tuple(volume.paths[name] for volume in volumes)
                           for name in options.formats}
        return RadarResult(paths_by_format, tuple(volumes), volumes[0].site, options.scan_strategy,
                           outdir / "radar" / "manifest.json", resources, transport)


def history_missing_columns(path) -> tuple[str, ...] | None:
    """What the radar needs that this history frame does not carry, from the frame's own header.

    ``()`` when the frame carries the vertical atmosphere ``rw_simradar``
    beams through (the variables and mass dimensions of its header admission,
    rw-simradar input_contract.rs ``validate_history``, as
    :func:`woof.simulated_radar_capabilities.input_contracts` lists them).
    The names it lacks when it is a 2D display frame, such as a quick-look
    frame (``quicklook = { history_vars }``) that keeps surface fields and
    reflectivity but no PH, W, U or V. ``None`` when the header cannot be
    read here (not a NetCDF file, a native-columns transport): the native
    admission decides, as before.

    Only the header is read, through the reader the history writer itself
    validates every frame with (:func:`woof.io.wrfout._validation_reader`).
    """
    import contextlib

    from woof.io.netcdf_serialization import netcdf4_session
    from woof.io.wrfout import _validation_reader
    from woof.simulated_radar_capabilities import input_contracts

    contract = input_contracts()["wrf"]
    try:
        reader, native = _validation_reader(path)
        # THE BREAKAGE THIS PREVENTS (2.9.10 pass 4 R7b route B, test_live_radar_scans_full_frames_and_skips_quick_look_frames):
        # this header read runs on a domain's writer thread while another thread holds a netCDF4 session (another domain's
        # frame write, a column-store read), and netCDF-C is not thread-safe: "NetCDF: Not a valid ID". Every netCDF4 session
        # in the process is opened under the process-wide lock (woof.io.netcdf_serialization); the native reader needs none.
        with (contextlib.nullcontext() if native else netcdf4_session()), reader(str(path)) as dataset:
            if getattr(dataset, "schema", None) == input_contracts()["native_columns"]["schema"]:
                return None
            names = set(dataset.variables)
            winds = getattr(dataset, "RADAR_NATIVE_WINDS", None)
            empty = [name for name in contract["required_dimensions"]
                     if name not in dataset.dimensions or len(dataset.dimensions[name]) == 0]
    except Exception:  # noqa: BLE001 - unreadable here: rw_simradar's own admission names it
        return None
    required = list(contract["required_variables"])
    alternatives = contract["reflectivity_alternatives"]
    required += [n for n in (alternatives[0] if alternatives[0][0] in names else alternatives[1])
                 if n not in required]
    if winds == contract["native_winds_marker"].split("=", 1)[1]:
        required += contract["native_winds_variables"]
    return tuple([n for n in required if n not in names] + [f"{n} (dimension)" for n in empty])


@functools.lru_cache(maxsize=8)
def _site_volume_bytes(config_json: str) -> int:
    from woof import bridges, rustwx

    # Pricing has no side effects: a stale staged artifact is reported by
    # the admission, never re-fetched by a disk estimate.
    with bridges.inspection_only():
        estimate = rustwx.estimate_simulated_radar(
            (), outdir=Path.cwd(), config=json.loads(config_json))
    return int(estimate["output"]["output_bytes_per_site_volume_upper_bound"])


def output_projection(experiment, frames_by_grid: dict) -> dict:
    """Disk bytes [simulated_radar] writes: one volume per site per frame.

    ``frames_by_grid`` is each grid's committed history frame count, the
    frames the radar scans. The per-site-volume figure is the native
    writer bound (``rw_simradar --estimate``). Automatic sites are chosen
    from each grid's coverage when its first history lands, so their count
    is not known here and the radar is named as unpriced instead of
    guessed.
    """
    options = experiment.simulated_radar
    if not options.enabled:
        return {"bytes": 0}
    if isinstance(options.sites, str):
        return {"bytes": None, "unpriced": (
            "simulated radar (sites = \"auto\" are chosen from domain coverage "
            "when the first history lands; list the sites to price their volumes "
            "before the run)")}
    try:
        per_volume = _site_volume_bytes(json.dumps(options.to_mapping(), sort_keys=True))
    except Exception as error:  # noqa: BLE001 - the admission names the cause
        first = str(error).splitlines()[0] if str(error) else type(error).__name__
        return {"bytes": None, "unpriced": f"simulated radar ({first})"}
    sites = len(options.sites)
    volumes = sites * sum(int(frames) for frames in frames_by_grid.values())
    return {"bytes": per_volume * volumes, "per_site_volume_bytes": per_volume,
            "sites": sites, "volumes": volumes,
            "basis": "native writer upper bound per site-volume (rw_simradar --estimate)"}


def run_with_radar(run, *args, radar_options, radar_outdir, progress_callback=None, **kwargs):
    """Give a synchronous history runner the same asynchronous radar path."""
    if not radar_options.enabled:
        return run(*args, progress_callback=progress_callback, **kwargs)
    with LiveSimulatedRadar(radar_options, radar_outdir) as radar:
        return run(*args, progress_callback=RadarProgress(progress_callback, radar), **kwargs)


class LiveSimulatedRadar:
    """Serialize radar work beside forecasting without retaining field arrays.

    A bounded queue holds file paths only. Native failures are stored because
    history landing observers are best effort; drain/close raise them on the
    forecast thread so a requested radar product cannot silently disappear.

    With ``timing = "scan"`` a history waits for its successor on the same
    grid, and the pair publishes the earlier one's volume once
    (``volume_paths``); the last history of each grid publishes at
    :meth:`close` with no successor. A forecast that stops early calls
    :meth:`cancel`, which drops queued histories, terminates the running
    ``rw_simradar`` and names on stderr every history left without a volume.

    ``runner`` is called as ``runner(paths, *, outdir, config, volume_paths,
    started)``; the default resolves and admits ``rw_simradar`` here, so a
    missing or stale binary refuses before the forecast's first step.

    Only histories that carry the vertical atmosphere are scanned
    (:func:`history_missing_columns`, read from each frame's own header as
    it lands). A 2D display frame, such as a quick-look frame between the
    full frames, is skipped and named at :meth:`close`; a run whose every
    frame lacks the columns refuses there, naming what they lack. THE
    BREAKAGE THIS PREVENTS (WOOF run_19793057ff2979e3, 2026-10-29: 12 h,
    2.25-0.75 km, full frames hourly, 5-min quick look, simulated radar):
    the first quick-look frame (d02 18:05Z) reached ``rw_simradar``, which
    refused it (radar_input_missing_columns: PH, W, U, V), and the stored
    failure stopped the whole forecast five minutes in, twice.
    """

    def __init__(self, options, outdir, *, runner=None, idle_timeout_seconds=None,
                 progress_interval_seconds=5.0, isolate_failures=None):
        self.options = options
        self.outdir = Path(outdir)
        if runner is None:
            from woof.rustwx import require_simulated_radar_binary, simulate_radar
            runner = functools.partial(simulate_radar, binary=require_simulated_radar_binary())
        self._runner = runner
        self._queue = queue.Queue(maxsize=64)
        self._failure = None
        self._closed = False
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._process = None
        self._current = None
        self._held = {}
        self._unfinished = []
        self._reported = False
        self._skipped = []
        self._scanned = 0
        self._snapshot_writers = {}
        self._scratch_slots = threading.BoundedSemaphore(RADAR_SCRATCH_FRAMES)
        self._owned_snapshots = {}
        self._scratch_deleted = []
        self._radar_map_frames = []
        self._idle_timeout = float(idle_timeout_seconds if idle_timeout_seconds is not None
                                   else os.environ.get("WOOF_RADAR_IDLE_TIMEOUT_SECONDS", "600"))
        self._progress_interval = min(float(progress_interval_seconds), self._idle_timeout / 4)
        if self._idle_timeout <= 0 or self._progress_interval <= 0:
            raise ValueError("radar idle timeout and progress interval must be positive")
        self._isolate_failures = inloop_enabled() if isolate_failures is None else bool(isolate_failures)
        self._isolated = None
        self._active = False
        self._last_volume_progress = time.monotonic()
        self._completed_frames = 0
        self._volume_keys = set()
        self._manifest_stamp = None
        self._monitor_stop = threading.Event()
        self._thread = threading.Thread(target=self._work, name="simulated-radar", daemon=True)
        self._thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor, name="simulated-radar-watch", daemon=True)
        self._monitor_thread.start()

    def capture(self, node, valid_time, template, metadata, *, refl_field=None, quicklook=False):
        """Snapshot the domain's current truth, with bounded disk and host staging.

        This is also callable by another forecast driver. A streamed domain reads
        its current store, never the resident state left behind at initialization.
        Only this consumer sees these private paths; they are not history outputs.
        """
        from woof.io.wrfout import AsyncDomainWrfoutWriter, wrfout_filename
        from woof.io.history_selection import HISTORY_VOCABULARY, HistorySelection
        import tempfile

        if not self.check():
            return
        with self._radar_wait() as pulse:
            while not self._scratch_slots.acquire(timeout=0.01):
                pulse()
                if not self.check():
                    return
                if self._closed or self._cancelled.is_set():
                    raise RuntimeError("radar snapshot requested after stop")
        path = None
        owned = False
        try:
            if not self.check():
                self._scratch_slots.release()
                return
            if self._closed or self._cancelled.is_set():
                raise RuntimeError("radar snapshot requested after close")
            if not hasattr(self, "_scratch_dir"):
                parent = self.outdir / ".radar-scratch"
                parent.mkdir(parents=True, exist_ok=True)
                self._scratch_dir = Path(tempfile.mkdtemp(prefix="frames-", dir=parent))
            gid = int(node.cfg.grid_id)
            path = self._scratch_dir / wrfout_filename(valid_time, gid)
            if path in self._owned_snapshots:
                raise RuntimeError(f"duplicate radar snapshot {path.name}")
            self._owned_snapshots[path] = {"domain": gid, "valid_time": valid_time.isoformat(), "quicklook": bool(quicklook)}
            owned = True
            writer = self._snapshot_writers.get(gid)
            if writer is None:
                writer = AsyncDomainWrfoutWriter(
                    **{key: getattr(template, key) for key in ("nx", "ny", "nz", "dx", "dy", "title", "soil_layers")},
                    global_attrs=template.global_attrs, grid_id=gid,
                    device=getattr(getattr(template, "_device", None), "id", None), landing_observer=self.output_committed)
                self._snapshot_writers[gid] = writer
            elif writer.global_attrs != template.global_attrs:
                writer.update_global_attrs(template.global_attrs)
            fields = RADAR_QUICKLOOK_FIELDS if quicklook else RADAR_SNAPSHOT_FIELDS
            selection = HistorySelection.from_mapping({"history_vars": [
                name for name in fields if name in HISTORY_VOCABULARY]})
            streamed = getattr(node.state, "_streamed_domain", None)
            if streamed is None:
                writer.submit(path, valid_time, node.state, extra_fields=metadata,
                              refl_field=refl_field, history_selection=selection)
            else:
                writer.submit(path, valid_time, None, frame=streamed.history_fields(),
                              extra_fields=metadata, refl_field=refl_field, history_selection=selection)
            # D2H must finish before the next step can mutate the state/store.
            # Native scan and rendering continue on the bounded radar worker.
            with self._radar_wait() as pulse:
                while getattr(writer, "pending", 0):
                    pulse()
                    time.sleep(0.01)
            writer.drain()
        except BaseException as error:
            if owned:
                self._release_snapshot(path)
            else:
                self._scratch_slots.release()
            if self._isolate_failures:
                self._isolate(error)
                return
            raise

    def _volume_progress(self):
        path = self.outdir / "radar/manifest.json"
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._manifest_stamp:
                return
            document = json.loads(path.read_text(encoding="utf-8"))
            keys = {(v.get("domain"), (v.get("site") or {}).get("id"), v.get("valid_time"),
                     v.get("generation"), tuple(f.get("sha256") for f in v.get("files", [])))
                    for v in document["volumes"] if v.get("files")}
        except (OSError, ValueError, KeyError, TypeError):
            return
        with self._lock:
            if keys - self._volume_keys:
                self._last_volume_progress = time.monotonic()
            self._volume_keys |= keys
            self._manifest_stamp = stamp

    def _stop_native(self):
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _isolate(self, error):
        with self._lock:
            if self._isolated is not None:
                return
            self._isolated = str(error)
            self._failure = error
        self.outdir.mkdir(parents=True, exist_ok=True)
        receipt = self.outdir / "radar-failure.json"
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps({"schema": "gpuwm.radar-failure/v1", "message": self._isolated,
                                         "completedFrames": self._completed_frames,
                                         "completedVolumes": len(self._volume_keys)}) + "\n", encoding="utf-8")
        temporary.replace(receipt)
        print(f"simulated radar isolated: {self._isolated}; the forecast continues", file=sys.stderr, flush=True)
        self._stop_native()

    def _monitor(self):
        while not self._monitor_stop.wait(self._progress_interval):
            self._volume_progress()
            with self._lock:
                stalled = self._active and time.monotonic() - self._last_volume_progress > self._idle_timeout
            if stalled:
                error = TimeoutError(f"Simulated radar completed no volume for {self._idle_timeout:g} s; "
                                     "the radar product was stopped, preserving the forecast")
                if self._isolate_failures:
                    self._isolate(error)
                else:
                    self._failure = error
                    self._stop_native()
                return

    @contextlib.contextmanager
    def _radar_wait(self):
        # WOOF watches the go log; ForecastWatchdog watches this owned heartbeat.
        # A one-shot _write_beat did not reach both (run_8aa07fb55650754a).
        # Refresh only while the MODEL is blocked on radar, never for background
        # radar alone. Only completed volumes reset radar's independent timeout.
        from woof.supervisor import read_heartbeat, write_heartbeat, utc_now
        path = self.outdir / "run-progress.json"
        try:
            before = read_heartbeat(path)
            if before.pid != os.getpid():
                before = None
        except (OSError, ValueError):
            before = None
        next_beat = [0.0]
        def pulse():
            now = time.monotonic()
            if now < next_beat[0]:
                return
            next_beat[0] = now + self._progress_interval
            if before is not None:
                try:
                    current = read_heartbeat(path)
                    if current.pid == os.getpid() and current.run_id == before.run_id:
                        write_heartbeat(path, dataclasses.replace(current, status="writing:simulated-radar",
                                        updated_at_utc=utc_now(), work_bytes=None, wait=None))
                except (OSError, ValueError):
                    pass
            print(f"simulated radar: waiting for bounded radar work; {self._completed_frames} frames completed", file=sys.stderr, flush=True)
        try:
            yield pulse
        finally:
            if before is not None:
                try:
                    current = read_heartbeat(path)
                    if (current.pid == os.getpid() and current.run_id == before.run_id
                            and current.status == "writing:simulated-radar"):
                        write_heartbeat(path, dataclasses.replace(current, status=before.status,
                                        updated_at_utc=utc_now(), work_bytes=before.work_bytes, wait=before.wait))
                except (OSError, ValueError):
                    pass

    def _release_snapshot(self, path):
        path = Path(path)
        if path not in self._owned_snapshots:
            return
        size = path.stat().st_size if path.exists() else 0
        path.unlink(missing_ok=True)
        (path.parent / "ready" / (path.name + ".json")).unlink(missing_ok=True)
        self._scratch_deleted.append({"path": str(path.relative_to(self.outdir)), "bytes": size})
        del self._owned_snapshots[path]
        self._scratch_slots.release()

    def _render_snapshot(self, path):
        metadata = self._owned_snapshots.get(Path(path))
        if metadata is None:
            return
        from woof.go_cli import render_command
        target = Path(os.environ.get("WOOF_RADAR_MAP_DIR", str(self.outdir.parent / "radar-png")))
        plan = {"run": self.outdir, "render": target,
                "render_products": "composite_reflectivity,1km_reflectivity"}
        completed = subprocess.run(render_command(plan, [Path(path)]), capture_output=True, text=True, timeout=self._idle_timeout)
        if completed.returncode:
            raise RuntimeError(f"radar maps failed: {completed.stderr[-2000:]}")
        self._radar_map_frames.append({**metadata, "complete": True, "exit_code": 0})
        target.mkdir(parents=True, exist_ok=True)
        index = target / "radar-maps.json"
        temporary = index.with_suffix(".tmp")
        temporary.write_text(json.dumps({"frames": self._radar_map_frames}) + "\n")
        temporary.replace(index)

    def output_committed(self, *, domain, valid_time, path):
        if os.environ.get("WOOF_RADAR_INLOOP") == "0" and Path(path).parent.name == "quicklook":
            return
        if self._cancelled.is_set():
            # A frame the stopping forecast still committed: named, not scanned.
            self._unfinished.append((int(domain), str(Path(path).resolve())))
            return
        if not self.check():
            return
        if self._closed:
            raise RuntimeError("simulated radar received history after close")
        # Read here, on the landing, while the frame is certainly on disk: a
        # consumer may move a quick-look frame off the box once it is drawn.
        missing = history_missing_columns(path)
        with self._lock:  # every domain's writer thread lands here
            if missing:
                self._skipped.append((int(domain), str(Path(path).resolve()), missing))
                return
            self._scanned += 1
        item = (int(domain), str(Path(path).resolve()))
        while not self._isolated:
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if not self.check():
                    return

    def skipped(self) -> list[tuple[int, str, tuple[str, ...]]]:
        """Histories not scanned because they carry no vertical atmosphere: (domain, path, what each lacks)."""
        return list(self._skipped)

    def _report_skipped(self):
        if not self._skipped:
            return
        from collections import Counter
        lacks = sorted({name for _, _, missing in self._skipped for name in missing})
        domains = ", ".join(f"d{domain:02d} x{count}" for domain, count
                            in sorted(Counter(d for d, _, _ in self._skipped).items()))
        print(f"simulated radar: {len(self._skipped)} history file(s) carry no vertical atmosphere "
              f"(no {', '.join(lacks)}; 2D display frames such as quick-look frames) and were not "
              f"scanned: {domains}. Radar volumes come from the full history frames.", file=sys.stderr)

    def _started(self, process):
        with self._lock:
            self._process = process
            stop = self._cancelled.is_set() or self._isolated is not None
        if stop:
            process.terminate()

    def _run(self, paths, volume_paths):
        targets = volume_paths if volume_paths is not None else paths
        config = self.options.to_mapping()
        if targets and all(self._owned_snapshots.get(Path(p), {}).get("quicklook") for p in targets):
            config = {**config, "fields": ["reflectivity", "velocity"]}
            config.update(quicklook_scan(config))
        with self._lock:
            self._active = True
            self._last_volume_progress = time.monotonic()
        try:
            self._runner(paths, outdir=self.outdir, config=config,
                         volume_paths=volume_paths, started=self._started)
            with self._lock:
                self._completed_frames += len(targets)
                self._last_volume_progress = time.monotonic()
            if self._isolated:
                return
            for path in (volume_paths if volume_paths is not None else paths):
                self._render_snapshot(path)
        finally:
            for path in (volume_paths if volume_paths is not None else paths):
                self._release_snapshot(path)
            with self._lock:
                self._process = None
                self._active = False

    def _scan(self, item):
        domain, path = item
        if self.options.timing != "scan":
            self._run([path], None)
            return
        previous = self._held.get(domain)
        self._held[domain] = path
        if previous is not None:
            # The successor bounds the earlier scan: publish it once, now.
            self._run([previous, path], [previous])

    def _work(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    if not self._cancelled.is_set() and self._failure is None:
                        # The last history of each grid has no successor.
                        for domain in sorted(self._held):
                            if self._cancelled.is_set():
                                break
                            self._current = (domain, self._held.pop(domain))
                            self._run([self._current[1]], [self._current[1]])
                            self._current = None
                    return
                if self._cancelled.is_set():
                    self._unfinished.append(item)
                    continue
                if self._failure is not None:
                    continue
                self._current = item
                self._scan(item)
                self._current = None
            except BaseException as error:
                if self._cancelled.is_set():
                    if self._current is not None:
                        self._unfinished.append(self._current)
                elif self._failure is None:
                    self._failure = error
                    if self._isolate_failures:
                        self._isolate(error)
                self._current = None
                if item is None:
                    return
            finally:
                self._queue.task_done()

    def check(self):
        if self._failure is not None:
            if self._isolate_failures:
                self._isolate(self._failure)
                return False
            raise RuntimeError(f"simulated radar failed: {self._failure}") from self._failure
        return True

    def drain(self):
        for writer in self._snapshot_writers.values():
            writer.drain()
        with self._radar_wait() as pulse:
            while self._queue.unfinished_tasks and not self._isolated:
                pulse()
                self.check()
                time.sleep(0.01)
        self.check()

    def close(self):
        """Finish every queued scan and raise a stored radar failure.

        After :meth:`cancel` it scans nothing and names, on stderr, every
        committed history left without a volume, including frames the
        stopping forecast committed after the cancel.
        """
        if self._cancelled.is_set():
            self._report_unfinished()
            return
        if not self._closed:
            for writer in self._snapshot_writers.values():
                writer.close()
            self._closed = True
            with self._radar_wait() as pulse:
                while True:
                    if self._isolated:
                        while True:
                            try:
                                self._queue.get_nowait()
                                self._queue.task_done()
                            except queue.Empty:
                                break
                    try:
                        self._queue.put(None, timeout=0.01)
                        break
                    except queue.Full:
                        pulse()
            try:
                with self._radar_wait() as pulse:
                    while self._thread.is_alive() and not self._isolated:
                        pulse()
                        self._thread.join(timeout=0.01)
                if self._isolated:
                    self._thread.join(timeout=0.2)
            except BaseException:
                # Ctrl-C while the last scans run stops them too.
                self.cancel()
                self._report_unfinished()
                raise
            self._report_skipped()
        self._monitor_stop.set()
        self._monitor_thread.join(timeout=0.2)
        if not self._thread.is_alive():
            for path in list(self._owned_snapshots):
                self._release_snapshot(path)
            if self._scratch_deleted:
                receipt = self.outdir / "radar-scratch-receipt.json"
                receipt.write_text(json.dumps({"maxFramesInFlight": RADAR_SCRATCH_FRAMES,
                                              "deleted": self._scratch_deleted}) + "\n")
        self.check()
        if self._skipped and not self._scanned:
            # Asked for and nothing to scan: a requested product never silently disappears.
            _, path, missing = self._skipped[0]
            raise RuntimeError(
                f"simulated radar failed: none of the {len(self._skipped)} history file(s) carries the vertical "
                f"atmosphere the radar beams through ({Path(path).name} lacks {', '.join(missing)}); write full "
                "history frames, or keep those fields in the history selection")

    def cancel(self):
        """Stop radar for a forecast that is stopping.

        Queued histories are not scanned and the running ``rw_simradar`` is
        terminated (killed if it has not exited 10 s later), so neither a
        failed forecast nor Ctrl-C waits for radar work nobody will use and
        no native process outlives the run. :meth:`close` then names what
        was left.
        """
        if self._cancelled.is_set():
            return
        self._closed = True
        self._monitor_stop.set()
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                self._unfinished.append(item)
            self._queue.task_done()
        self._queue.put(None)
        self._thread.join(timeout=30)
        if not self._thread.is_alive():
            for path in list(self._owned_snapshots):
                self._release_snapshot(path)

    def unfinished(self) -> list[tuple[int, str]]:
        """Committed histories a cancelled run left without a radar volume."""
        seen, result = set(), []
        for item in [*self._unfinished, *sorted(self._held.items())]:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _report_unfinished(self):
        left = self.unfinished()
        if not left or self._reported:
            return
        self._reported = True
        names = ", ".join(f"d{domain:02d} {Path(path).name}" for domain, path in left)
        print(f"simulated radar: the forecast stopped, so {len(left)} committed "
              f"history file(s) have no radar volume: {names}. "
              f"{self.outdir / 'radar' / 'manifest.json'} lists the volumes that "
              "were completed; `woof simulated-radar HISTORY ... --config CONFIG.toml "
              f"--outdir {self.outdir}` scans the rest.", file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.cancel()
        self.close()
        return False


class RadarProgress:
    """Preserve existing progress hooks and add the radar landing consumer."""

    def __init__(self, observer, radar):
        self.observer = observer
        self.radar = radar

    def __call__(self, *args, **kwargs):
        self.radar.check()
        if self.observer is not None:
            return self.observer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.observer, name)

    def output_committed(self, **kwargs):
        self.radar.output_committed(**kwargs)
        callback = getattr(self.observer, "output_committed", None)
        if callback is not None:
            callback(**kwargs)


def site_catalog():
    """Built-in simulation sites and world reference rows with known bands.

    Reference rows without antenna height need a coordinate-defined RadarSite.
    An unknown band remains null; it is not an S-band assumption.
    """
    from woof import rustwx
    result = subprocess.run([str(rustwx.require_simulated_radar_binary()), "--site-catalog"],
                            capture_output=True, text=True, timeout=60, env=rustwx.renderer_env())
    if result.returncode:
        raise rustwx.SimulatedRadarRefusal(result.stderr.strip())
    catalog = json.loads(result.stdout)
    if catalog.get("schema") != "simulated-radar.site-catalog/v1":
        raise rustwx.SimulatedRadarRefusal("native radar returned an incompatible site catalog")
    return catalog


def register_cli(subparsers):
    parser = subparsers.add_parser("simulated-radar", help="make radar volumes and PPI images from saved forecast history")
    parser.add_argument("history", nargs="*", type=Path, help="history files or run directories")
    parser.add_argument("--config", type=Path, help="TOML with a [simulated_radar] table")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="run root receiving radar/manifest.json")
    parser.add_argument("--sites", help="auto or comma-separated radar IDs")
    parser.add_argument("--formats", help="comma-separated output formats")
    parser.add_argument("--band", choices=("S", "C", "X"), help="override the site band")
    parser.add_argument("--wavelength-mm", type=float, help="select legacy S at 107.068735 mm or a 53.5/33.3 mm C/X table")
    parser.add_argument("--timing", choices=("history", "scan"),
                        help="history scans each saved snapshot; scan lets rays use neighboring "
                             "history times (overrides the --config table; default history)")
    parser.add_argument("--input-kind", choices=("wrf", "native-columns"), default="wrf",
                        help="full WRF histories or native-atmosphere.columns/v1 transports")
    parser.add_argument("--describe", action="store_true", help="inspect installed native capabilities, accepted inputs and routes as JSON")
    parser.add_argument("--list-sites", action="store_true", help="show simulation sites and world reference rows with band metadata")
    parser.add_argument("--estimate", action="store_true",
                        help="report native scan dimensions and memory admission without generating radar")
    parser.set_defaults(func=main)


def main(args):
    if getattr(args, "list_sites", False):
        print(json.dumps(site_catalog(), indent=2))
        return 0
    if args.describe:
        from woof.simulated_radar_capabilities import describe
        print(json.dumps(describe(), indent=2))
        return 0
    options = load_options(args.config).to_mapping() if args.config else {"enabled": True}
    options["enabled"] = True
    for key in ("sites", "formats"):
        value = getattr(args, key)
        if value is not None:
            options[key] = "auto" if key == "sites" and value == "auto" else value.split(",")
    if args.timing:
        options["timing"] = args.timing
    for key in ("band", "wavelength_mm"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    config = SimulatedRadarOptions.from_mapping(options)
    paths = []
    skipped = []
    for path in args.history:
        if path.is_dir():
            if args.input_kind == "native-columns":
                paths.extend(sorted(path.glob("*.nc")))
            else:
                from woof.io.wrfout import iter_wrfout_files
                # A run directory holds 2D display frames too (quicklook/):
                # only histories with the vertical atmosphere are scanned.
                for frame in iter_wrfout_files(path, include_temporaries=False):
                    missing = history_missing_columns(frame)
                    if missing:
                        skipped.append((frame, missing))
                    else:
                        paths.append(frame)
        elif path.is_file():
            paths.append(path)
        else:
            raise ValueError(f"history input does not exist: {path}")
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if skipped:
        lacks = sorted({name for _, missing in skipped for name in missing})
        print(f"simulated radar: {len(skipped)} history file(s) carry no vertical atmosphere "
              f"(no {', '.join(lacks)}; 2D display frames such as quick-look frames) and are not "
              f"scanned, e.g. {skipped[0][0].name}", file=sys.stderr)
        if not paths:
            raise ValueError(f"simulated-radar found no history with the vertical atmosphere: "
                             f"{skipped[0][0]} lacks {', '.join(skipped[0][1])}")
    if args.estimate:
        if paths and args.input_kind == "native-columns":
            raise ValueError("--estimate reads full canonical scene headers; native-columns must first "
                             "be converted with the shared adapter, or omit histories for geometry-only estimation")
        from woof.rustwx import estimate_simulated_radar
        print(json.dumps(estimate_simulated_radar(paths, outdir=args.outdir,
                                                  config=config.to_mapping()), indent=2))
        return 0
    if not paths:
        raise ValueError("simulated-radar needs at least one committed history file")
    if args.input_kind == "native-columns":
        import hashlib
        from woof.rustwx import canonical_radar_scene
        # Distinct source directories may use the same filename. Isolate each
        # conversion so a later source cannot overwrite an earlier scene.
        paths = [canonical_radar_scene(
            path, outdir=args.outdir / "radar" / "native-scenes" /
            hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]) for path in paths]
    from woof.rustwx import simulate_radar
    result = simulate_radar(paths, outdir=args.outdir, config=config.to_mapping())
    print(json.dumps(result, indent=2))
    return 0
