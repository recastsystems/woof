"""Write the any-point column store while the forecast runs: every output hour, every grid.

Every run writes the tiled compact column store for every output hour, so a sounding at any point
and any hour costs nothing extra; the tiled layout is rw-store version 2 (``RWSTORE2``, ``rw-store.columns.v2``,
tools/rustwx/crates/rw-store/src/columns.rs). Each committed history frame on the cadence (whole hours by default)
becomes ``<run>/columns/<domain>/<YYYYMMDDTHHMMSSZ>.wcol`` through ``rw_soundings --columns``
(:func:`woof.soundings.write_columns`), and at the end each grid's last frame too when the run ends off the hour.
Only frames that carry the 3-D state are taken (:class:`StateFrames`): with ``[output] full_history_interval_s``
the frames between the full-history leads hold the 2-D subset, and the store follows the full frames, said once.

Shaped like :class:`woof.grib2_live.LiveGrib2Export`: one serial queue on its own thread, independent of map
rendering, so a frame is written while the model integrates the next hour and a slow picture never delays it.
:meth:`LiveColumns.frame_committed` appends and returns; nothing raised here reaches the model loop, and a frame
that cannot be written is a warning, never a failed run.

Ensembles: :class:`MemberColumns` writes each member's store the same way, from the member's frame while the
ordinary writer's callback holds it (an ensemble keeps no member history files by default), under
``<run>/columns/members/<member>/<domain>/``.

Point forecasts: every file carries the surface section (precipitation, sea-level pressure, cloud, gust, CAPE and
the rest, exact f32, ``rw_soundings`` ABI ``columns-surface-v1``), and at :meth:`LiveColumns.stop` /
:meth:`MemberColumns.stop` each fixed grid's hourly files are folded into ``<store>/<domain>/series.wser``
(:func:`woof.soundings.write_series`), the time series an Open-Meteo-style point forecast or a meteogram reads in
three small reads. A series that cannot be written is said, never a failed run.

Settings: ``WOOF_COLUMNS=off`` (or ``run_options.columns = false``) writes no store; ``WOOF_COLUMNS_EVERY_MINUTES``
is the cadence (default 60, 0 writes every frame); ``WOOF_COLUMNS_THREADS`` the tile workers per frame (default a
quarter of the cores, at least 2: the forecast keeps the rest); ``WOOF_COLUMNS_TIMEOUT_SECONDS`` the longest one
frame's ``rw_soundings --columns`` may take before it is killed and said (default 600).
"""
from __future__ import annotations

import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})")
_DOMAIN = re.compile(r"wrfout_(d\d{2})")


def columns_enabled(run_options: dict | None = None, environ=None) -> bool:
    """On unless switched off by ``WOOF_COLUMNS=off`` or ``run_options.columns = false``."""
    env = (os.environ if environ is None else environ).get("WOOF_COLUMNS", "").strip().lower()
    if env in ("0", "off", "false", "no"):
        return False
    return bool((run_options or {}).get("columns", True))


def every_minutes(environ=None) -> int:
    text = (os.environ if environ is None else environ).get("WOOF_COLUMNS_EVERY_MINUTES", "60")
    try:
        return max(0, int(text))
    except ValueError:
        return 60


def tile_threads(environ=None) -> int:
    text = (os.environ if environ is None else environ).get("WOOF_COLUMNS_THREADS")
    if text:
        try:
            return max(1, int(text))
        except ValueError:
            pass
    return max(2, (os.cpu_count() or 8) // 4)


def frame_timeout_seconds(environ=None) -> float:
    """The longest one frame's store write may take. THE BREAKAGE: an unbounded ``rw_soundings --columns`` that hung
    at exit (as the same binary did in c665034731) or stalled on a full disk held finalize forever, so a finished
    forecast wrote no receipts and kept its box."""
    text = (os.environ if environ is None else environ).get("WOOF_COLUMNS_TIMEOUT_SECONDS", "600")
    try:
        return max(1.0, float(text))
    except ValueError:
        return 600.0


def _default_writer(timeout: float):
    from functools import partial

    from woof.soundings import write_columns
    return partial(write_columns, timeout=timeout)


def _default_series(timeout: float):
    from functools import partial

    from woof.soundings import write_series
    return partial(write_series, timeout=timeout)


def _series(writer, out: Path, warn) -> list:
    """Each fixed grid's point-forecast series under ``out`` on every core (the forecast has ended); never raises."""
    if writer is None or not Path(out).is_dir():
        return []
    try:
        return list(writer(out, threads=os.cpu_count() or 4))
    except Exception as error:  # a series not written never stops the run: readers use the hourly files
        warn("series_not_written", f"no point-forecast series for {out}: {type(error).__name__}: {error}"[:400],
             out=str(out))
        return [{"domain": "*", "error": f"{type(error).__name__}: {error}"[:400]}]


def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def on_cadence(path: Path, minutes: int) -> bool:
    m = _STAMP.search(path.name)
    if m is None or minutes <= 0:
        return True
    return (int(m[2]) * 60 + int(m[3])) % minutes == 0 and int(m[4]) == 0


class LiveColumns:
    """One serial column-store writer for a run's committed frames."""

    def __init__(self, out: Path, *, report: Callable[[dict], None] | None = None,
                 warn: Callable[..., None] | None = None, writer: Callable[..., list] | None = None,
                 minutes: int | None = None, threads: int | None = None, timeout: float | None = None,
                 frames: Callable[..., bool] | None = None, series: Callable[..., list] | None = None):
        self.out = Path(out)
        #: Which committed frames carry the 3-D state ``rw_soundings --columns`` reads (:class:`StateFrames`), or
        #: ``None`` to take every frame on the cadence. A frame that does not is never attempted.
        self.frames = frames
        self._skipped = 0
        self._noted: set[str] = set()      # grids whose state-less frames were said
        self._report = report or (lambda event: None)
        self._warn = warn or (lambda *a, **k: None)
        self._timeout = frame_timeout_seconds() if timeout is None else float(timeout)
        self._writer = _default_writer(self._timeout) if writer is None else writer
        #: Folds the store into each fixed grid's series at :meth:`stop` (``None`` in a test writer's run: the
        #: default series writer runs only beside the default store writer).
        self._series_writer = (_default_series(self._timeout) if writer is None else None) if series is None else series
        self._series: list = []
        self._abandoned = False
        self._minutes = every_minutes() if minutes is None else int(minutes)
        self._threads = tile_threads() if threads is None else int(threads)
        self._queue: queue.Queue = queue.Queue()
        self._last: dict[str, Path] = {}       # each grid's latest committed frame
        self._done: set[str] = set()           # frames written
        self._written: list[dict[str, Any]] = []
        self._failed: list[dict[str, Any]] = []
        self._seconds = 0.0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, name="gpuwm-live-columns", daemon=True)
        self._thread.start()

    def frame_committed(self, *, domain: int, valid_time, path) -> bool:
        """Queue one durable frame when it is on the cadence; remember it as its grid's latest either way.

        A frame written without the 3-D state (:attr:`frames` says so) is neither queued nor remembered: a frame
        between full-history leads, a quick-look frame, a selection that sheds ``P``. THE BREAKAGE: each such frame
        on the cadence ran ``rw_soundings --columns`` and warned "column store not written: variable not found in
        WRF file: P" (21 warnings per 24 h HRRR-recipe member with full history every 3 h, updraft factory
        2026-10-27). The grid's first one is said once, as a ``state_frames_only`` progress event."""
        frame = Path(path)
        m = _DOMAIN.search(frame.name)
        grid = m[1] if m else f"d{int(domain):02d}"
        if not self._carries_state(domain, valid_time, frame):
            with self._lock:
                self._skipped += 1
                first = grid not in self._noted
                self._noted.add(grid)
            if first:
                describe = getattr(self.frames, "describe", None)
                cadence = describe(domain) if callable(describe) else "this frame carries no 3-D state"
                self._report({"event": "state_frames_only", "domain": grid, "frame": str(frame),
                              "note": f"the {grid} column store takes the frames that carry the 3-D state: "
                                      f"{cadence}; the others are not attempted"})
            return False
        with self._lock:
            self._last[grid] = frame
        if on_cadence(frame, self._minutes):
            self._queue.put(frame)
            return True
        return False

    def _carries_state(self, domain, valid_time, frame: Path) -> bool:
        if self.frames is None:
            return True
        try:
            return bool(self.frames(int(domain), valid_time, frame))
        except Exception:  # noqa: BLE001 - an undecidable frame is attempted, as every frame was before
            return True

    def _write(self, frame: Path) -> None:
        key = str(frame)
        with self._lock:
            if key in self._done:
                return
            self._done.add(key)
        started = time.perf_counter()
        try:
            rows = self._writer(frame, self.out, threads=self._threads)
        except Exception as error:  # a frame not written never stops the run
            row = {"frame": key, "error": f"{type(error).__name__}: {error}"[:400]}
            with self._lock:
                self._failed.append(row)
            self._warn("columns_not_written", f"no column store for {frame.name}: {row['error']}", frame=key)
            return
        seconds = time.perf_counter() - started
        with self._lock:
            self._seconds += seconds
            for r in rows:
                self._written.append({"frame": key, "path": r.get("path"), "bytes": int(r.get("bytes", 0)),
                                      "ms": r.get("ms"), "surface_ms": r.get("surface_ms")})
        for r in rows:
            self._report({"event": "columns_written", "frame": key, "path": r.get("path"),
                          "bytes": int(r.get("bytes", 0)), "seconds": round(seconds, 3)})

    def _work(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            if not self._abandoned:
                self._write(frame)

    @property
    def restart_wait_seconds(self) -> float:
        """How long a restart waits for the frame in hand: its write is killed at :func:`frame_timeout_seconds`."""
        return self._timeout + 30.0

    def stop(self, timeout: float | None = None) -> dict:
        """Write what is queued, then each grid's last frame when the run ended off the cadence.

        Bounded: by default each frame still to write gets its own :func:`frame_timeout_seconds`. A writer still
        busy after that is said (``columns_not_written``) and left behind, and the frames after it are dropped, so
        a stuck store never holds the run's finalize."""
        with self._lock:
            last = list(self._last.values())
        for frame in last:
            if str(frame) not in self._done:
                self._queue.put(frame)
        if timeout is None:
            timeout = self._timeout * (self._queue.qsize() + 1) + 30.0
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            self._abandoned = True
            self._warn("columns_not_written", f"the column store writer did not finish in {timeout:.0f} s; "
                       "the run ends without the frames still queued", out=str(self.out))
        elif self._written:
            self._series = _series(self._series_writer, self.out, self._warn)
            for row in self._series:
                if row.get("path"):
                    self._report({"event": "series_written", "domain": row["domain"], "path": row["path"],
                                  "bytes": int(row.get("bytes", 0)), "times": row.get("times")})
        return self.summary()

    def halt(self, timeout: float = 5.0) -> dict:
        """A stopped or restarting run: the frames still queued are dropped and the frame in hand finishes
        (within ``timeout``). ``stopped`` in the summary says whether the writer has ended, so nothing is still
        reading the attempt's frames."""
        self._abandoned = True
        _drain(self._queue)
        self._queue.put(None)
        self._thread.join(timeout)
        return self.summary()

    def summary(self) -> dict:
        with self._lock:
            return {"schema": "gpuwm-live-columns.v1", "out": str(self.out), "every_minutes": self._minutes,
                    "stopped": not self._thread.is_alive(),
                    "files": len(self._written), "bytes": sum(r["bytes"] for r in self._written),
                    "seconds": round(self._seconds, 3), "failed": list(self._failed),
                    "skipped_without_state": self._skipped, "written": list(self._written),
                    "surface_ms": round(sum(float(r.get("surface_ms") or 0.0) for r in self._written), 1),
                    "series": list(self._series)}


#: What ``rw_soundings --columns`` reads from a WRF-shaped frame (rw-soundings input.rs ``wrf_grid``): the 3-D state,
#: the surface fields, and the grid (latitude, longitude, map rotation). A member's frame is cut to these.
SOUNDING_FIELDS = ("P", "PB", "PH", "PHB", "T", "QVAPOR", "U", "V", "W", "PSFC", "HGT", "T2", "Q2", "U10", "V10",
                   "XLAT", "XLONG", "COSALPHA", "SINALPHA")
#: What the store's surface section reads when the frame has it (rw-soundings surface.rs): accumulations, snow,
#: radiation, the PBL height, reflectivity, the window maxima and the hydrometeors behind cloud and visibility. A
#: member's frame keeps each one it has (an ensemble keeps no member history, so these are the only copy a member's
#: point forecast has); one it lacks leaves that plane out of the store, named, never filled.
SURFACE_FIELDS = ("RAINC", "RAINNC", "RAINSH", "SNOWNC", "GRAUPELNC", "HAILNC", "SNOWH", "SWDOWN", "PBLH", "REFC",
                  "WSPD10MAX", "GUST_EST_MAX", "REFL_10CM", "QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP",
                  "UP_HELI_MAX", "UP_HELI_MAX01", "UP_HELI_MAX02", "UP_HELI_MAX03")
_SURFACE_GRID = ("XLAT", "XLONG", "COSALPHA", "SINALPHA", "HGT")
#: The 2-D ones among them (squeezed to [ny, nx] when handed over with a length-1 leading axis).
_SURFACE_2D = ("RAINC", "RAINNC", "RAINSH", "SNOWNC", "GRAUPELNC", "HAILNC", "SNOWH", "SWDOWN", "PBLH", "REFC",
               "WSPD10MAX", "GUST_EST_MAX", "UP_HELI_MAX", "UP_HELI_MAX01", "UP_HELI_MAX02", "UP_HELI_MAX03")
#: The ones a frame cannot be stored without (the map rotation is optional).
STATE_FIELDS = tuple(n for n in SOUNDING_FIELDS if n not in ("COSALPHA", "SINALPHA"))


def carries_state(selection, lead_seconds=None) -> bool:
    """Whether a frame written with ``selection`` (a :class:`woof.io.history_selection.HistorySelection`) at
    ``lead_seconds`` keeps every field of :data:`STATE_FIELDS`: the subset it takes at that lead
    (:meth:`~woof.io.history_selection.HistorySelection.for_lead`, the writer's own rule) is asked by name."""
    if selection is None:
        return True
    chosen = selection.for_lead(lead_seconds)
    return all(chosen.keeps(name) for name in STATE_FIELDS)


def _lead_seconds(valid_time, path: Path, start) -> float | None:
    """Seconds from ``start`` to the frame's valid time (the object, its text, else the file's stamp), or None."""
    if not isinstance(start, datetime):
        return None
    valid = valid_time if isinstance(valid_time, datetime) else None
    if valid is None and valid_time is not None:
        try:
            valid = datetime.strptime(str(valid_time)[:19].replace("T", "_"), "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            valid = None
    if valid is None:
        m = _STAMP.search(Path(path).name)
        if m is None:
            return None
        valid = datetime.strptime(f"{m[1]}_{m[2]}:{m[3]}:{m[4]}", "%Y-%m-%d_%H:%M:%S")
    return (valid.replace(tzinfo=None) - start.replace(tzinfo=None)).total_seconds()


class StateFrames:
    """Which of a run's committed history frames carry the 3-D state the column store reads.

    Decided from the run's own ``[output]`` selection the way the writer decides what a frame holds
    (``WrfoutWriter._history_plan``), without opening the file: per grid the domain's own selection else the
    tree's (:func:`woof.io.history_selection.resolve`), at the frame's lead from the domain's own start (the
    writer's ``START_DATE``), so a frame between ``full_history_interval_s`` leads takes the between-hours subset;
    a quick-look frame (``quicklook/``) takes the domain's quick-look selection. A grid it does not know, or a
    frame whose lead cannot be read, counts as carrying it (it is attempted, as every frame was before)."""

    def __init__(self, grids: dict):
        #: ``{grid_id, or None for any grid: (selection, start, quick-look selection)}``
        self._grids = dict(grids)

    @classmethod
    def for_experiment(cls, experiment) -> "StateFrames":
        from woof.io.history_selection import resolve
        tree = getattr(experiment, "output", None)
        start = getattr(experiment, "start_time", None)
        return cls({int(d.grid_id): (resolve(tree, getattr(d, "output", None)),
                                     getattr(d, "start_time", None) or start,
                                     getattr(d, "quicklook_output", None))
                    for d in (getattr(experiment, "domains", None) or ())})

    @classmethod
    def for_selection(cls, selection) -> "StateFrames":
        """One standing selection for every frame, whatever its lead (an offline child's ``[output]``)."""
        return cls({None: (selection, None, None)})

    def _entry(self, domain):
        return self._grids.get(int(domain), self._grids.get(None))

    def __call__(self, domain, valid_time, path) -> bool:
        entry = self._entry(domain)
        if entry is None:
            return True
        selection, start, quicklook = entry
        if quicklook is not None and Path(path).parent.name == "quicklook":
            return carries_state(quicklook)
        return carries_state(selection, _lead_seconds(valid_time, Path(path), start))

    def describe(self, domain) -> str:
        """The cadence of the frames that do, in words, for the one line that says it."""
        entry = self._entry(domain)
        selection = entry[0] if entry is not None else None
        interval = float(getattr(selection, "full_history_interval_s", 0.0) or 0.0)
        if interval and carries_state(selection):
            return (f"full history every {interval:g} s ([output] full_history_interval_s), the frames between "
                    "hold the 2-D subset")
        if entry is not None and entry[2] is not None and carries_state(selection):
            return "the quick-look frames hold their subset only"
        return "this grid's [output] selection writes no 3-D state (P, PH, T, QVAPOR, U, V, W)"


def _host(array):
    """A host float32 copy (the caller's buffers are borrowed until the next step)."""
    import numpy as np
    get = getattr(array, "get", None)
    if callable(get) and not isinstance(array, np.ndarray):
        array = get()
    return np.array(array, dtype=np.float32, copy=True, order="C")


def sounding_frame(*, state=None, streamed_fields=None, metadata=None) -> dict:
    """One member frame cut to :data:`SOUNDING_FIELDS`, host float32, from what the ordinary history writer would
    have written: a streamed domain's history fields, else the resident state's own frame
    (:func:`woof.io.wrfout.state_frame` with the diagnostic pressure, the writer's call), plus the grid metadata."""
    if streamed_fields is not None:
        source = {str(k).upper(): v for k, v in dict(streamed_fields).items()}
    else:
        from woof.io.wrfout import state_frame
        source = state_frame(state, include_diagnostic_pressure=True)
    for name, value in (metadata or {}).items():
        source.setdefault(str(name).upper(), value)
    frame = {}
    for name in SOUNDING_FIELDS + SURFACE_FIELDS:
        if source.get(name) is not None:
            array = _host(source[name])
            if (name in _SURFACE_GRID or name in _SURFACE_2D) and array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            frame[name] = array
    missing = [n for n in SOUNDING_FIELDS if n not in frame and n not in ("COSALPHA", "SINALPHA")]
    if missing:
        raise ValueError(f"member frame has no {', '.join(missing)} for a column store")
    return frame


class MemberColumns:
    """Each ensemble member's any-point column store, written as its frames land.

    An ensemble keeps no member history files by default, so the deterministic path (a committed wrfout handed to
    ``rw_soundings --columns``) has nothing to read. The member's frame is cut to the fields a sounding needs while
    the ordinary writer's callback holds it (:func:`sounding_frame`), written as a small WRF-shaped frame under
    ``<root>/columns/.frames``, turned into ``<root>/columns/members/<member>/<domain>/<YYYYMMDDTHHMMSSZ>.wcol`` by
    the same writer as a deterministic run's, and deleted. One serial queue on its own thread, bounded so a slow
    disk holds the members back instead of filling host memory; a frame that cannot be written is counted and
    said, never a failed run.

    Cadence: whole hours by default (``WOOF_COLUMNS_EVERY_MINUTES``), plus the run's last frame when it ends off
    the cadence and ``end_time`` is known. Readers form the mean and the P10-P90 band from the members' columns.
    """

    def __init__(self, root, *, end_time=None, minutes: int | None = None, threads: int | None = None,
                 writer: Callable[..., list] | None = None, depth: int = 4, say: Callable[[str], None] | None = None,
                 timeout: float | None = None, series: Callable[..., list] | None = None):
        self.root = Path(root)
        self.out = self.root / "columns"
        self.scratch = self.out / ".frames"
        self._end = end_time
        self._minutes = every_minutes() if minutes is None else int(minutes)
        self._threads = tile_threads() if threads is None else int(threads)
        self._timeout = frame_timeout_seconds() if timeout is None else float(timeout)
        self._writer = _default_writer(self._timeout) if writer is None else writer
        self._series_writer = (_default_series(self._timeout) if writer is None else None) if series is None else series
        self._series: list = []
        self._say = say or (lambda line: print(line, flush=True))
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(depth)))
        self._lock = threading.Lock()
        self._written: list[dict[str, Any]] = []
        self._failed: list[dict[str, Any]] = []
        self._seconds = 0.0
        self._capture_seconds = 0.0
        self._stopped = False
        self._thread = threading.Thread(target=self._work, name="gpuwm-member-columns", daemon=True)
        self._thread.start()

    def due(self, valid_time) -> bool:
        if self._minutes <= 0:
            return True
        on = (valid_time.hour * 60 + valid_time.minute) % self._minutes == 0 and valid_time.second == 0
        return on or (self._end is not None and valid_time >= self._end)

    def submit(self, *, member: str, domain: str, valid_time, state=None, streamed_fields=None,
               metadata=None) -> bool:
        """Cut and queue one member frame when it is due; False when it is not."""
        if not hasattr(valid_time, "hour"):
            from datetime import datetime
            valid_time = datetime.strptime(str(valid_time).replace("T", "_").rstrip("Z")[:19], "%Y-%m-%d_%H:%M:%S")
        if self._stopped or not self.due(valid_time):
            return False
        started = time.perf_counter()
        try:
            frame = sounding_frame(state=state, streamed_fields=streamed_fields, metadata=metadata)
        except Exception as error:  # a store never stops a member
            self._fail(member, domain, valid_time, error)
            return False
        with self._lock:
            self._capture_seconds += time.perf_counter() - started
        self._queue.put((member, domain, valid_time, frame))
        return True

    def _fail(self, member, domain, valid_time, error) -> None:
        row = {"member": member, "domain": domain, "valid_time": str(valid_time),
               "error": f"{type(error).__name__}: {error}"[:400]}
        with self._lock:
            self._failed.append(row)
            first = len(self._failed) == 1
        if first:
            self._say(f"member columns: no store for {member} {domain} {valid_time}: {row['error']}")

    def _write(self, member: str, domain: str, valid_time, frame: dict) -> None:
        from woof.io.wrfout import WrfoutWriter
        started = time.perf_counter()
        stamp = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
        nz, ny, nx = frame["T"].shape
        directory = self.scratch / member
        directory.mkdir(parents=True, exist_ok=True)
        grid = re.match(r"d\d{2}", domain)
        path = directory / f"wrfout_{grid[0] if grid else 'd01'}_{stamp.replace(':', '_')}"
        rows = []
        try:
            # The process's one netCDF lock (woof.io.netcdf_serialization): the writer's publish check reopens
            # the frame with netCDF4, whose HDF5/netCDF-C is not thread-safe, beside the members' own per-domain
            # writers, which hold this lock. THE BREAKAGE: unguarded, a member's wrfout failed its inventory check
            # and this frame read "NetCDF: Not a valid ID" (2.9.7 R6, test_ensemble_member_release_gpu).
            from woof.io.netcdf_serialization import NETCDF4_IO_LOCK
            with NETCDF4_IO_LOCK:
                with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0, title="woof member columns",
                                  global_attrs={}, field_schema=frame) as w:
                    w.write_frame(stamp, frame)
            rows = self._writer(path, self.out, threads=self._threads, member=member, as_domain=domain)
        except Exception as error:  # a frame not written never stops the run
            self._fail(member, domain, valid_time, error)
            return
        finally:
            for leftover in directory.glob(path.name + "*"):
                leftover.unlink(missing_ok=True)
        seconds = time.perf_counter() - started
        with self._lock:
            self._seconds += seconds
            for r in rows:
                self._written.append({"member": member, "domain": domain, "path": r.get("path"),
                                      "bytes": int(r.get("bytes", 0))})

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._write(*item)

    def stop(self, timeout: float | None = None) -> dict:
        """Write what is queued, then say what was written. Bounded as :meth:`LiveColumns.stop` is: a writer still
        busy after each queued frame's :func:`frame_timeout_seconds` is said and left behind, never waited on."""
        if not self._stopped:
            self._stopped = True
            if timeout is None:
                timeout = self._timeout * (self._queue.qsize() + 1) + 30.0
            self._queue.put(None)
            self._thread.join(timeout)
            if self._thread.is_alive():
                self._say(f"member columns: the writer did not finish in {timeout:.0f} s; the run ends without "
                          "the frames still queued")
            else:
                import shutil
                shutil.rmtree(self.scratch, ignore_errors=True)
                if self._written:
                    self._series = _series(self._series_writer, self.out,
                                           lambda event, message, **_: self._say(f"member columns: {message}"))
            s = self.summary()
            self._say(f"member columns: {s['files']} files, {s['bytes'] / 1e6:.1f} MB, {s['members']} members, "
                      f"{s['seconds']:.1f} s writing, {s['capture_seconds']:.2f} s on the members' threads"
                      + (f", {len(s['failed'])} not written" if s["failed"] else "") + f" -> {self.out / 'members'}")
        return self.summary()

    def summary(self) -> dict:
        with self._lock:
            return {"schema": "gpuwm-member-columns.v1", "out": str(self.out / "members"),
                    "every_minutes": self._minutes, "files": len(self._written),
                    "members": len({r["member"] for r in self._written}),
                    "bytes": sum(r["bytes"] for r in self._written), "seconds": round(self._seconds, 3),
                    "capture_seconds": round(self._capture_seconds, 3), "failed": list(self._failed),
                    "series": list(self._series)}
