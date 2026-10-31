"""Lateral-boundary-condition files: the reader and the two-level pool.

The producer side of this seam is ``rw_mpas_lbc`` (and, for reference mints,
native ``init_atmosphere_model`` case 9): one CDF-5 file per boundary time,
each carrying FULL-MESH fields, ``lbc_u`` over every edge, the cell fields
over every cell, regardless of how few of those elements the consuming model
will read.  This module is the consumer's first half, a transcription of the
admission and timekeeping in ``mpas_atm_boundaries.F`` (MPAS v8.4.1):

* **admission**: the first admitted file is the LATEST file at-or-before
  the model time (``MPAS_STREAM_LATEST_BEFORE``); every subsequent one is the
  EARLIEST file STRICTLY after it (``MPAS_STREAM_EARLIEST_STRICTLY_AFTER``).
  A missing interval refuses by name: which rule, which model time, what the
  inventory actually holds.
* **the two-level pool**: level 2 holds the state read at the current
  interval's END; level 1 holds the TENDENCY ``(new - old) / dt`` computed in
  float32 exactly as the Fortran computes it (``dt`` formed in REAL(4), then
  reciprocal-multiplied).
* **time interpolation**, the state at model time ``t`` inside an interval
  is ``state(end) - (end - t) * tendency``: linear, anchored BACKWARD from
  the interval end, which is ``mpas_atm_get_bdy_state`` verbatim.

What this module deliberately does NOT do: derive the coupled fields
(``lbc_rho_zz = lbc_rho / zz``, edge-averaged ``lbc_rho_edge``, ``lbc_ru``,
``lbc_rtheta_m``) or touch the integration loop.  Those need mesh state and
live in :class:`woof.hex.regional_v841.RegionalDrivingState`, which wraps
this pool and computes them at every admission exactly as
``mpas_atm_update_bdy_tend`` does; the pool stays generic over field names
so that wrapper carries derived fields beside the file fields without this
module growing a mesh dependency.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .errors import MpasPortError

__all__ = [
    "LBC_REQUIRED_VARIABLES",
    "LBC_OPTIONAL_SCALAR_VARIABLES",
    "LbcAdmissionError",
    "LbcFile",
    "LbcFileError",
    "LbcInventory",
    "LbcPool",
    "read_lbc_file",
    "read_lbc_file_shared",
    "read_lbc_valid_time",
]

#: The v8.4.1 ``lbc`` stream under the ``lbcs`` package with the ``moist``
#: scalar group: variable name -> the dimension names its file slab carries.
#: Measured on the native case-9 reference files and pinned here; a file
#: missing one of these, or carrying it on other dimensions, is not an lbc
#: file this port understands.
LBC_REQUIRED_VARIABLES: Dict[str, Tuple[str, ...]] = {
    "lbc_qv": ("Time", "nCells", "nVertLevels"),
    "lbc_u": ("Time", "nEdges", "nVertLevels"),
    "lbc_w": ("Time", "nCells", "nVertLevelsP1"),
    "lbc_rho": ("Time", "nCells", "nVertLevels"),
    "lbc_theta": ("Time", "nCells", "nVertLevels"),
}

# MPAS-A v8.4.2 Registry.xml:2162-2196: additional allocated scalar
# packages. File presence is independent of the configured pool inventory.
LBC_OPTIONAL_SCALAR_VARIABLES: Dict[str, Tuple[str, ...]] = {
    "lbc_qc": ("Time", "nCells", "nVertLevels"),
    "lbc_qr": ("Time", "nCells", "nVertLevels"),
    "lbc_qi": ("Time", "nCells", "nVertLevels"),
    "lbc_qs": ("Time", "nCells", "nVertLevels"),
    "lbc_qg": ("Time", "nCells", "nVertLevels"),
    "lbc_ni": ("Time", "nCells", "nVertLevels"),
    "lbc_nr": ("Time", "nCells", "nVertLevels"),
    "lbc_nc": ("Time", "nCells", "nVertLevels"),
    "lbc_nifa": ("Time", "nCells", "nVertLevels"),
    "lbc_nwfa": ("Time", "nCells", "nVertLevels"),
    "lbc_tke": ("Time", "nCells", "nVertLevels"),
}

# MPAS-Model-rrfs Registry.xml:2500-2560 at 1d6e370e and 2659-2685 at 44c0849: the RRFS fork's lbc_scalars carry these too (NSSL 2-moment
# species, and TEMPO hail-aware graupel number and volume under tempo_hailaware_in).  Read only when the run's own
# scalar inventory names them (read_lbc_file ``extra_scalars``), so a v8.4.2 run reads the same fields as before.
# THE BREAKAGE THIS FIXES: hrrrv5 (TEMPO hail-aware) on a limited-area mesh refused at load with "undeclared
# boundary scalars: ['lbc_ng', 'lbc_volg']", and had it run, native's ng/volg boundary forcing would have been zero.
LBC_FORK_SCALAR_VARIABLES: Dict[str, Tuple[str, ...]] = {
    name: ("Time", "nCells", "nVertLevels")
    for name in ("lbc_qh", "lbc_ns", "lbc_ng", "lbc_nh", "lbc_nccn", "lbc_volg", "lbc_volh",
                 # registry.chemistry.xml lbc_scalars: smoke/dust (RRFS-SD) boundary species. Breakage it
                 # prevents: every limited-area RRFS run with smoke on refused "undeclared boundary scalars".
                 "lbc_smoke_fine", "lbc_dust_fine", "lbc_dust_coarse", "lbc_unspc_fine")
}

_XTIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


class LbcFileError(MpasPortError):
    """An lbc file is missing something the stream contract promises."""


class LbcAdmissionError(MpasPortError):
    """No file in the inventory satisfies the admission rule being applied."""


def _parse_xtime(raw: object, path: Path) -> datetime:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", "replace").strip()
    else:
        text = str(raw).strip()
    try:
        return datetime.strptime(text[:19], _XTIME_FORMAT)
    except ValueError as error:
        raise LbcFileError(
            f"{path} carries xtime {text!r}, which is not YYYY-MM-DD_HH:MM:SS; "
            "without a parseable valid time the file cannot be placed on the "
            "boundary timeline at all"
        ) from error


def _open_lbc(path: Path):
    """Open one boundary file through the engine's Rust NetCDF reader.

    ``woof.netcdf_bridge`` drives ``rw_netcdf``: no Python NetCDF library
    touches these bytes (the 2.5.0 Python boundary), and nothing here holds a
    native library object, so a same-host rank worker may read its own
    boundary files.  The netCDF4-python reader this replaced had to stay on
    the forecast coordinator under the free-threaded serial-I/O guard, which
    is why the door's regional N-rank path refused at step 1 with
    SerialExtensionIoError (rrfs-h-h3 skeptic finding, 2026-10-05).
    """

    from woof.netcdf_bridge import open_dataset

    return open_dataset(path)


def _read_xtime(dataset, path: Path) -> datetime:
    if "xtime" not in dataset.variables:
        raise LbcFileError(
            f"{path} has no xtime variable; an lbc file with no valid time "
            "cannot be admitted against any model time"
        )
    variable = dataset.variables["xtime"]
    variable.set_auto_mask(False)
    slab = np.asarray(variable[0])
    if slab.dtype.kind == "U":
        raw = "".join(str(item) for item in slab.reshape(-1)).encode("utf-8")
    elif slab.dtype.kind == "S":
        raw = slab.tobytes()
    elif slab.dtype.kind == "O":
        raw = "".join(
            item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item)
            for item in slab.reshape(-1)
        ).encode("utf-8")
    else:
        raw = bytes(slab)
    return _parse_xtime(raw, path)


def read_lbc_valid_time(path: str | Path) -> datetime:
    """The file's valid time, from its own ``xtime``, never the filename."""

    p = Path(path)
    return _read_xtime(_open_lbc(p), p)


@dataclass(frozen=True)
class LbcFile:
    """One boundary time, fields squeezed of their record axis, float32."""

    path: Path
    valid_time: datetime
    fields: Mapping[str, np.ndarray]

    @property
    def n_cells(self) -> int:
        return int(self.fields["lbc_theta"].shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.fields["lbc_u"].shape[0])

    @property
    def n_vert_levels(self) -> int:
        return int(self.fields["lbc_theta"].shape[1])


def _classic_float32_words(path: Path, dataset, schema: Mapping[str, Tuple[str, ...]],
                           rows=None) -> Dict[str, np.ndarray]:
    """Record 0 of every schema variable a classic file stores as whole float32 words, read on host threads.

    THE COST THIS REMOVES (mv3km 3 km limited area, 206,726 cells, V6000 RTX PRO 6000, 2026-10-28): each
    1.17 GB boundary file was decoded one variable per ``rw_netcdf`` process, widened to float64 in a
    temporary file and narrowed back, about 9 s on one core per file.  The first step admits two files
    (18-21 s of its 25 s while the card waited) and every boundary interval after it one more.  A CDF-5
    boundary file (``rw_mpas_lbc`` writes CDF-5) stores each variable's big-endian words at the offset
    its header names; ``woof.hex.classic_netcdf`` (the reader the mesh load uses) reads them on host
    threads, 0.2 s per file.  The words are the file's own float32 words, the bytes the bridge path
    returned (bit-compared on all 17 mv3km boundary variables).

    The bridge's inventory still supplies the names, dimensions and dtypes the caller checks; a
    variable this reader does not return the same way (non-classic file, packed or unsigned attributes,
    a file of other than one record, a shape or dtype the inventory disagrees with) is absent from the
    result and read through the bridge as before.  ``WOOF_HEX_CLASSIC_READ=off`` turns it off.

    ``rows`` (``name -> (indexes, global count)``, see ``read_lbc_file``): only those rows of each variable are read
    (``classic_netcdf.read_rows``); the result holds the rows without the record axis.
    """

    import os
    import struct

    if os.environ.get("WOOF_HEX_CLASSIC_READ", "").strip().lower() in ("off", "0", "no"):
        return {}
    from .classic_netcdf import extents, read_variables, readable
    from .host_parallel import threads
    try:
        found = extents(path)
    except (OSError, ValueError, KeyError, struct.error):
        return {}
    if found is None:
        return {}
    table, numrecs = found
    selected = []
    for name in schema:
        variable = dataset.variables.get(name)
        extent = table.get(name)
        if (variable is None or extent is None or not extent.record or numrecs != 1
                or not readable(extent, numrecs, variable.ncattrs())
                or tuple(extent.shape) != tuple(variable.shape)
                or np.dtype(extent.dtype).newbyteorder("=") != np.float32
                or variable.dtype != np.float32):
            continue
        selected.append(name)
    result: Dict[str, np.ndarray] = {}
    if rows is not None:
        from .classic_netcdf import read_rows
        selection = {}
        for name in selected:
            indexes, expected = rows(name)
            if table[name].shape[1] != expected:
                # project_lbc_file's refusal, raised before any word is read.
                raise ValueError(f"{path}: {name} has {table[name].shape[1]} entities, expected {expected}")
            selection[name] = indexes
        for name, data in read_rows(path, selected, table, selection, workers=threads()).items():
            data.setflags(write=False)
            result[name] = data
        return result
    words = read_variables(path, selected, table, workers=threads())
    for name, value in words.items():
        data = value[0]
        data.setflags(write=False)
        result[name] = data
    return result


def read_lbc_file(path: str | Path, *, extra_scalars: Sequence[str] = (), rows=None) -> LbcFile:
    """Read and validate one lbc file.

    Refusals name the missing piece: an absent variable, a variable on the
    wrong dimensions, or a non-float32 payload.  Each would otherwise surface
    as a shape error several frames inside the driver, long after the wrong
    file was accepted.  ``extra_scalars``: fork-only boundary scalars
    (LBC_FORK_SCALAR_VARIABLES) the run carries, read when the file has them.

    ``rows``: ``name -> (indexes, global count)`` for a same-host rank's own rows (partition_regional_v841
    ``layout_rows``).  Every field then holds only those rows, the words ``project_lbc_file`` cuts from the whole
    file, and the whole file is never held: classic float32 variables are gathered from the page cache, any other
    variable is decoded and cut at once.
    """

    p = Path(path)
    dataset = _open_lbc(p)
    # Raw bytes, not the mask-and-scale view: the tendency arithmetic is
    # pinned to the reference's float32, so the values must be the file's.
    dataset.set_auto_maskandscale(False)
    valid_time = _read_xtime(dataset, p)
    fields: Dict[str, np.ndarray] = {}
    variable_schema = dict(LBC_REQUIRED_VARIABLES)
    variable_schema.update({
        name: dims for name, dims in LBC_OPTIONAL_SCALAR_VARIABLES.items()
        if name in dataset.variables
    })
    variable_schema.update({
        name: LBC_FORK_SCALAR_VARIABLES[name] for name in extra_scalars
        if name in LBC_FORK_SCALAR_VARIABLES and name in dataset.variables
    })
    stored = _classic_float32_words(p, dataset, variable_schema, rows)
    for name, want_dims in variable_schema.items():
        if name not in dataset.variables:
            raise LbcFileError(
                f"{p} carries no {name}; the v8.4.1 lbc stream always "
                f"writes it, so this is not a complete lbc file and "
                f"admitting it would leave that field frozen at whatever "
                f"the previous interval held"
            )
        variable = dataset.variables[name]
        got_dims = tuple(variable.dimensions)
        if got_dims != want_dims:
            raise LbcFileError(
                f"{p} carries {name} on dimensions {got_dims}, not "
                f"{want_dims}; reading it anyway would transpose the "
                f"field into plausible-looking wrong numbers"
            )
        if variable.dtype != np.float32:
            raise LbcFileError(
                f"{p} carries {name} as {variable.dtype}, not float32; "
                f"the reference stream is single precision and a silent "
                f"widening here would make every tendency differ from "
                f"the reference arithmetic"
            )
        if name in stored:
            # The file's own float32 words, record 0 (see _classic_float32_words).
            fields[name] = stored.pop(name)
            continue
        decoded = np.asarray(variable[0])
        # The bridge may hand float32 back widened; float32 -> float64 ->
        # float32 is exact, but anything else is not the file's bytes.
        data = np.array(decoded, dtype=np.float32)
        if decoded.dtype != np.float32 and not np.array_equal(
            data.astype(decoded.dtype), decoded, equal_nan=True
        ):
            raise LbcFileError(
                f"{p}: {name} came back from the Rust reader as "
                f"{decoded.dtype} values that are not float32; the file's own "
                f"bytes are what the tendency arithmetic is pinned to"
            )
        if rows is not None:
            indexes, expected = rows(name)
            if data.shape[0] != expected:
                raise ValueError(f"{p}: {name} has {data.shape[0]} entities, expected {expected}")
            data = np.ascontiguousarray(data[indexes])
        data.setflags(write=False)
        fields[name] = data
    return LbcFile(path=p, valid_time=valid_time, fields=fields)


_SHARED_LBC_LOCK = threading.Lock()
_SHARED_LBC: "weakref.WeakValueDictionary[tuple, LbcFile]" = weakref.WeakValueDictionary()
_SHARED_LBC_GATES: Dict[tuple, threading.Lock] = {}


def read_lbc_file_shared(path: str | Path, *, extra_scalars: Sequence[str] = ()) -> LbcFile:
    """``read_lbc_file``, decoded once per process while any holder keeps it.

    The breakage: same-host rank threads admit the same boundary file at the
    same clock, and each read and kept its own whole copy before
    ``PartitionLbcPoolV841`` sliced its rows. On a 1,252,549-cell cut of NOAA's
    RRFS v2 conus3km on 8 ranks (VRFS, 2026-10-28) the forecast process grew to
    369 GB host RSS during the first step, about 2 GB/s for two minutes while the
    cards idled, from 8 ranks x (7.1 GB file + decode copy) per boundary time.
    The fields are read-only already, so every holder can share one decoded
    file. Values are the reader's own, so no output word can change. Entries are
    weak: the file is freed when its last holder lets go, so a one-rank run
    keeps exactly what it kept before.
    """

    p = Path(path)
    try:
        status = p.stat()
        key = (str(p.resolve()), status.st_size, status.st_mtime_ns, tuple(extra_scalars))
    except OSError:
        return read_lbc_file(p, extra_scalars=extra_scalars)
    with _SHARED_LBC_LOCK:
        hit = _SHARED_LBC.get(key)
        if hit is not None:
            return hit
        gate = _SHARED_LBC_GATES.setdefault(key, threading.Lock())
    with gate:
        with _SHARED_LBC_LOCK:
            hit = _SHARED_LBC.get(key)
            if hit is not None:
                return hit
        admitted = read_lbc_file(p, extra_scalars=extra_scalars)
        with _SHARED_LBC_LOCK:
            _SHARED_LBC[key] = admitted
            _SHARED_LBC_GATES.pop(key, None)
        return admitted


class LbcInventory:
    """The boundary files available to a run, indexed by their own xtime.

    The inventory holds ``(valid_time, path)`` and applies the two stream-
    manager admission rules.  It reads only each file's ``xtime`` at
    construction; field payloads are read on admission.
    """

    def __init__(self, paths: Sequence[str | Path]):
        from .lbc_stream import series_document, initial_paths
        self._series_dir = Path(paths[0]).parent if paths else None
        self._series = series_document(self._series_dir) if self._series_dir is not None else None
        if self._series is not None:
            paths = initial_paths(self._series_dir)
        if not paths:
            raise LbcAdmissionError(
                "the lbc inventory is empty; a regional run with no boundary "
                "files has nothing to relax toward and would integrate on a "
                "frozen boundary"
            )
        stamped: list[Tuple[datetime, Path]] = []
        for path in paths:
            p = Path(path)
            stamped.append((read_lbc_valid_time(p), p))
        stamped.sort(key=lambda pair: pair[0])
        for (t1, p1), (t2, p2) in zip(stamped, stamped[1:]):
            if t1 == t2:
                raise LbcAdmissionError(
                    f"two lbc files carry the same valid time "
                    f"{t1.strftime(_XTIME_FORMAT)}: {p1} and {p2}; the "
                    f"admission rules cannot order them and picking one "
                    f"silently would make the run depend on directory order"
                )
        self._entries: Tuple[Tuple[datetime, Path], ...] = tuple(stamped)

    @property
    def valid_times(self) -> Tuple[datetime, ...]:
        return tuple(t for t, _ in self._entries)

    @property
    def identity(self) -> Tuple[str, ...]:
        """The series this inventory admits from: its directory (a live series) and its files, as strings.

        Two inventories with the same identity admit the same files, so consumers in one process may share reads
        (partition_regional_v841.SharedLbcReads)."""
        return (str(self._series_dir),) + tuple(str(p) for _, p in self._entries)

    def _timeline(self) -> str:
        if self._series is not None:
            return ", ".join(self._series["valid_times"])
        return ", ".join(t.strftime(_XTIME_FORMAT) for t, _ in self._entries)

    def latest_before(self, when: datetime) -> Path:
        """``MPAS_STREAM_LATEST_BEFORE``: the latest file at-or-before ``when``."""

        if self._series is not None:
            from .lbc_stream import wait_file, FORMAT
            scheduled = [item for item in self._series["valid_times"]
                         if datetime.strptime(item, FORMAT) <= when]
            if scheduled:
                path = wait_file(self._series_dir, scheduled[-1], stage="prep.lbc.wait")
                if read_lbc_valid_time(path) != datetime.strptime(scheduled[-1], FORMAT):
                    raise LbcAdmissionError(f"{path}: xtime differs from its hash-checked marker; refuses wrong-time forcing")
                return path

        candidates = [(t, p) for t, p in self._entries if t <= when]
        if not candidates:
            raise LbcAdmissionError(
                f"no lbc file is valid at or before "
                f"{when.strftime(_XTIME_FORMAT)} (LATEST_BEFORE); the "
                f"inventory holds [{self._timeline()}].  This is the native "
                f"'Could not read from lbc_in stream on or before the "
                f"current date' failure, refused before the run starts "
                f"instead of inside it"
            )
        return candidates[-1][1]

    def earliest_strictly_after(self, when: datetime) -> Path:
        """``MPAS_STREAM_EARLIEST_STRICTLY_AFTER``: the earliest file after ``when``."""

        if self._series is not None:
            from .lbc_stream import wait_file, FORMAT
            scheduled = [item for item in self._series["valid_times"]
                         if datetime.strptime(item, FORMAT) > when]
            if scheduled:
                path = wait_file(self._series_dir, scheduled[0])
                actual = read_lbc_valid_time(path)
                if actual != datetime.strptime(scheduled[0], FORMAT):
                    raise LbcAdmissionError(f"{path}: xtime differs from its hash-checked marker; refuses wrong-time forcing")
                return path

        candidates = [(t, p) for t, p in self._entries if t > when]
        if not candidates:
            raise LbcAdmissionError(
                f"no lbc file is valid strictly after "
                f"{when.strftime(_XTIME_FORMAT)} (EARLIEST_STRICTLY_AFTER); "
                f"the inventory holds [{self._timeline()}].  The interval "
                f"the model is about to integrate has no far end, and "
                f"integrating past the last boundary file would freeze the "
                f"boundary at its final state without saying so"
            )
        return candidates[0][1]


def rkind_interval_seconds(delta: timedelta) -> np.float32:
    """Clock interval in RKIND, including fractional seconds.

    mpas_atm_boundaries.F:267-269,496-497,618-619 converts days, seconds
    and the fractional numerator/denominator separately, then adds left to
    right. Python's datetime clock carries its fraction in microseconds.
    """
    return np.float32(
        (np.float32(86400.0) * np.float32(delta.days) + np.float32(delta.seconds))
        + np.float32(delta.microseconds) / np.float32(1000000.0)
    )


def _interval_seconds_f32(start: datetime, end: datetime) -> np.float32:
    """MPAS clock conversion shared by raw and coupled LBC pools."""
    return rkind_interval_seconds(end - start)


class LbcPool:
    """The two-level value/tendency pool of ``mpas_atm_boundaries.F``.

    Level 2 holds the field values read at the CURRENT interval's end;
    level 1 holds the tendencies over the current interval.  ``start`` is
    ``mpas_atm_update_bdy_tend(firstCall=.true.)``; ``advance`` is every
    later call: shift, read the next file, form tendencies.
    """

    def __init__(
        self, inventory: LbcInventory, *, scalar_names: Sequence[str] = (),
        compute_tendencies: bool = True, reader=None,
    ):
        # MPAS-A v8.4.2 Registry.xml:2150-2196 declares the active array.
        # gen_inc.c:615-620 and mpas_field_routines.F:383-395 initialize it.
        self._compute_tendencies = bool(compute_tendencies)
        self._scalar_names = tuple(scalar_names)
        # The RRFS fork's Registry declares more (LBC_FORK_SCALAR_VARIABLES); a run names them only when its own
        # scalar inventory (the fork Registry's) carries those species.
        declared = {'lbc_qv'} | set(LBC_OPTIONAL_SCALAR_VARIABLES) | set(LBC_FORK_SCALAR_VARIABLES)
        unknown = set(self._scalar_names) - declared
        if unknown:
            raise LbcFileError(f"undeclared boundary scalars: {sorted(unknown)}")
        self._recycled: Dict[str, np.ndarray] = {}
        self._inventory = inventory
        # ``reader(path, extra_scalars)`` returns the admitted file; None reads it here (read_lbc_file_shared: one
        # decoded file per process while any holder keeps it, the reader's own words).  A same-host
        # rank installs partition_regional_v841's shared reader, which returns the rank's own rows of a file read once
        # per process and reads the next file ahead.  Either way the values are the file's own words.
        self.reader = reader
        self._state: LbcFile | None = None
        self._tendencies: Dict[str, np.ndarray] | None = None
        self._interval_start: datetime | None = None

    @property
    def inventory(self) -> LbcInventory:
        return self._inventory

    @property
    def scalar_names(self) -> Tuple[str, ...]:
        return self._scalar_names

    def _read(self, path: Path) -> LbcFile:
        if self.reader is None:
            return read_lbc_file_shared(path, extra_scalars=self._scalar_names)
        return self.reader(path, extra_scalars=self._scalar_names)

    # -- admission ---------------------------------------------------------

    def start(self, when: datetime) -> LbcFile:
        """Admit the latest file at-or-before ``when`` into level 2."""

        admitted = self._read(self._inventory.latest_before(when))
        if not self._compute_tendencies:
            self._state = admitted
            return admitted
        fields = dict(admitted.fields)
        for name in self._scalar_names:
            if name not in fields:
                fields[name] = np.zeros_like(fields['lbc_qv'])
        self._recycled = {
            name: np.zeros_like(value) for name, value in fields.items()
        }
        admitted = LbcFile(admitted.path, admitted.valid_time, fields)
        self._state = admitted
        self._tendencies = None
        self._interval_start = None
        return admitted

    def advance(self, when: datetime | None = None) -> LbcFile:
        """Shift level 2 to level 1, admit the next file, form tendencies.

        ``when`` defaults to the current interval end, which is what the
        native driver passes: the clock sits at the boundary it just
        reached.  The tendency arithmetic is float32 throughout, matching
        the reference: ``dt`` in REAL(4), inverted once, multiplied through.
        """

        if self._state is None:
            raise LbcAdmissionError(
                "advance was called on a pool that never started; the first "
                "admission is LATEST_BEFORE the run start and must happen "
                "before any interval can have two ends"
            )
        old = self._state
        clock = when if when is not None else old.valid_time
        admitted = self._read(self._inventory.earliest_strictly_after(clock))
        if not self._compute_tendencies:
            # CUDA consumers retain numerical pool storage on the device.
            self._interval_start = old.valid_time
            self._state = admitted
            return admitted
        # MPAS-A v8.4.2 mpas_atm_boundaries.F:148-151 rotates levels;
        # mpas_field_routines.F:3335-3347/shift_time_levs_array.inc recycles
        # the previous tendency buffer. mpas_io.F:858-878 warns and leaves
        # that allocated storage untouched when an input variable is absent.
        # Never replace an absent subsequent variable with a fresh zero.
        old_fields = dict(old.fields)
        fields = dict(admitted.fields)
        for name in old_fields.keys() | fields.keys() | set(self._scalar_names):
            if name not in old_fields:
                old_fields[name] = np.zeros_like(old_fields['lbc_qv'])
            if name not in fields:
                fields[name] = (self._recycled[name] if name in self._recycled
                                else np.zeros_like(old_fields[name]))
        admitted = LbcFile(admitted.path, admitted.valid_time, fields)
        dt = _interval_seconds_f32(old.valid_time, admitted.valid_time)
        inv_dt = np.float32(np.float32(1.0) / dt)
        tendencies: Dict[str, np.ndarray] = {}
        for name in old_fields:
            tendency = (admitted.fields[name] - old_fields[name]) * inv_dt
            tendency = np.asarray(tendency, dtype=np.float32)
            tendency.setflags(write=False)
            tendencies[name] = tendency
        self._interval_start = old.valid_time
        self._state = admitted
        self._tendencies = tendencies
        self._recycled = tendencies
        return admitted

    # -- what the pool holds ----------------------------------------------

    @property
    def interval_start(self) -> datetime:
        if self._interval_start is None:
            raise LbcAdmissionError(
                "the pool holds no complete interval yet; start then advance "
                "must both have run before an interval has two ends"
            )
        return self._interval_start

    @property
    def interval_end(self) -> datetime:
        if self._state is None:
            raise LbcAdmissionError("the pool has not started")
        return self._state.valid_time

    def tendency(self, name: str) -> np.ndarray:
        """The level-1 tendency for ``name``, per second."""

        self._require_field(name)
        if self._tendencies is None:
            raise LbcAdmissionError(
                f"the pool holds no tendencies yet, so the {name} tendency "
                f"does not exist; advance must admit the far end of the "
                f"first interval before any tendency is defined"
            )
        return self._tendencies[name]

    def state_at(self, name: str, when: datetime) -> np.ndarray:
        """The field at model time ``when``: linear, backward from interval end.

        ``mpas_atm_get_bdy_state`` verbatim: ``state(end) - dt * tend`` with
        ``dt = seconds(end - when)`` formed in float32.
        """

        self._require_field(name)
        if self._state is None or self._tendencies is None:
            raise LbcAdmissionError(
                f"state_at({name}) was asked of a pool with no complete "
                f"interval; start admits one end and advance the other, and "
                f"both are needed before boundary state exists at any time"
            )
        end = self._state.valid_time
        if when <= end:
            dt = _interval_seconds_f32(when, end)
        else:
            # Past the interval end the same linear form extrapolates
            # forward; the native driver never asks for this (it advances
            # first), and a consumer that does gets the arithmetic the
            # formula defines rather than a silent clamp.
            dt = np.float32(-_interval_seconds_f32(end, when))
        return np.asarray(
            self._state.fields[name] - dt * self._tendencies[name],
            dtype=np.float32,
        )

    def _require_field(self, name: str) -> None:
        schema = {**LBC_REQUIRED_VARIABLES, **LBC_OPTIONAL_SCALAR_VARIABLES, **LBC_FORK_SCALAR_VARIABLES}
        if name not in schema:
            raise LbcFileError(
                f"{name} is not an lbc stream field; the stream carries "
                f"required or optional fields {sorted(schema)}.  Derived coupled "
                f"fields (lbc_rho_zz, lbc_ru, lbc_rho_edge, lbc_rtheta_m) "
                f"are computed by "
                f"woof.hex.regional_v841.RegionalDrivingState, which is "
                f"the pool the regional driver consumes"
            )
        if self._state is not None and name not in self._state.fields:
            raise LbcFileError(
                f"the admitted file {self._state.path} carries no {name}; "
                "no boundary state or tendency exists for an absent species"
            )
