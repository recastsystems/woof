"""F000: the start state's surface diagnostics, from the forecast's own first physics call.

THE BREAKAGE THIS PREVENTS.  A start frame is written before any physics has run, so
every surface-layer and boundary-layer diagnostic in it is the init's value or the
seam's pre-first-call placeholder.  A start file that carries no 10 m wind (the HRRR
specific-humidity road writes ``u10``/``v10`` as the -1e30 missing value in every
cell) therefore published f000 with no 10 m wind: every f000 point sounding showed
"0-6 km shear NaN kt" and n/a helicity, the meteogram had no f000 barb, and the f000
10 m wind maps were empty (graded-plateau 1 km region, 2026-10-24, also on stock
0.4.4).  The fluxes and the boundary-layer depth (``hfx``, ``qfx``, ``lh``, ``pblh``)
were exact zero at f000 on every start.

THE FIX.  The forecast's first physics call runs at the start of step 1 on the start
state (MPAS-A's physics_driver order: surface layer, land surface and PBL before the
dynamics).  Its surface-layer output IS the scheme's diagnosis of the start state, by
the same scheme the forecast runs.  The drivers hold the f000 frame until step 1 has
run, read those fields once, and complete the frame:

* analysis fields (10 m wind, 2 m temperature): the start file's own value where it
  carries one; the first call's diagnosis in every cell where it does not
  (non-finite, or a missing-value magnitude);
* 2 m humidity: the first call's when the start file carried none;
* fluxes and PBL depth: the first call's, replacing the pre-first-call exact zero.

The model state is untouched; only the f000 history frame changes.  A run with no
step (``--hours 0``), a refused first step or a resumed start keeps the old frame and
says so in its receipt.  The MPAS suite route applies this in correct mode only:
exact mode writes native MPAS-A's f000 bit for bit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import threading
from typing import Any

import numpy as np

#: A value at or above this magnitude is MPAS's missing value (-1e30), not a wind or a
#: temperature.
MISSING_MAGNITUDE = 1.0e20


def missing_cells(value: Any) -> np.ndarray:
    """Cells that carry no value: non-finite or a missing-value magnitude."""

    array = np.asarray(value)
    with np.errstate(invalid="ignore"):
        return ~np.isfinite(array) | (np.abs(array) >= MISSING_MAGNITUDE)


def _exact_zero(value: Any) -> bool:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.itemsize == 4:
        return bool(np.all(array.view(np.uint32) == np.uint32(0)))
    return bool(np.all(array == 0))


def complete_start_frame(
    arrays: dict[str, Any],
    first_call: Mapping[str, Any] | None,
    *,
    analysis: Sequence[str],
    placeholders: Sequence[str],
    humidity: str | None = None,
    humidity_carried: bool | None = None,
) -> dict[str, Any]:
    """Complete one f000 frame in place from the first call's fields; return the record.

    ``analysis``: replaced cell by cell where the frame carries no value.
    ``placeholders``: replaced when the frame holds the exact-zero placeholder.
    ``humidity``: replaced when ``humidity_carried`` is False (None: when the frame's
    field is exact zero everywhere).
    """

    if first_call is None:
        return {"applied": False,
                "reason": "no first physics call ran before the frame was written "
                          "(no step, a refused first step, or a resumed start)"}
    record: dict[str, Any] = {
        "applied": True,
        "source": "the forecast's first physics call (step 1, on the start state)",
        "scope": "f000 history frame only; the model state is unchanged",
        "fields": {},
    }

    def take(name: str, how: str, mask: np.ndarray | None) -> None:
        start = np.asarray(arrays[name])
        value = np.asarray(first_call[name])
        if value.shape != start.shape:
            raise ValueError(f"f000 {name}: first call {value.shape} != frame {start.shape}")
        value = value.astype(start.dtype, copy=False)
        if mask is None:
            out = np.array(value, copy=True, order="C")
            cells = int(start.size)
        else:
            out = np.ascontiguousarray(np.where(mask, value, start), dtype=start.dtype)
            cells = int(np.count_nonzero(mask))
        left = int(np.count_nonzero(missing_cells(out)))
        arrays[name] = out
        record["fields"][name] = {
            "how": how, "cells_replaced": cells, "cells_without_value_after": left,
            "minimum": float(np.nanmin(out)) if out.size else None,
            "maximum": float(np.nanmax(out)) if out.size else None,
        }

    for name in analysis:
        if name not in arrays or name not in first_call:
            continue
        if _exact_zero(arrays[name]):
            # Exact zero in every cell is the init's placeholder, not an analysis (an IFS start
            # writes t2m = 0 K everywhere; run_2de68ba40c93de79's f000 2 m temperature was 0 K).
            take(name, "the start file carries only the exact-zero placeholder", None)
            continue
        mask = missing_cells(arrays[name])
        if mask.any():
            take(name, "the start file carries no value in these cells", mask)
    if humidity is not None and humidity in arrays and humidity in first_call:
        carried = humidity_carried
        if carried is None:
            carried = not _exact_zero(arrays[humidity])
        if not carried:
            take(humidity, "the start file carries no 2 m humidity", None)
        else:
            # A start file can carry q2 as non-finite or the missing value in some or all cells
            # (an IFS start writes q2 as NaN in every cell). Those cells take the first call's
            # diagnosis, as the analysis fields do. Breakage it prevents: hex global ensemble
            # run_2de68ba40c93de79 (IFS 2026-10-26 00Z, MPAS physics) published an all-NaN f000
            # q2, so the products pass refused the member ("history q2 is not finite").
            mask = missing_cells(arrays[humidity])
            if mask.any():
                take(humidity, "the start file carries no 2 m humidity in these cells", mask)
    for name in placeholders:
        if name in arrays and name in first_call and _exact_zero(arrays[name]):
            take(name, "pre-first-call placeholder (exact zero)", None)
    record["unchanged"] = sorted(
        name for name in (*analysis, *placeholders, *((humidity,) if humidity else ()))
        if name in arrays and name not in record["fields"])
    return record


class FirstCallHandoff:
    """One-shot hand-off of the first call's fields to the thread finishing the f000 frame.

    ``put`` is called once by the step loop (with the fields, or None when no first call
    will come); ``take`` blocks until then.  The loop's ``finally`` calls ``put(None)``
    so a waiting writer never hangs on a run that ended early.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._value: Mapping[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def put(self, value: Mapping[str, Any] | None) -> None:
        if self._event.is_set():
            return
        self._value = value
        self._event.set()

    def take(self) -> Mapping[str, Any] | None:
        self._event.wait()
        return self._value


# ---------------------------------------------------------------------------
# ArWen route (the frozen v2 seam): fields and reader
# ---------------------------------------------------------------------------
#: Seam diagnostics the first call computes from the start state.
ARWEN_ANALYSIS = ("u10", "v10", "t2")
ARWEN_HUMIDITY = "q2"
ARWEN_PLACEHOLDERS = ("hfx", "qfx", "lh", "pblh")
ARWEN_FIELDS = (*ARWEN_ANALYSIS, ARWEN_HUMIDITY, *ARWEN_PLACEHOLDERS)


def read_arwen_first_call(backend: Any, cp: Any, *, solve_cells: int | None = None,
                          names: Sequence[str] = ARWEN_FIELDS) -> dict[str, np.ndarray]:
    """The seam's surface diagnostics at the committed boundary after step 1, on the host.

    One device synchronisation and a few nCells-sized copies, once per run.  A
    limited-area seam is one column wider than its domain; the extra column is
    stripped as the history capture strips it.
    """

    snapshot_of = getattr(backend, "history_diagnostic_snapshot", None) or backend.diagnostic_snapshot
    snapshot = snapshot_of()
    if not isinstance(snapshot, Mapping):
        snapshot = {group: getattr(snapshot, group) for group in
                    ("surface", "soil", "precipitation", "gwdo", "metadata", "receipt")
                    if hasattr(snapshot, group)}
    wanted = set(names)
    found: dict[str, Any] = {}

    def walk(value: Any, leaf: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, str(key))
        elif leaf in wanted and leaf not in found and hasattr(value, "shape"):
            found[leaf] = value

    walk(snapshot, "")
    cp.cuda.Device().synchronize()
    host: dict[str, np.ndarray] = {}
    for name, value in found.items():
        array = cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)
        array = np.array(array, copy=True, order="C")
        if solve_cells is not None and array.ndim >= 1 and int(array.shape[-1]) == int(solve_cells) + 1:
            array = np.ascontiguousarray(array[..., : int(solve_cells)])
        host[name] = array
    return host


def complete_arwen_start_frame(arrays: dict[str, Any], first_call: Mapping[str, Any] | None,
                               f000_completion: Mapping[str, Any] | None) -> dict[str, Any]:
    """The WOOF-ArWen route's f000 completion: q2 counts as carried only when the start file had it."""

    q2 = (f000_completion or {}).get("q2") or {}
    carried = None if not q2 else (q2.get("source") == "start file q2")
    return complete_start_frame(arrays, first_call, analysis=ARWEN_ANALYSIS,
                                placeholders=ARWEN_PLACEHOLDERS, humidity=ARWEN_HUMIDITY,
                                humidity_carried=carried)


# ---------------------------------------------------------------------------
# MPAS suite route (diag_physics pool names)
# ---------------------------------------------------------------------------
MPAS_ANALYSIS = ("u10", "v10", "t2m")
MPAS_HUMIDITY = "q2"
MPAS_PLACEHOLDERS = ("hfx", "qfx", "lh", "hpbl", "ust", "ustm", "th2m")
MPAS_FIELDS = (*MPAS_ANALYSIS, MPAS_HUMIDITY, *MPAS_PLACEHOLDERS)
