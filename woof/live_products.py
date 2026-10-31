"""Draw every committed frame, on every grid, while the forecast runs.

THE DEFECT THIS CLOSES: during a nested run (12 km with 3 km and 1 km
nests) the page showed the 12 km grid only.  The early render
(:mod:`woof.first_products`) draws the first root frame, and every other
picture, every nest's included, waited for the forecast to end.  Measured
on a 5070 Ti: the whole default product set for one output time takes
5.4 s at 12 km (170x170), 8.3 s at 3 km (336x336) and 11.2 s at 1 km
(408x402), while that three-grid stack takes about 170 s of wall per
model hour, so drawing each frame as it lands keeps up many times over.

How it runs, and why each part is shaped so:

- **Bounded concurrent frames.** Frames enter one queue in commit order.
  Up to three CPU render processes overlap
  import and drawing. They divide the native product workers and memory
  allowance between them, reserving one processor for the forecast.
  Smaller hosts, a container's CPU quota and a host whose free memory
  holds fewer frames draw fewer at a time, down to one.  Each frame is
  priced from its own grid (:func:`live_frame_peak_bytes`) and starts
  only when the frames already drawing leave room for it, so a large
  grid's frames draw one or two at a time where a small grid's overlap.
  Publication remains serialized.
- **Memory read at every admission.**  On top of that budget, which is
  read once at the start, a frame starts only while the host's memory
  read NOW, less what the frames drawing are still expected to grow and
  this frame's expected peak (learned from the renders that ran), leaves
  the model's protected floor free.  The oldest waiting frame always
  runs when it would run alone, so at worst frames are drawn one at a
  time and catch up after the forecast; none is dropped.  Frames are
  admitted in queue order.  ``WOOF_LIVE_RENDER_ADAPTIVE=0`` restores the
  one-shot budget exactly (:class:`LiveRenderMemorySettings`).
- **Never blocks the model.**  :meth:`LiveProducts.frame_committed`
  appends to a queue and returns; it is called on the wrfout writer's
  thread and never raises.
- **The finalize stage's own command, with at most ONE HOUR of earlier
  frames.**  Each frame is drawn by :func:`woof.go_cli.render_command`
  from the same plan dict the end of run uses.  A frame on a whole hour
  is drawn beside every frame of its grid since the previous whole-hour
  frame, which is the hour it closes, when the request holds a window
  (:func:`requests_windows`); every other frame, and every frame of a
  request with no window, is drawn alone.
  Measured on the 5070 Ti host at 1 km (414x402): the renderer's import
  costs about 6 s per frame and drawing a frame's 204 pictures about
  3.5 s, so the last frame of nine drawn with its eight predecessors as
  baselines took 62 s against 12 s alone, and a live render carrying its
  grid's history fell further behind with every frame.  One hour of
  baselines is a fixed cost however long the run (one import on an
  hourly grid, four on a 15-minute one), and it is what the windowed
  family (``qpf_1h``, ``qpf_total``, the 1 h maxima) needs: those were
  drawn only after the forecast ended while every frame was drawn alone
  ("windowed accumulations need more than one stored whole-hour
  frame").  The frames between the hours are not optional on a
  sub-hourly grid: the history writer resets ``UP_HELI_MAX`` and the
  other 1 h maxima at each write (:mod:`woof.core.uh_diag`), so a
  whole-hour frame alone holds only the last interval of its hour.  The
  engine decides which windows the hour closes, and blocks the rest by
  name (``qpf_6h``, run maxima: "missing stored hour(s)", or on a
  sub-hourly grid "unevenly spaced"), so a window longer than the hour
  is never drawn short, the 10 m wind maxima included.
  :func:`windowed_passes` names what finalize still draws over each
  grid's whole series, the longer windows of the frames drawn live, and
  a frame the live pass did not finish is drawn at the end beside its
  grid's whole series (:func:`baseline_frames`).
- **A request made only of windows is drawn where windows end.**  A
  grid's first frame and its frames between whole hours close no
  window, so such a request drew nothing there: each live render exited
  1 with no picture, and handed the same frames the end-of-run batch
  did too, which stopped ``woof go``'s render stage before the frames
  that close windows were drawn.  Those frames are drawn neither live
  nor in the batch, and stay the baselines of every window
  (:func:`windows_only`, :func:`closes_a_window`).
- **Nothing half-written is published.**  Each frame is drawn into a
  dot-prefixed scratch under the render folder and moved onto its final
  ``<domain>/<product>/<valid-day>/`` name with :func:`os.replace` once
  the subprocess exits, exactly as the early render publishes.
- **The end-of-run render fills only what is missing.**  Every published
  frame is recorded in ``live-products.json`` with its pictures'
  digests; :func:`published_frames` lets finalize skip a frame only when
  the frame is unchanged on disk and every picture it recorded is still
  there with its digest.  At finalize the queue is drained (each frame
  costs seconds and the history is already written), bounded by a wait
  after which whatever is left is drawn in the end-of-run batch.
- **The map record is merged, never replaced.**  Each frame's
  ``render-georef.json`` entries are folded into the render folder's
  manifest through :mod:`woof.render_georef`.
- **Every route that writes history arms it.**  The run-plan observer
  (:class:`woof.runplan.RunObserver`) and the downscaled child
  (:class:`woof.offline_child_run._ChildProgress`) build it from the
  plan their finalize stage renders, beside the early render and on the
  same render slots. A downscaled child used to draw its analysis frame early
  and every other frame at the end: a 250 m child's second frame landed
  at 10:40 and was drawn at 13:16.
- **A stopped run stops drawing.**  :meth:`LiveProducts.halt` is the
  stop: nothing queued is drawn, the render in flight is ended and
  publishes nothing, and it returns within a few seconds, because the
  desktop's own stop kills the run 5 s after asking.  A render that
  ended on an interrupt (exit 130, or killed by a signal) is a stopped
  render and publishes nothing either: its output is whatever the
  renderer had staged when it was told to stop.

Telemetry never fails a run: every failure here is a ``warning`` event
and the frame is left for finalize.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from woof.first_products import (FirstProducts, WorkerEnd, finished_pictures,
                                 render_was_stopped, _write_receipt)
from woof.render_layout import fs_path, iter_rendered

#: The record of every frame drawn while the forecast ran.
LIVE_PRODUCTS_RECEIPT = "live-products.json"
LIVE_PRODUCTS_SCHEMA = "gpuwm.live-products.v1"

#: Where a frame is drawn before it is published; dot-prefixed so
#: :func:`woof.render_layout.iter_rendered` never lists it.
_SCRATCH_NAME = ".live-products-scratch"

#: How long finalize waits for the frame in flight.  One frame of the
#: finest grid measured 11.2 s; this is the point at which a wedged
#: render stops holding a finished forecast.
DEFAULT_WAIT_SECONDS = 600.0

#: How long a stop waits for the worker once its render has been ended.
#: The desktop and the terminal kill a run 5 s after asking it to stop,
#: and the stopped run's banner and report are written after this.
HALT_WAIT_SECONDS = 3.0

#: Whether a render ended because it was told to stop; one predicate
#: for the early render and this one (:mod:`woof.first_products`).
_render_was_stopped = render_was_stopped


#: Where the cgroup file systems are mounted, and the file naming the
#: cgroup this process runs in on each hierarchy (``<id>:<controllers>:
#: <path>``; the cgroup v2 line is ``0::<path>``).  A cgroup v2
#: directory holds ``cpu.max`` (``<quota> <period>`` in microseconds, or
#: ``max <period>`` for none); a cgroup v1 ``cpu`` controller directory
#: holds ``cpu.cfs_quota_us`` (``-1`` for none) and ``cpu.cfs_period_us``.
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"

#: The part of one live frame render's peak resident memory that does
#: not grow with its grid: the ``woof render`` front door's own peak
#: plus the peak of the ``rw_wrfbatch`` it runs, each read from the
#: kernel's resource usage.  Measured on an eight-processor worker drawing
#: the default products of a 49-level GFS nest (12 km at 72x58, 3 km at
#: 144x112): 346 and 366 MiB for a whole hour of each grid at the
#: three-frame budget (two native workers); at the one-frame budget
#: (seven), 482 MiB for the 12 km analysis frame and 616 MiB for the 3 km
#: whole hour with its four quarter-hour baselines.  The front door is
#: 62 MiB of each.  Rounded up past the largest, since a figure that
#: under-states the cost is the failure the cap exists to prevent.  It is
#: the whole price of those small frames: :func:`live_frame_peak_bytes`
#: adds what a larger grid costs on top of it.
LIVE_FRAME_PEAK_BYTES = 640 * 1024 * 1024

#: What each cell (west_east x south_north x bottom_top) of the frame
#: being drawn adds to that peak, and what each cell of an earlier frame
#: imported beside it as a window baseline adds.  Measured the same way on
#: the same worker, drawing the default products of a 3 km GFS frame of
#: 880x704x55 (34,073,600 cells, a 3.16 GB history file): 7,181 MiB for a
#: frame alone, whether the budget was one frame (seven native workers)
#: or three (two), because the import, not the product workers, sets the
#: peak; 7,848 and 7,863 MiB for a whole hour beside its two earlier
#: frames at the three- and one-frame budgets.  That is 219 bytes per
#: cell of the frame once the 62 MiB front door is taken off, and 10.5
#: per cell of a baseline, each rounded up.
LIVE_FRAME_BYTES_PER_CELL = 220
LIVE_BASELINE_BYTES_PER_CELL = 12

#: The share of available memory the concurrent frames may plan to use.
#: The same half the renderer gives its own product workers
#: (``BUDGET_FRACTION`` in rustwx's ``batch_render.rs``): a render is a
#: guest beside the forecast, whose host memory grows after the frames
#: are budgeted.
_LIVE_MEMORY_FRACTION = 0.5


# -- memory admission, read at every render -----------------------------
#
# THE BREAKAGE: production NYC 500 m 12Z on 2026-10-05 died at forecast
# hour 46.7 of 48 on a 32 GiB, 8 vCPU host (RTX PRO 4500), with no
# traceback.  The live frame slots read the host's available memory ONCE,
# when the forecast's renderer was set up, and admitted up to three
# renders against half of that for the rest of the run, however much the
# model and the host's other work had taken since.  The cause of that
# death is not confirmed; renders competing with the model for memory
# late in the run is the leading suspect, and a budget that cannot see
# the memory it budgets is a defect either way.
#
# So each render is admitted against memory as it is NOW: the host's
# MemAvailable (capped by a container's cgroup room), read at every
# admission, less what the renders already running are still expected to
# grow, less this render's expected peak, must leave the model's
# protected floor free (:func:`live_render_keep_free`).  A render's peak
# is learned from the renders that ran (their process tree's peak
# resident memory, ``rw_wrfbatch`` included).  The oldest waiting render
# always runs when no other render is running: the forecast runner waits
# for its live renders before it ends the stage, so holding them all
# would stall the run.  At worst renders run one at a time, nothing is
# dropped, and what the live pass has not drawn when the forecast ends
# is drawn then (the drain, then finalize).  This changes only WHEN a
# render runs, never what the forecast writes.
#
# Every number below is settable through the environment (the name
# beside it) so a host or worker can tune it; an unreadable value keeps
# the default.  The thresholds and their reasons were measured for the
# site worker's version of this budget, which this replaces inside the
# engine.

#: The kill switch.  ``0``, ``off``, ``false`` or ``no`` restores the
#: one-shot budget exactly as it was before this admission existed: the
#: memory read once at the start, frames priced from their cells only.
LIVE_RENDER_ADAPTIVE_ENV = "WOOF_LIVE_RENDER_ADAPTIVE"

_GIB = 1 << 30
_MIB = 1 << 20


class LiveRenderMemorySettings:
    """The memory admission's thresholds, each with its environment name.

    ``peak_default`` (``GPUWM_LIVE_RENDER_PEAK_DEFAULT_BYTES``, 4 GiB):
    one render's peak before any render of this run has finished.  The
    cells price a city 500 m nest frame at 1 to 2 GiB and an 880x704x55
    frame at 7.0 GiB; 4 GiB covers the common case without letting a
    second render start on a 32 GiB host that has only the model's floor
    left.  A frame's own cell price (:func:`live_frame_peak_bytes`) is
    used when it is larger.

    ``peaks_kept`` (``GPUWM_LIVE_RENDER_PEAKS_KEPT``, 16): once renders
    have finished, a render is reckoned at the largest of this many most
    recent measured peaks (or its cell price, if larger).

    ``model_margin_fraction`` and ``model_margin_bytes``
    (``GPUWM_LIVE_RENDER_MODEL_MARGIN_FRACTION``, 0.25;
    ``GPUWM_LIVE_RENDER_MODEL_MARGIN_BYTES``, 1 GiB): the model's floor
    is its measured resident memory plus this fraction of it plus these
    bytes, room for the history writes and a nest's growth between reads.

    ``small_host_bytes`` and ``small_host_fraction``
    (``GPUWM_LIVE_RENDER_SMALL_HOST_BYTES``, 64 GiB;
    ``GPUWM_LIVE_RENDER_SMALL_HOST_FRACTION``, 0.40): on a host under
    this size the model's floor is at least this share of the host.
    12.8 GiB on the 32 GiB host NYC died on, twice the 6.43 GiB the
    engine priced that forecast at.

    ``system_reserve`` (``GPUWM_LIVE_RENDER_SYSTEM_RESERVE_BYTES``,
    2 GiB): kept free beyond the model's floor for the kernel, the page
    cache the history writes go through, and the host's other work.  The
    kernel's OOM killer acts before MemAvailable reaches zero.

    ``hysteresis`` (``GPUWM_LIVE_RENDER_HYSTERESIS_BYTES``, 512 MiB): a
    render that was already refused starts only with this much to spare,
    so one near the line does not flap between refused and admitted.

    ``poll_seconds`` (``GPUWM_LIVE_RENDER_POLL_SECONDS``, 1.0): how often
    a refused render reads memory again.  ``sample_seconds``
    (``GPUWM_LIVE_RENDER_SAMPLE_SECONDS``, 0.5): how often the running
    renders' process trees are measured while any runs.
    """

    FIELDS = {
        "peak_default": ("GPUWM_LIVE_RENDER_PEAK_DEFAULT_BYTES", 4 * _GIB, "bytes"),
        "peaks_kept": ("GPUWM_LIVE_RENDER_PEAKS_KEPT", 16, "count"),
        "model_margin_fraction": ("GPUWM_LIVE_RENDER_MODEL_MARGIN_FRACTION", 0.25, "fraction"),
        "model_margin_bytes": ("GPUWM_LIVE_RENDER_MODEL_MARGIN_BYTES", 1 * _GIB, "bytes"),
        "small_host_bytes": ("GPUWM_LIVE_RENDER_SMALL_HOST_BYTES", 64 * _GIB, "bytes"),
        "small_host_fraction": ("GPUWM_LIVE_RENDER_SMALL_HOST_FRACTION", 0.40, "fraction"),
        "system_reserve": ("GPUWM_LIVE_RENDER_SYSTEM_RESERVE_BYTES", 2 * _GIB, "bytes"),
        "hysteresis": ("GPUWM_LIVE_RENDER_HYSTERESIS_BYTES", 512 * _MIB, "bytes"),
        "poll_seconds": ("GPUWM_LIVE_RENDER_POLL_SECONDS", 1.0, "seconds"),
        "sample_seconds": ("GPUWM_LIVE_RENDER_SAMPLE_SECONDS", 0.5, "seconds"),
    }

    def __init__(self, **values: Any):
        for name, (_env, default, _kind) in self.FIELDS.items():
            setattr(self, name, values.pop(name, default))
        if values:
            raise TypeError(f"unknown setting(s): {', '.join(sorted(values))}")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None
                 ) -> "LiveRenderMemorySettings":
        environ = os.environ if environ is None else environ
        values = {}
        for name, (env, default, kind) in cls.FIELDS.items():
            parsed = _parse_setting(environ.get(env), kind)
            values[name] = default if parsed is None else parsed
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELDS}


_BYTE_SUFFIXES = {"": 1, "k": 1 << 10, "kib": 1 << 10, "m": _MIB, "mib": _MIB,
                  "g": _GIB, "gib": _GIB, "t": 1 << 40, "tib": 1 << 40}


def _parse_setting(text: str | None, kind: str):
    """One environment value, or ``None`` (keep the default) if unreadable.

    Bytes take a plain number or a binary suffix (``2G``, ``512MiB``).
    """

    if text is None or not str(text).strip():
        return None
    word = str(text).strip().lower()
    try:
        if kind == "bytes":
            match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([a-z]*)", word)
            if match is None or match.group(2) not in _BYTE_SUFFIXES:
                return None
            value = int(float(match.group(1)) * _BYTE_SUFFIXES[match.group(2)])
            return value if value >= 0 else None
        if kind == "count":
            value = int(word)
            return value if value >= 1 else None
        value = float(word)
    except ValueError:
        return None
    if kind == "fraction":
        return value if 0.0 <= value <= 1.0 else None
    return value if value > 0 else None


def live_render_adaptive(environ: Mapping[str, str] | None = None) -> bool:
    """Whether renders are admitted against memory read at each admission."""

    environ = os.environ if environ is None else environ
    text = str(environ.get(LIVE_RENDER_ADAPTIVE_ENV, "1")).strip().lower()
    return text not in ("0", "off", "false", "no")


def live_render_model_floor(model_rss: float, host_total: float | None,
                            settings: LiveRenderMemorySettings) -> float:
    """The memory kept for the model: its resident memory with a margin,
    and on a small host at least a share of the host."""

    floor = (model_rss * (1.0 + settings.model_margin_fraction)
             + settings.model_margin_bytes)
    if host_total and host_total < settings.small_host_bytes:
        floor = max(floor, host_total * settings.small_host_fraction)
    return floor


def live_render_keep_free(model_rss: float, host_total: float | None,
                          settings: LiveRenderMemorySettings) -> float:
    """What must stay available while renders run: the part of the model's
    floor the model has not taken yet, and the system reserve.

    The model's resident memory is already outside MemAvailable, so only
    its growth room is kept free, not its whole floor again.
    """

    return (max(0.0, live_render_model_floor(model_rss, host_total, settings)
                - model_rss) + settings.system_reserve)


def _proc_stat(pid: int) -> tuple[int, int] | None:
    """``(parent pid, resident bytes)`` of one Linux process, or ``None``."""

    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            text = handle.read().decode("ascii", "replace")
    except OSError:
        return None
    # The command name is in parentheses and may hold spaces or ')'.
    fields = text[text.rfind(")") + 2:].split()
    try:
        return int(fields[1]), int(fields[21]) * _PAGE_BYTES
    except (IndexError, ValueError):
        return None


def _proc_peak(pid: int) -> int | None:
    """One Linux process's peak resident memory (``VmHWM``), or ``None``."""

    try:
        with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


try:
    _PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
except (AttributeError, ValueError, OSError):
    _PAGE_BYTES = 4096


def _own_resident_bytes() -> int | None:
    """This process's resident memory now, on any platform, or ``None``."""

    if os.path.exists("/proc/self/stat"):
        stat = _proc_stat(os.getpid())
        return None if stat is None else stat[1]
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                    ctypes.windll.kernel32.GetCurrentProcess(),
                    ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:  # noqa: BLE001 - unknown is no reading
            return None
    return None


class _HostMemoryProbe:
    """The readings the memory admission takes, in one place for tests.

    ``available()`` is the engine's one host probe (MemAvailable capped
    by the cgroup's room).  ``total()`` is the host's memory capped by
    the cgroup limit.  ``sample(pids)`` measures, on Linux, the model
    (this process and every process below it that is not a render) and
    each render's process tree by its root pid: ``(model_rss, {pid:
    (tree_rss, tree_peak)})``.  Off Linux the trees are unmeasured and
    the model is this process alone.
    """

    def available(self) -> int | None:
        return _available_memory_bytes()

    def total(self) -> int | None:
        try:
            from woof.core.preflight import _host_total_bytes_or_none

            return _host_total_bytes_or_none()
        except Exception:  # noqa: BLE001 - unknown host size, no small-host floor
            return None

    def sample(self, render_pids: Sequence[int]
               ) -> tuple[int | None, dict[int, tuple[int, int]]]:
        if not os.path.isdir("/proc/self"):
            return _own_resident_bytes(), {}
        me = os.getpid()
        children = _proc_children_reader()
        if children is None:
            return _own_resident_bytes(), {}
        table: dict[int, tuple[int, int]] = {}

        def tree(root: int) -> set[int]:
            members, todo = {root}, [root]
            while todo:
                for child in children(todo.pop()):
                    if child not in members:
                        members.add(child)
                        todo.append(child)
            for member in members:
                if member not in table:
                    stat = _proc_stat(member)
                    if stat is not None:
                        table[member] = stat
            return members

        trees = {}
        rendering: set[int] = set()
        for pid in render_pids:
            members = {member for member in tree(pid) if member in table}
            if not members:
                continue
            rendering |= members
            rss = sum(table[member][1] for member in members)
            # Each process's own high-water mark, so a peak between two
            # samples is still seen while the process lives.
            peak = sum(max(table[member][1], _proc_peak(member) or 0)
                       for member in members)
            trees[pid] = (rss, peak)
        mine = tree(me)
        model = (sum(table[member][1] for member in mine - rendering
                     if member in table)
                 if me in table else _own_resident_bytes())
        return model, trees


def _proc_children_reader() -> Callable[[int], list[int]] | None:
    """A function listing a Linux process's children, or ``None``.

    ``/proc/<pid>/task/<tid>/children`` where the kernel provides it
    (cheap: only the processes below this one are read); otherwise one
    scan of every ``/proc/<pid>/stat`` for this call.
    """

    def from_tasks(pid: int) -> list[int]:
        found: list[int] = []
        try:
            tasks = os.listdir(f"/proc/{pid}/task")
        except OSError:
            return found
        for task in tasks:
            try:
                with open(f"/proc/{pid}/task/{task}/children", encoding="ascii") as handle:
                    found.extend(int(word) for word in handle.read().split())
            except (OSError, ValueError):
                continue
        return found

    if os.path.exists(f"/proc/self/task/{os.getpid()}/children"):
        return from_tasks
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    by_parent: dict[int, list[int]] = {}
    for name in names:
        if name.isdigit():
            stat = _proc_stat(int(name))
            if stat is not None:
                by_parent.setdefault(stat[0], []).append(int(name))
    return lambda pid: by_parent.get(pid, [])


def _render_pid(ident: int) -> int | None:
    """The pid of the render a thread is running, if it is running one."""

    try:
        from woof.first_products import _RUNNING, _RUNNING_LOCK

        with _RUNNING_LOCK:
            process = _RUNNING.get(ident)
        return None if process is None else int(process.pid)
    except Exception:  # noqa: BLE001 - an unmeasured render keeps its estimate
        return None


def _read_words(path: Path) -> list[str] | None:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().split()
    except (OSError, UnicodeDecodeError):
        return None


def _quota_processors(quota_text: str, period_text: str) -> int | None:
    """Processors a ``quota``/``period`` pair grants, ``None`` for no quota.

    A fractional quota rounds up, since a 1.5-processor quota does run
    two threads.
    """

    try:
        quota, period = int(quota_text), int(period_text)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, -(-quota // period))


def _cgroup_ancestry(root: Path, relative: str) -> list[Path]:
    """``root/relative`` and every directory above it, up to ``root``.

    A container that mounts its own cgroup at ``root`` while
    ``/proc/self/cgroup`` still names the host path has no such
    directory below ``root``; ``root`` itself is always read.
    """

    parts = [part for part in relative.split("/") if part and part != ".."]
    return [root.joinpath(*parts[:depth]) for depth in range(len(parts), -1, -1)]


def _cgroup_memberships() -> list[tuple[str, str]]:
    """``(controllers, path)`` for each hierarchy this process is in."""

    try:
        with open(_PROC_SELF_CGROUP, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    memberships = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3:
            memberships.append((fields[1], fields[2]))
    return memberships


def _cgroup_cpu_limit() -> int | None:
    """The processors this process's cgroup CPU quota grants, or ``None``.

    The tightest quota anywhere between the process's own cgroup and the
    root of its hierarchy binds, on cgroup v2 (``cpu.max``) and on a
    cgroup v1 ``cpu`` controller (``cpu.cfs_quota_us`` over
    ``cpu.cfs_period_us``).

    THE BREAKAGE: a container limited to four processors on a 64-core
    host (``cpu.max`` of ``400000 100000``) still lists all 64 in its
    affinity mask, so live frames were budgeted as though 63 were free:
    three concurrent imports each given 21 native threads, throttled into
    the four processors the forecast also needs.  Reading only the root
    ``cpu.max`` missed the same limit set on a systemd slice or a nested
    cgroup (the root of a cgroup v2 host carries no ``cpu.max`` at all),
    and every limit on a cgroup v1 host.
    """

    root = Path(_CGROUP_ROOT)
    memberships = _cgroup_memberships() or [("", "/")]
    limits: list[int] = []
    for controllers, relative in memberships:
        if controllers == "":
            for directory in _cgroup_ancestry(root, relative):
                words = _read_words(directory / "cpu.max")
                if words and len(words) == 2 and words[0] != "max":
                    granted = _quota_processors(words[0], words[1])
                    if granted is not None:
                        limits.append(granted)
        elif "cpu" in controllers.split(","):
            mounts = [root / controllers, root / "cpu"]
            for mount in dict.fromkeys(mounts):
                for directory in _cgroup_ancestry(mount, relative):
                    quota = _read_words(directory / "cpu.cfs_quota_us")
                    period = _read_words(directory / "cpu.cfs_period_us")
                    if quota and period and len(quota) == 1 and len(period) == 1:
                        granted = _quota_processors(quota[0], period[0])
                        if granted is not None:
                            limits.append(granted)
    return min(limits) if limits else None


def _available_cpus() -> int:
    count = None
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = max(1, len(affinity(0)))
        except OSError:
            pass
    if count is None:
        count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    quota = _cgroup_cpu_limit()
    return count if quota is None else min(count, quota)


def _available_memory_bytes() -> int | None:
    """Available physical memory, from the engine's one host probe.

    :func:`woof.core.preflight.host_available_bytes`: ``MemAvailable``
    or ``GlobalMemoryStatusEx``, capped at a container's memory ceiling,
    which is the reading every other host-memory decision makes.
    ``None`` when the platform will not say.
    """

    try:
        from woof.core.preflight import host_available_bytes

        return host_available_bytes()
    except Exception:  # noqa: BLE001 - an unknown reading imposes no cap
        return None


def _live_memory_budget() -> int | None:
    """The bytes concurrent live frames may plan to hold, or ``None``.

    Half the available memory (:data:`_LIVE_MEMORY_FRACTION`); ``None``
    when the platform will not report it, which imposes no cap.
    """

    available = _available_memory_bytes()
    if available is None:
        return None
    return int(available * _LIVE_MEMORY_FRACTION)


def _frames_memory_affords(budget: int | None) -> int | None:
    """How many of the smallest live frames fit in ``budget``, or ``None``.

    THE BREAKAGE: frames were sized from processors alone, so an eight
    processor host with little free memory still drew three frames at
    once.  Each renderer gives its product workers a share of half the
    free memory but never fewer than one worker, so three imports on
    such a host each held a whole frame beside the forecast.  A platform
    that will not report free memory imposes no cap, as before.  A larger
    grid's frames are held to the same budget by their own price as they
    are drawn (:class:`_LiveRenderSlots`).
    """

    if budget is None:
        return None
    return max(1, budget // LIVE_FRAME_PEAK_BYTES)


def _render_width(budget: int | None) -> int:
    available = _available_cpus()
    # Two native workers per frame, one processor for the forecast, and at
    # most three simultaneous imports even on hosts with many more cores.
    width = min(3, max(1, (available - 1) // 2))
    # Race box override: the per-frame wrfout import is ~28 s on one core for a
    # full-CONUS frame, so three slots fall behind a 56 s/fh forecast (a rented box,
    # 2026-10-21: 190-200 s per 145-product frame). More slots keep up.
    override = os.environ.get("WOOF_LIVE_RENDER_SLOTS")
    if override:
        width = max(1, min(int(override), max(1, (available - 1) // 2)))
    affordable = _frames_memory_affords(budget)
    return width if affordable is None else min(width, affordable)


def _render_concurrency() -> int:
    return _render_width(_live_memory_budget())


#: The tag that opens a classic NetCDF header's dimension list.
_NC_DIMENSION = 0x0A


def _classic_dimensions(frame: Path) -> dict[str, int] | None:
    """The dimensions a classic (CDF-1, CDF-2 or CDF-5) header declares.

    Read straight off the header's first bytes, where the classic formats
    put the dimension list, so pricing a frame opens no NetCDF library:
    this runs on a live render thread inside the forecast's process, where
    every netCDF4 session must hold the writer's lock
    (:mod:`woof.io.netcdf_serialization`).  ``None`` for any other file.
    """

    with open(fs_path(frame), "rb") as handle:
        magic = handle.read(4)
        if magic[:3] != b"CDF" or magic[3:] not in (b"\x01", b"\x02", b"\x05"):
            return None
        width = 8 if magic[3:] == b"\x05" else 4

        def number(size: int = width) -> int:
            data = handle.read(size)
            if len(data) != size:
                raise ValueError("classic header ends early")
            return int.from_bytes(data, "big")

        number()  # the record count
        if number(4) != _NC_DIMENSION:
            return {}
        dimensions = {}
        count = number()
        if count > 1024:
            raise ValueError("not a dimension list")
        for _ in range(count):
            length = number()
            if length > 1024:
                raise ValueError("not a dimension name")
            name = handle.read(length).decode("utf-8", "replace")
            handle.read(-length % 4)
            dimensions[name] = number()
        return dimensions


#: The WRF grid-dimension globals a compressed frame is priced from.
_GRID_DIMENSION_ATTRS = {b"WEST-EAST_GRID_DIMENSION": "west_east",
                         b"SOUTH-NORTH_GRID_DIMENSION": "south_north",
                         b"BOTTOM-TOP_GRID_DIMENSION": "bottom_top"}


def _netcdf4_grid_dimensions(frame: Path) -> dict[str, int] | None:
    """The mass-grid dimensions of a compressed (NetCDF-4) history frame.

    woof's history writer stores frames as lossless NetCDF-4 by default
    (``woof.io.wrfout.HISTORY_COMPRESSION_ENV``), so the classic header
    read above finds nothing, and pricing from the file's size would count
    a third of the cells.  The frame's root group carries the WRF
    ``*_GRID_DIMENSION`` globals (staggered, one past the mass count) as
    compact attribute messages; this reads exactly those, opening no NetCDF
    library (see :func:`_classic_dimensions` for why).  ``None`` for a file
    laid out any other way.
    """

    with open(fs_path(frame), "rb") as handle:
        head = handle.read(96)
        if head[:8] != b"\x89HDF\r\n\x1a\n" or head[8] != 0:
            return None
        root = int.from_bytes(head[64:72], "little")
        handle.seek(root)
        prefix = handle.read(6)
        if prefix[:4] != b"OHDR" or prefix[4] != 2:
            return None
        flags = prefix[5]
        width = 1 << (flags & 3)
        if flags & 0x10:
            handle.read(16)
        if flags & 0x20:
            handle.read(4)
        size = int.from_bytes(handle.read(width), "little")
        if size > (16 << 20):
            return None
        body = handle.read(size)
    order = 2 if flags & 0x04 else 0
    found: dict[str, int] = {}
    at = 0
    while at + 4 + order <= len(body):
        kind = body[at]
        length = int.from_bytes(body[at + 1:at + 3], "little")
        start = at + 4 + order
        message = body[start:start + length]
        at = start + length
        if kind != 12 or len(message) < 8 or message[0] != 1:
            continue
        name_len = int.from_bytes(message[2:4], "little")
        type_len = int.from_bytes(message[4:6], "little")
        space_len = int.from_bytes(message[6:8], "little")
        name = message[8:8 + name_len].rstrip(b"\x00")
        key = _GRID_DIMENSION_ATTRS.get(name)
        if key is None:
            continue
        pad = lambda n: (n + 7) & ~7  # noqa: E731
        data = 8 + pad(name_len) + pad(type_len) + pad(space_len)
        dtype = message[8 + pad(name_len):8 + pad(name_len) + type_len]
        width_bytes = int.from_bytes(dtype[4:8], "little") if len(dtype) >= 8 else 4
        if width_bytes not in (4, 8):
            continue
        found[key] = int.from_bytes(message[data:data + width_bytes], "little",
                                    signed=True) - 1
    return found or None


def _frame_cells(frame: Path) -> int:
    """The cells (west_east x south_north x bottom_top) of a history frame.

    Read from the frame's classic header (WRF and woof's uncompressed
    frames are CDF-2), else from a compressed frame's grid-dimension
    globals.  A frame whose header does not say is priced from the file's
    size instead: every cell of an uncompressed history frame stores at
    least one four-byte value, so a quarter of its bytes is never fewer
    cells than it has.
    """

    try:
        dimensions = _classic_dimensions(frame)
    except (OSError, ValueError):
        dimensions = None
    if dimensions is None:
        try:
            dimensions = _netcdf4_grid_dimensions(frame)
        except (OSError, ValueError, IndexError):
            dimensions = None
    if dimensions and "west_east" in dimensions and "south_north" in dimensions:
        return (dimensions["west_east"] * dimensions["south_north"]
                * max(1, dimensions.get("bottom_top", 1)))
    try:
        return Path(fs_path(frame)).stat().st_size // 4
    except OSError:
        return 0


def live_frame_peak_bytes(frame: Path, baselines: Sequence[Path] = ()) -> int:
    """What drawing ``frame`` beside ``baselines`` holds at its peak.

    THE BREAKAGE: one figure (:data:`LIVE_FRAME_PEAK_BYTES`, 640 MiB),
    measured on 72x58 and 144x112 frames, priced every frame, so a host
    with 16 GiB free, whose half affords three of those, drew three
    880x704x55 frames at once, and each of those held 7.0 GiB: 21 GiB of
    renderers beside the forecast.  The import grows with the grid, so
    each frame is priced from its own cells and its baselines' on top of
    the fixed part.
    """

    price = LIVE_FRAME_PEAK_BYTES + LIVE_FRAME_BYTES_PER_CELL * _frame_cells(frame)
    for baseline in baselines:
        price += LIVE_BASELINE_BYTES_PER_CELL * _frame_cells(baseline)
    return int(price)


def live_render_requested(render_products: Any) -> bool:
    """Whether the end of the run would draw pictures at all.

    Unlike the first-frame render, an unset product spec is ON: the
    finalize stage draws the default set when nothing is named, and a
    run that will have pictures at the end has them as it goes.  Only
    ``none`` turns this off.
    """

    text = "" if render_products is None else str(render_products).strip()
    return text.lower() != "none"


#: How much nicer than the forecast a render beside it runs (POSIX).
#: THE BREAKAGE: live renders ran at the forecast's own priority (nice 0
#: on an 8x5090 host, 2.9.3), so on a host short of cores a render's ~170
#: worker threads competed with the model's host threads as equals.
LIVE_RENDER_NICE = 10


def _run_render(command: Sequence[str], *, width: int | None = None,
                finishing: bool = False, drawing: int | None = None,
                **options: Any) -> subprocess.CompletedProcess:
    """Spawn the render exactly as the early and finalize renders do.

    ``width`` is how many frames may draw at once (the memory share each
    renderer's workers get, ``RUSTWX_LIVE_RENDER_SLOTS``); ``drawing`` is
    how many are drawing as this one starts, this one included.  The
    processors are shared among the frames drawing NOW, never the planned
    width: split three ways while frames landed one at a time, a lone
    live render on an eight-processor box ran on two threads and the
    other five sat idle (2.9.5, RTX PRO 4500, 2026-10-23: the render after
    a 6 h nest ran 9 to 12 min at about 130% CPU).  ``None`` keeps the
    planned width, for a caller that does not count.
    """

    from woof.first_products import _run_render as run

    available = _available_cpus()
    concurrency = _render_concurrency() if width is None else width
    sharing = concurrency if drawing is None else max(1, min(concurrency, int(drawing)))
    options["env_overrides"] = {
        "RUSTWX_LIVE_RENDER_SLOTS": str(concurrency),
        # The renderer's product loop takes its width from this too, not from
        # the slots: divided by three slots, a lone frame on an eight-processor
        # box drew its products two at a time (2.25 km / 750 m HRRR nest,
        # 2026-10-24). The slots still divide each renderer's memory.
        "RUSTWX_LIVE_RENDER_DRAWING": str(sharing),
        "RAYON_NUM_THREADS": os.environ.get(
            "RAYON_NUM_THREADS", str(max(1, (available - (0 if finishing else 1)) // sharing))),
    }
    options.setdefault("nice", 0 if finishing else LIVE_RENDER_NICE)
    return run(command, **options)


def _frame_identity(frame: Path) -> dict[str, int]:
    status = Path(fs_path(frame)).stat()
    return {"frame_size": int(status.st_size),
            "frame_mtime_ns": int(status.st_mtime_ns)}


class LiveProducts:
    """Every committed frame of every grid, with bounded concurrent imports.

    ``render_plan`` is the dict finalize hands
    :func:`woof.go_cli._render_stage`.  ``report`` receives each
    published frame's record (on the worker thread) and ``warn`` takes
    ``(code, message, **fields)``.  ``first`` is the early render of the
    first root frame, when one is armed: once it is dispatched it is
    waited for before the next frame here.  ``slot`` is the lock both
    hold for a whole render (pass the semaphore given to that
    :class:`~woof.first_products.FirstProducts`). Each frame has private
    scratch; completed pictures and their receipts publish under one lock.
    """

    def __init__(self, render_plan: Mapping[str, Any], *,
                 report: Callable[[dict], None],
                 warn: Callable[..., None],
                 first=None,
                 runner: Callable[[Sequence[str]],
                                  subprocess.CompletedProcess] | None = None,
                 slot: threading.Lock | None = None,
                 own_group: bool = False,
                 windowed_slugs: Callable[[], Any] | None = None):
        self._plan = dict(render_plan)
        from woof.product_publish import LocalPublisher
        self._web = LocalPublisher(
            self._plan.get("publish_root") or Path(self._plan["render"]).parent / "www",
            cycle=self._plan.get("cycle"), command=self._plan.get("publish_cmd"),
            hydro=self._plan.get("hydro") or "auto")
        self._closed_at: dict[tuple[int, str], str] = {}
        self._report = report
        self._warn = warn
        self._first = first
        self._slot = shared_render_slots()[0] if slot is None else slot
        # An external Lock has one slot. Only our shared semaphore carries a
        # declared width; guessing from the host would divide a serial render.
        self._concurrency = (self._slot.width
                             if isinstance(self._slot, _LiveRenderSlots) else 1)
        self._active = 0
        self._runner = (partial(_run_render, width=self._concurrency)
                        if runner is None else runner)
        self._default_runner = self._runner if runner is None else None
        #: Each render starts in a process group of its own, for a host
        #: that ends it itself on a stop (:meth:`halt`); passed to the
        #: runner as ``own_group`` (:func:`woof.first_products._run_render`).
        self._own_group = bool(own_group)
        self._cond = threading.Condition()
        self._flags_lock = threading.Lock()
        self._queue: deque[dict[str, Any]] = deque()
        self._closed = False
        self._thread: threading.Thread | None = None
        #: Set when the worker returns; asked instead of the thread's
        #: ``is_alive()``, which a stop landing in :meth:`stop`'s wait
        #: makes wrong (:class:`woof.first_products.WorkerEnd`).
        self._ended = WorkerEnd()
        self._workers: list[tuple[threading.Thread, WorkerEnd]] = []
        self._stoppers: list[threading.Thread] = []
        self._entries: list[dict[str, Any]] = []
        self._render_seconds: list[float] = []
        # Held while a drawn frame is moved into the folder and recorded.
        # ``stop`` takes it to give up on a render it could not wait out,
        # so a late render never publishes while finalize draws into the
        # same folder.
        self._publish = threading.Lock()
        self._abandoned = False
        # Each grid's frames since its latest whole-hour frame (that frame
        # first), drawn or claimed by the early render: the baselines that
        # close the next whole hour's windows.
        self._open_hour: dict[str, list[Path]] = {}
        # The grids a frame has been committed for, and whether the
        # request is made only of windows (asked once, by the worker, of
        # ``windowed_slugs``: :func:`catalog_windowed_slugs` by default).
        self._seen_grids: set[str] = set()
        self._windowed_slugs = (catalog_windowed_slugs if windowed_slugs is None
                                else windowed_slugs)
        self._windows_only: bool | None = None
        # Whether the request holds any window at all, asked once of the
        # same listing: the hour's earlier frames are imported only then.
        self._draws_windows: bool | None = None
        # A resumed run's first frames close windows that opened before its
        # checkpoint.  Every grid's history saved up to the checkpoint is
        # noted as though it had been committed here, a nest the early
        # root render never claims included, so the first new whole hour
        # of each grid is drawn with the hour that precedes it.
        from woof.restart_render import history_before_restart
        for frame in history_before_restart(self._plan.get("restart")):
            self._closes_a_window(frame)
            self._closing_frames(frame)

    @property
    def render_dir(self) -> Path:
        return Path(self._plan["render"])

    @property
    def render_products(self) -> str:
        return str(self._plan.get("render_products") or "")

    @property
    def render_section(self) -> str | None:
        """The line the section products are cut along, or ``None``."""

        from woof.first_products import section_text

        return section_text(self._plan.get("render_section"))

    @property
    def published(self) -> list[dict[str, Any]]:
        """The records of every frame published so far."""

        with self._cond:
            return list(self._entries)

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._queue)

    @property
    def running(self) -> bool:
        """Whether the worker is still alive (after a halt: still drawing)."""

        return self._thread is not None and not self._ended.ended

    def render_threads(self) -> list:
        """The threads that start this render's processes, with their ends."""

        with self._cond:
            return list(self._workers)

    # -- the hook -------------------------------------------------------

    def frame_submitted(self, *, domain: int, valid_time: Any) -> None:
        """Hour-close timestamp, captured before the history writer is queued."""
        from woof.product_publish import utc_now, instant
        with self._cond:
            self._closed_at[(int(domain), instant(valid_time).isoformat())] = utc_now()
            if self._web.cycle is None:
                self._web.cycle = instant(valid_time)

    def frame_committed(self, *, domain: int, valid_time: Any, path: Any,
                        draw: bool = True) -> None:
        """Queue one durable frame and return at once.

        ``draw=False`` is a frame the early render has claimed.  It is not
        drawn here, but on a whole hour it is still the baseline of its
        grid's next whole-hour frame.
        """

        frame = Path(path)
        from woof.product_publish import instant
        # A quick-look frame (woof.io.wrfout.quicklook_path) carries the
        # subset only: it is drawn alone, with the quick-look maps, and is
        # never the baseline of an hour's window products.
        quicklook = is_quicklook_frame(frame)
        with self._cond:
            if self._closed:
                return
            if self._web.cycle is None:
                self._web.cycle = instant(valid_time)
            closes = False if quicklook else self._closes_a_window(frame)
            context = [] if quicklook else self._closing_frames(frame)
            if not draw:
                return
            self._queue.append({"domain": int(domain),
                                "valid_time": valid_time,
                                "hour_closed_at": self._closed_at.pop(
                                    (int(domain), instant(valid_time).isoformat()), None),
                                "frame": frame,
                                "context": context,
                                "closes": closes})
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._ended.run, args=(self._loop,),
                    name="gpuwm-live-products", daemon=True)
                self._thread.start()
            self._cond.notify_all()

    def _closing_frames(self, frame: Path) -> list[Path]:
        """The frames of the hour ``frame`` closes, and note ``frame``.

        On a whole hour: the grid's previous whole-hour frame and every
        frame committed after it, which on a 15-minute grid are the three
        frames whose interval maxima make up the hour.  Empty unless an
        earlier whole-hour frame of the grid was committed, and for a
        frame between hours, which is drawn alone.  Held under ``_cond``;
        frames of one grid arrive in time order from that grid's writer.
        """

        parsed = _grid_and_whole_hour(frame)
        if parsed is None:
            return []
        grid, whole = parsed
        hour = self._open_hour.get(grid)
        if not whole:
            if hour:
                hour.append(frame)
            return []
        self._open_hour[grid] = [frame]
        if not hour or hour[0].name >= frame.name:
            return []
        return [earlier for earlier in hour if earlier.name < frame.name]

    def _closes_a_window(self, frame: Path) -> bool:
        """Whether a window can end on ``frame``, and note its grid.

        A whole-hour frame after its grid's first, as
        :func:`closes_a_window` decides it over a finished series.  Held
        under ``_cond``.
        """

        parsed = _grid_and_whole_hour(frame)
        if parsed is None:
            return True
        grid, whole = parsed
        seen = grid in self._seen_grids
        self._seen_grids.add(grid)
        return whole and seen

    def _draws_windows_only(self) -> bool:
        """Whether the request is made only of windows, asked once."""

        # A catalog lookup must not hold the queue lock: frame commits and
        # stop requests still need to return while that subprocess runs.
        with self._flags_lock:
            if self._windows_only is None:
                try:
                    self._windows_only = windows_only(self.render_products,
                                                      self._windowed_slugs)
                except Exception:  # noqa: BLE001 - draw every frame as before
                    self._windows_only = False
            return self._windows_only

    def _requests_windows(self) -> bool:
        """Whether the request holds any window, asked once."""

        with self._flags_lock:
            if self._draws_windows is None:
                try:
                    self._draws_windows = requests_windows(self.render_products,
                                                           self._windowed_slugs)
                except Exception:  # noqa: BLE001 - import the hour as before
                    self._draws_windows = True
            return self._draws_windows

    def stop(self, timeout: float | None = DEFAULT_WAIT_SECONDS
             ) -> dict[str, Any]:
        """Stop taking frames and finish the queue, within ``timeout``.

        Called by finalize before it decides what is left to draw, and by
        a run that failed.  Frames still queued when the wait runs out are
        dropped and drawn in the end-of-run batch instead.
        """

        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        dropped = 0
        if thread is not None:
            self._ended.wait(timeout)
            with self._cond:
                dropped = len(self._queue)
                self._queue.clear()
                self._cond.notify_all()
            if not self._ended.ended:
                self._ended.wait(DEFAULT_WAIT_SECONDS)
            if not self._ended.ended:
                # Finalize is about to draw into the same folder.  A publish
                # already under way finishes first (it only moves files);
                # the render still running will publish nothing, so there
                # is never a second writer (the review's wedged-render case).
                # And that render is ended, not left drawing: it held its
                # process and its scratch in the picture folder past the
                # run that started it.
                self.halt()
                self._warn(
                    "live_products_timeout",
                    "the picture being drawn when the forecast ended was "
                    f"still being drawn after {timeout:.0f} s; it was ended "
                    "and is not published, and the end-of-run render draws "
                    "every frame it cannot prove is published",
                    render_dir=str(self.render_dir))
        self._remove_scratch_root()
        return self._summary(dropped)

    def halt(self, timeout: float | None = HALT_WAIT_SECONDS
             ) -> dict[str, Any]:
        """Stop now, for a run that was stopped.

        Unlike :meth:`stop` nothing queued is drawn.  Nothing is published
        after this returns (a publish already moving files finishes
        first), and the render in flight is ended rather than waited for
        (:func:`woof.first_products.end_render`), so a stopped run is
        not held past the 5 s the desktop gives it and no render outlives
        it.  Every frame is still on disk and draws with ``woof render``.
        Safe to call more than once and after :meth:`stop`.
        """

        from woof.first_products import end_render

        with self._cond:
            self._closed = True
            dropped = len(self._queue)
            self._queue.clear()
            self._cond.notify_all()
            thread = self._thread
        with self._publish:
            self._abandoned = True
        if thread is not None and not self._ended.ended:
            # Both processes receive the stop together. Giving each its own
            # full grace period in sequence could exceed the host's stop time.
            deadline = None if timeout is None else time.monotonic() + timeout
            with self._cond:
                if not self._stoppers:
                    for worker, ended in self._workers:
                        stopper = threading.Thread(
                            target=end_render, args=(worker,),
                            kwargs={"ended": ended}, daemon=True)
                        self._stoppers.append(stopper)
                        stopper.start()
                stopping = list(self._stoppers)
            for stopper in stopping:
                stopper.join(None if deadline is None else max(
                    0, deadline - time.monotonic()))
            self._ended.wait(None if deadline is None else max(
                0, deadline - time.monotonic()))
        self._remove_scratch_root()
        return self._summary(dropped)

    def _remove_scratch_root(self) -> None:
        # Workers may be between creating the shared root and their own child.
        # It is safe to remove the root only after every worker has returned.
        if self._thread is not None and not self._ended.ended:
            return
        try:
            (self.render_dir / _SCRATCH_NAME).rmdir()
        except OSError:
            pass

    def _summary(self, dropped: int) -> dict[str, Any]:
        seconds = self._render_seconds
        summary = {"published": len(self._entries), "dropped": dropped,
                   "render_seconds_total": round(sum(seconds), 3),
                   "render_seconds_max": round(max(seconds), 3) if seconds else None}
        record = self._memory_record()
        if record is not None:
            summary["memory_admission"] = record
        return summary

    # -- the worker -----------------------------------------------------

    def _loop(self) -> None:
        self._draws_windows_only()
        self._requests_windows()
        with self._cond:
            for index in range(self._concurrency):
                ended = WorkerEnd()
                worker = threading.Thread(
                    target=ended.run, args=(self._draw_loop,),
                    name=f"gpuwm-live-render-{index}", daemon=True)
                self._workers.append((worker, ended))
                worker.start()
        for worker, _ in self._workers:
            worker.join()

    def _draw_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue:
                    return
                item = self._queue.popleft()
                # The admission order is the queue's order: the place is
                # taken here, under the queue's lock, not when this thread
                # happens to reach the slots.
                turn = (self._slot.ticket()
                        if isinstance(self._slot, _LiveRenderSlots) else None)
            try:
                if not self._draw_one(item, turn):
                    return
            finally:
                if turn is not None:
                    self._slot.cancel(turn)

    def _draw_one(self, item: dict[str, Any], turn: object | None) -> bool:
        """Draw one queued frame; ``False`` once the pass was halted."""

        if not item.pop("closes", True) and self._draws_windows_only():
            # A request made only of windows has nothing to draw where
            # no window ends: its render exited 1 with no picture and
            # the frame was said as left to the end of the run, which
            # has nothing to draw there either.
            return True
        first = self._first
        if first is not None and first.dispatched:
            # Finish the early render before publishing later frames.
            first.wait()
        if self._abandoned:
            # Halted while this frame waited its turn.
            return False
        try:
            with self._admission(item, turn):
                if self._abandoned:
                    return False
                with self._cond:
                    self._active += 1
                try:
                    self._render(**item)
                finally:
                    with self._cond:
                        self._active -= 1
        except BaseException as error:  # noqa: BLE001 - never fail a run
            try:
                self._warn(
                    "live_products_failed",
                    f"drawing {Path(item['frame']).name} as it landed "
                    f"raised {type(error).__name__}: {error}; the "
                    "end-of-run render draws it",
                    frame=str(item["frame"]))
            except Exception:  # noqa: BLE001 - nor its warning
                pass
        return True

    def _admission(self, item: Mapping[str, Any], turn: object | None = None):
        """The slot one queued frame is drawn under.

        On the host's shared slots a frame waits until its own price fits
        beside the frames already drawing, and, with the memory admission
        on, until memory read now leaves the model its floor; a caller's
        own lock is one frame at a time, which no price can narrow.
        """

        if not isinstance(self._slot, _LiveRenderSlots):
            return self._slot
        frame = Path(item["frame"])
        price = live_frame_peak_bytes(
            frame, self._imported_baselines(item.get("context") or ()))
        if turn is None:
            return self._slot.admit(price)
        return self._slot.admit(price, turn)

    def _imported_baselines(self, context: Sequence[Path]) -> list[Path]:
        """The earlier frames a render imports beside its frame."""

        # The baselines must still be on disk; one that is gone is left out,
        # the engine blocks by name any window that needed it, and
        # finalize draws that window over the whole series.
        #
        # And they are imported only for a request that holds a window.
        # THE BREAKAGE: a 3 km run asked for composite reflectivity,
        # 2 m temperature and 10 m wind, no window among them, and every
        # whole hour was drawn beside the hour's six earlier frames: 77 to
        # 82 s against 14 s for every other frame, for baselines nothing
        # read.  The first of those frames, the analysis, stores no
        # REFL_10CM, and the lead-1h frame was published without its
        # reflectivity.
        return ([path for path in context if Path(fs_path(path)).is_file()]
                if context and self._requests_windows() else [])

    def _render(self, *, domain: int, valid_time: Any, frame: Path,
                context: Sequence[Path] = (), hour_closed_at=None) -> None:
        from woof.go_cli import render_command
        from woof.render import SCRATCH_SUFFIX, announce_missing_basemap
        from woof.product_publish import PRIORITY_PRODUCTS, utc_now

        started = time.perf_counter()
        render_start = utc_now()
        render_dir = self.render_dir
        # Once, not per frame: the first frame drawn (here or by the early
        # render) checks the renderer's map assets and the rest return.
        announce_missing_basemap(self._warn, render_dir, stage="as-drawn")
        scratch = render_dir / _SCRATCH_NAME / frame.name
        if scratch.exists():
            shutil.rmtree(fs_path(scratch, descend=True))
        scratch.mkdir(parents=True)
        baselines = self._imported_baselines(context)
        try:
            # A small native import for the headline set starts before the
            # expensive pressure volumes/parcel diagnostics of the full set.
            # Explicit subsets retain their meaning; never add unasked products.
            terms = {p.strip() for p in (self.render_products or "all").split(",")}
            plan = dict(self._plan)
            if is_quicklook_frame(frame):
                # The quick-look subset draws only the maps its variables
                # make; asking it for the full set failed every frame
                # ("produced no picture as it landed (render exited 1)").
                plan["render_products"] = quicklook_products(self.render_products)
                terms = set(plan["render_products"].split(","))
            priority = [p for p in PRIORITY_PRODUCTS if "all" in terms or p in terms]
            if len(priority) == len(PRIORITY_PRODUCTS) and terms != set(priority):
                priority_done = self._run_render(render_command(
                    {**plan, "render": scratch,
                     "render_products": ",".join(priority)}, [frame],
                    context_frames=baselines))
                if self._abandoned or _render_was_stopped(priority_done.returncode):
                    return
                pictures = finished_pictures(scratch, iter_rendered(scratch),
                                             priority_done.returncode)
                if pictures:
                    with self._publish:
                        if self._abandoned:
                            return
                        self._publish_web(scratch, pictures, domain=domain,
                            valid_time=valid_time, hour_closed_at=hour_closed_at,
                            render_start=render_start, render_end=utc_now(), complete=False)
            completed = self._run_render(render_command(
                {**plan, "render": scratch}, [frame],
                context_frames=baselines))
            render_end = utc_now()
            if self._abandoned:
                # Halted while it drew: the halt ended this render, and
                # nothing it left is published.
                return
            if _render_was_stopped(completed.returncode):
                self._warn(
                    "live_products_stopped",
                    f"drawing {frame.name} was stopped before it finished "
                    f"(render exited {completed.returncode}), so none of "
                    "it was published; the frame is on disk and draws "
                    "with woof render",
                    frame=str(frame))
                return
            written = finished_pictures(scratch, iter_rendered(scratch),
                                        completed.returncode)
            if not written:
                from woof.render_receipts import preserve_inactive_fire_skip
                with self._publish:
                    if preserve_inactive_fire_skip(scratch, render_dir, completed.returncode):
                        return
                self._warn(
                    "live_products_empty",
                    f"{frame.name} produced no picture as it landed (render "
                    f"exited {completed.returncode}); the end-of-run render "
                    "draws it",
                    frame=str(frame),
                    stderr=(completed.stderr or "")[-2000:])
                return
            with self._publish:
                if self._abandoned:
                    # stop() gave up waiting and finalize owns the folder.
                    return
                published = self._publish_frame(
                    scratch, render_dir, written, domain=domain,
                    valid_time=valid_time, frame=frame, started=started,
                    completed=completed, baselines=baselines)
                published["publication"] = self._publish_web(
                    render_dir, [render_dir / p["name"] for p in published["written"]],
                    domain=domain, valid_time=valid_time, hour_closed_at=hour_closed_at,
                    render_start=render_start, render_end=render_end,
                    complete=published["complete"])
        finally:
            shutil.rmtree(fs_path(scratch, descend=True), ignore_errors=True)
            shutil.rmtree(
                fs_path(scratch.with_name(scratch.name + SCRATCH_SUFFIX),
                        descend=True),
                ignore_errors=True)
        self._report(published)

    def publish_first(self, receipt):
        from woof.product_publish import instant
        with self._cond:
            closed = self._closed_at.pop((int(receipt["domain"]),
                instant(receipt["valid_time"]).isoformat()), None)
        return self._publish_web(self.render_dir,
            [self.render_dir / p["name"] for p in receipt["written"]],
            domain=receipt["domain"], valid_time=receipt["valid_time"],
            hour_closed_at=closed, render_start=receipt.get("render_start"),
            render_end=receipt.get("render_end"), complete=receipt["complete"])

    def _publish_web(self, source, pictures, **fields):
        try:
            return self._web.publish(source, pictures, **fields,
                timing_source=("frame_submitted" if fields["hour_closed_at"] else "unknown"))
        except Exception as error:
            self._warn("local_publish_failed", f"local publication failed: {error}")
            return None

    def _run_render(self, command):
        """One render of this pass, in its own process group when the pass was asked for that."""

        kwargs = {"own_group": True} if self._own_group else {}
        with self._cond:
            drawing = self._active
            finishing = self._closed and not self._queue and drawing == 1
        if self._runner is self._default_runner:
            if finishing:
                return _run_render(command, width=1, finishing=True, **kwargs)
            return self._runner(command, drawing=drawing, **kwargs)
        return self._runner(command, **kwargs)

    def _publish_frame(self, scratch: Path, render_dir: Path,
                       written: Sequence[Path], *, domain: int,
                       valid_time: Any, frame: Path,
                       started: float,
                       completed: subprocess.CompletedProcess,
                       baselines: Sequence[Path] = ()) -> dict[str, Any]:
        """Move one drawn frame into the folder and record it.  Held under
        ``_publish``; returns the stream event for the frame."""

        from woof import render_georef
        from woof.first_products import _sha256_file, render_outcome
        from woof.render_receipts import relocate_invocations

        published = []
        for source in written:
            relative = source.relative_to(scratch)
            target = render_dir / relative
            spelled = fs_path(target)
            Path(spelled).parent.mkdir(parents=True, exist_ok=True)
            os.replace(fs_path(source), spelled)
            published.append({"name": relative.as_posix(),
                              "sha256": _sha256_file(target),
                              "size_bytes": Path(spelled).stat().st_size})
        # The batch's map record, already keyed by the layout path
        # (woof.render re-keys it), folded into the folder's own.
        render_georef.fold(render_dir, render_georef.read(
            scratch / render_georef.GEOREF_FILENAME))
        relocate_invocations(scratch, render_dir, published)
        elapsed = time.perf_counter() - started
        entry = {
            "frame": str(frame),
            **_frame_identity(frame),
            "domain": int(domain),
            "valid_time": (valid_time.isoformat()
                           if hasattr(valid_time, "isoformat")
                           else str(valid_time)),
            "written": published,
            # The earlier frames imported beside this one, when it closed
            # a whole hour; their pictures are not delivered.
            "context": [str(path) for path in baselines],
            "render_seconds": round(elapsed, 6),
            "published_unix_ms": int(time.time() * 1000),
            # A renderer that failed partway leaves the frame incomplete:
            # its pictures are kept, and finalize draws the frame again.
            **render_outcome(completed, render_dir=render_dir,
                             published=published),
        }
        if not entry["complete"]:
            self._warn(
                "live_products_incomplete",
                f"drawing {frame.name} as it landed exited "
                f"{entry['exit_code']} after {len(published)} picture(s); "
                "they are kept, and the end-of-run render draws the frame "
                "again for the rest",
                frame=str(frame), exit_code=entry["exit_code"],
                products=entry["products"],
                diagnostics=entry["diagnostics"])
        with self._cond:
            self._entries.append(entry)
            self._render_seconds.append(elapsed)
            entries = list(self._entries)
        self._write_live_receipt(render_dir, entries)
        return {**entry, "pictures": len(published),
                "queued": self.pending}

    def _memory_record(self) -> dict[str, Any] | None:
        """The memory admission's record, when it is on, else ``None``."""

        if isinstance(self._slot, _LiveRenderSlots) and self._slot.adaptive:
            try:
                return self._slot.record()
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                return None
        return None

    def _write_live_receipt(self, render_dir: Path,
                            entries: Sequence[Mapping[str, Any]]) -> None:
        payload = {
            "schema": LIVE_PRODUCTS_SCHEMA,
            "render_products": self.render_products,
            "render_section": self.render_section,
            "frames": list(entries),
        }
        # How the renders were admitted, so a run on a small host shows
        # afterwards whether renders were held for memory and for how long.
        record = self._memory_record()
        if record is not None:
            payload["memory_admission"] = record
        _write_receipt(render_dir / LIVE_PRODUCTS_RECEIPT, payload)


class _EarlyRenderOnly:
    """The first-frame render alone, which the every-frame worker waits on.

    :class:`LandingRenders` overrides ``wait`` to finish the every-frame
    queue first; the worker must not wait on that (it would join itself).
    """

    def __init__(self, renders: "LandingRenders"):
        self._renders = renders

    @property
    def dispatched(self) -> bool:
        return self._renders.dispatched

    def wait(self, timeout: float | None = DEFAULT_WAIT_SECONDS):
        return FirstProducts.wait(self._renders, timeout)


class _ExclusiveSlots:
    """Every live frame slot at once, for the early render, each time.

    The early publisher has its own receipt lock. Holding every slot keeps a
    nest committed before the root from publishing alongside that first frame.

    A class, entered as often as it is asked. THE BREAKAGE: the guard was
    a ``@contextmanager`` generator made once per host and stored as
    ``FirstProducts._slot``, and such a generator serves one ``with``: the
    next raised ``AttributeError``. Anything that held the guard before
    the early render (a host checking that it excludes live frames) spent
    it, and the early render then failed with ``first_products_failed``
    and left its frame to the finalize stage.
    """

    def __init__(self, slots: threading.Semaphore, width: int):
        self._slots = slots
        self.width = width

    def __enter__(self) -> "_ExclusiveSlots":
        taken = 0
        try:
            while taken < self.width:
                self._slots.acquire()
                taken += 1
        except BaseException:
            for _ in range(taken):
                self._slots.release()
            raise
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        for _ in range(self.width):
            self._slots.release()
        return False


class _LiveRenderSlots(threading.Semaphore):
    """A host's live frame slots, each frame admitted by its own price.

    ``width`` frames at most run at once (the processors, and the memory
    for that many of the smallest frames).  Among those, a frame starts
    only while the frames already running leave room for its price
    (:func:`live_frame_peak_bytes`) within ``budget``, or when it would
    run alone.  Frames are admitted in the order they asked, so a large
    frame is not passed over indefinitely by smaller ones.  ``width`` of
    the smallest frames always fit: the width was sized to hold them.

    ``adaptive`` (the default, unless :data:`LIVE_RENDER_ADAPTIVE_ENV`
    turns it off) adds the memory admission described above
    :class:`LiveRenderMemorySettings`: on top of that one-shot budget, a
    frame starts only while memory read NOW leaves the model's floor free
    after it and the frames already drawing reach their expected peaks.
    Frames are then admitted in the order they took a :meth:`ticket`
    (the order they left the queue), and the oldest always runs when it
    would run alone.  With ``adaptive`` off this is exactly the one-shot
    budget it was before.
    """

    def __init__(self, width: int, budget: int | None = None, *,
                 adaptive: bool | None = None,
                 settings: LiveRenderMemorySettings | None = None,
                 probe: Any = None,
                 render_pid: Callable[[int], int | None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(width)
        self.width = width
        self.budget = (None if budget is None
                       else max(budget, width * LIVE_FRAME_PEAK_BYTES))
        self._memory = threading.Condition()
        self._reserved = 0
        self._running = 0
        self._waiting: deque[object] = deque()
        self.adaptive = live_render_adaptive() if adaptive is None else bool(adaptive)
        self.settings = (LiveRenderMemorySettings.from_env()
                         if settings is None else settings)
        self._probe = _HostMemoryProbe() if probe is None else probe
        self._render_pid = _render_pid if render_pid is None else render_pid
        self._clock = clock
        self._host_total: int | None = None
        self._host_total_read = False
        # Each running frame: its cell price, the thread drawing it, and
        # its tree's latest and peak resident memory once measured.
        self._flights: dict[object, dict[str, Any]] = {}
        self._peaks: deque[int] = deque(maxlen=max(1, int(self.settings.peaks_kept)))
        self._sampler: threading.Thread | None = None
        self._record: dict[str, Any] = {
            "renders_admitted": 0, "renders_held": 0,
            "longest_hold_seconds": 0.0, "model_rss_bytes": None,
            "keep_free_bytes": None, "available_bytes": None}

    def _fits(self, price: int) -> bool:
        return (self._running == 0 or self.budget is None
                or self._reserved + price <= self.budget)

    # -- the memory admission ------------------------------------------

    def ticket(self) -> object | None:
        """A place in the admission order, taken when a frame leaves the
        queue; ``None`` with the admission off (the old order: whichever
        thread reaches :meth:`admit` first)."""

        if not self.adaptive:
            return None
        turn = object()
        with self._memory:
            self._waiting.append(turn)
        return turn

    def cancel(self, turn: object | None) -> None:
        """Give up a ticket whose frame will not be drawn.  Idempotent."""

        if turn is None:
            return
        with self._memory:
            try:
                self._waiting.remove(turn)
            except ValueError:
                return
            self._memory.notify_all()

    def peak_estimate(self, price: int = 0) -> int:
        """What a frame of cell price ``price`` is reckoned to reach."""

        learned = max(self._peaks) if self._peaks else int(self.settings.peak_default)
        return max(int(price), learned)

    def _total(self) -> int | None:
        if not self._host_total_read:
            self._host_total_read = True
            try:
                self._host_total = self._probe.total()
            except Exception:  # noqa: BLE001 - unknown size, no small-host floor
                self._host_total = None
        return self._host_total

    def _sample_locked(self) -> int | None:
        """Measure the model and every running frame's tree; the model's
        resident memory, or ``None`` when it cannot be read."""

        pids = {}
        for flight in self._flights.values():
            pid = self._render_pid(flight["ident"]) if flight["ident"] else None
            if pid is not None:
                pids[pid] = flight
        try:
            model, trees = self._probe.sample(list(pids))
        except Exception:  # noqa: BLE001 - an unmeasured run keeps its estimates
            return None
        for pid, (rss, peak) in trees.items():
            flight = pids.get(pid)
            if flight is not None:
                flight["rss"] = int(rss)
                flight["peak"] = max(flight["peak"], int(peak), int(rss))
        if model is not None:
            self._record["model_rss_bytes"] = int(model)
        return model

    def _room(self, price: int, refused: bool) -> bool:
        """Whether a frame of cell price ``price`` may start now beside the
        frames drawing.  Held under ``_memory``."""

        if self._running == 0:
            return True
        try:
            available = self._probe.available()
        except Exception:  # noqa: BLE001 - no reading imposes no cap
            available = None
        self._record["available_bytes"] = available
        if available is None:
            return True
        model = self._sample_locked()
        keep = live_render_keep_free(model or 0, self._total(), self.settings)
        self._record["keep_free_bytes"] = int(keep)
        growth = sum(max(0, self.peak_estimate(flight["price"]) - flight["rss"])
                     for flight in self._flights.values())
        line = keep + (self.settings.hysteresis if refused else 0)
        return available - growth - self.peak_estimate(price) >= line

    def _sample_loop(self) -> None:
        while True:
            with self._memory:
                if not self._flights:
                    self._sampler = None
                    return
                self._sample_locked()
                self._memory.wait(self.settings.sample_seconds)

    def record(self) -> dict[str, Any]:
        """What the memory admission did, for the run's summary."""

        with self._memory:
            return {**self._record, "adaptive": self.adaptive,
                    "peak_estimate_bytes": self.peak_estimate(),
                    "peaks_measured": len(self._peaks),
                    "host_bytes": self._host_total,
                    "settings": self.settings.as_dict()}

    @contextmanager
    def admit(self, price: int, turn: object | None = None):
        """Hold one slot and ``price`` bytes of the budget for one frame."""

        if not self.adaptive:
            with self._admit_once(price):
                yield self
            return
        with self._admit_adaptive(price, turn):
            yield self

    @contextmanager
    def _admit_once(self, price: int):
        """The one-shot budget, unchanged (the kill switch's path)."""

        self.acquire()
        try:
            turn = object()
            with self._memory:
                self._waiting.append(turn)
                try:
                    while not (self._waiting[0] is turn and self._fits(price)):
                        self._memory.wait()
                finally:
                    self._waiting.remove(turn)
                    self._memory.notify_all()
                self._reserved += price
                self._running += 1
        except BaseException:
            self.release()
            raise
        try:
            yield self
        finally:
            with self._memory:
                self._reserved -= price
                self._running -= 1
                self._memory.notify_all()
            self.release()

    @contextmanager
    def _admit_adaptive(self, price: int, turn: object | None):
        if turn is None:
            turn = self.ticket()
        acquired = False
        flight: dict[str, Any] | None = None
        try:
            with self._memory:
                if turn not in self._waiting:
                    self._waiting.append(turn)
                # Only the head of the order takes a slot, so a later
                # frame can never hold the slot the oldest is waiting for.
                while self._waiting[0] is not turn:
                    self._memory.wait()
            self.acquire()
            acquired = True
            with self._memory:
                held_since = None
                while not (self._fits(price)
                           and self._room(price, held_since is not None)):
                    if held_since is None:
                        held_since = self._clock()
                        self._record["renders_held"] += 1
                    self._memory.wait(self.settings.poll_seconds)
                if held_since is not None:
                    self._record["longest_hold_seconds"] = max(
                        self._record["longest_hold_seconds"],
                        round(self._clock() - held_since, 3))
                self._waiting.remove(turn)
                flight = {"price": int(price), "ident": threading.get_ident(),
                          "rss": 0, "peak": 0}
                self._flights[turn] = flight
                self._reserved += price
                self._running += 1
                self._record["renders_admitted"] += 1
                self._memory.notify_all()
                if self._sampler is None:
                    self._sampler = threading.Thread(
                        target=self._sample_loop,
                        name="gpuwm-live-render-memory", daemon=True)
                    self._sampler.start()
        except BaseException:
            with self._memory:
                if turn in self._waiting:
                    self._waiting.remove(turn)
                self._memory.notify_all()
            if acquired:
                self.release()
            raise
        try:
            yield self
        finally:
            with self._memory:
                self._flights.pop(turn, None)
                if flight["peak"] > 0:
                    # A render that ran to its end: its measured peak.
                    self._peaks.append(flight["peak"])
                self._reserved -= price
                self._running -= 1
                self._memory.notify_all()
            self.release()


def shared_render_slots():
    """Live frame slots and the exclusive early-render guard for one host."""
    slot = _LiveRenderSlots(_render_concurrency(), _live_memory_budget())
    return slot, _ExclusiveSlots(slot, slot.width)


#: The early render's runner on every route. It holds every live slot
#: (:class:`_ExclusiveSlots`), so it is the one render on the host and
#: takes the whole live budget: ``width=1``, every processor but the
#: forecast's. THE BREAKAGE: the GUI and downscale routes armed it with
#: the ordinary batch runner, which leaves the renderer at half the
#: processors, so their first picture came out slower than ``woof go``'s.
early_render_runner = partial(_run_render, width=1)


def landing_render_wait_seconds() -> float:
    """The wait budget owned by LandingRenders.wait.

    LiveProducts.stop first drains the queue, then waits once more for its
    in-flight frame. FirstProducts.wait can then wait for the early frame.
    Each wait uses DEFAULT_WAIT_SECONDS. The supervisor adds its ordinary
    phase allowance for halting renderers and publishing their receipts.
    """
    return 3.0 * DEFAULT_WAIT_SECONDS


class LandingRenders(FirstProducts):
    """A forecast runner's own early render AND its every-frame render.

    THE DEFECT THIS CLOSES: a forecast runner with no host (``woof go``
    runs it as a subprocess, and so do the ``--wrfinput`` doors) armed
    only the first-frame render on its command line, so every frame after
    the analysis, and every nest's frames, waited for the finalize stage.
    The run-plan route, which hosts the runner in process, draws each
    frame as it lands; this gives a runner that same drawing without a
    host.

    It IS the early render (a :class:`woof.first_products.FirstProducts`,
    so every caller that armed one takes it unchanged) with the
    every-frame render behind it: a frame the first-frame render does not
    claim is queued on :attr:`live`, and :meth:`wait`, which every such
    runner already calls before it writes its report, finishes that queue
    first.  The finalize stage in the parent process reads both records
    off disk (:func:`published_frames`) and draws only what is missing.
    :meth:`halt` is a stopped run.
    """

    def __init__(self, render_plan: Mapping[str, Any], *,
                 report: Callable[[dict], None],
                 report_live: Callable[[dict], None],
                 warn: Callable[..., None],
                 runner: Callable[[Sequence[str]],
                                  subprocess.CompletedProcess] | None = None,
                 own_group: bool = False):
        slot, early_slot = shared_render_slots()
        def report_first(receipt):
            self.live.publish_first(receipt)
            report(receipt)
        super().__init__(render_plan, report=report_first, warn=warn,
                         runner=early_render_runner if runner is None else runner,
                         slot=early_slot, own_group=own_group)
        self.live = LiveProducts(render_plan, report=report_live, warn=warn,
                                 first=_EarlyRenderOnly(self), runner=runner,
                                 slot=slot, own_group=own_group)

    def frame_submitted(self, **event) -> None:
        self.live.frame_submitted(**event)

    def frame_committed(self, *, domain: int, valid_time: Any,
                        path: Any) -> bool:
        claimed = super().frame_committed(domain=domain,
                                          valid_time=valid_time, path=path)
        try:
            self.live.frame_committed(domain=domain, valid_time=valid_time,
                                      path=path, draw=not claimed)
        except Exception:  # noqa: BLE001 - telemetry never fails a run
            pass
        return claimed

    def render_threads(self) -> list:
        return super().render_threads() + self.live.render_threads()

    def wait(self, timeout: float | None = DEFAULT_WAIT_SECONDS
             ) -> dict[str, Any] | None:
        """Finish drawing the frames already written, then the first one's
        receipt, exactly as :meth:`FirstProducts.wait` returns it."""

        self.live.stop()
        return super().wait(timeout)

    def halt(self, timeout: float | None = HALT_WAIT_SECONDS
             ) -> dict[str, Any]:
        """A stopped run: nothing more is drawn by either render.

        BOTH are halted.  THE BREAKAGE: halting the every-frame render
        alone did not end the first-frame render when a stop landed while
        it drew (the analysis frame, early in every run).  Nothing ended
        that render or kept it from publishing after the stop, and the
        every-frame halt then waited on the shared lock it held.

        The every-frame render is closed first and joined last, as
        a downscaled child's own stop does (``halt_renders``): its
        worker may be waiting on the first-frame render, which is ended
        in between (:meth:`FirstProducts.halt`).  The summary is
        :meth:`LiveProducts.halt`'s, with ``ended`` saying whether no
        render of this run is still running.
        """

        self.live.halt(timeout=0)
        ended = super().halt(timeout)
        summary = self.live.halt(timeout)
        ended = ended and not self.live.running
        if ended:
            from woof.first_products import _render_scratch

            # Only the renderers' working stores belong to this cleanup.
            # An explicit render directory can also contain existing pictures.
            for scratch in _render_scratch(self.render_dir):
                shutil.rmtree(fs_path(scratch, descend=True), ignore_errors=True)
        return {**summary, "ended": ended}


def read_receipt(render_dir) -> dict[str, Any] | None:
    try:
        payload = json.loads((Path(render_dir) / LIVE_PRODUCTS_RECEIPT)
                             .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != LIVE_PRODUCTS_SCHEMA:
        return None
    if not isinstance(payload.get("frames"), list):
        return None
    return payload


def _entry_holds(entry: Mapping[str, Any], render_dir: Path) -> bool:
    """Whether one recorded frame is still exactly what was drawn.

    The frame by size and modification time (a rewritten history file
    changes both; hashing every frame of a long run again at finalize
    would cost minutes), every picture by its sha256.
    """

    from woof.first_products import _sha256_file

    # A record that does not say its render finished (one written before records kept the renderer's exit,
    # which said "done" for a render that failed partway as well) cannot show that every product was drawn.
    if entry.get("complete") is not True or entry.get("exit_code", 0) not in (0, None):
        return False
    frame = Path(str(entry.get("frame") or ""))
    try:
        if _frame_identity(frame) != {
                "frame_size": entry.get("frame_size"),
                "frame_mtime_ns": entry.get("frame_mtime_ns")}:
            return False
    except OSError:
        return False
    written = entry.get("written")
    if not isinstance(written, list) or not written:
        return False
    for picture in written:
        if not isinstance(picture, dict):
            return False
        path = render_dir / str(picture.get("name") or "")
        if not Path(fs_path(path)).is_file():
            return False
        if _sha256_file(path) != picture.get("sha256"):
            return False
    return True


def published_frames(frames: Sequence[Path], plan: Mapping[str, Any]
                     ) -> tuple[list[Path], list[Path], str | None]:
    """Split finalize's frames into still-to-draw and drawn-as-it-ran.

    Same contract as :func:`woof.first_products.published_frames`:
    ``(remaining, already, note)``, and a frame is skipped only on a
    record that still holds.
    """

    from woof.first_products import effective_products, section_text

    render_dir = Path(plan["render"])
    receipt = read_receipt(render_dir)
    if receipt is None:
        return list(frames), [], None
    if (effective_products(receipt.get("render_products"))
            != effective_products(plan.get("render_products"))):
        return (list(frames), [],
                "live-render record not used: it was drawn for a different "
                "product set; every frame is being rendered")
    if (section_text(receipt.get("render_section"))
            != section_text(plan.get("render_section"))):
        return (list(frames), [],
                "live-render record not used: its sections were cut along a "
                "different line; every frame is being rendered")
    proven = set()
    stale = 0
    partial = set()
    for entry in receipt["frames"]:
        if not isinstance(entry, dict):
            continue
        if _entry_holds(entry, render_dir):
            proven.add(Path(str(entry["frame"])).resolve())
        elif entry.get("complete") is False:
            partial.add(Path(str(entry["frame"])).resolve())
        else:
            stale += 1
    # A frame drawn whole later supersedes its incomplete record.
    incomplete = len(partial - proven)
    remaining = [frame for frame in frames
                 if Path(frame).resolve() not in proven]
    already = [frame for frame in frames if Path(frame).resolve() in proven]
    if not already and not stale and not incomplete:
        return list(frames), [], None
    clauses = []
    if already:
        clauses.append(f"{len(already)} frame(s) were drawn while the "
                       "forecast ran (pictures verified by digest)")
    if stale:
        clauses.append(f"{stale} recorded frame(s) no longer match and are "
                       "drawn again")
    if incomplete:
        clauses.append(f"{incomplete} frame(s) lost pictures when the "
                       "renderer failed partway and are drawn again")
    return remaining, already, "; ".join(clauses)


_FRAME_TIME = re.compile(
    r"_d(\d{2})_\d{4}-\d{2}-\d{2}_\d{2}[_:](\d{2})[_:](\d{2})$")


#: The maps a quick-look frame's subset makes (woof.experiment
#: QUICKLOOK_DEFAULT_VARS: reflectivity, updraft helicity, 10 m wind, 2 m
#: temperature and moisture, surface pressure and accumulated rain; no
#: sea-level pressure, so no map that overlays it).
QUICKLOOK_PRODUCTS = ("composite_reflectivity", "2m_temperature",
                      "10m_wind_speed_and_direction", "total_qpf")


def is_quicklook_frame(frame) -> bool:
    """Whether ``frame`` is a quick-look subset frame (woof.io.wrfout)."""
    from woof.io.wrfout import is_quicklook_frame as by_folder
    return by_folder(frame)


def quicklook_products(requested: str | None) -> str:
    """The run's asked maps that a quick-look frame can make, else all of those."""
    terms = [p.strip() for p in (requested or "all").split(",") if p.strip()]
    if "all" in terms:
        return ",".join(QUICKLOOK_PRODUCTS)
    chosen = [p for p in QUICKLOOK_PRODUCTS if p in terms]
    return ",".join(chosen or QUICKLOOK_PRODUCTS)


def _grid_and_whole_hour(frame: Path) -> tuple[str, bool] | None:
    """``(grid, whether the frame is on a whole hour)`` from its name."""

    match = _FRAME_TIME.search(Path(frame).name)
    if match is None:
        return None
    return match.group(1), match.group(2) == "00" and match.group(3) == "00"


def _grid_series(frames: Sequence[Path]
                 ) -> tuple[dict[str, list[Path]], dict[str, bool]]:
    """Frames by grid, and whether each grid's series is all whole hours.

    An unrecognised frame name is its own grid and counts as whole-hour,
    which costs time and never a picture.
    """

    grids: dict[str, list[Path]] = {}
    hourly: dict[str, bool] = {}
    for frame in frames:
        parsed = _grid_and_whole_hour(Path(frame))
        grid, whole = parsed if parsed is not None else (Path(frame).name, True)
        grids.setdefault(grid, []).append(Path(frame))
        hourly[grid] = hourly.get(grid, True) and whole
    return grids, hourly


def _grid_of(frame: Path) -> str:
    parsed = _grid_and_whole_hour(Path(frame))
    return parsed[0] if parsed is not None else Path(frame).name


def baseline_frames(remaining: Sequence[Path], drawn: Sequence[Path]
                    ) -> list[Path]:
    """The drawn frames the end-of-run batch needs beside ``remaining``.

    A baseline costs its whole import (about 6 s per frame at 1 km,
    414x402, on the 5070 Ti host) and buys only the windowed family, so
    the caller passes none for a request with no window
    (:func:`requests_windows`), and a drawn frame is passed only when it
    is on the SAME grid as a frame
    still to draw on a whole hour, the only frames a window ends on: a
    frame of another grid is no baseline for it, and a frame between
    hours closes no window.  Every drawn frame of that grid goes, so the
    batch's store holds the grid's whole series and each window it
    closes is folded from all of it; :func:`windowed_passes` then draws
    only what the frames drawn live still lack.

    On a grid that writes history more often than hourly the frames
    between the hours belong to every window: the history writer resets
    ``UP_HELI_MAX`` at each write, so a whole-hour frame holds only the
    last interval of its hour.  A batch handed that grid's remaining
    frames alone either refused their windows for the frames it lacked
    ("missing stored frame(s)") or, holding whole hours only, drew each
    1 h maximum from its last interval.
    """

    closing = {_grid_of(Path(frame)) for frame in remaining
               if _on_whole_hour(Path(frame))}
    return [Path(frame) for frame in drawn if _grid_of(Path(frame)) in closing]


def windowed_request(render_products: Any,
                     windowed_slugs: Callable[[], Any] | None = None
                     ) -> str:
    """The windowed members of one product request, or ``""``.

    The group keywords ``all`` and ``windowed`` stand for the engine's
    whole windowed group and become ``windowed``; a NAMED slug is kept
    only when the engine lists it as windowed.  ``windowed_slugs`` is
    asked for that list only when a named slug needs deciding (it costs
    a catalog listing), and is :func:`engine_windowed_slugs` in the
    finalize stage.  When the list cannot be had the named request is
    returned unchanged: the pass then redraws its instantaneous pictures
    (time, never a picture lost) rather than guessing which are windowed.
    """

    from woof.first_products import effective_products
    from woof.rustwx import product_spec_terms

    tokens = product_spec_terms(effective_products(render_products))
    if any(token.lower() in ("all", "windowed") for token in tokens):
        return "windowed"
    named = [token for token in tokens
             if token.lower() not in ("direct", "derived", "heavy")]
    if not named:
        return ""
    try:
        slugs = None if windowed_slugs is None else windowed_slugs()
    except Exception:  # noqa: BLE001 - the renderer decides instead
        slugs = None
    if not slugs:
        return ",".join(named)
    return ",".join(token for token in named if token in slugs)


def requests_windows(render_products: Any,
                     windowed_slugs: Callable[[], Any] | None = None) -> bool:
    """Whether the request holds any windowed product.

    Earlier frames are imported beside a frame for one reason, the
    windows it closes, so a request with none draws every frame alone.
    The keywords ``all`` and ``windowed`` hold the engine's windows; the
    other group keywords hold none.  A named slug counts when
    ``windowed_slugs`` (the renderer's own listing,
    :func:`catalog_windowed_slugs`, asked only for a named list) lists
    it, through the shared short names.  When the list cannot be had the
    answer is yes, and the frames are imported as they were: time, never
    a picture lost.
    """

    from woof.first_products import effective_products
    from woof.render import RUST_PRODUCT_ALIASES
    from woof.rustwx import GROUP_KEYWORDS, product_spec_terms

    tokens = product_spec_terms(effective_products(render_products))
    if any(token.lower() in ("all", "windowed") for token in tokens):
        return True
    named = [token for token in tokens if token.lower() not in GROUP_KEYWORDS]
    if not named:
        return False
    try:
        slugs = None if windowed_slugs is None else windowed_slugs()
    except Exception:  # noqa: BLE001 - import the frames as before
        slugs = None
    if not slugs:
        return True
    return any(RUST_PRODUCT_ALIASES.get(token, token) in slugs
               for token in named)


def windows_only(render_products: Any,
                 windowed_slugs: Callable[[], Any] | None = None) -> bool:
    """Whether every product the request asks for is a window.

    The keyword ``windowed`` is; any other group keyword draws products
    that are not.  A named slug counts when ``windowed_slugs`` (asked
    only for a named list, :func:`catalog_windowed_slugs`) lists it,
    through the shared short names.  When the list cannot be had the
    answer is no, and every frame is drawn as it was: time, never a
    picture lost.
    """

    from woof.first_products import effective_products
    from woof.render import RUST_PRODUCT_ALIASES
    from woof.rustwx import GROUP_KEYWORDS, product_spec_terms

    tokens = product_spec_terms(effective_products(render_products))
    if not tokens or any(token.lower() in GROUP_KEYWORDS
                         and token.lower() != "windowed" for token in tokens):
        return False
    named = [token for token in tokens if token.lower() != "windowed"]
    if not named:
        return True
    try:
        slugs = None if windowed_slugs is None else windowed_slugs()
    except Exception:  # noqa: BLE001 - draw every frame as before
        slugs = None
    if not slugs:
        return False
    return all(RUST_PRODUCT_ALIASES.get(token, token) in slugs
               for token in named)


def catalog_windowed_slugs() -> frozenset[str]:
    """The slugs the renderer's own listing calls windowed, with no frame.

    Read off the ``local_run`` rows of
    :func:`woof.runplan.render_catalog` (``rw_wrfbatch
    --list-products``, cached per process), so nothing is imported to
    ask it.  Empty when no renderer answers.
    """

    from woof.runplan import render_catalog

    local = render_catalog().get("local_run") or {}
    return frozenset(str(row.get("name")) for row in local.get("products") or ()
                     if isinstance(row, dict) and row.get("kind") == "windowed")


def closes_a_window(frame: Path, series: Sequence[Path]) -> bool:
    """Whether a window can end on ``frame``, one frame of ``series``.

    A whole-hour frame after its grid's first, the rule
    :func:`windowed_passes` asks its frames by: every window of the
    catalog ends on a whole hour, and no window ends on the first frame
    of a series.  An unrecognised frame name counts as closing one, which
    costs time and never a picture.
    """

    frame = Path(frame)
    parsed = _grid_and_whole_hour(frame)
    if parsed is None:
        return True
    grid, whole = parsed
    if not whole:
        return False
    first = min((Path(member).name for member in series
                 if _grid_of(Path(member)) == grid), default=frame.name)
    return frame.name > first


def engine_windowed_slugs(frame: Path, *, beside=None,
                          prefix: str | None = None) -> frozenset[str]:
    """The slugs the engine's catalog lists as windowed, from one frame.

    The store-aware listing prints every windowed slug of this build with
    kind ``windowed`` whatever the frame holds (a one-frame store lists
    them all as excluded), so one import of the smallest frame answers
    it; no copy of the vocabulary is kept in Python.

    ``beside`` and ``prefix`` place the import's working store as
    :func:`woof.render.scratch_store` does: a render stage passes its
    own delivery and token so the store is one its closing sweep owns.
    Omitted, the store sits beside the frame under this process's prefix.
    """

    from woof import rustwx
    from woof.render import scratch_store

    renderer = rustwx.find_renderer()
    if renderer is None:
        return frozenset()
    with scratch_store(Path(frame).parent if beside is None else beside,
                       prefix=prefix) as store:
        rows, _summary = rustwx.catalog_rows(renderer, (Path(frame),),
                                             store_root=store)
    return frozenset(row[0] for row in rows if row[1] == "windowed")


def engine_draws_windows(frames: Sequence[Path], *, beside=None,
                         prefix: str | None = None) -> bool | None:
    """Whether the engine draws any window from ``frames``, or ``None``.

    Asked of the engine's store-aware listing over the frames
    themselves, so the answer is what they hold, in the engine's own
    vocabulary: ``True`` when some windowed row is renderable, ``False``
    when none is (the frames hold no field a window folds, or their time
    axis defines no window).  ``None`` is a question that could not be
    put (no frame, no renderer, a listing that failed); a caller then
    draws as it would have without asking.  The listing imports every
    frame it is given, so it is for frames that are cheap to import,
    such as a cycle's 2-D boundaries.  ``beside`` and ``prefix`` place
    the working store as :func:`engine_windowed_slugs` does.
    """

    from woof import rustwx
    from woof.render import scratch_store

    frames = [Path(frame) for frame in frames]
    if not frames:
        return None
    renderer = rustwx.find_renderer()
    if renderer is None:
        return None
    try:
        with scratch_store(frames[0].parent if beside is None else beside,
                           prefix=prefix) as store:
            rows, _summary = rustwx.catalog_rows(renderer, frames,
                                                 store_root=store)
    except Exception:  # noqa: BLE001 - the caller draws as before
        return None
    return any(row[1] == "windowed" and row[2] == "renderable"
               for row in rows)


def windowed_passes(frames: Sequence[Path], drawn: Sequence[Path],
                    render_products: Any, *,
                    windowed_slugs: Callable[[Path], Any] | None = None,
                    recorded: Mapping[Path, Any] | None = None,
                    live_held: Mapping[Path, Any] | None = None,
                    prior_frames: Sequence[Path] = ()
                    ) -> list[tuple[list[Path], list[Path], str]]:
    """What finalize still draws of the windowed family, grid by grid.

    One ``(wanted, context, products)`` per grid.  Windows end on the
    grid's WHOLE-HOUR frames and are drawn over EVERY frame of the grid:
    on a 15-minute nest the history writer resets ``UP_HELI_MAX`` and the
    other 1 h maxima at each write, so a window drawn over the whole
    hours alone held only the last quarter hour of each hour, and the
    engine folds every frame inside a window.  Leaving such a grid out
    altogether, as this once did, left the 3 km nest of a 12/3 km run
    with no ``qpf_1h`` at all (0 of 49 frames) while its 12 km parent had
    12.

    ``wanted`` are the whole-hour frames after the grid's first (no
    window ends on the first frame of a series, and a launch that draws
    nothing is a failed launch) that were drawn live (``drawn``), each
    beside one hour of its series at most; the end-of-run batch draws
    every other frame beside the grid's whole series
    (:func:`baseline_frames`).  Less:

    - every frame whose recorded pictures (``recorded``, frame to
      product folders, :func:`recorded_products`) already hold each
      windowed product a NAMED request asks for: the live pass closed
      those windows while the forecast ran, and drawing them again is
      time spent on the same pictures;
    - every frame whose live render already held every earlier frame of
      its grid (``live_held``, frame to the frames its render imported,
      :func:`live_held`).  Every window the series closes there, the
      live render closed, which is how a group request, naming no one
      window, skips a frame.

    ``context`` is every other frame of the grid, the frames between its
    hours included.  An unrecognised frame name is its own grid and
    counts as whole-hour, which costs time and never a picture.

    ``prior_frames`` is a resumed run's history saved before its
    checkpoint (:func:`woof.restart_render.history_before_restart`).
    It joins each grid's series as context only, so the first whole
    hours after the checkpoint close their windows; it is never wanted.

    ``products`` is only the windowed part of the run's request
    (:func:`windowed_request`), so no instantaneous picture is drawn
    again; a request with no windowed member has no pass.
    ``windowed_slugs`` takes one frame and returns the engine's windowed
    slugs (:func:`engine_windowed_slugs`); it is called at most once, on
    the smallest frame of the passes, and only for a named list.
    """

    targets = {Path(frame).resolve() for frame in frames}
    prior_frames = [Path(frame) for frame in prior_frames
                    if Path(frame).resolve() not in targets]
    grids, _hourly = _grid_series([*prior_frames, *frames])
    drawn_set = {Path(frame).resolve() for frame in drawn}
    recorded = recorded or {}
    live_held = live_held or {}
    candidates = []
    for members in grids.values():
        members = sorted(members, key=lambda f: f.name)
        whole = [f for f in members if _on_whole_hour(f)]
        if len(whole) < 2:
            continue
        wanted = [f for f in whole[1:] if f.resolve() in drawn_set]
        if wanted:
            candidates.append((wanted, members))
    if not candidates:
        return []

    def slugs():
        if windowed_slugs is None:
            return None
        smallest = min((f for wanted, _ in candidates for f in wanted),
                       key=_frame_size)
        return windowed_slugs(smallest)

    products = windowed_request(render_products, slugs)
    if not products:
        return []
    needed = _named_request(products)
    passes = []
    for wanted, members in candidates:
        keep = [f for f in wanted
                if not (needed and needed <= recorded.get(f.resolve(),
                                                          frozenset()))
                and not _held_every_earlier(live_held, members, f)]
        if not keep:
            continue
        context = [f for f in members if f not in keep]
        passes.append((keep, context, products))
    return passes


def _held_every_earlier(live_held: Mapping[Path, Any],
                        members: Sequence[Path], frame: Path) -> bool:
    """Whether ``frame``'s live render imported every frame of its grid up to it."""

    imported = live_held.get(frame.resolve())
    if imported is None:
        return False
    imported = {Path(path).resolve() for path in imported}
    return all(f.resolve() in imported for f in members if f.name <= frame.name)


def _on_whole_hour(frame: Path) -> bool:
    parsed = _grid_and_whole_hour(Path(frame))
    return True if parsed is None else parsed[1]


def _named_request(products: str) -> frozenset[str] | None:
    """The slugs a request names, or ``None`` when a group keyword is in it."""

    from woof.render import RUST_PRODUCT_ALIASES
    from woof.rustwx import GROUP_KEYWORDS, product_spec_terms

    tokens = product_spec_terms(products)
    if not tokens or any(token.lower() in GROUP_KEYWORDS for token in tokens):
        return None
    # The folder a picture is filed under is the engine's slug, which is
    # what a shared short name (``t2``) is drawn as.
    return frozenset(RUST_PRODUCT_ALIASES.get(token, token) for token in tokens)


def recorded_products(render_dir) -> dict[Path, frozenset[str]]:
    """Each frame drawn live, and the product folders its pictures hold.

    Read off ``live-products.json``; a picture outside the nested layout
    (``<grid>/<product>/<valid-day>/<file>``) names no product and is not
    counted.  The record is trusted only for frames
    :func:`published_frames` has verified by digest, which is how
    :func:`windowed_passes` uses it.
    """

    receipt = read_receipt(render_dir)
    if receipt is None:
        return {}
    held: dict[Path, set[str]] = {}
    for entry in receipt["frames"]:
        if not isinstance(entry, dict) or not entry.get("frame"):
            continue
        folders = held.setdefault(Path(str(entry["frame"])).resolve(), set())
        for picture in entry.get("written") or ():
            if not isinstance(picture, dict):
                continue
            parts = Path(str(picture.get("name") or "")).parts
            if len(parts) >= 4:
                folders.add(parts[-3])
    return {frame: frozenset(folders) for frame, folders in held.items()}


def live_held(render_dir) -> dict[Path, frozenset[Path]]:
    """Each frame drawn live, and every frame its live render imported.

    The frame itself and the baselines drawn beside it (the record's
    ``context``), read off ``live-products.json``.  Trusted, like
    :func:`recorded_products`, only for frames :func:`published_frames`
    has verified by digest.
    """

    receipt = read_receipt(render_dir)
    if receipt is None:
        return {}
    held: dict[Path, frozenset[Path]] = {}
    for entry in receipt["frames"]:
        if not isinstance(entry, dict) or not entry.get("frame"):
            continue
        frame = Path(str(entry["frame"])).resolve()
        context = entry.get("context")
        context = context if isinstance(context, list) else []
        held[frame] = frozenset(
            {frame, *(Path(str(path)).resolve() for path in context if path)})
    return held


#: A product name that is its own folder in the nested layout.  ``var:``
#: generics and the storeless ``mesh:``/``meshdiff:``/``xsec:`` terms
#: are spelled differently on disk and are not checked.
_PRODUCT_FOLDER = re.compile(r"[A-Za-z0-9_]+\Z")


def unpictured_products(render_dir, render_products
                        ) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """``([(product, grids without one of its pictures)], every grid)``.

    Read off the pictures on disk under the nested layout, so it is what
    a reader opening the folder finds, whichever render drew what.  A
    product is reported only when some grid HAS pictures of it and
    another has none.  A product no grid drew is each render's own skip
    line to name, with the engine's reason, and a second closing line
    naming it would say the same thing twice.  A named request is
    checked for the products it names; a group keyword names none, so
    every product some grid drew is checked.  Nothing is reported for a
    folder with no nested picture at all.
    """

    from woof.first_products import effective_products
    from woof.render import RUST_PRODUCT_ALIASES
    from woof.rustwx import GROUP_KEYWORDS, product_spec_terms

    render_dir = Path(render_dir)
    held: dict[str, set[str]] = {}
    for picture in iter_rendered(render_dir):
        parts = picture.relative_to(render_dir).parts
        if len(parts) >= 4:
            held.setdefault(parts[0], set()).add(parts[-3])
    if not held:
        return [], []
    drawn = set().union(*held.values())
    tokens = product_spec_terms(effective_products(render_products))
    group = any(token.lower() in GROUP_KEYWORDS for token in tokens)
    named = [RUST_PRODUCT_ALIASES.get(token, token) for token in tokens
             if token.lower() not in GROUP_KEYWORDS
             and _PRODUCT_FOLDER.match(RUST_PRODUCT_ALIASES.get(token, token))]
    rows = []
    for product in dict.fromkeys([*named, *(sorted(drawn) if group else ())]):
        if product not in drawn:
            continue
        missing = sorted(grid for grid, products in held.items()
                         if product not in products)
        if missing:
            rows.append((product, missing))
    return rows, sorted(held)


def unpictured_note(render_dir, render_products) -> str | None:
    """The closing line naming what one grid has and another has none of.

    Each render names what IT skipped and calls that no failure, which is
    true of one invocation and hid that a whole grid had none: the 3 km
    nest of a 12/3 km run ended with no ``qpf_1h`` at all under a note
    that read as the first frame's skip, while its 12 km parent had
    twelve.  This line is about the finished folder, one clause per set
    of grids.
    """

    rows, _grids = unpictured_products(render_dir, render_products)
    if not rows:
        return None
    by_grids: dict[tuple[str, ...], list[str]] = {}
    for product, missing in rows:
        by_grids.setdefault(tuple(missing), []).append(product)
    clauses = [f"{', '.join(products)} on {', '.join(grids)}"
               for grids, products in by_grids.items()]
    summary = Path(render_dir) / "render-summary.json"
    return ("note: drawn on some grids and not on others: no picture of "
            + "; of ".join(clauses)
            + f".  The engine's reason for each skip is in {summary}")


def _frame_size(frame: Path) -> int:
    try:
        return Path(fs_path(frame)).stat().st_size
    except OSError:
        return 1 << 62


__all__ = ["DEFAULT_WAIT_SECONDS", "HALT_WAIT_SECONDS",
           "LIVE_PRODUCTS_RECEIPT", "LIVE_PRODUCTS_SCHEMA", "LandingRenders",
           "LiveProducts", "baseline_frames", "engine_draws_windows",
           "engine_windowed_slugs", "live_held", "live_render_requested",
           "published_frames", "read_receipt", "recorded_products",
           "requests_windows", "unpictured_note", "unpictured_products",
           "windowed_passes", "windowed_request"]
