"""What a run will write to disk, projected before it starts.

A run's disk is the source files it downloads, the files its preparation
writes from them, its history files, its checkpoints and its rendered
pictures. History prices the writer's selected variable shapes and each
domain's output window. Checkpoints price their independent state inventory.
The pictures scale with the history frames, since each picture
has a fixed size in pixels, but its compressed size varies with the
product and horizontal grid. Checkpoint bytes per cell were read off the
files a real run wrote (see :data:`MEASURED`). The download and the
preparation are priced by :mod:`woof.download_budget` from sizes
measured on real downloads and preparations: an 18 hour HRRR run
downloads about 22 GB, more than its history, and an event page layout
projected without them wrote nearly twice its projection.  A preparation
that composes through the mapped engine also stages its decoded frame
stream in a scratch folder while it runs, sized by the SOURCE grid
(:func:`woof.download_budget.compose_scratch_estimate`): a GDPS 48 hour
window stages about 82 GB for any domain that reaches the globe's stored
longitude cut, and about 5 GB elsewhere, where the atmospheric window
crops its pressure levels.  The stream is gone before
the forecast writes its history, so the peak is the download and the
preparation plus the larger of the stream and what the forecast writes.

``woof run-plan`` refuses, before it downloads anything, a run whose
projection is larger than the free space on the disk that will hold it,
because such a run fills the disk partway and ends with nothing usable,
and it can stop other work on that disk.  ``woof downscale`` refuses a
child the same way before the child starts
(:func:`projected_child_bytes`).
"""

from __future__ import annotations

import functools
import json
import math
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

#: Bytes one checkpoint writes per 3-D cell of a nest (189.3 measured).
CHECKPOINT_BYTES_PER_CELL = 190.0
#: Bytes one checkpoint writes per 3-D cell of the outermost grid, which
#: also carries the lateral boundary state (237.8 measured).
ROOT_CHECKPOINT_BYTES_PER_CELL = 238.0
#: The pictures the renderer's ``variables`` keyword
#: (:data:`woof.rustwx.VARIABLES_KEYWORD`) adds to each frame: one per
#: stored two-dimensional variable no named product draws.  The 3 km
#: parent and its 250 m child each drew 137 with the full history preset.
STORED_VARIABLE_PICTURES = 137
#: Citation for the independent run-coverage and price-ceiling checks,
#: not a pricing rate; runtime prices come from the per-product table.
PICTURES_MEASURED = (
    "file sizes of every picture a 3 hour 3 km HRRR run drew with all "
    "products, and two 250 m children of it drew with all products and with "
    "three products at history every minute, beside the count projected for "
    "each frame, on an RTX 5070 Ti on 2026-09-27; per-frame sizes are kept in "
    "tools/wiki_seed/runs/bytes-per-picture/sizes.json")
#: Where the numbers come from.
MEASURED = (
    "file sizes of a 2 hour 12, 3 and 1 km run (170x170, 336x336 and "
    "408x402 columns, 49 levels, Morrison, history every hour on the 12 km "
    "grid and every 15 minutes on the nests, every checkpoint kept) on an "
    "RTX 5070 Ti on 2026-09-25; per-file sizes are kept in "
    "tools/wiki_seed/runs/bytes-per-cell/sizes.json")

GIB = 1024 ** 3

#: Lossless compressed history frames (the default since woof 2.9.10:
#: :func:`woof.io.wrfout.resolve_history_compression` ``deflate``, byte
#: shuffle + zlib 4) are priced at this fraction of the frame's classic
#: layout.  Measured on all 105 frames of a 12-3-1 km GFS run
#: (2026-10-28):
#: the least compressible frame stored 1/2.74 = 0.365 of its classic bytes
#: (12 km, mean 1/3.04), 3 km 1/3.04 at worst, 1 km 1/3.36 at worst
#: (mean 1/3.54), 1 km quick look 1/3.90 at worst.  0.45 keeps a quarter
#: more than the worst frame measured, for weather that compresses less
#: (widespread hydrometeors), and still stops pricing a compressed run at
#: 2.7-3.9x the disk it fills, which refused runs that fit.
COMPRESSED_HISTORY_FRACTION = 0.45


def history_stored_fraction() -> float:
    """What one history frame's classic layout bytes become on the disk.

    :data:`COMPRESSED_HISTORY_FRACTION` when this process writes lossless
    compressed frames (the default; ``--history-compression none`` or
    ``GPUWM_HISTORY_COMPRESSION=none`` writes classic frames, and so does
    the ``python`` wrfout engine, which stores NETCDF4_CLASSIC uncompressed),
    else 1.  ``woof`` applies ``--history-compression`` to the process
    environment before any projection, so the price follows the writer.
    """
    from woof.io.wrfout import resolve_history_compression, resolve_wrfout_engine

    try:
        compressed = (resolve_history_compression() == "deflate"
                      and resolve_wrfout_engine() == "rust")
    except ValueError:
        # An unknown setting is refused by the writer itself; price it whole.
        return 1.0
    return COMPRESSED_HISTORY_FRACTION if compressed else 1.0


def history_frame_bytes(cfg, selection=None, *, include_reflectivity=True) -> int:
    """One frame's selected writer inventory, including its container header.

    Selection uses the writer's own predicate. A dropped volume costs no
    disk, and surface, soil and staggered fields retain their own extents.
    Checkpoints have an independent inventory and are never trimmed here.
    """
    from woof.io.history_layout import history_frame_bytes as frame_bytes

    return frame_bytes(cfg, selection, include_reflectivity=include_reflectivity)


def picture_table_path() -> Path:
    return Path(__file__).with_name("data") / "picture-bytes.v1.json"


@functools.lru_cache(maxsize=1)
def _picture_table() -> dict[str, Any]:
    return json.loads(picture_table_path().read_text(encoding="utf-8"))


def projected_picture_bytes(nx: int, ny: int, run_seconds: float,
                            interval_s: float, products: str = "all") -> int:
    """Default-size pictures from the shared product and grid-size model."""
    return _picture_projection(nx, ny, run_seconds, interval_s, products)[0]


def _picture_projection(nx, ny, run_seconds, interval_s, products, *, history_frames=None,
                        after_seconds=None, begin_seconds=0.0, end_seconds=None):
    """Price each selected product on the frames that can draw it.

    Forecasts and children share this calculation. The renderer's catalog
    owns group membership when available; the measured table supplies
    groups when it is not. Sizes interpolate across the table's horizontal
    grid brackets and hold the endpoints outside them, since image pixel
    dimensions stay fixed. This is an estimate, not a byte upper bound.

    ``after_seconds`` is how long the grid had run at the checkpoint a
    resumed run starts from: a window that closed by then was drawn
    before it, and ``history_frames`` are the frames still to come.
    """
    table = _picture_table()
    selected = _picture_products(products, run_hours=run_seconds / 3600)
    per_frame = None if selected is None else sum(row[2] for row in selected.values())
    if selected is None:
        selected = _picture_products(products, run_hours=run_seconds / 3600, measured=True)
    lo, hi = table["columns"]
    weight = min(1.0, max(0.0, (int(nx) * int(ny) - lo) / (hi - lo)))
    regular_frames = _frames(run_seconds, interval_s)
    frames = regular_frames if history_frames is None else int(history_frames)
    if frames <= 0:
        return 0, per_frame
    total = 0.0
    for name, (kind, first_hour, copies) in sorted(selected.items()):
        fallback = table["generic_bytes"] if kind == "generic" else [table["fallback_bytes"]] * 2
        small, large = table["products"].get(name, [kind, first_hour, *fallback])[2:]
        count = frames
        if kind == "windowed":
            if interval_s <= 0:
                continue
            # The renderer now uses every sub-hourly baseline, but each
            # window still ends on a whole hour after the initial frame.
            cadence = round(interval_s * 1000)
            begin = round(begin_seconds * 1000)
            hour = 3_600_000
            common = math.gcd(cadence, hour)
            if begin % common:
                count = 0
            else:
                # Solve begin + k*cadence == 0 (mod one hour), so a
                # phased history window prices only whole-hour frames.
                modulus = hour // common
                k = (0 if modulus == 1 else
                     (-begin // common * pow(cadence // common, -1, modulus)) % modulus)
                origin = begin + k * cadence
                step = math.lcm(cadence, hour)
                lower = max(hour, round(first_hour * hour), begin)
                if after_seconds is not None:
                    lower = max(lower, math.floor(after_seconds * 1000) + 1)
                upper = round(min(run_seconds, end_seconds if end_seconds is not None
                                  else run_seconds) * 1000)
                count = max(0, (upper - origin) // step
                            - max(0, -(-(lower - origin) // step)) + 1)
            # A child also writes its final step off the regular cadence.
            if (frames > regular_frames and run_seconds % 3600 == 0
                    and run_seconds >= max(3600, first_hour * 3600)):
                count += 1
        total += copies * count * (small + weight * (large - small))
    return math.ceil(total * table["headroom"]), per_frame


def _frames(run_seconds: float, interval_s: float) -> int:
    if not interval_s or interval_s <= 0:
        return 1
    return 1 + int(math.floor(run_seconds / interval_s + 1e-9))


def _live_seconds(exp, domain, run_seconds: float) -> float | None:
    """How long ``domain`` runs: from its own start to the run's end.

    A nest that starts late writes its first history frame when it starts
    and one per interval after that (``DomainClock.history_due`` counts from
    the domain's start), so it writes nothing for the time before it.
    Counting the whole run for it charged a nest starting 540 s into a
    600 s run for 11 frames where it writes 2, and refused runs that fit.
    ``None`` is a domain that starts after the run ends and writes nothing.
    A layout without per-domain starts runs every domain for the whole run.
    """
    offset = getattr(exp, "domain_start_offset_exact", None)
    start = 0.0 if offset is None else float(offset(int(domain.grid_id)))
    if start > run_seconds:
        return None
    return run_seconds - max(0.0, start)


def _frames_after_resume(exp, domain, run_seconds: float,
                         resume_seconds: float) -> int:
    """History frames ``domain`` still writes when a run resumes at ``resume_seconds``.

    A domain already live at the checkpoint keeps its history clock, which
    counts from its own start (``DomainClock.history_due``), and does not
    write the frame at the checkpoint instant again: the restore marks it
    committed.  A domain that starts after the checkpoint writes all of its
    frames, its first one included (:func:`_live_seconds`).  A layout
    without per-domain starts starts every domain with the run.
    """
    return history_frames(exp, domain, run_seconds, after_seconds=resume_seconds)


def history_frames(exp, domain, run_seconds: float | None = None, *,
                   after_seconds: float | None = None,
                   full_interval: float | None = None) -> int:
    """Count the writer's domain-local alarms, including begin/end and resume.

    With ``full_interval`` (an [output] full_history_interval_s), count only
    the alarms the writer gives the full history; the others are the
    between-hours subset.

    The first alarm rounds onto the same step lattice as DomainClock.
    Counting the closed interval algebraically avoids materializing a
    schedule for long runs. A resumed frame at the checkpoint is committed.
    """
    stop = Fraction(str(exp.run_seconds if run_seconds is None else run_seconds))
    offset = getattr(exp, "domain_start_offset_exact", None)
    start = Fraction(0) if offset is None else offset(int(domain.grid_id))
    start = max(Fraction(0), Fraction(start))
    if start > stop:
        return 0
    begin, window_end = _history_window_seconds(exp, domain)
    end = stop - start
    if window_end is not None:
        end = min(end, window_end)
    if end < begin:
        return 0
    interval = Fraction(str(domain.history_interval_s))
    if interval <= 0:
        return int(after_seconds is None or start + begin > Fraction(str(after_seconds)))
    after = None if after_seconds is None else Fraction(str(after_seconds)) - start

    def count(origin, period, limit):
        last = (limit - origin) // period
        # Inclusive begin and exclusive resume, on the same alarm lattice.
        first = max(0, -((origin - begin) // period))
        if after is not None:
            first = max(first, (after - origin) // period + 1)
        return max(0, int(last - first + 1))

    run = getattr(domain, "run", None)
    interval2 = Fraction(str(getattr(run, "history_interval2_s", 0)))
    if full_interval is not None:
        # Only the alarms on a full-history lead: the writer's own rule
        # (HistorySelection.for_lead, leads from the run's START_DATE).
        full = Fraction(str(full_interval))

        def on_full(origin, period, limit):
            last = (limit - origin) // period
            first = max(0, -((origin - begin) // period))
            if after is not None:
                first = max(first, (after - origin) // period + 1)
            hits = 0
            for k in range(int(first), int(last) + 1):
                lead = start + origin + k * period
                steps = lead / full
                if abs(float(steps - round(steps))) * float(full) < 0.5:
                    hits += 1
            return hits
        if interval2 <= 0:
            return on_full(begin, interval, end)
        change = Fraction(str(run.history_interval_change_s))
        return (on_full(begin, interval, min(end, change))
                + on_full(change + interval2, interval2, end))
    if interval2 <= 0:
        return count(begin, interval, end)
    change = Fraction(str(run.history_interval_change_s))
    return (count(begin, interval, min(end, change))
            + count(change + interval2, interval2, end))


def quicklook_frames(exp, domain, run_seconds: float | None = None, *,
                     after_seconds: float | None = None) -> int:
    """How many of ``domain``'s history alarms write only the quick-look subset.

    A domain with a quick-look stream (``quicklook = {...}``: its
    DomainConfig carries ``full_history_interval_s`` and
    ``quicklook_output``) rings its history alarm at the quick-look cadence.
    An alarm whose time since the domain's first history instant is a whole
    number of full intervals writes an ordinary full frame; every other one
    writes the quick-look subset to ``quicklook/``
    (:func:`woof.io.wrfout.quicklook_selection`).  0 for a domain without
    the stream, and for one whose run changes its history interval partway
    (``history_interval2_s``), which stays priced as full frames.

    Breakage it prevents: a WOOF run (GFS, 12-3-1 km, 1164 x 714 x 49 on
    the 1 km grid, 6 h, full frames hourly, quick-look every minute) was
    priced at 361 full frames on its 1 km grid, 1607.4 GiB of history, and
    refused on a disk with 1611.1 GiB free; it writes about 213 GiB.
    """
    if (getattr(domain, "full_history_interval_s", None) is None
            or getattr(domain, "quicklook_output", None) is None):
        return 0
    run = getattr(domain, "run", None)
    if Fraction(str(getattr(run, "history_interval2_s", 0) or 0)) > 0:
        return 0
    interval = Fraction(str(domain.history_interval_s))
    full = Fraction(str(domain.full_history_interval_s))
    if interval <= 0 or full <= interval or (full / interval).denominator != 1:
        return 0
    stop = Fraction(str(exp.run_seconds if run_seconds is None else run_seconds))
    offset = getattr(exp, "domain_start_offset_exact", None)
    start = Fraction(0) if offset is None else offset(int(domain.grid_id))
    start = max(Fraction(0), Fraction(start))
    if start > stop:
        return 0
    begin, window_end = _history_window_seconds(exp, domain)
    end = stop - start
    if window_end is not None:
        end = min(end, window_end)
    if end < begin:
        return 0
    # The alarms are begin + k * interval, k = first .. last (history_frames'
    # own lattice); alarm k is full when k is a multiple of full / interval.
    last = (end - begin) // interval
    first = 0
    if after_seconds is not None:
        after = Fraction(str(after_seconds)) - start
        first = max(first, (after - begin) // interval + 1)
    if last < first:
        return 0
    every = int(full / interval)
    on_full = last // every - (-(-first // every)) + 1
    return int(last - first + 1) - max(0, int(on_full))


def _history_window_seconds(exp, domain):
    from woof.core.clock import _history_window_ticks

    exact_step = getattr(exp, "dt_exact", None)
    step = (exact_step(int(domain.grid_id)) if exact_step is not None
            else Fraction(str(getattr(domain.run, "dt", 1.0))))
    # Legacy layout-shaped callers do not carry the optional windows.
    from types import SimpleNamespace
    window = SimpleNamespace(
        grid_id=domain.grid_id,
        history_begin_s=getattr(domain, "history_begin_s", 0.0),
        history_end_s=getattr(domain, "history_end_s", None))
    begin_ticks, end_ticks = _history_window_ticks(
        window, step.numerator, step.denominator)
    return (Fraction(begin_ticks, step.denominator),
            None if end_ticks is None else Fraction(end_ticks, step.denominator))


def _seconds_before_resume(exp, domain, resume_seconds: float) -> float | None:
    """How long ``domain`` had run at the checkpoint, or None when it starts after it.

    A domain that starts after the checkpoint draws every picture of its
    own; one live at it drew those whose frames and windows closed by then.
    """
    offset = getattr(exp, "domain_start_offset_exact", None)
    start = 0.0 if offset is None else float(offset(int(domain.grid_id)))
    if start > resume_seconds:
        return None
    return resume_seconds - max(0.0, start)


def projected_run_bytes(exp, *, keep_checkpoints: int | None,
                        fetch: Mapping[str, Any] | None, chain: str | None,
                        render: bool, render_products: str = "all",
                        download_present_bytes: int = 0,
                        resume_seconds: float | None = None) -> dict[str, Any]:
    """Every byte ``exp`` will write: download, preparation, history, checkpoints and pictures.

    The domain with the lowest grid id is the outermost grid.

    ``keep_checkpoints`` is how many complete checkpoint sets the run keeps
    (``None`` keeps every one).  A new set is written whole before an older
    one goes, so the peak holds one more set than the run keeps.

    ``fetch`` is the request the run downloads with (a ``[fetch]`` table,
    or ``None`` when it downloads nothing) and ``chain`` the preparation it
    takes (``None`` when it prepares nothing).  Both are required, so no
    caller can leave them out by omission, which is how they came to be
    left out of every projection before.  ``render`` is whether the run
    draws pictures at all, and is required for the same reason: a run
    told to draw none, or a route whose default draws none, was charged
    for every picture anyway, and refused for a disk it would never have
    filled.  ``download_present_bytes`` is what already lies in the
    download directory, and is not written again. ``render_products``
    is the route's resolved product request at the default image size.

    ``resume_seconds`` is the model time of the checkpoint a resumed run
    starts from.  The history frames, checkpoint sets and pictures written
    before it are already on the disk, so only what comes after it is
    priced, pictures on the same product table as a whole run: a 24 hour
    run resumed at hour 23 was charged for all 25 frames and 24 checkpoint
    sets, 3.5 GiB where it writes 0.14 GiB, and refused on a disk it fits.
    A checkpoint at time zero keeps the whole-run price, since the run can
    still write its first frame there.
    A part the table cannot price is named in ``unpriced`` and left out of
    ``total_bytes``.
    """
    from woof import download_budget

    run_seconds = float(exp.run_seconds)
    restart = float(getattr(exp, "restart_interval_s", 0.0) or 0.0)
    written = int(math.floor(run_seconds / restart + 1e-9)) if restart > 0 else 0
    resumed = resume_seconds is not None and float(resume_seconds) > 0.0
    if resumed and restart > 0:
        written = max(0, written - int(math.floor(float(resume_seconds) / restart + 1e-9)))
    held = written if keep_checkpoints is None else min(written, int(keep_checkpoints) + 1)
    rows, history, checkpoints, pictures = [], 0, 0.0, 0.0
    root_id = min((int(domain.grid_id) for domain in exp.domains), default=1)
    stored = history_stored_fraction()
    for domain in exp.domains:
        run = domain.run
        cells = int(run.nx) * int(run.ny) * int(run.nz)
        # History and pictures from this domain's own start; checkpoints
        # below stay whole-run, since a set carries a domain that has not
        # started yet as well.
        live = _live_seconds(exp, domain, run_seconds)
        frames = history_frames(exp, domain, run_seconds)
        per_cell = (ROOT_CHECKPOINT_BYTES_PER_CELL if int(domain.grid_id) == root_id
                    else CHECKPOINT_BYTES_PER_CELL)
        drawn_before = None
        if resumed:
            frames = _frames_after_resume(exp, domain, run_seconds, float(resume_seconds))
            drawn_before = _seconds_before_resume(exp, domain, float(resume_seconds))
        from woof.io.history_selection import resolve
        selection = resolve(getattr(exp, "output", None), getattr(domain, "output", None))
        frame_bytes = history_frame_bytes(run, selection)
        # A quick-look stream's subset frames (quicklook_frames) are priced
        # at the subset's own inventory; the rest are the domain's frames.
        quick = quicklook_frames(exp, domain, run_seconds,
                                 after_seconds=(float(resume_seconds) if resumed else None))
        regular = frames - quick
        h = frame_bytes * regular
        if selection.subhourly and regular:
            # Frames between full-history leads carry only the 2-D subset
            # (lane/upp-history-subset). Breakage it prevents: the HRRR
            # recipe's 15-minute cadence priced every frame as full history,
            # 147.7 GiB for a 2 h full-CONUS run that writes about a third of
            # that, and refused on a disk it fits (2.9.7 straggler fold, box J).
            interval = selection.full_history_interval_s
            full = min(regular, history_frames(
                exp, domain, run_seconds, full_interval=interval,
                after_seconds=(float(resume_seconds) if resumed else None)))
            between = history_frame_bytes(run, selection.for_lead(interval / 2.0))
            h = frame_bytes * full + between * (regular - full)
        quick_frame_bytes = (history_frame_bytes(run, domain.quicklook_output)
                             if quick else 0)
        h += quick_frame_bytes * quick
        c = cells * per_cell * held
        begin, end = _history_window_seconds(exp, domain)
        # A tick-zero analysis writes the zero reflectivity array. Only a
        # later activation precedes its domain's first reflectivity producer.
        start = run_seconds - live if live is not None else run_seconds + 1
        if frames and start > 0 and begin == 0 and (not resumed or float(resume_seconds) < start):
            h -= frame_bytes - history_frame_bytes(run, selection, include_reflectivity=False)
        # Compressed frames (the default) store the same words in a fraction of the layout.
        h = math.ceil(h * stored)
        # Full frames draw the run's products on the full cadence; a
        # quick-look frame draws only the maps its subset makes
        # (woof.live_products.quicklook_products), never a windowed one.
        picture_interval = float(domain.full_history_interval_s if quick
                                 else domain.history_interval_s)
        p = (_picture_projection(run.nx, run.ny, live, picture_interval,
                                 render_products,
                                 history_frames=regular,
                                 after_seconds=drawn_before,
                                 begin_seconds=float(begin),
                                 end_seconds=None if end is None else float(end))[0]
             if render and live is not None else 0)
        if quick and render and live is not None:
            from woof.live_products import quicklook_products
            p += _picture_projection(run.nx, run.ny, live, float(domain.history_interval_s),
                                     quicklook_products(render_products),
                                     history_frames=quick,
                                     after_seconds=drawn_before,
                                     begin_seconds=float(begin),
                                     end_seconds=None if end is None else float(end))[0]
        history += h
        checkpoints += c
        pictures += p
        rows.append({"grid_id": int(domain.grid_id), "cells": cells, "history_frames": frames,
                     "history_bytes": int(h), "history_frame_bytes": frame_bytes,
                     "history_selection": selection.spelling(),
                     **({"quicklook_frames": quick, "quicklook_frame_bytes": quick_frame_bytes}
                        if quick else {}),
                     "checkpoint_bytes": int(c), "picture_bytes": int(p)})
    download = download_budget.download_estimate(fetch)
    preparation = download_budget.preparation_estimate(
        exp, chain=chain, forcing_times=download.get("leads"))
    unpriced = [part for part, value in (("download", download["bytes"]),
                                         ("preparation", preparation["bytes"])) if value is None]
    scratch = download_budget.compose_scratch_estimate(
        exp, chain=chain, source=download.get("source"),
        forcing_times=download.get("leads"), points=download.get("points"))
    if scratch["bytes"] is None:
        unpriced.append("compose scratch")
    download_bytes = max(0, int(download["bytes"] or 0) - int(download_present_bytes or 0))
    preparation_bytes = int(preparation["bytes"] or 0)
    scratch_bytes = int(scratch["bytes"] or 0)
    # [simulated_radar] writes beside the history it scans: one volume per
    # site per committed frame, priced from the native writer bound.
    radar = None
    if getattr(getattr(exp, "simulated_radar", None), "enabled", False):
        from woof.simulated_radar import output_projection
        radar = output_projection(exp, {row["grid_id"]: row["history_frames"] for row in rows})
        if radar["bytes"] is None:
            unpriced.append(radar["unpriced"])
    radar_bytes = 0 if radar is None else int(radar["bytes"] or 0)
    # The frame stream lives while the preparation runs and is removed
    # before the forecast writes anything, so it and the forecast's own
    # files are never on the disk together.
    total = download_bytes + preparation_bytes + max(scratch_bytes,
                                                     history + checkpoints + pictures
                                                     + radar_bytes)
    # Keys only when radar is asked for, so a run without it projects
    # exactly what it did before radar existed.
    radar_keys = {} if radar is None else {"radar_bytes": radar_bytes, "radar": radar}
    return {"download_bytes": download_bytes, "preparation_bytes": preparation_bytes,
            "history_bytes": int(history), "checkpoint_bytes": int(checkpoints),
            "picture_bytes": int(pictures), **radar_keys,
            "compose_scratch_bytes": scratch_bytes,
            "compose_scratch_min_bytes": int(scratch["min_bytes"] or 0),
            "total_bytes": int(total), "checkpoint_sets_held": held,
            "resume_seconds": float(resume_seconds) if resumed else None,
            "download": dict(download, present_bytes=int(download_present_bytes or 0)),
            "preparation": preparation, "compose_scratch": scratch, "unpriced": unpriced,
            "history_stored_fraction": stored,
            "domains": rows, "basis": ("history uses the selected writer inventory and "
                "CDF-2 header allowance" + _stored_basis(stored) + "; checkpoints: " + MEASURED),
            "picture_basis": _picture_table()["basis"] if render else None}


def pictures_per_frame(render_products, *, run_hours: float | None = None) -> int | None:
    """Count the selected products using the renderer's tokenizer and catalog.

    A section's comma-separated levels belong to its one product. Aliases
    and overlapping groups count once. None means the renderer could not
    expand a group; pricing then uses the measured table's product rows.
    This is an upper count per frame, before individual windows close.
    """
    selected = _picture_products(render_products, run_hours=run_hours)
    return None if selected is None else sum(row[2] for row in selected.values())


def _picture_products(render_products, *, run_hours=None, measured=False):
    """One selection rule for counting and pricing: name -> (kind, first hour, copies)."""
    from woof.first_products import early_render_requested
    from woof.render import RUST_PRODUCT_ALIASES
    from woof.rustwx import GROUP_KEYWORDS, VARIABLES_KEYWORD, product_spec_terms

    if not early_render_requested(render_products):
        return {}
    table = _picture_table()
    if measured:
        local = [(name, row[0], row[1]) for name, row in table["products"].items()
                 if row[0] != "generic" and (run_hours is None or row[1] <= run_hours)]
        groups = set(GROUP_KEYWORDS)
    else:
        local, groups = _local_run_products(run_hours)
    result = {}
    for token in product_spec_terms(str(render_products)):
        word = token.strip().casefold()
        if not word or word == "none":
            continue
        if word == VARIABLES_KEYWORD:
            result[VARIABLES_KEYWORD] = ("generic", 0, STORED_VARIABLE_PICTURES)
        elif word in groups or word in GROUP_KEYWORDS:
            if local is None:
                return None
            members = [(name, kind, first) for name, kind, first in local
                       if word == "all" or kind == word]
            if not members and word != "all":
                if not measured:
                    return None
                result[word] = ("explicit", 0, 1)
            for name, kind, first in members:
                result[name] = (kind, first, 1)
        else:
            name = RUST_PRODUCT_ALIASES.get(token, token)
            row = table["products"].get(name, ["explicit", 0])
            result.setdefault(name, (row[0], row[1], 1))
    return result


def _local_run_products(run_hours: float | None):
    """The renderer's name, kind and first-hour rows, plus its group keywords."""
    from woof.runplan import render_catalog

    catalog = render_catalog()
    groups = {str(word).casefold() for word in catalog.get("group_keywords") or ()}
    local = catalog.get("local_run")
    if not isinstance(local, dict) or not isinstance(local.get("products"), list):
        return None, groups
    rows = []
    for row in local["products"]:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        try:
            first = float(row.get("minimum_hour") or 0)
        except (TypeError, ValueError):
            first = 0.0
        if run_hours is None or first <= float(run_hours):
            rows.append((str(row["name"]), str(row.get("kind") or ""), first))
    return rows, groups


def projected_child_bytes(cfg, *, history_frames: int, checkpoints_written: int,
                          keep_checkpoints: int | None,
                          render_products: str | None, history_selection=None) -> dict[str, Any]:
    """History, checkpoint and picture bytes one downscaled child will write.

    A child is one grid on its own clock, so its counts come from that
    clock rather than from an experiment: ``history_frames`` frames and
    ``checkpoints_written`` sets (every ``restart_interval_s`` and the last
    step), of which at most ``keep_checkpoints`` + 1 are on disk at once,
    because a new set is written whole before an older one goes (``None``
    keeps every set).

    ``render_products`` is the child's complete product request, passed to
    the same measured product/grid model as a forecast. Its reported
    ``pictures_per_frame`` remains the renderer-catalog upper count.

    A child's checkpoint carries no lateral boundary state (its boundaries
    are rebuilt from the parent archive each time it runs), so a set costs
    what a nest's does: :data:`CHECKPOINT_BYTES_PER_CELL`.  The same shape
    of answer as :func:`projected_run_bytes`, so :func:`disk_refusal` reads
    both.

    A child downloads nothing and prepares nothing on disk: it reads its
    parent's history and checkpoint where they already lie, and its
    initial state and boundaries are interpolated in memory.  Its download
    and preparation are therefore zero, stated rather than left out.
    """
    written = int(checkpoints_written)
    held = written if keep_checkpoints is None else min(written, int(keep_checkpoints) + 1)
    cells = int(cfg.nx) * int(cfg.ny) * int(cfg.nz)
    frames = int(history_frames)
    frame_bytes = history_frame_bytes(cfg, history_selection)
    stored = history_stored_fraction()
    history = math.ceil(frame_bytes * frames * stored)
    checkpoints = cells * CHECKPOINT_BYTES_PER_CELL * held
    drawn, count = _picture_projection(
        cfg.nx, cfg.ny, float(cfg.run_seconds), float(cfg.output_interval_s),
        render_products, history_frames=frames)
    picture_basis = _picture_table()["basis"]
    return {"download_bytes": 0, "preparation_bytes": 0,
            "history_bytes": int(history), "checkpoint_bytes": int(checkpoints),
            "picture_bytes": int(drawn),
            "total_bytes": int(history + checkpoints + drawn),
            "checkpoint_sets_held": held, "checkpoint_sets_written": written,
            "pictures_per_frame": count,
            "unpriced": [],
            "domains": [{"grid_id": int(cfg.grid_id), "cells": cells,
                         "history_frames": frames, "history_bytes": int(history),
                         "history_frame_bytes": frame_bytes,
                         "checkpoint_bytes": int(checkpoints),
                         "picture_bytes": int(drawn)}],
            "history_stored_fraction": stored,
            "basis": ("history uses the selected writer inventory and CDF-2 header "
                      "allowance" + _stored_basis(stored) + "; checkpoints: " + MEASURED),
            "picture_basis": picture_basis}


def _stored_basis(stored: float) -> str:
    return ("" if stored == 1.0 else
            f", times {stored:g} for lossless compressed frames "
            "(COMPRESSED_HISTORY_FRACTION, measured)")


def free_bytes(path: Path) -> int | None:
    """Free space on the disk that holds ``path`` (or its nearest existing parent)."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return None


def same_disk(first: Path, second: Path) -> bool:
    """Whether two paths (or their nearest existing parents) are on one disk."""

    def device(path: Path) -> int | None:
        probe = Path(path)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            return probe.stat().st_dev
        except OSError:
            return None

    a, b = device(first), device(second)
    return a is None or b is None or a == b


def _scratch_words(projection: dict[str, Any], stream: int, room: int,
                   folder, *, shares_run_disk: bool) -> str:
    """The sentence for a frame stream that ``room`` bytes cannot hold."""

    from woof.ingest.source_coverage import COMPOSE_SCRATCH_ENV

    scratch = projection.get("compose_scratch") or {}
    per_time = int(scratch.get("per_valid_time") or 0)
    times = scratch.get("valid_times")
    where = f" in {folder}" if folder is not None else ""
    shape = (f" ({times} valid times of {per_time / GIB:.1f} GiB)"
             if times and per_time else "")
    after = " once the download and the preparation's own files are on it" if shares_run_disk else ""
    return (f"this run's preparation stages its decoded frame stream, {stream / GIB:.1f} GiB"
            f"{shape}, in a scratch folder{where}, and the disk that holds that folder has "
            f"{max(room, 0) / GIB:.1f} GiB free{after}, so the preparation would stop at its "
            f"first valid time after the whole download.  Set {COMPOSE_SCRATCH_ENV} to an "
            "existing folder on a disk with room for the stream, or free that much space "
            "on this one")


def _disk_need(projection: dict[str, Any], stream: int, *, download_free, scratch_free):
    """(bytes the run directory's disk holds at its peak, the part before the stream or forecast)."""

    download = int(projection.get("download_bytes") or 0)
    base = int(projection.get("preparation_bytes") or 0) + (download if download_free is None else 0)
    forecast = (int(projection["history_bytes"]) + int(projection["checkpoint_bytes"])
                + int(projection.get("picture_bytes") or 0)
                + int(projection.get("radar_bytes") or 0))
    on_run_disk = 0 if scratch_free is not None else stream
    return base + max(on_run_disk, forecast), base, forecast


#: The ways out of :func:`disk_refusal` for a forecast run.
RUN_DISK_REMEDY = ("Free some disk, write history less often "
                   "(history_interval_s, nest_history_interval_s), or pick a smaller layout")


def disk_refusal(projection: dict[str, Any], free: int | None, *,
                 download_free: int | None = None, scratch_free: int | None = None,
                 scratch_folder=None, subject: str = "this run",
                 remedy: str = RUN_DISK_REMEDY) -> str | None:
    """The refusal for a run that would not fit on its disk, or None.

    ``free`` is the free space on the disk that holds the run directory.
    ``download_free`` is given only when the download lands on another
    disk; the download is then compared with that disk and the rest of the
    run with the run directory's.  ``scratch_free`` is given only when the
    preparation's compose scratch lands on another disk (``WOOF_COMPOSE_SCRATCH``
    names one); ``scratch_folder`` is that folder, named in the refusal.
    ``subject`` names what is refused and ``remedy`` gives the ways out in
    the refusing door's own words.

    The frame stream is refused on the part of it that does not depend on
    the atmospheric window (the whole stream for a global source whose
    target reaches its stored longitude cut, where no window is taken): that
    much is certain, and the engine would refuse it itself after the whole
    download and the first valid time's decode.
    """
    download = int(projection.get("download_bytes") or 0)
    if download_free is not None and download > download_free:
        return (f"{subject} downloads about {download / GIB:.1f} GiB into a directory on a disk "
                f"with {download_free / GIB:.1f} GiB free, so the download would stop partway "
                "when that disk fills.  Free some disk there, or download somewhere else")
    stream = int(projection.get("compose_scratch_min_bytes") or 0)
    if scratch_free is not None and stream > scratch_free:
        return _scratch_words(projection, stream, scratch_free, scratch_folder,
                              shares_run_disk=False)
    total, base, forecast = _disk_need(projection, stream, download_free=download_free,
                                       scratch_free=scratch_free)
    if free is None or total <= free:
        return None
    if scratch_free is None and stream > forecast and base + forecast <= free:
        # The frame stream is the one thing that does not fit.
        return _scratch_words(projection, stream, free - base, scratch_folder,
                              shares_run_disk=True)
    parts = [(download if download_free is None else 0, "of download"),
             (projection.get("preparation_bytes", 0), "of preparation"),
             (stream if scratch_free is None and stream > forecast else 0,
              "of decoded frame stream while it prepares"),
             (projection["history_bytes"], "of history"),
             (projection["checkpoint_bytes"], "of checkpoints"),
             (projection.get("picture_bytes", 0), "of pictures"),
             (projection.get("radar_bytes", 0), "of simulated radar")]
    words = ", ".join(f"{value / GIB:.1f} GiB {what}" for value, what in parts if value)
    return (f"{subject} would write about {total / GIB:.1f} GiB ({words}) and the disk "
            f"that holds its run directory has {free / GIB:.1f} GiB free, so it would stop "
            f"partway when the disk fills.  {remedy}")


def disk_warning(projection: dict[str, Any], free: int | None, *,
                 download_free: int | None = None, scratch_free: int | None = None,
                 scratch_folder=None) -> str | None:
    """A frame stream that may not fit, or None.

    Asked only after :func:`disk_refusal` returned None.  A regional
    source's stream is certain only in the layers its atmospheric window
    cannot crop; the rest is priced over the target's footprint.  When
    that estimate does not fit where the certain part does, the run may
    still fit (the window may crop more than the estimate) and is not
    refused, but the reader is told before the download.
    """
    stream = int(projection.get("compose_scratch_bytes") or 0)
    if stream <= int(projection.get("compose_scratch_min_bytes") or 0):
        return None
    if scratch_free is not None:
        if stream <= scratch_free:
            return None
        return ("may not fit: " + _scratch_words(projection, stream, scratch_free,
                                                 scratch_folder, shares_run_disk=False))
    total, base, forecast = _disk_need(projection, stream, download_free=download_free,
                                       scratch_free=None)
    if free is None or total <= free or stream <= forecast:
        return None
    return "may not fit: " + _scratch_words(projection, stream, free - base, scratch_folder,
                                            shares_run_disk=True)


__all__ = ["CHECKPOINT_BYTES_PER_CELL", "GIB", "MEASURED",
           "ROOT_CHECKPOINT_BYTES_PER_CELL", "RUN_DISK_REMEDY", "STORED_VARIABLE_PICTURES",
           "PICTURES_MEASURED", "picture_table_path", "projected_picture_bytes",
           "disk_refusal", "disk_warning", "free_bytes", "pictures_per_frame",
           "history_frame_bytes", "history_frames", "history_stored_fraction",
           "COMPRESSED_HISTORY_FRACTION", "projected_child_bytes", "projected_run_bytes",
           "quicklook_frames", "same_disk"]
