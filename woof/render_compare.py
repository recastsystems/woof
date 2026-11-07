"""``woof render --compare REFERENCE``: a run beside a reference model.

One sheet per product and valid time: the run's own product on the left,
the reference model's own field for the same cycle and lead on the right,
and -- for the continuous fields -- the run minus the reference in a third
panel.  Same grid, same projection and extent, same colour scale.

This module orchestrates and nothing else.  Every number and every pixel
is ``rw_compare``'s (``tools/rustwx/crates/rw-wrfbatch/src/bin/compare.rs``):
the wrfout import, the GRIB2 decode of the reference's messages, the
fetch from the public bucket, the grid match, the difference, the panels
and their composition into a sheet are all the Rust engine's.  What is
decided here is which files are one run's frames, which earlier frame an
hourly accumulation needs beside it, and where the sheets go.

The vocabulary -- which references exist, which products each sheet
covers -- is the engine's too, asked with ``--list-products`` rather than
transcribed, so a reference or product added to the engine's tables is
reachable from this door without a line changing here.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from woof.render import default_source_label

#: The brand a comparison sheet puts over the run's panel.
RUN_LABEL = "WOOF"

#: ``wrfout_d01_2026-10-03_06:00:00`` and ``wrfout_d01_2026-10-03_06_00_00``.
_WRFOUT_NAME = re.compile(
    r"^wrfout_(?P<domain>d\d{2})_"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}[:_-]\d{2}[:_-]\d{2})")

#: Where a run folder keeps its history frames, tried in order.
_RUN_FOLDER_FRAMES = ("", "wrfout", "run/wrfout", "chain/run/wrfout",
                      "out/wrfout", "forecast/wrfout")


def frame_identity(path: Path):
    """``(domain, valid time)`` from a history frame's name, or ``None``."""

    match = _WRFOUT_NAME.match(path.name)
    if match is None:
        return None
    digits = re.sub(r"\D", "", match.group("stamp"))
    try:
        valid = datetime.datetime.strptime(digits, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return match.group("domain"), valid


def _expand_patterns(inputs) -> list[Path]:
    """Each ``*``, ``?`` or ``[`` pattern becomes its matches, sorted.

    woof reads the pattern itself, so a quoted glob (the render line
    ``woof go --dry-run`` prints) and PowerShell, which expands none,
    name the same frames an expanding shell would.  A name that exists
    as written stays as written; a pattern that matches nothing is
    refused, since a literal ``wrfout_d01_*`` file can only fail later
    with a bare "No such file".
    """

    out: list[Path] = []
    for item in inputs:
        text = str(item)
        if not glob.has_magic(text) or Path(text).exists():
            out.append(Path(item))
            continue
        matched = sorted(glob.glob(text))
        if not matched:
            raise ValueError(
                f"{text} matches no file or folder; the pattern is read by "
                "woof itself, so spell the directory the way this system "
                "does (a drive-letter path on Windows)")
        out.extend(Path(name) for name in matched)
    return out


def expand_inputs(inputs) -> list[Path]:
    """Files stay files; a run folder becomes its history frames; a
    glob pattern becomes its matches (see :func:`_expand_patterns`).

    A folder is searched where a run keeps its frames (itself, then
    ``wrfout/``, ``out/wrfout/``, ``forecast/wrfout/``) and the first of
    those that holds any is taken whole, so a run folder and its
    ``wrfout`` directory name the same frames.
    """

    frames: list[Path] = []
    for item in _expand_patterns(inputs):
        path = Path(item)
        if not path.is_dir():
            frames.append(path)
            continue
        found: list[Path] = []
        for relative in _RUN_FOLDER_FRAMES:
            folder = path / relative if relative else path
            if not folder.is_dir():
                continue
            found = sorted(
                entry for entry in folder.iterdir()
                if entry.is_file() and frame_identity(entry) is not None)
            if found:
                break
        if not found:
            raise ValueError(
                f"{path} holds no wrfout_dNN_YYYY-MM-DD_HH:MM:SS history "
                f"frames (looked in {', '.join(name or '.' for name in _RUN_FOLDER_FRAMES)}). "
                "Supply a completed run folder, its wrfout directory, or explicit history files.")
        frames.extend(found)
    return frames


def group_frames(frames) -> list[tuple[list[Path], list[Path]]]:
    """``[(frames to compare, context frames)]``, one entry per run nest.

    Frames of one folder and one domain are one run's time series and go
    to the engine together.  Each compared frame's hourly accumulation
    needs the same run's frame one hour earlier; when that frame exists
    beside it and was not itself asked for, it rides along as context
    (imported, never drawn).
    """

    groups: dict[tuple[str, str], list[Path]] = {}
    for frame in frames:
        identity = frame_identity(frame)
        domain = identity[0] if identity else ""
        groups.setdefault((str(frame.parent.resolve()), domain), []).append(frame)
    out = []
    for (_, _), members in sorted(groups.items()):
        members = sorted(dict.fromkeys(members))
        named = {member.name for member in members}
        context: list[Path] = []
        for member in members:
            identity = frame_identity(member)
            if identity is None:
                continue
            domain, valid = identity
            earlier = valid - datetime.timedelta(hours=1)
            for separator in (":", "_"):
                name = (f"wrfout_{domain}_{earlier:%Y-%m-%d_%H}"
                        f"{separator}{earlier:%M}{separator}{earlier:%S}")
                candidate = member.with_name(name)
                if (name not in named and candidate.is_file()
                        and candidate not in context):
                    context.append(candidate)
                    break
        out.append((members, context))
    return out


#: The most engines one comparison runs at once.  One a physical core at most:
#: 40 frames on 8 cores took 29 s with 4 engines, 19 s with 8 and 19 s with 16.
COMPARE_WORKERS_MAX = 16
#: One engine's store and rasters in host memory, per byte of history frame on
#: disk (measured 16-20 GB for a 2.4 GB frame, WOOF 2026-10-31).
_ENGINE_HOST_BYTES_PER_FRAME_BYTE = 8


def available_cores() -> int:
    """The cores this process may use, its affinity where it has one."""

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # no affinity call on this platform
        return os.cpu_count() or 1


def compare_workers(frames, *, cores: int | None = None,
                    available_bytes: int | None = None) -> int:
    """How many engines draw these frames at once.

    THE BREAKAGE THIS CLOSES: ``rw_compare`` draws one sheet at a time and
    fetches the reference on that same thread, so finalize held a rented box
    at 0 % card and one busy core out of 16 for 3-6 minutes (WOOF city runs,
    2026-10-31: nyc-500m 189 s, sf-bay-fog-750m 342 s).  Frames are
    independent, so every core gets some of them: two cores an engine, because
    the engine's import is parallel; no more engines than the host's free
    memory holds, one frame's store each.
    """

    if cores is None:
        cores = available_cores()
    workers = max(1, min(COMPARE_WORKERS_MAX, cores // 2, len(frames)))
    if available_bytes is None:
        from woof.ingest.host_decode_window import available_host_bytes

        available_bytes = available_host_bytes()
        if available_bytes is None:
            return workers
    largest = max((frame.stat().st_size for frame in frames), default=0)
    needed = largest * _ENGINE_HOST_BYTES_PER_FRAME_BYTE
    if needed > 0:
        workers = max(1, min(workers, available_bytes // needed))
    return int(workers)


def deal_frames(frames, workers: int) -> list[list[Path]]:
    """``frames`` as ``workers`` lists, each a run of neighbouring valid times.

    A worker gets every nest's frames of its valid times, so no two engines
    ask for the same reference lead at once (the fetch cache's partial file
    is named by lead); and neighbouring times, so only the first frame of
    each list imports the hour before it as context.
    """

    timed = sorted({identity[1] for frame in frames
                    if (identity := frame_identity(frame)) is not None})
    if workers <= 1 or len(timed) <= 1:
        return [list(frames)] if frames else []
    # Whole valid times to each list; the first lists take the remainder.
    lists = min(workers, len(timed))
    each, extra = divmod(len(timed), lists)
    slot: dict = {}
    at = 0
    for index in range(lists):
        size = each + (index < extra)
        for valid in timed[at:at + size]:
            slot[valid] = index
        at += size
    dealt: list[list[Path]] = [[] for _ in range(lists)]
    for frame in frames:
        identity = frame_identity(frame)
        dealt[slot[identity[1]] if identity is not None else 0].append(frame)
    return [members for members in dealt if members]


def default_reference_cache() -> Path:
    """Where fetched reference subsets are kept between renders."""

    return Path.home() / ".woof" / "cache" / "compare-reference"


def reference_catalog(text: str) -> dict[str, dict]:
    """Reference names, labels, and product capabilities from the native table."""

    references: dict[str, dict] = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if fields[0] == "REFERENCE" and len(fields) >= 3:
            references[fields[1]] = {"label": fields[2], "products": {}}
        elif fields[0] == "CAPABILITY" and len(fields) >= 4:
            reference = references.get(fields[1])
            if reference is not None:
                reference["products"][fields[2]] = fields[3]
    return references


def selected_references(text: str, catalog: dict[str, dict]) -> list[str]:
    """Validate and preserve the caller's comma-separated reference order."""

    selected: list[str] = []
    for name in text.split(","):
        name = name.strip()
        if name not in catalog:
            raise ValueError(f"unknown comparison reference {name!r}; this "
                             f"native build knows: {', '.join(catalog)}")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("--compare named no references")
    return selected


def engine_command(engine: Path, args: argparse.Namespace, *, store: Path,
                   frames, context, size: tuple[int, int]) -> list[str]:
    """The exact ``rw_compare`` invocation for one run nest."""

    command = [
        str(engine), "--store-root", str(store), "--out-dir", str(args.out),
        "--reference", str(args.compare),
        "--products", str(args.products or "all"),
        "--difference", str(args.compare_difference),
        "--layout", str(args.layout),
        "--width", str(size[0]), "--height", str(size[1]),
        "--source-label", args.source_label or default_source_label(),
        "--run-label", args.compare_label or RUN_LABEL,
        "--reference-cache",
        str(args.compare_cache or default_reference_cache()),
    ]
    if args.compare_reference_dir is not None:
        command += ["--reference-dir", str(args.compare_reference_dir)]
    if args.compare_cycle:
        command += ["--cycle", str(args.compare_cycle)]
    if args.compare_offline:
        command.append("--offline")
    if args.compare_gallery is not None:
        command += ["--flat-dir", str(args.compare_gallery)]
    if getattr(args, "theme", None):
        command += ["--theme", str(args.theme)]
    for frame in context:
        command += ["--context", str(frame)]
    command += [str(frame) for frame in frames]
    return command


def _engine_environment(environment: dict, workers: int, cores: int) -> dict:
    """One engine's share of the cores: its import's own threads, no more."""

    if workers <= 1 or environment.get("RAYON_NUM_THREADS"):
        return environment
    return {**environment, "RAYON_NUM_THREADS": str(max(1, cores // workers))}


def _relay(line: str, tally: dict) -> None:
    """One engine line, said the way this door says things."""

    fields = line.rstrip("\n").split("\t")
    word = fields[0]
    if word == "RENDERED" and len(fields) >= 4:
        tally["rendered"].append(fields[3])
        print(f"render: {fields[3]}")
    elif word == "SKIPPED" and len(fields) >= 4:
        tally["skipped"] += 1
        what = "every product" if fields[1] == "*" else fields[1]
        print(f"render: skipped {what} for {Path(fields[2]).name}: "
              f"{fields[3]}", file=sys.stderr)
    elif word == "FAILED" and len(fields) >= 4:
        tally["failed"] += 1
        what = "every product" if fields[1] == "*" else fields[1]
        print(f"render: FAILED {what} for {Path(fields[2]).name}: "
              f"{fields[3]}", file=sys.stderr)
    elif word == "SOURCE" and len(fields) >= 6:
        if fields[2] == "observed":
            print(f"render: reference {fields[1]} observed {fields[3]} "
                  f"product {fields[4]} from {fields[5]}")
        else:
            print(f"render: reference {fields[1]} cycle {fields[2]} "
                  f"{fields[3]}Z {fields[4]} from {fields[5]}")
    elif word == "MATCH" and len(fields) >= 2:
        print("render: grid match " + " ".join(fields[1:]))
    elif word == "STATS" and len(fields) >= 3:
        print("render: difference " + " ".join(fields[1:]))
    elif word.startswith("FINISHED"):
        return
    elif word == "IMPORT_NOTE":
        return
    elif line.strip() and "Download" not in line:
        print(f"render: {line.rstrip()}", file=sys.stderr)


def compare_main(args: argparse.Namespace) -> int:
    """The ``--compare`` route of ``woof render``."""

    from woof import explain, render, rustwx, rustwx_lanes

    try:
        engine = rustwx_lanes.require_compare_bin()
    except RuntimeError as error:
        print("render: " + explain.render(
            str(error), explain=explain.explain_enabled(args),
            command="woof render"), file=sys.stderr)
        return 2
    if args.list_products:
        listing = subprocess.run(
            [str(engine), "--list-products"], capture_output=True, text=True,
            errors="replace")
        sys.stdout.write(listing.stdout)
        return listing.returncode
    if not args.wrfout:
        print("render: --compare needs at least one WRFOUT frame or run "
              "folder", file=sys.stderr)
        return 2
    try:
        listing = subprocess.run(
            [str(engine), "--list-products"], capture_output=True, text=True,
            errors="replace", env=rustwx.renderer_env())
        if listing.returncode != 0:
            raise ValueError("native comparison catalogue failed: "
                             + (listing.stderr.strip() or str(listing.returncode)))
        catalog = reference_catalog(listing.stdout)
        args.compare = ",".join(selected_references(args.compare, catalog))
        # The reference-model renderer has a fixed-canvas contract.
        # The single-run renderer's automatic canvas keeps that existing
        # comparison default at the reference renderer's native size.
        size = render.parse_size(args.size) or (1200, 900)
        frames = expand_inputs(args.wrfout)
    except ValueError as error:
        print(f"render: {error}", file=sys.stderr)
        return 2
    missing = [frame for frame in frames if not frame.is_file()]
    if missing:
        print("render: no such history frame: "
              + ", ".join(str(frame) for frame in missing), file=sys.stderr)
        return 2

    environment, device = rustwx.raster_device_env(rustwx.renderer_env())
    print(f"render: {device}")
    render._claim_run_dir(args, frames)
    tally = {"rendered": [], "skipped": 0, "failed": 0}
    dealt = deal_frames(frames, compare_workers(frames))
    engine_env = _engine_environment(environment, len(dealt), available_cores())
    relay_lock = threading.Lock()
    statuses: list[int] = []

    def work(worker_frames) -> None:
        # One worker's frames, a nest at a time, as one engine per nest.
        try:
            for members, context in group_frames(worker_frames):
                with render.scratch_store(args.out) as store:
                    command = engine_command(
                        engine, args, store=store, frames=members,
                        context=context, size=size)
                    process = subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, errors="replace", env=engine_env)
                    assert process.stdout is not None
                    for line in process.stdout:
                        with relay_lock:
                            _relay(line, tally)
                    if process.wait() != 0:
                        statuses.append(1)
        except OSError as error:
            with relay_lock:
                print(f"render: an engine could not run: {error}", file=sys.stderr)
            statuses.append(1)

    try:
        threads = [threading.Thread(target=work, args=(members,), daemon=True)
                   for members in dealt]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        render._publish_run_dir(args)
    status = 1 if statuses else 0
    print(f"render: {len(tally['rendered'])} comparison sheet(s) -> "
          f"{args.out}"
          + (f" ({tally['skipped']} skipped)" if tally["skipped"] else "")
          + (f" ({tally['failed']} FAILED)" if tally["failed"] else ""))
    if args.compare_gallery is not None and tally["rendered"]:
        print(f"render: flat copies -> {args.compare_gallery}")
    return status


def register_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``--compare`` family of ``woof render``."""

    parser.add_argument(
        "--compare", metavar="REFERENCE[,REFERENCE...]", default=None,
        help="draw each frame beside ordered native references (for example "
             "hrrr,mrms or hrrr,rrfs,mrms), on one grid and colour scale. "
             "Forecast references share the valid time; observation panels "
             "label their actual observation time. The native --list-products "
             "catalogue lists references and supported products. WRFOUT may "
             "be frames or a run folder. The map rasters are filled on a "
             "CUDA card: the first one CUDA_VISIBLE_DEVICES exposes (CUDA's "
             "device 0 when it is unset; the run names the card it uses). "
             "RUSTWX_RASTER_DEVICE=cpu fills them on the CPU")
    parser.add_argument(
        "--compare-difference", choices=("auto", "on", "off"),
        default="auto",
        help="run-minus-reference panels: 'auto' (default) draws continuous "
             "differences with one reference and only field panels with a "
             "reference list; 'on' adds a difference for each reference "
             "whose units have a difference ladder; 'off' never")
    parser.add_argument(
        "--compare-reference-dir", type=Path, default=None, metavar="DIR",
        help="read the reference's GRIB2 files from DIR (by their "
             "published names, flat or under the bucket's own folders) "
             "before fetching anything")
    parser.add_argument(
        "--compare-cache", type=Path, default=None, metavar="DIR",
        help="where fetched reference subsets are kept between renders "
             "(default ~/.woof/cache/compare-reference)")
    parser.add_argument(
        "--compare-cycle", default=None, metavar="YYYYMMDDHH",
        help="compare against THIS reference cycle instead of the run's "
             "own start time")
    parser.add_argument(
        "--compare-offline", action="store_true",
        help="never fetch: use only --compare-reference-dir and the cache")
    parser.add_argument(
        "--compare-gallery", type=Path, default=None, metavar="DIR",
        help="also copy every sheet, flat, into DIR; DIR may equal the output "
             "directory. Multi-reference sheets keep the requested panel order")
    parser.add_argument(
        "--compare-label", default=None, metavar="TEXT",
        help=f"the title over the run's panel (default {RUN_LABEL})")


__all__ = ["RUN_LABEL", "compare_main", "default_reference_cache",
           "default_source_label", "engine_command", "expand_inputs",
           "frame_identity", "group_frames", "register_arguments"]
