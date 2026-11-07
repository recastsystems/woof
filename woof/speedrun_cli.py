"""``woof speedrun`` -- run a named course end to end and seal the capsule.

The door drives the SHIPPED chain (``woof go`` as a subprocess, with the
course's own config and the course's own product list) rather than a
private sequence of stages.  That is deliberate: a benchmark that runs a
path no user runs measures a path nobody has.  Everything this module
adds is measurement and refusal -- the work itself is the front door a
reader already types.

What happens off the clock, and why:

* **Fetching.**  The bytes are staged before the door is called; the
  course names them and the door digests them, but no download is timed.
* **The device and stack probe.**  Asking the card its name is not
  forecasting.
* **The cache census.**  Reading the kernel cache decides which class the
  record belongs to; it happens before the clock so the decision cannot
  be made after seeing the time.
* **``go``'s own fetch stage.**  With ``--data-dir`` pointing at the
  staged bytes it revalidates them and downloads nothing.  Its wall is
  recorded in ``clock.off_clock`` and subtracted, by name, from the
  process wall.

Everything else -- authority, manifest, preparation, forecast (kernel
compile included), render -- is on the clock.
"""

from __future__ import annotations

from woof.backend import compilation_target as _compilation_target

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from woof import speedrun
from woof.speedrun import (CAPSULE_FILENAME, COMPILE_MODES,
                            ColdCacheRefusal, SpeedrunRefusal,
                            StagedInputsMissing)

#: What the door writes beside the run it measured.
DEFAULT_OUT = Path("speedrun")

#: Stage names ``woof go`` emits that are NOT on the speedrun clock.
OFF_CLOCK_STAGES = ("fetch",)


# ---------------------------------------------------------------------------
# Off-clock probes
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _engine_block(argv: list[str]) -> dict[str, Any]:
    import woof
    from woof.supervisor import git_commit

    return {
        "gpuwm_version": woof.__version__,
        "git_commit": git_commit() or "unavailable",
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "worktree_clean": _git("status", "--porcelain") == "",
        "python": platform.python_version(),
        "argv": list(argv),
    }


def _git(*args: str) -> str:
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parents[1]),
                              timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return done.stdout.strip() if done.returncode == 0 else "unavailable"


def _machine_block() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def _device_block() -> dict[str, Any]:
    """What card this is, measured, or an explicit unavailable marker.

    A capsule with a guessed device block would rank a record against a
    card nobody proved was there, so the failure mode is a named
    ``unavailable`` rather than a plausible string.

    This asks CuPy directly rather than going through
    :func:`woof.gpu_stack_identity.gpu_cuda_stack_identity`, and the
    reason is measured: that probe RAISES on a CUDA runtime outside
    ``CUDA_RUNTIME_RANGE`` (12.x), which is the right behaviour for the
    sealed native-WRF distribution it was written for and the wrong one
    here.  Both boxes this lane measured run CuPy 14 on CUDA 13, so
    routing the capsule through it would have thrown away the card's
    name, its driver and its memory in exchange for one boolean.  The
    range verdict is recorded as a FIELD instead, so a reader still
    learns the run was outside the certified family -- and still learns
    which card it was.
    """

    from woof.gpu_stack_identity import CUDA_RUNTIME_RANGE

    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            raise RuntimeError("CuPy reports no CUDA devices")
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
        low, high = CUDA_RUNTIME_RANGE
        block = {
            "status": "measured",
            "name": str(name),
            "device_count": count,
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": runtime_version,
            "cuda_runtime_in_certified_range": bool(
                low <= runtime_version < high),
            "certified_cuda_runtime_range": list(CUDA_RUNTIME_RANGE),
        }
    except Exception as error:  # noqa: BLE001 - any probe failure is the same
        return {"status": "unavailable", "reason": str(error),
                "name": "unavailable"}
    try:
        free, total = cp.cuda.Device(0).mem_info
        block["total_memory_gib"] = round(total / (1 << 30), 3)
        block["free_memory_gib_at_start"] = round(free / (1 << 30), 3)
        target = str(_compilation_target(0, xp=cp))
        if getattr(cp.cuda.runtime, "is_hip", False):
            block.update(backend="hip", architecture=target)
        else:
            block["compute_capability"] = f"{target[:-1]}.{target[-1]}"
    except Exception as error:  # noqa: BLE001
        block["memory_probe"] = f"unavailable: {error}"
    return block


def renderer_block() -> dict[str, Any]:
    """Which rw_wrfbatch drew the pictures, by path and by digest.

    Part of the capsule because it is part of what produced the output:
    two records whose product-set digests differ because one was drawn
    by a newer renderer are not evidence of non-determinism, and a
    reader cannot tell those apart without this.  It also records the
    tree-match verdict the render law's own guard applies -- MEASURED
    2026-08-20, a 3080 record rendered nothing at all because the
    staged binary came from another tree and the guard refused rather
    than drawing weather fields with something else.
    """

    from woof import rustwx

    try:
        path = rustwx.find_renderer()
    except Exception as error:  # noqa: BLE001
        return {"status": "unavailable", "reason": str(error)}
    if path is None:
        return {"status": "unavailable",
                "reason": "no rw_wrfbatch was resolvable on this machine"}
    usable, detail = rustwx.probe_renderer(Path(path))
    return {
        "status": "resolved",
        "path": str(path),
        "sha256": speedrun.digest_file(path),
        "bytes": Path(path).stat().st_size,
        "contract_check": "pass" if usable else "refused",
        "contract_detail": detail,
        "declared_by_env": os.environ.get(rustwx.RENDERER_ENV),
    }


def _numerical_stack_block() -> dict[str, Any]:
    from woof.certify import compile_platform

    try:
        return dict(compile_platform.compile_platform_fingerprint())
    except Exception as error:  # noqa: BLE001
        return {"status": "unavailable", "reason": str(error)}


# ---------------------------------------------------------------------------
# Staged inputs
# ---------------------------------------------------------------------------

def _staged_inputs(row: dict, staged: Path) -> list[dict[str, Any]]:
    """Digest the bytes the course consumes, or refuse naming what is absent.

    The digests go in the capsule because "same course" has to mean the
    same INPUT bytes as well as the same configuration: two records of
    the same grid over different weather are not the same work.
    """

    staged = Path(staged)
    if not staged.is_dir():
        raise StagedInputsMissing(
            f"the staged-input directory {staged} does not exist, so there "
            "are no bytes to start the clock on.\n\n"
            f"Remedy: stage them first (this is OFF the clock):\n"
            f"  {row['fetch_command']}\n"
            "then re-run with --staged pointing at that directory.")
    found = sorted(
        path for pattern in row["staged_inputs"]["patterns"]
        for path in staged.glob(pattern))
    expected = int(row["staged_inputs"]["count"])
    if len(found) != expected:
        listed = "\n".join(f"    {p.name}" for p in found) or "    (none)"
        raise StagedInputsMissing(
            f"course {row['id']!r} consumes {expected} staged input file(s) "
            f"and {staged} holds {len(found)}, so this run would integrate "
            "against a different set of boundary times than the course "
            "names -- a different job with the same name.\n\n"
            f"Found:\n{listed}\n\n"
            f"Remedy: stage the course's bytes (OFF the clock):\n"
            f"  {row['fetch_command']}")
    return [{"name": path.name,
             "bytes": path.stat().st_size,
             "sha256": speedrun.digest_file(path)}
            for path in found]


# ---------------------------------------------------------------------------
# The kernel cache class
# ---------------------------------------------------------------------------

def _cache_census(cache_dir: Path) -> dict[str, Any]:
    from woof import kernel_compile_notice as notice

    entries, undecodable, architectures = notice.scan_kernel_cache(cache_dir)
    capability = notice.current_compute_capability()
    # The census keys an architecture as the bare SM ("86", "120"), the
    # way CuPy spells the card.  Looking it up as "sm_86" found nothing,
    # so a cache full of this card's kernels counted 0 for this card and
    # a warm-class course refused on a warm cache as cold.
    token = None if capability is None else capability.replace(".", "")
    for_this_card = (
        entries if token is None
        else architectures.get(token, 0) + undecodable)
    return {
        "cache_dir": str(cache_dir),
        "entries": entries,
        "undecodable": undecodable,
        "architectures": dict(architectures),
        "compute_capability": capability,
        "entries_for_this_card": for_this_card,
    }


def _is_a_kernel_cache(path: Path, held: list[Path]) -> bool:
    """Every entry must be a regular, recognizable compiled kernel.

    A census may count unknown files conservatively for compile timing;
    deletion cannot.  One kernel beside staged inputs proves nothing
    about those inputs, and links/reparse points are never cache entries.
    """

    import stat
    from woof import kernel_compile_notice as notice

    try:
        for child in held:
            info = child.lstat()
            if (not stat.S_ISREG(info.st_mode)
                    or getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                return False
    except OSError:
        return False
    entries, undecodable, architectures = notice.scan_kernel_cache(path)
    return (entries == len(held) and undecodable == 0
            and bool(architectures))


def _prepare_cold_cache(path: Path) -> Path:
    """Empty a directory so this run pays the NVRTC compile for real.

    REFUSES rather than deletes.  ``--cold-cache-dir`` is a bare ``Path``
    on a command line already carrying relative paths (``--staged
    ./staged --out speedrun``), and the call site resolves it with
    ``.expanduser().resolve()``, which turns ``.`` into the working
    directory and ``~`` into ``$HOME`` before anything else sees them.
    What used to be here was an unguarded ``shutil.rmtree``: a typed
    ``--cold-cache-dir .`` deleted the working directory, staged GRIB and
    run output included, on a flag that reads as a cache knob -- in a
    tree whose fetch door advertises "nothing is deleted"
    (:func:`woof.fetch._force_quarantine_output`).

    The only directory this door will empty is one that is absent, empty,
    or shows positive evidence of being a CuPy kernel cache
    (:func:`_is_a_kernel_cache`).  The inherited cache, the working
    directory, the home directory and a filesystem root are refused by
    name even when they would pass that test, because each has a better
    remedy than a delete.  Refusing the inherited cache is what makes
    this flag's own promise -- "it never touches the inherited cache" --
    a property of the code rather than of the operator's typing.
    """

    from woof import kernel_compile_notice as notice

    path = path.expanduser().resolve()
    remedy = ("\n\nRemedy: name a directory that does not exist yet -- this "
              "door creates it -- e.g. --cold-cache-dir ./cold-kernel-cache.")
    if path == path.parent:
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} is a filesystem root." + remedy)
    if path == notice.cupy_kernel_cache_dir().expanduser().resolve():
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} IS the inherited CuPy kernel cache "
            f"(${notice.CUPY_CACHE_ENV}, or ~/.cupy/kernel_cache).  This "
            "flag exists to leave that cache alone -- its own help says it "
            "never touches it -- so emptying it is the one thing it must "
            "not do." + remedy)
    if path == Path.cwd().resolve():
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} is the working directory, which holds "
            "the staged inputs and the run output this record is made of."
            + remedy)
    if path == Path.home().resolve():
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} is the home directory." + remedy)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not path.is_dir():
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} is a file, not a directory." + remedy)
    try:
        held = sorted(path.iterdir())
    except OSError as error:
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} cannot be read ({error}), so this "
            "door cannot say what emptying it would delete." + remedy
        ) from error
    if not held:
        return path
    if not _is_a_kernel_cache(path, held):
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} contains entries that are not regular "
            "compiled CUDA kernels. Nothing was deleted." + remedy)
    if sorted(path.iterdir()) != held:
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} changed during validation. "
            "Nothing was deleted." + remedy)
    # Delete only the entries we inspected, never the entire directory.
    # A file arriving during clearing must survive rather than being
    # swept up by a recursive delete.
    for child in held:
        child.unlink()
    if any(path.iterdir()):
        raise ColdCacheRefusal(
            f"--cold-cache-dir {path} received new entries while clearing. "
            "Those entries were preserved; use a fresh directory." + remedy)
    return path


# ---------------------------------------------------------------------------
# Reading the run back
# ---------------------------------------------------------------------------

#: run-plan's own event stream, at the OUTPUT ROOT rather than inside the
#: run folder.  Same ``stage_finished`` grammar as `woof go`'s chain
#: events; different file, different directory.
RUN_PLAN_EVENTS_FILENAME = "events.jsonl"


def stage_walls(run_root: Path, *, case_dir: Path | None = None
                ) -> tuple[dict[str, float], dict[str, Any]]:
    """``{stage: wall_seconds}`` from whichever door recorded them.

    Both doors emit the same ``stage_finished`` grammar; only the file
    and its directory differ.  Asking for one and not the other is why
    two nested records read ``prepare 0:00  forecast 0:00  render 0:00``
    with the whole wall in an unattributed remainder.
    """

    from woof import chain_events

    candidates = [run_root / chain_events.CHAIN_EVENTS_FILENAME]
    if case_dir is not None:
        candidates.append(Path(case_dir) / RUN_PLAN_EVENTS_FILENAME)
    for path in candidates:
        if not path.is_file():
            continue
        events = last_run_events(chain_events.read_chain_events(path))
        walls = walls_by_phase(events)
        if walls:
            return walls, (chain_events.summarize(path) or {})
    return {}, {}


#: The event each run opens with.  run-plan APPENDS to one
#: ``events.jsonl`` per output root, so a case directory reused across
#: runs holds every run's stages in one file.
RUN_OPENING_EVENTS = ("plan_accepted", "boot", "started")


def last_run_events(events: list[Mapping[str, Any]]
                    ) -> list[Mapping[str, Any]]:
    """Only the events belonging to the most recent run in the stream.

    MEASURED on the 3080: a nested record whose own clock was 4:45
    reported ``forecast 13:26  render 4:01`` -- three earlier runs of the
    same case directory summed into one record, because the reader took
    the whole appended file.  Everything from the LAST opening event on
    is this run.
    """

    for index in range(len(events) - 1, -1, -1):
        if events[index].get("event") in RUN_OPENING_EVENTS:
            return list(events[index:])
    return list(events)


def walls_by_phase(events: list[Mapping[str, Any]]) -> dict[str, float]:
    """``{phase: wall_seconds}`` from a replayed ``stage_finished`` stream.

    Filed under the PHASE, not the stage, and summed rather than
    overwritten.  Both matter, and both were measured on test host A's five
    records: run-plan calls the render stage ``finalize`` (so a record's
    render cell read 0:00), and it emits ``prepare`` twice -- once for
    the authority phase, once for the manifest phase -- so a dict keyed
    on the stage silently dropped the first.  The phase is what the work
    IS; `woof go` already names its stages after their phases, so this
    reduction gives both doors one vocabulary.
    """

    walls: dict[str, float] = {}
    for record in events:
        if record.get("event") != "stage_finished":
            continue
        wall = record.get("wall_seconds")
        if not isinstance(wall, (int, float)) or isinstance(wall, bool):
            continue
        phases = record.get("phases")
        names = ([str(name) for name in phases]
                 if isinstance(phases, list) and phases
                 else [str(record.get("stage", "unnamed"))])
        share = float(wall) / len(names)
        for name in names:
            walls[name] = round(walls.get(name, 0.0) + share, 6)
    return walls


def _kernel_compile_seconds(run_root: Path) -> tuple[float | None, str]:
    """What the one-time NVRTC compile cost, as the RUN measured it.

    Read off the forecast's own ``progress.jsonl``: the runner emits a
    ``phase`` record named ``kernel_compile`` when -- and only when --
    the cache said a compile was coming AND step 1 really did overrun
    the steps after it.  Both halves are required upstream, and this
    module does not lower that bar: with no record, the capsule says
    the compile was not separately measured rather than inventing a
    number from the first step's wall.
    """

    from woof import progress_log

    for candidate in sorted(run_root.rglob(progress_log.STEP_LOG_FILENAME)):
        try:
            records = progress_log.read_step_log(candidate)
        except (OSError, ValueError):
            continue
        for record in records:
            if (record.get("event") == "phase"
                    and record.get("name") == "kernel_compile"):
                return (float(record["wall_seconds"]),
                        f"{candidate.name} phase kernel_compile "
                        f"({record.get('measured_as', 'runner measurement')})")
        for record in reversed(records):
            if record.get("event") == "run_end":
                excess = record.get("first_step_excess_seconds")
                if excess is not None:
                    return (float(excess),
                            f"{candidate.name} run_end "
                            "first_step_excess_seconds")
                return (None,
                        "the run reported no first-step excess, so no "
                        "kernel compile was separately measurable inside "
                        "this forecast")
    return (None, "no progress.jsonl was found under the run root")


#: Where the chain publishes each artifact inside one run folder.  These
#: are the paths `woof go` really writes -- ``plan["run"]`` is
#: ``<root>/run`` and ``plan["render"]`` is ``<root>/png`` -- and getting
#: them wrong is not a crash, it is a VOID verdict on a run that did all
#: the work.  MEASURED 2026-08-20: a test host A run wrote seven wrfout frames
#: and fifty-six pictures and was voided for having none, because this
#: reader looked for ``<root>/wrfout``, one level up.
FORECAST_SUBDIR = "run"
RENDER_SUBDIR = "png"
WRFOUT_SUBDIR = "wrfout"

#: Where the run's own validity verdict lives, in the order asked.  The
#: single-domain runner publishes ``report.json``; the DOMAIN TREE runner
#: publishes ``evidence/run-receipt.json`` and no report.json at all.
#: MEASURED 2026-08-20: asking only the first one voided a nested run
#: that had integrated eight frames across two nests and rendered
#: sixty-four pictures, with ``forecast_validity 'unavailable'``.
VERDICT_RECEIPTS = ("report.json", "evidence/run-receipt.json")

#: Kept as the single-domain spelling for consumers that import it.
REPORT_FILENAME = VERDICT_RECEIPTS[0]

#: ``run-plan``'s ``prepared`` route nests the go chain one level deeper
#: than ``woof go`` does: ``<output_root>/chain/run-.../``.
RUN_PLAN_CHAIN_SUBDIR = "chain"


def resolve_run_root(case_dir: str | Path, *, driver: str) -> Path:
    """This run's own timestamped folder, under whichever door wrote it."""

    from woof import run_stamp

    case_dir = Path(case_dir)
    root = (case_dir / RUN_PLAN_CHAIN_SUBDIR if driver == "run-plan"
            else case_dir)
    return run_stamp.latest(root) or case_dir


def run_artifacts(run_root: str | Path) -> dict[str, Any]:
    """Everything one finished run left behind, read where it leaves it.

    One function, so the capsule's evidence and any later consumer read
    the same layout from the same place.
    """

    run_root = Path(run_root)
    forecast = run_root / FORECAST_SUBDIR
    from woof.io.wrfout import iter_wrfout_files
    frames = sorted(iter_wrfout_files(forecast / WRFOUT_SUBDIR, "wrfout_d*",
                                     include_temporaries=False),
                    key=lambda path: (path.name, path.as_posix()))
    products, files = _rendered_products(run_root / RENDER_SUBDIR)
    return {
        "wrfout": [{"name": path.name, "bytes": path.stat().st_size}
                   for path in frames],
        "wrfout_paths": frames,
        "report": _first_verdict_receipt(forecast),
        "products_rendered": products,
        "files": files,
        "render_root": run_root / RENDER_SUBDIR,
        "forecast_root": forecast,
    }


def _first_verdict_receipt(forecast_root: Path) -> dict:
    """The run's own verdict document, whichever runner wrote it."""

    for relative in VERDICT_RECEIPTS:
        document = _read_json(forecast_root / relative)
        if document:
            return document
    return {}


def _rendered_products(render_root: Path) -> tuple[list[str],
                                                   list[dict[str, Any]]]:
    """Which products landed, and every PNG's bytes and digest.

    The product name is the folder the render layout files a picture
    under (``<run>/<domain>/<product>/<valid-day>/*.png``), so this
    reads the SHIPPED layout rather than parsing filenames.

    A nest that retires and re-arms files one segment deeper
    (``<domain>/<episode>/<product>/<valid-day>/``), so the position is
    asked of the tree rather than counted from the left: reading slot 1
    blindly would report ``episode-002`` as a product name and lose the
    real one out of the capsule.
    """

    from woof import render_layout

    files: list[dict[str, Any]] = []
    products: set[str] = set()
    if not render_root.is_dir():
        return [], []
    for png in sorted(render_root.rglob("*.png")):
        relative = png.relative_to(render_root)
        parts = relative.parts
        if len(parts) >= 3:
            deeper = (len(parts) > 3
                      and render_layout.episode_number(parts[1]) is not None)
            products.add(parts[2 if deeper else 1])
        files.append({"relpath": relative.as_posix(),
                      "bytes": png.stat().st_size,
                      "sha256": speedrun.digest_file(png)})
    return sorted(products), files


def _product_set_digest(files: list[dict[str, Any]]) -> str:
    return speedrun.digest_document(
        sorted((entry["relpath"], entry["sha256"]) for entry in files))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

#: The run-plan route that IS the `woof go` chain, entered with domain
#: trees allowed.  `woof.runplan` calls ``go_main(allow_tree=True)``.
RUN_PLAN_ROUTE = "prepared"

#: The run-plan document schema this door writes.
RUN_PLAN_SCHEMA = "gpuwm.run-plan.v1"


def course_driver(row: Mapping[str, Any]) -> str:
    """Which shipped door runs this course, from the course's own shape.

    ``woof go`` REFUSES a multi-domain config by name -- its printed
    relay binds three single-domain digests where a tree binds one
    preparation receipt -- and reaching past that refusal is not what
    this door is for.  ``woof run-plan``'s ``prepared`` route is the
    SAME chain entered with trees allowed, so a tree course goes there.

    Derived from ``domains`` rather than declared as a column: a future
    course gets the right door by describing itself accurately, which is
    one less field to get wrong.
    """

    return "run-plan" if int(row.get("domains", 1)) > 1 else "go"


def run_plan_document(row: Mapping[str, Any], *, config: Path,
                      case_dir: Path, staged: Path,
                      geog_root: Path | None) -> dict[str, Any]:
    """The run-plan JSON for a course, with the course's own product set."""

    options: dict[str, Any] = {
        "data_dir": str(staged),
        "render_products": ",".join(sorted(set(row["products"]))),
    }
    if geog_root is not None:
        options["geog_root"] = str(geog_root)
    return {
        "schema": RUN_PLAN_SCHEMA,
        "name": f"speedrun-{row['id']}",
        "route": RUN_PLAN_ROUTE,
        "config": {"path": str(config)},
        "output_root": str(case_dir),
        "run_options": options,
    }


def _go_command(config: Path, *, case_dir: Path, staged: Path,
                products: list[str], geog_root: Path | None) -> list[str]:
    command = [sys.executable, "-m", "woof.cli", "go", str(config),
               "--outdir", str(case_dir),
               "--data-dir", str(staged),
               "--products", ",".join(products)]
    if geog_root is not None:
        command += ["--geog-root", str(geog_root)]
    return command


def _run_plan_command(plan_path: Path) -> list[str]:
    return [sys.executable, "-m", "woof.cli", "run-plan", str(plan_path)]


def run_course(args) -> int:
    row = speedrun.course(args.course)
    assets = speedrun.course_assets(args.course)
    config = assets["experiment_config"]
    staged = Path(args.staged).expanduser().resolve()
    inputs = _staged_inputs(row, staged)

    declared_mode = args.compile_mode or row["compile_mode"]
    if args.cold_cache_dir is not None:
        cache_dir = _prepare_cold_cache(
            Path(args.cold_cache_dir).expanduser().resolve())
        os.environ["CUPY_CACHE_DIR"] = str(cache_dir)
    else:
        from woof import kernel_compile_notice as notice

        cache_dir = notice.cupy_kernel_cache_dir()
    census = _cache_census(cache_dir)
    measured_mode = speedrun.measured_compile_mode(
        census["entries_for_this_card"])
    speedrun.assert_compile_mode(
        declared_mode, measured=measured_mode, cache_dir=cache_dir,
        entries=census["entries"],
        entries_for_this_card=census["entries_for_this_card"])

    device = _device_block()
    stack = _numerical_stack_block()
    stack["renderer"] = renderer_block()

    out_root = Path(args.out).expanduser().resolve()
    case_dir = out_root / row["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    geog_root = (Path(args.geog_root).expanduser().resolve()
                 if args.geog_root else None)
    driver = course_driver(row)
    if driver == "run-plan":
        plan_path = case_dir / f"speedrun-{row['id']}.run-plan.json"
        plan_path.write_text(
            json.dumps(run_plan_document(
                row, config=config, case_dir=case_dir, staged=staged,
                geog_root=geog_root), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        command = _run_plan_command(plan_path)
    else:
        command = _go_command(config, case_dir=case_dir, staged=staged,
                              products=sorted(set(row["products"])),
                              geog_root=geog_root)

    print(f"speedrun: course {row['id']} -- {row['title']}")
    print(f"speedrun: driver {driver} ({row['domains']} domain(s))")
    print(f"speedrun: staged inputs {len(inputs)} file(s) from {staged} "
          "(OFF the clock)")
    print(f"speedrun: kernel cache {declared_mode.upper()} -- "
          f"{census['entries']} entry(s) in {cache_dir}, "
          f"{census['entries_for_this_card']} for this card")
    print("speedrun: THE CLOCK STARTS NOW")

    started_at = _utcnow()
    started = time.perf_counter()
    completed = subprocess.run(command)
    wall = time.perf_counter() - started
    finished_at = _utcnow()

    run_root = resolve_run_root(case_dir, driver=driver)
    stages, summary = stage_walls(run_root, case_dir=case_dir)
    off_clock = {name: stages.pop(name)
                 for name in OFF_CLOCK_STAGES if name in stages}
    accounted = sum(stages.values()) + sum(off_clock.values())
    clock_wall = round(wall - sum(off_clock.values()), 3)
    stages = {name: round(value, 3) for name, value in stages.items()}
    stages["orchestration"] = round(max(wall - accounted, 0.0), 3)

    compile_seconds, compile_source = _kernel_compile_seconds(run_root)
    artifacts = run_artifacts(run_root)
    report = artifacts["report"]
    frames = artifacts["wrfout"]
    products = artifacts["products_rendered"]
    files = artifacts["files"]

    capsule = speedrun.seal(speedrun.capsule_body(
        course_id=row["id"],
        engine=_engine_block(command),
        machine=_machine_block(),
        device=device,
        numerical_stack=stack,
        config={
            "experiment_config": str(config),
            "experiment_config_sha256": speedrun.digest_file(config),
            "wps_namelist": str(assets["wps_namelist"]),
            "wps_namelist_sha256": speedrun.digest_file(
                assets["wps_namelist"]),
            "command": command,
            "driver": driver,
            "exit_code": completed.returncode,
            "run_root": str(run_root),
        },
        compile_block={
            "mode": measured_mode,
            "declared_by_course": row["compile_mode"],
            "declared_by_run": declared_mode,
            "cache_dir": str(cache_dir),
            "cache_entries_before": census["entries"],
            "cache_entries_for_this_card_before": census[
                "entries_for_this_card"],
            "cache_architectures_before": census["architectures"],
            "kernel_compile_seconds": compile_seconds,
            "measured_as": compile_source,
            # STATED, and the same on every record this door writes.
            "included_in_clock": True,
        },
        clock={
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": clock_wall,
            "process_wall_seconds": round(wall, 3),
            "stages": stages,
            "off_clock": {name: round(value, 3)
                          for name, value in off_clock.items()},
            "off_clock_note":
                "fetch is not on the speedrun clock; with --data-dir it "
                "revalidates the staged bytes and downloads nothing.  Its "
                "wall is subtracted from process_wall_seconds by name.",
            "time_to_first_plot_seconds": summary.get(
                "time_to_first_plot_seconds"),
        },
        evidence={
            "staged_inputs": inputs,
            "wrfout_frames": len(frames),
            "wrfout": frames,
            "product_files": len(files),
            "products_rendered": products,
            "forecast_validity": report.get("status", "unavailable"),
            "go_exit_code": completed.returncode,
        },
        outputs={
            "render_root": str(artifacts["render_root"]),
            "product_set_sha256": _product_set_digest(files),
            "files": files,
        },
    ))

    capsule_path = run_root / CAPSULE_FILENAME
    capsule_path.write_text(
        json.dumps(capsule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    status, reasons = speedrun.evidence_verdict(capsule)
    _print_record(capsule, capsule_path, reasons)
    if completed.returncode != 0:
        print(f"speedrun: the chain exited {completed.returncode}; the "
              "capsule records that and is VOID", file=sys.stderr)
        return completed.returncode
    return 0 if status == speedrun.VALID else 1


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _print_record(capsule: dict, path: Path, reasons: list[str]) -> None:
    row = speedrun.record_row(capsule)
    print()
    print(f"speedrun: {row['status']}  {row['course']}  {row['wall']}  "
          f"on {row['device']}")
    print(f"  prepare {row['prepare']}   forecast {row['forecast']}   "
          f"render {row['render']}")
    print(f"  kernel compile {row['kernel_compile']} "
          f"({row['compile']} cache, inside the clock)")
    print(f"  determinism: {row['determinism']}")
    print(f"  capsule {path}")
    print(f"  capsule sha256 {row['capsule_sha256']}")
    for reason in reasons:
        print(f"  VOID: {reason}", file=sys.stderr)


# ---------------------------------------------------------------------------
# The read-only doors
# ---------------------------------------------------------------------------

def list_courses(args) -> int:
    table = speedrun.load_course_table()
    if args.json:
        print(json.dumps({
            "schema": speedrun.COURSE_TABLE_SCHEMA_ID,
            "courses": {
                course_id: dict(row, course_sha256=speedrun.course_digest(row),
                                product_set_sha256=speedrun.product_set_digest(
                                    row["products"]))
                for course_id, row in table.items()},
        }, indent=2, sort_keys=True))
        return 0
    for course_id, row in sorted(table.items()):
        print(f"{course_id}")
        print(f"  {row['title']}")
        print(f"  {row['summary']}")
        print(f"  source {row['source']}, {row['domains']} domain(s), "
              f"{row['forecast_seconds'] / 3600:g} h forecast, "
              f"{row['compile_mode']} kernel cache")
        print(f"  fits a card of {row['vram_gib_minimum']} GiB or more")
        print(f"  products ({row['product_set_id']}): "
              f"{', '.join(sorted(row['products']))}")
        print(f"  stage the bytes first (off the clock):")
        print(f"    {row['fetch_command']}")
        print()
    return 0


def verify_capsule(args) -> int:
    capsule = speedrun.load_capsule(args.verify)
    status, reasons = speedrun.evidence_verdict(capsule)
    row = speedrun.record_row(capsule)
    print(f"speedrun: seal OK -- {row['capsule_sha256']}")
    print(f"speedrun: {status}  {row['course']}  {row['wall']}  "
          f"on {row['device']}  (woof {row['version']} {row['commit']})")
    print(f"speedrun: comparability key {speedrun.comparability_key(capsule)}")
    for reason in reasons:
        print(f"  VOID: {reason}", file=sys.stderr)
    return 0 if status == speedrun.VALID else 1


def compare_capsules(args) -> int:
    left = speedrun.load_capsule(args.compare[0])
    right = speedrun.load_capsule(args.compare[1])
    result = speedrun.compare(left, right,
                              left_name=str(args.compare[0]),
                              right_name=str(args.compare[1]))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def screen_determinism(args) -> int:
    arm_a = speedrun.load_capsule(args.determinism[0])
    arm_b = speedrun.load_capsule(args.determinism[1])
    block = speedrun.determinism_screen(arm_a, arm_b)
    print(json.dumps(block, indent=2, sort_keys=True))
    return 0 if block["claim"] else 1


def leaderboard(args) -> int:
    """One markdown table per comparability class, from sealed capsules."""

    classes: dict[str, list[dict]] = {}
    for path in args.leaderboard:
        capsule = speedrun.load_capsule(path)
        classes.setdefault(speedrun.comparability_key(capsule), []).append(
            capsule)
    for key, capsules in sorted(classes.items()):
        head = capsules[0]
        print(f"### {head['course']['id']} -- {head['compile']['mode']} "
              "kernel cache")
        print(f"comparability key `{key}`")
        print()
        print("| # | wall | prepare | forecast | render | of which kernel "
              "compile | card | version | commit | determinism | capsule |")
        print("|---|------|---------|----------|--------|------------------"
              "-------|------|---------|--------|-------------|---------|")
        ranked = sorted(capsules, key=lambda c: c["clock"]["wall_seconds"])
        for rank, capsule in enumerate(ranked, start=1):
            row = speedrun.record_row(capsule)
            print(f"| {rank} | {row['wall']} | {row['prepare']} | "
                  f"{row['forecast']} | {row['render']} | "
                  f"{row['kernel_compile']} | {row['device']} | "
                  f"{row['version']} | `{row['commit']}` | "
                  f"{row['determinism']} | `{row['capsule_sha256'][:16]}` |")
        print()
    return 0


# ---------------------------------------------------------------------------
# Dispatch and registration
# ---------------------------------------------------------------------------

def speedrun_main(args) -> int:
    try:
        if args.list_courses:
            return list_courses(args)
        if args.verify is not None:
            return verify_capsule(args)
        if args.compare is not None:
            return compare_capsules(args)
        if args.determinism is not None:
            return screen_determinism(args)
        if args.leaderboard:
            return leaderboard(args)
        if args.course is None:
            args._speedrun_parser.error(
                "name a course to run, or pass --list to see them")
        return run_course(args)
    except SpeedrunRefusal as refusal:
        print(f"woof speedrun: {refusal}", file=sys.stderr)
        return 2


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "speedrun",
        help="run a named, reproducible course end to end and seal one "
             "capsule recording exactly what was run and on what -- the "
             "internal bench surface, whose clock starts when the bytes "
             "are already staged",
        description="A speedrun measures staged-inputs to finished-"
                    "products: preparation, forecast and render, with the "
                    "one-time NVRTC kernel compile inside the clock and "
                    "named separately.  Fetching is not on the clock.  "
                    "The capsule is content-sealed, so a record's numbers "
                    "cannot be retyped, and it carries the comparability "
                    "key that decides which other records it may be "
                    "ranked against.")
    parser.add_argument("course", nargs="?", default=None, metavar="COURSE",
                        help="the course id to run (`--list` shows them).  "
                             "A course is a row in the shipped course "
                             "table plus its two asset files; adding one "
                             "is table work")
    parser.add_argument("--list", action="store_true", dest="list_courses",
                        help="list the courses, their product sets and the "
                             "off-the-clock command that stages each "
                             "course's bytes")
    parser.add_argument("--json", action="store_true",
                        help="with --list, emit the table as JSON, "
                             "including each course's digest and product-"
                             "set digest")
    parser.add_argument("--staged", type=Path, default=None, metavar="DIR",
                        help="the directory holding this course's already-"
                             "staged input bytes.  Required to run a "
                             "course: the clock starts here, so the "
                             "download must have happened before the door "
                             "is called")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        metavar="DIR",
                        help=f"where the run tree and the capsule go "
                             f"(default {DEFAULT_OUT}/<course>).  The "
                             f"capsule is written as {CAPSULE_FILENAME} "
                             "inside the run's own timestamped folder")
    parser.add_argument("--geog-root", type=Path, default=None, metavar="DIR",
                        help="staged WPS_GEOG tree (default: the one "
                             "`woof fetch-geog` stages into)")
    parser.add_argument("--compile-mode", default=None, choices=COMPILE_MODES,
                        dest="compile_mode",
                        help="which kernel-cache class this record belongs "
                             "to (default: whatever the course declares).  "
                             "The door MEASURES the cache before the clock "
                             "starts and refuses a mismatch, because the "
                             "one-time NVRTC compile is roughly a minute "
                             "and it is always inside the clock -- a cold "
                             "record and a warm record are different "
                             "records and are never compared")
    parser.add_argument("--cold-cache-dir", type=Path, default=None,
                        dest="cold_cache_dir", metavar="DIR",
                        help="EMPTY this directory and point CUPY_CACHE_DIR "
                             "at it for the run, so a cold-cache record can "
                             "be set on a machine whose own cache is warm.  "
                             "Only a directory that is absent, empty, or "
                             "already a CuPy kernel cache is emptied; "
                             "anything else -- and the inherited cache, the "
                             "working directory and your home directory by "
                             "name -- is REFUSED rather than deleted")
    parser.add_argument("--verify", type=Path, default=None, metavar="CAPSULE",
                        help="verify one capsule's seal and evidence and "
                             "print its record line, instead of running "
                             "anything")
    parser.add_argument("--compare", type=Path, nargs=2, default=None,
                        metavar=("A", "B"),
                        help="compare two records.  REFUSED, by name, when "
                             "they are not records of the same course, the "
                             "same product set and the same compile mode")
    parser.add_argument("--determinism", type=Path, nargs=2, default=None,
                        metavar=("ARM_A", "ARM_B"),
                        help="the dual-run byte screen: two capsules from "
                             "two runs of one course on one machine.  These "
                             "cards carry no ECC, so this is the only thing "
                             "that may set a determinism claim")
    parser.add_argument("--leaderboard", type=Path, nargs="+", default=None,
                        metavar="CAPSULE",
                        help="emit the SPEEDRUN.md tables for these "
                             "capsules, one table per comparability class")
    parser.set_defaults(func=speedrun_main)
    parser.set_defaults(_speedrun_parser=parser)
    return parser


__all__ = ["DEFAULT_OUT", "FORECAST_SUBDIR", "OFF_CLOCK_STAGES",
           "RENDER_SUBDIR", "REPORT_FILENAME", "RUN_PLAN_CHAIN_SUBDIR",
           "RUN_OPENING_EVENTS", "RUN_PLAN_EVENTS_FILENAME", "RUN_PLAN_ROUTE",
           "RUN_PLAN_SCHEMA", "VERDICT_RECEIPTS",
           "WRFOUT_SUBDIR", "compare_capsules", "course_driver",
           "leaderboard", "list_courses", "register_cli", "renderer_block",
           "resolve_run_root", "run_artifacts", "run_course",
           "last_run_events", "walls_by_phase",
           "run_plan_document", "screen_determinism", "speedrun_main",
           "stage_walls",
           "verify_capsule"]
