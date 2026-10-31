"""``woof hex global-cycle``: run a global row every cycle into a parent store.

    woof hex global-cycle --row global-60km --store STORE            # the latest cycle, once
    woof hex global-cycle --row global-60km --store STORE --forever  # every cycle the source posts

WHAT WAS MISSING (B2, the 2026-10-16 global operations gap list): ``woof hex
global`` ran one named cycle.  Operating a global forecast every 6 h needed
someone to pick the cycle, wait until the source had posted it, keep the runs
somewhere consumers can find them, delete old ones and say what is going on.

THE PIECES, each one data or an existing door:

  latest    ``--cycle latest`` (here and on ``woof hex global``) picks the
            newest cycle of the init source that is due, from the source row's
            ``posting`` block in ``data/init-sources.v1.json`` (its cycle hours
            and measured posting lag), and confirms it by asking the source's
            own product URLs (HTTP HEAD, mirrors included).  A cycle that is
            due but not posted yet is WAITED FOR, polling, up to
            ``--max-wait-minutes``; it is never silently replaced by an older
            one, because a stale forecast that looks fresh is worse than a late
            one.  A newer cycle that posted early is taken.
  schedule  ``--forever`` runs the latest cycle, then sleeps until the next
            cycle is due and runs that, for as long as it lives.  A cycle that
            is already done in the store is not run again; a driver that falls
            more than one cycle behind runs the newest and records the skipped
            ones.
  store     ``STORE/<row>/cycles/<YYYYMMDDHH>/`` is each cycle's run folder (the
            global door's own layout: init, forecast, export, receipts) with a
            ``cycle.json`` manifest listing its history frames by valid time.
            ``STORE/<row>/latest`` (a symlink) and ``latest.json`` name the
            newest finished cycle; consumers (WOOF domains taking a hex
            parent, the export door) read those and never a half-written run.
  retention ``--keep N`` finished cycles stay; older cycle folders the driver
            made (they carry its ``cycle.json``) are deleted after each
            finished cycle, and the feed lists what went.
  feed      ``STORE/<row>/feed.json`` (atomic, one document: driver state, the
            cycle running and its stage, the next cycle and when it is due,
            every cycle in the store) and ``events.jsonl`` (append-only).

Each cycle runs ``woof hex global`` in its own process: the forecast door
binds a mesh once per process, so a second cycle in the same process would
be refused, and a crash in one cycle must not take the driver down.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import urllib.error
import urllib.request

from .errors import MpasPortError

FEED_SCHEMA = "gpuwm-hex.global-cycle-feed/v1"
MANIFEST_SCHEMA = "gpuwm-hex.global-cycle/v1"
RESOLUTION_SCHEMA = "gpuwm-hex.cycle-resolution/v1"
LOOKBACK_HOURS = 48
UTC = _dt.timezone.utc


class CycleDriverError(MpasPortError):
    """The cycle driver refuses, and says why."""


def utcnow() -> _dt.datetime:
    return _dt.datetime.now(UTC)


def iso(when: _dt.datetime) -> str:
    return when.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cycle_text(when: _dt.datetime) -> str:
    return when.strftime("%Y-%m-%dT%H")


# ---------------------------------------------------------------------------
# when a source posts
# ---------------------------------------------------------------------------
def posting(row: Mapping[str, Any]) -> dict[str, Any]:
    block = row.get("posting")
    if not block:
        raise CycleDriverError(
            f"init source {row['id']!r} declares no posting block (cycle hours and posting lag) in "
            f"data/init-sources.v1.json, so there is no 'latest' cycle to pick; pass --cycle YYYY-MM-DDTHH")
    return dict(block)


def cycles_back(block: Mapping[str, Any], now: _dt.datetime, hours: int = LOOKBACK_HOURS) -> list[_dt.datetime]:
    """The source's cycle times at or before ``now``, newest first, over ``hours``."""

    hours_of_day = sorted({int(h) for h in block["cycle_hours"]})
    found = []
    day = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    for back in range(hours // 24 + 2):
        for hour in reversed(hours_of_day):
            when = day - _dt.timedelta(days=back) + _dt.timedelta(hours=hour)
            if when <= now and now - when <= _dt.timedelta(hours=hours):
                found.append(when)
    return sorted(set(found), reverse=True)


def next_cycle(block: Mapping[str, Any], after: _dt.datetime) -> _dt.datetime:
    hours_of_day = sorted({int(h) for h in block["cycle_hours"]})
    day = after.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    for back in range(3):
        for hour in hours_of_day:
            when = day + _dt.timedelta(days=back, hours=hour)
            if when > after:
                return when
    raise CycleDriverError(f"no cycle after {iso(after)} in cycle hours {hours_of_day}")


def due_at(block: Mapping[str, Any], cycle: _dt.datetime) -> _dt.datetime:
    return cycle + _dt.timedelta(minutes=float(block["lag_minutes"]))


def head(url: str, timeout: float = 20.0) -> bool:
    """True when ``url`` answers a HEAD with a 2xx status."""

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gpuwm-hex-cycle/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def product_addresses(row: Mapping[str, Any], cycle: _dt.datetime, member: int | None) -> list[list[str]]:
    """Every https product of the cycle, each as its URL and mirrors (any one answering is enough)."""

    from .init_sources import Cycle, plan_products

    if row.get("members") is not None and member is None:
        control = (row["members"].get("control") or {}).get("number")
        member = control if control is not None else row["members"]["perturbed"]["first"]
    plans = plan_products(dict(row), Cycle(cycle.astimezone(UTC)), member)
    addresses = [[p.url, *p.url_mirrors] for p in plans if p.transport == "https" and p.url]
    if not addresses:
        raise CycleDriverError(
            f"init source {row['id']!r} has no https product to ask whether a cycle is posted")
    return addresses


def posted(row: Mapping[str, Any], cycle: _dt.datetime, member: int | None,
           ask: Callable[[str], bool] = head) -> tuple[bool, str | None]:
    """Whether every product of ``cycle`` answers; else the first address that does not."""

    addresses = product_addresses(row, cycle, member)

    def one(urls: list[str]) -> str | None:
        return None if any(ask(url) for url in urls) else urls[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(addresses))) as pool:
        missing = [m for m in pool.map(one, addresses) if m is not None]
    return (not missing, missing[0] if missing else None)


def resolve_latest(row: Mapping[str, Any], *, member: int | None = None, max_wait_minutes: float = 120.0,
                   poll_seconds: float = 60.0, not_before: _dt.datetime | None = None,
                   clock: Callable[[], _dt.datetime] | None = None, sleep: Callable[[float], None] | None = None,
                   ask: Callable[[str], bool] | None = None, say: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The newest cycle of ``row`` that is posted, waiting for the newest DUE one.

    Newest first: a cycle not yet due is taken only if it already answers; the
    first due cycle is the answer, waited for until it answers or
    ``max_wait_minutes`` past its due time, then refused by name.  Cycles at or
    before ``not_before`` are not candidates (the driver's last finished cycle)."""

    clock = clock if clock is not None else utcnow
    sleep = sleep if sleep is not None else time.sleep
    ask = ask if ask is not None else head
    block = posting(row)
    started = clock()
    checked = []
    for cycle in cycles_back(block, started):
        if not_before is not None and cycle <= not_before:
            break
        due = due_at(block, cycle)
        if clock() < due:
            ok, missing = posted(row, cycle, member, ask)
            checked.append({"cycle": cycle_text(cycle), "due_utc": iso(due), "posted": ok})
            if ok:
                return _resolution(row, cycle, block, started, clock(), checked, "posted before its usual time")
            continue
        deadline = max(started, due) + _dt.timedelta(minutes=max_wait_minutes)
        while True:
            ok, missing = posted(row, cycle, member, ask)
            if ok:
                checked.append({"cycle": cycle_text(cycle), "due_utc": iso(due), "posted": True})
                return _resolution(row, cycle, block, started, clock(), checked, "the newest due cycle")
            now = clock()
            if now >= deadline:
                checked.append({"cycle": cycle_text(cycle), "due_utc": iso(due), "posted": False,
                                "missing": missing})
                raise CycleDriverError(
                    f"{row['id']} {cycle_text(cycle)} is due since {iso(due)} (posting lag "
                    f"{block['lag_minutes']} min) but {missing} still does not answer after waiting until "
                    f"{iso(now)}; an older cycle is not run in its place (a stale forecast must not look "
                    f"fresh). Raise --max-wait-minutes, or pass --cycle for an older cycle explicitly")
            if say is not None:
                say(f"waiting for {row['id']} {cycle_text(cycle)}: {missing} not posted yet "
                    f"(due {iso(due)}, giving up at {iso(deadline)})")
            sleep(min(poll_seconds, max(1.0, (deadline - now).total_seconds())))
    raise CycleDriverError(
        f"no cycle of {row['id']} newer than {iso(not_before) if not_before else 'the lookback'} is due or "
        f"posted in the last {LOOKBACK_HOURS} h (cycle hours {block['cycle_hours']}, lag {block['lag_minutes']} min)")


def _resolution(row, cycle, block, started, finished, checked, why) -> dict[str, Any]:
    return {"schema": RESOLUTION_SCHEMA, "source": row["id"], "cycle": cycle_text(cycle), "why": why,
            "due_utc": iso(due_at(block, cycle)), "posting": dict(block), "checked": checked,
            "asked_utc": iso(started), "resolved_utc": iso(finished),
            "waited_seconds": round((finished - started).total_seconds(), 1)}


def source_row(name: str) -> dict[str, Any]:
    from .init_sources import load_table, resolve_source

    return resolve_source(load_table(None), name)


def resolve_cycle_argument(args: argparse.Namespace, say: Callable[[str], None] | None = None,
                           wait: bool = True) -> dict[str, Any] | None:
    """For the global door: turn ``--cycle latest`` into a cycle in place; None for an explicit cycle."""

    if str(getattr(args, "cycle", "") or "").lower() != "latest":
        return None
    resolution = resolve_latest(source_row(args.source), member=getattr(args, "member", None),
                                max_wait_minutes=float(args.max_wait_minutes) if wait else 0.0,
                                poll_seconds=float(args.poll_seconds), say=say)
    args.cycle = resolution["cycle"]
    return resolution


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Store:
    def __init__(self, root: Path, row: str) -> None:
        self.root = Path(root).expanduser().absolute() / row
        self.row = row
        self.cycles = self.root / "cycles"
        self.feed_path = self.root / "feed.json"
        self.events_path = self.root / "events.jsonl"

    def cycle_dir(self, cycle: str) -> Path:
        return self.cycles / cycle.replace("-", "").replace("T", "")

    def manifests(self) -> list[dict[str, Any]]:
        found = []
        if self.cycles.is_dir():
            for path in sorted(self.cycles.iterdir(), reverse=True):
                manifest = _read_json(path / "cycle.json")
                if manifest and manifest.get("schema") == MANIFEST_SCHEMA:
                    found.append({**manifest, "run_dir": str(path)})
        return found

    def done(self) -> list[dict[str, Any]]:
        return [m for m in self.manifests() if m.get("state") == "done"]

    def event(self, kind: str, **fields: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": FEED_SCHEMA, "event": kind, "utc": iso(utcnow()), **fields},
                                    sort_keys=True) + "\n")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """One driver per store row.  Prevents: two drivers (an overlapping cron line, a second
        --forever) running the same cycle into the same folder, the second refused by the
        forecast door mid-way and its failure recorded against a cycle that finished."""

        from .file_lock import lock as _flock, unlock as _funlock

        self.root.mkdir(parents=True, exist_ok=True)
        handle = (self.root / "driver.lock").open("a+")
        try:
            _flock(handle, blocking=False)
        except OSError:
            handle.close()
            raise CycleDriverError(f"another cycle driver holds {self.root / 'driver.lock'}; one driver "
                                   "per store row") from None
        try:
            yield
        finally:
            _funlock(handle)
            handle.close()


def frames(run_dir: Path, start: _dt.datetime) -> list[dict[str, Any]]:
    """The history frames a finished run left, by valid time and lead hour."""

    rows = []
    from .products_door import discover_histories, frame_stamp, ProductDoorError

    paths = []
    completed_native = set()
    folders = [run_dir / "forecast", *sorted((run_dir / "ensemble").glob("g*/mem*"))]
    for folder in folders:
        try:
            paths.extend(discover_histories(folder))
        except ProductDoorError:
            continue
        # Native MPAS owns stream I/O and closes its files at finalize. Only its
        # successful final receipt admits files without the CUDA writer's marker.
        try:
            if json.loads((folder / "native-physics-receipt.json").read_text()).get("completed") is True:
                completed_native.add(folder)
        except (OSError, ValueError):
            pass
    for path in paths:
        stamp = frame_stamp(path)
        try:
            valid = _dt.datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=UTC)
        except (TypeError, ValueError):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.parent not in completed_native:
            # The CUDA writer's receipt certifies one closed file: a stale or
            # invalid receipt (a failed rerun overwrote the frame) admits nothing.
            durable = _read_json(path.with_name(path.name + ".durable.json"))
            if not isinstance(durable, dict) or (
                    durable.get("schema"), durable.get("file"), durable.get("bytes"),
                    durable.get("mtime_ns")) != (
                    "gpuwm.hex-history-durable/v1", path.name, stat.st_size, stat.st_mtime_ns):
                continue  # a frame without its matching durability receipt is not published
        rows.append({"path": str(path), "valid_utc": iso(valid),
                     **({"member": path.parent.name} if path.parent.name.startswith("mem") else {}),
                     "lead_hours": round((valid - start).total_seconds() / 3600.0, 4),
                     "bytes": stat.st_size})
    return rows


def publish_latest(store: Store, run_dir: Path, manifest: Mapping[str, Any]) -> None:
    link = store.root / "latest"
    tmp = store.root / f".latest.{os.getpid()}"
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    os.symlink(os.path.relpath(run_dir, store.root), tmp)
    os.replace(tmp, link)
    _write_json(store.root / "latest.json", {"schema": MANIFEST_SCHEMA, "cycle": manifest["cycle"],
                                             "run_dir": str(run_dir), "manifest": str(run_dir / "cycle.json"),
                                             "published_utc": iso(utcnow())})


def apply_retention(store: Store, keep: int) -> list[str]:
    """Delete driver-made cycle folders older than the newest ``keep`` finished cycles."""

    if keep < 1:
        raise CycleDriverError("--keep must be at least 1: the latest finished cycle is what consumers read")
    done = store.done()
    if len(done) <= keep:
        return []
    oldest_kept = done[keep - 1]["cycle"]
    removed = []
    for manifest in store.manifests():
        if manifest["cycle"] < oldest_kept and manifest.get("state") != "running":
            shutil.rmtree(manifest["run_dir"], ignore_errors=True)
            removed.append(manifest["run_dir"])
    if store.cycles.is_dir():  # earlier attempts of a re-run cycle, renamed aside
        for path in store.cycles.iterdir():
            base = path.name.split(".attempt-")[0]
            if ".attempt-" in path.name and base < oldest_kept.replace("-", "").replace("T", ""):
                shutil.rmtree(path, ignore_errors=True)
                removed.append(str(path))
    return removed


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.row_id = args.row
        self.store = Store(args.store, args.row)
        self.source = source_row(args.source)
        self.block = posting(self.source) if str(args.cycle).lower() == "latest" else None
        self.feed: dict[str, Any] = {
            "schema": FEED_SCHEMA, "row": args.row, "source": self.source["id"], "store": str(self.store.root),
            "driver": {"pid": os.getpid(), "host": socket.gethostname(), "started_utc": iso(utcnow()),
                       "mode": "forever" if args.forever else "once", "keep": args.keep},
            "state": "starting", "current": None, "next": None, "removed": [], "skipped": []}

    def say(self, text: str) -> None:
        print(f"[global-cycle] {text}", flush=True)
        self.store.event("note", text=text)
        self.write_feed(note=text)

    def write_feed(self, **changes: Any) -> None:
        self.feed.update(changes)
        self.feed["updated_utc"] = iso(utcnow())
        self.feed["cycles"] = [{key: m.get(key) for key in (
            "cycle", "state", "started_utc", "finished_utc", "wall_seconds", "run_dir", "returncode", "why",
            "frame_count", "resolution")} for m in self.store.manifests()]
        done = self.store.done()
        self.feed["latest_done"] = done[0]["cycle"] if done else None
        current = self.feed.get("current")
        if current and current.get("run_dir"):
            status = _read_json(Path(current["run_dir"]) / "status.json")
            if status:
                current["stage"], current["stage_state"] = status.get("stage"), status.get("state")
        self.store.root.mkdir(parents=True, exist_ok=True)
        _write_json(self.store.feed_path, self.feed)

    def door_argv(self, cycle: str, run_dir: Path) -> list[str]:
        a = self.args
        argv = ["global", "--row", a.row, "--cycle", cycle, "--work-dir", str(run_dir), "--source", a.source]
        for flag, value in (("--physics-row", a.physics_row), ("--hours", a.hours),
                            ("--history-every-minutes", a.history_every_minutes), ("--devices", a.devices),
                            ("--cache-dir", a.cache_dir), ("--geog", a.geog), ("--table", a.table),
                            ("--observation-config", a.observation_config), ("--member", a.member),
                            ("--members", a.members), ("--members-per-card", a.members_per_card),
                            ("--init-jobs", a.init_jobs)):
            if value is not None:
                argv += [flag, str(value)]
        from woof.output_formats import argument_values
        argv += argument_values(a, relative=cycle)
        argv += ["--export-field-set", a.export_field_set]
        for value in a.forecast_arg or []:
            argv.append(f"--forecast-arg={value}")
        for stage in a.skip or []:
            argv += ["--skip", stage]
        return argv

    def run_cycle(self, cycle: str, resolution: Mapping[str, Any] | None) -> int:
        run_dir = self.store.cycle_dir(cycle)
        old = _read_json(run_dir / "cycle.json")
        if old and old.get("state") == "done" and not self.args.rerun:
            self.say(f"{cycle} is already done in the store ({run_dir}); not run again")
            return 0
        if run_dir.exists():
            attempt = 1
            while run_dir.with_name(f"{run_dir.name}.attempt-{attempt}").exists():
                attempt += 1
            run_dir.rename(run_dir.with_name(f"{run_dir.name}.attempt-{attempt}"))
        run_dir.mkdir(parents=True)
        start = _dt.datetime.strptime(cycle, "%Y-%m-%dT%H").replace(tzinfo=UTC)
        manifest = {"schema": MANIFEST_SCHEMA, "row": self.row_id, "source": self.source["id"], "cycle": cycle,
                    "state": "running", "started_utc": iso(utcnow()), "resolution": resolution,
                    "command": ["woof hex", *self.door_argv(cycle, run_dir)]}
        _write_json(run_dir / "cycle.json", manifest)
        self.store.event("cycle-start", cycle=cycle, run_dir=str(run_dir))
        self.write_feed(state="running", current={"cycle": cycle, "run_dir": str(run_dir),
                                                  "started_utc": manifest["started_utc"]})
        began = time.monotonic()
        with (run_dir / "driver-door.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "woof.hex.cli", *self.door_argv(cycle, run_dir)],
                                       stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(15)
                self.write_feed()  # carries the run's own stage from its status.json
        rc = process.returncode
        receipt = _read_json(run_dir / "global-receipt.json") or {}
        manifest.update(state="done" if rc == 0 else "failed", returncode=rc, finished_utc=iso(utcnow()),
                        wall_seconds=round(time.monotonic() - began, 1), receipt=str(run_dir / "global-receipt.json"),
                        stages=receipt.get("stages"), why=receipt.get("refused") or receipt.get("failed"))
        published = frames(run_dir, start)
        manifest.update(frames=published, frame_count=len(published))
        _write_json(run_dir / "cycle.json", manifest)
        self.store.event("cycle-end", cycle=cycle, state=manifest["state"], returncode=rc)
        if rc == 0:
            # latest names the NEWEST finished cycle: an older cycle run by hand never moves it back.
            newest = self.store.done()[0]
            publish_latest(self.store, Path(newest["run_dir"]), newest)
            removed = apply_retention(self.store, self.args.keep)
            if removed:
                self.store.event("retention", removed=removed)
                self.feed["removed"] = (self.feed.get("removed") or []) + removed
        self.write_feed(state="idle" if rc == 0 else "failed", current=None)
        return rc

    def once(self) -> int:
        if self.block is None:
            return self.run_cycle(self.args.cycle, None)
        resolution = resolve_latest(self.source, max_wait_minutes=self.args.max_wait_minutes,
                                    poll_seconds=self.args.poll_seconds, say=self.say)
        return self.run_cycle(resolution["cycle"], resolution)

    def forever(self) -> int:
        if self.block is None:
            raise CycleDriverError("--forever follows the source's schedule; it needs --cycle latest")
        attempted: _dt.datetime | None = None
        while True:
            done = self.store.done()
            last = (_dt.datetime.strptime(done[0]["cycle"], "%Y-%m-%dT%H").replace(tzinfo=UTC)
                    if done else None)
            # A cycle that failed here is not retried in a loop: the schedule moves to the next one
            # (rerun it by hand with --cycle and --rerun once the cause is fixed).
            if attempted is not None and (last is None or attempted > last):
                last = attempted
            if last is not None:
                upcoming = next_cycle(self.block, last)
                due = due_at(self.block, upcoming)
                self.write_feed(state="waiting", next={"cycle": cycle_text(upcoming), "due_utc": iso(due)})
                while utcnow() < due:
                    time.sleep(min(300.0, max(1.0, (due - utcnow()).total_seconds())))
            try:
                resolution = resolve_latest(self.source, max_wait_minutes=self.args.max_wait_minutes,
                                            poll_seconds=self.args.poll_seconds, not_before=last, say=self.say)
            except CycleDriverError as error:
                self.say(f"no cycle run: {error}")
                time.sleep(self.args.poll_seconds)
                continue
            chosen = _dt.datetime.strptime(resolution["cycle"], "%Y-%m-%dT%H").replace(tzinfo=UTC)
            if last is not None:
                skipped = []
                probe = next_cycle(self.block, last)
                while probe < chosen:
                    skipped.append(cycle_text(probe))
                    probe = next_cycle(self.block, probe)
                if skipped:
                    self.feed["skipped"] = (self.feed.get("skipped") or []) + skipped
                    self.store.event("skipped", cycles=skipped, ran=resolution["cycle"])
            attempted = chosen
            self.run_cycle(resolution["cycle"], resolution)


def run_global_cycle(args: argparse.Namespace) -> int:
    if args.store is None or args.row is None:
        raise CycleDriverError("--row and --store are required")
    driver = Driver(args)
    with driver.store.lock():
        driver.write_feed(state="starting")
        try:
            return driver.forever() if args.forever else driver.once()
        except CycleDriverError as error:
            driver.write_feed(state="refused", why=str(error))
            raise
        except BaseException as error:
            driver.write_feed(state="stopped", why=f"{type(error).__name__}: {error}")
            raise


def add_cycle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-wait-minutes", type=float, default=120.0,
                        help="with --cycle latest: how long past its due time to wait for the newest due "
                             "cycle to post before refusing (default 120)")
    parser.add_argument("--poll-seconds", type=float, default=60.0,
                        help="with --cycle latest: seconds between checks while waiting (default 60)")


def add_global_cycle_parser(commands: Any) -> None:
    from .global_door import _add_common

    parser = commands.add_parser(
        "global-cycle", help="run a global row every cycle into a parent store (latest, schedule, "
                             "retention, status feed)", description=__doc__.split("\n\n")[0])
    _add_common(parser)
    parser.set_defaults(cycle="latest")
    parser.add_argument("--store", type=Path, default=None, help="the parent store root (STORE/<row>/...)")
    parser.add_argument("--forever", action="store_true",
                        help="keep the source's schedule: run each cycle as it posts (default: one cycle)")
    parser.add_argument("--keep", type=int, default=4, help="finished cycles kept in the store (default 4)")
    parser.add_argument("--rerun", action="store_true", help="run a cycle again even when the store has it done")
    parser.add_argument("--skip", action="append", default=[], metavar="STAGE",
                        help="leave one stage of woof hex global out, repeatable")
    parser.set_defaults(handler=run_global_cycle)
