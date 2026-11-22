"""Leads written by a run on this machine, published as an as-posted fetch.

``python -m woof.local_posting`` is the as-posted fetch for a source whose
files are not downloaded but WRITTEN, one valid time at a time, by another
run beside this one -- a woof hex parent forecast exporting its frames
(``woof hex export --follow``) is the first such source.

THE BREAKAGE THIS PREVENTS: a WOOF-ArWen domain driven by a hex parent could only
be prepared once the whole parent forecast and its export had finished,
because ``woof prep --source hex-global`` read a complete frame set.  The
nest then ran strictly after its parent, so the first nest map came a whole
parent forecast late.  The engine already prepares a window as it posts
(``woof prep --as-posted POSTING_DIR``: the preparation starts on the first
leads, publishes its head, and the forecast runs while later boundary
intervals are built, waiting at a seam only when its source is behind).
What a locally written source lacked was the fetch half: nothing wrote the
posting folder that preparation reads.  This module writes it, in the same
documents the network fetch writes (:mod:`woof.fetch_as_posted`):

``<out>/fetch-manifest.json``  the window's planned files (``prep.primary_files``
                               and ``prep.supplement_files``), written first;
``<out>/posting/schedule.json`` ``gpuwm.posting-schedule.v1``, every lead
                               ``scheduled`` and moved to ``ready`` as it lands;
``<out>/posting/fNNN.json``    ``gpuwm.posted-lead.v1`` per lead, written after
                               the lead's file is in ``<out>/frames/`` and its
                               SHA-256 is computed, in lead order;
``<out>/posting/failed.json``  when the producer ends without a lead or the
                               window goes quiet past ``--quiet-minutes``.

A lead is ready when the producer's ready list (``--ready-list``, one JSON row
per finished file naming it under ``frame`` or ``path``) names its file, or,
without a list, when the file exists under ``--watch`` (the producer must then
publish each file by an atomic rename).  The file is hard-linked into
``<out>/frames/`` (copied across filesystems), so the input manifest the
preparation's seal writes names files inside the fetch folder, as it must.

Nothing here names a source.  A source opts in with its adapter row's
``local_posting`` column (``woof.source_adapters``), which is what lets
``woof prep --as-posted`` accept it without a network posting row; a new
locally produced source is that one column, not a code path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Mapping, Sequence

from woof.fetch_as_posted import MARKER_SCHEMA, marker_name
from woof.source_posting import (POSTING_DIRNAME, POSTING_FAILED_NAME,
                                  POSTING_SCHEDULE_NAME, SCHEDULE_SCHEMA)

FETCH_MANIFEST_NAME = "fetch-manifest.json"
LOCAL_POSTING_SCHEMA = "gpuwm.local-posting.v1"
FRAMES_DIRNAME = "frames"
SOURCE_BEHIND_CODE = "source_behind"
PRODUCER_ENDED_CODE = "producer_ended"
DEFAULT_NAME_TEMPLATE = "wrfout_d01_{valid:%Y-%m-%d_%H_%M_%S}"


def _instant(moment: datetime | None) -> str | None:
    return None if moment is None else moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def parse_cycle(text: str) -> datetime:
    for pattern in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d_%H:%M:%S",
                    "%Y%m%d%H"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise ValueError(f"--cycle {text!r}: give YYYY-MM-DDTHH")


class LocalPosting:
    """One window of a locally written source, published lead by lead."""

    def __init__(self, *, source: str, cycle: datetime, leads: Sequence[int],
                 out: Path, name_template: str = DEFAULT_NAME_TEMPLATE,
                 supplement_roles: Sequence[str] = (),
                 ready_list: Path | None = None, watch: Path | None = None,
                 producer_pid: int | None = None, quiet_minutes: float = 240.0,
                 poll_seconds: float = 1.0, progress=print,
                 lead_minutes: int = 60) -> None:
        if (ready_list is None) == (watch is None):
            raise ValueError("give exactly one of --ready-list and --watch")
        ordered = sorted({int(lead) for lead in leads})
        if not ordered or ordered[0] < 0:
            raise ValueError("a local posting needs one or more nonnegative leads")
        if int(lead_minutes) != lead_minutes or lead_minutes <= 0:
            raise ValueError("--lead-minutes must be a positive whole number of minutes")
        self.source = str(source)
        self.cycle = cycle
        #: One lead's span: lead N is valid at cycle + N x this.  An hourly
        #: posting (the default, and every network fetch) spells nothing
        #: new, so its declaration and documents are unchanged byte for
        #: byte; a sub-hourly one (a hex parent's 15-minute frames) states
        #: its span in the declaration and the schedule.
        self.lead_minutes = int(lead_minutes)
        self.leads = tuple(ordered)
        self.out = Path(out)
        self.folder = self.out / POSTING_DIRNAME
        self.frames = self.out / FRAMES_DIRNAME
        self.name_template = name_template
        self.supplement_roles = tuple(supplement_roles)
        self.ready_list = None if ready_list is None else Path(ready_list)
        self.watch = None if watch is None else Path(watch)
        self.producer_pid = producer_pid
        self.quiet_seconds = float(quiet_minutes) * 60.0
        self.poll_seconds = float(poll_seconds)
        self.progress = progress
        self.names = {lead: self.name(lead) for lead in self.leads}
        declaration = {
            "schema": LOCAL_POSTING_SCHEMA, "source": self.source,
            "cycle": cycle.strftime("%Y-%m-%dT%H"), "leads": list(self.leads),
            "name_template": name_template,
            "supplement_roles": list(self.supplement_roles),
            "ready": ("ready_list" if self.ready_list is not None else "watch"),
            **({"lead_minutes": self.lead_minutes} if self.lead_minutes != 60 else {}),
        }
        #: Binds what this posting promised (the same place a network fetch
        #: binds its route table): an as-posted head records it, and the seal
        #: re-reads it, so a window re-planned under another declaration is
        #: another plan.
        self.declaration_sha256 = hashlib.sha256(json.dumps(
            declaration, sort_keys=True).encode("utf-8")).hexdigest()
        self.declaration = declaration
        self.rows = {lead: {
            "lead": lead,
            "valid_time": _instant(self.valid(lead)),
            "expected_at": None, "late_at": None, "first_seen_at": None,
            "fetched_at": None, "endpoint": "local", "state": "scheduled",
            "last_answer": None,
        } for lead in self.leads}

    def valid(self, lead: int) -> datetime:
        return self.cycle + timedelta(minutes=int(lead) * self.lead_minutes)

    def name(self, lead: int) -> str:
        return self.name_template.format(valid=self.valid(lead), lead=int(lead))

    def relpath(self, lead: int) -> str:
        return f"{FRAMES_DIRNAME}/{self.names[lead]}"

    # -- documents -------------------------------------------------------

    def schedule(self) -> dict[str, object]:
        return {
            "schema": SCHEDULE_SCHEMA, "source": self.source, "member": None,
            "cycle": self.cycle.strftime("%Y-%m-%dT%H"), "as_posted": True,
            "shape": "local", "streams": True,
            "why": ("written lead by lead by a run on this machine and "
                    "published by woof.local_posting as each file is finished"),
            "ready_check": ("ready_list" if self.ready_list is not None else "file"),
            "late_after_minutes": self.quiet_seconds / 60.0,
            "poll_seconds": self.poll_seconds, "publication_lag": [],
            "expected_final_at": None,
            "leads": [self.rows[lead] for lead in self.leads],
            "table_sha256": self.declaration_sha256,
            "local_posting": self.declaration,
            **({"lead_minutes": self.lead_minutes} if self.lead_minutes != 60 else {}),
        }

    def write_schedule(self) -> None:
        _write_json(self.folder / POSTING_SCHEDULE_NAME, self.schedule())

    def fetch_manifest(self) -> dict[str, object]:
        primary = [self.relpath(lead) for lead in self.leads]
        return {
            "schema": LOCAL_POSTING_SCHEMA, "source": self.source,
            "cycle": self.cycle.strftime("%Y-%m-%dT%H"),
            "prep": {"primary_files": primary,
                     "supplement_files": (primary if self.supplement_roles else [])},
            "declaration_sha256": self.declaration_sha256,
        }

    def prep_arguments(self) -> list[str]:
        """The ``woof prep`` input flags naming every planned file."""

        flags: list[str] = []
        for lead in self.leads:
            flags.extend(("--input", str(self.out / self.relpath(lead))))
        for role in self.supplement_roles:
            for lead in self.leads:
                flags.extend(("--supplement", f"{role}={self.out / self.relpath(lead)}"))
        return flags

    def start(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.frames.mkdir(parents=True, exist_ok=True)
        (self.folder / POSTING_FAILED_NAME).unlink(missing_ok=True)
        for marker in self.folder.glob("f[0-9][0-9][0-9].json"):
            marker.unlink(missing_ok=True)
        _write_json(self.out / FETCH_MANIFEST_NAME, self.fetch_manifest())
        self.write_schedule()

    def fail(self, code: str, message: str, lead: int | None = None) -> None:
        _write_json(self.folder / POSTING_FAILED_NAME, {
            "schema": "gpuwm.posting-failed.v1", "code": code,
            "source": self.source, "cycle": self.cycle.strftime("%Y-%m-%dT%H"),
            "lead": lead, "message": message, "at": _instant(_now())})

    # -- finding finished files -----------------------------------------

    def _ready_files(self) -> dict[str, dict]:
        """``{file name: row}`` for every file the producer says is finished."""

        if self.ready_list is None:
            return {}
        try:
            text = self.ready_list.read_text(encoding="utf-8")
        except OSError:
            return {}
        found: dict[str, dict] = {}
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a line still being written
            path = row.get("frame") or row.get("path") if isinstance(row, dict) else None
            if path:
                found[Path(str(path)).name] = dict(row, _path=str(path))
        return found

    def _located(self, lead: int) -> tuple[Path, dict] | None:
        name = self.names[lead]
        if self.ready_list is not None:
            row = self._ready_files().get(name)
            return None if row is None else (Path(row["_path"]), row)
        path = self.watch / name
        return (path, {}) if path.is_file() else None

    def _place(self, lead: int, path: Path, row: Mapping[str, object]) -> dict:
        target = self.out / self.relpath(lead)
        if target.exists():
            target.unlink()
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        digest = _sha256(target)
        return {"role": "primary", "name": self.relpath(lead),
                "url": None, "endpoint": "local", "bytes": target.stat().st_size,
                "sha256": digest, "lead": lead, "source_path": str(path),
                "producer_digest": row.get("sha256")}

    def publish(self, lead: int, objects: list[dict]) -> None:
        row = self.rows[lead]
        now = _instant(_now())
        row.update(first_seen_at=row["first_seen_at"] or now, fetched_at=now,
                   state="ready", last_answer="posted")
        marker = {
            "schema": MARKER_SCHEMA, "source": self.source, "member": None,
            "cycle": self.cycle.strftime("%Y-%m-%dT%H"), "lead": lead,
            "valid_time": row["valid_time"], "objects": objects,
            "expected_at": None, "late_at": None,
            "first_seen_at": row["first_seen_at"],
            "posted_when_first_asked": None, "fetched_at": now,
        }
        _write_json(self.folder / marker_name(lead), marker)
        self.write_schedule()

    # -- the loop --------------------------------------------------------

    def run(self) -> int:
        self.start()
        started = time.monotonic()
        last = time.monotonic()
        self.progress(f"local posting {self.source} {self.cycle:%Y-%m-%dT%H}: "
                      f"{len(self.leads)} leads into {self.out}")
        for lead in self.leads:
            self.rows[lead]["state"] = "waiting"
            while True:
                located = self._located(lead)
                if located is not None:
                    break
                if not _pid_alive(self.producer_pid):
                    located = self._located(lead)  # it may have finished just before it ended
                    if located is not None:
                        break
                    message = (f"the producer (pid {self.producer_pid}) ended without "
                               f"{self.names[lead]} (lead {lead})")
                    self.fail(PRODUCER_ENDED_CODE, message, lead)
                    self.progress(f"local posting: {message}")
                    return 1
                if time.monotonic() - last > self.quiet_seconds:
                    message = (f"{self.names[lead]} (lead {lead}) is not in "
                               f"{self.quiet_seconds / 60.0:g} min after the previous lead")
                    self.fail(SOURCE_BEHIND_CODE, message, lead)
                    self.progress(f"local posting: {message}")
                    return 75
                time.sleep(self.poll_seconds)
            path, row = located
            objects = [self._place(lead, path, row)]
            self.publish(lead, objects)
            last = time.monotonic()
            self.progress(f"local posting: f{lead:03d} ready "
                          f"{time.monotonic() - started:.1f} s after start "
                          f"({objects[0]['bytes'] / 1e6:.1f} MB)")
        return 0


def _leads(text: str) -> list[int]:
    leads: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-", 1)
            leads.extend(range(int(low), int(high) + 1))
        elif part:
            leads.append(int(part))
    return leads


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m woof.local_posting",
        description="Publish a locally written source's leads as an as-posted "
                    "fetch folder, for `woof prep --as-posted OUT/posting`.")
    parser.add_argument("--source", required=True,
                        help="the source id the preparation names (e.g. hex-global)")
    parser.add_argument("--cycle", required=True, type=parse_cycle,
                        help="the producer's start, YYYY-MM-DDTHH (lead 0)")
    parser.add_argument("--leads", required=True, type=_leads,
                        help="leads, in --lead-minutes (default hours), e.g. 0-12 or 0,3,6")
    parser.add_argument("--lead-minutes", type=int, default=60,
                        help="one lead's span in minutes (default 60); a hex parent "
                             "writing every 15 min is posted with --lead-minutes 15")
    parser.add_argument("--out", required=True, type=Path,
                        help="the fetch folder (posting/, frames/, fetch-manifest.json)")
    found = parser.add_mutually_exclusive_group(required=True)
    found.add_argument("--ready-list", type=Path,
                       help="the producer's JSON-lines list of finished files")
    found.add_argument("--watch", type=Path,
                       help="the folder the producer renames finished files into")
    parser.add_argument("--name-template", default=DEFAULT_NAME_TEMPLATE,
                        help="each lead's file name; {valid:...} and {lead} "
                             # argparse %-formats help text (3.14 checks it at
                             # add_argument): the template's strftime %Y must be %%Y.
                             f"(default {DEFAULT_NAME_TEMPLATE.replace('%', '%%')!r})")
    parser.add_argument("--supplement-role", action="append", default=[],
                        metavar="ROLE", help="a supplement role the same files also "
                        "fill (hex-global: hex_in_band_surface)")
    parser.add_argument("--producer-pid", type=int, default=None,
                        help="stop with failed.json when this process ends first")
    parser.add_argument("--quiet-minutes", type=float, default=240.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--print-prep-arguments", action="store_true",
                        help="print the woof prep --input/--supplement flags and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    posting = LocalPosting(
        source=args.source, cycle=args.cycle, leads=args.leads, out=args.out,
        name_template=args.name_template, supplement_roles=args.supplement_role,
        ready_list=args.ready_list, watch=args.watch,
        producer_pid=args.producer_pid, quiet_minutes=args.quiet_minutes,
        poll_seconds=args.poll_seconds,
        progress=lambda line: print(line, flush=True),
        lead_minutes=args.lead_minutes)
    if args.print_prep_arguments:
        print(" ".join(posting.prep_arguments()))
        return 0
    return posting.run()


if __name__ == "__main__":
    sys.exit(main())
