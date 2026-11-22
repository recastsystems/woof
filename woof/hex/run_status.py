"""The machine surface a workspace reads while a long hex command runs.

Three files under the command's run directory:

``status.json``
    the current state, rewritten ATOMICALLY on every transition (temp file and
    ``os.replace``), so a reader that opens it at any instant sees a whole
    document, never a truncated one.

``events.jsonl``
    append-only, one JSON object per line, one line per event: a stage
    starting or ending, a skip and why, a step, a refusal, a failure.  It is
    the record a workspace replays to draw a timeline, and the record a
    reader greps after the fact.  Every line carries the schema id, so a
    reader can refuse a line it does not understand.

``<command>.log``
    the same events as human text, one line each.  Its path is printed first
    so a reader not driving a workspace can still find it.

This is the hex counterpart of woof global's ``status.py`` (schema
``gpuwm-global-status-v1``).  The document shape is the same -- command,
stage, stage_index, stage_count, step, step_count, started_utc,
updated_utc, eta_s, state, reason, log -- so the WOOF-ArWen Desktop global tab
and WOOF's global module read either one by switching the schema id they pin;
the hex schema adds ``events`` (the events.jsonl path) and ``row``.

THE STATES.  ``running``, ``done``, ``failed`` and ``refused``.  ``refused``
is apart from ``failed`` on purpose: a refusal is this package declining to
do something and naming why, and a workspace shows it as an answer, not as a
crash.

Nothing here knows what a workspace is, and nothing here draws one.  A
directory that cannot be created leaves the command running without a
machine surface rather than dying for want of one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Mapping

__all__ = ["STATUS_NAME", "EVENTS_NAME", "SCHEMA", "EVENTS_SCHEMA", "STATES",
           "StatusWriter", "read_status", "read_events", "utc_now"]

#: The file a workspace polls.
STATUS_NAME = "status.json"

#: The append-only event stream beside it.
EVENTS_NAME = "events.jsonl"

#: The status document's schema id.  A change of shape changes this string.
SCHEMA = "gpuwm-hex-status-v1"

#: The schema id every events.jsonl line carries.
EVENTS_SCHEMA = "gpuwm-hex-events-v1"

STATES = ("running", "done", "failed", "refused")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class StatusWriter:
    """Writes ``status.json``, ``events.jsonl`` and the command log."""

    outdir: Path
    command: str
    stages: tuple[str, ...] = ()
    row: str | None = None
    _path: Path | None = field(default=None, init=False)
    _events: Path | None = field(default=None, init=False)
    _log: Path | None = field(default=None, init=False)
    _stage_index: int = field(default=0, init=False)
    _stage: str | None = field(default=None, init=False)
    _step: int = field(default=0, init=False)
    _step_count: int = field(default=0, init=False)
    _eta_s: float | None = field(default=None, init=False)
    _started: str = field(default_factory=utc_now, init=False)
    _state: str = field(default="running", init=False)

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)
        try:
            self.outdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        self._path = self.outdir / STATUS_NAME
        self._events = self.outdir / EVENTS_NAME
        self._log = self.outdir / f"{self.command.replace(' ', '-')}.log"
        self.event("start", stages=list(self.stages), row=self.row)
        self._write("running")

    # -- what a command drives -------------------------------------------

    @property
    def log_path(self) -> Path | None:
        return self._log

    @property
    def events_path(self) -> Path | None:
        return self._events

    @property
    def state(self) -> str:
        return self._state

    def stage(self, name: str, *, step_count: int = 0, eta_s: float | None = None) -> None:
        """Enter a named stage.  ``step_count`` sizes its progress bar."""

        if self._stage is not None and self._state == "running":
            self.event("stage-end", stage=self._stage)
        if name in self.stages:
            self._stage_index = self.stages.index(name)
        else:
            self.stages = (*self.stages, name)
            self._stage_index = len(self.stages) - 1
        self._stage = name
        self._step = 0
        self._step_count = int(step_count)
        self._eta_s = eta_s
        self.event("stage-start", stage=name, step_count=self._step_count, eta_s=eta_s)
        self._write("running")

    def step(self, index: int, *, note: str | None = None) -> None:
        self._step = int(index)
        self.event("step", stage=self._stage, step=self._step, step_count=self._step_count,
                   note=note)
        self._write("running")

    def skip(self, name: str, reason: str) -> None:
        """A stage that does nothing, said out loud: a quiet skip is how a
        reader comes to believe a static was built when the cache was used."""

        self.event("stage-skip", stage=name, reason=reason)

    def note(self, line: str) -> None:
        self.event("note", stage=self._stage, text=line)

    def done(self, note: str | None = None) -> None:
        if self._stage is not None:
            self.event("stage-end", stage=self._stage)
        self.event("done", text=note)
        self._write("done")

    def failed(self, reason: str) -> None:
        self.event("failed", stage=self._stage, reason=reason)
        self._write("failed", reason=reason)

    def refused(self, reason: str) -> None:
        self.event("refused", stage=self._stage, reason=reason)
        self._write("refused", reason=reason)

    # -- the two writers -------------------------------------------------

    def event(self, kind: str, **payload: Any) -> None:
        """One line in events.jsonl and one in the human log."""

        record = {"schema": EVENTS_SCHEMA, "utc": utc_now(), "command": self.command,
                  "event": kind}
        record.update({key: value for key, value in payload.items() if value is not None})
        if self._events is not None:
            try:
                with self._events.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            except OSError:
                pass
        if self._log is not None:
            text = " ".join(f"{key}={value}" for key, value in record.items()
                            if key not in {"schema", "utc", "command", "event"})
            try:
                with self._log.open("a", encoding="utf-8") as stream:
                    stream.write(f"{record['utc']} {kind} {text}".rstrip() + "\n")
            except OSError:
                pass

    def _write(self, state: str, *, reason: str | None = None) -> None:
        self._state = state
        if self._path is None:
            return
        stage = (self.stages[self._stage_index]
                 if self.stages and self._stage_index < len(self.stages)
                 else self.command)
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "command": self.command,
            "row": self.row,
            "stage": stage,
            "stage_index": self._stage_index,
            "stage_count": len(self.stages),
            "stages": list(self.stages),
            "step": self._step,
            "step_count": self._step_count,
            "started_utc": self._started,
            "updated_utc": utc_now(),
            "eta_s": self._eta_s,
            "state": state,
        }
        if reason is not None:
            payload["reason"] = reason
        if self._log is not None:
            payload["log"] = str(self._log)
        if self._events is not None:
            payload["events"] = str(self._events)
        temp = self._path.with_suffix(".json.tmp")
        try:
            temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
            os.replace(temp, self._path)
        except OSError:
            pass


def read_status(outdir: Path) -> Mapping[str, Any]:
    """The current document, refused by name when its schema is not ours."""

    value = json.loads((Path(outdir) / STATUS_NAME).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError(f"{STATUS_NAME} schema {value.get('schema')!r} is not {SCHEMA}")
    return value


def read_events(outdir: Path) -> list[Mapping[str, Any]]:
    """Every event line, in order; a line of another schema is refused."""

    rows = []
    path = Path(outdir) / EVENTS_NAME
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != EVENTS_SCHEMA:
            raise ValueError(f"{EVENTS_NAME} line {number} schema {record.get('schema')!r} "
                             f"is not {EVENTS_SCHEMA}")
        rows.append(record)
    return rows
