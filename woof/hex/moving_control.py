"""Closed-loop seam between a running full-step LTS forecast and its moving-region controller.

A moving-region forecast must be steered by the model's own state while it runs: the
controller reads each history frame, tracks the storm, prepares the next mesh member on the
CPU, and only then asks the forecast to stop at a frame so the state can move.  Before this
seam every position check was a separate forecast process (construction, cold first step and
a 10 GB carrier per check, measured 2026-10-12 on the one-move proof: 583 s of process wall
for 30 forecast minutes).  With it one forecast process runs until a move is actually due.

The controller and the forecast share one directory, named by ``WOOF_HEX_MOVING_CONTROL``:

* ``progress.jsonl`` -- appended by the forecast after every history frame is closed:
  model seconds, history path and wall clock.  The controller tracks from these frames.
* ``stop-request.json`` -- written atomically by the controller.  At the first frame whose
  model time is at or after ``stop_at_seconds`` the forecast leaves its macro loop at that
  boundary (registers empty, every class at one instant) and the normal moving checkpoint
  export (``WOOF_HEX_MOVING_CHECKPOINT_OUT``) seals the state for the next member.
* ``seal-<seconds>.pkl`` -- recovery carriers, when ``WOOF_HEX_MOVING_SEAL_EVERY_SECONDS``
  is set: at frames on that cadence the same-mesh restart carrier is exported without
  stopping, so a crash or a lost card costs at most one cadence of forecast.  Only the newest
  seal this process wrote is kept; each removal is recorded in ``seals.jsonl``.

Nothing here changes numerics: a stop is a forecast that simply ends at an earlier frame,
and a seal is the same export a stop performs, taken without leaving the loop.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

CONTROL_ENV = "WOOF_HEX_MOVING_CONTROL"
SEAL_ENV = "WOOF_HEX_MOVING_SEAL_EVERY_SECONDS"
STOP_NAME = "stop-request.json"
PROGRESS_NAME = "progress.jsonl"
SEALS_NAME = "seals.jsonl"
AWAIT_NAME = "await-verdict.json"

# A resident moving process (woof.hex.mpasphys.moving_resident_v842) continues the next segment after a move.
RESIDENT_ENV = "WOOF_HEX_MOVING_RESIDENT"
#: In the running segment's control folder: the next segment's launch record (the controller writes it).
RESIDENT_NEXT = "resident-next.json"
#: In the running segment's control folder: its route can continue a move in this process (the process writes it).
RESIDENT_READY = "resident-ready.json"
#: In a segment folder: the resident process closed this segment and continues the next one.
RESIDENT_CLOSED = "resident-closed.json"
RESIDENT_SPEC_SCHEMA = "gpuwm-hex.moving-resident-next/v1"
#: The switches a segment's environment carries (moving_run.Controller.segment_env); a continuation applies the
#: next segment's values and clears the ones it does not name, as a fresh process's environment would.
RESIDENT_SEGMENT_VARIABLES = ("WOOF_HEX_MOVING_CHECKPOINT_IN", "WOOF_HEX_MOVING_CHECKPOINT_OUT",
                              "WOOF_HEX_MOVING_CHECKPOINT_WAIT", "WOOF_HEX_MOVING_CONTROL",
                              "WOOF_HEX_MOVING_OPERATOR", "WOOF_HEX_MOVING_SOURCE_GRID", "WOOF_HEX_MOVING_ZONE",
                              "WOOF_HEX_MOVING_SHOCK_ZONE", "WOOF_HEX_MESH_ROWS")


def wait_for_carrier(path, timeout_seconds: float) -> float:
    """Block until a moving carrier and its sealed receipt exist (pre-launched move segment).

    The receipt is written after the carrier is renamed into place, so its presence
    means the carrier is complete.  Returns the seconds waited.
    """
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".json")
    began = time.monotonic()
    while not (path.is_file() and receipt.is_file()):
        if time.monotonic() - began > timeout_seconds:
            raise TimeoutError(f"moving carrier {path} was not sealed within {timeout_seconds:.0f} s")
        time.sleep(1.0)
    waited = time.monotonic() - began
    print(json.dumps({"moving_prelaunch": {"carrier": str(path), "waited_seconds": waited}}), flush=True)
    if _DEFERRED_ADMISSION:
        readmit, require = _DEFERRED_ADMISSION.pop()
        _readmit_until_free(readmit, require)
    return waited


#: (readmit, require) set by the forecast door when a pre-launched segment's device
#: admission was refused only because the member it replaces still held the card.
_DEFERRED_ADMISSION: list = []


def defer_admission(readmit, require) -> None:
    _DEFERRED_ADMISSION[:] = [(readmit, require)]


def _readmit_until_free(readmit, require, timeout_seconds: float | None = None, poll_seconds: float = 2.0):
    """Take the deferred device admission again until the previous member's process has
    released the card (it exits right after sealing the carrier); refuse by the door's own
    sentence if it has not within the timeout (WOOF_HEX_MOVING_READMIT_SECONDS, default 600)."""
    import os

    if timeout_seconds is None:
        timeout_seconds = float(os.environ.get("WOOF_HEX_MOVING_READMIT_SECONDS", "600"))
    began = time.monotonic()
    verdict = readmit()
    while not verdict.admitted and time.monotonic() - began < timeout_seconds:
        time.sleep(poll_seconds)
        verdict = readmit()
    print(json.dumps({"moving_prelaunch_admission": {"admitted": bool(verdict.admitted),
                                                     "waited_seconds": time.monotonic() - began}}), flush=True)
    require(verdict)


class MovingControl:
    def __init__(self, directory: Path, seal_every_seconds: float = 0.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seal_every_seconds = float(seal_every_seconds)
        if self.seal_every_seconds < 0:
            raise ValueError("recovery seal cadence must be positive or zero (off)")
        self.stopped: dict[str, Any] | None = None
        self.seals: list[dict[str, Any]] = []
        self._own_seals: list[Path] = []
        self._lock = threading.Lock()

    @classmethod
    def from_environ(cls, environ=os.environ) -> "MovingControl | None":
        directory = environ.get(CONTROL_ENV)
        if not directory:
            return None
        return cls(Path(directory), float(environ.get(SEAL_ENV, "0") or 0))

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with self._lock, (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def stop_request(self) -> dict[str, Any] | None:
        path = self.directory / STOP_NAME
        if not path.is_file():
            return None
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            # A torn file cannot occur (the controller renames into place);
            # an unreadable one is a controller defect and must not stop a run.
            return None
        if not isinstance(request, dict) or "stop_at_seconds" not in request:
            return None
        return request

    def _await_verdict(self, model_seconds: float) -> dict[str, Any] | None:
        """Hold at this frame for the controller's verdict, only while it has asked for that.

        With member waiting enabled the controller holds every check, so a move
        decided on this frame can wait for its mesh and land on this frame.
        Without the request the forecast never waits. A missing reply is a
        controller failure, never permission to advance outside the fine region.
        """
        path = self.directory / AWAIT_NAME
        if not path.is_file():
            return None
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if float(model_seconds) < float(request.get("from_seconds", 0.0)):
            return None
        answer = self.directory / f"verdict-{int(model_seconds):08d}.json"
        deadline = time.monotonic() + float(request.get("timeout_seconds", 300.0))
        began = time.monotonic()
        while time.monotonic() < deadline:
            if answer.is_file():
                try:
                    verdict = json.loads(answer.read_text(encoding="utf-8"))
                except ValueError:
                    verdict = None
                if isinstance(verdict, dict):
                    self._append(PROGRESS_NAME, {"verdict": verdict, "model_seconds": float(model_seconds),
                                                 "waited_seconds": time.monotonic() - began})
                    return verdict
            time.sleep(0.5)
        self._append(PROGRESS_NAME, {"verdict": None, "model_seconds": float(model_seconds),
                                     "waited_seconds": time.monotonic() - began, "reason": "timeout; refusing to advance"})
        raise TimeoutError(f"moving controller did not answer frame {model_seconds:g} s before its verdict deadline")

    def frame_written(self, *, model_seconds: float, history_path: str, final: bool) -> None:
        """A history frame is closed on disk: the controller may read it now."""
        self._append(PROGRESS_NAME, {"model_seconds": float(model_seconds), "history": str(history_path),
                                     "final": bool(final), "wall_utc": time.time()})

    def at_frame(self, *, model_seconds: float, final: bool, seal: Callable[[Path], dict[str, Any]] | None,
                 wait_written: Callable[[], None] | None = None) -> bool:
        """At a frame boundary: return True when the forecast must stop here.

        ``wait_written`` blocks until this frame's history is on disk; it is called only
        when the controller has asked to give a verdict on the frame (it must read it).
        """
        if final or seal is None:
            return False
        request = self.stop_request()
        if request is None or float(model_seconds) < float(request["stop_at_seconds"]):
            if (self.directory / AWAIT_NAME).is_file() and wait_written is not None:
                wait_written()
            verdict = self._await_verdict(model_seconds)
            if verdict is not None and verdict.get("stop"):
                request = dict(verdict, stop_at_seconds=float(model_seconds))
        if request is not None and float(model_seconds) >= float(request["stop_at_seconds"]):
            self.stopped = dict(request, stopped_at_seconds=float(model_seconds), wall_utc=time.time())
            self._append(PROGRESS_NAME, {"stopped": self.stopped})
            return True
        cadence = self.seal_every_seconds
        if cadence and float(model_seconds) % cadence == 0:
            path = self.directory / f"seal-{int(model_seconds):08d}.pkl"
            receipt = seal(path)
            row = {"model_seconds": float(model_seconds), "path": str(path),
                   "bytes": receipt.get("bytes"), "sha256": receipt.get("sha256"),
                   "wall_seconds": receipt.get("wall_seconds")}
            self.seals.append(row)
            self._append(SEALS_NAME, row)
            for older in self._own_seals:
                for item in (older, older.with_suffix(older.suffix + ".json")):
                    if item.exists():
                        size = item.stat().st_size
                        item.unlink()
                        self._append(SEALS_NAME, {"removed": str(item), "bytes": size,
                                                  "reason": "a newer recovery seal of this process closed"})
            self._own_seals = [path]
        return False

    def receipt(self) -> dict[str, Any]:
        return {"directory": str(self.directory), "seal_every_seconds": self.seal_every_seconds,
                "stopped_on_request": self.stopped, "recovery_seals": self.seals}
