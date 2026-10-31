"""The start-time maps, drawn the moment the forecast writes its first history frame.

THE BREAKAGE THIS PREVENTS: the global and variable-resolution doors rendered their maps only after the whole forecast
(`forecast --no-products`, then `products`), so the first map of a 24 h run waited for all 24 hours to integrate.
On a g7e (RTX PRO 6000) box on 2026-10-23 the 60 km globe wrote its f000 frame 20 s into the forecast stage and
the f000 maps take 13 s to draw (71 pictures, all views); the run's first map still came only after the forecast
and the full products pass.

The forecast door lists each finished frame in ``history-ready.jsonl`` (its path, step and time, written after the
frame's durable sidecar), or, on an executor that keeps no such list, writes the sidecar itself (the variable-
resolution door's: 2026-10-23 its first look waited for a list that never came). FirstLook waits for the first, then draws that one frame with the products door in
a CPU-only process of its own, into the same tree and case the full products stage writes (same file names, so the
full pass replaces them with identical pictures). The cards are never touched; a forecast that ends without a frame
ends the wait. Nothing here can fail the run: a failed first look is reported, and the full pass draws every frame.

Then every later hour, the same way. THE BREAKAGE THIS PREVENTS (a hex global 60 km 24 h
run on woof hex 0.4.4, 2026-10-24): the first look rendered f000 only, so f000 reached the site 62.5 s into the
forecast, then nothing for 3 min, then f001-f024 together after the forecast and the full pass; a 10-day run would hold
every hour after f000 for the whole forecast. After f000 the thread follows the same list and draws each frame the
forecast closes the moment it is closed, one frame per draw and several draws at once (as many as the host's memory
and cores carry, ``live_slots``), so the drawing keeps level with a forecast on many cards; when the forecast ends
(``finish``) the frames it closed last are drawn the same way before the full pass begins. These draws run at low CPU
priority, carry no point soundings, and draw no windowed map (those need the whole series): the full pass draws those.
Each draw has its own scratch, so draws never share one; the files several draws share in the tree (a view's
georeference and receipts, the case's manifest and index) are written whole and merged under a lock (render_door).
The thread records what it drew in ``live-frames.json`` beside the forecast folder, which also tells a host (the WOOF
worker) that the door draws every hour itself.

The receipt also lets the full pass skip what is already drawn (``products --live-frames``). THE BREAKAGE THIS
PREVENTS (hex global 60 km 24 h, 2026-10-24): with every hour drawn live, the full pass still redrew all 25
frames' maps after the forecast (81 s), so the run's "done" came no sooner. The receipt names each frame drawn with
exit 0 (its size and mtime, so a frame the forecast rewrote is drawn again), the tree and case it went into, and the
products the draws made; the full pass then draws only the whole-run maps (windowed products), the frames no live
draw took, and the point soundings.

An ensemble runs one first look per member (``global_ensemble``). THE BREAKAGE THIS PREVENTS (a hex
global-120km ensemble, 2 members, GFS 2026-10-24 00Z, 6 h): the ensemble door ran no first look, so
no member's map existed until the forecast and every member's full products pass had ended, and all 610 maps reached
the site at the end. Members of one resident group share a forecast parent folder, so each member's look names its own
``receipt`` and ``log``; a ``gate`` shared by the members' looks bounds how many live draws run at once on the host.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

READY = "history-ready.jsonl"
DURABLE = "*history.*.nc.durable.json"
LIVE_RECEIPT = "live-frames.json"
#: The CPU priority (nice) the live draws after the first run at: the forecast is on the card.
LIVE_NICE = int(os.environ.get("WOOF_HEX_LIVE_FRAMES_NICE", "10"))
#: WOOF_HEX_LIVE_FRAMES=0 draws f000 only, as 0.4.4 did.
LIVE_ON = os.environ.get("WOOF_HEX_LIVE_FRAMES", "1") != "0"
#: Live draws in flight at once: ``auto`` (default) sizes it from the host (:func:`live_slots`); a number fixes it.
LIVE_SLOTS = os.environ.get("WOOF_HEX_LIVE_FRAMES_SLOTS", "auto")
#: Host memory one live draw (one frame, its views side by side) is budgeted, as a multiple of the frame's history
#: file. MEASURED: see the module's live-draw measurement note (RTX 5090, Arizona zoom globe, 550 km core).
LIVE_DRAW_MEMORY_FACTOR = 20.0
#: Cores one live draw keeps busy on average (measured beside the factor above).
LIVE_DRAW_CORES = 4
#: The share of the host's available memory live draws in flight may plan on (the forecast holds the rest).
LIVE_MEMORY_SHARE = 0.5


def available_memory_bytes() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo``; ``None`` where there is no such file."""

    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def live_slots(frame_bytes: int, *, available: int | None = None, cpus: int | None = None,
               setting: str | None = None) -> int:
    """Live draws in flight at once, from the host: half of its available memory at
    :data:`LIVE_DRAW_MEMORY_FACTOR` history files a draw, and :data:`LIVE_DRAW_CORES` cores a draw. One where the
    memory cannot be read (a guess that is too high is an out-of-memory kill beside the forecast)."""

    setting = LIVE_SLOTS if setting is None else setting
    if str(setting).strip().lower() != "auto":
        try:
            return max(1, int(setting))
        except (TypeError, ValueError):
            return 1
    if available is None:
        available = available_memory_bytes()
    if available is None or frame_bytes <= 0:
        return 1
    cpus = cpus or os.cpu_count() or 1
    by_memory = int(LIVE_MEMORY_SHARE * available // max(1.0, LIVE_DRAW_MEMORY_FACTOR * frame_bytes))
    by_cores = cpus // LIVE_DRAW_CORES
    return max(1, min(by_memory, by_cores))


def closed_frames(forecast: Path) -> list[Path]:
    """Every frame the forecast has closed, in the order it closed them: its history-ready list, else the frames whose
    durable sidecar names their full size."""
    out: list[Path] = []
    try:
        for line in (forecast / READY).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("path") and Path(row["path"]) not in out:
                out.append(Path(row["path"]))
    except OSError:
        pass
    if out:
        return out
    for sidecar in sorted(forecast.glob(DURABLE)):
        frame = sidecar.with_name(sidecar.name[: -len(".durable.json")])
        try:
            if frame.stat().st_size == int(json.loads(sidecar.read_text(encoding="utf-8"))["bytes"]):
                out.append(frame)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


class FirstLook(threading.Thread):
    def __init__(self, forecast: Path, *, grid: Path | str, init: Path | str, start_time: str, out: Path | str,
                 case: str, extra: Sequence[str] = (), poll_seconds: float = 0.5, log: Path | None = None,
                 receipt: Path | str | None = None, gate: threading.Semaphore | None = None):
        super().__init__(name="first-look", daemon=True)
        self.forecast, self.out, self.case = Path(forecast), Path(out), case
        self.argv = ["products", "--mesh", str(grid), "--init", str(init), "--simulation-start", start_time,
                     "--out", str(out), "--case", case, *extra]
        self.poll, self.log = poll_seconds, Path(log) if log else Path(forecast).parent / "first-look.log"
        self.receipt_path = Path(receipt) if receipt else self.forecast.parent / LIVE_RECEIPT
        self.gate = gate
        self.stopped = threading.Event()
        self.result: dict = {"status": "waiting"}
        self.live: dict = {"out": str(self.out), "case": case, "frames": [], "drawn": [], "stat": {}, "products": [],
                           "draws": []}
        self.lock = threading.Lock()
        self.running: dict[subprocess.Popen, dict] = {}
        self.slots: int | None = None

    def _first_frame(self) -> Path | None:
        ready = self.forecast / READY
        while True:
            stopping = self.stopped.is_set()  # read once more after the stop, so a frame listed just before it counts
            try:
                for line in ready.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row.get("path"):
                        return Path(row["path"])
            except (OSError, ValueError):
                pass
            # Executors that write no history-ready.jsonl (the variable-resolution door's) still write each frame's
            # durable sidecar after the frame itself: the earliest frame whose sidecar names its full size is done.
            for sidecar in sorted(self.forecast.glob(DURABLE)):
                frame = sidecar.with_name(sidecar.name[: -len(".durable.json")])
                try:
                    if frame.stat().st_size == int(json.loads(sidecar.read_text(encoding="utf-8"))["bytes"]):
                        return frame
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            if stopping:
                return None
            self.stopped.wait(self.poll)

    def run(self) -> None:
        began = time.monotonic()
        receipt = self.receipt_path
        if LIVE_ON:
            self._save(receipt)  # before the first look, so a host sees it first
        else:
            # A receipt left by an earlier run in this folder would let the full pass skip frames this run never rendered.
            receipt.unlink(missing_ok=True)
        frame = self._first_frame()
        if frame is None:
            self.result = {"status": "no-frame"}
            return
        waited = time.monotonic() - began
        if not LIVE_ON:
            argv = [sys.executable, "-m", "woof.hex.cli", self.argv[0], "--history", str(frame), *self.argv[1:]]
            drawn = time.monotonic()
            with self.gate or nullcontext(), self.log.open("w", encoding="utf-8") as stream:
                rc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
            self.result = {"status": "drawn" if rc == 0 else "failed", "returncode": rc, "frame": str(frame),
                           "frame_wait_seconds": round(waited, 1), "draw_seconds": round(time.monotonic() - drawn, 1),
                           "log": str(self.log)}
            return
        self.result = {"status": "drawing", "frame": str(frame), "frame_wait_seconds": round(waited, 1),
                       "log": str(self.log)}
        try:
            self._follow(first=frame)
        except Exception as error:  # nothing here fails the run: the full pass draws every frame
            self.live["stopped"] = f"{type(error).__name__}: {error}"
            self._save(receipt)

    def _launch(self, frames: list[Path], first: bool) -> bool:
        """Start one live draw of ``frames``; False when the shared gate (an ensemble's) has no room now."""
        if self.gate is not None and not self.gate.acquire(blocking=False):
            return False
        n = len(self.live["draws"]) + len(self.running)
        scratch = self.out.parent / f"{self.out.name}.live-scratch-{n:03d}"
        if first:
            # The first look: f000 with its point pictures, at normal priority, logged where hosts look for it.
            argv = [sys.executable, "-m", "woof.hex.cli", self.argv[0], "--history", *map(str, frames), *self.argv[1:],
                    "--scratch", str(scratch), "--keep-frame-cache"]
            log = self.log
        else:
            argv = [sys.executable, "-m", "woof.hex.cli", self.argv[0], *self.argv[1:], "--no-points",
                    "--keep-frame-cache", "--scratch", str(scratch), "--history", *map(str, frames)]
            nice = shutil.which("nice") if LIVE_NICE > 0 else None
            if nice:  # the nice(1) wrapper, not preexec_fn: the door process runs threads
                argv = [nice, "-n", str(LIVE_NICE), *argv]
            log = self.log.with_name(f"live-frames-{n:03d}.log")
        stream = log.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
        except OSError:
            stream.close()
            if self.gate is not None:
                self.gate.release()
            raise
        self.running[proc] = {"frames": frames, "began": time.monotonic(), "scratch": scratch, "stream": stream,
                              "first": first}
        return True

    def _reap(self) -> None:
        for proc, info in list(self.running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del self.running[proc]
            info["stream"].close()
            if self.gate is not None:
                self.gate.release()
            shutil.rmtree(info["scratch"], ignore_errors=True)
            seconds = round(time.monotonic() - info["began"], 1)
            if info["first"]:
                self.result.update(status="drawn" if rc == 0 else "failed", returncode=rc, draw_seconds=seconds)
            self._record(info["frames"], rc, seconds, self.receipt_path)

    def _follow(self, first: Path) -> None:
        """Draw every frame the forecast closes, each the moment it is closed, several draws at once, until finish()
        (the forecast has ended) and every closed frame is drawn.

        THE BREAKAGE THIS PREVENTS (an Arizona 96 -> 3 km zoom globe, 36 h, 8 x RTX PRO 6000, woof hex 0.4.10,
        2026-10-30): the forecast closed a frame every 22 s, but one draw ran at a time and took up to eight waiting
        frames (one import after another inside each view), so f000 took 361 s, f001-f008 15 min, and the forecast
        ended with 9 of 37 frames drawn; its products pass then waited 8.5 min for that draw before it began. One
        frame per draw, as many draws at once as the host's memory and cores carry (live_slots), keeps the maps
        level with the forecast."""

        drawn: set[str] = set()
        pending_first = [first]
        while True:
            self._reap()
            stopping = self.stopped.is_set()  # read before the list, so a frame listed just before the stop counts
            pending = pending_first + [p for p in closed_frames(self.forecast)
                                       if str(p) not in drawn and p != first]
            if self.slots is None:
                try:
                    size = first.stat().st_size
                except OSError:
                    size = 0
                self.slots = live_slots(size)
                self.live["slots"] = self.slots
            while pending and len(self.running) < self.slots:
                frame = pending[0]
                if not self._launch([frame], first=bool(pending_first) and frame == first):
                    break
                pending.pop(0)
                if pending_first:
                    pending_first = []
                drawn.add(str(frame))
            if stopping and not pending and not self.running:
                return
            time.sleep(self.poll)

    def _record(self, frames: list[Path], rc: int, seconds: float, receipt: Path) -> None:
        """One draw in the receipt: the frames it took, and on exit 0 each frame's identity and the products drawn."""
        with self.lock:
            self.live["frames"] += [p.name for p in frames]
            self.live["draws"].append({"frames": len(frames), "first": frames[0].name, "returncode": rc,
                                       "seconds": seconds})
            if rc == 0:
                for path in frames:
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    self.live["drawn"].append(path.name)
                    self.live["stat"][path.name] = [st.st_size, st.st_mtime_ns]
                try:
                    manifest = json.loads((self.out / self.case / "products-manifest.json").read_text(encoding="utf-8"))
                    products = set(self.live["products"])
                    for view in manifest.get("views") or ():
                        listed = view.get("products_drawn") or ()
                        products.update(p.strip() for p in (listed.split(",") if isinstance(listed, str) else listed)
                                        if p and p.strip())
                    self.live["products"] = sorted(products)
                except (OSError, ValueError, AttributeError):
                    pass
            self._save(receipt)

    def receipt(self) -> Path | None:
        """``live-frames.json`` when a live draw (f000 included) rendered a frame with exit 0, else None: what the
        full pass takes as ``products --live-frames``."""
        return self.receipt_path if self.live["drawn"] else None

    def _save(self, receipt: Path) -> None:
        try:
            temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"schema": "gpuwm-hex.live-frames/v1", **self.live}, indent=1) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, receipt)
        except OSError:
            pass

    def finish(self, timeout: float | None = None) -> dict:
        """The forecast has ended: draw the frames it closed that no draw has taken yet, let every draw land (up to
        ``timeout``), then stop."""
        self.stopped.set()
        self.join(timeout)
        if self.is_alive():
            for proc in list(self.running):
                if proc.poll() is None:
                    proc.terminate()
            self.join(30)
        receipt = self.receipt()
        return {**self.result, "live": {"frames": len(self.live["frames"]), "draws": len(self.live["draws"]),
                                        "drawn": len(self.live["drawn"]), "slots": self.slots,
                                        "receipt": str(receipt) if receipt else None}}
