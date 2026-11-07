"""Card memory reservations for nests added to a running hex run.

THE BREAKAGE THIS PREVENTS: two nests added close together (two trigger
firings one look apart, or a hand add beside a firing) each read the card's
free memory before either had allocated, both saw room, both were placed
beside the parent, and both ran out of memory together (shapes 2-3 checker,
2026-10-23).  A check and its claim must be one atomic act, so the fit is a
RESERVATION: under one file lock the ledger reads the card's measured free
memory, subtracts the part of every live reservation its owner has not yet
allocated, and records the new claim before the lock is released.  A second
claim in the same instant sees the first one.

The arithmetic is the box card queue's shared admission:
measured free (which already counts every process on the card, ours or not)
minus, for each live reservation, its budget less what its processes already
hold.  Reservations whose processes are all gone are dropped on the next
claim, so a killed nest never holds memory on paper.

A reservation's ``pids`` are the NEST's own processes only (add-nest, export,
prep, sim): ``unallocated`` subtracts what those hold.  A trigger firing is
reserved inside the parent forecast before any nest process exists, so the
parent's PID goes in a separate ``owner`` field that keeps the row alive until
add-nest adopts it and is never read for memory.  Counting the parent's PID
made every trigger claim count 0 unallocated bytes (the parent holds ~18 GB, a
nest needs ~5), so two firings one look apart both passed on a bare VM where
nvidia-smi PIDs match (2.9.7 release review 30, finding 1).

On a box whose card queue is present (a pool runner script) the nest's
GPU stage also takes a pool lease of the same size (``add-nest
--card-queue``), which makes the claim visible to every other lane on the
card, not only to other nests.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator

SCHEMA = "gpuwm-hex.card-ledger.v1"
MARGIN = 0.10

FreeBytes = Callable[[], "int | None"]
UsedBytes = Callable[[Iterable[int]], int]


def default_path(device: int = 0) -> Path:
    root = os.environ.get("WOOF_HEX_CARD_LEDGER_DIR") or tempfile.gettempdir()
    return Path(root) / f"gpuwm-hex-card-{int(device)}.json"


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    with open(lock, "a+b") as stream:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except ImportError:  # Windows: a byte-range lock on the lock file
            import msvcrt

            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) is CTRL_C_EVENT on Windows, never a probe
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def nvidia_free_bytes(device: int = 0) -> int | None:
    try:
        out = subprocess.run([*__import__('woof.gpu_smi', fromlist=['command']).command(), f"--id={device}", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return int(float(out.stdout.strip().splitlines()[0])) * 2**20
    except Exception:  # noqa: BLE001 - no card visible: the caller records an unmeasured fit
        return None


def nvidia_used_bytes(pids: Iterable[int]) -> int:
    """Card memory the given processes hold now (nvidia-smi compute-apps), 0 when unreadable."""

    wanted = {int(p) for p in pids}
    if not wanted:
        return 0
    try:
        out = subprocess.run([*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001 - unreadable: count nothing allocated (the conservative side)
        return 0
    total = 0
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) in wanted:
            try:
                total += int(float(parts[1])) * 2**20
            except ValueError:
                pass
    return total


class CardLedger:
    """Atomic memory reservations on one card (see the module text)."""

    def __init__(self, path: Path | None = None, *, device: int = 0,
                 free_bytes: FreeBytes | None = None, used_bytes: UsedBytes | None = None,
                 alive: Callable[[int], bool] = pid_alive, margin: float = MARGIN) -> None:
        self.path = Path(path) if path is not None else default_path(device)
        self.free_bytes = free_bytes or (lambda: nvidia_free_bytes(device))
        self.used_bytes = used_bytes or nvidia_used_bytes
        self.alive = alive
        self.margin = float(margin)

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return list(value.get("reservations") or [])

    def _write(self, rows: list[dict[str, Any]]) -> None:
        partial = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        partial.write_text(json.dumps({"schema": SCHEMA, "reservations": rows}, indent=1, sort_keys=True) + "\n",
                           encoding="utf-8")
        os.replace(partial, self.path)

    def _row_alive(self, row: dict[str, Any]) -> bool:
        pids = row.get("pids") or []
        if pids:
            return any(self.alive(p) for p in pids)
        owner = row.get("owner")
        return owner is not None and self.alive(owner)

    def _live(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if self._row_alive(r)]

    def unallocated(self, rows: list[dict[str, Any]]) -> int:
        return sum(max(0, int(r["bytes"]) - int(self.used_bytes(r.get("pids") or []))) for r in rows)

    def live_names(self) -> list[str]:
        with _locked(self.path):
            return [r["name"] for r in self._live(self._read())]

    def reserve(self, name: str, nbytes: int, pids: Iterable[int], *, force: bool = False,
                owner: int | None = None) -> dict[str, Any]:
        """Claim ``nbytes`` (+ margin) if it fits now; the verdict is returned either way.

        ``fits`` is None when the card's free memory cannot be read (no card on
        this host); the claim is then recorded so later claims still see it.
        ``force`` records the claim even when it does not fit (a queued nest
        whose parent has ended goes anyway: nothing is refused for size, and
        the engine's own admission still runs at sim start).
        ``owner`` keeps a row with no nest processes yet alive (the parent
        forecast for a trigger firing); its memory is never counted.
        """

        need = int(int(nbytes) * (1.0 + self.margin))
        with _locked(self.path):
            rows = self._live(self._read())
            rows = [r for r in rows if r["name"] != name]
            free = self.free_bytes()
            others = self.unallocated(rows)
            available = None if free is None else int(free) - others
            fits = None if available is None else need <= available
            verdict = {"name": name, "bytes": int(nbytes), "need_bytes": need, "free_bytes": free,
                       "others_unallocated_bytes": others, "available_bytes": available, "fits": fits}
            if fits is not False or force:
                row = {"name": name, "bytes": need, "pids": [int(p) for p in pids],
                       "created_unix": round(time.time(), 3)}
                if owner is not None:
                    row["owner"] = int(owner)
                rows.append(row)
            self._write(rows)
        return verdict

    def adopt(self, name: str, pids: Iterable[int]) -> bool:
        """Add processes to a reservation (the nest's own children take over its claim)."""

        with _locked(self.path):
            rows = self._read()
            found = False
            for row in rows:
                if row["name"] == name:
                    row["pids"] = sorted({*row.get("pids", []), *(int(p) for p in pids)})
                    found = True
            if found:
                self._write(rows)
        return found

    def release(self, name: str) -> None:
        with _locked(self.path):
            rows = [r for r in self._read() if r["name"] != name]
            self._write(rows)

    def holds(self, name: str) -> bool:
        with _locked(self.path):
            return any(r["name"] == name for r in self._live(self._read()))
