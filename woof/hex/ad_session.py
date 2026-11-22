# SPDX-License-Identifier: Apache-2.0
"""A sensitivity program kept loaded ACROSS commands (the session), and the Python object that holds one.

THE COST THIS REMOVES. One ``sensitivity`` command already ran several objectives on ONE loaded runner
(``ad_doors.LoadedProgram``: the runner's --resident route), but the program ended with the command, so a user who
asks for gradients one command at a time (any optimisation loop does) paid the program's start, its first-step
warm-up and its exit every time. Measured on the 3 km storm example, 59 steps, one 96 GB card (card run of
2026-11-14): the second objective on the loaded program took 283.7 s; the same kind of gradient
as a NEW command took 338.5 s in its runner stage (plan read and typed 11.1 s, start bytes, kernel set and images
8.7 s, a forward of 31.8 s against 18.1 s, a backward of 277.6 s against 263.6 s, 8.1 s for the process to end) and
399.7 s as the shell saw it.

WHAT A SESSION IS. The first ``sensitivity`` command on a row with a gradient line starts the runner detached, in a
folder of its own under the user's door cache (``sessions/<name>/``, owner-only; override
``WOOF_HEX_AD_SESSIONS``), and leaves it loaded when the command ends. The next ``sensitivity`` command on the same
card finds it and sends its objective as a request (the runner's own file mailbox: NAME.req, NAME.done; no network
port, no socket). The session:

* is named (``card0`` for card 0; ``--session-name``), and listed and stopped by name
  (``woof hex sensitivity --session-list``, ``--session-stop [NAME]``);
* ends by itself after ``--session-idle-s`` seconds with no request (default 600) and frees the card and the host
  memory it holds;
* answers only for the exact plan, build and settings it was started with: its key holds the answer check's whole
  operator key (the plan's sha256 and every file of it by size, inode and change times, the runner's sha256, the
  kernel set, the images, the row's words, the door program), the runner's command line and environment word for
  word (but the request's own: out, objective, dumped allocations, store folder), the cards and the user. A command
  whose key differs does not use it: the door stops it and starts fresh, and says which part differs;
* cannot give a stale answer: a request has a name no earlier request had, its answer is read only under that name
  from a program whose process (pid and start time) is the one the session names, and one command at a time talks to
  it (a file lock held for the command; a second command waits for the first);
* is dropped by itself: a session that ended, crashed or timed out between or during commands is noticed, said, and
  the command starts a fresh program.

A command's FIRST output line says which happened: ATTACHED or STARTED FRESH.

WHAT IT DOES NOT COVER. The first command on a plan runs the answer check in its own windows, which need the card:
the door ends the session's program before them, so that command leaves none and the next one starts it. A batch
job (SLURM_JOB_ID, PBS_JOBID or LSB_JOBID set) leaves none unless ``--session on`` is given. THE BREAKAGE THAT
PREVENTS: the session is detached from the job, so the scheduler would hand the card to the next job while the
session still held it.

``SensitivitySession`` is the same thing as one Python object for an optimisation loop: it builds the door's context
once and returns a gradient for each new target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:
    # POSIX only. THE BREAKAGE THIS PREVENTS (0.6.1 on Windows): the import at the top of this module stopped
    # `woof hex --help`, which builds every door, with "No module named 'fcntl'" once `resource` was fixed.
    # The session's lock is the one use; it says what it needs (_posix_lock).
    fcntl = None


def _posix_lock():
    """``fcntl``, or a refusal that names it: a card session locks its folder with ``flock``, which Windows lacks."""
    if fcntl is None:
        raise RuntimeError("a derivative command's card session locks its folder with fcntl.flock, which this platform "
                           "lacks: the derivative commands run on Linux")
    return fcntl

SESSIONS_ENV = "WOOF_HEX_AD_SESSIONS"
MODE_ENV = "WOOF_HEX_AD_SESSION"
SCHEMA = "gpuwm-hex.ad-session/v1"
DEFAULT_IDLE_S = 600.0
STOP_WAIT_S = 120.0
#: A batch scheduler's job: the scheduler owns the card, and a session outlives the job (see the module words).
SCHEDULER_ENV = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID")
#: Runner words that are a request's own, or the session's own store placement: not part of what the program is.
REQUEST_FLAGS = ("--out", "--objective", "--dump-grad", "--bstore", "--x0-dir", "--bstore-ram-gib")
#: Words of a command that choose the program (the quick comparison made before the plan is read; the full key
#: decides).
COMMAND_WORDS = ("mesh", "suite", "oro", "dt", "plan", "init", "case", "runner", "assets", "gradient_mesh",
                 "gradient_plan", "checked_forward", "boundary_layer", "first_gradient", "rank_plans",
                 # THE BREAKAGE THIS PREVENTS: a command with another --cloud-derivative than the loaded program's
                 # read ATTACHED in its first line and NOT ATTACHED when the full key was compared.
                 "cloud_derivative", "keep_forward")
_OPEN: list = []
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
_ENV_PREFIXES = ("AD_", "HEXAD_", "CUDA_")


class SessionFolderError(Exception):
    """The sessions folder is not this user's own, private folder: no session is used."""


class SessionLost(Exception):
    """The session's program ended before it answered (it crashed, timed out or was stopped): start fresh."""


# ------------------------------------------------------------------------------------------------------ the folder
def sessions_dir() -> Path:
    declared = os.environ.get(SESSIONS_ENV)
    if declared:
        return Path(declared).expanduser().absolute()
    from . import assets as _assets
    return _assets.cache_root() / "ad" / "sessions"


def private_dir(path: Path) -> Path:
    """``path`` as a folder only this user can enter (mode 0700, owned by this user, not a link), made if absent.
    THE BREAKAGE THIS PREVENTS: on a shared machine another user who can write the mailbox could send the program
    requests or plant an answer."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise SessionFolderError(f"{path} is not a folder of its own")
    if st.st_uid != os.geteuid():
        raise SessionFolderError(f"{path} belongs to another user")
    if stat.S_IMODE(st.st_mode) & 0o077:
        os.chmod(path, 0o700)
        if stat.S_IMODE(os.lstat(path).st_mode) & 0o077:
            raise SessionFolderError(f"{path} is open to other users and its mode cannot be set to 0700")
    return path


def session_name(args: argparse.Namespace | None, cards: list[str]) -> str:
    given = getattr(args, "session_name", None) if args is not None else None
    if given:
        if not _NAME.match(str(given)):
            raise SessionFolderError(f"--session-name {given!r}: letters, digits, dot, dash and underscore, at most 40")
        return str(given)
    return "card" + "-".join("".join(c if c.isalnum() else "_" for c in str(card)) for card in cards)


# --------------------------------------------------------------------------------------------------- the process
def _start_ticks(pid: int) -> int | None:
    """The process's start time in clock ticks since boot (None: no such process, or it has ended and is unreaped)."""
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii", errors="replace")
        fields = text[text.rindex(")") + 2:].split()
        return None if fields[0] in ("Z", "X") else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _threads_left(pid: int) -> bool:
    """Whether any thread of process ``pid`` still runs. THE BREAKAGE THIS PREVENTS (card jobs of 2026-11-14): the
    runner has helper threads, and a process whose first thread has ended reads as ended (state Z) in /proc while
    its other threads are still giving back the card and the page-locked host memory. A command that had stopped
    such a program (one it did not start: it attached to it, or found it loaded with other words) took it for gone
    and was refused for card memory 3 s early, and in a second run for host memory."""
    try:
        tasks = os.listdir(f"/proc/{int(pid)}/task")
    except OSError:
        return False
    for task in tasks:
        try:
            text = Path(f"/proc/{int(pid)}/task/{task}/stat").read_text(encoding="ascii", errors="replace")
            if text[text.rindex(")") + 2:].split()[0] not in ("Z", "X"):
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def _alive(pid: int | None, ticks: int | None) -> bool:
    """The process runs: its first thread does (and is the process the session names), or, once that has ended, any
    other thread of it still does (see _threads_left)."""
    if not pid:
        return False
    now = _start_ticks(pid)
    if now is not None:
        return ticks is None or now == ticks
    return _threads_left(pid)


def _serves(pid: int, folder: Path, settle_s: float = 0.5) -> bool:
    """Whether process ``pid`` is a runner serving ``folder`` (its command line names --resident FOLDER). A process
    whose first thread has ended reads an empty command line in /proc while its other threads run on; theirs is read
    then (the threads of one process share it).

    THE BREAKAGE THIS PREVENTS (test_a_program_that_is_whole_is_not_taken_for_an_ending_one failed 1 run in 40 alone):
    subprocess starts a program with posix_spawn (vfork), which gives control back to the parent once the child has
    replaced its memory and before the kernel has set the new argument area, so /proc/<pid>/cmdline of a program
    started a moment ago reads EMPTY (329 of 400 reads straight after Popen, Python 3.14), and a program
    just started read as serving nothing. An empty command line of a process whose first thread runs is read again
    for up to ``settle_s``; a command line that names anything is an answer at once."""
    for _ in range(max(1, int(settle_s / 0.002))):
        said = _serves_once(pid, folder)
        if said is not None:
            return said
        if _start_ticks(pid) is None:
            break
        time.sleep(0.002)
    return False


def _serves_once(pid: int, folder: Path) -> bool | None:
    """_serves read once: True or False, or None when every command line read was empty (not set yet, or ended)."""
    target = os.fsencode(str(folder))
    names = [f"/proc/{int(pid)}/cmdline"]
    try:
        names += [f"/proc/{int(pid)}/task/{task}/cmdline" for task in sorted(os.listdir(f"/proc/{int(pid)}/task"))]
    except OSError:
        pass
    seen = False
    for name in names:
        try:
            raw = Path(name).read_bytes()
        except OSError:
            continue
        seen = seen or bool(raw)
        words = raw.split(b"\0")
        if any(w == b"--resident" and i + 1 < len(words) and words[i + 1] == target for i, w in enumerate(words)):
            return True
    return False if seen else None


def _host_mib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text(encoding="ascii", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


class _Pid:
    """A program this command did not start: alive while /proc names the same process (pid and start time)."""

    def __init__(self, pid: int, ticks: int | None, folder: Path):
        self.pid, self.ticks, self.folder, self.returncode = int(pid), ticks, folder, None

    def poll(self) -> int | None:
        if self.returncode is None and not _alive(self.pid, self.ticks):
            self.returncode = 0 if _read_json(self.folder / "state.json").get("state") == "ended" else -1
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        t0 = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - t0 >= timeout:
                raise subprocess.TimeoutExpired(f"pid {self.pid}", timeout)
            time.sleep(0.05)
        return self.returncode

    def kill(self) -> None:
        if _alive(self.pid, self.ticks):
            try:
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass


def _read_json(path: Path) -> dict:
    for _ in range(3):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            return doc if isinstance(doc, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            time.sleep(0.02)   # written as it is read
    return {}


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_live(root: Path, name: str) -> dict | None:
    """The session ``name`` if its program is running: {"doc", "dir", "pid", "ticks", "state"}; else None. A program
    started in the folder whose session file names no process yet (a command ended between the two) is found by the
    runner's own state file and its command line."""
    folder = root / name
    doc = _read_json(folder / "session.json")
    if doc.get("schema") != SCHEMA or doc.get("uid") != os.geteuid():
        return None
    state = _read_json(folder / "state.json")
    pid, ticks = doc.get("pid"), doc.get("start_ticks")
    if not pid and state.get("pid") and _serves(state["pid"], folder):
        pid, ticks = state["pid"], _start_ticks(state["pid"])
    # a program whose first thread has ended serves no request: it is not loaded, it is ending (read_ending)
    if not pid or _start_ticks(pid) is None or not _alive(pid, ticks) or not _serves(pid, folder):
        return None
    return {"doc": doc, "dir": folder, "pid": int(pid), "ticks": ticks, "state": state.get("state"), "name": name,
            "served": state.get("served")}


def read_ending(root: Path, name: str) -> dict | None:
    """The program of session folder ``name`` when it is ENDING and not yet gone: its first thread has ended (it
    serves no request any more, and ``read_live`` does not find it) while other threads of it still run, giving back
    the card and the page-locked host memory; {"doc", "dir", "pid", "name"}, else None.

    THE BREAKAGE THIS PREVENTS: a command that started while a program stopped by ANOTHER command (or by its own
    idle time-out) was still giving its memory back found no session, read the card at once and was refused for
    memory that was free a few seconds later. The command that stops a program waits for its last thread; a command
    that only arrives while it ends did not know to (seen on the card jobs of 2026-11-14; a rerun passed)."""
    folder = root / name
    doc = _read_json(folder / "session.json")
    if doc.get("schema") != SCHEMA or doc.get("uid") != os.geteuid():
        return None
    pid = doc.get("pid") or _read_json(folder / "state.json").get("pid")
    if not pid or _start_ticks(pid) is not None or not _threads_left(pid) or not _serves(pid, folder):
        return None
    return {"doc": doc, "dir": folder, "pid": int(pid), "name": name}


def _tidy_lock(fd: int, root: Path, name: str) -> None:
    """The session's lock file removed, by the holder of its lock (``fd``), when no program is loaded or ending
    under the name: a lock file with no program behind it is litter in the user's folder. A waiter on the removed
    file takes the next one (``Use.lock``)."""
    try:
        path = root / f"{name}.lock"
        if (read_live(root, name) is None and read_ending(root, name) is None
                and os.path.samestat(os.fstat(fd), os.stat(path))):
            os.unlink(path)
    except OSError:
        pass


def stop_live(live: dict, wait_s: float = STOP_WAIT_S) -> dict:
    """Ask the session's program to end (the runner's stop file; it ends at once when idle, after its gradient when
    busy) and wait; end it by its own pid if it has not ended ``wait_s`` later."""
    handle = _Pid(live["pid"], live["ticks"], live["dir"])
    killed = False
    if handle.poll() is None:
        try:
            (live["dir"] / "stop").write_text("")
        except OSError:
            pass
        try:
            handle.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            handle.kill()
            killed = True
            try:
                handle.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                pass
    return {"name": live["name"], "pid": live["pid"], "ended_by_pid": killed, "ended": handle.poll() is not None}


# ----------------------------------------------------------------------------------------------------- the key
def command_words(ctx: Any) -> dict:
    """The words of a command that choose its program, as far as they are known before the plan is read."""
    from . import __version__
    args = ctx.args
    words = {k: (None if getattr(args, k, None) is None else str(getattr(args, k))) for k in COMMAND_WORDS}
    row = getattr(ctx, "row", None)
    if isinstance(row, dict):   # a route word typed as the row's default is the same program as no word
        from . import ad_doors as _d
        for key, pick in (("cloud_derivative", _d.cloud_derivative_route), ("boundary_layer", _d.boundary_layer_route)):
            name, _, _ = pick(row, words.get(key))
            if name:
                words[key] = name
    for k in ("plan", "init", "case", "runner", "assets", "gradient_plan", "rank_plans", "oro"):
        if words.get(k) is not None:
            words[k] = os.path.abspath(words[k])
    return {"version": __version__, "row": ctx.row_name, "steps": int(ctx.steps), "cards": list(ctx.cards),
            "words": words,
            "environ": sorted([k, v] for k, v in os.environ.items()
                              if k.startswith(_ENV_PREFIXES) and k != "CUDA_VISIBLE_DEVICES")}


def words_differ(then: dict, now: dict) -> list[str]:
    """The command words that differ between the loaded program's and this command's (command_words), by name:
    each word of ``words`` and each environment variable, not the group they sit in.
    THE BREAKAGE THIS PREVENTS: a command with another cloud word read "was started with other words (words)" in
    its first line and named no word (the build before's card job, 2026-11-14, legs cloud and cw)."""
    out: list[str] = []
    for key in sorted(set(then) | set(now)):
        a, b = then.get(key), now.get(key)
        if a == b:
            continue
        if key == "words" and isinstance(a, dict) and isinstance(b, dict):
            out += [w for w in sorted(set(a) | set(b)) if a.get(w) != b.get(w)] or ["words"]
        elif key == "environ" and isinstance(a, list) and isinstance(b, list):
            da, db = {str(x[0]): x[1] for x in a}, {str(x[0]): x[1] for x in b}
            out += [f"environment {v}" for v in sorted(set(da) | set(db)) if da.get(v) != db.get(v)] or ["environment"]
        else:
            out.append(key)
    return out


def program_key(ctx: Any, argv: list[str], env: dict, line: dict) -> dict | None:
    """What the session's program is: the answer check's operator key, the runner's line and environment (but the
    request's own words), the cards, the user and this program. None when the context cannot name its operator."""
    from . import ad_doors as d
    operator = d.check_key(ctx)
    if operator is None:
        return None
    masked = list(argv)
    for flag in REQUEST_FLAGS:
        for i, word in enumerate(masked[:-1]):
            if word == flag:
                masked[i + 1] = "<request>"
    key = {"schema": SCHEMA, "operator": operator, "argv": masked, "env": dict(sorted(env.items())),
           "environ": sorted([k, v] for k, v in os.environ.items()
                             if k.startswith(_ENV_PREFIXES) and k != "CUDA_VISIBLE_DEVICES"),
           "cards": list(ctx.cards), "uid": os.geteuid(), "library_path": d.library_path(),
           # the runner's own libraries (the physics library it loads from its folder): part of the build
           "runner_libraries": {p.name: d._content_sum(p)
                                for p in sorted(Path(ctx.runner).resolve().parent.glob("*.so*")) if p.is_file()},
           "empty_move": line.get("empty_move"),
           "session_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return json.loads(json.dumps(key, sort_keys=True, default=str))


_KEY_WORDS = {"argv": "runner words", "env": "runner environment", "environ": "environment", "cards": "cards",
              "runner_libraries": "runner libraries",
              "uid": "user", "library_path": "library folders", "empty_move": "start move",
              "session_code_sha256": "session program"}


def key_differs(old: dict, new: dict) -> list[str]:
    """The parts of two keys that differ, in the door's words."""
    from . import ad_doors as d
    out = [_KEY_WORDS.get(k, k) for k in sorted(set(old) | set(new)) if k != "operator" and old.get(k) != new.get(k)]
    a, b = old.get("operator") or {}, new.get("operator") or {}
    out += [d._KEY_WORDS.get(k, k) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return out


# ------------------------------------------------------------------------------------------ one command's use
class Use:
    """One command's dealing with the session on its card: found before any card is read (``open_for``), then
    attached to or started when the runner's line is known (``SessionProgram.start``)."""

    def __init__(self, ctx: Any, mode: str, why_off: str | None):
        self.ctx, self.mode, self.why_off = ctx, mode, why_off
        self.door = getattr(ctx, "door", "sensitivity")
        self.idle_s = float(getattr(ctx.args, "session_idle_s", None) or DEFAULT_IDLE_S)
        self.name: str | None = None
        self.root: Path | None = None
        self.live: dict | None = None        # the session found loaded, when this command means to attach to it
        self.how: str | None = None           # "attached" | "started fresh"
        self.why: str | None = None
        self.ended: str | None = None         # why this command ended the session's program, when it did
        self.stopped: list[dict] = []
        self.program_obj: "SessionProgram | None" = None
        self._fd: int | None = None
        self.defer = False                    # open_for(defer=True): the session's line waits for said()
        self.held_lines: list[str] = []
        self._stopped_here: list[dict] = []   # programs this command stopped (not ones it found ending)

    # ---------------------------------------------------------------------------------------- the session's line
    def say(self, text: str) -> None:
        """The session's line: printed now, or (``open_for(defer=True)``) held until the command's checks pass."""
        if self.defer:
            self.held_lines.append(text)
        else:
            print(text, flush=True)

    def said(self) -> None:
        """The command's card, host-memory and disk checks passed: the held line is printed now.

        THE BREAKAGE THIS PREVENTS: the line was printed before those checks, so on a 32 GB card the 3 km storm
        command first said "STARTED FRESH: ... this command leaves its program loaded for the next one", then was
        refused for the card's size, and no program was ever loaded (a fresh install of a 0.6.0 build)."""
        lines, self.held_lines, self.defer = self.held_lines, [], False
        for text in lines:
            print(text, flush=True)

    def refused(self) -> None:
        """The command stopped before its checks passed: the held line is not printed (nothing it says happened),
        a program this command stopped is said (that did happen), and the session's lock is given back, its file
        removed when no program is loaded for this card (``unlock(tidy=True)``).

        THE BREAKAGE THE REMOVAL PREVENTS: a 3 km command refused for the card's size left ``card<N>.lock`` in the
        user's sessions folder although no program was ever loaded (two 0.6.0 builds' fresh
        installs)."""
        if not self.defer:
            return
        self.held_lines, self.defer = [], False
        for info in self._stopped_here:
            print(f"woof hex {self.door}: session {info.get('name') or self.name}: the program found loaded (pid "
                  f"{info.get('pid')}) was stopped ({info.get('why')}); this command stopped before it started one "
                  "(the reason follows), so no program is loaded for this card now", flush=True)
        self.unlock(tidy=True)

    # -------------------------------------------------------------------------------------------------- the lock
    def lock(self) -> None:
        """One command at a time talks to the session on a card: a second command waits here for the first."""
        if self._fd is not None or self.root is None:
            return
        path = self.root / f"{self.name}.lock"
        while True:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _posix_lock().flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                try:
                    holder = os.pread(fd, 300, 0).decode("utf-8", "replace").strip()
                except OSError:
                    holder = ""
                print(f"woof hex {self.door}: session {self.name} is serving another command"
                      + (f" ({holder})" if holder else "") + ": waiting for it", flush=True)
                _posix_lock().flock(fd, fcntl.LOCK_EX)
            # the file this command locked must still be the one the name leads to: a command that ends with no
            # program loaded removes its lock file (unlock(tidy=True)), and a lock on a removed file excludes nobody
            try:
                if os.path.samestat(os.fstat(fd), os.stat(path)):
                    break
            except OSError:
                pass
            os.close(fd)
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"pid {os.getpid()}: {' '.join(sys.argv)[:240]}\n".encode("utf-8", "replace"), 0)
        self._fd = fd

    def unlock(self, tidy: bool = False) -> None:
        """The lock given back. ``tidy``: its file is removed first, while this command still holds it, when no
        program is loaded or ending for this card; the file beside a loaded program stays."""
        if self._fd is not None:
            try:
                if tidy and self.root is not None and self.name:
                    _tidy_lock(self._fd, self.root, self.name)
                os.close(self._fd)
            finally:
                self._fd = None

    # ---------------------------------------------------------------------------------- what the session holds
    def _count_held(self, live: dict | None) -> None:
        """The card and host memory the session's program holds is this command's to use (it attaches to it or stops
        it): the door's free-memory readings count it, so a command is not refused for memory its own session holds."""
        from . import ad_doors as d
        held = d._SESSION_HELD
        held["card"].clear()
        held["host"] = 0
        if live is not None:
            card = str(self.ctx.cards[0])
            mib = sum(d.process_card_mib(live["pid"]).values())
            try:   # all that is in use on the card now: the program's own, with what the driver keeps beside it
                total, free = d.total_mib(card), d.free_mib(card)
                if total is not None and free is not None:
                    mib = max(mib, int(total) - int(free))
            except (d.AdDoorError, SystemExit):
                pass
            held["card"][card] = mib
            held["host"] = _host_mib(live["pid"])
            self.held = {"card_mib": mib, "host_mib": held["host"]}

    def _stop(self, live: dict, why: str) -> None:
        info = stop_live(live)
        self.stopped.append({**info, "why": why})
        self._stopped_here.append(self.stopped[-1])
        self._count_held(None)
        self.live = None

    def wait_ending(self, wait_s: float = STOP_WAIT_S) -> dict | None:
        """A program of this session's folder that is ending as this command starts (``read_ending``): the command
        waits for its last thread, as the command that stopped it does, and counts it among the programs it stopped,
        so its card and host readings wait for the memory that program is giving back (ad_doors.Context: the card is
        admitted with a wait and the host memory is read once it has settled). None when no program is ending."""
        if self.root is None or self.name is None:
            return None
        ending = read_ending(self.root, self.name)
        if ending is None:
            return None
        print(f"woof hex {self.door}: session {self.name} (pid {ending['pid']}): its program was told to end before "
              "this command started and is still giving back the card and its host memory: waiting for it (at most "
              f"{wait_s:g} s)", flush=True)
        handle, t0, killed = _Pid(ending["pid"], None, ending["dir"]), time.monotonic(), False
        try:
            handle.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            handle.kill()
            killed = True
            try:
                handle.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                pass
        info = {"name": self.name, "pid": ending["pid"], "ended_by_pid": killed, "ended": handle.poll() is not None,
                "waited_s": round(time.monotonic() - t0, 1),
                "why": "it was ending when this command started (another command, or its own idle time-out, had "
                       "told it to end): this command waited for its last thread"}
        self.stopped.append(info)
        return info

    # ------------------------------------------------------------------------------------------------ the program
    def program(self, line: dict | None) -> "SessionProgram | None":
        """This command's session program (one object a command), or None when it uses none."""
        if self.mode != "use" or not line or not line.get("loaded_program"):
            return None
        if self.program_obj is None:
            self.program_obj = SessionProgram(self, line)
            self.ctx.session_program = self.program_obj
        return self.program_obj

    def block(self) -> dict:
        """The receipt's words on the session."""
        if self.mode != "use":
            return {"on": False, "why": self.why_off, "stopped": self.stopped}
        p = self.program_obj
        left = bool(p is not None and p.proc is not None and p.proc.poll() is None and self.ended is None)
        pid = getattr(getattr(p, "proc", None), "pid", None)
        return {"on": True, "name": self.name, "how": self.how, "why": self.why, "dir": str(self.root / self.name),
                "pid": pid, "idle_s": getattr(p, "idle_s", self.idle_s),
                # the program is not this command's child when it attached: its memory is read here, not in
                # the receipt's host_peak_mib
                "program_host_mib": _host_mib(pid) if left and pid else None,
                "left_loaded": left, "ended": self.ended, "stopped": self.stopped,
                "served_this_command": list(getattr(p, "served", []) or []),
                "stop_word": f"woof hex sensitivity --session-stop {self.name}",
                "what": "the runner kept loaded across commands for this plan, build and settings (woof.hex.ad_session)"}

    def finish(self) -> None:
        """The command's last words on the session, and its lock given back."""
        if self.mode == "use" and self.program_obj is not None:
            p = self.program_obj
            if self.ended:
                print(f"woof hex {self.door}: session {self.name}: ended ({self.ended}); the next sensitivity "
                      "command starts one", flush=True)
            elif p.proc is not None and p.proc.poll() is None:
                print(f"woof hex {self.door}: session {self.name}: left loaded (pid {p.proc.pid}); it ends after "
                      f"{p.idle_s:g} s with no request and frees the card; stop it now: woof hex sensitivity "
                      f"--session-stop {self.name}", flush=True)
        self._count_held(None)
        self.unlock(tidy=True)   # with no program left loaded for this card the lock file goes too


def wanted(ctx: Any) -> tuple[str, str | None]:
    """("use", None), or ("off", why) for a command that keeps no session."""
    args, door = ctx.args, getattr(ctx, "door", None)
    asked = getattr(args, "session", None) or os.environ.get(MODE_ENV) or None
    if door != "sensitivity":
        return "off", f"the {door} door runs its own program"
    if asked in ("off", "0"):
        return "off", "--session off"
    row, table = ctx.row, ctx.table
    line = (table["doors"].get(door) or {}).get("gradient_line") or {}
    if not row.get("gradient_line") or not line.get("loaded_program"):
        return "off", "the row has no gradient line with a loaded program"
    if getattr(args, "checked_forward", False):
        return "off", "--checked-forward runs one program a window"
    if getattr(ctx, "split", None) or len(ctx.cards) > 1:
        return "off", "a window on several cards runs one program a window"
    if asked != "on":
        for var in SCHEDULER_ENV:
            if os.environ.get(var):
                return "off", (f"a batch job ({var} is set): the scheduler owns the card, and a session would hold "
                               "it after the job; --session on keeps one anyway")
    return "use", None


def open_for(ctx: Any, defer: bool = False) -> Use:
    """Called by the door's context once the command's cards are known and before any card is read. A command that
    uses a session takes the card's session lock (held to its end) and prints its FIRST line: ATTACHED to the
    program found loaded for the same words, or STARTED FRESH. A command that uses none stops a session of this
    user found on its card (the card is needed), after any command that session is serving.

    ``defer`` (the door's context): the line is held until the command's card, host-memory and disk checks pass
    (``Use.said``) and dropped if the command is refused before then (``Use.refused``), so a refused command never
    says that it started or attached to a program."""
    for earlier in _OPEN:   # a context of this process that failed while it held a session lock gives it back
        earlier.unlock()
    _OPEN.clear()
    mode, why_off = wanted(ctx)
    use = Use(ctx, mode, why_off)
    use.defer = bool(defer)
    door = use.door
    _OPEN.append(use)
    try:
        use.name = session_name(ctx.args, ctx.cards)
        if mode == "off" and why_off and why_off.startswith("a batch job"):
            use.say(f"woof hex {door}: session: none kept ({why_off})")
        if mode != "use" and not sessions_dir().is_dir():   # no session was ever kept here: nothing to stop, nothing made
            return use
        use.root = private_dir(sessions_dir())
    except (SessionFolderError, OSError) as error:
        if mode == "use":
            use.say(f"woof hex {door}: session: not used ({error}): STARTED FRESH, and the program ends with this "
                    "command")
        use.mode, use.why_off, use.root = "off", f"the sessions folder cannot be used ({error})", None
        return use
    if mode != "use":
        live = read_live(use.root, use.name)
        if live is not None:
            use.lock()
            live = read_live(use.root, use.name)
            if live is not None:
                use._stop(live, why_off or "this command runs its own program on the card")
                use.say(f"woof hex {door}: session {use.name} (pid {live['pid']}) stopped: this command runs its own "
                        f"program on card {ctx.cards[0]} ({why_off})")
            use.unlock()
        else:   # none loaded: one that is ending is waited for, and counted as stopped by this command
            use.wait_ending()
        return use
    try:
        use.lock()
    except OSError as error:
        use.say(f"woof hex {door}: session: not used (its lock cannot be taken: {error}): STARTED FRESH, and the "
                "program ends with this command")
        use.mode, use.why_off = "off", f"the session lock cannot be taken ({error})"
        return use
    live = read_live(use.root, use.name)
    words = command_words(ctx)
    use.words = json.loads(json.dumps(words, default=str))
    tail = (f"this command leaves its program loaded for the next one (it ends after {use.idle_s:g} s with no "
            f"request and frees the card; stop it: woof hex sensitivity --session-stop {use.name}; --session off "
            "keeps none)")
    if live is None:
        ending = use.wait_ending()   # a program that is ending as this command starts: waited for, counted as stopped
        why = ("no program was loaded for this card" if ending is None else
               f"the program that was loaded for this card (pid {ending['pid']}) was ending when this command "
               f"started; waited {ending['waited_s']:g} s for it")
        use.how, use.why = "started fresh", why
        use.say(f"woof hex {door}: session {use.name}: STARTED FRESH: {why}; {tail}")
        return use
    then = live["doc"].get("command") or {}
    differ = words_differ(then, use.words)
    if differ:
        use._stop(live, "started with other words: " + ", ".join(differ))
        use.how, use.why = "started fresh", "the loaded program was started with other words: " + ", ".join(differ)
        use.say(f"woof hex {door}: session {use.name}: STARTED FRESH: the program found loaded (pid {live['pid']}) "
                f"was started with other words ({', '.join(differ)}) and was stopped; {tail}")
        return use
    use.live = live
    use._count_held(live)
    use.how, use.why = "attached", None
    held = getattr(use, "held", {})
    use.say(f"woof hex {door}: session {use.name}: ATTACHED to the program loaded at {live['doc'].get('started_utc')} "
            f"(pid {live['pid']}, {live.get('served') or 0} gradients served; it holds {held.get('card_mib', 0):,} MiB "
            f"of card {ctx.cards[0]} and {held.get('host_mib', 0):,} MiB of host memory, counted below as this "
            "command's): no program start")
    return use


# ------------------------------------------------------------------------------------------------- the program
def _loaded_program_class():
    from . import ad_doors as d

    class _SessionProgram(d.LoadedProgram):
        """``ad_doors.LoadedProgram`` whose runner outlives the command: ``start`` attaches to the session found
        loaded when its key is this command's, else starts a fresh detached program; ``request`` starts fresh by
        itself when the program is gone; ``stop`` leaves it loaded; ``end`` ends it."""
        REQUEST_WORDS = d.LoadedProgram.REQUEST_WORDS

        def __init__(self, use: Use, line: dict):
            lp = line.get("loaded_program") or {}
            super().__init__(use.root / use.name, use.idle_s, float(lp.get("stop_wait_s", STOP_WAIT_S)))
            self.use, self.line = use, line
            self.attached = False
            self.key: dict | None = None
            self._started_with: tuple | None = None
            self._n = 0

        # ------------------------------------------------------------------------------------------- the names
        def _unique(self, name: str) -> str:
            """A request name no earlier request of the session had, sorted after them (the runner serves the
            oldest name first): an answer under it can only be this request's."""
            self._n += 1
            clean = "".join(c if c.isalnum() else "-" for c in str(name))[:24]
            return f"q{time.time_ns():020d}-{os.getpid()}-{self._n:03d}-{clean}"

        def stage_label(self, first: bool, name: str) -> str:
            how = ("session " + self.use.name + (" attached" if self.attached else " started fresh")) if first \
                else f"session {self.use.name} request"
            return how

        # --------------------------------------------------------------------------------------- start or attach
        def start(self, argv: list[str], env: dict, plan: Path, card: str, name: str, out: Path) -> dict:
            use, ctx = self.use, self.use.ctx
            from .ad import rows as _rows
            getattr(_rows, "forget_plan_docs", lambda: None)()   # no parsed plan is held while a runner works
            self._started_with = (list(argv), dict(env), Path(plan), str(card))
            self.key = program_key(ctx, argv, env, self.line)
            live = use.live or read_live(use.root, use.name)
            if live is not None and self.key is not None:
                differ = key_differs(live["doc"].get("key") or {}, self.key)
                if not differ and not (live["dir"] / "stop").exists():
                    self.proc = _Pid(live["pid"], live["ticks"], live["dir"])
                    self.attached, self.first, self.t0 = True, live["doc"].get("first"), time.monotonic()
                    self.idle_s = float(live["doc"].get("idle_s") or self.idle_s)
                    use.how, use.why = "attached", None
                    return self.request(name, d._request_words(argv, out), out)
                why = ("the loaded program's " + ", ".join(differ) + " differ from this command's" if differ
                       else "the loaded program was already asked to stop")
                use._stop(live, why)
                use.how, use.why = "started fresh", why
                print(f"woof hex {use.door}: session {use.name}: NOT ATTACHED: {why}; it was stopped: STARTED FRESH",
                      flush=True)
                self._admit()
            elif live is not None:
                use._stop(live, "this command's operator cannot be named by a key")
                self._admit()
            if self.key is None:
                use.how, use.why = "started fresh", "this command's operator cannot be named by a key: no later " \
                                                    "command can attach to its program"
            return self._fresh(argv, env, plan, card, name, out)

        def _admit(self) -> None:
            """The card again, after a program of this user's was stopped on it (its memory comes back as it ends)."""
            ctx = self.use.ctx
            d.admit_card(ctx.card, ctx.need_mib(), max(float(getattr(ctx.args, "wait_card", 0.0) or 0.0), 90.0))

        def _fresh(self, argv: list[str], env: dict, plan: Path, card: str, name: str, out: Path) -> dict:
            d.refuse_no_local_gpu()
            d.refuse_hidden_card()
            d.refuse_planted_env("session program", before_download=False)
            use = self.use
            name = self._unique(name)
            shutil.rmtree(self.rdir, ignore_errors=True)
            private_dir(self.rdir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "bstore").mkdir(exist_ok=True)
            argv = list(argv)
            if "--x0-dir" in argv:   # the session's own copy: the command's folder may be gone before a later request
                move = self.rdir / "empty-move.json"
                shutil.copyfile(argv[argv.index("--x0-dir") + 1], move)
                argv[argv.index("--x0-dir") + 1] = str(move)
            self.argv = argv + ["--resident", str(self.rdir), "--resident-name", name,
                                "--resident-idle-s", f"{self.idle_s:g}"]
            child = {**os.environ, **env, "CUDA_VISIBLE_DEVICES": str(card), "HEXAD_LIBRARY_PATH": d.library_path()}
            text = " ".join(f"{k}={v}" for k, v in sorted(env.items())) + " " + " ".join(self.argv[1:]) + "\n"
            (self.rdir / "cmd").write_text(text)
            (out / "cmd").write_text(text)
            doc = {"schema": SCHEMA, "name": use.name, "uid": os.geteuid(), "key": self.key,
                   "key_sha256": hashlib.sha256(json.dumps(self.key, sort_keys=True).encode()).hexdigest()
                   if self.key else None,
                   "command": getattr(use, "words", None), "pid": None, "start_ticks": None, "first": name,
                   "idle_s": self.idle_s, "cards": list(use.ctx.cards), "plan": str(plan), "row": use.ctx.row_name,
                   "steps": int(use.ctx.steps), "started_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                   "started_by": {"pid": os.getpid(), "argv": sys.argv}}
            _write_json(self.rdir / "session.json", doc)
            with open(self.rdir / "stdout", "wb") as so, open(self.rdir / "stderr", "wb") as se:
                self.proc = subprocess.Popen(self.argv, cwd=plan, env=child, stdin=subprocess.DEVNULL, stdout=so,
                                             stderr=se, start_new_session=True)
            self._streams = ()
            doc["pid"], doc["start_ticks"] = self.proc.pid, _start_ticks(self.proc.pid)
            _write_json(self.rdir / "session.json", doc)
            self.t0, self.first, self.attached = time.monotonic(), name, False
            if use.how is None:
                use.how = "started fresh"
            try:
                return self._tidy(name, self._answer(name, out), out)
            except d.NotLoaded:
                use.ended = "the runner ended after its gradient: it serves no requests"
                raise

        # ----------------------------------------------------------------------------------------------- requests
        def request(self, name: str, words: dict, out: Path) -> dict:
            unique = name if name.startswith("q") and name[1:21].isdigit() else self._unique(name)
            use = self.use
            try:
                pid = _read_json(self.rdir / "session.json").get("pid")
                if self.proc is None or self.proc.poll() is not None or pid != self.proc.pid:
                    raise SessionLost("its program is not running" if pid == getattr(self.proc, "pid", None)
                                      else "another program was started in its folder")
                try:
                    doc = super().request(unique, words, out)
                except d.CardMemoryError:
                    raise
                except d.AdDoorError as error:
                    if self.proc.poll() is not None and not (self.rdir / f"{unique}.done").is_file():
                        raise SessionLost(str(error)) from error
                    raise
                return self._tidy(unique, doc, out)
            except SessionLost as lost:
                for stale in (f"{unique}.req", f"{unique}.req.part", f"{unique}.taken"):
                    (self.rdir / stale).unlink(missing_ok=True)
                if self._started_with is None:
                    raise d.AdDoorError(f"session {use.name}: {lost}") from lost
                if use.ended:   # this command ended it itself (a window of its own needed the card)
                    print(f"woof hex {use.door}: session {use.name}: STARTED FRESH again (this command ended its "
                          f"program: {use.ended})", flush=True)
                    use.how, use.why = "started fresh", f"this command had ended its program ({use.ended})"
                else:
                    print(f"woof hex {use.door}: session {use.name}: its program ended before it answered ({lost}); "
                          f"log {self.rdir / 'stderr'}: STARTED FRESH", file=sys.stderr, flush=True)
                    use.how, use.why = "started fresh", f"the session's program ended before it answered ({lost})"
                use.ended = None
                use._count_held(None)
                use.live = None
                argv, env, plan, card = self._started_with
                argv = list(argv)
                for word, value in words.items():
                    if f"--{word}" in argv:
                        argv[argv.index(f"--{word}") + 1] = str(value)
                self.proc = None
                self._admit()
                return self._fresh(argv, env, plan, card, name, out)

        def _tidy(self, name: str, doc: dict, out: Path) -> dict:
            """The answer is this request's (the runner names the request it served); its mailbox files and its
            emptied store folder go."""
            served = doc.get("resident")
            if isinstance(served, dict) and served.get("request") not in (None, name):
                raise d.AdDoorError(f"session {self.use.name}: the window receipt in {doc.get('args', {}).get('out')} "
                                    f"is request {served.get('request')}'s, not {name}'s: not taken")
            for suffix in (".done", ".taken"):
                (self.rdir / f"{name}{suffix}").unlink(missing_ok=True)
            shutil.rmtree(Path(out) / "bstore", ignore_errors=True)
            doc["loaded_program"] = {**(doc.get("loaded_program") or {}), "session": self.use.name,
                                     "attached": self.attached}
            return doc

        # -------------------------------------------------------------------------------------------- the ending
        def _clear_stores(self) -> None:
            for item in self.served:
                shutil.rmtree(Path(item["out"]) / "bstore", ignore_errors=True)

        def stop(self) -> dict:
            """The command's end: the program stays loaded (``end`` ends it)."""
            self._clear_stores()
            running = self.proc is not None and self.proc.poll() is None
            return {"on": self.proc is not None, "session": self.use.name, "dir": str(self.rdir),
                    "pid": getattr(self.proc, "pid", None), "attached": self.attached, "left_loaded": running,
                    "ended": self.use.ended, "idle_s": self.idle_s, "served": self.served,
                    "card_peak_mib": max(self.peak.values(), default=0) or None,
                    "stderr": str(self.rdir / "stderr"),
                    "what": "the session's runner served every objective of this call and stays loaded for the next "
                            "command; it ends after idle_s seconds with no request"}

        def end(self, why: str) -> None:
            """End the session's program now (this command needs the card for a window of its own, or the caller
            closes the session)."""
            if self.proc is None or self.proc.poll() is not None or self.use.ended:
                return
            self._clear_stores()
            stop_live({"pid": self.proc.pid, "ticks": _start_ticks(self.proc.pid), "dir": self.rdir,
                       "name": self.use.name}, self.stop_wait_s)
            try:
                self.proc.wait(timeout=30.0)   # a program this command started is its child: reaped here
            except subprocess.TimeoutExpired:
                pass
            self.use.ended = why
            self.use._count_held(None)
            self.use.live = None

    return _SessionProgram


def SessionProgram(use: Use, line: dict):   # noqa: N802  (a class made on first use: it extends ad_doors.LoadedProgram)
    global _CLASS
    if _CLASS is None:
        _CLASS = _loaded_program_class()
    return _CLASS(use, line)


_CLASS = None


# ------------------------------------------------------------------------------------- list and stop by name
def list_sessions() -> list[dict]:
    try:
        root = private_dir(sessions_dir())
    except (SessionFolderError, OSError):
        return []
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        live = read_live(root, folder.name)
        if live is not None:
            doc = live["doc"]
            out.append({"name": folder.name, "pid": live["pid"], "state": live["state"], "served": live.get("served"),
                        "row": doc.get("row"), "steps": doc.get("steps"), "plan": doc.get("plan"),
                        "cards": doc.get("cards"), "idle_s": doc.get("idle_s"), "started_utc": doc.get("started_utc"),
                        "host_mib": _host_mib(live["pid"]), "dir": str(folder)})
    return out


def stop_sessions(name: str | None = None, wait_s: float = STOP_WAIT_S) -> list[dict]:
    """Stop the session ``name`` (every session of this user when None or "all"), after the command it is serving."""
    for own in _OPEN:   # a lock this very process holds would make it wait for itself
        own.unlock()
    try:
        root = private_dir(sessions_dir())
    except (SessionFolderError, OSError):
        return []
    done = []
    for item in list_sessions():
        if name not in (None, "all") and item["name"] != name:
            continue
        fd = os.open(root / f"{item['name']}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _posix_lock().flock(fd, fcntl.LOCK_EX)
            live = read_live(root, item["name"])
            if live is not None:
                done.append(stop_live(live, wait_s))
            _tidy_lock(fd, root, item["name"])   # the stopped session's lock file goes with it
        finally:
            os.close(fd)
    return done


class _ListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        rows = list_sessions()
        for r in rows:
            print(f"session {r['name']}: pid {r['pid']}, {r['state']}, {r['served'] or 0} gradients served, row "
                  f"{r['row']}, {r['steps']} steps, card {','.join(r['cards'] or [])}, {r['host_mib']:,} MiB of host "
                  f"memory, loaded {r['started_utc']}, ends after {r['idle_s']:g} s idle; plan {r['plan']}")
        if not rows:
            print("no session is loaded")
        parser.exit(0)


class _StopAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        done = stop_sessions(values)
        for r in done:
            print(f"session {r['name']} (pid {r['pid']}) "
                  + ("stopped" if r["ended"] and not r["ended_by_pid"] else
                     "ended by its pid (it did not stop when asked)" if r["ended"] else "did NOT end"))
        if not done:
            print("no session is loaded" if values in (None, "all") else f"no session {values} is loaded")
        parser.exit(0 if all(r["ended"] for r in done) else 1)


def add_session_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("the session (the program kept loaded across commands)")
    g.add_argument("--session", default=None, choices=("on", "off"),
                   help="on a row with a gradient line the command leaves its program loaded and the next sensitivity "
                        "command for the same plan, build and settings uses it (its first line says ATTACHED or "
                        "STARTED FRESH); off: the program ends with the command (default: on, but off in a batch "
                        f"job; {MODE_ENV}=off sets it for a shell)")
    g.add_argument("--session-idle-s", type=float, default=None,
                   help=f"seconds with no request after which a session this command starts ends and frees the card "
                        f"(default {DEFAULT_IDLE_S:g})")
    g.add_argument("--session-name", default=None, help="the session's name (default: card<N> for card N)")
    g.add_argument("--session-list", nargs=0, action=_ListAction, help="list this user's loaded sessions and exit")
    g.add_argument("--session-stop", nargs="?", metavar="NAME", action=_StopAction,
                   help="stop the session NAME (every session of this user without a name), after the command it is "
                        "serving, and exit")


# ------------------------------------------------------------------------------------------- the Python object
class SensitivitySession:
    """One object that holds the session and returns gradients for new targets (an optimisation loop's entry).

        from woof.hex.ad_session import SensitivitySession
        with SensitivitySession("runs/opt", row="hrrrv5-q3.96.38278.n40.60w91.30", steps=59) as s:
            g = s.gradient("theta_m", circle="40.6,-91.3,150", levels="1-10")
            g["fields"]["dJ_dtheta"]            # numpy, [levels, cells]
            h = s.gradient("qv", circle="40.6,-91.3,150")

    The door's context (the row, the plan's kept facts, the field map, the mesh) is built once; each ``gradient`` is
    the ``sensitivity`` door's own answer for that target in its own folder (``<out>/g0001``, ...: sensitivity.nc,
    receipt.json), with the same answer check rule and the same bytes as the command. The first gradient attaches to
    the session found loaded for the same plan, build and settings, or starts it; ``close`` (and leaving the ``with``
    block) ends the program unless ``leave_loaded`` is true. ``words`` are further ``sensitivity`` command words
    (for example ["--boundary-layer", "taped"])."""

    def __init__(self, out: str | os.PathLike, *, row: str | None = None, steps: int | None = None,
                 card: int | None = None, idle_s: float = DEFAULT_IDLE_S, leave_loaded: bool = False,
                 words: list[str] | tuple[str, ...] = ()):
        from . import ad_doors as d
        argv = ["sensitivity", "--out", str(out), "--session", "on", "--session-idle-s", f"{float(idle_s):g}"]
        argv += (["--row", row] if row else []) + (["--steps", str(int(steps))] if steps else [])
        argv += (["--card", str(int(card))] if card is not None else []) + [str(w) for w in words]
        parser = argparse.ArgumentParser(prog="woof hex")
        d.add_ad_parsers(parser.add_subparsers(dest="command"))
        self.args = parser.parse_args(argv)
        if getattr(self.args, "objectives", None) is not None:
            raise d.AdDoorError("SensitivitySession takes one target a gradient() call, not --objectives")
        table = d.load_table()
        first = d.first_gradient_route(self.args, table)
        d.refuse_speed_words(self.args, table)
        route = d.gradient_route(self.args, "sensitivity")
        if route["carried"]:
            raise d.AdDoorError("SensitivitySession runs on the mesh itself: a gradient carried from a companion mesh "
                                "is one command a gradient")
        self.leave_loaded = bool(leave_loaded)
        self.ctx = d.Context(self.args, "sensitivity")
        self.ctx.gradient_route, self.ctx.first_gradient = route, first
        self.use: Use = self.ctx.session
        self.out = self.ctx.out
        self.gradients = 0
        self.use.unlock()
        self.per_call: str | None = None
        if self.use.mode != "use":
            if not str(self.use.why_off or "").startswith("the row has no gradient line"):
                raise d.AdDoorError(f"no session on this row or command: {self.use.why_off}")
            # A row with no gradient line keeps no program loaded: each gradient() is the sensitivity command's own
            # program, the same answer and bytes as the command, as GUIDE section 2 says of the 12 km row.
            # THE BREAKAGE THIS PREVENTS: the shipped example (examples/differentiable/own_measure.py, GUIDE section
            # 2, run-32gb.sh step 3) stopped at once on the 12 km row with "no session on this row or command"
            # (a review of 0.6.0, 2026-11-15).
            self.per_call = self.use.why_off
            print(f"woof hex sensitivity: no session on this row ({self.use.why_off}): each gradient() runs its "
                  "own program, as the command does", flush=True)

    def gradient(self, metric: str | None = None, *, levels: str | None = None, box: str | None = None,
                 circle: str | None = None, draw_map: bool = False) -> dict:
        """dJ/d(initial state) of the mean of ``metric`` over the region and levels at the window end: {"fields":
        {name: numpy array}, "units", "objective", "receipt", "out", "valid", "passed", "seconds", "latCell",
        "lonCell"} (the cell centres in degrees, as sensitivity.nc holds them)."""
        from . import ad_doors as d
        self.gradients += 1
        ctx = self.ctx
        args = argparse.Namespace(**vars(self.args))
        args.metric, args.levels, args.box, args.circle = metric or args.metric, levels, box, circle
        args.no_map = not draw_map
        ctx.args = args
        ctx.out = self.out / f"g{self.gradients:04d}"
        ctx.out.mkdir(parents=True, exist_ok=True)
        ctx.stages, ctx.windows, ctx.started = [], [], time.time()
        t0 = time.monotonic()
        self.use.lock()
        try:
            code, info = d._sensitivity(ctx, args)
        finally:
            self.use.unlock()
        rec = info["rec"]
        return {"fields": {name: value[0] for name, value in info["out_fields"].items()}, "units": info["units"],
                "objective": info["objective"], "receipt": rec, "out": ctx.out, "valid": rec.get("valid"),
                "passed": code == 0, "seconds": round(time.monotonic() - t0, 3),
                "session": rec.get("session"), "latCell": ctx.geometry["latCell"], "lonCell": ctx.geometry["lonCell"]}

    def close(self) -> None:
        program = self.use.program_obj
        if program is not None and not self.leave_loaded:
            self.use.lock()
            try:
                program.end("the Python session object was closed")
            finally:
                self.use.unlock()

    def __enter__(self) -> "SensitivitySession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
