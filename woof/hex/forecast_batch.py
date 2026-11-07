"""Member batch: an ensemble's member forecasts on worker slots, several per card.

usage: python -m woof.hex.forecast_batch ROSTER.json RECEIPT.json

A storm-scale cycle forecasts 30+ members of one mesh for one short window.
The box's granted cards each carry ``workers_per_card`` worker slots; members
flow from ONE queue to whichever slot is free, so a card integrates one
member while its other slots prepare the next and write the last one's
history, and no card idles while another still has a pack to finish.  The
slots of a card start ``stagger_seconds`` apart so their phases interleave
instead of every slot preparing at once and then integrating at once.

Each member runs through ``woof.hex.cli forecast`` with the member's own
arguments: exactly the separate process's door, driver, receipts and
destinations.  A slot process imports the package and the drivers once and
FORKS one child per member, so a member starts from the state a fresh
process has after its imports (no bind, no CUDA context, no cache of the
previous member) without paying the interpreter and imports again.
Nothing of one member survives into the next: the child exits.  THE
BREAKAGE A RESIDENT LOOP HAD (measured 2026-10-13, a 167,266-cell storm-scale
patch): members
run one after another in one process kept every finished member's device
state reachable (6.8 GB per member: the class executor's closures, and
woof.hex.cuda_dynamics_v841._REFERENCE_WIND holds its arrays for the life of
the process), so a worker's second member peaked 6 GB higher than its first;
and the mesh bind refuses a second bind of the same row in one process.  The
proof of identity is per member: every variable of every history frame
compared byte for byte against the same member run as its own process.

Kernels are not fused across members: each member is its own forecast on
its own context, and the card interleaves the slots' kernels.

The tail (fewer members queued than live slots) is spread evenly over the
cards instead of going to whichever slots free first, and a card whose
slots have all ended while another card still works is handed back to the
box queue at once, through the per-card release command the queue exports
to a granted command as GPU_MUTEX_RELEASE (``$GPU_MUTEX_RELEASE CARD``).
THE BREAKAGE BOTH PREVENT (a rented 4-card host, 2026-10-14, a free
ensemble batch of 31 members on 4 cards x 3 slots): card 2's three slots finished their second
members first (01:46:02-01:46:40Z) and took three of the last eight, which
then ran three to a card (681-704 s each) while card 3 ran one member alone
(367 s); cards 3, 0 and 1 then sat empty but locked from 01:52, 01:55 and
01:56Z until the batch exited at 01:58:26Z, while two other jobs on the
same host waited for one card each.  See _Dispatch.

Roster (JSON):
  {"schema": "gpuwm-hex.member-batch/v2", "workers_per_card": 5, "stagger_seconds": 20, "min_free_gib": 28,
   "tail_wait_seconds": null, "release_idle_cards": true,
   "members": [{"run_id": "mem001", "argv": [door forecast flags, --init/--out/--scratch included],
                "log": ".../forecast.log", "result": ".../result.json"}, ...]}

Cards: GPU_MUTEX_CARDS (the box queue's grant, physical indices) paired with
CUDA_VISIBLE_DEVICES (the same cards, as the queue exports them).
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback

SCHEMA = "gpuwm-hex.member-batch/v2"
#: A member the door refuses at admission (before it creates a destination)
#: is run again this many times, this many seconds later.
ADMISSION_RETRIES = 3
ADMISSION_RETRY_SECONDS = 30.0
#: A slot sends its next member only once the card has this much free
#: memory (or after MIN_FREE_WAIT_SECONDS): the driver's own floor for the
#: 167,266-cell storm-scale patch is 27.4 GB free at its check, so a member started into less
#: is refused after its host preparation.
MIN_FREE_GIB = 28.0
MIN_FREE_WAIT_SECONDS = 300.0
#: In the tail a running member counts as leaving its card once it has run
#: this fraction of the median member wall time measured so far.
TAIL_FINISHING_FRACTION = 0.75
#: A free slot defers to a lighter card for at most this fraction of that
#: median (TAIL_WAIT_DEFAULT_SECONDS before any member has finished), unless
#: the roster sets tail_wait_seconds (0: no deferring, the pre-2026-10-14 order).
TAIL_WAIT_FRACTION = 0.25
TAIL_WAIT_DEFAULT_SECONDS = 120.0


def _card_free_bytes(visible):
    try:
        text = subprocess.run([*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                               "-i", str(visible)], capture_output=True, text=True, timeout=20).stdout
        return float(text.strip().splitlines()[0]) * 2**20
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _wait_for_card_memory(visible, need_bytes):
    began = time.time()
    while time.time() - began < MIN_FREE_WAIT_SECONDS:
        free = _card_free_bytes(visible)
        if free is None or free >= need_bytes:
            break
        time.sleep(5.0)
    return round(time.time() - began, 1)


def _move_destinations_aside(member, attempt):
    moved = []
    argv = [str(item) for item in member["argv"]]
    for flag in ("--out", "--scratch"):
        path = _flag(argv, flag)
        if path and Path(path).exists():
            target = Path(f"{path}.memory-refused-{attempt}")
            Path(path).rename(target)
            moved.append(str(target))
    return moved
RECEIPT_SCHEMA = "gpuwm-hex.member-batch-receipt/v2"


class BatchRefusal(ValueError):
    """The roster cannot run as one batch."""


def _flag(argv, name):
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


def load_roster(path):
    roster = json.loads(Path(path).read_text(encoding="utf-8"))
    if roster.get("schema") != SCHEMA:
        raise BatchRefusal(f"expected roster schema {SCHEMA}")
    members = roster.get("members") or []
    if not members:
        raise BatchRefusal("the roster names no members")
    ids, destinations = set(), set()
    for member in members:
        missing = {"run_id", "argv", "log", "result"} - set(member)
        if missing:
            raise BatchRefusal(f"member record lacks {sorted(missing)}")
        if member["run_id"] in ids:
            raise BatchRefusal(f"member {member['run_id']} appears twice")
        ids.add(member["run_id"])
        argv = [str(item) for item in member["argv"]]
        if argv and argv[0] == "forecast":
            raise BatchRefusal("argv holds the forecast door's flags only (no 'forecast' subcommand)")
        for flag in ("--init", "--out", "--scratch"):
            if _flag(argv, flag) is None:
                raise BatchRefusal(f"member {member['run_id']} has no {flag}")
        for flag in ("--out", "--scratch"):
            path = str(Path(_flag(argv, flag)).resolve())
            if path in destinations:
                raise BatchRefusal(f"two members share {path}; prevents one member overwriting another's history")
            destinations.add(path)
        for flag in ("--members", "--samehost-devices", "--preflight"):
            if any(item == flag or item.startswith(flag + "=") for item in argv):
                raise BatchRefusal(f"{flag}: a batch member is one single-card forecast")
    if int(roster.get("workers_per_card", 5)) < 1:
        raise BatchRefusal("workers_per_card must be at least 1")
    return roster


def granted_cards(environ=None):
    """[(physical index, CUDA_VISIBLE_DEVICES entry)] for the cards this batch holds."""

    environ = os.environ if environ is None else environ
    visible = [item for item in environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "").split(",") if item]
    granted = [item for item in environ.get("GPU_MUTEX_CARDS", "").replace(" ", "").split(",") if item]
    if granted and visible and len(granted) != len(visible):
        raise BatchRefusal(f"GPU_MUTEX_CARDS {granted} and CUDA_VISIBLE_DEVICES {visible} name different counts")
    if not visible:
        visible = granted or ["0"]
    if not granted:
        granted = [item if item.isdecimal() else str(index) for index, item in enumerate(visible)]
    return list(zip(granted, visible))


# ---------------------------------------------------------------- dispatch
class _Dispatch:
    """The batch's ONE member queue, and whether a free slot on a card starts the next member now.

    Bulk (at least as many members queued as live slots): any free slot takes the next member at once.

    Tail (fewer queued than live slots, so not every slot gets another member): a member goes to the card with the
    fewest members still running.  Whichever slots free first used to take the tail, which piles it onto the card
    whose previous round ended first (measured 2026-10-14: three of the last eight members ran three to a card on card 2,
    681-704 s each, while card 3 ran one alone in 367 s and then sat empty; see the module docstring).  A running
    member that has run TAIL_FINISHING_FRACTION of the median member wall time so far counts as leaving, so a round
    about to end does not make its card look full.  A free slot defers while another live card carries fewer such
    members and will have a slot soon (a free slot, or a member about to leave); deferring only ever points at a
    strictly lighter card, so it cannot cycle, and a slot defers at most ``wait_limit()`` seconds, so a card whose
    members run longer than the estimate cannot hold the tail back longer than that.
    """

    def __init__(self, count, cards, per_card, wait_seconds=None, clock=time.monotonic):
        self.queue = deque(range(count))
        self.live = {card: int(per_card) for card in cards}
        self.running = {card: {} for card in cards}     # card -> {token: start (clock)}
        self.tail_counts = {card: 0 for card in cards}
        self.wait_seconds = None if wait_seconds is None else float(wait_seconds)
        self.tail_began = None
        self.tail_began_unix = None
        self.tail_members = None
        self.deferred = []                               # seconds each deferred tail start waited
        self.walls = []
        self.clock = clock
        self.cond = threading.Condition()
        self._tokens = iter(range(1, 1 << 62))

    def _estimate(self):
        return statistics.median(self.walls) if self.walls else None

    def wait_limit(self):
        if self.wait_seconds is not None:
            return self.wait_seconds
        estimate = self._estimate()
        return TAIL_WAIT_DEFAULT_SECONDS if estimate is None else TAIL_WAIT_FRACTION * estimate

    def _load(self, card, now, estimate):
        """Members running on ``card`` that are not about to leave it."""
        return sum(1 for began in self.running[card].values()
                   if estimate is None or now - began < TAIL_FINISHING_FRACTION * estimate)

    def poll(self, card, waited=None):
        """With ``cond`` held: ("start", member index, token), ("wait", None, None) or ("done", None, None).
        ``waited``: None on a slot's first request for this member, else the seconds it has deferred so far."""

        if not self.queue:
            return "done", None, None
        slots = sum(self.live.values())
        tail = len(self.queue) < slots
        if tail and self.tail_began is None:
            self.tail_began, self.tail_began_unix, self.tail_members = self.clock(), time.time(), len(self.queue)
            print(f"[forecast-batch] tail: {len(self.queue)} members queued for {slots} live slots on "
                  f"{sum(1 for n in self.live.values() if n > 0)} cards; each goes to the least-loaded card",
                  flush=True)
        if tail and (waited or 0.0) < self.wait_limit():
            now, estimate = self.clock(), self._estimate()
            mine = self._load(card, now, estimate)
            for other, slots_live in self.live.items():
                if other == card or slots_live <= 0:
                    continue
                load = self._load(other, now, estimate)
                soon = slots_live > len(self.running[other]) or load < len(self.running[other])
                if load < mine and soon:
                    return "wait", None, None
        index = self.queue.popleft()
        token = next(self._tokens)
        self.running[card][token] = self.clock()
        if tail:
            self.tail_counts[card] += 1
            if waited is not None:
                self.deferred.append(round(waited, 1))
        return "start", index, token

    def take(self, card, poll_seconds=1.0):
        """(member index, token) for a free slot on ``card``, or (None, None) once the queue is empty."""

        with self.cond:
            began = waited = None
            while True:
                verdict, index, token = self.poll(card, waited)
                if verdict != "wait":
                    return index, token
                began = self.clock() if began is None else began
                self.cond.wait(timeout=poll_seconds)
                waited = self.clock() - began

    def finished(self, card, token, wall_seconds=None):
        with self.cond:
            self.running[card].pop(token, None)
            if wall_seconds is not None:
                self.walls.append(float(wall_seconds))
            self.cond.notify_all()

    def requeue(self, index, card, token):
        """A member refused before it integrated goes back on the queue."""

        with self.cond:
            self.running[card].pop(token, None)
            self.queue.append(index)
            self.cond.notify_all()

    def retire(self, card):
        """A slot ended (queue empty, or its process died)."""

        with self.cond:
            self.live[card] = max(0, self.live.get(card, 0) - 1)
            self.cond.notify_all()

    def summary(self):
        return {"tail_began_unix": self.tail_began_unix, "tail_members": self.tail_members,
                "tail_members_per_card": dict(self.tail_counts), "tail_wait_seconds": self.wait_limit(),
                "deferred_starts": len(self.deferred), "deferred_seconds": self.deferred}


def _worker_command():
    """The slot process (a seam: the coordinator tests run a stand-in)."""

    return [sys.executable, "-m", "woof.hex.forecast_batch", "--worker"]


def _release_argv(card):
    """The box queue's per-card release, ``$GPU_MUTEX_RELEASE CARD`` (the queue exports it to a granted command and
    withholds it from a nested step of a larger grant), or None when there is none."""

    command = os.environ.get("GPU_MUTEX_RELEASE")
    return [command, str(card)] if command else None


def _release_card(card):
    """Hand ``card`` back to the box queue; the receipt row says whether the queue took it and why not."""

    argv = _release_argv(card)
    began = time.time()
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        released, message = done.returncode == 0, (done.stdout + done.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        released, message = False, f"{type(error).__name__}: {error}"
    return {"card": card, "released": released, "unix": began, "seconds": round(time.time() - began, 1),
            "message": message[-600:]}


# ---------------------------------------------------------------- coordinator
def run_batch(roster_path, receipt_path):
    started = time.time()
    roster = load_roster(roster_path)
    members = roster["members"]
    per_card = int(roster.get("workers_per_card", 5))
    cards = granted_cards()
    dispatch = _Dispatch(len(members), [card for card, _ in cards], per_card,
                         wait_seconds=roster.get("tail_wait_seconds"))
    rows = [None] * len(members)
    retries = [[] for _ in members]
    workers = []
    released, releasing = [], set()
    log_dir = Path(receipt_path).resolve().parent / (Path(receipt_path).stem + ".workers")
    log_dir.mkdir(parents=True, exist_ok=True)
    if not roster.get("release_idle_cards", True):
        release = "off: the roster sets release_idle_cards false"
    elif _release_argv(cards[0][0]) is None:
        release = ("off: GPU_MUTEX_RELEASE is not set (not under a box queue grant, a nested step of a larger "
                   "grant, or a queue without a per-card release); every card drops when the batch exits")
    else:
        release = "on"
    receipt = {"schema": RECEIPT_SCHEMA, "roster": str(Path(roster_path).resolve()),
               "cards": [card for card, _ in cards], "workers_per_card": per_card,
               "members": rows, "workers": [], "card_release": release, "released_cards": released,
               "status": "incomplete"}

    stagger = float(roster.get("stagger_seconds", 0.0))
    min_free = float(roster.get("min_free_gib", MIN_FREE_GIB)) * 2**30

    def slot_ended(worker):
        """A slot's loop ended.  Once every slot of its card has ended cleanly (queue empty, process exited 0)
        while a slot on another card still works, the card goes back to the box queue now instead of when the
        batch exits.  A card with a dead slot keeps its grant (its member process may still hold the card), and
        the batch's last card is left to the queue's own exit drop."""

        card = worker["card"]
        with dispatch.cond:
            worker["ended"] = True
            mine = [other for other in workers if other["card"] == card]
            if (release != "on" or card in releasing
                    or any(not other["ended"] or other["died"] or other.get("returncode") != 0 for other in mine)
                    or all(other["ended"] for other in workers)):
                return
            releasing.add(card)
        row = _release_card(card)
        with dispatch.cond:
            released.append(row)
        verdict = "handed back to the box queue" if row["released"] else "kept (the queue refused)"
        print(f"[forecast-batch] card {card}: every slot done, {verdict}: {row['message']}", flush=True)

    def drive(worker):
        process = worker["process"]
        try:
            # Workers of one card start their first members apart, so the card
            # integrates one member while the next prepares instead of every
            # worker preparing at once and then integrating at once.
            time.sleep(stagger * worker["slot"])
            while True:
                index, token = dispatch.take(worker["card"])
                if index is None:
                    try:
                        process.stdin.write(json.dumps({"exit": True}) + "\n")
                        process.stdin.flush()
                    except OSError:
                        pass
                    try:
                        worker["returncode"] = process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        worker["returncode"] = process.wait()
                        worker["died"] = True
                    return
                member = members[index]
                gate_wait = _wait_for_card_memory(worker["visible"], min_free)
                sent = time.time()
                try:
                    process.stdin.write(json.dumps(member) + "\n")
                    process.stdin.flush()
                    line = process.stdout.readline()
                except OSError:
                    line = ""
                if not line:
                    rows[index] = {"run_id": member["run_id"], "returncode": 1, "card": worker["card"],
                                   "worker": worker["name"], "error": "worker exited during the member",
                                   "sent_unix": sent, "done_unix": time.time()}
                    _record_result(member, 1)
                    worker["died"] = True
                    dispatch.finished(worker["card"], token)
                    return
                row = json.loads(line)
                row.update(card=worker["card"], worker=worker["name"], sent_unix=sent, done_unix=time.time())
                row["card_memory_gate_seconds"] = gate_wait
                if row.get("memory_refused") and len(retries[index]) < ADMISSION_RETRIES:
                    # The door (admission) or the driver (its device-memory floor)
                    # refused before the member integrated anything: the card's
                    # co-resident members were near their peaks.  THE BREAKAGE THIS
                    # PREVENTS (a rented box, 2026-10-13, 6 slots on one card): 6 of 30 members
                    # failed on the driver floor (27.4 GB free required, 16.8-27.0 GB
                    # free) and were simply lost.  The member's partial destinations
                    # move aside and it goes back on the queue.
                    row["moved_aside"] = _move_destinations_aside(member, len(retries[index]) + 1)
                    retries[index].append(row)
                    dispatch.finished(worker["card"], token)
                    time.sleep(ADMISSION_RETRY_SECONDS)
                    dispatch.requeue(index, worker["card"], token)
                    continue
                row["memory_retries"] = retries[index]
                rows[index] = row
                worker["members"].append(member["run_id"])
                dispatch.finished(worker["card"], token, row.get("wall_seconds"))
        finally:
            dispatch.retire(worker["card"])
            slot_ended(worker)

    for card, visible in cards:
        for slot in range(per_card):
            name = f"card{card}.w{slot}"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible, GPU_MUTEX_CARDS=str(card),
                       WOOF_CARD=str(card), HEX_BATCH_WORKER=name)
            # Several members share this card: each hands its pool's unused
            # blocks back at macro boundaries (fullstep_lts_forecast_v841,
            # WOOF_HEX_RELEASE_POOL) instead of holding its transients' peak.
            env.setdefault("WOOF_HEX_RELEASE_POOL", "macro")
            stderr = (log_dir / f"{name}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(_worker_command(),
                                       env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=stderr, text=True, bufsize=1)
            workers.append({"name": name, "card": card, "visible": visible, "slot": slot, "process": process,
                            "stderr": stderr, "members": [], "died": False, "ended": False,
                            "started_unix": time.time()})
    threads = [threading.Thread(target=drive, args=(worker,), name=worker["name"]) for worker in workers]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        for worker in workers:
            try:
                worker["returncode"] = worker["process"].wait(timeout=120)
            except subprocess.TimeoutExpired:
                worker["process"].kill()
                worker["returncode"] = worker["process"].wait()
            worker["stderr"].close()
        for index, member in enumerate(members):
            if rows[index] is None:
                rows[index] = {"run_id": member["run_id"], "returncode": 1, "error": "never ran (every worker exited)"}
                _record_result(member, 1)
        receipt["workers"] = [{key: worker[key] for key in ("name", "card", "members", "died", "returncode",
                                                            "started_unix")} for worker in workers]
        receipt["dispatch"] = dispatch.summary()
        receipt["wall_seconds"] = time.time() - started
        receipt["status"] = "passed" if all(row["returncode"] == 0 for row in rows) else "failed"
        Path(receipt_path).write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    return 0 if receipt["status"] == "passed" else 1


def _record_result(member, returncode):
    path = Path(member["result"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"returncode": int(returncode), "batch": True}, indent=2) + "\n")


# ---------------------------------------------------------------- worker
class _PoolPeak:
    """The worker's device-pool high-water mark while one member runs."""

    def __init__(self, interval=0.25):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._thread = None

    def __enter__(self):
        import cupy as cp
        pool = cp.get_default_memory_pool()

        def sample():
            while not self._stop.wait(self.interval):
                self.peak = max(self.peak, int(pool.total_bytes()))

        self._thread = threading.Thread(target=sample, name="hex-batch-pool-peak", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        import cupy as cp
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, int(cp.get_default_memory_pool().total_bytes()))


def _run_member(member, cli):
    """In the member's own child process: the door command, output to the member's log."""

    log_path = Path(member["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    began = time.time()
    returncode, error, peak = 1, None, None
    try:
        command = ["forecast", *[str(item) for item in member["argv"]]]
        with _PoolPeak() as sampler:
            if os.environ.get("HEX_BATCH_PROFILE"):
                import cProfile
                profiler = cProfile.Profile()
                try:
                    returncode = int(profiler.runcall(cli.main, command))
                finally:
                    profiler.dump_stats(str(log_path) + ".prof")
            else:
                returncode = int(cli.main(command))
        peak = sampler.peak
    except SystemExit as stop:
        returncode = stop.code if isinstance(stop.code, int) else 1
        error = f"SystemExit: {stop.code}"
    except BaseException as failure:  # one member's failure is that member's
        traceback.print_exc()
        returncode, error = 1, f"{type(failure).__name__}: {failure}"
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    _record_result(member, returncode)
    refused = False
    if returncode != 0:
        # Refused for device memory before integrating anything: at the door's
        # admission (no destination yet) or at the driver's own floor.
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            refused = any((line.startswith("ADMISSION ") and "REFUSED" in line)
                          or "the driver's own device-memory floor refused this run" in line
                          # forecast_door.py also reports a floor refusal this way (P11-30m mem022, 2026-11-02)
                          or "the card's free device memory fell between the door's admission and the driver's floor" in line
                          for line in lines)
        except OSError:
            refused = False
    import resource
    return {"run_id": member["run_id"], "returncode": returncode, "error": error,
            "began_unix": began, "wall_seconds": time.time() - began,
            "device_pool_peak_bytes": peak, "pid": os.getpid(), "memory_refused": refused,
            "host_rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}


#: Modules a member imports anyway; the slot imports them once before forking.
#: None of them touches the CUDA runtime at import (a CUDA context does not
#: survive fork; each child creates its own).
_PRELOAD = ("numpy", "netCDF4", "cupy", "woof.hex.drivers.run_cuda_v841_forecast",
            "woof.hex.drivers.fullstep_lts_forecast_v841", "woof.hex.cuda_fullstep_lts_active",
            "woof.hex.cuda_driver", "woof.hex.cuda_arwen_physics_v841", "woof.hex.fullstep_lts_classes",
            "woof.hex.partition_local_mesh_v841", "woof.core.mpas_column_batch", "woof.core.physics")


def _member_in_child(job, cli):
    """Fork; the child runs the member and exits; the slot reads its row."""

    import atexit
    read_end, write_end = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        os.close(read_end)
        code = 0
        try:
            row = _run_member(job, cli)
        except BaseException as failure:
            row = {"run_id": job.get("run_id"), "returncode": 1, "error": f"{type(failure).__name__}: {failure}"}
            code = 1
        try:
            with os.fdopen(write_end, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(row))
        finally:
            try:
                # What a separate process runs at exit (kernel prewarm manifest), threading's hooks
                # included, so an idle compile worker is stopped rather than joined forever.
                from .cli import run_exit_handlers
                run_exit_handlers()
            finally:
                os._exit(code)
    os.close(write_end)
    with os.fdopen(read_end, "r", encoding="utf-8") as stream:
        text = stream.read()
    _, status = os.waitpid(pid, 0)
    if text:
        return json.loads(text)
    _record_result(job, 1)
    return {"run_id": job.get("run_id"), "returncode": 1, "pid": pid,
            "error": f"member process ended without a result (wait status {status})"}


def worker_main():
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)   # anything printed between members goes to the slot log
    import importlib
    from . import cli
    for name in _PRELOAD:
        try:
            importlib.import_module(name)
        except Exception as error:  # the child imports it again and reports its own failure
            print(f"[forecast-batch] preload {name}: {type(error).__name__}: {error}", flush=True)
    for line in sys.stdin:
        job = json.loads(line)
        if job.get("exit"):
            break
        row = _member_in_child(job, cli)
        protocol.write(json.dumps(row) + "\n")
        protocol.flush()
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--worker"]:
        return worker_main()
    if len(argv) != 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    return run_batch(argv[0], argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
