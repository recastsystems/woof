"""Bounded UUID process waves and a complete-roster DA forecast barrier.

The numerical worker is the same complete-tree leg used by the serial
controller. This module schedules processes only. It never analyses a
partial roster or removes a worker's input restart.
"""
from __future__ import annotations

import hashlib
import ast
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import zipfile

GIB = 1024**3
SCHEMA = "gpuwm-da.member-leg-result.v1"
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
HOST_RESERVE_BYTES = 48*GIB
DECODER_RAM_LIMIT_BYTES = 160*GIB
CLOCK_FIELDS = ("elapsed_ticks", "tick_den", "step_count", "elapsed_seconds",
                "domain_start_ticks", "domain_lifecycle", "dtbc_fp32_bits")


class MemberWaveFailed(RuntimeError):
    """No forecast barrier was published; input state remains reusable."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


#: Threads that hash a roster's artifacts at once.  hashlib and file reads
#: release the GIL, so a 32-member barrier hashes its restarts in parallel
#: instead of one file after another on one core.
HASH_THREADS = 32


def sha256_many(paths, *, threads=HASH_THREADS):
    """``{str(path): sha256}`` for every readable path, hashed concurrently.

    A path that cannot be read is left out; the caller's own check then
    reaches it and refuses it exactly as a serial check would.  Digests
    are the same bytes :func:`sha256` returns.
    """
    from concurrent.futures import ThreadPoolExecutor

    unique = list(dict.fromkeys(str(path) for path in paths))

    def one(path):
        try:
            return path, sha256(path)
        except OSError:
            return path, None

    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(threads), len(unique)))) as pool:
        return {path: digest for path, digest in pool.map(one, unique)
                if digest is not None}


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def visible_card_uuids(environ=None):
    """The physical GPU UUIDs CUDA_VISIBLE_DEVICES names, or None.

    None when the variable is unset, empty, or names cards by index (an
    index is relative to the enumeration order and is not an identity
    nvidia-smi can be asked about).  A mixed list is refused: it would
    silently drop the indexed cards.
    """
    value = (os.environ if environ is None else environ).get(
        "CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not entries:
        return None
    uuids = [entry for entry in entries if entry.startswith("GPU-")]
    if not uuids:
        return None
    if len(uuids) != len(entries):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES mixes GPU UUIDs with indices; name the "
            "packed forecast cards one way")
    return uuids


def query_cards(uuids=None):
    """Read physical card identities without opening a CUDA context.

    ``uuids`` None takes the cards CUDA_VISIBLE_DEVICES names, when it
    names them by UUID, and every card on the box otherwise.  The breakage
    this prevents: a cycle granted two cards of a shared eight-card box
    (a rented box, 2026-10-06 18:13Z and 18:20Z) priced its packed members
    against the TIGHTEST of all eight, a card another tenant held at
    27.7 GiB, and refused itself twice ("one packed member (12.7 GiB
    priced) does not fit the tightest card (0.9 GiB after the reserve)")
    while its own two cards were empty.
    """
    if uuids is None:
        uuids = visible_card_uuids()
    result = subprocess.run(
        [*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-gpu=uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15)
    cards = []
    for line in result.stdout.splitlines():
        uuid, total, free = [value.strip() for value in line.split(",")]
        cards.append({"uuid": uuid, "total_bytes": int(float(total)*1024**2),
                      "free_bytes": int(float(free)*1024**2)})
    if uuids is not None:
        found = {card["uuid"]: card for card in cards}
        if len(set(uuids)) != len(uuids) or any(uuid not in found for uuid in uuids):
            raise ValueError("forecast devices must name distinct full physical GPU UUIDs")
        cards = [found[uuid] for uuid in uuids]
    return cards


def _roster_clocks(jobs):
    if not jobs or [job.get("member") for job in jobs] != list(range(len(jobs))):
        raise ValueError("member jobs must preserve the complete ordered zero-based roster")
    for job in jobs:
        if type(job["member"]) is not int:
            raise ValueError("member IDs must be integers, not booleans or wave slots")
        if not isinstance(job.get("argv"), list) or not job["argv"] or any(
                not isinstance(word, str) or not word for word in job["argv"]):
            raise ValueError("member job requires a literal argument vector")
        if not re.fullmatch(r"[0-9a-f]{64}", job.get("request_hash", "")):
            raise ValueError("each member job must bind its immutable request hash")
        if any(not math.isfinite(job[key]) for key in ("t_start", "t_end")) or job["t_end"] <= job["t_start"]:
            raise ValueError("member job has invalid forecast clocks")
    if len({(job["t_start"], job["t_end"]) for job in jobs}) != 1:
        raise ValueError("all members must reach the same forecast analysis clock")
    declared_sizes = [job.get("ensemble_members") for job in jobs]
    if any(size is not None for size in declared_sizes) and (any(type(size) is not int for size in declared_sizes) or set(declared_sizes)!={len(jobs)}):
        raise ValueError("forecast jobs are missing part of the fixed ensemble roster")


def available_host_memory_bytes():
    from woof.ingest.preparation_workers import host_available_bytes
    return host_available_bytes()


def plan_wave(jobs, devices, members_per_card, *, member_peak_bytes=0,
              vram_reserve_bytes=4*GIB, decoder_budget_bytes=None,
              host_available_bytes=None, host_reserve_bytes=HOST_RESERVE_BYTES,
              check_roster=True):
    if check_roster:
        _roster_clocks(jobs)
    identities = [card if isinstance(card, str) else card["uuid"] for card in devices]
    if not identities or len(set(identities)) != len(identities) or any(not UUID.fullmatch(uuid) for uuid in identities):
        raise ValueError("packing needs distinct full physical GPU UUIDs")
    if members_per_card not in (1, 2, 4):
        raise ValueError("member packing is 1, 2 or 4 processes per card")
    if type(member_peak_bytes) is not int or member_peak_bytes < 0:
        raise ValueError("member VRAM envelope must be a nonnegative integer byte count")
    if member_peak_bytes:
        for card in devices:
            if not isinstance(card, dict):
                raise ValueError("a VRAM admission requires measured card capacities")
            available = min(card["free_bytes"], card["total_bytes"])-vram_reserve_bytes
            if members_per_card*member_peak_bytes > available:
                raise ValueError("selected DA member pack exceeds actual card capacity minus reserve")
    width = len(identities)*members_per_card
    concurrent = min(width, len(jobs))
    available = available_host_memory_bytes() if host_available_bytes is None else host_available_bytes
    if type(available) is not int or available <= 0:
        raise ValueError("available host RAM could not be measured before member launch")
    if type(host_reserve_bytes) is not int or host_reserve_bytes < 0:
        raise ValueError("host RAM reserve must be a nonnegative integer byte count")
    if len(jobs)>4 and host_reserve_bytes<HOST_RESERVE_BYTES:
        raise ValueError("production ensemble waves must reserve at least 48 GiB of host RAM")
    limit = min(DECODER_RAM_LIMIT_BYTES, max(0, available-host_reserve_bytes))
    budget = min(5*GIB, limit//concurrent) if decoder_budget_bytes is None else decoder_budget_bytes
    if type(budget) is not int or budget <= 0 or concurrent*budget > limit:
        raise ValueError("concurrent decoder budgets exceed available host RAM after reserve or the 160 GiB bound")
    waves = []
    for offset in range(0, len(jobs), width):
        waves.append([dict(job, gpu_uuid=identities[slot % len(identities)],
                           card_slot=slot//len(identities))
                      for slot, job in enumerate(jobs[offset:offset+width])])
    return {"members": len(jobs), "members_per_card": members_per_card,
            "device_uuids": identities, "mps_required": members_per_card == 4,
            "decoder_threads": 12, "decoder_budget_bytes": budget,
            "decoder_ram_bound_bytes": concurrent*budget,
            "host_available_bytes": available, "host_reserve_bytes": host_reserve_bytes,
            "member_peak_bytes": member_peak_bytes, "waves": waves}


def _plain_array_archive(path):
    """Reject pickle/object payloads from NPY headers without allocating fields."""
    import numpy as np
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names or len(names)!=len(set(names)) or any(not name.endswith(".npy") for name in names):
                raise ValueError("array payload is not one plain NPZ inventory")
            for name in names:
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        _, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        _, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    elif version == (3, 0):
                        # NPY v3 has the v2 length prefix with UTF-8 headers.
                        # NumPy 2.4 exposes no version-taking header reader.
                        prefix = stream.read(4)
                        if len(prefix) != 4:
                            raise ValueError("truncated NPY v3 header length")
                        length = struct.unpack("<I", prefix)[0]
                        if length > 10_000:
                            raise ValueError("NPY v3 header exceeds the plain payload limit")
                        encoded = stream.read(length)
                        if len(encoded) != length:
                            raise ValueError("truncated NPY v3 header")
                        header = ast.literal_eval(encoded.decode("utf-8"))
                        if not isinstance(header, dict) or set(header)!={"descr", "fortran_order", "shape"}:
                            raise ValueError("invalid NPY v3 header fields")
                        if type(header["fortran_order"]) is not bool or not isinstance(header["shape"], tuple) or any(type(n) is not int or n<0 for n in header["shape"]):
                            raise ValueError("invalid NPY v3 shape or storage order")
                        dtype = np.dtype(header["descr"])
                    else:
                        raise ValueError("unsupported NPY payload header version")
                    if dtype.hasobject:
                        raise ValueError("array payload refuses object arrays and pickle content")
    except (OSError, zipfile.BadZipFile, ValueError, SyntaxError, TypeError, UnicodeError) as error:
        raise MemberWaveFailed(f"plain array payload refused: {error}") from error


def _declared_artifacts(jobs):
    """Every artifact path the roster's result receipts declare, resolved.

    Only a hashing prepass: :func:`validate_results` still checks each
    path's ownership, size and digest itself, in roster order.
    """
    paths = []
    for job in jobs:
        path = Path(job["result_path"])
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            for item in result.get("files") or ():
                artifact = Path(item["path"])
                if not artifact.is_absolute():
                    artifact = path.parent/artifact
                paths.append(artifact.resolve())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return paths


def validate_results(jobs):
    """Verify every result before handing any trajectory to the analysis."""
    try:
        _roster_clocks(jobs)
    except (ValueError, KeyError, TypeError) as error:
        raise MemberWaveFailed(f"incomplete forecast roster or clocks: {error}") from error
    results = []
    roster_artifacts = set()
    digests = sha256_many(_declared_artifacts(jobs))
    for job in jobs:
        results.append(_validate_result(job, digests, roster_artifacts))
    return results


def validate_job(job):
    """Verify one job outside a member roster: the unanalysed control.

    The same checks :func:`validate_results` makes of each member (request
    hash, clocks, owned and hashed inventory, plain arrays, native restart
    identity), without the roster's own (a complete zero-based member
    list), which the control is not part of.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", job.get("request_hash", "")):
        raise MemberWaveFailed("the job must bind its immutable request hash")
    if any(not math.isfinite(job[key]) for key in ("t_start", "t_end")) \
            or job["t_end"] <= job["t_start"]:
        raise MemberWaveFailed("the job has invalid forecast clocks")
    return _validate_result(job, sha256_many(_declared_artifacts([job])),
                            set())


def _validate_result(job, digests, roster_artifacts):
    """One job's result, checked; ``roster_artifacts`` gains its files."""
    if True:
        path = Path(job["result_path"])
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MemberWaveFailed(f"member {job['member']}: result is missing or unreadable") from error
        if result.get("schema") != SCHEMA or result.get("complete") is not True:
            raise MemberWaveFailed(f"member {job['member']}: unfinished forecast result")
        if type(result.get("member")) is not type(job["member"]) \
                or result["member"] != job["member"]:
            raise MemberWaveFailed("member barrier has a missing, reordered or substituted identity")
        if result.get("request_hash") != job["request_hash"]:
            raise MemberWaveFailed(f"member {job['member']}: forecast request changed")
        if any(result.get(key) != job[key] for key in ("t_start", "t_end")):
            raise MemberWaveFailed(f"member {job['member']}: forecast clock changed")
        output_root = Path(result["output_root"]).resolve()
        expected_root = Path(job.get("output_root", path.parent)).resolve()
        if output_root != expected_root:
            raise MemberWaveFailed(f"member {job['member']}: output ownership root changed")
        if path.is_symlink() or not path.resolve().is_relative_to(output_root):
            raise MemberWaveFailed(f"member {job['member']}: result receipt is outside its owned root")
        files = result.get("files")
        if not isinstance(files, list) or not files:
            raise MemberWaveFailed(f"member {job['member']}: no complete restart and array inventory")
        seen = set()
        for item in files:
            artifact = Path(item["path"])
            if not artifact.is_absolute():
                artifact = path.parent/artifact
            resolved = artifact.resolve()
            if not resolved.is_relative_to(output_root) or artifact.is_symlink() or str(resolved) in seen:
                raise MemberWaveFailed(f"member {job['member']}: unowned or duplicated output artifact")
            if str(resolved) in roster_artifacts:
                raise MemberWaveFailed(f"member {job['member']}: output artifact is substituted from another member")
            seen.add(str(resolved))
            if not resolved.is_file() or resolved.stat().st_size != item["bytes"] or (
                    digests.get(str(resolved)) or sha256(resolved)) != item["sha256"]:
                raise MemberWaveFailed(f"member {job['member']}: output artifact changed or is missing: {artifact}")
        restart = Path(result["restart"]).resolve()
        if str(restart) not in seen:
            raise MemberWaveFailed(f"member {job['member']}: tree restart root missing from the output inventory")
        payload = result.get("arrays")
        if isinstance(payload, dict) and "path" in payload:
            payload_path = Path(payload["path"]).resolve()
            inventory = next((item for item in files if Path(item["path"]).resolve()==payload_path), None)
            if inventory != payload:
                raise MemberWaveFailed(f"member {job['member']}: array inventory record changed")
        elif result.get("arrays_path") is not None:
            # The independently owned GPU fixture publishes array statistics
            # separately, while its file inventory owns the actual NPZ bytes.
            payload_path = Path(result["arrays_path"]).resolve()
        else:
            raise MemberWaveFailed(f"member {job['member']}: typed array payload is missing")
        if str(payload_path) not in seen:
            raise MemberWaveFailed(f"member {job['member']}: array payload is outside the owned inventory")
        _plain_array_archive(payload_path)
        from woof.io.restart import tree_restart_members, read_restart_header, RESTART_FORMAT_VERSION
        try:
            members = tree_restart_members(restart)
            if "expected_domain_ids" in job and sorted(members) != sorted(job["expected_domain_ids"]):
                raise MemberWaveFailed(f"member {job['member']}: restart tree membership changed")
            declared = result.get("clocks")
            if declared is None and isinstance(result.get("checkpoint"), dict):
                declared = {gid: item["clock"] for gid,item in result["checkpoint"].items()}
            if not isinstance(declared, dict) or set(declared)!={str(gid) for gid in members}:
                raise MemberWaveFailed(f"member {job['member']}: complete tree clocks are missing")
            for gid, member in members.items():
                if str(member.resolve()) not in seen:
                    raise MemberWaveFailed(f"member {job['member']}: a child restart is missing from the inventory")
                header = read_restart_header(member)
                if header.get("format_version") != RESTART_FORMAT_VERSION:
                    raise MemberWaveFailed(f"member {job['member']}: restart uses a retired complete-tree format")
                components = header.get("experiment_fingerprint_components")
                if not isinstance(components, dict) or components.get("trajectory")!=str(job["member"]):
                    raise MemberWaveFailed(f"member {job['member']}: native restart trajectory identity changed")
                if job.get("expected_trajectory_fingerprint") is not None and header.get("experiment_fingerprint")!=job["expected_trajectory_fingerprint"]:
                    raise MemberWaveFailed(f"member {job['member']}: native restart ensemble fingerprint changed")
                ticks,den = header.get("elapsed_ticks"),header.get("tick_den")
                if type(ticks) is not int or type(den) is not int or den<=0 or ticks/den!=job["t_end"] or header.get("elapsed_seconds")!=job["t_end"]:
                    raise MemberWaveFailed(f"member {job['member']}: native tree restart clock changed")
                required = {"elapsed_ticks", "tick_den", "elapsed_seconds", "domain_start_ticks", "domain_lifecycle"}
                if not required.issubset(declared[str(gid)]):
                    raise MemberWaveFailed(f"member {job['member']}: native tree clock inventory is incomplete")
                if any(value!=header.get(key) for key,value in declared[str(gid)].items() if key in CLOCK_FIELDS):
                    raise MemberWaveFailed(f"member {job['member']}: declared and native tree restart clocks differ")
        except MemberWaveFailed:
            raise
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            raise MemberWaveFailed(f"member {job['member']}: complete tree restart is unreadable: {error}") from error
        roster_artifacts.update(seen)
        return result


def mps_available() -> bool:
    """Whether an owned MPS controller answers on this host now: the pipe
    directory named by CUDA_MPS_PIPE_DIRECTORY exists and the controller
    lists its servers.  The same test :func:`_require_mps` refuses on."""
    try:
        _require_mps()
    except (MemberWaveFailed, OSError, subprocess.SubprocessError):
        return False
    return True


def _require_mps():
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if not pipe or not Path(pipe).is_dir():
        raise MemberWaveFailed("four DA members per card require an active owned MPS controller")
    subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n", encoding="utf-8",  # text-pipe-ok: one command, LF only
                   capture_output=True, check=True, timeout=10)


#: Linux launcher that binds a member worker's life to its controller.
#:
#: Breakage it prevents: a packed DA controller killed during a leg left its
#: member forecast workers running as orphans (ppid 1) on every forecast
#: card, and the next job queued on cards the GPU mutex reported free.
#:
#: It runs in the forked child before the real worker and replaces itself
#: with it (same pid, same new session), so nothing else changes: the
#: coordinator still owns and signals the worker's process group, and a
#: terminal Ctrl-C still does not reach a worker mid-write.  PR_SET_PDEATHSIG
#: survives the exec.  SIGKILL is the death signal because nothing can use a
#: worker's output once its controller is gone (the barrier lives in the
#: controller and a worker never removes its inputs), and a handler the
#: forecast code installs cannot swallow it.  The parent check after prctl
#: closes the race where the controller died between fork and prctl.
_PARENT_DEATH_LAUNCHER = (
    "import ctypes, os, signal, sys\n"
    "parent = int(sys.argv[1])\n"
    "if ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:\n"
    "    sys.exit('DA member worker refused: PR_SET_PDEATHSIG failed (errno %d); '\n"
    "             'it could outlive its controller and hold its GPU' % ctypes.get_errno())\n"
    "if os.getppid() != parent:\n"
    "    sys.exit('DA member worker not started: its controller already exited')\n"
    "try:\n"
    "    os.execvp(sys.argv[2], sys.argv[2:])\n"
    "except OSError as error:\n"
    "    sys.exit('DA member worker could not start %r: %s' % (sys.argv[2], error))\n")


def _launch_argv(argv):
    """The argument vector that starts one member worker.

    On Linux the worker is started through :data:`_PARENT_DEATH_LAUNCHER`,
    so a controller that dies by SIGKILL, a crash or any signal it does not
    trap takes its workers with it.  PR_SET_PDEATHSIG fires when the parent
    THREAD that forked the child exits, not the process.  The forking thread
    here is whichever thread runs :func:`run_wave` (the main thread in
    ``tools.da_cycle_prepared``), and run_wave never returns or raises while
    a worker it started is alive: its ``finally`` stops every active worker
    first.  So the forking thread outlives every worker unless the whole
    controller dies, which is exactly when the workers must die.

    Other platforms have no prctl and no parent-death signal; the vector is
    returned unchanged there.  Packed member waves run on Linux GPU hosts,
    and the coordinator's own signal and exception paths still stop owned
    workers on every platform.
    """
    argv = list(argv)
    if not sys.platform.startswith("linux"):
        return argv
    return [sys.executable, "-I", "-S", "-c", _PARENT_DEATH_LAUNCHER, str(os.getpid()), *argv]


class ControllerSignalled(BaseException):
    """The controller received a termination signal while it owned workers."""

    def __init__(self, signum):
        super().__init__(f"DA controller received signal {signum}; owned member workers stopped")
        self.signum = signum


#: Signals whose default action kills the controller without unwinding, so
#: run_wave's ``finally`` would never stop the workers it started.  SIGINT
#: is not listed: its Python default already raises KeyboardInterrupt and
#: unwinds through that ``finally``.
_TRAPPED_SIGNALS = ("SIGTERM", "SIGHUP")


def _trap_termination(state, previous):
    """While workers are owned, turn SIGTERM/SIGHUP into an unwind.

    Only signals still at their default disposition are trapped; a handler
    the controller installed itself is left alone, and an ignored signal
    does not kill the controller.  Python allows handlers only on the main
    thread; from another thread the parent-death launcher is the guard.
    """
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return

    def handler(signum, unused_frame):
        if state.get("signum") is None:
            state["signum"] = signum
            if not state.get("cleaning"):
                raise ControllerSignalled(signum)

    for name in _TRAPPED_SIGNALS:
        number = getattr(signal, name, None)
        if number is not None and signal.getsignal(number) is signal.SIG_DFL:
            previous[number] = signal.signal(number, handler)


def _release_trap(state, previous):
    """Restore the controller's dispositions and re-deliver a trapped signal.

    The re-delivered signal meets its default action, so the controller
    still ends as killed by that signal, as it did before workers were
    owned, but only after every owned worker was stopped and the leg
    receipt written.
    """
    for number, handler in previous.items():
        signal.signal(number, handler)
    previous.clear()
    if state.get("signum") is not None:
        signal.raise_signal(state["signum"])


def _stop(process):
    if isinstance(process, ServerJob):
        process = process.server.process
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def _member_environment(gpu_uuid, plan):
    environment = os.environ.copy()
    environment.update(__import__("woof.gpu_smi",fromlist=["visibility_environment"]).visibility_environment(gpu_uuid))
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID",
        GPUWM_MAPPED_ENGINE_THREADS="12", GPUWM_MAPPED_ENGINE_MEMORY_BUDGET_BYTES=str(plan["decoder_budget_bytes"]),
        WOOF_PREPROCESS_THREADS="12", RAYON_NUM_THREADS="12",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    return environment


class _LogHandle:
    """The per-job log of a card-server job: a name the refusal reads."""

    def __init__(self, path):
        self.name = str(path)
        self.closed = False

    def close(self):
        self.closed = True


class ServerJob:
    """A member job running in a :class:`CardServer`, seen as a process.

    ``poll`` answers from the job's ``.done`` marker (or the server's death);
    ``pid`` and ``wait`` are the server's, so stopping a job stops its
    card's server -- the state of a server whose job was abandoned is not
    trusted with another member.
    """

    def __init__(self, server, index, done_path):
        self.server = server
        self.index = index
        self.done_path = Path(done_path)
        self.returncode = None
        self.pid = server.process.pid

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.done_path.exists():
            try:
                done = json.loads(self.done_path.read_text(encoding="utf-8"))
                self.returncode = int(done.get("rc", 1))
            except (OSError, ValueError):
                return None
            return self.returncode
        code = self.server.process.poll()
        if code is not None:
            if self.done_path.exists():
                return self.poll()
            self.returncode = code if code else 1
        return self.returncode

    def wait(self, timeout=None):
        return self.server.process.wait(timeout=timeout)


class CardServer:
    """One long-lived member worker per card for a whole packed run.

    Breakage it removes: on a rented box's 9 km CONUS run every member was a new
    process that imported the engine, opened a CUDA context, loaded its
    kernels and re-read and re-hashed the whole 4 GB prepared cache before
    its first step, about 40 s of host work around 25 s of stepping, with
    the card idle meanwhile (26 percent of card-time busy over a 3-cycle
    run).  The server pays that once; each job is the same complete-tree
    member leg the serial route runs in one process
    (:func:`woof.da.member_transport.serve`).
    """

    def __init__(self, gpu_uuid, directory, *, decoder_budget_bytes=5*GIB,
                 python=None, argv=None):
        self.gpu_uuid = str(gpu_uuid)
        self.queue = Path(directory)
        self.queue.mkdir(parents=True, exist_ok=False)
        self.index = 0
        self.outstanding = []
        self.log = (self.queue / "server.log").open("xb")
        argv = ([python or sys.executable, "-m", "woof.da.member_transport",
                 "--serve", str(self.queue)] if argv is None
                else [*argv, str(self.queue)])
        try:
            self.process = subprocess.Popen(
                _launch_argv(argv),
                env=_member_environment(self.gpu_uuid, {
                    "decoder_budget_bytes": decoder_budget_bytes}),
                stdin=subprocess.DEVNULL, stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log.close()
            raise

    def submit(self, job, log_path):
        """Queue one sealed member job; returns its :class:`ServerJob`."""
        if self.process.poll() is not None:
            raise MemberWaveFailed(
                f"the card server on {self.gpu_uuid} has exited "
                f"(code {self.process.returncode})"
                + _log_tail(self.queue / "server.log"))
        argv = job["argv"]
        request = argv[argv.index("--request") + 1]
        digest = argv[argv.index("--request-sha256") + 1]
        result = argv[argv.index("--result") + 1]
        index = self.index
        self.index += 1
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).touch(exist_ok=False)
        atomic_json(self.queue / f"job-{index:05d}.json", {
            "schema": "gpuwm-da.card-server-job.v1", "request": request,
            "request_sha256": digest, "result": result,
            "log": str(Path(log_path).resolve())})
        job = ServerJob(self, index, self.queue / f"job-{index:05d}.done")
        self.outstanding.append(job)
        return job

    def load(self):
        """Jobs queued or running in this server."""
        self.outstanding = [job for job in self.outstanding
                            if job.poll() is None]
        return len(self.outstanding)

    def idle(self):
        """True when the server has no job queued or running."""
        return self.process.poll() is None and self.load() == 0

    def close(self, timeout_seconds=60.0):
        """Ask the server to stop after its queue; stop it if it lingers."""
        try:
            (self.queue / "stop").touch()
            self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _stop(self.process)
        finally:
            if not self.log.closed:
                self.log.close()


#: Jobs a card server holds at once: the one it runs and the next, whose
#: request it reads and proves while the first steps
#: (woof.da.member_transport.serve).
SERVER_QUEUE_DEPTH = 2


class CardServerPool:
    """The card servers of one card: one per member slot on that card.

    Breakage it prevents: with one server per card, a card packed with two
    or four members (``--forecast-members-per-card``) queued them all on
    that one process, so the slot scheduler counted them as running side by
    side while they ran one after another -- packing silently undone, and
    the control's job (which occupies one slot) held every member queued
    behind it.  ``submit`` hands each job to a server with nothing running.
    """

    def __init__(self, servers):
        self.servers = list(servers)

    def submit(self, job, log_path):
        live = [server for server in self.servers
                if server.process.poll() is None]
        if live:
            server = min(live, key=lambda s: s.load())
            if server.load() < SERVER_QUEUE_DEPTH:
                return server.submit(job, log_path)
        dead = [s for s in self.servers if s.process.poll() is not None]
        if dead:
            raise MemberWaveFailed(
                f"the card server on {dead[0].gpu_uuid} has exited "
                f"(code {dead[0].process.returncode})"
                + _log_tail(dead[0].queue / "server.log"))
        raise MemberWaveFailed(
            "every card server on this card is busy; the slot scheduler "
            "gave a card more jobs than it has member slots")

    def close(self, timeout_seconds=60.0):
        for server in self.servers:
            server.close(timeout_seconds=timeout_seconds)


def start_card_servers(devices, directory, *, per_card=1, **kwargs):
    """``{uuid: CardServerPool}``: ``per_card`` servers on every forecast card."""
    if int(per_card) < 1:
        raise ValueError("a card needs at least one server")
    started = []
    servers = {}
    try:
        for card in devices:
            uuid = card if isinstance(card, str) else card["uuid"]
            pool = []
            for slot in range(int(per_card)):
                pool.append(CardServer(
                    uuid, Path(directory) / uuid / f"slot-{slot}", **kwargs))
                started.append(pool[-1])
            servers[uuid] = CardServerPool(pool)
    except BaseException:
        for server in started:
            _stop(server.process)
        raise
    return servers


class OwnedJob:
    """One worker process launched outside a member roster, on one card.

    The unanalysed control runs this way: its leg depends only on its own
    restart, so the controller starts it as soon as the previous leg's
    barrier passes, on a card the analysis leaves free, and the next
    roster takes the card back when it ends (``occupied`` of
    :func:`run_wave`).
    """

    def __init__(self, job, gpu_uuid, log_path, *, decoder_budget_bytes=5*GIB,
                 server=None):
        if Path(job["result_path"]).exists():
            raise MemberWaveFailed("fresh result path required; a stale job cannot clear the barrier")
        self.job = dict(job, gpu_uuid=gpu_uuid)
        self.gpu_uuid = gpu_uuid
        self.started = time.monotonic()
        self.ended = None
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if server is not None:
            # Queued on the card's server: it runs before any member the
            # next roster queues there, which is what occupying means.
            self.process = server.submit(job, log_path)
            self.log = _LogHandle(log_path)
            return
        self.log = log_path.open("xb")
        try:
            # Through the parent-death launcher too.  The forking thread
            # is the controller's background thread (its pool lives until
            # the run ends) or the main thread; the control is stopped or
            # waited for before either exits.
            self.process = subprocess.Popen(
                _launch_argv(job["argv"]), cwd=job.get("cwd"),
                env=_member_environment(gpu_uuid, {"decoder_budget_bytes": decoder_budget_bytes}),
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                start_new_session=True)
        except BaseException:
            self.log.close()
            raise

    def poll(self):
        code = self.process.poll()
        if code is not None and self.ended is None:
            self.ended = time.monotonic()
            self.log.close()
        return code

    def wait(self, timeout_seconds):
        """Wait for the job; stop it and raise on failure or timeout."""
        deadline = time.monotonic() + float(timeout_seconds)
        while self.poll() is None:
            if time.monotonic() > deadline:
                self.stop()
                raise MemberWaveFailed("the owned job timed out")
            time.sleep(0.05)
        if self.process.returncode != 0:
            raise MemberWaveFailed(
                f"the owned job failed with exit code {self.process.returncode}"
                + _log_tail(self.log.name))
        return validate_job(self.job)

    def stop(self):
        _stop(self.process)
        self.poll()
        if not self.log.closed:
            self.log.close()

    def record(self):
        return {"member": self.job["member"], "gpu_uuid": self.gpu_uuid,
                "exit_code": self.process.returncode, "argv": self.job["argv"],
                "wall_seconds": (None if self.ended is None
                                 else round(self.ended - self.started, 3))}


def _log_tail(path, lines: int = 40) -> str:
    """The end of a failed job's log, carried in the refusal: a stopped run
    clears its packed job directories (and the log with them), so the
    message is the only place the worker's own error survives."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return "\n(log %s unreadable: %s)" % (path, error)
    tail = "\n".join(text.splitlines()[-lines:])
    return "\n--- last %d lines of %s ---\n%s" % (lines, path, tail)


def run_wave(jobs, devices, members_per_card, output, timeout_seconds,
             *, member_peak_bytes=0, decoder_budget_bytes=None, check_mps=True,
             host_available_bytes=None, host_reserve_bytes=HOST_RESERVE_BYTES,
             occupied=None, servers=None):
    """Launch every member as a card slot frees, stop owned failed workers,
    then publish one barrier.

    A card holds ``members_per_card`` workers; the next member starts the
    moment one of them ends, rather than when the slowest worker of a whole
    wave does.  ``jobs`` may hold futures of job dicts (requests sealed
    while earlier members already run); each is resolved when its turn to
    launch comes.  ``occupied`` maps a card's UUID to an :class:`OwnedJob`
    already running there (the control): that slot is the job's until it
    ends.  ``check_mps=False`` is a CPU scheduler-test seam only.
    Production calls and the GPU proof both leave the check enabled. Jobs
    own numerical result publication; this coordinator owns the
    complete-roster receipt.

    Workers never outlive the coordinator.  An exception or SIGINT unwinds
    through the ``finally`` below, which stops every owned worker's process
    group (TERM, then KILL after a short grace).  SIGTERM and SIGHUP are
    trapped while workers are owned and take the same path before the
    signal is re-delivered.  SIGKILL or a hard crash is covered on Linux by
    the parent-death launcher (:func:`_launch_argv`).  Only processes this
    call started are signalled, plus an occupying job when the barrier
    fails.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("forecast leg timeout must be finite and positive")
    state, previous = {}, {}
    try:
        _trap_termination(state, previous)
        return _run_wave(jobs, devices, members_per_card, output, timeout_seconds,
                         state, member_peak_bytes=member_peak_bytes,
                         decoder_budget_bytes=decoder_budget_bytes, check_mps=check_mps,
                         host_available_bytes=host_available_bytes,
                         host_reserve_bytes=host_reserve_bytes, occupied=occupied,
                         servers=servers)
    finally:
        _release_trap(state, previous)


def _run_wave(jobs, devices, members_per_card, output, timeout_seconds, state,
              *, member_peak_bytes, decoder_budget_bytes, check_mps,
              host_available_bytes, host_reserve_bytes, occupied, servers=None):
    sealed = all(isinstance(job, dict) for job in jobs)
    plan = plan_wave(jobs if sealed else [
        {"member": n} for n in range(len(jobs))], devices, members_per_card,
                     member_peak_bytes=member_peak_bytes,
                     decoder_budget_bytes=decoder_budget_bytes,
                     host_available_bytes=host_available_bytes,
                     host_reserve_bytes=host_reserve_bytes,
                     check_roster=sealed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"plan.json", plan)
    start = time.monotonic()
    occupied = dict(occupied or {})
    identities = list(plan["device_uuids"])
    # With card servers each slot also holds the NEXT member's job, so the
    # server reads and proves its request while the current one steps
    # (SERVER_QUEUE_DEPTH).  An occupying job (the control) takes its slot
    # and that slot's queue place, so no member waits behind it.
    depth = SERVER_QUEUE_DEPTH if servers else 1
    free = {uuid: members_per_card*depth for uuid in identities}
    for uuid in occupied:
        if uuid not in free:
            raise ValueError("an occupying job is not on one of this roster's cards")
        free[uuid] -= depth
    active, records, failure, result = [], [], None, None
    resolved = [job if isinstance(job, dict) else None for job in jobs]
    launched = 0
    width = len(identities)*members_per_card
    try:
        if plan["mps_required"] and check_mps:
            _require_mps()
        while launched < len(jobs) or active or occupied:
            if time.monotonic()-start > timeout_seconds:
                raise MemberWaveFailed("member forecast timed out before a complete analysis barrier")
            for job, process, log, uuid, began in list(active):
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    bad = [(job["member"], code)]
                    log.close()
                    raise MemberWaveFailed(f"member forecast failed; analysis barrier retained: {bad}"
                                           + _log_tail(log.name))
                log.close()
                active.remove((job, process, log, uuid, began))
                free[uuid] += 1
                records.append({"member": job["member"], "gpu_uuid": uuid,
                                "wave": job["launch"] // width,
                                "launch": job["launch"], "exit_code": code,
                                "argv": job["argv"],
                                "launched_seconds": round(began - start, 3),
                                "wall_seconds": round(time.monotonic() - began, 3)})
            for uuid, owned in list(occupied.items()):
                code = owned.poll()
                if code is None:
                    continue
                if code != 0:
                    raise MemberWaveFailed(
                        f"the job occupying {uuid} failed with exit code {code}; "
                        "analysis barrier retained" + _log_tail(owned.log.name))
                del occupied[uuid]
                free[uuid] += depth
            while launched < len(jobs):
                uuid = max(identities, key=lambda u: (free[u], -identities.index(u)))
                if free[uuid] <= 0:
                    break
                job = resolved[launched]
                if job is None:
                    source = jobs[launched]
                    if not source.done():
                        break
                    job = resolved[launched] = source.result()
                    if job.get("member") != launched:
                        raise MemberWaveFailed("member jobs must preserve the complete ordered zero-based roster")
                available_now = available_host_memory_bytes() if host_available_bytes is None else host_available_bytes
                required_now = plan["decoder_budget_bytes"]+host_reserve_bytes
                if type(available_now) is not int or available_now < required_now:
                    raise MemberWaveFailed("available host RAM fell below this member wave's decoder budgets plus reserve")
                if Path(job["result_path"]).exists():
                    raise MemberWaveFailed("fresh result path required; a stale member cannot clear the barrier")
                log_path = output/f"member-{job['member']:03d}.log"
                if servers and uuid in servers:
                    process = servers[uuid].submit(job, log_path)
                    log = _LogHandle(log_path)
                else:
                    log = log_path.open("xb")
                    try:
                        process = subprocess.Popen(_launch_argv(job["argv"]), cwd=job.get("cwd"),
                            env=_member_environment(uuid, plan),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                active.append((dict(job, gpu_uuid=uuid, launch=launched),
                               process, log, uuid, time.monotonic()))
                free[uuid] -= 1
                launched += 1
            time.sleep(0.05)
        records.sort(key=lambda row: row["member"])
        result = validate_results(resolved)
        if time.monotonic()-start > timeout_seconds:
            result = None
            raise MemberWaveFailed("member forecast and output validation exceeded the leg timeout")
        return result
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        # A second termination signal during cleanup is recorded, not
        # raised, so it cannot abandon the remaining workers half-stopped.
        state["cleaning"] = True
        stop_errors = []
        for job, process, log, _uuid, _began in active:
            try:
                _stop(process)
            except Exception as error:  # keep stopping the rest of the roster
                stop_errors.append(f"member {job['member']} pid {process.pid}: "
                                   f"{type(error).__name__}: {error}")
            finally:
                log.close()
        for owned in occupied.values():
            if failure is not None:
                try:
                    owned.stop()
                except Exception as error:
                    stop_errors.append(f"occupying job {owned.job['member']}: "
                                       f"{type(error).__name__}: {error}")
        atomic_json(output/"forecast-leg-receipt.json", {
            "schema": "gpuwm-da.member-wave.v1", "complete": result is not None,
            "analysis_permitted": result is not None, "failure": failure,
            "stop_errors": stop_errors,
            "forecast_wall_seconds_including_io": time.monotonic()-start,
            "analysis_wall_seconds": None, "members": len(jobs), "records": records,
            "scheduling": ("per-card slots: a member starts when a slot on any "
                           "card frees, not at a wave boundary"),
            "inputs_deleted": False, "plan": plan})
