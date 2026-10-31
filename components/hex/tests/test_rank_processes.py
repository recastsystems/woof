"""The rank-process links and communicator (woof.hex.rank_processes), on the CPU.

THE BREAKAGE THE MODULE PREVENTS: same-host rank threads shared one interpreter, so a plain install on a Python with
the GIL ran more cards slower (8x RTX 5090, 2026-10-28: 4 cards 0.70 s/step against 2 cards 0.51).  These check the
pieces the process executor stands on: values and large arrays arrive intact and in rank order, and a rank that
stops makes its peers raise instead of waiting forever.
"""
from __future__ import annotations

import socket
import sys
import threading

import numpy as np
import pytest

from woof.hex import rank_processes as rp

pytestmark = pytest.mark.skipif(not rp.supported()[0], reason="needs Linux memfd and fd passing")


def _world(size, timeout=20.0):
    links = {rank: {} for rank in range(size)}
    for peer in range(1, size):
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        links[0][peer] = rp.Link(a, timeout=timeout, name=f"0-{peer}")
        links[peer][0] = rp.Link(b, timeout=timeout, name=f"{peer}-0")
    return [rp.ProcessComm(rank, size, links[rank]) for rank in range(size)]


def _run(comms, operation):
    results, errors = [None] * len(comms), []

    def one(rank):
        try:
            results[rank] = operation(comms[rank])
        except BaseException as error:  # noqa: BLE001 - collected for the assertion
            errors.append((rank, error))

    threads = [threading.Thread(target=one, args=(rank,)) for rank in range(len(comms))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    return results, errors


def test_collectives_keep_rank_order():
    comms = _world(4)

    def operation(comm):
        rank = comm.Get_rank()
        comm.Barrier()
        return dict(
            allgather=comm.allgather(rank * 10),
            gather=comm.gather(rank, root=0),
            gather2=comm.gather(rank, root=2),
            bcast=comm.bcast("zero" if rank == 0 else None, root=0),
            bcast3=comm.bcast("three" if rank == 3 else None, root=3),
            scatter=comm.scatter([f"to{peer}" for peer in range(4)] if rank == 0 else None, root=0),
            alltoall=comm.alltoall([(rank, peer) for peer in range(4)]),
        )

    results, errors = _run(comms, operation)
    assert not errors
    for rank, result in enumerate(results):
        assert result["allgather"] == [0, 10, 20, 30]
        assert result["gather"] == ([0, 1, 2, 3] if rank == 0 else None)
        assert result["gather2"] == ([0, 1, 2, 3] if rank == 2 else None)
        assert result["bcast"] == "zero" and result["bcast3"] == "three"
        assert result["scatter"] == f"to{rank}"
        assert result["alltoall"] == [(peer, rank) for peer in range(4)]


def test_large_arrays_travel_out_of_band_with_exact_bytes():
    comms = _world(3)
    rng = np.random.default_rng(7)
    words = {rank: rng.integers(0, 2**32, size=(55, 50_000), dtype=np.uint32) for rank in range(3)}
    words[1][0, :3] = np.array([0x7FC00001, 0x80000000, 0xFF800000], dtype=np.uint32)   # NaN payload, -0, -inf bits

    def operation(comm):
        rank = comm.Get_rank()
        frame = {"field": words[rank].view(np.float32), "small": np.arange(4) + rank, "text": f"rank {rank}"}
        return comm.gather(frame, root=0)

    results, errors = _run(comms, operation)
    assert not errors
    gathered = results[0]
    for rank in range(3):
        assert gathered[rank]["field"].view(np.uint32).tobytes() == words[rank].tobytes()
        assert list(gathered[rank]["small"]) == [rank, rank + 1, rank + 2, rank + 3]
        assert gathered[rank]["text"] == f"rank {rank}"
    assert comms[1]._links[0].sent_bytes > words[1].nbytes   # the array really crossed


def test_a_stopped_rank_makes_its_peers_raise():
    comms = _world(3, timeout=10.0)

    def operation(comm):
        if comm.Get_rank() == 2:
            comm.abort("rank 2 failed on purpose")
            return None
        return comm.allgather(comm.Get_rank())

    _results, errors = _run(comms, operation)
    assert errors and all(isinstance(error, rp.ProcessCommAborted) for _, error in errors)


def test_child_visible_devices_keeps_the_parent_numbering(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert rp.child_visible_devices(2) == "6"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert rp.child_visible_devices(3) == "3"


def test_executor_default_is_processes_and_threads_stay_selectable(monkeypatch):
    from types import SimpleNamespace
    from woof.hex.mpasphys import samehost_v842 as samehost

    monkeypatch.delenv("WOOF_HEX_SAMEHOST_RANKS", raising=False)
    assert samehost.executor_choice(SimpleNamespace(samehost_ranks=None))[0] == "processes"
    assert samehost.executor_choice(SimpleNamespace(samehost_ranks="threads"))[0] == "threads"
    monkeypatch.setenv("WOOF_HEX_SAMEHOST_RANKS", "threads")
    assert samehost.executor_choice(SimpleNamespace(samehost_ranks=None))[0] == "threads"
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    assert samehost.rank_host_parallelism("threads") == ("serialised (GIL)" if gil else "rank threads (free-threaded Python)")


def test_peer_halo_transport_is_not_dropped_by_the_process_executor(monkeypatch):
    """--samehost-transport peer was accepted and ignored on the default process executor (host-staged halos)."""
    from types import SimpleNamespace
    import pytest
    from woof.hex.mpasphys import samehost_v842 as samehost

    monkeypatch.delenv("WOOF_HEX_SAMEHOST_RANKS", raising=False)
    choice = samehost.executor_choice
    assert choice(SimpleNamespace(samehost_ranks=None, samehost_transport="peer"))[0] == "threads"
    assert choice(SimpleNamespace(samehost_ranks="threads", samehost_transport="peer"))[0] == "threads"
    with pytest.raises(ValueError, match="peer needs --samehost-ranks threads"):
        choice(SimpleNamespace(samehost_ranks="processes", samehost_transport="peer"))
    assert choice(SimpleNamespace(samehost_ranks=None, samehost_transport="staged"))[0] == "processes"
    assert choice(SimpleNamespace(samehost_ranks=None, samehost_transport="auto"))[0] == "processes"


def test_rank_threads_on_a_gil_python_say_so_once(monkeypatch, capsys):
    from woof.hex import partition_samehost_v841 as samehost

    monkeypatch.delenv("WOOF_HEX_RANK_GIL_WARNED", raising=False)
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)
    assert samehost.warn_serialised_ranks(4) is True
    assert samehost.warn_serialised_ranks(4) is False
    assert "shares one core" in capsys.readouterr().err
    monkeypatch.delenv("WOOF_HEX_RANK_GIL_WARNED", raising=False)
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)
    assert samehost.warn_serialised_ranks(4) is False


def _sleepers(group, ranks):
    import subprocess
    for rank in ranks:
        group.processes[rank] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    for rank, process in group.processes.items():
        watcher = threading.Thread(target=group._watch, args=(rank, process), daemon=True)
        watcher.start()
        group._watchers.append(watcher)


def test_a_dead_rank_stops_the_others_before_rank0_releases_and_rank0_is_not_waited_for(monkeypatch):
    """THE BREAKAGE (an RTX 5090 host, x1.2562 hrrrv5, 4 rank processes, 2026-10-29, tip-s4-oom-hang.log): rank 2 died, live peers
    wrote round numbers over rank 0's released halo flags, and rank 0 hung 4+ min in a CUDA module load.  Now the other
    ranks are killed and reaped before the failure callbacks run, and a rank 0 that does not respond within the grace
    ends the process with the failure named."""
    import os
    import signal
    import time

    group = rp.RankProcesses([0, 0, 0], {0: [1, 2], 1: [0, 2], 2: [0, 1]}, module="unused", timeout=5.0, job="t")
    group.rank0_grace = 0.5
    seen, exits = [], []
    group.on_failure(lambda reason: seen.append((reason, group.processes[2].poll())))
    monkeypatch.setattr(group, "_exit_now", lambda: exits.append(time.monotonic()))
    _sleepers(group, (1, 2))
    began = time.monotonic()
    os.kill(group.processes[1].pid, signal.SIGKILL)
    for watcher in group._watchers:
        watcher.join(10.0)
    assert seen and seen[0][0].startswith("rank 1 process") and "code -9" in seen[0][0]
    assert seen[0][1] is not None          # rank 2 was already reaped when rank 0's release ran
    assert len(seen) == 1                  # the second exit (the killed rank 2) is not a new failure
    assert exits and exits[0] - began < 5.0

    responsive = rp.RankProcesses([0, 0], {0: [1], 1: [0]}, module="unused", timeout=5.0, job="t")
    responsive.rank0_grace = 0.5
    ended = []
    monkeypatch.setattr(responsive, "_exit_now", lambda: ended.append(True))
    responsive.on_failure(lambda reason: responsive.rank0_responded())   # rank 0 reached its abort point
    _sleepers(responsive, (1,))
    os.kill(responsive.processes[1].pid, signal.SIGKILL)
    for watcher in responsive._watchers:
        watcher.join(10.0)
    assert responsive.failure and not ended


def test_aborted_deferred_checks_queue_no_device_work():
    """After a rank-process abort, rank 0's deferred checks raise at once (no cupy.stack, so no module load that
    synchronizes a context waiting on a dead rank's halo) and do not replace the abort as the named cause."""
    from woof.hex.mpasphys.deferred_checks_v842 import DeferredChecks, ForecastAborted

    checks = DeferredChecks(None)          # no array module: any device work would fail with AttributeError
    checks._pending.append((None, ("never read", RuntimeError), None))
    checks.abort("rank 1 process (pid 7) exited with code -9")
    with pytest.raises(ForecastAborted, match="rank 1 process"):
        checks.flush(5)
    with pytest.raises(ForecastAborted):
        checks.defer(None, "never")
    checks.poll(block=True)
    assert checks.prefer_earlier(RuntimeError, RuntimeError("halo"), None) is False
    assert not checks._pending
