"""Rank-process CPU windows: start-up vs step 1 vs stepping, with system time per thread group.

THE QUESTION (NOAA RRFS v2 conus3km, V8X6000, 2026-10-29): rank processes spent more CPU time in the kernel than in
user code (rank 0 775.7 s system / 428 s user in a 543 s 4-card run) and the exit totals could not say when or on
which thread.  These tests hold the instrument to what it reports; host-only, no card.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types

import pytest

from woof.hex import rank_processes as rp

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc and getrusage")


@pytest.fixture(autouse=True)
def fresh_marks(monkeypatch):
    monkeypatch.setattr(rp, "_CPU_MARKS", [])


def _burn_user(seconds):
    end = time.process_time() + seconds
    while time.process_time() < end:
        sum(range(2000))


def _name_thread(name):
    # Python 3.14 names OS threads after threading names; older Pythons do not, so set it (PR_SET_NAME) here.
    import ctypes
    ctypes.CDLL(None).prctl(15, name.encode()[:15], 0, 0, 0)


def _burn_system(seconds, name=None):
    if name is not None:
        _name_thread(name)
    # Reading /dev/zero is the kernel filling this process's buffer: its time is system time.
    fd = os.open("/dev/zero", os.O_RDONLY)
    try:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            os.read(fd, 1 << 20)
    finally:
        os.close(fd)


@linux
def test_windows_split_at_marks_and_close_at_exit():
    rp.cpu_mark("step 1 begin")
    _burn_user(0.3)
    rp.cpu_mark("step 2 begin")
    rp.cpu_mark("step 2 begin")   # a repeated label keeps its first reading
    record = rp.process_cpu_record(3)
    windows = record["cpu_windows"]
    assert windows["marks"] == ["step 1 begin", "step 2 begin"]
    names = [w["window"] for w in windows["windows"]]
    assert names == ["process start -> step 1 begin", "step 1 begin -> step 2 begin", "step 2 begin -> exit"]
    step1 = windows["windows"][1]
    assert step1["cpu_user_seconds"] >= 0.2
    assert step1["wall_seconds"] >= 0.2
    first = windows["windows"][0]
    assert first["wall_seconds"] is None or first["wall_seconds"] > 0
    # The windows add up to the exit totals (the last window closes a moment after the record's own read).
    total = sum(w["cpu_user_seconds"] + w["cpu_system_seconds"] for w in windows["windows"])
    assert total == pytest.approx(record["cpu_user_seconds"] + record["cpu_system_seconds"], abs=0.05)


@linux
def test_system_time_is_attributed_to_the_thread_that_spent_it():
    rp.cpu_mark("before")
    worker = threading.Thread(target=_burn_system, args=(0.6, "hex-sys-burner"))
    worker.start()
    worker.join()
    keeper = threading.Event()
    alive = threading.Thread(target=lambda: (_burn_system(0.4, "hex-live-burner_7"), keeper.wait()))
    alive.start()
    time.sleep(0.6)
    try:
        windows = rp.cpu_windows()
    finally:
        keeper.set()
        alive.join()
    window = windows["windows"][-1]
    rows = {row["threads"]: row for row in window["threads"]}
    # A live thread is named without its pool index and carries its kernel time.
    assert rows["hex-live-burner"]["cpu_system_seconds"] >= 0.1
    assert rows["hex-live-burner"]["count"] == 1
    # A thread that ended inside the window is not in /proc any more: its time is the remainder, not lost.
    assert "hex-sys-burner" not in rows
    assert window["ended_threads_cpu_system_seconds"] + window["ended_threads_cpu_user_seconds"] >= 0.2
    assert window["cpu_system_seconds"] >= rows["hex-live-burner"]["cpu_system_seconds"]


@linux
def test_thread_group_names():
    assert rp._thread_name("rank-request_3", 11, 10) == "rank-request"
    assert rp._thread_name("cuda-EvtHandlr", 11, 10) == "cuda-EvtHandlr"
    assert rp._thread_name("python3.14", 10, 10) == "main"
    assert rp._thread_name("123", 11, 10) == "123"


def test_off_linux_the_record_says_nothing(monkeypatch):
    monkeypatch.setattr(rp.sys, "platform", "win32")
    rp.cpu_mark("step 1 begin")
    assert rp._CPU_MARKS == []
    assert rp.cpu_windows() is None


@linux
def test_an_unreadable_proc_never_fails_the_record(monkeypatch):
    def broken():
        raise OSError("no /proc")
    monkeypatch.setattr(rp, "_thread_counters", broken)
    rp.cpu_mark("step 1 begin")   # no exception, no mark
    assert rp._CPU_MARKS == []
    assert "error" in rp.cpu_windows()


def test_mpas_rank_execution_marks_start_up_step_one_and_the_last_step(monkeypatch):
    from woof.hex.mpasphys import process_ranks_v842 as pr
    from woof.hex.mpasphys import samehost_v842

    labels = []
    monkeypatch.setattr(rp, "cpu_mark", labels.append)
    monkeypatch.setattr(samehost_v842.SameHostRankExecution, "before_step",
                        lambda self, step: setattr(self, "_step", step))
    monkeypatch.setattr(samehost_v842.SameHostRankExecution, "after_step",
                        lambda self, dycore: self.step_stages.append({}))
    execution = pr.ProcessRankExecution.__new__(pr.ProcessRankExecution)
    execution.step_stages, execution.rank, execution.devices = [], 1, [0, 0]
    execution.last_step = 4
    for step in range(1, 5):
        execution.before_step(step)
        execution.after_step(types.SimpleNamespace())
    assert labels == ["step 1 begin", "step 2 begin", "step 4 end"]


@linux
def test_windows_carry_the_process_own_memory_at_each_mark():
    rp.cpu_mark("step 1 begin")
    record = rp.process_cpu_record(0)
    windows = record["cpu_windows"]["windows"]
    for window in windows:
        memory = window["memory_at_end_mib"]
        assert memory["VmHWM"] >= memory["VmRSS"] > 0
        assert set(memory) <= set(rp._STATUS_MEMORY_KEYS)
    assert record["memory_at_exit_mib"]["VmHWM"] == record["max_rss_mib"]


@linux
def test_a_rank_process_records_its_own_peak_not_the_parent_rss_it_was_launched_from():
    # THE BREAKAGE (NOAA RRFS v2 conus3km, V8XB, 2026-10-29): getrusage's ru_maxrss survives fork and exec, so every
    # rank process recorded the forecast parent's RSS at launch (104.3 GiB on ranks 1-7) while its own VmHWM was
    # 35.2-36.0 GiB, and the receipts sized host RAM from it.  Here the parent holds 512 MiB resident and launches a
    # child the way RankProcesses does (subprocess, fork + exec); the child's max_rss_mib must be its own peak.
    import json
    import subprocess
    held_mib = 512
    ballast = bytearray(held_mib << 20)
    ballast[::4096] = b"\x01" * len(ballast[::4096])   # touch every page so it is resident
    child = ("import json; from woof.hex import rank_processes as rp; "
             "print(json.dumps(rp.process_cpu_record(1)))")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, env=env, timeout=120,
                         check=True)
    record = json.loads(out.stdout.strip().splitlines()[-1])
    del ballast
    assert record["max_rss_source"].startswith("VmHWM")
    assert record["max_rss_mib"] < held_mib / 2, record   # the child's own interpreter, not the parent's 512 MiB
    assert record["max_rss_mib"] == record["memory_at_exit_mib"]["VmHWM"]
    # The inherited figure stays in the record, labelled, and shows what the old field reported.
    assert record["rusage_max_rss_mib"] >= held_mib * 0.9, record
    assert "parent" in record["rusage_max_rss_note"]


@linux
def test_an_unreadable_status_leaves_the_peak_unknown_not_inherited(monkeypatch):
    monkeypatch.setattr(rp, "_own_memory_mib", lambda: None)
    record = rp.process_cpu_record(2)
    assert record["max_rss_mib"] is None
    assert record["max_rss_source"].startswith("unavailable")
    assert record["rusage_max_rss_mib"] > 0


def test_the_parent_counts_what_it_releases_once_the_ranks_inputs_are_cut():
    # The forecast parent (rank 0) held the global inputs and every child's cut for the whole run (NOAA RRFS v2
    # conus3km, V8XB, 2026-10-29: parent 148.6 GiB VmHWM against 35-36 GiB per child). samehost_v842 releases them
    # after preparation and the launch receipt says how much: arrays rank 0 still references are not counted.
    import numpy as np
    from dataclasses import dataclass
    from woof.hex.mpasphys import samehost_v842 as sh

    mib = 1 << 20
    shared_profile = np.zeros(mib // 8)                     # 1 MiB, referenced by rank 0's local inputs too
    mesh = types.SimpleNamespace(arrays={"cellsOnEdge": np.zeros(2 * mib // 8)})
    @dataclass(slots=True)
    class State:
        theta: object
        qv: object

    state = State(theta=np.zeros(4 * mib // 8), qv=np.zeros(4 * mib // 8)[::2])   # a view counts once
    inputs = (mesh, None, shared_profile, state, None, None, ())
    local0 = (types.SimpleNamespace(partition_source_mesh=mesh), None, shared_profile, None, None, None, ())
    local1 = (None, None, np.zeros(3 * mib // 8), None, None, None, ())
    cuts = [(local0, 0.1), (local1, 0.1)]
    prepared = [(None, None, local0, 0.0), (None, None, local1, 0.0)]
    released = sh._released_host_bytes(inputs, cuts, prepared, "processes")
    assert released == dict(global_inputs=8.0, rank_inputs=3.0)
    # Rank threads keep every rank's inputs, so only the global arrays are released.
    assert sh._released_host_bytes(inputs, cuts, prepared, "threads") == dict(global_inputs=8.0, rank_inputs=0.0)
