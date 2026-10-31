# SPDX-License-Identifier: Apache-2.0
"""woof.hex.device_memory_release on a stand-in CuPy (no card needed).

The card measurements (bytes returned, bitwise output, step time) are in the
pool lane's report; these tests pin the bracket's contract: what it returns,
when, what each knob switches off, and that it leaves a failing or captured
phase alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from woof.hex.device_memory_release import (
    HEADROOM_KNOB,
    LMEM_KNOB,
    LMEM_MIN_FRAME_KNOB,
    LMEM_WAIT_KNOB,
    POOL_KNOB,
    POOL_MIN_BYTES_KNOB,
    DevicePhaseRelease,
    knob_enabled,
)

#: The mechanism tests below use tiny fake sizes: release every idle block and every grown frame, whatever the room.
_EVERY = {POOL_MIN_BYTES_KNOB: "0", LMEM_MIN_FRAME_KNOB: "0", HEADROOM_KNOB: "1"}


class _Pool:
    def __init__(self) -> None:
        self.used = 100
        self.held = 100
        self.frees = 0

    def used_bytes(self) -> int:
        return self.used

    def total_bytes(self) -> int:
        return self.held

    def free_all_blocks(self) -> None:
        self.frees += 1
        self.held = self.used


class _CUDARuntimeError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"status {status}")
        self.status = status


class _Runtime:
    cudaLimitStackSize = 1
    CUDARuntimeError = _CUDARuntimeError

    def __init__(self) -> None:
        self.stack = 1024
        self.sets: list[int] = []
        self.refuse = 0          # how many grow requests the driver refuses
        self.refuse_status = 2   # cudaErrorMemoryAllocation

    def getDevice(self) -> int:
        return 0

    free_bytes = 90 << 30        # a card with room unless a test says otherwise
    total_bytes = 96 << 30

    def memGetInfo(self):
        return self.free_bytes, self.total_bytes

    def deviceGetLimit(self, limit: int) -> int:
        assert limit == self.cudaLimitStackSize
        return self.stack

    def deviceSetLimit(self, limit: int, value: int) -> None:
        assert limit == self.cudaLimitStackSize
        if value > self.stack and self.refuse:
            self.refuse -= 1
            raise _CUDARuntimeError(self.refuse_status)
        self.sets.append(value)
        self.stack = value


def _fake_cp(capturing: bool = False):
    pool = _Pool()
    runtime = _Runtime()
    stream = SimpleNamespace(is_capturing=lambda: capturing)
    cuda = SimpleNamespace(runtime=runtime, get_current_stream=lambda: stream)
    return SimpleNamespace(cuda=cuda, get_default_memory_pool=lambda: pool), pool, runtime


def test_knobs_default_on_and_switch_off():
    assert knob_enabled(POOL_KNOB, {}) and knob_enabled(LMEM_KNOB, {})
    for off in ("0", "off", "OFF", "false", "no"):
        assert not knob_enabled(POOL_KNOB, {POOL_KNOB: off})
    assert knob_enabled(POOL_KNOB, {POOL_KNOB: "1"})


def test_phase_returns_cached_blocks_on_both_sides_and_shrinks_the_frame_store():
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    pool.held = 400  # 300 cached by the previous phase
    with release.phase("slot:microphysics"):
        assert pool.held == pool.used == 100  # returned before the phase allocates
        pool.used, pool.held = 150, 900       # the phase's own transient blocks
        runtime.stack = 43440                 # a kernel grew the local-memory frame
        pool.used = 150
    assert runtime.stack == 1024 and runtime.sets == [1024]
    assert pool.held == pool.used == 150
    receipt = release.receipt()
    assert receipt["pool_release"]["bytes_returned"] == 300 + 750
    assert receipt["lmem_release"]["resets"] == 1
    assert receipt["lmem_release"]["widest_stack_bytes"] == 43440
    assert receipt["by_phase"]["slot:microphysics"]["lmem_resets"] == 1


def test_frame_growth_outside_a_phase_is_kept():
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    runtime.stack = 2048  # the dycore's own frame, grown between phases
    with release.phase("arwen:phase1"):
        runtime.stack = 7216
    assert runtime.stack == 2048


def test_knobs_off_touch_nothing():
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ={POOL_KNOB: "0", LMEM_KNOB: "0"})
    pool.held = 400
    with release.phase("slot:radiation_sw"):
        runtime.stack = 9000
    assert pool.frees == 0 and runtime.sets == [] and pool.held == 400


def test_each_knob_alone():
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ={**_EVERY, LMEM_KNOB: "off"})
    pool.held = 400
    with release.phase("p"):
        runtime.stack = 9000
    assert runtime.sets == [] and pool.held == 100
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ={**_EVERY, POOL_KNOB: "off"})
    pool.held = 400
    with release.phase("p"):
        runtime.stack = 9000
    assert runtime.sets == [1024] and pool.frees == 0


def test_failed_phase_is_left_alone():
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    with pytest.raises(FloatingPointError):
        with release.phase("p"):
            pool.held, runtime.stack = 500, 9000
            raise FloatingPointError("refused")
    assert runtime.stack == 9000 and pool.held == 500


def test_capture_and_nesting_do_nothing():
    cp, pool, runtime = _fake_cp(capturing=True)
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    pool.held = 400
    with release.phase("p"):
        runtime.stack = 9000
    assert pool.frees == 0 and runtime.sets == []
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    with release.phase("outer"):
        with release.phase("inner"):
            runtime.stack = 9000
        assert runtime.sets == []  # the outer phase owns the boundary
    assert runtime.sets == [1024]


def test_nesting_is_per_thread():
    """Same-host ranks step on their own threads: one rank's open phase does
    not make another rank's bracket a pass-through."""

    import threading

    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    inside = threading.Event()
    done = threading.Event()

    def other_rank():
        with release.phase("arwen:phase1"):
            inside.set()
            done.wait(5)

    worker = threading.Thread(target=other_rank)
    worker.start()
    assert inside.wait(5)
    pool.held = 400
    with release.phase("arwen:phase2"):
        assert pool.held == pool.used  # this thread's entry still returned blocks
        runtime.stack = 7216
    assert runtime.stack == 1024
    done.set()
    worker.join(5)
    assert release.receipt()["phases"] == 2


def test_wide_frame_store_is_grown_back_on_entry_before_the_phase_runs():
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    with release.phase("slot:microphysics"):
        runtime.stack = 43440
    assert runtime.sets == [1024]
    with release.phase("slot:microphysics"):
        # grown at entry, where a refusal can be handled, not inside a launch
        assert runtime.stack == 43440
    assert runtime.sets == [1024, 43440, 1024]
    receipt = release.receipt()["lmem_release"]
    assert receipt["regrows"] == 1 and receipt["resets"] == 2
    assert receipt["pinned_after_regrow_wait"] is False


def test_min_frame_knob_keeps_narrow_frames():
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    with release.phase("slot:radiation_lw"):
        runtime.stack = 2048  # by default every grown store is released
    assert runtime.stack == 1024
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ={**_EVERY, LMEM_MIN_FRAME_KNOB: "4096"})
    with release.phase("slot:radiation_lw"):
        runtime.stack = 2048  # longwave's frame: under the knob, kept
    assert runtime.stack == 2048 and runtime.sets == []
    with release.phase("arwen:phase2"):
        runtime.stack = 7216  # WSM6: released back to the narrow store
    assert runtime.stack == 2048


def test_refused_regrow_returns_own_blocks_waits_then_keeps_the_store(capsys):
    cp, pool, runtime = _fake_cp()
    now = [0.0]
    naps: list[float] = []

    def sleep(seconds: float) -> None:
        naps.append(seconds)
        now[0] += seconds

    release = DevicePhaseRelease(cp, environ=dict(_EVERY), sleep=sleep, clock=lambda: now[0])
    with release.phase("slot:microphysics"):
        runtime.stack = 43440
    runtime.refuse = 3  # a co-tenant took the memory between steps
    pool.held = 400
    with release.phase("slot:microphysics"):
        assert runtime.stack == 43440
    assert pool.held == pool.used and naps == [2.0, 2.0]
    assert runtime.stack == 43440  # kept from now on
    with release.phase("slot:microphysics"):
        pass
    assert runtime.stack == 43440
    receipt = release.receipt()["lmem_release"]
    assert receipt["refused_regrows"] == 1 and receipt["pinned_after_regrow_wait"] is True
    assert receipt["regrow_wait_seconds"] == 4.0
    assert "refused to regrow" in capsys.readouterr().err


def test_regrow_wait_runs_out_with_an_error_naming_the_knob():
    cp, pool, runtime = _fake_cp()
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    release = DevicePhaseRelease(cp, environ={**_EVERY, LMEM_WAIT_KNOB: "5"}, sleep=sleep,
                                 clock=lambda: now[0])
    with release.phase("slot:microphysics"):
        runtime.stack = 43440
    runtime.refuse = 100
    with pytest.raises(RuntimeError, match=LMEM_KNOB):
        with release.phase("slot:microphysics"):
            pass


def test_other_driver_errors_are_not_waited_on():
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ=dict(_EVERY))
    with release.phase("slot:microphysics"):
        runtime.stack = 43440
    runtime.refuse, runtime.refuse_status = 1, 1  # cudaErrorInvalidValue
    with pytest.raises(_CUDARuntimeError):
        with release.phase("slot:microphysics"):
            pass


def test_defaults_keep_small_idle_pools_and_narrow_frames():
    """U-MPASPHYS-1: by default a boundary keeps under 8 GiB of idle blocks and frames under 4 KiB."""
    from woof.hex.device_memory_release import POOL_MIN_BYTES_DEFAULT
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ={})
    assert release.min_frame == 4096 and release.pool_min_bytes == POOL_MIN_BYTES_DEFAULT == 8 << 30
    with release.phase("slot:gwdo"):
        runtime.stack = 2736          # GWDO/PBL frame: kept, no reset/regrow pair per step
    assert runtime.stack == 2736 and runtime.sets == []
    assert release.receipt()["pool_release"]["releases"] == 0   # fake pool idles far below 4 GiB


def test_card_pressure_returns_idle_blocks_under_the_keep():
    """U-MPASPHYS-1: the keep yields when the card has under a tenth of its memory free."""
    cp, pool, runtime = _fake_cp()
    release = DevicePhaseRelease(cp, environ={})
    runtime.free_bytes = 5 << 30
    with release.phase("slot:pbl"):
        pool.held = 400            # this phase left idle blocks behind
    assert pool.held == pool.used and release.receipt()["pool_release"]["releases"] == 1


def test_card_pressure_is_read_at_most_once_a_second_per_card():
    """Host-bound V5090 2026-10-27: two cudaMemGetInfo calls per physics slot, ~1 ms each on a shared host,
    on the thread that issues the card's work.  The reading is reused for POOL_PRESSURE_RECHECK_SECONDS."""
    from woof.hex.device_memory_release import POOL_PRESSURE_RECHECK_SECONDS
    cp, pool, runtime = _fake_cp()
    reads = []
    real = runtime.memGetInfo
    runtime.memGetInfo = lambda: (reads.append(1), real())[1]
    now = [100.0]
    release = DevicePhaseRelease(cp, environ={}, clock=lambda: now[0])
    for _ in range(5):                       # five slots inside one second: one reading
        with release.phase("slot:pbl"):
            pool.held = pool.used + 400      # idle under the keep: the reading decides
    assert len(reads) == 1 and release.receipt()["pool_release"]["releases"] == 0
    runtime.free_bytes = 5 << 30             # another process takes the card's memory
    now[0] += POOL_PRESSURE_RECHECK_SECONDS  # the next boundary past the recheck sees it
    with release.phase("slot:pbl"):
        pool.held = pool.used + 400
    assert len(reads) == 2 and pool.held == pool.used
    assert release.receipt()["pool_release"]["releases"] >= 1


_GIB = 1 << 30


def test_big_idle_set_is_kept_while_the_card_has_room():
    """PROFILE-3KM item 4 (mv3km, RTX PRO 6000): the dycore's transients exceed the 8 GiB keep, and every step the
    surface-layer boundary returned them (25 synchronising cudaFree) on a card with 55 GB free.  Kept now."""
    cp, pool, runtime = _fake_cp()
    runtime.free_bytes, runtime.total_bytes = 55 * _GIB, 96 * _GIB
    release = DevicePhaseRelease(cp, environ={})
    pool.used = 30 * _GIB
    for _ in range(3):                         # three steps' surface-layer boundaries
        pool.held = pool.used + 12 * _GIB      # the dycore left 12 GiB of idle blocks
        with release.phase("slot:surface_layer"):
            pass
        assert pool.held == pool.used + 12 * _GIB
    receipt = release.receipt()["pool_release"]
    assert pool.frees == 0 and receipt["releases"] == 0 and receipt["keeps"] == 6


def test_big_idle_set_is_returned_when_the_card_is_short():
    """A tight card still releases: under 15 % free, or less free than the idle set it would keep."""
    now = [0.0]
    for free, idle in ((13 * _GIB, 9 * _GIB),     # under the 15 % headroom of a 96 GB card
                       (20 * _GIB, 30 * _GIB)):   # 20 % free, but less than the idle set
        cp, pool, runtime = _fake_cp()
        runtime.free_bytes, runtime.total_bytes = free, 96 * _GIB
        release = DevicePhaseRelease(cp, environ={}, clock=lambda: now[0])
        pool.used = 40 * _GIB
        pool.held = pool.used + idle
        with release.phase("slot:surface_layer"):
            assert pool.held == pool.used
        assert release.receipt()["pool_release"]["bytes_returned"] == idle
    # Small idle sets keep the U-MPASPHYS-1 rule: kept down to a tenth free.
    cp, pool, runtime = _fake_cp()
    runtime.free_bytes, runtime.total_bytes = 12 * _GIB, 96 * _GIB
    release = DevicePhaseRelease(cp, environ={})
    pool.held = pool.used + 2 * _GIB
    with release.phase("slot:pbl"):
        pass
    assert pool.frees == 0


def test_headroom_one_returns_big_idle_sets_at_every_boundary():
    cp, pool, runtime = _fake_cp()
    runtime.free_bytes, runtime.total_bytes = 55 * _GIB, 96 * _GIB
    release = DevicePhaseRelease(cp, environ={HEADROOM_KNOB: "1"})
    pool.used, pool.held = 30 * _GIB, 42 * _GIB
    with release.phase("slot:surface_layer"):
        assert pool.held == pool.used
    assert release.receipt()["headroom"]["fraction"] == 1.0


def test_wide_store_is_kept_while_the_card_has_room_and_set_back_once_it_is_short():
    """WSM6's 7,216 B store: kept past its phase on a roomy card (no reset now, no regrow next step); the first
    boundary that finds the card short sets it back, and the next WSM6 entry grows it back where a refusal is handled."""
    cp, pool, runtime = _fake_cp()
    now = [0.0]
    release = DevicePhaseRelease(cp, environ={}, clock=lambda: now[0])
    with release.phase("slot:microphysics"):
        runtime.stack = 7216
    with release.phase("slot:pbl"):
        assert runtime.stack == 7216
    with release.phase("slot:microphysics"):
        assert runtime.stack == 7216
    assert runtime.sets == []
    lmem = release.receipt()["lmem_release"]
    assert lmem["kept"] == 1 and lmem["resets"] == 0 and lmem["regrows"] == 0
    runtime.free_bytes = 10 * _GIB               # a co-tenant took the card's memory
    now[0] += 1.0
    with release.phase("slot:pbl"):
        assert runtime.stack == 1024             # set back at this boundary's entry
    with release.phase("slot:microphysics"):
        assert runtime.stack == 7216             # grown back at entry
    assert runtime.sets == [1024, 7216, 1024]    # and, still short, reset at its exit
    lmem = release.receipt()["lmem_release"]
    assert lmem["resets"] == 2 and lmem["regrows"] == 1
    assert release.receipt()["by_phase"]["slot:microphysics"]["lmem_resets"] == 2
