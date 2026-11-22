# SPDX-License-Identifier: Apache-2.0
"""The external-tendency slot (woof/hex/ad/external_tendency.py), without a card.

What these hold: with nothing given the slot does not exist (so every run and plan is the bytes it was); the kind has a
window rule by name; a table is read one step slice at a time and refuses past its end; the add is tend += ext with a
zero word leaving tend untouched (a -0 tendency stays -0); the assembler route adds into copies, never into the
schemes' own rates; and the RRFS rate-table upload keeps the source text the sealer's rule matches.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from woof.hex.ad import external_tendency as et
from woof.hex.ad.record import coverage

SRC = Path(et.__file__).resolve().parents[1]
RUST = SRC.parents[1] / "rust" / "hexad" / "src" / "runner" / "wrules.rs"


class _Arr(np.ndarray):
    """A host stand-in for a CuPy array: .data.ptr, .device.id, c_contiguous flags."""

    @property
    def data(self):  # noqa: D401 - CuPy's spelling
        class _P:
            ptr = self.__array_interface__["data"][0]
        return _P()

    @property
    def device(self):
        class _D:
            id = 0
        return _D()


class _FakeCP:
    ndarray = _Arr
    float32 = np.float32

    class cuda:  # noqa: N801 - CuPy's spelling
        class Device:
            id = 0

            def synchronize(self):
                pass

        @staticmethod
        def get_current_stream():
            class _S:
                ptr = 0

                def synchronize(self):
                    pass
            return _S()

    @staticmethod
    def asarray(x):
        return np.ascontiguousarray(x).view(_Arr)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype).view(_Arr)

    @staticmethod
    def asnumpy(x):
        return np.asarray(x)


class _Cache:
    def __init__(self):
        self.launches = []

    def raw_kernel(self, name, source, *, module_key):
        assert name == et.KIND and module_key == et.MODULE_KEY and et.KIND in source

        def launch(grid, block, args):
            n, ext, tend = int(args[0]), np.asarray(args[1]).reshape(-1), np.asarray(args[2]).reshape(-1)
            self.launches.append((n, ext.ctypes.data, tend.ctypes.data))
            live = ext[:n] != 0
            tend[:n][live] = tend[:n][live] + ext[:n][live]
        return launch


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(et.ENV_TABLE, raising=False)
    monkeypatch.delenv(et.ENV_RECEIPT, raising=False)
    et.uninstall()
    yield
    et.uninstall()


def test_absent_is_nothing():
    # No callback, no table: no slot, no rates copied (the default every run and plan is recorded on).
    rates = {"rthblten": object()}
    assert et.slot_for(cp=None, cache=None, shape=(2, 3)) is None
    assert et.substitute_rates(None, rates, None) is rates

    class Owner:
        pass
    owner = Owner()
    assert et.rates_for(owner, rates, cp=None, cache=None, mass=np.zeros((2, 3))) is rates
    assert owner._external_tendency_slot is None


def test_kind_has_a_window_rule():
    assert coverage.why_covered(et.KIND) == "rule"
    # (WOOF 1.1.7 round 4) the runner's Rust source is hex's (rust/hexad); WOOF folds the built runner only
    if not RUST.is_file():
        pytest.skip("rust/hexad is not folded into WOOF; the runner ships built in the helper archive")
    text = RUST.read_text()
    assert f'"{et.KIND}" if !self.kernels.contains_key(f)' in text
    assert "fn external_tendency_rule" in text and "c: 1.0" in text


def test_table_steps_and_add(tmp_path, monkeypatch):
    nz, nc, steps = 3, 5, 2
    theta = np.zeros((steps, nz, nc), np.float32)
    theta[0, 1, 2] = 0.5
    theta[1, 0, 0] = -0.25
    table = tmp_path / "t.npz"
    np.savez(table, theta=theta)
    receipt = tmp_path / "ext.json"
    monkeypatch.setenv(et.ENV_TABLE, str(table))
    monkeypatch.setenv(et.ENV_RECEIPT, str(receipt))
    cache = _Cache()
    slot = et.slot_for(cp=_FakeCP, cache=cache, shape=(nz, nc))
    assert slot.fields == ("theta",) and slot.nsteps == steps and receipt.is_file()
    tend = np.full((nz, nc), -0.0, np.float32).view(_Arr)
    slot.add({"theta": tend})
    assert tend[1, 2] == np.float32(0.5)
    # every word the table leaves zero keeps its -0 bit for bit
    rest = np.delete(tend.reshape(-1), 1 * nc + 2)
    assert np.all(np.signbit(rest)) and np.all(rest == 0)
    # step 1 reads slice 1 of the one table allocation (pointer offset k nz nc 4)
    tend2 = np.zeros((nz, nc), np.float32).view(_Arr)
    slot.add({"theta": tend2})
    assert tend2[0, 0] == np.float32(-0.25)
    base = slot.tables["theta"].data.ptr
    assert [l[1] - base for l in cache.launches] == [0, nz * nc * 4]
    with pytest.raises(RuntimeError, match="covers 2 steps"):
        slot.add({"theta": tend2})


def test_preload_is_the_slots_table(tmp_path, monkeypatch):
    # Uploaded before the step loop (boundary 0 of a recording); the slot then uses that allocation, no second upload.
    nz, nc = 2, 3
    table = tmp_path / "t.npz"
    np.savez(table, u=np.arange(2 * nz * nc, dtype=np.float32).reshape(2, nz, nc))
    et.preload(_FakeCP)                       # unset: nothing
    assert not et._PRELOADED
    monkeypatch.setenv(et.ENV_TABLE, str(table))
    try:
        et.preload(_FakeCP)
        first = et._PRELOADED[str(table)]["u"]
        slot = et.slot_for(cp=_FakeCP, cache=_Cache(), shape=(nz, nc))
        assert slot.tables["u"] is first and slot.nsteps == 2
        with pytest.raises(ValueError, match="not \\(nsteps, nz, ncol\\)"):
            et.slot_for(cp=_FakeCP, cache=_Cache(), shape=(nz, nc + 1))
    finally:
        et._PRELOADED.clear()


def test_forecast_preloads_before_the_step_loop():
    text = (SRC / "drivers" / "run_cuda_v841_forecast.py").read_text()
    assert text.index("_external_tendency.preload(cp)") < text.index("for step in range(start_step + 1, steps + 1):")


def test_table_refuses_wrong_shape(tmp_path, monkeypatch):
    table = tmp_path / "t.npz"
    np.savez(table, theta=np.zeros((2, 3, 4), np.float32))
    monkeypatch.setenv(et.ENV_TABLE, str(table))
    with pytest.raises(ValueError, match="not \\(nsteps, nz, ncol\\)"):
        et.slot_for(cp=_FakeCP, cache=_Cache(), shape=(3, 5))
    np.savez(table, wind=np.zeros((2, 3, 5), np.float32))
    with pytest.raises(ValueError, match="not slot fields"):
        et.slot_for(cp=_FakeCP, cache=_Cache(), shape=(3, 5))


def test_callback_route_copies_rates(tmp_path):
    nz, nc = 2, 4
    seen = []

    def callback(k, state):
        seen.append((k, state.nz, state.ncol))
        return {"qv": _FakeCP.asarray(np.full((nz, nc), 1e-6, np.float32)), "u": None}

    et.install(callback, fields=("qv", "u"), save=str(tmp_path / "s.npz"))
    rates = {name: np.zeros((nz, nc), np.float32).view(_Arr) for name in et.RATE_OF.values()}
    before = {k: v.copy() for k, v in rates.items()}

    class Owner:
        pass
    owner = Owner()
    cache = _Cache()
    out = et.rates_for(owner, rates, cp=_FakeCP, cache=cache, mass=np.zeros((nz, nc)))
    assert out is not rates and all(out[k] is not rates[k] for k in rates)
    assert all(np.array_equal(rates[k], before[k]) for k in rates)          # the schemes' rates untouched
    assert np.all(out["rqvblten"] == np.float32(1e-6)) and not np.any(out["rublten"])
    assert len(cache.launches) == 1 and seen == [(0, nz, nc)]               # u was None: no launch
    et.rates_for(owner, rates, cp=_FakeCP, cache=cache, mass=np.zeros((nz, nc)))
    assert seen[-1][0] == 1
    path = et.write_table(tmp_path / "s.npz")
    with np.load(path) as data:
        assert data.files == ["qv"] and data["qv"].shape == (2, nz, nc)


def test_callback_refuses_bad_arrays():
    et.install(lambda k, s: {"theta": np.zeros((2, 4), np.float64).view(_Arr)})
    slot = et.slot_for(cp=_FakeCP, cache=_Cache(), shape=(2, 4))
    with pytest.raises(TypeError, match="not float32"):
        slot.add({"theta": np.zeros((2, 4), np.float32).view(_Arr)})
    et.install(lambda k, s: {"wind": None}, fields=("theta",))
    slot = et.slot_for(cp=_FakeCP, cache=_Cache(), shape=(2, 4))
    with pytest.raises(ValueError, match="outside its fields"):
        slot.add({})
    with pytest.raises(ValueError, match="not slot fields"):
        et.install(lambda k, s: {}, fields=("w",))


def test_allocation_ids_finds_recording_as_line(tmp_path):
    import json
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "catalog-a.json").write_text(json.dumps({"allocations": [
        {"id": 7, "recorded_base": 1000, "bytes": 4096, "memory": "device", "freed_clock_ns": None},
        {"id": 9, "recorded_base": 1000, "bytes": 64, "memory": "device", "freed_clock_ns": 5},
        {"id": 11, "recorded_base": 8192, "bytes": 1 << 20, "memory": "device", "freed_clock_ns": None}]}))
    receipt = tmp_path / "ext.jsonl"
    warm = {"pid": 1, "tables": {"theta": {"ptr": 99_000_000, "bytes": 400}}}
    a = {"pid": 2, "tables": {"theta": {"ptr": 1024, "bytes": 400}, "qv": {"ptr": 8192 + 512, "bytes": 400}},
         "state": {"qv": {"ptr": 9000, "bytes": 40}}}
    receipt.write_text(json.dumps(warm) + "\n" + json.dumps(a) + "\n")
    got = et.allocation_ids(plan, receipt)
    assert got["pid"] == 2
    assert got["tables"]["theta"] == {"allocation": 7, "offset": 24, "bytes": 400}   # live allocation, not the freed one
    assert got["tables"]["qv"] == {"allocation": 11, "offset": 512, "bytes": 400}
    assert got["state"]["qv"]["allocation"] == 11
    receipt.write_text(json.dumps(warm) + "\n")
    with pytest.raises(ValueError, match="no line"):
        et.allocation_ids(plan, receipt)


def test_rate_table_upload_keeps_the_sealers_text():
    # seal_phys_stage2.TABLE_TEXT_RULES finds the RRFS stage-sources rate table by this exact line.
    text = (SRC / "mpasphys" / "rrfs" / "driver_rrfs.py").read_text()
    assert "        table = cp.asarray(np.array([array.data.ptr for array in table_arrays], dtype=np.uint64))\n" in text
    assert text.index("external_tendency.slot_for(") < text.index("table = cp.asarray(np.array([array.data.ptr")


def test_callback_sees_the_runs_clock():
    """The step length and model time reach the callback's state view from the step that calls the slot.

    THE BREAKAGE THIS PREVENTS (hex 0.6.2, 20 Nov 2026): no route handed the slot its step length, so
    ``state.dt`` was always None and ``state.time_seconds`` did not exist; the learned correction stopped every forecast
    whose door had no ``--dt`` word ("the step length is unknown (pass --dt)"), ``woof hex global`` among them."""
    nz, nc = 2, 4
    seen = []
    et.install(lambda k, state: seen.append((k, state.dt, state.time_seconds)) or {})
    rates = {name: np.zeros((nz, nc), np.float32).view(_Arr) for name in et.RATE_OF.values()}

    class Owner:
        pass
    owner = Owner()
    et.rates_for(owner, rates, cp=_FakeCP, cache=_Cache(), mass=np.zeros((nz, nc)), dt=720.0, time_seconds=0.0)
    et.rates_for(owner, rates, cp=_FakeCP, cache=_Cache(), mass=np.zeros((nz, nc)), dt=720.0, time_seconds=720.0)
    slot = et.slot_for(cp=_FakeCP, cache=_Cache(), shape=(nz, nc))
    slot.add({}, dt=60.0, time_seconds=1800.0)
    assert seen == [(0, 720.0, 0.0), (1, 720.0, 720.0), (0, 60.0, 1800.0)]


def test_every_route_hands_the_slot_its_clock():
    """Each engine call site of the slot passes the step's dt and model time (see the test above)."""
    for rel, call in (("mpasphys/resident_v842.py", "external_tendency.rates_for("),
                      ("mpasphys/rrfs/resident_rrfs.py", "external_tendency.rates_for("),
                      ("mpasphys/rrfs/driver_rrfs.py", "self.external.add(")):
        text = (SRC / rel).read_text()
        at = text.index(call)
        window = text[at:at + 400]
        assert "dt=getattr(call, 'dt', None)" in window, rel
        assert "time_seconds=getattr(call, 'time_seconds', None)" in window, rel
