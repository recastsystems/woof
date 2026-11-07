"""GBM PBL (bl_pbl_physics=12): the device port against WRF v4.6.1 itself.

The fixture (woof/data/gbmpbl/oracle/) is what WRF's OWN compiled object
``phys/module_bl_gbmpbl.o`` from the scalar referee build wrote, driven by
``tools/gbmpbl_wrf461_oracle/run_gbmpbl.F90`` on 86 columns for 8 steps
each, carrying TKE_PBL and the advanced state between steps.  build.sh
also recompiles the pristine source at -O0 and requires the two runs to
agree byte for byte.

The port replays every set with its own carried state and must match every
word of every step: 0 ULP on every tendency, TKE, EL, exchange coefficient,
PBLH, UST and KPBL, under strict numerics. The fast default uses the common compiler policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np
import pytest

from tests.conftest import requires_gpu

from woof.verify.gbmpbl_oracle import (COMPARED, ORACLE_DIR, read_cases,
                                        read_oracle, replay_set)

CASES = ORACLE_DIR / "gbm-cases.bin"
ORACLE = ORACLE_DIR / "gbm-oracle.bin"
META = ORACLE_DIR / "gbm-cases.json"
SUMS = ORACLE_DIR / "FIXTURES.sha256"

#: The regimes the brief names, each needing at least one column.
REGIMES = ("sc-", "cbl-", "stable-", "neutral-", "decoupled-", "shallow-",
           "top-", "ice-", "layered-", "calm-")


def test_fixture_bytes_are_the_recorded_oracle_run():
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        got = hashlib.sha256((ORACLE_DIR / name).read_bytes()).hexdigest()
        assert got == digest, name


def test_fixture_covers_every_named_regime_with_64_plus_columns():
    meta = json.loads(META.read_text(encoding="utf-8"))
    tags = [t for s in meta["sets"] for t in s["tags"]]
    assert len(tags) >= 64
    for prefix in REGIMES:
        assert any(t.startswith(prefix) for t in tags), prefix
    cases = read_cases(CASES)
    oracle = read_oracle(ORACLE)
    assert [c["ncol"] for c in cases] == [s["ncol"] for s in meta["sets"]]
    for case, orc in zip(cases, oracle):
        assert (case["ncol"], case["nz"], case["nstep"]) == (
            orc["ncol"], orc["nz"], orc["nstep"])
        assert orc["nstep"] >= 2
        for step in orc["steps"]:
            for name, value in step.items():
                if value.dtype.kind == "f":
                    assert np.all(np.isfinite(value)), name


def test_oracle_exercises_merging_cloud_top_and_both_surfaces():
    """The fixture reaches the branches that make GBM GBM."""
    cases = read_cases(CASES)
    oracle = read_oracle(ORACLE)
    xland = np.concatenate([c["xland"] for c in cases])
    assert set(np.unique(xland)) == {1.0, 2.0}
    kpbl = np.concatenate([s["steps"][-1]["kpbl"] for s in oracle])
    assert kpbl.min() == 1 and kpbl.max() >= 30
    # cloud-top radiative cooling entering the TKE budget needs cloud at a
    # convective layer top: qc > 1e-8 with RTHRATEN < 0
    cloudy_cooling = any(
        np.any((c["qc"] > 1e-8) & (c["rthraten"] < 0)) for c in cases)
    assert cloudy_cooling
    # the 1000 m2/s diffusivity ceiling binds somewhere
    assert max(float(s["steps"][-1]["exch_h"].max()) for s in oracle) == 1000.0


@requires_gpu
def test_port_matches_wrf_word_for_word_on_every_step():
    cases = read_cases(CASES)
    oracle = read_oracle(ORACLE)
    worst = {name: 0 for name in COMPARED}
    rows = []
    for index, (case, orc) in enumerate(zip(cases, oracle)):
        for row in replay_set(case, orc):
            rows.append({"set": index, **row})
            for name in COMPARED:
                worst[name] = max(worst[name], row[name])
    receipt = os.environ.get("WOOF_GBMPBL_PARITY_RECEIPT")
    if receipt:
        from woof import wrf_exact
        with open(receipt, "w", encoding="utf-8") as fh:
            json.dump({"wrf_exact": wrf_exact.ENABLED, "worst": worst,
                       "steps": rows}, fh, indent=1)
    assert worst == {name: 0 for name in COMPARED}, worst
    assert all(count == 0 for row in rows for count in row["different_words"].values()), rows


def _next_float_literal(value: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", value))[0] ^ 1
    flipped = struct.unpack("<f", struct.pack("<I", bits))[0]
    return f"{float(np.float32(flipped)).hex()}f"


@requires_gpu
def test_a_one_bit_change_in_a_kernel_constant_is_caught(monkeypatch):
    """Flip the last bit of czero (module_bl_gbmpbl.F:428) and replay."""
    from woof.core import gbmpbl

    original = gbmpbl.module_source()
    needle = "#define GB_CZERO  5.869f"
    assert original.count(needle) == 1
    planted = original.replace(
        needle, f"#define GB_CZERO  {_next_float_literal(5.869)}")
    monkeypatch.setattr(gbmpbl, "module_source", lambda: planted)
    gbmpbl._module.cache_clear()
    try:
        case = read_cases(CASES)[0]
        orc = read_oracle(ORACLE)[0]
        orc = dict(orc, steps=orc["steps"][:2])
        rows = replay_set(case, orc)
    finally:
        gbmpbl._module.cache_clear()
    moved = sorted({name for row in rows for name in COMPARED if row[name]})
    assert moved, "a one-bit change in czero went unnoticed"
    assert "tke_pbl" in moved


@requires_gpu
def test_unused_roots_matches_its_wrf_source_routine():
    import cupy as cp
    from woof.core import gbmpbl
    fixture = np.fromfile(ORACLE_DIR / "roots-oracle.bin", dtype="<f4").reshape(-1, 5)
    inp = cp.asarray(np.ascontiguousarray(fixture[:, :3]))
    out = cp.empty((len(fixture), 2), dtype=cp.float32)
    fn = gbmpbl._module(cp.cuda.Device().id).get_function("gbmpbl_roots_probe")
    fn(((len(fixture) + 31)//32,), (32,), (inp, out, np.int32(len(fixture))))
    assert cp.asnumpy(out).tobytes() == np.ascontiguousarray(fixture[:, 3:]).tobytes()
