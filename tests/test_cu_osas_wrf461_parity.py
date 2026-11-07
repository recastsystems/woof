"""cu_physics = 95 (WRF v4.6.1 old GFS SAS) on the card, against WRF, word for word.

The fixtures of record (tests/data/oracles/osas/) are the byte-pinned
``module_cu_osas.F`` run by ``tools/osas_wrf461_oracle`` through WRF's own
preprocessing and the WRF 0 ULP referee's compile flags, with XKT2 pinned:
96 analytic soundings (deep tropical, midlatitude severe, shallow-only,
capped, dry, near-saturated; land and sea) on a 44-level and a 29-level
grid, six calls each (DT 60 s / STEPCU 1 three times chained, DT 30 s /
STEPCU 3 twice chained, DT 20 s / STEPCU 5).  About 40 columns per call
convect deeply and about 50 more mix shallowly.

Graded for EQUALITY of every float32 output word (RTHCUTEN, RQVCUTEN,
RQCCUTEN, RQICUTEN, RAINCV, PRATEC, HTOP, HBOT) and of every binary64 word
of the two stage records (after OSASCNV, after SHALCV), with the kernel in
its WRF-faithful parity setting.  The shipped identity's one deliberate
difference -- QL restored with T1 and Q1 when the column's rain evaporates
to nothing -- is graded separately: it may move exactly those columns and
nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from woof.core import column_cumulus_r8 as cc
from woof import wrf_exact

strict_words = pytest.mark.skipif(not wrf_exact.ENABLED, reason="word equality requires strict arithmetic")

FIXTURES = Path(__file__).parent / "data" / "oracles" / "osas"
GRIDS = ("k44", "k29")
SCHEME = cc.COLUMN_CUMULUS_SCHEMES[95]
FLAG_WRF_QL_BACKOUT = 1


def _fixture(grid):
    with np.load(FIXTURES / f"osas-{grid}.npz") as data:
        return {key: data[key] for key in data.files}


def _call(fx, n):
    dt, stepcu, itimestep, chain = fx[f"c{n}_scalars"]
    inputs = {name: fx[f"c{n}_in_{name}"] for name in
              cc.LEVEL_INPUTS + cc.IFACE_INPUTS + ("xland", "xkt2")}
    outputs = {name: fx[f"c{n}_out_{name}"] for name in
               cc.LEVEL_OUTPUTS + cc.SURFACE_OUTPUTS}
    stage = {key[len(f"c{n}_stage_"):]: fx[key] for key in fx
             if key.startswith(f"c{n}_stage_")}
    return float(dt), int(stepcu), int(itimestep), inputs, outputs, stage


def _run(inputs, dt, stepcu, *, flags, stages=True, budget_bytes=None, scheme=SCHEME):
    import cupy as cp

    dev = {name: cp.asarray(np.ascontiguousarray(inputs[name], dtype=np.float32))
           for name in cc.LEVEL_INPUTS + cc.IFACE_INPUTS + ("xland",)}
    draw = cp.asarray(np.ascontiguousarray(inputs["xkt2"], dtype=np.float64))
    kwargs = {} if budget_bytes is None else {"budget_bytes": budget_bytes}
    out = cc.run_columns(scheme, dev, draw, dt=dt, stepcu=stepcu, flags=flags,
                         stages=stages, **kwargs)
    return {name: cp.asnumpy(value) for name, value in out.items()}


def _stage_view(stages, kx):
    """The kernel's (ncol, 6*kx+5) stage records as the oracle's arrays."""
    t = stages.T
    return {"t1": t[0:kx], "q1": t[kx:2 * kx], "ql1": t[2 * kx:3 * kx],
            "ql2": t[3 * kx:4 * kx], "rn": t[4 * kx], "cldwrk": t[4 * kx + 1],
            "kbot": t[4 * kx + 2].astype(np.int32),
            "ktop": t[4 * kx + 3].astype(np.int32),
            "kuo": t[4 * kx + 4].astype(np.int32),
            "t1_shallow": t[4 * kx + 5:5 * kx + 5],
            "q1_shallow": t[5 * kx + 5:6 * kx + 5]}


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint64 if a.dtype.itemsize == 8 else np.uint32)


def _bad_columns(got, want):
    """Column indices where any word differs, per field."""
    bad = {}
    for name, w in want.items():
        g = got[name]
        diff = _bits(g) != _bits(np.asarray(w, dtype=g.dtype))
        cols = np.nonzero(diff.any(axis=0) if diff.ndim == 2 else diff)[0]
        if cols.size:
            bad[name] = cols.tolist()
    return bad


def _backout_columns(inputs, stage):
    """Columns whose rain evaporated to nothing after QL detrainment: WRF
    leaves QL raised there while T1/Q1 are backed out."""
    qi, qc = inputs["qi"].astype(np.float32), inputs["qc"].astype(np.float32)
    ql1_in = (qi / (np.float32(1.0) + qi)).astype(np.float64)
    ql2_in = (qc / (np.float32(1.0) + qc)).astype(np.float64)
    moved = (np.any(stage["ql1"] != ql1_in, axis=0)
             | np.any(stage["ql2"] != ql2_in, axis=0))
    return np.nonzero(moved & (stage["rn"] <= 0.0))[0]


@pytest.mark.gpu
def test_the_gfuncphys_tables_are_wrfs_words():
    import cupy as cp

    receipt = json.loads((FIXTURES / "tables.json").read_text(encoding="utf-8"))
    tb = cp.asnumpy(cc.gfs_funcphys_tables(SCHEME.module))
    assert cc.table_sha256(tb) == receipt["sha256"]
    assert cc.GFS_FUNCPHYS_SHA256 == receipt["sha256"]
    for n, name in enumerate(receipt["coefficients"]):
        assert tb[n] == float.fromhex(receipt["coefficients"][name]), name


@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
@strict_words
def test_every_call_of_every_column_is_wrfs_words(grid):
    fx = _fixture(grid)
    kx = int(fx["kx"])
    for n in range(int(fx["ncall"])):
        dt, stepcu, _, inputs, want, stage = _call(fx, n)
        got = _run(inputs, dt, stepcu, flags=FLAG_WRF_QL_BACKOUT)
        bad = _bad_columns(got, want)
        assert not bad, f"{grid} call {n}: outputs differ in columns {bad}"
        got_stage = _stage_view(got["stages"], kx)
        bad = _bad_columns(got_stage, stage)
        assert not bad, f"{grid} call {n}: stage records differ in columns {bad}"


@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
@strict_words
def test_the_shipped_backout_moves_only_the_backed_out_columns(grid):
    fx = _fixture(grid)
    kx = int(fx["kx"])
    for n in range(int(fx["ncall"])):
        dt, stepcu, _, inputs, want, stage = _call(fx, n)
        got = _run(inputs, dt, stepcu, flags=SCHEME.flags)
        backed = set(_backout_columns(inputs, stage).tolist())
        bad = _bad_columns(got, want)
        moved = set()
        for name, cols in bad.items():
            assert name in ("rqccuten", "rqicuten"), (
                f"{grid} call {n}: the shipped identity moved {name} {cols}")
            moved.update(cols)
        assert moved <= backed, (
            f"{grid} call {n}: columns {sorted(moved - backed)} moved but "
            "their rain did not evaporate to nothing")
        for c in backed:
            # The restored column carries no condensate tendency at all
            # beyond the float round trip every column has.
            for name, qname in (("rqccuten", "qc"), ("rqicuten", "qi")):
                q = inputs[qname][:, c].astype(np.float32)
                ql = (q / (np.float32(1.0) + q)).astype(np.float64)
                rdelt = 1.0 / float(np.float32(dt) * np.float32(stepcu))
                expect = ((ql / (1.0 - ql) - q.astype(np.float64)) * rdelt
                          ).astype(np.float32)
                assert np.array_equal(_bits(got[name][:, c]), _bits(expect))


@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("ncol", (65, 96))
def test_a_chunked_launch_equals_a_single_one(grid, ncol):
    # The staged output kernel must use the final chunk's active width,
    # including a one-column tail, while the pool keeps its full stride.
    fx = _fixture(grid)
    dt, stepcu, _, inputs, want, stage = _call(fx, 0)
    inputs = {name: value[..., :ncol] for name, value in inputs.items()}
    want = {name: value[..., :ncol] for name, value in want.items()}
    stage = {name: value[..., :ncol] for name, value in stage.items()}
    whole = _run(inputs, dt, stepcu, flags=FLAG_WRF_QL_BACKOUT)
    per_column = 8 * (SCHEME.r8_per_level * int(fx["kx"]) + SCHEME.r8_extra)
    chunked = _run(inputs, dt, stepcu, flags=FLAG_WRF_QL_BACKOUT,
                   budget_bytes=per_column * 70)
    for name in whole:
        assert np.array_equal(_bits(whole[name]), _bits(chunked[name])), name
    if wrf_exact.ENABLED:
        assert not _bad_columns(chunked, want)
        assert not _bad_columns(_stage_view(chunked["stages"], int(fx["kx"])), stage)


@pytest.mark.gpu
@strict_words
def test_a_one_bit_change_in_a_kernel_constant_is_caught():
    """C0 (module_cu_osas.F:557, the autoconversion rate .002) moved by one
    unit in the last place of its float32 word: the parity check must fail."""
    import cupy as cp

    from woof.core.kernels import module_source

    src = module_source(SCHEME.module)
    old = "const R8 c0 = OSAS_F(.002);"
    assert src.count(old) == 1
    planted = src.replace(
        old, "const R8 c0 = R8((double)__int_as_float(__float_as_int(.002f) + 1));")
    module = cp.RawModule(code=planted, options=("-std=c++17",))
    fx = _fixture("k44")
    dt, stepcu, _, inputs, want, _ = _call(fx, 0)

    import woof.core.kernels as kernels
    real = kernels.load_module

    def fake(name):
        return module if name == SCHEME.module else real(name)

    tb = cc.gfs_funcphys_tables(SCHEME.module)
    assert tb is not None
    kernels.load_module = fake
    try:
        got = _run(inputs, dt, stepcu, flags=FLAG_WRF_QL_BACKOUT, stages=False)
    finally:
        kernels.load_module = real
    bad = _bad_columns(got, want)
    assert bad, "a one-ULP change to C0 went unnoticed by the oracle comparison"
    assert "raincv" in bad


@pytest.mark.gpu
def test_the_default_draw_is_the_host_replica_and_is_keyed_on_step_and_column():
    import cupy as cp

    ncol = 4099
    a = cp.asnumpy(cc.random_input(ncol, step=7, module_name=SCHEME.module))
    b = cp.asnumpy(cc.random_input(ncol, step=7, module_name=SCHEME.module))
    c = cp.asnumpy(cc.random_input(ncol, step=8, module_name=SCHEME.module))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.array_equal(a, cc.random_input_host(ncol, step=7))
    assert a.min() >= 0.0 and a.max() < 1.0
    assert abs(a.mean() - 0.5) < 0.02
    strict = cp.asnumpy(cc.random_input(ncol, step=7, module_name=SCHEME.module,
                                        strict_value=0.1))
    assert np.all(strict == 0.1)


@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
@strict_words
def test_the_default_arm_is_wrfs_arithmetic_on_its_own_draw(grid):
    """The default arm differs from WRF in where XKT2 comes from and in the
    back-out, nothing else: fed the counter-based draw it makes itself, the
    oracle and the shipped kernel agree word for word outside the backed-out
    columns."""
    import cupy as cp

    fx = _fixture(f"{grid}-default")
    ncol = int(fx["im"]) * int(fx["jm"])
    for n in range(int(fx["ncall"])):
        dt, stepcu, itimestep, inputs, want, stage = _call(fx, n)
        draw = cp.asnumpy(cc.random_input(ncol, step=itimestep,
                                          module_name=SCHEME.module))
        assert np.array_equal(draw, inputs["xkt2"]), "the device draw moved"
        got = _run(inputs, dt, stepcu, flags=SCHEME.flags, stages=False)
        backed = set(_backout_columns(inputs, stage).tolist())
        for name, cols in _bad_columns(got, want).items():
            assert name in ("rqccuten", "rqicuten") and set(cols) <= backed, (
                f"{grid} default call {n}: {name} differs in columns {cols}")

@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
def test_production_strict_launch_uses_wrfs_backout_without_test_flags(grid):
    from woof import wrf_exact
    if not wrf_exact.ENABLED:
        pytest.skip("this production-arm check requires GPUWM_WRF_EXACT=1")
    fx = _fixture(grid)
    for n in range(int(fx["ncall"])):
        dt, stepcu, _, inputs, want, stage = _call(fx, n)
        got = _run(inputs, dt, stepcu, flags=None)
        assert not _bad_columns(got, want)
        assert not _bad_columns(_stage_view(got["stages"], int(fx["kx"])), stage)

@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
def test_the_full_strict_callable_reads_injected_draws(grid):
    import cupy as cp
    from types import SimpleNamespace
    from woof import wrf_exact
    if not wrf_exact.ENABLED:
        pytest.skip("injected draws are the strict interface")
    fx = _fixture(grid)
    dt, stepcu, _, inputs, want, _ = _call(fx, 0)
    kx, jm, im = int(fx["kx"]), int(fx["jm"]), int(fx["im"])
    dev = {n: cp.asarray(a) for n, a in inputs.items()}
    shape = (kx, jm, im)
    state = SimpleNamespace(
        p=dev["p"].reshape(shape), w=dev["w"].reshape(kx + 1, jm, im),
        qc=dev["qc"].reshape(shape), qi=dev["qi"].reshape(shape))
    atmosphere = {n: dev[key].reshape(shape) for n, key in (
        ("temperature", "t"), ("qv", "qv"), ("qc", "qc"), ("qi", "qi"),
        ("u", "u"), ("v", "v"), ("pressure", "p"), ("exner", "pi"),
        ("rho", "rho"), ("dz", "dz8w"))}
    atmosphere["p_interface"] = dev["p8w"].reshape(kx + 1, jm, im)
    cfg = SimpleNamespace(dt=dt, mp_physics=8)
    callable_ = cc.ColumnCumulus(95)
    callable_.bind_driver(SimpleNamespace(stepcu=stepcu))
    callable_.inject_random(dev["xkt2"])
    result = callable_(atmosphere=atmosphere,
                       fields={"xland": dev["xland"]}, state=state, cfg=cfg)
    got = {name: cp.asnumpy(getattr(result, name)).reshape(kx, jm * im)
           for name in cc.LEVEL_OUTPUTS}
    got["raincv"] = cp.asnumpy(result.raincv).reshape(-1)
    got["pratec"] = cp.asnumpy(result.pratec).reshape(-1)
    got["htop"] = cp.asnumpy(callable_.last_htop).reshape(-1)
    got["hbot"] = cp.asnumpy(callable_.last_hbot).reshape(-1)
    assert not _bad_columns(got, want)
    assert result.nca_seconds is None and result.persistent_pratec

@pytest.mark.gpu
@pytest.mark.parametrize("injected", (False, True))
def test_osas_full_state_restart_roundtrip_and_continuation(tmp_path, injected, monkeypatch):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _physics_state, _assert_restart_equal
    from datetime import datetime, timezone
    import hashlib

    # Pin only archive provenance so the whole archive can be compared as bytes.
    # The real model, writer, reader and numeric member payloads are unchanged.
    class ProvenanceTimeMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, datetime)

    class FixedProvenanceTime(datetime, metaclass=ProvenanceTimeMeta):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 8, tzinfo=timezone.utc)

    monkeypatch.setattr(restart, "datetime", FixedProvenanceTime)

    def fresh():
        state, cfg, driver = _physics_state(
            cp, cu_physics=95, dx=12000., dy=12000., ztop=20000.,
            nz=29, cudt_minutes=1.)
        if injected:
            driver.cumulus_callable.inject_random(
                cp.asarray(cc.random_input_host(cfg.nx * cfg.ny, step=19)))
        return state, cfg, driver

    state_a, cfg_a, _ = fresh()
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz", state_a, cfg_a)
    state_b, cfg_b, _ = fresh()
    run_steps(state_b, cfg_b, 20)
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)
    state_c, cfg_c, driver_c = fresh()
    restart.restore_restart(mid, state_c, cfg_c)
    roundtrip = restart.write_restart(tmp_path / "roundtrip.npz", state_c, cfg_c)
    _assert_restart_equal(roundtrip, mid)
    assert roundtrip.read_bytes() == mid.read_bytes(), "restart archive roundtrip"
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz", state_c, cfg_c)
    _assert_restart_equal(resumed, reference)
    assert resumed.read_bytes() == reference.read_bytes(), "restart archive continuation"
    import os
    receipt_root = os.environ.get("WOOF_OSAS_RECEIPTS")
    if receipt_root:
        from woof import wrf_exact
        arm = "strict" if wrf_exact.ENABLED else "default"
        root = Path(receipt_root)
        root.mkdir(parents=True, exist_ok=True)
        receipt = {
            "arm": arm, "injected": injected, "split_steps": [20, 20],
            "uninterrupted_steps": 40, "archive_bytes_identical": True,
            "provenance_stamp": "2026-10-08T00:00:00+00:00",
            "archive_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (mid, roundtrip, reference, resumed)},
            "members": {}
        }
        with np.load(reference, allow_pickle=False) as data:
            for name in data.files:
                receipt["members"][name] = {
                    "dtype": str(data[name].dtype), "shape": list(data[name].shape),
                    "sha256": hashlib.sha256(data[name].tobytes()).hexdigest()}
        (root / f"restart-{arm}-injected-{int(injected)}.json").write_text(
            json.dumps(receipt, indent=2) + "\n")
    assert driver_c.call_counts["cumulus"] >= 6
    header = restart.read_restart_header(mid)
    for name in ("scratch/cu_pratec", "scratch/cu_rthcuten",
                 "cumulus/last_htop", "cumulus/last_hbot"):
        assert name in header["array_manifest"]
    if injected:
        assert "cumulus/_injected" in header["array_manifest"]
