"""The ACM2 PBL (bl_pbl_physics = 7) against WRF v4.6.1's own words.

The fixtures (``tests/data/oracles/acm2``, PROVENANCE.md there) are WRF's
ACMPBL from the byte-pinned ``phys/module_bl_acm.F``, preprocessed and
compiled exactly as WRF's serial gfortran build compiles ``phys/`` and
byte-identical to the fixtures written by the referee ``wrf.exe``'s own
object (tools/acm2_wrf461_oracle/build.sh): 74 columns on 40 levels for six
consecutive steps from ``itimestep = 1`` (dt 60 s) and 74 columns on 60
levels for four steps from ``itimestep = 41`` (dt 180 s), each step's
exact float32 inputs recorded beside its outputs.

Graded on the card, every word: RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN,
RQCBLTEN, RQIBLTEN, EXCH_H, EXCH_M (every level), PBLH, KPBL, REGIME,
RMOL, compared as bit patterns (the sign of zero included), under strict
numerics. These constructed inputs do not establish real-forecast replay.
The fixture pins and the CPU-side checks are in
tests/test_acm2_integration.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from woof.verify.acm2_oracle import (STEP_OUT_2D, STEP_OUT_3D,
                                      load_fixture, load_inputs, ulp_distance)

ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(os.environ.get("ACM2_ORACLE_DIR")
              or ROOT / "tests/data/oracles/acm2")
FIXTURES = ("acm2-l40", "acm2-l60")

#: Kernel field name for each oracle output.
_KERNEL_NAME = {"rublten": "du", "rvblten": "dv", "rthblten": "dtheta",
                "rqvblten": "dqv", "rqcblten": "dqc", "rqiblten": "dqi"}


def _fixture(name):
    stem = ORACLE / name
    if not Path(f"{stem}.out.bin").exists():
        pytest.skip(f"ACM2 oracle fixture {stem} is absent")
    return load_fixture(stem)


# --------------------------------------------------------------------- card

def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    return cp


def _run_step(cp, fx, st, module=None):
    from woof.core.acm2pbl import acm2_pbl_step
    s3, s2 = fx.static3d, fx.static2d
    d = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))
    pblh = d(st["state2d"]["pblh"])
    regime = d(st["state2d"]["regime"])
    rmol = d(st["state2d"]["rmol"])
    exch_h = cp.zeros((fx.nz, fx.ncol), dtype=cp.float32)
    exch_m = cp.zeros((fx.nz, fx.ncol), dtype=cp.float32)
    out = acm2_pbl_step(
        u=d(st["state3d"]["u"]), v=d(st["state3d"]["v"]),
        theta=d(st["state3d"]["th"]), t=d(st["state3d"]["t"]),
        qv=d(st["state3d"]["qv"]), qc=d(st["state3d"]["qc"]),
        qi=d(st["state3d"]["qi"]), rho=d(s3["rho"]), dz8w=d(s3["dz8w"]),
        ust=d(s2["ust"]), hfx=d(s2["hfx"]), qfx=d(s2["qfx"]),
        wspd=d(s2["wspd"]), pblh=pblh, regime=regime, rmol=rmol,
        exch_h=exch_h, exch_m=exch_m, dtpbl=float(fx.constants["dtpbl"]),
        xtime=st["xtime"], ep1=fx.constants["ep1"], g=fx.constants["g"],
        rd=fx.constants["rd"], cpd=fx.constants["cpd"], module=module)
    got = {name: cp.asnumpy(out[_KERNEL_NAME[name]])
           for name in _KERNEL_NAME}
    got["exch_h"] = cp.asnumpy(exch_h)
    got["exch_m"] = cp.asnumpy(exch_m)
    got2 = {"pblh": cp.asnumpy(pblh), "regime": cp.asnumpy(regime),
            "rmol": cp.asnumpy(rmol)}
    return got, got2, cp.asnumpy(out["kpbl"])


def _grade(cp, fx, module=None):
    """Per field: (mismatched words, total words, max ULP)."""
    table = {name: [0, 0, 0] for name in (*STEP_OUT_3D, *STEP_OUT_2D,
                                          "kpbl")}
    for st in fx.steps:
        got3, got2, kpbl = _run_step(cp, fx, st, module)
        for name in STEP_OUT_3D:
            want = st["out3d"][name]
            bad = got3[name].view(np.uint32) != want.view(np.uint32)
            row = table[name]
            row[0] += int(bad.sum())
            row[1] += bad.size
            if bad.any():
                row[2] = max(row[2], int(ulp_distance(got3[name],
                                                      want).max()))
        for name in STEP_OUT_2D:
            want = st["out2d"][name]
            bad = got2[name].view(np.uint32) != want.view(np.uint32)
            row = table[name]
            row[0] += int(bad.sum())
            row[1] += bad.size
            if bad.any():
                row[2] = max(row[2], int(ulp_distance(got2[name],
                                                      want).max()))
        bad = kpbl != st["kpbl"]
        table["kpbl"][0] += int(bad.sum())
        table["kpbl"][1] += bad.size
        if bad.any():
            table["kpbl"][2] = max(table["kpbl"][2],
                                   int(np.abs(kpbl - st["kpbl"]).max()))
    return table


@pytest.mark.parametrize("name", FIXTURES)
def test_every_step_of_every_column_is_wrfs_words(name):
    cp = _cupy()
    fx = _fixture(name)
    table = _grade(cp, fx)
    receipt = os.environ.get("ACM2_PARITY_RECEIPT")
    if receipt:
        from woof import wrf_exact
        with open(receipt, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"fixture": name,
                                 "strict": bool(wrf_exact.ENABLED),
                                 "fields": table}) + "\n")
    bad = {k: v for k, v in table.items() if v[0]}
    assert not bad, f"{name}: mismatched (count, total, max ulp): {bad}"


def test_wrfs_fatal_is_raised_naming_the_column():
    """fatal.in.bin column 2 (flat 1) stops WRF with 'RIBX never exceeds
    RIC' (fatal.stderr.txt); the launcher raises naming that column."""
    cp = _cupy()
    from woof.core.acm2pbl import AcmFatal
    fx = load_inputs(ORACLE / "fatal")
    st = {"xtime": fx.xtime0,
          "state3d": {k: fx.static3d[k]
                      for k in ("u", "v", "th", "t", "qv", "qc", "qi")},
          "state2d": {k: fx.static2d[k] for k in ("pblh", "regime", "rmol")}}
    with pytest.raises(AcmFatal, match=r"flat 1\)"):
        _run_step(cp, fx, st)
    # the ordinary column alone runs
    keep = slice(0, 1)
    one = type(fx)(1, fx.nz, fx.nsteps, fx.xtime0, fx.constants,
                   {k: v[:, keep] for k, v in fx.static3d.items()},
                   {k: v[keep] for k, v in fx.static2d.items()}, [])
    st1 = {"xtime": fx.xtime0,
           "state3d": {k: v[:, keep] for k, v in st["state3d"].items()},
           "state2d": {k: v[keep] for k, v in st["state2d"].items()}}
    _run_step(cp, one, st1)


def test_a_planted_one_bit_change_is_caught():
    """RIC moved by one bit (0.25 -> 0x1.000002p-2) must fail the grade."""
    cp = _cupy()
    from woof.core import acm2pbl
    text = acm2pbl.module_source()
    line = "#define ACM_RIC     0.25f"
    assert text.count(line) == 1
    planted = text.replace(line, "#define ACM_RIC     0x1.000002p-2f")
    module = acm2pbl.compile_module(planted, cp.cuda.Device().id)
    table = _grade(cp, _fixture("acm2-l40"), module=module)
    mismatches = sum(v[0] for v in table.values())
    print("planted RIC bit: mismatched words", mismatches, "fields", table)
    receipt = os.environ.get("ACM2_PARITY_RECEIPT")
    if receipt:
        with open(receipt, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"control": "RIC-one-bit", "mismatches": mismatches,
                                 "fields": table}) + "\n")
    assert mismatches > 0


def test_shared_loader_preserves_subnormals_and_records_the_compiled_image():
    cp = _cupy()
    from woof.core.kernels import compile_no_ftz
    from woof.certify.kernel_manifest import kernel_manifest, source_sha256
    source = """extern "C" __global__ void rounded_probe(const float* a, float* b) {
        b[0] = a[3] * a[4];
        float product = a[0] * a[1];
        b[1] = product + a[2];
    }"""
    key = "acm2-test:rounded-probe"
    module = compile_no_ftz(source, module_key=key, filename="rounded_probe.cu",
                            device=cp.cuda.Device().id)
    values = cp.asarray([float.fromhex("0x1.000002p+0"),
                         float.fromhex("0x1.fffffep-1"), -1.0,
                         float.fromhex("0x1p-126"), 0.5], dtype=cp.float32)
    output = cp.empty(2, dtype=cp.float32)
    module.get_function("rounded_probe")((1,), (1,), (values, output))
    assert output.get().view(np.uint32).tolist() == [0x00400000, 0]
    entry = kernel_manifest()[key]
    assert entry["source_sha256"] == source_sha256(source)
    from woof.numerics import effective_options
    assert entry["options"] == list(effective_options(
        ("-std=c++17", "--fmad=false", "--ftz=false")))
    assert entry["compiled_image"]["status"] == "resolved"
    assert len(entry["compiled_image"]["sha256"]) == 64


def test_tile_boundary_and_top_exchange_values_are_preserved():
    """Cross the runtime's actual tile boundary with varied WRF columns."""
    cp = _cupy()
    from woof.core.physics_inventory import (ACM2_BLOCK,
                                              ACM2_TILE_BLOCKS_PER_SM)
    from woof.core.acm2pbl import acm2_pbl_step
    fx = _fixture("acm2-l40")
    st = fx.steps[1]
    tile = (cp.cuda.Device().attributes["MultiProcessorCount"]
            * ACM2_TILE_BLOCKS_PER_SM * ACM2_BLOCK)
    plane = (1, tile + 3)
    idx = np.arange(plane[1]) % fx.ncol
    d3 = lambda a: cp.asarray(np.ascontiguousarray(a[:, idx][:, None, :]))
    d2 = lambda a: cp.asarray(np.ascontiguousarray(a[idx][None, :]))
    pblh, regime, rmol = (d2(st["state2d"][n])
                          for n in ("pblh", "regime", "rmol"))
    eh = cp.full((fx.nz, *plane), np.float32(17.25))
    em = cp.full((fx.nz, *plane), np.float32(19.75))
    out = acm2_pbl_step(
        u=d3(st["state3d"]["u"]), v=d3(st["state3d"]["v"]),
        theta=d3(st["state3d"]["th"]), t=d3(st["state3d"]["t"]),
        qv=d3(st["state3d"]["qv"]), qc=d3(st["state3d"]["qc"]),
        qi=d3(st["state3d"]["qi"]), rho=d3(fx.static3d["rho"]),
        dz8w=d3(fx.static3d["dz8w"]), ust=d2(fx.static2d["ust"]),
        hfx=d2(fx.static2d["hfx"]), qfx=d2(fx.static2d["qfx"]),
        wspd=d2(fx.static2d["wspd"]), pblh=pblh, regime=regime, rmol=rmol,
        exch_h=eh, exch_m=em, dtpbl=float(fx.constants["dtpbl"]),
        xtime=st["xtime"])
    for name, field in _KERNEL_NAME.items():
        got = out[field].get()[:, 0, :]
        want = st["out3d"][name][:, idx]
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), name
    for name, a in (("exch_h", eh), ("exch_m", em)):
        assert np.array_equal(a.get()[:-1, 0, :].view(np.uint32),
                              st["out3d"][name][:-1, idx].view(np.uint32))
    assert bool((eh[-1] == np.float32(17.25)).all())
    assert bool((em[-1] == np.float32(19.75)).all())
    for name, a in (("pblh", pblh), ("regime", regime), ("rmol", rmol)):
        assert np.array_equal(a.get()[0].view(np.uint32),
                              st["out2d"][name][idx].view(np.uint32))
    assert np.array_equal(out["kpbl"].get()[0], st["kpbl"][idx])
    print("tile boundary: columns", plane[1], "tile", tile, "all words equal")


def test_the_runtime_frame_fits_the_advertised_bound():
    cp = _cupy()
    from woof.core.acm2pbl import _module, MODULE_KEY, CONSTANT_ASSET
    from woof.core.preflight import CHAINED_TRANSLATION_UNIT_FRAMES
    from woof.certify.kernel_manifest import kernel_manifest
    from woof.certify.compile_platform import compile_platform_fingerprint
    kernel = _module(cp.cuda.Device().id).get_function("acm2_pbl_column")
    driver = cp.cuda.driver
    frame = int(driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel.ptr))
    bound = CHAINED_TRANSLATION_UNIT_FRAMES["acm2pbl_composed"].max_local_size_bytes
    assert frame <= bound, (frame, bound)
    document = {"local_size_bytes": frame, "bound": bound,
                "constant_asset_sha256": CONSTANT_ASSET.sha256,
                "platform": compile_platform_fingerprint(),
                "manifest": kernel_manifest()[MODULE_KEY]}
    print("ACM2 frame", json.dumps(document, sort_keys=True))
    receipt = os.environ.get("ACM2_PARITY_RECEIPT")
    if receipt:
        with open(receipt, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"control": "frame", **document}) + "\n")


def _device_call(cp, fx):
    """Stage inputs before a stream capture, including the native fatal case."""
    s3, s2 = fx.static3d, fx.static2d
    d = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))
    args = {name: d(s3[key]) for name, key in (
        ("u", "u"), ("v", "v"), ("theta", "th"), ("t", "t"),
        ("qv", "qv"), ("qc", "qc"), ("qi", "qi"), ("rho", "rho"),
        ("dz8w", "dz8w"))}
    args.update({name: d(s2[name]) for name in (
        "ust", "hfx", "qfx", "wspd", "pblh", "regime", "rmol")})
    args.update(exch_h=cp.zeros((fx.nz, fx.ncol), dtype=cp.float32),
                exch_m=cp.zeros((fx.nz, fx.ncol), dtype=cp.float32),
                dtpbl=float(fx.constants["dtpbl"]), xtime=fx.xtime0)
    return args


def test_model_health_path_is_cuda_graph_capturable():
    """A healthy model call must issue no blocking status or finite read."""
    cp = _cupy()
    from woof.core.acm2pbl import acm2_pbl_step
    from woof.core.health_ledger import HealthLedger, deferring
    args = _device_call(cp, _fixture("acm2-l40"))
    ledger = HealthLedger(label="acm2 graph")
    stream = cp.cuda.Stream(non_blocking=True)
    with stream, deferring(ledger):
        acm2_pbl_step(**args)  # warm the kernel, allocator and ledger slot
    stream.synchronize()
    ledger.drain()
    with stream, deferring(ledger):
        stream.begin_capture()
        out = acm2_pbl_step(**args)
        graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    ledger.drain()
    assert all(bool(cp.isfinite(out[name]).all()) for name in _KERNEL_NAME.values())


def test_model_health_path_retains_native_fatal_until_drain():
    cp = _cupy()
    from woof.core.acm2pbl import AcmFatal, acm2_pbl_step
    from woof.core.health_ledger import HealthLedger, deferring
    args = _device_call(cp, load_inputs(ORACLE / "fatal"))
    ledger = HealthLedger(label="acm2 fatal")
    with deferring(ledger):
        acm2_pbl_step(**args)
    with pytest.raises(AcmFatal, match="RIBX never exceeds RIC"):
        ledger.drain()
