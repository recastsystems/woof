"""Grell-Devenyi (cu_physics = 93): the kernel against WRF's own object.

The oracle is tools/gd_wrf461_oracle: run_gd.F90 drives GRELLDRV of WRF
v4.6.1's compiled module_cu_gd.o (the wrf.exe objects of the strict and
scalar builds, word-identical to each other) over 120 single-call columns,
a 12 x 11 non-periodic domain (the edge band) and a 6-step stepcu = 3
cadence run, each with F_QI on and off.  Two answer sets are committed:

* ``stock`` -- WRF as shipped.  The kernel's ``wrf_closure_count = 1`` arm
  must reproduce every output word;
* ``fixed`` -- the same source with patches/gd-closure-count.patch, the
  optional arithmetic experiment (``wrf_closure_count = 0``), never the default.

"0 ULP" here is every word: tendencies, GD_CLOUD/GD_CLOUD2, RAINCV, PRATEC,
RAINC, HTOP/HBOT, the eight APR_* accumulators, ktop_deep and all 288
ensemble words per column (+0 and -0 compare equal; nothing else is
forgiven).

Tiers: the CPU tier compiles the exact gd.cu as C++ (g++, skipped where
absent) and grades it; the GPU tier launches it through the adapter's own
no-FTZ compile route.  Run the GPU tier once plainly (the default arm) and
once under GPUWM_WRF_EXACT=1 (the strict arm); both must be 0 ULP.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE_TOOLS = ROOT / "tools" / "gd_wrf461_oracle"
FIXTURES = ROOT / "tests" / "data" / "oracles" / "gd"
KERNEL = ROOT / "woof" / "core" / "kernels" / "gd.cu"
sys.path.insert(0, str(ORACLE_TOOLS))
import gd_cases  # noqa: E402
from validate_kernel import compare  # noqa: E402

NAMES = ("columns", "edge", "cadence")
ARMS = (("stock", 1), ("fixed", 0))


def _fixture_input(tmp_path: Path, name: str, fqi: int) -> Path:
    """The committed input with its F_QI header word set (byte 28)."""
    raw = bytearray(gzip.decompress((FIXTURES / f"gd-{name}.in.gz").read_bytes()))
    raw[28:32] = int(fqi).to_bytes(4, "little")
    out = tmp_path / f"gd-{name}-fqi{fqi}.in"
    out.write_bytes(bytes(raw))
    return out


def _oracle(tmp_path: Path, name: str, fqi: int, arm: str) -> Path:
    out = tmp_path / f"gd-{name}-fqi{fqi}.{arm}.out"
    out.write_bytes(gzip.decompress(
        (FIXTURES / f"gd-{name}-fqi{fqi}.{arm}.out.gz").read_bytes()))
    return out


def _steps(path: Path, name: str):
    inp = gd_cases.read_input(next(path.parent.glob(f"gd-{name}-fqi*.in")))
    return gd_cases.read_output(path, inp["nx"], inp["ny"], inp["nz"],
                                inp["nsteps"])


# ---------------------------------------------------------------- fixtures
def test_fixture_files_match_their_recorded_digests():
    sums = {}
    for line in (FIXTURES / "SHA256SUMS").read_text().splitlines():
        digest, fname = line.split()
        sums[fname] = digest
    assert len(sums) == 3 + 12
    for fname, digest in sums.items():
        data = gzip.decompress((FIXTURES / (fname + ".gz")).read_bytes())
        assert hashlib.sha256(data).hexdigest() == digest, fname


def test_the_fixtures_exercise_convection_and_the_edge_band(tmp_path):
    _fixture_input(tmp_path, "columns", 1)
    steps = _steps(_oracle(tmp_path, "columns", 1, "stock"), "columns")
    raining = int(np.count_nonzero(steps[0]["pratec"] > 0))
    assert raining >= 64, raining
    _fixture_input(tmp_path, "edge", 1)
    edge = _steps(_oracle(tmp_path, "edge", 1, "stock"), "edge")[0]
    band = gd_cases.band_flags(12, 11, False, False)
    assert np.all(edge["pratec"][band < 2] == 0)
    assert np.count_nonzero(edge["pratec"][band == 2] > 0) >= 8


def test_the_closure_count_fix_touches_water_columns_only(tmp_path):
    for name in NAMES:
        inp = gd_cases.read_input(_fixture_input(tmp_path, name, 1))
        stock = _steps(_oracle(tmp_path, name, 1, "stock"), name)[-1]
        fixed = _steps(_oracle(tmp_path, name, 1, "fixed"), name)[-1]
        moved = stock["pratec"].view(np.uint32) != fixed["pratec"].view(np.uint32)
        for f in gd_cases.LEVEL_OUTPUTS:
            moved |= np.any(stock[f].view(np.uint32)
                            != fixed[f].view(np.uint32), axis=1)
        xland = inp["steps"][0]["xland"]
        assert not np.any(moved & (xland < 1.5)), name
        if name == "columns":
            assert np.count_nonzero(moved & (xland > 1.5)) >= 8


# ---------------------------------------------------------------- CPU tier
@pytest.fixture(scope="module")
def host_binary(tmp_path_factory):
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ not available for the host-compiled kernel")
    out = tmp_path_factory.mktemp("gdhost") / "gd_host"
    subprocess.run([gxx, "-std=c++17", "-O2", "-ffp-contract=off",
                    f"-I{ROOT / 'woof' / 'core' / 'kernels'}", "-o", str(out),
                    str(ORACLE_TOOLS / "gd_host_harness.cpp")], check=True)
    return out


def _host_run(binary, tmp_path, name, fqi, count, kernel_dir=None):
    inp = _fixture_input(tmp_path, name, fqi)
    out = tmp_path / f"host-{name}-{fqi}-{count}.out"
    subprocess.run([str(binary), str(inp), str(out), str(count)], check=True)
    return out


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("fqi", (1, 0))
@pytest.mark.parametrize("arm,count", ARMS)
def test_host_compiled_kernel_is_wrf_word_for_word(host_binary, tmp_path,
                                                   name, fqi, arm, count):
    got = _steps(_host_run(host_binary, tmp_path, name, fqi, count), name)
    want = _steps(_oracle(tmp_path, name, fqi, arm), name)
    bad = {k: v for k, v in compare(want, got).items() if v[0]}
    assert not bad, bad


def _mutated_binary(tmp_path, old, new):
    src = KERNEL.read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    kdir = tmp_path / "kernels"
    shutil.copytree(KERNEL.parent, kdir,
                    ignore=shutil.ignore_patterns("__pycache__", "*.py"))
    (kdir / "gd.cu").write_text(src.replace(old, new), encoding="utf-8")
    out = tmp_path / "gd_host_mut"
    subprocess.run(["g++", "-std=c++17", "-O2", "-ffp-contract=off",
                    f"-I{kdir}", "-o", str(out),
                    str(ORACLE_TOOLS / "gd_host_harness.cpp")], check=True)
    return out


def test_a_planted_one_bit_closure_change_is_caught(host_binary, tmp_path):
    # Closure 2 of the ensemble is .9 x closure 1; flip the last bit of .9
    # (0x3f666666 -> 0x3f666667).  The grade must see it.
    mutant = _mutated_binary(
        tmp_path, "xff_ens3[2] = FMUL(.9f, xff_ens3[1]);",
        "xff_ens3[2] = FMUL(__uint_as_float(0x3f666667u), xff_ens3[1]);")
    got = _steps(_host_run(mutant, tmp_path, "columns", 1, 0), "columns")
    want = _steps(_oracle(tmp_path, "columns", 1, "fixed"), "columns")
    bad = {k: v for k, v in compare(want, got).items() if v[0]}
    assert "xf_ens" in bad and "pratec" in bad, bad


# ------------------------------------------------- source/metadata checks
def test_the_workspace_slot_count_is_the_kernels():
    src = KERNEL.read_text(encoding="utf-8")
    body = src[src.index("#define GD_SLOT(name)"):src.index("#undef GD_SLOT")]
    slots = len(re.findall(r"GD_SLOT\(\w+\)", body)) - 1   # minus the define
    from woof.core.gf import GD_NSLOT
    assert slots == GD_NSLOT
    assert f"#define GD_NSLOT {GD_NSLOT}" in src


def test_the_adapter_band_is_the_oracles():
    from woof.core.gf import gd_band_flags
    for ny, nx, px, py in ((11, 12, False, False), (1, 120, True, True),
                           (20, 9, True, False), (9, 9, False, True)):
        assert np.array_equal(gd_band_flags(ny, nx, px, py),
                              gd_cases.band_flags(nx, ny, px, py))


def test_93_is_admitted_by_table_rows():
    from woof.config import (CU_SCHEMES, CUMULUS_ADVECTIVE_FORCING_SCHEMES,
                              CUMULUS_QC_QI_SCHEMES, GRELL_FAMILY_CU_SCHEMES)
    from woof.config import CU_STOCK_CALLABLES
    from woof.physics_registry import physics_registry
    assert 93 in CU_SCHEMES and 93 in CUMULUS_ADVECTIVE_FORCING_SCHEMES
    assert 93 in CUMULUS_QC_QI_SCHEMES and 93 in GRELL_FAMILY_CU_SCHEMES
    assert CU_STOCK_CALLABLES[93] == "woof.core.gf.GrellDevenyi"
    row = physics_registry()["components"]["cumulus"]["options"]["grell-devenyi"]
    assert row["selectors"] == {"cu_physics": 93}
    assert row["consumers"]["stock_callable_class"] == "woof.core.gf.GrellDevenyi"
    assert row["consumers"]["restart_algorithm_identity"].startswith(
        "grell-devenyi-wrf461")


def test_validate_run_config_admits_wrf_s_gd_settings():
    from dataclasses import replace

    from woof.config import RunConfig, validate_run_config
    base = RunConfig(nx=12, ny=12, nz=40, dx=12000.0, dy=12000.0, ztop=16000.0, dt=60.0, run_seconds=0.0, moist=True, mp_physics=8, cu_physics=93, cudt_minutes=0.0)
    validate_run_config(base)
    # WRF has no PBL requirement for GD and honours a nonzero cudt.
    validate_run_config(replace(base, bl_pbl_physics=0, sf_sfclay_physics=0,
                                sf_surface_physics=0, cudt_minutes=5.0))
    with pytest.raises(ValueError, match="cu_physics is 3 or 5"):
        validate_run_config(replace(base, clos_choice=2))


def test_the_ensemble_dimensions_refuse_with_their_breakage():
    from woof.config import GRELL_ENSEMBLE_DEFAULTS, grell_ensemble_refusal
    for key, default in GRELL_ENSEMBLE_DEFAULTS.items():
        assert grell_ensemble_refusal(key, default) is None
        why = grell_ensemble_refusal(key, default + 1)
        assert why and ("module_cu_gd.F" in why or "massflx_stats" in why)


# ---------------------------------------------------------------- GPU tier
def _gpu():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:                          # noqa: BLE001
        pytest.skip(f"no CUDA device: {exc}")
    return cp


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("fqi", (1, 0))
@pytest.mark.parametrize("arm,count", ARMS)
def test_cuda_kernel_is_wrf_word_for_word(tmp_path, name, fqi, arm, count):
    cp = _gpu()
    from woof.core.gf import gd_launch

    inp = gd_cases.read_input(_fixture_input(tmp_path, name, fqi))
    nz = inp["nz"]

    def run(lvin, scin, iin):
        ens = cp.empty((lvin.shape[0], 2 * 144), dtype=cp.float32)
        lev, sca, isc = gd_launch(cp.asarray(lvin), cp.asarray(scin),
                                  cp.asarray(iin), nz=nz,
                                  wrf_closure_count=count, ens=ens)
        return (cp.asnumpy(lev), cp.asnumpy(sca), cp.asnumpy(isc),
                cp.asnumpy(ens))

    got = gd_cases.emulate_driver(inp, run)
    want = _steps(_oracle(tmp_path, name, fqi, arm), name)
    bad = {k: v for k, v in compare(want, got).items() if v[0]}
    assert not bad, bad


def test_cuda_planted_one_bit_change_is_caught(tmp_path):
    cp = _gpu()
    from woof import nvrtc_ptx_cache as compiler
    from woof.core.gf import GD_MODULE_OPTIONS, gd_module_source

    src = gd_module_source()
    old = "xff_ens3[2] = FMUL(.9f, xff_ens3[1]);"
    assert src.count(old) == 1
    src = src.replace(old, "xff_ens3[2] = FMUL(__uint_as_float(0x3f666667u), xff_ens3[1]);")
    ptx, _ = compiler.compile_using_nvrtc(src, GD_MODULE_OPTIONS, None, "gd_mut.cu")
    mod = cp.cuda.function.Module()
    mod.load(ptx.encode() if isinstance(ptx, str) else ptx)
    fn = mod.get_function("gd_grelldrv")
    inp = gd_cases.read_input(_fixture_input(tmp_path, "columns", 1))
    nz = inp["nz"]

    def run(lvin, scin, iin):
        from woof.core import gf
        ncol = lvin.shape[0]
        lev = cp.empty((ncol, 6, nz), cp.float32)
        sca = cp.empty((ncol, 13), cp.float32)
        isc = cp.empty((ncol, 5), cp.int32)
        ens = cp.empty((ncol, 288), cp.float32)
        ws = cp.empty(gf.gd_workspace_floats(nz, ncol), cp.float32)
        fn(((ncol + 63) // 64,), (64,),
           (cp.asarray(lvin), cp.asarray(scin), cp.asarray(iin), lev, sca,
            isc, ens, ws, gf.GD_XLV, gf.GD_CP, gf.GD_G, gf.GD_RV,
            np.int32(0), np.int32(ncol), np.int32(nz)))
        return (cp.asnumpy(lev), cp.asnumpy(sca), cp.asnumpy(isc),
                cp.asnumpy(ens))

    got = gd_cases.emulate_driver(inp, run)
    want = _steps(_oracle(tmp_path, "columns", 1, "fixed"), "columns")
    bad = {k: v for k, v in compare(want, got).items() if v[0]}
    assert "xf_ens" in bad and "pratec" in bad, bad

def test_gd_uses_shared_noftz_route():
    import inspect
    from woof.core import gf
    from woof.core.kernels import module_options, _NO_FTZ_MODULES

    assert 'load_module("gd")' in inspect.getsource(gf._gd_module)
    assert "gd" in _NO_FTZ_MODULES
    assert set(module_options("gd")) == {"-std=c++17", "--ftz=false", "--fmad=false"}

def test_grade_refuses_a_missing_step_or_field_shape(tmp_path):
    _fixture_input(tmp_path, "cadence", 1)
    want = _steps(_oracle(tmp_path, "cadence", 1, "stock"), "cadence")
    with pytest.raises(ValueError, match="step counts differ"):
        compare(want, want[:-1])
    truncated = [dict(step) for step in want]
    truncated[0]["rthcuten"] = truncated[0]["rthcuten"][:, :-1]
    with pytest.raises(ValueError, match="shapes differ"):
        compare(want, truncated)


@pytest.mark.parametrize("cudt", (0.0, 3.0))
def test_gd_engine_dispatch_and_cadence(cudt, tmp_path):
    cp = _gpu()
    from woof.config import RunConfig, validate_run_config
    from woof.core.gf import GrellDevenyi
    from woof.core.physics import initialize_physics
    from test_gf_engine_smoke import _state_for

    cfg = validate_run_config(RunConfig(
        nx=12, ny=12, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
        dt=60.0, run_seconds=0.0, moist=True, mp_physics=10,
        cu_physics=93, cudt_minutes=cudt))
    state = _state_for(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=290.0)
    assert isinstance(driver.cumulus_callable, GrellDevenyi)
    # Initial history is written before any physics call.
    from woof.io.wrf_output_schema import GD_OUTPUT_FIELDS
    initial = driver.output_fields()
    for row in GD_OUTPUT_FIELDS.values():
        assert row.netcdf_name in initial
        assert bool(cp.all(initial[row.netcdf_name] == 0))
    for step in range(4):
        state.elapsed_seconds = step * cfg.dt
        tendencies = driver.compute(state, cfg)
        assert bool(cp.isfinite(tendencies.rtheta).all())
        assert bool(cp.isfinite(tendencies.rqv).all())
    assert driver.call_counts["cumulus"] == (4 if cudt == 0.0 else 2)
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import GD_OUTPUT_FIELDS
    output=driver.output_fields()
    priced=produced_history_shapes(cfg)
    for key,row in GD_OUTPUT_FIELDS.items():
        assert row.netcdf_name in output and row.netcdf_name in priced
        assert output[row.netcdf_name].shape==priced[row.netcdf_name]
        assert cp.shares_memory(output[row.netcdf_name], driver.cumulus_callable.cu_diag[key])
    assert output["KTOP_DEEP"].dtype==cp.int32
    from woof.io.wrfout import WrfoutWriter
    from woof.netcdf_bridge import Dataset
    frame={row.netcdf_name:cp.asnumpy(output[row.netcdf_name]) for row in GD_OUTPUT_FIELDS.values()}
    target=tmp_path/"gd-diagnostics.nc"
    with WrfoutWriter(target,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,dx=cfg.dx,dy=cfg.dy) as writer:
        writer.write_frame("2026-10-08_00:00:00",frame)
    with Dataset(target) as ds:
        for row in GD_OUTPUT_FIELDS.values():
            variable=ds.variables[row.netcdf_name]
            variable.set_auto_maskandscale(False)
            read=np.asarray(variable[:],dtype=row.dtype)[0]
            assert np.array_equal(read.view(np.uint32),frame[row.netcdf_name].view(np.uint32))
            assert variable.description==row.description
            assert variable.units==row.units
            assert np.dtype(variable.dtype)==np.dtype(row.dtype)
    diag = driver.cumulus_callable.cu_diag
    for name in ("gd_cloud", "gd_cloud2", "ktop_deep", "apr_gr", "apr_w"):
        assert bool(cp.isfinite(diag[name]).all()), name
    assert bool(cp.isfinite(driver.rainc).all())

def test_gd_driver_constants_use_the_shared_wrf_constants():
    from woof.core import constants, gf
    got=np.array([gf.GD_XLV,gf.GD_CP,gf.GD_G,gf.GD_RV],np.float32)
    want=np.array([constants.XLV,constants.CP,constants.G,constants.RV],np.float32)
    assert np.array_equal(got.view(np.uint32),want.view(np.uint32))
    assert got.view(np.uint32).tolist()==[0x4a189680,0x447b2000,0x411cf5c3,0x43e6cccd]

def test_gd_has_a_named_composition_with_scoped_evidence():
    from woof.physics_registry import physics_registry
    row=physics_registry()["templates"]["thompson-mp8-ysu-mm5-noah-grell-devenyi-rte-rrtmgp-v1"]
    assert row["components"]["cumulus"]=="grell-devenyi"
    assert row["maturity"]=="implemented-unverified"
    assert physics_registry()["components"]["cumulus"]["options"]["grell-devenyi"]["scientific_evidence"] == "none"


def test_gd_checkpoint_preserves_held_rates_and_diagnostics(tmp_path):
    """Off-cadence restore retains canonical APR and diagnostic state."""
    cp = _gpu()
    from woof.config import RunConfig, validate_run_config
    from woof.core.physics import initialize_physics
    from woof.io import restart
    from test_gf_engine_smoke import _state_for
    from test_restart import _assert_restart_equal
    cfg = validate_run_config(RunConfig(
        nx=12, ny=12, nz=16, dx=12000., dy=12000., ztop=9000.,
        dt=60., run_seconds=0., moist=True, mp_physics=10,
        cu_physics=93, cudt_minutes=3.))
    def fresh():
        state = _state_for(cfg)
        driver = initialize_physics(state, cfg, landmask=1., tsk=290.)
        # APR are cumulative native diagnostics, including when a column
        # subsequently declines convection. Distinct nonzero words expose
        # a missing carrier even in this stable atmosphere.
        driver.fields['gd_sfc'][4:12] = cp.arange(1, 9, dtype=cp.float32)[:, None, None] * .125
        driver.fields['gd_sfc'][3] = cp.float32(2.125)
        return state, driver
    def calls(state, driver, start, end):
        for step in range(start, end):
            state.elapsed_seconds = step * cfg.dt
            driver.compute(state, cfg)
    state, driver = fresh()
    calls(state, driver, 0, 2)
    mid = restart.write_restart(tmp_path/'mid.npz', state, cfg)
    calls(state, driver, 2, 4)
    reference = restart.write_restart(tmp_path/'reference.npz', state, cfg)
    resumed, restored = fresh()
    restart.restore_restart(mid, resumed, cfg)
    calls(resumed, restored, 2, 4)
    result = restart.write_restart(tmp_path/'resumed.npz', resumed, cfg)
    _assert_restart_equal(result, reference)
    assert bool(cp.any(restored.fields['gd_sfc'][4:12] != 0.))
    assert bool(cp.all(restored.fields['gd_sfc'][3] == cp.float32(2.125)))
    assert restored.call_counts['cumulus'] == 2


@pytest.mark.parametrize('feedback', [0, 1])
def test_gd_runs_across_native_nest_coupling(feedback):
    """A real GD driver executes on each side of both coupling modes."""
    cp = _gpu()
    from types import SimpleNamespace as NS
    from woof.config import RunConfig, validate_run_config
    from woof.core.physics import initialize_physics
    from woof.core.model import FeedbackScratch
    from woof.core.nest import NestCoupler
    from test_gf_engine_smoke import _state_for
    from test_nest_coupler import _clock
    nodes = []
    for grid_id, size, dt, dx in ((1, 24, 9., 12000.), (2, 30, 3., 4000.)):
        cfg = validate_run_config(RunConfig(
            nx=size, ny=size, nz=16, dx=dx, dy=dx, ztop=9000.,
            dt=dt, run_seconds=0., moist=True, mp_physics=10,
            cu_physics=93, cudt_minutes=0., nested=grid_id == 2,
            specified=grid_id == 1, grid_id=grid_id))
        state = _state_for(cfg)
        driver = initialize_physics(state, cfg, landmask=1., tsk=290.)
        driver.fields['gd_sfc'][4:12] = cp.float32(grid_id * .125)
        node = NS(cfg=NS(grid_id=grid_id, parent_id=grid_id-1,
                         parent_grid_ratio=1 if grid_id == 1 else 3,
                         i_parent_start=5, j_parent_start=5, run=cfg),
                  state=state, driver=driver,
                  parent=None if grid_id == 1 else nodes[0],
                  clock=_clock(grid_id, grid_id-1,
                               step_ticks=9 if grid_id == 1 else 3,
                               dt=dt, advanced=grid_id == 1))
        nodes.append(node)
    parent, child = nodes
    child.clock.prepare_step()
    coupler = NestCoupler(child, feedback=feedback)
    parent.driver.compute(parent.state, parent.cfg.run)
    before = parent.driver.fields['gd_sfc'].copy()
    coupler.force(child)
    for step in range(3):
        child.state.elapsed_seconds = step * child.cfg.run.dt
        child.driver.compute(child.state, child.cfg.run)
        child.clock.advance()
        if step < 2:
            child.clock.prepare_step()
    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)
    assert coupler.force_count == 1
    assert coupler.feedback_count == feedback
    assert parent.driver.call_counts['cumulus'] == 1
    assert child.driver.call_counts['cumulus'] == 3
    # Native GD diagnostics are domain-local, not child prognostics fed
    # into the parent by the ordinary mass/moisture feedback transaction.
    cp.testing.assert_array_equal(parent.driver.fields['gd_sfc'], before)
    assert bool(cp.isfinite(parent.state.thp).all())
    assert bool(cp.isfinite(child.driver.fields['gd_lev']).all())
