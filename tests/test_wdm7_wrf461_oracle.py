"""Production WDM7 column checks against the pinned box-local WRF oracle."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools/wdm7_wrf461_oracle"
def tool(name):
    spec = importlib.util.spec_from_file_location("wdm7_test_" + name,
                                                 TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wdm7_native_inventory():
    from woof.config import MP_PHYSICS_ACCEPTED, MYNN_QN_FLAG_SPECIES
    from woof.wrf_physics_inventory import stock_wrf_physics_inventory
    assert 26 in MP_PHYSICS_ACCEPTED
    assert MYNN_QN_FLAG_SPECIES[26] == ("qnc",)
    inv = stock_wrf_physics_inventory(26)
    fields = {f.netcdf_name: f.state_key for f in inv.wrfinput_fields}
    assert fields["QHAIL"] == "qh" and "QNHAIL" not in fields
    assert {n: fields[n] for n in ("QNCCN", "QNCLOUD", "QNRAIN")} == {
        "QNCCN": "nn", "QNCLOUD": "nc", "QNRAIN": "nr"}
    assert all(f.units == "  kg(-1)" for f in inv.wrfinput_fields
               if f.collection == "scalar")

def test_shared_wdm_controls_remain_bound_to_restart_identity():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.core.model import restart_identity_payload
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold
    scaffold = load_scaffold()
    for mp in (16, 26):
        cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0,
                        ztop=10000.0, dt=1.0, run_seconds=10.0,
                        moist=True, mp_physics=mp, wdm6_ccn_conc=5.0e8)
        exp = replace(scaffold, domains=(
            replace(scaffold.domains[0], run=cfg),))
        run = restart_identity_payload(exp)["domains"][0]["run"]
        assert run["wdm6_ccn_conc"] == 5.0e8
        assert run["wdm6_hail_opt"] == 0

def test_wdm7_history_consumes_the_runtime_reflectivity_menu():
    from woof import prepared_single_domain_forecast as prepared
    from woof.core.physics_inventory import REFL_10CM_MICROPHYSICS
    assert prepared.REFL_10CM_MICROPHYSICS is REFL_10CM_MICROPHYSICS
    assert 26 in prepared.REFL_10CM_MICROPHYSICS

def test_wdm7_legacy_radiation_reads_its_ice_and_snow_membership():
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
    assert legacy_cloud_fraction_flags(26) == (True, True)
    assert legacy_cloud_fraction_flags(26) == legacy_cloud_fraction_flags(16)

def test_wdm7_uses_its_own_translation_unit():
    from woof.core.kernels import module_source_int_defines
    source = module_source_int_defines("wdm7", (("WDM6_KMAX", 80),))
    assert "void wdm7_column(" in source
    assert "#define WDM_HAIL 1" in source


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("config", ("dt20", "native_step2"))
def test_wdm7_matches_stock_fixture(tmp_path, config):
    import gzip
    from woof.numerics import is_strict
    if not is_strict():
        pytest.skip("bitwise native replay requires GPUWM_NUMERICS=strict")
    data = ROOT / "tests/data/oracles/wdm7"
    inp = tmp_path / f"{config}.in"
    ref = tmp_path / "reference.out"
    inp.write_bytes(gzip.decompress((data / f"{config}.in.gz").read_bytes()))
    ref.write_bytes(gzip.decompress((data / f"{config}.stock.out.gz").read_bytes()))
    run = tool("run_woof")
    out = tmp_path / "gpu.out"
    run.run(inp, out)
    io = tool("oracle_io")
    case = io.read_input(inp)
    got = io.read_output(out, case.ncol, case.nz, case.nsteps)
    want = io.read_output(ref, case.ncol, case.nz, case.nsteps)
    for (gl, gs), (wl, ws) in zip(got, want):
        for level, other in ((gl, wl), (gs, ws)):
            for name, value in level.items():
                expected = other[name]
                if name in ("re_cloud", "re_ice", "re_snow"):
                    expected = (expected * np.float32(1e6)).astype(np.float32)
                assert np.array_equal(value.view(np.uint32), expected.view(np.uint32)), name


@pytest.mark.gpu
@requires_gpu
def test_refreshed_rain_fall_speed_cannot_create_column_water(tmp_path, monkeypatch):
    """Real GFS step 283 created 23--24 kg/m2 under the native CFL schedule."""
    import functools
    import gzip
    from woof.core import wdm6

    inp = tmp_path / "gfs.in"
    inp.write_bytes(gzip.decompress((ROOT / "tests/data/oracles/wdm7/gfs-step283.in.gz").read_bytes()))
    io = tool("oracle_io")
    case = io.read_input(inp)
    # Exercise the production forecast transport selection through the raw
    # column harness, with its exact captured meteorology and timestep.
    monkeypatch.setattr(wdm6, "launch_wdm7", functools.partial(
        wdm6.launch_wdm7, positive_rain_transport=True))
    out = tmp_path / "fixed.out"
    tool("run_woof").run(inp, out)
    levels, surface = io.read_output(out, case.ncol, case.nz, 1)[0]
    species = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    weights = case.levels["den"].astype(np.float64) * case.levels["delz"]
    before = np.sum(weights * sum(case.levels[k].astype(np.float64)
                                 for k in species), axis=0)
    after = np.sum(weights * sum(levels[k].astype(np.float64)
                                for k in species), axis=0)
    rain = surface["rain"].ravel().astype(np.float64) - case.surface["rain"]
    # FP32 transport and phase transfers close within their accumulation
    # roundoff. Reverting the flux remedy misses by nearly the initial water.
    np.testing.assert_allclose(after + rain, before,
                               rtol=32 * np.finfo(np.float32).eps, atol=0)
    assert levels["qr"].max() < 0.01
