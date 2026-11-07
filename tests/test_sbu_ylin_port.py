"""SBU-YLin native formulas, IO, unadvected RIMI, and strict GPU replay."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
ORACLE = REPO / "tools" / "sbu_ylin_wrf461_oracle"
FIXTURE = REPO / "woof" / "data" / "sbu_ylin" / "oracle"
WRF_SCHEME_SHA256 = (
    "d06a0b8fa01465c09ac218f94e2afc409b4ef3a65cf5782fc79cbe8a9d25e06f")

# Fast mode permits contraction. Strict mode alone proves native word identity.
DEFAULT_ATOL = {"th": 5.e-4, "qv": 1.e-8, "qc": 1.e-8, "qr": 1.e-8,
                "qi": 1.e-8, "qs": 1.e-8, "rimi": 1.e-5,
                "rainnc": 1.e-5, "rainncv": 1.e-5}


def _fixture_sha(name: str) -> str:
    return hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()


# --------------------------------------------------------------- the oracle

def test_the_oracle_pins_the_wrf_scheme_source():
    assert (ORACLE / "module_mp_sbu_ylin.F.sha256").read_text().split()[0] \
        == WRF_SCHEME_SHA256


def test_the_fixture_is_the_one_the_oracle_wrote():
    sums = {}
    for line in (FIXTURE / "oracle-sha256sums.txt").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        sums[Path(path).name] = digest
    for name in ("sbu-ylin-cases.txt", "sbu-ylin-pristine.txt",
                 "sbu-ylin-declared.txt"):
        assert _fixture_sha(name) == sums[name], name
    assert sums["module_mp_sbu_ylin.F"] == WRF_SCHEME_SHA256


def test_the_fixture_covers_every_family_and_at_least_64_columns():
    from tools.sbu_ylin_wrf461_oracle.validate_sbu_ylin_oracle import (
        read_cases)
    cases = read_cases(FIXTURE / "sbu-ylin-cases.txt")
    names = (FIXTURE / "sbu-ylin-case-names.txt").read_text().split()
    assert cases["ncol"] == len(names) >= 64
    families = {n.rsplit("-", 1)[0] for n in names}
    for family in ("deep_convective", "stratiform", "cold_snow",
                   "mixed_phase_riming", "warm_rain", "dry",
                   "supersaturated", "melting_virga"):
        assert family in families
    # zero hydrometeors, small negatives, signed zeros, long time steps
    assert any(n.startswith("edge0") for n in names)
    assert any(n.startswith("edge4") for n in names)
    assert float(cases["dt"].max()) >= 300.0
    assert np.any(np.signbit(cases["levels"][..., 2:6])
                  & (cases["levels"][..., 2:6] == 0.0))


def test_the_declared_fixes_each_apply_exactly_once(tmp_path):
    import subprocess
    import sys
    src = tmp_path / "module_mp_sbu_ylin.F"
    # A synthetic file carrying each fix's anchor exactly once proves the
    # patcher's one-match rule; the real file is pinned by sha256 above.
    from tools.sbu_ylin_wrf461_oracle.apply_declared_fixes import FIXES
    src.write_text("".join(old for _, old, _ in FIXES), encoding="ascii")
    out = tmp_path / "fixed.F"
    subprocess.run([sys.executable, str(ORACLE / "apply_declared_fixes.py"),
                    str(src), str(out)], check=True)
    text = out.read_text(encoding="ascii")
    for tag, old, new in FIXES:
        assert new in text, tag
    src.write_text("".join(old + old for _, old, _ in FIXES), encoding="ascii")
    bad = subprocess.run([sys.executable,
                          str(ORACLE / "apply_declared_fixes.py"),
                          str(src), str(out)])
    assert bad.returncode == 3


# --------------------------------------------------------- the GPU column

@pytest.fixture(scope="module")
def measured():
    from tools.sbu_ylin_wrf461_oracle.validate_sbu_ylin_oracle import measure
    return measure(FIXTURE)


@requires_gpu
def test_the_strict_arm_is_bitwise_the_pristine_wrf_build(measured):
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("strict arm: run with GPUWM_WRF_EXACT=1")
    rep = measured["pristine"]
    bad = {k: v for k, v in rep.items()
           if not k.startswith("_") and v["words_differing"]}
    assert bad == {}, bad
    assert rep["_total_words_differing"] == 0


@requires_gpu
def test_the_default_arm_is_within_its_stated_tolerance(measured):
    from woof.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip("default arm: run without GPUWM_WRF_EXACT")
    from tools.sbu_ylin_wrf461_oracle.validate_sbu_ylin_oracle import (
        read_cases, read_outputs, run_port, LEVEL_OUTPUTS, SURFACE_OUTPUTS)
    got = run_port(read_cases(FIXTURE / "sbu-ylin-cases.txt"))
    want = read_outputs(FIXTURE / "sbu-ylin-pristine.txt")
    for key, names in (("levels", LEVEL_OUTPUTS), ("surface", SURFACE_OUTPUTS)):
        for j, name in enumerate(names):
            np.testing.assert_allclose(got[key][..., j], want[key][..., j],
                rtol=1.e-4, atol=DEFAULT_ATOL[name], equal_nan=True, err_msg=name)


@requires_gpu
def test_historical_formula_changes_remain_negative_controls(measured):
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("word identity requires strict numerics")
    assert measured["pristine"]["_total_words_differing"] == 0
    assert measured["declared_vs_pristine"]["_total_words_differing"] > 0
    assert measured["declared"] == measured["declared_vs_pristine"]


@requires_gpu
def test_a_planted_one_ulp_literal_is_caught():
    from tools.sbu_ylin_wrf461_oracle.validate_sbu_ylin_oracle import measure
    planted = measure(FIXTURE, plant=True)
    assert planted["pristine"]["_total_words_differing"] > 0
    assert planted["columns_differing_from_declared"]


def test_registry_consumers_and_transport_contract():
    from woof.physics_registry import consumer_row_for_selector
    from woof.core import microphysics_transition
    row = consumer_row_for_selector("microphysics", "column_kernel", 13)
    assert row["species"] == ["qv", "qc", "qr", "qi", "qs"]
    assert row["state_field_io"]["rimi"]["advected"] is False
    assert row["state_field_io"]["rimi"]["restart"] is True
    assert row["state_field_io"]["rimi"]["history"] is True
    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    from woof.state_digest import lazy_inventory_member_class
    assert "rimi" not in COUPLED_SCALAR_STATE_FIELDS
    assert 13 in microphysics_transition.PORTED_MP_PHYSICS
    assert 13 not in microphysics_transition.UNVALIDATED_MIXED_EDGE_SELECTORS


def test_namelist_selector_13_is_registered():
    from woof.namelist_import import _MP_MAP
    assert 13 in _MP_MAP


def test_stock_namelist_import_selects_13(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair, _replace_namelist_column
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    inp = _replace_namelist_column(INPUT_TEXT, "mp_physics", [13, 13])
    # The stock 4/4 radiation pair is imported through its WRF implementation.
    text, report = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "mp13.toml"
    out.write_text(text)
    exp = load_experiment(out)
    assert [domain.run.mp_physics for domain in exp.domains] == [13, 13]
    assert all(domain.run.ra_rrtmg_variant == "rrtmg_legacy" for domain in exp.domains)


def test_output_word_comparator_catches_one_bit():
    from tools.sbu_ylin_wrf461_oracle.validate_sbu_ylin_oracle import compare
    ref = {"levels": np.ones((1, 1, 7), dtype=np.float32),
           "surface": np.ones((1, 2), dtype=np.float32)}
    got = {name: value.copy() for name, value in ref.items()}
    got["levels"].view(np.uint32)[0, 0, 6] ^= np.uint32(1)
    rep = compare(got, ref)
    assert rep["rimi"]["words_differing"] == 1
    assert rep["rimi"]["max_ulp"] == 1
    assert rep["_total_words_differing"] == 1


def test_mp13_transport_allocation_and_accounting():
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.device_inventory import state_array_shapes
    from woof.core.preflight import scratch_slot_registry
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=500., dy=500., ztop=6000.,
                    dt=10., run_seconds=0., moist=True, mp_physics=13, km_opt=3)
    state = DomainState(cfg, array_module=np)
    assert state.rimi.shape == (8, 8, 8)
    assert not hasattr(state, "rimi0")
    assert getattr(state, "qg", None) is None
    assert "smag_rrimi" not in scratch_slot_registry(cfg)
    assert state_array_shapes(cfg)["rimi"] == (8, 8, 8)
    assert "rimi0" not in state_array_shapes(cfg)


@requires_gpu
def test_rimi_is_unadvected_and_survives_restart_and_history(tmp_path):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import advance_scalars_stage, init_moist_balanced, moist_species
    from woof.ingest.nest_init import seed_rk_time_t_copies
    from woof.io import restart
    from woof.io.history_layout import live_state_history_fields
    nx, ny, nz = 16, 8, 12
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500., dy=500., ztop=6000.,
                    dt=10., run_seconds=0., moist=True, mp_physics=13)
    coord = make_vertical_coord(nz)
    base = make_base_state(coord, lambda z: 300. + .003*np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    def make_state():
        return init_moist_balanced(cfg, coord, base, lambda z: np.full(nz, 1.e-3))
    state = make_state()
    blob = np.zeros((nz, ny, nx), np.float32)
    blob[4:8, 3:6, 6:10] = 1.e-4
    for name in ("qi", "rimi"):
        getattr(state, name)[...] = cp.asarray(blob)
    assert "rimi" not in moist_species(state)
    assert "rimi0" not in seed_rk_time_t_copies(state)
    before = state.rimi.copy()
    chm = state.c1h[:, None, None]*state.total_mu()[None]+state.c2h[:, None, None]
    ru = cp.zeros((nz, ny, nx+1), cp.float32)
    ru[...] = .5*(cfg.dx/cfg.dt)*chm[:, :, :1]
    rv = cp.zeros((nz, ny+1, nx), cp.float32)
    ww = cp.zeros((nz+1, ny, nx), cp.float32)
    advance_scalars_stage(state, cfg, ru, rv, ww, cfg.dt, final=True)
    assert state.rimi.tobytes() == before.tobytes()
    assert state.rimi.tobytes() != state.qi.tobytes()
    assert live_state_history_fields(state)["RIMI"] is state.rimi
    path = restart.write_restart(tmp_path/"restart.npz", state, cfg)
    assert "state/rimi" in restart.read_restart_header(path)["array_manifest"]
    fresh = make_state()
    restart.restore_restart(path, fresh, cfg)
    assert fresh.rimi.tobytes() == state.rimi.tobytes()


@requires_gpu
def test_exported_in_file_helpers_match_pristine_fortran():
    import cupy as cp
    from woof.core.kernels import get_kernel
    lines = (FIXTURE / "sbu-ylin-leaves.txt").read_text().splitlines()
    n = int(lines[0])
    words = np.array([[int(x, 16) for x in line.split()] for line in lines[1:]], np.uint32)
    inputs = cp.asarray(np.ascontiguousarray(words[:, :3]).view(np.float32))
    outputs = cp.empty((n, 3), cp.float32)
    get_kernel("sbu_ylin", "sbu_ylin_helpers")(((n+63)//64,), (64,),
                                               (inputs, outputs, np.int32(n)))
    assert cp.asnumpy(outputs).view(np.uint32).tobytes() == words[:, 3:].tobytes()



def test_wrfinput_admission_and_rimi_restore():
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.ingest import wrfinput as wi
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=500., dy=500., ztop=6000.,
                    dt=10., run_seconds=0., moist=True, mp_physics=13)
    wi.check_supported_schemes({"MP_PHYSICS":13, "SF_SURFACE_PHYSICS":2}, source="test")
    names = {"QVAPOR","QCLOUD","QRAIN","QICE","QSNOW","RIMI"}
    assert wi.active_moisture_inventory(cfg) == (frozenset(names),frozenset(names))
    assert 13 not in wi.UNSUPPORTED_MICROPHYSICS
    state = DomainState(cfg, array_module=np)
    raw = {name: np.full((8,8,8),.125,np.float32) for name in names}
    wi._restore_active_moisture(state,raw,cfg,np)
    assert state.rimi.tobytes() == raw["RIMI"].tobytes()
    assert not hasattr(state, "rimi0")
    assert wi.WRFINPUT_DIMENSIONS["RIMI"] == ("bottom_top","south_north","west_east")

def test_column_preparation_scratch_is_rebuilt():
    from woof.io.restart import classify_scratch_slot
    for name in ("column_mp_theta","column_mp_rho","column_mp_pii",
                 "column_mp_z","column_mp_dz8w","column_mp_z8w",
                 "refl_sbu_snow","refl_sbu_graupel"):
        assert classify_scratch_slot(name) == "rebuild", name

@requires_gpu
def test_real_column_call_can_write_a_restart(tmp_path):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.microphysics import apply
    from woof.core.physics import initialize_physics
    from woof.io import restart
    cfg=RunConfig(nx=8,ny=8,nz=12,dx=1000.,dy=1000.,ztop=6000.,
                  dt=10.,run_seconds=0.,moist=True,mp_physics=13)
    coord=make_vertical_coord(cfg.nz)
    base=make_base_state(coord,lambda z:300.+.003*np.asarray(z),
                         p_surf=cfg.p_surf,ztop=cfg.ztop)
    def make():
        return init_moist_balanced(cfg,coord,base,lambda z:np.full(cfg.nz,.01))
    state=make()
    initialize_physics(state,cfg)
    state.qc[...] = cp.float32(2.e-4)
    state.qs[...] = cp.float32(1.e-3)
    apply(state,cfg,cfg.dt,refl_10cm_due=True)
    from types import SimpleNamespace
    from woof.state_digest import canonical_state_digest
    digest=canonical_state_digest(state,SimpleNamespace(dtbc_fp32=np.float32(cfg.dt)))
    assert "state/rimi" in digest["field_order"]
    path=restart.write_restart(tmp_path/"column-restart.npz",state,cfg)
    fresh=make()
    initialize_physics(fresh,cfg)
    restart.restore_restart(path,fresh,cfg)
    for name in ("qv","qc","qr","qi","qs","rimi"):
        assert getattr(fresh,name).tobytes() == getattr(state,name).tobytes(),name

def test_reflectivity_work_buffers_are_excluded_from_frame_digest():
    from types import SimpleNamespace
    from woof.state_digest import _canonical_extra_manifest
    state=SimpleNamespace(_scratch={
        "refl_10cm":np.ones((2,2,2),np.float32),
        "refl_sbu_snow":np.zeros((2,2,2),np.float32),
        "refl_sbu_graupel":np.zeros((2,2,2),np.float32)})
    assert set(_canonical_extra_manifest(state)) == {"scratch/refl_10cm"}
    state._scratch["refl_unregistered_future"]=np.zeros((2,2,2),np.float32)
    with pytest.raises(RuntimeError,match="not a concrete registered member"):
        _canonical_extra_manifest(state)
