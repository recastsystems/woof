"""Production hail columns, literal tendencies, and the one-bit control."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
from conftest import requires_gpu
pytestmark = [pytest.mark.gpu, requires_gpu]
ROOT=Path(__file__).resolve().parents[1]


def test_hail_zero_out_uses_the_native_first_scalar():
    import cupy as cp
    import numpy as np
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.microphysics import microphysics_zero_out
    cfg = RunConfig(nx=4, ny=4, nz=24, dx=1000., dy=1000., ztop=12000.,
                    dt=10., run_seconds=10., moist=True, mp_physics=38,
                    mp_zero_out=1, mp_zero_out_all=1, mp_zero_out_thresh=1e-12)
    state = DomainState(cfg)
    for name in ("ni", "nr", "ng", "qvolg"):
        getattr(state, name).fill(np.float32(1e-13))
    microphysics_zero_out(state, cfg)
    np.testing.assert_array_equal(cp.asnumpy(state.ni[:, 1:-1, 1:-1]), np.float32(1e-13))
    for name in ("nr", "ng", "qvolg"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(state, name)[:, 1:-1, 1:-1]), 0.)


def check(tmp_path, strict, name, *extra):
    out=tmp_path/(name+".json")
    command=[sys.executable,str(ROOT/"tools/thompson_wrf461_oracle/validate_hail.py"),
        "--backend","gpu","--fixtures",str(ROOT/"woof/data/thompson/oracle-hail"),
        "--out",str(out),*extra]
    run=subprocess.run(command,cwd=ROOT,env={**os.environ,"GPUWM_WRF_EXACT":str(strict),
        "GPUWM_NUMERICS":"strict" if strict else "fast"},
        capture_output=True,text=True)
    assert out.exists(),run.stdout+run.stderr
    return run.returncode,json.loads(out.read_text()),out
def test_fast_replay_reports_literal_tendency_residue(tmp_path):
    from woof.core.kf import kf_phase_mode_for_microphysics, KFPhaseMode
    assert kf_phase_mode_for_microphysics(38) == KFPhaseMode.SEPARATE_ICE_SNOW
    rc,receipt,_=check(tmp_path,0,"default","--tolerance","3e-6","--tendency-tolerance","1e-5")
    # Fast contraction exposes a qvten cancellation residue in the hail
    # fixture. Keep reporting the original tolerance failure rather than
    # labeling a strict-mode helper run as the default or raising a limit.
    assert rc==1
    assert receipt["numerics"]["mode"] == "fast"
    assert not receipt["default_tendency_tolerance_pass"]
    assert receipt["tendencies"]["gh-hail"]["qvten"]["max_relative"] > 1e-5
    assert receipt["literal_tendency_coverage"]==14


def test_hail_initial_history_preserves_native_zero(monkeypatch):
    import cupy as cp
    import numpy as np
    from types import SimpleNamespace
    from woof import wrf_exact
    from woof.core.refl import analysis_refl_10cm
    monkeypatch.setattr(wrf_exact, "ENABLED", False)
    # Real WRF mp=38 does not seed the source-absent moments. Initial
    # history precedes its first microphysics diagnostic, even with mass.
    state = SimpleNamespace(qv=cp.full((4, 2, 2), .01, cp.float32))
    result = analysis_refl_10cm(state, SimpleNamespace(mp_physics=38))
    np.testing.assert_array_equal(cp.asnumpy(result), np.zeros((4, 2, 2), np.float32))
def test_hail_default_leaves_native_inactive_radii_untouched():
    import cupy as cp
    import numpy as np
    from test_thompson_aerosol_adapter import _build_case
    from woof.core.microphysics import microphysics_init
    from woof.core.microphysics_aerosol import _apply_thompson_aerosol
    state, cfg, dt, *_ = _build_case(cp, "aero-warm-overlap")
    cfg.mp_physics = 38
    state.ng = cp.zeros_like(state.qg)
    state.qvolg = cp.zeros_like(state.qg)
    for field in ("effc", "effi", "effs"):
        getattr(state, field).fill(np.float32(37.))
    microphysics_init(state, cfg)
    _apply_thompson_aerosol(state, cfg, dt)
    for field in ("effc", "effi", "effs"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(state, field)), np.float32(37.))


def test_strict_residue_and_planted_ng_bit(tmp_path):
    rc,baseline,path=check(tmp_path,1,"strict")
    assert rc==1  # The remaining strict misses are recorded, not hidden.
    assert baseline["input_origin"] == "synthetic-column-fixtures"
    assert baseline["real_native_capture"] is False
    assert baseline["unmeasured_literal_tendencies"]==[]
    warm=baseline["scenarios"]["gh-warm-melt"]["fields"]
    assert all(m["max_ulp"]==0 for m in warm.values())
    assert all(m["max_ulp"]==0 for m in baseline["tendencies"]["gh-warm-melt"].values())
    rc,mutant,_=check(tmp_path,1,"mutant","--mutation","--compare-port",str(path))
    assert rc==1
    changed=mutant["changed_from_port_baseline"]
    assert {case for case,field in changed if field=="ng_per_kg"}==set(baseline["scenarios"])


def test_bare_default_hail_initialization():
    import cupy as cp
    import numpy as np
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.microphysics import microphysics_init
    cfg=RunConfig(nx=4, ny=4, nz=24, dx=1000., dy=1000., ztop=12000.,
                  dt=10., run_seconds=10., moist=True, mp_physics=38)
    state=DomainState(cfg)
    state.p[...] = cp.asarray(np.linspace(95000.,30000.,cfg.nz,dtype=np.float32)[:,None,None])
    state.thb[...] = 300.
    state.phb[...] = cp.asarray(np.linspace(0.,9.81*12000.,cfg.nz+1,dtype=np.float32))
    state.qv[...] = 0.006
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
    from woof.core.physics import microphysics_scheme_sr_available
    assert legacy_cloud_fraction_flags(38) == (True, True)
    assert microphysics_scheme_sr_available(38)
    from woof.io.restart import classify_scratch_slot
    for slot in ("mp_thompson_aero_ngten", "mp_thompson_aero_qgten",
                 "mp_thompson_aero_qbten", "mp_thompson_aero_qvten",
                 "mp_thompson_aero_qsten", "mp_thompson_aero_tten",
                 "mp_thompson_graupel_mvd", "mp_thompson_graupel_mvd_max",
                 "mp_thompson_graupel_density"):
        assert classify_scratch_slot(slot) == "rebuild"
    receipt=microphysics_init(state,cfg)
    assert receipt["thompson_aerosol_profile"]
    assert bool(cp.all(cp.isfinite(state.nwfa))) and bool(cp.any(state.nwfa>0))
    assert state.ng.shape==state.qg.shape==state.qvolg.shape
    assert state.existing_scratch("mp_thompson_graupel_mvd_max") is not None
