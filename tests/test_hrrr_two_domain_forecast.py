"""CPU contracts for the native-HRRR two-domain forecast harness."""

from datetime import datetime

import numpy as np

from woof.core.clock import build_schedule, resolve_clock
from woof.experiment import VerticalConfig
from woof.physics_compat import THOMPSON_RTE_RRTMGP_PROFILE_ID
from woof.ingest.hrrr_surface import surface_fields_to_device
from tools.hrrr_two_domain_forecast import _experiment
from tools.hrrr_build_native_static import benchmark_grid
from tools.hrrr_single_domain_benchmark import _experiment as benchmark_experiment


ETA = np.linspace(1.0, 0.0, 50)


def test_restored_numpy_surface_fields_are_uploaded_before_gpu_assignment():
    class RecordingCupy:
        float32 = np.float32

        def __init__(self):
            self.seen = []

        def asarray(self, value, *, dtype):
            self.seen.append((value, dtype))
            return ("device", id(value), dtype)

    fields = {
        "T2": np.full((2, 3), 290.0),
        "U10": np.full((2, 4), 5.0),
        "V10": np.full((3, 3), -2.0),
    }
    cp = RecordingCupy()

    uploaded = surface_fields_to_device(
        type("Met", (), {"fields": fields})(), cp)

    assert list(uploaded) == ["T2", "U10", "V10"]
    assert cp.seen == [
        (fields["T2"], np.float32),
        (fields["U10"], np.float32),
        (fields["V10"], np.float32),
    ]
    assert all(value[0] == "device" for value in uploaded.values())


def test_two_domain_forecast_geometry_physics_and_schedule_are_frozen():
    exp = _experiment(ETA, run_seconds=300.0, sample_interval_s=60.0)
    d01, d02 = exp.domains
    assert (d01.run.nx, d01.run.ny, d01.run.dx, d01.run.dt) == (
        199, 199, 2999.4213047435587, 15.0)
    assert (d02.run.nx, d02.run.ny, d02.run.dx, d02.run.dt) == (
        300, 300, 999.8071015811862, 5.0)
    assert (d02.i_parent_start, d02.j_parent_start) == (50, 50)
    assert (d02.parent_grid_ratio, d02.parent_time_step_ratio) == (3, 3)
    for domain in exp.domains:
        cfg = domain.run
        assert (cfg.mp_physics, cfg.ra_lw_physics,
                cfg.ra_sw_physics) == (6, 0, 1)
        assert (cfg.sf_sfclay_physics, cfg.sf_surface_physics,
                cfg.bl_pbl_physics, cfg.cu_physics) == (91, 2, 1, 0)
        assert cfg.km_opt == 4 and cfg.diff_6th_opt == 2
    schedule = build_schedule(exp, resolve_clock(exp, lbc_interval_s=3600.0))
    assert schedule.periods == 20
    assert schedule.op_counts(schedule.full_ops()) == {
        "STEP": 80, "FORCE": 20, "FEEDBACK": 20}


def test_single_domain_500x500_benchmark_geometry_and_physics_are_frozen():
    vertical = VerticalConfig(
        eta_levels=tuple(float(value) for value in ETA),
        p_top=10_000.0, hybrid_opt=2, etac=0.2)
    # 4193eb0da, development branch: named HRRR defaults now select the
    # MP28/GSD/MYNN/RUC suite. This historical 500x500 benchmark keeps its
    # explicit MP8/YSU/MM5/Noah full-radiation profile across route changes.
    # 3e5245839, development branch: monthly source defaults now cover Noah
    # as well as RUC. This frozen benchmark keeps both prescribed fields off.
    exp = benchmark_experiment(
        vertical, run_seconds=300.0,
        start_time=datetime(2026, 7, 20, 6),
        physics_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID,
        usemonalb=False, rdlai2d=False)
    assert exp.start_time == datetime(2026, 7, 20, 6)
    assert len(exp.domains) == 1
    dc = exp.domains[0]
    cfg = dc.run
    assert (cfg.nx, cfg.ny, cfg.nz) == (500, 500, 49)
    assert (cfg.dx, cfg.dy, cfg.dt) == (
        999.8071015811862, 999.8071015811862, 5.0)
    assert dc.parent_id == 0 and cfg.specified and not cfg.nested
    # 1.8 froze this benchmark at the full-radiation suite:
    # Thompson mp8 with longwave AND shortwave on, nocturnally valid.  It
    # replaced (6, 0, 1) -- WSM6 with longwave OFF -- whose frozen
    # downward longwave cratered nocturnal skin temperature (the 1.7.1
    # headline).  The 4/4 pair below does not say WHICH implementation
    # runs: the engine is a setting, and the default moved from the
    # legacy RRTMG pair to RTE+RRTMGP without moving these numbers.
    assert (cfg.mp_physics, cfg.ra_lw_physics,
            cfg.ra_sw_physics) == (8, 4, 4)
    assert (cfg.sf_sfclay_physics, cfg.sf_surface_physics,
            cfg.bl_pbl_physics, cfg.cu_physics) == (91, 2, 1, 0)
    assert cfg.usemonalb is False and cfg.rdlai2d is False
    assert cfg.radt_minutes == 12.0
    assert dc.history_interval_s == 300.0
    grid = benchmark_grid()
    assert grid.latlon_mass()[0].shape == (500, 500)
    schedule = build_schedule(exp, resolve_clock(exp, lbc_interval_s=3600.0))
    assert schedule.periods == 60
    assert schedule.op_counts(schedule.full_ops()) == {
        "STEP": 60, "FORCE": 0, "FEEDBACK": 0}
