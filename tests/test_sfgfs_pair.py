import pytest
from woof.config import RunConfig, validate_run_config


def test_natural_gfs_pair_binds_the_published_boundary():
    cfg = RunConfig(nx=8, ny=8, nz=12, dx=3000., dy=3000., ztop=10000.,
                    dt=10., run_seconds=80., sf_sfclay_physics=3,
                    bl_pbl_physics=3, sf_surface_physics=2)
    validate_run_config(cfg)


@pytest.mark.gpu
@pytest.mark.parametrize("fractional", [0, 1])
def test_natural_gfs_pair_restart_retains_every_word(tmp_path, fractional):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _physics_state, _assert_restart_equal

    def build():
        if fractional:
            from test_ruc_runtime import _build
            state, cfg, driver = _build(
                nx=6, ny=4, nz=16, sf_sfclay_physics=3,
                bl_pbl_physics=3, fractional_seaice=1, ice_rows=1)
            # Exercise the blended branch with partial ice, rather than
            # just entering a wrapper whose fractions are all zero or one.
            driver.fields["xice"][0, :4] = cp.float32(0.6)
            return state, cfg, driver
        return _physics_state(cp, sf_sfclay_physics=3, bl_pbl_physics=3,
                              cu_physics=0)
    full, cfg, _ = build()
    run_steps(full, cfg, 8)
    expected = restart.write_restart(tmp_path / "full.npz", full, cfg)
    first, cfg, _ = build()
    run_steps(first, cfg, 4)
    middle = restart.write_restart(tmp_path / "middle.npz", first, cfg)
    resumed, cfg, _ = build()
    restart.restore_restart(middle, resumed, cfg)
    run_steps(resumed, cfg, 4)
    actual = restart.write_restart(tmp_path / "resumed.npz", resumed, cfg)
    _assert_restart_equal(actual, expected)
