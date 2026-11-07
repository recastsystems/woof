"""Native cloud-bound handoffs, including no-cloud and held KF columns."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def contracts(monkeypatch):
    pytest.importorskip("cupy")
    from woof.core import physics, gf
    monkeypatch.setattr(physics, "cp", np)
    monkeypatch.setattr(physics.health_ledger, "check_finite",
                        lambda a, **kw: bool(np.isfinite(a).all()))
    monkeypatch.setattr(physics, "_is_native_kf_result", lambda *a: False)
    return physics, gf


@pytest.mark.parametrize("kf", [False, True])
def test_due_cloud_bounds_overwrite_and_kf_preserves_held_columns(contracts, monkeypatch, kf):
    physics, gf = contracts
    shape = (4, 1, 3)
    old_base = np.array([[2., 3., 2.]], dtype="f4")
    old_top = np.array([[4., 4., 3.]], dtype="f4")
    # First column is a native no-cloud reset, second changes its extent.
    # Preserve the .001 marker that the Grell drivers append to levels.
    new_base = np.array([[5., 1.001, 5.]], dtype="f4")
    new_top = np.array([[1., 3.001, 1.]], dtype="f4")
    result = physics.CumulusResult(
        rthcuten=np.zeros(shape, dtype="f4"), rqvcuten=np.zeros(shape, dtype="f4"),
        cloud_base=new_base, cloud_top=new_top,
        nca_seconds=np.zeros((1, 3), dtype="f4") if kf else None)
    driver = SimpleNamespace(
        cumulus_callable=lambda **kw: result,
        fields={"hbot": old_base.copy(), "htop": old_top.copy()},
        cu_nca=np.array([[-100., 0., 60.]], dtype="f4"))
    class HandoffComplete(Exception):
        pass
    def stop(*args):
        raise HandoffComplete
    monkeypatch.setattr(gf, "is_native_gf_result", stop)
    with pytest.raises(HandoffComplete):
        physics.PhysicsDriver._run_cumulus(
            driver, {}, SimpleNamespace(p=np.zeros(shape, dtype="f4")),
            SimpleNamespace(dt=10.))
    expect_base, expect_top = new_base.copy(), new_top.copy()
    if kf:
        expect_base[0, 2], expect_top[0, 2] = old_base[0, 2], old_top[0, 2]
    np.testing.assert_array_equal(driver.fields["hbot"], expect_base)
    np.testing.assert_array_equal(driver.fields["htop"], expect_top)


@pytest.mark.parametrize("gfdl", [False, True])
def test_new_sas_returns_native_bounds_only_to_selected_gfdl(contracts, monkeypatch, gfdl):
    import sys
    from woof.core import column_cumulus
    from woof.config import RunConfig
    monkeypatch.setitem(sys.modules, "cupy", np)
    nz, ny, nx = 4, 2, 3
    vol = np.ones((nz, ny, nx), dtype="f4")
    iface = np.ones((nz + 1, ny, nx), dtype="f4")
    surface = np.ones((ny, nx), dtype="f4")
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000., dy=3000., ztop=8000.,
                    dt=10., run_seconds=0., cu_physics=96,
                    ra_lw_physics=99 if gfdl else 0, ra_sw_physics=0)
    def launch(scheme, **kw):
        for value in kw["outputs"].values():
            if value is not None:
                value.fill(0.)
        kw["sfc"].fill(0.)
        kw["sfc"][column_cumulus.SURFACE_OUTPUTS.index("hbot")] = 2.
        kw["sfc"][column_cumulus.SURFACE_OUTPUTS.index("htop")] = 4.
    monkeypatch.setattr(column_cumulus, "launch_columns", launch)
    result = column_cumulus.ColumnCumulus(96)(
        atmosphere={**{name: vol for name in (
            "temperature", "qv", "qc", "qi", "pressure", "exner", "rho", "dz", "u", "v")},
            "p_interface": iface},
        fields={name: surface for name in ("xland", "pblh", "hfx", "qfx")},
        state=SimpleNamespace(p=vol, w=iface, qc=vol, qi=vol), cfg=cfg)
    if gfdl:
        np.testing.assert_array_equal(result.cloud_base, np.full((ny, nx), 2., dtype="f4"))
        np.testing.assert_array_equal(result.cloud_top, np.full((ny, nx), 4., dtype="f4"))
    else:
        assert result.cloud_base is result.cloud_top is None
