"""Hour-0 refl10cm comes from the initial hydrometeors, not an allocated zero.

THE BREAKAGE THIS PREVENTS: both history paths wrote ``refl10cm`` as zeros
at step 0 (native MPAS-A's allocated default), so an init carrying storms
showed none at f000 in maps and scores (the ensemble's hour-0 CSI 0 and
-60 dBZ max bias).  The backend now evaluates the active scheme's own
10 cm reflectivity of the initial state through the engine's between-steps
operator, on a column view of the hex arrays
(``InitialReflectivityViewV841``).  These CPU tests run the engine's host
(float64 column mirror) path of the same operator on that view: a synthetic
storm column must give the scheme diagnostic itself, and a selection with
no microphysics must say so instead of inventing a field.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.cuda_arwen_physics_v841 import (
    InitialReflectivityViewV841,
    PersistentTwoPhaseCudaPhysicsBackendV841,
)

obsop = pytest.importorskip("woof.da.obsop")
npref = pytest.importorskip("woof.verify.npref")
constants = pytest.importorskip("woof.core.constants")


def _storm_columns():
    """Two columns, bottom to top: a convective one (rain, snow, graupel) and a dry one."""
    nz = 12
    p = np.linspace(100000.0, 20000.0, nz, dtype=np.float32)
    temperature = np.linspace(300.0, 220.0, nz, dtype=np.float32)
    theta = (temperature * (np.float32(constants.P0) / p) ** np.float32(constants.RCP)).astype(np.float32)
    qv = np.linspace(0.016, 0.0001, nz, dtype=np.float32)
    qr = np.where(np.arange(nz) < 5, 4.0e-3, 0.0).astype(np.float32)
    qs = np.where((np.arange(nz) > 5) & (np.arange(nz) < 10), 1.5e-3, 0.0).astype(np.float32)
    qg = np.where((np.arange(nz) > 3) & (np.arange(nz) < 8), 2.5e-3, 0.0).astype(np.float32)
    qc = np.full(nz, 5.0e-4, dtype=np.float32)
    qi = np.zeros(nz, dtype=np.float32)

    def two(storm, dry):
        return np.ascontiguousarray(np.stack([storm, dry], axis=1))

    zero = np.zeros(nz, dtype=np.float32)
    species = {"qv": two(qv, qv), "qc": two(qc, zero), "qr": two(qr, zero), "qi": two(qi, zero),
               "qs": two(qs, zero), "qg": two(qg, zero)}
    return two(theta, theta), two(p, p), species


def test_a_storm_column_gives_the_wsm6_diagnostic_itself():
    theta, pressure, species = _storm_columns()
    view = InitialReflectivityViewV841(theta=theta, pressure=pressure, species=species, melting=False, cp=np)
    cfg = SimpleNamespace(mp_physics=6, wsm6_hail_opt=0)
    refl = np.asarray(obsop.simulated_reflectivity(view, cfg)).reshape(theta.shape)

    # the scheme's own column routine, fed the same initial state directly
    t = (theta * (pressure / np.float32(constants.P0)) ** np.float32(constants.RCP)).astype(np.float64)
    for column in range(2):
        expected = npref.np_refl10cm_wsm6_column(
            species["qv"][:, column].astype(np.float64), species["qr"][:, column].astype(np.float64),
            species["qs"][:, column].astype(np.float64), species["qg"][:, column].astype(np.float64),
            t[:, column], pressure[:, column].astype(np.float64), hail_opt=0)
        np.testing.assert_array_equal(refl[:, column], np.asarray(expected, dtype=np.float32))
    assert refl[:, 0].max() > 30.0, "a column with 4 g/kg rain must show a storm at f000"
    assert refl[:, 1].max() < -30.0, "a dry column stays at the clear-air floor"


def test_the_view_reads_the_initial_arrays_and_writes_none_of_them():
    theta, pressure, species = _storm_columns()
    before = {name: value.copy() for name, value in species.items()}
    view = InitialReflectivityViewV841(theta=theta, pressure=pressure, species=species, melting=False, cp=np)
    assert view.qr.shape == (theta.shape[0], 1, theta.shape[1])
    np.testing.assert_array_equal(view.qr[:, 0, :], species["qr"])
    obsop.simulated_reflectivity(view, SimpleNamespace(mp_physics=6, wsm6_hail_opt=0))
    for name, value in before.items():
        np.testing.assert_array_equal(species[name], value)
    # a species the selection does not carry is absent, so the operator refuses by name
    assert view.nr is None and view.qh is None
    with pytest.raises(AttributeError):
        view.not_a_field


def test_a_shape_mismatch_is_refused_by_name():
    theta, pressure, species = _storm_columns()
    species["qr"] = species["qr"][:, :1]
    with pytest.raises(ValueError, match="qr"):
        InitialReflectivityViewV841(theta=theta, pressure=pressure, species=species, melting=False, cp=np)


def test_no_microphysics_reports_why_instead_of_inventing_a_field():
    backend = PersistentTwoPhaseCudaPhysicsBackendV841.__new__(PersistentTwoPhaseCudaPhysicsBackendV841)
    backend.computes_reflectivity = lambda: False
    refl, reason = backend.initial_refl10cm(atmosphere=None, scalar_names=("qv",))
    assert refl is None and "no microphysics" in reason


def test_both_history_paths_ask_the_backend_at_step_zero():
    from pathlib import Path

    drivers = Path(__file__).resolve().parents[1] / "../../woof/hex/drivers"
    for name in ("samehost_forecast_v841.py", "run_cuda_v841_full_physics_x4.py"):
        text = (drivers / name).read_text()
        assert "initial_refl10cm(" in text, f"{name} no longer computes the hour-0 reflectivity"


def test_the_background_history_handoff_carries_the_step():
    # The f000 frame's refl10cm_provenance attribute is chosen by the step the
    # writer is handed; the background writer was handed the arrays alone, so
    # a measured f000 file (G3, 30 km, 2026-10-12) still said "computed
    # inside the due step's WSM6 call" for a field computed from the init.
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "../../woof/hex/drivers/run_cuda_v841_forecast.py").read_text()
    assert '"receipt": {"step": step}' in source
    writer = (Path(__file__).resolve().parents[1] / "../../woof/hex/drivers/run_cuda_v841_full_physics_x4.py").read_text()
    assert '(snapshot.get("receipt") or {}).get("step") == 0' in writer


def test_the_view_carries_the_dry_specific_volume_nssl_reads():
    """THE BREAKAGE THIS PREVENTS (2026-11-09): NSSL two-moment's reflectivity forms its air density as
    ``1 / state.alt``; the view carried no ``alt`` and the forecast stopped at its first frame with
    ``AttributeError: alt``.  The view holds the quotient the seam hands a step, shaped as its other fields."""
    theta = np.full((6, 4), 300.0, dtype=np.float32)
    pressure = np.full((6, 4), 9.0e4, dtype=np.float32)
    rho = np.linspace(1.2, 0.4, 24, dtype=np.float32).reshape(6, 4)
    species = {"qv": np.zeros((6, 4), dtype=np.float32)}
    view = InitialReflectivityViewV841(theta=theta, pressure=pressure, species=species, melting=False, cp=np,
                                       rho_dry=rho)
    assert view.alt.shape == (6, 1, 4) and view.alt.dtype == np.float32
    np.testing.assert_array_equal(view.alt[:, 0, :], np.float32(1.0) / rho)
    plain = InitialReflectivityViewV841(theta=theta, pressure=pressure, species=species, melting=False, cp=np)
    assert not hasattr(plain, "alt")
