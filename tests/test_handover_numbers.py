"""Hydrometeor mass handed to a child never arrives with number zero.

The defect (init-from-any-run proof, GFS 2024-05-06 12Z): a hex -> ArWen 3 km
child (Morrison) started with rain and cloud mass at number 0 (an empty first
map, 341,746 NaN reflectivity values) and took 469-518 such columns per hour
through its ring; a WOOF-ArWen -> ArWen 1 km downscale dropped the droplet number
and started 60-70 dBZ rings.  These tests fail on the tree before the fix.
"""
import numpy as np
import pytest

from woof.ingest import handover_numbers as hn


def _column(q_rain=1e-3, n_rain=0.0):
    shape = (3, 2, 2)
    masses = {"qr": np.full(shape, q_rain, np.float32),
              "qi": np.full(shape, 1e-5, np.float32),
              "qs": np.full(shape, 2e-4, np.float32),
              "qg": np.zeros(shape, np.float32)}
    numbers = {"nr": np.full(shape, n_rain, np.float32),
               "ni": np.zeros(shape, np.float32),
               "ns": np.zeros(shape, np.float32),
               "ng": np.zeros(shape, np.float32)}
    return masses, numbers, shape


def test_morrison_mass_without_number_is_closed_and_carried_numbers_kept():
    masses, numbers, shape = _column()
    numbers["nr"][0] = 1234.0  # a number carried from the parent
    receipt = hn.close_numbers(10, masses, numbers, rho=np.ones(shape),
                               temperature=np.full(shape, 285.0))
    assert receipt["species"]["nr"] == 8 and receipt["species"]["ng"] == 0
    assert np.all(numbers["nr"][0] == 1234.0)
    for mass, number in hn.MORRISON_HANDOVER_NUMBERS:
        has = masses[mass] > 0
        assert np.all(numbers[number][has] > 0) and np.all(np.isfinite(numbers[number]))
        assert np.all(numbers[number][~has] == 0)
    # Marshall-Palmer-sized rain, not the 2.8 mm slope-limit drops N = 0 gave.
    lam = np.cbrt(np.pi * hn.MORRISON_RHOW * numbers["nr"][1] / masses["qr"][1])
    assert np.all(1.0 / lam < 1.0e-3)


def test_strict_process_keeps_real_exe_zeros(monkeypatch):
    from woof import correct_mode
    monkeypatch.setattr(correct_mode, "STRICT", True)
    assert not hn.closure_active({hn.HANDOVER_NUMBERS_KEY: True})
    assert hn.handover_number_pairs(10, None) == ()
    masses, numbers, shape = _column()
    assert hn.close_numbers(10, masses, numbers, rho=np.ones(shape),
                            temperature=np.full(shape, 285.0)) is None
    assert np.all(numbers["nr"] == 0)


# --- exact mode WITHOUT GPUWM_WRF_EXACT (release review 2.9.7, HIGH 1) ------
# physics.mode = "exact" is byte-identical to the code before the closure: it
# authors the HN1 key False, and every route reads the run's key, not the
# process.  Before this, closure_active() read only the env var, so an exact
# run still diagnosed numbers and forced nr/ni/nc back on the specified ring.

def _exact_cfg(mp):
    from woof import correct_mode
    from woof.config import RunConfig
    cfg = RunConfig(nx=13, ny=12, nz=4, dx=12000., dy=12000., ztop=10000.,
                    dt=1., run_seconds=120., moist=True, mp_physics=mp,
                    specified=True, mp28_aerosol_source="synthetic",
                    **correct_mode.resolve("exact").settings)
    assert cfg.hydrometeor_handover_numbers is False
    return cfg


def test_exact_mode_is_a_correction_key_off():
    from woof import correct_mode
    assert "hydrometeor_handover_numbers" in correct_mode.CORRECTION_KEYS
    assert correct_mode.FINDING_BY_KEY["hydrometeor_handover_numbers"] == "HN1"
    assert correct_mode.resolve("correct", ["HN1"]).settings[
        "hydrometeor_handover_numbers"] is False
    assert not correct_mode.STRICT
    assert not hn.closure_active(_exact_cfg(10))
    assert hn.closure_active(None) and hn.closure_active({})
    assert hn.handover_number_pairs(10, _exact_cfg(10)) == ()
    # Thompson's real.exe seeds hold in every mode.
    assert hn.handover_number_pairs(28, _exact_cfg(28)) == (
        ("qc", "nc"), ("qr", "nr"), ("qi", "ni"))


@pytest.mark.parametrize("mp", [8, 10, 28])
def test_exact_mode_downscale_leaves_the_numbers_as_handed(mp):
    from woof.offline_child import close_child_numbers
    masses, numbers, shape = _column()
    mixing = {"qv": np.full(shape, 0.01, np.float32), "qc": masses["qr"].copy(),
              "nc": np.zeros(shape, np.float32), **masses, **numbers}
    before = {k: v.copy() for k, v in mixing.items()}
    fields = {"T": np.full(shape, 5.0, np.float32),
              "P": np.zeros(shape, np.float32),
              "PB": np.full(shape, 85000.0, np.float32)}
    assert close_child_numbers(mp, mixing, fields, cfg=_exact_cfg(mp)) is None
    for name, value in before.items():
        assert mixing[name].tobytes() == value.tobytes(), name


def test_exact_mode_ring_and_cold_start_are_the_296_rows():
    from woof.boundary_fields import (SCALAR_ARRAY_BOUNDARY_FIELDS,
                                       boundary_hydrometeor_fields)
    from woof.ingest.microphysics_cold_start import cold_start_seeded_numbers
    five = ("qc", "qr", "qi", "qs", "qg")
    assert boundary_hydrometeor_fields(_exact_cfg(10), five) == five
    assert "nr" not in cold_start_seeded_numbers(_exact_cfg(10))
    # 2.9.6's scalar-array list: ns/ng stay forced back where a ring has them.
    assert set(SCALAR_ARRAY_BOUNDARY_FIELDS) == {
        "nwfa", "nifa", "nc", "nr", "ni", "qke_adv"}


@pytest.mark.parametrize("mp, numbers", [(8, ("nr", "ni")),
                                         (28, ("nc", "nr", "ni"))])
def test_exact_mode_specified_ring_leaves_the_numbers_to_their_tendency(
        monkeypatch, mp, numbers):
    from types import SimpleNamespace
    from woof.ingest import lateral_bc
    five = ("qc", "qr", "qi", "qs", "qg")
    fields = dict.fromkeys(("u", "v", "theta", "phi", "mu", "qv", *five,
                            *numbers), 0)
    interval = SimpleNamespace(fields=fields)
    monkeypatch.setattr(lateral_bc, "_active_device_interval",
                        lambda *a: (interval, 0., None, None))
    monkeypatch.setattr(lateral_bc, "_launch_mu_boundary_values", lambda *a: None)
    monkeypatch.setattr(lateral_bc, "_ring_at_record_end", lambda *a: True)
    monkeypatch.setattr(lateral_bc, "close_ring_numbers",
                        lambda *a: pytest.fail("ring closure ran in exact mode"))
    calls = []
    monkeypatch.setattr(lateral_bc, "_launch_finalize_field",
                        lambda s, name, *a: calls.append(name))
    state = SimpleNamespace(qv=object(), lateral_boundaries=object(),
                            _lateral_boundary_device=SimpleNamespace(clock=object()))
    cfg = _exact_cfg(mp)
    monkeypatch.setattr(lateral_bc, "specified_relaxes_w", lambda c: False)
    lateral_bc.apply_state_boundary_values(state, cfg)
    assert calls == ["u", "v", "theta", "phi", "qv", *five]


# The ring half is pinned in tests/test_lateral_hydrometeor_boundaries.py (mp=10 row).


def test_downscale_child_numbers_are_closed_on_the_child_state():
    from woof.offline_child import close_child_numbers
    masses, numbers, shape = _column()
    mixing = {"qv": np.full(shape, 0.01, np.float32), **masses, **numbers}
    fields = {"T": np.full(shape, 5.0, np.float32),
              "P": np.zeros(shape, np.float32),
              "PB": np.full(shape, 85000.0, np.float32)}
    receipt = close_child_numbers(10, mixing, fields, morr_rimed_ice=1, cfg=None)
    assert receipt["cells_total"] > 0
    assert np.all(mixing["nr"] > 0) and np.all(mixing["ni"] > 0)


def test_morrison_droplet_number_is_the_schemes_own_diagnostic():
    rho = np.array([1.0, 0.5])
    np.testing.assert_allclose(hn.morrison_droplet_number(rho), 250.0e6 / rho)


def test_thompson_child_route_seeds_from_mass():
    shape = (2, 2, 2)
    masses = {"qr": np.full(shape, 1e-4, np.float32),
              "qi": np.full(shape, 1e-6, np.float32)}
    numbers = {"nr": np.zeros(shape, np.float32), "ni": np.zeros(shape, np.float32)}
    hn.close_numbers(8, masses, numbers, rho=np.ones(shape),
                     temperature=np.full(shape, 260.0))
    assert np.all(numbers["nr"] > 0) and np.all(numbers["ni"] > 0)


def test_a_number_rounded_to_almost_nothing_beside_its_mass_is_closed():
    # The 151 dBZ first-frame cell of the 1 km downscale: SINT carried the
    # rain mass and left its number at 2.6e-16 per kg.
    masses, numbers, shape = _column(q_rain=6.4e-8, n_rain=2.6e-16)
    hn.close_numbers(10, masses, numbers, rho=np.ones(shape),
                     temperature=np.full(shape, 290.0))
    least = hn.morrison_least_number("rain", masses["qr"])
    assert np.all(numbers["nr"] >= least)
