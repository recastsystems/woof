"""A remap must preserve forces, budgets and exact unchanged columns separately."""

from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.moving_balance import (
    project_columns,
    conservative_float32,
    source_pressure_acceleration,
)
from woof.hex.moving_physics import field_intensive, remap_step_carriers
from test_moving_vertical import mapper


def test_already_conserved_cast_keeps_words_without_building_correction_indices(monkeypatch):
    exact=np.array([[1.00000001,-0.,1.e-42],[2.00000003,3.00000004,4.00000005]])
    expected=exact.astype(np.float32)
    measure=np.array([1.,2.,3.])
    target=float(np.sum(expected.astype(float)*measure))
    def unnecessary(*args,**kwargs):
        raise AssertionError('a closed cast budget needs no correction index list')
    monkeypatch.setattr(np,'flatnonzero',unnecessary)
    actual=conservative_float32(exact,measure,target,eligible=np.ones_like(exact,dtype=bool))
    assert actual.tobytes()==expected.tobytes()
    assert float(np.sum(actual.astype(float)*measure))==target


def test_physics_layer_rate_follows_height_and_leaves_soil_coordinates_alone():
    w = mapper(
        [[0], [2], [4], [6], [8]], [[0], [1], [3], [5], [8]], endpoint_alignment=False
    )
    rate = np.array([1, 3, 5, 7], np.float32)[:, None, None]
    np.testing.assert_allclose(
        field_intensive(rate, w, "pbl/dtheta")[:, 0, 0], [1, 2, 4, 6.5]
    )
    np.testing.assert_array_equal(field_intensive(rate, w, "fields/tslb"), rate)
    np.testing.assert_array_equal(
        field_intensive(np.ones_like(rate), w, "state/h_diabatic"), np.ones_like(rate)
    )


def test_feedback_carrier_uses_atmospheric_heights():
    w = mapper([[0], [2], [4], [6]], [[0], [1], [3], [6]], endpoint_alignment=False)
    rate = np.array([[1], [3], [5]], np.float32)
    layout = SimpleNamespace(cell_l2g=np.array([0]), n_owned_cells=1)
    payload = dict(
        previous=[None],
        forcing=[dict(rthdynten=rate, rqvdynten=rate, time_seconds=60.0)],
        layouts=[layout],
    )
    got = remap_step_carriers(payload, w, [layout])["forcing"][0]
    np.testing.assert_allclose(got["rthdynten"][:, 0], [1, 2, 4.5])
    assert got["time_seconds"] == 60.0


def test_projection_preserves_nonhydrostatic_acceleration_and_two_column_budgets():
    nl, nc = 20, 7
    z = np.arange(nl)[:, None] * 400.0 + 200.0
    r = np.broadcast_to(1.2 * np.exp(-z / 8500), (nl, nc)).copy()
    h = r * (300.0 + 0.004 * z)
    p = 1e5 * (287 * h / 1e5) ** 1.4
    ref = SimpleNamespace(
        rho_base=0.98 * r,
        rho_theta_base=0.99 * h,
        exner_base=(287 * 0.99 * h / 1e5) ** 0.4,
    )
    v = SimpleNamespace(
        zz=np.ones_like(r),
        dzw=np.full(nl, 400.0),
        rdzu=np.full(nl, 1 / 400.0),
        fzm=np.full(nl, 0.5),
        fzp=np.full(nl, 0.5),
    )
    q = 0.01 * np.exp(-z / 3000.0) * np.ones_like(r)
    pb = 287 * ref.rho_theta_base * ref.exner_base
    old = dict(
        rho=r,
        pressure_perturbation=p - pb,
        density_perturbation=r - ref.rho_base,
        scalars=q[None],
    )
    acc = source_pressure_acceleration(old, v, [0])
    noisy_r = r * (1 + 0.002 * np.sin(z / 2000.0))
    noisy_r *= (v.dzw @ r) / (v.dzw @ noisy_r)
    noisy_h = h * (1 + 0.002 * np.cos(z / 2000.0))
    noisy_h *= (v.dzw @ h) / (v.dzw @ noisy_h)
    r1, h1, rec = project_columns(noisy_r, noisy_h, r * q, acc, v, ref)
    new = dict(
        rho=r1,
        pressure_perturbation=1e5 * (287 * h1 / 1e5) ** 1.4 - pb,
        density_perturbation=r1 - ref.rho_base,
        scalars=(r * q / r1)[None],
    )
    np.testing.assert_allclose(
        source_pressure_acceleration(new, v, [0]), acc, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(v.dzw @ r1, v.dzw @ noisy_r, rtol=1e-13)
    np.testing.assert_allclose(v.dzw @ h1, v.dzw @ noisy_h, rtol=1e-13)
    assert np.all(r1 > 0) and np.all(h1 > 0)
    assert rec["iterations"] < 8


def test_conservative_rounding_closes_products_without_touching_exact_entries():
    rng = np.random.default_rng(7)
    exact = rng.uniform(0.2, 1.3, (30, 132))
    exact[:, 10:20] = 1.0
    measure = rng.uniform(1e5, 1e8, exact.shape)
    target = float(np.sum(exact * measure))
    out = conservative_float32(exact, measure, target)
    assert abs(np.sum(out.astype(float) * measure) / target - 1) < 1e-9
    np.testing.assert_array_equal(out[:, 10:20], exact[:, 10:20])
    nearest = exact.astype(np.float32)
    assert np.all(
        abs(out.astype(float) - exact) <= abs(np.spacing(nearest).astype(float))
    )


def test_vector_and_interface_caches_use_their_physical_vertical_layout():
    from woof.hex.moving_physics import carry_horizontal_vector

    w = mapper([[0], [2], [4], [6]], [[0], [1], [3], [6]], endpoint_alignment=False)
    mesh = SimpleNamespace(arrays=dict(latCell=np.zeros(1), lonCell=np.zeros(1)))
    u = np.array([1, 3, 5], np.float32)[:, None, None]
    v = np.zeros((3, 2, 1), np.float32)
    mapped_u, mapped_v = carry_horizontal_vector(u, v, w, mesh, mesh, atmospheric=True)
    np.testing.assert_allclose(mapped_u[:, 0, 0], [1, 2, 4.5])
    np.testing.assert_array_equal(mapped_v, v)
    interface = np.array([0, 2, 4, 6], np.float32)[:, None, None]
    np.testing.assert_array_equal(
        field_intensive(interface, w, "seam/interface_rate")[:, 0, 0], [0, 1, 3, 6]
    )


def test_tracer_rounding_conserves_mass_against_already_rounded_density():
    rng = np.random.default_rng(19)
    rho = rng.uniform(0.1, 1.2, (20, 132)).astype(np.float32)
    tracer_mass = rng.uniform(0, 0.001, rho.shape)
    measure = rng.uniform(1e5, 1e8, rho.shape)
    target = np.sum(tracer_mass * measure)
    q = conservative_float32(tracer_mass / rho, measure * rho, target)
    assert abs(np.sum(q.astype(float) * rho * measure) / target - 1) < 5e-10
    with pytest.raises(ValueError, match="conservation tolerance"):
        conservative_float32(np.ones((2, 2)), np.ones((2, 2)), 4.01)


def test_tracer_only_redistribution_is_not_overwritten_by_dry_identity_copy():
    from woof.hex.moving_balance import remap_balanced_thermodynamics

    nl, nc = 4, 132
    z = np.broadcast_to(np.arange(nl + 1)[:, None], (nl + 1, nc))
    w = mapper(z, z, endpoint_alignment=False)
    original_map = w.mass_intensive

    def independently_reconciled_product(values, density=None):
        result = original_map(values, density=density)
        if np.max(values) < 0.1:  # An independent passive-tracer correction.
            packet = result[:, 0] * 0.01
            result[:, 0] += packet
            result[:, 1] -= packet
        return result

    w.mass_intensive = independently_reconciled_product
    r = np.ones((nl, nc), np.float32)
    h = r * np.float32(300.0)
    ex = (np.float32(287.0 / 1e5) * h) ** np.float32(0.4)
    v = SimpleNamespace(
        dzw=np.ones(nl),
        zz=r,
        rdzu=np.ones(nl),
        fzm=np.full(nl, 0.5),
        fzp=np.full(nl, 0.5),
    )
    ref = SimpleNamespace(rho_base=r, rho_theta_base=h, exner_base=ex)
    q = np.full((1, nl, nc), 0.001, np.float32)
    payload = dict(
        vertical=v,
        scalar_names=["passive"],
        state=dict(
            rho=r,
            rho_theta=h,
            scalars=q,
            exner=ex,
            density_perturbation=r * 0,
            pressure_perturbation=r * 0,
        ),
    )
    rho, _, got, _, _, receipts, record = remap_balanced_thermodynamics(
        payload, w, SimpleNamespace(vertical=v, reference=ref)
    )
    assert record["unchanged_columns"] == nc
    np.testing.assert_allclose(got[0, :, 0], q[0, :, 0] * 1.01, rtol=1e-7)
    np.testing.assert_allclose(got[0, :, 1], q[0, :, 1] * 0.99, rtol=1e-7)
    before = receipts["passive"]["before"][0]
    assert abs(receipts["passive"]["absolute_error"][0] / before) < 5e-10


def test_balance_scope_needs_the_saved_pressure_split_and_honours_the_knob():
    # An analysis transfer carries prognostic fields only: it keeps the previous
    # path (by name in its receipt) instead of failing on a missing key.
    from woof.hex.moving_balance import balance_scope

    w = mapper([[0], [2], [4]], [[0], [1], [4]], endpoint_alignment=False)
    moving = dict(rho=1, pressure_perturbation=1, density_perturbation=1, exner=1)
    analysis = dict(rho=1, rho_theta=1, scalars=1)
    assert balance_scope(w, moving, {})[0] is True
    apply, reason = balance_scope(w, analysis, {})
    assert apply is False and "saved native pressure split" in reason
    apply, reason = balance_scope(w, moving, {"WOOF_HEX_MOVING_BALANCE": "0"})
    assert apply is False and "comparison knob" in reason
    flat = SimpleNamespace(redistributes_vertical=False)
    assert balance_scope(flat, moving, {})[0] is False


def test_sparse_budget_chooses_nonprefix_adjacent_floats_without_relaxing_tolerance():
    ulp=float(np.spacing(np.float32(1.)))
    exact=np.full(3,1.+.49*ulp)
    measure=np.array([8.,5.,3.])/ulp
    target=float(measure.sum()+5.)
    result=conservative_float32(exact,measure,target)
    assert float(np.sum(result.astype(float)*measure))==target
    np.testing.assert_array_equal(result,np.array([1.,np.nextafter(np.float32(1.),np.float32(2.)),1.],dtype=np.float32))
    assert np.all(abs(result.astype(float)-exact)<ulp)
    with pytest.raises(ValueError,match='conservation tolerance'):
        conservative_float32(exact,measure,target+0.5)


def test_heavy_quantum_overshoot_closes_within_one_float_per_value():
    # run_0bd003fbf0fc2910 (2026-10-26): a sparse hydrometeor over fine cells and a few coarse
    # background cells; the index-order prefix stopped on a heavy quantum and overshot by twice
    # the tolerance. Seed 2 of this shape refused on hex 0.4.6.
    rng = np.random.default_rng(2)
    n = 2046
    q = rng.uniform(0, 6e-4, n) * rng.uniform(0, 1, n) ** 2
    mass = np.where(rng.uniform(size=n) < 0.1, 10 ** rng.uniform(11, 12.3, n),
                    10 ** rng.uniform(9, 10.5, n))
    target = float(np.sum(q * mass))
    out = conservative_float32(q, mass, target)
    assert abs(float(np.sum(out.astype(float) * mass)) - target) <= 5e-10 * target
    nearest = q.astype(np.float32)
    assert np.all(abs(out.astype(float) - q) <= abs(np.spacing(nearest).astype(float)))
