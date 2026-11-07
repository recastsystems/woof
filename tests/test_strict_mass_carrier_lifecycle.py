"""Cold transport capacity does not imply a completed acoustic producer."""
from types import SimpleNamespace

import pytest

from woof.state_serialization_contract import carried_wrf_muts, carried_wrf_mu_halo
from woof.io.restart import RestartMismatchError, validate_strict_mass_cold
from tilestream.physics_inventory import carrier_scalars, set_carrier_scalars


def test_reusing_a_tile_resets_the_first_acoustic_producer_status():
    mass = object()
    tile = SimpleNamespace(elapsed_seconds=0., physics=None,
                           existing_scratch=lambda slot: mass)
    assert carried_wrf_muts(tile) is mass
    assert carried_wrf_mu_halo(tile) is mass
    for cold in (True, False, True):
        set_carrier_scalars(tile, dict(elapsed_seconds=0., wrf_mass_carriers_cold=cold))
        assert (carried_wrf_muts(tile) is None) == cold
        assert (carried_wrf_mu_halo(tile) is None) == cold
        assert carrier_scalars(tile)['wrf_mass_carriers_cold'] is cold
    set_carrier_scalars(tile, dict(elapsed_seconds=3.))
    assert carried_wrf_muts(tile) is mass


def test_cold_marker_requires_a_real_transport_plane():
    assert validate_strict_mass_cold({}) is False
    assert validate_strict_mass_cold(dict(wrf_mass_carriers_cold=True,
        array_manifest={'scratch/wrf_exact_muts': {}})) is True
    with pytest.raises(RestartMismatchError, match='requires its allocated'):
        validate_strict_mass_cold(dict(wrf_mass_carriers_cold=True))
    with pytest.raises(RestartMismatchError, match='must be a boolean'):
        validate_strict_mass_cold(dict(wrf_mass_carriers_cold='false'))


@pytest.mark.parametrize("template_status", (None, True, False))
@pytest.mark.parametrize("stored_status", (True, False))
def test_canonical_header_reads_the_store_status_instead_of_stale_template_status(
        template_status, stored_status):
    from woof.core.streamed_state import CanonicalStoreState
    template = SimpleNamespace()
    if template_status is not None:
        template._wrf_mass_carriers_cold = template_status
    facade = CanonicalStoreState(template, SimpleNamespace(nx=2, ny=2, nz=4),
        store={}, geography={}, scalars={'elapsed_seconds': 3., 'wrf_mass_carriers_cold': stored_status},
        inventory={}, geography_inventory={})
    assert facade._wrf_mass_carriers_cold is stored_status


@pytest.mark.gpu
def test_dry_graph_key_keeps_the_first_diagnosis_separate():
    pytest.importorskip("cupy")
    from woof.config import RunConfig
    from tilestream.graphcap import cadence_key
    cfg = RunConfig(nx=24, ny=24, nz=29, dx=3000., dy=3000.,
                    ztop=20000., dt=3., run_seconds=9.)
    state = SimpleNamespace(physics=None)
    established = cadence_key(state, cfg)
    state._wrf_mass_carriers_cold = True
    assert cadence_key(state, cfg) != established
    state._wrf_mass_carriers_cold = False
    assert cadence_key(state, cfg) == established


@pytest.mark.gpu
@pytest.mark.parametrize('kind', ('u', 'v'))
def test_nest_coupling_ignores_cold_capacity_until_the_halo_is_produced(kind):
    """A reserved, unwritten ring must behave exactly like an absent ring."""
    import cupy as cp
    from woof.core.nest_round_trip import couple_round_trip, couple_halo_faces

    ny, nx, nz = 4, 5, 3
    mu = cp.arange(ny * nx, dtype=cp.float32).reshape(ny, nx) / 7
    state = SimpleNamespace(
        thp=cp.zeros((nz, ny, nx), cp.float32), has_msf=False,
        mub2d=cp.full((ny, nx), 85000., cp.float32), mup=mu,
        c1h=cp.asarray([1., .7, .2], cp.float32),
        c2h=cp.asarray([0., 100., 500.], cp.float32),
        c1f=cp.asarray([1., .8, .4, 0.], cp.float32),
        c2f=cp.asarray([0., 50., 200., 1000.], cp.float32),
        existing_scratch=lambda slot: None)
    shape = (nz, ny, nx + 1) if kind == 'u' else (nz, ny + 1, nx)
    field = (cp.arange(nz * shape[1] * shape[2], dtype=cp.float32)
             .reshape(shape) + 1) / 17
    setattr(state, kind, field)
    run = SimpleNamespace(specified=True)
    expected = field.copy()
    couple_round_trip(state, run, {kind: expected})
    expected_face = cp.full(shape, 11., cp.float32)
    couple_halo_faces(state, kind, expected_face)

    # Poison unproduced transport memory so reading it cannot pass by chance.
    reserved = cp.full((ny, nx), cp.nan, cp.float32)
    state.existing_scratch = lambda slot: reserved
    state._wrf_mass_carriers_cold = True
    actual = field.copy()
    couple_round_trip(state, run, {kind: actual})
    actual_face = cp.full(shape, 11., cp.float32)
    couple_halo_faces(state, kind, actual_face)
    assert cp.asnumpy(actual).tobytes() == cp.asnumpy(expected).tobytes()
    assert cp.asnumpy(actual_face).tobytes() == cp.asnumpy(expected_face).tobytes()

    state._wrf_mass_carriers_cold = False
    couple_halo_faces(state, kind, actual_face)
    face = actual_face[:, :, 0] if kind == 'u' else actual_face[:, 0, :]
    assert bool(cp.isnan(face).all()), 'a produced halo must be consumed'
