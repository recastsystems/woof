"""CU99 keeps native failures while allowing the driver to defer readback."""
import pytest

from conftest import requires_gpu


def test_old_kf_species_cover_current_microphysics_packages():
    from woof.config import MP_PHYSICS_ACCEPTED
    from woof.core.kf import KFOldSpecies, kf_old_species_for_microphysics

    assert all(isinstance(kf_old_species_for_microphysics(mp), KFOldSpecies)
               for mp in MP_PHYSICS_ACCEPTED)
    assert kf_old_species_for_microphysics(3) == KFOldSpecies(False, False, False, True)
    assert kf_old_species_for_microphysics(5) == KFOldSpecies(False, True, False, True)
    assert kf_old_species_for_microphysics(4) == KFOldSpecies(False, True, True, True)


@requires_gpu
def test_old_kf_fatal_bits_survive_multiple_deferred_calls():
    import cupy as cp
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core.kf import raise_kf_old_fatal

    ledger = HealthLedger(label="CU99 negative control")
    with deferring(ledger):
        raise_kf_old_fatal(cp.asarray([0, 1, 0], dtype=cp.int32))
        raise_kf_old_fatal(cp.asarray([2, 0, 3], dtype=cp.int32))
    assert ledger.records == 2 and ledger.drains == 0
    with pytest.raises(RuntimeError) as caught:
        ledger.drain()
    assert 'qg = 0 at the surface' in str(caught.value)
    assert 'MASS DOES NOT BALANCE' in str(caught.value)
    assert 'QVERR' in str(caught.value)
    ledger.drain()


@requires_gpu
def test_old_kf_native_fatal_without_a_ledger_stops_immediately():
    import cupy as cp
    from woof.core.kf import raise_kf_old_fatal

    raise_kf_old_fatal(cp.zeros(257, dtype=cp.int32))
    with pytest.raises(RuntimeError, match='QVERR'):
        raise_kf_old_fatal(cp.asarray([0, 3], dtype=cp.int32))


@requires_gpu
@pytest.mark.parametrize('microphysics', [0, 1, 3, 5, 6, 50])
def test_old_kf_output_validation_defers_real_device_status(microphysics):
    from types import SimpleNamespace
    import cupy as cp
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core.kf import KainFritsch, kf_old_species_for_microphysics
    from woof.core.physics import (_NativeKFCumulusResult,
        _is_native_kf_result, _validate_native_kf_result)

    owner = KainFritsch(99)
    species = kf_old_species_for_microphysics(microphysics)
    shape = (12, 2, 3)
    volume = lambda: cp.zeros(shape, cp.float32)
    surface = lambda: cp.zeros(shape[1:], cp.float32)
    state = SimpleNamespace(qi=volume() if species.f_qi else None,
        qs=volume() if species.f_qs else None,
        scratch=lambda dims, name: cp.zeros(dims, cp.float32))
    cfg = SimpleNamespace(cu_physics=99, mp_physics=microphysics)
    result = _NativeKFCumulusResult(owner=owner,
        rthcuten=volume(), rqvcuten=volume(), rqccuten=volume(),
        rqrcuten=volume() if species.f_qr else None,
        rqicuten=volume() if species.f_qi else None,
        rqscuten=volume() if species.f_qs else None,
        nca_seconds=surface(), pratec=surface())
    assert _is_native_kf_result(result, owner, state, cfg, shape)
    assert not _is_native_kf_result(result, KainFritsch(99), state, cfg, shape)
    ledger = HealthLedger(label='CU99 tendency negative control')
    with deferring(ledger):
        _validate_native_kf_result(result, state)
        result.rqvcuten[0, 0, 0] = cp.nan
        _validate_native_kf_result(result, state)
    assert ledger.records == 2 and ledger.drains == 0
    with pytest.raises(FloatingPointError, match='cumulus rqvcuten'):
        ledger.drain()
