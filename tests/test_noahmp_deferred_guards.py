"""Native fatal conditions survive delayed health reads and later healthy slabs."""
import pytest
from conftest import requires_gpu

pytestmark = requires_gpu


def test_land_solver_rejects_ice_at_drain_without_a_stage_host_read(monkeypatch):
    import cupy as cp
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core.noahmp_energy_slab import _refuse_unadmitted
    ledger = HealthLedger()
    original = cp.asnumpy
    def no_read(*args, **kwargs):
        raise AssertionError('Noah-MP stage synchronized its validation')
    with deferring(ledger), monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', no_read)
        _refuse_unadmitted(cp.asarray([1, 0]), cp.ones(2, dtype=cp.int32))
        _refuse_unadmitted(cp.zeros(2, dtype=cp.int32), cp.ones(2, dtype=cp.int32))
    with pytest.raises(NotImplementedError, match='glacier'):
        ledger.drain()


@pytest.mark.parametrize('code,kind', [(1, 'ShortwaveBalanceError'),
                                       (2, 'EnergyBalanceError'),
                                       (3, 'WaterBalanceError')])
def test_budget_fault_survives_a_later_healthy_slab(code, kind, monkeypatch):
    import cupy as cp
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core import noahmp_sflx_post_slab as post
    ledger = HealthLedger()
    def no_read(*args, **kwargs):
        raise AssertionError('Noah-MP stage synchronized its validation')
    with deferring(ledger), monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', no_read)
        post._raise_for_status(cp, cp.asarray([0, code], dtype=cp.int32), None, None)
        post._raise_for_status(cp, cp.zeros(2, dtype=cp.int32), None, None)
    with pytest.raises(getattr(post, kind)):
        ledger.drain()
