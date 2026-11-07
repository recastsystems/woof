"""Real device transfers, failure retention and executor publication gates."""
from dataclasses import replace
from datetime import datetime
import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_pending_clean_copy_cannot_erase_later_failure(monkeypatch):
    import cupy as cp
    from woof.core.health_ledger import HealthLedger

    ledger = HealthLedger()
    finish = ledger._finish_pending
    monkeypatch.setattr(ledger, '_finish_pending',
                        lambda *, wait: finish(wait=True) if wait else None)

    def refuse(flags):
        raise FloatingPointError(f'first failure {flags:#x}')

    ledger.record('scheme', cp.asarray([0], dtype=cp.uint32), refuse)
    ledger.poll()
    for flag in (2, 0, 4, 0):
        ledger.record('scheme', cp.asarray([flag], dtype=cp.uint32), refuse)
        ledger.poll()
    assert ledger._pending is not None
    with pytest.raises(FloatingPointError, match='0x6'):
        ledger.drain()
    # A caught failure was cleared, including the asynchronous snapshot.
    ledger.drain()


def test_poll_uses_no_blocking_host_array_copy(monkeypatch):
    import cupy as cp
    from woof.core.health_ledger import HealthLedger

    ledger = HealthLedger()
    ledger.record('scheme', cp.zeros(1, dtype=cp.uint32), lambda _: None)
    with monkeypatch.context() as scoped:
        scoped.setattr(cp, 'asnumpy', lambda *a, **k: pytest.fail('blocking copy'))
        for _ in range(10):
            ledger.poll()
    ledger.close()


def _model():
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.experiment import DomainConfig, ExperimentConfig, ProjectionConfig, VerticalConfig
    from woof.verify.cases import wk82
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    cfg = replace(wk82.default_config(), nx=24, ny=24, nz=40, dt=3.,
                  run_seconds=6., output_interval_s=6., mp_physics=6,
                  cu_physics=11, kf_edrates=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82.wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    exp = ExperimentConfig(
        name='health_publication', start_time=datetime(2026, 10, 16, 9),
        run_seconds=6., vertical=VerticalConfig((), 0., 1, .2),
        projection=ProjectionConfig('lambert', 35., -97., 30., 60., -97.),
        feedback=0, restart_interval_s=6., domains=(DomainConfig(
            grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
            parent_grid_ratio=1, parent_time_step_ratio=1,
            history_interval_s=6., run=cfg, time_step=3),))
    return assemble_idealized_tree(exp, wk82.build(cfg, coord, base))


@pytest.mark.parametrize('consumer', ['history', 'restart', 'return'])
def test_executor_failure_cannot_publish_or_return(consumer, monkeypatch):
    import cupy as cp
    from woof.core.model import execute_experiment
    from woof.core import health_ledger

    model = _model()
    published = []
    original = health_ledger.HealthLedger._finish_pending
    monkeypatch.setattr(health_ledger.HealthLedger, '_finish_pending',
                        lambda self, *, wait: original(self, wait=True) if wait else None)

    def refuse(flags):
        raise FloatingPointError(f'scheme refused {flags}')

    calls = []
    def step(state, cfg, **kwargs):
        calls.append(1)
        health_ledger.read_status(cp.asarray([1 if len(calls) == 1 else 0], dtype=cp.uint32),
                                  site='scheme', describe=refuse)

    def publish(*args):
        if model.root.clock.step_count:
            published.append(consumer)

    kwargs = {consumer + '_handler': publish} if consumer != 'return' else {}
    with pytest.raises(FloatingPointError, match='scheme refused 1'):
        execute_experiment(model, steppers={1: step}, validate_state=False, **kwargs)
    assert not published
    assert len(calls) == 2


def test_resident_stability_check_is_deferred_and_preserves_transient_nan(monkeypatch):
    import cupy as cp
    from woof.core.model import execute_experiment
    from woof.core import health_ledger

    model = _model()
    original_finish = health_ledger.HealthLedger._finish_pending
    monkeypatch.setattr(health_ledger.HealthLedger, '_finish_pending',
                        lambda self, *, wait: original_finish(self, wait=True) if wait else None)
    asnumpy = cp.asnumpy
    def checked_copy(*args, **kwargs):
        assert all(frame.function != 'on_step' for frame in inspect.stack()), 'step blocked on health'
        return asnumpy(*args, **kwargs)
    monkeypatch.setattr(cp, 'asnumpy', checked_copy)
    calls = []
    def step(state, cfg, **kwargs):
        calls.append(1)
        state.thp.fill(np.float32(np.nan if len(calls) == 1 else 0.))
    with pytest.raises(RuntimeError, match='non-finite state by model step 1'):
        execute_experiment(model, steppers={1: step})
    assert len(calls) == 2


def test_poll_drain_respects_nondefault_stream():
    import cupy as cp
    from woof.core.health_ledger import HealthLedger

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        ledger = HealthLedger()
        ledger.record('scheme', cp.zeros(1, dtype=cp.uint32), lambda _: None)
        ledger.poll()
        def refuse(flags):
            raise FloatingPointError(f'stream failure {flags}')
        ledger.record('scheme', cp.ones(1, dtype=cp.uint32), refuse)
    with pytest.raises(FloatingPointError, match='stream failure 1'):
        ledger.drain()
