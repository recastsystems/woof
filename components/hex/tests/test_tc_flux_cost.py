"""Curve-only native calls must close the shared event-ring sample."""
from types import SimpleNamespace


def test_curve_only_native_cost_closes_samples_without_field_binding():
    from woof.hydro.cost import HydroCostMeter
    from woof.tc.surface import SurfaceBundle
    from woof.hex.mpasphys.tc_surface import NativeSurface

    class Event:
        done = True
        def record(self, stream):
            pass
        def synchronize(self):
            pass

    meter = HydroCostMeter(ring=2, events=Event, elapsed=lambda a, b: .25)
    # A stream sentinel avoids importing CUDA in this event-lifecycle check.
    original_record = meter._record
    meter._record = lambda index, stream: original_record(index, object())
    surface = object.__new__(SurfaceBundle)
    surface.cfg = SimpleNamespace(tc_spray='off')
    surface.cost = meter
    adapter = object.__new__(NativeSurface)
    adapter.cfg = surface.cfg
    adapter.tc_surface = surface
    adapter.fields = lambda: (_ for _ in ()).throw(AssertionError('unneeded field binding'))
    for _ in range(5):
        meter.before_lsm_start()
        meter.before_lsm_end()
        adapter.spray(None)
    meter.finish()
    assert meter.calls == meter.sampled == 5
    assert meter.unsampled == 0
    assert meter.before_ms == 1.25


def test_class_receipts_select_active_card_and_executed_steps(monkeypatch):
    import sys
    import woof.hex.tc_flux as tc
    active = SimpleNamespace(_seam=SimpleNamespace(_driver=SimpleNamespace(tc_surface=object())))
    current = []
    class Device:
        def __init__(self, index):
            self.index = index
        def __enter__(self):
            current.append(self.index)
        def __exit__(self, *args):
            current.pop()
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(Device=Device)))
    monkeypatch.setattr(tc, 'final_receipt', lambda backend, steps: dict(card=current[-1], steps=steps))
    stacks = [dict(backend=object()), dict(backend=active), dict(backend=active)]
    rows = tc.class_receipts(stacks, [40, 20, 10], devices=[0, 2, 1])
    assert [(r['index'], r['receipt']) for r in rows] == [
        (1, dict(card=2, steps=20)), (2, dict(card=1, steps=10))]
    assert not current


def test_inactive_classes_do_not_import_cuda(monkeypatch):
    import sys
    from woof.hex.tc_flux import class_receipts
    monkeypatch.setitem(sys.modules, 'cupy', None)
    assert class_receipts([dict(backend=object())], [100], devices=[0]) == []
