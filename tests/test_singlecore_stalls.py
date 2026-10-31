"""The sizing audit keeps exclusions and overlap errors after indexing."""
from dataclasses import replace
import pytest
from woof.core import preflight as pf


def test_index_matches_the_full_audit_for_every_literal_and_prefix():
    audit = pf.SCRATCH_SLOT_LIFETIME_AUDIT
    slots = {'unclassified_slot'}
    for row in audit:
        for pattern in row.patterns:
            slots.add(pattern.rstrip('*'))
            slots.add(pattern.rstrip('*') + 'probe')
    for slot in slots:
        matches = [row for row in audit if any(
            slot.startswith(p[:-1]) if p.endswith('*') else slot == p
            for p in row.patterns)]
        if len(matches) > 1:
            with pytest.raises(RuntimeError, match='overlaps'):
                pf.scratch_slot_lifetime(slot)
        else:
            assert pf.scratch_slot_lifetime(slot) is (matches[0] if matches else None)


def test_audit_replacement_invalidates_cached_misses_and_keeps_overlap_errors(monkeypatch):
    slot = 'index_audit_probe'
    assert pf.scratch_slot_lifetime(slot) is None
    row = replace(pf.SCRATCH_SLOT_LIFETIME_AUDIT[0], patterns=(slot, 'index_audit_*'))
    monkeypatch.setattr(pf, 'SCRATCH_SLOT_LIFETIME_AUDIT', (row,))
    assert pf.scratch_slot_lifetime(slot) is row  # two patterns of ONE row
    other = replace(row, patterns=('index_*',))
    monkeypatch.setattr(pf, 'SCRATCH_SLOT_LIFETIME_AUDIT', (row, other))
    with pytest.raises(RuntimeError, match='overlaps'):
        pf.scratch_slot_lifetime(slot)


def test_core_handle_factory_installs_prebuilt_loader_without_a_supervisor(monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof import kernel_aot
    from woof.core import device_cache
    calls = []
    fake = SimpleNamespace(cuda=SimpleNamespace(compiler=object(),
        Device=lambda: SimpleNamespace(id=0)))
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    monkeypatch.setattr(kernel_aot, 'install', lambda cp: calls.append(cp))
    device_cache._install_prebuilt_images.cache_clear()
    try:
        @device_cache.cuda_cache()
        def handle():
            assert calls == [fake]
            return object()
        assert handle() is handle()
        assert calls == [fake]
    finally:
        device_cache._install_prebuilt_images.cache_clear()
