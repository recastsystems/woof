"""Sea ice is decided for the domain when preparation builds it in row slabs.

THE BREAKAGE THIS PREVENTS (2.9.12, 2026-11-06, the HiresW Alaska row,
1385 x 1133 in 19 preparation slabs): the ice model switches on where the
fields it is handed hold ocean ice, and each slab handed it only its own rows.
Slab 0 had none and the Arctic slabs did, so preparation stopped with "slab 12
produced carrier 'fields/ice_concentration_change' that slab 0 did not; the
manifest is not slab-invariant".
"""
from __future__ import annotations

import pytest

from woof.ingest import prepared_store
from woof.ocean import seaice


def test_a_slab_disagreement_rebuilds_once_with_the_domain_answer(monkeypatch):
    calls = []

    def once(path, *args, **kwargs):
        calls.append(seaice.DOMAIN_OCEAN_ICE.get())
        if not seaice.DOMAIN_OCEAN_ICE.get():
            raise prepared_store._SlabIceDisagreement("slab 12 holds ocean ice that slab 0 lacks")
        return "store"

    monkeypatch.setattr(prepared_store, "_store_from_prepared_cache_once", once)
    assert prepared_store.store_from_prepared_cache("cache", expected_identity={}) == "store"
    assert calls == [False, True]
    assert seaice.DOMAIN_OCEAN_ICE.get() is False


def test_a_domain_whose_slabs_agree_is_built_once(monkeypatch):
    calls = []
    monkeypatch.setattr(prepared_store, "_store_from_prepared_cache_once",
                        lambda path, *a, **k: calls.append(1) or "store")
    assert prepared_store.store_from_prepared_cache("cache") == "store"
    assert calls == [1]


def test_a_second_disagreement_is_not_retried(monkeypatch):
    def once(path, *args, **kwargs):
        raise prepared_store._SlabIceDisagreement("slabs still disagree")

    monkeypatch.setattr(prepared_store, "_store_from_prepared_cache_once", once)
    with pytest.raises(prepared_store.PreparedStoreError, match="still disagree"):
        prepared_store.store_from_prepared_cache("cache")
    assert seaice.DOMAIN_OCEAN_ICE.get() is False
