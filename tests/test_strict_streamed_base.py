"""Strict slabs use WRF theta while checkpoints retain the prepared base."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.core.state import adopt_wrf_theta_carrier
from woof.state_serialization_contract import STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS
from tilestream.restart_stream import domain_setup_from_stream, _HeaderView

@pytest.mark.parametrize('shape', [(4,), (4, 3, 5)])
def test_strict_slab_adopts_theta_and_preserves_original_base(monkeypatch, shape):
    import woof.wrf_exact as exact
    monkeypatch.setattr(exact, 'ENABLED', True)
    base = np.linspace(290., 464., np.prod(shape), dtype=np.float32).reshape(shape)
    state = SimpleNamespace(thb=base.copy(), thp=np.full((4, 3, 5), 2.5, np.float32))
    full = base if base.ndim == 3 else base[:, None, None]
    expected = (full + state.thp) - np.float32(300.)
    adopt_wrf_theta_carrier(state)
    np.testing.assert_array_equal(state.thp, expected)
    np.testing.assert_array_equal(state._setup_thb, base)
    assert np.all(state.thb == 300.)
    adopt_wrf_theta_carrier(state)
    np.testing.assert_array_equal(state.thp, expected)


def test_streamed_header_uses_domain_original_base_instead_of_last_slab():
    template = SimpleNamespace(**{k: np.ones(4, np.float32) for k in STATE_SETUP_ARRAYS},
                               **{k: 0 for k in STATE_SETUP_SCALARS})
    template._setup_thb = np.full((4, 2, 5), 999., np.float32)
    original = np.arange(4*7*5, dtype=np.float32).reshape(4, 7, 5) + 290.
    live = np.full_like(original, 300.)
    setup = domain_setup_from_stream({'setup/thb': live}, template, setup_thb=original)
    view = _HeaderView(setup, template)
    np.testing.assert_array_equal(view.thb, live)
    np.testing.assert_array_equal(view._setup_thb, original)


def test_default_theta_adoption_is_a_noop(monkeypatch):
    import woof.wrf_exact as exact
    monkeypatch.setattr(exact, 'ENABLED', False)
    state = SimpleNamespace(thb=np.array([290., 464.], np.float32),
                            thp=np.ones((2, 1, 1), np.float32))
    base, perturbation = state.thb.copy(), state.thp.copy()
    adopt_wrf_theta_carrier(state)
    np.testing.assert_array_equal(state.thb, base)
    np.testing.assert_array_equal(state.thp, perturbation)
    assert not hasattr(state, '_setup_thb')


def test_domain_scalar_carrier_compares_its_words():
    from woof.ingest.prepared_store import _hold_domain_carrier, PreparedStoreError
    scalar = np.array(1.25, dtype=np.float32)
    _hold_domain_carrier(scalar, np.array(1.25, np.float32), 'test', 0)
    _hold_domain_carrier(scalar, np.array(1.25, np.float32), 'test', 1)
    with pytest.raises(PreparedStoreError):
        _hold_domain_carrier(scalar, np.array(1.5, np.float32), 'test', 2)

# Release census replacements (reviewed against the merged sources).
# Strict theta/base adoption replaces the retired streamed-base refusal suite.
