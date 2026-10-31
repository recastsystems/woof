"""The f000 start-frame reflectivity is formed in column slabs, with the whole-state words.

THE BREAKAGE THIS PREVENTS: a same-host multi-card run holds its GLOBAL f000 frame on rank 0, and the TEMPO
start reflectivity copied the whole global state to rank 0's card beside step 1 (~8 GB on NOAA's conus3km over
4 x 96 GB cards, where rank 0 has ~6 GB left).  ``refl10cm_staged`` forms it one slab at a time.  These tests
go red if a slab edge drops, repeats or shifts a column, or if the f000 thread stops using the slabs.
"""

from __future__ import annotations

import inspect

import numpy as np

from woof.hex.mpasphys.rrfs import tempo_refl_v303 as refl


class _HostCupy:
    """numpy with cupy's asnumpy, so the slab assembly runs on the CPU tier."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(a, stream=None):
        return np.array(a, copy=True)


def _columnwise(fields, *, pin_set, hailaware, melting, cp):
    # A per-column stand-in for the kernel: each output word depends on its own column only.
    t = fields['theta'] * np.float32(1.01)
    return (np.cumsum(fields['qr'] + fields['qs'] * np.float32(2.0), axis=0) * np.float32(1e4) + t
            - np.float32(300.0)).astype(np.float32)


def _state(nz=7, ncells=103):
    rng = np.random.default_rng(20261029)
    state = {name: rng.uniform(0.0, 1e-3, (nz, ncells)).astype(np.float32)
             for name in ('qv', 'qr', 'nr', 'qs', 'qg')}
    state['theta'] = rng.uniform(280.0, 320.0, (nz, ncells)).astype(np.float32)
    state['p'] = rng.uniform(2e4, 1e5, (nz, ncells)).astype(np.float32)
    zgrid = np.cumsum(rng.uniform(100.0, 600.0, (nz + 1, ncells)), axis=0).astype(np.float32)
    return state, zgrid


def test_slabs_give_the_whole_state_words(monkeypatch):
    monkeypatch.setattr(refl, 'refl10cm', _columnwise)
    cp = _HostCupy()
    state, zgrid = _state()
    whole = _columnwise(state, pin_set='x', hailaware=True, melting=False, cp=cp)
    whole_max, whole_1km = refl.column_radar(whole, zgrid, cp=cp)
    for columns in (1, 10, 32, 103, 4096):
        got = refl.refl10cm_staged(state, zgrid, cp=cp, columns=columns)
        assert got['refl10cm'].view(np.uint32).tolist() == whole.view(np.uint32).tolist(), columns
        assert got['refl10cm_max'].view(np.uint32).tolist() == whole_max.view(np.uint32).tolist(), columns
        assert got['refl10cm_1km'].view(np.uint32).tolist() == whole_1km.view(np.uint32).tolist(), columns


def test_without_zgrid_only_the_column_maximum(monkeypatch):
    monkeypatch.setattr(refl, 'refl10cm', _columnwise)
    cp = _HostCupy()
    state, _ = _state()
    whole = _columnwise(state, pin_set='x', hailaware=True, melting=False, cp=cp)
    got = refl.refl10cm_staged(state, None, cp=cp, columns=16)
    assert 'refl10cm_1km' not in got
    expected = np.maximum(whole, np.float32(-35.0)).max(axis=0)
    assert got['refl10cm_max'].view(np.uint32).tolist() == expected.view(np.uint32).tolist()


def test_the_f000_thread_forms_the_frame_in_slabs():
    source = inspect.getsource(refl.StartFrameReflectivity._run)
    assert 'refl10cm_staged(' in source
    assert 'refl10cm(state' not in source
