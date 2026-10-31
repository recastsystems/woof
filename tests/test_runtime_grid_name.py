"""``GPUWM_GRID_NAME``: what a run's maps call its outermost domain instead of ``d01``.

The breakage (WOOF Hex with Regional, 2026-10-28): a nest added to a hex run is
its own one-domain WOOF run, so every nest map was headlined ``d01 3 km`` while
the site files the nest as d05.  ``woof hex add-nest`` sets the name for its
``woof sim``; the frames carry it and ``rw_wrfbatch`` prints it in the headline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from woof import runtime
from woof.io import wrfout


@pytest.fixture
def bare_attrs(monkeypatch):
    monkeypatch.setattr(wrfout, "wrf_global_attrs", lambda *args, **kwargs: {"GRID_ID": 1})
    monkeypatch.setattr(wrfout, "initial_condition_global_attrs", lambda *args, **kwargs: {})

    def attrs(grid_id=None):
        domain = None
        if grid_id is not None:
            domain = SimpleNamespace(grid_id=grid_id, parent_id=1, i_parent_start=1, j_parent_start=1,
                                     parent_grid_ratio=3, run=SimpleNamespace(grid_id=grid_id, dt=6.0))
        return runtime._global_wrf_attrs(object(), runtime.datetime(2026, 10, 28, 18), domain=domain)

    return attrs


def test_unset_writes_nothing(monkeypatch, bare_attrs):
    monkeypatch.delenv(runtime.GRID_NAME_ENV, raising=False)
    assert runtime.GRID_NAME_ATTR not in bare_attrs()
    assert runtime.GRID_NAME_ATTR not in bare_attrs(1)


def test_the_outermost_domain_takes_the_name_and_nests_keep_theirs(monkeypatch, bare_attrs):
    monkeypatch.setenv(runtime.GRID_NAME_ENV, "  nest ")
    assert bare_attrs()[runtime.GRID_NAME_ATTR] == "nest"
    assert bare_attrs(1)[runtime.GRID_NAME_ATTR] == "nest"
    assert runtime.GRID_NAME_ATTR not in bare_attrs(2)
    assert runtime.GRID_NAME_ATTR == "GPUWM_GRID_NAME"


@pytest.mark.parametrize("bad", ["a|b", "x" * 33, "nest\x07"])
def test_a_name_the_headline_cannot_draw_is_refused(monkeypatch, bad):
    monkeypatch.setenv(runtime.GRID_NAME_ENV, bad)
    with pytest.raises(ValueError, match="GPUWM_GRID_NAME"):
        runtime.grid_name_for(1)
