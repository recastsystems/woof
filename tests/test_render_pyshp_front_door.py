"""The pyshp prerequisite is answered BEFORE the work, not after it.

The DA nowcast's render stage is the last thing a run does -- after the
survey, the fetch, the preparation, the free forecast and every DA cycle
-- and it opened its basemap with a bare ``import shapefile``.  A run
that was always going to fail therefore failed at the end, with
``ModuleNotFoundError: No module named 'shapefile'`` and no remedy:
the most work destroyed and the least information given of any failure
in the reachability audit.

These tests bind the fix in both directions, which is the only way a
capability check is worth anything: the refusal must FIRE when pyshp is
absent and must STAY SILENT when it is present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from woof import rustwx


def _nowcast_args(**overrides):
    """The narrowest args object validate_analysis_flags reads."""

    defaults = dict(
        reflectivity_analysis=False, hydrometeors=False,
        positivity_policy=None, dealias=False, dealias_refinement=None,
        dealias_engine=None, clear_air_analysis=False, goes_cwp=None,
        cwp_vertical_loc_m=None, surface_obs=None, sfc_t2_sigma_k=None,
        sfc_wspd_sigma_ms=None, stop_after=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# The capability check itself, both directions
# ---------------------------------------------------------------------------

def test_pyshp_available_answers_true_when_the_reader_imports():
    """The instrument must not report a gap that is not there."""

    pytest.importorskip("shapefile")
    assert rustwx.pyshp_available() is True
    # ... and the gate it drives stays silent.
    rustwx.require_pyshp()


def test_pyshp_available_answers_false_when_the_reader_is_absent(
        monkeypatch):
    """The other direction, which is the one that was never asked."""

    import importlib.util

    real = importlib.util.find_spec

    def absent(name, *args, **kwargs):
        if name == "shapefile":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", absent)
    assert rustwx.pyshp_available() is False
    with pytest.raises(ImportError) as caught:
        rustwx.require_pyshp()
    assert "pyshp" in str(caught.value)


def test_the_remedy_names_a_command_that_can_actually_supply_it():
    """A remedy that cannot supply what it names is worse than none.

    The reader arrives by pip, and since 2.8.0 so does the geometry pyshp
    reads: the ``recast-woof-data`` package every install pulls carries it.  So
    the pip line is the whole fix, and the sentence says where the
    geometry comes from rather than sending the reader to
    ``woof fetch-bridges`` for files the install already has.
    """

    remedy = rustwx.PYSHP_REMEDY
    assert "pip install" in remedy
    assert "recast-woof[render]" in remedy
    assert "recast-woof-data" in remedy
    assert "fetch-bridges" not in remedy


# ---------------------------------------------------------------------------
# The position: the front door, before the forecast
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The direct doors: the renderer run on its own
# ---------------------------------------------------------------------------
