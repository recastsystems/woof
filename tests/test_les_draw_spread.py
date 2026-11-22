"""The engine-side draw aggregator, pinned to the oracle's published one.

CPU-only.  ``woof/verify/draw_spread.py`` exists because the oracle's
``moist_spread.py`` cannot aggregate engine receipts -- it indexes
``wrfout_sha256`` unconditionally -- and patching the oracle's instrument
so it can also score the engine it exists to check is what
``INSTRUMENT-HISTORY.md`` forbids.

That leaves two independent reductions of the same statistic, which is
worse than one shared one unless they are pinned together.  These tests
pin them: the engine's aggregator is run over the ORACLE's own committed
multi-draw receipts, with the oracle's own metric and identity lists, and
must reproduce the oracle's published aggregate JSON exactly -- the
capped-moist reference family (n = 4 once the campaign completed) and
the n = 2 qv-12 fallback, both banked.

Nothing here cuts a band.  Neither aggregator can.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from woof.verify import draw_spread

_ROOT = Path(__file__).resolve().parents[1]
_ORACLE = _ROOT / "tools" / "wrf_em_les_oracle"
_FIXTURES = (_ROOT / "docs" / "internal" / "receipts" / "les"
             / "moist-2026-08-04")

#: ``(published aggregate, its source receipts)`` -- the banked
#: multi-draw sets.  The reference family (n = 3 when this pin was
#: written; the completed campaign banked r4 and republished at n = 4,
#: which is what the aggregate on disk says) and the n = 2 registered
#: fallback: different draw counts, so the n < 2 / n >= 2 branches of
#: the statistic are both exercised on real data.
_PUBLISHED = (
    ("moist_spread_capped_km3_100m.json",
     ("capped_moist_km3_100m_r1.json", "capped_moist_km3_100m_r2.json",
      "capped_moist_km3_100m_r3.json", "capped_moist_km3_100m_r4.json")),
    ("moist_spread_capped12_km3_100m.json",
     ("capped12_moist_km3_100m_r1.json", "capped12_moist_km3_100m_r2.json")),
    ("moist_spread_cap20_km3_100m.json",
     ("cap20_moist_km3_100m_r1.json", "cap20_moist_km3_100m_r2.json")),
)

_HAVE = all((_FIXTURES / agg).exists()
            and all((_FIXTURES / r).exists() for r in srcs)
            for agg, srcs in _PUBLISHED)
oracle_fixtures = pytest.mark.skipif(
    not _HAVE,
    reason="the banked WRF moist-oracle aggregates live under a "
           "release-excluded path and are absent from a published clone")

_HAVE_TOOL = (_ORACLE / "moist_spread.py").exists()
oracle_tool = pytest.mark.skipif(
    not _HAVE_TOOL, reason="the oracle recipe is not present in this tree")


def _load_oracle_aggregator():
    """Import ``moist_spread`` by path.

    By path because ``tools/`` is not importable and must not become so:
    nothing in ``woof`` may depend on the oracle side.  Only this test,
    which exists to compare the two, reaches across.
    """
    spec = importlib.util.spec_from_file_location(
        "_oracle_moist_spread", _ORACLE / "moist_spread.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle_shaped(paths, label):
    """This module's ``aggregate``, driven with the ORACLE's own lists.

    The nested engagement metrics and identity keys become dotted paths,
    which is the same traversal the oracle spells with a second mechanism.
    """
    oracle = _load_oracle_aggregator()
    metrics = tuple(oracle.METRICS) + tuple(
        "%s.%s" % (outer, inner) for outer, inner in oracle.NESTED)
    identity = ("km_opt", "mp_physics", "nx", "ny", "nz", "dx_m",
                "surface_heat_flux", "window_minutes", "t_end_minutes",
                "n_frames_in_window")
    flags = (("saturated_branch_all_engaged_somewhere",
              "saturated_branch.engaged_somewhere", "all"),
             ("saturated_branch_any_engaged_everywhere",
              "saturated_branch.engaged_everywhere", "any"))
    return draw_spread.aggregate_paths(
        paths, label=label, metrics=metrics, identity=identity, flags=flags)


# ------------------------------------------------- equivalence, on real data

@oracle_fixtures
@oracle_tool
@pytest.mark.parametrize("published,sources", _PUBLISHED)
def test_reproduces_the_published_oracle_aggregate(published, sources):
    """Every statistic the oracle published, reproduced to the bit.

    ``==`` on floats, not ``approx``: both routines are plain-Python
    ``sum``/``math.sqrt`` over the same committed numbers in the same
    order, so they agree exactly or one of them has drifted.
    """
    want = json.loads((_FIXTURES / published).read_text())
    got = _oracle_shaped([_FIXTURES / s for s in sources],
                         want["label"])

    assert got["n_draws"] == want["n_draws"]
    assert got["source_receipts"] == want["source_receipts"]
    assert set(got["metrics"]) == set(want["metrics"])
    for name in sorted(want["metrics"]):
        mine, theirs = got["metrics"][name], want["metrics"][name]
        assert set(mine) == set(theirs), name
        for field in ("n", "mean", "sd", "min", "max", "cv"):
            a, b = mine[field], theirs[field]
            if a is None or b is None:
                assert a is None and b is None, (name, field)
                continue
            assert a == b, (name, field, a, b)


@oracle_fixtures
@oracle_tool
@pytest.mark.parametrize("published,sources", _PUBLISHED)
def test_reproduces_the_published_configuration_identity(published, sources):
    """A spread over draws is only a spread if the draws are one
    configuration.  The identity block, and the list of anything that
    differs, must come out the same as the oracle's."""
    want = json.loads((_FIXTURES / published).read_text())
    got = _oracle_shaped([_FIXTURES / s for s in sources], want["label"])
    assert got["configuration"] == want["configuration"]
    assert got["configuration_differs_on"] == want["configuration_differs_on"]
    assert got["configuration_differs_on"] == []


@oracle_fixtures
@oracle_tool
@pytest.mark.parametrize("published,sources", _PUBLISHED)
def test_reproduces_the_published_engagement_flags(published, sources):
    want = json.loads((_FIXTURES / published).read_text())
    got = _oracle_shaped([_FIXTURES / s for s in sources], want["label"])
    for key in ("saturated_branch_all_engaged_somewhere",
                "saturated_branch_any_engaged_everywhere"):
        assert got[key] == want[key], key
    assert got["note"] == want["note"]


@oracle_fixtures
@oracle_tool
def test_stats_is_bit_identical_to_the_oracle_routine():
    """The statistic itself, on the awkward inputs, not only on the
    fixtures: empties, singletons, a NaN, a None, and a sample whose mean
    is exactly zero (where cv must be null rather than a division)."""
    oracle = _load_oracle_aggregator()
    samples = (
        [],
        [1.0],
        [1.0, 2.0],
        [1.0, 2.0, 4.0, 8.0],
        [1.0, None, 3.0],
        [1.0, float("nan"), 3.0],
        [-1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1274.4514270009245, 1274.4660317780388, 1274.4766812268485],
    )
    for vals in samples:
        mine = draw_spread.stats(list(vals))
        theirs = oracle.stats(list(vals))
        assert set(mine) == set(theirs)
        for field in mine:
            a, b = mine[field], theirs[field]
            if isinstance(a, float) and math.isnan(a):
                assert isinstance(b, float) and math.isnan(b), field
                continue
            assert a == b, (vals, field)


def test_single_draw_reports_no_spread_rather_than_zero():
    """One draw has no spread.  Reporting sd = 0.0 for it is the same
    error class as a resolved fraction reading 1.000 because its
    denominator was absent."""
    s = draw_spread.stats([0.5])
    assert s["n"] == 1 and s["sd"] is None and s["cv"] is None
    assert s["mean"] == s["min"] == s["max"] == 0.5


def test_absent_metric_reports_n_zero_and_not_a_zero_value():
    out = draw_spread.aggregate(
        [{"a": 1.0}, {"a": 2.0}], label="x", metrics=("a", "missing"),
        identity=(), flags=())
    assert out["metrics"]["a"]["n"] == 2
    assert out["metrics"]["missing"] == dict(
        n=0, mean=None, sd=None, min=None, max=None, cv=None)


# ------------------------------------------------------------ engine shape

def _engine_receipt(**over):
    """A receipt shaped like ``cloud_topped_boundary_layer``'s."""
    r = {
        "zi_thetav_load_m": 1274.45, "cloud_base_m": 1274.45,
        "cloud_top_m": 1738.7, "cloud_fraction_max": 0.314,
        "sat_fraction_max": 0.0195, "n2_moist_fraction_max": 0.3144,
        "wthv_res_max": 0.1986, "wthv_total_min": -0.0476,
        "wthv_res_max_over_qs": 0.8275, "wqv_res_max": 1.265e-4,
        "wqv_total_max": 1.3e-4, "resolved_fraction_wthv": 0.83,
        "resolved_fraction_wqv": 0.79, "qv_surface": 0.01346,
        "qc_profile_max": 1.1e-4, "qr_profile_max": 1.0e-6,
        "qc_max_pointwise_run": 3.74e-3, "qr_max_pointwise_run": 5.1e-5,
        "first_cloud_seconds": 60.0, "lwp_kg_m2": 0.02527,
        "rwp_kg_m2": 0.0, "rainnc_mm_end": 3.8e-7,
        "lwp_trend_pct_per_h": -42.1,
        "w_max": 4.1, "cfl_max": 0.31, "mass_drift_rel": 9.6e-10,
        "window_minutes": 30.0, "window_frames": 31,
        "n2_engaged_somewhere": True, "n2_engaged_everywhere": False,
        "mutation_control": "none",
        "sounding": {"sha256": "993fedb1"},
        "config": {"km_opt": 3, "mp_physics": 1, "nx": 96, "ny": 96,
                   "nz": 64, "dx": 100.0, "dt": 0.5, "ztop": 2400.0,
                   "run_seconds": 7200.0, "c_s": 0.18, "c_k": 0.1,
                   "tke_heat_flux": 0.24, "isfflx": 0,
                   "bl_pbl_physics": 0},
        # environmental fields, which must NOT be spread
        "wall_seconds": 231.4, "steps_per_second": 62.2,
        "vram_device_gib": 14.4, "vram_pool_gib": 0.79,
    }
    for key, value in over.items():
        r[key] = value
    return r


def test_engine_defaults_aggregate_a_matched_draw_set():
    a = _engine_receipt()
    b = _engine_receipt(lwp_kg_m2=0.02644, cloud_top_m=1778.56)
    out = draw_spread.aggregate([a, b], label="km3_100m",
                                source_names=["r1.json", "r2.json"],
                                registration="deadbeef")
    assert out["n_draws"] == 2 and out["registration"] == "deadbeef"
    assert out["configuration_differs_on"] == []
    assert out["metrics"]["cloud_top_m"]["n"] == 2
    assert out["metrics"]["cloud_top_m"]["sd"] == pytest.approx(
        abs(1778.56 - 1738.7) / math.sqrt(2))
    assert out["saturated_branch_all_engaged_somewhere"] is True
    assert out["saturated_branch_any_engaged_everywhere"] is False
    assert out["note"] == draw_spread.NOTE


def test_a_configuration_difference_is_reported_not_absorbed():
    """Two draws that are not the same case do not have a spread.  The
    aggregator still computes one -- and names the difference at the top of
    its own output, so the number cannot be read without it."""
    out = draw_spread.aggregate(
        [_engine_receipt(),
         _engine_receipt(config=dict(_engine_receipt()["config"], km_opt=2))],
        label="mixed")
    assert "config.km_opt" in out["configuration_differs_on"]
    assert out["configuration"]["config.km_opt"] == {"DIFFERS": [2, 3]}


def test_environmental_fields_are_never_spread():
    """Wall clock and device-wide VRAM measure the MACHINE, not the run.
    Pooling their scatter into a sigma would inflate every band cut from
    that sigma, and the band method divides by exactly that sigma."""
    assert draw_spread.EXCLUDED_FROM_SPREAD
    for name in draw_spread.EXCLUDED_FROM_SPREAD:
        assert name not in draw_spread.MOIST_METRICS, name
    # the per-process pool figure is a property of the run and stays
    # screened -- but it is still not a physical metric, so it is not here
    assert "vram_pool_gib" not in draw_spread.MOIST_METRICS


def test_registration_defaults_to_none_and_is_recorded_not_enforced():
    """This tool publishes spread and gates nothing.  A missing
    registration makes the output say it is a measurement; it does not
    make the tool refuse to do arithmetic, because that would be a gate
    this lane never registered."""
    out = draw_spread.aggregate([_engine_receipt()], label="one")
    assert out["registration"] is None
    assert out["n_draws"] == 1
    assert out["metrics"]["lwp_kg_m2"]["sd"] is None


def test_cli_writes_sorted_json(tmp_path):
    import subprocess
    import sys
    paths = []
    for i, lwp in enumerate((0.0252, 0.0264, 0.0246)):
        p = tmp_path / f"r{i + 1}.json"
        p.write_text(json.dumps(_engine_receipt(lwp_kg_m2=lwp)))
        paths.append(str(p))
    out = tmp_path / "spread.json"
    res = subprocess.run(
        [sys.executable, "-m", "woof.verify.draw_spread",
         "km3_100m", str(out), *paths, "--registration", "abc1234"],
        cwd=_ROOT, capture_output=True, text=True, check=True)
    got = json.loads(out.read_text())
    assert got["n_draws"] == 3
    assert got["registration"] == "abc1234"
    assert got["metrics"]["lwp_kg_m2"]["n"] == 3
    assert got["source_receipts"] == ["r1.json", "r2.json", "r3.json"]
    assert draw_spread.NOTE in res.stdout
    assert "registration: abc1234" in res.stdout
