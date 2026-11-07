"""These tests name ``--engine matplotlib`` wherever they assert the
matplotlib suite's own filename grammar.

`woof enprod --engine` defaults to ``auto``, which takes the Rust
``rw_ensbatch`` on a checkout that has built it -- so a test asserting
``refl-ens-mean_d02-3km_<stamp>.png`` is asserting a property of one
engine and has to say which.  ``tests/test_enprod_rust_engine.py`` covers
the resolver and the rust route.

``woof enprod`` ensemble product tests -- CPU-only.

The mathematics tests are deliberately hand-computable: a probability
over five members is a fifth, and a test that only checks "roughly 0.6"
would pass for an implementation that divided by the wrong count.  Every
probability assertion here is an exact equality against a number worked
out on paper in the test body's comment.

The rendering tests use the module's own synthetic-ensemble generator
(:func:`woof.da.enprod.write_synthetic_ensemble`), so the files under
test are written by the project's ``WrfoutWriter`` and read by the
mandated ``wrf`` package exactly as a production ensemble would be.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import woof.cli as cli
from woof import render_layout
from woof.da import enprod
from woof.render import plot_context

_THRESHOLD = 40.0


# ---------------------------------------------------------------------------
# Exceedance probability -- exact, hand-computable
# ---------------------------------------------------------------------------


def test_point_probability_is_the_exact_member_fraction():
    """Five members, five columns, one exact fifth each.

    Column 0: no member above 40      -> 0/5
    Column 1: one member above 40     -> 1/5
    Column 2: three members above 40  -> 3/5
    Column 3: every member above 40   -> 5/5
    Column 4: every member EXACTLY 40 -> 0/5 (the test is strictly >)
    """

    stack = np.array([
        [[0.0, 50.0, 50.0, 50.0, 40.0]],
        [[0.0, 0.0, 50.0, 50.0, 40.0]],
        [[0.0, 0.0, 50.0, 50.0, 40.0]],
        [[0.0, 0.0, 0.0, 50.0, 40.0]],
        [[0.0, 0.0, 0.0, 50.0, 40.0]],
    ])
    probability = enprod.exceedance_probability(stack, _THRESHOLD)
    assert_array_equal(probability, np.array([[0.0, 0.2, 0.6, 1.0, 0.0]]))


def test_a_threshold_equal_value_is_not_an_exceedance():
    """``>`` and not ``>=``: the 40 dBZ product means above 40 dBZ."""

    stack = np.full((5, 1, 1), _THRESHOLD)
    assert_array_equal(enprod.exceedance_probability(stack, _THRESHOLD),
                       np.zeros((1, 1)))
    stack[0, 0, 0] = np.nextafter(_THRESHOLD, np.inf)
    assert_array_equal(enprod.exceedance_probability(stack, _THRESHOLD),
                       np.full((1, 1), 0.2))


def test_neighborhood_probability_spreads_one_member_hit_by_the_disc():
    """One member's single hot cell, radius 1 cell -> a plus of 1/5.

    Radius 1.0 admits the four edge neighbours (dy^2+dx^2 = 1) and not
    the diagonals (= 2), so exactly five cells reach the hit and each
    reports one member in five.
    """

    stack = np.zeros((5, 3, 3))
    stack[0, 1, 1] = 100.0
    probability = enprod.exceedance_probability(stack, 50.0, radius_cells=1.0)
    expected = np.array([[0.0, 0.2, 0.0],
                         [0.2, 0.2, 0.2],
                         [0.0, 0.2, 0.0]])
    assert_array_equal(probability, expected)


def test_a_radius_reaching_the_diagonals_fills_the_block():
    """Radius 1.5 admits dy^2+dx^2 = 2, so all nine cells see the hit."""

    stack = np.zeros((5, 3, 3))
    stack[0, 1, 1] = 100.0
    probability = enprod.exceedance_probability(stack, 50.0, radius_cells=1.5)
    assert_array_equal(probability, np.full((3, 3), 0.2))


def test_the_neighborhood_does_not_invent_exceedances_past_the_edge():
    """A corner hit reaches three cells, not five.

    This pins the VALUES at the edge, and that is all it can pin: for a
    MAXIMUM operator, replicating an edge value outward can never change
    an in-domain result, because the replicated cell only duplicates a
    value already inside that point's clipped footprint.  The corner
    "plus" below therefore passes under replicate padding too, so the
    claim this test used to make -- that it rules padding out -- was
    false.  What actually distinguishes the two is the FOOTPRINT, and
    ``test_the_neighborhood_footprint_is_clipped_not_padded`` below
    measures it.
    """

    stack = np.zeros((5, 3, 3))
    stack[0, 0, 0] = 100.0
    probability = enprod.exceedance_probability(stack, 50.0, radius_cells=1.0)
    expected = np.array([[0.2, 0.2, 0.0],
                         [0.2, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])
    assert_array_equal(probability, expected)


def test_the_neighborhood_footprint_is_clipped_not_padded():
    """The discriminator a maximum of values cannot supply.

    A replicate-padded implementation gives every point the FULL disc:
    five cells at radius 1 everywhere, corners included.  A clipped one
    gives the corner three.  Counting is the only way to tell them apart,
    so the count is a published function and this is the test of it.
    """

    footprint = enprod.neighborhood_footprint((3, 3), 1.0)
    full_disc = len(enprod.disc_offsets(1.0))
    assert full_disc == 5
    assert footprint[1, 1] == 5, "an interior point sees the whole disc"
    assert footprint[0, 0] == 3, "a corner sees three; padding would say 5"
    assert footprint[0, 1] == 4
    assert not np.all(footprint == full_disc)

    # And the maximum is taken over exactly that footprint: every point's
    # value equals the brute-force maximum over its clipped disc.
    rng = np.random.default_rng(11)
    field = rng.normal(size=(4, 6))
    got = enprod.neighborhood_max(field, 2.0)
    ny, nx = field.shape
    for j in range(ny):
        for i in range(nx):
            cells = [field[j + dy, i + dx]
                     for dy, dx in enprod.disc_offsets(2.0)
                     if 0 <= j + dy < ny and 0 <= i + dx < nx]
            assert got[j, i] == max(cells)
            assert enprod.neighborhood_footprint(field.shape, 2.0)[j, i] \
                == len(cells)


def test_probability_never_decreases_as_the_radius_grows():
    """Monotonicity, everywhere, for every pair of increasing radii."""

    rng = np.random.default_rng(20260730)
    stack = rng.normal(30.0, 15.0, (5, 14, 18))
    previous = enprod.exceedance_probability(stack, _THRESHOLD,
                                             radius_cells=0.0)
    for radius in (0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0):
        current = enprod.exceedance_probability(stack, _THRESHOLD,
                                                radius_cells=radius)
        assert np.all(current >= previous), radius
        previous = current
    # A radius spanning the domain saturates at "some member, somewhere".
    assert np.all(previous > 0.0)


def test_disc_footprints_are_nested_which_is_why_probability_is_monotone():
    previous = set(enprod.disc_offsets(0.0))
    assert previous == {(0, 0)}
    for radius in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
        current = set(enprod.disc_offsets(radius))
        assert previous <= current, radius
        previous = current
    # And the disc is a disc: (3, 3) is 4.24 cells out, outside radius 3.
    assert (3, 3) not in set(enprod.disc_offsets(3.0))
    assert (3, 0) in set(enprod.disc_offsets(3.0))


def test_neighborhood_max_of_a_zero_radius_is_the_field_itself():
    rng = np.random.default_rng(7)
    field = rng.normal(size=(5, 6))
    assert_array_equal(enprod.neighborhood_max(field, 0.0), field)


# ---------------------------------------------------------------------------
# Mean and spread
# ---------------------------------------------------------------------------


def test_mean_and_spread_are_the_sample_statistics():
    stack = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]])
    assert enprod.ensemble_mean(stack)[0, 0] == 3.0
    # Sample standard deviation (ddof=1) of 1..5 is sqrt(10/4) = 1.5811...
    assert enprod.ensemble_spread(stack)[0, 0] == pytest.approx(
        np.sqrt(2.5), rel=0, abs=1e-12)
    # The population form differs, and this test exists so a change from
    # one to the other cannot happen quietly.
    assert enprod.ensemble_spread(stack, ddof=0)[0, 0] == pytest.approx(
        np.sqrt(2.0), rel=0, abs=1e-12)


def test_a_single_member_has_no_spread_and_says_so():
    with pytest.raises(enprod.EnsembleRefusal, match="more than 1 member"):
        enprod.ensemble_spread(np.zeros((1, 2, 2)))


def test_the_mean_does_not_silently_drop_a_nan_member():
    """The objection was never masking; it was masking without saying so.

    A bare ``np.nanmean`` averages a different member count at every grid
    point and never reports which -- that is what made propagating the NaN
    the better of the two implicit choices.  The policy is neither: the
    mean is over the finite members AND the count it used is published,
    which is what ``missingness_report`` and the panel caption are for.
    ``--nan-policy refuse`` is the fail-closed alternative.
    """

    stack = np.ones((5, 1, 2))
    stack[2, 0, 1] = np.nan
    mean = enprod.ensemble_mean(stack)
    assert mean[0, 0] == 1.0
    assert mean[0, 1] == 1.0, "four finite members still average to 1.0"

    report = enprod.missingness_report(stack, member_numbers=[1, 2, 3, 4, 5])
    assert report["nonfinite_values"] == 1
    assert report["members_affected"] == [3]
    assert report["min_finite_members"] == 4
    assert report["fully_missing_points"] == 0
    caption = enprod.coverage_caption(report)
    assert "member(s) 3" in caption and "min 4 of 5" in caption

    with pytest.raises(enprod.EnsembleRefusal, match="non-finite"):
        enprod.ensemble_mean(stack, nan_policy="refuse")


def test_a_point_with_no_finite_member_stays_blank():
    stack = np.full((3, 1, 2), np.nan)
    stack[:, 0, 0] = 5.0
    mean = enprod.ensemble_mean(stack)
    assert mean[0, 0] == 5.0
    assert np.isnan(mean[0, 1]), "nothing to say is the answer, not zero"


# ---------------------------------------------------------------------------
# Probability-matched mean
# ---------------------------------------------------------------------------


def _pooled_reference(stack):
    """Ebert's every-M-th value of the pooled descending sort."""

    n_members = stack.shape[0]
    pooled = np.sort(np.asarray(stack).reshape(-1))[::-1]
    return pooled[::n_members]


def test_pmm_values_are_exactly_the_pooled_member_distribution():
    rng = np.random.default_rng(11)
    stack = rng.gamma(2.0, 12.0, (5, 9, 11))
    pmm = enprod.probability_matched_mean(stack)
    assert_array_equal(np.sort(pmm.reshape(-1))[::-1],
                       _pooled_reference(stack))


def test_pmm_carries_the_mean_pattern_rank_for_rank():
    rng = np.random.default_rng(12)
    stack = rng.gamma(2.0, 12.0, (5, 9, 11))
    mean = stack.mean(axis=0)
    pmm = enprod.probability_matched_mean(stack)
    order = np.argsort(-mean.reshape(-1), kind="stable")
    # Walking the grid in the MEAN's descending order must walk the PMM
    # in descending order too: same pattern, remapped amplitudes.
    assert np.all(np.diff(pmm.reshape(-1)[order]) <= 0.0)
    assert np.all(np.diff(mean.reshape(-1)[order]) <= 0.0)


def test_pmm_restores_the_peak_a_plain_mean_smears_away():
    """The reason PMM exists, as a test.

    Two members with the same storm in different places: the mean halves
    the peak, and the probability-matched mean puts a member-strength
    peak back at the mean's maximum.
    """

    grid = np.zeros((2, 1, 9))
    grid[0, 0, 3] = 60.0
    grid[1, 0, 5] = 60.0
    mean = enprod.ensemble_mean(grid)
    pmm = enprod.probability_matched_mean(grid)
    assert mean.max() == 30.0
    assert pmm.max() == 60.0
    assert pmm.sum() == pytest.approx(60.0)


def test_pmm_is_deterministic_when_the_mean_has_ties():
    stack = np.zeros((5, 2, 3))
    stack[0] = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    first = enprod.probability_matched_mean(stack)
    second = enprod.probability_matched_mean(stack.copy())
    assert_array_equal(first, second)
    assert_array_equal(np.sort(first.reshape(-1))[::-1],
                       _pooled_reference(stack))


# ---------------------------------------------------------------------------
# Paintball colours
# ---------------------------------------------------------------------------


def test_member_colour_is_keyed_to_the_member_not_to_its_position():
    """Filtering the roster must not repaint the members that remain.

    Colour is the only label a paintball plot carries; if member 7
    changed colour when members 4 and 5 dropped out, two paintball plots
    of one ensemble could not be compared.
    """

    full = {number: enprod.member_color(number) for number in range(1, 31)}
    subset = [3, 7, 11, 29]
    assert [enprod.member_color(n) for n in subset] == [full[n] for n in subset]


# ---------------------------------------------------------------------------
# Filename tokens and the subtitle
# ---------------------------------------------------------------------------


def test_the_ensemble_max_is_a_named_product_end_to_end():
    """The live DA ensemble menu needs an "Ensemble max" map (outsider check
    2026-11-03 finding 3): enprod must accept it, expand it once per
    field, name its file, and reduce it with a NaN member masked."""

    assert "max" in enprod.PRODUCTS
    products, _ = enprod.parse_products("all")
    assert "max" in products
    assert enprod.parse_products("max") == (("max",), False)
    assert enprod.ensemble_token("max") == "ens-max"
    requests = enprod.expand_requests(("refl", "uh"), ("max",),
                                      (30.0, 40.0), (0.0, 5.0))
    assert [(r.field, r.threshold, r.radius_km) for r in requests] == [
        ("refl", None, None), ("uh", None, None)]

    stack = np.array([
        [[10.0, -5.0, np.nan]],
        [[np.nan, -2.0, np.nan]],
        [[np.inf, -7.0, np.nan]],
    ])
    out = enprod.ensemble_max(stack)
    assert out[0, 0] == 10.0
    assert out[0, 1] == -2.0
    assert np.isnan(out[0, 2])
    with pytest.raises(enprod.EnsembleRefusal):
        enprod.ensemble_max(stack, nan_policy="refuse")


def test_a_threshold_product_without_a_threshold_is_an_error():
    with pytest.raises(ValueError, match="threshold-based"):
        enprod.ensemble_token("prob")


def test_the_subtitle_extends_v1_0_1_with_the_member_count_and_the_stamp():
    base = plot_context("d02", 3000.0, "1974-04-03_18:00:00", "WOOF")
    context = enprod.ensemble_plot_context("d02", 3000.0,
                                           "1974-04-03_18:00:00", "WOOF", 30)
    assert context.startswith(base)
    assert "30 members" in context
    assert enprod.EXPERIMENTAL_STAMP in context


def test_the_stamp_carries_the_running_engine_version_not_a_frozen_one():
    """A version on a product is a provenance claim, so it must be true.

    ``EXPERIMENTAL_STAMP`` was the literal ``"EXPERIMENTAL (v1.2
    ensemble, uncalibrated)"``, frozen at the release the suite was
    written for and stamped onto every ensemble panel every release
    since -- so a 1.8.7 plot claimed to have come from a 1.2 ensemble.
    The uncalibrated warning is still true and stays; the version now
    comes from the engine that produced the plot.
    """
    import woof

    stamp = enprod.experimental_stamp()
    assert f"v{woof.__version__} ensemble" in stamp
    assert "EXPERIMENTAL" in stamp and "uncalibrated" in stamp
    # The frozen literal, gone.  Asserted as an absence because the
    # failure mode was a string that stayed correct-looking for six
    # minor releases.
    assert "v1.2 ensemble" not in stamp or woof.__version__ == "1.2"


def test_the_stamp_is_read_at_call_time_not_import_time(monkeypatch):
    """Detector both directions: move the version, the stamp moves."""
    import woof

    monkeypatch.setattr(woof, "__version__", "9.9.9-probe")
    assert "v9.9.9-probe ensemble" in enprod.experimental_stamp()
    assert "v9.9.9-probe" in enprod.ensemble_plot_context(
        "d02", 3000.0, "1974-04-03_18:00:00", "WOOF", 30)
    # The compatibility ATTRIBUTE moves too: it is PEP 562 module
    # __getattr__ over experimental_stamp(), not a constant evaluated at
    # import.  A restored `EXPERIMENTAL_STAMP = experimental_stamp()`
    # module constant -- the exact contradiction the function's docstring
    # disclaims -- fails here, because enprod was imported long before
    # this monkeypatch.
    assert "v9.9.9-probe ensemble" in enprod.EXPERIMENTAL_STAMP


def test_a_neighborhood_radius_needs_a_declared_grid_spacing():
    assert enprod.radius_in_cells(5.0, 1000.0) == 5.0
    assert enprod.radius_in_cells(5.0, 250.0) == 20.0
    assert enprod.radius_in_cells(0.0, None) == 0.0
    with pytest.raises(enprod.EnsembleRefusal, match="no usable DX"):
        enprod.radius_in_cells(5.0, None)


# ---------------------------------------------------------------------------
# Request expansion
# ---------------------------------------------------------------------------


def test_all_products_drops_pmm_from_the_fields_that_disclaim_it():
    products, explicit = enprod.parse_products("all")
    assert explicit is False
    requests = enprod.expand_requests(("refl", "uh"), products, (), (0.0,),
                                      pmm_explicit=False)
    pmm_fields = {r.field for r in requests if r.product == "pmm"}
    assert pmm_fields == {"refl"}


def test_a_field_with_no_threshold_given_uses_its_own_default():
    requests = enprod.expand_requests(("refl", "uh"), ("prob",), (), (0.0,))
    defaults = {r.field: r.threshold for r in requests}
    assert defaults == {"refl": 40.0, "uh": 75.0}


# ---------------------------------------------------------------------------
# The manifest: fail closed, and name what failed
# ---------------------------------------------------------------------------


def _write_manifest(root, document):
    root.mkdir(parents=True, exist_ok=True)
    (root / enprod.MANIFEST_FILENAME).write_text(
        json.dumps(document), encoding="utf-8")
    return root


def _member_records(root, count, *, status="complete"):
    records = []
    for number in range(1, count + 1):
        (root / f"member_{number:03d}").mkdir(parents=True, exist_ok=True)
        records.append({"member": number, "dir": f"member_{number:03d}",
                        "seed": 1000 + number, "status": status,
                        "provenance": {}})
    return records


def test_a_missing_manifest_is_a_refusal_that_says_how_to_get_one(tmp_path):
    with pytest.raises(enprod.EnsembleRefusal) as excinfo:
        enprod.load_manifest(tmp_path)
    message = str(excinfo.value)
    assert enprod.MANIFEST_FILENAME in message
    assert "--make-fixture" in message


def test_a_foreign_schema_warns_and_the_fields_still_decide(
        tmp_path, capsys):
    """Warn-not-block: the schema string is metadata; every field this
    loader uses is validated on use.  An empty roster is still refused
    (there is nothing to ensemble)."""
    root = _write_manifest(tmp_path / "ens", {
        "schema": "some-other-manifest.v9", "n_members": 1, "members": []})
    with pytest.raises(enprod.EnsembleRefusal, match="no usable member"):
        enprod.load_manifest(root)
    err = capsys.readouterr().err
    assert "warning:" in err and "some-other-manifest.v9" in err


def test_the_schema_string_is_found_under_any_of_its_plausible_keys(tmp_path):
    """The sibling lane owns the writer; the VALUE is the contract."""

    root = tmp_path / "ens"
    root.mkdir()
    records = _member_records(root, 2)
    for key in ("schema", "schema_version", "format", "kind"):
        _write_manifest(root, {key: enprod.MANIFEST_SCHEMA,
                               "n_members": 2, "members": records})
        assert enprod.load_manifest(root).n_members == 2


def test_a_manifest_with_no_schema_warns_and_is_read(tmp_path, capsys):
    root = tmp_path / "ens"
    root.mkdir()
    records = _member_records(root, 2)
    _write_manifest(root, {"n_members": 2, "members": records})
    manifest = enprod.load_manifest(root)
    err = capsys.readouterr().err
    assert "warning:" in err and "no schema string" in err
    assert manifest.n_members == 2


def test_a_member_count_that_disagrees_with_the_roster_warns(
        tmp_path, capsys):
    """Warn-not-block: the roster is authoritative; the redundant
    scalar drift is named in one line and the true count is stamped."""
    root = tmp_path / "ens"
    root.mkdir()
    _write_manifest(root, {"schema": enprod.MANIFEST_SCHEMA,
                           "n_members": 30,
                           "members": _member_records(root, 5)})
    manifest = enprod.load_manifest(root)
    err = capsys.readouterr().err
    assert "warning:" in err and "n_members=30" in err
    assert manifest.n_members == 5
    assert len(manifest.members) == 5


def test_missing_member_directories_are_named_all_of_them(tmp_path):
    """A 5-member ensemble missing two members is not a 3-member
    ensemble; the refusal names both, in one run."""

    root = tmp_path / "ens"
    root.mkdir()
    records = _member_records(root, 5)
    for number in (2, 4):
        (root / f"member_{number:03d}").rmdir()
    _write_manifest(root, {"schema": enprod.MANIFEST_SCHEMA,
                           "n_members": 5, "members": records})
    with pytest.raises(enprod.EnsembleRefusal) as excinfo:
        enprod.load_manifest(root)
    message = str(excinfo.value)
    assert "member 2:" in message and "member 4:" in message
    assert "member 3:" not in message
    assert "2 of 5" in message


def test_an_unaccepted_status_is_skipped_with_a_warning(tmp_path, capsys):
    """Warn-not-block: the unaccepted member is dropped in one line
    that still teaches the flag; the panels stamp the count actually
    averaged."""
    root = tmp_path / "ens"
    root.mkdir()
    records = _member_records(root, 3)
    records[1]["status"] = "crashed"
    _write_manifest(root, {"schema": enprod.MANIFEST_SCHEMA,
                           "n_members": 3, "members": records})
    manifest = enprod.load_manifest(root)
    err = capsys.readouterr().err
    assert "warning:" in err and "'crashed'" in err
    assert "--accept-status" in err
    assert manifest.n_members == 2
    assert [m.number for m in manifest.members] == [1, 3]
    # And the operator who means it can accept the member explicitly.
    manifest = enprod.load_manifest(root,
                                    accept_status=("complete", "crashed"))
    assert manifest.n_members == 3
    assert [m.number for m in manifest.members] == [1, 2, 3]


def test_a_member_with_no_status_is_not_a_member_this_suite_averages(tmp_path):
    root = tmp_path / "ens"
    root.mkdir()
    records = _member_records(root, 2)
    del records[0]["status"]
    _write_manifest(root, {"schema": enprod.MANIFEST_SCHEMA,
                           "n_members": 2, "members": records})
    with pytest.raises(enprod.EnsembleRefusal, match="member 1: no 'status'"):
        enprod.load_manifest(root)


def test_a_duplicated_member_number_is_refused(tmp_path):
    root = tmp_path / "ens"
    root.mkdir()
    records = _member_records(root, 2)
    records[1]["member"] = 1
    _write_manifest(root, {"schema": enprod.MANIFEST_SCHEMA,
                           "n_members": 2, "members": records})
    with pytest.raises(enprod.EnsembleRefusal, match="already declared"):
        enprod.load_manifest(root)


def test_the_seed_and_provenance_survive_the_read(tmp_path):
    root = tmp_path / "ens"
    root.mkdir()
    records = _member_records(root, 2)
    records[0]["provenance"] = {"perturbation": "smooth-random", "cycle": 3}
    _write_manifest(root, {"schema": enprod.MANIFEST_SCHEMA,
                           "n_members": 2, "members": records})
    manifest = enprod.load_manifest(root)
    assert manifest.members[0].seed == 1001
    assert manifest.members[0].provenance["perturbation"] == "smooth-random"


# ---------------------------------------------------------------------------
# End to end, against wrfouts the project's own writer produced
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ensemble_root(tmp_path_factory):
    pytest.importorskip(
        "wrf", reason="woof enprod requires the wrf package (wrf-rust)")
    root = tmp_path_factory.mktemp("enprod") / "ens"
    enprod.write_synthetic_ensemble(root, n_members=5, nx=16, ny=12,
                                    dx=3000.0, domain_id=2, seed=3)
    return root


_STAMPS = ("1970-01-01_18-00-00", "1970-01-01_19-00-00")


def test_the_suite_writes_exactly_the_expected_filenames(ensemble_root,
                                                         tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["enprod", str(ensemble_root), "--field", "refl",
                   "--products", "all", "--neighborhood-km", "0,5",
                   "--timeidx", "0", "--out", str(out), "--dpi", "72",
                   "--engine", "matplotlib"])
    assert rc == 0
    produced = sorted(p.name for p in render_layout.iter_rendered(out))
    assert produced == sorted(
        f"refl-{token}_d02-3km_{_STAMPS[0]}.png"
        for token in ("ens-mean", "ens-spread", "ens-p40dbz",
                      "ens-p40dbz-r5km", "ens-paintball40dbz", "ens-pmm"))
    for png in render_layout.iter_rendered(out):
        assert png.stat().st_size > 5_000, png.name


def test_the_default_field_pair_carries_its_own_thresholds(ensemble_root,
                                                           tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["enprod", str(ensemble_root), "--field", "refl,uh",
                   "--products", "prob", "--timeidx", "1",
                   "--out", str(out), "--dpi", "72",
                   "--engine", "matplotlib"])
    assert rc == 0
    produced = sorted(p.name for p in render_layout.iter_rendered(out))
    assert produced == [
        f"refl-ens-p40dbz_d02-3km_{_STAMPS[1]}.png",
        f"uh-ens-p75m2s2_d02-3km_{_STAMPS[1]}.png",
    ]


def test_every_shared_valid_time_is_rendered(ensemble_root, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["enprod", str(ensemble_root), "--field", "refl",
                   "--products", "mean", "--out", str(out), "--dpi", "72",
                   "--engine", "matplotlib"])
    assert rc == 0
    assert sorted(p.name for p in render_layout.iter_rendered(out)) == [
        f"refl-ens-mean_d02-3km_{stamp}.png" for stamp in _STAMPS]


def test_two_requests_that_collide_on_one_filename_are_refused(ensemble_root,
                                                               tmp_path):
    """v1.0.1's claims map.  40 and 40.0000001 dBZ are two forecasts and
    one filename token; the second write would report success."""

    out = tmp_path / "png"
    rc = cli.main(["enprod", str(ensemble_root), "--field", "refl",
                   "--products", "prob", "--threshold", "40,40.0000001",
                   "--timeidx", "0", "--out", str(out), "--dpi", "72",
                   "--engine", "matplotlib"])
    assert rc == 1
    assert sorted(p.name for p in render_layout.iter_rendered(out)) == [
        f"refl-ens-p40dbz_d02-3km_{_STAMPS[0]}.png"]


def test_a_member_missing_from_disk_refuses_the_whole_suite(ensemble_root,
                                                            tmp_path):
    """No products at all: a 4-member mean labelled "5 members" is worse
    than no plot."""

    import shutil

    root = tmp_path / "ens"
    shutil.copytree(ensemble_root, root)
    shutil.rmtree(root / "member_003")
    out = tmp_path / "png"
    rc = cli.main(["enprod", str(root), "--field", "refl", "--products",
                   "mean", "--out", str(out), "--dpi", "72",
                   "--engine", "matplotlib"])
    assert rc == 2
    assert not out.exists() or not render_layout.iter_rendered(out)


def test_members_with_surplus_frames_render_the_intersection(
        ensemble_root, tmp_path, capsys):
    """Warn-not-block: a surplus frame in one member cannot corrupt a
    reduction computed on the shared times; it is named and ignored."""
    import shutil

    root = tmp_path / "ens"
    shutil.copytree(ensemble_root, root)
    # Give member 4 an extra frame nobody else has.
    enprod.write_synthetic_ensemble(
        tmp_path / "extra", n_members=1, nx=16, ny=12, dx=3000.0,
        domain_id=2, stamps=("1970-01-01_20:00:00",), seed=99)
    shutil.copy(
        next((tmp_path / "extra" / "member_000").glob("wrfout_d02_*.nc")),
        root / "member_004" / "wrfout_d02_1970-01-01_20-00-00.nc")
    out = tmp_path / "png"
    rc = cli.main(["enprod", str(root), "--field", "refl", "--products",
                   "mean", "--out", str(out), "--dpi", "72",
                   "--engine", "matplotlib"])
    captured = capsys.readouterr()
    assert rc == 0
    assert render_layout.iter_rendered(out)
    assert "warning:" in captured.err
    assert "surplus frames" in captured.err
    # The surplus valid time itself must not have been rendered.
    assert not list(out.glob("*20-00-00*"))


def test_the_fixture_generator_writes_a_manifest_this_module_reads(tmp_path):
    pytest.importorskip("wrf")
    root = tmp_path / "ens"
    manifest_path = enprod.write_synthetic_ensemble(
        root, n_members=3, nx=8, ny=8, dx=1000.0,
        stamps=("1970-01-01_18:00:00",))
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["schema"] == enprod.MANIFEST_SCHEMA
    assert document["n_members"] == 3
    # The fixture writes the writer of record's spelling: zero-based
    # ``index``, ``member_dir``, and the ``DONE`` terminal status of
    # woof.ensemble.manifest.  A fixture in any other spelling would
    # test a contract nothing produces.
    assert document["members"][0]["index"] == 0
    assert document["members"][0]["member_dir"] == "member_000"
    assert {record["status"] for record in document["members"]} == {"DONE"}
    manifest = enprod.load_manifest(root)
    assert [m.number for m in manifest.members] == [0, 1, 2]
    # Every member carries the seed that produced it -- the reproduction
    # handle the manifest exists to preserve.
    assert len({m.seed for m in manifest.members}) == 3


def test_the_cli_warns_on_a_one_member_fixture_and_writes_it(
        tmp_path, capsys):
    """Warn-not-block: a one-member synthetic fixture writes; spread
    and probability panels refuse themselves later, per panel."""
    rc = cli.main(["enprod", "--make-fixture", str(tmp_path / "ens"),
                   "--members", "1"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "warning:" in captured.err
    assert (tmp_path / "ens" / enprod.MANIFEST_FILENAME).is_file()


def test_the_experimental_stamp_reaches_stdout(ensemble_root, tmp_path,
                                               capsys):
    cli.main(["enprod", str(ensemble_root), "--field", "refl", "--products",
              "mean", "--timeidx", "0", "--out", str(tmp_path / "png"),
              "--dpi", "72", "--engine", "matplotlib"])
    assert enprod.EXPERIMENTAL_STAMP in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The rust route's own door: one --dpi meaning, one nest choice, one delivery
# ---------------------------------------------------------------------------

_PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")


def _staged_engine(monkeypatch, seen, *, products=("mean",)):
    """A fake rw_ensbatch that records its request and writes one panel."""
    from woof import rustwx_lanes

    def fake_ensemble(engine, manifest, *, store_root, out_dir, **kwargs):
        seen.update(kwargs)
        seen["out_dir"] = Path(out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        written, rendered = [], []
        for product in products:
            panel = Path(out_dir) / f"ens{len(written):02d}" / \
                f"ens_{product}_{kwargs['field']}" / "ensemble" / \
                f"arwen_ens_{product}.png"
            panel.parent.mkdir(parents=True, exist_ok=True)
            panel.write_bytes(_PNG_MAGIC + b"fake ensemble panel")
            written.append(panel)
            rendered.append((product, panel))
        return written, [], [], {"rendered": rendered}

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin",
                        lambda: Path("rw_ensbatch"))
    monkeypatch.setattr(rustwx_lanes, "probe_ensemble_bin",
                        lambda path: (True, "--abi matches the contract"))
    monkeypatch.setattr(rustwx_lanes, "run_ensemble_renderer", fake_ensemble)


def _bare_ensemble(tmp_path):
    root = tmp_path / "ens"
    root.mkdir(exist_ok=True)
    (root / enprod.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    return root


def test_dpi_has_one_meaning_on_both_engines(tmp_path, monkeypatch, capsys):
    """``--dpi`` was accepted and dropped with a warn on the rust route."""
    seen: dict = {}
    _staged_engine(monkeypatch, seen)
    out = tmp_path / "png"
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
                   "--products", "mean", "--out", str(out),
                   "--engine", "rust", "--dpi", "300"])
    assert rc == 0
    assert (seen["width"], seen["height"]) == (2400, 1800)
    err = capsys.readouterr().err
    assert "has no effect" not in err


def test_the_default_dpi_is_the_engines_own_size(tmp_path, monkeypatch):
    seen: dict = {}
    _staged_engine(monkeypatch, seen)
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
                   "--products", "mean", "--out", str(tmp_path / "png"),
                   "--engine", "rust"])
    assert rc == 0
    assert (seen["width"], seen["height"]) == (1200, 900)


def test_panel_pixels_names_its_floor_instead_of_clamping_quietly(capsys):
    assert enprod.panel_pixels(150) == (1200, 900)
    assert enprod.panel_pixels(300) == (2400, 1800)
    assert enprod.panel_pixels(1) == (320, 240)
    assert "320x240" in capsys.readouterr().err


def test_the_nest_choice_reaches_the_rust_engine(tmp_path, monkeypatch):
    """``--domain`` was refused as unimplemented; it is the disambiguation."""
    seen: dict = {}
    _staged_engine(monkeypatch, seen)
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
                   "--products", "mean", "--out", str(tmp_path / "png"),
                   "--engine", "rust", "--domain", "d02"])
    assert rc == 0
    assert seen["domain"] == "d02"


def _mixed_nest_roster(tmp_path):
    from types import SimpleNamespace

    members = []
    for number in (1, 2):
        directory = tmp_path / f"member_{number:03d}"
        for token in ("d01", "d02"):
            path = directory / f"wrfout_{token}_1974-04-03_18_00_00"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        members.append(SimpleNamespace(number=number, directory=directory))
    return SimpleNamespace(members=tuple(members))


def test_a_mixed_nest_ensemble_is_refused_identically_at_both_doors(
        tmp_path, monkeypatch, capsys):
    """One request fact, one sentence, whichever engine was asked for."""
    roster = _mixed_nest_roster(tmp_path)
    monkeypatch.setattr(enprod, "load_manifest", lambda *a, **k: roster)
    seen: dict = {}
    _staged_engine(monkeypatch, seen)
    messages = []
    for engine in ("rust", "matplotlib"):
        rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)),
                       "--field", "refl", "--products", "mean",
                       "--out", str(tmp_path / f"png-{engine}"),
                       "--engine", engine])
        assert rc == 2, engine
        messages.append([line for line in capsys.readouterr().err.splitlines()
                         if "more than one domain" in line])
    assert messages[0] == messages[1] != []
    assert "--domain dNN" in messages[0][0]
    assert "d01, d02" in messages[0][0]


def test_naming_the_nest_admits_the_same_ensemble(tmp_path, monkeypatch):
    roster = _mixed_nest_roster(tmp_path)
    assert enprod.domain_choice_problem(roster, "d02") is None
    assert enprod.domain_choice_problem(roster, None) is not None


def test_the_rust_route_delivers_into_the_layout_and_leaves_nothing_behind(
        tmp_path, monkeypatch):
    """One --out, two nests, and the engine's own token carried neither."""
    seen: dict = {}
    _staged_engine(monkeypatch, seen, products=("mean", "spread"))
    out = tmp_path / "png"
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
                   "--products", "mean,spread", "--out", str(out),
                   "--engine", "rust"])
    assert rc == 0
    delivered = render_layout.iter_rendered(out)
    assert len(delivered) == 2
    for path in delivered:
        parts = path.relative_to(out).parts
        assert len(parts) == 4, parts
        assert parts[1] in {"refl-ens-mean", "refl-ens-spread"}
    # The engine drew into scratch, and scratch is gone.
    assert not seen["out_dir"].exists() or not list(seen["out_dir"].rglob("*.png"))
    assert "ensemble" not in {part for path in delivered
                              for part in path.relative_to(out).parts}


def test_the_rust_route_publishes_a_render_receipt(tmp_path, monkeypatch):
    from woof import render_receipts

    seen: dict = {}
    _staged_engine(monkeypatch, seen)
    out = tmp_path / "png"
    assert cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
                     "--products", "mean", "--out", str(out),
                     "--engine", "rust"]) == 0
    summary = render_receipts.read_summary(out)
    assert summary is not None
    assert summary["schema"] == "gpuwm.render-summary.v1"
    assert summary["rendered_png_count"] == 1
    assert [row["name"] for row in summary["rendered_families"]] == ["refl-ens-mean"]
    assert len(list((out / ".render-receipts").glob("*.json"))) == 1


def test_the_matplotlib_route_publishes_a_render_receipt(tmp_path, monkeypatch):
    from woof import render_receipts

    out = tmp_path / "png"
    panel = enprod.delivered_path(out, "refl", "ens-mean", "d02-3km",
                                 "1974-04-03_18:00:00")
    panel.parent.mkdir(parents=True, exist_ok=True)
    panel.write_bytes(_PNG_MAGIC + b"fake matplotlib panel")
    monkeypatch.setattr(enprod, "run_suite", lambda *a, **k: ([panel], []))
    assert cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
                     "--products", "mean", "--out", str(out),
                     "--engine", "matplotlib"]) == 0
    summary = render_receipts.read_summary(out)
    assert summary is not None
    assert summary["rendered_png_count"] == 1
    assert [row["name"] for row in summary["rendered_families"]] == ["refl-ens-mean"]


def test_the_matplotlib_route_reports_itself_as_a_workaround(tmp_path, monkeypatch, capsys):
    out = tmp_path / "png"
    monkeypatch.setattr(enprod, "run_suite", lambda *a, **k: ([], []))
    cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
              "--products", "mean", "--out", str(out),
              "--engine", "matplotlib"])
    err = capsys.readouterr().err
    assert "WORKAROUND:" in err
    assert "rw_ensbatch" in err


def test_auto_with_no_engine_refuses_before_a_member_is_read(tmp_path, monkeypatch, capsys):
    from woof import rustwx_lanes

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin", lambda: None)
    monkeypatch.setattr(enprod, "run_suite",
                        lambda *a, **k: pytest.fail("a member was read"))
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
                   "--products", "mean", "--out", str(tmp_path / "png"),
                   "--engine", "auto"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "enprod: refused:" in err
    assert "rw_ensbatch" in err and "--engine matplotlib" in err


def test_the_module_docstring_states_the_engine_contract_the_code_has(
        monkeypatch):
    """The module docstring is a door: it is read before the code is.

    It described three behaviours this release removed -- that ``auto``
    degrades to the matplotlib suite and reports the degradation by
    name, and that ``--domain`` and ``--dpi`` bring a caller here on
    purpose because the rust route refuses or warns about them.  All
    three are gone from the code, so each sentence is pinned to the
    function that answers it and the retired ones are pinned absent.
    """

    from woof import rustwx_lanes

    doc = inspect.getdoc(enprod) or ""
    # auto answers rust or a refusal; nothing degrades to this suite.
    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin", lambda: None)
    with pytest.raises(RuntimeError) as refusal:
        rustwx_lanes.resolve_ensemble_engine("auto")
    assert "--engine matplotlib" in str(refusal.value)
    assert "resolve_lane_engine" in doc
    # the one way in says it is a workaround, every run.
    assert "ensemble_workaround_notice" in doc
    assert "WORKAROUND:" in rustwx_lanes.ensemble_workaround_notice()
    # --dpi is the same picture on both routes, from the stated anchor.
    assert enprod.panel_pixels(300.0) == (2400, 1800)
    assert "1200x900" in doc and "--dpi 150" in doc
    # --domain is the nest choice the engine is given.
    assert "--domain dNN" in doc
    assert enprod.domain_choice_problem is not None
    for retired in ("degrade", "bring a caller here",
                    "refused or warned about"):
        assert retired not in doc, retired


def test_the_field_vocabulary_is_the_engines(tmp_path, monkeypatch):
    """A threshold kept in two places meant 30 degC here and 303 K there."""
    from woof import rustwx_lanes

    monkeypatch.setattr(
        rustwx_lanes, "list_ensemble_fields",
        lambda engine: {"t2": {"title": "2 m temperature", "units": "K",
                               "default_threshold": 303.0, "unit_slug": "k"}})
    seen: dict = {}
    _staged_engine(monkeypatch, seen, products=("prob",))
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "t2",
                   "--products", "prob", "--out", str(tmp_path / "png"),
                   "--engine", "rust"])
    assert rc == 0
    assert seen["threshold"] == 303.0
    delivered = render_layout.iter_rendered(tmp_path / "png")
    assert any("p303k" in path.name for path in delivered), \
        [path.name for path in delivered]


def test_a_field_the_engine_knows_is_not_refused_at_argument_check(
        tmp_path, monkeypatch):
    """The vocabulary is the engine's, so it may be wider than this module's."""
    from woof import rustwx_lanes

    monkeypatch.setattr(
        rustwx_lanes, "list_ensemble_fields",
        lambda engine: {"refl": {"title": "composite reflectivity",
                                 "units": "dBZ", "default_threshold": 40.0,
                                 "unit_slug": "dbz"},
                        "hail": {"title": "maximum hail diameter",
                                 "units": "mm", "default_threshold": 25.0,
                                 "unit_slug": "mm"}})
    seen: dict = {}
    _staged_engine(monkeypatch, seen)
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "hail",
                   "--products", "mean", "--out", str(tmp_path / "png"),
                   "--engine", "rust"])
    assert rc == 0
    assert seen["field"] == "hail"


def test_an_unknown_field_names_the_breakage_and_the_way_out(tmp_path, capsys):
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)),
                   "--field", "not_a_field", "--products", "mean",
                   "--out", str(tmp_path / "png"), "--engine", "matplotlib"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--list-fields" in err


def test_the_fallback_refuses_a_field_only_the_engine_can_draw(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(enprod, "FIELDS", dict(enprod.FIELDS))
    enprod.FIELDS["hail"] = enprod.FieldSpec(
        name="hail", title="maximum hail diameter", units="mm",
        unit_slug="mm", default_threshold=25.0, extract=None)
    rc = cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "hail",
                   "--products", "mean", "--out", str(tmp_path / "png"),
                   "--engine", "matplotlib"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "cannot draw hail" in err and "--engine rust" in err


def test_the_engine_vocabulary_reader_parses_its_own_row_grammar(monkeypatch):
    from woof import rustwx_lanes

    class Result:
        returncode = 0
        stdout = ("FIELD\tt2\t2 m temperature\tK\tdefault_threshold=303\t"
                  "unit_slug=k\nPRODUCTS\tmean\tspread\n")
        stderr = ""

    monkeypatch.setattr(rustwx_lanes.subprocess, "run", lambda *a, **k: Result())
    rows = rustwx_lanes.list_ensemble_fields(Path("rw_ensbatch"))
    assert rows == {"t2": {"title": "2 m temperature", "units": "K",
                           "default_threshold": 303.0, "unit_slug": "k"}}


def test_two_panels_of_one_product_do_not_collide_on_one_delivered_name(
        tmp_path, monkeypatch):
    """Two forecasts and one file is the overwrite the claims map prevents."""
    from woof import rustwx_lanes

    out = tmp_path / "png"

    def fake_ensemble(engine, manifest, *, store_root, out_dir, **kwargs):
        paths, rendered = [], []
        for index in (0, 1):
            panel = Path(out_dir) / f"frame{index}" / "arwen_ens_mean.png"
            panel.parent.mkdir(parents=True, exist_ok=True)
            panel.write_bytes(_PNG_MAGIC + str(index).encode())
            paths.append(panel)
            rendered.append(("mean", panel))
        return paths, [], [], {"rendered": rendered}

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin",
                        lambda: Path("rw_ensbatch"))
    monkeypatch.setattr(rustwx_lanes, "probe_ensemble_bin",
                        lambda path: (True, "ok"))
    monkeypatch.setattr(rustwx_lanes, "run_ensemble_renderer", fake_ensemble)
    assert cli.main(["enprod", str(_bare_ensemble(tmp_path)), "--field", "refl",
                     "--products", "mean", "--out", str(out),
                     "--engine", "rust"]) == 0
    delivered = render_layout.iter_rendered(out)
    assert len(delivered) == 2
    assert len({path.name for path in delivered}) == 2
