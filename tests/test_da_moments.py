"""The multi-moment analysis contract, pinned by the cycle that broke it.

A rung-2 real-radar LETKF cycle updated nine fields --
``('thp','qv','u','v','qc','qr','qi','qs','qg')`` -- on a Morrison
two-moment state.  Reflectivity assimilation created hydrometeor mass in
cells the background had left clear, the number concentrations stayed at
the background's exact zeros, and Morrison's slope closure evaluated the
resulting pairs to NaN.  Reflectivity OmA for that analysis was not
recoverable.

The tests here reproduce that signature end to end with the real filter
and the real reflectivity mirror, and then require it to be unreachable
through the module that writes an analysis.

CPU-only by construction: no CuPy import anywhere in this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.da import moments
from woof.da.letkf import (GriddedObs, GridGeometry, LetkfConfig,
                            Localization, analyze)
from woof.da.moments import MomentPolicyError
from woof.ensemble.increments import (apply_increments,
                                       apply_increments_to_checkpoint)
from woof.verify.npref import np_refl10cm_morrison_column

#: The node's own field list, reproduced exactly.
NODE_ANALYSIS_FIELDS = ("thp", "qv", "u", "v", "qc", "qr", "qi", "qs", "qg")

#: The threshold the test host 8 forensics counted offenders at, and the
#: threshold the reflectivity operator's species gate uses.  The guard's
#: own default is the SCHEME's 1e-14, which is stricter; this is here so
#: the regression is stated in the units the evidence was.
NODE_Q_THRESHOLD = 1.0e-9

_MORRISON_PAIRS = (("qc", "nc"), ("qr", "nr"), ("qi", "ni"),
                   ("qs", "ns"), ("qg", "ng"))


# ---------------------------------------------------------------------------
# Part 1 -- the state vector is the scheme's, not a typed-out list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mp_physics, expected_numbers", [
    (10, ("nc", "nr", "ni", "ns", "ng")),
    (18, ("qndrop", "qnr", "qni", "qns", "qng", "qnh")),
    (8, ("nr", "ni")),
])
def test_the_analysis_field_list_carries_the_schemes_own_moments(
        mp_physics, expected_numbers):
    """Keyed on the scheme, because that is the dimension that broke.

    A list that is right for Morrison is wrong for NSSL -- different
    spellings, and NSSL additionally advances two predicted-volume
    moments -- so a single hard-coded list cannot be right for both, and
    was silently wrong for the one it was used on.
    """

    fields = moments.analysis_fields(mp_physics)
    for name in expected_numbers:
        assert name in fields, (mp_physics, name)
    scheme = moments.scheme_moments(mp_physics)
    for pair in scheme.pairs:
        assert pair.mass in fields and pair.number in fields


def test_the_nssl_field_list_includes_its_predicted_volume_moments():
    fields = moments.analysis_fields(18)
    assert "qvolg" in fields and "qvolh" in fields
    assert "qnn" in fields, "predicted CCN is prognostic and is analysed"


def test_the_nssl_moment_set_follows_the_schemes_own_selectors():
    """Derived, not enumerated: hail off must remove hail's moments.

    The list comes from ``resolve_nssl2_mode``, which is WRF's own
    option-18 consistency pass, so a run without hail does not get an
    analysis field list that names ``qh``.
    """

    with_hail = moments.analysis_fields(18, nssl_hail_on=1)
    without = moments.analysis_fields(18, nssl_hail_on=0, nssl_density_on=1)
    assert {"qh", "qnh", "qvolh"} <= set(with_hail)
    assert not ({"qh", "qnh", "qvolh"} & set(without))
    assert "qvolg" in without


def test_a_single_moment_policy_reproduces_the_node_field_list():
    """The defect's own list is what the truncating policy produces.

    Which is the point: it is reachable only by naming the policy, and
    naming it turns the moment-consistency guard from optional into
    required.
    """

    fields = moments.analysis_fields(
        10, policy="single-moment-with-repair")
    assert fields == NODE_ANALYSIS_FIELDS


def test_an_unknown_scheme_is_refused_rather_than_guessed():
    with pytest.raises(MomentPolicyError, match="no moment structure"):
        moments.scheme_moments(99)


def test_an_unknown_moment_policy_is_refused():
    with pytest.raises(MomentPolicyError, match="unknown moment policy"):
        moments.analysis_fields(10, policy="whatever")


# ---------------------------------------------------------------------------
# Part 1 -- validation refuses the truncating update
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mp_physics, carried", [
    (10, ("qc", "nc", "qr", "nr", "qi", "ni", "qs", "ns", "qg", "ng")),
    (18, ("qc", "qndrop", "qr", "qnr", "qs", "qns", "qg", "qng", "qvolg")),
])
def test_a_mass_only_update_of_a_two_moment_state_is_refused(mp_physics,
                                                             carried):
    """The test host 8 field list, against a background that carries moments."""

    with pytest.raises(MomentPolicyError) as caught:
        moments.validate_analysis_fields(
            NODE_ANALYSIS_FIELDS + ("qh",), available=carried,
            mp_physics=mp_physics)
    message = str(caught.value)
    assert "leaves its prognostic moment" in message
    assert "q > 0 with N = 0" in message


def test_the_refusal_names_every_truncated_species():
    with pytest.raises(MomentPolicyError) as caught:
        moments.validate_analysis_fields(
            NODE_ANALYSIS_FIELDS,
            available=[name for pair in _MORRISON_PAIRS for name in pair],
            mp_physics=10)
    message = str(caught.value)
    for mass, number in _MORRISON_PAIRS:
        assert f"{mass} without {number}" in message


def test_the_scheme_is_detected_from_the_states_own_spellings():
    """No config needed: ``nr`` and ``qnr`` are different schemes' moments.

    The module that writes an analysis is handed a checkpoint and no
    namelist.  A guard that could be switched off by omitting an argument
    is a guard that will be.
    """

    morrison = moments.pairs_present(("qr", "nr", "qs", "ns"))
    assert {pair.number for pair in morrison} == {"nr", "ns"}
    nssl = moments.pairs_present(("qr", "qnr", "qg", "qng", "qvolg"))
    assert {pair.number for pair in nssl} == {"qnr", "qng"}
    assert [pair.volume for pair in nssl if pair.mass == "qg"] == ["qvolg"]


def test_a_full_moment_update_validates():
    receipt = moments.validate_analysis_fields(
        moments.analysis_fields(10),
        available=[name for pair in _MORRISON_PAIRS for name in pair],
        mp_physics=10)
    assert receipt["repair_required"] is False
    assert receipt["mass_only_species"] == []


def test_a_single_moment_policy_marks_the_repair_as_required():
    receipt = moments.validate_analysis_fields(
        NODE_ANALYSIS_FIELDS,
        available=[name for pair in _MORRISON_PAIRS for name in pair],
        mp_physics=10, policy="single-moment-with-repair")
    assert receipt["repair_required"] is True
    assert set(receipt["mass_only_species"]) == {
        mass for mass, _ in _MORRISON_PAIRS}


def test_analysing_no_hydrometeors_at_all_is_consistent():
    """Withholding the whole set is a complete choice, not a truncation."""

    receipt = moments.validate_analysis_fields(
        ("thp", "qv", "u", "v"),
        available=[name for pair in _MORRISON_PAIRS for name in pair],
        mp_physics=10)
    assert receipt["repair_required"] is False


# ---------------------------------------------------------------------------
# Part 2 -- the guard and the scheme-consistent repair
# ---------------------------------------------------------------------------


def _broken_state(shape=(2, 3, 4), *, masses=None):
    """A Morrison state whose named species hold mass with zero number."""

    state = {name: np.zeros(shape) for pair in _MORRISON_PAIRS
             for name in pair}
    for field, value in (masses or {"qr": 4.0011e-06}).items():
        state[field][0, 0, 0] = value
    return state


@pytest.mark.parametrize("mass, number, value", [
    ("qr", "nr", 1.18e-3),
    ("qs", "ns", 4.39e-3),
    ("qg", "ng", 6.44e-3),
    ("qc", "nc", 2.15e-3),
    ("qi", "ni", 6.19e-4),
])
def test_every_species_offender_is_counted_and_repaired(mass, number, value):
    """Five species, because the node saw three and the pair rule is general.

    The masses are the node's own reported maxima where it reported them.
    """

    state = _broken_state(masses={mass: value})
    report = moments.moment_consistency_report(state, mp_physics=10)
    assert report["offending_cells_total"] == 1
    entry = next(item for item in report["species"]
                 if item["mass_field"] == mass)
    assert entry["offending_cells"] == 1
    assert entry["max_offending_mass_kg_kg"] == pytest.approx(value)

    repaired, receipt = moments.repair_moments(state, mp_physics=10)
    assert receipt["repaired"] is True
    assert receipt["repaired_cells_total"] == 1
    assert number in repaired
    assert repaired[number][0, 0, 0] > 0.0
    # And the repair is the scheme's own bound, not a floor: doubling the
    # mass at a fixed lambda_min doubles the number.
    doubled = moments.repair_moments(
        _broken_state(masses={mass: 2.0 * value}), mp_physics=10)[0]
    assert doubled[number][0, 0, 0] == pytest.approx(
        2.0 * repaired[number][0, 0, 0], rel=1e-9)


def test_the_repair_is_the_schemes_own_limiter_not_a_reimplementation():
    """Oracle: the scheme mirror, called independently of the DA path.

    ``_np_morrison_slopes`` is the float64 transcription of the WRF
    limiter.  The repair must equal what that limiter produces for the
    same broken pair -- if it does not, the analysis has been handed a
    number the scheme would immediately overwrite with a different one.
    """

    from woof.verify.npref import _np_morrison_slopes

    masses = {"qc": 2.15e-3, "qr": 1.18e-3, "qi": 6.19e-4,
              "qs": 4.39e-3, "qg": 6.44e-3}
    state = _broken_state(masses=masses)
    repaired, _ = moments.repair_moments(state, mp_physics=10)

    q = {key: np.array([masses[f"q{key}"]]) for key in "crisg"}
    n = {key: np.array([0.0]) for key in "crisg"}
    _, _, bounded = _np_morrison_slopes(
        q, n, np.array([1.0]), np.array([273.15]),
        reset_cloud_number=False)
    for key, mass in (("c", "qc"), ("r", "qr"), ("i", "qi"),
                      ("s", "qs"), ("g", "qg")):
        number = dict(_MORRISON_PAIRS)[mass]
        assert repaired[number][0, 0, 0] == pytest.approx(
            float(bounded[key][0]), rel=1e-12), mass


@pytest.mark.parametrize("rho, temperature", [(0.3, 220.0), (1.4, 310.0)])
def test_the_repair_does_not_depend_on_the_density_it_was_given(rho,
                                                                temperature):
    """Stated in the module and pinned here rather than asserted.

    Density and temperature reach the limiter only through the cloud
    branch's ``pgam``, multiplied by the cloud number -- which at an
    offending cell is zero.  If that ever stopped being true the repair
    would silently become density-dependent, and this is what would say so.
    """

    from woof.verify.npref import _np_morrison_slopes

    masses = {"qc": 2.15e-3, "qr": 1.18e-3}
    state = _broken_state(masses=masses)
    repaired, _ = moments.repair_moments(state, mp_physics=10)
    q = {key: np.array([masses.get(f"q{key}", 0.0)]) for key in "crisg"}
    n = {key: np.array([0.0]) for key in "crisg"}
    _, _, bounded = _np_morrison_slopes(
        q, n, np.array([rho]), np.array([temperature]),
        reset_cloud_number=False)
    assert repaired["nc"][0, 0, 0] == pytest.approx(float(bounded["c"][0]),
                                                    rel=1e-12)
    assert repaired["nr"][0, 0, 0] == pytest.approx(float(bounded["r"][0]),
                                                    rel=1e-12)


def test_healthy_moments_are_left_exactly_alone():
    """A repair that rewrote sound pairs would be a second analysis."""

    state = _broken_state(masses={"qr": 1.0e-3})
    state["qs"][0, 1, 1] = 2.0e-3
    state["ns"][0, 1, 1] = 5.0e3          # healthy pair, unusual number
    before = np.array(state["ns"], copy=True)
    repaired, receipt = moments.repair_moments(state, mp_physics=10)
    assert "ns" not in repaired, "no offender in snow, so nothing to write"
    assert receipt["repaired_cells_total"] == 1
    np.testing.assert_array_equal(state["ns"], before)


def test_a_consistent_state_reports_consistent_and_repairs_nothing():
    state = _broken_state(masses={"qr": 1.0e-3})
    state["nr"][0, 0, 0] = 1.0e4
    report = moments.moment_consistency_report(state, mp_physics=10)
    assert report["consistent"] is True
    repaired, receipt = moments.repair_moments(state, mp_physics=10)
    assert repaired == {} and receipt["repaired"] is False


def test_mass_below_the_schemes_own_threshold_is_not_an_offender():
    """The scheme treats it as absent, so it demands no number moment."""

    scheme = moments.scheme_moments(10)
    state = _broken_state(masses={"qr": 0.5 * scheme.q_threshold})
    assert moments.moment_consistency_report(
        state, mp_physics=10)["consistent"] is True
    state = _broken_state(masses={"qr": 2.0 * scheme.q_threshold})
    assert moments.moment_consistency_report(
        state, mp_physics=10)["consistent"] is False


def test_nssl_detects_offenders_and_refuses_rather_than_inventing_a_number():
    """The second scheme, and the accurate limit of what this tree can repair.

    NSSL sets number from mass through its own fixed intercepts in
    ``module_mp_nssl_2mom.F``.  That routine is not ported here, and a
    number invented at this seam would be a science decision wearing a
    repair's clothes.
    """

    shape = (2, 3, 4)
    state = {name: np.zeros(shape) for name in
             ("qc", "qndrop", "qr", "qnr", "qs", "qns", "qg", "qng",
              "qvolg")}
    state["qr"][0, 0, 0] = 1.18e-3
    state["qg"][0, 1, 2] = 6.44e-3
    report = moments.moment_consistency_report(state, mp_physics=18,
                                               nssl_hail_on=0,
                                               nssl_density_on=1)
    assert report["offending_cells_total"] == 2
    with pytest.raises(MomentPolicyError) as caught:
        moments.repair_moments(state, mp_physics=18, nssl_hail_on=0,
                               nssl_density_on=1)
    message = str(caught.value)
    assert "no repair authority" in message
    assert "qr: 1" in message and "qg: 1" in message


# ---------------------------------------------------------------------------
# Part 2a -- the guard fails CLOSED on a moment that is not finite
#
# Re-verification RV3-01.  ``moment_consistency_report`` defined an
# offender as ``(mass > threshold) & (number <= 0.0)``.  IEEE comparison
# makes ``NaN <= 0.0`` False and ``NaN > threshold`` False, so a cell with
# ``qr = 1e-3`` and ``nr = NaN`` was neither active nor an offender: the
# probe drove a finite, unrelated ``thp`` increment through the real live
# writer with ``mp_physics=10``, the full-moment policy and repair
# enabled, the write SUCCEEDED, the receipt said the moments were
# consistent, and ``nr`` was still NaN.  Every probe below is that
# transcript, permanent.
# ---------------------------------------------------------------------------


#: Both signs of infinity and a NaN.  Keyed on the value because the three
#: fail differently under ordering: ``+inf > threshold`` is True (so the
#: cell IS active and its number IS read), ``-inf`` is not above threshold
#: at all, and NaN compares False against everything.  A guard written for
#: one of them is not a guard for the other two.
_NONFINITE = (float("nan"), float("inf"), float("-inf"))


@pytest.mark.parametrize("bad", _NONFINITE)
@pytest.mark.parametrize("field", ["qr", "nr"])
def test_a_nonfinite_moment_is_never_reported_consistent(field, bad):
    """The RV3-01 probe, over both halves of the pair and all three values.

    ``qr = 1e-3`` with ``nr = NaN`` was reported ``consistent=True``.  It
    is the same unevaluable state as ``nr = 0`` -- Morrison's
    ``lam = (six_c * N / q)**(1/3)`` is NaN for a NaN ``N`` exactly as it
    is zero for a zero one -- and the guard must say so.
    """

    state = _broken_state(masses={"qr": 1.0e-3})
    state["nr"][0, 0, 0] = 1.0e4          # the pair is healthy to start
    assert moments.moment_consistency_report(
        state, mp_physics=10)["consistent"] is True
    state[field][0, 0, 0] = bad
    report = moments.moment_consistency_report(state, mp_physics=10)
    assert report["consistent"] is False, report
    assert report["nonfinite_cells_total"] == 1, report
    rain = [entry for entry in report["species"]
            if entry["mass_field"] == "qr"][0]
    assert rain["nonfinite_cells"] == 1
    if field == "qr":
        assert rain["nonfinite_mass_cells"] == 1
    else:
        assert rain["nonfinite_number_cells"] == 1


def test_a_nonfinite_number_below_the_activity_threshold_is_not_counted():
    """The gate's other side: the scheme reads no moment there.

    Without this the test above would pass for a guard that simply
    refused every non-finite value anywhere in the array, which is a
    different (and unrequested) contract -- and one the ``-inf`` case
    above could not distinguish from the real one.
    """

    state = _broken_state(masses={"qr": 0.0})
    state["nr"][0, 0, 0] = np.nan
    report = moments.moment_consistency_report(state, mp_physics=10)
    assert report["nonfinite_cells_total"] == 0
    assert report["consistent"] is True


def test_a_nonfinite_volume_moment_is_refused_too():
    """NSSL's third moment.  Its pair is (mass, number, VOLUME)."""

    shape = (2, 3, 4)
    state = {name: np.zeros(shape) for name in
             ("qc", "qndrop", "qr", "qnr", "qs", "qns", "qg", "qng",
              "qvolg")}
    state["qg"][0, 1, 2] = 6.44e-3
    state["qng"][0, 1, 2] = 5.0e3         # number is healthy
    state["qvolg"][0, 1, 2] = np.nan      # the volume moment is not
    report = moments.moment_consistency_report(
        state, mp_physics=18, nssl_hail_on=0, nssl_density_on=1)
    assert report["offending_cells_total"] == 0, (
        "the number moment is sound; only the volume is not")
    assert report["nonfinite_cells_total"] == 1
    graupel = [entry for entry in report["species"]
               if entry["mass_field"] == "qg"][0]
    assert graupel["volume_field"] == "qvolg"
    assert graupel["nonfinite_volume_cells"] == 1


def test_the_repair_refuses_a_nonfinite_moment_instead_of_bounding_it():
    """Morrison HAS a limiter, and the limiter cannot repair this.

    ``repair_moments`` relies on the same report, so enabling the repair
    did not close the hole -- and the limiter's own bound,
    ``N = q * lam_min**3 / six_c``, is NaN for a NaN ``q`` and has
    nothing to bound for a NaN ``N``.  A refusal is the only accurate
    outcome, from the one scheme whose authority IS ported.
    """

    state = _broken_state(masses={"qr": 1.0e-3})
    state["nr"][0, 0, 0] = np.nan
    with pytest.raises(MomentPolicyError) as caught:
        moments.repair_moments(state, mp_physics=10)
    assert "not a finite number" in str(caught.value)
    assert np.isnan(state["nr"][0, 0, 0]), (
        "a refused repair must leave the state exactly as it found it")


def test_nssl_refuses_a_nonfinite_moment_as_a_nonfinite_one():
    """The second scheme, and the refusal it gets is the accurate one.

    NSSL would refuse this state anyway for want of a ported repair
    authority.  The finiteness check runs first so the message names what
    is actually wrong: porting NSSL's intercepts would not make a NaN
    repairable, and an operator told "no repair authority" would go
    looking for the wrong fix.
    """

    shape = (2, 3, 4)
    state = {name: np.zeros(shape) for name in
             ("qc", "qndrop", "qr", "qnr", "qs", "qns", "qg", "qng",
              "qvolg")}
    state["qr"][0, 0, 0] = 1.18e-3
    state["qnr"][0, 0, 0] = np.nan
    with pytest.raises(MomentPolicyError) as caught:
        moments.repair_moments(state, mp_physics=18, nssl_hail_on=0,
                               nssl_density_on=1)
    message = str(caught.value)
    assert "not a finite number" in message
    assert "no repair authority" not in message


# ---------------------------------------------------------------------------
# Part 2 -- the guard is in the module that WRITES an analysis
# ---------------------------------------------------------------------------


def _duck_state(shape=(2, 3, 4)):
    import types

    state = types.SimpleNamespace()
    for pair in _MORRISON_PAIRS:
        for name in pair:
            setattr(state, name, np.zeros(shape, np.float32))
    state.thp = np.zeros(shape, np.float32)
    return state


def test_the_live_state_writer_refuses_the_node_field_list():
    state = _duck_state()
    increments = {"qr": np.full((2, 3, 4), 1.0e-3, np.float32),
                  "thp": np.zeros((2, 3, 4), np.float32)}
    with pytest.raises(MomentPolicyError, match="leaves its prognostic"):
        apply_increments(state, increments)
    assert float(state.qr.max()) == 0.0, (
        "a refused analysis must leave the state alone")


def test_the_live_state_writer_repairs_under_the_declared_policy():
    state = _duck_state()
    increments = {"qr": np.full((2, 3, 4), 1.0e-3, np.float32),
                  "thp": np.zeros((2, 3, 4), np.float32)}
    receipt = apply_increments(state, increments,
                               moment_policy="single-moment-with-repair")
    assert receipt["moments"]["repaired"] is True
    assert receipt["moments"]["repaired_cells_total"] == 24
    assert float(state.nr.min()) > 0.0
    assert receipt["moments"]["authority"].startswith("WRF v4.6.1")


def test_the_live_state_writer_refuses_when_the_repair_is_disabled():
    state = _duck_state()
    increments = {"qr": np.full((2, 3, 4), 1.0e-3, np.float32)}
    with pytest.raises(ValueError, match="moment_repair is off"):
        apply_increments(state, increments,
                         moment_policy="single-moment-with-repair",
                         moment_repair=False)


def _write_morrison_checkpoint(path, shape=(2, 3, 4)):
    payload = {f"state/{name}": np.zeros(shape, np.float32)
               for pair in _MORRISON_PAIRS for name in pair}
    payload["state/thp"] = np.zeros(shape, np.float32)
    np.savez(path, **payload)
    return path


def test_the_checkpoint_writer_refuses_the_node_field_list(tmp_path):
    background = _write_morrison_checkpoint(tmp_path / "gpuwmrst_000.npz")
    analysis = tmp_path / "analysis.npz"
    with pytest.raises(MomentPolicyError, match="leaves its prognostic"):
        apply_increments_to_checkpoint(
            background, {"qs": np.full((2, 3, 4), 4.39e-3, np.float32)},
            analysis)
    assert not analysis.exists()
    assert not analysis.with_name(analysis.name + ".staged").exists()


@pytest.mark.parametrize("moment_repair", [True, False])
@pytest.mark.parametrize("bad", _NONFINITE)
def test_the_live_writer_refuses_a_nonfinite_paired_moment(bad,
                                                           moment_repair):
    """RV3-01's transcript against the real live writer.

    ``mp_physics=10``, the default full-moment policy, a finite and
    entirely unrelated ``thp`` increment -- and ``nr`` non-finite in the
    background beside active ``qr``.  The write used to succeed with a
    receipt claiming the moments were consistent.

    Keyed on ``moment_repair`` because the repair switch was the reason
    the hole stayed open: ``repair_moments`` reads the same report, so
    turning the repair ON did not close it either.  Both settings refuse.
    """

    state = _duck_state()
    state.qr[0, 0, 0] = 1.0e-3
    state.nr[0, 0, 0] = bad
    increments = {"thp": np.full((2, 3, 4), 0.5, np.float32)}
    with pytest.raises(ValueError, match="not a finite number"):
        apply_increments(state, increments, mp_physics=10,
                         moment_repair=moment_repair)
    assert not np.isfinite(state.nr[0, 0, 0]), (
        "the refused write must leave the poisoned moment exactly as it "
        "found it rather than 'repairing' it to another unevaluable value")
    assert float(state.thp.max()) == 0.0, (
        "and must not have applied the finite increment either")


@pytest.mark.parametrize("bad", _NONFINITE)
def test_the_checkpoint_writer_refuses_a_nonfinite_paired_moment(bad,
                                                                 tmp_path):
    """The same, through the writer a cycling run actually uses.

    ``_require_finite_result`` does not catch it: that checks the fields
    the INCREMENT names, and the pair's other half is precisely the field
    the increment did not name.
    """

    background = _write_morrison_checkpoint(tmp_path / "gpuwmrst_000.npz")
    with np.load(background, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["state/qr"][0, 0, 0] = 1.0e-3
    payload["state/nr"][0, 0, 0] = bad
    np.savez(background, **payload)

    analysis = tmp_path / "analysis.npz"
    with pytest.raises(ValueError, match="not a finite number"):
        apply_increments_to_checkpoint(
            background, {"thp": np.full((2, 3, 4), 0.5, np.float32)},
            analysis, mp_physics=10)
    assert not analysis.exists()
    assert not analysis.with_name(analysis.name + ".staged").exists()
    with np.load(background, allow_pickle=False) as data:
        assert not np.isfinite(data["state/nr"][0, 0, 0])
        assert float(data["state/thp"].max()) == 0.0


def test_the_checkpoint_writer_still_repairs_a_merely_depleted_pair(tmp_path):
    """The finiteness refusal must not swallow the repairable case.

    Same writer, same policy, same species -- the difference is that
    ``nr`` is a finite zero rather than a NaN, and that IS what Morrison's
    limiter has authority over.  Without this the tests above would be
    satisfied by a guard that simply refused every analysis.
    """

    background = _write_morrison_checkpoint(tmp_path / "gpuwmrst_000.npz")
    receipt = apply_increments_to_checkpoint(
        background, {"qr": np.full((2, 3, 4), 1.0e-3, np.float32)},
        tmp_path / "analysis.npz",
        moment_policy="single-moment-with-repair", mp_physics=10)
    assert receipt["moments"]["repaired"] is True
    assert receipt["moments"]["nonfinite_cells_total"] == 0
    with np.load(tmp_path / "analysis.npz", allow_pickle=False) as data:
        assert float(data["state/nr"].min()) > 0.0


def test_the_checkpoint_writer_repairs_and_stamps_the_repair(tmp_path):
    background = _write_morrison_checkpoint(tmp_path / "gpuwmrst_000.npz")
    analysis = tmp_path / "analysis.npz"
    receipt = apply_increments_to_checkpoint(
        background, {"qs": np.full((2, 3, 4), 4.39e-3, np.float32)},
        analysis, moment_policy="single-moment-with-repair")
    assert receipt["moments"]["repaired_cells_total"] == 24
    with np.load(analysis, allow_pickle=False) as data:
        assert float(data["state/ns"].min()) > 0.0
        assert data["state/ns"].dtype == np.float32
    # The pair is consistent by the guard's own report, read back off disk.
    with np.load(analysis, allow_pickle=False) as data:
        stored = {key[len("state/"):]: data[key] for key in data.files
                  if key.startswith("state/")}
    assert moments.moment_consistency_report(
        stored, mp_physics=10)["consistent"] is True


# ---------------------------------------------------------------------------
# The OSSE: reflectivity assimilation into background-clear cells
# ---------------------------------------------------------------------------


def _reflectivity_osse(analysis_fields, *, members=12, seed=20260730):
    """A twin experiment whose analysis creates mass where member 0 had none.

    This is the node's own geometry, at test scale.  MEMBER 0 is exactly
    clear over the western half -- every hydrometeor mass and every number
    concentration is a hard zero there, which is what the forensics found
    at 100% of the in-mask offending cells.  Other members carry a
    displaced storm that reaches into that half, so the ensemble has real
    covariance there, and a high reflectivity-like observation in the
    western half pulls member 0's mass up from nothing.

    The increments go through the product's own positivity policy before
    they are applied, because that is the path a cycle takes and because
    a clip is one of the two ways mass appears in a clear cell.
    """

    from woof.da.positivity import apply_positivity

    rng = np.random.default_rng(seed)
    nz, ny, nx = 3, 6, 8
    shape = (nz, ny, nx)
    grid = GridGeometry(dx_m=2000.0, dy_m=2000.0,
                        heights_m=np.array([500.0, 2000.0, 4500.0]))

    prior = {
        "thp": rng.standard_normal((members,) + shape) * 0.5,
        "qv": 8.0e-3 + rng.standard_normal((members,) + shape) * 5.0e-4,
        "u": rng.standard_normal((members,) + shape),
        "v": rng.standard_normal((members,) + shape),
    }

    # A storm whose position varies member to member.  Member 0's is
    # pushed entirely into the eastern half, so its western columns are
    # exact zeros; the rest of the ensemble reaches across.
    storm = np.zeros((members,) + shape)
    for member in range(members):
        edge = nx - 2 if member == 0 else max(1, (member * 7) % nx)
        storm[member, :, :, edge:] = np.abs(
            rng.standard_normal((nz, ny, nx - edge))) * 1.0e-3 + 1.0e-4
    for mass, number, scale, per_kg in (("qc", "nc", 0.4, 2.5e8),
                                        ("qr", "nr", 1.0, 5.0e3),
                                        ("qi", "ni", 0.3, 1.0e5),
                                        ("qs", "ns", 0.6, 5.0e3),
                                        ("qg", "ng", 0.6, 5.0e3)):
        prior[mass] = storm * scale
        # Number scales with mass, so it carries the same ensemble spread
        # and is exactly zero wherever the mass is.
        prior[number] = prior[mass] * (per_kg / 1.0e-3)

    # One observation in member 0's clear half, at a value the ensemble
    # is far below: this is the increment that creates mass from nothing.
    mask = np.zeros(shape, dtype=bool)
    mask[1, ny // 2, 1] = True
    simulated = 1.0e4 * prior["qr"] + 6.0e3 * prior["qs"]
    values = np.where(mask, 45.0, np.nan)
    errors = np.where(mask, 3.0, np.nan)
    obs = GriddedObs(name="reflectivity", values=values, errors=errors,
                     simulated=simulated, mask=mask)

    config = LetkfConfig(
        localization=Localization(horizontal_m=20000.0, vertical_m=8000.0),
        analysis_fields=tuple(analysis_fields), rtps_alpha=0.0)
    increments = analyze(prior, [obs], grid, config)
    background = {name: values_[0] for name, values_ in prior.items()}
    member_increments = {name: increments[name][0] for name in increments}
    member_increments, _ = apply_positivity(
        background, member_increments, policy="clip")
    analysis = {name: (background[name] + member_increments[name]
                       if name in member_increments else background[name])
                for name in background}
    return prior, analysis, mask


def _hz_column_finite(state, mask):
    """H_Z over every masked column, through Morrison's own mirror."""

    nz, ny, nx = state["qr"].shape
    finite = True
    saw = 0
    for j in range(ny):
        for i in range(nx):
            if not mask[:, j, i].any():
                continue
            saw += 1
            dbz = np_refl10cm_morrison_column(
                np.full(nz, 8.0e-3), state["qr"][:, j, i],
                state["nr"][:, j, i], state["qs"][:, j, i],
                state["ns"][:, j, i], state["qg"][:, j, i],
                state["ng"][:, j, i], np.full(nz, 285.0),
                np.full(nz, 8.0e4))
            finite = finite and bool(np.all(np.isfinite(dbz)))
    assert saw, "the probe must actually evaluate a masked column"
    return finite


def test_the_node_signature_reproduces_with_a_mass_only_analysis():
    """The defect itself, with the real filter and the real operator.

    This is the control: without it the tests below could pass because
    the OSSE never creates mass in a clear cell, and would then be
    proving nothing.
    """

    prior, analysis, mask = _reflectivity_osse(NODE_ANALYSIS_FIELDS)
    background = {name: values[0] for name, values in prior.items()}
    where = tuple(index[0] for index in np.nonzero(mask))
    assert background["qr"][where] == 0.0
    assert background["nr"][where] == 0.0
    assert analysis["qr"][where] > NODE_Q_THRESHOLD, (
        "the analysis must create rain mass in a background-clear cell, "
        "or this probe is not the node's case")
    assert analysis["nr"][where] == 0.0, (
        "and leave the number at the background's zero")
    report = moments.moment_consistency_report(
        analysis, mp_physics=10, q_threshold=NODE_Q_THRESHOLD)
    assert report["offending_cells_total"] > 0
    assert not _hz_column_finite(analysis, mask), (
        "Morrison's slope closure must produce NaN here -- that is the "
        "observed-and-correct operator behaviour the defect exposes")


def test_a_full_moment_analysis_leaves_h_z_finite_in_the_obs_mask():
    """The physically right fix: analyse both moments of the pair."""

    _, analysis, mask = _reflectivity_osse(moments.analysis_fields(10))
    report = moments.moment_consistency_report(
        analysis, mp_physics=10, q_threshold=NODE_Q_THRESHOLD)
    assert report["consistent"], report
    assert _hz_column_finite(analysis, mask)


def test_the_repaired_single_moment_analysis_also_leaves_h_z_finite():
    """The declared-truncation path, closed by the scheme's own limiter."""

    _, analysis, mask = _reflectivity_osse(NODE_ANALYSIS_FIELDS)
    repaired, receipt = moments.repair_moments(analysis, mp_physics=10)
    assert receipt["repaired_cells_total"] > 0
    analysis = {**analysis, **repaired}
    assert moments.moment_consistency_report(
        analysis, mp_physics=10,
        q_threshold=NODE_Q_THRESHOLD)["consistent"]
    assert _hz_column_finite(analysis, mask)


def test_the_node_signature_cannot_survive_the_analysis_writer(tmp_path):
    """The regression, stated as the evidence stated it.

    ``q > 1e-9`` with ``N == 0`` after an analysis: refused outright under
    the default policy, and repaired out of existence under the declared
    single-moment one.  There is no third path through the module that
    writes an analysis.
    """

    prior, _, mask = _reflectivity_osse(NODE_ANALYSIS_FIELDS)
    background = tmp_path / "gpuwmrst_000.npz"
    np.savez(background, **{f"state/{name}": values[0].astype(np.float32)
                            for name, values in prior.items()})
    increments = {name: np.full(prior["qr"].shape[1:], 1.0e-3, np.float32)
                  for name in ("qr", "qs", "qg")}

    with pytest.raises(MomentPolicyError):
        apply_increments_to_checkpoint(background, increments,
                                       tmp_path / "a.npz")

    receipt = apply_increments_to_checkpoint(
        background, increments, tmp_path / "b.npz",
        moment_policy="single-moment-with-repair")
    assert receipt["moments"]["repaired_cells_total"] > 0
    with np.load(tmp_path / "b.npz", allow_pickle=False) as data:
        stored = {key[len("state/"):]: data[key] for key in data.files
                  if key.startswith("state/")}
    report = moments.moment_consistency_report(
        stored, mp_physics=10, q_threshold=NODE_Q_THRESHOLD)
    assert report["offending_cells_total"] == 0
    assert _hz_column_finite(stored, mask)


# ---------------------------------------------------------------------------
# The property the fix must not break
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_number_fields", [0, 5])
def test_rho_one_inactive_increments_stay_bitwise_zero(n_number_fields):
    """The localisation gate's guarantee, with the moments in the vector.

    Keyed on whether the number moments are analysed, because that is
    what changed: a state vector three times longer must not perturb the
    exactly-zero increment at a gridpoint no observation reaches.
    """

    rng = np.random.default_rng(4242)
    members, nz, ny, nx = 6, 2, 4, 12
    shape = (nz, ny, nx)
    fields = ("thp", "qr", "qs", "qg", "qc", "qi")
    if n_number_fields:
        fields = fields + ("nr", "ns", "ng", "nc", "ni")
    prior = {name: rng.standard_normal((members,) + shape) + 1.0
             for name in fields}
    mask = np.zeros(shape, dtype=bool)
    mask[0, 0, 0] = True
    simulated = prior["thp"] * 1.3
    obs = GriddedObs(name="probe",
                     values=np.where(mask, 2.0, np.nan),
                     errors=np.where(mask, 1.0, np.nan),
                     simulated=simulated, mask=mask)
    config = LetkfConfig(
        localization=Localization(horizontal_m=2500.0, vertical_m=1000.0),
        analysis_fields=fields, rtps_alpha=0.0, prior_inflation=1.0)
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0,
                        heights_m=np.array([250.0, 1800.0]))
    increments = analyze(prior, [obs], grid, config)
    far = (slice(None), slice(None), slice(None), slice(nx - 2, nx))
    for name in fields:
        values = increments[name][far]
        assert np.count_nonzero(values) == 0, (
            f"{name}: rho=1 must leave every inactive increment bitwise "
            "zero, whatever the length of the state vector")
        assert not np.signbit(values).any(), "and positive zero at that"


# ---------------------------------------------------------------------------
# Part 4 -- Thompson: the scheme that refused, and the block that answers
# ---------------------------------------------------------------------------
#
# A real-radar cycle on aerosol-aware Thompson (mp_physics=28) hit the
# same signature from the other side: the hydrometeor analysis left 1 834
# cells holding mass with a number moment clipped to zero -- 212 cloud,
# 1 056 rain, 566 ice -- and because no repair authority for the scheme
# was registered, the guard refused and the reflectivity and hydrometeor
# analyses could not run at all.  The authority was there the whole time,
# inside the scheme: mp_thompson's entry block sets exactly these numbers
# from exactly these masses before it computes a single process rate.

#: 1/alt on the state, i.e. the density the scheme is handed.
_THOMPSON_RHO = 0.9

_THOMPSON_PAIRS = (("qc", "nc"), ("qr", "nr"), ("qi", "ni"))


def _thompson_state(shape=(2, 3, 4), *, masses=None, numbers=None,
                    rho=_THOMPSON_RHO):
    """An mp=28 state whose named species hold mass with zero number."""

    state = {name: np.zeros(shape) for pair in _THOMPSON_PAIRS
             for name in pair}
    state.update({name: np.zeros(shape)
                  for name in ("qv", "qs", "qg", "nwfa", "nifa")})
    state["alt"] = np.full(shape, 1.0 / rho)
    for field, value in (masses or {}).items():
        state[field][0, 0, 0] = value
    for field, value in (numbers or {}).items():
        state[field][0, 0, 0] = value
    return state


def test_the_aerosol_aware_scheme_is_registered_with_its_own_moments():
    scheme = moments.scheme_moments(28)
    assert scheme.name == "Thompson aerosol-aware"
    assert [pair.mass for pair in scheme.pairs] == ["qc", "qr", "qi"]
    assert scheme.number_fields == ("nc", "nr", "ni")
    # The aerosol tracers are prognostic and have no mass to pair with,
    # so they are carried and never treated as half of a broken pair.
    assert scheme.unpaired == ("nwfa", "nifa")
    # R1, the scheme's own activity gate, not the module's default.
    assert scheme.q_threshold == 1.0e-12
    assert scheme.repair_authority == moments.THOMPSON_REPAIR_AUTHORITY


def test_the_guard_stops_refusing_the_cycle_that_could_not_run():
    """The 1 834-cell signature, reproduced and then repaired."""

    shape = (8, 16, 16)
    state = _thompson_state(shape)
    rng = np.random.default_rng(20260918)
    # An increment that adds condensate mass to cells the background left
    # clear, which is what a reflectivity analysis does.
    for mass, count in (("qc", 212), ("qr", 1056), ("qi", 566)):
        flat = rng.choice(state[mass].size, size=count, replace=False)
        state[mass].reshape(-1)[flat] = 1.0e-4

    report = moments.moment_consistency_report(state, mp_physics=28)
    assert report["offending_cells_total"] == 1834
    assert report["consistent"] is False

    repaired, receipt = moments.repair_moments(state, mp_physics=28)
    assert receipt["repaired"] is True
    assert receipt["repaired_cells_total"] == 1834
    assert receipt["authority"] == moments.THOMPSON_REPAIR_AUTHORITY
    for mass, number in _THOMPSON_PAIRS:
        active = state[mass] > 1.0e-12
        assert np.all(repaired[number][active] > 0.0)

    after = dict(state)
    after.update(repaired)
    assert moments.moment_consistency_report(
        after, mp_physics=28)["consistent"] is True


def test_the_repaired_rain_number_is_the_schemes_own_millimetre_drop():
    """Invert the number back through the scheme's own distribution.

    module_mp_thompson.F:1883 states the rule in one line -- rain mass
    that arrives without a number is a 1 mm median volume drop.  The
    repaired number must reproduce that diameter, because the diameter
    is the physics and the number is only how the scheme stores it.
    """

    mass = 1.0e-3
    repaired, _ = moments.repair_moments(
        _thompson_state(masses={"qr": mass}), mp_physics=28)
    am_r = 3.1415926536 * 1000.0 / 6.0
    per_volume = repaired["nr"][0, 0, 0] * _THOMPSON_RHO
    lamr = (am_r * 6.0 * per_volume / (mass * _THOMPSON_RHO)) ** (1.0 / 3.0)
    assert (3.672 / lamr) == pytest.approx(1.0e-3, rel=1.0e-6)


def test_the_repaired_ice_number_is_the_schemes_own_ceiling():
    """:1855-1856 -- a 5 um crystal, capped at 999e3 m^-3.

    Any ice mass a radar analysis creates is far above the ~5e-9
    kg m^-3 where that cap starts to bind, so the repaired ice number is
    the ceiling itself, and it is a per-VOLUME ceiling: the state's own
    per-kilogram number is the ceiling divided by the density.
    """

    repaired, _ = moments.repair_moments(
        _thompson_state(masses={"qi": 1.0e-4}), mp_physics=28)
    assert repaired["ni"][0, 0, 0] == pytest.approx(
        999.0e3 / _THOMPSON_RHO, rel=1.0e-6)


def test_the_repaired_cloud_number_is_the_schemes_largest_droplet():
    """:1830-1841 -- nc floors at 2 m^-3, so the size clamp decides.

    With no droplets at all the entry block's first lambda implies drops
    far larger than the scheme allows, the ``xDc > 2*D0r`` arm fires, and
    the rediagnosed number is the one that puts the distribution at a
    100 um mean diameter.  Checked against the clamp's own arithmetic
    rather than against a number copied out of a previous run.
    """

    from woof.core.thompson_aerosol_contract import (AM_R, CCE2, CCG1,
                                                      OCG2)

    mass = 1.0e-3
    repaired, _ = moments.repair_moments(
        _thompson_state(masses={"qc": mass}), mp_physics=28)
    nu_c = 15                      # MIN(15, NINT(1000e6/2) + 2)
    lamc = CCE2[nu_c] / (50.0e-6 * 2.0)
    expected = CCG1[nu_c] * OCG2[nu_c] * (mass * _THOMPSON_RHO) / AM_R \
        * lamc ** 3.0
    assert repaired["nc"][0, 0, 0] * _THOMPSON_RHO == pytest.approx(
        expected, rel=1.0e-5)
    assert expected < 1999.0e6, "the droplet ceiling must not be what bound"


def test_removing_all_the_mass_zeroes_the_number():
    """The reverse case, and the same block's answer to it.

    :1900-1904: at or below R1 the entry block zeroes the mass AND the
    number.  An increment that takes all of a species' mass away leaves
    the number standing, and the repair puts it where the scheme will.
    """

    state = _thompson_state(masses={"qr": 1.0e-14}, numbers={"nr": 5.0e4})
    report = moments.moment_consistency_report(state, mp_physics=28)
    assert report["stranded_number_cells_total"] == 1
    assert report["offending_cells_total"] == 0
    # Not a NaN risk, so not a refusal: the state stays "consistent" and
    # the cell is repaired anyway.
    assert report["consistent"] is True

    repaired, receipt = moments.repair_moments(state, mp_physics=28)
    assert receipt["stranded_cells_repaired"] == 1
    assert repaired["nr"][0, 0, 0] == 0.0


def test_the_thompson_repair_is_a_no_op_on_consistent_cells():
    """Healthy pairs are not a second analysis nobody asked for."""

    state = _thompson_state(masses={"qr": 1.0e-3}, numbers={"nr": 8.0e4})
    state["qi"][1, 1, 1] = 2.0e-5
    state["ni"][1, 1, 1] = 4.0e5
    report = moments.moment_consistency_report(state, mp_physics=28)
    assert report["consistent"] is True
    assert report["stranded_number_cells_total"] == 0
    repaired, receipt = moments.repair_moments(state, mp_physics=28)
    assert repaired == {}
    assert receipt["repaired"] is False
    assert receipt["repaired_cells_total"] == 0


@pytest.mark.parametrize("rho", [0.4, 0.9, 1.25])
def test_the_thompson_repair_depends_on_density_only_where_it_must(rho):
    """Cloud and rain cancel it; ice keeps it, through the ceiling alone.

    Stated as a property rather than as three numbers, because the
    property is what makes the repair safe to run on a state whose
    density this module had to reconstruct.
    """

    masses = {"qc": 1.0e-3, "qr": 1.0e-3, "qi": 1.0e-4}
    reference = moments.repair_moments(
        _thompson_state(masses=masses, rho=_THOMPSON_RHO), mp_physics=28)[0]
    repaired = moments.repair_moments(
        _thompson_state(masses=masses, rho=rho), mp_physics=28)[0]
    for number in ("nc", "nr"):
        assert repaired[number][0, 0, 0] == pytest.approx(
            reference[number][0, 0, 0], rel=1.0e-6)
    assert repaired["ni"][0, 0, 0] == pytest.approx(
        reference["ni"][0, 0, 0] * _THOMPSON_RHO / rho, rel=1.0e-6)


def test_a_thompson_repair_without_a_density_is_refused_by_name():
    state = _thompson_state(masses={"qr": 1.0e-3})
    del state["alt"]
    with pytest.raises(MomentPolicyError, match="no 'alt'"):
        moments.repair_moments(state, mp_physics=28)


def test_the_two_schemes_that_share_their_spellings_are_not_guessed_at():
    """nc/nr/ni belong to Morrison and to Thompson-28 alike.

    Before the Thompson repair existed, that spelling set could only be
    Morrison's, and the detected-structure arm said so.  It can now be
    either, and the two schemes answer a depleted pair differently, so
    the arm names the ambiguity instead of picking.
    """

    state = _thompson_state(masses={"qr": 1.0e-3})
    with pytest.raises(MomentPolicyError, match="Pass mp_physics"):
        moments.repair_moments(state)
    # A snow or graupel moment still identifies Morrison on its own.
    morrison = moments.repair_moments(_broken_state())[1]
    assert morrison["authority"] == moments.MORRISON_REPAIR_AUTHORITY


def test_every_scalar_the_mirror_uses_is_the_device_headers_own():
    """One spelling of each constant, enforced rather than asserted.

    The mirror and the CUDA header are two transcriptions of the same
    Fortran.  If they can drift, the analysis and the forecast disagree
    about what the scheme is, so the header is parsed and compared here.
    """

    import re
    from pathlib import Path

    from woof.core import thompson_entry

    header = (Path(thompson_entry.__file__).resolve().parents[1]
              / "core" / "kernels" / "thompson_aerosol_common.cuh")
    defines = dict(re.findall(
        r"#define\s+(THOMPSON_AA_\w+)\s+([0-9.eE+-]+)f?\s",
        header.read_text()))
    for name, (define, _) in thompson_entry.CUH_SCALARS.items():
        assert define in defines, f"{define} is not in the device header"
        from woof.core import thompson_aerosol_contract as contract
        from woof.core import thompson_aerosol_state as aerosol_state
        for owner in (thompson_entry, contract, aerosol_state):
            ours = getattr(owner, name, None)
            if ours is not None:
                break
        assert ours is not None, f"nothing in the tree spells {name}"
        assert float(np.float32(ours)) == pytest.approx(
            float(np.float32(float(defines[define]))), rel=1.0e-7), (
                f"{name} disagrees with {define}")


def _thompson_duck_state(shape=(2, 3, 4), rho=_THOMPSON_RHO):
    import types

    state = types.SimpleNamespace()
    for pair in _THOMPSON_PAIRS:
        for name in pair:
            setattr(state, name, np.zeros(shape, np.float32))
    for name in ("thp", "qv", "qs", "qg", "nwfa", "nifa"):
        setattr(state, name, np.zeros(shape, np.float32))
    state.alt = np.full(shape, np.float32(1.0 / rho), np.float32)
    return state


def test_the_live_state_writer_repairs_a_thompson_analysis():
    """The door the cycle actually goes through, at mp_physics=28."""

    state = _thompson_duck_state()
    increments = {name: np.full((2, 3, 4), 1.0e-4, np.float32)
                  for name in ("qc", "qr", "qi")}
    receipt = apply_increments(state, increments, mp_physics=28,
                               moment_policy="single-moment-with-repair")
    assert receipt["moments"]["repaired"] is True
    assert receipt["moments"]["repaired_cells_total"] == 72
    assert receipt["moments"]["authority"] == (
        moments.THOMPSON_REPAIR_AUTHORITY)
    for number in ("nc", "nr", "ni"):
        assert float(getattr(state, number).min()) > 0.0
