"""The spectral-comparison pin owner's deferred guards, each one measured.

Four defects in the v2 comparison class, one in its receipt gate surface, and
one in its reader.  Every threshold this suite pins was MEASURED first; the
measurement and the margin derivation are recorded beside the constant in
``woof.verify.spectral_compare`` and in
``docs/public/SPECTRAL_VERIFICATION.md``.

1. BAND-EDGE ROUNDING.  ``_band_mask`` compared a wavelength recovered as
   ``1/|k|`` against a declared edge with no allowance, while the Nyquist
   edge in the same module already carried an 8-ulp one.  Measured over
   2,414,712 retained modes on 48 grids (n 64..400, dx 250..5000 m), 80 modes
   were graded into the wrong band, and every one of them was a |j|=1
   domain-scale mode -- the most energetic mode the disk retains.  Which band
   it lands in depended on the float spelling of dx, so n=100/dx=1000 and
   n=200/dx=500 -- the same physical wavelength -- disagreed.

2. ZERO-VARIANCE GUARD.  The guard was ``denominator > 0.0``, which only
   catches an exactly-zero band power.  Measured: a plane held at a constant
   273.15 K detrends to float cancellation residue, not to zero, and scores
   ``spectral_correlation = 0.9999999999999997`` with ``status = "ok"`` --
   a field with no structure at all passing a 0.95 correlation gate.  Whether
   it does depends on whether the constant is a dyadic rational: 1.0 and
   101325.0 give exact zero, 273.15 does not.

3. GATE-METRIC DUPLICATION.  The synthetic ``partition`` row carried a
   verbatim copy of the ``total`` component's ``reference_power`` so the
   ``minimum_reference_power`` floor could be enforced.  That copy was
   gate-addressable, so a gate declaring ``component = "partition"``,
   ``metric = "reference_power"`` silently graded the total-KE row, and the
   calibrator built two gates out of one measurement.

4. CROSS-BOX TOLERANCE.  Measured 2026-08-20 on sha256-identical input and
   module bytes: the Windows desktop (Python 3.13.7, numpy 2.2.6, UCRT) and
   test host 1 (Python 3.14.4, numpy 2.3.5, glibc 2.43) agree on 103 of
   279 metric values to the bit and differ on the rest.  The receipt
   self-hash is therefore a THIS-BOX identity, exactly as the portable frame
   header rule already says for a decode; what travels is the metric values
   under a declared tolerance.

5. NAN CEILING.  The reader's ceiling on non-finite cells is zero, which is
   right -- one NaN cell propagates through the FFT into every retained mode,
   so every band power, correlation and gate on the whole plane becomes NaN.
   The refusal did not say that, did not say how many cells or where, did not
   separate "empty" from "non-finite", and named no remedy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.verify import spectral_compare
from woof.verify import spectral_io
from woof.verify import spectral_receipt

DX = 1000.0
BANDS = (
    spectral_compare.WavelengthBand("convective", 2.0, 8.0),
    spectral_compare.WavelengthBand("storm", 8.0, 32.0),
    spectral_compare.WavelengthBand("meso", 32.0, 100.0),
    spectral_compare.WavelengthBand("synoptic", 100.0, None),
)


def wave(n: int, cycles: int, *, axis: str = "x") -> np.ndarray:
    coordinate = np.arange(n, dtype=np.float64)
    one_d = np.cos(2.0 * np.pi * cycles * coordinate / n)
    return (np.tile(one_d, (n, 1)) if axis == "x"
            else np.tile(one_d[:, None], (1, n)))


def band_row(result, name, component="scalar"):
    for row in spectral_compare.metric_rows(result):
        if row["band"] == name and row["component"] == component:
            return row
    raise AssertionError((name, component))


# ---------------------------------------------------------------------------
# 1. Band-edge rounding
# ---------------------------------------------------------------------------


def _exact_band(n: int, dx: float, jy: int, jx: int) -> str | None:
    """Which band a mode belongs to under exact rational arithmetic.

    ``lambda = n*dx/sqrt(jy^2+jx^2)``, so ``lambda >= edge`` is
    ``(n*dx)^2 >= edge^2 * (jy^2+jx^2)`` -- integers, no rounding.
    """

    modes = jy * jy + jx * jx
    span = int(n * dx) ** 2
    for band in BANDS:
        low = None if band.minimum_km is None else int(band.minimum_km * 1000) ** 2
        high = None if band.maximum_km is None else int(band.maximum_km * 1000) ** 2
        if ((low is None or span >= low * modes)
                and (high is None or span < high * modes)):
            return band.name
    return None


def test_a_mode_exactly_on_a_band_edge_is_graded_in_the_declared_band():
    """n=100, dx=1000 m: the |j|=1 mode is exactly 100 km, the synoptic floor.

    ``1/(1/100000)`` evaluates to 99999.99999999999, one ulp low, so the
    four domain-scale modes were graded 'meso' instead of 'synoptic'.  The
    same physical wavelength on n=200/dx=500 was graded 'synoptic'.
    """

    misgraded = []
    checked = 0
    for n, dx in ((100, 1000.0), (200, 500.0), (100, 2000.0), (250, 400.0)):
        geometry = spectral_compare.mode_geometry(n, n, dx_m=dx, dy_m=dx)
        valid = np.asarray(geometry["valid"], dtype=bool)
        assigned = np.full(valid.shape, "", dtype=object)
        for band in BANDS:
            assigned[spectral_compare._band_mask(geometry, band)] = band.name
        indices = (np.fft.fftfreq(n) * n).round().astype(int)
        for row in range(n):
            for column in range(n):
                if not valid[row, column]:
                    continue
                checked += 1
                exact = _exact_band(n, dx, int(indices[row]), int(indices[column]))
                if (assigned[row, column] or None) != exact:
                    misgraded.append(
                        (n, dx, int(indices[row]), int(indices[column]),
                         exact, assigned[row, column] or None))
    assert checked > 90000
    assert misgraded == []


def test_the_band_edge_allowance_is_the_one_the_nyquist_edge_already_used():
    assert spectral_compare.BAND_EDGE_ULP_ALLOWANCE == 8


def test_a_gapped_band_pair_still_drops_a_mode_the_campaign_did_not_claim():
    """Snapping must not smuggle a mode into a gap the campaign declared.

    Bands 2-8 km and 10-40 km leave 8-10 km unclaimed.  A mode at exactly
    8 km is excluded by the lower band's exclusive maximum and is below the
    upper band's inclusive minimum, so it stays uncovered -- by declaration,
    not by rounding.
    """

    gapped = spectral_compare.parse_bands([
        {"name": "low", "minimum_km": 2.0, "maximum_km": 8.0},
        {"name": "high", "minimum_km": 10.0, "maximum_km": 40.0},
    ])
    geometry = spectral_compare.mode_geometry(64, 64, dx_m=1000.0, dy_m=1000.0)
    coverage = spectral_compare._uncovered_modes(geometry, gapped)
    assert coverage["uncovered_mode_count"] > 0
    wavelength = np.asarray(geometry["wavelength_m"], dtype=np.float64)
    on_edge = np.asarray(geometry["valid"], dtype=bool) & (
        np.abs(wavelength - 8000.0) <= 8.0 * np.spacing(8000.0))
    assert np.count_nonzero(on_edge) == 4
    assert not np.any(spectral_compare._band_mask(geometry, gapped[0]) & on_edge)


# ---------------------------------------------------------------------------
# 2. Zero-variance guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("constant", [273.15, 300.7, 1013.25, 101325.0, 0.0])
def test_a_constant_plane_is_unresolved_not_a_perfect_correlation(constant):
    plane = np.full((64, 64), constant, dtype=np.float64)
    result = spectral_compare.compare_scalar(
        plane, plane, dx_m=DX, bands=BANDS)
    scored = 0
    for row in spectral_compare.metric_rows(result):
        assert row["status"] == "unresolved", row
        assert row.get("spectral_correlation") is None
        if row["mode_count"] == 0:
            # A band this grid cannot resolve at all; already unresolved.
            continue
        scored += 1
        assert "no resolvable variance" in str(row["note"])
    assert scored >= 2


def test_the_zero_variance_floor_sits_between_the_two_measured_populations():
    """Measured 2026-08-20, both ends of the separation this floor divides.

    Constant planes (84 cases, n=8..512, 12 physically-plausible constants):
    valid-disk power over the plane's own mean square peaked at 2.784e-31.
    Real structure at the float32 storage quantization limit -- the finest
    structure a wrfout file can carry at all -- measured 1.24e-14.  The
    declared floor is 1e-22: 8.6 decades above the noise ceiling, 8.1 decades
    below the signal floor, near the geometric centre of the measured gap.
    """

    floor = spectral_compare.VARIANCE_FLOOR_RELATIVE_POWER
    assert floor == 1e-22
    assert floor > 1e4 * 2.784e-31
    assert floor < 1e-4 * 1.235e-14


def test_structure_at_the_float32_quantization_limit_still_resolves():
    generator = np.random.default_rng(11)
    step = float(np.spacing(np.float32(273.15)))
    plane = (np.float32(273.15)
             + np.float32(step) * generator.integers(
                 0, 4, (128, 128)).astype(np.float32)).astype(np.float64)
    result = spectral_compare.compare_scalar(
        plane, plane, dx_m=DX, bands=BANDS)
    resolved = [row for row in spectral_compare.metric_rows(result)
                if row["status"] == "ok"]
    assert resolved, "the guard must not refuse a real float32-limited field"
    assert all(row["spectral_correlation"] == pytest.approx(1.0, abs=1e-12)
               for row in resolved)


def test_a_gate_on_a_zero_variance_field_is_incomplete_never_pass(tmp_path):
    plane = np.full((64, 64), 273.15, dtype=np.float64)
    np.savez(tmp_path / "left.npz", W=plane)
    np.savez(tmp_path / "reference.npz", W=plane)
    spec = tmp_path / "spec.toml"
    spec.write_text(_SPEC_TEMPLATE.format(gate=_CORRELATION_GATE),
                    encoding="utf-8")
    receipt = spectral_receipt.score_registration(
        spectral_receipt.make_registration(spec))
    assert receipt["verdict"] == "incomplete"
    assert receipt["gates"]["passed"] == 0


# ---------------------------------------------------------------------------
# 3. Gate-metric duplication
# ---------------------------------------------------------------------------


def test_the_partition_row_no_longer_republishes_the_total_reference_power():
    n = 96
    u = wave(n, 6)
    v = np.roll(wave(n, 6, axis="y"), 3, axis=0) * 0.5
    result = spectral_compare.compare_vector(
        1.03 * u, 1.01 * v, u, v, dx_m=DX, bands=BANDS)
    rows = spectral_receipt._gate_rows(
        {"pair": "p", "field": "wind", "result": result})
    partition = [row for row in rows if row["component"] == "partition"]
    assert partition
    for row in partition:
        assert "reference_power" not in row
        assert "gate_reference_power" in row


def test_a_gate_naming_a_metric_its_component_does_not_carry_is_refused(tmp_path):
    spec = tmp_path / "spec.toml"
    spec.write_text(
        _SPEC_TEMPLATE.format(gate=_PARTITION_REFERENCE_POWER_GATE),
        encoding="utf-8")
    with pytest.raises(ValueError) as error:
        spectral_receipt.make_registration(spec)
    message = str(error.value)
    assert "partition" in message
    assert "reference_power" in message
    assert "divergent_energy_fraction_difference" in message


def test_every_declared_component_metric_pair_is_reachable_and_unique(tmp_path):
    """The table is the contract: what it declares, a real scoring run emits."""

    n = 96
    u = wave(n, 6)
    v = np.roll(wave(n, 6, axis="y"), 3, axis=0) * 0.5
    vector = spectral_compare.compare_vector(
        1.03 * u, 1.01 * v, u, v, dx_m=DX, bands=BANDS)
    scalar = spectral_compare.compare_scalar(
        1.03 * u, u, dx_m=DX, bands=BANDS)
    emitted: dict[str, set[str]] = {}
    for result in (scalar, vector):
        for row in spectral_receipt._gate_rows(
                {"pair": "p", "field": "f", "result": result}):
            if row["status"] != "ok":
                continue
            emitted.setdefault(row["component"], set()).update(
                key for key in row
                if key in spectral_receipt.metrics_for_component(
                    row["component"]))
    for component, metrics in spectral_receipt.COMPONENT_METRICS.items():
        assert emitted.get(component) == set(metrics), component


def test_a_wildcard_gate_gives_every_matched_pair_its_own_row_id(tmp_path):
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        field = wave(64, 4) * (1.0 if name == "a" else 1.01)
        np.savez(directory / "left.npz", W=field)
        np.savez(directory / "reference.npz", W=wave(64, 4))
    spec = tmp_path / "spec.toml"
    spec.write_text(_TWO_PAIR_SPEC, encoding="utf-8")
    receipt = spectral_receipt.score_registration(
        spectral_receipt.make_registration(spec))
    rows = receipt["gates"]["rows"]
    assert len(rows) == 2
    assert len({row["row_id"] for row in rows}) == 2
    assert len({row["id"] for row in rows}) == 1


def test_a_metric_row_states_its_band_once():
    result = spectral_compare.compare_scalar(
        wave(64, 4), wave(64, 4), dx_m=DX, bands=BANDS)
    row = spectral_compare.metric_rows(result)[0]
    assert row["band"]
    assert "name" not in row


# ---------------------------------------------------------------------------
# 4. Cross-box tolerance
# ---------------------------------------------------------------------------


def test_the_cross_box_rule_is_declared_and_derived_from_the_measurement():
    rule = spectral_compare.CROSS_BOX_RULE
    assert rule["rule"] == "gpuwm-spectral-cross-box-v1"
    assert rule["tolerance"] == spectral_compare.CROSS_BOX_TOLERANCE == 1e-12
    assert rule["receipt_sha256_is_portable"] is False
    assert set(rule["exact_across_boxes"]) == {
        "mode_count", "minimum_wavelength_km", "maximum_wavelength_km"}
    assert rule["measured_worst_bounded"] == 1.1102e-14
    assert rule["measured_worst_unbounded_ratio"] == 7.7498e-15
    assert rule["measured_worst_power_over_reference_power"] == 7.7304e-16
    assert rule["measured_worst_exact"] == 0.0
    assert "192" in rule["measured_on"]
    # The declared tolerance clears the measured worst case by 90x.
    worst = max(rule["measured_worst_bounded"],
                rule["measured_worst_unbounded_ratio"])
    assert rule["tolerance"] / worst > 50.0


def test_a_bounded_metric_that_is_analytically_zero_is_not_judged_relatively():
    """Found by running the real cross-box door on two boxes' receipts.

    When the candidate is a pure rescaling of the reference, every phase
    metric is analytically zero and comes out as float cancellation --
    2.4e-14 degrees on one box against 2.5e-14 on the other.  Judged against
    each other that is a 3.6% disagreement; judged against the 180 degrees
    the metric can span, it is 1.4e-16, which is what it is.
    """

    assert spectral_compare.BOUNDED_METRIC_SCALE[
        "weighted_mean_absolute_phase_error_degrees"] == 180.0
    reference = wave(128, 6) + 0.3 * wave(128, 20, axis="y")
    left = spectral_compare.compare_scalar(
        1.05 * reference, reference, dx_m=DX, bands=BANDS)
    right = json.loads(json.dumps(left))
    for band in right["bands"]:
        if band["status"] != "ok":
            continue
        # Both are analytic zeros; the ratio between them is meaningless.
        assert abs(band["weighted_mean_absolute_phase_error_degrees"]) < 1e-10
        band["weighted_mean_absolute_phase_error_degrees"] *= 1.5
    assert spectral_compare.cross_box_differences(left, right) == []


def test_two_results_inside_the_declared_tolerance_compare_equal():
    reference = wave(128, 6) + 0.3 * wave(128, 20, axis="y")
    candidate = 1.05 * reference + 0.1 * wave(128, 40)
    left = spectral_compare.compare_scalar(
        candidate, reference, dx_m=DX, bands=BANDS)
    right = json.loads(json.dumps(left))
    assert spectral_compare.cross_box_differences(left, right) == []
    # A perturbation one decade inside the tolerance is still agreement.
    for band in right["bands"]:
        if band["status"] == "ok":
            band["power_ratio"] *= 1.0 + 1e-13
    assert spectral_compare.cross_box_differences(left, right) == []


def test_a_difference_above_the_tolerance_is_named_not_absorbed():
    reference = wave(128, 6) + 0.3 * wave(128, 20, axis="y")
    candidate = 1.05 * reference
    left = spectral_compare.compare_scalar(
        candidate, reference, dx_m=DX, bands=BANDS)
    right = json.loads(json.dumps(left))
    target = next(band for band in right["bands"] if band["status"] == "ok")
    target["power_ratio"] *= 1.0 + 1e-9
    differences = spectral_compare.cross_box_differences(left, right)
    assert [item["metric"] for item in differences] == ["power_ratio"]
    assert differences[0]["band"] == target["name"]
    assert differences[0]["difference"] > spectral_compare.CROSS_BOX_TOLERANCE


def test_mode_count_disagreement_is_a_defect_not_a_tolerance():
    reference = wave(128, 6)
    left = spectral_compare.compare_scalar(
        1.02 * reference, reference, dx_m=DX, bands=BANDS)
    right = json.loads(json.dumps(left))
    target = next(band for band in right["bands"] if band["status"] == "ok")
    target["mode_count"] = int(target["mode_count"]) + 1
    differences = spectral_compare.cross_box_differences(left, right)
    assert [item["metric"] for item in differences] == ["mode_count"]
    assert differences[0]["exact"] is True


# ---------------------------------------------------------------------------
# 4b. The Helmholtz leakage note, as a number in the receipt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cycles,tolerated", [(4, 0.35), (8, 0.35), (16, 0.35)])
def test_the_declared_leakage_model_bounds_the_measured_leakage(
        cycles, tolerated):
    """The taper moves a pure mode; the model says how much, and it is right.

    Measured 2026-08-20 over 38 cases (n=64..256, 2..32 wavelengths across
    the window, crop 0 and 8): with no crop the fitted coefficient rises
    from 0.314 at m=2 to 0.340 at m=32, and rotational and divergent leak by
    the same amount to every printed digit.
    """

    n = 192
    parallel = wave(n, cycles)
    zero = np.zeros_like(parallel)
    divergent = spectral_compare.compare_vector(
        parallel, zero, parallel, zero, dx_m=DX,
        bands=(spectral_compare.WavelengthBand("all"),))
    rotational = spectral_compare.compare_vector(
        zero, parallel, zero, parallel, dx_m=DX,
        bands=(spectral_compare.WavelengthBand("all"),))
    divergent_leak = 1.0 - divergent["bands"][0][
        "left_divergent_energy_fraction"]
    rotational_leak = rotational["bands"][0]["left_divergent_energy_fraction"]
    # The two partitions leak by the same amount.
    assert divergent_leak == pytest.approx(rotational_leak, rel=1e-6)
    estimate = spectral_compare.HELMHOLTZ_LEAKAGE_COEFFICIENT / (cycles ** 2)
    assert divergent_leak <= estimate
    assert divergent_leak >= tolerated * estimate


def test_every_vector_band_carries_the_leakage_it_is_exposed_to():
    n = 192
    result = spectral_compare.compare_vector(
        wave(n, 8), wave(n, 8, axis="y"), wave(n, 8),
        wave(n, 8, axis="y"), dx_m=DX, bands=BANDS)
    assert result["helmholtz_leakage"]["coefficient"] == 0.34
    assert result["helmholtz_leakage"]["scored_window_m"] == n * DX
    for band in result["bands"]:
        across = band["wavelengths_across_scored_window"]
        estimate = band["helmholtz_leakage_estimate"]
        if across is None:
            assert estimate is None
            continue
        if across < 1.0:
            # A wavelength longer than the window has no partition to trust.
            assert estimate is None
            continue
        assert estimate == pytest.approx(0.34 / (across * across))
    convective = next(band for band in result["bands"]
                      if band["name"] == "convective")
    # The 2-8 km band on a 192 km window: at least 24 wavelengths across it,
    # so the taper is worth well under a tenth of a percent there.
    assert convective["wavelengths_across_scored_window"] >= 24.0
    assert convective["helmholtz_leakage_estimate"] < 6e-4


# ---------------------------------------------------------------------------
# 5. Receipt threading: code identity and the declared tolerance
# ---------------------------------------------------------------------------


def _score(tmp_path: Path) -> dict:
    np.savez(tmp_path / "left.npz", W=1.02 * wave(64, 4))
    np.savez(tmp_path / "reference.npz", W=wave(64, 4))
    spec = tmp_path / "spec.toml"
    spec.write_text(_SPEC_TEMPLATE.format(gate=""), encoding="utf-8")
    return spectral_receipt.score_registration(
        spectral_receipt.make_registration(spec))


def test_the_receipt_binds_the_evaluator_commit_through_the_capsule_builder(
        tmp_path):
    """A receipt binds arithmetic and input bytes; it did not bind the code.

    The certification capsule already resolves that identity for every
    emitting route.  Building a second one here would let a capsule and a
    receipt from one run disagree about which commit produced the numbers,
    so the receipt calls the capsule's own builder.
    """

    receipt = _score(tmp_path)
    code = receipt["code"]
    assert set(code) >= {"gpuwm_version", "git_commit"}
    from woof.certify import capsule

    assert code == capsule.code_identity()


def test_the_receipt_declares_the_tolerance_its_numbers_reproduce_under(
        tmp_path):
    receipt = _score(tmp_path)
    declared = receipt["reproducibility"]
    assert declared["rule"] == "gpuwm-spectral-cross-box-v1"
    assert declared["tolerance"] == 1e-12
    assert declared["receipt_sha256_is_portable"] is False
    assert spectral_receipt.validate_receipt(receipt)["receipt_sha256"] == (
        receipt["receipt_sha256"])


def test_a_receipt_whose_declared_rule_was_edited_is_refused(tmp_path):
    receipt = _score(tmp_path)
    tampered = json.loads(json.dumps(receipt))
    tampered["reproducibility"]["tolerance"] = 1e-3
    with pytest.raises(ValueError, match="receipt hash"):
        spectral_receipt.validate_receipt(tampered)


def test_the_registration_policy_hash_survives_a_move_between_boxes(tmp_path):
    """The raw registration hash binds resolved paths; the policy one does not.

    Found against the artifact: the same source TOML registered on the
    Windows desktop and on test host 1 produced registration hashes
    31607149b0ed and a419270a4e9c, because one spells the inputs
    ``C:\\Users\\<user>\\...`` and the other ``/home/<user>/...``.  A
    cross-box comparison keyed on that hash refuses every real pair.
    """

    here = tmp_path / "here"
    there = tmp_path / "there"
    for directory in (here, there):
        directory.mkdir()
        np.savez(directory / "left.npz", W=1.02 * wave(64, 4))
        np.savez(directory / "reference.npz", W=wave(64, 4))
        (directory / "spec.toml").write_text(
            _SPEC_TEMPLATE.format(gate=""), encoding="utf-8")
    one = spectral_receipt.make_registration(here / "spec.toml")
    other = spectral_receipt.make_registration(there / "spec.toml")
    assert one["registration_sha256"] != other["registration_sha256"]
    assert (one["registration_policy_sha256"]
            == other["registration_policy_sha256"])


def test_the_cross_box_door_agrees_on_two_scorings_and_names_a_real_move(
        tmp_path):
    from woof.verify import spectral_cli

    np.savez(tmp_path / "left.npz", W=1.02 * wave(64, 4))
    np.savez(tmp_path / "reference.npz", W=wave(64, 4))
    spec = tmp_path / "spec.toml"
    spec.write_text(_SPEC_TEMPLATE.format(gate=""), encoding="utf-8")
    first = spectral_receipt.score_registration(
        spectral_receipt.make_registration(spec))
    spectral_receipt.write_json_atomic(tmp_path / "a.json", first)
    spectral_receipt.write_json_atomic(tmp_path / "b.json", first)
    import argparse

    arguments = argparse.Namespace(
        receipt=tmp_path / "a.json", other=tmp_path / "b.json",
        tolerance=None)
    assert spectral_cli._cross_box(arguments) == 0

    moved = json.loads(json.dumps(first))
    target = next(band for band in moved["comparisons"][0]["result"]["bands"]
                  if band["status"] == "ok")
    target["power_ratio"] *= 1.0 + 1e-9
    moved.pop("receipt_sha256")
    moved["receipt_sha256"] = spectral_receipt.canonical_hash(moved)
    spectral_receipt.write_json_atomic(tmp_path / "c.json", moved)
    arguments.other = tmp_path / "c.json"
    assert spectral_cli._cross_box(arguments) == 1


def test_the_cross_box_door_refuses_two_different_campaigns(tmp_path):
    from woof.verify import spectral_cli
    import argparse

    receipts = []
    for index, maximum in enumerate((32.0, 30.0)):
        directory = tmp_path / f"campaign{index}"
        directory.mkdir()
        np.savez(directory / "left.npz", W=1.02 * wave(64, 4))
        np.savez(directory / "reference.npz", W=wave(64, 4))
        spec = directory / "spec.toml"
        spec.write_text(
            _SPEC_TEMPLATE.format(gate="").replace(
                "maximum_km = 32.0", f"maximum_km = {maximum}"),
            encoding="utf-8")
        receipt = spectral_receipt.score_registration(
            spectral_receipt.make_registration(spec))
        path = directory / "receipt.json"
        spectral_receipt.write_json_atomic(path, receipt)
        receipts.append(path)
    arguments = argparse.Namespace(
        receipt=receipts[0], other=receipts[1], tolerance=None)
    with pytest.raises(ValueError) as error:
        spectral_cli._cross_box(arguments)
    message = str(error.value)
    assert "different campaign policy" in message
    assert "ONE source" in message


# ---------------------------------------------------------------------------
# 6. The NaN ceiling refuses by name, with the remedy
# ---------------------------------------------------------------------------


def test_the_nan_ceiling_names_the_count_the_place_and_the_remedy(tmp_path):
    plane = wave(64, 4).copy()
    plane[10, 10] = math.nan
    plane[11, 12] = math.inf
    source = tmp_path / "field.npz"
    np.savez(source, W=plane)
    with pytest.raises(ValueError) as error:
        spectral_io.load_array(source, variable="W")
    message = str(error.value)
    assert "2 of 4096" in message
    assert "(10, 10)" in message
    assert "every retained Fourier mode" in message
    assert "crop_cells" in message
    assert "nan=1" in message and "inf=1" in message


def test_an_empty_array_refuses_as_an_empty_array_not_as_a_nan(tmp_path):
    source = tmp_path / "empty.npz"
    np.savez(source, W=np.zeros((0, 8), dtype=np.float64))
    with pytest.raises(ValueError, match="carries no cells"):
        spectral_io.load_array(source, variable="W")


def test_a_masked_source_says_which_cells_were_masked(tmp_path):
    values = np.ma.masked_array(
        wave(16, 2), mask=np.zeros((16, 16), dtype=bool))
    values.mask[3, 4] = True
    with pytest.raises(ValueError) as error:
        spectral_io._finite_array(values, label="unit")
    message = str(error.value)
    assert "1 of 256" in message
    assert "(3, 4)" in message


def test_the_real_front_door_refuses_a_nan_plane_without_a_traceback(tmp_path):
    """Verified against the artifact: the CLI, not the module."""

    import subprocess
    import sys

    plane = wave(64, 4).copy()
    plane[5, 6] = math.nan
    np.savez(tmp_path / "left.npz", W=plane)
    np.savez(tmp_path / "reference.npz", W=wave(64, 4))
    spec = tmp_path / "spec.toml"
    spec.write_text(_SPEC_TEMPLATE.format(gate=""), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "woof.cli", "spectral", "run", str(spec),
         "--registration", str(tmp_path / "registration.json"),
         "--receipt", str(tmp_path / "receipt.json")],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "1 of 4096" in completed.stderr
    assert "every retained Fourier mode" in completed.stderr


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

_SPEC_TEMPLATE = '''schema = "gpuwm.spectral-comparison-source/v1"
title = "pin backlog control"
default_dx_m = 1000.0

[[bands]]
name = "storm"
minimum_km = 8.0
maximum_km = 32.0

[[pairs]]
name = "p1"
left_path = "left.npz"
reference_path = "reference.npz"

[[fields]]
name = "w"
kind = "scalar"
variable = "W"
reduction = "plane"
{gate}'''

_CORRELATION_GATE = '''
[[gates]]
id = "w-correlation"
field = "w"
component = "scalar"
band = "storm"
metric = "spectral_correlation"
minimum = 0.95
'''

_PARTITION_REFERENCE_POWER_GATE = '''
[[gates]]
id = "bad"
field = "w"
component = "partition"
band = "storm"
metric = "reference_power"
minimum = 1.0
'''

_TWO_PAIR_SPEC = '''schema = "gpuwm.spectral-comparison-source/v1"
title = "wildcard gate control"
default_dx_m = 1000.0

[[bands]]
name = "storm"
minimum_km = 8.0
maximum_km = 32.0

[[pairs]]
name = "pair-a"
left_path = "a/left.npz"
reference_path = "a/reference.npz"

[[pairs]]
name = "pair-b"
left_path = "b/left.npz"
reference_path = "b/reference.npz"

[[fields]]
name = "w"
kind = "scalar"
variable = "W"
reduction = "plane"

[[gates]]
id = "w-power"
pair = "*"
field = "w"
component = "scalar"
band = "storm"
metric = "power_ratio"
minimum = 0.5
maximum = 2.0
'''
