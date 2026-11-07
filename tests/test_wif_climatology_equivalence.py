"""The Rust WIF ingest path against the NumPy oracle-of-record.

WHAT THIS GATE PREVENTS (gate law): the data path silently drifting off
the numbers commit 94260bf44 bound against WRF-4.7.1 real.exe on test host 4.
The NumPy expressions in ``woof/ingest/wif_climatology.py`` are the ones
that were measured there; the Rust engine is what a run actually uses.
Nothing else compares the two, so without this test a change to the
bridge -- or a rebuild against a different library -- could move QNWFA on
every mp=28 climatology run with no test saying so.

The tolerances below are NOT taste.  Decode, horizontal and the surface
emission are held BIT-EXACT, because nothing in either implementation
rounds differently there and a single differing bit means a real
divergence.  Only the vertical stage is allowed a residual, and its bound
is the oracle's own margin against real.exe: the Rust route evaluates
WRF's linear window as ``lagrange_setup`` writes it, the reference
evaluates the algebraically identical ``y0 + w*(y1-y0)``, and the two
disagree at the last bits of an FP32 log-p interpolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.ingest import wif_climatology as wif

pytest.importorskip("numpy")

_equiv = pytest.importorskip(
    "tools.wif_climatology_equivalence",
    reason="the equivalence harness ships beside the module it gates")


def _bridge_or_skip():
    try:
        wif._cpu_backend()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"CPU preprocessing bridge unavailable: {exc}")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    _bridge_or_skip()
    path = tmp_path_factory.mktemp("wif") / "equivalence.dat"
    # Small but structurally identical to the shipped dataset: global
    # cylindrical-equidistant, 12 months x N levels x three fields, and a
    # longitude axis that spans exactly 360 degrees so the seam is live.
    _equiv.write_synthetic_dataset(path, 31, 48, 8, noise=0.0)
    return path


def test_the_two_decoders_agree_bit_for_bit(dataset):
    rust = wif.load_wif_climatology(dataset, backend="rust")
    reference = wif.load_wif_climatology(dataset, backend="numpy")
    for name in ("qnwfa", "qnifa", "pressure", "latitude", "longitude"):
        assert np.array_equal(getattr(rust, name), getattr(reference, name)), (
            f"the Rust and NumPy decoders disagree on {name}; a decode "
            "difference makes every later stage's agreement meaningless")


def test_the_horizontal_and_surface_stages_are_bit_exact(dataset):
    rust = wif.load_wif_climatology(dataset, backend="rust")
    reference = wif.load_wif_climatology(dataset, backend="numpy")
    lat2d, lon2d, pb, phb = _equiv.model_grid(9, 9, 12)
    rust_fields, rust_receipt = wif.wif_fields_for_grid(
        rust, lat2d, lon2d, "2026-08-25_12:00:00", pb, phb, backend="rust")
    ref_fields, ref_receipt = wif.wif_fields_for_grid(
        reference, lat2d, lon2d, "2026-08-25_12:00:00", pb, phb,
        backend="numpy")
    assert rust_receipt["engine"] == "rust"
    assert ref_receipt["engine"] == "numpy"
    # nwfa2d is horizontal + temporal + one elementwise scaling and NO
    # vertical interpolation, so it isolates those stages exactly.
    for name in ("nwfa2d", "nifa2d"):
        assert np.array_equal(rust_fields[name], ref_fields[name]), (
            f"{name} is produced without any vertical interpolation, so a "
            "difference here is a horizontal or temporal divergence, not "
            "an FP32 window-form residual")


def test_the_vertical_stage_holds_the_oracle_margin(dataset):
    rust = wif.load_wif_climatology(dataset, backend="rust")
    reference = wif.load_wif_climatology(dataset, backend="numpy")
    lat2d, lon2d, pb, phb = _equiv.model_grid(9, 9, 12)
    rust_fields, _ = wif.wif_fields_for_grid(
        rust, lat2d, lon2d, "2026-08-25_12:00:00", pb, phb, backend="rust")
    ref_fields, _ = wif.wif_fields_for_grid(
        reference, lat2d, lon2d, "2026-08-25_12:00:00", pb, phb,
        backend="numpy")
    # 1.032e-05 is the reference's OWN pointwise relative margin against
    # real.exe for QNWFA (commit 94260bf44).  The engine swap is allowed
    # to sit inside that margin and nowhere near outside it.
    oracle_margin = 1.032e-05
    for name in ("nwfa", "nifa"):
        got = np.asarray(rust_fields[name], dtype=np.float64)
        want = np.asarray(ref_fields[name], dtype=np.float64)
        scale = np.maximum(np.abs(want), np.finfo(np.float32).tiny)
        relative = float(np.max(np.abs(got - want) / scale))
        assert relative <= oracle_margin, (
            f"{name} moved {relative:.3e} between the Rust engine and the "
            f"oracle-of-record, past the {oracle_margin:.3e} margin the "
            "reference itself holds against WRF real.exe; the engine swap "
            "is no longer inside the ported numbers")


def test_the_seam_is_the_thing_the_new_operator_exists_for(dataset):
    """The dateline column is what the pre-existing bilinear got wrong.

    ``interpolate_regular`` clamps its last column, so a target between
    the last and first source columns takes column ``nx-1`` twice instead
    of wrapping to column 0.  On a global source that is a wrong number,
    not a rounding difference, and this asserts the new operator does not
    reproduce it.
    """

    _bridge_or_skip()
    backend = wif._cpu_backend()
    nlat, nlon = 5, 8
    deltalon = 360.0 / nlon
    field = np.arange(nlat * nlon, dtype=np.float32).reshape(nlat, nlon)
    # A point three quarters of the way from the LAST column to the first.
    target_lon = np.array([[-180.0 + deltalon * (nlon - 1 + 0.75)]])
    target_lat = np.array([[0.0]])
    got = backend.interpolate_regular_cyclic(
        field, target_lat, target_lon,
        startlat=-90.0, deltalat=180.0 / (nlat - 1),
        startlon=-180.0, deltalon=deltalon)
    row = nlat // 2
    expected = np.float32(0.25 * field[row, nlon - 1] + 0.75 * field[row, 0])
    assert got.shape == (1, 1)
    assert np.float32(got[0, 0]) == expected, (
        "the cyclic operator did not wrap at the seam; it read the last "
        "column twice, which is the bounded operator's behaviour and the "
        "reason this entry exists")


def test_a_missing_bridge_refuses_instead_of_running_the_oracle():
    """No silent demotion to NumPy.

    The receipt names the Rust engine.  A run that quietly fell back to
    the NumPy reference would publish that receipt over different bits,
    which is exactly the drift the 2.5.0 data-path law forbids.
    """

    with pytest.raises(wif.WifClimatologyError) as caught:
        wif._resolve_backend("fallback")
    assert "rust" in str(caught.value)


def test_the_receipt_names_which_engine_produced_the_fields(dataset):
    rust = wif.load_wif_climatology(dataset, backend="rust")
    lat2d, lon2d, pb, phb = _equiv.model_grid(5, 5, 8)
    _, receipt = wif.wif_fields_for_grid(
        rust, lat2d, lon2d, "2026-08-25_12:00:00", pb, phb, backend="rust")
    assert receipt["schema"] == "wrf-v4.7.1-wif-climatology-ingest-v2"
    assert receipt["engine"] == "rust"
    for symbol in ("gpuwm_wps_intermediate_read",
                   "gpuwm_regular_cyclic_bilinear_f32",
                   "gpuwm_wrf_vert_interp_f32"):
        assert symbol in receipt["engine_detail"], (
            "the receipt must name the entries that produced the numbers; "
            "a receipt that only says 'rust' cannot be audited")
