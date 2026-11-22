"""The WOOF-ArWen moist LES case, validated against the banked WRF oracle.

CPU-only, and deliberately so: every claim here is checkable before any
WOOF-ArWen moist run exists, which is the order the P1 instrument-accuracy rule
requires (``tools/wrf_em_les_oracle/INSTRUMENT-HISTORY.md``).

Three things are pinned.

**The initial condition.** The case rebuilds the ratified capped-moist
``input_sounding`` from committed constants; its sha256 must be
``993fedb1...``, the asset the oracle campaign actually ran, and the
capped-dry anchor and the base dry asset must come out of the same
constructor at their own committed hashes.  A case that initialises from
a re-derivation of the reference sounding is not running the reference
case.

**The reduction.** ``cloud_topped_boundary_layer.reduce_moist_profiles``
is a transcription of the oracle's ``same_instrument_moist.reduce_moist``.
Transcriptions rot.  These tests run both routines on the four banked
oracle fixtures and require *bit* equality on every scalar, and separately
require the transcription to reproduce the numbers the oracle published in
its own receipt JSONs.  The essential case is ``zi_thetav_load_m``:
one name, two heights (cloud base in a cloudy run, the inversion in a
clear one), so a drift here would silently compare two different
quantities.

**The npz contract.** The writer must emit every array the reducer
requires, at the documented shapes, with the SGS carrier present only
where it exists.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from woof.verify.cases import cloud_topped_boundary_layer as ctbl

_ROOT = Path(__file__).resolve().parents[1]
_ORACLE = _ROOT / "tools" / "wrf_em_les_oracle"
_FIXTURES = (_ROOT / "docs" / "internal" / "receipts" / "les"
             / "moist-2026-08-04")

#: The banked capped-family draws: the three capped-moist draws of the
#: reference (8 K cap, qv 14 / 5.6) family and the capped-dry anchor, which
#: runs the same namelist over a sounding with qv identically zero.  The
#: anchor is in the list because it is the only fixture that exercises the
#: OTHER reading of ``zi_thetav_load_m``.
_BANKED = ("capped_moist_km3_100m_r1", "capped_moist_km3_100m_r2",
           "capped_moist_km3_100m_r3", "capped_dry_km3_100m_r1")

#: The oracle receipts live under a release-excluded path; a published
#: clone does not carry them.  Same convention as
#: ``tests/test_les_receipt_fields.py``.
_HAVE_FIXTURES = all((_FIXTURES / f"{n}_moist_profiles.npz").exists()
                     and (_FIXTURES / f"{n}.json").exists()
                     for n in _BANKED)
oracle_fixtures = pytest.mark.skipif(
    not _HAVE_FIXTURES,
    reason="the banked WRF moist-oracle fixtures live under a "
           "release-excluded path and are absent from a published clone")

_HAVE_ORACLE_TOOL = (_ORACLE / "same_instrument_moist.py").exists()
oracle_tool = pytest.mark.skipif(
    not _HAVE_ORACLE_TOOL,
    reason="the oracle recipe is not present in this tree")

#: The oracle instrument's 30-minute default window, in seconds.
_WINDOW_S = 30.0 * 60.0


def _load_oracle_reducer():
    """Import ``same_instrument_moist`` by path.

    By path rather than by package import because ``tools/`` is not
    importable and must not become so: the oracle side has to stay able to
    disagree with the engine it scores, so nothing in ``woof`` may depend
    on it.  Only this test, which exists to compare the two, reaches
    across.
    """
    spec = importlib.util.spec_from_file_location(
        "_oracle_same_instrument_moist",
        _ORACLE / "same_instrument_moist.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_arrays(name):
    d = np.load(_FIXTURES / f"{name}_moist_profiles.npz")
    return dict(
        z_mass=d["z_mass"], t_seconds=d["t_seconds"],
        wthv_res_t=d["wthv_res"], wthv_sgs_t=d["wthv_sgs"],
        wqv_res_t=d["wqv_res"], wqv_sgs_t=d["wqv_sgs"],
        qv_t=d["qv"], qc_t=d["qc"], qr_t=d["qr"],
        cloud_t=d["cloud_frac"], sat_t=d["sat_frac"],
        n2_t=d["n2_moist_frac"], lwp_t=d["lwp"])


# --------------------------------------------------------------------- IC

def test_capped_moist_sounding_is_the_ratified_asset():
    """The reference asset, byte for byte.

    ``993fedb1...`` is what ``capped_family_assets.sha256`` records and
    what every capped-moist run's provenance block names.
    """
    text = ctbl.sounding_text(capped=True, moist=True)
    assert hashlib.sha256(text.encode()).hexdigest() == \
        ctbl.SOUNDING_SHA256_CAPPED_MOIST
    assert ctbl.SOUNDING_SHA256_CAPPED_MOIST == (
        "993fedb14ab5d57684535342c7d66507437a5e792186a79c8d565a825da0fd17")


def test_capped_dry_anchor_and_base_asset_reproduce():
    """The other two members of the same constructor.

    The capped-dry anchor is the control the oracle ran to separate the
    vapour column from everything else; the base asset is the dry matched
    family's own sounding, which is also the file form of the dry case's
    ``_theta_profile``.  All three from one code path means the WOOF-ArWen case
    cannot be running a different theta profile from the anchor it will be
    compared against.
    """
    dry = ctbl.sounding_text(capped=True, moist=False)
    base = ctbl.sounding_text(capped=False, moist=False)
    assert hashlib.sha256(dry.encode()).hexdigest() == \
        ctbl.SOUNDING_SHA256_CAPPED_DRY
    assert hashlib.sha256(base.encode()).hexdigest() == \
        ctbl.SOUNDING_SHA256_BASE
    committed = _ORACLE / "input_sounding.arwen_cbl"
    if committed.exists():
        assert base == committed.read_text()


def test_capped_arms_differ_only_in_the_vapour_column():
    """"The two arms differ in the vapour column and in nothing else" is
    the property that makes the capped-dry run a control rather than a
    second experiment.  Checked on the columns, not asserted."""
    moist = [ln.split() for ln
             in ctbl.sounding_text(capped=True, moist=True).splitlines()]
    dry = [ln.split() for ln
           in ctbl.sounding_text(capped=True, moist=False).splitlines()]
    assert len(moist) == len(dry) == 121
    # surface line: pressure and theta identical, qv differs
    assert moist[0][:2] == dry[0][:2]
    assert moist[0][2] != dry[0][2]
    for a, b in zip(moist[1:], dry[1:]):
        assert a[0] == b[0] and a[1] == b[1]      # height, theta
        assert a[3] == b[3] and a[4] == b[4]      # u, v
        assert a[2] != b[2]                       # qv


def test_realised_inversion_jump_is_the_receipted_8_45_K():
    """The requested 8.00 K cap lands as 8.45 K on the sounding's own
    levels -- the 8 K plus 0.45 K of the underlying 3 K/km lapse across
    the 150 m ramp (MOIST-CASE-REFERENCE-SETTINGS.md §6.3).  The receipts
    quote the realised number, so the case must produce it."""
    z, theta, _ = ctbl.sounding_profiles(capped=True, moist=True)
    below = theta[z <= ctbl.INVERSION_BASE_M][-1]
    above = theta[z >= ctbl.INVERSION_BASE_M + ctbl.INVERSION_DEPTH_M][0]
    assert below == pytest.approx(301.50, abs=5e-3)
    assert above == pytest.approx(309.95, abs=5e-3)
    assert above - below == pytest.approx(8.45, abs=1e-2)


def test_vapour_step_colocates_with_the_thermal_cap():
    """``--transition-m 1500`` is passed explicitly on the oracle side
    because the auto-detector would find the sounding's FIRST theta
    departure, still 1025 m in the capped profile.  The step must sit at
    the cap."""
    z, _, qv = ctbl.sounding_profiles(capped=True, moist=True)
    assert qv[z <= 1500.0][-1] == pytest.approx(14.0e-3)
    assert qv[z > 1500.0][0] == pytest.approx(5.6e-3)
    assert float(qv[z <= 1025.0].min()) == pytest.approx(14.0e-3)


# -------------------------------------------------------------- reduction

@oracle_fixtures
@oracle_tool
@pytest.mark.parametrize("name", _BANKED)
def test_reduction_is_bit_identical_to_the_oracle_routine(name):
    """One routine, applied to both sides -- proven, not asserted.

    Both implementations are run on the same banked arrays and every
    scalar must agree to the bit.  ``==`` and not ``approx``: a
    transcription that has drifted by an ULP has drifted.
    """
    oracle = _load_oracle_reducer()
    args = _fixture_arrays(name)
    mine = ctbl.reduce_moist_profiles(window_s=_WINDOW_S, **args)
    theirs = oracle.reduce_moist(window_s=_WINDOW_S, **args)
    assert set(mine) == set(theirs)
    for key in sorted(k for k in mine if k != "_profiles"):
        a, b = mine[key], theirs[key]
        # The dry anchor's resolved-fraction denominators are exactly zero
        # -- no vapour flux exists to take a share of -- so both routines
        # return NaN there.  Two NaNs from one expression is agreement;
        # `==` alone would call it a difference.
        if isinstance(a, float) and np.isnan(a):
            assert isinstance(b, float) and np.isnan(b), key
            continue
        assert a == b, key
    for key in sorted(mine["_profiles"]):
        np.testing.assert_array_equal(
            mine["_profiles"][key], theirs["_profiles"][key], err_msg=key)


@oracle_fixtures
@pytest.mark.parametrize("name", _BANKED)
def test_reduction_reproduces_the_published_oracle_numbers(name):
    """The transcription, run on the oracle's own arrays, must reproduce
    the numbers the oracle published in its own receipt -- exactly.

    This is the validation that has to pass *before* a WOOF-ArWen run exists:
    it separates "the two engines disagree" from "the two reductions
    disagree", and it can only be done while one side's arrays are the
    only arrays there are.
    """
    published = json.loads((_FIXTURES / f"{name}.json").read_text())
    R = ctbl.reduce_moist_profiles(window_s=_WINDOW_S,
                                   **_fixture_arrays(name))
    assert R["n_samples"] == published["n_frames_in_window"]
    for mine, theirs in (("zi_thetav_m", "zi_thetav_load_m"),
                         ("wthv_res_max", "wthv_res_max"),
                         ("wthv_total_min", "wthv_total_min"),
                         ("wqv_res_max", "wqv_res_max"),
                         ("qv_surface", "qv_surface"),
                         ("qc_profile_max", "qc_max_profile"),
                         ("cloud_fraction_max", "cloud_fraction_max"),
                         ("cloud_base_m", "cloud_base_m"),
                         ("cloud_top_m", "cloud_top_m"),
                         ("sat_fraction_max", "sat_fraction_max"),
                         ("lwp_kg_m2", "lwp_kg_m2")):
        assert R[mine] == published[theirs], mine
    branch = published["saturated_branch"]
    assert R["n2_moist_fraction_max"] == branch["max_level_fraction_window"]
    assert R["n2_engaged_somewhere"] == branch["engaged_somewhere"]
    assert R["n2_engaged_everywhere"] == branch["engaged_everywhere"]


@oracle_fixtures
def test_zi_is_cloud_base_in_the_moist_run_and_the_inversion_in_the_dry():
    """ONE NAME, TWO HEIGHTS, measured rather than restated.

    The two fixtures differ only in their vapour column and run the same
    namelist, so this is the cleanest possible demonstration that the
    metric changes meaning with the regime and not with the model.
    """
    moist = ctbl.reduce_moist_profiles(
        window_s=_WINDOW_S, **_fixture_arrays("capped_moist_km3_100m_r1"))
    dry = ctbl.reduce_moist_profiles(
        window_s=_WINDOW_S, **_fixture_arrays("capped_dry_km3_100m_r1"))

    # cloudy: the buoyancy-flux minimum IS cloud base, to every digit
    assert moist["cloud_base_m"] is not None
    assert moist["zi_thetav_m"] == moist["cloud_base_m"]
    assert moist["zi_thetav_m"] == pytest.approx(1274.4514, abs=1e-3)

    # clear: no cloud at all, so the same metric is the inversion, which
    # the 1500 m cap base puts at 1526.3 m
    assert dry["cloud_base_m"] is None
    assert dry["cloud_fraction_max"] == 0.0
    assert dry["zi_thetav_m"] == pytest.approx(1526.3284, abs=1e-3)

    # and therefore the naive dry-vs-moist difference is 252 m of NOTHING
    assert dry["zi_thetav_m"] - moist["zi_thetav_m"] == \
        pytest.approx(251.877, abs=1e-2)


@oracle_fixtures
def test_the_reference_family_engages_the_saturated_branch_partially():
    """The clamp-coverage discipline the moist instrument exists to
    exercise: engaged somewhere and NOT engaged everywhere.  An instrument
    that reports either extreme has not demonstrated it can see the
    switch, and a case that produces either extreme cannot qualify it."""
    for name in _BANKED[:3]:
        R = ctbl.reduce_moist_profiles(window_s=_WINDOW_S,
                                       **_fixture_arrays(name))
        assert R["n2_engaged_somewhere"] is True
        assert R["n2_engaged_everywhere"] is False
        assert R["cloud_top_m"] > R["cloud_base_m"]


# ------------------------------------------------------------ npz contract

def _synthetic_samples(nt: int, nz: int, *, km_opt: int):
    """Frames shaped like ``_moist_slab_profiles`` output, with no device.

    Values are arbitrary; only the shapes and the key set are under test.
    """
    rng = np.random.default_rng(0)
    out = []
    for it in range(nt):
        s = {"t_seconds": 60.0 * (it + 1)}
        for k in ("wthv_res_load", "wthv_res_novload", "wqv_res", "qv",
                  "qc", "qr", "theta", "thetav", "cloud_frac", "sat_frac",
                  "n2_moist_frac", "e_sgs"):
            s[k] = rng.standard_normal(nz)
        for k in ("wthv_sgs_load", "wthv_sgs_novload", "wqv_sgs"):
            s[k] = rng.standard_normal(nz + 1)
        for k in ("lwp", "rwp", "rainnc"):
            s[k] = float(rng.standard_normal())
        out.append(s)
    return out


@pytest.mark.parametrize("km_opt", (2, 3))
def test_npz_carries_every_required_array_at_the_contract_shapes(km_opt):
    nt, nz = 7, 11
    arrays = ctbl.moist_npz_arrays(
        _synthetic_samples(nt, nz, km_opt=km_opt),
        z_mass=np.arange(nz, dtype=float),
        z_w=np.arange(nz + 1, dtype=float), km_opt=km_opt)
    for name in ctbl.NPZ_REQUIRED:
        assert name in arrays, name
    assert arrays["z_mass"].shape == (nz,)
    assert arrays["t_seconds"].shape == (nt,)
    for name in ("wthv_res", "wqv_res", "qv", "qc", "qr", "theta",
                 "thetav", "cloud_frac", "sat_frac", "n2_moist_frac"):
        assert arrays[name].shape == (nt, nz), name
    for name in ("wthv_sgs", "wqv_sgs"):
        assert arrays[name].shape == (nt, nz + 1), name
    assert arrays["lwp"].shape == (nt,)
    for name in arrays:
        assert arrays[name].dtype == np.float64, name


def test_sgs_tke_carrier_is_written_only_where_it_exists():
    """An all-zero ``e_sgs`` is read by the reducer as an ABSENT carrier,
    not a zero one.  km_opt=3 has no prognostic SGS TKE, so writing the
    zeros would be writing a carrier that does not exist; the key is
    omitted instead, which is the unambiguous statement."""
    kwargs = dict(z_mass=np.arange(4.0), z_w=np.arange(5.0))
    assert "e_sgs" not in ctbl.moist_npz_arrays(
        _synthetic_samples(3, 4, km_opt=3), km_opt=3, **kwargs)
    assert "e_sgs" in ctbl.moist_npz_arrays(
        _synthetic_samples(3, 4, km_opt=2), km_opt=2, **kwargs)


@oracle_fixtures
@oracle_tool
def test_the_oracle_reducer_accepts_this_writer_s_key_set():
    """The reducer's own ``REQUIRED`` tuple is the contract; read it from
    the oracle file rather than restating it here, so the two cannot drift
    apart without this failing."""
    oracle = _load_oracle_reducer()
    arrays = ctbl.moist_npz_arrays(
        _synthetic_samples(3, 4, km_opt=2),
        z_mass=np.arange(4.0), z_w=np.arange(5.0), km_opt=2)
    assert set(oracle.REQUIRED) <= set(arrays)
    assert tuple(oracle.REQUIRED) == ctbl.NPZ_REQUIRED


# ----------------------------------------------------------------- SGS arm

def _sgs_flux_reference(field, khv, fnm, fnp, phi, nz):
    """The dry case's SGS loop, transcribed literally.

    ``convective_boundary_layer._slab_profiles`` :316-320, unrolled with no
    slicing at all.  It is the authority the vectorized version must equal:
    the moist arm's whole SGS half rides on this index algebra, and an
    off-by-one in it produces a plausible profile rather than an error.
    """
    out = np.zeros(nz + 1)
    for kw in range(1, nz):
        rdz = 2.0 * 9.81 / (phi[kw + 1] - phi[kw - 1])
        k_w = fnm[kw] * khv[kw] + fnp[kw] * khv[kw - 1]
        flux = -k_w * (field[kw] - field[kw - 1]) * rdz
        out[kw] = float(flux.mean())
    return out


def test_sgs_flux_matches_the_dry_case_loop():
    nz, ny, nx = 9, 5, 4
    rng = np.random.default_rng(7)
    field = rng.standard_normal((nz, ny, nx))
    khv = np.abs(rng.standard_normal((nz, ny, nx))) + 0.1
    fnm = rng.random(nz) + 0.5
    fnp = rng.random(nz) + 0.5
    # a monotone geopotential, as a real column has
    phi = np.cumsum(np.abs(rng.standard_normal((nz + 1, ny, nx))) + 1.0,
                    axis=0) * 100.0

    got = ctbl._sgs_flux(field, khv, fnm, fnp, phi, nz)
    want = _sgs_flux_reference(field, khv, fnm, fnp, phi, nz)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    # the w-level ends are untouched by construction, on both sides
    assert got[0] == 0.0 and got[nz] == 0.0


def test_sgs_flux_is_zero_without_a_live_k_field():
    """No ``smag_khv`` means no SGS flux to report, not a crash: the slot is
    filled by the closure during a step, so a frame taken before one exists
    has nothing to read."""
    out = ctbl._sgs_flux(np.zeros((4, 2, 2)), None, np.zeros(4),
                         np.zeros(4), np.zeros((5, 2, 2)), 4)
    assert out.shape == (5,)
    assert not out.any()


def test_sgs_flux_geometry_identity_with_the_oracle_spelling():
    """``2*G/(phi[k+1]-phi[k-1])`` and ``1/(z_mass[k]-z_mass[k-1])`` are one
    geometry in two spellings -- the WOOF-ArWen construction and the oracle's.
    Stated in the docstring; measured here, because "obviously the same" is
    how unstagger conventions get mixed."""
    rng = np.random.default_rng(3)
    nz = 12
    z_w = np.cumsum(np.abs(rng.standard_normal(nz + 1)) + 1.0) * 30.0
    z_mass = 0.5 * (z_w[:-1] + z_w[1:])
    phi = z_w * 9.81
    for kw in range(1, nz):
        mine = 2.0 * 9.81 / (phi[kw + 1] - phi[kw - 1])
        theirs = 1.0 / (z_mass[kw] - z_mass[kw - 1])
        assert mine == pytest.approx(theirs, rel=1e-14)


# ------------------------------------------------------------- case config

def test_config_matches_the_oracle_matched_moist_namelist():
    """The knobs the oracle namelist pins, checked one by one.

    ``namelist.moist_match_km3_100m`` (sha256 ``55c21819...``) differs from
    the certified dry ``namelist.match_km3_100m`` by exactly two lines,
    ``mp_physics`` and the iofields filename.  The WOOF-ArWen config must carry
    the same delta over the same dry base.
    """
    cfg = ctbl.make_config()
    assert (cfg.nx, cfg.ny, cfg.nz) == (96, 96, 64)
    assert cfg.dx == 100.0 and cfg.dy == 100.0 and cfg.ztop == 2400.0
    assert cfg.dt == 0.5 and cfg.run_seconds == 7200.0
    assert cfg.km_opt == 3 and cfg.mix_isotropic == 1
    assert cfg.c_s == 0.18 and cfg.c_k == 0.10
    assert cfg.isfflx == 0
    assert cfg.tke_heat_flux == 0.24
    assert cfg.tke_drag_coefficient == 0.0013
    assert cfg.bl_pbl_physics == 0 and cfg.sf_sfclay_physics == 0
    assert cfg.time_step_sound == 4
    # the moist delta, and nothing else
    assert cfg.moist is True and cfg.mp_physics == 1
    assert cfg.moist_adv_opt == 1
    # diff_6th_opt=1 is refused with moisture (config.py:1855); the oracle
    # namelist runs diff_6th_opt=0 and so must this
    assert cfg.diff_6th_opt == 0


def test_config_rejects_a_closure_the_campaign_has_no_arm_for():
    with pytest.raises(ValueError, match="km_opt"):
        ctbl.make_config(km_opt=1)


def test_km2_arm_selectable():
    assert ctbl.make_config(km_opt=2).km_opt == 2


def test_gates_are_validity_only():
    """No moist LES band is cut in this module.  The oracle's AC-CAP
    criteria were cut from the oracle's own draws and are not this
    engine's to inherit; the dry case makes the same argument for the same
    reason (``convective_boundary_layer.py:17-23``)."""
    assert set(ctbl.GATES) == {"w_max", "cfl_max", "mass_drift_rel"}


def test_receipt_field_partition_is_shared_with_the_dry_case():
    """The dual-run corruption screen must see one partition, not two:
    this case re-exports the dry case's, it does not define a second."""
    from woof.verify.cases import convective_boundary_layer as cbl
    assert ctbl.ENVIRONMENTAL_FIELDS is cbl.ENVIRONMENTAL_FIELDS
    assert ctbl.partition_receipt_fields is cbl.partition_receipt_fields


def test_print_sounding_emits_the_asset_bytes(tmp_path):
    """``--print-sounding | sha256sum`` is the documented way to check that
    this engine is about to initialise from the ratified asset.  It writes
    BYTES: routing it through the text layer on Windows turns every "\\n"
    into "\\r\\n" and the digest then matches nothing, which would make the
    one check the switch exists for silently useless."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-m",
         "woof.verify.cases.cloud_topped_boundary_layer",
         "--print-sounding"],
        cwd=_ROOT, capture_output=True, check=True).stdout
    assert hashlib.sha256(out).hexdigest() == \
        ctbl.SOUNDING_SHA256_CAPPED_MOIST
    assert b"\r" not in out


def test_case_module_declares_the_verify_contract():
    """Registration is by static inspection of top-level names
    (``woof/verify/cases/__init__.py``): a module binding ``GATES`` and
    ``run`` is a verify case, with no table to edit."""
    from woof.verify import cases
    path = Path(cases.__file__).parent / "cloud_topped_boundary_layer.py"
    assert cases.VERIFY in cases.capabilities_of(path)
    assert cases.SCRIPT in cases.capabilities_of(path)
