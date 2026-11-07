"""The mp=28 WIF climatology ingest, held to WRF's numbers.

The port under test is :mod:`woof.ingest.wif_climatology` -- WRF's
``QNWFA_QNIFA_SIGMA_MONTHLY.dat`` input path (metgrid ``constants_name``
horizontal + real.exe monthly/vertical) -- whose end-to-end oracle is a
WRF-4.7.1 real.exe run of the summer reference case (test host 4,
``cases/summer-20260825/run-ctrl``): against that wrfinput the port's
QNWFA agreed to a pointwise relative max of 1.03e-5, QNIFA to 7.84e-6,
QNWFA2D to 1.26e-6 (bit-exact when fed metgrid's own horizontal output),
QNIFA2D bit-exact zero -- float32 arithmetic-order noise, no structural
term.  These tests pin the pieces that comparison exercised, on synthetic
inputs small enough for stage 1, plus the fail-closed admission around the
new ``wif_climatology_path`` configuration key.
"""

import struct

import numpy as np
import pytest

from woof.config import RunConfig, validate_aerosol_source_options
from woof.ingest.wif_climatology import (
    WifClimatology,
    WifClimatologyError,
    four_pt_bilinear,
    load_wif_climatology,
    monthly_interp_to_date,
    monthly_interp_weights,
    orient_bottom_up,
    read_wps_intermediate,
    vert_interp_wif_column_grid,
    wif_fields_for_grid,
    wif_surface_emission,
)

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def _cfg(**overrides):
    base = dict(nx=2, ny=2, nz=4, dx=1.0, dy=1.0, ztop=1.0,
                dt=1.0, run_seconds=0.0)
    base.update(overrides)
    return RunConfig(**base)


# ---------------------------------------------------------------------------
# WPS intermediate reader (IFV=5)
# ---------------------------------------------------------------------------

def _write_ifv5(path, fields):
    """Write a minimal IFV=5 intermediate file: (name, xlvl, ny, nx, data)."""
    def rec(payload):
        return struct.pack(">i", len(payload)) + payload + struct.pack(
            ">i", len(payload))

    with open(path, "wb") as handle:
        for name, xlvl, data in fields:
            ny, nx = data.shape
            handle.write(rec(struct.pack(">i", 5)))
            header = (b"2007:01:15_00:00:00     "         # hdate, 24
                      + struct.pack(">f", 0.0)            # xfcst
                      + b"synthetic".ljust(32)            # map_source
                      + name.encode().ljust(9)            # field
                      + b"#/kg".ljust(25)                 # units
                      + b"test field".ljust(46)           # desc
                      + struct.pack(">fiii", xlvl, nx, ny, 0))
            handle.write(rec(header))
            proj = (b"SWCORNER"
                    + struct.pack(">fffff", -90.0, -180.0,
                                  180.0 / (ny - 1), 360.0 / nx, 6371.0))
            handle.write(rec(proj))
            handle.write(rec(struct.pack(">i", 0)))       # is_wind_earth_rel
            handle.write(rec(data.astype(">f4").tobytes()))


def _synthetic_dat(tmp_path, nlev=3, ny=5, nx=8):
    """A full 12-month synthetic dataset on a tiny global lat-lon grid."""
    rng = np.random.default_rng(20260825)
    entries = []
    arrays = {}
    for prefix, lo, hi in (("QNWFA", 1.0e8, 4.0e9), ("QNIFA", 1.0e2, 6.0e5)):
        arrays[prefix] = rng.uniform(
            lo, hi, size=(12, nlev, ny, nx)).astype(np.float32)
    # Pressure decreasing with level index (bottom-up in the file) so the
    # orientation test has a definite answer either way.
    base_p = np.array([9.5e4, 5.0e4, 1.0e4], dtype=np.float32)[:nlev]
    arrays["P_WIF"] = np.broadcast_to(
        base_p[None, :, None, None] , (12, nlev, ny, nx)).astype(
        np.float32) + rng.uniform(0, 100, (12, nlev, ny, nx)).astype(
        np.float32)
    for prefix in ("QNWFA", "QNIFA", "P_WIF"):
        for month_index, month in enumerate(_MONTHS):
            for level in range(nlev):
                entries.append((f"{prefix}_{month}", float(level + 1),
                                arrays[prefix][month_index, level]))
    path = tmp_path / "wif_synthetic.dat"
    _write_ifv5(path, entries)
    return path, arrays


def test_reader_roundtrips_the_ifv5_layout(tmp_path):
    path, arrays = _synthetic_dat(tmp_path)
    clim = load_wif_climatology(path)
    assert isinstance(clim, WifClimatology)
    np.testing.assert_array_equal(clim.qnwfa, arrays["QNWFA"])
    np.testing.assert_array_equal(clim.qnifa, arrays["QNIFA"])
    np.testing.assert_array_equal(clim.pressure, arrays["P_WIF"])
    assert clim.latitude[0] == -90.0 and clim.latitude[-1] == 90.0
    assert clim.longitude[0] == -180.0


def test_reader_refuses_a_missing_month(tmp_path):
    path, _ = _synthetic_dat(tmp_path)
    records, _lat, _lon = read_wps_intermediate(path)
    assert "QNWFA_AUG" in records
    truncated = [(name, xlvl, values)
                 for name, by_level in records.items()
                 for xlvl, values in by_level.items()
                 if name != "QNWFA_AUG"]
    bad = tmp_path / "missing_month.dat"
    _write_ifv5(bad, truncated)
    with pytest.raises(WifClimatologyError, match="QNWFA_AUG"):
        load_wif_climatology(bad)


def test_reader_refuses_a_non_latlon_projection(tmp_path):
    path = tmp_path / "lambert.dat"

    def rec(payload):
        return struct.pack(">i", len(payload)) + payload + struct.pack(
            ">i", len(payload))

    with open(path, "wb") as handle:
        handle.write(rec(struct.pack(">i", 5)))
        header = (b"2007:01:15_00:00:00     " + struct.pack(">f", 0.0)
                  + b"synthetic".ljust(32) + b"QNWFA_JAN".ljust(9)
                  + b"#/kg".ljust(25) + b"test".ljust(46)
                  + struct.pack(">fiii", 1.0, 2, 2, 3))
        handle.write(rec(header))
        handle.write(rec(b"SWCORNER" + struct.pack(">fffff", 1, 1, 1, 1, 1)))
        handle.write(rec(struct.pack(">i", 0)))
        handle.write(rec(np.zeros((2, 2), dtype=">f4").tobytes()))
    with pytest.raises(WifClimatologyError, match="iproj=3"):
        read_wps_intermediate(path)


# ---------------------------------------------------------------------------
# Temporal: monthly_interp_to_date
# ---------------------------------------------------------------------------

def test_monthly_weights_match_the_oracle_date():
    """2026-08-25: the exact bracketing real.exe printed for the oracle run.

    Aug 15 is julian day 227 and Sep 15 is 258 in 2026 (non-leap), the
    target Aug 25 is 237, so month1=8/month2=9 weighted 21/31 and 10/31 --
    the same integers the test host 4 comparison receipt carried.
    """
    assert monthly_interp_weights("2026-08-25_18:00:00") == (8, 9, 21, 10, 31)


def test_monthly_weights_ignore_the_hour_and_wrap_the_year():
    assert (monthly_interp_weights("2026-08-25_00:00:00")
            == monthly_interp_weights("2026-08-25_23:00:00"))
    month1, month2, w1, w2, den = monthly_interp_weights(
        "2026-01-05_00:00:00")
    assert (month1, month2) == (12, 1)
    assert w1 + w2 == den == 31


def test_monthly_interp_is_float32_linear():
    monthly = np.zeros((12, 2, 2), dtype=np.float32)
    monthly[7] = 31.0   # August
    monthly[8] = 62.0   # September
    out = monthly_interp_to_date(monthly, "2026-08-25_18:00:00")
    expected = np.float32(
        (np.float32(62.0) * np.float32(10) + np.float32(31.0) * np.float32(21))
        / np.float32(31))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.full((2, 2), expected))


# ---------------------------------------------------------------------------
# Horizontal: metgrid four_pt
# ---------------------------------------------------------------------------

def test_four_pt_reproduces_source_points_and_wraps_the_seam():
    lat = np.arange(-90.0, 90.1, 45.0)
    lon = np.arange(-180.0, 180.0, 45.0)  # global: 8 columns * 45 = 360
    field = np.arange(lat.size * lon.size, dtype=np.float32).reshape(
        lat.size, lon.size)
    # On-node: exact values back.
    got = four_pt_bilinear(field, lat, lon,
                           np.array([[0.0]]), np.array([[-90.0]]))
    np.testing.assert_array_equal(got, field[2:3, 2:3])
    # Across the antimeridian seam: average of the last and first columns.
    got = four_pt_bilinear(field, lat, lon,
                           np.array([[0.0]]), np.array([[157.5]]))
    expected = np.float32((field[2, 7] + field[2, 0]) / 2.0)
    assert got[0, 0] == pytest.approx(expected, rel=1e-6)


def test_four_pt_refuses_latitude_off_the_grid():
    lat = np.arange(0.0, 10.1, 1.0)
    lon = np.arange(0.0, 10.1, 1.0)
    field = np.zeros((lat.size, lon.size), dtype=np.float32)
    with pytest.raises(WifClimatologyError, match="latitude"):
        four_pt_bilinear(field, lat, lon,
                         np.array([[45.0]]), np.array([[5.0]]))


# ---------------------------------------------------------------------------
# Orientation and vertical: real.exe's assembled column
# ---------------------------------------------------------------------------

def test_orientation_flips_top_down_data_and_keeps_bottom_up():
    p_bottom_up = np.linspace(9.0e4, 1.0e4, 6, dtype=np.float32).reshape(
        6, 1, 1) * np.ones((6, 2, 2), dtype=np.float32)
    field = np.arange(6, dtype=np.float32).reshape(6, 1, 1) * np.ones(
        (6, 2, 2), dtype=np.float32)
    kept_p, (kept_f,) = orient_bottom_up(p_bottom_up, field)
    np.testing.assert_array_equal(kept_p, p_bottom_up)
    np.testing.assert_array_equal(kept_f, field)
    flipped_p, (flipped_f,) = orient_bottom_up(p_bottom_up[::-1],
                                               field[::-1])
    np.testing.assert_array_equal(flipped_p, p_bottom_up)
    np.testing.assert_array_equal(flipped_f, field)


def test_vertical_is_linear_in_log_pressure():
    p = np.array([1.0e5, 1.0e4], dtype=np.float32).reshape(2, 1, 1)
    f = np.array([10.0, 20.0], dtype=np.float32).reshape(2, 1, 1)
    target = np.array([np.sqrt(1.0e5 * 1.0e4)], dtype=np.float32).reshape(
        1, 1, 1)
    out = vert_interp_wif_column_grid(f, p, target)
    assert out[0, 0, 0] == pytest.approx(15.0, rel=1e-6)


def test_vertical_holds_constant_below_the_deepest_point():
    """extrap_type=2 for 'Q': constant below the assembled column."""
    p = np.array([9.0e4, 5.0e4], dtype=np.float32).reshape(2, 1, 1)
    f = np.array([7.0, 3.0], dtype=np.float32).reshape(2, 1, 1)
    target = np.array([9.8e4], dtype=np.float32).reshape(1, 1, 1)
    out = vert_interp_wif_column_grid(f, p, target)
    assert out[0, 0, 0] == np.float32(7.0)


def test_vertical_refuses_targets_above_the_source_top():
    p = np.array([9.0e4, 5.0e4], dtype=np.float32).reshape(2, 1, 1)
    f = np.array([7.0, 3.0], dtype=np.float32).reshape(2, 1, 1)
    target = np.array([1.0e4], dtype=np.float32).reshape(1, 1, 1)
    with pytest.raises(WifClimatologyError, match="above"):
        vert_interp_wif_column_grid(f, p, target)


def test_vertical_zap_close_drops_the_too_close_level_not_the_last():
    """A level within 500 Pa of the one below it is skipped (:6075-6080)."""
    p = np.array([9.0e4, 8.97e4, 5.0e4], dtype=np.float32).reshape(3, 1, 1)
    f = np.array([1.0, 100.0, 1.0], dtype=np.float32).reshape(3, 1, 1)
    target = np.array([8.97e4], dtype=np.float32).reshape(1, 1, 1)
    out = vert_interp_wif_column_grid(f, p, target)
    # The 8.97e4 level (value 100) was zapped: the answer interpolates
    # between 9.0e4 and 5.0e4 and must stay near 1, nowhere near 100.
    assert out[0, 0, 0] < 2.0


def test_vertical_force_sfc_skips_levels_below_the_first_eta_level():
    """force_sfc_in_vinterp=1 (:6053-6066): data levels between the
    surface slot and the pressure of eta level 1 are removed, so the first
    eta level interpolates between the surface value and the first level
    above it."""
    p = np.array([9.6e4, 9.5e4, 5.0e4], dtype=np.float32).reshape(3, 1, 1)
    f = np.array([10.0, 1000.0, 20.0], dtype=np.float32).reshape(3, 1, 1)
    # eta level 1 pressure ABOVE (numerically below) the 9.5e4 data level:
    # that level is skipped entirely.
    target = np.array([9.4e4, 7.0e4], dtype=np.float32).reshape(2, 1, 1)
    out = vert_interp_wif_column_grid(f, p, target)
    assert 10.0 < out[0, 0, 0] < 20.0     # not influenced by the 1000
    assert 10.0 < out[1, 0, 0] < 20.0


def test_surface_emission_is_the_wrf_formula():
    w_level1 = np.full((2, 2), 2.0e9, dtype=np.float32)
    phb = np.zeros((3, 2, 2), dtype=np.float32)
    phb[1] = np.float32(9.81) * np.float32(100.0)   # z1 = 100 m
    qnwfa2d, qnifa2d = wif_surface_emission(w_level1, phb)
    expected = np.float32(2.0e9) * np.float32(0.000196) * (
        np.float32(50.0) / np.float32(100.0))
    np.testing.assert_array_equal(qnwfa2d, np.full((2, 2), expected))
    np.testing.assert_array_equal(qnifa2d, np.zeros((2, 2), dtype=np.float32))


def test_full_pipeline_shapes_and_receipt(tmp_path):
    path, _ = _synthetic_dat(tmp_path)
    clim = load_wif_climatology(path)
    target_lat = np.array([[10.0, 10.0], [20.0, 20.0]])
    target_lon = np.array([[-100.0, -90.0], [-100.0, -90.0]])
    pd = np.stack([np.full((2, 2), 9.0e4), np.full((2, 2), 6.0e4),
                   np.full((2, 2), 2.0e4)]).astype(np.float32)
    phb = np.zeros((4, 2, 2), dtype=np.float32)
    phb[1] = 9.81 * 50.0
    fields, receipt = wif_fields_for_grid(
        clim, target_lat, target_lon, "2026-08-25_18:00:00", pd, phb)
    assert fields["nwfa"].shape == (3, 2, 2)
    assert fields["nifa"].shape == (3, 2, 2)
    assert fields["nwfa2d"].shape == (2, 2)
    np.testing.assert_array_equal(fields["nifa2d"],
                                  np.zeros((2, 2), dtype=np.float32))
    assert (fields["nwfa"] > 0).all() and (fields["nifa"] > 0).all()
    assert receipt["temporal"]["month1"] == 8
    assert receipt["temporal"]["weight1_days"] == 21
    assert receipt["num_wif_levels"] == 3


# ---------------------------------------------------------------------------
# Fail-closed admission around wif_climatology_path
# ---------------------------------------------------------------------------

def test_default_selectors_still_refuse_exactly_as_before():
    for name, value in (("aer_init_opt", 1), ("aer_init_opt", 2),
                        ("wif_input_opt", 1), ("wif_input_opt", 2)):
        with pytest.raises(NotImplementedError):
            validate_aerosol_source_options(_cfg(**{name: value}))


def test_the_climatology_pair_no_longer_demands_an_explicit_path(tmp_path):
    """RE-BASELINED (development branch).

    The old pin asserted that ``aer_init_opt=1``/``wif_input_opt=1`` was
    admitted ONLY alongside a non-empty ``wif_climatology_path``, and that
    the empty case refused with "wif_climatology_path is empty".  That
    refusal existed because there was no other way to find the dataset.
    There is now: ``resolve_wif_climatology`` searches the config path, two
    environment overrides, and WRF's own bare-relative rule.  Keeping the
    old refusal would have made the CORRECT configuration the one that
    needs the most typing.

    Nothing was loosened, only relocated -- the requirement that the dataset
    actually EXIST is enforced at the point that reads it, by the resolver's
    ``explicit_required``, which is what the two tests below cover.
    """
    validate_aerosol_source_options(_cfg(
        aer_init_opt=1, wif_input_opt=1,
        wif_climatology_path=str(tmp_path / "wif.dat")))
    validate_aerosol_source_options(_cfg(aer_init_opt=1, wif_input_opt=1))


def test_an_explicitly_named_missing_dataset_raises_rather_than_degrades():
    """The relocated half of the retired "path is empty" refusal.

    An operator who names a dataset that is not there gets an error, not the
    synthetic profile.  This is the thompson_aerosol_contract posture: an
    override that is silently ignored is how a run ends up using an initial
    condition nobody chose.
    """
    from woof.ingest.wif_climatology import (
        MissingWifClimatologyDataset, resolve_wif_climatology)

    with pytest.raises(MissingWifClimatologyDataset,
                       match="there is no such file"):
        resolve_wif_climatology("nowhere/QNWFA_QNIFA_SIGMA_MONTHLY.dat")


def test_an_explicit_climatology_request_refuses_when_nothing_resolves(
        tmp_path, monkeypatch):
    """``mp28_aerosol_source='climatology'`` refuses instead of falling back."""
    from woof.ingest import wif_climatology

    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    # THE STAGED ROOT is a rung too (merge: static-dataset-door +
    # wif-default).  `woof fetch-tables --wif` installs the dataset into
    # $WOOF_WIF_DATA_ROOT, defaulting to ~/.woof/wif, and
    # resolve_wif_climatology searches it ahead of the cwd.  Unsetting the
    # two file/root overrides is therefore no longer enough to guarantee
    # "nothing resolves": on any machine that has actually staged the
    # dataset -- the reference host has -- this test would otherwise
    # resolve the real 225 MB copy and pass, or fail, for a reason it is
    # not about.  Point the staged root at an empty directory instead.
    with pytest.raises(wif_climatology.MissingWifClimatologyDataset,
                       match="requires the dataset"):
        wif_climatology.resolve_wif_climatology(
            cwd=tmp_path, explicit_required=True,
            env={"WOOF_WIF_DATA_ROOT": str(tmp_path / "no-staged-wif")})


def test_the_strict_refusal_names_the_selector_that_actually_asked(
        tmp_path, monkeypatch):
    """There are two spellings of the strict request, and only one is a field.

    A namelist carrying ``(aer_init_opt, wif_input_opt) = (1, 1)`` reaches
    this resolver with ``mp28_aerosol_source`` at its default 'auto'
    (woof/ingest/real.py resolves the pair to 'climatology'), so a refusal
    that named the field would report a setting the operator never touched.
    """
    from woof.ingest import wif_climatology
    from woof.ingest.analyzed_numbers import WIF_CLIMATOLOGY_NAMELIST_PHRASE

    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    with pytest.raises(wif_climatology.MissingWifClimatologyDataset) as named:
        wif_climatology.resolve_wif_climatology(
            cwd=tmp_path, explicit_required=True,
            requested_by=WIF_CLIMATOLOGY_NAMELIST_PHRASE,
            env={"WOOF_WIF_DATA_ROOT": str(tmp_path / "no-staged-wif")})
    said = str(named.value)
    assert said.startswith(
        "(aer_init_opt, wif_input_opt) = (1, 1) requires the dataset")
    assert "mp28_aerosol_source" not in said.split("Searched")[0]


def test_a_dataset_path_the_selection_would_not_read_is_refused():
    """RE-BASELINED.  The old pin refused a path under the 0/0 selectors.

    0/0 no longer means "no climatology"; it means "auto", which reads the
    path.  The refusal it guarded -- carrying a dataset path nothing would
    open -- still exists and still names the same breakage; the only
    configuration in which it is now true is an explicit synthetic request.
    """
    with pytest.raises(ValueError, match="no code would open"):
        validate_aerosol_source_options(_cfg(
            mp28_aerosol_source="synthetic",
            wif_climatology_path="somewhere/wif.dat"))


def test_default_config_takes_the_climatology_when_it_resolves():
    """RE-BASELINED -- this is the default flip, at the config layer.

    The old assertion was that a default RunConfig "executes not one
    instruction of the new module".  That was the defect.  What a default
    RunConfig now carries: WRF's Registry defaults on both namelist
    selectors (unchanged, and essential for the prepared-forecast
    runner's exact-equality switch rows), an empty explicit path
    (unchanged), and ``mp28_aerosol_source='auto'`` -- the field that makes
    the dataset the default without moving either namelist selector.
    """
    cfg = _cfg()
    assert cfg.wif_climatology_path == ""
    assert (cfg.aer_init_opt, cfg.wif_input_opt) == (0, 0)
    assert cfg.mp28_aerosol_source == "auto"
    validate_aerosol_source_options(cfg)


def test_the_receipt_statement_names_no_working_or_home_directory(tmp_path):
    """THE BREAKAGE: the namelist import wrote the search's absolute cwd and
    home paths into the WRF-input door's preparation receipt, so a
    ``--restart`` launched from another directory hashed a different receipt
    and was refused as "written for a different run" (5 km radar DA,
    2026-10-23).  The statement the receipt binds is the same from any
    directory; the run's own warning keeps the paths."""
    import shutil

    from woof.ingest.wif_climatology import (
        WIF_CLIMATOLOGY_FILE, portable_wif_statement, resolve_wif_climatology)

    def resolve(where, home):
        where.mkdir(parents=True, exist_ok=True)
        return resolve_wif_climatology(
            cwd=where, env={"WOOF_WIF_DATA_ROOT": str(home)})

    a = resolve(tmp_path / "a", tmp_path / "home-a")
    b = resolve(tmp_path / "b", tmp_path / "home-b")
    assert not a.resolved and not b.resolved
    assert a.fallback_reason != b.fallback_reason
    assert str(tmp_path / "a") in a.fallback_reason
    assert portable_wif_statement(a) == portable_wif_statement(b)
    assert str(tmp_path) not in portable_wif_statement(a)

    # Found in the working directory: the same bytes state the same
    # sentence from either directory, other bytes do not.
    source, _ = _synthetic_dat(tmp_path)
    for where in ("a", "b"):
        shutil.copyfile(source, tmp_path / where / WIF_CLIMATOLOGY_FILE)
    a = resolve(tmp_path / "a", tmp_path / "home-a")
    b = resolve(tmp_path / "b", tmp_path / "home-b")
    assert a.resolved and b.resolved and a.path != b.path
    assert portable_wif_statement(a) == portable_wif_statement(b)
    assert str(tmp_path) not in portable_wif_statement(a)
    with open(tmp_path / "b" / WIF_CLIMATOLOGY_FILE, "ab") as extra:
        extra.write(b"\0")
    b = resolve(tmp_path / "b", tmp_path / "home-b")
    assert portable_wif_statement(a) != portable_wif_statement(b)


def test_one_decode_and_one_horizontal_interpolation_serve_every_forcing_time(tmp_path, monkeypatch):
    """Breakage: a GFS preparation decoded the WIF file and interpolated all 12 months onto the
    same grid once per forcing time (13x for a 36 h window, ~12 s of a 50 s preparation)."""
    from woof.ingest import wif_climatology as wif

    path, _ = _synthetic_dat(tmp_path)
    clim = load_wif_climatology(path, backend="numpy")
    assert load_wif_climatology(path, backend="numpy") is clim
    assert not clim.qnwfa.flags.writeable
    calls = []
    real = wif.four_pt_bilinear
    monkeypatch.setattr(wif, "four_pt_bilinear", lambda *a: calls.append(1) or real(*a))
    target_lat = np.array([[10.0, 10.0], [20.0, 20.0]])
    target_lon = np.array([[-100.0, -90.0], [-100.0, -90.0]])
    pd = np.stack([np.full((2, 2), 9.0e4), np.full((2, 2), 6.0e4),
                   np.full((2, 2), 2.0e4)]).astype(np.float32)
    phb = np.zeros((4, 2, 2), dtype=np.float32)
    phb[1] = 9.81 * 50.0
    dates = ("2026-08-25_18:00:00", "2026-08-25_21:00:00", "2026-09-02_00:00:00")
    cached = [wif_fields_for_grid(clim, target_lat, target_lon, d, pd, phb, backend="numpy")[0]
              for d in dates]
    assert len(calls) == 3  # three species, once
    for d, held in zip(dates, cached):
        wif._HORIZONTAL.clear()  # each date against an uncached interpolation
        fresh = wif_fields_for_grid(clim, target_lat, target_lon, d, pd, phb, backend="numpy")[0]
        for name in fresh:
            assert held[name].tobytes() == fresh[name].tobytes()
    # Another grid is interpolated afresh.
    wif_fields_for_grid(clim, target_lat + 1.0, target_lon, dates[0], pd, phb, backend="numpy")
    assert len(calls) == 3 + 3 * len(dates) + 3
