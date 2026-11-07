"""WRF RRTM longwave (ra_lw_physics=1) through the seams that ship.

Scope, stated once so no reader has to infer it: this file establishes
that the port RUNS and that what it computes carries the physical
signatures a longwave scheme must carry.  It is NOT an oracle comparison:
nothing here compares a number against the WRF v4.6.1 Fortran.  That is
tests/test_rrtm_lw_wrf461_parity.py, which holds every output word to the
unmodified module's.

Almost every assertion here is self-consistency -- the port compared with
itself under a changed input -- and self-consistency is blind to an error
shared by every column.  A review found exactly such an error (the Planck
table read one row too warm, biasing every flux by 1.3-2.5%) while all of
these tests passed, so the file now carries one check anchored OUTSIDE the
port: :func:`test_the_planck_table_integrates_to_the_stefan_boltzmann_
blackbody` requires RRTM's band-summed Planck table to reproduce
``sigma*T^4/pi``.  That class of defect is visible here now; a
wrong-but-smooth absorption coefficient or a uniformly scaled band still
is not.

What the assertions are worth is guarded by three mutation controls:
:func:`test_the_signature_assertions_fail_when_the_longwave_is_stubbed`
zeroes the band optical depths and then TAUCLOUD, and
:func:`test_the_blackbody_identity_catches_a_one_row_planck_offset`
reintroduces the index-base slip and requires the blackbody check to go
red while the signature checks stay green.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config
from woof.core import rrtm_lw
from woof.core.rrtm_lw import rrtm_layer_count, rrtm_longwave_columns
from woof.core.rrtm_tables import load_rrtm_lw_tables
from woof.namelist_import import import_namelists

MODEL = Path(__file__).resolve().parents[1]
F = np.float32

#: Fortran declarations from module_ra_rrtm.F:38-63 and :116-145, as
#: (rows, combined g-points) after the CMBGBn reduction.
_DECLARED_ABSA = {
    1: (65, 8), 2: (65, 14), 3: (650, 16), 4: (585, 14), 5: (585, 16),
    6: (65, 8), 7: (585, 12), 8: (35, 8), 9: (715, 12), 10: (65, 6),
    11: (65, 8), 12: (585, 8), 13: (585, 4), 14: (65, 2), 15: (585, 2),
    16: (585, 2),
}
_DECLARED_ABSB = {
    1: (235, 8), 2: (235, 14), 3: (1175, 16), 4: (1410, 14), 5: (1175, 16),
    7: (235, 12), 8: (265, 8), 9: (235, 12), 10: (235, 6), 11: (235, 8),
    14: (235, 2),
}


# ---------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------


def test_the_reduction_lands_on_wrfs_declared_combined_shapes() -> None:
    tables = load_rrtm_lw_tables()
    assert {band: array.shape for band, array in tables.absa.items()} == \
        _DECLARED_ABSA
    assert {band: array.shape for band, array in tables.absb.items()} == \
        _DECLARED_ABSB
    for band, array in tables.selfrefc.items():
        assert array.shape == (10, int(tables.ngc[band - 1])), band
    for name, array in {**tables.absa, **tables.absb,
                        **tables.selfrefc, **tables.combined}.items():
        assert array.dtype == np.float32, name
        assert np.isfinite(array).all(), name


def test_the_gpoint_bookkeeping_matches_the_module_head_tables() -> None:
    """NGC/NGS/NGN/NGB and RWGT are one consistent partition."""

    tables = load_rrtm_lw_tables()
    ngc, ngs, ngn, ngb = (tables.ngc, tables.ngs, tables.statics["NGN"],
                          tables.ngb)
    assert int(ngc.sum()) == 140
    assert np.array_equal(np.cumsum(ngc), ngs)
    assert int(ngn.sum()) == 256
    assert np.array_equal(np.bincount(ngb, minlength=17)[1:], ngc)
    # Within a combined band the RWGT entries of one group sum to 1: they
    # are WT(IG)/WTSM(group), and WTSM is that group's WT total.
    for band in range(1, 17):
        base = 0 if band == 1 else int(ngs[band - 2])
        weights = tables.rwgt[(band - 1) * 16:band * 16]
        start = 0
        for igc in range(int(ngc[band - 1])):
            stop = start + int(ngn[base + igc])
            total = float(weights[start:stop].sum())
            expected = 1.0 if int(ngc[band - 1]) < 16 else float(stop - start)
            assert total == pytest.approx(expected, rel=1e-5), (band, igc)
            start = stop
        assert start == 16, band


def test_the_buffer_layer_count_follows_rrtminit() -> None:
    """NLAYERS = kme + NINT(p_top_hPa/4) - 1 with kme = nz+1."""

    assert rrtm_layer_count(40, 5000.0) == 40 + 13   # NINT(12.5) = 13
    assert rrtm_layer_count(40, 1000.0) == 40 + 3    # NINT(2.5)  = 3
    assert rrtm_layer_count(50, 10000.0) == 50 + 25
    with pytest.raises(ValueError):
        rrtm_layer_count(40, 0.0)
    with pytest.raises(ValueError):
        rrtm_layer_count(2, 5000.0)


# ---------------------------------------------------------------------------
# Column solver
# ---------------------------------------------------------------------------

NZ = 40
_HEIGHT = np.linspace(0.0, 16000.0, NZ, dtype=np.float32)


def _bottom_up_profile(surface_temperature: float = 288.15):
    pw = np.linspace(1013.0, 50.0, NZ + 1, dtype=np.float32)
    p = (F(0.5) * (pw[:-1] + pw[1:])).astype(F)
    t = np.maximum((surface_temperature - F(6.5e-3) * _HEIGHT).astype(F),
                   F(210.0))
    tw = np.empty(NZ + 1, F)
    tw[1:NZ] = F(0.5) * (t[:-1] + t[1:])
    tw[0] = t[0] + F(0.5) * (t[0] - t[1])
    tw[NZ] = t[NZ - 1] + F(0.5) * (t[NZ - 1] - t[NZ - 2])
    qv = (F(0.012) * np.exp(-_HEIGHT / F(2500.0))).astype(F)
    dz = np.full(NZ, 400.0, F)
    return p, pw, t, tw, qv, dz


def _column_block(*, ncol: int = 1, surface_temperature: float = 288.15):
    """``(kwargs, layer_index_helper)`` for :func:`rrtm_longwave_columns`.

    Layer arrays come back TOP-DOWN, the frame WRF's RRTMLWRAD packs.
    """
    p, pw, t, tw, qv, dz = _bottom_up_profile(surface_temperature)

    def top_down(array):
        return np.ascontiguousarray(
            np.repeat(array[::-1][None, :], ncol, axis=0).astype(F))

    zeros = np.zeros((ncol, NZ), F)
    kwargs = dict(
        t=top_down(t), p_mb=top_down(p), dz=top_down(dz), qv=top_down(qv),
        qc=zeros.copy(), qr=zeros.copy(), qi=zeros.copy(),
        qs=zeros.copy(), qg=zeros.copy(), cldfra=zeros.copy(),
        tw=top_down(tw), pw_mb=top_down(pw),
        tsfc=np.full(ncol, surface_temperature, F),
        emiss=np.full(ncol, 0.95, F),
        r=287.0, g=9.81, co2vmr=379.0e-6, n2ovmr=319.0e-9,
        ch4vmr=1774.0e-9, nlayers=rrtm_layer_count(NZ, 5000.0))
    return kwargs


def _top_down_index(bottom_up_level: int) -> int:
    """Column index of a bottom-up model level in the TOP-DOWN block."""
    return NZ - 1 - bottom_up_level


def _seed_cloud(kwargs, *, base: int, top: int, water: bool):
    """Fill bottom-up levels ``base..top`` with condensate and cloud."""
    rows = [_top_down_index(level) for level in range(base, top + 1)]
    for row in rows:
        if water:
            kwargs["qc"][:, row] = F(4.0e-4)
        else:
            kwargs["qi"][:, row] = F(2.0e-4)
            kwargs["qs"][:, row] = F(1.0e-4)
        kwargs["cldfra"][:, row] = F(1.0)
    return rows


def _heating_k_per_day(result):
    """Bottom-up model-layer heating rate in K/day."""
    return np.asarray(result["tten"]) * 86400.0


def _assert_finite_and_physical(result) -> None:
    for name, value in result.items():
        assert np.isfinite(np.asarray(value)).all(), name
        assert np.asarray(value).dtype == np.float32, name
    glw = np.asarray(result["glw"])
    olr = np.asarray(result["olr"])
    assert np.all(glw > 50.0) and np.all(glw < 600.0), glw
    assert np.all(olr > 50.0) and np.all(olr < 400.0), olr


def test_a_clear_column_cools_at_every_layer_with_physical_fluxes() -> None:
    result = rrtm_longwave_columns(**_column_block())
    _assert_finite_and_physical(result)
    heating = _heating_k_per_day(result)
    # A cloud-free moist troposphere radiates to space at every level.
    assert np.all(heating < 0.0), heating
    assert np.all(heating > -20.0), heating
    # The surface downward flux cannot exceed a blackbody at the warmest
    # temperature in the column.
    sigma_t4 = 5.670374e-8 * 288.15 ** 4
    assert float(np.asarray(result["glw"])[0]) < sigma_t4


def _cloud_signature(kwargs, *, base: int, top: int, water: bool):
    """Return ``(clear, cloudy)`` results for the same profile."""
    clear = rrtm_longwave_columns(**{k: (v.copy() if hasattr(v, "copy")
                                         else v) for k, v in kwargs.items()})
    _seed_cloud(kwargs, base=base, top=top, water=water)
    return clear, rrtm_longwave_columns(**kwargs)


def _assert_cloud_top_cools_and_base_warms(clear, cloudy, base, top) -> None:
    heating = _heating_k_per_day(cloudy)[0]
    reference = _heating_k_per_day(clear)[0]
    assert heating[top] < reference[top] - 5.0, (heating[top], reference[top])
    assert heating[base] > reference[base] + 2.0, (heating[base],
                                                   reference[base])
    assert heating[top] < heating[base]


def test_a_water_cloud_cools_its_top_and_warms_its_base() -> None:
    base, top = 26, 30
    clear, cloudy = _cloud_signature(
        _column_block(), base=base, top=top, water=True)
    _assert_finite_and_physical(cloudy)
    _assert_cloud_top_cools_and_base_warms(clear, cloudy, base, top)
    # A low deck emits downward at near-surface temperature, so the
    # surface sees more longwave than the clear column does.
    assert float(np.asarray(cloudy["glw"])[0]) > \
        float(np.asarray(clear["glw"])[0]) + 10.0


def test_an_ice_cloud_lowers_the_outgoing_longwave() -> None:
    base, top = 6, 14
    clear, cloudy = _cloud_signature(
        _column_block(), base=base, top=top, water=False)
    _assert_finite_and_physical(cloudy)
    _assert_cloud_top_cools_and_base_warms(clear, cloudy, base, top)
    # A cold high cloud replaces warm surface emission at the top of the
    # atmosphere.
    assert float(np.asarray(cloudy["olr"])[0]) < \
        float(np.asarray(clear["olr"])[0]) - 50.0


def test_a_warmer_surface_raises_the_downward_longwave() -> None:
    cold = rrtm_longwave_columns(**_column_block(surface_temperature=278.15))
    warm = rrtm_longwave_columns(**_column_block(surface_temperature=298.15))
    assert float(np.asarray(warm["glw"])[0]) > \
        float(np.asarray(cold["glw"])[0]) + 20.0
    assert float(np.asarray(warm["olr"])[0]) > \
        float(np.asarray(cold["olr"])[0]) + 10.0


def test_lowering_the_surface_emissivity_lowers_the_outgoing_flux() -> None:
    """SEMISS enters RTRN's surface source, so OLR must respond to it."""

    black = _column_block()
    black["emiss"] = np.full(1, 1.0, F)
    grey = _column_block()
    grey["emiss"] = np.full(1, 0.80, F)
    assert float(np.asarray(rrtm_longwave_columns(**grey)["olr"])[0]) < \
        float(np.asarray(rrtm_longwave_columns(**black)["olr"])[0])


def test_the_flux_divergence_identity_closes_over_the_whole_column() -> None:
    """What RRTM conserves: HTR is the net flux divergence, layer by layer.

    ``HTR(L) = HEATFAC * (FNET(L) - FNET(L+1)) / (PZ(L) - PZ(L+1))``
    (module_ra_rrtm.F:6594-6595).  Summing the mass-weighted heating over
    every layer -- the Cavallo buffer included -- must telescope to the
    difference between the surface and top-of-atmosphere net fluxes.  A
    transcription that dropped or double-counted a layer fails this.
    """
    kwargs = _column_block()
    _seed_cloud(kwargs, base=26, top=30, water=True)
    result = rrtm_longwave_columns(**kwargs)
    tables = load_rrtm_lw_tables()
    heatfac = float(tables.statics["HEATFAC"][0])
    pz = np.asarray(result["pz"])[0].astype(np.float64)
    htr = np.asarray(result["htr"])[0].astype(np.float64)
    fnet = (np.asarray(result["totuflux"])[0]
            - np.asarray(result["totdflux"])[0]).astype(np.float64)
    nlayers = pz.shape[0] - 1
    integrated = float(
        (htr[:nlayers] * (pz[:nlayers] - pz[1:]) / heatfac).sum())
    telescoped = float(fnet[0] - fnet[nlayers])
    assert integrated == pytest.approx(telescoped, rel=2.0e-5,
                                       abs=1.0e-3), (integrated, telescoped)
    # Guard against the identity holding on zeros.
    assert abs(telescoped) > 10.0


def test_a_column_block_gives_each_column_its_own_answer() -> None:
    """Vectorising over columns must not smear one column into another."""

    block = _column_block(ncol=3)
    _seed_cloud({"qc": block["qc"][1:2], "qi": block["qi"][1:2],
                 "qs": block["qs"][1:2], "cldfra": block["cldfra"][1:2]},
                base=26, top=30, water=True)
    block["tsfc"] = np.asarray([288.15, 288.15, 300.0], F)
    result = rrtm_longwave_columns(**block)
    glw = np.asarray(result["glw"])
    olr = np.asarray(result["olr"])
    # The cloudy column sees more downward longwave at the ground.
    assert glw[1] > glw[0] + 10.0
    # GLW is a downwelling flux and TBOUND enters only the UPWARD stream,
    # so the warm-surface column must move OLR and leave GLW alone.
    assert float(glw[2]) == pytest.approx(float(glw[0]), rel=1e-6)
    assert olr[2] > olr[0] + 2.0
    assert olr[1] < olr[0]
    single = rrtm_longwave_columns(**_column_block(ncol=1))
    assert float(glw[0]) == pytest.approx(
        float(np.asarray(single["glw"])[0]), rel=1e-6)


# ---------------------------------------------------------------------------
# The Planck source, against an external identity
# ---------------------------------------------------------------------------
#
# Everything above this line is self-consistency: it compares the port
# with itself under a changed input.  The two checks below are the one
# place in this file where an assertion is anchored to a number that
# exists outside the port, which is what it takes to see an offset shared
# by every column.  A one-row TOTPLNK slip -- the defect these were
# written for -- moves every flux by 1.3-2.5% and every signature test
# above still passes.

#: Stefan-Boltzmann, CODATA 2018 (W m-2 K-4).
_SIGMA_SB = 5.670374419e-8
#: TOTPLNK is tabulated in W cm-2 sr-1 (cm-1)-1; RRTM carries the cm-2
#: through the whole transfer and converts once in FLUXFAC (pi*2e4).
#: Here the same 1e4 puts a band-summed Planck integral in W m-2 sr-1.
_TOTPLNK_W_CM2_TO_W_M2 = 1.0e4


def _band_summed_planck(temperature):
    """``sum_b DELWAVE_b * B_b(T)`` in W m-2 sr-1, via the SHIPPED lookup.

    Calls :func:`rrtm_lw._planck_index` and :func:`rrtm_lw._planck_bands`
    -- the exact two functions ``rtrn_columns`` uses -- so a regression in
    either one lands here rather than in a copy of the arithmetic.
    """
    tables = load_rrtm_lw_tables()
    totplnk = tables.statics["TOTPLNK"].reshape(
        (rrtm_lw._TOTPLNK_ROWS, 16), order="F")
    delwave = tables.statics["DELWAVE"]
    t = np.atleast_1d(np.asarray(temperature, F))
    row, frac = rrtm_lw._planck_index(np, t, t.dtype)
    bands = np.asarray(rrtm_lw._planck_bands(totplnk, delwave, row, frac))
    return bands.sum(axis=-1) * _TOTPLNK_W_CM2_TO_W_M2


def test_the_planck_table_integrates_to_the_stefan_boltzmann_blackbody(
) -> None:
    """RRTM's 16 bands span the thermal spectrum, so their sum is a
    blackbody.

    ``sum_b DELWAVE_b * TOTPLNK(T,b)`` is the Planck function integrated
    over 10-3000 cm-1, which for a blackbody is ``sigma*T^4/pi``.  The
    table reproduces that to a few 1e-5, and nothing about the identity
    comes from this repository: it is the check that says WHICH row of
    TOTPLNK belongs to a given temperature, independent of the solver.

    The tolerance is set by the table's own accuracy, not by the port.
    160 K is the loosest point at 4.5e-5 relative (the band edges cut off
    least of the spectrum at the warm end); 288 K and above sit near
    1.6e-5.  Reading one row too warm costs 1.3% at 288 K and 2.5% at
    160 K, i.e. 250 to 500 times this tolerance, so the gap between
    "correct" and "off by one row" is not a matter of tuning.
    """
    for temperature in (160.0, 200.0, 250.0, 288.0, 300.0, 339.0):
        got = float(_band_summed_planck(temperature)[0])
        blackbody = _SIGMA_SB * temperature ** 4 / np.pi
        assert got == pytest.approx(blackbody, rel=5.0e-5), (
            temperature, got, blackbody, got / blackbody - 1.0)
    # Fractional temperatures exercise the interpolation between rows,
    # where linear interpolation of a convex T^4 sits slightly high.
    for temperature in (199.25, 250.7, 288.4):
        got = float(_band_summed_planck(temperature)[0])
        blackbody = _SIGMA_SB * temperature ** 4 / np.pi
        assert got == pytest.approx(blackbody, rel=5.0e-5), (
            temperature, got, blackbody, got / blackbody - 1.0)


def test_the_blackbody_identity_catches_a_one_row_planck_offset(
        monkeypatch) -> None:
    """Mutation control for the check above: reading the Fortran's
    1-based ``INDBOUND`` as a 0-based NumPy row must fail it.

    This is the third mutation in this file and the only one aimed at a
    defect that leaves every physical signature intact.  With it applied,
    all the cloud/cooling/emissivity assertions above still pass and the
    solver still runs; only the external identity notices.
    """
    real_index = rrtm_lw._planck_index

    def one_row_too_warm(xp, temperature, dtype):
        row, frac = real_index(xp, temperature, dtype)
        return row + 1, frac

    monkeypatch.setattr(rrtm_lw, "_planck_index", one_row_too_warm)
    # The identity goes red, and by a margin that names the defect.
    with pytest.raises(AssertionError):
        test_the_planck_table_integrates_to_the_stefan_boltzmann_blackbody()
    ratios = [float(_band_summed_planck(t)[0])
              / (_SIGMA_SB * t ** 4 / np.pi)
              for t in (160.0, 288.0)]
    assert ratios[0] > 1.02 and ratios[1] > 1.013, ratios
    # And the solver it feeds runs one kelvin hot everywhere while every
    # signature assertion in this file stays green -- which is exactly
    # why the identity had to be added.
    hot = rrtm_longwave_columns(**_column_block())
    monkeypatch.undo()
    right = rrtm_longwave_columns(**_column_block())
    assert float(np.asarray(hot["glw"])[0]) > \
        float(np.asarray(right["glw"])[0]) * 1.01
    test_a_clear_column_cools_at_every_layer_with_physical_fluxes()


def test_a_surface_hotter_than_the_planck_table_is_clamped_not_crashed(
) -> None:
    """MM5ATM clamps TBOUND to 339.99 (F:3949-3953) so the lookup stays
    inside TOTPLNK's 181 rows; the clamp only works if the row is 0-based.

    A desert skin temperature above 339 K is reachable, and before the
    index base was fixed this raised ``IndexError: index 181 is out of
    bounds for axis 0 with size 181`` instead of clamping.
    """
    kwargs = _column_block()
    kwargs["tsfc"] = np.full(1, 345.0, F)
    result = rrtm_longwave_columns(**kwargs)
    _assert_finite_and_physical(result)
    # The clamp is a ceiling, not a discard: a 345 K surface over the
    # same atmosphere still emits far more than a 288 K one.
    mild = rrtm_longwave_columns(**_column_block())
    assert float(np.asarray(result["olr"])[0]) > \
        float(np.asarray(mild["olr"])[0]) + 20.0
    # The top row is genuinely reachable: 339.99 K maps to the pair
    # (179, 180), the last two rows of the table.
    row, frac = rrtm_lw._planck_index(
        np, np.full(1, rrtm_lw._TBOUND_MAX, F), np.dtype(F))
    assert int(row[0]) == rrtm_lw._TOTPLNK_ROWS - 2
    assert float(frac[0]) == pytest.approx(0.99, abs=1e-4)
    # WRF does not clamp LEVEL or LAYER temperatures at all (see the
    # module docstring); the port's own clip holds the LAST row (340 K),
    # which a 340.0 K TBOUND legally indexes (tests/
    # test_rrtm_lw_wrf461_parity.py), so drive one past the table and
    # check the row, not a flux.
    row, _f = rrtm_lw._planck_index(np, np.full(1, 400.0, F), np.dtype(F))
    assert int(row[0]) == rrtm_lw._TOTPLNK_ROWS - 1
    row, _f = rrtm_lw._planck_index(np, np.full(1, 100.0, F), np.dtype(F))
    assert int(row[0]) == 0


# ---------------------------------------------------------------------------
# Mutation control
# ---------------------------------------------------------------------------


def test_the_signature_assertions_fail_when_the_longwave_is_stubbed(
        monkeypatch) -> None:
    """The physics checks above must be essential, not decorative.

    Two mutations, each aimed at the assertion it should break.  Both
    leave the transfer running and every flux finite and in bounds, so a
    finiteness-only test would sail through either one.
    """
    # (1) Transparent gases: zero every band optical depth.  A clear
    # column can no longer cool.
    real_taumol = rrtm_lw.rrtm_taumol

    def transparent(**kwargs):
        taug, pfrac = real_taumol(**kwargs)
        return np.zeros_like(taug), pfrac

    monkeypatch.setattr(rrtm_lw, "rrtm_taumol", transparent)
    stubbed = rrtm_longwave_columns(**_column_block())
    assert np.isfinite(np.asarray(stubbed["tten"])).all()
    with pytest.raises(AssertionError):
        heating = _heating_k_per_day(stubbed)
        assert np.all(heating < 0.0), heating
    # An atmosphere with no absorption emits nothing downward, which is
    # the tell that the stub reached the transfer rather than being
    # silently ignored.
    assert float(np.asarray(stubbed["glw"])[0]) == pytest.approx(0.0)
    monkeypatch.undo()

    # (2) Invisible clouds: zero TAUCLOUD after MM5ATM builds it.  The
    # cloud-top-cooling / cloud-base-warming couplet must vanish.
    real_mm5atm = rrtm_lw.mm5atm_columns

    def cloudless(*args, **kwargs):
        profile = real_mm5atm(*args, **kwargs)
        profile["taucloud"] = np.zeros_like(profile["taucloud"])
        profile["cldfrac"] = np.zeros_like(profile["cldfrac"])
        return profile

    monkeypatch.setattr(rrtm_lw, "mm5atm_columns", cloudless)
    base, top = 26, 30
    clear, cloudy = _cloud_signature(
        _column_block(), base=base, top=top, water=True)
    _assert_finite_and_physical(cloudy)
    with pytest.raises(AssertionError):
        _assert_cloud_top_cools_and_base_warms(clear, cloudy, base, top)
    # With TAUCLOUD zeroed the seeded condensate leaves no trace at all.
    assert float(np.asarray(cloudy["glw"])[0]) == pytest.approx(
        float(np.asarray(clear["glw"])[0]), rel=1e-6)


# ---------------------------------------------------------------------------
# The shipped adapters
# ---------------------------------------------------------------------------


def _atmosphere_and_state(nz=NZ, ny=2, nx=3):
    p, pw, t, tw, qv, dz = _bottom_up_profile()

    def field3d(column):
        return np.ascontiguousarray(
            np.broadcast_to(column[:, None, None].astype(F),
                            (column.shape[0], ny, nx)).copy())

    p_pa = field3d(p * F(100.0))
    p_interface = field3d(pw * F(100.0))
    z_interface = field3d(
        np.concatenate([[F(0.0)], np.cumsum(np.full(nz, 400.0, F))]))
    atmosphere = {
        "temperature": field3d(t), "pressure": p_pa,
        "p_interface": p_interface, "z_interface": z_interface,
        "dz": field3d(np.full(nz, 400.0, F)),
        "exner": field3d((p / F(1000.0)) ** F(287.0 / 1004.0)),
        "qv": field3d(qv), "qc": np.zeros((nz, ny, nx), F),
        "qi": np.zeros((nz, ny, nx), F),
    }
    fnm = np.full(nz, 0.5, F)
    fnp = np.full(nz, 0.5, F)
    state = SimpleNamespace(elapsed_seconds=0.0, fnm=fnm, fnp=fnp,
                            p_top=float(pw[-1] * 100.0))
    fields = {"tsk": np.full((ny, nx), 288.15, F),
              "emiss": np.full((ny, nx), 0.95, F),
              "albedo": np.full((ny, nx), 0.2, F),
              "glw": np.full((ny, nx), -999.0, F)}
    cfg = SimpleNamespace(bl_pbl_physics=1, icloud_bl=0, dt=60.0,
                          radt=12.0, radt_minutes=12.0)
    return atmosphere, fields, state, cfg


def test_the_composed_adapter_publishes_both_streams_and_owns_glw(
        monkeypatch) -> None:
    """The 1/1 pair through its shipped adapter, on the NumPy shim.

    Runs the real :class:`RRTMDudhiaRadiation` -- column packing, t8w,
    cal_cldfra1, chunking, the merge with Dudhia -- and checks the
    division of labour the composition exists to provide: RRTM supplies
    RTHRATENLW/GLW/OLR, Dudhia supplies the shortwave, and the GLW the
    caller carried in is overwritten rather than passed through.
    """
    monkeypatch.setitem(sys.modules, "cupy", np)
    from woof.core.rrtm_lw import RRTMDudhiaRadiation

    atmosphere, fields, state, cfg = _atmosphere_and_state()
    # The WRF 4.8 solar geometry (the default generation) is a compiled
    # device unit the NumPy shim cannot run; the WRF 4.6.1 generation's
    # host arithmetic serves this test, which is about which scheme owns
    # which stream, not about COSZEN (tests/test_solar_generation_gpu.py
    # grades both generations on the card).
    cfg.solar_generation = "wrf_461"
    ny, nx = fields["tsk"].shape
    latitude = np.full((ny, nx), 39.0, F)
    longitude = np.full((ny, nx), -87.0, F)
    adapter = RRTMDudhiaRadiation(
        datetime(2011, 4, 27, 18), latitude, longitude,
        p_top=state.p_top, icloud=1, swrad_scat=1.0)
    result = adapter(atmosphere=atmosphere, fields=fields, state=state,
                     cfg=cfg)

    assert adapter.publishes_olr is True
    assert adapter.glw_provenance == "scheme"
    for name in ("rthratenlw", "rthratensw", "swdown", "glw", "olr"):
        value = np.asarray(getattr(result, name))
        assert np.isfinite(value).all(), name
    # Longwave from RRTM: a cooling potential-temperature tendency
    # everywhere in this clear profile.
    assert np.all(np.asarray(result.rthratenlw) < 0.0)
    # Shortwave from Dudhia, daytime at this longitude and hour.
    assert np.all(np.asarray(result.rthratensw) >= 0.0)
    assert np.any(np.asarray(result.rthratensw) > 0.0)
    assert np.all(np.asarray(result.swdown) > 0.0)
    # GLW is the scheme's, not the -999 sentinel the caller carried in.
    glw = np.asarray(result.glw)
    assert np.all(glw > 100.0) and np.all(glw < 600.0), glw
    assert np.all(np.asarray(fields["glw"]) == -999.0)
    assert np.all(np.asarray(result.olr) > 50.0)
    identity = adapter.restart_identity
    assert identity["algorithm"] == "wrf-v4.6.1-rrtm-lw+dudhia-sw"
    assert identity["longwave"]["wrf_source"].endswith("RRTMLWRAD/RRTM")


def test_the_longwave_adapter_alone_returns_zero_shortwave(
        monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", np)
    from woof.core.rrtm_lw import RRTMLongwaveRadiation

    atmosphere, fields, state, cfg = _atmosphere_and_state()
    ny, nx = fields["tsk"].shape
    adapter = RRTMLongwaveRadiation(
        datetime(2011, 4, 27, 18), np.full((ny, nx), 39.0, F),
        np.full((ny, nx), -87.0, F), p_top=state.p_top)
    result = adapter(atmosphere=atmosphere, fields=fields, state=state,
                     cfg=cfg)
    assert np.array_equal(np.asarray(result.rthratensw),
                          np.zeros_like(np.asarray(result.rthratensw)))
    assert np.array_equal(np.asarray(result.swdown),
                          np.zeros_like(np.asarray(result.swdown)))
    assert np.all(np.asarray(result.rthratenlw) < 0.0)


def test_the_auto_sized_chunk_follows_free_memory() -> None:
    """``column_chunk=None`` sizes the solver pass from free memory.

    The fixed 512-column default re-dispatched the whole per-chunk CuPy
    op graph hundreds of times per radiation step on a production nest
    (~3.2M kernel launches, ~93% host-dispatch gap -- the task #185
    profile).  The auto-sizer must scale with free memory, allow a smaller
    block when the device is starved, cap at the block's own columns,
    and leave an explicit pin untouched.  Numerics are chunk-invariant,
    which :func:`test_column_chunking_does_not_change_the_answer` pins.
    """
    free = 8 << 30
    nlayers = 60

    class _Pool:
        @staticmethod
        def free_bytes():
            return 0

    class _FakeCupy:
        __name__ = "cupy"
        cuda = SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (free, 24 << 30)))

        @staticmethod
        def get_default_memory_pool():
            return _Pool()

    fit = rrtm_lw.auto_column_chunk(200000, nlayers,
        int(free * rrtm_lw._TRANSIENT_MEMORY_FRACTION))
    assert fit > rrtm_lw.DEFAULT_COLUMN_CHUNK
    resolved = rrtm_lw._resolve_column_chunk(None, 200000, nlayers, _FakeCupy)
    assert resolved == fit
    # Cap: a block smaller than the fit runs in one pass.
    assert rrtm_lw._resolve_column_chunk(None, 96, nlayers, _FakeCupy) == 96
    # No artificial floor: a starved device uses the block that fits.
    free = 1 << 20
    assert (rrtm_lw._resolve_column_chunk(None, 200000, nlayers, _FakeCupy)
            == rrtm_lw.auto_column_chunk(200000, nlayers, free // 2))
    # An explicit pin wins over the fit.
    free = 8 << 30
    assert rrtm_lw._resolve_column_chunk(2048, 200000, nlayers,
                                         _FakeCupy) == 2048
    # Host (NumPy) blocks have no pool to protect: one pass.
    assert rrtm_lw._resolve_column_chunk(None, 777, nlayers, np) == 777


def test_the_auto_size_is_resolved_once_and_never_inside_capture() -> None:
    """The 8d94a2ff7 regression, pinned at the resolver seam.

    The perf fix called ``cudaMemGetInfo`` on EVERY solve, and that call
    is illegal inside CUDA graph capture -- both graph-replay rungs of
    the 2.0 tiles gate raised ``GraphCaptureError`` on it.  The contract
    now is: the memory query runs on eager solves only and its answer is
    cached per ``(ncol, nlayers)``; a captured solve reuses the cache; a
    COLD capture (no eager solve yet) uses the floor without touching
    the query OR the cache, so a later eager call still auto-sizes; an
    explicit pin bypasses all of it.  The fake cupy's ``memGetInfo``
    detonates while "capturing", which is exactly what the CUDA runtime
    does, so a revert of any branch goes red here without a card.
    """
    from woof.core.rrtm_lw import DEFAULT_COLUMN_CHUNK, RRTMLongwaveRadiation

    free = 8 << 30
    nlayers = 60
    capturing = False
    queries = []

    class _Pool:
        @staticmethod
        def free_bytes():
            return 0

    def _mem_get_info():
        if capturing:
            raise RuntimeError(
                "operation not permitted when stream is capturing")
        queries.append(True)
        return free, 24 << 30

    class _Stream:
        @staticmethod
        def is_capturing():
            return capturing

    class _FakeCupy:
        __name__ = "cupy"
        cuda = SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=_mem_get_info),
            get_current_stream=staticmethod(lambda: _Stream()))

        @staticmethod
        def get_default_memory_pool():
            return _Pool()

    adapter = RRTMLongwaveRadiation.__new__(RRTMLongwaveRadiation)
    adapter.column_chunk = None
    adapter._auto_chunk = {}

    fit = rrtm_lw.auto_column_chunk(200000, nlayers,
        int(free * rrtm_lw._TRANSIENT_MEMORY_FRACTION))
    # Eager: resolves from free memory and caches -- ONE query.
    assert adapter._column_chunk_for(200000, nlayers, _FakeCupy) == fit
    assert adapter._column_chunk_for(200000, nlayers, _FakeCupy) == fit
    assert len(queries) == 1
    # Captured with the cache warm: the cached value, and NO query (the
    # fake raises if one is attempted).
    capturing = True
    assert adapter._column_chunk_for(200000, nlayers, _FakeCupy) == fit
    # Cold capture on a fresh adapter: the floor, no query, no cache
    # poisoning -- the next eager call still auto-sizes.
    cold = RRTMLongwaveRadiation.__new__(RRTMLongwaveRadiation)
    cold.column_chunk = None
    cold._auto_chunk = {}
    assert cold._column_chunk_for(200000, nlayers,
                                  _FakeCupy) == DEFAULT_COLUMN_CHUNK
    assert cold._column_chunk_for(96, nlayers, _FakeCupy) == 96
    assert cold._auto_chunk == {}
    capturing = False
    assert cold._column_chunk_for(200000, nlayers, _FakeCupy) == fit
    # An explicit pin never consults anything, capturing or not.
    pinned = RRTMLongwaveRadiation.__new__(RRTMLongwaveRadiation)
    pinned.column_chunk = 2048
    pinned._auto_chunk = {}
    capturing = True
    assert pinned._column_chunk_for(200000, nlayers, _FakeCupy) == 2048


def test_the_longwave_solve_runs_under_stream_capture_bit_exact() -> None:
    """The regression test the graph-replay rungs paid for, on a card.

    An eager solve, then the SAME solve captured into a CUDA graph and
    replayed.  Before the fix the captured solve raised out of
    ``cudaMemGetInfo`` at :func:`_resolve_column_chunk`; now the capture
    must complete and the replayed bytes must equal the eager bytes,
    which is the chunk-invariance guarantee doing its job across the
    eager/captured boundary.
    """
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    from woof.core.rrtm_lw import RRTMLongwaveRadiation

    atmosphere, fields, state, cfg = _atmosphere_and_state()
    atmosphere = {name: cp.asarray(value)
                  for name, value in atmosphere.items()}
    fields = {name: cp.asarray(value) for name, value in fields.items()}
    # A real DomainState's fnm/fnp are DEVICE arrays (state.py allocates
    # them with the state's own module), so the adapter's cp.asarray on
    # them is a no-op in production.  The CPU fixture's NumPy pair would
    # instead be a pageable H2D copy, which capture refuses -- that would
    # test a state shape the model never runs.
    state.fnm = cp.asarray(state.fnm, dtype=cp.float32)
    state.fnp = cp.asarray(state.fnp, dtype=cp.float32)
    ny, nx = fields["tsk"].shape
    adapter = RRTMLongwaveRadiation(
        datetime(2011, 4, 27, 18), np.full((ny, nx), 39.0, F),
        np.full((ny, nx), -87.0, F), p_top=state.p_top)
    call = dict(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        # Eager warmup: caches the auto-sized chunk, the device constant
        # tables and the pool blocks, exactly as a real run's first
        # radiation call precedes any tilestream capture.
        eager_tten, eager_glw, eager_olr = adapter.longwave(**call)
        stream.synchronize()
        stream.begin_capture()
        try:
            captured_tten, captured_glw, captured_olr = adapter.longwave(
                **call)
        finally:
            graph = stream.end_capture()
        graph.launch(stream)
        stream.synchronize()

    assert cp.array_equal(captured_tten, eager_tten)
    assert cp.array_equal(captured_glw, eager_glw)
    assert cp.array_equal(captured_olr, eager_olr)

    # COLD capture: a fresh adapter whose first solve is the captured one
    # takes the floor fallback instead of the memory query, and the
    # bytes still match by chunk invariance.
    cold = RRTMLongwaveRadiation(
        datetime(2011, 4, 27, 18), np.full((ny, nx), 39.0, F),
        np.full((ny, nx), -87.0, F), p_top=state.p_top)
    with stream:
        stream.begin_capture()
        try:
            cold_tten, cold_glw, cold_olr = cold.longwave(**call)
        finally:
            graph = stream.end_capture()
        graph.launch(stream)
        stream.synchronize()
    assert cold._auto_chunk == {}, (
        "a capture-time fallback must not poison the eager cache")
    assert cp.array_equal(cold_tten, eager_tten)
    assert cp.array_equal(cold_glw, eager_glw)
    assert cp.array_equal(cold_olr, eager_olr)


def test_column_chunking_does_not_change_the_answer(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", np)
    from woof.core.rrtm_lw import RRTMLongwaveRadiation

    atmosphere, fields, state, cfg = _atmosphere_and_state()
    ny, nx = fields["tsk"].shape
    kwargs = dict(latitude_deg=np.full((ny, nx), 39.0, F),
                  longitude_deg=np.full((ny, nx), -87.0, F),
                  p_top=state.p_top)
    whole = RRTMLongwaveRadiation(
        datetime(2011, 4, 27, 18), column_chunk=4096, **kwargs)
    split = RRTMLongwaveRadiation(
        datetime(2011, 4, 27, 18), column_chunk=2, **kwargs)
    call = dict(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    assert np.array_equal(np.asarray(whole(**call).glw),
                          np.asarray(split(**call).glw))
    assert np.array_equal(np.asarray(whole(**call).rthratenlw),
                          np.asarray(split(**call).rthratenlw))


# ---------------------------------------------------------------------------
# Admission: config, namelist import, registry
# ---------------------------------------------------------------------------


def _radiation_config(lw: int, sw: int) -> RunConfig:
    return RunConfig(
        nx=8, ny=8, nz=20, dx=12000.0, dy=12000.0, dt=60.0,
        run_seconds=600.0, ztop=20000.0, moist=True, mp_physics=6, ra_physics=0, ra_lw_physics=lw,
        ra_sw_physics=sw, sf_sfclay_physics=1, sf_surface_physics=2,
        bl_pbl_physics=1)


def test_config_admits_classic_longwave_with_each_implemented_spectrum() -> None:
    # The shared factory now composes spectra; the old pair-only guard is
    # retired by that implementation, not part of classic LW's contract.
    for shortwave in (0, 1, 4, 90):
        validate_run_config(_radiation_config(1, shortwave))


def _classic_namelist_pair(tmp_path: Path):
    wps = tmp_path / "namelist.wps"
    inp = tmp_path / "namelist.input"
    wps.write_text("""\
&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date = '2011-04-27_12:00:00',
 end_date = '2011-04-27_18:00:00',
 interval_seconds = 21600,
/
&geogrid
 parent_id = 1,
 parent_grid_ratio = 1,
 i_parent_start = 1,
 j_parent_start = 1,
 e_we = 101,
 e_sn = 81,
 geog_data_res = 'default',
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat = 39.7,
 ref_lon = -83.9,
 truelat1 = 30.0,
 truelat2 = 60.0,
 stand_lon = -83.9,
 geog_data_path = '/geog',
/
""", encoding="utf-8")
    inp.write_text("""\
&time_control
 run_hours = 6,
 start_year = 2011,
 start_month = 4,
 start_day = 27,
 start_hour = 12,
 end_year = 2011,
 end_month = 4,
 end_day = 27,
 end_hour = 18,
 input_from_file = .true.,
 history_interval = 60,
 restart_interval = 60,
/
&domains
 time_step = 60,
 max_dom = 1,
 e_we = 101,
 e_sn = 81,
 e_vert = 9,
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,
 p_top_requested = 5000,
 dx = 12000.0,
 dy = 12000.0,
 grid_id = 1,
 parent_id = 0,
 i_parent_start = 1,
 j_parent_start = 1,
 parent_grid_ratio = 1,
 parent_time_step_ratio = 1,
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = 6,
 ra_lw_physics = 1,
 ra_sw_physics = 1,
 radt = 12,
 icloud = 1,
 swrad_scat = 1.0,
 sf_sfclay_physics = 91,
 sf_surface_physics = 2,
 bl_pbl_physics = 1,
 bldt = 0,
 cu_physics = 1,
 cudt = 5,
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 w_damping = 1,
 epssm = 0.5,
 diff_opt = 2,
 km_opt = 4,
 mix_full_fields = .true.,
 diff_6th_opt = 2,
 diff_6th_factor = 0.12,
 diff_6th_slopeopt = 1,
 base_temp = 290.0,
 damp_opt = 3,
 zdamp = 5000.0,
 dampcoef = 0.2,
 khdif = 0,
 kvdif = 0,
 non_hydrostatic = .true.,
 use_theta_m = 0,
 moist_adv_opt = 1,
/
&bdy_control
 spec_bdy_width = 5,
 specified = .true.,
 nested = .false.,
/
""", encoding="utf-8")
    return wps, inp


def test_a_classic_wrf_namelist_imports_rrtm_natively(tmp_path) -> None:
    """ra_lw_physics=1 must survive import as itself, not a substitute."""

    toml_text, report = import_namelists(
        *_classic_namelist_pair(tmp_path), name="classic-rrtm-dudhia")
    assert "ra_lw_physics = 1" in toml_text
    assert "ra_sw_physics = 1" in toml_text
    radiation_substitutions = [
        s for s in report.substitutions
        if "ra_lw_physics" in s.key or "ra_sw_physics" in s.key]
    assert radiation_substitutions == [], radiation_substitutions


def test_the_registry_row_is_accurate_about_the_oracle() -> None:
    """The longwave half has a WRF column oracle; the row says so, and no more.

    The maturity stays implemented-unverified: that rung is about a matched
    forecast trajectory, and the Dudhia shortwave half of this pair has no
    oracle in this file's scope.
    """
    registry = json.loads(
        (MODEL / "woof" / "physics_registry_v2.json").read_text(
            encoding="utf-8"))
    row = registry["components"]["radiation"]["options"]["wrf-rrtm-dudhia"]
    assert row["implemented"] is True
    assert row["maturity"] == "implemented-unverified"
    assert row["reachability"]["state"] == "template"
    assert row["selectors"] == {"ra_lw_physics": 1, "ra_sw_physics": 1}
    warnings = " ".join(row["warnings"])
    assert "NO ORACLE COMPARISON" not in warnings
    assert "BITWISE AGAINST WRF ON ITS COLUMN ORACLE" in warnings
    assert "tests/test_rrtm_lw_wrf461_parity.py" in warnings
    assert (MODEL / "tests" / "test_rrtm_lw_wrf461_parity.py").is_file()
    # What the oracle does NOT cover stays stated.
    assert "NOT COVERED" in warnings
    assert "matched WRF forecast trajectory" in warnings
    route = registry["runner_routes"]["tools.prepared_domain_tree_forecast"]
    assert "wrf-rrtm-dudhia" in route["allowed_component_options"]["radiation"]
