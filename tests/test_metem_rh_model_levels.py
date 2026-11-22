"""met_em with RH and no SPECHUMD on levels that are not isobaric.

real.exe takes such a file: with neither FLAG_QV nor FLAG_SH it forms qv
from RH, TT and the 3-D ``p_gc`` with ``rh_to_mxrat1``
(module_initialize_real.F:1113-1142) and forces ``use_sh_qv = .FALSE.``
(:1119).  ``woof run --met-em`` refused it ("level 1 of PRES spans
1.993e+04 Pa ... not an isobaric level"), which is what the outside
runner's Arwen Global model-level sets are.  The reader now forms the
specific humidity that conversion implies and carries it on the FLAG_SH
lane, which reads PRES per column.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.ingest import real as R


def _bolton_specific_humidity(t, p, rh):
    """An independent transcription of WRF rh_to_mxrat1 (liquid, method 1).

    module_initialize_real.F:7404-7425: RH clipped to 0..100;
    es = 0.01 RH 0.6112*10 exp(17.67 (T-273.15)/(T-29.65)) hPa;
    qv = max(0.622 es / (p/100 - es), 1e-6); :7490-7498 with the Registry
    defaults, qv = 3e-6 where p < 10000 Pa and qv > 1e-5; then
    q = qv / (1 + qv).
    """
    t = np.asarray(t, np.float64)
    p = np.asarray(p, np.float64)
    rh = np.clip(np.asarray(rh, np.float64), 0.0, 100.0)
    es = 0.01 * rh * 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qv = np.where(es < p / 100.0,
                  np.maximum(0.622 * es / (p / 100.0 - es), 1.0e-6), 1.0e-6)
    qv = np.where((p < 10000.0) & (qv > 1.0e-5), 3.0e-6, qv)
    return qv / (1.0 + qv)


def test_derived_specific_humidity_is_wrfs_rh_to_mxrat1():
    rng = np.random.default_rng(1122)
    t = rng.uniform(200.0, 310.0, (6, 4, 5))
    p = rng.uniform(5000.0, 105000.0, t.shape)
    rh = rng.uniform(-3.0, 104.0, t.shape)
    got = R.relative_humidity_to_specific_humidity(t, p, rh)
    np.testing.assert_allclose(got, _bolton_specific_humidity(t, p, rh),
                               rtol=1e-12, atol=0.0)


def test_the_specific_humidity_lane_gives_back_the_clipped_rh_real_exe_interpolates():
    """real.exe interpolates rh_gc after rh_to_mxrat1 clipped it in place.

    The FLAG_SH lane re-diagnoses RH from qv = q / (1 - q); on the derived
    q that is the clipped RH again, to float32 storage of q.
    """
    rng = np.random.default_rng(7)
    t = rng.uniform(225.0, 305.0, (8, 6, 6))
    p = np.sort(rng.uniform(15000.0, 104000.0, t.shape), axis=0)[::-1]
    rh = rng.uniform(1.0, 103.0, t.shape)
    q = R.relative_humidity_to_specific_humidity(t, p, rh).astype(np.float32)
    back = R._mixing_ratio_to_relative_humidity(
        t, p, R._specific_humidity_to_mixing_ratio(q))
    np.testing.assert_allclose(back, np.clip(rh, 0.0, 100.0),
                               rtol=2e-6, atol=1e-5)


def _rh_lane_and_derived_lane(rh_profile):
    from test_metem_differential import _synthetic
    snapshot, cfg, coord, terrain, orography = _synthetic(400.0, 600.0)
    fields = dict(snapshot.fields)
    shape = fields["TT"].shape
    fields["RH"] = np.broadcast_to(
        np.asarray(rh_profile, np.float32)[:, None, None], shape).copy()
    rh_snapshot = replace(snapshot, fields=fields)
    pressure = np.broadcast_to(
        snapshot.levels_hpa[:, None, None] * 100.0, shape).astype(np.float32)
    derived = {name: value for name, value in fields.items()
               if name not in ("RH", "RH2")}
    derived["PRES"] = pressure
    derived["SPFH"] = R.relative_humidity_to_specific_humidity(
        fields["TT"], pressure, fields["RH"]).astype(np.float32)
    derived["Q2"] = R.relative_humidity_to_specific_humidity(
        fields["T2"], fields["PSFC"], fields["RH2"]).astype(np.float32)
    derived_snapshot = replace(snapshot, fields=derived)
    common = dict(source_orography=orography, p_top=5000.0,
                  preprocess_backend="cpu", state_backend="preprocess")
    by_rh = R.initialize_real(rh_snapshot, cfg, coord, terrain, **common)
    by_q = R.initialize_real(
        derived_snapshot, cfg, coord, terrain, analyzed_species=(),
        use_sh_qv=False, flag_sh_surface_fallback=False, **common)
    return by_rh.state, by_q.state


def test_on_isobaric_levels_the_derived_lane_reproduces_the_rh_lane():
    """One file, two roads: RH on its flat ladder, and q on the 3-D PRES.

    On a pressure-level file both are real.exe's RH path.  With a dry
    stratosphere (no stratospheric cap fires, as in both real Arwen Global
    sets: 0 capped cells) the two states are the same bytes.
    """
    by_rh, by_q = _rh_lane_and_derived_lane([92.0, 71.0, 38.0, 12.0, 4.0, 2.0])
    for name in ("thp", "qv", "mup", "php"):
        assert np.array_equal(np.asarray(getattr(by_rh, name)),
                              np.asarray(getattr(by_q, name))), name


def test_where_the_stratospheric_cap_fires_only_the_top_vapour_differs():
    """55 % RH at 80 and 40 hPa: rh_to_mxrat1 caps those source qv.

    The moisture integral reads the capped qv on both roads, so theta and
    the dry column mass are the same bytes; the RH the derived lane
    re-diagnoses at the capped levels is that of 3e-6, so vapour differs
    on the top two model levels, the ones those source levels reach
    (real.exe keeps the RH it calls "probably missing" there); below them
    only the moist pressure it feeds back moves qv, by under 1e-4 of it.
    """
    by_rh, by_q = _rh_lane_and_derived_lane([55.0] * 6)
    for name in ("thp", "mup"):
        assert np.array_equal(np.asarray(getattr(by_rh, name)),
                              np.asarray(getattr(by_q, name))), name
    a = np.asarray(by_rh.qv, np.float64)
    b = np.asarray(by_q.qv, np.float64)
    assert float(np.max(np.abs(a[-2:] - b[-2:]))) < 5e-5
    below = np.abs(a[:-2] - b[:-2]) / np.abs(a[:-2])
    assert float(np.max(below)) < 1e-4


def test_rh_files_follow_real_exe_use_sh_qv_false_and_skip_the_flag_sh_surface_test():
    from woof.ingest.metem import RH_DERIVED_LANE
    from woof.metem_door import metgrid_initialization_controls
    run = SimpleNamespace(controls={"domains": {"use_sh_qv": [True]}})

    def case(lane):
        return SimpleNamespace(
            path=Path("met_em.d01.test"), lane=lane,
            snapshot=SimpleNamespace(levels_hpa=np.array([950., 700., 300.]), fields={}),
            attributes={"FLAG_PSFC": 1, "FLAG_SOILHGT": 1},
            geometry={"num_metgrid_levels": 4})
    run.controls["domains"]["sfcp_to_sfcp"] = [True]
    derived = metgrid_initialization_controls(case(RH_DERIVED_LANE), run)
    assert derived["use_sh_qv"] is False
    assert derived["flag_sh_surface_fallback"] is False
    plain = metgrid_initialization_controls(case("rh"), run)
    assert plain["use_sh_qv"] is False and "flag_sh_surface_fallback" not in plain
    specific = metgrid_initialization_controls(case("specific_humidity"), run)
    assert specific["use_sh_qv"] is True and "flag_sh_surface_fallback" not in specific


def test_an_rh_file_is_priced_with_the_pressure_the_derived_lane_uploads():
    from woof.metem_door import metgrid_analysis_shapes
    shapes = {name: (1, 7, 11, 13) for name in ("TT", "GHT", "RH", "PRES")}
    shapes.update(UU=(1, 7, 11, 14), VV=(1, 7, 12, 13),
                  PSFC=(1, 11, 13), SOILHGT=(1, 11, 13))
    metadata = SimpleNamespace(path=Path("met_em.test"), nx=13, ny=11,
                               global_attributes={}, variables=shapes)
    priced = metgrid_analysis_shapes(metadata)
    assert priced["PRES"] == (6, 11, 13)
    assert priced["RH"] == (6, 11, 13) and priced["RH2"] == (11, 13)
