"""RRTM longwave (ra_lw_physics=1) against the byte-unmodified WRF v4.6.1.

``tools/rrtm_lw_wrf461_oracle`` compiles ``phys/module_ra_rrtm.F`` at the
pinned commit, unmodified, at -O0 with gfortran 13.3.0 / glibc 2.39, and
drives it on 79 columns in 7 RRTMLWRAD calls: deep convection, stable nights
(clear, fog, dry desert), arctic and mid-latitude snow and ice, tropical,
stratocumulus and storm-track ocean, 580-705 hPa terrain, and edge cases (a
1085 hPa surface, a 342 K surface WRF resets to 339.99 K, a surface at exactly
340 K, a hot desert layer clamping JT and INDSELF, layer pressures whose
logarithm lands exactly on SETCOEF's 4.56/5.76/6.62 switches, TAUCLOUD around
MM5ATM's 0.01 overcast test, negative and subnormal mixing ratios, integer-
kelvin temperatures, icloud = 0, Kessler's registry flags).  The model top
takes 25, 13, 3, 1 and 0 Cavallo buffer layers across the sets.

MEASURED on that fixture (development branch, 2026-10-07): every word WOOF
writes -- RTHRATEN, RTHRATENC, GLW, OLR, and every MM5ATM / SETCOEF / GASABS /
RTRN intermediate and every rrtm_lookuptable table -- is WRF's word, on the
NumPy path, on the GPU under GPUWM_WRF_EXACT=1 and on the GPU under default
arithmetic.  The one default-arithmetic difference is TAUCLOUD on 10 layers of
the subnormal-mixing-ratio column (two sets), where the card flushes a
subnormal cloud or snow path and TAUCLOUD comes back up to 2.3e-26 smaller;
no output moves.  Before the fixes in this lane the same comparison gave
OLR 2.28 W m-2 and RTHRATEN 1.3e-4 K s-1 off on the 340 K column and every
output off by ULPs elsewhere; the causes were BPADE and FLUXFAC formed in
binary64, NumPy's / CUDA's EXP, LOG and ** in place of WRF's float32 libm,
pairwise and tree sums in place of WRF's accumulation loops, P/100 spelled
P*0.01, the trace gases evaluated in binary64, one association in eight
TAUGB bands, and a Planck clamp that read the 339 K row at 340 K.

Stage fields that WRF never writes are excluded where they occur: SELFFAC,
SELFFRAC and INDSELF above LAYTROP (SETCOEF skips them and GASABS never reads
them).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from tools.rrtm_lw_wrf461_oracle.oracle_io import read_inputs

FIXTURE = Path(__file__).resolve().parent / "data" / "oracles" / "rrtm_lw"
FINAL = ("rthraten", "rthratenc", "glw", "olr")


def _fixture():
    sets = read_inputs(FIXTURE / "columns.bin")
    labels = json.loads((FIXTURE / "labels.json").read_text(encoding="utf-8"))
    names = list(labels)
    with np.load(FIXTURE / "wrf461-O0.npz") as archive:
        wrf = {key: archive[key] for key in archive.files}
    return sets, names, labels, wrf


def _run(cs, xp, stages=True):
    from woof.core.rrtm import rrtm_default_trace_gases
    from woof.core.rrtm_lw import rrtm_layer_count, rrtmlwrad_columns

    f = {k: xp.asarray(v) for k, v in cs.fields.items()}
    gases = rrtm_default_trace_gases(cs.yr)
    out = rrtmlwrad_columns(
        t3d=f["t3d"], p3d=f["p3d"], pi3d=f["pi3d"], dz8w=f["dz8w"],
        qv=f["qv"], qc=f["qc"], qr=f["qr"], qi=f["qi"], qs=f["qs"],
        qg=f["qg"], cldfra=f["cldfra"], p8w=f["p8w"], t8w=f["t8w"],
        tsk=f["tsk"], emiss=f["emiss"],
        nlayers=rrtm_layer_count(cs.nz, cs.p_top), co2vmr=gases["co2"],
        n2ovmr=gases["n2o"], ch4vmr=gases["ch4"], icloud=cs.icloud,
        stages=stages)

    def host(value):
        if isinstance(value, dict):
            return {k: host(v) for k, v in value.items()}
        return value.get() if hasattr(value, "get") else np.asarray(value)

    return host(out)


def _stage_pairs(name, wrf, woof):
    """(field, WRF array, WOOF array, mask) for every graded intermediate."""
    st = woof["stages"]
    laytrop = wrf[f"{name}/laytrop"]
    nlay = wrf[f"{name}/pavel"].shape[1]
    trop = np.arange(1, nlay + 1)[None, :] <= laytrop[:, None]
    pairs = []
    for field in ("o3prof", "pavel", "tavel", "pz", "tz", "cldfrac",
                  "taucloud", "coldry", "tbound", "colh2o", "colco2",
                  "colo3", "coln2o", "colch4", "colo2", "co2mult", "fac00",
                  "fac01", "fac10", "fac11", "forfac", "jp", "jt", "jt1",
                  "laytrop", "layswtch", "laylow", "totdflux", "totuflux",
                  "htr", "htrc"):
        pairs.append((field, wrf[f"{name}/{field}"], st[field], None))
    pairs.append(("wkl", wrf[f"{name}/wkl"], st["wkl"][:, :, :7], None))
    for field in ("selffac", "selffrac", "indself"):
        pairs.append((field, wrf[f"{name}/{field}"], st[field], trop))
    pairs.append(("tten", wrf[f"{name}/tten_topdown"][:, ::-1], st["tten"],
                  None))
    pairs.append(("ttenc", wrf[f"{name}/ttenc_topdown"][:, ::-1],
                  st["ttenc"], None))
    return pairs


def _bits(array):
    array = np.asarray(array)
    if array.dtype.kind == "f":
        return np.ascontiguousarray(array.astype(np.float32)).view(np.int32)
    return np.asarray(array, dtype=np.int64)


def _mismatches(want, got, mask=None):
    w, g = _bits(want), _bits(got)
    if w.shape != g.shape:
        return f"shape {w.shape} != {g.shape}"
    if mask is not None:
        w, g = w[mask], g[mask]
    bad = int((w != g).sum())
    return bad or None


def _grade(xp, *, flushed_ok=False):
    sets, names, _labels, wrf = _fixture()
    failures = []
    for cs, name in zip(sets, names):
        woof = _run(cs, xp)
        for field in FINAL:
            bad = _mismatches(wrf[f"{name}/{field}"], woof[field])
            if bad:
                failures.append(f"{name}/{field}: {bad} words differ")
        for field, want, got, mask in _stage_pairs(name, wrf, woof):
            if flushed_ok and field == "taucloud":
                # Default arithmetic flushes subnormals: a layer whose
                # cloud, rain, ice or snow mixing ratio is subnormal loses
                # that path (RO*Q flushes before the 0.75 power) and its
                # TAUCLOUD comes back smaller, by at most ~2e-26 here.  That
                # is the only difference allowed, it must stay far below
                # MM5ATM's 0.01 overcast test, and no output may move
                # (checked above and below).
                w = np.asarray(want, np.float32)
                g = np.asarray(got, np.float32)
                differ = w.view(np.int32) != g.view(np.int32)
                tiny = np.finfo(np.float32).tiny
                subq = np.zeros_like(differ)
                for species in ("qc", "qr", "qi", "qs"):
                    q = cs.fields[species]
                    subq[:, :cs.nz] |= (q != 0) & (np.abs(q) < tiny)
                if (np.any(differ & ~subq)
                        or np.abs(w[differ] - g[differ]).max(initial=0.0)
                        > 1.0e-20):
                    failures.append(f"{name}/taucloud: non-flush difference")
                continue
            bad = _mismatches(want, got, mask)
            if bad:
                failures.append(f"{name}/{field}: {bad}")
    return failures


def test_every_output_and_stage_is_wrfs_word_on_the_numpy_path():
    assert _grade(np) == []


def test_the_lookup_tables_are_wrfs():
    from woof.core.rrtm_lw import RRTM_BPADE, RRTM_FLUXFAC
    from woof.core.rrtm_tables import load_rrtm_lw_tables

    _sets, _names, _labels, wrf = _fixture()
    t = load_rrtm_lw_tables()
    assert _mismatches(wrf["tables/BPADE"], [RRTM_BPADE]) is None
    assert _mismatches(wrf["tables/FLUXFAC"], [RRTM_FLUXFAC]) is None
    assert _mismatches(wrf["tables/HEATFAC"], t.statics["HEATFAC"][:1]) is None
    for name, mine in (("TAU", t.tau), ("TF", t.tf), ("TRANS", t.trans),
                       ("CORR1", t.corr1), ("CORR2", t.corr2),
                       ("RWGT", t.rwgt)):
        assert _mismatches(wrf[f"tables/{name}"], mine) is None, name
    for band, arr in t.absa.items():
        assert _mismatches(wrf[f"tables/ABSA{band}"], arr.T) is None, band
    for band, arr in t.absb.items():
        assert _mismatches(wrf[f"tables/ABSB{band}"], arr.T) is None, band
    for band, arr in t.selfrefc.items():
        assert _mismatches(wrf[f"tables/SELFREFC{band}"], arr.T) is None
    graded = 0
    for name, arr in t.combined.items():
        key = f"tables/{name}"
        if key in wrf:
            assert _mismatches(wrf[key].reshape(-1), arr.reshape(-1)) is None
            graded += 1
    assert graded == len(t.combined)


def test_the_trace_gases_and_buffer_count_are_wrfs():
    from woof.core.rrtm import rrtm_default_trace_gases
    from woof.core.rrtm_lw import rrtm_layer_count

    sets, names, _labels, wrf = _fixture()
    years = set()
    for cs, name in zip(sets, names):
        gases = rrtm_default_trace_gases(cs.yr)
        want = wrf[f"{name}/gases_f64"]
        got = np.asarray([gases["co2"], gases["n2o"], gases["ch4"]])
        assert want.tobytes() == got.tobytes(), (name, want, got)
        assert rrtm_layer_count(cs.nz, cs.p_top) == int(
            wrf[f"{name}/_header"][2])
        years.add(cs.yr)
    assert len(years) >= 3


def test_a_340_kelvin_surface_reads_the_last_planck_row():
    """TBOUND = 340.0 is legal (only > 340 resets) and indexes row 181."""
    from woof.core.rrtm_lw import _planck_bands, _planck_index
    from woof.core.rrtm_tables import load_rrtm_lw_tables

    t = load_rrtm_lw_tables()
    totplnk = t.statics["TOTPLNK"].reshape((181, 16), order="F")
    delwave = t.statics["DELWAVE"]
    temps = np.asarray([340.0, 339.0, 339.5], np.float32)
    row, frac = _planck_index(np, temps, temps.dtype)
    assert row.tolist() == [180, 179, 179]
    got = _planck_bands(totplnk, delwave, row, frac)
    assert got[0].tobytes() == (delwave * totplnk[180]).tobytes()
    assert got[1].tobytes() == (delwave * totplnk[179]).tobytes()


def test_the_sums_run_in_fortran_order():
    from woof.core.rrtm_lw import _sequential_sum

    rng = np.random.default_rng(7)
    values = (rng.standard_normal((64, 140)) * 10.0 ** rng.integers(
        -6, 6, (64, 140))).astype(np.float32)
    want = np.zeros(64, np.float32)
    for j in range(140):
        want = (want + values[:, j]).astype(np.float32)
    assert _sequential_sum(np, values).tobytes() == want.tobytes()
    # negative control: NumPy's pairwise sum is a different order
    assert values.sum(axis=1).tobytes() != want.tobytes()


@pytest.mark.gpu
@requires_gpu
def test_every_output_and_stage_is_wrfs_word_on_the_gpu():
    import cupy as cp

    from woof import wrf_exact

    assert _grade(cp, flushed_ok=not wrf_exact.ENABLED) == []


@pytest.mark.gpu
@requires_gpu
def test_the_gpu_sum_runs_in_fortran_order():
    import cupy as cp

    from woof.core.rrtm_lw import _sequential_sum

    rng = np.random.default_rng(11)
    values = (rng.standard_normal((257, 140)) * 10.0 ** rng.integers(
        -6, 6, (257, 140))).astype(np.float32)
    want = np.zeros(257, np.float32)
    for j in range(140):
        want = (want + values[:, j]).astype(np.float32)
    got = _sequential_sum(cp, cp.asarray(values)).get()
    assert got.tobytes() == want.tobytes()
    assert cp.asarray(values).sum(axis=1).get().tobytes() != want.tobytes()


def test_stock_wrf_moves_only_taucloud_on_this_fixture():
    """WRF's own -O2 -ftree-vectorize build calls libmvec's vector powf.

    Measured: 5 TAUCLOUD words move by 1 ULP against the -O0 reference and
    every output stays identical, so WOOF equals stock WRF's outputs too.
    """
    _sets, names, _labels, wrf = _fixture()
    with np.load(FIXTURE / "wrf461-O2vec-final.npz") as archive:
        o2 = {key: archive[key] for key in archive.files}
    moved = 0
    for name in names:
        for field in FINAL:
            assert _mismatches(wrf[f"{name}/{field}"],
                               o2[f"{name}/{field}"]) is None
        moved += _mismatches(wrf[f"{name}/taucloud"],
                             o2[f"{name}/taucloud"]) or 0
    assert moved == 5
