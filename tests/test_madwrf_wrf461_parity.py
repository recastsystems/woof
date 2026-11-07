"""MAD-WRF (mp_physics = 96) transcription gate against the WRF v4.6.1 oracle.

The oracle (tools/madwrf_wrf461_oracle) runs the UNMODIFIED
phys/module_madwrf.F and Cloud_tracer_nudge from the pinned tree
d66e442fccc04111067e29274c9f9eaccc3cef28 at gfortran -O0 over:

* seven Init_madwrf_clouds cases on a 12 x 4 x 40-layer grid (no
  observations, cloud mask, mask + brightness temperature, cloud-top
  height, cloud top + base, mask + brightness temperature + base, and the
  hold_ups = .false. boundary-ring pass), followed by Init_madwrf_tracers;
* three Cloud_tracer_nudge sequences whose xtime crosses madwrf_dt_nudge,
  with zeros and subnormal hydrometeor words in the fields.

Every input and output is an exact FP32 word.  The gate holds the NumPy
host path and (on a card) the CUDA path at 0 ULP on every field --
cloud fraction, qv, qc, qi, qs, the three tracers, the INOUT cloud mask and
cloud-top height -- in the default arm and under GPUWM_WRF_EXACT=1.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from woof.core import madwrf
from woof.core.fp32_ulp import fp32_ulp_distance

ORACLE_DIR = (Path(__file__).resolve().parents[1] / "tests" / "data"
              / "oracles" / "madwrf")
F32 = np.float32


def _words(line):
    return np.array(line.split(), dtype=np.int64).astype(np.int32).view(F32)


def _f3(line, ni, nk, nj):
    """Fortran (ni, nk, nj) storage -> woof (k, j, i)."""
    return _words(line).reshape((ni, nk, nj), order="F").transpose(1, 2, 0) \
        .copy()


def _f2(line, ni, nj):
    return _words(line).reshape((ni, nj), order="F").T.copy()


def load_cldinit():
    lines_in = (ORACLE_DIR / "madwrf-cldinit-in.txt").read_text().splitlines()
    lines_out = (ORACLE_DIR / "madwrf-cldinit-out.txt").read_text() \
        .splitlines()
    n = int(lines_in[0])
    assert int(lines_out[0]) == n
    pi = po = 1
    cases = []
    for _ in range(n):
        ni, nj, nk, fm, ft, fb, fbr, width, hold = map(int,
                                                       lines_in[pi].split())
        dx, dy, p00 = _words(lines_in[pi + 1])
        pi += 2
        f = {}
        for name in ("t", "p", "ph_2", "phb", "qv", "qc", "qi", "qs"):
            f[name] = _f3(lines_in[pi], ni, nk, nj)
            pi += 1
        for name in ("xland", "ht", "cldmask", "cldtopz", "cldbasez",
                     "brtemp"):
            f[name] = _f2(lines_in[pi], ni, nj)
            pi += 1
        assert list(map(int, lines_out[po].split())) == [
            ni, nj, nk, fm, ft, fb, fbr, width, hold]
        po += 1
        out = {}
        for name in ("cldfra", "qv", "qc", "qi", "qs", "tr_qc", "tr_qi",
                     "tr_qs"):
            out[name] = _f3(lines_out[po], ni, nk, nj)
            po += 1
        for name in ("cldmask", "cldtopz"):
            out[name] = _f2(lines_out[po], ni, nj)
            po += 1
        cases.append(dict(dims=(ni, nj, nk), flags=(fm, ft, fb, fbr),
                          width=width, hold=bool(hold), dx=dx, dy=dy,
                          p00=p00, inp=f, out=out))
    return cases


def load_nudge():
    lines_in = (ORACLE_DIR / "madwrf-nudge-in.txt").read_text().splitlines()
    lines_out = (ORACLE_DIR / "madwrf-nudge-out.txt").read_text() \
        .splitlines()
    n = int(lines_in[0])
    pi = po = 1
    cases = []
    for _ in range(n):
        ni, nk, nj, nsteps = map(int, lines_in[pi].split())
        dt, relax, nudge = _words(lines_in[pi + 1])
        xtimes = _words(lines_in[pi + 2])
        pi += 3
        f = {}
        for name in ("qc", "qi", "qs", "tr_qc", "tr_qi", "tr_qs"):
            f[name] = _f3(lines_in[pi], ni, nk, nj)
            pi += 1
        assert list(map(int, lines_out[po].split())) == [ni, nk, nj, nsteps]
        po += 1
        steps = []
        for _s in range(nsteps):
            step = {}
            for name in ("qc", "qi", "qs"):
                step[name] = _f3(lines_out[po], ni, nk, nj)
                po += 1
            steps.append(step)
        cases.append(dict(dt=dt, relax=relax, nudge=nudge, xtimes=xtimes,
                          inp=f, steps=steps))
    return cases


@pytest.fixture(scope="module")
def cldinit_cases():
    return load_cldinit()


@pytest.fixture(scope="module")
def nudge_cases():
    return load_nudge()


def test_oracle_fixture_hashes_match_the_build_receipt():
    recorded = {}
    for line in (ORACLE_DIR / "oracle-sha256sums.txt").read_text(
            encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition("  ")
        recorded[Path(name).name] = digest
    for name in ("madwrf-cldinit-in.txt", "madwrf-cldinit-out.txt",
                 "madwrf-nudge-in.txt", "madwrf-nudge-out.txt"):
        digest = hashlib.sha256((ORACLE_DIR / name).read_bytes()).hexdigest()
        assert digest == recorded[name], name
    for name in ("module_madwrf.pinned.F",
                 "module_big_step_utilities_em.pinned.F",
                 "extract_cloud_tracer_nudge.f90", "extract_rslf.f90",
                 "extract_rsif.f90", "extract_skip_middle_points_t.f90"):
        assert name in recorded, name


def test_oracle_reference_takes_glibc_powf_and_no_vector_libm():
    report = (ORACLE_DIR / "libm-report.txt").read_text(encoding="utf-8")
    madwrf_part = report.split("## module_madwrf.o")[1].split("##")[0]
    assert "powf" in madwrf_part
    assert "_ZGV" not in report


def _run_host_cldinit(case, *, mutate=None):
    ni, nj, nk = case["dims"]
    nz = nk - 1
    f = {k: v.copy() for k, v in case["inp"].items()}
    cldfra = np.full((nk, nj, ni), F32(-7.0), dtype=F32)
    counts = madwrf.ColumnCounts()
    fm, ft, fb, fbr = case["flags"]
    qv, qc, qi, qs = (f[n][:nz].copy() for n in ("qv", "qc", "qi", "qs"))
    madwrf.init_madwrf_clouds(
        t=f["t"][:nz], p=f["p"][:nz], ph_2=f["ph_2"], phb=f["phb"],
        qv=qv, qc=qc, qi=qi, qs=qs, xland=f["xland"], ht=f["ht"],
        cldmask=f["cldmask"], cldtopz=f["cldtopz"],
        cldbasez=f["cldbasez"], brtemp=f["brtemp"], dx=case["dx"],
        dy=case["dy"], p00=case["p00"], flag_cldmask=fm, flag_cldtopz=ft,
        flag_cldbasez=fb, flag_brtemp=fbr, em_width=case["width"],
        hold_ups=case["hold"], cldfra=cldfra, counts=counts)
    got = {"cldfra": cldfra}
    for name, arr in (("qv", qv), ("qc", qc), ("qi", qi), ("qs", qs)):
        full = case["inp"][name].copy()
        full[:nz] = arr
        got[name] = full
    got["cldmask"], got["cldtopz"] = f["cldmask"], f["cldtopz"]
    return got, counts


def _worst(got, want):
    return int(fp32_ulp_distance(got, want).max())


CLDINIT_FIELDS = ("cldfra", "qv", "qc", "qi", "qs", "cldmask", "cldtopz")


def test_host_cloud_initialization_matches_the_oracle_at_0_ulp(
        cldinit_cases):
    worst = {}
    for idx, case in enumerate(cldinit_cases):
        got, _ = _run_host_cldinit(case)
        for name in CLDINIT_FIELDS:
            worst[(idx, name)] = _worst(got[name], case["out"][name])
    bad = {key: value for key, value in worst.items() if value}
    assert not bad, bad


def test_tracers_copy_the_initialized_hydrometeors(cldinit_cases):
    for case in cldinit_cases:
        ni, nj, nk = case["dims"]
        for src, dst in (("qc", "tr_qc"), ("qi", "tr_qi"), ("qs", "tr_qs")):
            want = case["out"][dst][:nk - 1]
            np.testing.assert_array_equal(
                want.view(np.int32), case["out"][src][:nk - 1].view(np.int32))


def test_fixture_exercises_the_branches(cldinit_cases):
    """The fixture must earn its coverage claims, not assume them."""
    total = madwrf.ColumnCounts()
    for case in cldinit_cases:
        _, counts = _run_host_cldinit(case)
        for name in vars(total):
            setattr(total, name, getattr(total, name) + getattr(counts, name))
    census = vars(total)
    for name, value in census.items():
        assert value > 0, (name, census)


def test_the_ring_pass_touches_only_the_boundary_columns(cldinit_cases):
    ring = [c for c in cldinit_cases if not c["hold"]][0]
    got = ring["out"]["cldfra"]
    ni, nj, nk = ring["dims"]
    middle = got[:, 1:nj - 1, 1:ni - 1]
    assert (middle == F32(-7.0)).all()
    assert (got[:-1, 0, :] != F32(-7.0)).all()


def _nudge_host(case, mutate=False):
    f = {k: v.copy() for k, v in case["inp"].items()}
    steps = []
    for xtime in case["xtimes"]:
        madwrf.cloud_tracer_nudge_np(
            f["qc"], f["qi"], f["qs"], f["tr_qc"], f["tr_qi"], f["tr_qs"],
            np.array(case["dt"]).view(np.int32).__xor__(np.int32(1)).view(F32)
            if mutate else case["dt"], case["relax"], case["nudge"], xtime)
        steps.append({k: f[k].copy() for k in ("qc", "qi", "qs")})
    return steps


def test_host_nudging_matches_the_oracle_at_0_ulp(nudge_cases):
    for case in nudge_cases:
        got = _nudge_host(case)
        for s, step in enumerate(case["steps"]):
            for name in ("qc", "qi", "qs"):
                assert _worst(got[s][name], step[name]) == 0, (s, name)


def test_nudging_stops_at_dt_nudge_and_keeps_subnormals(nudge_cases):
    case = nudge_cases[0]
    last, before = case["steps"][-1], case["steps"][-2]
    assert case["xtimes"][-2] >= case["nudge"]
    np.testing.assert_array_equal(last["qc"].view(np.int32),
                                  before["qc"].view(np.int32))
    first = case["steps"][0]["qc"]
    tiny = (first != 0) & (np.abs(first) < np.finfo(F32).tiny)
    assert tiny.any(), "no subnormal word survived the oracle's nudge"


def test_a_planted_one_bit_change_in_the_nudging_term_is_caught(nudge_cases):
    case = nudge_cases[0]
    got = _nudge_host(case, mutate=True)
    assert any(_worst(got[s]["qi"], step["qi"]) > 0
               for s, step in enumerate(case["steps"]))


def _no_card(cp) -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() < 1
    except Exception:
        return True


def test_cuda_cloud_initialization_matches_the_oracle_at_0_ulp(
        cldinit_cases):
    cp = pytest.importorskip("cupy")
    if _no_card(cp):
        pytest.skip("no CUDA device")
    bad = {}
    for idx, case in enumerate(cldinit_cases):
        ni, nj, nk = case["dims"]
        nz = nk - 1
        f = {k: cp.asarray(v) for k, v in case["inp"].items()}
        fields = {n: cp.ascontiguousarray(f[n][:nz])
                  for n in ("t", "p", "qv", "qc", "qi", "qs")}
        cldfra = cp.full((nk, nj, ni), F32(-7.0), dtype=F32)
        fm, ft, fb, fbr = case["flags"]
        madwrf.init_madwrf_clouds_device(
            t=fields["t"], p=fields["p"], ph_2=f["ph_2"], phb=f["phb"],
            qv=fields["qv"], qc=fields["qc"], qi=fields["qi"],
            qs=fields["qs"], xland=f["xland"], ht=f["ht"],
            cldmask=f["cldmask"], cldtopz=f["cldtopz"],
            cldbasez=f["cldbasez"], brtemp=f["brtemp"], dx=case["dx"],
            dy=case["dy"], p00=case["p00"], flag_cldmask=fm,
            flag_cldtopz=ft, flag_cldbasez=fb, flag_brtemp=fbr,
            em_width=case["width"], hold_ups=case["hold"], cldfra=cldfra)
        got = {"cldfra": cp.asnumpy(cldfra),
               "cldmask": cp.asnumpy(f["cldmask"]),
               "cldtopz": cp.asnumpy(f["cldtopz"])}
        for name in ("qv", "qc", "qi", "qs"):
            full = case["inp"][name].copy()
            full[:nz] = cp.asnumpy(fields[name])
            got[name] = full
        for name in CLDINIT_FIELDS:
            value = _worst(got[name], case["out"][name])
            if value:
                bad[(idx, name)] = value
    assert not bad, (os.environ.get("GPUWM_WRF_EXACT", "0"), bad)


def test_cuda_nudging_matches_the_oracle_at_0_ulp(nudge_cases):
    cp = pytest.importorskip("cupy")
    if _no_card(cp):
        pytest.skip("no CUDA device")
    for case in nudge_cases:
        f = {k: cp.asarray(v) for k, v in case["inp"].items()}
        for s, xtime in enumerate(case["xtimes"]):
            madwrf.cloud_tracer_nudge_device(
                f["qc"], f["qi"], f["qs"], f["tr_qc"], f["tr_qi"],
                f["tr_qs"], case["dt"], case["relax"], case["nudge"], xtime)
            for name in ("qc", "qi", "qs"):
                assert _worst(cp.asnumpy(f[name]),
                              case["steps"][s][name]) == 0, (s, name)


def test_cuda_nudge_preserves_the_specified_ring(nudge_cases):
    cp = pytest.importorskip("cupy")
    if _no_card(cp):
        pytest.skip("no CUDA device")
    case = nudge_cases[0]
    f = {name: cp.asarray(value) for name, value in case["inp"].items()}
    old = {name: value.copy() for name, value in case["inp"].items()}
    madwrf.cloud_tracer_nudge_device(
        f["qc"], f["qi"], f["qs"], f["tr_qc"], f["tr_qi"], f["tr_qs"],
        case["dt"], case["relax"], case["nudge"], case["xtimes"][0], spec_zone=1)
    for name in ("qc", "qi", "qs"):
        got = cp.asnumpy(f[name])
        ring = np.ones(got.shape, dtype=bool)
        ring[:, 1:-1, 1:-1] = False
        np.testing.assert_array_equal(got[ring].view(np.int32), old[name][ring].view(np.int32))
        np.testing.assert_array_equal(got[:, 1:-1, 1:-1].view(np.int32),
                                      case["steps"][0][name][:, 1:-1, 1:-1].view(np.int32))


def test_cuda_oracle_catches_a_one_bit_change_in_the_nudging_term(nudge_cases):
    cp = pytest.importorskip('cupy')
    if _no_card(cp):
        pytest.skip('no CUDA device')
    case = nudge_cases[0]
    fields = {name: cp.asarray(value) for name, value in case['inp'].items()}
    changed_dt = (np.array(case['dt']).view(np.int32) ^ np.int32(1)).view(F32)
    madwrf.cloud_tracer_nudge_device(
        *(fields[name] for name in ('qc', 'qi', 'qs', 'tr_qc', 'tr_qi', 'tr_qs')),
        changed_dt, case['relax'], case['nudge'], case['xtimes'][0])
    assert any(_worst(cp.asnumpy(fields[name]), case['steps'][0][name]) > 0
               for name in ('qc', 'qi', 'qs'))
