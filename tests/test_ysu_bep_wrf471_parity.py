"""YSU's flag_bep arm (sf_urban_physics 2/3) against WRF v4.7.1.

``tools/urban_wrf471_oracle/run_ysu_bep.F90`` calls ``bl_ysu_run`` from the
byte-unmodified ``phys/physics_mmm/bl_ysu.F90`` at MMM-physics
``20240626-MPASv8.2`` with ``flag_bep = .true.`` on the 24 columns of the
shipped v4.6.1 YSU fixture, each carrying a BEP forcing set shaped like the one
``module_sf_noahdrv.F:1679-1720`` hands the PBL.  This module runs
``launch_ysu(..., bep=..., frc_urb2d=...)`` (``ysu_column_bep``) on the same
inputs and pins the measured distance.

That ``bl_ysu.F90`` differs from the v4.6.1 copy the non-BEP kernel was
transcribed from by exactly one line (``we(i) = 0.`` at :605, which the
kernel's zero-initialised ``we`` already is): every flag_bep expression is
identical in the two tags.

**The numbers are a measurement, asserted for equality** (the
``tests/test_ysu_wrf461_parity.py`` convention): the BEP arm inherits the
non-BEP kernel's CUDA-libm residue (expf/powf/cbrtf are CUDA's in ysu.cu, not
glibc's), so it is not bitwise, and a change in either direction must be
stated in a commit.

Reference arm.  WRF's own driver always passes ``ctopo = ctopo2 = 1``
(``module_bl_ysu.F:404``) and in that arm :1313 removes the urban fraction of
the surface drag.  ``ysu.cu`` carries the ctopo-absent form of ``ad(1)``, so
its BEP arm spells that removal on ``fric`` directly; it is graded against the
ctopo arm (what WRF does), and the WRF-against-WRF distance to the ctopo-absent
call (where WRF removes nothing) is pinned too, to show the arm matters.

Which WRF.  woof carries ONE declared divergence in this arm
(``tests/test_ysu_bep_rural_drag.py``): :1313 removes only the urban
fraction of YSU's own drag while the BEP couple already folds the rural drag
into ``a_u_bep``, so WRF counts the rural surface drag twice; ``ysu.cu``
removes the whole of YSU's own drag.  The port is graded against
``ysu_bep_fix/columns``, WRF built by
``tools/urban_wrf471_oracle/build_ysu_bep_fix.sh`` with exactly that one
change (the ``frc_urb1d(i)*`` factor dropped from :1313).  The stock build,
``ysu_bep/columns``, still reproduces byte for byte from the same script and
stays here as the record of what WRF does: the two differ in the u and v
tendencies of the columns that are not wholly urban and nowhere else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.fp32_ulp import fp32_ulp_distance
from woof.wrf_exact import ENABLED as WRF_EXACT

_ORACLE = Path(__file__).resolve().parents[1] / "tests" / "data" / "oracles" / "urban" / "bep"
#: WRF with the one-line rural-drag fix woof carries: what the port is graded against.
FIXTURE = _ORACLE / "ysu_bep_fix" / "columns"
#: Byte-unmodified WRF v4.7.1: the defect's record.
STOCK_FIXTURE = _ORACLE / "ysu_bep" / "columns"

#: port output -> oracle word.  dtheta is compared with ttnp/pi2d, the
#: tendency module_bl_ysu.F:452 hands the solver (the v4.6.1 test's rthblten).
LEVEL_MAP = {"du": "utnp", "dv": "vtnp", "dqv": "qvtnp", "dqc": "qctnp",
             "dqi": "qitnp", "exch_h": "exch_hx", "exch_m": "exch_mx"}
SURFACE_MAP = {"hpbl": "hpbl", "wstar": "wstar", "delta": "delta"}

#: Columns whose disagreement is a branch, not a rounding -- the v4.6.1
#: test's cases 7 and 13 (subnormal br / ust under CuPy's -ftz=true).  Case 12
#: (ust = hfx = qfx = 0) is NOT held out: it takes WRF's path under BEP, as
#: every column of the plain kernel has since golden/ports-pbl removed the
#: zero-flux short circuit.
BRANCH_DIVERGENCE_CASES = (7, 13)

#: Measured on an RTX 4090 (sm_89, NVRTC 13.4, cupy 14.2.0), over the 22
#: columns that take WRF's branches.  Same classes as the non-BEP table in
#: tests/test_ysu_wrf461_parity.py: dtheta/hpbl/wstar/delta 1 ULP, exch 7 ULP
#: (CUDA libm vs glibc through the diffusivities), momentum and vapor
#: tendencies up to ~1e4 ULP but <= 4.3e-08 m/s2 and 6.2e-11 kg/kg/s absolute
#: (near-total cancellations), dqv/dqc/dqi maxima at lanes where WRF wrote a
#: subnormal and the -ftz=true kernel writes zero.  The dv row was 1974573
#: (2.6e-04 m/s2, 12% of the field) until the v solve used tridi2n's own
#: hybrid elimination (see ysu_tridi2n_v); that is the gate firing on a real
#: transcription error, kept here as its record.
#:
#: RE-MEASURED 2026-09-30 against the fixed build (ysu_bep_fix, the declared
#: rural-drag divergence): du 8738 -> 1821 on test host 2 (RTX 5090, sm_120,
#: NVRTC 13.4.92), every other field unchanged.  The 8738 was the stock
#: build's u row; both builds share every non-momentum word, and the port
#: carries the fix, so it now sits closer to the build it transcribes.
#:
#: RE-MEASURED 2026-10-05 (development branch, sweep rows 9 and 10): the BEP
#: entry point now takes WRF's ctopo drag arm with ctopo = 1, as WRF's driver
#: does, and the unit compiles with --fmad=false and glibc's powf/expf, so the
#: arm is WRF's word everywhere except the flushed subnormal lanes (dqv, dqc,
#: dqi maxima below).  Measured on an RTX PRO 6000 (sm_120) under NVRTC
#: 13.4.92 and 12.9.86, identical: the per-build and per-architecture rows
#: this table used to carry (du 1821 / 1457 / 5825, dtheta 1 / 91) were the
#: compiler's contraction and CUDA libm, and there is no compiler-sensitive
#: field left.
BASELINE_MAX_ULP: dict[str, int] = {
    "du": 0,
    "dv": 0,
    "dtheta": 0,
    "dqv": 15070,
    "dqc": 207470,
    "dqi": 30808,
    "exch_h": 0,
    "exch_m": 0,
    "hpbl": 0,
    "wstar": 0,
    "delta": 0,
}

#: Under GPUWM_WRF_EXACT=1 (no flush to zero) the arm is WRF's word on every
#: field, the subnormal lanes included.
STRICT_MAX_ULP: dict[str, int] = {field: 0 for field in BASELINE_MAX_ULP}

#: One row per NVRTC build measured, kept so a build that ever reads
#: differently names itself.  Both builds read the same table.
BASELINE_MAX_ULP_BY_NVRTC_BUILD: dict[str, dict[str, int]] = {
    "13.4.92": BASELINE_MAX_ULP,
    "12.9.86": dict(BASELINE_MAX_ULP),
}
COMPILER_SENSITIVE_FIELDS: tuple[str, ...] = ()

#: Where an architecture reads a build differently from that build's row.
#: Empty since the contraction and CUDA libm left the unit.
BASELINE_MAX_ULP_BY_ARCH_AND_BUILD: dict[tuple[str, str], dict[str, int]] = {}
#: Fields an architecture row may move from its build's row.
ARCH_SENSITIVE_FIELDS: tuple[str, ...] = ()


def _recorded_baseline() -> tuple[str, dict[str, int]]:
    """The row for the NVRTC build compiling these kernels, on this card's
    architecture.

    An architecture row wins where one is recorded; otherwise the build's
    row.  An unrecorded build is graded against the 13.4.92 row and, on a
    mismatch, the failure names the build and the architecture, so the fix
    is to measure it and add its row -- never to relax the comparison.
    """
    import cupy as cp

    from woof.certify.compile_platform import nvrtc_build

    build = nvrtc_build()
    capability = str(cp.cuda.Device().compute_capability)
    label = f"{build} on sm_{capability}"
    if WRF_EXACT:
        # GPUWM_WRF_EXACT=1 compiles --ftz=false, so the subnormal lanes the
        # default row carries (dqv, dqc, dqi) are written as WRF wrote them:
        # every field is WRF's word (measured on an RTX 5070 Ti, sm_120, with the
        # strict wrf_461 arm).
        return label + " strict", STRICT_MAX_ULP
    if (capability, build) in BASELINE_MAX_ULP_BY_ARCH_AND_BUILD:
        return label, BASELINE_MAX_ULP_BY_ARCH_AND_BUILD[(capability, build)]
    return label, BASELINE_MAX_ULP_BY_NVRTC_BUILD.get(build, BASELINE_MAX_ULP)


def test_compiler_rows_differ_only_where_the_compiler_is_allowed_to():
    for build, row in BASELINE_MAX_ULP_BY_NVRTC_BUILD.items():
        assert set(row) == set(BASELINE_MAX_ULP), build
        for field, value in row.items():
            if field not in COMPILER_SENSITIVE_FIELDS:
                assert value == BASELINE_MAX_ULP[field], (build, field)
    for (capability, build), row in BASELINE_MAX_ULP_BY_ARCH_AND_BUILD.items():
        assert build in BASELINE_MAX_ULP_BY_NVRTC_BUILD, (capability, build)
        base = BASELINE_MAX_ULP_BY_NVRTC_BUILD[build]
        assert set(row) == set(base), (capability, build)
        moved = {field for field in row if row[field] != base[field]}
        assert moved and moved <= set(ARCH_SENSITIVE_FIELDS), (
            capability, build, sorted(moved))

#: WRF against WRF: flag_bep with ctopo = 1 vs flag_bep with ctopo absent.
#: 9.5e-03 m/s2 on utnp: the drag removal at bl_ysu.F90:1313 is not a
#: rounding, which is why the BEP arm carries it.  One row per build: the
#: fixed build removes the whole of YSU's drag in the ctopo arm, so its gap
#: to the (unchanged) ctopo-absent arm moves.
WRF_CTOPO_GAP_MAX_ULP: dict[str, int] = {"du": 1944797275, "dv": 9941925}
WRF_CTOPO_GAP_MAX_ULP_STOCK: dict[str, int] = {"du": 1944377117, "dv": 9036004}


def load_bin_fixture(directory: Path) -> dict[str, np.ndarray]:
    """One oracle case through woof.verify.urban_oracle.load_case, with each
    array transposed from Fortran index order to C order (dims reversed, so a
    Fortran ``(ncol, nz)`` array comes back ``(nz, ncol)``) and scalars as
    one-element arrays."""
    from woof.verify.urban_oracle import load_case

    return {name: (np.ascontiguousarray(np.asarray(value).T)
                   if np.ndim(value) else np.atleast_1d(value))
            for name, value in load_case(directory).items()}


def _fixture(directory: Path = FIXTURE):
    fx = load_bin_fixture(directory)
    nz, ncase = int(fx["meta_nz"][0]), int(fx["meta_ncase"][0])

    def lev(name):
        return np.ascontiguousarray(fx[name].reshape(nz, 1, ncase))

    def sfc(name):
        return np.ascontiguousarray(fx[name].reshape(1, ncase))

    tx, pi2d = lev("tx"), lev("pi2d")
    inputs = {
        "u": lev("ux"), "v": lev("vx"),
        "theta": np.ascontiguousarray(tx / pi2d),   # bl_ysu.F90:419
        "qv": lev("qvx"), "qc": lev("qcx"), "qi": lev("qix"),
        "p": lev("p2d"),
        "p_interface": np.ascontiguousarray(
            fx["p2di"].reshape(nz + 1, 1, ncase)),
        "exner": pi2d, "dz": lev("dz8w"), "rthraten": lev("rthraten"),
    }
    for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim",
                 "psih", "xland", "u10", "v10"):
        inputs[name] = sfc(name)
    bep = {name: lev(name) for name in (
        "a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "b_u_bep", "b_v_bep",
        "b_t_bep", "b_q_bep", "sf_bep", "vl_bep")}
    ref = {}
    for arm in ("ctopo_", "noctopo_"):
        for word in set(LEVEL_MAP.values()) | {"ttnp"}:
            ref[arm + word] = lev(arm + word)
        for word in set(SURFACE_MAP.values()) | {"kpbl"}:
            ref[arm + word] = sfc(arm + word)
        ref[arm + "rthblten"] = np.ascontiguousarray(
            ref[arm + "ttnp"] / pi2d)                # module_bl_ysu.F:452
    return dict(nz=nz, ncase=ncase, dt=float(fx["meta_dt"][0]),
                topdown=fx["topdown"].astype(np.int32), inputs=inputs,
                bep=bep, frc=sfc("frc_urb2d"), ref=ref)


def _port(fx, *, bep: bool = True):
    import cupy as cp

    from woof.core.ysu import launch_ysu

    dev = {k: cp.asarray(v) for k, v in fx["inputs"].items()}
    kw = {}
    if bep:
        kw = dict(bep={k: cp.asarray(v) for k, v in fx["bep"].items()},
                  frc_urb2d=cp.asarray(fx["frc"]))
    merged = {}
    for flag in sorted(set(int(f) for f in fx["topdown"])):
        out = launch_ysu(
            dev["u"], dev["v"], dev["theta"], dev["qv"], dev["qc"], dev["qi"],
            dev["p"], dev["p_interface"], dev["exner"], dev["dz"],
            dev["rthraten"], psfc=dev["psfc"], znt=dev["znt"],
            ust=dev["ust"], hfx=dev["hfx"], qfx=dev["qfx"],
            wspd=dev["wspd"], br=dev["br"], psim=dev["psim"],
            psih=dev["psih"], xland=dev["xland"], u10=dev["u10"],
            v10=dev["v10"], dt=fx["dt"], ysu_topdown_pblmix=flag,
            # The WRF 4.7.1 flag_bep reference grades the legacy kernel; the
            # WRF 4.8 generation's BEP columns are graded against WRF 4.8 by
            # tools/ysu_generations_oracle/gate.py.
            ysu_version="wrf_461", **kw)
        take = fx["topdown"] == flag
        for name, value in out.items():
            host = cp.asnumpy(value)
            merged.setdefault(name, np.zeros_like(host))
            merged[name][..., take] = host[..., take]
    return merged


def _mask(fx):
    return np.asarray([c + 1 not in BRANCH_DIVERGENCE_CASES
                       for c in range(fx["ncase"])])


def _measure(fx, port, arm="ctopo_"):
    mask = _mask(fx)
    out = {}
    for name, word in LEVEL_MAP.items():
        out[name] = int(fp32_ulp_distance(
            np.ascontiguousarray(port[name][:, :, mask], np.float32),
            np.ascontiguousarray(fx["ref"][arm + word][:, :, mask])).max())
    out["dtheta"] = int(fp32_ulp_distance(
        np.ascontiguousarray(port["dtheta"][:, :, mask], np.float32),
        np.ascontiguousarray(fx["ref"][arm + "rthblten"][:, :, mask])).max())
    for name, word in SURFACE_MAP.items():
        out[name] = int(fp32_ulp_distance(
            np.ascontiguousarray(port[name][:, mask], np.float32),
            np.ascontiguousarray(fx["ref"][arm + word][:, mask])).max())
    return out


def test_fixture_reaches_the_bep_arm():
    """Every urban fraction class and a nonzero forcing are in the fixture,
    and the two WRF arms really differ (else the ctopo gate is vacuous)."""
    fx = _fixture()
    frc = fx["frc"].reshape(-1)
    assert {0.0, 1.0} <= set(frc.tolist()) and ((frc > 0) & (frc < 1)).any()
    assert (fx["bep"]["vl_bep"] < 1).any() and (fx["bep"]["sf_bep"] < 1).any()
    assert np.abs(fx["bep"]["b_t_bep"]).max() > 0
    gap = {}
    for name, word in LEVEL_MAP.items():
        gap[name] = int(fp32_ulp_distance(
            fx["ref"]["ctopo_" + word], fx["ref"]["noctopo_" + word]).max())
    assert gap["du"] > 0 and gap["dv"] > 0, gap
    if WRF_CTOPO_GAP_MAX_ULP:
        assert {k: gap[k] for k in ("du", "dv")} == WRF_CTOPO_GAP_MAX_ULP
    stock = _fixture(STOCK_FIXTURE)
    stock_gap = {name: int(fp32_ulp_distance(
        stock["ref"]["ctopo_" + word], stock["ref"]["noctopo_" + word]).max())
        for name, word in LEVEL_MAP.items() if name in ("du", "dv")}
    assert stock_gap == WRF_CTOPO_GAP_MAX_ULP_STOCK


def test_stock_and_fixed_wrf_differ_only_in_the_rural_drag():
    """WRF against WRF: the one-line fix moves the u and v tendencies of
    every column that is not wholly urban, and not one other word.  On a
    wholly urban column (frc = 1) YSU's own drag was already removed whole,
    so the two builds agree bitwise there too."""
    fixed, stock = _fixture(), _fixture(STOCK_FIXTURE)
    for name, value in fixed["inputs"].items():
        assert np.array_equal(value.view(np.uint32),
                              stock["inputs"][name].view(np.uint32)), name
    for name, value in fixed["bep"].items():
        assert np.array_equal(value.view(np.uint32),
                              stock["bep"][name].view(np.uint32)), name
    for arm in ("ctopo_", "noctopo_"):
        for word in set(LEVEL_MAP.values()) | {"ttnp"} | set(SURFACE_MAP.values()) | {"kpbl"}:
            a, b = fixed["ref"][arm + word], stock["ref"][arm + word]
            if arm == "ctopo_" and word in ("utnp", "vtnp"):
                continue
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), (arm, word)
    frc = fixed["frc"].reshape(-1)
    for word in ("utnp", "vtnp"):
        a, b = fixed["ref"]["ctopo_" + word], stock["ref"]["ctopo_" + word]
        moved = (a.view(np.uint32) != b.view(np.uint32)).any(axis=(0, 1))
        assert not moved[frc == 1.0].any(), word
        # Case 12 has ust = 0 and case 13 a subnormal ust (fric underflows
        # to zero): no drag to double.
        ust = fixed["inputs"]["ust"].reshape(-1)
        rural = (frc < 1.0) & (ust >= np.finfo(np.float32).tiny)
        assert moved[rural].all(), (word, np.flatnonzero(rural & ~moved))


@requires_gpu
def test_ysu_bep_arm_matches_wrf_flag_bep_measured_table():
    fx = _fixture()
    port = _port(fx)
    measured = _measure(fx, port)
    build, recorded = _recorded_baseline()
    print("measured", measured, "under NVRTC", build)
    assert measured == recorded, (
        f"measured {measured}\nrecorded {recorded}\n"
        f"kernel compiler: NVRTC {build}; rows recorded for "
        f"{sorted(BASELINE_MAX_ULP_BY_NVRTC_BUILD)} and, by architecture, "
        f"{sorted(BASELINE_MAX_ULP_BY_ARCH_AND_BUILD)}")
    kpbl = port["kpbl"][:, _mask(fx)]
    np.testing.assert_array_equal(
        kpbl, fx["ref"]["ctopo_kpbl"][:, _mask(fx)].astype(np.int32))


@requires_gpu
def test_bep_arm_is_not_the_plain_kernel():
    """With the forcing withheld the kernel must land far from WRF's
    flag_bep words: the gate above grades the arm, not the plain column."""
    fx = _fixture()
    plain = _measure(fx, _port(fx, bep=False))
    forced = _measure(fx, _port(fx))
    assert plain["dtheta"] > 1000 * max(forced["dtheta"], 1), (plain, forced)
