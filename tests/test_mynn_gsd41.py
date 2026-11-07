"""The GSD MYNN v4.1 generation (bl_mynn_version = "gsd_41").

Rows are ported from NOAA-EMC/HRRR tag v4.1.21,
sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_bl_mynn.F.  The default generation
(``wrf_461``) compiles mynn_pbl.cu without the MYNN_GSD41 define, so its
machine code is the one every earlier run used; the identity half of each
check below says so numerically as well.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig, validate_run_config

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "woof" / "core" / "kernels" / "mynn_pbl.cu"


def _cfg(**kwargs):
    base = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                dt=6.0, run_seconds=60.0)
    base.update(kwargs)
    return RunConfig(**base)


def test_default_generation_is_wrf_461_and_names_are_closed():
    assert _cfg().bl_mynn_version == "wrf_461"
    assert _cfg().bl_mynn_gsd41_unsquared_qtke is False
    validate_run_config(_cfg(bl_mynn_version="gsd_41"))
    with pytest.raises(ValueError, match="bl_mynn_version"):
        validate_run_config(_cfg(bl_mynn_version="hrrr"))
    with pytest.raises(ValueError, match="bl_mynn_gsd41_unsquared_qtke"):
        validate_run_config(_cfg(bl_mynn_gsd41_unsquared_qtke=1))


def test_gsd41_cloud_runs_on_every_radiation_and_needs_the_main_mass_flux():
    legacy = dict(bl_pbl_physics=5, sf_sfclay_physics=5, ra_physics=4,
                  ra_rrtmg_variant="rrtmg_legacy", bl_mynn_version="gsd_41")
    validate_run_config(_cfg(**legacy))
    # U-E4: every radiation coupling carries the gsd_41 merge
    # (test_gsd41_merge_matches_the_legacy_adapter_rule below).
    validate_run_config(_cfg(**dict(legacy, ra_rrtmg_variant="rte-rrtmgp")))
    for pair in ((5, 5), (1, 1)):
        validate_run_config(_cfg(**dict(legacy, ra_physics=0, ra_lw_physics=pair[0],
                                        ra_sw_physics=pair[1])))
    with pytest.raises(ValueError, match="sibling unit"):
        validate_run_config(_cfg(**dict(legacy, bl_mynn_mixscalars=1,
                                        mp_physics=28)))


def test_gsd41_refuses_the_v461_stochastic_kernels():
    with pytest.raises(ValueError, match="specialised from the WRF v4.6.1"):
        validate_run_config(_cfg(bl_mynn_version="gsd_41", spp_pbl=1))


def test_gsd41_define_gates_exactly_the_v461_dew_limiter():
    src = KERNEL.read_text(encoding="utf-8")
    blocks = re.findall(r"#if !defined\(MYNN_GSD41\)\n(.*?)#endif", src, re.S)
    limiter = [b for b in blocks if "qvflux = mynn_max2(" in b]
    assert len(limiter) == 1
    assert "if (qvflux < 0.0f)" in limiter[0]


# ---------------------------------------------------------------------------
# P3: a downward surface vapour flux (dew, frost) reaches the vapour
# equation under gsd_41 and is deleted under wrf_461.
# ---------------------------------------------------------------------------

def _tendency_column(ncol: int, nz: int, flqv: float):
    import cupy as cp
    from woof.core.mynn_pbl import (
        MYNN_TENDENCIES_INTERFACE_INPUTS, MYNN_TENDENCIES_LAYER_INPUTS)
    f = np.float32
    z = np.arange(nz, dtype=f)
    col = {}
    col["dz"] = np.full(nz, 40.0, f) + 10.0 * z
    col["rho"] = (1.15 - 0.01 * z).astype(f)
    col["u"] = (3.0 + 0.2 * z).astype(f)
    col["v"] = (1.0 - 0.05 * z).astype(f)
    col["th"] = (285.0 + 0.3 * z).astype(f)
    col["p"] = (95000.0 - 450.0 * z).astype(f)
    col["exner"] = ((col["p"] / 100000.0) ** 0.2857).astype(f)
    col["tk"] = (col["th"] * col["exner"]).astype(f)
    col["sqv"] = (0.008 - 0.0002 * z).astype(f)
    col["qv"] = (col["sqv"] / (1.0 - col["sqv"])).astype(f)
    col["thl"] = col["th"].copy()
    for name in ("sqc", "sqi", "sqs", "ozone", "tcd", "qcd", "diss_heat",
                 "sub_thl", "sub_sqv", "sub_u", "sub_v", "det_thl",
                 "det_sqv", "det_sqc", "det_u", "det_v", "dfm"):
        col[name] = np.zeros(nz, f)
    # Interface diffusivity only in the lowest levels, so the held top value
    # cannot leak water and the column budget closes on the surface flux.
    dfh = np.zeros(nz, f)
    dfh[1:6] = (np.array([5.0, 4.0, 3.0, 2.0, 1.0], f) / col["dz"][1:6]).astype(f)
    col["dfh"] = dfh
    col["dfm"] = dfh.copy()
    values = {name: cp.asarray(np.tile(col[name], (ncol, 1)))
              for name in MYNN_TENDENCIES_LAYER_INPUTS}
    for name in MYNN_TENDENCIES_INTERFACE_INPUTS:
        values[name] = cp.zeros((ncol, nz + 1), dtype=cp.float32)
    scal = dict(delt=20.0, psfc=95500.0, ust=0.1, wspd=2.0, uoce=0.0,
                voce=0.0, flt=-0.01, flqv=flqv, flqc=0.0)
    for name, value in scal.items():
        values[name] = cp.full((ncol,), value, dtype=cp.float32)
    values.update(qc=cp.zeros_like(values["sqc"]), qi=cp.zeros_like(values["sqi"]))
    return values, col


def _column_vapour_change(values, col, version):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_tendencies_default_cuda
    out = mynn_tendencies_default_cuda(
        values, bl_mynn_edmf=1, bl_mynn_edmf_mom=1, bl_mynn_version=version)
    dqv = cp.asnumpy(out.dqv)
    mass = (col["rho"] * col["dz"]).astype(np.float64)
    specific_rate = dqv.astype(np.float64)
    if version == "gsd_41":
        # Analysis of the GPU result, not a numerical host implementation:
        # invert the updated mixing ratio to read the specific-water budget.
        updated = col["qv"].astype(np.float64)[None] + 20.0 * specific_rate
        specific_rate = (updated / (1.0 + updated) - col["sqv"][None]) / 20.0
    return dqv, (specific_rate * mass).sum(axis=1) * 20.0


@requires_gpu
def test_gsd41_transport_is_independent_of_density_weighting():
    values, col = _tendency_column(4, 12, 4.0e-5)
    a, _ = _column_vapour_change(values, col, "wrf_461")
    b, _ = _column_vapour_change(values, col, "gsd_41")
    assert not np.array_equal(a.view(np.uint32), b.view(np.uint32))
    values["rho"] *= 0.25
    changed, _ = _column_vapour_change(values, col, "gsd_41")
    assert np.array_equal(b.view(np.uint32), changed.view(np.uint32))


@requires_gpu
def test_dew_flux_is_deleted_by_wrf461_and_applied_by_gsd41():
    flqv = 2.0e-5  # kg/kg m/s, about 60 W/m2 of latent heat
    up_values, col = _tendency_column(4, 12, flqv)
    _, total_up = _column_vapour_change(up_values, col, "gsd_41")
    zero_values, _ = _tendency_column(4, 12, 0.0)
    _, total_zero = _column_vapour_change(zero_values, col, "gsd_41")
    down_values, _ = _tendency_column(4, 12, -flqv)
    dq461, total461 = _column_vapour_change(down_values, col, "wrf_461")
    dqgsd, totalgsd = _column_vapour_change(down_values, col, "gsd_41")
    # The upward flux reaches the fork column.
    assert np.all(total_up > 0)
    # v4.6.1: the column keeps every gram of the dew (the residue is FP32
    # rounding of q near 8 g/kg, about 1e-4 of the flux).
    assert np.all(np.abs(total461) < 1e-3 * total_up)
    # gsd_41: the column loses what the same flux upward would add; the
    # specific-humidity solve is linear, even though P21 converts its output.
    assert np.allclose(totalgsd - total_zero, -(total_up - total_zero), rtol=1e-3)
    assert np.all(dqgsd[:, 0] < dq461[:, 0])


# ---------------------------------------------------------------------------
# P1: mixing length option 2 of the GSD MYNN v4.1 generation, against that
# generation's own Fortran (tools/mynn_pbl_gsd41_oracle, GNU Fortran -O0).
# ---------------------------------------------------------------------------

ORACLE_DIR = ROOT / "tests" / "data" / "oracles" / "mynn"
_GSD41_CASES, _GSD41_NZ = 12, 14


def _gsd41_fields(name: str):
    import csv
    with (ORACLE_DIR / name).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == _GSD41_CASES * _GSD41_NZ
    return {
        key: np.asarray([np.float32(row[key]) for row in rows]).reshape(
            _GSD41_CASES, _GSD41_NZ)
        for key in rows[0] if key not in ("case", "k")
    }


def _gsd41_values(fields):
    values = {name: fields[name] for name in (
        "dz", "u", "v", "qke", "dtv", "theta", "vt", "vq", "cldfra",
        "edmf_w", "edmf_a")}
    values["zw"] = np.concatenate((fields["zw"][:, :1], fields["zw_next"]),
                                  axis=1)
    for name in ("rmo", "flt", "flq", "zi", "psig_bl"):
        values[name] = fields[name][:, 0]
    ncol = _GSD41_CASES
    # Read by neither generation's option 2 under gsd_41 (fltv is rebuilt
    # from flt, flq, vt and vq); held at values that would show if read.
    values["fltv"] = np.full(ncol, 7.0, np.float32)
    values["xland"] = np.ones(ncol, np.float32)
    values["dx"] = np.full(ncol, 3000.0, np.float32)
    return values


@requires_gpu
@pytest.mark.parametrize("csv_name, unsquared", (
    ("mixlength2-gsd41.csv", True),      # the v4.1.21 source as written
    ("mixlength2-gsd41-sq.csv", False),  # the default: :995 squared
))
def test_gsd41_mixlength2_cuda_matches_its_fortran(csv_name, unsquared):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_mixlength_default_cuda
    fields = _gsd41_fields(csv_name)
    actual = mynn_mixlength_default_cuda(
        {name: cp.asarray(value) for name, value in _gsd41_values(fields).items()},
        bl_mynn_mixlength=2, bl_mynn_version="gsd_41",
        bl_mynn_gsd41_unsquared_qtke=unsquared)
    # qkw is sqrt alone: exact.  el passes through tanhf and powf.  The
    # oracle was built on glibc 2.43, whose float tanhf and powf are the
    # correctly rounded CORE-MATH routines; the kernel transcribes glibc
    # 2.39 (fdlibm tanhf, woof/core/kernels/mynn_pbl.cu mynn_tanhf), the
    # library the WRF v4.6.1 oracle CSVs were recorded on.  MEASURED: the
    # unmodified v4.6.1 option-2 harness rebuilt on the same node differs
    # from its own recorded CSV in 2 of 96 el values by 1 ULP, the same
    # class as the 4 of 168 here (at most 2 ULP).
    from woof.core.fp32_ulp import fp32_ulp_distance
    got = cp.asnumpy(actual.qkw)
    np.testing.assert_array_equal(got, fields["qkw"], err_msg="qkw")
    distance = fp32_ulp_distance(cp.asnumpy(actual.el), fields["el"])
    # development branch: 0 on the RTX 5090 (sm_120, NVRTC 12.9) afte
    # with the fork CSVs rebuilt on glibc 2.39 (mixlength2-gsd41-provenance.txt).
    assert int(distance.max()) == 0, int(distance.max())


@requires_gpu
def test_gsd41_mixlength2_differs_from_wrf461_option2():
    # The selector must reach the kernel: the same columns under wrf_461
    # give the v4.6.1 length, not the oracle's.
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_mixlength_default_cuda
    fields = _gsd41_fields("mixlength2-gsd41-sq.csv")
    values = _gsd41_values(fields)
    values["fltv"] = (fields["flt"][:, 0]).astype(np.float32)
    actual = mynn_mixlength_default_cuda(
        {name: cp.asarray(value) for name, value in values.items()},
        bl_mynn_mixlength=2)
    assert not np.array_equal(cp.asnumpy(actual.el), fields["el"])


@requires_gpu
def test_gsd41_option1_does_not_substitute_the_wrf461_length():
    # The family referee gate grades the fork's option 1 over twelve steps.
    # This negative control catches the former stock-length substitution.
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_mixlength_default_cuda
    fields = _gsd41_fields("mixlength2-gsd41-sq.csv")
    values = {name: cp.asarray(value)
              for name, value in _gsd41_values(fields).items()}
    a = mynn_mixlength_default_cuda(values, bl_mynn_mixlength=1)
    a_el = cp.asnumpy(a.el).copy()
    b = mynn_mixlength_default_cuda(values, bl_mynn_mixlength=1,
                                    bl_mynn_version="gsd_41")
    assert not np.array_equal(a_el.view(np.uint32),
                              cp.asnumpy(b.el).view(np.uint32))


# ---------------------------------------------------------------------------
# Cloud block: stratus subgrid cloud (P2, P12, P14), against the generation's
# own mym_condensation; the decay memory (P9); the radiation merge (P8/A1).
# ---------------------------------------------------------------------------

@requires_gpu
def test_gsd41_plume_condensation_matches_its_fortran():
    import csv
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.mynn_pbl_gpu import mynn_pbl_kernel

    path = ORACLE_DIR / "plume-condensation-gsd41.csv"
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    fields = {key: np.asarray([np.float32(row[key]) for row in rows])
              for key in rows[0] if key != "case"}
    qc = cp.asarray(fields["qc_in"])
    thv = cp.empty_like(qc)
    mynn_pbl_kernel("mynn_gsd41_condensation_edmf_columns", "gsd_41")(
        (1,), (32,),
        tuple(cp.asarray(fields[key]) for key in ("qt", "thl", "p", "zagl"))
        + (qc, thv, np.int32(len(rows))))
    worst = {key: int(fp32_ulp_distance(cp.asnumpy(value), fields[key]).max())
             for key, value in (("qc", qc), ("thv", thv))}
    assert worst == {"qc": 0, "thv": 0}, worst
    assert fields["qc"][-1] == 0.0  # shallow plume clears condensate
    assert np.count_nonzero(fields["qc"]) > 8

_COND_CASES, _COND_NZ = 8, 30


def _cond_fields():
    import csv
    path = ORACLE_DIR / "condensation-gsd41.csv"
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == _COND_CASES * _COND_NZ
    return {
        key: np.asarray([np.float32(row[key]) for row in rows]).reshape(
            _COND_CASES, _COND_NZ)
        for key in rows[0] if key not in ("case", "k")
    }


@requires_gpu
def test_gsd41_condensation_cuda_matches_its_fortran():
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.mynn_pbl_gpu import mynn_condensation_default_cuda
    f = _cond_fields()
    ncol, nz = _COND_CASES, _COND_NZ
    zeros = np.zeros((ncol, nz), np.float32)
    values = {
        "dz": f["dz"], "th": f["th"], "thl": f["thl"], "qw": f["qw"],
        "qv": f["qw"], "qc": zeros, "qi": zeros, "qs": zeros, "p": f["p"],
        "exner": f["exner"], "tsq": zeros, "qsq": zeros, "cov": zeros,
        "sh": np.full((ncol, nz), 0.5, np.float32), "el": f["el"],
        "rstoch": zeros, "vt": zeros, "vq": zeros, "sgm": f["sgm_in"],
        "zw": np.concatenate((f["zw"][:, :1], f["zw_next"]), axis=1),
        "xland": np.ones(ncol, np.float32), "dx": f["dx"][:, 0],
        "pblh": f["pblh"][:, 0], "hfx": f["hfx"][:, 0],
        "rmo": np.zeros(ncol, np.float32),
    }
    out = mynn_condensation_default_cuda(
        {k: cp.asarray(v) for k, v in values.items()},
        bl_mynn_version="gsd_41")
    budget = {}
    for mine, theirs in (("qc_bl", "qc_bl"), ("cldfra", "cldfra_bl"),
                         ("vt", "vt"), ("vq", "vq"), ("sgm", "sgm")):
        got = cp.asnumpy(getattr(out, mine))
        distance = fp32_ulp_distance(got, f[theirs])
        budget[mine] = int(distance.max())
    # One rounded instruction per Fortran operator; MEASURED bit for bit
    # (0 ULP on all 240 levels of every output) on the RTX 4090.
    assert budget == {"qc_bl": 0, "cldfra": 0, "vt": 0, "vq": 0, "sgm": 0}, budget
    assert np.all(cp.asnumpy(out.qi_bl) == 0.0)
    # The inversion column carries real cloud (the row's point).
    assert f["cldfra_bl"][1].max() > 0.2


@requires_gpu
def test_gsd41_cloud_decay_limits_the_fall_and_floors_the_water():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_pbl_kernel
    ncol, nz = 3, 4
    f32 = np.float32
    new = np.array([[0.1, 0.004, 0.3, 0.0]] * ncol, f32)
    old = np.array([[0.4, 0.006, 0.2, 0.0]] * ncol, f32)
    qc = np.array([[2e-4, 3e-4, 0.0, 0.0]] * ncol, f32)
    u = np.full((ncol, nz), 6.0, f32)
    v = np.full((ncol, nz), 8.0, f32)
    dx = np.full(ncol, 3000.0, f32)
    delt = np.full(ncol, 20.0, f32)
    cf_d, qc_d = cp.asfortranarray(cp.asarray(new)), cp.asfortranarray(cp.asarray(qc))
    mynn_pbl_kernel("mynn_gsd41_cloud_decay", "gsd_41")(
        (1,), (64,),
        (cf_d, qc_d, cp.asfortranarray(cp.asarray(old)),
         cp.asfortranarray(cp.asarray(u)), cp.asfortranarray(cp.asarray(v)),
         cp.asarray(dx), cp.asarray(delt), np.int32(nz), np.int32(ncol)))
    cf, qcv = cp.asnumpy(cf_d), cp.asnumpy(qc_d)
    ts = f32(min(1800.0, 3.0 * 3000.0 / 10.0))
    assert np.allclose(cf[:, 0], f32(0.4) - f32(0.25 * 20.0) / ts)  # held
    assert np.all(cf[:, 1] == 0.0) and np.all(qcv[:, 1] == 0.0)  # cleared
    assert np.all(cf[:, 2] == f32(0.3))       # rising cloud is not touched
    assert np.all(qcv[:, 2] == f32(1e-8))     # water floored under cloud
    assert np.all(cf[:, 3] == 0.0) and np.all(qcv[:, 3] == 0.0)


@requires_gpu
def test_gsd41_radiation_merge_multiplies_by_fraction_where_no_resolved_cloud():
    import cupy as cp
    from woof.core.rrtmg_legacy import _adapter_kernel
    f32 = np.float32
    qc = np.array([0.0, 5e-5, 0.0, 0.0, 0.0], f32)
    qi = np.array([0.0, 0.0, 2e-6, 0.0, 0.0], f32)
    qc_bl = np.array([3e-4, 3e-4, 3e-4, 3e-4, 3e-4], f32)
    cf_bl = np.array([0.4, 0.4, 0.4, 0.4, 0.0005], f32)
    t = np.array([280.0, 280.0, 280.0, 261.5, 280.0], f32)
    cldfra = np.full(5, 0.9, f32)
    d = {k: cp.asarray(v.copy()) for k, v in
         dict(qc=qc, qi=qi, qc_bl=qc_bl, cf_bl=cf_bl, t=t, cldfra=cldfra).items()}
    null = np.uint64(0)
    _adapter_kernel("rla_mynn_gsd41")(
        (1,), (32,),
        (np.int64(5), d["qc"], d["qi"], d["qc_bl"], d["cf_bl"], d["t"],
         d["cldfra"], np.int32(0), null, null, np.int32(0),
         f32(1e-6), f32(1e-8), f32(0.001)))
    oqc, oqi, ocf = (cp.asnumpy(d[k]) for k in ("qc", "qi", "cldfra"))
    assert oqc[0] == f32(f32(3e-4) * f32(1.0)) * f32(0.4) and oqi[0] == 0.0
    assert oqc[1] == f32(5e-5) and oqi[1] == 0.0      # resolved liquid: none
    assert oqc[2] == 0.0 and oqi[2] == f32(2e-6)      # resolved ice: none
    liq = f32(f32(f32(261.5) - f32(254.0)) / f32(15.0))
    assert oqc[3] == f32(f32(f32(3e-4) * liq) * f32(0.4))
    assert oqi[3] == f32(f32(f32(3e-4) * f32(f32(1.0) - liq)) * f32(0.4))
    assert oqc[4] == 0.0                                # fraction under 0.001
    assert np.array_equal(ocf, cf_bl)                    # replaced after step 1


# ---------------------------------------------------------------------------
# P15 surface TKE source and 1/L, P19 dissipative heating.
# ---------------------------------------------------------------------------

@requires_gpu
def test_gsd41_surface_keeps_the_surface_layer_inverse_l_and_kansas_forms():
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.mynn_pbl_gpu import mynn_pbl_kernel
    f32 = np.float32
    rmol_in = np.array([0.08, 0.5, -0.02, -0.2, 0.0], f32)  # z/L to 10 at 40 m
    ncol, nz = rmol_in.size, 4
    lay = lambda v: cp.asfortranarray(cp.full((ncol, nz), v, dtype=cp.float32))
    col = lambda v: cp.full((ncol,), v, dtype=cp.float32)
    outs = [cp.zeros((ncol,), cp.float32) for _ in range(10)]
    mynn_pbl_kernel("mynn_driver_surface_columns", "gsd_41")(
        (1,), (64,),
        (lay(1.1), lay(0.98), lay(40.0), lay(0.006), col(0.2), col(-20.0),
         col(-1.0e-6), col(285.0), *outs, np.int32(nz), np.int32(ncol),
         cp.asarray(rmol_in)))
    rmol, zet, pmz, phh = (cp.asnumpy(outs[i]) for i in (6, 7, 8, 9))
    assert np.array_equal(rmol, rmol_in)            # not recomputed
    z = (f32(0.5) * f32(40.0) * rmol_in).astype(f32)
    assert np.array_equal(zet, z)                    # not clipped
    stable = z >= 0
    assert np.array_equal(pmz[stable], (f32(1) + f32(4) * z[stable]).astype(f32))
    assert np.array_equal(phh[stable], (f32(1) + f32(5) * z[stable]).astype(f32))
    zu = z[~stable].astype(np.float64)
    want_pmz = (1.0 / (1.0 - 16.0 * zu) ** 0.25 - zu).astype(f32)
    want_phh = (1.0 / np.sqrt(1.0 - 16.0 * zu)).astype(f32)
    assert int(fp32_ulp_distance(pmz[~stable], want_pmz).max()) <= 2
    assert int(fp32_ulp_distance(phh[~stable], want_phh).max()) <= 2
    # At z/L = 10 the v4.6.1 form would be negative; this one is 41.
    assert pmz[1] == f32(41.0)


@requires_gpu
def test_gsd41_dissipative_heating_is_half_and_capped_at_2e5():
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.mynn_pbl_gpu import mynn_pbl_kernel
    f32 = np.float32
    ncol, nz = 2, 5
    el = np.array([[0.0, 40.0, 30.0, 10.0, 2.0]] * ncol, f32)
    qke = np.array([[3.0, 2.0, 0.5, 1e-3, 1e-3], [30.0, 20.0, 10.0, 1.0, 0.1]], f32)
    p = np.full((ncol, nz), 90000.0, f32)
    out = cp.zeros((ncol, nz), dtype=cp.float32, order="F")
    mynn_pbl_kernel("mynn_driver_diss_heat_columns", "gsd_41")(
        (1,), (64,),
        (cp.asfortranarray(cp.asarray(el)), cp.asfortranarray(cp.asarray(qke)),
         cp.asfortranarray(cp.asarray(p)), out, np.int32(nz), np.int32(ncol)))
    got = cp.asnumpy(out)
    blend = np.maximum(0.5 * (el[:, :-1].astype(np.float64) + el[:, 1:]), 1.0)
    want = np.minimum(np.maximum(0.5 * qke[:, :-1].astype(np.float64) ** 1.5
                                 / (24.0 * blend) / 1004.5, 0.0), 2e-5).astype(f32)
    assert int(fp32_ulp_distance(got[:, :-1], want).max()) <= 2
    assert np.all(got[:, -1] == 0.0)
    assert got[1, 0] == f32(2e-5)  # strong TKE: the 2e-5 K/s cap binds


# ---------------------------------------------------------------------------
# P16 PBL height input and KPBL.
# ---------------------------------------------------------------------------

@requires_gpu
def test_gsd41_pbl_height_reads_liquid_water_theta_v_with_subgrid_cloud():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_pbl_kernel
    f32 = np.float32
    n = 4
    th = np.full(n, 290.0, f32)
    exner = np.full(n, 0.95, f32)
    sqv = np.full(n, 0.008, f32)
    sqc = np.array([0.0, 2e-6, 0.0, 0.0], f32)   # resolved liquid in cell 1
    sqi = np.zeros(n, f32)
    tk = np.array([275.6, 275.6, 261.5, 275.6], f32)
    qc_bl = np.full(n, 4e-4, f32)
    cf_bl = np.array([0.5, 0.5, 0.5, 0.0005], f32)
    out = cp.zeros(n, cp.float32)
    mynn_pbl_kernel("mynn_gsd41_thvl_columns", "gsd_41")(
        (1,), (32,), tuple(cp.asarray(a) for a in (th, exner, sqv, sqc, sqi, tk,
                                                   qc_bl, cf_bl))
        + (out, np.int32(1), np.int32(n)))
    got = cp.asnumpy(out)
    xlvcp, xlscp = 2.5e6 / 1004.5, 2.85e6 / 1004.5
    fac = 1.0 + 0.61 * 0.008
    want = np.array([
        (290.0 - xlvcp / 0.95 * 4e-4 * 1.0 * 0.5) * fac,      # subgrid liquid
        (290.0 - xlvcp / 0.95 * 2e-6) * fac,                   # resolved cloud
        (290.0 - xlvcp / 0.95 * 4e-4 * 0.5 * 0.5
         - xlscp / 0.95 * 4e-4 * 0.5 * 0.5) * fac,             # mixed phase
        290.0 * fac])                                          # fraction 0
    assert np.allclose(got, want, rtol=2e-6)
    assert got[0] < got[3]  # subgrid cloud lowers theta-v-l: PBLH at cloud top


@requires_gpu
def test_gsd41_kpbl_blends_the_two_heights_levels():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_pblh_scale_columns_cuda
    f32 = np.float32
    ncol, nz = 2, 20
    dz = np.full((ncol, nz), 100.0, f32)
    zw = np.concatenate((np.zeros((ncol, 1), f32), np.cumsum(dz, axis=1)), axis=1)
    thv = np.where(zw[:, :-1] + 50.0 < 1240.0, 300.0, 303.0).astype(f32)
    qke = np.tile(np.maximum(2.0 - 0.15 * np.arange(nz), 1e-3), (ncol, 1)).astype(f32)
    args = [cp.asarray(a) for a in (thv, qke, zw, dz)] + [
        cp.asarray(np.ones(ncol, f32)), cp.asarray(np.full(ncol, 3000.0, f32))]
    a = mynn_pblh_scale_columns_cuda(*args)
    zi_a, kzi_a = cp.asnumpy(a.zi).copy(), cp.asnumpy(a.kzi).copy()
    b = mynn_pblh_scale_columns_cuda(*args, bl_mynn_version="gsd_41")
    assert np.array_equal(cp.asnumpy(b.zi), zi_a)   # the height is the same
    # v4.6.1: the level below the first interface at or above zi.  gsd_41:
    # the blend of the two rounded levels, here at least as high.
    assert np.all(cp.asnumpy(b.kzi) >= 1)
    assert np.all(np.abs(cp.asnumpy(b.kzi) - kzi_a) <= 2)


# ---------------------------------------------------------------------------
# P1 with P17 in the assembled turbulence call, against the generation's
# mym_turbulence at level 2.5 (levflag 2, mixing length 2).
# ---------------------------------------------------------------------------

@requires_gpu
def test_gsd41_turbulence_matches_its_fortran():
    import csv
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.mynn_pbl_gpu import mynn_turbulence_default_cuda
    path = ORACLE_DIR / "turbulence2-gsd41-sq.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="ascii")))
    ncase, nz = 5, 14
    assert len(rows) == ncase * nz
    f = {k: np.asarray([np.float32(r[k]) for r in rows]).reshape(ncase, nz)
         for k in rows[0] if k not in ("case", "k")}
    zeros = np.zeros((ncase, nz), np.float32)
    values = {
        "dz": f["dz"],
        "zw": np.concatenate((f["zw"][:, :1], f["zw_next"]), axis=1),
        "u": f["u"], "v": f["v"], "thl": f["thl"],
        "thetav": (f["theta"] * (1.0 + 0.61 * f["qw"])).astype(np.float32),
        "ql": f["ql"], "qw": f["qw"], "qke": f["qke"], "tsq": zeros,
        "qsq": zeros, "cov": zeros, "vt": f["vt"], "vq": f["vq"],
        "theta": f["theta"], "cldfra": f["cldfra"], "edmf_w": f["edmf_w"],
        "edmf_a": f["edmf_a"], "tkeprodtd": zeros,
        "xland": np.ones(ncase, np.float32),
        "dx": np.full(ncase, 3000.0, np.float32), "rmo": f["rmo"][:, 0],
        "flt": f["flt"][:, 0], "fltv": f["flt"][:, 0], "flq": f["flq"][:, 0],
        "zi": f["zi"][:, 0], "psig_bl": f["psig_bl"][:, 0],
        "psig_shcu": f["psig_shcu"][:, 0],
    }
    out = mynn_turbulence_default_cuda(
        {k: cp.asarray(v) for k, v in values.items()}, bl_mynn_mixlength=2,
        bl_mynn_version="gsd_41")
    worst = {}
    for name in ("el", "dfm", "dfh", "dfq", "pdk", "pdt", "pdq", "pdc", "sh"):
        got, want = cp.asnumpy(getattr(out, name)), f[name]
        if name in ("sh", "pdk", "pdt", "pdq", "pdc"):
            got, want = got[:, 1:], want[:, 1:]
        worst[name] = int(fp32_ulp_distance(got, want).max())
    # The interface arithmetic of this kernel is the v4.6.1 port's plain
    # operators (NVRTC may contract), as on the default path; MEASURED at
    # most 3 ULP on the RTX 4090 over 70 interfaces.
    # development branch: 0 with the fork's a2den algebra, --fmad=false and the
    # glibc 2.39 rebuild of turbulence2-gsd41-sq.csv.
    assert max(worst.values()) == 0, worst


# ---------------------------------------------------------------------------
# U-E4: the gsd_41 radiation merge on every coupling.  The legacy RRTMG
# adapter's kernel (rla_mynn_gsd41) is the reference; the shared merge every
# other radiation coupling calls (woof.core.mynn_radiation) must match it.
# ---------------------------------------------------------------------------

def _gsd41_columns(xp):
    rng = np.random.default_rng(41)
    n = 4096
    f = np.float32
    qc = np.where(rng.random(n) < 0.5, f(0.0), rng.random(n).astype(f) * f(2e-6))
    qi = np.where(rng.random(n) < 0.5, f(0.0), rng.random(n).astype(f) * f(2e-8))
    qc_bl = rng.random(n).astype(f) * f(1e-3)
    qi_bl = rng.random(n).astype(f) * f(1e-4)
    cldfra_bl = np.where(rng.random(n) < 0.2, f(0.0005), rng.random(n).astype(f))
    t = (f(230.0) + rng.random(n).astype(f) * f(50.0)).astype(f)
    cldfra = rng.random(n).astype(f)
    return [xp.asarray(a) for a in (qc, qi, qc_bl, qi_bl, cldfra_bl, t, cldfra)]


def _reference_gsd41(qc, qi, qc_bl, cldfra_bl, t, cldfra, first_step):
    """rla_mynn_gsd41, element by element in FP32."""
    f = np.float32
    qc, qi, cldfra = qc.copy(), qi.copy(), cldfra.copy()
    for i in range(qc.size):
        fb = cldfra_bl[i]
        if not first_step or fb > f(0.001):
            cldfra[i] = fb
        if qc[i] < f(1e-6) and qi[i] < f(1e-8) and fb > f(0.001):
            liq = min(f(1.0), max(f(0.0), f(f(t[i] - f(254.0)) / f(15.0))))
            qc[i] = f(qc[i] + f(f(qc_bl[i] * liq) * fb))
            qi[i] = f(qi[i] + f(f(qc_bl[i] * f(f(1.0) - liq)) * fb))
    return qc, qi, cldfra


@pytest.mark.parametrize("itimestep", [1, 7])
def test_gsd41_merge_matches_the_legacy_adapter_rule(itimestep):
    from woof.core.mynn_radiation import merge_mynn_bl_clouds, mynn_bl_cloud_supplied
    qc, qi, qc_bl, qi_bl, cldfra_bl, t, cldfra = _gsd41_columns(np)
    want = _reference_gsd41(qc, qi, qc_bl, cldfra_bl, t, cldfra, itimestep == 1)
    liquid, ice = mynn_bl_cloud_supplied(qc, qi, qc_bl=qc_bl, qi_bl=qi_bl, cldfra_bl=cldfra_bl,
                                         bl_pbl_physics=5, icloud_bl=1, bl_mynn_version="gsd_41",
                                         temperature=t)
    got = merge_mynn_bl_clouds(qc.copy(), qi.copy(), cldfra.copy(), qc_bl=qc_bl, qi_bl=qi_bl,
                               cldfra_bl=cldfra_bl, bl_pbl_physics=5, icloud_bl=1,
                               itimestep=itimestep, bl_mynn_version="gsd_41", temperature=t)
    for a, b in zip(got, want):
        assert a.tobytes() == b.tobytes()
    assert np.array_equal(liquid, got[0] != qc) and np.array_equal(ice, got[1] != qi)
    # The v4.6.1 merge is untouched by the selector's default.
    v461 = merge_mynn_bl_clouds(qc.copy(), qi.copy(), cldfra.copy(), qc_bl=qc_bl, qi_bl=qi_bl,
                                cldfra_bl=cldfra_bl, bl_pbl_physics=5, icloud_bl=1, itimestep=itimestep)
    assert not np.array_equal(v461[0], got[0])


@requires_gpu
@pytest.mark.parametrize("itimestep", [1, 7])
def test_gsd41_merge_is_the_legacy_adapter_kernel_on_device(itimestep):
    import cupy as cp
    from woof.core.mynn_radiation import merge_mynn_bl_clouds
    from woof.core.rrtmg_legacy import _adapter_kernel
    qc, qi, qc_bl, qi_bl, cldfra_bl, t, cldfra = _gsd41_columns(cp)
    n = qc.size
    kqc, kqi, kcf = qc.copy(), qi.copy(), cldfra.copy()
    null = np.uint64(0)
    _adapter_kernel("rla_mynn_gsd41")(((n + 127) // 128,), (128,), (
        np.int64(n), kqc, kqi, qc_bl, cldfra_bl, t, kcf, np.int32(1 if itimestep == 1 else 0),
        null, null, np.int32(0), np.float32(1e-6), np.float32(1e-8), np.float32(0.001)))
    got = merge_mynn_bl_clouds(qc.copy(), qi.copy(), cldfra.copy(), qc_bl=qc_bl, qi_bl=qi_bl,
                               cldfra_bl=cldfra_bl, bl_pbl_physics=5, icloud_bl=1,
                               itimestep=itimestep, bl_mynn_version="gsd_41", temperature=t)
    for a, b in zip(got, (kqc, kqi, kcf)):
        assert cp.asnumpy(a).tobytes() == cp.asnumpy(b).tobytes()
