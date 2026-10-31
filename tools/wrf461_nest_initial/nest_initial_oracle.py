"""WRF v4.6.1 med_nest_initial for an input-file nest, in FP32 on the host.

usage: python nest_initial_oracle.py INPUT_DIR NATIVE_RUN [I_PARENT_START J_PARENT_START RATIO]

INPUT_DIR holds the real.exe wrfinput_d01/d02 the native two-domain run read;
NATIVE_RUN holds that wrf.exe 4.6.1 strict run's wrfout_d02 frames.  The script
rebuilds the nest's first history frame from the input files alone, by WRF's
own transaction, and compares it word for word (the transaction WOOF's door
has to perform; see Downloads/UAE-OU-CASE-2026-10-28/w461-2dom/RESUME.md).

What WRF does to an input-file nest before the nest's t = 0 history, as
measured here (UAE 9 km/3 km, physics off, feedback = 1, every field below
0 differing words against wrf.exe 4.6.1 strict on a development machine):

1. med_initialdata_input reads wrfinput_d02 AND runs start_domain on it: MUB,
   PB, T_INIT, ALB and PHB are rebuilt from the file's HGT by the cold-start
   formula (start_em.F:554-636; woof.ingest.wrfinput.wrf_start_base_state),
   t_1 = t_2, ph_1 = ph_2, mu_1 = mu_2, and P/AL are re-derived (start_em.F
   :774-842).  So mub_save is the REBUILT MUB (16,027 of 31,500 words differ
   from the file's) and adjust_tempqv's pp is that re-derived P, not the
   file's (every word differs).
2. ht_int/mub_fine are SINT captures of the PARENT's ht and its rebuilt MUB
   (the parent ran start_domain too); the blend (nest_init_utils.F:712-785)
   is FP32, r_blend_zones = 1./(blend_width+1).
3. adjust_tempqv (nest_init_utils.F:812-890) in FP32 with glibc powf/expf.
4. med_nest_feedback (nest -> parent; no coupling on the nest).
5. start_domain(nest) with press_adj: base rebuilt from the BLENDED HGT;
   AL/P from ph_1 (file PH), mu_1 (file MU), t_1 = the ADJUSTED t_2 and the
   adjusted QV; then MU_2 += al(1)/(alt(1)*alb(1))*g*(ht - ht_fine).
6. The parent's first step, then med_nest_force couples and uncouples the
   nest (couple_or_uncouple_em: x*m then x*(1./m), m from the rebuilt MUB and
   the press-adjusted MU), and only then the nest writes its t = 0 history.
   THM, QVAPOR, PH, U and V (interior) carry that round trip.
"""
import glob
import sys

import netCDF4
import numpy as np

from woof.core.nest_interp import register_nest, sint
from woof.core.noahmp_libm import logf_array, powf_array
from woof.ingest.wrfinput import _glibc_expf_array as expf_array
from woof.ingest.wrfinput import wrf_start_base_state

f32 = np.float32


def A(x):
    return np.asarray(x, np.float32)


R_D, R_V, G = f32(287.0), f32(461.6), f32(9.81)
RVOVRD = f32(R_V / R_D)
CP = f32(f32(f32(7.0) * R_D) / f32(2.0))
CPOVCV = f32(CP / f32(CP - R_D))
P1000, T0 = f32(100000.0), f32(300.0)
TWO7 = f32(f32(2.0) / f32(7.0))
BASE_NAMES = ("P00", "T00", "TLP", "TISO", "TLP_STRAT", "P_STRAT", "P_TOP",
              "C3H", "C4H", "C3F", "C4F", "C1H", "C2H", "DNW", "HGT")


def blend(coarse, fine, sbw=5, width=5):
    """blend_terrain, FP32: the smallest matching blend_cell wins, from the original."""
    fine, coarse = A(fine), A(coarse)
    sq = fine.ndim == 2
    F, C = (fine[None], coarse[None]) if sq else (fine, coarse)
    _, ny, nx = F.shape
    ide, jde = nx + 1, ny + 1
    i1 = np.arange(1, nx + 1)[None, None, :]
    j1 = np.arange(1, ny + 1)[None, :, None]
    rz = f32(f32(1.0) / f32(width + 1))
    out = F.copy()
    for bc in range(width, 0, -1):
        hit = ((i1 == sbw + bc) | (j1 == sbw + bc)
               | (i1 == ide - sbw - bc) | (j1 == jde - sbw - bc))
        out = np.where(hit, A(A(A(f32(bc) * F) + A(f32(width + 1 - bc) * C)) * rz), out)
    spec = (i1 <= sbw) | (j1 <= sbw) | (i1 >= ide - sbw) | (j1 >= jde - sbw)
    out = A(np.where(spec, C, out))
    return out[0] if sq else out


def es_of(tc):
    return A(f32(610.78) * expf_array(A(A(f32(17.0809) * tc) / A(f32(234.175) + tc))))


def adjust_tempqv(th, qv, pp, mub_save, mub, c3h, c4h, p_top):
    """nest_init_utils.F:812-890 (use_theta_m = 0), Fortran order, FP32."""
    po = A(A(A(c4h + A(c3h * mub_save[None])) + p_top) + pp)
    tc = A(A(A(th + f32(300.0)) * powf_array(A(po / f32(1.e5)), TWO7)) - f32(273.15))
    e = A(A(qv * po) / A(f32(0.622) + qv))
    rh = A(e / es_of(tc))
    pn = A(A(A(c4h + A(c3h * mub[None])) + p_top) + pp)
    tl = A(th + f32(300.0))
    s, dp = A(pn + po), A(pn - po)
    dth1 = A(A(A(f32(-191.86e-3) * tl) / s) * dp)
    dth = A(A(A(f32(-191.86e-3) * A(tl + A(f32(0.5) * dth1))) / s) * dp)
    th_new = A(A(tl + dth) - f32(300.0))
    tc = A(A(A(tl + dth) * powf_array(A(pn / f32(1.e5)), TWO7)) - f32(273.15))
    e = A(rh * es_of(tc))
    return th_new, A(A(f32(0.622) * e) / A(pn - e))


def start_eos(base, ph1, mu1, t1, qv, cols, p_top):
    """start_em.F:803-842, hypsometric_opt = 2, FP32: AL then P."""
    c3h, c4h, c3f, c4f = cols
    MUB, PB, ALB, PHB = (A(base[n]) for n in ("MUB", "PB", "ALB", "PHB"))
    mt = A(MUB + mu1)[None]

    def lev(c3, c4):
        return A(A(A(c3 * mt) + c4) + p_top)
    pfu, pfd, phm = lev(c3f[1:], c4f[1:]), lev(c3f[:-1], c4f[:-1]), lev(c3h, c4h)
    num = A(A(A(ph1[1:] - ph1[:-1]) + PHB[1:]) - PHB[:-1])
    al = A(A(A(num / phm) / logf_array(A(pfd / pfu))) - ALB)
    qvf = A(f32(1.0) + A(RVOVRD * qv))
    x = A(A(A(R_D * A(T0 + t1)) * qvf) / A(P1000 * A(al + ALB)))
    return al, A(A(P1000 * powf_array(x, CPOVCV)) - PB)


def main():
    IN, NAT = sys.argv[1], sys.argv[2]
    ip, jp, ratio = (int(v) for v in (sys.argv[3:6] if len(sys.argv) > 5 else (26, 26, 3)))
    d1 = netCDF4.Dataset(IN + "/wrfinput_d01")
    d2 = netCDF4.Dataset(IN + "/wrfinput_d02")
    nat_ds = netCDF4.Dataset(sorted(glob.glob(NAT + "/wrfout_d02_*"))[0])

    def r(ds, n):
        return A(ds[n][0])

    def nat(n):
        return A(nat_ds[n][0])

    def raw_of(ds, ht=None):
        raw = {n: np.asarray(ds[n][0]) for n in BASE_NAMES}
        if ht is not None:
            raw["HGT"] = ht
        return raw

    def cmp(name, mine, theirs):
        d = mine.view(np.uint32) != theirs.view(np.uint32)
        print(f"{name}: {int(d.sum())} of {d.size} words differ")

    ny, nx = r(d2, "HGT").shape
    pny, pnx = r(d1, "HGT").shape
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=ip, j_parent_start=jp,
                        child_nx=nx, child_ny=ny, parent_nx=pnx, parent_ny=pny,
                        stagger="", wrapper="interp")
    col = lambda n: r(d2, n)[:, None, None]  # noqa: E731
    c3h, c4h, c3f, c4f, c1h, c2h, c1f, c2f = (col(n) for n in (
        "C3H", "C4H", "C3F", "C4F", "C1H", "C2H", "C1F", "C2F"))
    p_top = f32(np.asarray(d2["P_TOP"][0]).reshape(-1)[0])
    th, qv, ph, mu = r(d2, "T"), r(d2, "QVAPOR"), r(d2, "PH"), r(d2, "MU")
    ht_fine = r(d2, "HGT")
    # 1. input + start_domain on the file
    base_file = wrf_start_base_state(raw_of(d2), 2)
    _, p_file = start_eos(base_file, ph, mu, th, qv, (c3h, c4h, c3f, c4f), p_top)
    mub_save = A(base_file["MUB"])
    # 2. parent captures (the parent's REBUILT MUB) and the FP32 blend
    base_parent = wrf_start_base_state(raw_of(d1), 2)
    ht = blend(A(sint(r(d1, "HGT"), reg, preparation=True)), ht_fine)
    mub = blend(A(sint(A(base_parent["MUB"]), reg, preparation=True)), mub_save)
    # 3. adjust_tempqv
    th_a, qv_a = adjust_tempqv(th, qv, p_file, mub_save, mub, c3h, c4h, p_top)
    # 5. start_domain(nest) on the blended terrain, press_adj
    base = wrf_start_base_state(raw_of(d2, ht), 2)
    al, p = start_eos(base, ph, mu, th_a, qv_a, (c3h, c4h, c3f, c4f), p_top)
    alb = A(base["ALB"])
    mu_adj = A(mu + A(A(A(al[0] / A(A(al[0] + alb[0]) * alb[0])) * G) * A(ht - ht_fine)))
    # 6. med_nest_force: couple and uncouple
    MB = A(base["MUB"])

    def rt(x, c1, c2):
        m = A(A(A(c1 * MB[None]) + c2) + A(c1 * mu_adj[None]))
        return A(A(x * m) * A(f32(1.0) / m))
    for name, mine in (("HGT", ht), ("MUB", MB), ("PB", A(base["PB"])),
                       ("PHB", A(base["PHB"])), ("MU", mu_adj), ("P", p),
                       ("THM", rt(th_a, c1h, c2h)), ("QVAPOR", rt(qv_a, c1h, c2h)),
                       ("PH", rt(ph, c1f, c2f))):
        cmp(name, mine, nat(name))
    a = A(A(c1h * MB[None]) + c2h)
    b = A(c1h * mu_adj[None])
    for name, axis, msf in (("U", 2, r(d2, "MAPFAC_UY")), ("V", 1, r(d2, "MAPFAC_VX"))):
        lo = (slice(None),) * axis + (slice(None, -1),)
        hi = (slice(None),) * axis + (slice(1, None),)
        s = A(A(A(a[hi] + a[lo]) + b[hi]) + b[lo])
        inner = (slice(None),) * axis + (slice(1, -1),)
        m = msf[None][inner[:1] + inner[1:]] if axis == 2 else msf[None][:, 1:-1, :]
        if axis == 2:
            m = msf[None][:, :, 1:-1]
        x = r(d2, name)[inner]
        x = A(A(x * A(A(f32(0.5) * s) / m)) * A(A(f32(2.0) * m) / s))
        cmp(f"{name} (interior faces)", x, nat(name)[inner])


if __name__ == "__main__":
    main()
