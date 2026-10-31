"""WRF v4.6.1's start of a nest read from its own wrfinput, for the WRF-file door.

wrf.exe does not integrate a nest's wrfinput as the file holds it.  When a
nest opens with ``input_from_file`` (the only nest start the door admits),
``med_nest_initial`` (share/mediation_integrate.F:509-952) runs this
transaction before the nest's first step:

1. ``med_initialdata_input`` reads wrfinput_d0N and runs ``start_domain`` on
   it: the base state (MUB, PB, T_INIT, ALB, PHB) is rebuilt from the file's
   HGT (start_em.F:554-636) and P/AL are re-derived from the file's T, PH,
   MU and QVAPOR (start_em.F:774-842).  That rebuilt MUB is ``mub_save``
   and that P is the ``pp`` adjust_tempqv reads -- neither is the file's.
2. ``ht_int``/``mub_fine`` are SINT captures of the PARENT's HGT and its
   rebuilt MUB; when the file says SAVE_TOPO_FROM_REAL = 1, blend_terrain
   (nest_init_utils.F:712-785) blends them into the nest's outer
   spec_bdy_width + blend_width rows, in FP32 with
   ``r_blend_zones = 1./(blend_width+1)``.
3. ``adjust_tempqv`` (nest_init_utils.F:812-890) moves T and QVAPOR to the
   blended column mass at constant relative humidity.
4. ``start_domain(nest)`` with ``press_adj``: the base state is rebuilt from
   the BLENDED HGT, P/AL are re-derived from the file's PH and MU and the
   adjusted T/QV, and then
   ``MU_2 += al(1)/(alt(1)*alb(1))*g*(ht - ht_fine)`` (start_em.F:878-894).

Steps 1-3 change only host words of the nest's file, so
:func:`nest_input_start` applies them to the restored file before it is
uploaded: the restore then rebuilds the base state from the blended HGT
and diagnoses P/AL exactly as step 4 does.  :func:`press_adj_mu` is step
4's mass move, applied after that diagnosis.

Every operation is FP32 in Fortran's order with glibc 2.39's ``powf``,
``logf`` and ``expf`` (the pinned strict wrf.exe 4.6.1 recipe).  Measured
on the UAE 9 km/3 km pair (physics off, feedback = 1) against that build:
the nest's HGT, MUB, PB, PHB, MU and P are wrf.exe's word for word
(tools/wrf461_nest_initial/nest_initial_oracle.py).  Before this, the door
restored each nest file as it is: at t = 0 the nest's P differed from
wrf.exe in 1,381,567 words and its QVAPOR in 826,343.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

f32 = np.float32

_R_D, _R_V, _G = f32(287.0), f32(461.6), f32(9.81)
_RVOVRD = f32(_R_V / _R_D)
_CP = f32(f32(f32(7.0) * _R_D) / f32(2.0))
_CPOVCV = f32(_CP / f32(_CP - _R_D))
_P1000, _T0 = f32(100000.0), f32(300.0)
_TWO7 = f32(f32(2.0) / f32(7.0))


def _a(x):
    return np.asarray(x, np.float32)


@dataclass(frozen=True)
class NestInputStart:
    """One door nest after med_nest_initial's host half.

    ``restored`` is the nest's file with WRF's blended HGT and adjusted
    T/QVAPOR in place of the file's words; ``ht_fine`` is the file's own
    HGT, which press_adj reads.
    """
    restored: object
    ht_fine: np.ndarray
    receipt: MappingProxyType
    #: The file's own T: WRF's history T is th_phy_m_t0, which the input
    #: sets and neither adjust_tempqv nor feedback touches, so the nest's
    #: first frame writes it while THM carries the adjusted t_2.
    file_t: np.ndarray | None = None
    #: MU after press_adj, from the start diagnosis on the blended base.
    #: Formed here because the nest's own children may feed back into its
    #: terrain before the tree restores it, and WRF's press_adj ran first.
    mu_press_adj: np.ndarray | None = None


def blend_terrain_fp32(coarse, fine, *, spec_bdy_width: int, blend_width: int):
    """blend_terrain (nest_init_utils.F:712-785) in FP32.

    The blend_cell loop runs from blend_width down to 1 and the last match
    wins, so a cell on two frames takes the smaller blend_cell; the
    spec_bdy_width rows then take the parent's word.
    """
    fine, coarse = _a(fine), _a(coarse)
    if fine.shape != coarse.shape or fine.ndim != 2:
        raise ValueError("blend_terrain needs two equal (ny, nx) planes")
    ny, nx = fine.shape
    ide, jde = nx + 1, ny + 1
    sbw, width = int(spec_bdy_width), int(blend_width)
    i1 = np.arange(1, nx + 1)[None, :]
    j1 = np.arange(1, ny + 1)[:, None]
    rz = f32(f32(1.0) / f32(width + 1))
    out = fine.copy()
    for cell in range(width, 0, -1):
        hit = ((i1 == sbw + cell) | (j1 == sbw + cell)
               | (i1 == ide - sbw - cell) | (j1 == jde - sbw - cell))
        mixed = _a(_a(_a(f32(cell) * fine) + _a(f32(width + 1 - cell) * coarse)) * rz)
        out = np.where(hit, mixed, out)
    spec = (i1 <= sbw) | (j1 <= sbw) | (i1 >= ide - sbw) | (j1 >= jde - sbw)
    return _a(np.where(spec, coarse, out))


def _expf(x):
    from woof.ingest.wrfinput import _glibc_expf_array
    return _glibc_expf_array(x)


def _es(tc):
    return _a(f32(610.78) * _expf(_a(_a(f32(17.0809) * tc) / _a(f32(234.175) + tc))))


def adjust_tempqv_fp32(th, qv, pp, mub_save, mub, c3h, c4h, p_top):
    """adjust_tempqv (nest_init_utils.F:812-890, use_theta_m = 0) in FP32.

    ``th`` is WRF's t_2 (perturbation potential temperature), ``pp`` the
    perturbation pressure start_domain derived on the file, ``mub_save``
    the file's rebuilt MUB and ``mub`` the blended one.  Returns the
    adjusted (th, qv).
    """
    from woof.core.noahmp_libm import powf_array

    th, qv, pp = _a(th), _a(qv), _a(pp)
    c3h, c4h = _a(c3h)[:, None, None], _a(c4h)[:, None, None]
    p_old = _a(_a(_a(c4h + _a(c3h * _a(mub_save)[None])) + p_top) + pp)
    tc = _a(_a(_a(th + f32(300.0)) * powf_array(_a(p_old / f32(1.e5)), _TWO7))
            - f32(273.15))
    e = _a(_a(qv * p_old) / _a(f32(0.622) + qv))
    rh = _a(e / _es(tc))
    p_new = _a(_a(_a(c4h + _a(c3h * _a(mub)[None])) + p_top) + pp)
    thloc = _a(th + f32(300.0))
    total, step = _a(p_new + p_old), _a(p_new - p_old)
    dth1 = _a(_a(_a(f32(-191.86e-3) * thloc) / total) * step)
    dth = _a(_a(_a(f32(-191.86e-3) * _a(thloc + _a(f32(0.5) * dth1))) / total) * step)
    th_new = _a(_a(thloc + dth) - f32(300.0))
    tc = _a(_a(_a(thloc + dth) * powf_array(_a(p_new / f32(1.e5)), _TWO7))
            - f32(273.15))
    e = _a(rh * _es(tc))
    return th_new, _a(_a(f32(0.622) * e) / _a(p_new - e))


def start_perturbation_pressure(base, raw, *, hypsometric_opt: int):
    """start_em.F:774-842's AL then P on a cold start (t_1 = t_2, ph_1 =
    ph_2, mu_1 = mu_2), use_theta_m = 0, FP32.  Returns (al, p)."""
    from woof.core.noahmp_libm import logf_array, powf_array

    def col(name):
        return _a(np.asarray(raw[name]).reshape(-1))[:, None, None]

    p_top = f32(np.asarray(raw["P_TOP"]).reshape(-1)[0])
    ph1, mu1, t1, qv = (_a(raw[n]) for n in ("PH", "MU", "T", "QVAPOR"))
    mub, pb, alb, phb = (_a(base[n]) for n in ("MUB", "PB", "ALB", "PHB"))
    c1h, c2h, c3h, c4h, c3f, c4f = (col(n) for n in (
        "C1H", "C2H", "C3H", "C4H", "C3F", "C4F"))
    if hypsometric_opt == 2:
        mt = _a(mub + mu1)[None]

        def level(c3, c4):
            return _a(_a(_a(c3 * mt) + c4) + p_top)
        pfu, pfd = level(c3f[1:], c4f[1:]), level(c3f[:-1], c4f[:-1])
        phm = level(c3h, c4h)
        thick = _a(_a(_a(ph1[1:] - ph1[:-1]) + phb[1:]) - phb[:-1])
        al = _a(_a(_a(thick / phm) / logf_array(_a(pfd / pfu))) - alb)
    elif hypsometric_opt == 1:
        rdnw = col("RDNW")
        column = _a(_a(_a(c1h * mub[None]) + c2h) + _a(c1h * mu1[None]))
        lead = _a(f32(-1.0) / column)
        al = _a(lead * _a(_a(alb * _a(c1h * mu1[None]))
                          + _a(rdnw * _a(ph1[1:] - ph1[:-1]))))
    else:
        raise ValueError(f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    qvf = _a(f32(1.0) + _a(_RVOVRD * qv))
    ratio = _a(_a(_a(_R_D * _a(_T0 + t1)) * qvf) / _a(_P1000 * _a(al + alb)))
    return al, _a(_a(_P1000 * powf_array(ratio, _CPOVCV)) - pb)


def press_adj_mu(mu, al1, alb1, ht, ht_fine):
    """start_em.F:878-894: ``MU_2 + al(1)/(alt(1)*alb(1))*g*(ht - ht_fine)``,
    FP32 in Fortran's order, ``alt = al + alb``."""
    al1, alb1 = _a(al1), _a(alb1)
    alt1 = _a(al1 + alb1)
    move = _a(_a(_a(al1 / _a(alt1 * alb1)) * _G) * _a(_a(ht) - _a(ht_fine)))
    return _a(_a(mu) + move)


def nest_input_start(nest, parent, *, i_parent_start: int, j_parent_start: int,
                     ratio: int, spec_bdy_width: int, blend_width: int,
                     hypsometric_opt: int) -> NestInputStart:
    """Steps 1-3 of med_nest_initial on a door nest's restored file.

    ``parent`` is the parent's restored file AS WRF HOLDS IT when the nest
    opens: for a nest of a nest that is the parent's own
    :class:`NestInputStart` restored file (its blended HGT).
    """
    from woof.core.nest_interp import register_nest, sint
    from woof.ingest.wrfinput import wrf_start_base_state

    raw, praw = nest.raw, parent.raw
    ny, nx = _a(raw["HGT"]).shape
    pny, pnx = _a(praw["HGT"]).shape
    reg = register_nest(nri=int(ratio), nrj=int(ratio),
                        i_parent_start=int(i_parent_start),
                        j_parent_start=int(j_parent_start),
                        child_nx=nx, child_ny=ny, parent_nx=pnx, parent_ny=pny,
                        stagger="", wrapper="interp")
    p_top = f32(np.asarray(raw["P_TOP"]).reshape(-1)[0])
    ht_fine = _a(raw["HGT"]).copy()
    # 1. start_domain on the file: mub_save and adjust_tempqv's pp.
    file_base = wrf_start_base_state(raw, int(hypsometric_opt))
    _, p_file = start_perturbation_pressure(
        file_base, raw, hypsometric_opt=int(hypsometric_opt))
    mub_save = _a(file_base["MUB"])
    # 2. the parent's HGT and REBUILT MUB on the nest grid, blended.
    save_topo = np.asarray(raw.get("SAVE_TOPO_FROM_REAL", 0)).reshape(-1)
    blended = bool(save_topo.size and int(save_topo[0]) == 1)
    ht, mub = ht_fine, mub_save
    if blended:
        parent_base = wrf_start_base_state(praw, int(hypsometric_opt))
        width = dict(spec_bdy_width=int(spec_bdy_width), blend_width=int(blend_width))
        ht = blend_terrain_fp32(
            _a(sint(_a(praw["HGT"]), reg, preparation=True)), ht_fine, **width)
        mub = blend_terrain_fp32(
            _a(sint(_a(parent_base["MUB"]), reg, preparation=True)), mub_save, **width)
    # 3. adjust_tempqv.
    th, qv = adjust_tempqv_fp32(
        raw["T"], raw["QVAPOR"], p_file, mub_save, mub,
        np.asarray(raw["C3H"]).reshape(-1), np.asarray(raw["C4H"]).reshape(-1), p_top)
    updated = dict(raw)
    updated.update(HGT=ht, T=th, QVAPOR=qv)
    # 4. start_domain(nest) on the blended terrain: AL from the file's PH
    # and MU with the adjusted T/QV, then press_adj.
    start_base = wrf_start_base_state(updated, int(hypsometric_opt))
    al, _ = start_perturbation_pressure(
        start_base, updated, hypsometric_opt=int(hypsometric_opt))
    mu_adj = press_adj_mu(raw["MU"], al[0], _a(start_base["ALB"])[0], ht, ht_fine)
    receipt = {
        "transaction": "WRF 4.6.1 med_nest_initial, input_from_file",
        "blend_terrain": ("SAVE_TOPO_FROM_REAL = 1: parent HGT and rebuilt MUB "
                          f"blended over spec_bdy_width {int(spec_bdy_width)} + "
                          f"blend_width {int(blend_width)} rows")
                         if blended else "SAVE_TOPO_FROM_REAL = 0: not blended",
        "adjust_tempqv": "T and QVAPOR moved to the blended column mass at constant RH",
        "press_adj": "MU += al(1)/(alt(1)*alb(1))*g*(HGT - file HGT) after the start diagnosis",
        "terrain_cells_changed": int(np.count_nonzero(ht.view(np.uint32) != ht_fine.view(np.uint32))),
    }
    return NestInputStart(replace(nest, raw=MappingProxyType(updated)),
                          ht_fine, MappingProxyType(receipt),
                          _a(raw["T"]).copy(), mu_adj)


def apply_press_adj(state, start: NestInputStart) -> None:
    """Step 4's mass move on the restored nest, after its initial feedback.

    WRF feeds the nest back before its start_domain, so the restriction
    reads the file's MU; MU_1 = MU_2 follows (start_em.F:890-894).
    """
    import cupy as cp

    state.mup[...] = cp.asarray(start.mu_press_adj)
    if getattr(state, "mup0", None) is not None:
        state.mup0[...] = state.mup


#: WRF v4.6.1's nest feedback for the words a WRF input file carries
#: (inc/nest_feedbackup_interp.inc, generated from the Registry's ``u``
#: flags): ``avg`` is copy_fcn's cell average, ``pick`` copy_fcnm's centre
#: point, ``ipick`` copy_fcni's.  U, V, W, PH, THM, MU and every moist and
#: scalar species are restricted on the card by the tree's live feedback
#: (woof.core.nest.NestCoupler.feedback_commit); MUB, PHB, ALB and PB are
#: fed back too, but the parent's start_domain then rebuilds them from the
#: fed-back HGT, as the door's restore does.  HGT is fed back by
#: :func:`feed_back_file_fields` itself.
FILE_FEEDBACK = MappingProxyType({
    **{name: ("avg", "") for name in (
        "Q2", "T2", "TH2", "PSFC", "U10", "V10", "QNWFA2D", "QNIFA2D",
        "TOPOSLPX", "TOPOSLPY", "LANDUSEF", "SOILCTOP", "SOILCBOT",
        "H_DIABATIC", "RAINC", "RAINNC", "PCB")},
    **{name: ("pick", "") for name in (
        "LU_INDEX", "SHDMAX", "SHDMIN", "SHDAVG", "TSLB", "SMOIS", "SH2O",
        "SMCREL", "SEAICE", "VEGFRA", "ACRUNOFF", "SNOW", "SNOWH", "CANWAT",
        "FRC_URB2D", "OM_TMP", "OM_S", "OM_U", "OM_V", "MAPFAC_M",
        "MAPFAC_MX", "MAPFAC_MY", "F", "E", "SINALPHA", "COSALPHA", "TSK",
        "CLAT", "TMN", "XLAND", "SNOWC", "TML", "T0ML", "HML", "H0ML",
        "HUML", "HVML", "TMOML", "LANDMASK", "LAKEMASK")},
    **{name: ("pick", "x") for name in (
        "MAPFAC_U", "MAPFAC_UX", "MAPFAC_UY", "XLAT_U", "XLONG_U")},
    **{name: ("pick", "y") for name in (
        "MAPFAC_V", "MAPFAC_VX", "MF_VX_INV", "MAPFAC_VY", "XLAT_V", "XLONG_V")},
    **{name: ("ipick", "") for name in ("IVGTYP", "ISLTYP")},
})

#: Fed-back words WRF's parent smoother then filters (Registry ``s`` flag,
#: inc/nest_feedbackup_smooth.inc), among the file words above and HGT.
_SMOOTHED = frozenset({"HGT", "PCB"})

#: mark_domain's word (interp_fcn.F, "Beverly Hills * 100").
NEST_POS_MARK = f32(9021000.0)


def _span(ipos, nested, ratio, spec_zone, stag):
    """copy_fcn/copy_fcnm's 1-based parent loop bounds on one axis."""
    lo = int(ipos) + int(spec_zone)
    hi = int(ipos) + int(nested) // int(ratio) - (0 if stag else 1) - int(spec_zone)
    return lo, hi


def _feedback_host(cfld, nfld, *, kind, stagger, ratio, ipos, jpos, spec_zone):
    """copy_fcn (mass), copy_fcnm or copy_fcni on host arrays, WRF's order.

    ``cfld``/``nfld`` are the parent and nest words (leading level axis
    optional).  Returns the parent array with the restricted rectangle
    written.
    """
    c = np.array(cfld, copy=True)
    n = np.asarray(nfld)
    xstag, ystag = stagger == "x", stagger == "y"
    nx_mass = n.shape[-1] - (1 if xstag else 0)
    ny_mass = n.shape[-2] - (1 if ystag else 0)
    r = int(ratio)
    ci0, ci1 = _span(ipos, nx_mass, r, spec_zone, xstag)
    cj0, cj1 = _span(jpos, ny_mass, r, spec_zone, ystag)
    if ci1 < ci0 or cj1 < cj0:
        return c
    ci = np.arange(ci0, ci1 + 1)
    cj = np.arange(cj0, cj1 + 1)
    istag, jstag = (0 if xstag else 1), (0 if ystag else 1)
    odd = r % 2 != 0
    if kind == "avg":
        if xstag or ystag:
            raise ValueError("the file feedback averages mass words only")
        w = f32(f32(1.0) / f32(r * r))
        if odd:
            ni = (ci - ipos) * r + r // 2 + 1
            nj = (cj - jpos) * r + r // 2 + 1
            points = [((ij - 1) % r - r // 2, (ij - 1) // r - r // 2)
                      for ij in range(1, r * r + 1)]
        else:
            ni = (ci - ipos) * r + istag
            nj = (cj - jpos) * r + jstag
            points = [((ij - 1) % r, (ij - 1) // r) for ij in range(1, r * r + 1)]
        acc = np.zeros(n.shape[:-2] + (cj.size, ci.size), np.float32)
        for di, dj in points:
            donor = n[..., (nj + dj - 1)[:, None], (ni + di - 1)[None, :]]
            acc = (acc + (w * donor.astype(np.float32)).astype(np.float32)).astype(np.float32)
        c[..., (cj - 1)[:, None], (ci - 1)[None, :]] = acc.astype(c.dtype)
        return c
    if odd:
        ni = (ci - ipos) * r + istag + 1
        nj = (cj - jpos) * r + jstag + 1
    else:
        ni = (ci - ipos) * r + 1 + (r // 2 - 1)
        nj = (cj - jpos) * r + 1 + (r // 2 - 1)
    c[..., (cj - 1)[:, None], (ci - 1)[None, :]] = n[..., (nj - 1)[:, None], (ni - 1)[None, :]]
    return c


def _smooth_host(field, *, smooth_option, ratio, ipos, jpos, nest_nx, nest_ny):
    """WRF's parent smoother (interp_fcn.F sm121 / smdsm), mass words, FP32."""
    if int(smooth_option) == 0:
        return field
    c = np.array(field, dtype=np.float32, copy=True)
    squeeze = c.ndim == 2
    if squeeze:
        c = c[None]
    r = int(ratio)
    i0, i1 = int(ipos) + 2, int(ipos) + int(nest_nx) // r - 2 - 1
    j0, j1 = int(jpos) + 2, int(jpos) + int(nest_ny) // r - 2 - 1
    if i1 < i0 or j1 < j0:
        return field
    I = slice(i0 - 1, i1)
    J = slice(j0 - 1, j1)
    Ip, Im = slice(i0, i1 + 1), slice(i0 - 2, i1 - 1)
    Jp, Jm = slice(j0, j1 + 1), slice(j0 - 2, j1 - 1)
    if int(smooth_option) == 1:
        new = c.copy()
        new[:, J, I] = (f32(0.25) * ((c[:, Jp, I] + f32(2.0) * c[:, J, I]).astype(np.float32)
                                     + c[:, Jm, I])).astype(np.float32)
        c[:, J, I] = (f32(0.25) * ((new[:, J, Ip] + f32(2.0) * new[:, J, I]).astype(np.float32)
                                   + new[:, J, Im])).astype(np.float32)
    elif int(smooth_option) == 2:
        for xnu in (f32(0.50), f32(-0.52)):
            new = (c[:, J, I] + xnu * (((c[:, Jp, I] + c[:, Jm, I]).astype(np.float32)
                                        * f32(0.5)) - c[:, J, I])).astype(np.float32)
            c[:, J, I] = new
            new = (c[:, J, I] + xnu * (((c[:, J, Ip] + c[:, J, Im]).astype(np.float32)
                                        * f32(0.5)) - c[:, J, I])).astype(np.float32)
            c[:, J, I] = new
    else:
        raise ValueError(f"smooth_option must be 0, 1 or 2, got {smooth_option}")
    return c[0] if squeeze else c


def feed_back_file_fields(parent, nest, *, i_parent_start: int, j_parent_start: int,
                          ratio: int, spec_zone: int, smooth_option: int):
    """med_nest_feedback's file words, nest into parent, before the parent restarts.

    ``nest`` is the nest's file after :func:`nest_input_start` (blended
    HGT).  HGT is averaged (copy_fcn), smoothed when smooth_option asks,
    and then kept only where mark_domain set NEST_POS
    (share/mediation_feedback_domain.F: ``WHERE (nest_pos /= 9021000.)
    ht = ht_coarse``); every other word in :data:`FILE_FEEDBACK` that both
    files carry takes the nest's restriction.  The parent's restore then
    rebuilds its base state from the fed-back HGT, as its second
    start_domain does.
    """
    raw, nraw = dict(parent.raw), nest.raw
    ny, nx = _a(raw["HGT"]).shape
    nny, nnx = _a(nraw["HGT"]).shape
    geometry = dict(ratio=int(ratio), ipos=int(i_parent_start),
                    jpos=int(j_parent_start), spec_zone=int(spec_zone))
    smoothing = dict(smooth_option=int(smooth_option), ratio=int(ratio),
                     ipos=int(i_parent_start), jpos=int(j_parent_start),
                     nest_nx=nnx, nest_ny=nny)
    mark = np.zeros((ny, nx), np.float32) if "NEST_POS" not in raw else _a(raw["NEST_POS"]).copy()
    r = int(ratio)
    mi0, mi1 = int(i_parent_start) + 1, int(i_parent_start) + nnx // r - 1 - 1
    mj0, mj1 = int(j_parent_start) + 1, int(j_parent_start) + nny // r - 1 - 1
    if mi1 >= mi0 and mj1 >= mj0:
        mark[mj0 - 1:mj1, mi0 - 1:mi1] = NEST_POS_MARK
    raw["NEST_POS"] = mark
    coarse = _a(raw["HGT"]).copy()
    ht = _feedback_host(coarse, nraw["HGT"], kind="avg", stagger="", **geometry)
    ht = _smooth_host(ht, **smoothing)
    raw["HGT"] = np.where(mark != NEST_POS_MARK, coarse, ht).astype(np.float32)
    fed = ["HGT"]
    for name, (kind, stagger) in FILE_FEEDBACK.items():
        if name not in raw or name not in nraw:
            continue
        value = _feedback_host(raw[name], nraw[name], kind=kind, stagger=stagger, **geometry)
        if name in _SMOOTHED:
            value = _smooth_host(value, **smoothing)
        raw[name] = value
        fed.append(name)
    return replace(parent, raw=MappingProxyType(raw))


def restart_parent_domain(parent, parent_restored) -> None:
    """The parent's second start_domain after its nest's initial feedback.

    med_nest_initial restarts the parent (share/mediation_integrate.F:800-838)
    on its first trip: t_1/ph_1/mu_1 take the fed-back t_2/ph_2/mu_2, P and
    AL are re-derived over the whole parent from them and the base its
    restore rebuilt from the fed-back HGT, and the cold-start W is
    re-diagnosed from the fed-back HGT, U and V (start_em.F:1475-1532).
    """
    from woof.core.diagnostics import update_diagnostics
    from woof.ingest.wrfinput_startup import initialize_wrfinput_vertical_velocity

    state, run = parent.state, parent.cfg.run
    update_diagnostics(state, int(run.hypsometric_opt))
    initialize_wrfinput_vertical_velocity(
        state, parent_restored, run,
        periodic_x=not (run.open_x or run.specified or run.nested),
        periodic_y=not (run.open_y or run.specified or run.nested))
    for current, initial in (("thp", "thp0"), ("php", "php0"), ("mup", "mup0"),
                             ("u", "u0"), ("v", "v0"), ("w", "w0")):
        source, target = getattr(state, current, None), getattr(state, initial, None)
        if source is not None and target is not None:
            target[...] = source


def finish_strict_nest_start(node, start: NestInputStart, parent_restored) -> None:
    """Strict mode's part of the nest start, after press_adj.

    The nest's first step reads the p/al its start_domain left (before
    press_adj moved MU and the first force round-tripped its words), not a
    re-diagnosis.  Each domain's first frame writes WRF's th_phy_m_t0 as T:
    the file's word, which the nest's adjust_tempqv and the parent's
    feedback leave alone.  (The first force's round trip itself is the
    coupler's, woof.core.nest_round_trip.)
    """
    node.state._wrf_start_diagnostics_carried = True
    node.state._wrf_start_history_t = start.file_t
    parent = node.parent
    if parent is not None and getattr(parent.state, "_wrf_start_history_t", None) is None:
        parent.state._wrf_start_history_t = _a(parent_restored.raw["T"]).copy()
