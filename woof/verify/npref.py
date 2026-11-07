"""Float64 NumPy mirrors of the CUDA kernels, for verification only.

Every GPU kernel test compares device FP32 output against the reference
implementation here.  Each function mirrors one kernel's math exactly (same
discretization, full float64).  Later tasks append their own references:
advection fluxes (Task 8), acoustic pieces (Tasks 10-11), diffusion (Task 13).
"""

from __future__ import annotations

import math
import numpy as np

from woof.core import constants as c
from woof.core.morrison_constants import rimed_ice_constants


def _prof3(a):
    """Broadcast a base-state column (1-D) to (n, 1, 1); pass 3-D through."""
    a = np.asarray(a, dtype=np.float64)
    return a[:, None, None] if a.ndim == 1 else a


def np_calc_p_alpha(thp, php, mup, base, coord, qv=None, hypsometric_opt=1):
    """Mirror of ``calc_p_alpha`` (woof/core/kernels/diagnostics.cu).

    General hybrid/terrain form (WRF v4.6.1 ``calc_p_rho_phi``,
    dyn_em/module_big_step_utilities_em.F:1025-1052, NONHYDROSTATIC
    branch), keyed on ``hypsometric_opt`` exactly as WRF:

    - ``hypsometric_opt=1`` (the frozen woof default): the total specific
      volume comes from the discrete hydrostatic relation with the hybrid
      column-mass increment, ``alt = -d(phi)/d(eta) / (c1h*mu + c2h)`` with
      ``mu = mub + mu'``, algebraically WRF's
      ``al = -1/(c1*muts+c2)*(alb*c1*mu' + rdnw*dphi')`` (F:1029) given the
      discrete base-state identity ``alb = -rdnw*dphb/(c1h*mub+c2h)``.
      This branch is bitwise the pre-hypsometric_opt mirror.
    - ``hypsometric_opt=2`` (the reference WRF run's Registry default):
      the log-pressure hypsometric form, verbatim F:1042-1051 ::

          pfu = c3f(k+1)*MUTS + c4f(k+1) + ptop
          pfd = c3f(k  )*MUTS + c4f(k  ) + ptop
          phm = c3h(k  )*MUTS + c4h(k  ) + ptop
          al  = (dphi' + dphb)/phm/LOG(pfd/pfu) - alb

      with ``MUTS = mub + mu'`` the total dry column mass and the pressure
      then evaluated on ``al + alb`` exactly as WRF's moist_nonhydro block
      (F:1061-1066, use_theta_m=0 arm).  Requires the coord's
      pressure-weighted c4 coefficients finalized for ``base.p_top``.

    Base profiles may be Phase 1 flat columns (1-D, scalar ``mub``) or
    per-column terrain fields (3-D, 2-D ``mub``); for ``hybrid_opt=0``
    (c1h = 1, c2h = 0) opt 1 reduces bitwise to the Phase 1
    ``-d(phi)/d(eta) / mu``.

    Moist form (Task 5): with a vapor mixing ratio ``qv (nz,ny,nx)`` the
    EOS uses the moist potential temperature ``theta_m = theta *
    (1 + Rv/Rd*qv)`` (ARW Tech Note ch. 2; the ratio is computed from the
    constants, never hardcoded).  ``alt`` stays the DRY specific volume
    (the geopotential/dry-mass relation is unchanged); only the pressure
    picks up theta_m.  ``qv=None`` is the dry Phase 1 path, bitwise.

    Parameters: perturbation fields ``thp (nz,ny,nx)``, ``php (nz+1,ny,nx)``,
    ``mup (ny,nx)`` as float64 arrays, plus the ``BaseState`` and
    ``VerticalCoord`` used to build the run.  Returns ``(p, al, alt)``,
    each ``(nz, ny, nx)`` float64.
    """
    thb, phb, alb = _prof3(base.thb), _prof3(base.phb), _prof3(base.alb)
    rdnw = np.asarray(coord.rdnw, dtype=np.float64)[:, None, None]
    c1h = np.asarray(coord.c1h, dtype=np.float64)[:, None, None]
    c2h = np.asarray(coord.c2h, dtype=np.float64)[:, None, None]

    th = thb + thp                       # total theta        (nz, ny, nx)
    if qv is not None:                   # theta_m in the EOS (moist)
        th = th * (1.0 + c.RVOVRD * np.asarray(qv, dtype=np.float64))
    mu = np.asarray(base.mub, dtype=np.float64) + mup   # total dry mass
    ph = phb + php                       # total geopotential (nz+1, ny, nx)

    if hypsometric_opt == 1:
        alt = -(ph[1:] - ph[:-1]) * rdnw / (c1h * mu + c2h)  # rdnw < 0
        al = alt - alb
    elif hypsometric_opt == 2:
        if coord.p_top is None or coord.p_top != base.p_top:
            raise ValueError(
                "hypsometric_opt=2 needs the coord's pressure-weighted "
                f"c4 coefficients finalized for base.p_top={base.p_top}; "
                f"coord.p_top={coord.p_top}")
        p_top = float(base.p_top)
        c3f = np.asarray(coord.c3f, dtype=np.float64)[:, None, None]
        c4f = np.asarray(coord.c4f, dtype=np.float64)[:, None, None]
        c3h = np.asarray(coord.c3h, dtype=np.float64)[:, None, None]
        c4h = np.asarray(coord.c4h, dtype=np.float64)[:, None, None]
        pfu = c3f[1:] * mu + c4f[1:] + p_top
        pfd = c3f[:-1] * mu + c4f[:-1] + p_top
        phm = c3h * mu + c4h + p_top
        al = (ph[1:] - ph[:-1]) / phm / np.log(pfd / pfu) - alb
        alt = al + alb
    else:
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    p = c.P0 * ((c.RD * th) / (c.P0 * alt)) ** c.GAMMA
    return p, al, alt


def _flux5(qm3, qm2, qm1, q0, qp1, qp2, vel):
    """WRF 5th-order upwind face flux; face f lies between cells f-1 and f."""
    return (vel * (37.0 * (q0 + qm1) - 8.0 * (qp1 + qm2) + (qp2 + qm3))
            - np.abs(vel) * (10.0 * (q0 - qm1) - 5.0 * (qp1 - qm2)
                             + (qp2 - qm3))) / 60.0


def _flux3(qm2, qm1, q0, qp1, vel):
    """VERTICAL 3rd-order face flux; face f lies between cells f-1 and f.

    Carries WRF's upwind-on-(-vel) sign: ``vel`` is the eta-directed
    (Omega-signed) flux, so the dissipation term has the opposite sign to
    ``_flux5``'s.  Only used vertically (``_fz_half_levels`` and
    ``np_flux_div_w``).
    """
    return (vel * (7.0 * (q0 + qm1) - (qp1 + qm2))
            + np.abs(vel) * (3.0 * (q0 - qm1) - (qp1 - qm2))) / 12.0


def _flux5v(qm3, qm2, qm1, q0, qp1, qp2, vel):
    """VERTICAL 5th-order face flux: WRF ``flux5`` called with ``-vel``
    (module_advect_em.F ``vert_order == 5``), so its dissipation carries
    the Omega-signed sign of ``_flux3``, opposite to ``_flux5``'s.
    Mirrors the kernels' ``flux5v``.
    """
    return (vel * (37.0 * (q0 + qm1) - 8.0 * (qp1 + qm2) + (qp2 + qm3))
            + np.abs(vel) * (10.0 * (q0 - qm1) - 5.0 * (qp1 - qm2)
                             + (qp2 - qm3))) / 60.0


def _flux3h(qm2, qm1, q0, qp1, vel):
    """HORIZONTAL 3rd-order face flux (WRF ``flux3`` with the ``flux5``
    upwinding sign): used two cells in from an open lateral boundary where
    WRF degrades the 5th-order stencil (module_advect_em.F degrade blocks).
    """
    return (vel * (7.0 * (q0 + qm1) - (qp1 + qm2))
            - np.abs(vel) * (3.0 * (q0 - qm1) - (qp1 - qm2))) / 12.0


def _open_cell_fluxes(q, vel, axis):
    """WRF open-boundary face fluxes along ``axis`` for a CELL-type field.

    ``q`` has n cells along ``axis``; ``vel`` carries the advecting flux at
    the n+1 faces.  Faces 0 and n are never consumed (the boundary cells
    take the open advective term instead) and return 0; one face in is
    2nd-order centered, two in is the 3rd-order ``_flux3h``, the interior
    is the full ``_flux5``: WRF's ``degrade_xs/xe/ys/ye`` blocks
    (module_advect_em.F, horz_order == 5).  Requires n >= 7 so the order
    bands cannot overlap.
    """
    q = np.moveaxis(np.asarray(q, dtype=np.float64), axis, -1)
    vel = np.moveaxis(np.asarray(vel, dtype=np.float64), axis, -1)
    n = q.shape[-1]
    if n < 7:
        raise ValueError(f"open-boundary advection needs >= 7 cells along "
                         f"the open axis, got {n}")
    fx = np.zeros(np.broadcast_shapes(q.shape[:-1], vel.shape[:-1])
                  + (n + 1,))
    f = np.arange(3, n - 2)
    fx[..., f] = _flux5(q[..., f - 3], q[..., f - 2], q[..., f - 1],
                        q[..., f], q[..., f + 1], q[..., f + 2],
                        vel[..., f])
    fx[..., 1] = 0.5 * vel[..., 1] * (q[..., 1] + q[..., 0])
    fx[..., 2] = _flux3h(q[..., 0], q[..., 1], q[..., 2], q[..., 3],
                         vel[..., 2])
    fx[..., n - 2] = _flux3h(q[..., n - 4], q[..., n - 3], q[..., n - 2],
                             q[..., n - 1], vel[..., n - 2])
    fx[..., n - 1] = 0.5 * vel[..., n - 1] * (q[..., n - 1] + q[..., n - 2])
    return np.moveaxis(fx, -1, axis)


def _open_face_fluxes(q, vel, axis, spec=False):
    """WRF open-boundary fluxes along ``axis`` for a FACE-type field (the
    boundary-normal momentum): ``q`` has n+1 faces along ``axis``; ``vel``
    the advecting flux at the n cell centers between them.  Center 0 /
    n-1 are 2nd-order (WRF ``fqx(ids+1)``/``fqx(ide)``), centers 1 / n-2
    3rd-order, the rest full 5th-order.  Requires n >= 7.

    ``spec`` applies WRF's "specified uses upstream normal wind at
    boundaries" substitution in the 2nd-order fluxes (advect_u F:690-723,
    advect_v F:1978-2013): fqx(ids+1) takes ub = q(1) when q(1) < 0 and
    fqx(ide) takes ub = q(n-1) when q(n-1) > 0."""
    q = np.moveaxis(np.asarray(q, dtype=np.float64), axis, -1)
    vel = np.moveaxis(np.asarray(vel, dtype=np.float64), axis, -1)
    n = q.shape[-1] - 1
    if n < 7:
        raise ValueError(f"open-boundary advection needs >= 7 cells along "
                         f"the open axis, got {n}")
    fx = np.zeros(np.broadcast_shapes(q.shape[:-1], vel.shape[:-1]) + (n,))
    m = np.arange(2, n - 2)
    fx[..., m] = _flux5(q[..., m - 2], q[..., m - 1], q[..., m],
                        q[..., m + 1], q[..., m + 2], q[..., m + 3],
                        vel[..., m])
    qa = (np.where(q[..., 1] < 0.0, q[..., 1], q[..., 0])
          if spec else q[..., 0])
    fx[..., 0] = 0.5 * vel[..., 0] * (qa + q[..., 1])
    fx[..., 1] = _flux3h(q[..., 0], q[..., 1], q[..., 2], q[..., 3],
                         vel[..., 1])
    fx[..., n - 2] = _flux3h(q[..., n - 3], q[..., n - 2], q[..., n - 1],
                             q[..., n], vel[..., n - 2])
    qb = (np.where(q[..., n - 1] > 0.0, q[..., n - 1], q[..., n])
          if spec else q[..., n])
    fx[..., n - 1] = 0.5 * vel[..., n - 1] * (q[..., n - 1] + qb)
    return np.moveaxis(fx, -1, axis)


def _open_cell_term(q, vel, d, axis):
    """WRF's open-boundary advective term for the two boundary CELLS along
    ``axis`` (module_advect_em.F "the computations that don't require cb",
    with field_old == field per the rk_tendency call): upwind-outbound
    one-sided advection plus the cell's normal-flux divergence,

      low :  -(min(<vel>_cell0, 0)*(q1 - q0) + q0*(vel1 - vel0)) / d
      high:  -(max(<vel>_celln, 0)*(qn - qn-1) + qn*(veln+1 - veln)) / d

    Returns ``(t_low, t_high)`` with the ``axis`` dimension removed."""
    q = np.moveaxis(np.asarray(q, dtype=np.float64), axis, -1)
    vel = np.moveaxis(np.asarray(vel, dtype=np.float64), axis, -1)
    n = q.shape[-1]
    ub = np.minimum(0.5 * (vel[..., 0] + vel[..., 1]), 0.0)
    t_low = -(ub * (q[..., 1] - q[..., 0])
              + q[..., 0] * (vel[..., 1] - vel[..., 0])) / d
    ub = np.maximum(0.5 * (vel[..., n - 1] + vel[..., n]), 0.0)
    t_high = -(ub * (q[..., n - 1] - q[..., n - 2])
               + q[..., n - 1] * (vel[..., n] - vel[..., n - 1])) / d
    return t_low, t_high


def _fz_half_levels(q, vel, fnm, fnp, vorder=3):
    """Vertical fluxes at w-levels 0..nz for a half-level field ``q``.

    ``vel (nz+1, ...)`` is the advecting flux at the w-levels.  Zero flux
    at the boundary faces, WRF's stretched-grid fnm/fnp-weighted 2nd-order
    face value one face in (module_advect_em.F vert_order 3, vflux =
    rom*(fzm(k)*f(k)+fzp(k)*f(k-1)) at k=kts+1/ktf; 0.5/0.5 on a uniform
    grid), then the ``vorder`` ladder, exactly the kernels'
    ``zface_half``: 3 is 3rd-order upwind in the interior; 5 is WRF's
    ``vert_order == 5`` ladder (flux3 at faces 2 and nz-2, ``_flux5v`` at
    faces 3 .. nz-3).  ``fnm``/``fnp`` are the (nz,) weight arrays
    (``coord.fnm``/``coord.fnp``).
    """
    if vorder not in (3, 5):
        raise ValueError(f"vorder must be 3 or 5, got {vorder!r}")
    nz = q.shape[0]
    fnm = np.asarray(fnm, dtype=np.float64)
    fnp = np.asarray(fnp, dtype=np.float64)
    fz = np.zeros((nz + 1,) + q.shape[1:])
    if nz >= 2:
        fz[1] = vel[1] * (fnm[1] * q[1] + fnp[1] * q[0])
        fz[nz - 1] = vel[nz - 1] * (fnm[nz - 1] * q[nz - 1]
                                    + fnp[nz - 1] * q[nz - 2])
        fz[2:nz - 1] = _flux3(q[0:nz - 3], q[1:nz - 2], q[2:nz - 1], q[3:nz],
                              vel[2:nz - 1])
    if vorder == 5 and nz >= 6:
        fz[3:nz - 2] = _flux5v(q[0:nz - 5], q[1:nz - 4], q[2:nz - 3],
                               q[3:nz - 2], q[4:nz - 1], q[5:nz],
                               vel[3:nz - 2])
    return fz


def np_flux_div_scalar(q, ru, rv, rw, coord, dx, dy,
                       open_x=False, open_y=False, msf=None, spec=False, vorder=3):
    """Mirror of ``flux_div_scalar`` (woof/core/kernels/advection.cu).

    Returns the tendency increment ``-dFx/dx - dFy/dy - dFeta/deta`` for a
    scalar at mass points, shape ``(nz, ny, nx)``, periodic in x/y.

    ``open_x``/``open_y`` (Task 11 prerequisite) select WRF's open-boundary
    treatment along that axis, transcribed from v4.6.1 ``advect_scalar``
    (module_advect_em.F, horz_order 5): degraded near-boundary stencils
    (``_open_cell_fluxes``), no advection normal to the boundary at the two
    boundary cells, which instead take the upwind-outbound open advective
    term (``_open_cell_term``; corner cells take both axes' terms).

    ``msf`` (Task 3) is the mass-point map factor ``msft (ny, nx)``: WRF
    ``advect_scalar``'s ``mrdx/mrdy = msftx*rdx/rdy`` weighting of the
    horizontal divergence (the vertical term is unweighted, the working
    tendency is the my-divided form of tech note eqn 2.26).
    """
    nz, ny, nx = q.shape
    m = 1.0 if msf is None else np.asarray(msf, dtype=np.float64)[None]
    if open_x:
        fx = _open_cell_fluxes(q, ru, axis=2)
        tx = m * (-(fx[:, :, 1:] - fx[:, :, :-1]) / dx)
        # WRF non-cb open terms carry plain rdx (F:4119-4126), no msf;
        # under specified the boundary cells get NO x tendency at all
        # (bounds ids+1..ide-2, F:4037-4038) -- non-cb is open-only.
        if spec:
            tx[:, :, 0] = tx[:, :, -1] = 0.0
        else:
            tx[:, :, 0], tx[:, :, -1] = _open_cell_term(q, ru, dx, axis=2)
    else:
        xf = np.arange(nx + 1)            # u-face f is between cells f-1, f
        qx = lambda off: q[:, :, (xf + off) % nx]
        fx = _flux5(qx(-3), qx(-2), qx(-1), qx(0), qx(1), qx(2), ru)
        tx = m * (-(fx[:, :, 1:] - fx[:, :, :-1]) / dx)
    if open_y:
        fy = _open_cell_fluxes(q, rv, axis=1)
        ty = m * (-(fy[:, 1:, :] - fy[:, :-1, :]) / dy)
        if spec:
            ty[:, 0, :] = ty[:, -1, :] = 0.0
        else:
            ty[:, 0, :], ty[:, -1, :] = _open_cell_term(q, rv, dy, axis=1)
    else:
        yf = np.arange(ny + 1)
        qy = lambda off: q[:, (yf + off) % ny, :]
        fy = _flux5(qy(-3), qy(-2), qy(-1), qy(0), qy(1), qy(2), rv)
        ty = m * (-(fy[:, 1:, :] - fy[:, :-1, :]) / dy)
    fz = _fz_half_levels(q, rw, coord.fnm, coord.fnp, vorder)
    rdnw = np.asarray(coord.rdnw, dtype=np.float64)[:, None, None]
    return tx + ty - (fz[1:] - fz[:-1]) * rdnw


def np_flux_div_u(u, ru, rv, rw, coord, dx, dy, open_x=False, open_y=False,
                  msf=None, spec=False, vorder=3):
    """Mirror of ``flux_div_u``: tendency increment at u-points.

    ``msf`` (Task 3) is the u-point map factor ``msfu (ny, nx+1)``: WRF
    ``advect_u`` weights BOTH horizontal divergence terms by msfux at the
    u point; the vertical term is unweighted (tech note eqn 2.23).

    Returns shape ``(nz, ny, nx+1)``; under periodic x the redundant column
    nx duplicates column 0 (only u[..., :nx] is ever read).

    ``open_x``/``open_y`` transcribe WRF v4.6.1 ``advect_u``
    (module_advect_em.F, horz_order 5) for open boundaries:

    - ``open_x``: the two boundary-normal faces (0 and nx) get NO x/y
      advection (the ``MAX(ids+1,its)``/``MIN(ide-1,ite)`` loop-bound
      exclusions; the cb radiative term of ``np_open_u_radiative`` stands in
      for the x-advection there); their VERTICAL advection is retained with
      the boundary cell's Omega (the zero-gradient ghost of WRF's rom)
      unless ``open_y`` is also set, whose vertical bounds exclude them (the
      Fortran's ``open_ys/ye`` i-bounds with the ``periodic_x`` override
      commented out for open x).  x-fluxes degrade near the boundary
      (``_open_face_fluxes``).
    - ``open_y``: y-advection uses degraded stencils and covers rows
      1..ny-2 only; boundary rows 0/ny-1 instead take the non-cb open
      y-term ``-(min/max(vb,0)*du/dy + 0.5*u*(dvm+dvp))/dy``
      (module_advect_em.F:1292/1314).
    """
    nz, ny, nxp1 = u.shape
    nx = nxp1 - 1
    rdnw = np.asarray(coord.rdnw, dtype=np.float64)[:, None, None]
    if not (open_x or open_y):
        uu = u[:, :, :nx]                 # periodic degrees of freedom
        # x-faces at mass centers m = 0..nx-1, between u-points m and m+1.
        mx = np.arange(nx)
        ux = lambda off: uu[:, :, (mx + 1 + off) % nx]
        velx = 0.5 * (ru[:, :, :nx] + ru[:, :, 1:])
        fx = _flux5(ux(-3), ux(-2), ux(-1), ux(0), ux(1), ux(2), velx)
        tx = -(fx - np.roll(fx, 1, axis=2)) / dx
        # y-faces at corners (v-face row g, u-point column i).
        yf = np.arange(ny + 1)
        uy = lambda off: uu[:, (yf + off) % ny, :]
        vely = 0.5 * (np.roll(rv, 1, axis=2) + rv)
        fy = _flux5(uy(-3), uy(-2), uy(-1), uy(0), uy(1), uy(2), vely)
        ty = -(fy[:, 1:, :] - fy[:, :-1, :]) / dy
        # z-faces at corners (w-level, u-point column).
        velz = 0.5 * (np.roll(rw, 1, axis=2) + rw)
        fz = _fz_half_levels(uu, velz, coord.fnm, coord.fnp, vorder)
        tz = -(fz[1:] - fz[:-1]) * rdnw
        hor = tx + ty
        if msf is not None:
            hor = np.asarray(msf, dtype=np.float64)[None, :, :nx] * hor
        core = hor + tz
        return np.concatenate([core, core[:, :, :1]], axis=2)

    u = np.asarray(u, dtype=np.float64)
    ru = np.asarray(ru, dtype=np.float64)
    rv = np.asarray(rv, dtype=np.float64)
    rw = np.asarray(rw, dtype=np.float64)
    out = np.zeros((nz, ny, nx + 1))

    # Advected faces and the mass columns flanking each: with open x the
    # boundary faces are excluded from horizontal advection entirely.
    if open_x:
        faces = np.arange(1, nx)          # interior u faces only
        im = faces - 1                    # left mass column (in-domain)
        ip = faces                        # right mass column
    else:
        faces = np.arange(nx)             # periodic core (nx duplicated)
        im = (faces - 1) % nx
        ip = faces % nx

    mf = (np.ones((ny, nx + 1)) if msf is None
          else np.asarray(msf, dtype=np.float64))[None]

    # ---- x advection: fluxes at mass centers (msfu-weighted, WRF mrdx
    # F:740).
    velx = 0.5 * (ru[:, :, :nx] + ru[:, :, 1:])        # centers 0..nx-1
    if open_x:
        fxm = _open_face_fluxes(u, velx, axis=2, spec=spec)
        out[:, :, 1:nx] += (mf[:, :, 1:nx]
                            * (-(fxm[:, :, 1:] - fxm[:, :, :-1]) / dx))
    else:
        mx = np.arange(nx)
        ux = lambda off: u[:, :, (mx + 1 + off) % nx]
        fxm = _flux5(ux(-3), ux(-2), ux(-1), ux(0), ux(1), ux(2), velx)
        tx = -(fxm - np.roll(fxm, 1, axis=2)) / dx
        out[:, :, :nx] += mf[:, :, :nx] * tx
        out[:, :, nx] += mf[:, :, nx] * tx[:, :, 0]

    # ---- y advection on the advected faces (corner-averaged rv).
    vely = 0.5 * (rv[:, :, im] + rv[:, :, ip])         # (nz, ny+1, nfaces)
    uf = u[:, :, faces]
    if open_y:
        fy = _open_cell_fluxes(uf, vely, axis=1)
        ty = np.zeros_like(uf)
        ty[:, 1:ny - 1] = -(fy[:, 2:ny] - fy[:, 1:ny - 1]) / dy
        if not spec:
            # boundary rows: WRF's non-cb open y-term (u_old == u);
            # open-only -- specified leaves rows 0/ny-1 with no y tendency.
            vw = 0.5 * (rv[:, 0, im] + rv[:, 0, ip])
            vb = np.minimum(vw, 0.0)
            dsum = ((rv[:, 1, im] - rv[:, 0, im])
                    + (rv[:, 1, ip] - rv[:, 0, ip]))
            ty[:, 0] = -(vb * (uf[:, 1] - uf[:, 0])
                         + 0.5 * uf[:, 0] * dsum) / dy
            vw = 0.5 * (rv[:, ny, im] + rv[:, ny, ip])
            vb = np.maximum(vw, 0.0)
            dsum = ((rv[:, ny, im] - rv[:, ny - 1, im])
                    + (rv[:, ny, ip] - rv[:, ny - 1, ip]))
            ty[:, ny - 1] = -(vb * (uf[:, ny - 1] - uf[:, ny - 2])
                              + 0.5 * uf[:, ny - 1] * dsum) / dy
    else:
        yf = np.arange(ny + 1)
        uy = lambda off: uf[:, (yf + off) % ny, :]
        fy = _flux5(uy(-3), uy(-2), uy(-1), uy(0), uy(1), uy(2), vely)
        ty = -(fy[:, 1:, :] - fy[:, :-1, :]) / dy
    # WRF weights u y-divergence AND the non-cb open y-term by msfux at
    # the tendency face (mrdy F:633/1296).
    out[:, :, faces] += mf[:, :, faces] * ty
    if not open_x:
        out[:, :, nx] += mf[:, :, nx] * ty[:, :, 0]

    # ---- z advection.
    velz = 0.5 * (rw[:, :, im] + rw[:, :, ip])
    fz = _fz_half_levels(uf, velz, coord.fnm, coord.fnp, vorder)
    tz = -(fz[1:] - fz[:-1]) * rdnw
    out[:, :, faces] += tz
    if not open_x:
        out[:, :, nx] += tz[:, :, 0]
    elif not open_y:
        # boundary-normal faces keep vertical advection, with the boundary
        # cell's Omega (WRF's zero-gradient rom ghost); under open_y the
        # Fortran bounds exclude them.
        for face, cell in ((0, 0), (nx, nx - 1)):
            fzb = _fz_half_levels(u[:, :, face], rw[:, :, cell],
                                  coord.fnm, coord.fnp, vorder)
            out[:, :, face] += -(fzb[1:] - fzb[:-1]) * rdnw[:, :, 0]
    return out


def np_flux_div_v(v, ru, rv, rw, coord, dx, dy, open_x=False, open_y=False,
                  msf=None, spec=False, vorder=3):
    """Mirror of ``flux_div_v``: tendency increment at v-points.

    ``msf`` (Task 3) is the v-point map factor ``msfv (ny+1, nx)`` (WRF
    ``advect_v``'s msfvy weighting of the horizontal divergence; the
    vertical (msfvy/msfvx) ratio is identically 1 for the isotropic single
    msfv woof carries).  Periodic-dup convention: row ny must equal row 0.

    Returns shape ``(nz, ny+1, nx)``; under periodic y row ny duplicates
    row 0.  ``open_x``/``open_y`` transcribe WRF v4.6.1 ``advect_v``, the
    exact y/x mirror of :func:`np_flux_div_u` with one asymmetry kept from
    the Fortran: the boundary-normal v faces (rows 0/ny) lose their
    VERTICAL advection whenever ``open_y`` is set (``advect_v``'s
    ``open_ys/ye`` j-bounds on the vertical loop), whereas u keeps it
    unless open_y is ALSO set.
    """
    nz, nyp1, nx = v.shape
    ny = nyp1 - 1
    rdnw = np.asarray(coord.rdnw, dtype=np.float64)[:, None, None]
    if not (open_x or open_y):
        vv = v[:, :ny, :]
        # x-faces at corners (u-face column f, v-point row j).
        xf = np.arange(nx + 1)
        vx = lambda off: vv[:, :, (xf + off) % nx]
        velx = 0.5 * (np.roll(ru, 1, axis=1) + ru)
        fx = _flux5(vx(-3), vx(-2), vx(-1), vx(0), vx(1), vx(2), velx)
        tx = -(fx[:, :, 1:] - fx[:, :, :-1]) / dx
        # y-faces at mass centers m = 0..ny-1, between v-points m and m+1.
        my = np.arange(ny)
        vy = lambda off: vv[:, (my + 1 + off) % ny, :]
        vely = 0.5 * (rv[:, :ny, :] + rv[:, 1:, :])
        fy = _flux5(vy(-3), vy(-2), vy(-1), vy(0), vy(1), vy(2), vely)
        ty = -(fy - np.roll(fy, 1, axis=1)) / dy
        # z-faces at corners (w-level, v-point row).
        velz = 0.5 * (np.roll(rw, 1, axis=1) + rw)
        fz = _fz_half_levels(vv, velz, coord.fnm, coord.fnp, vorder)
        tz = -(fz[1:] - fz[:-1]) * rdnw
        hor = tx + ty
        if msf is not None:
            hor = np.asarray(msf, dtype=np.float64)[None, :ny, :] * hor
        core = hor + tz
        return np.concatenate([core, core[:, :1, :]], axis=1)

    v = np.asarray(v, dtype=np.float64)
    ru = np.asarray(ru, dtype=np.float64)
    rv = np.asarray(rv, dtype=np.float64)
    rw = np.asarray(rw, dtype=np.float64)
    out = np.zeros((nz, ny + 1, nx))

    if open_y:
        faces = np.arange(1, ny)          # interior v faces only
        jm = faces - 1                    # south mass row
        jp = faces                        # north mass row
    else:
        faces = np.arange(ny)
        jm = (faces - 1) % ny
        jp = faces % ny

    mf = (np.ones((ny + 1, nx)) if msf is None
          else np.asarray(msf, dtype=np.float64))[None]

    # ---- y advection: fluxes at mass centers (msfv-weighted, WRF mrdy
    # F:2050).
    vely = 0.5 * (rv[:, :ny, :] + rv[:, 1:, :])        # centers 0..ny-1
    if open_y:
        fym = _open_face_fluxes(v, vely, axis=1, spec=spec)
        out[:, 1:ny, :] += (mf[:, 1:ny, :]
                            * (-(fym[:, 1:, :] - fym[:, :-1, :]) / dy))
    else:
        my = np.arange(ny)
        vy = lambda off: v[:, (my + 1 + off) % ny, :]
        fym = _flux5(vy(-3), vy(-2), vy(-1), vy(0), vy(1), vy(2), vely)
        ty = -(fym - np.roll(fym, 1, axis=1)) / dy
        out[:, :ny, :] += mf[:, :ny, :] * ty
        out[:, ny, :] += mf[:, ny, :] * ty[:, 0, :]

    # ---- x advection on the advected faces (corner-averaged ru).
    velx = 0.5 * (ru[:, jm, :] + ru[:, jp, :])         # (nz, nfaces, nx+1)
    vf = v[:, faces, :]
    if open_x:
        fx = _open_cell_fluxes(vf, velx, axis=2)
        tx = np.zeros_like(vf)
        tx[:, :, 1:nx - 1] = -(fx[:, :, 2:nx] - fx[:, :, 1:nx - 1]) / dx
        if not spec:
            # WRF's non-cb open x-term for v uses the corner-averaged ru AT
            # the boundary face itself (module_advect_em.F:2763/2785 uw),
            # unlike the flank-face average of the scalar/w terms;
            # open-only -- specified leaves columns 0/nx-1 with no x
            # tendency (F:2660-2661).
            ub = np.minimum(velx[:, :, 0], 0.0)
            tx[:, :, 0] = -(ub * (vf[:, :, 1] - vf[:, :, 0])
                            + vf[:, :, 0]
                            * (velx[:, :, 1] - velx[:, :, 0])) / dx
            ub = np.maximum(velx[:, :, nx], 0.0)
            tx[:, :, nx - 1] = -(ub * (vf[:, :, nx - 1] - vf[:, :, nx - 2])
                                 + vf[:, :, nx - 1]
                                 * (velx[:, :, nx]
                                    - velx[:, :, nx - 1])) / dx
    else:
        xf = np.arange(nx + 1)
        vx = lambda off: vf[:, :, (xf + off) % nx]
        fx = _flux5(vx(-3), vx(-2), vx(-1), vx(0), vx(1), vx(2), velx)
        tx = -(fx[:, :, 1:] - fx[:, :, :-1]) / dx
    # WRF weights v x-divergence AND the non-cb open x-term by msfvy at
    # the tendency face (mrdx F:2676/2766).
    out[:, faces, :] += mf[:, faces, :] * tx
    if not open_y:
        out[:, ny, :] += mf[:, ny, :] * tx[:, 0, :]

    # ---- z advection: excluded at the boundary-normal faces under open_y.
    velz = 0.5 * (rw[:, jm, :] + rw[:, jp, :])
    fz = _fz_half_levels(vf, velz, coord.fnm, coord.fnp, vorder)
    tz = -(fz[1:] - fz[:-1]) * rdnw
    out[:, faces, :] += tz
    if not open_y:
        out[:, ny, :] += tz[:, 0, :]
    return out


def np_flux_div_w(w, ru, rv, rw, coord, dx, dy, open_x=False, open_y=False,
                  msf=None, spec=False, vorder=3):
    """Mirror of ``flux_div_w``: tendency increment at w-points.

    ``msf`` (Task 3) is the mass-point map factor ``msft (ny, nx)`` (WRF
    ``advect_w``'s msftx weighting of the horizontal divergence; the
    vertical Omega term is unweighted, tech note eqn 2.25).

    Returns shape ``(nz+1, ny, nx)``. The surface has zero tendency; the
    model top retains WRF's horizontal flux extrapolation and its vertical
    lid contribution (``advect_w``:4854-4859, 6025-6028). The vertical flux
    of w lives at mass levels; divergence uses ``coord.rdn``. A rigid-lid
    update applies its boundary condition after this tendency calculation.

    ``open_x``/``open_y`` transcribe WRF v4.6.1 ``advect_w``: the same
    open-boundary structure as :func:`np_flux_div_scalar` (degraded
    stencils, boundary cells take the non-cb open term) with the advecting
    velocities averaged to w levels; vertical advection is unmodified.
    """
    nzp1, ny, nx = w.shape
    nz = nzp1 - 1
    out = np.zeros_like(np.asarray(w, dtype=np.float64))
    if nz < 2:
        return out
    m = 1.0 if msf is None else np.asarray(msf, dtype=np.float64)[None]
    wk = w[1:nz]                          # interior w-levels 1..nz-1
    # x-faces: ru interpolated to the w level with WRF's fnm/fnp weights
    # (advect_w F:5004/:4531).
    fnm = np.asarray(coord.fnm, dtype=np.float64)[1:nz, None, None]
    fnp = np.asarray(coord.fnp, dtype=np.float64)[1:nz, None, None]
    velx = fnm * ru[1:nz] + fnp * ru[0:nz - 1]
    if open_x:
        fx = _open_cell_fluxes(wk, velx, axis=2)
        tx = m * (-(fx[:, :, 1:] - fx[:, :, :-1]) / dx)
        # WRF non-cb open terms carry plain rdx (F:5695-5806), no msf;
        # specified: no x tendency at boundary cells (F:5570-5571).
        if spec:
            tx[:, :, 0] = tx[:, :, -1] = 0.0
        else:
            tx[:, :, 0], tx[:, :, -1] = _open_cell_term(wk, velx, dx,
                                                        axis=2)
    else:
        xf = np.arange(nx + 1)
        wx = lambda off: wk[:, :, (xf + off) % nx]
        fx = _flux5(wx(-3), wx(-2), wx(-1), wx(0), wx(1), wx(2), velx)
        tx = m * (-(fx[:, :, 1:] - fx[:, :, :-1]) / dx)
    vely = fnm * rv[1:nz] + fnp * rv[0:nz - 1]
    if open_y:
        fy = _open_cell_fluxes(wk, vely, axis=1)
        ty = m * (-(fy[:, 1:, :] - fy[:, :-1, :]) / dy)
        if spec:
            ty[:, 0, :] = ty[:, -1, :] = 0.0
        else:
            ty[:, 0, :], ty[:, -1, :] = _open_cell_term(wk, vely, dy,
                                                        axis=1)
    else:
        yf = np.arange(ny + 1)
        wy = lambda off: wk[:, (yf + off) % ny, :]
        fy = _flux5(wy(-3), wy(-2), wy(-1), wy(0), wy(1), wy(2), vely)
        ty = m * (-(fy[:, 1:, :] - fy[:, :-1, :]) / dy)
    # Vertical fluxes at mass levels m = 0..nz-1, between w-points m, m+1;
    # 2nd-order centered at m = 0 and m = nz-1, 3rd-order upwind between.
    velz = 0.5 * (rw[:nz] + rw[1:])
    fzm = np.zeros((nz, ny, nx))
    fzm[0] = 0.5 * velz[0] * (w[1] + w[0])
    fzm[nz - 1] = 0.5 * velz[nz - 1] * (w[nz] + w[nz - 1])
    fzm[1:nz - 1] = _flux3(w[0:nz - 2], w[1:nz - 1], w[2:nz], w[3:nz + 1],
                           velz[1:nz - 1])
    if vorder not in (3, 5):
        raise ValueError(f"vorder must be 3 or 5, got {vorder!r}")
    if vorder == 5 and nz >= 5:
        # WRF advect_w vert_order 5: flux5 on w(k-3..k+2) for k = kts+3 ..
        # ktf-1, i.e. mass faces m = 2 .. nz-3 on w[m-2 .. m+3]; faces 1
        # and nz-2 keep flux3, 0 and nz-1 the 2nd-order form.  A column
        # with nz < 5 has no flux5 face and runs the order-3 faces.
        fzm[2:nz - 2] = _flux5v(w[0:nz - 4], w[1:nz - 3], w[2:nz - 2],
                                w[3:nz - 1], w[4:nz], w[5:nz + 1],
                                velz[2:nz - 2])
    rdn = np.asarray(coord.rdn, dtype=np.float64)[1:nz, None, None]
    tz = -(fzm[1:] - fzm[:-1]) * rdn
    out[1:nz] = tx + ty + tz
    # WRF includes k=ktf+1: extrapolate horizontal advecting fluxes from
    # the final two mass levels, then add the one-sided vertical lid term.
    wtop = w[nz:nz + 1]
    fnm_top = np.asarray(coord.fnm, dtype=np.float64)[nz - 1]
    fnp_top = np.asarray(coord.fnp, dtype=np.float64)[nz - 1]
    velx_top = ((2.0 - fnm_top) * ru[nz - 1]
                - fnp_top * ru[nz - 2])[None]
    if open_x:
        fx_top = _open_cell_fluxes(wtop, velx_top, axis=2)
        tx_top = m * (-(fx_top[:, :, 1:] - fx_top[:, :, :-1]) / dx)
        if spec:
            tx_top[:, :, 0] = tx_top[:, :, -1] = 0.0
        else:
            tx_top[:, :, 0], tx_top[:, :, -1] = _open_cell_term(
                wtop, velx_top, dx, axis=2)
    else:
        xf_top = np.arange(nx + 1)
        wx_top = lambda off: wtop[:, :, (xf_top + off) % nx]
        fx_top = _flux5(wx_top(-3), wx_top(-2), wx_top(-1), wx_top(0),
                        wx_top(1), wx_top(2), velx_top)
        tx_top = m * (-(fx_top[:, :, 1:] - fx_top[:, :, :-1]) / dx)
    vely_top = ((2.0 - fnm_top) * rv[nz - 1]
                - fnp_top * rv[nz - 2])[None]
    if open_y:
        fy_top = _open_cell_fluxes(wtop, vely_top, axis=1)
        ty_top = m * (-(fy_top[:, 1:, :] - fy_top[:, :-1, :]) / dy)
        if spec:
            ty_top[:, 0, :] = ty_top[:, -1, :] = 0.0
        else:
            ty_top[:, 0, :], ty_top[:, -1, :] = _open_cell_term(
                wtop, vely_top, dy, axis=1)
    else:
        yf_top = np.arange(ny + 1)
        wy_top = lambda off: wtop[:, (yf_top + off) % ny, :]
        fy_top = _flux5(wy_top(-3), wy_top(-2), wy_top(-1), wy_top(0),
                        wy_top(1), wy_top(2), vely_top)
        ty_top = m * (-(fy_top[:, 1:, :] - fy_top[:, :-1, :]) / dy)
    out[nz] = (tx_top + ty_top)[0] + (
        2.0 * np.asarray(coord.rdn, dtype=np.float64)[nz - 1] * fzm[nz - 1])
    return out


def _flux_upwind(qm1, q0, cr):
    """WRF ``flux_upwind`` (module_advect_em.F advect_scalar_pd): CFL-clamped
    1st-order upwind face value times the Courant number ``cr``; face f lies
    between cells f-1 (``qm1``) and f (``q0``).  The min/max clamps cap the
    per-step upwind outflow at half a cell (transcribed verbatim)."""
    return (0.5 * np.minimum(1.0, cr + np.abs(cr)) * qm1
            + 0.5 * np.maximum(-1.0, cr - np.abs(cr)) * q0)


def np_pd_fluxes(q, q0, ru, rv, rw, mut, coord, dx, dy, dt, msft=None,
                 open_x=False, open_y=False, vorder=3):
    """Mirror of ``pd_fluxes`` (woof/core/kernels/pd_advection.cu).

    ``msft`` (Task 3) is the mass-point map factor: the horizontal upwind
    fluxes and Courant numbers then use the PHYSICAL face spacing
    ``dx*2/(msft_A + msft_B)`` (WRF ``advect_scalar_pd``'s "ADT eqn 48
    d/dx"); the eta face is unweighted.

    Positive-definite flux decomposition per WRF ``advect_scalar_pd``
    (Skamarock 2006): at every face, the CFL-clamped 1st-order upwind flux
    of the *time-t* scalar ``q0`` (WRF ``field_old``) and the correction
    ``F_corr = F_high - F_upwind1``, where F_high is exactly the unlimited
    5th-order horizontal / vorder-ladder vertical flux of the RK stage estimate
    ``q`` used by ``flux_div_scalar`` (including woof's 0.5-centered
    ``flux2`` faces one cell in from the eta boundaries, where WRF's PD
    routine uses fnm/fnp weights -- identical on the uniform grid; keeping
    woof's stencil makes F_upwind1 + F_corr recombine bitwise with the
    unlimited kernel).  The upwind Courant numbers divide by the hybrid
    face mass increment ``c1h*<mu>_face + c2h`` built from the post-acoustic
    stage mass ``mut`` (WRF ``muts``); the eta-face spacing is
    ``dz = 2/(rdnw[kf] + rdnw[kf-1])`` (negative), so the Omega-signed
    upwinding follows automatically.  Boundary eta faces carry zero flux.

    ``open_x``/``open_y`` select WRF's specified/open treatment along the
    axis (module_advect_em.F advect_scalar_pd): the boundary-normal faces
    carry zero flux, the high-order flux degrades to 2nd order one face in
    and horizontal flux3 two faces in (the degrade bands), and nothing
    wraps.

    Returns ``(fxl, fxc, fyl, fyc, fzl, fzc)`` float64: low/correction flux
    pairs on x faces ``(nz,ny,nx+1)``, y faces ``(nz,ny+1,nx)``, eta faces
    ``(nz+1,ny,nx)``; periodic in x/y by default (face nx/ny duplicates
    face 0).
    """
    q = np.asarray(q, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    nz, ny, nx = q.shape
    mut = np.broadcast_to(np.asarray(mut, dtype=np.float64), (ny, nx))
    c1h = np.asarray(coord.c1h, dtype=np.float64)[:, None, None]
    c2h = np.asarray(coord.c2h, dtype=np.float64)[:, None, None]
    rdnw = np.asarray(coord.rdnw, dtype=np.float64)

    msft2 = (np.ones((ny, nx)) if msft is None
             else np.asarray(msft, dtype=np.float64))

    xf = np.arange(nx + 1)               # u-face f between cells f-1, f
    qx = lambda a, off: a[:, :, (xf + off) % nx]
    if open_x:
        fxh = _open_cell_fluxes(q, ru, axis=2)
    else:
        fxh = _flux5(qx(q, -3), qx(q, -2), qx(q, -1), qx(q, 0), qx(q, 1),
                     qx(q, 2), np.asarray(ru, dtype=np.float64))
    mux = 0.5 * (mut[:, xf % nx] + mut[:, (xf - 1) % nx])   # (ny, nx+1)
    mfx = c1h * mux[None] + c2h
    dxf = dx * 2.0 / (msft2[:, xf % nx] + msft2[:, (xf - 1) % nx])
    crx = np.asarray(ru, dtype=np.float64) * dt / dxf / mfx
    fxl = mfx * (dxf / dt) * _flux_upwind(qx(q0, -1), qx(q0, 0), crx)
    if open_x:                           # boundary-normal faces: zero flux
        fxl[:, :, 0] = fxl[:, :, nx] = 0.0
    fxc = fxh - fxl

    yf = np.arange(ny + 1)
    qy = lambda a, off: a[:, (yf + off) % ny, :]
    if open_y:
        fyh = _open_cell_fluxes(q, rv, axis=1)
    else:
        fyh = _flux5(qy(q, -3), qy(q, -2), qy(q, -1), qy(q, 0), qy(q, 1),
                     qy(q, 2), np.asarray(rv, dtype=np.float64))
    muy = 0.5 * (mut[yf % ny, :] + mut[(yf - 1) % ny, :])   # (ny+1, nx)
    mfy = c1h * muy[None] + c2h
    dyf = dy * 2.0 / (msft2[yf % ny, :] + msft2[(yf - 1) % ny, :])
    cry = np.asarray(rv, dtype=np.float64) * dt / dyf / mfy
    fyl = mfy * (dyf / dt) * _flux_upwind(qy(q0, -1), qy(q0, 0), cry)
    if open_y:
        fyl[:, 0, :] = fyl[:, ny, :] = 0.0
    fyc = fyh - fyl

    rw = np.asarray(rw, dtype=np.float64)
    fzh = _fz_half_levels(q, rw, coord.fnm, coord.fnp, vorder)
    fzl = np.zeros_like(fzh)
    if nz >= 2:
        kf = np.arange(1, nz)
        dzf = (2.0 / (rdnw[kf] + rdnw[kf - 1]))[:, None, None]   # < 0
        mfz = (c1h[kf] * mut[None] + c2h[kf])
        crz = rw[1:nz] * dt / (dzf * mfz)
        fzl[1:nz] = mfz * (dzf / dt) * _flux_upwind(q0[:nz - 1], q0[1:],
                                                    crz)
        if vorder == 5:
            # Production order 5 (kernels/pd_vertical_sl.cu): the upwind
            # flux above stands at face Courant <= 1; the fork's
            # semi-Lagrangian upstream sum replaces it above Courant 1.
            for face in range(2, nz - 1):
                vel = rw[face]
                cr = vel*dt/dzf[face-1]/mfz[face-1]
                lower, upper = (-2 if face == nz-2 else -3), (2 if face == 2 else 3)
                shift = np.clip(np.floor(np.abs(cr))*np.sign(cr), lower, upper).astype(int)
                for j, i in np.argwhere(np.abs(cr) > 1.0):
                    first = face-shift[j,i] if cr[j,i] > 1.0 else face
                    last = face-1 if cr[j,i] > 1.0 else face-shift[j,i]-1
                    fzl[face,j,i] = vel[j,i]*q0[first:last+1,j,i].sum()/abs(cr[j,i])
    fzc = fzh - fzl
    return fxl, fxc, fyl, fyc, fzl, fzc


def np_pd_renorm_apply(q0, mu_old, fxl, fxc, fyl, fyc, fzl, fzc,
                       coord, dx, dy, dt, msft=None,
                       open_x=False, open_y=False):
    """Mirror of ``pd_renorm_apply`` (pd_advection.cu): WRF's PD limiter.

    ``msft`` (Task 3): ``ph_low``/``flux_out`` weight the horizontal
    divergence by ``msft^2`` (WRF msftx*msfty) and the vertical by
    ``msft`` (msfty, the eta flux carries 1/my), and the returned
    tendency weights the horizontal divergence by ``msft`` (WRF's
    "un-canceled" msftx).

    Per cell: the upwind-updated coupled scalar ``ph_low = (c1h*mu_old +
    c2h)*q0 - dt*div(F_low)`` and the
    total outgoing correction ``flux_out``; where applying the full
    corrections would overdraw the cell (``flux_out > ph_low``), ALL
    outgoing correction fluxes of that cell are scaled by ``r =
    ph_low/(flux_out + eps)`` clamped to [0, 1] (WRF scales in place; each
    face has exactly one donor cell -- the one it drains, by flux sign --
    so the functional form here is identical).  Returns the tendency
    increment ``-div(F_upwind1 + r*F_corr)`` (nz, ny, nx) float64.

    ``open_x``/``open_y`` (WRF specified/open bounds): the limiter skips
    the outermost cells (module_advect_em.F:7697-7715 -- their scale stays
    1) and the returned tendency's horizontal divergences skip the
    boundary cells (x: ids+1..ide-2, F:7817-7821; y: jds+1..jde-2,
    F:7852-7856; the vertical covers every cell, F:7787-7791).
    """
    q0 = np.asarray(q0, dtype=np.float64)
    nz, ny, nx = q0.shape
    rdx, rdy = 1.0 / dx, 1.0 / dy
    rdnw = np.asarray(coord.rdnw, dtype=np.float64)[:, None, None]
    c1h = np.asarray(coord.c1h, dtype=np.float64)[:, None, None]
    c2h = np.asarray(coord.c2h, dtype=np.float64)[:, None, None]
    mu_old = np.broadcast_to(np.asarray(mu_old, dtype=np.float64), (ny, nx))
    fxl, fxc = np.asarray(fxl, float), np.asarray(fxc, float)
    fyl, fyc = np.asarray(fyl, float), np.asarray(fyc, float)
    fzl, fzc = np.asarray(fzl, float), np.asarray(fzc, float)

    m = (np.ones((ny, nx)) if msft is None
         else np.asarray(msft, dtype=np.float64))[None]
    m2 = m * m

    chm0 = c1h * mu_old[None] + c2h
    div_low = (m2 * (rdx * (fxl[:, :, 1:] - fxl[:, :, :-1])
                     + rdy * (fyl[:, 1:, :] - fyl[:, :-1, :]))
               + m * rdnw * (fzl[1:] - fzl[:-1]))
    ph_low = chm0 * q0 - dt * div_low
    # outgoing corrections: rdnw < 0 flips the eta-face sign selection
    # exactly as WRF's comment ("z flux is opposite sign in mass
    # coordinate") -- both eta terms are nonnegative contributions.
    flux_out = dt * (m2 * (rdx * (np.maximum(0.0, fxc[:, :, 1:])
                                  - np.minimum(0.0, fxc[:, :, :-1]))
                           + rdy * (np.maximum(0.0, fyc[:, 1:, :])
                                    - np.minimum(0.0, fyc[:, :-1, :])))
                     + m * rdnw * (np.minimum(0.0, fzc[1:])
                                   - np.maximum(0.0, fzc[:-1])))
    scale = np.where(flux_out > ph_low,
                     np.clip(ph_low / (flux_out + 1e-20), 0.0, 1.0), 1.0)
    if open_x:                          # WRF limiter bounds: boundary cells
        scale[:, :, 0] = scale[:, :, -1] = 1.0     # never renormalize
    if open_y:
        scale[:, 0, :] = scale[:, -1, :] = 1.0

    xf = np.arange(nx + 1)
    sx = np.where(fxc > 0.0, scale[:, :, (xf - 1) % nx],
                  np.where(fxc < 0.0, scale[:, :, xf % nx], 1.0))
    yf = np.arange(ny + 1)
    sy = np.where(fyc > 0.0, scale[:, (yf - 1) % ny, :],
                  np.where(fyc < 0.0, scale[:, yf % ny, :], 1.0))
    sz = np.ones_like(fzc)
    if nz >= 2:
        # eta face kf: positive (downward) drains the upper cell kf,
        # negative (upward, Omega-signed) drains the lower cell kf-1.
        sz[1:nz] = np.where(fzc[1:nz] > 0.0, scale[1:nz],
                            np.where(fzc[1:nz] < 0.0, scale[:nz - 1], 1.0))
    fx = sx * fxc + fxl
    fy = sy * fyc + fyl
    fz = sz * fzc + fzl
    tx = -m * rdx * (fx[:, :, 1:] - fx[:, :, :-1])
    ty = -m * rdy * (fy[:, 1:, :] - fy[:, :-1, :])
    if open_x:                          # WRF applied-tendency bounds
        tx[:, :, 0] = tx[:, :, -1] = 0.0
    if open_y:
        ty[:, 0, :] = ty[:, -1, :] = 0.0
    return tx + ty - rdnw * (fz[1:] - fz[:-1])


def np_kessler_column(t, qv, qc, qr, rho, pii, z, dz8w, dt, rainnc=0.0):
    """Mirror of ``kessler_column`` (woof/core/kernels/kessler.cu) for ONE
    column, float64: WRF v4.6.1 ``phys/module_mp_kessler.F`` transcribed
    line by line (0-based k; the Fortran tile kts..kte is the nz half
    levels).

    Order exactly as the Fortran: (1) terminal fall speed ``vt = 36.34 *
    (max(0, 0.001*rho*qr))^0.1364 * sqrt(rho[0]/rho)`` and the stable split
    count ``nfall = max(1, nint(0.5 + crmax/0.75))``; (2) the time-split
    sedimentation loop -- each split step records the surface precipitation
    ``ppt = rho[0]*qr[0]*vt[0]*dtfall/rhowater`` (RAINNC accumulates
    ppt*1000 mm; RAINNCV is overwritten, keeping the last split step),
    then flux-upstream fallout on the ``rdzk`` spacings (half-level z
    differences; the top level repeats the spacing below and skips the
    1/rho -- the file's quirk, kept verbatim), then recomputes vt/nfall
    from the fallen field unless this was the last split step; (3) the
    production/adjustment sweep -- autoconversion + accretion (c1 = 0.001
    1/s above the FILE's threshold c2 = 0.001 kg/kg, accretion c3 = 2.2,
    exponent c4 = 0.875, all on the PRE-sedimentation qc/qr), then the
    Teten saturation adjustment (es from svp1/svp2/svp3/svpt0; the
    hardcoded 1004./287. pressure exponent and 2.5e6/(1004.*pii) latent
    factor are the file's own literals and are kept, while f5 uses the
    passed cp = CP), rain evaporation ``ern`` (capped by the saturation
    deficit and by qr), and the coupled t/qv/qc/qr update with its clamps.

    Inputs are (nz,) float64: full potential temperature ``t`` (K), mixing
    ratios (kg/kg), dry density ``rho`` (WRF moist_physics_prep_em:
    1/(al+alb)), full-pressure Exner ``pii``, half-level heights ``z`` and
    layer depths ``dz8w`` (m); ``rainnc`` is the accumulated surface rain
    (mm) carried across calls.  Returns ``(t, qv, qc, qr, rainnc, rainncv)``
    as new arrays/floats.
    """
    c1k, c2k, c3k, c4k = 0.001, 0.001, 2.2, 0.875
    max_cr_sedimentation = 0.75
    f5 = c.SVP2 * (c.SVPT0 - c.SVP3) * c.XLV / c.CP
    nint = lambda x: int(np.floor(x + 0.5))     # Fortran NINT, x >= 0 here

    t = np.asarray(t, dtype=np.float64).copy()
    qv = np.asarray(qv, dtype=np.float64).copy()
    qc = np.asarray(qc, dtype=np.float64).copy()
    qr = np.asarray(qr, dtype=np.float64).copy()
    rho = np.asarray(rho, dtype=np.float64)
    pii = np.asarray(pii, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    dz8w = np.asarray(dz8w, dtype=np.float64)
    nz = t.shape[0]
    rainnc = float(rainnc)
    rainncv = 0.0

    # --- terminal velocity and the stable time-split count
    prodk = qr.copy()
    rhok = rho
    vtden = np.sqrt(rhok[0] / rhok)
    rdzw = 1.0 / dz8w
    vt = 36.34 * np.maximum(0.0, qr * 0.001 * rhok) ** 0.1364 * vtden
    crmax = float(np.max(vt * dt * rdzw))
    rdzk = np.empty(nz)
    rdzk[:nz - 1] = 1.0 / (z[1:] - z[:nz - 1])
    rdzk[nz - 1] = 1.0 / (z[nz - 1] - z[nz - 2])

    nfall = max(1, nint(0.5 + crmax / max_cr_sedimentation))
    dtfall = dt / nfall
    time_sediment = dt

    # --- time-split sedimentation (flux upstream); the Fortran's in-place
    # bottom-up loop reads prodk[k+1] before it is overwritten, so the
    # vectorized old-array form below is identical.
    while nfall > 0:
        time_sediment -= dtfall
        factor = dtfall * rdzk / rhok
        factor[nz - 1] = dtfall * rdzk[nz - 1]

        ppt = rhok[0] * prodk[0] * vt[0] * dtfall / c.RHOWATER
        rainncv = ppt * 1000.0
        rainnc += ppt * 1000.0

        flux = rhok * prodk * vt
        prodk[:nz - 1] = prodk[:nz - 1] - factor[:nz - 1] * (flux[:nz - 1]
                                                             - flux[1:])
        prodk[nz - 1] = prodk[nz - 1] - factor[nz - 1] * prodk[nz - 1] \
            * vt[nz - 1]

        if nfall > 1:                            # not the last split step
            nfall -= 1
            vt = 36.34 * np.maximum(0.0, prodk * 0.001 * rhok) ** 0.1364 \
                * vtden
            crmax = float(np.max(vt * time_sediment * rdzw))
            nfall_new = max(1, nint(0.5 + crmax / max_cr_sedimentation))
            if nfall_new != nfall:
                nfall = nfall_new
                dtfall = time_sediment / nfall
        else:
            nfall = 0                            # prodk is the fallen qr

    # --- production of rain from qc, saturation adjustment, evaporation
    factorn = 1.0 / (1.0 + c3k * dt * np.maximum(0.0, qr) ** c4k)
    qrprod = qc * (1.0 - factorn) + factorn * c1k * dt \
        * np.maximum(qc - c2k, 0.0)
    rcgs = 0.001 * rho

    qc = np.maximum(qc - qrprod, 0.0)
    qr = (qr + prodk) - qr                       # the file's own expression
    qr = np.maximum(qr + qrprod, 0.0)

    temp = pii * t
    pressure = 1.0e5 * pii ** (1004.0 / 287.0)
    gam = 2.5e6 / (1004.0 * pii)
    es = 1000.0 * c.SVP1 * np.exp(c.SVP2 * (temp - c.SVPT0)
                                  / (temp - c.SVP3))
    es = np.minimum(es, 0.99 * pressure)         # WOOF's es >= p guard
    qvs = c.EP2 * es / (pressure - es)
    prod = (qv - qvs) / (1.0 + pressure / (pressure - es) * qvs * f5
                         / (temp - c.SVP3) ** 2)
    ern = np.minimum(np.minimum(
        dt * (((1.6 + 124.9 * (rcgs * qr) ** 0.2046)
               * (rcgs * qr) ** 0.525)
              / (2.55e8 / (pressure * qvs) + 5.4e5))
        * (np.maximum(qvs - qv, 0.0) / (rcgs * qvs)),
        np.maximum(-prod - qc, 0.0)), qr)

    product = np.maximum(prod, -qc)
    t = t + gam * (product - ern)
    qv = np.maximum(qv - product + ern, 0.0)
    qc = qc + product
    qr = qr - ern
    return t, qv, qc, qr, rainnc, rainncv


# ---------------------------------------------------------------------------
# Morrison two-moment column microphysics
# ---------------------------------------------------------------------------

_MORR_PI = np.pi
_MORR_RHOW = 997.0
_MORR_RHOI = 500.0
_MORR_RHOS = 100.0
_MORR_RHOSU = 85000.0 / (287.15 * 273.15)
_MORR_QSMALL = 1.0e-14
_MORR_DCS = 125.0e-6
_MORR_MI0 = 4.0 / 3.0 * _MORR_PI * _MORR_RHOI * (10.0e-6) ** 3
_MORR_MG0 = 1.6e-10
_MORR_CI = _MORR_RHOI * _MORR_PI / 6.0
_MORR_CS = _MORR_RHOS * _MORR_PI / 6.0


def _np_morrison_polysvp(t, ice=False):
    """Flatau/Goff-Gratch saturation pressure used by Morrison.

    Transcribed from WRF v4.6.1 ``module_mp_morr_two_moment.F:4066-4149``.
    The model regimes are above the low-temperature Goff-Gratch switch,
    but that branch is retained so the mirror defines the complete helper.
    """
    t = np.asarray(t, dtype=np.float64)
    x = t - 273.15
    if ice:
        coeff = (6.11147274, 0.503160820, 0.188439774e-1,
                 0.420895665e-3, 0.615021634e-5, 0.602588177e-7,
                 0.385852041e-9, 0.146898966e-11, 0.252751365e-14)
        poly = np.full_like(t, coeff[-1])
        for value in coeff[-2::-1]:
            poly = value + x * poly
        ratio = 273.16 / t
        goff = 10.0 ** (-9.09718 * (ratio - 1.0)
                        - 3.56654 * np.log10(ratio)
                        + 0.876793 * (1.0 - 1.0 / ratio)
                        + np.log10(6.1071)) * 100.0
        return np.where(t >= 195.8, poly * 100.0, goff)
    coeff = (6.11239921, 0.443987641, 0.142986287e-1,
             0.264847430e-3, 0.302950461e-5, 0.206739458e-7,
             0.640689451e-10, -0.952447341e-13, -0.976195544e-15)
    poly = np.full_like(t, coeff[-1])
    for value in coeff[-2::-1]:
        poly = value + x * poly
    ratio = 373.16 / t
    goff = 10.0 ** (-7.90298 * (ratio - 1.0)
                    + 5.02808 * np.log10(ratio)
                    - 1.3816e-7
                    * (10.0 ** (11.344 * (1.0 - 1.0 / ratio)) - 1.0)
                    + 8.1328e-3
                    * (10.0 ** (-3.49149 * (ratio - 1.0)) - 1.0)
                    + np.log10(1013.246)) * 100.0
    return np.where(t >= 202.0, poly * 100.0, goff)


def _np_morrison_slopes(q, n, rho, temperature, *, pressure=None,
                        reset_cloud_number=True, morr_rimed_ice=1):
    """Return bounded lambda/number fields for the five species.

    This is the repeated PSD limiter at WRF source lines 1525-1638,
    2122-2267, and 3855-4000.  Number concentrations are per kg dry air.
    """
    q = {name: np.asarray(value, np.float64) for name, value in q.items()}
    n = {name: np.maximum(np.asarray(value, np.float64), 0.0)
         for name, value in n.items()}
    nz = np.asarray(temperature).size
    lam = {name: np.zeros(nz, np.float64)
           for name in ("c", "r", "i", "s", "g")}
    pgam = np.full(nz, 2.0, np.float64)

    # INUM=1 is imposed once on entry to the warm/cold branch.  A LAMC
    # rebound then alters NC3D at fixed QC, and that altered concentration
    # feeds PRC/MNUCCC/PGAM for the rest of the process section
    # (WRF 1493-1586, 2108-2215).  Callers doing sedimentation/final bounds
    # must not silently restore 250 cm-3 here.
    if reset_cloud_number:
        n["c"] = 250.0e6 / rho
    active = q["c"] >= _MORR_QSMALL
    if np.any(active):
        dens = (rho[active] if pressure is None else
                np.asarray(pressure, np.float64)[active]
                / (287.15 * np.asarray(temperature, np.float64)[active]))
        pg = 0.0005714 * (n["c"][active] / 1.0e6 * dens) + 0.2714
        pgam[active] = np.clip(1.0 / (pg * pg) - 1.0, 2.0, 10.0)
        raw = (_MORR_PI / 6.0 * _MORR_RHOW * n["c"][active]
               * np.vectorize(math.gamma)(pgam[active] + 4.0)
               / (q["c"][active]
                  * np.vectorize(math.gamma)(pgam[active] + 1.0))) \
            ** (1.0 / 3.0)
        lo = (pgam[active] + 1.0) / 60.0e-6
        hi = (pgam[active] + 1.0) / 1.0e-6
        bounded = np.clip(raw, lo, hi)
        lam["c"][active] = bounded
        rebounded = raw != bounded
        if np.any(rebounded):
            aa = np.flatnonzero(active)[rebounded]
            n["c"][aa] = (bounded[rebounded] ** 3 * q["c"][aa]
                            * np.vectorize(math.gamma)(pgam[aa] + 1.0)
                            / (np.vectorize(math.gamma)(pgam[aa] + 4.0)
                               * (_MORR_PI / 6.0 * _MORR_RHOW)))

    rimed = rimed_ice_constants(morr_rimed_ice)
    specs = {
        "r": (_MORR_PI * _MORR_RHOW, 1.0 / 2800.0e-6,
              1.0 / 20.0e-6),
        "i": (6.0 * _MORR_CI, 1.0 / (2.0 * _MORR_DCS + 100.0e-6),
              1.0 / 1.0e-6),
        "s": (6.0 * _MORR_CS, 1.0 / 2000.0e-6, 1.0 / 10.0e-6),
        "g": (6.0 * rimed.cg, 1.0 / 2000.0e-6, 1.0 / 20.0e-6),
    }
    for name, (six_c, lo, hi) in specs.items():
        active = q[name] >= _MORR_QSMALL
        raw = np.zeros(nz)
        raw[active] = np.cbrt(six_c * n[name][active] / q[name][active])
        lam[name][active] = np.clip(raw[active], lo, hi)
        # q = Gamma(4)*C*N/lambda^3 = 6*C*N/lambda^3.
        n[name][active] = q[name][active] * lam[name][active] ** 3 / six_c
        n[name][~active] = 0.0
    return lam, pgam, n


def _np_morrison_fall_speeds(kind, q, n, rho, temperature, *,
                             morr_rimed_ice=1):
    """Mass/number-weighted fall speeds (source 3376-3503)."""
    # Sedimentation clamps only the local DLAM* slopes.  It does not rebound
    # and persist the provisional number moments (WRF 3376-3432).
    lam, pgam, n = _np_morrison_slopes(
        q, n, rho, temperature, reset_cloud_number=False,
        morr_rimed_ice=morr_rimed_ice)
    mu = 1.496e-6 * temperature ** 1.5 / (temperature + 120.0)
    dens54 = (_MORR_RHOSU / rho) ** 0.54
    if kind == "c":
        acn = c.G * _MORR_RHOW / (18.0 * mu)
        safe_lam = np.where(q["c"] >= _MORR_QSMALL, lam["c"], 1.0)
        vm = acn * np.vectorize(math.gamma)(4.0 + 2.0 + pgam) \
            / (safe_lam ** 2.0
               * np.vectorize(math.gamma)(pgam + 4.0))
        vn = acn * np.vectorize(math.gamma)(1.0 + 2.0 + pgam) \
            / (safe_lam ** 2.0
               * np.vectorize(math.gamma)(pgam + 1.0))
        active = q["c"] >= _MORR_QSMALL
        return np.where(active, vm, 0.0), np.where(active, vn, 0.0), n
    rimed = rimed_ice_constants(morr_rimed_ice)
    params = {
        "r": (841.99667, 0.8, 9.1, 0.54),
        "i": (700.0, 1.0, 1.2, 0.35),
        "s": (11.72, 0.41, 1.2, 0.54),
        "g": (rimed.ag, rimed.bg, 20.0, 0.54),
    }
    aa, bb, cap, exponent = params[kind]
    an = aa * (_MORR_RHOSU / rho) ** exponent
    safe_lam = np.where(q[kind] >= _MORR_QSMALL, lam[kind], 1.0)
    vm = an * math.gamma(4.0 + bb) / 6.0 / safe_lam ** bb
    vn = an * math.gamma(1.0 + bb) / safe_lam ** bb
    active = q[kind] >= _MORR_QSMALL
    density_cap = cap * ((_MORR_RHOSU / rho) ** 0.54
                         if kind != "i"
                         else (_MORR_RHOSU / rho) ** 0.35)
    return (np.where(active, np.minimum(vm, density_cap), 0.0),
            np.where(active, np.minimum(vn, density_cap), 0.0), n)


def _np_morrison_advect_sedimentation(qdens, ndens, velocities, dz, dt):
    """Advect every Morrison moment with WRF's shared sedimentation clock.

    This is the flux-form block at WRF v4.6.1
    ``module_mp_morr_two_moment.F:3505-3667``.  Empty levels below
    precipitation inherit the fall speed from the level above, and the
    largest mass/number Courant number across *all* categories selects one
    ``NSTEP`` used by every category.  Inputs and outputs are densities
    (mass or number per cubic metre); vertical storage is bottom to top.
    """
    qwork = {name: np.asarray(value, np.float64).copy()
             for name, value in qdens.items()}
    nwork = {name: np.asarray(value, np.float64).copy()
             for name, value in ndens.items()}
    dz = np.asarray(dz, np.float64)
    nz = dz.size
    if set(qwork) != set(nwork) or set(qwork) != set(velocities):
        raise ValueError("sedimentation mass, number, and speed keys differ")
    if any(value.shape != (nz,) for value in (*qwork.values(),
                                               *nwork.values())):
        raise ValueError("sedimentation fields must match dz")

    speeds = {}
    max_courant = 0.0
    for name, (vm_in, vn_in) in velocities.items():
        vm = np.asarray(vm_in, np.float64).copy()
        vn = np.asarray(vn_in, np.float64).copy()
        if vm.shape != (nz,) or vn.shape != (nz,):
            raise ValueError("sedimentation speeds must match dz")
        for k in range(nz - 2, -1, -1):
            if vm[k] < 1.0e-10:
                vm[k] = vm[k + 1]
            if vn[k] < 1.0e-10:
                vn[k] = vn[k + 1]
        speeds[name] = (vm, vn)
        max_courant = max(max_courant,
                           float(np.max(np.maximum(vm, vn) * dt / dz)))

    # Fortran: INT(RGVM*DT/DZQ(K)+1.), accumulated column-wide.
    nstep = max(int(max_courant + 1.0), 1)
    dts = dt / nstep
    exported = {name: 0.0 for name in qwork}
    top = nz - 1
    for _ in range(nstep):
        for name in qwork:
            vm, vn = speeds[name]
            fm = vm * qwork[name]
            fn = vn * nwork[name]
            exported[name] += fm[0] * dts
            qwork[name][top] -= fm[top] * dts / dz[top]
            nwork[name][top] -= fn[top] * dts / dz[top]
            for k in range(top - 1, -1, -1):
                qwork[name][k] += (fm[k + 1] - fm[k]) * dts / dz[k]
                nwork[name][k] += (fn[k + 1] - fn[k]) * dts / dz[k]
    return qwork, nwork, exported, nstep


def _np_morrison_final_phase_cleanup(q, n, temperature, rho,
                                     ice_to_snow, pressure=None,
                                     xlv_stale=None, cpm_stale=None):
    """Apply WRF's post-sedimentation ice/snow and phase cleanup.

    The ice-size conversion is from source lines 3679-3689; instantaneous
    melting and homogeneous freezing are lines 3799-3849.  Keeping this
    block after fallout is important because each category sediments with
    its own WRF terminal velocity before the local phase transfer.
    """
    q = {name: np.asarray(value, np.float64).copy()
         for name, value in q.items()}
    n = {name: np.asarray(value, np.float64).copy()
         for name, value in n.items()}
    t = np.asarray(temperature, np.float64).copy()
    rho = np.asarray(rho, np.float64)
    pressure = (rho * c.RD * t if pressure is None else
                np.asarray(pressure, np.float64))
    convert = np.asarray(ice_to_snow, bool)
    nz = t.size
    if (rho.shape != (nz,) or convert.shape != (nz,)):
        raise ValueError("Morrison phase-cleanup fields must share one column")
    xlv_stale = (3.1484e6 - 2370.0 * t if xlv_stale is None else
                 np.asarray(xlv_stale, np.float64))
    cpm_stale = (c.CP * (1.0 + 0.887 * q["qv"]) if cpm_stale is None else
                 np.asarray(cpm_stale, np.float64))
    if xlv_stale.shape != (nz,) or cpm_stale.shape != (nz,):
        raise ValueError("Morrison stale thermodynamics must share one column")
    for k in range(nz):
        xlv = xlv_stale[k]
        xls = xlv + 0.3353e6
        xlf = xls - xlv
        cpm = cpm_stale[k]

        # ``convert`` already includes WRF's pre-update T3D test.  Source
        # 3679-3689 applies the latched tendency without a second T check.
        if convert[k] and q["qi"][k] > 0.0:
            q["qs"][k] += q["qi"][k]
            n["ns"][k] += n["ni"][k]
            q["qi"][k] = 0.0
            n["ni"][k] = 0.0

        # The second low-RH tiny-category cleanup occurs after fallout and
        # the ice-to-snow tendency application (WRF 3729-3761).
        ew = min(0.99 * pressure[k],
                 float(_np_morrison_polysvp(t[k], False)))
        ei = min(ew, 0.99 * pressure[k],
                 float(_np_morrison_polysvp(t[k], True)))
        qvs = c.EP2 * ew / (pressure[k] - ew)
        qvi = c.EP2 * ei / (pressure[k] - ei)
        if q["qv"][k] / qvs < 0.9:
            for mass in ("qr", "qc"):
                if q[mass][k] < 1.0e-8:
                    amount = q[mass][k]
                    q["qv"][k] += amount
                    q[mass][k] = 0.0
                    t[k] -= amount * xlv / cpm
        if q["qv"][k] / qvi < 0.9:
            for mass in ("qi", "qs", "qg"):
                if q[mass][k] < 1.0e-8:
                    amount = q[mass][k]
                    q["qv"][k] += amount
                    q[mass][k] = 0.0
                    t[k] -= amount * xls / cpm

        for mass, moment in (("qc", "nc"), ("qr", "nr"),
                             ("qi", "ni"), ("qs", "ns"),
                             ("qg", "ng")):
            if q[mass][k] < _MORR_QSMALL:
                q[mass][k] = 0.0
                n[moment][k] = 0.0

        if q["qi"][k] >= _MORR_QSMALL and t[k] >= 273.15:
            amount = q["qi"][k]
            q["qi"][k] = 0.0
            q["qr"][k] += amount
            n["nr"][k] += n["ni"][k]
            n["ni"][k] = 0.0
            t[k] -= amount * xlf / cpm
        if t[k] <= 233.15 and q["qc"][k] >= _MORR_QSMALL:
            amount = q["qc"][k]
            q["qc"][k] = 0.0
            q["qi"][k] += amount
            n["ni"][k] += n["nc"][k]
            n["nc"][k] = 0.0
            t[k] += amount * xlf / cpm
        if t[k] <= 233.15 and q["qr"][k] >= _MORR_QSMALL:
            amount = q["qr"][k]
            q["qr"][k] = 0.0
            q["qg"][k] += amount
            n["ng"][k] += n["nr"][k]
            n["nr"][k] = 0.0
            t[k] += amount * xlf / cpm
    return q, n, t


def _np_morrison_level_moments(q, n, rhoa, temperature, pressure,
                               *, reset_cloud=True, morr_rimed_ice=1):
    """Scalar PSD reconstruction used by the WRF process section.

    The cloud-number rebound is stateful inside a process section: INUM=1
    is imposed once, then a 1--60 um LAMC clamp alters NC3D at fixed QC
    (WRF v4.6.1 lines 1493-1586 and 2108-2215).
    """
    nn = {name: max(float(value), 0.0) for name, value in n.items()}
    if reset_cloud:
        nn["nc"] = 250.0e6 / rhoa
    lam = {name: 0.0 for name in ("c", "r", "i", "s", "g")}
    pgam = 2.0
    if q["qc"] >= _MORR_QSMALL:
        rho_cloud = pressure / (287.15 * temperature)
        ac = 0.0005714 * (nn["nc"] / 1.0e6 * rho_cloud) + 0.2714
        pgam = min(max(1.0 / ac ** 2 - 1.0, 2.0), 10.0)
        raw = (_MORR_PI / 6.0 * _MORR_RHOW * nn["nc"]
               * math.gamma(pgam + 4.0)
               / (q["qc"] * math.gamma(pgam + 1.0))) ** (1.0 / 3.0)
        lo = (pgam + 1.0) / 60.0e-6
        hi = (pgam + 1.0) / 1.0e-6
        lam["c"] = min(max(raw, lo), hi)
        if lam["c"] != raw:
            nn["nc"] = (lam["c"] ** 3 * q["qc"]
                         * math.gamma(pgam + 1.0)
                         / ((_MORR_PI / 6.0 * _MORR_RHOW)
                            * math.gamma(pgam + 4.0)))
    rimed = rimed_ice_constants(morr_rimed_ice)
    specs = {
        "r": ("qr", "nr", _MORR_PI * _MORR_RHOW,
              1.0 / 2800.0e-6, 1.0 / 20.0e-6),
        "i": ("qi", "ni", 6.0 * _MORR_CI,
              1.0 / (2.0 * _MORR_DCS + 100.0e-6), 1.0 / 1.0e-6),
        "s": ("qs", "ns", 6.0 * _MORR_CS,
              1.0 / 2000.0e-6, 1.0 / 10.0e-6),
        "g": ("qg", "ng", 6.0 * rimed.cg,
              1.0 / 2000.0e-6, 1.0 / 20.0e-6),
    }
    for kind, (mass, number, six_c, lo, hi) in specs.items():
        if q[mass] >= _MORR_QSMALL:
            raw = np.cbrt(six_c * nn[number] / q[mass])
            lam[kind] = min(max(raw, lo), hi)
            if lam[kind] != raw:
                nn[number] = q[mass] * lam[kind] ** 3 / six_c
        else:
            nn[number] = 0.0
    return lam, pgam, nn


def _np_morrison_apply_level(q, n, temperature, pressure, rhoa, dt,
                             qvs, qvi, xlv, xls, cpm, warm, *,
                             morr_rimed_ice=1):
    """WRF rate-vector skeleton for one Morrison level.

    Rates are diagnosed from one begin-of-process snapshot, then mutated in
    WRF order, mass-only donor limited, and collectively applied.  Number
    rates are intentionally never multiplied by mass donor ratios
    (WRF v4.6.1 lines 1643-2044 and 2270-3267).
    """
    xlf = xls - xlv

    # WRF 1424-1479 evaluates these before the warm branch's small-particle
    # melt at 1498-1514.  They deliberately retain the pre-melt temperature.
    mu = 1.496e-6 * temperature ** 1.5 / (temperature + 120.0)
    dv = 8.794e-5 * temperature ** 1.81 / pressure
    sc = mu / (rhoa * dv)
    kap = 1.414e3 * mu
    ab = 1.0 + xlv * qvs / (c.RV * temperature ** 2) * xlv / cpm
    abi = 1.0 + xls * qvi / (c.RV * temperature ** 2) * xls / cpm

    # WRF performs these tiny warm-phase melts before reconstructing PSDs,
    # but the warm/cold decision has already been latched by the caller.
    if warm:
        for mass, moment in (("qs", "ns"), ("qg", "ng")):
            if q[mass] < 1.0e-6:
                amount = q[mass]
                q["qr"] += amount
                n["nr"] += n[moment]
                q[mass] = 0.0
                n[moment] = 0.0
                temperature -= amount * xlf / cpm

    rimed = rimed_ice_constants(morr_rimed_ice)
    lam, pgam, n = _np_morrison_level_moments(
        q, n, rhoa, temperature, pressure, reset_cloud=True,
        morr_rimed_ice=morr_rimed_ice)
    stale_lami = 0.0 if warm else lam["i"]
    cloud_nc_for_sedimentation = n["nc"]
    r = {name: 0.0 for name in (
        "prc nprc nprc1 pra npra nragg pre pracs npracs pracg npracg "
        "psmlt evpms pgmlt evpmg nsmlts nsmltr ngmltg ngmltr nsubr "
        "mnuccc nnuccc nsagg psacws npsacws psacwi npsacwi psacwg "
        "npsacwg qmults nmults qmultr nmultr qmultg nmultg qmultrg "
        "nmultrg pgsacw pgracs psacr nscng ngracs mnuccr nnuccr "
        "prci nprci prai nprai nnuccd mnuccd prd prds prdg eprd "
        "eprds eprdg piacr niacr praci piacrs niacrs pracis nsubi "
        "nsubs nsubg").split()}

    dens54 = (_MORR_RHOSU / rhoa) ** 0.54
    ain = 700.0 * (_MORR_RHOSU / rhoa) ** 0.35
    arn, asn, agn = (841.99667 * dens54, 11.72 * dens54,
                     rimed.ag * dens54)
    n0r, n0i = n["nr"] * lam["r"], n["ni"] * lam["i"]
    n0s, n0g = n["ns"] * lam["s"], n["ng"] * lam["g"]

    # Rates shared by both phase branches (WRF 1671-1842/2407-2425,
    # 2795-2821, and in-section rain evaporation).
    if q["qc"] >= 1.0e-6:
        r["prc"] = (1350.0 * q["qc"] ** 2.47
                    * (n["nc"] / 1.0e6 * rhoa) ** -1.79)
        r["nprc1"] = r["prc"] / (4.0 / 3.0 * _MORR_PI * _MORR_RHOW
                                    * (25.0e-6) ** 3)
        r["nprc"] = min(r["prc"] / (q["qc"] / n["nc"]), n["nc"] / dt)
        r["nprc1"] = min(r["nprc1"], r["nprc"])
    if q["qr"] >= 1.0e-8 and q["qc"] >= 1.0e-8:
        r["pra"] = 67.0 * (q["qc"] * q["qr"]) ** 1.15
        r["npra"] = r["pra"] / (q["qc"] / n["nc"])
    if q["qr"] >= 1.0e-8:
        fb = (1.0 if 1.0 / lam["r"] < 300.0e-6 else
              2.0 - np.exp(2300.0 * (1.0 / lam["r"] - 300.0e-6)))
        r["nragg"] = -5.78 * fb * n["nr"] * q["qr"] * rhoa
    if q["qr"] >= _MORR_QSMALL:
        epsr = (2.0 * _MORR_PI * n0r * rhoa * dv
                * (0.78 / lam["r"] ** 2
                   + 0.308 * np.sqrt(arn * rhoa / mu)
                   * sc ** (1.0 / 3.0) * math.gamma(2.9)
                   / lam["r"] ** 2.9))
        if q["qv"] < qvs:
            r["pre"] = min(epsr * (q["qv"] - qvs) / ab, 0.0)

    if warm:
        # Above-freezing rain/frozen collisions are evaluated only to drive
        # accelerated melt and shed-drop number, then their mass rates reset.
        for mass, moment, kind, an, bb in (
                ("qs", "ns", "s", asn, 0.41),
                ("qg", "ng", "g", agn, rimed.bg)):
            if q["qr"] >= 1.0e-8 and q[mass] >= 1.0e-8:
                umx = min(an * math.gamma(4.0 + bb) / 6.0
                          / lam[kind] ** bb,
                          (1.2 if kind == "s" else 20.0) * dens54)
                unx = min(an * math.gamma(1.0 + bb) / lam[kind] ** bb,
                          (1.2 if kind == "s" else 20.0) * dens54)
                umr = min(arn * math.gamma(4.8) / 6.0 / lam["r"] ** 0.8,
                          9.1 * dens54)
                unr = min(arn * math.gamma(1.8) / lam["r"] ** 0.8,
                          9.1 * dens54)
                mass_rate = (_MORR_PI ** 2 * _MORR_RHOW
                             * np.sqrt((1.2 * umr - 0.95 * umx) ** 2
                                       + 0.08 * umx * umr)
                             * rhoa * n0r * (n[moment] * lam[kind])
                             / lam["r"] ** 3
                             * (5.0 / (lam["r"] ** 3 * lam[kind])
                                + 2.0 / (lam["r"] ** 2 * lam[kind] ** 2)
                                + 0.5 / (lam["r"] * lam[kind] ** 3)))
                if kind == "s":
                    r["pracs"] = mass_rate
                else:
                    r["pracg"] = mass_rate
                    collected_n = (_MORR_PI / 2.0 * rhoa
                                   * np.sqrt(1.7 * (unr - unx) ** 2
                                             + 0.3 * unr * unx)
                                   * n0r * (n[moment] * lam[kind])
                                   * (1.0 / (lam["r"] ** 3 * lam[kind])
                                      + 1.0 / (lam["r"] ** 2
                                               * lam[kind] ** 2)
                                      + 1.0 / (lam["r"]
                                               * lam[kind] ** 3)))
                    r["npracg"] = collected_n - mass_rate / 5.2e-7
        if q["qs"] >= 1.0e-8:
            accel = -4187.0 / xlf * (temperature - 273.15) * r["pracs"]
            r["psmlt"] = (2.0 * _MORR_PI * n0s * kap
                           * (273.15 - temperature) / xlf
                           * (0.86 / lam["s"] ** 2
                              + 0.28 * np.sqrt(asn * rhoa / mu)
                              * sc ** (1.0 / 3.0) * math.gamma(2.705)
                              / lam["s"] ** 2.705) + accel)
            if q["qv"] / qvs < 1.0:
                epss = (2.0 * _MORR_PI * n0s * rhoa * dv
                        * (0.86 / lam["s"] ** 2
                           + 0.28 * np.sqrt(asn * rhoa / mu)
                           * sc ** (1.0 / 3.0) * math.gamma(2.705)
                           / lam["s"] ** 2.705))
                r["evpms"] = max((q["qv"] - qvs) * epss / ab, r["psmlt"])
                r["psmlt"] -= r["evpms"]
        if q["qg"] >= 1.0e-8:
            accel = -4187.0 / xlf * (temperature - 273.15) * r["pracg"]
            r["pgmlt"] = (2.0 * _MORR_PI * n0g * kap
                           * (273.15 - temperature) / xlf
                           * (0.86 / lam["g"] ** 2
                              + 0.28 * np.sqrt(agn * rhoa / mu)
                              * sc ** (1.0 / 3.0)
                              * math.gamma(2.5 + rimed.bg / 2.0)
                              / lam["g"] ** (2.5 + rimed.bg / 2.0))
                           + accel)
            if q["qv"] / qvs < 1.0:
                epsg = (2.0 * _MORR_PI * n0g * rhoa * dv
                        * (0.86 / lam["g"] ** 2
                           + 0.28 * np.sqrt(agn * rhoa / mu)
                           * sc ** (1.0 / 3.0)
                           * math.gamma(2.5 + rimed.bg / 2.0)
                           / lam["g"] ** (2.5 + rimed.bg / 2.0)))
                r["evpmg"] = max((q["qv"] - qvs) * epsg / ab, r["pgmlt"])
                r["pgmlt"] -= r["evpmg"]

        # WRF 1924-1930: collision mass rates do not directly transfer mass.
        r["pracg"] = 0.0
        r["pracs"] = 0.0
        loss = (r["prc"] + r["pra"]) * dt
        if loss > q["qc"] and q["qc"] >= _MORR_QSMALL:
            ratio = q["qc"] / loss
            r["prc"] *= ratio
            r["pra"] *= ratio
        loss = (-r["psmlt"] - r["evpms"] + r["pracs"]) * dt
        if loss > q["qs"] and q["qs"] >= _MORR_QSMALL:
            ratio = q["qs"] / loss
            for name in ("psmlt", "evpms", "pracs"):
                r[name] *= ratio
        loss = (-r["pgmlt"] - r["evpmg"] + r["pracg"]) * dt
        if loss > q["qg"] and q["qg"] >= _MORR_QSMALL:
            ratio = q["qg"] / loss
            for name in ("pgmlt", "evpmg", "pracg"):
                r[name] *= ratio
        loss = (-r["pracs"] - r["pracg"] - r["pre"] - r["pra"]
                - r["prc"] + r["psmlt"] + r["pgmlt"]) * dt
        if loss > q["qr"] and q["qr"] >= _MORR_QSMALL:
            ratio = ((q["qr"] / dt + r["pracs"] + r["pracg"]
                      + r["pra"] + r["prc"] - r["psmlt"]
                      - r["pgmlt"]) / (-r["pre"]))
            r["pre"] *= ratio

        tqv = -r["pre"] - r["evpms"] - r["evpmg"]
        tt = (r["pre"] * xlv + (r["evpms"] + r["evpmg"]) * xls
              + (r["psmlt"] + r["pgmlt"] - r["pracs"]
                 - r["pracg"]) * xlf) / cpm
        tqc = -r["pra"] - r["prc"]
        tqr = (r["pre"] + r["pra"] + r["prc"] - r["psmlt"]
               - r["pgmlt"] + r["pracs"] + r["pracg"])
        tqi = 0.0
        tqs = r["psmlt"] + r["evpms"] - r["pracs"]
        tqg = r["pgmlt"] + r["evpmg"] - r["pracg"]
        if r["pre"] < 0.0:
            r["nsubr"] = max(-1.0, r["pre"] * dt / q["qr"]) * n["nr"] / dt
        if r["evpms"] + r["psmlt"] < 0.0:
            r["nsmlts"] = (max(-1.0, (r["evpms"] + r["psmlt"])
                                  * dt / q["qs"]) * n["ns"] / dt)
        if r["psmlt"] < 0.0:
            r["nsmltr"] = max(-1.0, r["psmlt"] * dt / q["qs"]) * n["ns"] / dt
        if r["evpmg"] + r["pgmlt"] < 0.0:
            r["ngmltg"] = (max(-1.0, (r["evpmg"] + r["pgmlt"])
                                  * dt / q["qg"]) * n["ng"] / dt)
        if r["pgmlt"] < 0.0:
            r["ngmltr"] = max(-1.0, r["pgmlt"] * dt / q["qg"]) * n["ng"] / dt
        tnc = -r["npra"] - r["nprc"]
        tnr = (r["nprc1"] + r["nragg"] - r["npracg"] + r["nsubr"]
               - r["nsmltr"] - r["ngmltr"])
        tni = 0.0
        tns = r["nsmlts"]
        tng = r["ngmltg"]
    else:
        # Cold rate evaluation: every expression below reads q/n/lam from
        # the one snapshot above.  No prognostic state is changed here.
        if q["qc"] >= _MORR_QSMALL and temperature < 269.15:
            nacnt = np.exp(-2.80 + 0.262 * (273.15 - temperature)) * 1000.0
            slip = 7.37 * temperature / (288.0 * 10.0 * pressure) / 100.0
            rin = 0.1e-6
            dap = (4.0 * _MORR_PI * 1.38e-23 / (6.0 * _MORR_PI * rin)
                   * temperature * (1.0 + slip / rin) / mu)
            cdist = n["nc"] / math.gamma(pgam + 1.0)
            bigg = np.exp(0.66 * (273.15 - temperature)) - 1.0
            r["mnuccc"] = (_MORR_PI ** 2 / 3.0 * _MORR_RHOW * dap
                            * nacnt * cdist * math.gamma(pgam + 5.0)
                            / lam["c"] ** 4
                            + _MORR_PI ** 2 / 36.0 * _MORR_RHOW * 100.0
                            * cdist * math.gamma(pgam + 7.0)
                            / lam["c"] ** 6 * bigg)
            r["nnuccc"] = (2.0 * _MORR_PI * dap * nacnt * cdist
                            * math.gamma(pgam + 2.0) / lam["c"]
                            + _MORR_PI / 6.0 * 100.0 * cdist
                            * math.gamma(pgam + 4.0) / lam["c"] ** 3 * bigg)
            r["nnuccc"] = min(r["nnuccc"], n["nc"] / dt)
        if q["qs"] >= 1.0e-8:
            cons15 = (-1108.0 * 0.1 * _MORR_PI ** ((1.0 - 0.41) / 3.0)
                      * _MORR_RHOS ** ((-2.0 - 0.41) / 3.0) / (4.0 * 720.0))
            r["nsagg"] = (cons15 * asn * rhoa ** ((2.0 + 0.41) / 3.0)
                           * q["qs"] ** ((2.0 + 0.41) / 3.0)
                           * (n["ns"] * rhoa) ** ((4.0 - 0.41) / 3.0) / rhoa)
        cons13 = math.gamma(3.41) * _MORR_PI / 4.0 * 0.7
        cons14 = math.gamma(3.0 + rimed.bg) * _MORR_PI / 4.0 * 0.7
        cons16 = math.gamma(4.0) * _MORR_PI / 4.0 * 0.7
        if q["qs"] >= 1.0e-8 and q["qc"] >= _MORR_QSMALL:
            r["psacws"] = cons13 * asn * q["qc"] * rhoa * n0s / lam["s"] ** 3.41
            r["npsacws"] = cons13 * asn * n["nc"] * rhoa * n0s / lam["s"] ** 3.41
        if q["qg"] >= 1.0e-8 and q["qc"] >= _MORR_QSMALL:
            r["psacwg"] = (cons14 * agn * q["qc"] * rhoa * n0g
                             / lam["g"] ** (3.0 + rimed.bg))
            r["npsacwg"] = (cons14 * agn * n["nc"] * rhoa * n0g
                              / lam["g"] ** (3.0 + rimed.bg))
        if (q["qi"] >= 1.0e-8 and q["qc"] >= _MORR_QSMALL
                and 1.0 / lam["i"] >= 100.0e-6):
            r["psacwi"] = cons16 * ain * q["qc"] * rhoa * n0i / lam["i"] ** 4
            r["npsacwi"] = cons16 * ain * n["nc"] * rhoa * n0i / lam["i"] ** 4

        if q["qr"] >= 1.0e-8 and q["qs"] >= 1.0e-8:
            ums = min(asn * math.gamma(4.41) / 6.0 / lam["s"] ** 0.41, 1.2 * dens54)
            uns = min(asn * math.gamma(1.41) / lam["s"] ** 0.41, 1.2 * dens54)
            umr = min(arn * math.gamma(4.8) / 6.0 / lam["r"] ** 0.8, 9.1 * dens54)
            unr = min(arn * math.gamma(1.8) / lam["r"] ** 0.8, 9.1 * dens54)
            vrelm = np.sqrt((1.2 * umr - 0.95 * ums) ** 2 + 0.08 * ums * umr)
            vreln = np.sqrt(1.7 * (unr - uns) ** 2 + 0.3 * unr * uns)
            r["pracs"] = min(_MORR_PI ** 2 * _MORR_RHOW * vrelm * rhoa
                              * n0r * n0s / lam["r"] ** 3
                              * (5.0 / (lam["r"] ** 3 * lam["s"])
                                 + 2.0 / (lam["r"] ** 2 * lam["s"] ** 2)
                                 + 0.5 / (lam["r"] * lam["s"] ** 3)),
                              q["qr"] / dt)
            r["npracs"] = (_MORR_PI / 2.0 * rhoa * vreln * n0r * n0s
                            * (1.0 / (lam["r"] ** 3 * lam["s"])
                               + 1.0 / (lam["r"] ** 2 * lam["s"] ** 2)
                               + 1.0 / (lam["r"] * lam["s"] ** 3)))
            if q["qs"] >= 0.1e-3 and q["qr"] >= 0.1e-3:
                r["psacr"] = (_MORR_PI ** 2 * _MORR_RHOS * vrelm * rhoa
                               * n0r * n0s / lam["s"] ** 3
                               * (5.0 / (lam["s"] ** 3 * lam["r"])
                                  + 2.0 / (lam["s"] ** 2 * lam["r"] ** 2)
                                  + 0.5 / (lam["s"] * lam["r"] ** 3)))
        if q["qr"] >= 1.0e-8 and q["qg"] >= 1.0e-8:
            umg = min(agn * math.gamma(4.0 + rimed.bg) / 6.0
                      / lam["g"] ** rimed.bg, 20.0 * dens54)
            ung = min(agn * math.gamma(1.0 + rimed.bg)
                      / lam["g"] ** rimed.bg, 20.0 * dens54)
            umr = min(arn * math.gamma(4.8) / 6.0 / lam["r"] ** 0.8, 9.1 * dens54)
            unr = min(arn * math.gamma(1.8) / lam["r"] ** 0.8, 9.1 * dens54)
            vrelm = np.sqrt((1.2 * umr - 0.95 * umg) ** 2 + 0.08 * umg * umr)
            vreln = np.sqrt(1.7 * (unr - ung) ** 2 + 0.3 * unr * ung)
            r["pracg"] = min(_MORR_PI ** 2 * _MORR_RHOW * vrelm * rhoa
                              * n0r * n0g / lam["r"] ** 3
                              * (5.0 / (lam["r"] ** 3 * lam["g"])
                                 + 2.0 / (lam["r"] ** 2 * lam["g"] ** 2)
                                 + 0.5 / (lam["r"] * lam["g"] ** 3)),
                              q["qr"] / dt)
            r["npracg"] = (_MORR_PI / 2.0 * rhoa * vreln * n0r * n0g
                            * (1.0 / (lam["r"] ** 3 * lam["g"])
                               + 1.0 / (lam["r"] ** 2 * lam["g"] ** 2)
                               + 1.0 / (lam["r"] * lam["g"] ** 3)))

        # Hallett-Mossop rate mutation (WRF 2601-2713).
        if 265.16 < temperature < 270.16:
            fmult = ((270.16 - temperature) / 2.0 if temperature > 268.16
                     else (temperature - 265.16) / 3.0)
            if q["qs"] >= 0.1e-3 and (q["qc"] >= 0.5e-3 or q["qr"] >= 0.1e-3):
                if r["psacws"] > 0.0:
                    r["nmults"] = 35.0e4 * r["psacws"] * fmult * 1000.0
                    r["qmults"] = min(r["nmults"] * (4.0 / 3.0 * _MORR_PI
                                                       * _MORR_RHOI * (5.0e-6) ** 3),
                                       r["psacws"])
                    r["psacws"] -= r["qmults"]
                if r["pracs"] > 0.0:
                    r["nmultr"] = 35.0e4 * r["pracs"] * fmult * 1000.0
                    r["qmultr"] = min(r["nmultr"] * (4.0 / 3.0 * _MORR_PI
                                                       * _MORR_RHOI * (5.0e-6) ** 3),
                                       r["pracs"])
                    r["pracs"] -= r["qmultr"]
            if q["qg"] >= 0.1e-3 and (q["qc"] >= 0.5e-3 or q["qr"] >= 0.1e-3):
                if r["psacwg"] > 0.0:
                    r["nmultg"] = 35.0e4 * r["psacwg"] * fmult * 1000.0
                    r["qmultg"] = min(r["nmultg"] * (4.0 / 3.0 * _MORR_PI
                                                       * _MORR_RHOI * (5.0e-6) ** 3),
                                       r["psacwg"])
                    r["psacwg"] -= r["qmultg"]
                if r["pracg"] > 0.0:
                    r["nmultrg"] = 35.0e4 * r["pracg"] * fmult * 1000.0
                    r["qmultrg"] = min(r["nmultrg"] * (4.0 / 3.0 * _MORR_PI
                                                         * _MORR_RHOI * (5.0e-6) ** 3),
                                        r["pracg"])
                    r["pracg"] -= r["qmultrg"]

        # Ordered snow-to-graupel redirects (WRF 2719-2763).
        if r["psacws"] > 0.0 and q["qs"] >= 0.1e-3 and q["qc"] >= 0.5e-3:
            cons17 = (3.0 * _MORR_RHOSU * _MORR_PI * 0.7 ** 2
                      * math.gamma(2.82) / (rimed.rhog - _MORR_RHOS))
            r["pgsacw"] = min(r["psacws"], cons17 * dt * n0s * q["qc"] ** 2
                                * asn ** 2 / (rhoa * lam["s"] ** 2.82))
            embryo = max(_MORR_RHOS / (rimed.rhog - _MORR_RHOS)
                         * r["pgsacw"], 0.0)
            r["nscng"] = min(embryo / _MORR_MG0 * rhoa, n["ns"] / dt)
            r["psacws"] -= r["pgsacw"]
        if r["pracs"] > 0.0 and q["qs"] >= 0.1e-3 and q["qr"] >= 0.1e-3:
            snow6 = _MORR_RHOS ** 2 * (4.0 / lam["s"]) ** 6
            rain6 = _MORR_RHOW ** 2 * (4.0 / lam["r"]) ** 6
            frac_snow = min(max(snow6 / (snow6 + rain6), 0.0), 1.0)
            r["pgracs"] = (1.0 - frac_snow) * r["pracs"]
            r["ngracs"] = min((1.0 - frac_snow) * r["npracs"],
                               n["nr"] / dt, n["ns"] / dt)
            r["pracs"] -= r["pgracs"]
            r["npracs"] -= r["ngracs"]
            r["psacr"] *= 1.0 - frac_snow

        if temperature < 269.15 and q["qr"] >= _MORR_QSMALL:
            bigg = np.exp(0.66 * (273.15 - temperature)) - 1.0
            r["mnuccr"] = (20.0 * _MORR_PI ** 2 * _MORR_RHOW * 100.0
                            * n["nr"] * bigg / lam["r"] ** 6)
            r["nnuccr"] = min(_MORR_PI * n["nr"] * 100.0 * bigg
                                / lam["r"] ** 3, n["nr"] / dt)
        if q["qi"] >= 1.0e-8 and q["qv"] / qvi >= 1.0:
            # PRCI forms from the UNCAPPED NPRCI; the NI3D/DT limit applies
            # to the number rate only afterwards (F:2833-2836 order).
            r["nprci"] = (4.0 / (_MORR_DCS * _MORR_RHOI)
                          * (q["qv"] - qvi) * rhoa * n0i
                          * np.exp(-lam["i"] * _MORR_DCS) * dv / abi)
            r["prci"] = (_MORR_PI * _MORR_RHOI * _MORR_DCS ** 3 / 6.0
                           * r["nprci"])
            r["nprci"] = min(r["nprci"], n["ni"] / dt)
        if q["qs"] >= 1.0e-8 and q["qi"] >= _MORR_QSMALL:
            cons23 = _MORR_PI / 4.0 * 0.1 * math.gamma(3.41)
            r["prai"] = cons23 * asn * q["qi"] * rhoa * n0s / lam["s"] ** 3.41
            r["nprai"] = min(cons23 * asn * n["ni"] * rhoa * n0s
                              / lam["s"] ** 3.41, n["ni"] / dt)
        if q["qr"] >= 1.0e-8 and q["qi"] >= 1.0e-8:
            cons24 = _MORR_PI / 4.0 * math.gamma(3.8)
            cons25 = _MORR_PI ** 2 / 24.0 * _MORR_RHOW * math.gamma(6.8)
            niacr = min(cons24 * n["ni"] * n0r * arn / lam["r"] ** 3.8 * rhoa,
                        n["nr"] / dt, n["ni"] / dt)
            piacr = cons25 * n["ni"] * n0r * arn / lam["r"] ** 6.8 * rhoa
            praci = cons24 * q["qi"] * n0r * arn / lam["r"] ** 3.8 * rhoa
            if q["qr"] >= 0.1e-3:
                r["niacr"], r["piacr"], r["praci"] = niacr, piacr, praci
            else:
                r["niacrs"], r["piacrs"], r["pracis"] = niacr, piacr, praci
        if ((q["qv"] / qvs >= 0.999 and temperature <= 265.15)
                or q["qv"] / qvi >= 1.08):
            target = min(0.005 * np.exp(0.304 * (273.15 - temperature))
                         * 1000.0, 500.0e3) / rhoa
            if target > n["ni"] + n["ns"] + n["ng"]:
                r["nnuccd"] = (target - n["ni"] - n["ns"] - n["ng"]) / dt
                r["mnuccd"] = r["nnuccd"] * _MORR_MI0
                # Mirror of morrison.cu's cold-trap nucleation bound: below
                # ~159 K the extrapolated liquid POLYSVP drops under its ice
                # curve, qvi == qvs, and the 0.999*QVS trigger fires with qv
                # at or below ice saturation while the one-sided FUDGEF
                # rescale skips the dum <= 0 / sum_dep > 0 sign pair (the
                # qv = -1.6e-4 kg/kg polar-night model-top abort, T255
                # native suite, 2026-09-01).  The nucleated mass is bounded
                # by the vapor excess over ice saturation, the number moment
                # scaling with it so both sides of the process shrink
                # together.  Above the trap the bound also engages in cold
                # first nucleation whose excess is under the Cooper embryo
                # mass (the 1.08 arm near 200 K, a reachable WRF tropopause
                # state): the dum > 0 FUDGEF branch already caps the mass
                # in both models there, so qv and qi match WRF and only
                # the number moment shrinks -- crystals keep MI0 seed mass
                # instead of WRF's mass-starved count.
                avail = max(q["qv"] - qvi, 0.0) / dt
                if r["mnuccd"] > avail:
                    r["nnuccd"] *= avail / r["mnuccd"]
                    r["mnuccd"] = avail

        epsi = (2.0 * _MORR_PI * n0i * rhoa * dv / lam["i"] ** 2
                if q["qi"] >= _MORR_QSMALL else 0.0)
        epss = (2.0 * _MORR_PI * n0s * rhoa * dv
                * (0.86 / lam["s"] ** 2
                   + 0.28 * np.sqrt(asn * rhoa / mu) * sc ** (1.0 / 3.0)
                   * math.gamma(2.705) / lam["s"] ** 2.705)
                if q["qs"] >= _MORR_QSMALL else 0.0)
        epsg = (2.0 * _MORR_PI * n0g * rhoa * dv
                * (0.86 / lam["g"] ** 2
                   + 0.28 * np.sqrt(agn * rhoa / mu) * sc ** (1.0 / 3.0)
                   * math.gamma(2.5 + rimed.bg / 2.0)
                   / lam["g"] ** (2.5 + rimed.bg / 2.0))
                if q["qg"] >= _MORR_QSMALL else 0.0)
        dep = (q["qv"] - qvi) / abi
        tail = (1.0 - np.exp(-lam["i"] * _MORR_DCS)
                * (1.0 + lam["i"] * _MORR_DCS)
                if q["qi"] >= _MORR_QSMALL else 0.0)
        r["prd"] = epsi * dep * tail
        if q["qs"] >= _MORR_QSMALL:
            r["prds"] = epss * dep + epsi * dep * (1.0 - tail)
        else:
            r["prd"] += epsi * dep * (1.0 - tail)
        r["prdg"] = epsg * dep
        sum_dep = r["prd"] + r["prds"] + r["prdg"] + r["mnuccd"]
        dum = (q["qv"] - qvi) / dt
        if sum_dep != 0.0 and ((dum > 0.0 and sum_dep > dum * 0.9999)
                               or (dum < 0.0 and sum_dep < dum * 0.9999)):
            ratio = 0.9999 * dum / sum_dep
            for name in ("prd", "prds", "prdg", "mnuccd"):
                r[name] *= ratio
        for depname, subname in (("prd", "eprd"), ("prds", "eprds"),
                                 ("prdg", "eprdg")):
            if r[depname] < 0.0:
                r[subname], r[depname] = r[depname], 0.0

        # Joint WRF donor ratios; mass rates only (3086-3187).
        loss = (r["prc"] + r["pra"] + r["mnuccc"] + r["psacws"]
                + r["psacwi"] + r["qmults"] + r["psacwg"]
                + r["pgsacw"] + r["qmultg"]) * dt
        if loss > q["qc"] and q["qc"] >= _MORR_QSMALL:
            ratio = q["qc"] / loss
            for name in ("prc", "pra", "mnuccc", "psacws", "psacwi",
                         "qmults", "qmultg", "psacwg", "pgsacw"):
                r[name] *= ratio
        loss = (-r["prd"] - r["mnuccc"] + r["prci"] + r["prai"]
                - r["qmults"] - r["qmultg"] - r["qmultr"]
                - r["qmultrg"] - r["mnuccd"] + r["praci"]
                + r["pracis"] - r["eprd"] - r["psacwi"]) * dt
        if loss > q["qi"] and q["qi"] >= _MORR_QSMALL:
            ratio = ((q["qi"] / dt + r["prd"] + r["mnuccc"]
                      + r["qmults"] + r["qmultg"] + r["qmultr"]
                      + r["qmultrg"] + r["mnuccd"] + r["psacwi"])
                     / (r["prci"] + r["prai"] + r["praci"]
                        + r["pracis"] - r["eprd"]))
            for name in ("prci", "prai", "praci", "pracis", "eprd"):
                r[name] *= ratio
        loss = ((r["pracs"] - r["pre"]) + (r["qmultr"] + r["qmultrg"]
                - r["prc"]) + (r["mnuccr"] - r["pra"]) + r["piacr"]
                + r["piacrs"] + r["pgracs"] + r["pracg"]) * dt
        if loss > q["qr"] and q["qr"] >= _MORR_QSMALL:
            ratio = ((q["qr"] / dt + r["prc"] + r["pra"])
                     / (-r["pre"] + r["qmultr"] + r["qmultrg"]
                        + r["pracs"] + r["mnuccr"] + r["piacr"]
                        + r["piacrs"] + r["pgracs"] + r["pracg"]))
            for name in ("pre", "pracs", "qmultr", "qmultrg", "mnuccr",
                         "piacr", "piacrs", "pgracs", "pracg"):
                r[name] *= ratio
        loss = (-r["prds"] - r["psacws"] - r["prai"] - r["prci"]
                - r["pracs"] - r["eprds"] + r["psacr"] - r["piacrs"]
                - r["pracis"]) * dt
        if loss > q["qs"] and q["qs"] >= _MORR_QSMALL:
            ratio = ((q["qs"] / dt + r["prds"] + r["psacws"] + r["prai"]
                      + r["prci"] + r["pracs"] + r["piacrs"] + r["pracis"])
                     / (-r["eprds"] + r["psacr"]))
            r["eprds"] *= ratio
            r["psacr"] *= ratio
        loss = (-r["psacwg"] - r["pracg"] - r["pgsacw"] - r["pgracs"]
                - r["prdg"] - r["mnuccr"] - r["eprdg"] - r["piacr"]
                - r["praci"] - r["psacr"]) * dt
        if loss > q["qg"] and q["qg"] >= _MORR_QSMALL:
            ratio = ((q["qg"] / dt + r["psacwg"] + r["pracg"]
                      + r["pgsacw"] + r["pgracs"] + r["prdg"]
                      + r["mnuccr"] + r["psacr"] + r["piacr"] + r["praci"])
                     / (-r["eprdg"]))
            r["eprdg"] *= ratio

        tqv = (-r["pre"] - r["prd"] - r["prds"] - r["mnuccd"]
               - r["eprd"] - r["eprds"] - r["prdg"] - r["eprdg"])
        tt = (r["pre"] * xlv
              + (r["prd"] + r["prds"] + r["mnuccd"] + r["eprd"]
                 + r["eprds"] + r["prdg"] + r["eprdg"]) * xls
              + (r["psacws"] + r["psacwi"] + r["mnuccc"] + r["mnuccr"]
                 + r["qmults"] + r["qmultg"] + r["qmultr"] + r["qmultrg"]
                 + r["pracs"] + r["psacwg"] + r["pracg"] + r["pgsacw"]
                 + r["pgracs"] + r["piacr"] + r["piacrs"]) * xlf) / cpm
        tqc = (-r["pra"] - r["prc"] - r["mnuccc"] - r["psacws"]
               - r["psacwi"] - r["qmults"] - r["qmultg"]
               - r["psacwg"] - r["pgsacw"])
        tqi = (r["prd"] + r["eprd"] + r["psacwi"] + r["mnuccc"]
               - r["prci"] - r["prai"] + r["qmults"] + r["qmultg"]
               + r["qmultr"] + r["qmultrg"] + r["mnuccd"]
               - r["praci"] - r["pracis"])
        tqr = (r["pre"] + r["pra"] + r["prc"] - r["pracs"]
               - r["mnuccr"] - r["qmultr"] - r["qmultrg"]
               - r["piacr"] - r["piacrs"] - r["pracg"] - r["pgracs"])
        tqs = (r["prai"] + r["psacws"] + r["prds"] + r["pracs"]
               + r["prci"] + r["eprds"] - r["psacr"] + r["piacrs"]
               + r["pracis"])
        tqg = (r["pracg"] + r["psacwg"] + r["pgsacw"] + r["pgracs"]
               + r["prdg"] + r["eprdg"] + r["mnuccr"] + r["piacr"]
               + r["praci"] + r["psacr"])
        tnc = (-r["nnuccc"] - r["npsacws"] - r["npra"] - r["nprc"]
               - r["npsacwi"] - r["npsacwg"])
        tni = (r["nnuccc"] - r["nprci"] - r["nprai"] + r["nmults"]
               + r["nmultg"] + r["nmultr"] + r["nmultrg"]
               + r["nnuccd"] - r["niacr"] - r["niacrs"])
        tnr = (r["nprc1"] - r["npracs"] - r["nnuccr"] + r["nragg"]
               - r["niacr"] - r["niacrs"] - r["npracg"] - r["ngracs"])
        tns = r["nsagg"] + r["nprci"] - r["nscng"] - r["ngracs"] + r["niacrs"]
        tng = r["nscng"] + r["ngracs"] + r["nnuccr"] + r["niacr"]
        if r["eprd"] < 0.0:
            r["nsubi"] = max(-1.0, r["eprd"] * dt / q["qi"]) * n["ni"] / dt
        if r["eprds"] < 0.0:
            r["nsubs"] = max(-1.0, r["eprds"] * dt / q["qs"]) * n["ns"] / dt
        if r["pre"] < 0.0:
            r["nsubr"] = max(-1.0, r["pre"] * dt / q["qr"]) * n["nr"] / dt
        if r["eprdg"] < 0.0:
            r["nsubg"] = max(-1.0, r["eprdg"] * dt / q["qg"]) * n["ng"] / dt
        tni += r["nsubi"]
        tns += r["nsubs"]
        tnr += r["nsubr"]
        tng += r["nsubg"]

    # WRF saturation adjustment is evaluated from the collectively
    # predicted state, then folded into the same one state update.
    t_pred = temperature + dt * tt
    qv_pred = q["qv"] + dt * tqv
    qc_pred = max(q["qc"] + dt * tqc, 0.0)
    ew_pred = min(0.99 * pressure, float(_np_morrison_polysvp(t_pred, False)))
    qss_pred = c.EP2 * ew_pred / (pressure - ew_pred)
    pcc = ((qv_pred - qss_pred)
           / (1.0 + xlv ** 2 * qss_pred / (cpm * c.RV * t_pred ** 2))) / dt
    if pcc * dt + qc_pred < 0.0:
        pcc = -qc_pred / dt
    tqv -= pcc
    tqc += pcc
    tt += pcc * xlv / cpm

    qnew = {
        "qv": q["qv"] + dt * tqv, "qc": q["qc"] + dt * tqc,
        "qr": q["qr"] + dt * tqr, "qi": q["qi"] + dt * tqi,
        "qs": q["qs"] + dt * tqs, "qg": q["qg"] + dt * tqg,
    }
    nnew = {
        "nc": n["nc"] + dt * tnc, "nr": n["nr"] + dt * tnr,
        "ni": n["ni"] + dt * tni, "ns": n["ns"] + dt * tns,
        "ng": n["ng"] + dt * tng,
    }
    return (qnew, nnew, temperature + dt * tt, stale_lami,
            cloud_nc_for_sedimentation)


def _np_morrison_seed_cumulus_numbers(
        nr, ns, ni, qrcuten, qscuten, qicuten, rho, dt):
    """WRF Morrison's KF number-moment injection (F:1327-1343).

    Inputs and outputs are number concentrations per kg dry air.  The three
    cumulus inputs are the raw, uncoupled KF mass-mixing-ratio rates in s-1;
    WRF applies them in rain, snow, ice statement order before entry cleanup
    and PSD reconstruction.  The comparison is deliberately ``>= 1e-10``.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in
              (nr, ns, ni, qrcuten, qscuten, qicuten, rho)]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays):
        raise ValueError("Morrison cumulus number-seed arrays must share shape")
    if (not np.isfinite(dt) or dt <= 0.0
            or not all(np.isfinite(value).all() for value in arrays)
            or np.any(arrays[-1] <= 0.0)):
        raise ValueError("Morrison cumulus number seeding requires finite "
                         "arrays, positive density, and positive dt")
    nr_out, ns_out, ni_out = (value.copy() for value in arrays[:3])
    qrcu, qscu, qicu, rhoa = arrays[3:]
    rain = qrcu >= 1.0e-10
    snow = qscu >= 1.0e-10
    ice = qicu >= 1.0e-10
    nr_out[rain] += 1.8e5 * (
        qrcu[rain] * dt
        / (_MORR_PI * _MORR_RHOW * rhoa[rain] ** 3)) ** 0.25
    ns_out[snow] += 3.0e5 * (
        qscu[snow] * dt
        / (100.0 * _MORR_PI * rhoa[snow] ** 3)) ** 0.25
    ni_out[ice] += qicu[ice] * dt / (
        _MORR_CI * (80.0e-6) ** 3)
    return nr_out, ns_out, ni_out


def np_morrison_column(theta, qv, qc, qr, qi, qs, qg,
                       nc, nr, ni, ns, ng, rho, pii, pressure, dz, dt,
                       rainnc=0.0, snownc=0.0, graupelnc=0.0,
                       morr_rimed_ice=1, *, qrcuten=None, qscuten=None,
                       qicuten=None):
    """Float64 one-column mirror of woof's WRF Morrison port.

    The ordering follows ``MORR_TWO_MOMENT_MICRO``: time-varying
    thermodynamics and fixed droplet number (WRF lines 1270-1496), bounded
    gamma/exponential PSDs (1525-1638 and 2122-2267), warm/cold conversion
    processes and donor limiters (1667-3316), scheme liquid saturation
    adjustment (2048-2070 and 3249-3267), internal sedimentation substeps
    (3356-3677), and final phase/number bounds (3683-4055).

    All input arrays are ``(nz,)`` and copied.  Mixing ratios are kg/kg dry
    air, moments kg-1, precipitation increments numerically kg m-2 == mm.
    ``rho`` is accepted to mirror the WRF wrapper but, as its line 589
    documents, Morrison recomputes air density from pressure/temperature.
    ``morr_rimed_ice`` is WRF's scalar hail (1, default) / graupel (0)
    selector (Registry.EM_COMMON:2663-2666; Morrison F:337-411).  The optional
    ``qrcuten``/``qscuten``/``qicuten`` triplet carries raw KF mass rates;
    all three must be supplied together and seed Nr/Ns/Ni in WRF statement
    order before process calculations (F:1327-1343).
    """
    names = ("theta", "qv", "qc", "qr", "qi", "qs", "qg",
             "nc", "nr", "ni", "ns", "ng", "rho", "pii",
             "pressure", "dz")
    values = (theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
              rho, pii, pressure, dz)
    a = {name: np.asarray(value, np.float64).copy()
         for name, value in zip(names, values)}
    nz = a["theta"].size
    if nz < 2:
        raise ValueError("Morrison column requires nz >= 2")
    if any(value.shape != (nz,) for value in a.values()):
        raise ValueError("all Morrison column inputs must have shape (nz,)")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    # Validate once at the public mirror boundary.  WRF Registry defaults
    # this scalar to 1 = hail (Registry.EM_COMMON:2663-2666).
    rimed_ice_constants(morr_rimed_ice)

    t = a["theta"] * a["pii"]
    p = a["pressure"]
    rhoa = p / (c.RD * t)       # WRF source line 1325; input rho unused
    q = {name: np.maximum(a[name], 0.0)
         for name in ("qv", "qc", "qr", "qi", "qs", "qg")}
    n = {name: np.maximum(a[name], 0.0)
         for name in ("nc", "nr", "ni", "ns", "ng")}
    cu_values = (qrcuten, qscuten, qicuten)
    if any(value is not None for value in cu_values):
        if not all(value is not None for value in cu_values):
            raise ValueError("qrcuten, qscuten, and qicuten must be supplied "
                             "together")
        cu = [np.asarray(value, dtype=np.float64) for value in cu_values]
        if any(value.shape != (nz,) for value in cu):
            raise ValueError("Morrison cumulus tendencies must have shape "
                             f"{(nz,)}")
        n["nr"], n["ns"], n["ni"] = \
            _np_morrison_seed_cumulus_numbers(
                n["nr"], n["ns"], n["ni"], *cu, rhoa, dt)
    ice_to_snow = np.zeros(nz, dtype=bool)
    stale_lami = np.zeros(nz, dtype=np.float64)
    cloud_nc_for_sedimentation = np.zeros(nz, dtype=np.float64)
    xlv_stale = np.zeros(nz, dtype=np.float64)
    cpm_stale = np.zeros(nz, dtype=np.float64)
    # A mass donor that is exhausted algebraically can leave a positive
    # O(FP32 roundoff) remnant in the device collective update.  WRF then
    # sees that category when selecting its discontinuous column NSTEP even
    # though the float64 rate sum is exactly zero.  Record that FP32-floor
    # branch mask without adding mass to the mirror state.
    sediment_floor = {name: np.zeros(nz, dtype=bool)
                      for name in ("qc", "qr", "qi", "qs", "qg")}

    for k in range(nz):
        # Thermodynamics are diagnosed before WRF's entry cleanup.
        xlv = 3.1484e6 - 2370.0 * t[k]
        xls = 3.15e6 - 2370.0 * t[k] + 0.3337e6
        cpm = c.CP * (1.0 + 0.887 * q["qv"][k])
        xlv_stale[k] = xlv
        cpm_stale[k] = cpm
        ew = min(0.99 * p[k], float(_np_morrison_polysvp(t[k], False)))
        ei = min(ew, 0.99 * p[k], float(_np_morrison_polysvp(t[k], True)))
        qvs = c.EP2 * ew / (p[k] - ew)
        qvi = c.EP2 * ei / (p[k] - ei)

        # WRF 1345-1377 entry cleanup precedes the one phase decision.
        if q["qv"][k] / qvs < 0.9:
            for mass in ("qr", "qc"):
                if q[mass][k] < 1.0e-8:
                    amount = q[mass][k]
                    q["qv"][k] += amount
                    q[mass][k] = 0.0
                    t[k] -= amount * xlv / cpm
        if q["qv"][k] / qvi < 0.9:
            for mass in ("qi", "qs", "qg"):
                if q[mass][k] < 1.0e-8:
                    amount = q[mass][k]
                    q["qv"][k] += amount
                    q[mass][k] = 0.0
                    t[k] -= amount * xls / cpm
        for mass, moment in (("qc", "nc"), ("qr", "nr"),
                             ("qi", "ni"), ("qs", "ns"), ("qg", "ng")):
            if q[mass][k] < _MORR_QSMALL:
                q[mass][k] = 0.0
                n[moment][k] = 0.0

        warm = bool(t[k] >= 273.15)  # WRF 1486/3327: latch once.
        qk = {name: float(q[name][k]) for name in q}
        nk = {name: float(n[name][k]) for name in n}
        (qnew, nnew, tnew, stale_lami[k],
         cloud_nc_for_sedimentation[k]) = _np_morrison_apply_level(
            qk, nk, float(t[k]), float(p[k]), float(rhoa[k]), float(dt),
            float(qvs), float(qvi), float(xlv), float(xls), float(cpm), warm,
            morr_rimed_ice=morr_rimed_ice)
        if not warm:
            fp32 = np.finfo(np.float32).eps
            for mass in sediment_floor:
                sediment_floor[mass][k] = (
                    qk[mass] >= _MORR_QSMALL
                    and qnew[mass] <= 64.0 * fp32 * qk[mass])
        # WRF 3679-3690 uses stale cold-branch LAMI; warm LAMI is zero.
        ice_to_snow[k] = (not warm and qk["qi"] >= _MORR_QSMALL
                          and stale_lami[k] >= 1.0e-10
                          and 1.0 / stale_lami[k] >= 2.0 * _MORR_DCS)
        for name, value in qnew.items():
            q[name][k] = value
        for name, value in nnew.items():
            n[name][k] = value
        t[k] = tnew

    # WRF internal sedimentation includes all five categories and both
    # moments on one shared column-wide NSTEP (source 3356-3667).  Fall
    # speeds are evaluated once, then fluxes are recomputed every substep.
    velocities = {}
    for kind, mass, moment in (("c", "qc", "nc"), ("r", "qr", "nr"),
                               ("i", "qi", "ni"), ("s", "qs", "ns"),
                               ("g", "qg", "ng")):
        q_short = {"c": q["qc"], "r": q["qr"], "i": q["qi"],
                   "s": q["qs"], "g": q["qg"]}
        proxy = q_short[kind].copy()
        proxy[sediment_floor[mass]] = np.maximum(
            proxy[sediment_floor[mass]], _MORR_QSMALL)
        q_short[kind] = proxy
        n_short = {"c": cloud_nc_for_sedimentation,
                   "r": n["nr"], "i": n["ni"],
                   "s": n["ns"], "g": n["ng"]}
        vm, vn, _ = _np_morrison_fall_speeds(
            kind, q_short, n_short, rhoa, t,
            morr_rimed_ice=morr_rimed_ice)
        velocities[kind] = (vm, vn)
    qdens = {kind: q[mass] * rhoa for kind, mass in
             (("c", "qc"), ("r", "qr"), ("i", "qi"),
              ("s", "qs"), ("g", "qg"))}
    ndens = {kind: n[moment] * rhoa for kind, moment in
             (("c", "nc"), ("r", "nr"), ("i", "ni"),
              ("s", "ns"), ("g", "ng"))}
    # INUM=1 overrides only cloud DUMFNC: omit the local NC3DTEN from the
    # sedimenting provisional moment (WRF 3367-3374).
    ndens["c"] = cloud_nc_for_sedimentation * rhoa
    qdens, ndens, exported, _ = _np_morrison_advect_sedimentation(
        qdens, ndens, velocities, a["dz"], dt)
    for kind, mass, moment in (("c", "qc", "nc"), ("r", "qr", "nr"),
                               ("i", "qi", "ni"), ("s", "qs", "ns"),
                               ("g", "qg", "ng")):
        q[mass] = np.maximum(qdens[kind] / rhoa, 0.0)
        if kind == "c":
            n[moment] = np.maximum(
                n[moment] + ndens[kind] / rhoa
                - cloud_nc_for_sedimentation, 0.0)
        else:
            n[moment] = np.maximum(ndens[kind] / rhoa, 0.0)
    precip = sum(exported.values())
    snow = exported["i"] + exported["s"]
    graupel = exported["g"]

    q, n, t = _np_morrison_final_phase_cleanup(
        q, n, t, rhoa, ice_to_snow, a["pressure"],
        xlv_stale=xlv_stale, cpm_stale=cpm_stale)

    # Final zeroing/PSD bounds and WRF ice-number cap (3766-4055).
    for mass, moment in (("qc", "nc"), ("qr", "nr"), ("qi", "ni"),
                         ("qs", "ns"), ("qg", "ng")):
        empty = q[mass] < _MORR_QSMALL
        q[mass][empty] = 0.0
        n[moment][empty] = 0.0
    lam, pgam, bounded = _np_morrison_slopes(
        {"c": q["qc"], "r": q["qr"], "i": q["qi"],
         "s": q["qs"], "g": q["qg"]},
        {"c": n["nc"], "r": n["nr"], "i": n["ni"],
         "s": n["ns"], "g": n["ng"]}, rhoa, t,
        pressure=a["pressure"], reset_cloud_number=False,
        morr_rimed_ice=morr_rimed_ice)
    safe = {name: np.where(value > 0.0, value, 1.0)
            for name, value in lam.items()}
    effc = np.where(q["qc"] >= _MORR_QSMALL,
                    (pgam + 3.0) / (2.0 * safe["c"]) * 1.0e6, 25.0)
    effr = np.where(q["qr"] >= _MORR_QSMALL, 1.5e6 / safe["r"], 25.0)
    effi = np.where(q["qi"] >= _MORR_QSMALL, 1.5e6 / safe["i"], 25.0)
    effs = np.where(q["qs"] >= _MORR_QSMALL, 1.5e6 / safe["s"], 25.0)
    n = {"nc": bounded["c"], "nr": bounded["r"],
         "ni": np.minimum(bounded["i"], 0.3e6 / rhoa),
         "ns": bounded["s"], "ng": bounded["g"]}
    n["nc"] = 250.0e6 / rhoa

    out = {"theta": t / a["pii"], **q, **n,
           "effc": effc, "effr": effr, "effi": effi, "effs": effs,
           "rho": a["rho"], "pii": a["pii"],
           "pressure": a["pressure"], "dz": a["dz"]}
    out.update({"precip_step": precip, "snow_step": snow,
                "graupel_step": graupel,
                "rainnc": float(rainnc) + precip,
                "snownc": float(snownc) + snow,
                "graupelnc": float(graupelnc) + graupel,
                "sr": (snow + graupel) / (precip + 1.0e-12)})
    return out


# ---------------------------------------------------------------------------
# h_diabatic: microphysics latent heating retained as an RK dynamics tendency
# (WRF Registry.EM_COMMON:1389, "MICROPHYSICS LATENT HEATING", K s-1)
# ---------------------------------------------------------------------------

#: WRF namelist defaults the mechanism consumes: ``no_mp_heating``
#: (Registry.EM_COMMON:2630, default 0 = heating ON) and ``mp_tend_lim``
#: (Registry.EM_COMMON:2642, default 10. K/s -- the clamp on the per-step
#: microphysics theta increment in ``moist_physics_finish_em``).
NO_MP_HEATING_DEFAULT = 0
MP_TEND_LIM_DEFAULT = 10.0


def np_moist_physics_finish(th_after, th_saved, dt, *, no_mp_heating=0,
                            mp_tend_lim=MP_TEND_LIM_DEFAULT):
    """Mirror of WRF ``moist_physics_finish_em``
    (dyn_em/module_big_step_utilities_em.F:5593-5784), ``use_theta_m = 0``
    branch (woof's dry-theta prognostic; the oracle run's Registry-default
    ``use_theta_m = 1`` frame is the registered project-wide convention
    deviation, see woof.core.moist).

    ``th_saved`` is the pre-microphysics FULL theta that
    ``moist_physics_prep_em`` parked in the h_diabatic array
    (:5503 "use h_diabatic to temporarily save pre-microphysics full
    theta", :5523-5526); ``th_after`` is the scheme's updated full theta.
    With ``no_mp_heating == 0`` (:5682) the theta increment ``mpten =
    th_after - th_saved`` (:5688) is clamped to ``+/- mp_tend_lim*dt``
    (:5706-5707), that product formed as WRF forms it from two REALs
    (``float32(float32(mp_tend_lim) * float32(dt))``), applied ONCE directly to the prognostic theta
    (``t_new = t_new + mpten``, :5743), and stored as the heating rate
    ``h_diabatic = mpten/dt`` (:5745) for the NEXT step's RK tendencies.
    With ``no_mp_heating = 1`` theta is left untouched (:5775, commented
    out on purpose) and ``h_diabatic = 0`` (:5776).

    Returns ``(th_new, h_diabatic)``, both float64.
    """
    th_after = np.asarray(th_after, dtype=np.float64)
    th_saved = np.asarray(th_saved, dtype=np.float64)
    if no_mp_heating == 0:
        # WRF's bound is the REAL product of two REALs: one float32
        # rounding of float32 operands, not the double product rounded.
        lim = float(np.float32(np.float32(mp_tend_lim) * np.float32(dt)))
        mpten = th_after - th_saved                       # :5688
        mpten = np.minimum(lim, mpten)                    # :5706
        mpten = np.maximum(-lim, mpten)                   # :5707
        return th_saved + mpten, mpten / dt               # :5743, :5745
    return th_saved.copy(), np.zeros_like(th_saved)       # :5775-5776


def np_h_diabatic_tendency(h_diabatic, mut, c1h, c2h, msft=None):
    """Mirror of ``rk_addtend_dry``'s h_diabatic term (dyn_em/
    module_em.F:1076-1080): the coupled theta tendency contribution
    ``(c1(k)*mut + c2(k)) * h_diabatic / msfty``, added EVERY RK step with
    the step's total dry column mass ``mut = mub + mu_2`` (rk_step_prep's
    ``CALL calculate_full``, module_em.F:143 / solve_em.F:652-666).
    ``msft=None`` is the identity
    map factor.  Returns the (nz, ny, nx) float64 contribution to be ADDED
    to the coupled slow theta tendency.
    """
    h_diabatic = np.asarray(h_diabatic, dtype=np.float64)
    mut = np.asarray(mut, dtype=np.float64)
    c1h = np.asarray(c1h, dtype=np.float64)[:, None, None]
    c2h = np.asarray(c2h, dtype=np.float64)[:, None, None]
    hd = (c1h * mut[None] + c2h) * h_diabatic
    if msft is not None:
        hd = hd / np.asarray(msft, dtype=np.float64)[None]
    return hd


def np_small_step_finish_theta(thp, th_pp, mu_s, mu_new, thb, c1h, c2h,
                               h_diabatic=None, dt_stage=0.0):
    """Mirror of the theta fold in ``dycore._finish_small_steps`` with WRF
    ``small_step_finish``'s h_diabatic removal (dyn_em/
    module_small_step_em.F:408-426): on the FINAL RK step only
    (``rk_step == rk_order``, :416) the coupled update subtracts
    ``dts*number_of_small_timesteps*(c1h(k)*mut + c2h(k))*h_diabatic``
    (:421) -- exactly what the per-substep integration of the
    ``rk_addtend_dry`` term added over the stage (advance_mu_t
    ``t = t + msfty*dts*ft``, :1142; the map factors cancel), so the net
    h_diabatic contribution to theta(t+dt) is zero and the heating enters
    the state once, via ``moist_physics_finish_em``'s direct update.

    ``mu_s`` is the stage-reference total dry mass (WRF ``mut``),
    ``mu_new`` the post-substep mass (WRF ``muts``); ``dt_stage`` is
    ``dts*number_of_small_timesteps`` (the full dt on the final stage) and
    0 on stages 1-2 (:408-415, no removal).  Returns the new uncoupled
    perturbation theta, float64.
    """
    thp = np.asarray(thp, dtype=np.float64)
    th_pp = np.asarray(th_pp, dtype=np.float64)
    mu_s = np.asarray(mu_s, dtype=np.float64)
    mu_new = np.asarray(mu_new, dtype=np.float64)
    thb = _prof3(thb)
    c1h = np.asarray(c1h, dtype=np.float64)[:, None, None]
    c2h = np.asarray(c2h, dtype=np.float64)[:, None, None]
    th_num = (c1h * mu_s[None] + c2h) * (thb + thp) + th_pp
    if h_diabatic is not None and dt_stage:
        th_num = th_num - (dt_stage * (c1h * mu_s[None] + c2h)
                           * np.asarray(h_diabatic, dtype=np.float64))
    return th_num / (c1h * mu_new[None] + c2h) - thb


def np_calc_ww_cp(ru, rv, dnw, c1h, rdx, rdy, msft=None, top_residual=None):
    """WRF ``calc_ww_cp`` in float32 scalars, one rounding per operation.

    The CPU oracle for :func:`woof.core.dycore._omega_ref`: the Fortran
    loop of dyn_em/module_big_step_utilities_em.F transcribed operator for
    operator on NumPy float32 scalars, which round every product and sum
    to single precision with no contraction, so the device kernel is
    checked against it on the uint32 view rather than with a tolerance.
    Per column, in the Fortran's order:

    * ``divv(k) = (msftx*dnw(k)) * (rdx*(ru(i+1) - ru(i)) + rdy*(rv(j+1)
      - rv(j)))``, the map factor and thickness multiplied first;
    * ``dmdt = dmdt + divv(k)`` sequentially from the surface up;
    * ``ww(1) = 0``, ``ww(kte) = 0``, and for the interior
      ``ww(k) = (ww(k-1) - (dnw(k-1)*c1h(k-1))*dmdt) - divv(k-1)``.

    ``ru`` is (nz, ny, nx+1), ``rv`` is (nz, ny+1, nx), ``dnw``/``c1h`` are
    (nz,), ``msft`` is (ny, nx) or None (WRF with ``msftx = 1``); every
    array must already be float32 and ``rdx``/``rdy`` float32 scalars (the
    single-precision ``1./dx`` of solve_em), because a float64 operand would
    silently widen the chain.  Returns ``ww`` (nz+1, ny, nx) float32.  When
    ``top_residual`` is an (ny, nx) float32 array it receives what the
    recurrence WOULD have produced at the top level, the value WRF discards
    in favour of the exact zero: the column closure, which is zero only to
    the rounding of the two sums it cancels.
    """
    f32 = np.float32
    arrays = {"ru": ru, "rv": rv, "dnw": dnw, "c1h": c1h}
    if msft is not None:
        arrays["msft"] = msft
    for name, arr in arrays.items():
        if arr.dtype != np.float32:
            raise TypeError(f"np_calc_ww_cp: {name} must be float32, "
                            f"got {arr.dtype}")
    for name, val in (("rdx", rdx), ("rdy", rdy)):
        if not isinstance(val, np.float32):
            raise TypeError(f"np_calc_ww_cp: {name} must be a numpy "
                            f"float32 scalar, got {type(val).__name__}")
    nz, ny, nxp = ru.shape
    nx = nxp - 1
    if rv.shape != (nz, ny + 1, nx):
        raise ValueError(f"np_calc_ww_cp: rv shape {rv.shape} does not "
                         f"match ru shape {ru.shape}")
    ww = np.zeros((nz + 1, ny, nx), dtype=np.float32)
    divv = np.empty(nz, dtype=np.float32)
    zero = f32(0.0)
    for j in range(ny):
        for i in range(nx):
            m = None if msft is None else msft[j, i]
            dmdt = zero
            for k in range(nz):
                du = ru[k, j, i + 1] - ru[k, j, i]
                dv = rv[k, j + 1, i] - rv[k, j, i]
                bracket = (rdx * du) + (rdy * dv)
                weight = dnw[k] if m is None else m * dnw[k]
                d = weight * bracket
                divv[k] = d
                dmdt = dmdt + d
            w = zero
            ww[0, j, i] = w
            for k in range(1, nz):
                w = w - (dnw[k - 1] * c1h[k - 1]) * dmdt
                w = w - divv[k - 1]
                ww[k, j, i] = w
            if top_residual is not None:
                t = w - (dnw[nz - 1] * c1h[nz - 1]) * dmdt
                top_residual[j, i] = t - divv[nz - 1]
            ww[nz, j, i] = zero
    return ww


def random_acoustic_state(seed=0, nz=8, ny=2, nx=12, stretch=None,
                          hybrid_opt=0, hill_height=0.0,
                          msf_amp=0.0, f_amp=0.0, moist=False,
                          mp_physics=0):
    """Small consistent ``DomainState`` + ``RunConfig`` for acoustic tests.

    GPU only (allocates device arrays).  Builds a discretely balanced base
    state, fills the reference (t*) fields, large-step tendencies, and the
    acoustic ``_pp`` perturbation fields with reproducible random values,
    and recomputes the EOS diagnostics so ``p/al/alt`` are consistent.
    Periodic duplicates (u face nx, v row ny) are enforced.  Amplitudes are
    chosen so every fp32 intermediate in the acoustic kernels stays O(1),
    keeping the device-vs-float64 comparison meaningful at tight tolerances.
    Reused by Task 11 (carries w/ph state and tendencies as well).

    ``stretch`` (Task 2, Phase 2) selects the tanh-stretched eta grid of
    ``make_vertical_coord``: nonuniform spacing makes fnm != fnp and
    rdn != rdnw, so the acoustic matching tests can observe the
    interpolation-weight and spacing orientations that are degenerate on
    the uniform grid.

    ``hybrid_opt``/``hill_height`` (Task 4) select the WRF cubic-B hybrid
    coordinate and a bell-ridge terrain (halfwidth 1.5 km on the 500 m
    grid), making the c1/c2-weighted couplings and the terrain
    base-pressure-gradient paths of the acoustic kernels observable; the
    defaults keep the exact Phase 1 flat/sigma state.

    ``msf_amp``/``f_amp`` (Phase 3 Task 3): nonzero values fill random map
    factors ``1 + msf_amp*U(0,1)`` (periodic duplicates enforced) and
    Coriolis parameters ``f_amp*U(0,1)`` via ``set_map_coriolis``, making
    the msf-weighted acoustic/advection paths and the Coriolis kernel
    observable; the defaults keep the exact Phase 2 identity fields.

    ``moist``/``mp_physics`` let device-reference tests allocate the matching
    WRF Registry moisture package; callers fill the desired mixing ratios.
    """
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest
    from woof.core.terrain import bell_hill

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=3.0, run_seconds=0.0, hybrid_opt=hybrid_opt,
                    terrain_opt=(1 if hill_height > 0.0 else 0),
                    hill_height=hill_height, hill_halfwidth=1500.0,
                    moist=moist, mp_physics=mp_physics)
    coord = make_vertical_coord(nz, stretch=stretch, hybrid_opt=hybrid_opt,
                                etac=cfg.etac)
    terrain_z = bell_hill(cfg) if hill_height > 0.0 else None
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           p_surf=cfg.p_surf, ztop=cfg.ztop,
                           terrain_z=terrain_z)
    s = init_at_rest(cfg, coord, base)
    rng = np.random.default_rng(seed)

    def fill(name, amp, xdup=False, ydup=False):
        arr = getattr(s, name)
        vals = amp * rng.standard_normal(arr.shape)
        if xdup:
            vals[..., -1] = vals[..., 0]
        if ydup:
            vals[:, -1, :] = vals[:, 0, :]
        arr[...] = cp.asarray(vals, dtype=arr.dtype)

    # Reference (t*) state.
    fill("u", 0.05, xdup=True)
    fill("v", 0.05, ydup=True)
    fill("w", 0.05)
    s.w[0] = 0.0
    s.w[-1] = 0.0
    fill("thp", 0.5)
    fill("php", 20.0)
    fill("mup", 20.0)
    update_diagnostics(s)                     # consistent p, al, alt
    # Large-step tendencies (constant forcing over the acoustic substeps).
    fill("ru_t", 0.2, xdup=True)
    fill("rv_t", 0.2, ydup=True)
    fill("rw_t", 0.2)
    fill("rth_t", 0.2)
    fill("rph_t", 0.2)
    fill("rmu_t", 1e-3)
    # Acoustic perturbation fields.
    fill("u_pp", 0.2, xdup=True)
    fill("v_pp", 0.2, ydup=True)
    fill("w_pp", 0.2)
    fill("th_pp", 0.2)
    fill("mu_pp", 0.5)
    fill("ph_pp", 1.5e-3)
    s.ph_pp[0] = 0.0                          # fixed surface: phi''(sfc) = 0
    fill("p_pp", 1e-3)
    fill("p_pp_old", 1e-3)
    # Acoustic specific volume al'' (diagnosed by calc_p_pp in a real run;
    # random here).  Amplitude chosen so its d(pb)/dx term over terrain is
    # comparable to the other pressure-gradient terms.
    fill("al_pp", 3e-7)
    if msf_amp > 0.0 or f_amp > 0.0:
        msft = 1.0 + msf_amp * rng.random((ny, nx))
        msfu = 1.0 + msf_amp * rng.random((ny, nx + 1))
        msfv = 1.0 + msf_amp * rng.random((ny + 1, nx))
        msfu[:, -1] = msfu[:, 0]              # periodic duplicates
        msfv[-1, :] = msfv[0, :]
        s.set_map_coriolis(msft=msft, msfu=msfu, msfv=msfv,
                           f=f_amp * rng.random((ny, nx)),
                           e=f_amp * rng.random((ny, nx)))
    return s, cfg


#: Reference/base/coordinate arrays snapshotted by :func:`s_meta`, every
#: device array that stays fixed during the acoustic substeps.
_META_FIELDS = ("u", "v", "w", "thp", "php", "p", "al", "alt",
                "ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t", "mup",
                "thb", "pb", "alb", "phb", "mub2d", "ht",
                "dnw", "rdnw", "dn", "rdn", "fnp", "fnm", "znu", "znw",
                "c1h", "c2h", "c1f", "c2f",
                "msft", "msfu", "msfv", "f", "e")


def s_meta(s):
    """Float64 NumPy snapshot of the non-``_pp`` arrays of a ``DomainState``.

    These are the RK stage reference fields, base-state profiles (1-D flat
    columns or 3-D over terrain; dry mass always the 2-D ``mub2d``), and
    coordinate/hybrid arrays: everything the acoustic mirrors need beyond
    the evolving ``_pp`` perturbations (which the caller captures
    separately, before the substep).  Scalars ``cf1``/``cf2``/``cf3``
    included.
    """
    import cupy as cp

    m = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
         for n in _META_FIELDS}
    # Optional WRF ``moist`` mass fields are fixed for every acoustic
    # substep in an RK stage.  Number moments deliberately stay out: WRF's
    # calc_cq traverses the Registry moist array, not its scalar array.
    for n in _CQ_NSSL_MASS_SPECIES:
        value = getattr(s, n, None)
        if value is not None:
            m[n] = cp.asnumpy(value).astype(np.float64)
    for n in ("cf1", "cf2", "cf3", "cfn", "cfn1"):
        m[n] = float(getattr(s, n))
    return m


def _uv_pgrad(pp, meta, pe, ax3, n, rd, *, top_lid):
    """Acoustic horizontal pressure-gradient term (WRF ``dpxy``).

    Evaluated on the ``n`` periodic faces normal to 3-D axis ``ax3``
    (2 = x faces for u, 1 = y faces for v) with grid spacing ``1/rd``;
    ``pe`` is the divergence-damped p''.  General hybrid/terrain form of
    WRF ``advance_uv`` (map factors 1, dry cqu = cqv = 1): the coupled face
    mass is ``c1h*<mu>_face + c2h``, the mu'' term is c1h-weighted, and the
    acoustic specific volume al'' rides the base-pressure gradient
    (``(al''_A + al''_B) * d(pb)``, nonzero on eta surfaces over terrain).
    A legacy ``pp`` snapshot without ``al_pp`` is accepted only for a
    horizontally uniform base pressure (1-D ``pb``), where that term
    vanishes identically.
    """
    A = np.arange(n)
    B = (A - 1) % n
    ax2 = ax3 - 1                             # matching axis of 2-D fields
    t3 = lambda f, idx: np.take(f, idx, axis=ax3)
    t2 = lambda f, idx: np.take(f, idx, axis=ax2)

    nz = pe.shape[0]
    ph_pp, php, alt = pp["ph_pp"], meta["php"], meta["alt"]
    c1h = meta["c1h"][:, None, None]
    c2h = meta["c2h"][:, None, None]
    mu_full = meta["mub2d"] + meta["mup"]
    muf = 0.5 * (t2(mu_full, A) + t2(mu_full, B))
    # (c1h*mu_face + c2h) * (d(phi'')/dx + alpha_t* d(pe)/dx
    #                        + alpha'' d(pb)/dx), phi'' averaged to half
    # levels via the sum of the two full-level differences (WRF form).
    dph = ((t3(ph_pp[1:], A) - t3(ph_pp[1:], B))
           + (t3(ph_pp[:-1], A) - t3(ph_pp[:-1], B)))
    dpe = t3(pe, A) - t3(pe, B)
    al_pp = pp.get("al_pp")
    if al_pp is None:
        if np.asarray(meta["pb"]).ndim != 1:
            raise KeyError("al_pp required in the pp snapshot over terrain "
                           "(base pressure varies on eta surfaces)")
        albpb = 0.0
    else:
        pb = np.broadcast_to(_prof3(meta["pb"]), alt.shape)
        albpb = (t3(al_pp, A) + t3(al_pp, B)) * (t3(pb, A) - t3(pb, B))
    dpxy = 0.5 * rd * (c1h * muf[None] + c2h) * (
        dph + (t3(alt, A) + t3(alt, B)) * dpe + albpb)
    # d(phi'_t*)/dx * (d(pe)/d(eta) - c1h*mu''), with pe at full levels
    # (dpn): surface value extrapolated with cf1..cf3; WRF also
    # extrapolates the rigid-lid top, while an open top remains zero.
    psum = t3(pe, A) + t3(pe, B)
    dpn = np.zeros((nz + 1,) + psum.shape[1:])
    dpn[0] = 0.5 * (meta["cf1"] * psum[0] + meta["cf2"] * psum[1]
                    + meta["cf3"] * psum[2])
    fnm = meta["fnm"][1:nz, None, None]
    fnp = meta["fnp"][1:nz, None, None]
    dpn[1:nz] = 0.5 * (fnm * psum[1:nz] + fnp * psum[0:nz - 1])
    if top_lid:
        dpn[nz] = 0.5 * (
            meta["cf1"] * psum[nz - 1]
            + meta["cf2"] * psum[nz - 2]
            + meta["cf3"] * psum[nz - 3])
    # Term 4's coefficient is the FULL t* half-level geopotential -- WRF
    # calc_php's 0.5*(phb(k)+phb(k+1)+ph(k)+ph(k+1))
    # (module_big_step_utilities_em.F:1261; consumed at
    # module_small_step_em.F:861-862/935-936) -- so the base phb joins the
    # perturbation php over terrain (3-D phb).  With a flat 1-D base the
    # phb face difference is identically zero and the perturbation-only
    # expression is kept, matching the kernel's base3d branch bitwise.
    php_h = 0.5 * (php[:-1] + php[1:])        # phi'_t* at half levels
    dphp = t3(php_h, A) - t3(php_h, B)
    phb = np.asarray(meta["phb"])
    if phb.ndim == 3:
        phb_h = 0.5 * (phb[:-1] + phb[1:])
        dphp = dphp + (t3(phb_h, A) - t3(phb_h, B))
    dmu = 0.5 * (t2(pp["mu_pp"], A) + t2(pp["mu_pp"], B))
    rdnw = meta["rdnw"][:, None, None]
    dpxy += rd * dphp * (rdnw * (dpn[1:] - dpn[:-1]) - c1h * dmu[None])
    return dpxy


def _specified_frame_mask(shape, spec_zone):
    """Boolean mask of a WRF specified frame on any horizontal staggering."""
    ny, nx = shape[-2:]
    j, i = np.ogrid[:ny, :nx]
    return ((i < spec_zone) | (i >= nx - spec_zone)
            | (j < spec_zone) | (j >= ny - spec_zone))


def np_spec_bdyupdate(field, tendency, dtau, spec_zone):
    """WRF ``spec_bdyupdate(field, tendency, dts_rk)`` on the frame.

    This is the nested ELSE branch at ``solve_em.F:1602-1611``.  The
    interior is untouched; Y sides own corners through the same frame mask
    used by the explicit u/v/mu/theta boundary updates.
    """
    out = np.asarray(field).copy()
    tendency = np.asarray(tendency)
    if out.shape != tendency.shape or out.ndim != 3:
        raise ValueError("field and tendency must be like-shaped 3-D arrays")
    mask = _specified_frame_mask(out.shape[-2:], spec_zone)
    out[:, mask] += dtau * tendency[:, mask]
    return out


def _boundary_forced_cfg(cfg):
    """WRF ``specified_bdy`` mirror: specified OR nested."""
    return (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False))


_CQ_MASS_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
_CQ_NSSL_MASS_SPECIES = _CQ_MASS_SPECIES + ("qh",)
_CQ_MASS_SPECIES_BY_MP = {
    0: _CQ_MASS_SPECIES[:1],
    1: _CQ_MASS_SPECIES[:3],
    6: _CQ_MASS_SPECIES,
    8: _CQ_MASS_SPECIES,
    # Milbrandt-Yau (Registry.EM_COMMON:3025) binds moist:qv,qc,qr,qi,qs,
    # qg,qh -- hail in the MOIST package exactly like NSSL, all six number
    # moments in ``scalar`` -- so its calc_cq is the same seven-mass sum,
    # matching the production arm's n_mass=7 for (9, 18)
    # (woof/core/acoustic.py prepare_moist_cq).  The row was missing while
    # 9 was an accepted selector: this mirror refused a scheme the device
    # path integrates (found by the widened ACCEPTED_MP_PHYSICS census,
    # tests/test_mp28_runtime_reachability.py, 2026-08-30).
    9: _CQ_NSSL_MASS_SPECIES,
    10: _CQ_MASS_SPECIES,
    # WDM6 (Registry.EM_COMMON:3031) carries WSM6's six moist masses with
    # qnn/qnc/qnr in ``scalar``, so its row is byte-for-byte mp=6's --
    # missing for the same reason and found by the same census.
    14: _CQ_MASS_SPECIES[:5],
    16: _CQ_MASS_SPECIES,
    26: _CQ_NSSL_MASS_SPECIES,
    27: _CQ_NSSL_MASS_SPECIES,
    18: _CQ_NSSL_MASS_SPECIES,
    # Thompson aerosol-aware (Registry.EM_COMMON:3036) adds qnc/qnwfa/qnifa
    # to the ``scalar`` package only, so its ``moist`` package -- and thus
    # its calc_cq sum -- is byte-for-byte mp=8's.
    28: _CQ_MASS_SPECIES,
    # P3 one-category (Registry.EM_COMMON:3038):
    #     moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib
    # -- FOUR masses, and the first ported package with qi and no qs/qg.
    # WRF's calc_cq sums the ``moist`` array alone
    # (module_big_step_utilities_em.F:822-830 loops
    # PARAM_FIRST_SCALAR..n_moist over it), so the row is the qv/qc/qr/qi
    # prefix and nothing else.
    #
    # The rime pair stays out for two independent reasons, either alone
    # sufficient.  It is Registry ``scalar``, reaching the scheme from the
    # scalar array (solve_em.F:3849 ``QIR_CURR=scalar(...,P_QIR)``) exactly
    # like every number moment.  And qir is a COMPONENT of qi rather than
    # extra mass -- module_mp_p3.F:2666/:5099 form the rime mass FRACTION
    # as ``qirim/qitot`` -- so adding it would double count the rimed ice,
    # while qib is a rime VOLUME in m3 kg-1 (module_mp_p3.F:1946) and is
    # not a mass at all.
    #
    # This is the reference arm for the device's presence-keyed P3 branch
    # (woof/core/acoustic.py prepare_moist_cq), which keeps the six-mass
    # kernel mode and hands the absent qs/qg slots a zero plane: summing
    # four species here is the same number, and
    # tests/test_acoustic_npref.py gates the two against each other.
    50: _CQ_MASS_SPECIES[:4],
    # mp=53 (p3_1cat_3mom, Registry.EM_COMMON:3041) declares the same
    # moist:qv,qc,qr,qi package; qnc and qzi are scalars, outside calc_cq.
    53: _CQ_MASS_SPECIES[:4],
}


def np_calc_cq(moisture, mp_physics):
    """WRF ``calc_cq`` factors for the configured Registry moist package.

    ``mp_physics`` selects qv only for 0, qv/qc/qr for Kessler (1), all six
    mass species through qg for WSM6 (6), Thompson (8), Morrison (10),
    WDM6 (16) or aerosol-aware Thompson (28), those six plus hail mass qh
    for Milbrandt-Yau (9) and NSSL option 18, and qv/qc/qr/qi -- one ice
    category, no snow, no graupel -- for P3 (50).
    Number moments deliberately do not participate: WRF registers them as
    ``scalar``, outside
    the ``moist`` array traversed by ``calc_cq``; P3's rime mass qir and
    rime volume qib are ``scalar`` too, and qir is a component of qi
    besides.  Surface/top cqw entries are
    one; only interior w levels are consumed after ``pg_buoy_w`` transforms
    WRF's stored half-sum into ``1/(1+qtot_w)``.
    """
    try:
        names = _CQ_MASS_SPECIES_BY_MP[int(mp_physics)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"unsupported mp_physics={mp_physics!r} for np_calc_cq") from exc
    missing = [name for name in names if moisture.get(name) is None]
    if missing:
        raise ValueError(
            f"np_calc_cq mp_physics={mp_physics} requires {missing}")
    active = [np.asarray(moisture[name], dtype=np.float64) for name in names]
    if active[0].ndim != 3 or any(a.shape != active[0].shape for a in active):
        raise ValueError("active cq moisture species must be like-shaped 3-D arrays")
    qtot = np.zeros_like(active[0])
    for species in active:                      # WRF PARAM_FIRST_SCALAR order
        qtot = qtot + species
    nz, ny, nx = qtot.shape
    u_core = 1.0 / (1.0 + 0.5 * (qtot + np.roll(qtot, 1, axis=2)))
    v_core = 1.0 / (1.0 + 0.5 * (qtot + np.roll(qtot, 1, axis=1)))
    cqu = np.concatenate([u_core, u_core[:, :, :1]], axis=2)
    cqv = np.concatenate([v_core, v_core[:, :1, :]], axis=1)
    cqw = np.ones((nz + 1, ny, nx), dtype=np.float64)
    cqw[1:nz] = 1.0 / (1.0 + 0.5 * (qtot[1:] + qtot[:-1]))
    return cqu, cqv, cqw


def _np_stage_cq(meta, cfg):
    """Stage-fixed cq tuple, or ``None`` on the exact dry/attribution path."""
    if (not getattr(cfg, "moist_cq", True)
            or meta.get("qv") is None):
        return None
    return np_calc_cq(meta, mp_physics=cfg.mp_physics)


def _np_zero_grad_specified(field, spec_zone):
    """NumPy WRF ``zero_grad_bdy`` with Y-side corner ownership."""
    out = np.asarray(field, dtype=np.float64).copy()
    ny, nx = out.shape[-2:]
    source_i = np.clip(np.arange(nx), spec_zone, nx - 1 - spec_zone)
    for d in range(spec_zone):
        lo, hi = d, ny - 1 - d
        cols = source_i[d:nx - d]
        out[..., lo, d:nx - d] = out[..., spec_zone, cols]
        out[..., hi, d:nx - d] = out[..., ny - 1 - spec_zone, cols]
        if hi > lo + 1:
            rows = np.arange(d + 1, ny - d - 1)
            inner_j = np.clip(rows, spec_zone, ny - 1 - spec_zone)
            out[..., rows, lo] = out[..., inner_j, spec_zone]
            out[..., rows, nx - 1 - d] = (
                out[..., inner_j, nx - 1 - spec_zone])
    return out


def np_advance_uv(pp, meta, cfg, dtau, first=False):
    """Mirror of ``advance_uv`` (woof/core/kernels/acoustic.cu).

    Forward step of the coupled perturbation momenta u''/v'' (ARW Tech Note
    eqns 3.7-3.8; WRF ``advance_uv``): large-step tendency as constant
    forcing minus the horizontal pressure gradient evaluated from the
    damped p'' (``p_pp + smdiv*(p_pp - p_pp_old)``; undamped when
    ``first``).  ``pp`` holds the float64 ``_pp`` fields captured before
    the substep, ``meta`` the :func:`s_meta` snapshot.  Returns updated
    ``(u_pp, v_pp)`` with the periodic faces duplicated.

    Open lateral boundaries (Task 9, ``cfg.open_x``/``open_y``): the
    boundary-normal momentum at the two boundary faces skips the pressure
    gradient and advances by the large-step tendency alone (WRF
    ``advance_uv``'s ``i_start_up``/``j_start_vp`` exclusions,
    module_small_step_em.F), that tendency is the radiative term installed
    by :func:`np_open_u_radiative`/dycore, so the boundary faces integrate
    the gravity-wave radiation equation over the acoustic loop.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    smdiv = 0.0 if first else cfg.smdiv
    pe = pp["p_pp"] + smdiv * (pp["p_pp"] - pp["p_pp_old"])
    du = _uv_pgrad(pp, meta, pe, ax3=2, n=nx, rd=1.0 / cfg.dx,
                   top_lid=cfg.top_lid)
    dv = _uv_pgrad(pp, meta, pe, ax3=1, n=ny, rd=1.0 / cfg.dy,
                   top_lid=cfg.top_lid)
    cq = _np_stage_cq(meta, cfg)
    if cq is None:
        u_core = (pp["u_pp"][:, :, :nx]
                  + dtau * (meta["ru_t"][:, :, :nx] - du))
        v_core = (pp["v_pp"][:, :ny, :]
                  + dtau * (meta["rv_t"][:, :ny, :] - dv))
    else:
        cqu, cqv, _ = cq
        u_core = (pp["u_pp"][:, :, :nx]
                  + dtau * (meta["ru_t"][:, :, :nx]
                             - cqu[:, :, :nx] * du))
        v_core = (pp["v_pp"][:, :ny, :]
                  + dtau * (meta["rv_t"][:, :ny, :]
                             - cqv[:, :ny, :] * dv))
    u_new = np.concatenate([u_core, u_core[:, :, :1]], axis=2)
    v_new = np.concatenate([v_core, v_core[:, :1, :]], axis=1)
    if getattr(cfg, "open_x", False):
        for f in (0, nx):
            u_new[:, :, f] = pp["u_pp"][:, :, f] + dtau * meta["ru_t"][:, :, f]
    if getattr(cfg, "open_y", False):
        for f in (0, ny):
            v_new[:, f, :] = pp["v_pp"][:, f, :] + dtau * meta["rv_t"][:, f, :]
    if _boundary_forced_cfg(cfg):
        umask = _specified_frame_mask(u_new.shape, cfg.spec_zone)
        vmask = _specified_frame_mask(v_new.shape, cfg.spec_zone)
        u_new[:, umask] = (pp["u_pp"][:, umask]
                           + dtau * meta["ru_t"][:, umask])
        v_new[:, vmask] = (pp["v_pp"][:, vmask]
                           + dtau * meta["rv_t"][:, vmask])
    return u_new, v_new


def np_advance_mu_th(pp, meta, cfg, dtau):
    """Mirror of ``advance_mu_th`` (woof/core/kernels/acoustic.cu).

    ARW Tech Note eqns 3.9-3.10 / WRF ``advance_mu_t``: ``pp["u_pp"]`` and
    ``pp["v_pp"]`` must already hold the post-``advance_uv`` momenta.  The
    mu'' forward step integrates the column divergence of the *total*
    coupled momentum (perturbation + reference (c1h*mu_face + c2h)*u_t*,
    the fixed large-step part of R_mu); Omega'' (``ww_pp``) integrates the
    perturbation-only divergence upward from Omega''(surface) = 0 with the
    c1h-weighted column-mass tendency (WRF ``ww(k) = ww(k-1) -
    dnw(k-1)*(c1h(k-1)*(dmdt + mu_tend) + dvdxi)``); the coupled
    (mu*theta)'' forward step adds the ``rth_t`` forcing and the divergence
    of the perturbation momenta advecting theta_t*.  Returns
    ``(mu_pp, ww_pp, th_pp)`` after the step, float64.

    Open lateral boundaries (``cfg.open_x``/``open_y``): the periodic
    wraps become WRF's zero-gradient ghost reads (share/module_bc.F
    ``set_physical_bc``) -- the boundary-face reference mass is the
    boundary cell's own (WRF ``muu``/``muv`` under the ghost copy) and the
    theta_t* advection clamps its cross-boundary neighbour to the boundary
    value.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    open_x = getattr(cfg, "open_x", False)
    open_y = getattr(cfg, "open_y", False)
    u_pp, v_pp = pp["u_pp"], pp["v_pp"]
    c1h = meta["c1h"][:, None, None]
    c2h = meta["c2h"][:, None, None]
    mu_full = meta["mub2d"] + meta["mup"]

    # Reference column mass at the faces (face f between cells f-1 and f),
    # coupled per level with the hybrid increment c1h*mu + c2h.
    muu = 0.5 * (mu_full + np.roll(mu_full, 1, axis=1))
    muu = np.concatenate([muu, muu[:, :1]], axis=1)      # (ny, nx+1)
    muv = 0.5 * (mu_full + np.roll(mu_full, 1, axis=0))
    muv = np.concatenate([muv, muv[:1, :]], axis=0)      # (ny+1, nx)
    if open_x:
        muu[:, 0] = mu_full[:, 0]
        muu[:, nx] = mu_full[:, nx - 1]
    if open_y:
        muv[0, :] = mu_full[0, :]
        muv[ny, :] = mu_full[ny - 1, :]
    # Map factors (Task 3, WRF advance_mu_t): the reference coupled momenta
    # divide by their face msf (u_pp/v_pp already carry theirs), the
    # divergence carries msftx*msfty = msft^2, and the Omega'' recurrence
    # divides by msfty = msft; identity with the default map factors.
    msft = meta["msft"]
    m2 = (msft * msft)[None]
    fu = (c1h * muu[None] + c2h) * meta["u"] / meta["msfu"][None]
    fv = (c1h * muv[None] + c2h) * meta["v"] / meta["msfv"][None]
    div_pp = m2 * (rdx * (u_pp[:, :, 1:] - u_pp[:, :, :-1])
                   + rdy * (v_pp[:, 1:, :] - v_pp[:, :-1, :]))
    div_rf = m2 * (rdx * (fu[:, :, 1:] - fu[:, :, :-1])
                   + rdy * (fv[:, 1:, :] - fv[:, :-1, :]))
    dnw = meta["dnw"][:, None, None]
    dmdt_pp = np.sum(dnw * div_pp, axis=0)
    dmdt = dmdt_pp + np.sum(dnw * div_rf, axis=0)
    mu_new = pp["mu_pp"] + dtau * (dmdt + meta["rmu_t"])

    # Omega'' at w levels; zero at surface and top by construction
    # (sum_k dnw[k] = -1 makes the column integral close, rmu_t aside).
    ww_new = np.zeros((nz + 1, ny, nx))
    layer = (c1h * (dmdt_pp[None] + meta["rmu_t"][None])
             + div_pp) / msft[None]
    ww_new[1:nz] = -np.cumsum(dnw[:nz - 1] * layer[:nz - 1], axis=0)

    t_ref = _prof3(meta["thb"]) + meta["thp"]
    wdtn = np.zeros((nz + 1, ny, nx))
    fnm = meta["fnm"][1:nz, None, None]
    fnp = meta["fnp"][1:nz, None, None]
    wdtn[1:nz] = ww_new[1:nz] * (fnm * t_ref[1:nz] + fnp * t_ref[0:nz - 1])
    tip = np.roll(t_ref, -1, axis=2)
    tim = np.roll(t_ref, 1, axis=2)
    if open_x:                             # zero-gradient theta ghosts
        tip[:, :, -1] = t_ref[:, :, -1]
        tim[:, :, 0] = t_ref[:, :, 0]
    hx = 0.5 * rdx * (u_pp[:, :, 1:] * (tip + t_ref)
                      - u_pp[:, :, :-1] * (t_ref + tim))
    tjp = np.roll(t_ref, -1, axis=1)
    tjm = np.roll(t_ref, 1, axis=1)
    if open_y:
        tjp[:, -1, :] = t_ref[:, -1, :]
        tjm[:, 0, :] = t_ref[:, 0, :]
    hy = 0.5 * rdy * (v_pp[:, 1:, :] * (tjp + t_ref)
                      - v_pp[:, :-1, :] * (t_ref + tjm))
    rdnw = meta["rdnw"][:, None, None]
    # WRF: t += msfty*dts*ft - dts*msfty*(msftx*(hx+hy) + rdnw*d(wdtn)).
    th_new = pp["th_pp"] + dtau * msft[None] * (
        meta["rth_t"]
        - (msft[None] * (hx + hy) + rdnw * (wdtn[1:] - wdtn[:-1])))
    if _boundary_forced_cfg(cfg):
        mask = _specified_frame_mask(mu_new.shape, cfg.spec_zone)
        mu_new[mask] = (pp["mu_pp"][mask]
                        + dtau * meta["rmu_t"][mask])
        # WRF's boundary T is theta-300 coupled.  This mirror carries the
        # kernel's algebraically equivalent full-theta th_pp, so X=mu*theta
        # adds rth_t + 300*c1h*rmu_t on the frame (still with no msft).
        th_new[:, mask] = (pp["th_pp"][:, mask]
                           + dtau * (meta["rth_t"][:, mask]
                                     + 300.0 * c1h[:, 0, 0, None]
                                     * meta["rmu_t"][mask][None]))
        ww_new[:, mask] = pp["ww_pp"][:, mask]
    return mu_new, ww_new, th_new


def np_advance_w_phi(pp, new, meta, cfg, dtau):
    """Mirror of ``calc_coefs`` + ``advance_w_phi`` (acoustic.cu).

    Vertically implicit (Crank-Nicolson off-centered by ``cfg.epssm``)
    coupled w''/phi'' step: ARW Tech Note eqns 3.11-3.14; WRF
    ``module_small_step_em.F`` subroutines ``calc_coef_w`` / ``advance_w``
    in the general hybrid/terrain form (dry cqw = 1, WRF's open top by
    default with the historical rigid lid selectable by ``cfg.top_lid``;
    map factors from ``meta["msft"]`` per WRF
    advance_w, identity by default): every column-mass coupling carries the hybrid
    increments ``c1h*mu + c2h`` (half levels) / ``c1f*mu + c2f`` (full
    levels), the buoyancy's mu'' average is c1f-weighted, and the lower
    boundary is the kinematic terrain condition ``w''(0) = <(u''.grad ht)>``
    (cf1..cf3-weighted lowest half levels; identically zero when flat).
    Includes the WRF ``damp_opt=3`` implicit w-only Rayleigh damper
    (Klemp-Dudhia-Hassiotis 2008): after the tridiagonal solve,
    ``w'' <- (w'' - dampwt*(c1f*mut+c2f)*w_t*)/(1+dampwt)`` with
    ``dampwt = dtau*dampcoef*sin^2(pi/2*(z - (ztop_col - zdamp))/zdamp)``
    and per-column heights from the t* geopotential.
    ``pp`` holds the float64 ``_pp`` fields captured *before* the substep,
    ``new`` the post-``advance_mu_th`` ``mu_pp``/``ww_pp``/``th_pp`` plus
    the post-``advance_uv`` ``u_pp``/``v_pp`` (surface BC), and ``meta``
    the :func:`s_meta` snapshot.  Under ``top_lid=True``, w''(nz) = 0 and
    phi''(nz) = 0 exactly as in the legacy mirror; the default open branch
    retains WRF's top rhs, forcing, coupling, and phi'' update.  Returns
    updated ``(w_pp, ph_pp)`` float64.
    """
    nz = cfg.nz
    eps = cfg.epssm
    c1h = meta["c1h"][:, None, None]
    c2h = meta["c2h"][:, None, None]
    c1f = meta["c1f"][:, None, None]
    c2f = meta["c2f"][:, None, None]
    mut = meta["mub2d"] + meta["mup"]                      # (ny, nx), t* mass
    muts = mut + new["mu_pp"]                              # after the substep
    chm_t = c1h * mut[None] + c2h                # (nz, ny, nx) half-level t*
    chm_ts = c1h * muts[None] + c2h
    cfm_t = c1f * mut[None] + c2f                # (nz+1, ny, nx) full-level
    cfm_ts = c1f * muts[None] + c2f
    muave = 0.5 * ((1.0 + eps) * new["mu_pp"] + (1.0 - eps) * pp["mu_pp"])
    th_ref = _prof3(meta["thb"]) + meta["thp"]             # theta_t*
    # Off-centered (mu*theta)'' average, normalized as WRF t_2ave (with full
    # theta the +c1h*muave*t0 numerator shift cancels identically).
    t2ave = (0.5 * ((1.0 + eps) * new["th_pp"] + (1.0 - eps) * pp["th_pp"])
             / (chm_ts * th_ref))
    c2a = c.GAMMA * meta["p"] / meta["alt"]                # gamma*p/alpha, t*
    ph_ref = _prof3(meta["phb"]) + meta["php"]             # full geopotential
    cq = _np_stage_cq(meta, cfg)
    cqw = (np.ones_like(pp["ph_pp"]) if cq is None else cq[2])

    rdnw = meta["rdnw"][:, None, None]
    rdnk = meta["rdn"][1:nz, None, None]                   # interior w levels

    # RHS of the phi'' equation: large-step forcing, the explicit half of the
    # g*w term, minus Omega''*d(phi_t*)/d(eta) averaged to full levels.
    # Map factor at mass points (Task 3, WRF advance_w msfty; identity by
    # default).
    msft = meta["msft"][None]

    rhs = np.zeros_like(pp["ph_pp"])
    rhs[1:] = dtau * (meta["rph_t"][1:]
                      + 0.5 * c.G * (1.0 - eps) * pp["w_pp"][1:])
    ww = new["ww_pp"]
    wdwn = 0.5 * (ww[1:] + ww[:-1]) * rdnw * (ph_ref[1:] - ph_ref[:-1])
    fnm = meta["fnm"][1:nz, None, None]
    fnp = meta["fnp"][1:nz, None, None]
    rhs[1:nz] -= dtau * (fnm * wdwn[1:] + fnp * wdwn[:-1])
    rhs[1:] = pp["ph_pp"][1:] + msft * rhs[1:] / cfm_t[1:]
    if cfg.top_lid:
        rhs[nz] = 0.0                                      # rigid lid only

    # Forcing rows of the tridiagonal system for the new w''.  The lower
    # boundary is the kinematic terrain BC on the coupled w'' (WRF
    # advance_w), built from the post-advance_uv momenta.
    w = pp["w_pp"].copy()
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    ht = meta["ht"]
    uc = (meta["cf1"] * new["u_pp"][0] + meta["cf2"] * new["u_pp"][1]
          + meta["cf3"] * new["u_pp"][2])                  # (ny, nx+1)
    vc = (meta["cf1"] * new["v_pp"][0] + meta["cf2"] * new["v_pp"][1]
          + meta["cf3"] * new["v_pp"][2])                  # (ny+1, nx)
    w[0] = meta["msft"] * (
        0.5 * rdy * ((np.roll(ht, -1, 0) - ht) * vc[1:, :]
                     + (ht - np.roll(ht, 1, 0)) * vc[:-1, :])
        + 0.5 * rdx * ((np.roll(ht, -1, 1) - ht) * uc[:, 1:]
                       + (ht - np.roll(ht, 1, 1)) * uc[:, :-1]))
    ph_old = pp["ph_pp"]
    c2k, c2km = c2a[1:nz], c2a[:nz - 1]
    rdnwk, rdnwkm = rdnw[1:nz], rdnw[:nz - 1]
    dph_up = ((1.0 + eps) * (rhs[2:] - rhs[1:nz])
              + (1.0 - eps) * (ph_old[2:] - ph_old[1:nz]))
    dph_dn = ((1.0 + eps) * (rhs[1:nz] - rhs[:nz - 1])
              + (1.0 - eps) * (ph_old[1:nz] - ph_old[:nz - 1]))
    # WRF advance_w: the implicit pressure-gradient and buoyancy chunks
    # carry msft_inv = 1/msfty; the rw_t forcing is already (1/my)-coupled.
    w[1:nz] += (dtau * meta["rw_t"][1:nz]
                + (1.0 / msft) * cqw[1:nz]
                * (0.5 * dtau * c.G * rdnk
                   * (c2k * rdnwk / chm_t[1:nz] * dph_up
                      - c2km * rdnwkm / chm_t[:nz - 1] * dph_dn))
                + dtau * c.G * (1.0 / msft)
                * (rdnk * (c2k * meta["alt"][1:nz] * t2ave[1:nz]
                           - c2km * meta["alt"][:nz - 1]
                           * t2ave[:nz - 1])
                   - c1f[1:nz] * muave[None]))
    if cfg.top_lid:
        w[nz] = 0.0                                        # legacy rigid lid
    else:
        # WRF v4.6.1 module_small_step_em.F:1420-1431.  Unlike the
        # interior rows, the open top has its one-sided pressure/buoyancy
        # forcing and no cqw multiplier.
        dph_top = ((1.0 + eps) * (rhs[nz] - rhs[nz - 1])
                   + (1.0 - eps) * (ph_old[nz] - ph_old[nz - 1]))
        w[nz] += (dtau * meta["rw_t"][nz]
                  + (1.0 / msft[0])
                  * (-dtau * c.G * c2a[nz - 1] * rdnw[nz - 1] ** 2
                     / chm_t[nz - 1] * dph_top
                     - dtau * c.G
                     * (2.0 * rdnw[nz - 1] * c2a[nz - 1]
                        * meta["alt"][nz - 1] * t2ave[nz - 1]
                        + c1f[nz] * muave)))

    # Tridiagonal factors (WRF calc_coef_w) and Thomas solve.
    # Row k couples w[k-1], w[k], w[k+1]; each c2a is divided by its own
    # half-level (c1h*mut + c2h) and the (c1f*mut + c2f) of the full level
    # whose phi'' update carries the coupled w.
    cof = (0.5 * dtau * c.G * (1.0 + eps)) ** 2
    a = np.zeros_like(w)
    alpha = np.zeros_like(w)
    gam = np.zeros_like(w)
    a[2:nz] = (-cqw[2:nz] * cof * meta["rdn"][2:nz, None, None]
               * rdnw[1:nz - 1] * c2a[1:nz - 1]
               / (chm_t[1:nz - 1] * cfm_t[1:nz - 1]))
    if not cfg.top_lid:
        a[nz] = (-2.0 * cof * rdnw[nz - 1] ** 2 * c2a[nz - 1]
                 / (chm_t[nz - 1] * cfm_t[nz - 1]))
    b_int = 1.0 + cqw[1:nz] * cof * rdnk * (
        rdnwk * c2k / (chm_t[1:nz] * cfm_t[1:nz])
        + rdnwkm * c2km / (chm_t[:nz - 1] * cfm_t[1:nz]))
    c_int = (-cqw[1:nz] * cof * rdnk * rdnwk * c2k
             / (chm_t[1:nz] * cfm_t[2:nz + 1]))
    for k in range(1, nz):
        alpha[k] = 1.0 / (b_int[k - 1] - a[k] * gam[k - 1])
        gam[k] = c_int[k - 1] * alpha[k]
    b_top = (1.0 + 2.0 * cof * meta["rdnw"][nz - 1] ** 2 * c2a[nz - 1]
             / (chm_t[nz - 1] * cfm_t[nz]))
    alpha[nz] = 1.0 / (b_top - a[nz] * gam[nz - 1])
    for k in range(1, nz + 1):
        w[k] = (w[k] - a[k] * w[k - 1]) * alpha[k]
    for k in range(nz - 1, 0, -1):
        w[k] = w[k] - gam[k] * w[k + 1]

    # WRF damp_opt=3 (KDH 2008): implicit w-only Rayleigh damping of the
    # solved w'' against the coupled reference w_t* (advance_w); per-column
    # heights from the t* geopotential, layer depth zdamp below the top.
    if cfg.damp_opt == 3 and cfg.dampcoef > 0.0:
        zf = (_prof3(meta["phb"]) + meta["php"]) / c.G     # (nz+1, ny, nx)
        hbot = zf[nz] - cfg.zdamp
        arg = np.clip((zf - hbot[None]) / cfg.zdamp, 0.0, None) * (0.5 * np.pi)
        dampwt = np.where(zf >= hbot[None],
                          dtau * cfg.dampcoef * np.sin(arg) ** 2, 0.0)
        w[1:] = ((w[1:] - dampwt[1:] * cfm_t[1:] * meta["w"][1:])
                 / (1.0 + dampwt[1:]))

    # phi'' forward update from the implicit half of the g*w term
    # (tech note eq 3.11; WRF: + msfty*0.5*dts*g*(1+epssm)*w/C_f(muts));
    # phi''(surface) is never touched.
    ph = pp["ph_pp"].copy()
    ph[1:] = (rhs[1:]
              + msft * 0.5 * dtau * c.G * (1.0 + eps) * w[1:] / cfm_ts[1:])
    if _boundary_forced_cfg(cfg):
        mask = _specified_frame_mask(muts.shape, cfg.spec_zone)
        mu_old = muts - dtau * meta["rmu_t"]
        ratio = ((c1f * mu_old[None] + c2f)
                 / (c1f * muts[None] + c2f))
        ph_spec = (pp["ph_pp"] * ratio
                   # rph_t is a mu-only scalar boundary tendency: no msft.
                   + dtau * meta["rph_t"] / cfm_ts
                   + meta["php"] * (ratio - 1.0))
        ph[:, mask] = ph_spec[:, mask]
        if getattr(cfg, "specified", False):
            w = _np_zero_grad_specified(w, cfg.spec_zone)
        else:
            # solve_em.F:1602-1611: nested children advance w_2 in the
            # specified zone on every acoustic small step.  advance_w skips
            # that frame, so the operand is the pre-substep w_2, not the
            # mirror's provisional whole-domain implicit result.
            w_spec = np_spec_bdyupdate(
                pp["w_pp"], meta["rw_t"], dtau, cfg.spec_zone)
            w[:, mask] = w_spec[:, mask]
    return w, ph


def np_calc_p_pp(th_pp, ph_pp, mu_pp, meta):
    """Mirror of ``calc_p_pp``: linearized-EOS p''/alpha'' (WRF
    ``calc_p_rho``), general hybrid form.

    With full theta the WRF t0-offset terms cancel:
    ``alpha'' = -(alpha_t*.(c1h*mu'') + d(phi'')/d(eta)) / (c1h*mu_ts+c2h)``
    and ``p'' = c2a*(alpha_t*.((mu.theta)'' - c1h*mu''.theta_t*)
    / ((c1h*mu_ts+c2h).theta_t*) - alpha'')``.  Inputs are the post-substep
    fields; returns ``(p'', alpha'')`` float64: alpha'' feeds the next
    substep's ``advance_uv`` base-pressure-gradient term.
    """
    rdnw = meta["rdnw"][:, None, None]
    c1h = meta["c1h"][:, None, None]
    c2h = meta["c2h"][:, None, None]
    chm_ts = c1h * (meta["mub2d"] + meta["mup"] + mu_pp)[None] + c2h
    th_ref = _prof3(meta["thb"]) + meta["thp"]
    c2a = c.GAMMA * meta["p"] / meta["alt"]
    al = -(meta["alt"] * (c1h * mu_pp[None])
           + rdnw * (ph_pp[1:] - ph_pp[:-1])) / chm_ts
    p = c2a * (meta["alt"] * (th_pp - c1h * mu_pp[None] * th_ref)
               / (chm_ts * th_ref) - al)
    return p, al


def np_acoustic_substep(pp, cfg, dtau, first=False):
    """Float64 mirror of one full :func:`woof.core.acoustic.acoustic_substep`.

    ``pp`` is a :func:`snapshot` taken before the substep (it carries both
    the ``_pp`` fields and the :func:`s_meta` reference state, so it serves
    as the mirrors' ``pp`` and ``meta`` arguments at once).  Returns a dict
    of the updated perturbation fields.
    """
    u_new, v_new = np_advance_uv(pp, pp, cfg, dtau, first)
    mu_new, ww_new, th_new = np_advance_mu_th(
        {**pp, "u_pp": u_new, "v_pp": v_new}, pp, cfg, dtau)
    new = {"mu_pp": mu_new, "ww_pp": ww_new, "th_pp": th_new,
           "u_pp": u_new, "v_pp": v_new}
    w_new, ph_new = np_advance_w_phi(pp, new, pp, cfg, dtau)
    p_new, al_new = np_calc_p_pp(th_new, ph_new, mu_new, pp)
    return {"u_pp": u_new, "v_pp": v_new, "w_pp": w_new, "ph_pp": ph_new,
            "mu_pp": mu_new, "th_pp": th_new, "ww_pp": ww_new,
            "p_pp": p_new, "al_pp": al_new, "p_pp_old": pp["p_pp"]}


#: Evolving acoustic perturbation fields captured by :func:`snapshot`.
_PP_FIELDS = ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp",
              "p_pp", "p_pp_old", "ww_pp", "al_pp")


def snapshot(s):
    """Float64 snapshot of *all* fields the acoustic mirrors consume.

    The ``_pp`` perturbations (which the substep updates in place) plus the
    :func:`s_meta` reference/base/coordinate arrays, in one dict.
    """
    import cupy as cp

    snap = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
            for n in _PP_FIELDS}
    snap.update(s_meta(s))
    return snap


def build_isothermal_rest_state(nx, nz, dx, T, pulse_amp, pulse_x0,
                                pulse_halfwidth):
    """Isothermal (constant temperature ``T``) atmosphere at rest with a
    Gaussian p'' pulse, for acoustic wave-propagation tests.  GPU only.

    The sounding ``theta(z) = T*exp(G*z/(CP*T))`` keeps the temperature,
    and hence the sound speed ``sqrt(GAMMA*RD*T)``, uniform (surface
    pressure P0 makes theta(0) = T).  The pulse is planted in ``th_pp`` such
    that the linearized EOS diagnoses ``p'' = pulse_amp * gaussian(x)`` (Pa)
    at every level; ``p_pp``/``p_pp_old`` are initialized consistently.
    Returns ``(state, cfg, dx)``.
    """
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import DTYPE, init_at_rest

    cfg = RunConfig(nx=nx, ny=1, nz=nz, dx=dx, dy=dx, ztop=8000.0,
                    dt=1.0, run_seconds=0.0)
    coord = make_vertical_coord(nz)
    base = make_base_state(coord,
                           lambda z: T * np.exp(c.G * np.asarray(z, float)
                                                / (c.CP * T)),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_at_rest(cfg, coord, base)
    update_diagnostics(s)                          # p, al, alt at rest
    meta = s_meta(s)

    x = (np.arange(nx) + 0.5) * dx
    gauss = pulse_amp * np.exp(-((x - pulse_x0) / pulse_halfwidth) ** 2)
    # Invert p'' = c2a*alt*th_pp/(mu_t*.theta_t*) (mu'' = phi'' = 0), with
    # c2a*alt = GAMMA*p: th_pp = p''*mu_t*.theta_t*/(GAMMA*p).
    th_ref = meta["thb"][:, None, None] + meta["thp"]
    mut = meta["mub2d"] + meta["mup"]
    th_pp = gauss[None, None, :] * mut[None] * th_ref / (c.GAMMA * meta["p"])
    s.th_pp[...] = cp.asarray(th_pp, dtype=DTYPE)
    p_pp, al_pp = np_calc_p_pp(th_pp, np.zeros((nz + 1, cfg.ny, nx)),
                               np.zeros((cfg.ny, nx)), meta)
    s.p_pp[...] = cp.asarray(p_pp, dtype=DTYPE)
    s.al_pp[...] = cp.asarray(al_pp, dtype=DTYPE)
    s.p_pp_old[...] = s.p_pp
    return s, cfg, dx


def _diff2_spacings(zf, stagger):
    """Inverse vertical spacings for :func:`np_add_diff2`, float64.

    Derived independently of ``woof.core.diffusion._dz_spacings``
    (final-review carry-over T18: the mirror must not transcribe the
    model's spacing construction, or a shared conceptual error would
    cancel in the comparison test).  From first principles: the flux-form
    second derivative needs, for the ``nlev`` levels a field lives on, the
    inverse distances between successive level CENTERS (``rdzf``, the
    nlev-1 interior flux faces) and the inverse thickness of each level's
    control volume (``rdzc``, whose edges are the midpoints between
    centers, for half-level fields those edges are exactly the w levels
    ``zf``).  Half-level fields (nlev = nzf-1) live at the layer midpoints
    of ``zf``; w-staggered fields (``stagger="z"``, nlev = nzf) live on
    ``zf`` itself, with the BC-pinned boundary entries of ``rdzc`` unused
    (zero).
    """
    zf = np.asarray(zf, dtype=np.float64)
    mid = 0.5 * (zf[:-1] + zf[1:])        # midpoints between w levels
    centers, edges = (zf, mid) if stagger == "z" else (mid, zf)
    rdzf = 1.0 / np.diff(centers)
    if stagger == "z":
        rdzc = np.zeros(centers.size)
        rdzc[1:-1] = 1.0 / np.diff(edges)
    else:
        rdzc = 1.0 / np.diff(edges)
    return rdzf, rdzc


def np_add_diff2(f, kh, kv, dx, dy, zf, stagger=""):
    """Mirror of ``add_diff2`` (woof/core/kernels/diffusion.cu).

    Returns the (uncoupled) tendency increment ``kh*(f_xx + f_yy) +
    kv*(f_z)_z``: 2nd-order horizontal Laplacian on coordinate surfaces,
    periodic in x/y, and a physical-space vertical second derivative from
    the per-level base-state dz implied by the full-level heights ``zf``.
    ``stagger`` as in ``launch_add_diff2``: half-level fields get zero-flux
    top/bottom boundaries; ``"z"`` (w-points) gets zero tendency at the
    BC-pinned boundary levels; the redundant staggered column/row of ``"x"``
    / ``"y"`` fields duplicates column/row 0.
    """
    f = np.asarray(f, dtype=np.float64)
    nlev, nys, nxs = f.shape
    nx = nxs - 1 if stagger == "x" else nxs
    ny = nys - 1 if stagger == "y" else nys
    q = f[:, :ny, :nx]                    # periodic degrees of freedom
    lap = kh * ((np.roll(q, -1, 2) - 2.0 * q + np.roll(q, 1, 2)) / dx ** 2
                + (np.roll(q, -1, 1) - 2.0 * q + np.roll(q, 1, 1)) / dy ** 2)
    rdzf, rdzc = _diff2_spacings(zf, stagger)
    flux = np.zeros((nlev + 1, ny, nx))   # face m between levels m-1 and m
    flux[1:nlev] = (q[1:] - q[:-1]) * rdzf[:, None, None]
    lap += kv * (flux[1:] - flux[:-1]) * rdzc[:, None, None]
    if stagger == "z":
        lap[0] = 0.0                      # BC-pinned w levels: no tendency
        lap[-1] = 0.0
    if stagger == "x":
        lap = np.concatenate([lap, lap[:, :, :1]], axis=2)
    if stagger == "y":
        lap = np.concatenate([lap, lap[:, :1, :]], axis=1)
    return lap


def random_field_setup(nz=8, ny=2, nx=12, seed=0):
    """Random mass-point field + spacing kwargs for the diffusion tests.

    Returns ``(f, meta)``: ``f (nz, ny, nx)`` float64 standard-normal
    (O(1) amplitude keeps the FP32-vs-float64 comparison tight) and
    ``meta = {"dx", "dy", "zf"}`` accepted verbatim by both
    ``launch_add_diff2`` and :func:`np_add_diff2`; ``zf (nz+1,)`` is a
    gently stretched column of full-level heights, exercising the
    per-level vertical spacing.
    """
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((nz, ny, nx))
    zf = 6400.0 * np.linspace(0.0, 1.0, nz + 1) ** 1.25
    return f, {"dx": 100.0, "dy": 100.0, "zf": zf}


def np_diff6(f, mut, c1, c2, factor, dt, opt, stagger="",
             open_x=False, open_y=False, phb=None, msfu=None, msfv=None,
             slopeopt=0, thresh=0.10, dx=0.0, dy=0.0):
    """Mirror of ``diff6`` (woof/core/kernels/diff6.cu): WRF 6th-order
    horizontal numerical diffusion, transcribed from v4.6.1
    ``dyn_em/module_big_step_utilities_em.F`` ``sixth_order_diffusion``
    (map factors 1 on the tendency; ``diff_6th_slopeopt`` taper via the
    slope kwargs below), float64, periodic in x/y.

    Face fluxes are Xue (2000) eq. 3, ``dflux_p0 = 10*(f_i - f_{i-1}) -
    5*(f_{i+1} - f_{i-2}) + (f_{i+2} - f_{i-3})`` (face left of point i);
    with ``opt == 2`` (monotonic) any flux whose sign is up-gradient,
    ``dflux*(f_i - f_{i-1}) <= 0``, is zeroed.  The coupled tendency is
    ``coef*(mu_p1*dflux_p1 - mu_p0*dflux_p0)`` per direction with ``coef =
    factor * 0.015625 / (2*dt)`` (the factor/2^6 normalization: a full-dt
    integration removes ``factor`` of a 2-D 2dx checkerboard's amplitude)
    and the hybrid face mass ``c1[k]*<MUT> + c2[k]`` averaged to the flux
    location per WRF: x-fluxes of u live at mass centers, y-fluxes of v
    likewise, the cross fluxes of u/v at corners (4-point MUT average),
    and both fluxes of mass/w-point fields at the cell faces (2-point).
    No grid spacing enters: the per-step damping rate is by construction
    resolution-independent.

    ``stagger`` as in the other mirrors: ``""`` mass points, ``"x"`` u,
    ``"y"`` v (redundant periodic column/row duplicated on output), ``"z"``
    w points (nlev = nz+1; the BC-pinned boundary levels k = 0 and nlev-1
    get zero tendency, matching WRF's k loop kts+1..ktf for 'w').  ``c1``/
    ``c2`` carry nlev entries (c1h/c2h for half-level fields, c1f/c2f for
    w).  Returns the coupled tendency, same shape as ``f``.

    ``open_x``/``open_y`` mirror the application pair ``launch_diff6``
    (with ``bnd_x``/``bnd_y``) + ``dycore._zero_open_strips(width=3)``,
    shared by the open and specified/nested loop bounds: WRF computes
    ids+3..ide-4 on non-staggered axes and ids+3..ide-3 on the boundary-
    normal staggered axis, so the outer 3 entries per non-periodic side
    are zeroed on EVERY axis and stagger, and the outermost computed
    staggered face (u's nx-3 under open_x, v's ny-3 under open_y) is
    computed with WRF's accurate read of the stored true boundary datum
    ``field(ide)``/``field(jde)`` -- u column nx / v row ny -- exactly as
    the Fortran's dflux_p1 (module_big_step_utilities_em.F:6465-6467 x /
    :6547-6549 y; loop bounds :6354-6358/:6381-6385).

    ``slopeopt >= 1`` with a 3-D ``phb`` mirrors WRF's diff_6th_slopeopt
    terrain taper (sixth_order_diffusion,
    module_big_step_utilities_em.F:6487-6501 x / :6569-6583 y): every face
    flux is scaled by ``slopedamp = max(1 - dzmax/(thresh*9.81*dx), 0)``
    with ``dzmax`` the msf-scaled BASE-state (phb) face geopotential jump
    at the field's own level index (msfux scales x slopes, msfvy y
    slopes; the u/v variants take the max over their two adjacent mass
    faces).  ``msfu``/``msfv`` default to identity.
    """
    f = np.asarray(f, dtype=np.float64)
    nlev, nys, nxs = f.shape
    nx = nxs - 1 if stagger == "x" else nxs
    ny = nys - 1 if stagger == "y" else nys
    q = f[:, :ny, :nx]                    # periodic degrees of freedom
    mut = np.broadcast_to(np.asarray(mut, dtype=np.float64), (ny, nx))
    c1 = np.asarray(c1, dtype=np.float64)[:, None, None]
    c2 = np.asarray(c2, dtype=np.float64)[:, None, None]
    coef = factor * 0.015625 / (2.0 * dt)

    slope = (slopeopt >= 1 and phb is not None
             and np.asarray(phb).ndim == 3)
    if slope:
        ph = np.asarray(phb, dtype=np.float64)[:nlev]
        mfu = (np.ones((ny, nx)) if msfu is None
               else np.asarray(msfu, dtype=np.float64)[:, :nx])
        mfv = (np.ones((ny, nx)) if msfv is None
               else np.asarray(msfv, dtype=np.float64)[:ny, :])
        dzfx = np.abs(ph - np.roll(ph, 1, axis=2)) * mfu[None]
        dzfy = np.abs(ph - np.roll(ph, 1, axis=1)) * mfv[None]
        if stagger == "x":                # u: max over the two mass faces
            dzx = np.maximum(dzfx, np.roll(dzfx, 1, axis=2))
            dzy = np.maximum(dzfy, np.roll(dzfy, 1, axis=2))
        elif stagger == "y":              # v: max over rows j and j-1
            dzx = np.maximum(dzfx, np.roll(dzfx, 1, axis=1))
            dzy = np.maximum(dzfy, np.roll(dzfy, 1, axis=1))
        else:                             # mass/w points: the face itself
            dzx, dzy = dzfx, dzfy
        sdx = np.maximum(1.0 - dzx / (thresh * 9.81 * dx), 0.0)
        sdy = np.maximum(1.0 - dzy / (thresh * 9.81 * dy), 0.0)
    else:
        sdx = sdy = np.ones((1, 1, 1))

    def flux_p0(ax):
        """Xue's flux at the face LEFT of each point along axis ``ax``."""
        r = lambda s: np.roll(q, -s, axis=ax)          # r(s)[i] = q[i+s]
        dflux = (10.0 * (q - r(-1)) - 5.0 * (r(1) - r(-2))
                 + (r(2) - r(-3)))
        # Accurate boundary-datum read at the outermost computed staggered
        # face: WRF's dflux_p1 there (this array's face nx-2 / ny-2, the
        # p0 face of point nx-2 / ny-2) reads field(ide)/field(jde) --
        # the stored last column/row, which the periodic core lacks.
        if ax == 2 and stagger == "x" and open_x:
            dflux[:, :, nx - 2] = (
                10.0 * (q[:, :, nx - 2] - q[:, :, nx - 3])
                - 5.0 * (q[:, :, nx - 1] - q[:, :, nx - 4])
                + (f[:, :, nx] - q[:, :, nx - 5]))
        if ax == 1 and stagger == "y" and open_y:
            dflux[:, ny - 2, :] = (
                10.0 * (q[:, ny - 2, :] - q[:, ny - 3, :])
                - 5.0 * (q[:, ny - 1, :] - q[:, ny - 4, :])
                + (f[:, ny, :] - q[:, ny - 5, :]))
        if opt == 2:                                   # monotonic: zero any
            dflux = np.where(dflux * (q - r(-1)) <= 0.0, 0.0, dflux)
        return dflux                                   # up-gradient flux

    m = lambda dj, di: np.roll(mut, (-dj, -di), axis=(0, 1))
    corner = 0.25 * (m(-1, -1) + m(-1, 0) + m(0, -1) + m(0, 0))
    if stagger == "x":                    # u: x at mass centers, y at corners
        wx, wy = m(0, -1), corner
    elif stagger == "y":                  # v: x at corners, y at mass centers
        wx, wy = corner, m(-1, 0)
    else:                                 # mass/w points: both at cell faces
        wx = 0.5 * (m(0, -1) + m(0, 0))
        wy = 0.5 * (m(-1, 0) + m(0, 0))

    out = np.zeros((nlev, ny, nx))
    for ax, w, sd in ((2, wx[None], sdx), (1, wy[None], sdy)):
        dflux = flux_p0(ax)
        out += coef * (np.roll(sd, -1, axis=ax)
                       * (c1 * np.roll(w, -1, axis=ax) + c2)
                       * np.roll(dflux, -1, axis=ax)
                       - sd * (c1 * w + c2) * dflux)
    if stagger == "z":
        out[0] = 0.0                      # BC-pinned w levels (WRF kts+1..ktf)
        out[-1] = 0.0
    if stagger == "x":
        out = np.concatenate([out, out[:, :, :1]], axis=2)
    if stagger == "y":
        out = np.concatenate([out, out[:, :1, :]], axis=1)
    if open_x:                            # WRF non-periodic bounds: 3 per
        out[:, :, :3] = 0.0               # side on every axis and stagger
        out[:, :, -3:] = 0.0
    if open_y:
        out[:, :3, :] = 0.0
        out[:, -3:, :] = 0.0
    return out


def np_smag2d_km(u, v, dx, dy, c_s, prandtl=1.0 / 3.0):
    """Mirror of ``smag2d_km`` (woof/core/kernels/smag2d.cu).

    WRF km_opt=4 (module_diffusion_em.F ``cal_deform_and_div`` +
    ``smag2d_km``, coordinate-surface / flat-metric reduction: zx = zy = 0,
    map factors 1): at mass points,

      D11 = 2*du/dx,  D22 = 2*dv/dy                (mass points)
      D12 = du/dy + dv/dx                          (corner points)
      def2 = 0.25*(D11-D22)^2 + (<D12>_4corners)^2
      K_m  = min((c_s*mlen)^2 * sqrt(def2), 10*mlen),  mlen = sqrt(dx*dy)
      K_h  = K_m / prandtl                         (scalars; prandtl = 1/3
                                                    per share/
                                                    module_model_constants.F)

    ``u (nz, ny, nx+1)`` / ``v (nz, ny+1, nx)`` are the staggered winds
    (periodic core; the duplicate column/row is never read).  Returns
    ``(xkmh, xkhh)``, each ``(nz, ny, nx)`` float64 at mass points.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ny, nx = v.shape[1] - 1, u.shape[2] - 1
    uc, vc = u[:, :, :nx], v[:, :ny, :]
    rdx, rdy = 1.0 / dx, 1.0 / dy
    d11 = 2.0 * rdx * (np.roll(uc, -1, axis=2) - uc)
    d22 = 2.0 * rdy * (np.roll(vc, -1, axis=1) - vc)
    # D12 at the SW corner of mass cell (j, i): u rows j-1/j at face i,
    # v columns i-1/i at face j.
    d12 = (rdy * (uc - np.roll(uc, 1, axis=1))
           + rdx * (vc - np.roll(vc, 1, axis=2)))
    d12m = 0.25 * (d12 + np.roll(d12, -1, axis=2) + np.roll(d12, -1, axis=1)
                   + np.roll(np.roll(d12, -1, axis=1), -1, axis=2))
    def2 = 0.25 * (d11 - d22) ** 2 + d12m ** 2
    mlen = np.sqrt(dx * dy)
    xkmh = np.minimum(c_s * c_s * mlen * mlen * np.sqrt(def2), 10.0 * mlen)
    return xkmh, xkmh / prandtl


def np_smag2d_hd(f, xk, mut, c1, c2, dx, dy, stagger="",
                 open_x=False, open_y=False):
    """Mirror of the ``smag_hd_*`` kernels (woof/core/kernels/smag2d.cu).

    WRF ``horizontal_diffusion`` (module_big_step_utilities_em.F, map
    factors 1): variable-K 2nd-order flux-form mixing on coordinate
    surfaces, periodic in x/y, with the eddy diffusivity ``xk (nz, ny, nx)``
    at mass points averaged to the flux faces exactly as the Fortran and
    the hybrid coupling ``c1[k]*mut + c2[k]`` (c1h/c2h for half-level
    fields, c1f/c2f for ``stagger="z"``).  Exception, transcribed from WRF
    v4.6.1 exactly: the ``stagger="y"`` (v) branch's normal y-direction
    fluxes carry NO mass coupling (module_big_step_utilities_em.F
    horizontal_diffusion 'v', mkrdym/mkrdyp) -- a WRF asymmetry vs the 'u'
    branch's mass-coupled normal fluxes.  ``stagger`` as in
    ``launch_add_diff2``: ``""`` mass points, ``"x"`` u, ``"y"`` v, ``"z"``
    w (BC-pinned boundary levels get no tendency); the redundant staggered
    column/row duplicates column/row 0.  Returns the COUPLED tendency
    increment, same shape as ``f``, float64.

    ``open_x``/``open_y`` mirror the application pair ``launch_smag2d_hd``
    + ``dycore._zero_open_strips(width=1)``: WRF's open loop bounds
    (ids+1 / ide-1|2 per stagger) zero the outer entry of each open axis
    side, and the boundary-normal staggered face nx-1 (u, open_x) / ny-1
    (v, open_y) is computed with WRF's accurate boundary-datum read
    field(i+1) = field(ide) -- the stored last column/row -- instead of
    the periodic wrap (module_big_step_utilities_em.F:2786-2787/2819 for
    'u', 2834-2837/2861 for 'v').
    """
    f = np.asarray(f, dtype=np.float64)
    xk = np.asarray(xk, dtype=np.float64)
    mut = np.asarray(mut, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)[:, None, None]
    c2 = np.asarray(c2, dtype=np.float64)[:, None, None]
    rdx, rdy = 1.0 / dx, 1.0 / dy
    ny, nx = xk.shape[1:]
    chm = c1 * mut[None] + c2                  # coupled mass at mass points

    def cavg(a):                               # mass -> SW-corner 4-average
        return 0.25 * (a + np.roll(a, 1, axis=1) + np.roll(a, 1, axis=2)
                       + np.roll(np.roll(a, 1, axis=1), 1, axis=2))

    if stagger == "x":                         # WRF 'u' branch
        uc = f[:, :, :nx]
        up1 = np.roll(uc, -1, axis=2)
        if open_x:                             # accurate east boundary datum
            up1[:, :, nx - 1] = f[:, :, nx]    # field(i+1) = u(ide)
        hx = chm * xk * rdx * (up1 - uc)
        tx = rdx * (hx - np.roll(hx, 1, axis=2))
        gy = (cavg(chm) * cavg(xk) * rdy
              * (uc - np.roll(uc, 1, axis=1)))
        ty = rdy * (np.roll(gy, -1, axis=1) - gy)
        lap = tx + ty
        out = np.concatenate([lap, lap[:, :, :1]], axis=2)
    elif stagger == "y":                       # WRF 'v' branch
        vc = f[:, :ny, :]
        vp1 = np.roll(vc, -1, axis=1)
        if open_y:                             # accurate north boundary datum
            vp1[:, ny - 1, :] = f[:, ny, :]    # field(j+1) = v(jde)
        # WRF quirk (see docstring): the v normal (y) fluxes have no chm.
        hy = xk * rdy * (vp1 - vc)
        ty = rdy * (hy - np.roll(hy, 1, axis=1))
        gx = (cavg(chm) * cavg(xk) * rdx
              * (vc - np.roll(vc, 1, axis=2)))
        tx = rdx * (np.roll(gx, -1, axis=2) - gx)
        lap = tx + ty
        out = np.concatenate([lap, lap[:, :1, :]], axis=1)
    elif stagger == "z":                       # WRF 'w' branch
        nzf = f.shape[0]                       # nz + 1 w levels
        wc = f[1:nzf - 1]                      # interior w levels 1..nz-1
        chw = chm[1:nzf - 1]                   # c1f/c2f coupling at w levels
        kz = 0.25 * (xk[1:] + xk[:-1])         # half levels k-1/k -> w level

        def face(ax, rd):
            sh = lambda a: np.roll(a, 1, axis=ax)
            g = (0.5 * (chw + sh(chw)) * (kz + sh(kz)) * rd
                 * (wc - sh(wc)))
            return rd * (np.roll(g, -1, axis=ax) - g)

        out = np.zeros_like(f)
        out[1:nzf - 1] = face(2, rdx) + face(1, rdy)
    else:                                      # WRF 'm' (scalar) branch
        gx = (0.5 * (xk + np.roll(xk, 1, axis=2))
              * 0.5 * (chm + np.roll(chm, 1, axis=2)) * rdx
              * (f - np.roll(f, 1, axis=2)))
        tx = rdx * (np.roll(gx, -1, axis=2) - gx)
        gy = (0.5 * (xk + np.roll(xk, 1, axis=1))
              * 0.5 * (chm + np.roll(chm, 1, axis=1)) * rdy
              * (f - np.roll(f, 1, axis=1)))
        ty = rdy * (np.roll(gy, -1, axis=1) - gy)
        out = tx + ty
    if open_x:                                 # WRF open bounds (width 1)
        out[:, :, :1] = 0.0
        out[:, :, -1:] = 0.0
    if open_y:
        out[:, :1, :] = 0.0
        out[:, -1:, :] = 0.0
    return out


#: Open-boundary radiation speed c* (m/s) of the Klemp-Wilhelmson gravity-
#: wave radiative BC: WRF's cb = 25 (share/module_model_constants.F:47,
#: consumed by the open radiative blocks in dyn_em/module_advect_em.F),
#: adjudicated over the plan's original 30 (the published KW78 value).
#: Must match dycore.OPEN_CB.
OPEN_CB = 25.0

#: WRF w_damp constants (share/module_model_constants.F:88-89 and the
#: Registry w_crit_cfl default): damping strength (m/s^2), activation
#: Courant number (w_beta, non-IEVA path), and the default reference
#: Courant number the excess is measured against (namelist w_crit_cfl).
W_DAMP_ALPHA = 0.3
W_DAMP_BETA = 1.0
W_CRIT_CFL = 1.0


def np_open_u_radiative(ru_t, u, mut, coord, dx, cb=OPEN_CB, msfu=None):
    """Mirror of ``open_u_radiative`` (woof/core/kernels/openbc.cu).

    WRF open (gravity-wave radiative) lateral BC for the boundary-normal
    velocity, transcribed from ``dyn_em/module_advect_em.F`` ``advect_u``
    (the ``open_xs``/``open_xe`` radiative blocks; called with u_old = u,
    the RK stage estimate, per ``module_em.F`` rk_tendency): the one-sided
    radiative term ``-rdx*ub*du`` with the outbound-only phase speed

      west:  ub = min(ru_b - cb*(c1h*mut + c2h), 0)
      east:  ub = max(ru_b + cb*(c1h*mut + c2h), 0)

    is ADDED to the coupled tendency at the two boundary u faces (WRF
    ``tendency = tendency + ...``, module_advect_em.F:1252/1267), the
    open-aware advection bounds excluded the x-advection at those faces
    and the radiative term stands in for it; the retained contributions
    (e.g. vertical advection when only x is open) remain in the sum.
    ``ru_b = (c1h*mut + c2h)*u/msfu`` couples the boundary-face u with the
    boundary CELL's column mass (WRF's muu under the zero-gradient mu ghost
    copy). Optional ``msfu`` is WRF's staggered ``msfuy``; its division
    applies to ``ru_b`` and leaves ``cb*(c1h*mut+c2h)`` unchanged. Omitting
    it preserves the map-factor-one expressions. Interior faces are
    untouched. Returns a new float64 array.
    """
    ru_t = np.asarray(ru_t, dtype=np.float64).copy()
    u = np.asarray(u, dtype=np.float64)
    nz, ny, nxp1 = u.shape
    mut = np.broadcast_to(np.asarray(mut, dtype=np.float64), (ny, nxp1 - 1))
    c1h = np.asarray(coord.c1h, dtype=np.float64)[:, None]
    c2h = np.asarray(coord.c2h, dtype=np.float64)[:, None]
    mw = c1h * mut[None, :, 0] + c2h                     # (nz, ny)
    if msfu is None:
        ub = np.minimum(mw * u[:, :, 0] - cb * mw, 0.0)
    else:
        msfu = np.broadcast_to(np.asarray(msfu, dtype=np.float64), (ny, nxp1))
        ub = np.minimum((mw * u[:, :, 0]) / msfu[None, :, 0] - cb * mw, 0.0)
    ru_t[:, :, 0] += -(1.0 / dx) * ub * (u[:, :, 1] - u[:, :, 0])
    me = c1h * mut[None, :, -1] + c2h
    if msfu is None:
        ub = np.maximum(me * u[:, :, -1] + cb * me, 0.0)
    else:
        ub = np.maximum((me * u[:, :, -1]) / msfu[None, :, -1] + cb * me, 0.0)
    ru_t[:, :, -1] += -(1.0 / dx) * ub * (u[:, :, -1] - u[:, :, -2])
    return ru_t


def np_open_v_radiative(rv_t, v, mut, coord, dy, cb=OPEN_CB, msfv=None):
    """Mirror of ``open_v_radiative``: y analogue of
    :func:`np_open_u_radiative` (WRF ``advect_v`` ``open_ys``/``open_ye``;
    additive per module_advect_em.F:2721/2736). Optional ``msfv`` is WRF's
    staggered ``msfvx`` and divides only the coupled meridional momentum.
    Omitting it preserves the map-factor-one expressions."""
    rv_t = np.asarray(rv_t, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64)
    nz, nyp1, nx = v.shape
    mut = np.broadcast_to(np.asarray(mut, dtype=np.float64), (nyp1 - 1, nx))
    c1h = np.asarray(coord.c1h, dtype=np.float64)[:, None]
    c2h = np.asarray(coord.c2h, dtype=np.float64)[:, None]
    ms = c1h * mut[None, 0, :] + c2h                     # (nz, nx)
    if msfv is None:
        vb = np.minimum(ms * v[:, 0, :] - cb * ms, 0.0)
    else:
        msfv = np.broadcast_to(np.asarray(msfv, dtype=np.float64), (nyp1, nx))
        vb = np.minimum((ms * v[:, 0, :]) / msfv[None, 0, :] - cb * ms, 0.0)
    rv_t[:, 0, :] += -(1.0 / dy) * vb * (v[:, 1, :] - v[:, 0, :])
    mn = c1h * mut[None, -1, :] + c2h
    if msfv is None:
        vb = np.maximum(mn * v[:, -1, :] + cb * mn, 0.0)
    else:
        vb = np.maximum((mn * v[:, -1, :]) / msfv[None, -1, :] + cb * mn, 0.0)
    rv_t[:, -1, :] += -(1.0 / dy) * vb * (v[:, -1, :] - v[:, -2, :])
    return rv_t


def np_emdiv_uv(u_pp, v_pp, mudf, coord, cfg, msfu=None, msfv=None):
    """Reference for :func:`woof.core.dycore.apply_emdiv_filter`: WRF's
    external-mode divergence damping (module_small_step_em.F ``mudf_xy``,
    advance_uv lines 809/868 and 880/942):

      u'' += c1h(k) * (-emdiv * dx * (mudf_i - mudf_{i-1}) / msfu_i)
      v'' += c1h(k) * (-emdiv * dy * (mudf_j - mudf_{j-1}) / msfv_j)

    with ``mudf (ny, nx)`` the PREVIOUS acoustic substep's column-mass
    tendency (advance_mu_t: dmdt + mu_tend; zeroed by small_step_prep, so
    the first substep of every RK stage adds nothing).  ``msfu``/``msfv``
    (Task 3, WRF's /msfuy and *msfvx_inv) default to 1.  Boundary-normal
    faces at open boundaries are excluded exactly like the acoustic
    pressure gradient (the mudf_xy loop shares advance_uv's bounds);
    periodic faces wrap.  Returns updated ``(u_pp, v_pp)`` float64.
    """
    u_pp = np.asarray(u_pp, dtype=np.float64).copy()
    v_pp = np.asarray(v_pp, dtype=np.float64).copy()
    mudf = np.asarray(mudf, dtype=np.float64)
    ny, nx = mudf.shape
    c1h = np.asarray(coord.c1h, dtype=np.float64)[:, None, None]
    gx = -cfg.emdiv * cfg.dx * (mudf - np.roll(mudf, 1, axis=1))
    if msfu is not None:
        gx = gx / np.asarray(msfu, dtype=np.float64)[:, :nx]
    if _boundary_forced_cfg(cfg):
        sz = cfg.spec_zone
        u_pp[:, sz:ny - sz, sz:nx + 1 - sz] += (
            c1h * gx[None, sz:ny - sz, sz:nx + 1 - sz])
    elif getattr(cfg, "open_x", False):
        u_pp[:, :, 1:nx] += c1h * gx[None, :, 1:]
    else:
        u_pp[:, :, :nx] += c1h * gx[None]
        u_pp[:, :, nx] += (c1h[:, :, 0] * gx[None, :, 0])
    gy = -cfg.emdiv * cfg.dy * (mudf - np.roll(mudf, 1, axis=0))
    if msfv is not None:
        gy = gy / np.asarray(msfv, dtype=np.float64)[:ny, :]
    if _boundary_forced_cfg(cfg):
        sz = cfg.spec_zone
        v_pp[:, sz:ny + 1 - sz, sz:nx - sz] += (
            c1h * gy[None, sz:ny + 1 - sz, sz:nx - sz])
    elif getattr(cfg, "open_y", False):
        v_pp[:, 1:ny, :] += c1h * gy[None, 1:, :]
    else:
        v_pp[:, :ny, :] += c1h * gy[None]
        v_pp[:, ny, :] += (c1h[:, 0, :] * gy[None, 0, :])
    return u_pp, v_pp


def np_w_damp(rw_t, ww, w, mut, coord, dt, *, w_crit_cfl=W_CRIT_CFL,
              zadvect_implicit=0):
    """Mirror of ``w_damp`` (openbc.cu): WRF's vertical-velocity limiter,
    transcribed from WRF 4.7.1 ``dyn_em/module_big_step_utilities_em.F``
    ``w_damp`` (w_damping = 1, map factors 1).

    On interior w levels k = 1..nz-1 (Fortran k = 2, kde-1) the vertical
    Courant number is ``vert_cfl = |ww/(c1f*mut + c2f) * rdnw[k] * dt|``;
    where it exceeds the activation value (``w_crit_cfl`` when
    ``zadvect_implicit > 0``, else ``w_beta = 1``), the coupled w
    tendency gets ``-sign(1, w)*w_alpha*(vert_cfl - w_crit_cfl)*
    (c1f*mut + c2f)``.  Returns a new float64 array.
    """
    rw_t = np.asarray(rw_t, dtype=np.float64).copy()
    ww = np.asarray(ww, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    nzp1 = ww.shape[0]
    nz = nzp1 - 1
    mut = np.broadcast_to(np.asarray(mut, dtype=np.float64), ww.shape[1:])
    c1f = np.asarray(coord.c1f, dtype=np.float64)[1:nz, None, None]
    c2f = np.asarray(coord.c2f, dtype=np.float64)[1:nz, None, None]
    rdnw = np.asarray(coord.rdnw, dtype=np.float64)[1:nz, None, None]
    cfm = c1f * mut[None] + c2f
    vert_cfl = np.abs(ww[1:nz] / cfm * rdnw * dt)
    onset = w_crit_cfl if zadvect_implicit > 0 else W_DAMP_BETA
    rw_t[1:nz] += np.where(
        vert_cfl > onset,
        -np.copysign(1.0, w[1:nz]) * W_DAMP_ALPHA
        * (vert_cfl - w_crit_cfl) * cfm, 0.0)
    return rw_t


def np_coriolis_curvature(ru, rv, u, v, w, mut, msft, msfu, msfv, f, e,
                          coord, dx, dy, *, sina=None, cosa=None,
                          boundary_x=False, boundary_y=False):
    """Mirror of ``coriolis_curvature`` (kernels/coriolis_map.cu): the WRF
    v4.6.1 ``coriolis`` + ``curvature`` large-step tendencies
    (module_big_step_utilities_em.F; periodic bounds, isotropic map
    factors, full sina/cosa rotation terms, see the kernel header).

    ``ru``/``rv`` are the msf-coupled stage momenta, ``u``/``v``/``w`` the
    uncoupled winds, ``mut (ny, nx)`` the total dry mass (the coupled
    ``rw = (c1f*mut + c2f)*w/msft`` is formed here), ``sina``/``cosa`` the
    local map-rotation angle at mass points (geo_em SINALPHA/COSALPHA;
    ``None`` selects WRF's unrotated identity sina = 0 / cosa = 1),
    ``coord`` anything carrying ``c1f``/``c2f``/``fnm``/``fnp``.  Returns
    the tendency increments ``(dru_t, drv_t, drw_t)`` float64 with the
    periodic duplicate faces (u face nx / v row ny) copied from face 0 /
    row 0.
    """
    ru = np.asarray(ru, dtype=np.float64)
    rv = np.asarray(rv, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mut = np.asarray(mut, dtype=np.float64)
    msft = np.asarray(msft, dtype=np.float64)
    msfu = np.asarray(msfu, dtype=np.float64)
    msfv = np.asarray(msfv, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    nz, ny, nxp1 = ru.shape
    nx = nxp1 - 1
    sina = (np.zeros((ny, nx)) if sina is None
            else np.asarray(sina, dtype=np.float64))
    cosa = (np.ones((ny, nx)) if cosa is None
            else np.asarray(cosa, dtype=np.float64))
    rdx, rdy = 1.0 / dx, 1.0 / dy
    c1f = np.asarray(coord.c1f, dtype=np.float64)[:, None, None]
    c2f = np.asarray(coord.c2f, dtype=np.float64)[:, None, None]
    fnm = np.asarray(coord.fnm, dtype=np.float64)[1:nz, None, None]
    fnp = np.asarray(coord.fnp, dtype=np.float64)[1:nz, None, None]

    # Coupled w flux (WRF couple_momentum 'w') and the shared stencil sums.
    rw = (c1f * mut[None] + c2f) * w / msft[None]          # (nz+1, ny, nx)
    rws = rw[:nz] + rw[1:]                # levels k + k+1 per column
    rvs = rv[:, :ny, :] + rv[:, 1:, :]    # v rows j + j+1 per column
    rus = ru[:, :, :nx] + ru[:, :, 1:]    # u faces i + i+1 per column
    us = u[:, :, :nx] + u[:, :, 1:]
    vs = v[:, :ny, :] + v[:, 1:, :]

    # WRF curvature vxgm at mass points: v cross grad m.
    vxgm = (0.5 * us * (msfv[1:, :] - msfv[:ny, :])[None] * rdy
            - 0.5 * vs * (msfu[:, 1:] - msfu[:, :nx])[None] * rdx)

    # u faces 0..nx-1 (face nx duplicates face 0).  Coriolis: WRF
    # module_big_step_utilities_em.F:3726-3729, the e-term carries the
    # x-averaged cosa factor.
    favg = 0.5 * (f + np.roll(f, 1, axis=1))
    eavg = 0.5 * (e + np.roll(e, 1, axis=1))
    cavg = 0.5 * (cosa + np.roll(cosa, 1, axis=1))
    rv4 = 0.25 * (rvs + np.roll(rvs, 1, axis=2))
    rw4 = 0.25 * (rws + np.roll(rws, 1, axis=2))
    vx = 0.5 * (vxgm + np.roll(vxgm, 1, axis=2))
    dru_core = (favg[None] * rv4 - eavg[None] * cavg[None] * rw4
                + vx * rv4 - u[:, :, :nx] * c.RERADIUS * rw4)
    dru = np.concatenate([dru_core, dru_core[:, :, :1]], axis=2)
    if boundary_x:
        # WRF coriolis/curvature i_start/i_end exclude the two
        # boundary-normal u faces for open, specified, and nested domains.
        dru[:, :, 0] = 0.0
        dru[:, :, -1] = 0.0

    # v faces 0..ny-1 (row ny duplicates row 0).  Coriolis: WRF :3800-3803:
    # the y-averaged +e*sina*<rw> rotation term ((msfvy/msfvx) == 1).
    favg = 0.5 * (f + np.roll(f, 1, axis=0))
    eavg = 0.5 * (e + np.roll(e, 1, axis=0))
    savg = 0.5 * (sina + np.roll(sina, 1, axis=0))
    ru4 = 0.25 * (rus + np.roll(rus, 1, axis=1))
    rw4 = 0.25 * (rws + np.roll(rws, 1, axis=1))
    vx = 0.5 * (vxgm + np.roll(vxgm, 1, axis=1))
    drv_core = (-favg[None] * ru4 + eavg[None] * savg[None] * rw4
                - vx * ru4 - v[:, :ny, :] * c.RERADIUS * rw4)
    drv = np.concatenate([drv_core, drv_core[:, :1, :]], axis=1)
    if boundary_y:
        drv[:, 0, :] = 0.0
        drv[:, -1, :] = 0.0

    # interior w levels 1..nz-1 (fnm/fnp interpolation to full levels).
    # Coriolis: WRF :3839-3844, +e*(cosa*<ru> - sina*<rv>),
    # (msftx/msfty) == 1.
    drw = np.zeros((nz + 1, ny, nx))
    ruf = 0.5 * (fnm * rus[1:] + fnp * rus[:nz - 1])
    uf = 0.5 * (fnm * us[1:] + fnp * us[:nz - 1])
    rvf = 0.5 * (fnm * rvs[1:] + fnp * rvs[:nz - 1])
    vf = 0.5 * (fnm * vs[1:] + fnp * vs[:nz - 1])
    drw[1:nz] = (e[None] * (cosa[None] * ruf - sina[None] * rvf)
                 + c.RERADIUS * (ruf * uf + rvf * vf))
    return dru, drv, drw


def np_rhs_ph_hadv(meta, cfg):
    """Mirror of ``dycore.add_rhs_ph_hadv`` (float64).

    WRF ``rhs_ph`` horizontal advection of the FULL geopotential ph+phb
    with the ``h_sca_adv_order`` stencil
    (module_big_step_utilities_em.F:1435).  Order 2 is the frozen <=2
    two-face branch (:1516-1584), excluding the entire normal-direction
    contribution on outer open/specified rows; order 5 is the <=6 branch's 1/60-weighted
    7-point centered stencil (:1786-1795 y / :1949-1959 x) applied
    everywhere when periodic and with WRF's specified narrowing
    otherwise, y rows 1/ny-2 2nd order (:1882-1906), rows 2/ny-3 4th
    order (:1819-1850), interior [3, ny-4]; x cols 1/nx-2 2nd order
    (:2023-2048), cols 2/nx-3 NOTHING (the 4th-order x pickups are gated
    on open_xs/open_xe only, :1973/:1997, binding v4.6.1 quirk),
    interior [3, nx-4]; spec rows/cols themselves get nothing.

    ``meta`` is an :func:`s_meta` snapshot.  Returns the (nz+1, ny, nx)
    contribution ADDED to ``rph_t`` (full levels 1..nz; the surface row is
    zero).  WRF's top row uses ``cfn/cfn1``-extrapolated u/v and a 1/2
    horizontal coefficient rather than the interior 1/4 coefficient
    (module_big_step_utilities_em.F:1542-1582, 1786-1795, 1949-1959).
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    order = getattr(cfg, "h_sca_adv_order", 2)
    open_x = getattr(cfg, "open_x", False) or _boundary_forced_cfg(cfg)
    open_y = getattr(cfg, "open_y", False) or _boundary_forced_cfg(cfg)
    c1f = meta["c1f"][1:nz + 1, None, None]
    c2f = meta["c2f"][1:nz + 1, None, None]
    ph_i = (_prof3(meta["phb"]) + meta["php"])[1:nz + 1]
    ph_i = np.broadcast_to(ph_i, (nz, ny, nx))
    mut = meta["mub2d"] + meta["mup"]
    mux = 0.5 * (mut + np.roll(mut, 1, axis=1))        # faces 0..nx-1
    muy = 0.5 * (mut + np.roll(mut, 1, axis=0))        # rows 0..ny-1
    ua = np.empty((nz, ny, nx), dtype=np.float64)
    ua[:-1] = meta["u"][1:nz, :, :nx] + meta["u"][:nz - 1, :, :nx]
    ua[-1] = (meta["cfn"] * meta["u"][nz - 1, :, :nx]
              + meta["cfn1"] * meta["u"][nz - 2, :, :nx])
    fcx = (c1f * mux[None] + c2f) * ua * meta["msfu"][None, :, :nx]
    va = np.empty((nz, ny, nx), dtype=np.float64)
    va[:-1] = meta["v"][1:nz, :ny, :] + meta["v"][:nz - 1, :ny, :]
    va[-1] = (meta["cfn"] * meta["v"][nz - 1, :ny, :]
              + meta["cfn1"] * meta["v"][nz - 2, :ny, :])
    fcy = (c1f * muy[None] + c2f) * va * meta["msfv"][None, :ny, :]
    msft = meta["msft"][None]
    hcoef = np.full((nz, 1, 1), 0.25, dtype=np.float64)
    hcoef[-1] = 0.5
    out = np.zeros((nz + 1, ny, nx))

    if order == 2:
        dphx = ph_i - np.roll(ph_i, 1, axis=2)
        if open_x:
            dphx[:, :, 0] = 0.0
        fx = fcx * dphx
        fxp = np.roll(fx, -1, axis=2)
        if open_x:
            fxp[:, :, -1] = 0.0
        hx = hcoef * rdx * (fxp + fx) / msft
        if open_x:
            hx[:, :, (0, -1)] = 0.0
        out[1:nz + 1] -= hx
        dphy = ph_i - np.roll(ph_i, 1, axis=1)
        if open_y:
            dphy[:, 0, :] = 0.0
        fy = fcy * dphy
        fyp = np.roll(fy, -1, axis=1)
        if open_y:
            fyp[:, -1, :] = 0.0
        hy = hcoef * rdy * (fyp + fy) / msft
        if open_y:
            hy[:, (0, -1), :] = 0.0
        out[1:nz + 1] -= hy
        return out

    if order != 5:
        raise ValueError(f"h_sca_adv_order must be 2 or 5, got {order}")
    if getattr(cfg, "open_x", False) or getattr(cfg, "open_y", False):
        raise NotImplementedError(
            "h_sca_adv_order=5 with radiative open boundaries is not "
            "wired (periodic and specified only)")

    csx = fcx + np.roll(fcx, -1, axis=2)
    csy = fcy + np.roll(fcy, -1, axis=1)

    def cdiff(ax, w1, w2, w3, denom):
        r = lambda s: np.roll(ph_i, -s, axis=ax)       # r(s)[i] = ph[i+s]
        return (w1 * (r(1) - r(-1)) + w2 * (r(2) - r(-2))
                + w3 * (r(3) - r(-3))) / denom

    hx = hcoef * rdx * csx * cdiff(2, 45.0, -9.0, 1.0, 60.0)
    hy = hcoef * rdy * csy * cdiff(1, 45.0, -9.0, 1.0, 60.0)
    if _boundary_forced_cfg(cfg):
        dphx = ph_i - np.roll(ph_i, 1, axis=2)
        fx2 = fcx * dphx
        h2x = hcoef * rdx * (np.roll(fx2, -1, axis=2) + fx2)
        dphy = ph_i - np.roll(ph_i, 1, axis=1)
        fy2 = fcy * dphy
        h2y = hcoef * rdy * (np.roll(fy2, -1, axis=1) + fy2)
        hx[:, :, :3] = 0.0
        hx[:, :, nx - 3:] = 0.0
        hx[:, :, 1] = h2x[:, :, 1]
        hx[:, :, nx - 2] = h2x[:, :, nx - 2]
        d4y = cdiff(1, 8.0, -1.0, 0.0, 12.0)
        hy[:, :3, :] = 0.0
        hy[:, ny - 3:, :] = 0.0
        hy[:, 2, :] = hcoef[:, 0] * rdy * csy[:, 2, :] * d4y[:, 2, :]
        hy[:, ny - 3, :] = (hcoef[:, 0] * rdy * csy[:, ny - 3, :]
                            * d4y[:, ny - 3, :])
        hy[:, 1, :] = h2y[:, 1, :]
        hy[:, ny - 2, :] = h2y[:, ny - 2, :]
    out[1:nz + 1] -= (hx + hy) / msft
    return out


# ---------------------------------------------------------------------------
# Phase 3 Task 8: real-data vertical interpolation and specified boundaries
# ---------------------------------------------------------------------------

def np_vertical_interpolate_logp(field, source_pressure, target_pressure, *,
                                 below="constant", above="error"):
    """Float64 mirror of the retained linear-log(p) vertical kernel.

    Interior values are linear in ``log(p)``: WRF's ``interp_type=2`` with
    ``lagrange_order=1`` and no surface pseudo-level, NOT WRF real's default
    configuration (that full machinery lives in
    :func:`np_wrf_real_vert_interp`).  ``below='temperature'`` is
    ``module_initialize_real.F:lagrange_setup``'s ``t_extrap_type=2``
    standard-atmosphere extrapolation applied to potential temperature;
    ``below='constant'`` is ``extrap_type=2``.  A target above the source
    top is fatal, matching WRF's unbracketed-target initialization error
    rather than silently extending the top value.
    """
    values = np.asarray(field, dtype=np.float64)
    source = np.asarray(source_pressure, dtype=np.float64)
    target = np.asarray(target_pressure, dtype=np.float64)
    if values.ndim != 3 or target.ndim != 3:
        raise ValueError("field and target_pressure must be (level, y, x)")
    try:
        source = np.broadcast_to(source, values.shape)
    except ValueError as exc:
        raise ValueError("source_pressure is not broadcastable to field") from exc
    if target.shape[1:] != values.shape[1:]:
        raise ValueError("source and target horizontal shapes differ")
    if below not in ("constant", "temperature"):
        raise ValueError("below must be 'constant' or 'temperature'")
    if above != "error":
        raise ValueError("above must be 'error'")
    if (not np.isfinite(values).all() or not np.isfinite(source).all()
            or not np.isfinite(target).all()):
        raise ValueError("vertical interpolation inputs must be finite")
    if np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("vertical interpolation pressures must be positive")

    ns, ny, nx = values.shape
    out = np.empty(target.shape, dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            order = np.argsort(source[:, j, i])[::-1]
            p = source[order, j, i]
            q = values[order, j, i]
            if np.any(np.diff(p) >= 0.0):
                raise ValueError("source pressure must be unique in every column")
            lp = np.log(p)
            for k, pt in enumerate(target[:, j, i]):
                if pt > p[0]:
                    if below == "constant":
                        out[k, j, i] = q[0]
                    else:
                        # WRF v4.6.1 lagrange_setup, t_extrap_type=2.
                        t1 = q[0] * (p[0] / c.P0) ** c.RCP
                        depth = pt - p[0]
                        pavg = 0.5 * (pt + p[0])
                        dhdp = (11880.516 * 0.1902632
                                * (pavg / 100.0) ** (0.1902632 - 1.0))
                        dt = dhdp * (depth / 100.0) * 0.0065
                        out[k, j, i] = (t1 + dt) * (c.P0 / pt) ** c.RCP
                elif pt < p[-1]:
                    raise ValueError("target pressure lies above source top")
                else:
                    lower = np.nonzero((p[:-1] >= pt) & (pt >= p[1:]))[0]
                    if lower.size == 0:
                        raise ValueError("could not bracket target pressure")
                    n = int(lower[0])
                    weight = (np.log(pt) - lp[n]) / (lp[n + 1] - lp[n])
                    out[k, j, i] = q[n] + weight * (q[n + 1] - q[n])
    return out


def _np_lagrange_interp(x, y, order, target_x):
    """WRF ``lagrange_interp`` (module_initialize_real.F:6661-6699).

    Full Lagrange polynomial through ``order + 1`` points, with WRF's
    zero-denominator term skip.
    """
    px = 0.0
    for term in range(order + 1):
        numer = 1.0
        denom = 1.0
        for k in range(order + 1):
            if k == term:
                continue
            numer *= target_x - x[k]
            denom *= x[term] - x[k]
        if denom != 0.0:
            px += y[term] * numer / denom
    return px


def np_wrf_real_vert_interp(field, surface_value, source_pressure,
                            surface_pressure, target_pressure, *,
                            interp_in_logp=True, extrap="constant",
                            force_sfc_in_vinterp=1, zap_close_levels=500.0,
                            vboundb=4):
    """Float64 authority for WRF real's default vertical interpolation.

    Transcription of ``module_initialize_real.F:vert_interp`` (v4.6.1) with
    the reference run's ratified Registry defaults ``use_surface=.true.`` and
    ``use_levels_below_ground=.true.`` (Registry.EM_COMMON:2285,2287), no
    metgrid max-wind/tropopause extra levels (flags 0 for this data source),
    and ``lowest_lev_from_sfc=.false.``:

    * Column assembly (:5940-6079): the met-style source column is
      ``[below-ground isobaric levels, surface pseudo-level, above-surface
      levels]``.  A below-ground level within ``zap_close_levels`` Pa of the
      surface is dropped (:5957-5961); ``force_sfc_in_vinterp=N`` removes
      above-surface input levels down to the target dry pressure of eta
      level N so the surface analysis bounds the lowest layer(s)
      (:5978-6002, :6047-6060); the level left just above the surface is
      dropped when within ``zap_close_levels`` of it (:6010-6016,
      :6067-6073, including WRF's asymmetric per-branch re-checking).
    * ``lagrange_setup`` (:6339-6574) with ``lagrange_order=2``
      (Registry:2288): targets with 1-based index ``<= vboundb`` (vboundb=4,
      :6418) interpolate linearly between the trapping input levels
      (:6565-6568); higher targets average the two overlapping second-order
      Lagrange polynomials when both fit, else use the one that fits
      (:6537-6563).
    * ``interp_in_logp`` selects the LOG(p) polynomial space
      (``interp_type=2``, :6309-6315) versus plain pressure
      (``interp_type=1``, :6303-6308, forced for the full-pressure field at
      :1807).
    * Below-ground targets (no trapping interval and target pressure above
      the deepest column entry): ``extrap="temperature"`` is the
      ``t_extrap_type=2`` / var ``'T'`` CRC standard-atmosphere branch
      (:6459-6467); ``extrap="constant"`` is ``extrap_type=2`` (:6486-6488).
      Both start from the deepest column entry (a below-ground isobaric
      level when one is retained, otherwise the surface).
    * A target above the source top is fatal (:6495-6502).

    ``field``/``source_pressure`` are ``(nsource, ny, nx)`` bottom-up
    (strictly descending pressure) WITHOUT the surface; ``surface_value``/
    ``surface_pressure`` carry the surface pseudo-level (met_em level 1);
    ``target_pressure`` is ``(ntarget, ny, nx)``.
    """
    field = np.asarray(field, dtype=np.float64)
    sfc_value = np.asarray(surface_value, dtype=np.float64)
    source = np.asarray(source_pressure, dtype=np.float64)
    sfc_pressure = np.asarray(surface_pressure, dtype=np.float64)
    target = np.asarray(target_pressure, dtype=np.float64)
    if field.ndim != 3 or target.ndim != 3:
        raise ValueError("field and target_pressure must be (level, y, x)")
    if source.shape != field.shape:
        raise ValueError("source_pressure shape does not match field")
    if sfc_value.shape != field.shape[1:] or sfc_pressure.shape != field.shape[1:]:
        raise ValueError("surface fields must be (y, x)")
    if target.shape[1:] != field.shape[1:]:
        raise ValueError("source and target horizontal shapes differ")
    if extrap not in ("constant", "temperature"):
        raise ValueError("extrap must be 'constant' or 'temperature'")
    if np.any(np.diff(source, axis=0) >= 0.0):
        raise ValueError("source pressure must be strictly descending")
    if force_sfc_in_vinterp > target.shape[0]:
        raise ValueError("force_sfc_in_vinterp exceeds target level count")

    nsource, ny, nx = field.shape
    ntarget = target.shape[0]
    out = np.empty(target.shape, dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            pd = source[:, j, i]
            val = field[:, j, i]
            psfc = float(sfc_pressure[j, i])
            above = np.nonzero(pd < psfc)[0]
            if above.size == 0:
                raise ValueError("no source level above the surface")
            m_above = int(above[0])
            ox: list[float] = []
            oy: list[float] = []
            if m_above > 0:
                # Surface sits inside the column: below-ground levels first.
                ox = list(pd[:m_above])
                oy = list(val[:m_above])
                if ox[-1] - psfc < zap_close_levels:
                    ox.pop()
                    oy.pop()
                ox.append(psfc)
                oy.append(float(sfc_value[j, i]))
                knext = m_above
                if force_sfc_in_vinterp > 0:
                    pforce = target[force_sfc_in_vinterp - 1, j, i]
                    for ko in range(m_above, nsource):
                        if pd[ko] <= pforce:
                            knext = ko
                            break
                kst = (knext + 1 if ox[-1] - pd[knext] < zap_close_levels
                       else knext)
                ox.extend(pd[kst:])
                oy.extend(val[kst:])
            else:
                # Surface is the lowest level; iterative close-level check.
                ox = [psfc]
                oy = [float(sfc_value[j, i])]
                knext = 0
                if force_sfc_in_vinterp > 0:
                    pforce = target[force_sfc_in_vinterp - 1, j, i]
                    for ko in range(nsource):
                        if pd[ko] <= pforce:
                            knext = ko
                            break
                for ko in range(knext, nsource):
                    if (ox[-1] - pd[ko] < zap_close_levels
                            and ko < nsource - 1):
                        continue
                    ox.append(pd[ko])
                    oy.append(val[ko])
            count = len(ox)
            x = np.log(ox) if interp_in_logp else np.asarray(ox)
            for kt in range(ntarget):
                pt = float(target[kt, j, i])
                xt = math.log(pt) if interp_in_logp else pt
                found = -1
                for loop in range(count - 1):
                    a = xt - x[loop]
                    b = xt - x[loop + 1]
                    if a * b <= 0.0:
                        found = loop
                        break
                if found < 0:
                    if pt > ox[0]:
                        if extrap == "temperature":
                            t1 = oy[0] * (ox[0] / c.P0) ** c.RCP
                            pavg = 0.5 * (pt + ox[0])
                            dhdp = (11880.516 * 0.1902632
                                    * (pavg / 100.0) ** (0.1902632 - 1.0))
                            dt = dhdp * ((pt - ox[0]) / 100.0) * 0.0065
                            out[kt, j, i] = (t1 + dt) * (c.P0 / pt) ** c.RCP
                        else:
                            out[kt, j, i] = oy[0]
                    else:
                        raise ValueError("target pressure lies above source top")
                elif kt + 1 >= 1 + vboundb:
                    fits_upper = found + 2 <= count - 1
                    fits_lower = found - 1 >= 0
                    if fits_upper and fits_lower:
                        upper = _np_lagrange_interp(
                            x[found:found + 3], oy[found:found + 3], 2, xt)
                        lower = _np_lagrange_interp(
                            x[found - 1:found + 2], oy[found - 1:found + 2],
                            2, xt)
                        out[kt, j, i] = 0.5 * (upper + lower)
                    elif fits_upper:
                        out[kt, j, i] = _np_lagrange_interp(
                            x[found:found + 3], oy[found:found + 3], 2, xt)
                    elif fits_lower:
                        out[kt, j, i] = _np_lagrange_interp(
                            x[found - 1:found + 2], oy[found - 1:found + 2],
                            2, xt)
                    else:
                        raise ValueError(
                            "no second-order window fits the column")
                else:
                    out[kt, j, i] = _np_lagrange_interp(
                        x[found:found + 2], oy[found:found + 2], 1, xt)
    return out


def _lbc_weights(width, spec_zone, relax_zone, dt, spec_exp):
    """WRF ``module_bc_em.F:lbc_fcx_gcx`` in float64."""
    fcx = np.zeros(width, dtype=np.float64)
    gcx = np.zeros(width, dtype=np.float64)
    for loop in range(spec_zone + 1, spec_zone + relax_zone + 1):
        index = loop - 1
        if index >= width:
            break
        ramp = (spec_zone + relax_zone - loop) / (relax_zone - 1)
        sponge = np.exp(-(loop - (spec_zone + 1)) * spec_exp)
        fcx[index] = 0.1 / dt * ramp * sponge
        gcx[index] = 1.0 / dt / 50.0 * ramp * sponge
    return fcx, gcx


def np_specified_relaxation(field, tendency, boundary, *, dtbc, dt,
                            spec_zone=1, relax_zone=4, spec_exp=0.0,
                            apply_relax=True):
    """Mirror WRF ``spec_bdytend`` then ``relax_bdytend_core``.

    Boundary side layouts are west/east ``(nz,ny,width)`` and south/north
    ``(nz,width,nx)``; east/north distance zero is the outermost edge.
    The loop bounds preserve WRF's corner ownership (Y sides own corners).
    """
    field = np.asarray(field, dtype=np.float64)
    out = np.asarray(tendency, dtype=np.float64).copy()
    if field.ndim != 3 or out.shape != field.shape:
        raise ValueError("field and tendency must have the same 3-D shape")
    nz, ny, nx = field.shape
    width = boundary.west.value.shape[-1]
    if width < spec_zone + relax_zone:
        raise ValueError("boundary width is smaller than spec_zone + relax_zone")
    fcx, gcx = _lbc_weights(width, spec_zone, relax_zone, dt, spec_exp)

    def forcing(side, index):
        return side.value[index] + dtbc * side.tendency[index]

    # spec_bdytend: Y sides first and own all four corners.
    for d in range(spec_zone):
        js, jn = d, ny - 1 - d
        for k in range(nz):
            for i in range(d, nx - d):
                out[k, js, i] = boundary.south.tendency[k, d, i]
                out[k, jn, i] = boundary.north.tendency[k, d, i]
        iw, ie = d, nx - 1 - d
        for k in range(nz):
            for j in range(d + 1, ny - d - 1):
                out[k, j, iw] = boundary.west.tendency[k, j, d]
                out[k, j, ie] = boundary.east.tendency[k, j, d]

    if not apply_relax:
        return out

    # relax_bdytend_core: spec_zone <= distance < relax_zone.
    for d in range(spec_zone, relax_zone):
        js, jn = d, ny - 1 - d
        for k in range(nz):
            for i in range(d, nx - d):
                side = boundary.south
                f0 = forcing(side, (k, d, i)) - field[k, js, i]
                f1 = forcing(side, (k, d, max(i - 1, 0))) - field[k, js, max(i - 1, 0)]
                f2 = forcing(side, (k, d, min(i + 1, nx - 1))) - field[k, js, min(i + 1, nx - 1)]
                f3 = forcing(side, (k, d - 1, i)) - field[k, js - 1, i]
                f4 = forcing(side, (k, d + 1, i)) - field[k, js + 1, i]
                out[k, js, i] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)

                side = boundary.north
                f0 = forcing(side, (k, d, i)) - field[k, jn, i]
                f1 = forcing(side, (k, d, max(i - 1, 0))) - field[k, jn, max(i - 1, 0)]
                f2 = forcing(side, (k, d, min(i + 1, nx - 1))) - field[k, jn, min(i + 1, nx - 1)]
                f3 = forcing(side, (k, d - 1, i)) - field[k, jn + 1, i]
                f4 = forcing(side, (k, d + 1, i)) - field[k, jn - 1, i]
                out[k, jn, i] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)

        iw, ie = d, nx - 1 - d
        for k in range(nz):
            for j in range(d + 1, ny - d - 1):
                side = boundary.west
                f0 = forcing(side, (k, j, d)) - field[k, j, iw]
                f1 = forcing(side, (k, j - 1, d)) - field[k, j - 1, iw]
                f2 = forcing(side, (k, j + 1, d)) - field[k, j + 1, iw]
                f3 = forcing(side, (k, j, d - 1)) - field[k, j, iw - 1]
                f4 = forcing(side, (k, j, d + 1)) - field[k, j, iw + 1]
                out[k, j, iw] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)

                side = boundary.east
                f0 = forcing(side, (k, j, d)) - field[k, j, ie]
                f1 = forcing(side, (k, j - 1, d)) - field[k, j - 1, ie]
                f2 = forcing(side, (k, j + 1, d)) - field[k, j + 1, ie]
                f3 = forcing(side, (k, j, d - 1)) - field[k, j, ie + 1]
                f4 = forcing(side, (k, j, d + 1)) - field[k, j, ie - 1]
                out[k, j, ie] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)
    return out


def np_rk_specified_relaxation(field, tendencies, boundary, *, dtbc, dt,
                               hold_relaxation, spec_zone=1, relax_zone=4,
                               spec_exp=0.0):
    """Mirror WRF's three-stage specified/relaxation tendency policy.

    ``spec_bdytend`` replaces the specified-zone tendency on every RK stage.
    The dry u/v/theta/phi relaxation increment is computed from the stage-1
    field once and held in the ``*_save`` slot for all three stages; pass
    ``hold_relaxation=True`` for those fields.  Mu and moist scalars receive
    relaxation only on stage 1, so pass ``False`` for them.
    """
    if len(tendencies) != 3:
        raise ValueError("tendencies must contain exactly three RK stages")
    zeros = np.zeros_like(np.asarray(field, dtype=np.float64))
    spec_only = np_specified_relaxation(
        field, zeros, boundary, dtbc=dtbc, dt=dt, spec_zone=spec_zone,
        relax_zone=relax_zone, spec_exp=spec_exp, apply_relax=False)
    combined = np_specified_relaxation(
        field, zeros, boundary, dtbc=dtbc, dt=dt, spec_zone=spec_zone,
        relax_zone=relax_zone, spec_exp=spec_exp, apply_relax=True)
    held = combined - spec_only

    result = []
    for rk_stage, tendency in enumerate(tendencies):
        stage = np_specified_relaxation(
            field, tendency, boundary, dtbc=dtbc, dt=dt,
            spec_zone=spec_zone, relax_zone=relax_zone, spec_exp=spec_exp,
            apply_relax=False)
        if hold_relaxation or rk_stage == 0:
            stage += held
        result.append(stage)
    return tuple(result)


def np_flow_dependent_boundary(field, u_flux, v_flux, spec_zone=1):
    """NumPy mirror of WRF ``share/module_bc.F:flow_dep_bdy``."""
    out = np.asarray(field, dtype=np.float64).copy()
    u_flux = np.asarray(u_flux, dtype=np.float64)
    v_flux = np.asarray(v_flux, dtype=np.float64)
    if out.ndim != 3:
        raise ValueError("flow-dependent boundary field must be 3-D")
    nz, ny, nx = out.shape
    if u_flux.shape != (nz, ny, nx + 1):
        raise ValueError("u_flux must have shape (nz, ny, nx + 1)")
    if v_flux.shape != (nz, ny + 1, nx):
        raise ValueError("v_flux must have shape (nz, ny + 1, nx)")

    for d in range(spec_zone):
        cols = np.arange(d, nx - d)
        inner_i = np.clip(cols, spec_zone, nx - 1 - spec_zone)
        out[:, d, cols] = np.where(
            v_flux[:, d, cols] < 0.0,
            out[:, spec_zone, inner_i], 0.0)
        jn = ny - 1 - d
        out[:, jn, cols] = np.where(
            v_flux[:, jn + 1, cols] > 0.0,
            out[:, ny - 1 - spec_zone, inner_i], 0.0)

        rows = np.arange(d + 1, ny - d - 1)
        inner_j = np.clip(rows, spec_zone, ny - 1 - spec_zone)
        out[:, rows, d] = np.where(
            u_flux[:, rows, d] < 0.0,
            out[:, inner_j, spec_zone], 0.0)
        ie = nx - 1 - d
        out[:, rows, ie] = np.where(
            u_flux[:, rows, ie + 1] > 0.0,
            out[:, inner_j, nx - 1 - spec_zone], 0.0)
    return out


def np_clock_scaled_diff6_factor(cfg):
    """Float64 mirror of dycore's model-clock sixth-order factor."""
    clock_dt = cfg.clock_dt if cfg.clock_dt > 0.0 else cfg.dt
    factor = float(cfg.diff_6th_factor)
    if clock_dt == cfg.dt:
        return factor
    if not 0.0 <= factor <= 1.0:
        raise ValueError(
            "clock-scaled diff_6th_factor must lie in [0, 1], got "
            f"{factor}")
    if factor == 1.0:
        return 1.0
    return -np.expm1((cfg.dt / clock_dt) * np.log1p(-factor))


def np_advect_1d_rk3(q0, u0, dx, t_end, cfl):
    """1-D periodic RK3 (WRF dt/3, dt/2, dt) + 5th-order upwind driver.

    Float64 reference for the convergence-order test; uses the same
    ``_flux5`` stencil as the flux-divergence mirrors.
    """
    q = np.asarray(q0, dtype=np.float64).copy()

    def rhs(qs):
        r = lambda off: np.roll(qs, -off)          # r(off)[i] = qs[(i+off)%n]
        f = _flux5(r(-3), r(-2), r(-1), r(0), r(1), r(2), u0)
        return -(np.roll(f, -1) - f) / dx          # -(F[i+1] - F[i]) / dx

    nsub = max(1, int(np.ceil(t_end / (cfl * dx / abs(u0)))))
    dt = t_end / nsub
    for _ in range(nsub):
        q1 = q + (dt / 3.0) * rhs(q)
        q2 = q + (dt / 2.0) * rhs(q1)
        q = q + dt * rhs(q2)
    return q


# ---------------------------------------------------------------------------
# WRF MM5 surface layer (Phase 3 Task 9)
# ---------------------------------------------------------------------------

_SFCLAY_OUTPUTS = (
    "znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol", "regime",
    "psim", "psih", "fm", "fh", "lh", "u10", "v10", "th2", "t2",
    "q2", "chs", "chs2", "cqs2", "flhc", "flqc", "qgh", "rmol",
    "wspd", "br", "gz1oz0", "cpm", "ck", "cka", "cd", "cda",
)


def _sf_psim_classic_full(zeta):
    """Paulson momentum integral used to initialize SFCLAY's table."""
    x = (1.0 - 16.0 * zeta) ** 0.25
    return (2.0 * np.log(0.5 * (1.0 + x))
            + np.log(0.5 * (1.0 + x * x)) - 2.0 * np.arctan(x)
            + 2.0 * np.arctan(1.0))


def _sf_psih_classic_full(zeta):
    y = np.sqrt(1.0 - 16.0 * zeta)
    return 2.0 * np.log(0.5 * (1.0 + y))


def _sf_classic_table(func, zeta):
    """Analytic evaluation of the 0.01-spaced initialized lookup table."""
    x = min(max(-float(zeta), 0.0), 9.9999) * 100.0
    n = int(x)
    r = x - n
    return func(-0.01 * n) + r * (func(-0.01 * (n + 1))
                                   - func(-0.01 * n))


def _sf_psim_stable_full(zeta):
    # Cheng and Brutsaert (2005), exactly as sf_sfclayrev.F90.
    return -6.1 * np.log(zeta + (1.0 + zeta ** 2.5) ** (1.0 / 2.5))


def _sf_psih_stable_full(zeta):
    return -5.3 * np.log(zeta + (1.0 + zeta ** 1.1) ** (1.0 / 1.1))


def _sf_psim_unstable_full(zeta):
    x = (1.0 - 16.0 * zeta) ** 0.25
    psimk = (2.0 * np.log(0.5 * (1.0 + x))
             + np.log(0.5 * (1.0 + x * x)) - 2.0 * np.arctan(x)
             + 2.0 * np.arctan(1.0))
    ym = (1.0 - 10.0 * zeta) ** 0.33  # the file uses exponent .33
    psimc = (1.5 * np.log((ym * ym + ym + 1.0) / 3.0)
             - np.sqrt(3.0) * np.arctan((2.0 * ym + 1.0) / np.sqrt(3.0))
             + 4.0 * np.arctan(1.0) / np.sqrt(3.0))
    return (psimk + zeta * zeta * psimc) / (1.0 + zeta * zeta)


def _sf_psih_unstable_full(zeta):
    y = np.sqrt(1.0 - 16.0 * zeta)
    psihk = 2.0 * np.log((1.0 + y) / 2.0)
    yh = (1.0 - 34.0 * zeta) ** 0.33
    psihc = (1.5 * np.log((yh * yh + yh + 1.0) / 3.0)
             - np.sqrt(3.0) * np.arctan((2.0 * yh + 1.0) / np.sqrt(3.0))
             + 4.0 * np.arctan(1.0) / np.sqrt(3.0))
    return (psihk + zeta * zeta * psihc) / (1.0 + zeta * zeta)


def _sf_rev_table(func, zeta, unstable=False):
    """Revised scheme's 0.01 lookup, including its >=9.99 full fallback."""
    x = (-zeta if unstable else zeta) * 100.0
    n = int(x)
    r = x - n
    if n + 1 < 1000:
        sign = -1.0 if unstable else 1.0
        z0, z1 = sign * 0.01 * n, sign * 0.01 * (n + 1)
        return func(z0) + r * (func(z1) - func(z0))
    return func(zeta)


def _sf_psim_stable(zeta):
    return _sf_rev_table(_sf_psim_stable_full, max(float(zeta), 0.0))


def _sf_psih_stable(zeta):
    return _sf_rev_table(_sf_psih_stable_full, max(float(zeta), 0.0))


def _sf_psim_unstable(zeta):
    return _sf_rev_table(_sf_psim_unstable_full, min(float(zeta), 0.0),
                         unstable=True)


def _sf_psih_unstable(zeta):
    return _sf_rev_table(_sf_psih_unstable_full, min(float(zeta), 0.0),
                         unstable=True)


def _sf_log_zratio(num, z0):
    """Defined-behaviour ``log(num/z0)``; mirrors ``sfclay.cu sf_log_zratio``.

    In float64 the quotient of float32-ranged inputs never overflows, so the
    only degenerate case left here is ``z0 <= 0``.  WRF v4.6.1 leaves zero
    roughness undefined (module_sf_sfclay.F:494 ``GZ1OZ0 = ALOG(ZA/ZNT)`` at
    ``ZNT=0`` sends Inf into ``U10 = Inf/Inf = NaN``); the port floors the
    logarithm's roughness at FLT_MIN instead -- a huge-but-finite contrast
    whose exchange coefficients go smoothly to zero.  Documented divergence.

    A NaN argument is NOT that case and must not be treated as one.  The
    floor below is written ``not z0 > 0.0``, which is true for NaN, so a NaN
    roughness would leave here as ``log(num / FLT_MIN)`` -- an ordinary
    finite contrast around 90, indistinguishable from a real answer.  That
    is laundering already-corrupted state into a plausible number, and it is
    the one way a caller can lose the evidence that something upstream broke.
    Zero roughness is a physical input the port defines; NaN roughness is a
    defect, so it propagates.
    """
    if np.isnan(num) or np.isnan(z0):
        return num + z0
    if not z0 > 0.0:
        z0 = 1.1754943508222875e-38
    return np.log(num / z0)


def _sf_zolri_residual(zeta, ri, z, z0):
    """``zolri2`` from sf_sfclayrev.F90; returns residual and limited zeta."""
    if zeta * ri < 0.0:
        zeta = 0.0
    zeta0 = zeta * z0 / z
    zeta3 = zeta + zeta0
    if ri < 0.0:
        fm = (_sf_log_zratio(z + z0, z0)
              - (_sf_psim_unstable(zeta3) - _sf_psim_unstable(zeta0)))
        fh = (_sf_log_zratio(z + z0, z0)
              - (_sf_psih_unstable(zeta3) - _sf_psih_unstable(zeta0)))
    else:
        fm = (_sf_log_zratio(z + z0, z0)
              - (_sf_psim_stable(zeta3) - _sf_psim_stable(zeta0)))
        fh = (_sf_log_zratio(z + z0, z0)
              - (_sf_psih_stable(zeta3) - _sf_psih_stable(zeta0)))
    return zeta * fh / (fm * fm) - ri, zeta


def _sf_zolri(ri, z, z0):
    """Ten-iteration regula-falsi inversion copied from revised SFCLAY."""
    x1, x2 = (-5.0, 0.0) if ri < 0.0 else (0.0, 5.0)
    fx1, x1 = _sf_zolri_residual(x1, ri, z, z0)
    fx2, x2 = _sf_zolri_residual(x2, ri, z, z0)
    result = x1 if abs(fx1) < abs(fx2) else x2
    iteration = 0
    while abs(x1 - x2) > 0.01:
        if iteration == 10 or fx1 == fx2:
            return result
        if abs(fx2) < abs(fx1):
            x1 = x1 - fx1 / (fx2 - fx1) * (x2 - x1)
            fx1, x1 = _sf_zolri_residual(x1, ri, z, z0)
            result = x1
        else:
            x2 = x2 - fx2 / (fx2 - fx1) * (x2 - x1)
            fx2, x2 = _sf_zolri_residual(x2, ri, z, z0)
            result = x2
        iteration += 1
    return result


def _sf_rev_heat_psi(zol, za, roughness, height):
    """Revised heat integral between roughness and ``height+roughness``."""
    zeta_hi = zol * (height + roughness) / za
    zeta_0 = zol * roughness / za
    if zol > 0.0:
        return _sf_psih_stable(zeta_hi) - _sf_psih_stable(zeta_0)
    if zol < 0.0:
        return _sf_psih_unstable(zeta_hi) - _sf_psih_unstable(zeta_0)
    return 0.0


def _np_sfclay_point(u, v, temp, qv, pressure, dz8w, psfc, tsk, znt,
                      pblh, mavail, xland, qsfc, zol, ust, ustm, mol,
                      hfx_old, qfx_old, lakemask, dx, option, isfflx,
                      isftcflx, iz0tlnd):
    """One float64 surface point, line-for-line with the two WRF files."""
    karman = 0.4
    # WRF's EP_1, which is rvovrd - 1 with rvovrd the float32 quotient.
    # Spelling it c.RV / c.RD here made it a double divide -- a different
    # number from the one sfclay.cu forms as RV / RD - 1.0f, and from the one
    # WRF folds.  The mirror stays float64; only the constant is WRF's.
    ep1 = c.RVOVRD - 1.0
    xka = 2.4e-5
    salinity = 0.98
    land = xland < 1.5
    old_ust, old_mol, old_zol = ust, mol, zol
    old_ustm = ustm

    psfc_kpa = psfc / 1000.0
    thgb = tsk * (c.P0 / psfc) ** c.RCP
    thx = temp * (c.P0 / pressure) ** c.RCP
    thvx = thx * (1.0 + ep1 * qv)
    tv = temp * (1.0 + ep1 * qv)
    cpm = c.CP * (1.0 + 0.8 * qv)

    es = c.SVP1 * np.exp(c.SVP2 * (tsk - c.SVPT0) / (tsk - c.SVP3))
    if not land and lakemask == 0.0:
        es *= salinity
    if not land or qsfc <= 0.0:
        qsfc = c.EP2 * es / (psfc_kpa - es)
    es_air = c.SVP1 * np.exp(c.SVP2 * (temp - c.SVPT0)
                             / (temp - c.SVP3))
    qgh = c.EP2 * es_air / (pressure / 1000.0 - es_air)

    rho = psfc / (c.RD * tv)
    za = 0.5 * dz8w
    if option == 91:
        gz1 = _sf_log_zratio(za, znt)
        gz2 = _sf_log_zratio(2.0, znt)
        gz10 = _sf_log_zratio(10.0, znt)
    else:
        gz1 = _sf_log_zratio(za + znt, znt)
        gz2 = _sf_log_zratio(2.0 + znt, znt)
        gz10 = _sf_log_zratio(10.0 + znt, znt)

    tskv = thgb * (1.0 + ep1 * qsfc)
    dthv = thvx - tskv
    wspd0 = np.hypot(u, v)
    if land:
        fluxc = max(hfx_old / rho / c.CP + ep1 * tskv * qfx_old / rho,
                    0.0)
        vconv = (c.G / tsk * pblh * fluxc) ** 0.33
    else:
        vconv = np.sqrt(max(-dthv, 0.0))
    vsgd = 0.32 * max(dx / 5000.0 - 1.0, 0.0) ** 0.33
    wspd = max(np.sqrt(wspd0 * wspd0 + vconv * vconv + vsgd * vsgd),
               0.1)
    br = c.G / thx * za * dthv / (wspd * wspd)
    if old_mol < 0.0:
        br = min(br, 0.0)

    psim = psih = psim10 = psih10 = psim2 = psih2 = 0.0
    pq = pq2 = pq10 = 0.0
    zol = old_zol

    if option == 91:
        if br >= 0.2:
            regime = 1.0
            psim = max(-10.0 * gz1, -10.0)
            psih = psim
            psim10 = max(10.0 / za * psim, -10.0)
            psih10 = psim10
            psim2 = max(2.0 / za * psim, -10.0)
            psih2 = psim2
            if old_ust < 0.01:
                za_over_l = br * gz1
            else:
                za_over_l = (karman * c.G / thx * za * old_mol
                             / (old_ust * old_ust))
            rmol = min(za_over_l, 9.999) / za
        elif br > 0.0:
            regime = 2.0
            psim = max(-5.0 * br * gz1 / (1.1 - 5.0 * br), -10.0)
            psih = psim
            psim10 = max(10.0 / za * psim, -10.0)
            psih10 = psim10
            psim2 = max(2.0 / za * psim, -10.0)
            psih2 = psim2
            zol = br * gz1 / (1.00001 - 5.0 * br)
            if zol > 0.5:
                zol = ((1.89 * gz1 + 44.2) * br * br
                       + (1.18 * gz1 - 1.37) * br)
                zol = min(zol, 9.999)
            rmol = zol / za
        elif br == 0.0:
            regime = 3.0
            if old_ust < 0.01:
                zol = br * gz1
            else:
                zol = karman * c.G / thx * za * old_mol / old_ust ** 2
            rmol = zol / za
        else:
            regime = 4.0
            if old_ust < 0.01:
                zol = br * gz1
            else:
                zol = karman * c.G / thx * za * old_mol / old_ust ** 2
            zol10 = np.clip(10.0 / za * zol, -9.9999, 0.0)
            zol2 = np.clip(2.0 / za * zol, -9.9999, 0.0)
            zol = np.clip(zol, -9.9999, 0.0)
            psim = _sf_classic_table(_sf_psim_classic_full, zol)
            psih = _sf_classic_table(_sf_psih_classic_full, zol)
            psim10 = _sf_classic_table(_sf_psim_classic_full, zol10)
            psih10 = _sf_classic_table(_sf_psih_classic_full, zol10)
            psim2 = _sf_classic_table(_sf_psim_classic_full, zol2)
            psih2 = _sf_classic_table(_sf_psih_classic_full, zol2)
            psih = min(psih, 0.9 * gz1)
            psim = min(psim, 0.9 * gz1)
            psih2 = min(psih2, 0.9 * gz2)
            psim10 = min(psim10, 0.9 * gz10)
            psih10 = min(psih10, 0.9 * gz10)
            rmol = zol / za
    else:
        zol = 0.0
        if br > 0.0:
            zol = _sf_zolri(min(br, 250.0), za, znt)
        elif br < 0.0:
            zol = (br * gz1 if old_ust < 0.001
                   else _sf_zolri(max(br, -250.0), za, znt))
        zeta_z = zol * (za + znt) / za
        zeta10 = zol * (10.0 + znt) / za
        zeta2 = zol * (2.0 + znt) / za
        zeta0 = zol * znt / za
        scalar_rough_zeta = zol * (0.01 / za) if land else zeta0
        if br > 0.0:
            regime = 1.0
            psim = _sf_psim_stable(zeta_z) - _sf_psim_stable(zeta0)
            psih = _sf_psih_stable(zeta_z) - _sf_psih_stable(zeta0)
            psim10 = _sf_psim_stable(zeta10) - _sf_psim_stable(zeta0)
            psih10 = _sf_psih_stable(zeta10) - _sf_psih_stable(zeta0)
            psim2 = _sf_psim_stable(zeta2) - _sf_psim_stable(zeta0)
            psih2 = _sf_psih_stable(zeta2) - _sf_psih_stable(zeta0)
            pq = _sf_psih_stable(zol) - _sf_psih_stable(scalar_rough_zeta)
            pq2 = (_sf_psih_stable(2.0 / za * zol)
                   - _sf_psih_stable(scalar_rough_zeta))
            pq10 = (_sf_psih_stable(10.0 / za * zol)
                    - _sf_psih_stable(scalar_rough_zeta))
        elif br == 0.0:
            regime = 3.0
            zol = 0.0
        else:
            regime = 4.0
            psim = _sf_psim_unstable(zeta_z) - _sf_psim_unstable(zeta0)
            psih = _sf_psih_unstable(zeta_z) - _sf_psih_unstable(zeta0)
            psim10 = (_sf_psim_unstable(zeta10)
                      - _sf_psim_unstable(zeta0))
            psih10 = (_sf_psih_unstable(zeta10)
                      - _sf_psih_unstable(zeta0))
            psim2 = _sf_psim_unstable(zeta2) - _sf_psim_unstable(zeta0)
            psih2 = _sf_psih_unstable(zeta2) - _sf_psih_unstable(zeta0)
            pq = (_sf_psih_unstable(zol)
                  - _sf_psih_unstable(scalar_rough_zeta))
            pq2 = (_sf_psih_unstable(2.0 / za * zol)
                   - _sf_psih_unstable(scalar_rough_zeta))
            pq10 = (_sf_psih_unstable(10.0 / za * zol)
                    - _sf_psih_unstable(scalar_rough_zeta))
            psih = min(psih, 0.9 * gz1)
            psim = min(psim, 0.9 * gz1)
            psih2 = min(psih2, 0.9 * gz2)
            psim10 = min(psim10, 0.9 * gz10)
            psih10 = min(psih10, 0.9 * gz10)
        rmol = zol / za

    dtg = thx - thgb
    psix = gz1 - psim
    psix10 = gz10 - psim10
    psit = max(gz1 - psih, 2.0) if option == 91 else gz1 - psih
    psit2 = gz2 - psih2
    zl = 0.01 if land else znt
    psiq = np.log(karman * old_ust * za / xka + za / zl) - (psih if option == 91 else pq)
    psiq2 = np.log(karman * old_ust * 2.0 / xka + 2.0 / zl) - (psih2 if option == 91 else pq2)
    psiq10 = np.log(karman * old_ust * 10.0 / xka + 10.0 / zl) - (psih10 if option == 91 else pq10)

    # Fairall (2003) scalar roughness over water, present in both files.
    if not land:
        visc = (1.32 + 0.009 * (temp - 273.15)) * 1.0e-5
        restar = old_ust * znt / visc
        z0t = np.clip(5.5e-5 * restar ** (-0.60), 2.0e-9, 1.0e-4)
        if option == 91:
            psiq = max(np.log((za + z0t) / z0t) - psih, 2.0)
            psit = max(np.log((za + z0t) / z0t) - psih, 2.0)
            psiq2 = max(np.log((2.0 + z0t) / z0t) - psih2, 2.0)
            psit2 = max(np.log((2.0 + z0t) / z0t) - psih2, 2.0)
            psiq10 = max(np.log((10.0 + z0t) / z0t) - psih10, 2.0)
        else:
            psih_t = _sf_rev_heat_psi(zol, za, z0t, za)
            psih_t2 = _sf_rev_heat_psi(zol, za, z0t, 2.0)
            psih_t10 = _sf_rev_heat_psi(zol, za, z0t, 10.0)
            # The revised file reuses its PSIH/PSIH2/PSIH10 inout arrays
            # for the scalar roughness corrections (lines 545-580).
            psih, psih2, psih10 = psih_t, psih_t2, psih_t10
            psit = np.log((za + z0t) / z0t) - psih_t
            psit2 = np.log((2.0 + z0t) / z0t) - psih_t2
            psiq = psit
            psiq2 = psit2
            psiq10 = np.log((10.0 + z0t) / z0t) - psih_t10

    if isftcflx == 1 and not land:
        z0q = 1.0e-4
        if option == 91:
            psiq = np.log(za / z0q) - psih
            psiq2 = np.log(2.0 / z0q) - psih2
            psiq10 = np.log(10.0 / z0q) - psih10
        else:
            ph = _sf_rev_heat_psi(zol, za, z0q, za)
            ph2 = _sf_rev_heat_psi(zol, za, z0q, 2.0)
            ph10 = _sf_rev_heat_psi(zol, za, z0q, 10.0)
            psih, psih2, psih10 = ph, ph2, ph10
            psiq = np.log((za + z0q) / z0q) - ph
            psiq2 = np.log((2.0 + z0q) / z0q) - ph2
            psiq10 = np.log((10.0 + z0q) / z0q) - ph10
        psit, psit2 = psiq, psiq2
    elif isftcflx == 2 and not land:
        visc = (1.32 + 0.009 * (temp - 273.15)) * 1.0e-5
        restar = old_ust * znt / visc
        gz0t = 0.4 * (7.3 * restar ** 0.25 * np.sqrt(0.71) - 5.0)
        gz0q = 0.4 * (7.3 * restar ** 0.25 * np.sqrt(0.60) - 5.0)
        if option == 91:
            psit, psiq = gz1 - psih + gz0t, gz1 - psih + gz0q
            psit2, psiq2 = gz2 - psih2 + gz0t, gz2 - psih2 + gz0q
            psiq10 = gz10 - psih + gz0q  # file's PSIH (not PSIH10)
        else:
            z0t, z0q = znt / np.exp(gz0t), znt / np.exp(gz0q)
            psit = np.log((za + z0t) / z0t) - _sf_rev_heat_psi(zol, za, z0t, za)
            psit2 = np.log((2.0 + z0t) / z0t) - _sf_rev_heat_psi(zol, za, z0t, 2.0)
            psih = _sf_rev_heat_psi(zol, za, z0q, za)
            psih2 = _sf_rev_heat_psi(zol, za, z0q, 2.0)
            psih10 = _sf_rev_heat_psi(zol, za, z0q, 10.0)
            psiq = np.log((za + z0q) / z0q) - psih
            psiq2 = np.log((2.0 + z0q) / z0q) - psih2
            psiq10 = np.log((10.0 + z0q) / z0q) - psih10

    ck = (karman / psix10) * (karman / psiq10)
    cd = (karman / psix10) ** 2
    cka = (karman / psix) * (karman / psiq)
    cda = (karman / psix) ** 2

    if iz0tlnd >= 1 and land:
        visc = (1.32 + 0.009 * (temp - 273.15)) * 1.0e-5
        restar = old_ust * znt / visc
        czil = 10.0 ** (-0.40 * (znt / 0.07)) if iz0tlnd == 1 else 0.1
        if option == 91:
            add = czil * karman * np.sqrt(restar)
            psit = psiq = gz1 - psih + add
            psit2 = psiq2 = gz2 - psih2 + add
        else:
            z0t = znt / np.exp(czil * karman * np.sqrt(restar))
            psih = _sf_rev_heat_psi(zol, za, z0t, za)
            psih2 = _sf_rev_heat_psi(zol, za, z0t, 2.0)
            psih10 = _sf_rev_heat_psi(zol, za, z0t, 10.0)
            psiq = psit = np.log((za + z0t) / z0t) - psih
            psiq2 = psit2 = np.log((2.0 + z0t) / z0t) - psih2

    ust = 0.5 * old_ust + 0.5 * karman * wspd / psix
    # TKE coupling (module_sf_sfclay.F:800-804,
    # physics_mmm/sf_sfclayrev.F90:759-763): the same relaxation on the
    # wind speed WITHOUT vconv/vsgd and without WSPD's 0.1 floor, and with
    # none of the land floor UST takes below.
    wspdi = np.sqrt(u * u + v * v)
    ustm = 0.5 * old_ustm + 0.5 * karman * wspdi / psix
    u10, v10 = u * psix10 / psix, v * psix10 / psix
    th2 = thgb + dtg * psit2 / psit
    q2 = qsfc + (qv - qsfc) * psiq2 / psiq
    t2 = th2 * (psfc / c.P0) ** c.RCP
    if land:
        ust = max(ust, 0.1 if option == 91 else 0.001)
    mol = karman * dtg / psit
    fm, fh = psix, psit

    # Updated ocean roughness is carried to the next call; current-call
    # transfer denominators above deliberately used the incoming value.
    znt_out = znt
    if isfflx and not land:
        znt_out = min(0.0185 * ust * ust / c.G + 0.11 * 1.5e-5 / ust,
                      2.85e-3)
        if isftcflx != 0:
            zw = min((ust / 1.06) ** 0.3, 1.0)
            zn1 = 0.011 * ust * ust / c.G + 1.59e-5
            zn2 = (10.0 * np.exp(-9.5 * ust ** (-0.3333))
                   + 0.11 * 1.5e-5 / max(ust, 0.01))
            znt_out = np.clip((1.0 - zw) * zn1 + zw * zn2,
                              1.27e-7, 2.85e-3)

    if isfflx:
        flqc = rho * mavail * ust * karman / psiq
        if abs(thx - thgb) > 1.0e-5:
            flhc = cpm * rho * ust * mol / (thx - thgb)
        else:
            flhc = 0.0
        qfx = flqc * (qsfc - qv)
        lh = c.XLV * qfx
        hfx = flhc * (thgb - thx)
        chs = ust * karman / psiq
        cqs2 = ust * karman / psiq2
        chs2 = ust * karman / psit2
    else:
        hfx = qfx = lh = flhc = flqc = chs = cqs2 = chs2 = 0.0

    return {
        "znt": znt_out, "ust": ust, "ustm": ustm, "mol": mol, "hfx": hfx,
        "qfx": qfx, "qsfc": qsfc, "zol": zol, "regime": regime,
        "psim": psim, "psih": psih, "fm": fm, "fh": fh, "lh": lh,
        "u10": u10, "v10": v10, "th2": th2, "t2": t2, "q2": q2,
        "chs": chs, "chs2": chs2, "cqs2": cqs2, "flhc": flhc,
        "flqc": flqc, "qgh": qgh, "rmol": rmol, "wspd": wspd,
        "br": br, "gz1oz0": gz1, "cpm": cpm, "ck": ck, "cka": cka,
        "cd": cd, "cda": cda, "theta_air": thx,
        "theta_ground": thgb,
    }


def np_sfclay(u, v, t, qv, p, dz8w, psfc, tsk, znt, pblh, mavail,
              xland, *, option=91, qsfc=None, zol=None, ust=None, ustm=None,
              mol=None, hfx=None, qfx=None, lakemask=None, dx=1000.0,
              isfflx=True, isftcflx=0, iz0tlnd=0):
    """Float64 mirror of :mod:`woof.core.sfclay` and ``sfclay.cu``.

    Inputs are broadcast-compatible surface arrays (normally ``(ny,nx)``)
    containing WRF's lowest mass-level wind, temperature, vapor, pressure,
    layer depth and surface/LSM fields.  ``xland`` follows WRF (1 land,
    2 water).  Incoming ``ust``/``mol`` and heat/moisture fluxes are the
    previous-step values used by the WRF stability iteration, as is the
    incoming ``ustm`` the TKE-coupling relaxation averages against
    (Registry.EM_COMMON:1954 state, cold-started at zero).  Incoming
    ``zol`` is also preserved by classic option 91's strong-stable branch.
    The returned dictionary contains float64 arrays for every device result
    plus ``theta_air``/``theta_ground`` verification intermediates.
    """
    if option not in (1, 91):
        raise ValueError(f"sfclay option must be 1 or 91, got {option}")
    if isftcflx not in (0, 1, 2):
        raise ValueError(f"isftcflx must be 0, 1, or 2, got {isftcflx}")
    if iz0tlnd not in (0, 1, 2):
        raise ValueError(f"iz0tlnd must be 0, 1, or 2, got {iz0tlnd}")

    base = [np.asarray(a, dtype=np.float64)
            for a in (u, v, t, qv, p, dz8w, psfc, tsk, znt, pblh,
                      mavail, xland)]
    shape = np.broadcast_shapes(*(a.shape for a in base))

    def arr(value, default):
        if value is None:
            value = default
        return np.broadcast_to(np.asarray(value, dtype=np.float64), shape)

    values = [arr(a, 0.0) for a in base]
    extras = [arr(qsfc, 0.0), arr(zol, 0.0), arr(ust, 0.1),
              arr(ustm, 0.0), arr(mol, 0.0),
              arr(hfx, 0.0), arr(qfx, 0.0), arr(lakemask, 0.0)]
    result = {name: np.empty(shape, dtype=np.float64)
              for name in _SFCLAY_OUTPUTS + ("theta_air", "theta_ground")}
    for index in np.ndindex(shape):
        point = _np_sfclay_point(
            *(a[index] for a in values), *(a[index] for a in extras),
            float(dx), int(option), bool(isfflx), int(isftcflx),
            int(iz0tlnd))
        for name, value in point.items():
            result[name][index] = value
    return result


# ===========================================================================
# Noah LSM float64 mirror (Phase 3 Task 10)
# ===========================================================================
# Transcribed line-faithfully from WRF v4.6.1 phys/module_sf_noahdrv.F
# (subroutine lsm: per-column prep + post-SFLX updates) and
# phys/module_sf_noahlsm.F (SFLX and its full subtree).  One call
# integrates ONE column in float64; woof/core/kernels/noah.cu is the
# FP32 device twin.  UA_PHYS / FASDAS / WRF_HYDRO / urban canopy /
# SFCDIF_off are not ported (see woof/core/noah.py); SFLX_GLACIAL is
# (_noah_glacial, since 2026-10-12).
#
# Constant discipline: module_sf_noahlsm's OWN parameters are kept as the
# file literals below (RD = 287.04, SIGMA = 5.67e-8, PENMAN's local
# CP = 1004.6, ...); quantities the Fortran takes from
# module_model_constants (CP = 1004.5, STBOLT, XLF, XLV) come from
# woof.core.constants so they are never hardcoded twice.

_NRD = 287.04             # noahlsm module parameter RD
_NSIGMA = 5.67e-8         # noahlsm SIGMA
_NCPH2O = 4.218e3         # noahlsm CPH2O
_NCPICE = 2.106e3         # noahlsm CPICE
_NLSUBF = 3.335e5         # noahlsm LSUBF
_NEMISSI_S = 0.95         # noahlsm EMISSI_S (snow emissivity)
_NTFREEZ = 273.15         # SFLX TFREEZ
_NLVH2O = 2.501e6         # SFLX LVH2O
_NLSUBS = 2.83e6          # SFLX/PENMAN/SNOPAC LSUBS
_NR = 287.04              # SFLX local R (SHEAT denominator)
_NCP_PEN = 1004.6         # PENMAN's local CP
_NELCP = 2.4888e3         # PENMAN ELCP
_NLSUBC = 2.501e6         # PENMAN/SNOPAC LSUBC
_NCP = c.CP               # module_model_constants CP = 1004.5
_NSTBOLT = 5.67051e-8     # module_model_constants STBOLT (NOAHRES)
_NXLF = 3.50e5            # module_model_constants XLF (SNOPCX)
_NXLV = 2.5e6             # module_model_constants XLV (POTEVP)


from woof.core.noah import noah_frh2o as np_noah_frh2o


def _noah_csnow(dsnow):
    """CSNOW: snow thermal conductivity (doubled Dyachkova 1960)."""
    return 2.0 * 0.11631 * (0.328 * 10.0 ** (2.25 * dsnow))


def _noah_snow_new(temp, newsn, snowh, sndens):
    """SNOW_NEW: depth/density update for fresh snowfall."""
    snowhc = snowh * 100.0
    newsnc = newsn * 100.0
    tempc = temp - 273.15
    if tempc <= -15.0:
        dsnew = 0.05
    else:
        dsnew = 0.05 + 0.0017 * (tempc + 15.0) ** 1.5
    hnewc = newsnc / dsnew
    if snowhc + hnewc < 1.0e-3:
        sndens = max(dsnew, sndens)
    else:
        sndens = (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc)
    snowhc = snowhc + hnewc
    return snowhc * 0.01, sndens


def _noah_snfrac(sneqv, snup, salp):
    """SNFRAC (non-UA): fractional snow cover."""
    if sneqv < snup:
        rsnow = sneqv / snup
        return 1.0 - (math.exp(-salp * rsnow) - rsnow * math.exp(-salp))
    return 1.0


def _noah_alcalc(alb, snoalb, embrd, sncovr, dt, snowng, snotime1,
                 lvcoef):
    """ALCALC: snow albedo (Livneh aging) + emissivity blend."""
    snacca, snaccb = 0.94, 0.58
    albedo = alb + sncovr * (snoalb - alb)
    emissi = embrd + sncovr * (_NEMISSI_S - embrd)
    snoalb1 = snoalb + lvcoef * (0.85 - snoalb)
    snoalb2 = snoalb1
    if snowng:
        snotime1 = 0.0
    else:
        snotime1 = snotime1 + dt
        snoalb2 = snoalb1 * (snacca ** ((snotime1 / 86400.0) ** snaccb))
    snoalb2 = max(snoalb2, alb)
    albedo = alb + sncovr * (snoalb2 - alb)
    if albedo > snoalb2:
        albedo = snoalb2
    return albedo, emissi, snotime1


def _noah_tdfcnd(smc, qz, smcmax, sh2o, bexp, psisat, soiltyp,
                 opt_thcnd):
    """TDFCND: soil thermal conductivity (Peters-Lidard / Johansen; the
    McCumber-Pielke option-2 branch for soil types 3 and 4)."""
    if opt_thcnd == 1 or (opt_thcnd == 2 and soiltyp != 4
                          and soiltyp != 3):
        satratio = smc / smcmax
        thkice = 2.2
        thkw = 0.57
        thko = 2.0
        thkqtz = 7.7
        thks = (thkqtz ** qz) * (thko ** (1.0 - qz))
        xunfroz = sh2o / smc
        xu = xunfroz * smcmax
        thksat = (thks ** (1.0 - smcmax) * thkice ** (smcmax - xu)
                  * thkw ** xu)
        gammd = (1.0 - smcmax) * 2700.0
        thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
        akei = satratio
        if satratio > 0.1:
            akel = math.log10(satratio) + 1.0
        else:
            akel = 0.0
        ake = ((smc - sh2o) * akei + sh2o * akel) / smc
        return ake * (thksat - thkdry) + thkdry
    psif = psisat * 100.0 * (smcmax / smc) ** bexp
    pf = math.log10(abs(psif))
    if pf <= 5.1:
        return 420.0 * math.exp(-(pf + 2.7))
    return 0.1744


def _noah_snowz0(sncovr, z0brd, snowh):
    """SNOWZ0 (non-UA): roughness length over (partial) snow."""
    z0s = 0.001
    burial = 7.0 * z0brd - snowh
    if burial <= 0.0007:
        z0eff = z0s
    else:
        z0eff = burial / 7.0
    return (1.0 - sncovr) * z0brd + sncovr * z0eff


def _noah_penman(sfctmp, sfcprs, ch, t2v, th2, prcp, fdown, ssoil, q2,
                 q2sat, dqsdt2, snowng, frzgra, emissi_in, sncovr):
    """PENMAN: potential evaporation + partial sums (AOASIS = 1)."""
    emissi = emissi_in
    elcp1 = (1.0 - sncovr) * _NELCP + sncovr * _NELCP * _NLSUBS / _NLSUBC
    lvs = (1.0 - sncovr) * _NLSUBC + sncovr * _NLSUBS
    flx2 = 0.0
    delta = elcp1 * dqsdt2
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    rr = emissi * t24 * 6.48e-8 / (sfcprs * ch) + 1.0
    rho = sfcprs / (_NRD * t2v)
    rch = rho * _NCP_PEN * ch
    if not snowng:
        if prcp > 0.0:
            rr = rr + _NCPH2O * prcp / rch
    else:
        rr = rr + _NCPICE * prcp / rch
    fnet = fdown - emissi * _NSIGMA * t24 - ssoil
    if frzgra:
        flx2 = -_NLSUBF * prcp
        fnet = fnet - flx2
    rad = fnet / rch + th2 - sfctmp
    a = elcp1 * (q2sat - q2)
    epsca = (a * rr + rad * delta) / (delta + rr)
    # Fei-Mike AOASIS oasis factor: 1.0 without the urban model
    etp = epsca * rch / lvs
    return etp, rch, rr, epsca, t24, flx2, lvs


def _noah_canres(solar, ch, sfctmp, q2, sh2o, zsoil, nsoil, smcwlt,
                 smcref, rsmin, nroot, q2sat, dqsdt2, topt, rsmax, rgl,
                 hs, xlai, sfcprs, emissi):
    """CANRES: Jarvis canopy resistance -> plant coefficient PC."""
    slv = 2.501000e6
    ff = 0.55 * 2.0 * solar / (rgl * xlai)
    rcs = (ff + rsmin / rsmax) / (1.0 + ff)
    rcs = max(rcs, 0.0001)
    rct = 1.0 - 0.0016 * ((topt - sfctmp) ** 2.0)
    rct = max(rct, 0.0001)
    rcq = 1.0 / (1.0 + hs * (q2sat - q2))
    rcq = max(rcq, 0.01)
    rcsoil = 0.0
    gx = (sh2o[0] - smcwlt) / (smcref - smcwlt)
    gx = min(max(gx, 0.0), 1.0)
    part = [0.0] * nsoil
    part[0] = (zsoil[0] / zsoil[nroot - 1]) * gx
    for k in range(1, nroot):
        gx = (sh2o[k] - smcwlt) / (smcref - smcwlt)
        gx = min(max(gx, 0.0), 1.0)
        part[k] = ((zsoil[k] - zsoil[k - 1]) / zsoil[nroot - 1]) * gx
    for k in range(nroot):
        rcsoil = rcsoil + part[k]
    rcsoil = max(rcsoil, 0.0001)
    rc = rsmin / (xlai * rcs * rct * rcq * rcsoil)
    rr = ((4.0 * emissi * _NSIGMA * _NRD / _NCP) * (sfctmp ** 4.0)
          / (sfcprs * ch) + 1.0)
    delta = (slv / _NCP) * dqsdt2
    pc = (rr + delta) / (rr * (1.0 + rc * ch) + delta)
    return rc, pc


def _noah_devap(etp1, smc, shdfac, smcmax, smcdry, fxexp):
    """DEVAP: direct soil evaporation."""
    sratio = (smc - smcdry) / (smcmax - smcdry)
    if sratio > 0.0:
        fx = sratio ** fxexp
        fx = max(min(fx, 1.0), 0.0)
    else:
        fx = 0.0
    return fx * (1.0 - shdfac) * etp1


def _noah_transp(nsoil, etp1, sh2o, cmc, shdfac, smcwlt, cmcmax, pc,
                 cfactr, smcref, nroot, rtdis):
    """TRANSP: per-layer plant transpiration."""
    et = [0.0] * nsoil
    if cmc != 0.0:
        etp1a = shdfac * pc * etp1 * (1.0 - (cmc / cmcmax) ** cfactr)
    else:
        etp1a = shdfac * pc * etp1
    gx = [0.0] * nroot
    sgx = 0.0
    for i in range(nroot):
        gx[i] = (sh2o[i] - smcwlt) / (smcref - smcwlt)
        gx[i] = max(min(gx[i], 1.0), 0.0)
        sgx = sgx + gx[i]
    sgx = sgx / nroot
    denom = 0.0
    for i in range(nroot):
        rtx = rtdis[i] + gx[i] - sgx
        gx[i] = gx[i] * max(rtx, 0.0)
        denom = denom + gx[i]
    if denom <= 0.0:
        denom = 1.0
    for i in range(nroot):
        et[i] = etp1a * gx[i] / denom
    return et


def _noah_evapo(smc, nsoil, cmc, etp1, dt, sh2o, smcmax, pc, smcwlt,
                smcref, shdfac, cmcmax, smcdry, cfactr, nroot, rtdis,
                fxexp):
    """EVAPO: direct evaporation + transpiration + canopy evaporation."""
    edir = 0.0
    ec = 0.0
    ett = 0.0
    et = [0.0] * nsoil
    if etp1 > 0.0:
        if shdfac < 1.0:
            edir = _noah_devap(etp1, smc[0], shdfac, smcmax, smcdry,
                               fxexp)
        if shdfac > 0.0:
            et = _noah_transp(nsoil, etp1, sh2o, cmc, shdfac, smcwlt,
                              cmcmax, pc, cfactr, smcref, nroot, rtdis)
            for k in range(nsoil):
                ett = ett + et[k]
            if cmc > 0.0:
                ec = shdfac * ((cmc / cmcmax) ** cfactr) * etp1
            else:
                ec = 0.0
            cmc2ms = cmc / dt
            ec = min(cmc2ms, ec)
    eta1 = edir + ett + ec
    return eta1, edir, ec, et, ett


def _noah_fac2mit(smcmax):
    """FAC2MIT: instability limit for the double SRT/SSTEP pass."""
    flimit = 0.90
    if smcmax == 0.395:
        flimit = 0.59
    elif smcmax == 0.434 or smcmax == 0.404:
        flimit = 0.85
    elif smcmax == 0.465 or smcmax == 0.406:
        flimit = 0.86
    elif smcmax == 0.476 or smcmax == 0.439:
        flimit = 0.74
    elif smcmax == 0.200 or smcmax == 0.464:
        flimit = 0.80
    return flimit


def _noah_wdfcnd(smc, smcmax, bexp, dksat, dwsat, sicemax):
    """WDFCND: soil water diffusivity + hydraulic conductivity."""
    factr1 = 0.05 / smcmax
    factr2 = smc / smcmax
    factr1 = min(factr1, factr2)
    expon = bexp + 2.0
    wdf = dwsat * factr2 ** expon
    if sicemax > 0.0:
        vkwgt = 1.0 / (1.0 + (500.0 * sicemax) ** 3.0)
        wdf = vkwgt * wdf + (1.0 - vkwgt) * dwsat * factr1 ** expon
    expon = (2.0 * bexp) + 3.0
    wcnd = dksat * factr2 ** expon
    return wdf, wcnd


def _noah_srt(edir, et, sh2o, sh2oa, nsoil, pcpdrp, zsoil, dwsat,
              dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice):
    """SRT: RHS + tridiagonal coefficients of the soil-water equation."""
    cvfrz = 3
    rhstt = [0.0] * nsoil
    ai = [0.0] * nsoil
    bi = [0.0] * nsoil
    ci = [0.0] * nsoil
    dmax = [0.0] * nsoil
    sicemax = 0.0
    for ks in range(nsoil):
        if sice[ks] > sicemax:
            sicemax = sice[ks]
    pddum = pcpdrp
    runoff1 = 0.0
    runoff2 = 0.0
    if pcpdrp != 0.0:
        dt1 = dt / 86400.0
        smcav = smcmax - smcwlt
        dmax[0] = -zsoil[0] * smcav
        dice = -zsoil[0] * sice[0]
        dmax[0] = dmax[0] * (1.0 - (sh2oa[0] + sice[0] - smcwlt)
                             / smcav)
        dd = dmax[0]
        for ks in range(1, nsoil):
            dice = dice + (zsoil[ks - 1] - zsoil[ks]) * sice[ks]
            dmax[ks] = (zsoil[ks - 1] - zsoil[ks]) * smcav
            dmax[ks] = dmax[ks] * (1.0 - (sh2oa[ks] + sice[ks]
                                          - smcwlt) / smcav)
            dd = dd + dmax[ks]
        val = 1.0 - math.exp(-kdt * dt1)
        ddt = dd * val
        px = pcpdrp * dt
        if px < 0.0:
            px = 0.0
        infmax = (px * (ddt / (px + ddt))) / dt
        fcr = 1.0
        if dice > 1.0e-2:
            acrt = cvfrz * frzx / dice
            ssum = 1.0
            ialp1 = cvfrz - 1
            for j in range(1, ialp1 + 1):
                k = 1
                for jj in range(j + 1, ialp1 + 1):
                    k = k * jj
                ssum = ssum + (acrt ** (cvfrz - j)) / float(k)
            fcr = 1.0 - math.exp(-acrt) * ssum
        infmax = infmax * fcr
        mxsmc = sh2oa[0]
        wdf, wcnd = _noah_wdfcnd(mxsmc, smcmax, bexp, dksat, dwsat,
                                 sicemax)
        infmax = max(infmax, wcnd)
        infmax = min(infmax, px / dt)
        if pcpdrp > infmax:
            runoff1 = pcpdrp - infmax
            pddum = infmax
    mxsmc = sh2oa[0]
    wdf, wcnd = _noah_wdfcnd(mxsmc, smcmax, bexp, dksat, dwsat, sicemax)
    ddz = 1.0 / (-0.5 * zsoil[1])
    ai[0] = 0.0
    bi[0] = wdf * ddz / (-zsoil[0])
    ci[0] = -bi[0]
    dsmdz = (sh2o[0] - sh2o[1]) / (-0.5 * zsoil[1])
    rhstt[0] = (wdf * dsmdz + wcnd - pddum + edir + et[0]) / zsoil[0]
    ddz2 = 0.0
    for k in range(1, nsoil):
        denom2 = zsoil[k - 1] - zsoil[k]
        if k != nsoil - 1:
            slopx = 1.0
            mxsmc2 = sh2oa[k]
            wdf2, wcnd2 = _noah_wdfcnd(mxsmc2, smcmax, bexp, dksat,
                                       dwsat, sicemax)
            denom = zsoil[k - 1] - zsoil[k + 1]
            dsmdz2 = (sh2o[k] - sh2o[k + 1]) / (denom * 0.5)
            ddz2 = 2.0 / denom
            ci[k] = -wdf2 * ddz2 / denom2
        else:
            slopx = slope
            wdf2, wcnd2 = _noah_wdfcnd(sh2oa[nsoil - 1], smcmax, bexp,
                                       dksat, dwsat, sicemax)
            dsmdz2 = 0.0
            ci[k] = 0.0
        numer = (wdf2 * dsmdz2 + slopx * wcnd2 - wdf * dsmdz - wcnd
                 + et[k])
        rhstt[k] = numer / (-denom2)
        ai[k] = -wdf * ddz / denom2
        bi[k] = -(ai[k] + ci[k])
        if k == nsoil - 1:
            runoff2 = slopx * wcnd2
        if k != nsoil - 1:
            wdf = wdf2
            wcnd = wcnd2
            dsmdz = dsmdz2
            ddz = ddz2
    return rhstt, runoff1, runoff2, ai, bi, ci


def _noah_rosr12(a, b, c_, d, nsoil):
    """ROSR12: tridiagonal solve; returns P (the solution increment)."""
    p = [0.0] * nsoil
    delta = [0.0] * nsoil
    c_ = list(c_)
    c_[nsoil - 1] = 0.0
    p[0] = -c_[0] / b[0]
    delta[0] = d[0] / b[0]
    for k in range(1, nsoil):
        p[k] = -c_[k] * (1.0 / (b[k] + a[k] * p[k - 1]))
        delta[k] = ((d[k] - a[k] * delta[k - 1])
                    * (1.0 / (b[k] + a[k] * p[k - 1])))
    p[nsoil - 1] = delta[nsoil - 1]
    for k in range(1, nsoil):
        kk = nsoil - k - 1
        p[kk] = p[kk] * p[kk + 1] + delta[kk]
    return p


def _noah_sstep(sh2oin, cmc, rhstt, rhsct, dt, nsoil, smcmax, cmcmax,
                zsoil, smc, sice, ai, bi, ci, flags):
    """SSTEP: advance SH2O/SMC/CMC with the tridiagonal solution.
    Mutates ``smc``; returns (sh2oout, cmc, runoff3)."""
    rhstt = [r * dt for r in rhstt]
    ai = [x * dt for x in ai]
    bi = [1.0 + x * dt for x in bi]
    ci = [x * dt for x in ci]
    p = _noah_rosr12(ai, bi, ci, list(rhstt), nsoil)
    sh2oout = [0.0] * nsoil
    wplus = 0.0
    ddz = -zsoil[0]
    for k in range(nsoil):
        if k != 0:
            ddz = zsoil[k - 1] - zsoil[k]
        sh2oout[k] = sh2oin[k] + p[k] + wplus / ddz
        stot = sh2oout[k] + sice[k]
        if stot > smcmax:
            if k == 0:
                ddz = -zsoil[0]
            else:
                ddz = -zsoil[k] + zsoil[k - 1]
            wplus = (stot - smcmax) * ddz
        else:
            wplus = 0.0
        if stot < 0.02 or (min(stot, smcmax) - sice[k]) < 0.0:
            flags["clamped"] = True      # the 0.02/0.0 floors create water
        smc[k] = max(min(stot, smcmax), 0.02)
        sh2oout[k] = max(smc[k] - sice[k], 0.0)
    runoff3 = wplus
    cmc = cmc + dt * rhsct
    if cmc < 1.0e-20:
        cmc = 0.0
    cmc = min(cmc, cmcmax)
    return sh2oout, cmc, runoff3


def _noah_smflx(smc, nsoil, cmc, dt, prcp1, zsoil, sh2o, slope, kdt,
                frzfact, smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
                cmcmax, edir, ec, et, flags):
    """SMFLX: canopy water + SRT/SSTEP (single or Kalnay-Kanamitsu
    double pass).  Mutates ``smc``/``sh2o``; returns
    (cmc, runoff1, runoff2, runoff3, drip)."""
    dummy = 0.0
    rhsct = shdfac * prcp1 - ec
    drip = 0.0
    trhsct = dt * rhsct
    excess = cmc + trhsct
    if excess > cmcmax:
        drip = excess - cmcmax
    pcpdrp = (1.0 - shdfac) * prcp1 + drip / dt
    sice = [smc[i] - sh2o[i] for i in range(nsoil)]
    fac2 = 0.0
    for i in range(nsoil):
        fac2 = max(fac2, sh2o[i] / smcmax)
    flimit = _noah_fac2mit(smcmax)
    thresh = 0.0001 * 1000.0 * (-zsoil[0]) * smcmax
    if (abs(pcpdrp * dt - thresh) < 1e-3 * thresh
            or abs(fac2 - flimit) < 5e-4):
        flags["near_branch"] = True
    if (pcpdrp * dt) > thresh or fac2 > flimit:
        rhstt, runoff1, runoff2, ai, bi, ci = _noah_srt(
            edir, et, sh2o, sh2o, nsoil, pcpdrp, zsoil, dwsat, dksat,
            smcmax, bexp, dt, smcwlt, slope, kdt, frzfact, sice)
        sh2ofg, _, runoff3 = _noah_sstep(
            sh2o, dummy, rhstt, rhsct, dt, nsoil, smcmax, cmcmax,
            zsoil, smc, sice, ai, bi, ci, flags)
        sh2oa = [(sh2o[k] + sh2ofg[k]) * 0.5 for k in range(nsoil)]
        rhstt, runoff1, runoff2, ai, bi, ci = _noah_srt(
            edir, et, sh2o, sh2oa, nsoil, pcpdrp, zsoil, dwsat, dksat,
            smcmax, bexp, dt, smcwlt, slope, kdt, frzfact, sice)
        sh2onew, cmc, runoff3 = _noah_sstep(
            sh2o, cmc, rhstt, rhsct, dt, nsoil, smcmax, cmcmax, zsoil,
            smc, sice, ai, bi, ci, flags)
        sh2o[:] = sh2onew
    else:
        rhstt, runoff1, runoff2, ai, bi, ci = _noah_srt(
            edir, et, sh2o, sh2o, nsoil, pcpdrp, zsoil, dwsat, dksat,
            smcmax, bexp, dt, smcwlt, slope, kdt, frzfact, sice)
        sh2onew, cmc, runoff3 = _noah_sstep(
            sh2o, cmc, rhstt, rhsct, dt, nsoil, smcmax, cmcmax, zsoil,
            smc, sice, ai, bi, ci, flags)
        sh2o[:] = sh2onew
    return cmc, runoff1, runoff2, runoff3, drip


def _noah_tbnd(tu, tb, zsoil, zbot, k, nsoil):
    """TBND: temperature at the bottom interface of layer k (0-based)."""
    if k == 0:
        zup = 0.0
    else:
        zup = zsoil[k - 1]
    if k == nsoil - 1:
        zb = 2.0 * zbot - zsoil[k]
    else:
        zb = zsoil[k + 1]
    return tu + (tb - tu) * (zup - zsoil[k]) / (zup - zb)


def _noah_tmpavg(tup, tm, tdn, zsoil, k):
    """TMPAVG: freezing-aware average layer temperature (k 0-based)."""
    t0 = 2.7315e2
    if k == 0:
        dz = -zsoil[0]
    else:
        dz = zsoil[k - 1] - zsoil[k]
    dzh = dz * 0.5
    if tup < t0:
        if tm < t0:
            if tdn < t0:
                return (tup + 2.0 * tm + tdn) / 4.0
            x0 = (t0 - tm) * dzh / (tdn - tm)
            return 0.5 * (tup * dzh + tm * (dzh + x0)
                          + t0 * (2.0 * dzh - x0)) / dz
        if tdn < t0:
            xup = (t0 - tup) * dzh / (tm - tup)
            xdn = dzh - (t0 - tm) * dzh / (tdn - tm)
            return 0.5 * (tup * xup + t0 * (2.0 * dz - xup - xdn)
                          + tdn * xdn) / dz
        xup = (t0 - tup) * dzh / (tm - tup)
        return 0.5 * (tup * xup + t0 * (2.0 * dz - xup)) / dz
    if tm < t0:
        if tdn < t0:
            xup = dzh - (t0 - tup) * dzh / (tm - tup)
            return 0.5 * (t0 * (dz - xup) + tm * (dzh + xup)
                          + tdn * dzh) / dz
        xup = dzh - (t0 - tup) * dzh / (tm - tup)
        xdn = (t0 - tm) * dzh / (tdn - tm)
        return 0.5 * (t0 * (2.0 * dz - xup - xdn) + tm * (xup + xdn)) \
            / dz
    if tdn < t0:
        xdn = dzh - (t0 - tm) * dzh / (tdn - tm)
        return (t0 * (dz - xdn) + 0.5 * (t0 + tdn) * xdn) / dz
    return (tup + 2.0 * tm + tdn) / 4.0


def _noah_snksrc(qtot, tavg, smc, sh2o, zsoil, smcmax, psisat, bexp,
                 dt, k):
    """SNKSRC: freeze/thaw heat source-sink; returns (tsnsr, new sh2o)."""
    dh2o, hlice = 1.0000e3, 3.3350e5
    if k == 0:
        dz = -zsoil[0]
    else:
        dz = zsoil[k - 1] - zsoil[k]
    free = np_noah_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat)
    xh2o = sh2o + qtot * dt / (dh2o * hlice * dz)
    if xh2o < sh2o and xh2o < free:
        if free > sh2o:
            xh2o = sh2o
        else:
            xh2o = free
    if xh2o > sh2o and xh2o > free:
        if free < sh2o:
            xh2o = sh2o
        else:
            xh2o = free
    if xh2o < 0.0:
        xh2o = 0.0
    if xh2o > smc:
        xh2o = smc
    tsnsr = -dh2o * hlice * dz * (xh2o - sh2o) / dt
    return tsnsr, xh2o


def _noah_hrt(stc, smc, smcmax, nsoil, zsoil, yy, zz1, tbot, zbot,
              psisat, sh2o, dt, bexp, soiltyp, opt_thcnd, f1, df1,
              quartz, csoil, vegtyp, isurban):
    """HRT: RHS + tridiagonal coefficients of the soil heat equation
    (ITAVG = .TRUE. path); mutates SH2O through SNKSRC."""
    t0, cair, cice, ch2o = 273.15, 1004.0, 2.106e6, 4.2e6
    if vegtyp == isurban:
        csoil_loc = 3.0e6
    else:
        csoil_loc = csoil
    rhsts = [0.0] * nsoil
    ai = [0.0] * nsoil
    bi = [0.0] * nsoil
    ci = [0.0] * nsoil
    hcpct = (sh2o[0] * ch2o + (1.0 - smcmax) * csoil_loc
             + (smcmax - smc[0]) * cair + (smc[0] - sh2o[0]) * cice)
    ddz = 1.0 / (-0.5 * zsoil[1])
    ai[0] = 0.0
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct)
    bi[0] = -ci[0] + df1 / (0.5 * zsoil[0] * zsoil[0] * hcpct * zz1)
    dtsdz = (stc[0] - stc[1]) / (-0.5 * zsoil[1])
    ssoil = df1 * (stc[0] - yy) / (0.5 * zsoil[0] * zz1)
    denom = zsoil[0] * hcpct
    rhsts[0] = (df1 * dtsdz - ssoil) / denom
    qtot = -1.0 * rhsts[0] * denom
    sice = smc[0] - sh2o[0]
    tsurf = (yy + (zz1 - 1.0) * stc[0]) / zz1
    tbk = _noah_tbnd(stc[0], stc[1], zsoil, zbot, 0, nsoil)
    if sice > 0.0 or stc[0] < t0 or tsurf < t0 or tbk < t0:
        tavg = _noah_tmpavg(tsurf, stc[0], tbk, zsoil, 0)
        tsnsr, sh2o[0] = _noah_snksrc(qtot, tavg, smc[0], sh2o[0],
                                      zsoil, smcmax, psisat, bexp, dt,
                                      0)
        rhsts[0] = rhsts[0] - tsnsr / denom
    ddz2 = 0.0
    df1k = df1
    for k in range(1, nsoil):
        hcpct = (sh2o[k] * ch2o + (1.0 - smcmax) * csoil_loc
                 + (smcmax - smc[k]) * cair
                 + (smc[k] - sh2o[k]) * cice)
        if k != nsoil - 1:
            df1n = _noah_tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp,
                                psisat, soiltyp, opt_thcnd)
            if vegtyp == isurban:
                df1n = 3.24
            denom = 0.5 * (zsoil[k - 1] - zsoil[k + 1])
            dtsdz2 = (stc[k] - stc[k + 1]) / denom
            ddz2 = 2.0 / (zsoil[k - 1] - zsoil[k + 1])
            ci[k] = -df1n * ddz2 / ((zsoil[k - 1] - zsoil[k]) * hcpct)
            tbk1 = _noah_tbnd(stc[k], stc[k + 1], zsoil, zbot, k, nsoil)
        else:
            df1n = _noah_tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp,
                                psisat, soiltyp, opt_thcnd)
            if vegtyp == isurban:
                df1n = 3.24
            denom = 0.5 * (zsoil[k - 1] + zsoil[k]) - zbot
            dtsdz2 = (stc[k] - tbot) / denom
            ci[k] = 0.0
            tbk1 = _noah_tbnd(stc[k], tbot, zsoil, zbot, k, nsoil)
        denom = (zsoil[k] - zsoil[k - 1]) * hcpct
        rhsts[k] = (df1n * dtsdz2 - df1k * dtsdz) / denom
        qtot = -1.0 * denom * rhsts[k]
        sice = smc[k] - sh2o[k]
        tavg = _noah_tmpavg(tbk, stc[k], tbk1, zsoil, k)
        if sice > 0.0 or stc[k] < t0 or tbk < t0 or tbk1 < t0:
            tsnsr, sh2o[k] = _noah_snksrc(qtot, tavg, smc[k], sh2o[k],
                                          zsoil, smcmax, psisat, bexp,
                                          dt, k)
            rhsts[k] = rhsts[k] - tsnsr / denom
        ai[k] = -df1k * ddz / ((zsoil[k - 1] - zsoil[k]) * hcpct)
        bi[k] = -(ai[k] + ci[k])
        tbk = tbk1
        df1k = df1n
        dtsdz = dtsdz2
        ddz = ddz2
    return rhsts, ai, bi, ci


def _noah_hstep(stcin, rhsts, dt, nsoil, ai, bi, ci):
    """HSTEP: advance the soil temperatures."""
    rhsts = [r * dt for r in rhsts]
    ai = [x * dt for x in ai]
    bi = [1.0 + x * dt for x in bi]
    ci = [x * dt for x in ci]
    p = _noah_rosr12(ai, bi, ci, list(rhsts), nsoil)
    return [stcin[k] + p[k] for k in range(nsoil)]


def _noah_shflx(stc, smc, smcmax, nsoil, t1, dt, yy, zz1, zsoil, tbot,
                zbot, psisat, sh2o, bexp, f1, df1, quartz, csoil,
                vegtyp, isurban, soiltyp, opt_thcnd):
    """SHFLX: update STC (in place), then skin temperature + soil heat
    flux; returns (t1, ssoil)."""
    rhsts, ai, bi, ci = _noah_hrt(stc, smc, smcmax, nsoil, zsoil, yy,
                                  zz1, tbot, zbot, psisat, sh2o, dt,
                                  bexp, soiltyp, opt_thcnd, f1, df1,
                                  quartz, csoil, vegtyp, isurban)
    stcf = _noah_hstep(stc, rhsts, dt, nsoil, ai, bi, ci)
    stc[:] = stcf
    t1 = (yy + (zz1 - 1.0) * stc[0]) / zz1
    ssoil = df1 * (stc[0] - t1) / (0.5 * zsoil[0])
    return t1, ssoil


def _noah_snowpack(esd, dtsec, snowh, sndens, tsnow, tsoil, ua_phys=False, snomlt=0.):
    """SNOWPACK: compaction (Koren polynomial) + melt-season densifying."""
    c1k, c2k = 0.01, 21.0
    snowhc = snowh * 100.0
    esdc = esd * 100.0
    dthr = dtsec / 3600.0
    tsnowc = tsnow - 273.15
    tsoilc = tsoil - 273.15
    tavgc = 0.5 * (tsnowc + tsoilc)
    if esdc > 1.0e-2:
        esdcx = esdc
    else:
        esdcx = 1.0e-2
    bfac = dthr * c1k * math.exp(0.08 * tavgc - c2k * sndens)
    ipol = 4
    pexp = 0.0
    for j in range(ipol, 0, -1):
        pexp = (1.0 + pexp) * bfac * esdcx / float(j + 1)
    pexp = pexp + 1.0
    dsx = sndens * pexp
    if dsx > 0.40:
        dsx = 0.40
    if dsx < 0.05:
        dsx = 0.05
    sndens = dsx
    if tsnowc >= 0.0:
        dw = 0.13 * dthr / 24.0
        if ua_phys and tsoilc>=0.:
            cm=snomlt*100.
            dw=min(dw,0.13*cm/(esdcx+0.13*cm))
        sndens = sndens * (1.0 - dw) + dw
        if sndens >= 0.40:
            sndens = 0.40
    snowhc = esdc / sndens
    return snowhc * 0.01, sndens


def _noah_nopac(etp, prcp, smc, smcmax, smcwlt, smcref, smcdry, cmc,
                cmcmax, nsoil, dt, shdfac, sbeta, q2, t1, sfctmp, t24,
                th2, fdown, f1, emissi, stc, epsca, bexp, pc, rch, rr,
                cfactr, sh2o, slope, kdt, frzfact, psisat, zsoil,
                dksat, dwsat, tbot, zbot, nroot, rtdis, quartz, fxexp,
                csoil, vegtyp, isurban, soiltyp, opt_thcnd, flags):
    """NOPAC: snow-free land update.  Mutates smc/sh2o/stc; returns a
    dict of the Fortran outputs (evap components still kinematic)."""
    prcp1 = prcp * 0.001
    etp1 = etp * 0.001
    dew = 0.0
    edir1 = 0.0
    ec1 = 0.0
    et1 = [0.0] * nsoil
    ett1 = 0.0
    eta = 0.0
    if etp > 0.0:
        eta1, edir1, ec1, et1, ett1 = _noah_evapo(
            smc, nsoil, cmc, etp1, dt, sh2o, smcmax, pc, smcwlt,
            smcref, shdfac, cmcmax, smcdry, cfactr, nroot, rtdis,
            fxexp)
        cmc, runoff1, runoff2, runoff3, drip = _noah_smflx(
            smc, nsoil, cmc, dt, prcp1, zsoil, sh2o, slope, kdt,
            frzfact, smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
            cmcmax, edir1, ec1, et1, flags)
        eta = eta1 * 1000.0
    else:
        dew = -etp1
        prcp1 = prcp1 + dew
        cmc, runoff1, runoff2, runoff3, drip = _noah_smflx(
            smc, nsoil, cmc, dt, prcp1, zsoil, sh2o, slope, kdt,
            frzfact, smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
            cmcmax, edir1, ec1, et1, flags)
    if etp <= 0.0:
        beta = 0.0
        eta = etp
        if etp < 0.0:
            beta = 1.0
    else:
        beta = eta / etp
    edir = edir1 * 1000.0
    ec = ec1 * 1000.0
    et = [x * 1000.0 for x in et1]
    ett = ett1 * 1000.0
    df1 = _noah_tdfcnd(smc[0], quartz, smcmax, sh2o[0], bexp, psisat,
                       soiltyp, opt_thcnd)
    if vegtyp == isurban:
        df1 = 3.24
    df1 = df1 * math.exp(sbeta * shdfac)
    yynum = fdown - emissi * _NSIGMA * t24
    yy = sfctmp + (yynum / rch + th2 - sfctmp - beta * epsca) / rr
    zz1 = df1 / (-0.5 * zsoil[0] * rch * rr) + 1.0
    t1, ssoil = _noah_shflx(stc, smc, smcmax, nsoil, t1, dt, yy, zz1,
                            zsoil, tbot, zbot, psisat, sh2o, bexp, f1,
                            df1, quartz, csoil, vegtyp, isurban,
                            soiltyp, opt_thcnd)
    flx1 = _NCPH2O * prcp * (t1 - sfctmp)
    flx3 = 0.0
    return dict(eta=eta, t1=t1, beta=beta, cmc=cmc, dew=dew, drip=drip,
                flx1=flx1, flx3=flx3, ssoil=ssoil, runoff1=runoff1,
                runoff2=runoff2, runoff3=runoff3, edir=edir, ec=ec,
                et=et, ett=ett, snomlt=0.0)


def _noah_snopac(etp, prcp, prcpf, snowng, smc, smcmax, smcwlt, smcref,
                 smcdry, cmc, cmcmax, nsoil, dt, df1, q2, t1, sfctmp,
                 t24, th2, fdown, f1, stc, epsca, sfcprs, bexp, pc,
                 rch, rr, cfactr, sncovr, esd, sndens, snowh, sh2o,
                 slope, kdt, frzfact, psisat, zsoil, dwsat, dksat,
                 tbot, zbot, shdfac, nroot, rtdis, quartz, fxexp,
                 csoil, emissi, ribb, flx2, isurban, vegtyp, soiltyp,
                 opt_thcnd, flags):
    """SNOPAC: snowpack-present land update.  Mutates smc/sh2o/stc;
    returns a dict (evap components kinematic; reslin computed here
    with the branch-exact identity)."""
    ua_phys=flags.get('ua_phys',False)
    flx4=flags.get('flx4',0.)
    etpn=flags.get('etpn',0.)
    esdmin = 1.0e-6
    snoexp = 2.0
    dew = 0.0
    edir1 = 0.0
    ec1 = 0.0
    et1 = [0.0] * nsoil
    ett1 = 0.0
    etns = 0.0
    etns1 = 0.0
    esnow = 0.0
    esnow1 = 0.0
    esnow2 = 0.0
    edir = 0.0
    ec = 0.0
    et = [0.0] * nsoil
    ett = 0.0
    prcp1 = prcpf * 0.001
    beta = 1.0
    if etp <= 0.0:
        if ribb >= 0.1 and fdown > 150.0:
            etp = ((min(etp * (1.0 - ribb), 0.0) * sncovr / 0.980
                    + etp * (0.980 - sncovr)) / 0.980)
        if etp == 0.0:
            beta = 0.0
        if abs(etp) < 1e-9:
            flags["near_branch"] = True
        etp1 = etp * 0.001
        dew = -etp1
        esnow2 = etp1 * dt
        etanrg = etp * ((1.0 - sncovr) * _NLSUBC + sncovr * _NLSUBS)
    else:
        etp1 = etp * 0.001
        if sncovr < 1.0:
            etns1, edir1, ec1, et1, ett1 = _noah_evapo(
                smc, nsoil, cmc, etp1, dt, sh2o, smcmax, pc, smcwlt,
                smcref, shdfac, cmcmax, smcdry, cfactr, nroot, rtdis,
                fxexp)
            edir1 = edir1 * (1.0 - sncovr)
            ec1 = ec1 * (1.0 - sncovr)
            for k in range(nsoil):
                et1[k] = et1[k] * (1.0 - sncovr)
            ett1 = ett1 * (1.0 - sncovr)
            etns1 = etns1 * (1.0 - sncovr)
            edir = edir1 * 1000.0
            ec = ec1 * 1000.0
            et = [x * 1000.0 for x in et1]
            ett = ett1 * 1000.0
            etns = etns1 * 1000.0
        esnow = (etpn if ua_phys else etp) * sncovr
        esnow1 = esnow * 0.001
        esnow2 = esnow1 * dt
        etanrg = esnow * _NLSUBS + etns * _NLSUBC
    flx1 = 0.0
    if snowng:
        flx1 = _NCPICE * prcp * (t1 - sfctmp)
    else:
        if prcp > 0.0:
            flx1 = _NCPH2O * prcp * (t1 - sfctmp)
    dsoil = -0.5 * zsoil[0]
    dtot = snowh + dsoil
    denom = 1.0 + df1 / (dtot * rr * rch)
    t12a = ((fdown - flx1 - flx2 - emissi * _NSIGMA * t24) / rch
            + th2 - sfctmp - etanrg / rch) / rr
    t12b = df1 * stc[0] / (dtot * rr * rch)
    t12 = (sfctmp + t12a + t12b) / denom
    stc1_old = stc[0]
    if abs(t12 - _NTFREEZ) < 0.05:
        flags["near_branch"] = True
    snomlt = 0.0
    ex = 0.0
    if t12 <= _NTFREEZ:
        ebal_case = 1
        t1 = t12
        ssoil = df1 * (t1 - stc[0]) / dtot
        if esd - esnow2 < 0.0:
            flags["clamped"] = True     # sublimated more than the pack
        esd = max(0.0, esd - esnow2)
        flx3 = 0.0
        if ua_phys: flx4=0.
        ex = 0.0
        snomlt = 0.0
        # exact linearized identity for this branch
        reslin = (fdown - flx1 - flx2 - emissi * _NSIGMA * t24
                  + rch * (th2 - sfctmp) - etanrg
                  - rch * rr * (t12 - sfctmp)
                  - df1 * (t12 - stc1_old) / dtot)
    else:
        t1 = (_NTFREEZ * max(0.01, sncovr ** snoexp)
              + t12 * (1.0 - max(0.01, sncovr ** snoexp)))
        beta = 1.0
        ssoil = df1 * (t1 - stc[0]) / dtot
        if esd - esnow2 <= esdmin:
            ebal_case = 3
            flags["clamped"] = True     # pack (up to esdmin) discarded
            esd = 0.0
            ex = 0.0
            snomlt = 0.0
            flx3 = 0.0
            if ua_phys: flx4=0.
        else:
            esd = esd - esnow2
            etp3 = etp * _NLSUBC        # kept: the Fortran computes it
            seh = rch * (t1 - th2)
            t14 = t1 * t1
            t14 = t14 * t14
            flx3 = (fdown - flx1 - flx2 - emissi * _NSIGMA * t14
                    - ssoil - seh - etanrg)
            flx3_def = flx3
            if flx3 <= 0.0:
                flx3 = 0.0
            if abs(flx3_def) < 2.0:
                flags["near_branch"] = True
            if ua_phys:
                if flx4>0. and flx3>0.:
                    if flx3>=flx4: flx3=flx3-flx4
                    else: flx4=flx3;flx3=0.
                else: flx4=0.
            ex = flx3 * 0.001 / _NLSUBF
            snomlt = ex * dt
            if abs(esd - snomlt - esdmin) < 2e-6:
                flags["near_branch"] = True
            if esd - snomlt >= esdmin:
                esd = esd - snomlt
                ebal_case = 2 if flx3 == flx3_def else 3
            else:
                ex = esd / dt
                flx3 = ex * 1000.0 * _NLSUBF
                snomlt = esd
                esd = 0.0
                ebal_case = 3
        prcp1 = prcp1 + ex
        seh = rch * (t1 - th2)
        t14 = t1 * t1
        t14 = t14 * t14
        reslin = (fdown - flx1 - flx2 - emissi * _NSIGMA * t14 - ssoil
                  - seh - etanrg - flx3)
    cmc, runoff1, runoff2, runoff3, drip = _noah_smflx(
        smc, nsoil, cmc, dt, prcp1, zsoil, sh2o, slope, kdt, frzfact,
        smcmax, bexp, smcwlt, dksat, dwsat, shdfac, cmcmax, edir1,
        ec1, et1, flags)
    zz1 = 1.0
    yy = stc[0] - 0.5 * ssoil * zsoil[0] * zz1 / df1
    t11 = t1
    _t11, _ssoil1 = _noah_shflx(stc, smc, smcmax, nsoil, t11, dt, yy,
                                zz1, zsoil, tbot, zbot, psisat, sh2o,
                                bexp, f1, df1, quartz, csoil, vegtyp,
                                isurban, soiltyp, opt_thcnd)
    if esd > 0.0:
        snowh, sndens = _noah_snowpack(esd, dt, snowh, sndens, t1, yy, ua_phys, snomlt)
    else:
        esd = 0.0
        snowh = 0.0
        sndens = 0.0
        sncovr = 0.0
    if ua_phys: flags['flx4']=flx4
    return dict(eta=0.0, t1=t1, beta=beta, cmc=cmc, dew=dew, drip=drip,
                flx1=flx1, flx3=flx3, ssoil=ssoil, runoff1=runoff1,
                runoff2=runoff2, runoff3=runoff3, edir=edir, ec=ec,
                et=et, ett=ett, etns=etns, esnow=esnow, esd=esd,
                snowh=snowh, sndens=sndens, sncovr=sncovr,
                snomlt=snomlt, etp=etp, ebal_case=ebal_case,
                reslin=reslin)


def _noah_glacial(ffrozp, dt, nsoil, sldpth, lwdn, solnet, sfcprs, prcp,
                  sfctmp, q2, th2, q2sat, dqsdt2, alb, snoalb, tbot, z0brd,
                  emissi, embrd, t1, stc, snowh, sneqv, sncovr, ch,
                  snotime1, ribb, lvcoef):
    """SFLX_GLACIAL (WRF v4.6.1 phys/module_sf_noahlsm_glacial_only.F) in
    float64: the land-ice column the driver's ICE == -1 arm calls.  Mutates
    ``stc``; returns a dict of what the driver consumes.  kernels/noah.cu's
    noah_sflx_glacial is the FP32 twin; both are graded against WRF's own
    words by tests/test_noah_wrf461_parity.py."""
    tfreez = _NTFREEZ
    zsoil = [-sldpth[0]]
    for kz in range(1, nsoil):
        zsoil.append(-sldpth[kz] + zsoil[kz - 1])
    if sneqv < 0.10:
        sneqv, snowh = 0.10, 0.50
    sndens = sneqv / snowh
    sncond = _noah_csnow(sndens)
    snowng = frzgra = False
    if prcp > 0.0:
        if ffrozp > 0.5:
            snowng = True
        elif t1 <= tfreez:
            frzgra = True
    if snowng or frzgra:
        sn_new = prcp * dt * 0.001
        sneqv = sneqv + sn_new
        prcpf = 0.0
        snowh, sndens = _noah_snow_new(sfctmp, sn_new, snowh, sndens)
        if sncovr > 0.99:
            if stc[0] < tfreez - 5.0:
                sndens = 0.2
            if snowng and t1 < 273.0 and sfctmp < 273.0:
                sndens = 0.2
        sncond = _noah_csnow(sndens)
    else:
        prcpf = prcp
    del prcpf  # SNOPAC's PRCP1 is computed and never read
    sncovr = 1.0
    albedo, emissi, snotime1 = _noah_alcalc(alb, snoalb, embrd, 1.0, dt,
                                            snowng, snotime1, lvcoef)
    df1 = sncond
    dsoil = -(0.5 * zsoil[0])
    dtot = snowh + dsoil
    df1 = (snowh / dtot) * sncond + (dsoil / dtot) * df1
    if dtot > 2.0 * dsoil:
        dtot = 2.0 * dsoil
    ssoil = df1 * (t1 - stc[0]) / dtot
    burial = 7.0 * z0brd - snowh
    z0 = 0.001 if burial <= 0.0007 else burial / 7.0
    fdown = solnet + lwdn
    t2v = sfctmp * (1.0 + 0.61 * q2)
    rch = sfcprs / (_NRD * t2v) * _NCP_PEN * ch
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    # PENMAN (glacial): latent heat by T1, not by snow cover
    if t1 > 273.15:
        elcp1, lvs = _NELCP, _NLSUBC
    else:
        elcp1, lvs = _NELCP * _NLSUBS / _NLSUBC, _NLSUBS
    delta = elcp1 * dqsdt2
    a = elcp1 * (q2sat - q2)
    rr = emissi * t24 * 6.48e-8 / (sfcprs * ch) + 1.0
    if not snowng:
        if prcp > 0.0:
            rr = rr + _NCPH2O * prcp / rch
    else:
        rr = rr + _NCPICE * prcp / rch
    flx2 = -_NLSUBF * prcp if frzgra else 0.0
    fnet = fdown - emissi * _NSIGMA * t24 - ssoil - flx2
    rad = fnet / rch + th2 - sfctmp
    etp = ((a * rr + rad * delta) / (delta + rr)) * rch / lvs
    # SNOPAC (glacial)
    esdmin = 1.0e-6
    snomlt = dew = esnow = esnow2 = 0.0
    if etp <= 0.0:
        if ribb >= 0.1 and fdown > 150.0:
            etp = (min(etp * (1.0 - ribb), 0.0) / 0.980
                   + etp * (0.980 - 1.0)) / 0.980
        etp1 = etp * 0.001
        dew = -etp1
        esnow2 = etp1 * dt
        etanrg = etp * _NLSUBS
    else:
        esnow = etp
        esnow2 = esnow * 0.001 * dt
        etanrg = esnow * _NLSUBS
    flx1 = 0.0
    if snowng:
        flx1 = _NCPICE * prcp * (t1 - sfctmp)
    elif prcp > 0.0:
        flx1 = _NCPH2O * prcp * (t1 - sfctmp)
    dtot = snowh + dsoil
    denom = 1.0 + df1 / (dtot * rr * rch)
    t12a = ((fdown - flx1 - flx2 - emissi * _NSIGMA * t24) / rch
            + th2 - sfctmp - etanrg / rch) / rr
    t12b = df1 * stc[0] / (dtot * rr * rch)
    t12 = (sfctmp + t12a + t12b) / denom
    flx3 = 0.0
    if t12 <= tfreez:
        t1 = t12
        ssoil = df1 * (t1 - stc[0]) / dtot
        sneqv = max(0.0, sneqv - esnow2)
    else:
        t1 = tfreez
        if dtot > 2.0 * dsoil:
            dtot = 2.0 * dsoil
        ssoil = df1 * (t1 - stc[0]) / dtot
        if sneqv - esnow2 <= esdmin:
            sneqv = 0.0
        else:
            sneqv = sneqv - esnow2
            seh = rch * (t1 - th2)
            flx3 = (fdown - flx1 - flx2 - emissi * _NSIGMA * t1 ** 4
                    - ssoil - seh - etanrg)
            flx3 = max(flx3, 0.0)
            ex = flx3 * 0.001 / _NLSUBF
            snomlt = ex * dt
            if sneqv - snomlt >= esdmin:
                sneqv = sneqv - snomlt
            else:
                ex = sneqv / dt
                flx3 = ex * 1000.0 * _NLSUBF
                snomlt = sneqv
                sneqv = 0.0
    zz1 = 1.0
    yy = stc[0] - 0.5 * ssoil * zsoil[0] * zz1 / df1
    # SHFLX (glacial) = HRTICE + HSTEP
    zbot = -25.0
    ai, bi, ci, rhsts = ([0.0] * nsoil for _ in range(4))
    hcpct = 1.0e6 * (0.8194 - 0.1309 * 0.5 * zsoil[0])
    ddz = 1.0 / (-0.5 * zsoil[1])
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct)
    bi[0] = -ci[0] + df1 / (0.5 * zsoil[0] * zsoil[0] * hcpct * zz1)
    dtsdz = (stc[0] - stc[1]) / (-0.5 * zsoil[1])
    ssoil_h = df1 * (stc[0] - yy) / (0.5 * zsoil[0] * zz1)
    rhsts[0] = (df1 * dtsdz - ssoil_h) / (zsoil[0] * hcpct)
    ddz2 = 0.0
    df1k = df1
    for k in range(1, nsoil):
        zmd = 0.5 * (zsoil[k] + zsoil[k - 1])
        hcpct = 1.0e6 * (0.8194 - 0.1309 * zmd)
        df1n = 0.32333 - 0.10073 * zmd
        if k != nsoil - 1:
            dtsdz2 = (stc[k] - stc[k + 1]) / (0.5 * (zsoil[k - 1]
                                                     - zsoil[k + 1]))
            ddz2 = 2.0 / (zsoil[k - 1] - zsoil[k + 1])
            ci[k] = -df1n * ddz2 / ((zsoil[k - 1] - zsoil[k]) * hcpct)
        else:
            dtsdz2 = (stc[k] - tbot) / (0.5 * (zsoil[k - 1] + zsoil[k])
                                        - zbot)
            ci[k] = 0.0
        rhsts[k] = ((df1n * dtsdz2 - df1k * dtsdz)
                    / ((zsoil[k] - zsoil[k - 1]) * hcpct))
        ai[k] = -df1k * ddz / ((zsoil[k - 1] - zsoil[k]) * hcpct)
        bi[k] = -(ai[k] + ci[k])
        df1k, dtsdz, ddz = df1n, dtsdz2, ddz2
    stc[:] = _noah_hstep(stc, rhsts, dt, nsoil, ai, bi, ci)
    if sneqv >= 0.10:
        snowh, sndens = _noah_snowpack(sneqv, dt, snowh, sndens, t1, yy)
    else:
        sneqv, snowh = 0.10, 0.50
    eta_kinematic = etp
    q1 = q2 + eta_kinematic * _NCP / rch
    sheat = -(ch * _NCP * sfcprs) / (_NR * t2v) * (th2 - t1)
    esnow = esnow * _NLSUBS
    etp = etp * _NLSUBS
    eta = esnow if etp > 0.0 else etp
    return dict(t1=t1, sneqv=sneqv, snowh=snowh, sncovr=sncovr,
                albedo=albedo, emissi=emissi, z0=z0, snotime1=snotime1,
                eta=eta, sheat=sheat, eta_kinematic=eta_kinematic,
                fdown=fdown, esnow=esnow, dew=dew, etp=etp,
                ssoil=-1.0 * ssoil, flx1=flx1, flx2=flx2, flx3=flx3,
                snomlt=snomlt, runoff1=snomlt / dt, q1=q1)


def _noah_sflx(ffrozp, dt, nsoil, sldpth, lwdn, soldn, solnet, sfcprs,
               prcp, sfctmp, q2, th2, q2sat, dqsdt2, vegtyp, soiltyp,
               shdfac, shdmin, shdmax, alb, snoalb, tbot, z0brd, cmc,
               t1, stc, smc, sh2o, snowh, sneqv, ch, xlai, snotime1,
               ribb, params, isurban, opt_thcnd, rdlai2d, usemonalb,
               flags, ua_phys=False):
    """SFLX: the unified Noah column update.  Mutates stc/smc/sh2o
    (lists); returns a dict of everything the WRF driver consumes plus
    the energy-identity diagnostics."""
    from woof.core.noah import GEN, SOIL_COLS, VEG_COLS

    zsoil = [0.0] * nsoil
    zsoil[0] = -sldpth[0]
    for kz in range(1, nsoil):
        zsoil[kz] = -sldpth[kz] + zsoil[kz - 1]

    # ---- REDPRM (from the packed tables; SLOPETYP = 1 as the driver)
    sv = {n: params.soil[soiltyp - 1][i]
          for i, n in enumerate(SOIL_COLS)}
    vv = {n: params.veg[vegtyp - 1][i] for i, n in enumerate(VEG_COLS)}
    g = params.gen
    csoil = g[GEN["csoil"]]
    bexp = sv["bexp"]
    dksat = sv["dksat"]
    dwsat = sv["dwsat"]
    f1 = sv["f1"]
    psisat = sv["psisat"]
    quartz = sv["quartz"]
    smcdry = sv["smcdry"]
    smcmax = sv["smcmax"]
    smcref = sv["smcref"]
    smcwlt = sv["smcwlt"]
    zbot = g[GEN["zbot"]]
    salp = g[GEN["salp"]]
    sbeta = g[GEN["sbeta"]]
    refdk = g[GEN["refdk"]]
    frzk = g[GEN["frzk"]]
    fxexp = g[GEN["fxexp"]]
    refkdt = g[GEN["refkdt"]]
    kdt = refkdt * dksat / refdk
    slope = g[GEN["slope"]]
    lvcoef = g[GEN["lvcoef"]]
    frzfact = (smcmax / smcref) * (0.412 / 0.468)
    frzx = frzk * frzfact
    topt = g[GEN["topt"]]
    cmcmax = g[GEN["cmcmax"]]
    cfactr = g[GEN["cfactr"]]
    rsmax = g[GEN["rsmax"]]
    nroot = int(vv["nroot"])
    snup = vv["snup"]
    rsmin = vv["rsmin"]
    rgl = vv["rgl"]
    hs = vv["hs"]
    emissmin = vv["emissmin"]
    emissmax = vv["emissmax"]
    laimin = vv["laimin"]
    laimax = vv["laimax"]
    z0min = vv["z0min"]
    z0max = vv["z0max"]
    albedomin = vv["albedomin"]
    albedomax = vv["albedomax"]
    if vegtyp == params.bare:
        shdfac = 0.0
    if nroot > nsoil:
        raise ValueError("too many root layers")
    rtdis = [0.0] * nsoil
    for i in range(nroot):
        rtdis[i] = -sldpth[i] / zsoil[nroot - 1]

    # ---- urban parameter overrides (plain Noah, no UCM)
    if vegtyp == isurban:
        shdfac = 0.05
        rsmin = 400.0
        smcmax = 0.45
        smcref = 0.42
        smcwlt = 0.40
        smcdry = 0.40

    # ---- background emissivity / LAI / albedo / roughness interp
    if shdfac >= shdmax:
        embrd = emissmax
        if not rdlai2d:
            xlai = laimax
        if not usemonalb:
            alb = albedomin
        z0brd = z0max
    elif shdfac <= shdmin:
        embrd = emissmin
        if not rdlai2d:
            xlai = laimin
        if not usemonalb:
            alb = albedomax
        z0brd = z0min
    else:
        if shdmax > shdmin:
            interp_fraction = (shdfac - shdmin) / (shdmax - shdmin)
            interp_fraction = min(interp_fraction, 1.0)
            interp_fraction = max(interp_fraction, 0.0)
            embrd = ((1.0 - interp_fraction) * emissmin
                     + interp_fraction * emissmax)
            if not rdlai2d:
                xlai = ((1.0 - interp_fraction) * laimin
                        + interp_fraction * laimax)
            if not usemonalb:
                alb = ((1.0 - interp_fraction) * albedomax
                       + interp_fraction * albedomin)
            z0brd = ((1.0 - interp_fraction) * z0min
                     + interp_fraction * z0max)
        else:
            embrd = 0.5 * emissmin + 0.5 * emissmax
            if not rdlai2d:
                xlai = 0.5 * laimin + 0.5 * laimax
            if not usemonalb:
                alb = 0.5 * albedomin + 0.5 * albedomax
            z0brd = 0.5 * z0min + 0.5 * z0max

    # ---- snowpack density / precipitation type
    snowng = False
    frzgra = False
    if sneqv <= 1.0e-7:
        if sneqv > 0.0:
            flags["clamped"] = True      # sub-1e-7 pack discarded
        sneqv = 0.0
        sndens = 0.0
        snowh = 0.0
        sncond = 1.0
    else:
        sndens = sneqv / snowh
        if sndens > 1.0:
            raise ValueError(
                "Physical snow depth is less than snow water equiv.")
        sncond = _noah_csnow(sndens)
    if 0.0 < sneqv <= 2.0e-7:
        flags["near_branch"] = True
    if prcp > 0.0:
        if ffrozp > 0.5:
            snowng = True
        else:
            if t1 <= _NTFREEZ:
                frzgra = True
            if abs(t1 - _NTFREEZ) < 0.05:
                flags["near_branch"] = True
    if snowng or frzgra:
        sn_new = prcp * dt * 0.001
        sneqv = sneqv + sn_new
        prcpf = 0.0
        snowh, sndens = _noah_snow_new(sfctmp, sn_new, snowh, sndens)
        sncond = _noah_csnow(sndens)
    else:
        prcpf = prcp

    fvb=gama=fbur=fgsn=flx4=etpn=0.0
    flags['ua_phys']=ua_phys
    # ---- snow cover fraction, snow albedo, emissivity
    sncovr_in_arg = None
    if sneqv == 0.0:
        sncovr = 0.0
        albedo = alb
        emissi = embrd
    else:
        sncovr = _noah_snfrac(sneqv, snup, salp)
        if ua_phys:
            top,bottom=params.canopy[vegtyp-1]
            if abs(snowh-bottom)<1.e-6: flags['near_branch']=True
            fgsn=sneqv/0.02 if sneqv<0.02 else 1.
            if bottom>0. and snowh>bottom:
                if bottom<=0.5: fbur=(snowh-0.4*bottom)/(0.4*(top-bottom))
                else: fbur=(snowh-bottom)/(top-bottom)
            fbur=min(max(fbur,0.),1.)
            xlai=xlai*(1.-fbur)
            fvb=shdfac*fgsn*(1.-fbur)
            gama=math.exp(-1.*xlai)
            ru=0. if sfctmp<=t1 else 100.*shdfac*fgsn*min((sfctmp-t1)/5.,1.)*(1.-math.exp(-xlai))
            ch=ch/(1.+ru*ch)
        if abs(sncovr - 0.98) < 0.005 or abs(sncovr - 0.97) < 0.005:
            flags["near_branch"] = True
        sncovr = min(sncovr, 0.98)
        albedo, emissi, snotime1 = _noah_alcalc(
            alb, snoalb, embrd, sncovr, dt, snowng, snotime1, lvcoef)

    # ---- surface thermal conductivity + first-guess soil heat flux
    df1 = _noah_tdfcnd(smc[0], quartz, smcmax, sh2o[0], bexp, psisat,
                       soiltyp, opt_thcnd)
    if vegtyp == isurban:
        df1 = 3.24
    df1 = df1 * math.exp(sbeta * shdfac)
    if sncovr > 0.97:
        df1 = sncond
    dsoil = -(0.5 * zsoil[0])
    dtot = 0.0
    if sneqv == 0.0:
        ssoil = df1 * (t1 - stc[0]) / dsoil
    else:
        dtot = snowh + dsoil
        frcsno = snowh / dtot
        frcsoi = dsoil / dtot
        df1h = (sncond * df1) / (frcsoi * sncond + frcsno * df1)
        df1a = frcsno * sncond + frcsoi * df1
        df1 = df1a * sncovr + df1 * (1.0 - sncovr)
        ssoil = df1 * (t1 - stc[0]) / dtot

    # ---- roughness over snow
    if sncovr > 0.0:
        z0 = _noah_snowz0(sncovr, z0brd, snowh)
    else:
        z0 = z0brd

    if ua_phys:
        fv=shdmax*(1.-fbur)
        a1=(1.-fv)**2*((1.-fgsn**2)*math.log(0.01)+fgsn**2*math.log(0.001))
        a2=(1.-(1.-fv)**2)*math.log(z0brd)
        z0=math.exp(a1+a2)
    # ---- Penman potential evaporation
    fdown = solnet + lwdn
    t2v = sfctmp * (1.0 + 0.61 * q2)
    etp, rch, rr, epsca, t24, flx2, lvs = _noah_penman(
        sfctmp, sfcprs, ch, t2v, th2, prcp, fdown, ssoil, q2, q2sat,
        dqsdt2, snowng, frzgra, emissi, sncovr)
    if ua_phys:
        fnet=fdown-emissi*_NSIGMA*t24-ssoil
        if sneqv>0. and fnet>0. and soldn>0.:
            total=(1.-albedo)*soldn*fvb
            under=min(total,((1.-0.5)*(1.-0.13)*soldn*gama)*fvb)
            flx4=min(total-under,min(250.,0.5*(1.-albedo)*soldn))
        signck=(stc[0]-273.15)*(sfctmp-273.15)
        if flx4>0. and (signck<=0. or stc[0]<273.15):
            if fnet>=flx4: fnetn=fnet-flx4
            else: flx4=fnet;fnetn=0.
        else: flx4=0.;fnetn=0.
        if frzgra: fnetn=fnetn-flx2
        elcp1=(1.-sncovr)*_NELCP+sncovr*_NELCP*_NLSUBS/_NLSUBC
        delta=elcp1*dqsdt2
        radn=fnetn/rch+th2-sfctmp
        a=elcp1*(q2sat-q2)
        etpn=((a*rr+radn*delta)/(delta+rr))*rch/lvs
        flags["flx4"]=flx4;flags["etpn"]=etpn
    if abs(etp) < 1e-9:
        flags["near_branch"] = True

    # ---- canopy resistance
    if shdfac > 0.0 and xlai > 0.0:
        rc, pc = _noah_canres(soldn, ch, sfctmp, q2, sh2o, zsoil,
                              nsoil, smcwlt, smcref, rsmin, nroot,
                              q2sat, dqsdt2, topt, rsmax, rgl, hs,
                              xlai, sfcprs, emissi)
    else:
        rc = 0.0
        pc = 0.0
    if abs(smc[0] - smcdry) < 1e-4:
        flags["near_branch"] = True

    # ---- NOPAC / SNOPAC
    esnow = 0.0
    if sneqv == 0.0:
        p = _noah_nopac(etp, prcp, smc, smcmax, smcwlt, smcref, smcdry,
                        cmc, cmcmax, nsoil, dt, shdfac, sbeta, q2, t1,
                        sfctmp, t24, th2, fdown, f1, emissi, stc,
                        epsca, bexp, pc, rch, rr, cfactr, sh2o, slope,
                        kdt, frzx, psisat, zsoil, dksat, dwsat,
                        tbot, zbot, nroot, rtdis, quartz, fxexp,
                        csoil, vegtyp, isurban, soiltyp, opt_thcnd,
                        flags)
        eta_kinematic = p["eta"]
        ebal_case = 0
        etns = 0.0
        sndens_out = 0.0
        snomlt = 0.0
    else:
        p = _noah_snopac(etp, prcp, prcpf, snowng, smc, smcmax, smcwlt,
                         smcref, smcdry, cmc, cmcmax, nsoil, dt, df1,
                         q2, t1, sfctmp, t24, th2, fdown, f1, stc,
                         epsca, sfcprs, bexp, pc, rch, rr, cfactr,
                         sncovr, sneqv, sndens, snowh, sh2o, slope,
                         kdt, frzx, psisat, zsoil, dwsat, dksat,
                         tbot, zbot, shdfac, nroot, rtdis, quartz,
                         fxexp, csoil, emissi, ribb, flx2, isurban,
                         vegtyp, soiltyp, opt_thcnd, flags)
        esnow = p["esnow"]
        etns = p["etns"]
        etp = p["etp"]
        sneqv = p["esd"]
        snowh = p["snowh"]
        sndens_out = p["sndens"]
        sncovr = p["sncovr"]
        snomlt = p["snomlt"]
        ebal_case = p["ebal_case"]
        eta_kinematic = esnow + etns - 1000.0 * p["dew"]
    t1 = p["t1"]
    cmc = p["cmc"]
    flx1 = p["flx1"]
    flx3 = p["flx3"]
    ssoil = p["ssoil"]
    runoff1 = p["runoff1"]
    runoff2 = p["runoff2"]
    runoff3 = p["runoff3"]
    dew = p["dew"]
    drip = p["drip"]

    q1 = q2 + eta_kinematic * _NCP / rch
    sheat = -(ch * _NCP * sfcprs) / (_NR * t2v) * (th2 - t1)

    if ua_phys:
        flx4=flags["flx4"]
        sheat=sheat+flx4
    # ---- kinematic -> energy conversions
    edir = p["edir"] * _NLVH2O
    ec = p["ec"] * _NLVH2O
    et = [x * _NLVH2O for x in p["et"]]
    ett = p["ett"] * _NLVH2O
    esnow_e = esnow * _NLSUBS
    etp_e = etp * ((1.0 - sncovr) * _NLVH2O + sncovr * _NLSUBS)
    if etp_e > 0.0:
        eta_e = edir + ec + ett + esnow_e
    else:
        eta_e = etp_e
    if etp_e == 0.0:
        beta = 0.0
    else:
        beta = eta_e / etp_e

    # ---- energy identity residual (case 0; SNOPAC computed its own)
    if ebal_case == 0:
        reslin = (fdown - emissi * _NSIGMA * t24
                  + rch * (th2 - sfctmp) - eta_e
                  - rch * rr * (t1 - sfctmp) - ssoil)
    else:
        reslin = p["reslin"]

    ssoil_out = -1.0 * ssoil
    runoff3 = runoff3 / dt
    runoff2 = runoff2 + runoff3
    soilm = -1.0 * smc[0] * zsoil[0]
    for k in range(1, nsoil):
        soilm = soilm + smc[k] * (zsoil[k - 1] - zsoil[k])
    soilwm = -1.0 * (smcmax - smcwlt) * zsoil[0]
    soilww = -1.0 * (smc[0] - smcwlt) * zsoil[0]
    smav = [0.0] * nsoil
    for k in range(nsoil):
        smav[k] = (smc[k] - smcwlt) / (smcmax - smcwlt)
    if nroot >= 2:
        for k in range(1, nroot):
            soilwm = soilwm + (smcmax - smcwlt) * (zsoil[k - 1]
                                                   - zsoil[k])
            soilww = soilww + (smc[k] - smcwlt) * (zsoil[k - 1]
                                                   - zsoil[k])
    if soilwm < 1.0e-6:
        soilwm = 0.0
        soilw = 0.0
        soilm = 0.0
    else:
        soilw = soilww / soilwm

    return dict(flx4=flx4,fvb=fvb,fbur=fbur,fgsn=fgsn,t1=t1, cmc=cmc, sneqv=sneqv, snowh=snowh,
                sncovr=sncovr, albedo=albedo, emissi=emissi, z0=z0,
                z0brd=z0brd, alb=alb, xlai=xlai, snotime1=snotime1,
                eta_kinematic=eta_kinematic, eta=eta_e, sheat=sheat,
                etp=etp_e, ssoil=ssoil_out, flx1=flx1, flx2=flx2,
                flx3=flx3, beta=beta, runoff1=runoff1, runoff2=runoff2,
                runoff3=runoff3, snomlt=snomlt, q1=q1, soilw=soilw,
                soilm=soilm, smav=smav, dew=dew, drip=drip,
                edir=edir, ec=ec, ett=ett, esnow=esnow_e, rc=rc, pc=pc,
                fdown=fdown, ebal_case=ebal_case, reslin=reslin,
                kdt=kdt, frzx=frzx, smcmax=smcmax, smcwlt=smcwlt,
                ffrozp=ffrozp)


def np_noah_column(col, params, dt, dzs, isurban=None, isice=None,
                   xice_threshold=0.5, frpcpn=False, usemonalb=False,
                   rdlai2d=False, opt_thcnd=1, ua_phys=False):
    """Float64 mirror of one Noah column step: the WRF v4.6.1 driver
    ``lsm`` prep + ``SFLX`` + post-update, for a single (i, j).

    ``col`` is a dict of scalars (plus length-4 arrays ``smois``,
    ``tslb``, ``sh2o``) named exactly as woof.core.noah.launch_noah's
    device fields.  Returns the updated state dict plus diagnostics:
    ``skip`` (0 = land run, 1 = water, 2 = sea ice, 3 = land ice, run
    through SFLX_GLACIAL), ``ebal_case`` (0 NOPAC, 1 SNOPAC sub-freezing,
    2 SNOPAC melting unclipped, 3 SNOPAC clipped), ``reslin`` (the
    branch-exact discrete energy-identity residual, W/m^2),
    ``near_branch``/``clamped`` flags, and the WRF ``noahres``.
    """
    if isurban is None or isice is None:
        # the dataset's own ISURBAN/ISICE, as launch_noah resolves them
        from woof.core.noah import noah_category_ids
        dataset_isurban, dataset_isice = noah_category_ids(params)
        isurban = dataset_isurban if isurban is None else isurban
        isice = dataset_isice if isice is None else isice
    tresh, a2, a3, a4 = 0.95, 17.67, 273.15, 29.65
    a23m4 = a2 * (a3 - a4)
    flags = dict(near_branch=False, clamped=False)
    out = {k: (np.array(v, np.float64) if k in ("smois", "tslb", "sh2o")
               else v) for k, v in col.items()}
    out.update(skip=0, ebal_case=-1, reslin=0.0, noahres=0.0,
               smstav=0.0, smstot=0.0, smcrel=np.zeros(4),
               chklowq=1.0, runoff1=0.0, runoff2t=0.0, etp=0.0,
               esnow_w=0.0, ec_w=0.0, edir_w=0.0, ett_w=0.0,
               flx1=0.0, flx2=0.0, flx3=0.0, fdown=0.0, q1=0.0,
               dew=0.0, drip=0.0, sncovr=col["snowc"], kdt=0.0,
               frzx=0.0, smcmax=0.0, smcwlt=0.0, beta=0.0, pc=0.0,
               rc=0.0, near_branch=False, clamped=False)

    nsoil = 4
    sldpth = [float(d) for d in dzs]
    psfc = col["psfc"]
    sfcprs = col["sfcprs"]
    q2k = col["qv1"] / (1.0 + col["qv1"])
    q2sat = col["qgh"] / (1.0 + col["qgh"])
    sfctmp = col["sfctmp"]
    zlvl = 0.5 * col["dz8w1"]
    apes = (1.0e5 / psfc) ** c.RCP          # CAPA = R_d/CP
    apelm = (1.0e5 / sfcprs) ** c.RCP
    sfcth2 = sfctmp * apelm
    th2 = sfcth2 / apes
    emissi = col["emiss"]
    lwdn = col["glw"] * emissi
    soldn = col["swdown"]
    solnet = soldn * (1.0 - col["albedo"])
    prcp = col["rainbl"] / dt
    vegtyp = int(col["ivgtyp"])
    soiltyp = int(col["isltyp"])
    shdfac = col["vegfra"] / 100.0
    t1 = col["tsk"]
    chk = col["chs"]
    shmin = col["shdmin"] / 100.0
    shmax = col["shdmax"] / 100.0
    sneqv = col["snow"] * 0.001
    snowhk = col["snowh"]
    sncovr = col["snowc"]
    if frpcpn:
        ffrozp = col["sr"]
    else:
        ffrozp = 1.0 if sfctmp <= 273.15 else 0.0
        if prcp > 0.0 and abs(sfctmp - 273.15) < 0.05:
            flags["near_branch"] = True

    if (col["xland"] - 1.5) >= 0.0:          # open water
        out["skip"] = 1
        return out
    if col["xice"] >= xice_threshold:        # sea ice: WRF ICE=1 branch
        out["skip"] = 2
        out["sh2o"] = np.ones(nsoil)
        out["lai"] = 0.01
        return out
    dqsdt2 = q2sat * a23m4 / (sfctmp - a4) ** 2
    if col["snow"] > 0.0:
        sfctsno = sfctmp
        e2sat = 611.2 * math.exp(6174.0 * (1.0 / 273.15
                                           - 1.0 / sfctsno))
        q2sati = 0.622 * e2sat / (sfcprs - e2sat)
        q2sati = q2sati / (1.0 + q2sati)
        if abs(t1 - 273.14) < 0.05 or abs(t1 - 273.0) < 0.05:
            flags["near_branch"] = True
        if t1 > 273.14:
            q2sat = q2sat * (1.0 - col["snowc"]) + q2sati * col["snowc"]
            dqsdt2 = (dqsdt2 * (1.0 - col["snowc"])
                      + q2sati * 6174.0 / (sfctsno ** 2) * col["snowc"])
        else:
            q2sat = q2sati
            dqsdt2 = q2sati * 6174.0 / (sfctsno ** 2)
        if abs(soldn - 10.0) < 0.5 and col["snowc"] > 0.0:
            flags["near_branch"] = True
        if t1 > 273.0 and col["snowc"] > 0.0 and soldn > 10.0:
            dqsdt2 = dqsdt2 * (1.0 - col["snowc"])

    tbot = col["tmn"]
    if soiltyp == 14 and col["xice"] == 0.0:
        soiltyp = 7
    snoalb1 = col["snoalb"]
    cmc = col["canwat"] / 1000.0
    albbrd = col["albbck"]
    z0brd = col["z0"]
    embrd = col["embck"]
    snotime1 = col["snotime"]
    ribb = col["rib"]
    smc = [float(x) for x in np.asarray(col["smois"], np.float64)]
    stc = [float(x) for x in np.asarray(col["tslb"], np.float64)]
    swc = [float(x) for x in np.asarray(col["sh2o"], np.float64)]
    if (sneqv != 0.0 and snowhk == 0.0) or (snowhk <= sneqv):
        snowhk = 5.0 * sneqv
    xlai = col["lai"]
    if rdlai2d:
        if shdfac > 0.0 and xlai <= 0.0:
            xlai = 0.01

    if vegtyp == isice:
        # LAND ICE, module_sf_noahdrv.F:1137-1190 (ICE == -1): the shared
        # prep above, then SFLX_GLACIAL and the same post-update with the
        # soil held saturated, XLAI = 0.01, SOILM = SOILW = 0, RUNOFF2 = 0.
        from woof.core.noah import GEN
        g = _noah_glacial(ffrozp, dt, nsoil, sldpth, lwdn, solnet, sfcprs,
                          prcp, sfctmp, q2k, th2, q2sat, dqsdt2, albbrd,
                          snoalb1, tbot, z0brd, emissi, embrd, t1, stc,
                          snowhk, sneqv, sncovr, chk, snotime1, ribb,
                          params.gen[GEN["lvcoef"]])
        out["lai"] = 0.01
        out["canwat"] = cmc * 1000.0
        out["snow"] = g["sneqv"] * 1000.0
        out["snowh"] = g["snowh"]
        out["albedo"] = g["albedo"]
        out["albbck"] = albbrd
        out["z0"] = z0brd
        out["emiss"] = g["emissi"]
        out["znt"] = g["z0"]
        out["tsk"] = g["t1"]
        out["hfx"] = g["sheat"]
        out["potevp"] = col["potevp"] + g["etp"] * (dt / (_NXLV
                                                          * c.RHOWATER))
        out["qfx"] = g["eta_kinematic"]
        out["lh"] = g["eta"]
        out["grdflx"] = g["ssoil"]
        out["snowc"] = g["sncovr"]
        out["chs2"] = col["cqs2"]
        out["snotime"] = g["snotime1"]
        out["qsfc"] = g["q1"] / (1.0 - g["q1"])
        out["smois"] = np.ones(nsoil)
        out["tslb"] = np.array(stc)
        out["sh2o"] = np.ones(nsoil)
        out["noahres"] = ((solnet + lwdn) - g["sheat"] + g["ssoil"]
                          - g["eta"]
                          - (g["emissi"] * _NSTBOLT * g["t1"] ** 4)
                          - g["flx1"] - g["flx2"] - g["flx3"])
        out["smstav"] = 0.0
        out["smstot"] = 0.0
        out["smcrel"] = np.ones(nsoil)
        out["sfcrunoff"] = col["sfcrunoff"] + g["runoff1"] * dt * 1000.0
        out["udrunoff"] = col["udrunoff"]
        if ffrozp > 0.5:
            out["acsnow"] = col["acsnow"] + prcp * dt
        if out["snow"] > 0.0:
            out["acsnom"] = col["acsnom"] + g["snomlt"] * 1000.0
            out["snopcx"] = (col["snopcx"]
                             - g["snomlt"] * 1000.0 * _NXLF / dt)
        out.update(skip=3, ebal_case=4, reslin=0.0, chklowq=1.0,
                   runoff1=g["runoff1"], runoff2t=0.0, etp=g["etp"],
                   esnow_w=g["esnow"], flx1=g["flx1"], flx2=g["flx2"],
                   flx3=g["flx3"], fdown=g["fdown"], q1=g["q1"],
                   dew=g["dew"], sncovr=g["sncovr"])
        return out

    r = _noah_sflx(ffrozp, dt, nsoil, sldpth, lwdn, soldn, solnet,
                   sfcprs, prcp, sfctmp, q2k, th2, q2sat, dqsdt2,
                   vegtyp, soiltyp, shdfac, shmin, shmax, albbrd,
                   snoalb1, tbot, z0brd, cmc, t1, stc, smc, swc,
                   snowhk, sneqv, chk, xlai, snotime1, ribb, params,
                   isurban, opt_thcnd, rdlai2d, usemonalb, flags, ua_phys)

    # ---- driver post-SFLX state/flux updates
    out["lai"] = r["xlai"]
    out["canwat"] = r["cmc"] * 1000.0
    out["snow"] = r["sneqv"] * 1000.0
    out["snowh"] = r["snowh"]
    out["albedo"] = r["albedo"]
    out["albbck"] = r["alb"]
    out["z0"] = r["z0brd"]
    out["emiss"] = r["emissi"]
    out["znt"] = r["z0"]
    out["tsk"] = r["t1"]
    out["hfx"] = r["sheat"]
    out["potevp"] = col["potevp"] + r["etp"] * (dt / (_NXLV
                                                      * c.RHOWATER))
    out["qfx"] = r["eta_kinematic"]
    out["lh"] = r["eta"]
    out["grdflx"] = r["ssoil"]
    out["snowc"] = r["sncovr"]
    out["chs2"] = col["cqs2"]
    out["snotime"] = r["snotime1"]
    out["qsfc"] = r["q1"] / (1.0 - r["q1"])
    out["smois"] = np.array(smc)
    out["tslb"] = np.array(stc)
    out["sh2o"] = np.array(swc)
    out["noahres"] = ((solnet + lwdn) - r["sheat"] + r["ssoil"]
                      - r["eta"]
                      - (r["emissi"] * _NSTBOLT * r["t1"] ** 4)
                      - r["flx1"] - r["flx2"] - r["flx3"])
    if ua_phys:
        for field in ("flx4","fvb","fbur","fgsn"): out[field]=r[field]
        out["noahres"]-=r["flx4"]
    out["smstav"] = r["soilw"]
    out["smstot"] = r["soilm"] * 1000.0
    out["smcrel"] = np.array(r["smav"])
    out["sfcrunoff"] = col["sfcrunoff"] + r["runoff1"] * dt * 1000.0
    out["udrunoff"] = col["udrunoff"] + r["runoff2"] * dt * 1000.0
    if r["ffrozp"] > 0.5:
        out["acsnow"] = col["acsnow"] + prcp * dt
    if out["snow"] > 0.0:
        out["acsnom"] = col["acsnom"] + r["snomlt"] * 1000.0
        out["snopcx"] = (col["snopcx"]
                         - r["snomlt"] * 1000.0 * _NXLF / dt)

    # ---- diagnostics
    out["skip"] = 0
    out["ebal_case"] = r["ebal_case"]
    out["reslin"] = r["reslin"]
    out["chklowq"] = 1.0
    out["runoff1"] = r["runoff1"]
    out["runoff2t"] = r["runoff2"]
    out["etp"] = r["etp"]
    out["esnow_w"] = r["esnow"]
    out["ec_w"] = r["ec"]
    out["edir_w"] = r["edir"]
    out["ett_w"] = r["ett"]
    out["flx1"] = r["flx1"]
    out["flx2"] = r["flx2"]
    out["flx3"] = r["flx3"]
    out["fdown"] = r["fdown"]
    out["q1"] = r["q1"]
    out["dew"] = r["dew"]
    out["drip"] = r["drip"]
    out["sncovr"] = r["sncovr"]
    out["kdt"] = r["kdt"]
    out["frzx"] = r["frzx"]
    out["smcmax"] = r["smcmax"]
    out["smcwlt"] = r["smcwlt"]
    out["beta"] = r["beta"]
    out["pc"] = r["pc"]
    out["rc"] = r["rc"]
    out["near_branch"] = flags["near_branch"]
    out["clamped"] = flags["clamped"]
    return out


# ---- YSU PBL (Phase 3 Task 11) ---------------------------------------------

def _ysu_thomas(lower, diag, upper, rhs):
    """Float64 in-column Thomas solve used by :func:`np_ysu_column`.

    This is ``tridin_ysu`` from WRF v4.6.1
    ``phys/physics_mmm/bl_ysu.F90`` with its pressure-coordinate matrix
    already assembled.  Inputs are copied because WRF overwrites its work
    arrays during forward elimination and back substitution.
    """
    lower = np.asarray(lower, dtype=np.float64)
    diag = np.asarray(diag, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    out = np.asarray(rhs, dtype=np.float64).copy()
    nz = diag.size
    gamma = np.zeros(nz, dtype=np.float64)
    inv = 1.0 / diag[0]
    gamma[0] = upper[0] * inv
    out[0] *= inv
    for k in range(1, nz - 1):
        inv = 1.0 / (diag[k] - lower[k] * gamma[k - 1])
        gamma[k] = upper[k] * inv
        out[k] = (out[k] - lower[k] * out[k - 1]) * inv
    inv = 1.0 / (diag[-1] - lower[-1] * gamma[-2])
    out[-1] = (out[-1] - lower[-1] * out[-2]) * inv
    for k in range(nz - 2, -1, -1):
        out[k] -= gamma[k] * out[k + 1]
    return out


def np_ysu_column(u, v, theta, qv, qc, qi, p, p_interface, exner, dz, *,
                  psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland,
                  u10, v10, dt, rthraten=None,
                  ysu_topdown_pblmix=1, ysu_version="wrf_480"):
    """Float64 mirror of the YSU CUDA column kernel.

    The transcription authority is WRF v4.6.1 ``module_bl_ysu.F`` and the
    scheme body it calls in ``phys/physics_mmm/bl_ysu.F90``.  This ports the
    operational non-BEP, non-top-down path: bulk-Richardson PBL diagnosis,
    unstable countergradient terms, explicit entrainment, stable/free-air
    local mixing, and the implicit pressure-coordinate solves for momentum,
    potential temperature, vapor, cloud water, and cloud ice.  Inputs use
    Python's zero-based, surface-to-top convention; ``kpbl`` in the result is
    deliberately the WRF one-based level index.

    ``theta`` is potential temperature (the WRF wrapper receives temperature,
    converts to theta internally, then converts the returned temperature
    tendency back to theta tendency).  Surface heat/moisture fluxes are WRF
    ``HFX`` (W m-2) and ``QFX`` (kg m-2 s-1).  The returned mapping contains
    float64 tendencies plus ``hpbl``, ``kpbl``, ``exch_h/m``, ``wstar``, and
    entrainment-layer ``delta``.
    """
    u, v, theta, qv, qc, qi, p, exner, dz = (
        np.asarray(a, dtype=np.float64).copy()
        for a in (u, v, theta, qv, qc, qi, p, exner, dz))
    p_interface = np.asarray(p_interface, dtype=np.float64).copy()
    nz = theta.size
    if nz < 4:
        raise ValueError(f"YSU requires nz >= 4, got {nz}")
    if any(a.shape != (nz,) for a in (u, v, theta, qv, qc, qi, p,
                                      exner, dz)):
        raise ValueError("YSU column inputs must all have shape (nz,)")
    if p_interface.shape != (nz + 1,):
        raise ValueError("p_interface must have shape (nz + 1,)")
    rthraten = (np.zeros(nz, dtype=np.float64) if rthraten is None else
                np.asarray(rthraten, dtype=np.float64).copy())
    if rthraten.shape != (nz,) or not np.isfinite(rthraten).all():
        raise ValueError("rthraten must be a finite (nz,) array")
    if ysu_topdown_pblmix not in (0, 1, False, True):
        raise ValueError("ysu_topdown_pblmix must be 0 or 1")
    if dt <= 0.0 or np.any(dz <= 0.0):
        raise ValueError("YSU requires dt > 0 and positive layer depths")

    # No zero-flux exit: bl_ysu_run has none, and a calm column (ust = hfx =
    # qfx = 0) runs the whole scheme in WRF (a 703 m PBL with nine levels on
    # the WRF v4.6.1 oracle's case 12).  YSU without a surface layer is
    # refused at the configuration door (WRF fatals on that pairing,
    # woof.wrf461_compatibility PBL_SURFACE_LAYER_AUTHORITY[(1, 0)]), so
    # no supported run reaches this column with uninitialised fluxes.

    # WRF constants local to bl_ysu_run.
    xkzminm, xkzminh, xkzmax = 0.1, 0.01, 1000.0
    rimin, rlam, prmin, prmax = -100.0, 30.0, 0.25, 4.0
    brcr_ub, brcr_sb, cori = 0.0, 0.25, 1.0e-4
    afac = bfac = 6.8
    phifac, sfcfrac = 8.0, 0.1
    d1, d2, d3 = 0.02, 0.05, 0.001
    h1, h2 = 1.0 / 3.0, 2.0 / 3.0
    zfmin, aphi5, aphi16, tmin = 1.0e-8, 5.0, 16.0, 1.0e-2
    gamcrt, gamcrq, karman = 3.0, 2.0e-3, 0.4
    ep1 = c.RVOVRD - 1.0

    temp = theta * exner
    thli = (temp - c.XLV * qc / c.CP - 2.834e6 * qi / c.CP) / exner
    thv = theta * (1.0 + ep1 * qv)
    zq = np.empty(nz + 1, dtype=np.float64)
    zq[0] = 0.0
    zq[1:] = np.cumsum(dz)
    za = 0.5 * (zq[:-1] + zq[1:])
    dza = np.empty(nz, dtype=np.float64)
    dza[0] = za[0]
    dza[1:] = np.diff(za)
    delp = p_interface[:-1] - p_interface[1:]
    if np.any(delp <= 0.0) or np.any(dza <= 0.0):
        raise ValueError("YSU pressure and height coordinates must decrease/increase")

    rho = psfc / (c.RD * temp[0] * (1.0 + ep1 * qv[0]))
    govrth = c.G / theta[0]
    wspd1 = np.hypot(u[0], v[0]) + 1.0e-9
    sflux = hfx / rho / c.CP + qfx / rho * ep1 * theta[0]
    dt2, rdt = 2.0 * dt, 1.0 / (2.0 * dt)

    def diagnose(thermal, brcrit, clamp=True):
        """WRF bulk-Richardson crossing; returns hpbl and 1-based kpbl.

        ``clamp`` is bl_ysu.F90's ``if(hpbl(i).lt.zq(i,2)) kpbl(i) = 1``,
        which follows the sweep at :646 and :823 but NOT at :718-728 --
        that sweep's result reaches the theta-li scan unclamped, and WRF
        interpolates and clamps it once afterwards at :764-765.
        """
        brup = float(br)
        brdn = brup
        kp = 1
        crossed = False
        for kk in range(1, nz):
            if not crossed:
                brdn = brup
                spdk2 = max(u[kk] ** 2 + v[kk] ** 2, 1.0)
                brup = ((thv[kk] - thermal) * (c.G * za[kk] / thv[0])
                        / spdk2)
                kp = kk + 1                 # WRF level number
                crossed = brup > brcrit
        if brdn >= brcrit:
            frac = 0.0
        elif brup <= brcrit:
            frac = 1.0
        else:
            frac = (brcrit - brdn) / (brup - brdn)
        kh = kp - 1                          # Python index of upper level
        hp = za[kh - 1] + frac * (za[kh] - za[kh - 1])
        if clamp and hp < zq[1]:
            kp = 1
        return float(hp), int(kp), brdn, brup

    # First guess, followed by WRF's similarity scales and thermal excess.
    thermal = float(thv[0])
    thermalli = float(thli[0])
    hpbl, kpbl, brdn, brup = diagnose(thermal, brcr_ub)
    pblflg = kpbl > 1
    sfcflg = br <= 0.0
    fm, fh = float(psim), float(psih)
    zol1 = max(br * fm * fm / fh, rimin)
    zol1 = min(zol1, -zfmin) if sfcflg else max(zol1, zfmin)
    hol1 = zol1 * hpbl / za[0] * sfcfrac
    if sfcflg:
        phim = (1.0 - aphi16 * hol1) ** (-0.25)
        phih = (1.0 - aphi16 * hol1) ** (-0.5)
        wstar3 = govrth * max(sflux, 0.0) * hpbl
        wstar = np.cbrt(max(wstar3, 0.0))
    else:
        phim = phih = 1.0 + aphi5 * hol1
        wstar3 = 0.0
        wstar = 0.0
    ust3 = ust ** 3
    wscale = np.cbrt(max(ust3 + phifac * karman * wstar3 * 0.5, 0.0))
    wscale = min(wscale, ust * aphi16)
    wscale = max(wscale, ust / aphi5)

    hgamt = hgamq = hgamu = hgamv = 0.0
    if sfcflg and sflux > 0.0:
        gamfac = bfac / rho / wscale
        hgamt = min(gamfac * hfx / c.CP, gamcrt)
        hgamq = min(gamfac * qfx, gamcrq)
        vpert = (hgamt + ep1 * theta[0] * hgamq) / bfac * afac
        thermal += max(vpert, 0.0) * min(za[0] / (sfcfrac * hpbl), 1.0)
        thermalli += max(vpert, 0.0) * min(
            za[0] / (sfcfrac * hpbl), 1.0)
        hgamt, hgamq = max(hgamt, 0.0), max(hgamq, 0.0)
        cg = (-15.9 * ust * ust / max(wspd, 1.0e-9) * wstar3
              / max(wscale ** 4, 1.0e-20))
        hgamu, hgamv = cg * u[0], cg * v[0]
        # bl_ysu.F90:703-728 guards all three thermal-enhanced statements
        # with if(pblflg(i)), and :684-698 can only LOWER pblflg -- WRF has
        # no path that raises it here.  A column whose FIRST guess sat below
        # zq(i,2) (:646-647) therefore keeps kpbl=1 and stays in the local-K
        # regime for the whole step, however far the thermal excess could
        # have pushed the enhanced sweep.
        #
        # The sweep is the WHOLE of :703-728: it leaves hpbl at the
        # zq(i,1) of :706 and never touches pblflg.  WRF interpolates hpbl
        # and applies the zq(i,2) clamp exactly once, at :754-768, AFTER
        # the theta-li scan -- so the scan below must see this sweep's
        # kpbl >= 2 and an unchanged pblflg.  The hpbl this returns is
        # dead for this call: the post-scan ``if pblflg`` block recomputes
        # it from the same brdn/brup before any reader, as :764 does.
        if pblflg:
            hpbl, kpbl, brdn, brup = diagnose(thermal, brcr_ub,
                                              clamp=False)
    else:
        pblflg = False

    # WRF v4.6.1 bl_ysu.F90:732-768 applies this to every column.  Liquid
    # loading can revive a stable-surface PBL before cloud-top entrainment.
    if ysu_topdown_pblmix:
        kpblold = kpbl
        definebrup = False
        for fk in range(kpblold, nz):  # Fortran kpblold:kte-1
            k = fk - 1
            spdk2 = max(u[k] ** 2 + v[k] ** 2, 1.0)
            bruptmp = ((thli[k] - thermalli)
                        * (c.G * za[k] / thli[0]) / spdk2)
            stable_li = bruptmp >= brcr_ub
            if definebrup:
                kpbl = fk
                brup = bruptmp
                definebrup = False
            if not stable_li:
                brdn = bruptmp
                definebrup = True
                pblflg = True
    if pblflg:
        if brdn >= brcr_ub:
            frac = 0.0
        elif brup <= brcr_ub:
            frac = 1.0
        else:
            frac = (brcr_ub - brdn) / (brup - brdn)
        kh = kpbl - 1
        hpbl = za[kh - 1] + frac * (za[kh] - za[kh - 1])
        # bl_ysu.F90:765 and :766 are two INDEPENDENT statements.  The
        # second kills pblflg whenever kpbl <= 1, whatever hpbl came back
        # as -- including the case where hpbl was interpolated from WRF's
        # own za(i,0) overrun above and came back large.  Nesting :766
        # inside :765 let the theta-li revival (:745-749 raises pblflg on a
        # final unstable iteration without ever reassigning kpbl) survive
        # with kpbl == 1, and :833's k = kpbl(i)-1 is then one level below
        # the column.
        if hpbl < zq[1]:
            kpbl = 1
        if kpbl <= 1:
            pblflg = False

    # WRF stable-boundary-layer enhancement when the first diagnosis falls
    # below the first interface.  Land uses Ricr=0.25; water uses the Rossby
    # dependent limit.
    if (not sfcflg) and hpbl < zq[1]:
        brcrit = brcr_sb
        if xland >= 1.5:
            ross = max(np.hypot(u10, v10), 1.0e-9) / (cori * znt)
            brcrit = min(0.16 * (1.0e-7 * ross) ** (-0.18), 0.3)
        hpbl, kpbl, brdn, brup = diagnose(thermal, brcrit)
        # WRF deliberately leaves pblflg false here: kpbl still separates
        # stable-PBL from free-air K profiles, but countergradient and
        # entrainment terms remain convective-only.

    # Entrainment parameters for a resolved convective PBL.
    wm2 = we = bfxpbl = hfxpbl = qfxpbl = 0.0
    ufxpbl = vfxpbl = delta = 0.0
    wstar3_2 = topdown_radsum = 0.0
    cloudflg = False
    if pblflg:
        kt = kpbl - 2                       # level immediately below PBL top
        wm3 = wstar3 + 5.0 * ust3
        wm2 = max(wm3, 0.0) ** h2
        bfxpbl = -0.15 * thv[0] / c.G * wm3 / hpbl
        dthv = max(thv[kt + 1] - thv[kt], tmin)
        we = max(bfxpbl / dthv, -np.sqrt(wm2))
        # v4.6.1 F90 839-897.  ``kpbl < nz`` makes the source k+2 hazard
        # explicit: top-level PBLs have no layer available above cloud top.
        if (ysu_topdown_pblmix and kpbl < nz
                and qc[kt] + qi[kt] > 1.0e-5):
            cloudflg = True
            ptop = p_interface[kt + 1]
            templ = thli[kt] * (ptop / 100000.0) ** c.RCP
            rvls = (100.0 * 6.112
                    * np.exp(17.67 * (templ - 273.16) / (templ - 29.65))
                    * (c.EP2 / ptop))
            qsum = qv[kt] + qc[kt]
            temps = templ + ((qsum - rvls)
                             / (c.CP / c.XLV + c.EP2 * c.XLV * rvls
                                / (c.RD * templ ** 2)))
            rvls = (100.0 * 6.112
                    * np.exp(17.67 * (temps - 273.15) / (temps - 29.65))
                    * (c.EP2 / ptop))
            rcldb = max(qsum - rvls, 0.0)
            kabove = kt + 2
            dthv_cloud = (
                thli[kabove] + theta[kabove] * ep1
                * (qv[kabove] + qc[kabove])
                - (thli[kt] + theta[kt] * ep1 * qsum))
            dthv_cloud = max(dthv_cloud, 0.1)
            tmp1 = c.XLV / c.CP * rcldb / (exner[kt] * dthv_cloud)
            from woof.ysu_generations import ysu_generation
            ysu_generation(ysu_version)
            ent_eff = (min(0.2 * 8.0 * tmp1, 0.4) if ysu_version == "wrf_480"
                       else 0.2 * 8.0 * tmp1 + 0.2)
            for kk in range(kt + 1):
                radflux = rthraten[kk] * exner[kk]
                radflux *= (c.CP / c.G
                            * (p_interface[kk] - p_interface[kk + 1]))
                if radflux < 0.0:
                    topdown_radsum += abs(radflux)
            topdown_radsum = max(topdown_radsum, 0.0)
            rho2 = p[kt] / (c.RD * temp[kt] * (1.0 + ep1 * qv[kt]))

            # Preserve the executable value after each overwritten bfx0
            # assignment in the v4.6.1 source.
            bfx0 = max(sflux, 0.0)
            wm3 = govrth * bfx0 * hpbl + 5.0 * ust3
            wm2 = wm3 ** h2
            bfxpbl = -0.15 * thv[0] / c.G * wm3 / hpbl
            dthv = max(thv[kt + 1] - thv[kt], tmin)
            we = max(bfxpbl / dthv, -np.sqrt(wm2))

            bfx0 = max(topdown_radsum / rho2 / c.CP, 0.0)
            wm3_top = c.G / thv[kt] * bfx0 * hpbl
            wm2_top = wm3_top ** h2
            wm2 += wm2_top
            bfxpbl = -ent_eff * bfx0
            dthv = max(thv[kt + 1] - thv[kt], 0.1)
            we += max(bfxpbl / dthv, -np.sqrt(wm2_top))
            wstar3_2 = c.G / thv[kt] * bfx0 * hpbl
            wscale = np.cbrt(ust3 + phifac * karman
                             * (wstar3 + wstar3_2) * 0.5)
            wscale = min(wscale, ust * aphi16)
            wscale = max(wscale, ust / aphi5)
            gamfac = bfac / rho / wscale
            hgamt = min(gamfac * hfx / c.CP, gamcrt)
            hgamq = min(gamfac * qfx, gamcrq)
            gamfac = bfac / rho2 / wscale
            hgamt2 = min(gamfac * topdown_radsum / c.CP, gamcrt)
            hgamt = max(hgamt, 0.0) + max(hgamt2, 0.0)
            cg = (-15.9 * ust * ust / max(wspd, 1.0e-9)
                  * (wstar3 + wstar3_2) / max(wscale ** 4, 1.0e-20))
            hgamu, hgamv = cg * u[0], cg * v[0]
        dth = max(theta[kt + 1] - theta[kt], tmin)
        dq = min(qv[kt + 1] - qv[kt], 0.0)
        hfxpbl, qfxpbl = we * dth, we * dq
        dux, dvx = u[kt + 1] - u[kt], v[kt + 1] - v[kt]
        if dux > tmin:
            ufxpbl = max(we * dux, -ust * ust)
        elif dux < -tmin:
            ufxpbl = min(we * dux, ust * ust)
        if dvx > tmin:
            vfxpbl = max(we * dvx, -ust * ust)
        elif dvx < -tmin:
            vfxpbl = min(we * dvx, ust * ust)
        delb = govrth * d3 * hpbl
        delta = min(d1 * hpbl + d2 * wm2 / delb, 100.0)

    entfac = np.full(nz, 1.0e30, dtype=np.float64)
    if pblflg and delta > 0.0:
        for k in range(nz):
            if k + 1 >= kpbl:
                entfac[k] = ((zq[k + 1] - hpbl) / delta) ** 2

    # Diffusivities at interfaces above each full level k (the final entry is
    # unused, matching WRF).  Background K is momentum=.1, scalars=.01.
    xkzm = np.zeros(nz, dtype=np.float64)
    xkzh = np.zeros(nz, dtype=np.float64)
    xkzq = np.zeros(nz, dtype=np.float64)
    xkzm[:-1], xkzh[:-1], xkzq[:-1] = xkzminm, xkzminh, xkzminh
    xkzml = np.zeros(nz, dtype=np.float64)
    xkzhl = np.zeros(nz, dtype=np.float64)
    zfacent = np.zeros(nz, dtype=np.float64)

    for k in range(nz):
        if k + 1 < kpbl:
            zfac = np.clip(1.0 - ((zq[k + 1] - za[0])
                                  / (hpbl - za[0])), zfmin, 1.0)
            zfacent[k] = (1.0 - zfac) ** 3
            wsk = np.cbrt(max(ust3 + phifac * karman * wstar3
                              * (1.0 - zfac), 0.0))
            wsk2 = np.cbrt(max(phifac * karman * wstar3_2 * zfac, 0.0))
            if sfcflg:
                prfac = bfac * karman * sfcfrac
                if ust3 > 0.0:
                    prfac2 = (15.9 * (wstar3 + wstar3_2) / ust3
                              / (1.0 + 4.0 * karman
                                 * (wstar3 + wstar3_2) / ust3))
                else:
                    # ust**3 == 0 exactly makes WRF's own quotient 0/0
                    # (bl_ysu.F90:948) -- NaN, not an answer.  Take the
                    # defined algebraic limit instead, as ysu.cu does.
                    prfac2 = (15.9 / (4.0 * karman)
                              if wstar3 + wstar3_2 > 0.0 else 0.0)
                prnumfac = (-3.0 * max(zq[k + 1] - sfcfrac * hpbl, 0.0) ** 2
                            / hpbl ** 2)
            else:
                prfac = prfac2 = prnumfac = 0.0
                wsk = max(ust / (1.0 + aphi5 * zol1 * zq[k + 1] / za[0]),
                          0.001)
            prnum0 = np.clip(phih / phim + prfac, prmin, prmax)
            km = (wsk * karman * zq[k + 1] * zfac ** 2
                  + wsk2 * karman * (hpbl - zq[k + 1])
                  * (1.0 - zfac) ** 2)
            if k == kpbl - 2 and cloudflg and we < 0.0:
                km = 0.0
            prnum = 1.0 + (prnum0 - 1.0) * np.exp(prnumfac)
            kq = km / prnum
            prnum0 /= 1.0 + prfac2 * karman * sfcfrac
            prnum = 1.0 + (prnum0 - 1.0) * np.exp(prnumfac)
            kh = km / prnum
            xkzm[k] = min(km + xkzminm, xkzmax)
            xkzh[k] = min(kh + xkzminh, xkzmax)
            xkzq[k] = min(kq + xkzminh, xkzmax)

    for k in range(nz - 1):
        if k + 1 >= kpbl:
            ss = (((u[k + 1] - u[k]) ** 2 + (v[k + 1] - v[k]) ** 2)
                  / dza[k + 1] ** 2 + 1.0e-9)
            govrthv = c.G / (0.5 * (thv[k + 1] + thv[k]))
            ri = govrthv * (thv[k + 1] - thv[k]) / (ss * dza[k + 1])
            # WRF moist Richardson correction in a layer cloudy at both ends.
            if ((qc[k] + qi[k]) > 1.0e-5
                    and (qc[k + 1] + qi[k + 1]) > 1.0e-5):
                qmean = 0.5 * (qv[k] + qv[k + 1])
                tmean = 0.5 * (temp[k] + temp[k + 1])
                alph = c.XLV * qmean / c.RD / tmean
                chi = c.XLV ** 2 * qmean / c.CP / c.RV / tmean ** 2
                ri = ((1.0 + alph)
                      * (ri - c.G ** 2 / ss / tmean / c.CP
                         * ((chi - alph) / (1.0 + chi))))
            zk = karman * zq[k + 1]
            rlamdz = min(max(0.1 * dza[k + 1], rlam), 300.0)
            rlamdz = min(dza[k + 1], rlamdz)
            rl2 = (zk * rlamdz / (rlamdz + zk)) ** 2
            dk = rl2 * np.sqrt(ss)
            if ri < 0.0:
                ri = max(ri, rimin)
                sri = np.sqrt(-ri)
                km = dk * (1.0 + 8.0 * (-ri) / (1.0 + 1.746 * sri))
                kh = dk * (1.0 + 8.0 * (-ri) / (1.0 + 1.286 * sri))
            else:
                kh = dk / (1.0 + 5.0 * ri) ** 2
                km = kh * min(1.0 + 2.1 * ri, prmax)
            xkzm[k] = min(km + xkzminm, xkzmax)
            xkzh[k] = min(kh + xkzminh, xkzmax)
            xkzml[k], xkzhl[k] = xkzm[k], xkzh[k]
    xkzq[kpbl - 1:nz - 1] = xkzh[kpbl - 1:nz - 1]

    def scalar_matrix(kcoef, surface_rhs, values, kind):
        lower = np.zeros(nz, dtype=np.float64)
        diag = np.zeros(nz, dtype=np.float64)
        upper = np.zeros(nz, dtype=np.float64)
        rhs = np.zeros(nz, dtype=np.float64)
        diag[0], rhs[0] = 1.0, surface_rhs
        for k in range(nz - 1):
            dtodsd, dtodsu = dt2 / delp[k], dt2 / delp[k + 1]
            dsig, rdz = p[k] - p[k + 1], 1.0 / dza[k + 1]
            tem1 = dsig * kcoef[k] * rdz
            if pblflg and k + 1 < kpbl:
                if kind == "heat":
                    grad = -hgamt / hpbl - hfxpbl * zfacent[k] / kcoef[k]
                else:
                    grad = -qfxpbl * zfacent[k] / kcoef[k]
                nonlocal_flux = tem1 * grad
                rhs[k] += dtodsd * nonlocal_flux
                rhs[k + 1] = values[k + 1] - dtodsu * nonlocal_flux
            elif pblflg and k + 1 >= kpbl and entfac[k] < 4.6:
                knew = -we * dza[kpbl - 1] * np.exp(-entfac[k])
                knew = np.sqrt(max(knew * xkzhl[k], 0.0))
                kcoef[k] = np.clip(knew, xkzminh, xkzmax)
                rhs[k + 1] = values[k + 1]
            else:
                rhs[k + 1] = values[k + 1]
            dsdz2 = dsig * kcoef[k] * rdz * rdz
            upper[k] = -dtodsd * dsdz2
            lower[k + 1] = -dtodsu * dsdz2
            diag[k] -= upper[k]
            diag[k + 1] = 1.0 - lower[k + 1]
        return lower, diag, upper, rhs

    # Heat uses theta-300 exactly like WRF; the offset improves precision.
    kh_work = xkzh.copy()
    heat_surface = theta[0] - 300.0 + hfx / (c.CP / c.G) / delp[0] * dt2
    lo_h, di_h, up_h, rhs_h = scalar_matrix(
        kh_work, heat_surface, theta - 300.0, "heat")
    theta_new = _ysu_thomas(lo_h, di_h, up_h, rhs_h) + 300.0
    dtheta = (theta_new - theta) * rdt

    # Vapor has the same scalar matrix form, a distinct Kq below the PBL,
    # and the surface moisture-flux source.  Cloud/ice reuse that matrix.
    kq_work = xkzq.copy()
    q_surface = qv[0] + qfx * c.G / delp[0] * dt2
    lo_q, di_q, up_q, rhs_q = scalar_matrix(
        kq_work, q_surface, qv, "moisture")
    qv_new = _ysu_thomas(lo_q, di_q, up_q, rhs_q)
    qc_new = _ysu_thomas(lo_q, di_q, up_q, qc)
    qi_new = _ysu_thomas(lo_q, di_q, up_q, qi)

    # Momentum matrix: surface stress is implicit; countergradient and
    # entrainment terms mirror the heat assembly with Km.
    lower = np.zeros(nz, dtype=np.float64)
    diag = np.zeros(nz, dtype=np.float64)
    upper = np.zeros(nz, dtype=np.float64)
    rhs_u = np.zeros(nz, dtype=np.float64)
    rhs_v = np.zeros(nz, dtype=np.float64)
    fric = (ust * ust / wspd1 * rho * c.G / delp[0] * dt2
            * (wspd1 / max(wspd, 1.0e-9)) ** 2)
    diag[0] = 1.0 + fric
    rhs_u[0], rhs_v[0] = u[0], v[0]
    km_work = xkzm.copy()
    for k in range(nz - 1):
        dtodsd, dtodsu = dt2 / delp[k], dt2 / delp[k + 1]
        dsig, rdz = p[k] - p[k + 1], 1.0 / dza[k + 1]
        tem1 = dsig * km_work[k] * rdz
        if pblflg and k + 1 < kpbl:
            gu = -hgamu / hpbl - ufxpbl * zfacent[k] / km_work[k]
            gv = -hgamv / hpbl - vfxpbl * zfacent[k] / km_work[k]
            fu, fv = tem1 * gu, tem1 * gv
            rhs_u[k] += dtodsd * fu
            rhs_u[k + 1] = u[k + 1] - dtodsu * fu
            rhs_v[k] += dtodsd * fv
            rhs_v[k + 1] = v[k + 1] - dtodsu * fv
        elif pblflg and k + 1 >= kpbl and entfac[k] < 4.6:
            km_work[k] = np.sqrt(max(kh_work[k] * xkzml[k], 0.0))
            km_work[k] = np.clip(km_work[k], xkzminm, xkzmax)
            rhs_u[k + 1], rhs_v[k + 1] = u[k + 1], v[k + 1]
        else:
            rhs_u[k + 1], rhs_v[k + 1] = u[k + 1], v[k + 1]
        dsdz2 = dsig * km_work[k] * rdz * rdz
        upper[k] = -dtodsd * dsdz2
        lower[k + 1] = -dtodsu * dsdz2
        diag[k] -= upper[k]
        diag[k + 1] = 1.0 - lower[k + 1]
    u_new = _ysu_thomas(lower, diag, upper, rhs_u)
    v_new = _ysu_thomas(lower, diag, upper, rhs_v)

    exch_h = np.zeros(nz, dtype=np.float64)
    exch_m = np.zeros(nz, dtype=np.float64)
    exch_h[1:], exch_m[1:] = kh_work[:-1], km_work[:-1]
    return dict(du=(u_new - u) * rdt, dv=(v_new - v) * rdt,
                dtheta=dtheta, dqv=(qv_new - qv) * rdt,
                dqc=(qc_new - qc) * rdt, dqi=(qi_new - qi) * rdt,
                hpbl=float(hpbl), kpbl=int(kpbl), exch_h=exch_h,
                exch_m=exch_m, wstar=float(wstar), delta=float(delta),
                topdown_radsum=float(topdown_radsum),
                wstar3_2=float(wstar3_2), cloudflg=bool(cloudflg))


# ---------------------------------------------------------------------------
# Phase 3 Task 7: metgrid-style horizontal interpolation and wind rotation
# ---------------------------------------------------------------------------

def _regular_coordinates(latitude, longitude, target_lat, target_lon):
    """Return zero-based regular-grid coordinates in float64.

    ERA5 longitudes use 0..360 while WRF grids conventionally use -180..180;
    target longitudes are shifted by whole turns toward the source-axis centre.
    Both axes may be increasing or decreasing, but must be uniformly spaced.
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("source latitude and longitude must be 1-D")
    if latitude.size < 2 or longitude.size < 2:
        raise ValueError("source axes must each contain at least two points")
    if target_lat.shape != target_lon.shape:
        raise ValueError("target latitude and longitude shapes differ")
    if not (np.isfinite(latitude).all() and np.isfinite(longitude).all()
            and np.isfinite(target_lat).all() and np.isfinite(target_lon).all()):
        raise ValueError("coordinates must be finite")
    dlat = np.diff(latitude)
    dlon = np.diff(longitude)
    if dlat[0] == 0.0 or dlon[0] == 0.0:
        raise ValueError("source axes must be strictly monotonic")
    if not np.allclose(dlat, dlat[0], rtol=1.0e-11, atol=1.0e-12):
        raise ValueError("source latitude axis must be uniform")
    if not np.allclose(dlon, dlon[0], rtol=1.0e-11, atol=1.0e-12):
        raise ValueError("source longitude axis must be uniform")

    lon_mid = 0.5 * (longitude[0] + longitude[-1])
    unwrapped_lon = target_lon + 360.0 * np.round((lon_mid - target_lon) / 360.0)
    y = (target_lat - latitude[0]) / dlat[0]
    x = (unwrapped_lon - longitude[0]) / dlon[0]
    eps = 2.0e-10
    if (np.min(x) < -eps or np.max(x) > longitude.size - 1 + eps
            or np.min(y) < -eps or np.max(y) > latitude.size - 1 + eps):
        raise ValueError("target points fall outside the source grid")
    return np.clip(y, 0.0, latitude.size - 1.0), np.clip(
        x, 0.0, longitude.size - 1.0)


def _wps_oned_np(x, a, b, cc, d):
    """WPS v4.6.0 ``interp_module.F:oned`` in float64."""
    regular = ((1.0 - x)
               * (b + x * (0.5 * (cc - a) + x * (0.5 * (cc + a) - b)))
               + x * (cc + (1.0 - x)
                      * (0.5 * (b - d) + (1.0 - x)
                         * (0.5 * (b + d) - cc))))
    out = np.zeros_like(regular, dtype=np.float64)
    out = np.where(x == 0.0, b, out)
    out = np.where(x == 1.0, cc, out)
    both = b * cc != 0.0
    only_a = both & (a != 0.0) & (d == 0.0)
    only_d = both & (a == 0.0) & (d != 0.0)
    neither = both & (a == 0.0) & (d == 0.0)
    all_four = both & (a != 0.0) & (d != 0.0)
    out = np.where(neither, b * (1.0 - x) + cc * x, out)
    out = np.where(
        only_a, b + x * (0.5 * (cc - a) + x * (0.5 * (cc + a) - b)), out)
    out = np.where(
        only_d,
        cc + (1.0 - x) * (0.5 * (b - d) + (1.0 - x)
                           * (0.5 * (b + d) - cc)),
        out,
    )
    return np.where(all_four, regular, out)


def interpolate_regular_np(field, latitude, longitude, target_lat, target_lon,
                           method="parabolic"):
    """Float64 mirror of regular-grid bilinear/overlapping-parabolic GPU interp.

    ``parabolic`` is WPS ``sixteen_pt``: four calls to its one-dimensional
    overlapping-parabolic ``oned`` in x followed by one call in y. Edge
    stencil indices are clamped exactly like the Fortran routine.
    """
    field = np.asarray(field, dtype=np.float64)
    y, x = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    if field.ndim < 2 or field.shape[-2:] != (len(latitude), len(longitude)):
        raise ValueError("field trailing dimensions do not match source axes")
    lead = (slice(None),) * (field.ndim - 2)
    expand = (None,) * (field.ndim - 2)
    if method == "nearest":
        iy = np.rint(y).astype(np.int64)
        ix = np.rint(x).astype(np.int64)
        return field[lead + (iy, ix)]
    if method == "bilinear":
        iy = np.minimum(np.floor(y).astype(np.int64), len(latitude) - 2)
        ix = np.minimum(np.floor(x).astype(np.int64), len(longitude) - 2)
        fy = (y - iy)[expand]
        fx = (x - ix)[expand]
        lower = ((1.0 - fx) * field[lead + (iy, ix)]
                 + fx * field[lead + (iy, ix + 1)])
        upper = ((1.0 - fx) * field[lead + (iy + 1, ix)]
                 + fx * field[lead + (iy + 1, ix + 1)])
        return (1.0 - fy) * lower + fy * upper
    if method != "parabolic":
        raise ValueError("method must be 'nearest', 'bilinear', or 'parabolic'")

    iy = np.floor(y).astype(np.int64)
    ix = np.floor(x).astype(np.int64)
    fy = (y - iy)[expand]
    fx = (x - ix)[expand]
    xindices = [np.clip(ix + offset, 0, len(longitude) - 1)
                for offset in (-1, 0, 1, 2)]
    yindices = [np.clip(iy + offset, 0, len(latitude) - 1)
                for offset in (-1, 0, 1, 2)]
    rows = []
    for jy in yindices:
        values = [np.where(field[lead + (jy, jx)] == 0.0, 1.0e-20,
                           field[lead + (jy, jx)])
                  for jx in xindices]
        rows.append(_wps_oned_np(fx, *values))
    result = _wps_oned_np(fy, *rows)
    return np.where(result == 1.0e-20, 0.0, result)


def masked_nearest_np(field, latitude, longitude, target_lat, target_lon,
                      source_landmask, target_landmask, *, surface="match",
                      fill_value=0.0, search_radius=8, strict=True):
    """Nearest valid same-surface interpolation mirror for a 2-D field."""
    field = np.asarray(field, dtype=np.float64)
    source_landmask = np.asarray(source_landmask, dtype=bool)
    target_landmask = np.asarray(target_landmask, dtype=bool)
    y, x = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    if field.shape != source_landmask.shape or field.shape != (
            len(latitude), len(longitude)):
        raise ValueError("field/source_landmask shape does not match source axes")
    if target_landmask.shape != y.shape:
        raise ValueError("target_landmask shape does not match target coordinates")
    if surface == "match":
        active = np.ones_like(target_landmask)
        desired_land = target_landmask
    elif surface == "land":
        active = target_landmask
        desired_land = np.ones_like(target_landmask)
    elif surface == "water":
        active = ~target_landmask
        desired_land = np.zeros_like(target_landmask)
    else:
        raise ValueError("surface must be 'match', 'land', or 'water'")

    center_y = np.rint(y).astype(np.int64)
    center_x = np.rint(x).astype(np.int64)
    best_distance = np.full(y.shape, np.inf, dtype=np.float64)
    best_value = np.full(y.shape, float(fill_value), dtype=np.float64)
    for dj in range(-search_radius, search_radius + 1):
        jy = center_y + dj
        in_y = (jy >= 0) & (jy < len(latitude))
        jy_safe = np.clip(jy, 0, len(latitude) - 1)
        for di in range(-search_radius, search_radius + 1):
            ix = center_x + di
            inside = in_y & (ix >= 0) & (ix < len(longitude))
            ix_safe = np.clip(ix, 0, len(longitude) - 1)
            value = field[jy_safe, ix_safe]
            valid = (active & inside & np.isfinite(value)
                     & (source_landmask[jy_safe, ix_safe] == desired_land))
            distance = (y - jy_safe) ** 2 + (x - ix_safe) ** 2
            take = valid & (distance < best_distance)
            best_distance = np.where(take, distance, best_distance)
            best_value = np.where(take, value, best_value)
    if strict and np.any(active & ~np.isfinite(best_distance)):
        raise ValueError("no matching source surface within search_radius")
    return best_value


def rotate_earth_to_grid_np(u_earth, v_earth, sinalpha, cosalpha):
    """WRF earth-relative -> grid-relative vector rotation in float64."""
    u = np.asarray(u_earth, dtype=np.float64)
    v = np.asarray(v_earth, dtype=np.float64)
    sina = np.asarray(sinalpha, dtype=np.float64)
    cosa = np.asarray(cosalpha, dtype=np.float64)
    return u * cosa + v * sina, v * cosa - u * sina


def rotate_grid_to_earth_np(u_grid, v_grid, sinalpha, cosalpha):
    """Inverse of :func:`rotate_earth_to_grid_np` in float64."""
    u = np.asarray(u_grid, dtype=np.float64)
    v = np.asarray(v_grid, dtype=np.float64)
    sina = np.asarray(sinalpha, dtype=np.float64)
    cosa = np.asarray(cosalpha, dtype=np.float64)
    return u * cosa - v * sina, v * cosa + u * sina


def era5_rh_to_water_np(relative_humidity, temperature):
    """Convert ERA5 mixed-phase RH to WRF's water-saturation convention.

    Float64 mirror of WPS v4.6 ``ungrib/src/rrpr.F:fix_gfs_rh`` (ECMWF/ERA5
    branch): below freezing RH is multiplied by ``r/ews`` with ``ews`` the
    Bolton 1980 liquid saturation vapor pressure (hPa), ``eis`` the Murphy
    and Koop 2005 ice saturation vapor pressure (hPa), and ``r`` a linear
    liquid-to-ice blend over 273.15 down to 253.15 K (pure ice below).
    """
    rh = np.asarray(relative_humidity, dtype=np.float64)
    t = np.asarray(temperature, dtype=np.float64)
    if rh.shape != t.shape:
        raise ValueError("relative_humidity and temperature shapes differ")
    eis = 0.01 * np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t)
                        - 0.00728332 * t)
    ews = 6.112 * np.exp(17.67 * (t - 273.15) / ((t - 273.15) + 243.5))
    frac = (273.15 - t) / 20.0
    r = np.where(t > 253.15, frac * eis + (1.0 - frac) * ews, eis)
    return np.where(t <= 273.15, rh * (r / ews), rh)


# ---------------------------------------------------------------------------
# RTE+RRTMGP gas optics (Phase 4 Task 3)
# ---------------------------------------------------------------------------

def np_rrtmgp_col_dry(vmr_h2o, plev):
    """Dry-air molecular column (# cm-2).

    Transcription of RTE+RRTMGP commit fa107a1
    ``rte/kernels/mo_gas_optics_utils.F90:127-152``.  The constants are the
    upstream 2018-SI values in ``mo_gas_optics_constants.F90:11-35``.
    """
    h2o = np.asarray(vmr_h2o, dtype=np.float64)
    plev = np.asarray(plev, dtype=np.float64)
    if plev.shape != (h2o.shape[0], h2o.shape[1] + 1):
        raise ValueError("plev must have shape (ncol,nlay+1)")
    fact = 1.0 / (1.0 + h2o)
    m_air = (0.028964 + 0.018016 * h2o) * fact
    return (np.abs(np.diff(plev, axis=1)) * 6.02214076e23 * fact
            / (10000.0 * m_air * 9.80665))


def _np_rrtmgp_validate_range(name, value, lower, upper, unit):
    """Mirror the frontend's pre-interpolation range rejection."""
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} range contains non-finite values")
    observed = (float(np.min(value)), float(np.max(value)))
    if observed[0] < lower or (upper is not None and observed[1] > upper):
        bound = (f"[{lower:.9g}, {upper:.9g}]" if upper is not None
                 else f"[{lower:.9g}, infinity)")
        raise ValueError(
            f"{name} range [{observed[0]:.9g}, {observed[1]:.9g}] {unit} "
            f"is outside allowed range {bound} {unit}")


class _RRTMGPGasOptics:
    def __init__(self, tau, ssa=None, g=None, col_dry=None):
        self.tau = tau
        self.ssa = ssa
        self.g = g
        self.col_dry = col_dry


def _rrtmgp_interpolation(tables, play, tlay, col_gas):
    """Float64 mirror of ``interpolation`` (gas kernels lines 37-170)."""
    ncol, nlay = play.shape
    dtemp = (tables.temp_ref[-1] - tables.temp_ref[0]) / (tables.ntemp - 1)
    jtemp_1 = np.trunc(
        (tlay - (tables.temp_ref[0] - dtemp)) / dtemp).astype(np.int32)
    jtemp_1 = np.clip(jtemp_1, 1, tables.ntemp - 1)
    jtemp = jtemp_1 - 1
    ftemp = (tlay - tables.temp_ref[jtemp]) / dtemp

    press_log = np.log(tables.press_ref)
    dpress = (press_log[-1] - press_log[0]) / (tables.npres - 1)
    locpress = 1.0 + (np.log(play) - press_log[0]) / dpress
    jpress_1 = np.clip(np.trunc(locpress), 1,
                       tables.npres - 1).astype(np.int32)
    jpress = jpress_1 - 1
    fpress = locpress - jpress_1
    tropo = play > tables.press_ref_trop

    col_mix = np.empty((2, ncol, nlay, tables.nflav), np.float64)
    jeta = np.empty((2, ncol, nlay, tables.nflav), np.int32)
    fminor = np.empty((2, 2, ncol, nlay, tables.nflav), np.float64)
    fmajor = np.empty((2, 2, 2, ncol, nlay, tables.nflav),
                      np.float64)
    itropo = np.where(tropo, 0, 1)
    for iflav, (igas1, igas2) in enumerate(tables.flavor):
        for itemp in range(2):
            jt = jtemp + itemp
            ratio = (tables.vmr_ref[itropo, igas1, jt]
                     / tables.vmr_ref[itropo, igas2, jt])
            mix = col_gas[:, :, igas1] + ratio * col_gas[:, :, igas2]
            col_mix[itemp, :, :, iflav] = mix
            eta = np.full_like(mix, 0.5)
            np.divide(col_gas[:, :, igas1], mix, out=eta,
                      where=mix > 2.0 * np.finfo(np.float64).tiny)
            loceta = eta * (tables.neta - 1)
            # Fortran: min(int(loceta)+1,neta-1), then convert to zero base.
            je = np.minimum(np.trunc(loceta).astype(np.int32),
                            tables.neta - 2)
            jeta[itemp, :, :, iflav] = je
            feta = loceta - np.trunc(loceta)
            ftemp_term = (1.0 - ftemp) if itemp == 0 else ftemp
            fminor[0, itemp, :, :, iflav] = (1.0 - feta) * ftemp_term
            fminor[1, itemp, :, :, iflav] = feta * ftemp_term
            fmajor[0, 0, itemp, :, :, iflav] = \
                (1.0 - fpress) * fminor[0, itemp, :, :, iflav]
            fmajor[1, 0, itemp, :, :, iflav] = \
                (1.0 - fpress) * fminor[1, itemp, :, :, iflav]
            fmajor[0, 1, itemp, :, :, iflav] = \
                fpress * fminor[0, itemp, :, :, iflav]
            fmajor[1, 1, itemp, :, :, iflav] = \
                fpress * fminor[1, itemp, :, :, iflav]
    return (jtemp, jpress, tropo, jeta, col_mix, fminor, fmajor)


def _rrtmgp_interp2(k, start, end, jtemp, jeta, fminor):
    """Lines 741-763, with a contiguous contributor/g-point interval."""
    inds = np.arange(start, end + 1)
    je1, je2 = int(jeta[0]), int(jeta[1])
    jt = int(jtemp)
    return (fminor[0, 0] * k[jt, je1, inds]
            + fminor[1, 0] * k[jt, je1 + 1, inds]
            + fminor[0, 1] * k[jt + 1, je2, inds]
            + fminor[1, 1] * k[jt + 1, je2 + 1, inds])


def _rrtmgp_interp3(tables, k, gstart, gend, jtemp, jpress, itropo,
                    jeta, fmajor, scaling):
    """Lines 765-803, with Python's zero-based pressure offsets."""
    g = np.arange(gstart, gend + 1)
    jt, jp = int(jtemp), int(jpress) + int(itropo)
    je1, je2 = int(jeta[0]), int(jeta[1])
    low = scaling[0] * (
        fmajor[0, 0, 0] * k[jt, je1, jp, g]
        + fmajor[1, 0, 0] * k[jt, je1 + 1, jp, g]
        + fmajor[0, 1, 0] * k[jt, je1, jp + 1, g]
        + fmajor[1, 1, 0] * k[jt, je1 + 1, jp + 1, g])
    high = scaling[1] * (
        fmajor[0, 0, 1] * k[jt + 1, je2, jp, g]
        + fmajor[1, 0, 1] * k[jt + 1, je2 + 1, jp, g]
        + fmajor[0, 1, 1] * k[jt + 1, je2, jp + 1, g]
        + fmajor[1, 1, 1] * k[jt + 1, je2 + 1, jp + 1, g])
    return low + high


def np_rrtmgp_gas_optics(tables, play, plev, tlay, vmr):
    """Float64 gas absorption/Rayleigh mirror for one LW or SW table.

    This transcribes RRTMGP ``compute_tau_absorption`` and its major/minor
    helpers (``mo_gas_optics_rrtmgp_kernels.F90:176-501``), plus Rayleigh
    scattering (lines 506-565).  Arrays use ``(ncol,nlay,ngpt)`` internally;
    ``vmr[...,0]`` is ignored because slot zero is the dry-air column.
    """
    play = np.ascontiguousarray(np.asarray(play, dtype=np.float64))
    plev = np.ascontiguousarray(np.asarray(plev, dtype=np.float64))
    tlay = np.ascontiguousarray(np.asarray(tlay, dtype=np.float64))
    vmr = np.ascontiguousarray(np.asarray(vmr, dtype=np.float64))
    if play.shape != tlay.shape or plev.shape != (play.shape[0],
                                                  play.shape[1] + 1):
        raise ValueError("inconsistent play/plev/tlay shapes")
    if vmr.shape != (*play.shape, tables.ngas + 1):
        raise ValueError("vmr must have shape (ncol,nlay,ngas+1)")
    _np_rrtmgp_validate_range("play", play, float(np.min(tables.press_ref)),
                              float(np.max(tables.press_ref)), "Pa")
    _np_rrtmgp_validate_range("plev", plev, 0.0, None, "Pa")
    _np_rrtmgp_validate_range("tlay", tlay, float(np.min(tables.temp_ref)),
                              float(np.max(tables.temp_ref)), "K")
    idx_h2o = tables.gas_index["h2o"]
    col_dry = np_rrtmgp_col_dry(vmr[:, :, idx_h2o], plev)
    col_gas = vmr * col_dry[:, :, None]
    col_gas[:, :, 0] = col_dry
    (jtemp, jpress, tropo, jeta, col_mix,
     fminor, fmajor) = _rrtmgp_interpolation(tables, play, tlay, col_gas)

    ncol, nlay = play.shape
    tau_abs = np.zeros((ncol, nlay, tables.ngpt), np.float64)
    # Major species: mo_gas_optics_rrtmgp_kernels.F90:345-396.
    for icol in range(ncol):
        for ilay in range(nlay):
            iatm = 0 if tropo[icol, ilay] else 1
            for gstart, gend in tables.band_lims_gpt:
                iflav = int(tables.gpoint_flavor[iatm, gstart])
                tau_abs[icol, ilay, gstart:gend + 1] += _rrtmgp_interp3(
                    tables, tables.kmajor, gstart, gend,
                    jtemp[icol, ilay], jpress[icol, ilay], iatm,
                    jeta[:, icol, ilay, iflav],
                    fmajor[:, :, :, icol, ilay, iflav],
                    col_mix[:, icol, ilay, iflav])

    # Minor species: lines 402-501.  Layer-limit construction in the caller
    # is equivalent to selecting the appropriate atmosphere per cell here.
    for iatm in range(2):
        if iatm == 0:
            limits = tables.minor_limits_gpt_lower
            density = tables.minor_scales_with_density_lower
            complement = tables.scale_by_complement_lower
            idx_minor = tables.idx_minor_lower
            idx_scaling = tables.idx_minor_scaling_lower
            starts = tables.kminor_start_lower
            kminor = tables.kminor_lower
            mask = tropo
        else:
            limits = tables.minor_limits_gpt_upper
            density = tables.minor_scales_with_density_upper
            complement = tables.scale_by_complement_upper
            idx_minor = tables.idx_minor_upper
            idx_scaling = tables.idx_minor_scaling_upper
            starts = tables.kminor_start_upper
            kminor = tables.kminor_upper
            mask = ~tropo
        for imnr, (gstart, gend) in enumerate(limits):
            iflav = int(tables.gpoint_flavor[iatm, gstart])
            width = int(gend - gstart)
            for icol, ilay in np.argwhere(mask):
                scaling = col_gas[icol, ilay, idx_minor[imnr]]
                if density[imnr]:
                    scaling *= 0.01 * play[icol, ilay] / tlay[icol, ilay]
                    if idx_scaling[imnr] > 0:
                        vmr_fact = 1.0 / col_gas[icol, ilay, 0]
                        dry_fact = 1.0 / (
                            1.0 + col_gas[icol, ilay, idx_h2o] * vmr_fact)
                        special = (col_gas[icol, ilay, idx_scaling[imnr]]
                                   * vmr_fact * dry_fact)
                        scaling *= ((1.0 - special) if complement[imnr]
                                    else special)
                coeff = _rrtmgp_interp2(
                    kminor, int(starts[imnr]), int(starts[imnr]) + width,
                    jtemp[icol, ilay], jeta[:, icol, ilay, iflav],
                    fminor[:, :, icol, ilay, iflav])
                tau_abs[icol, ilay, gstart:gend + 1] += scaling * coeff

    if tables.kind == "lw":
        return _RRTMGPGasOptics(tau_abs, col_dry=col_dry)

    # Rayleigh: lines 506-565; combine_abs_and_rayleigh is frontend lines
    # 1954-2020 (g=0, ssa=tau_rayleigh/tau_total).
    tau_rayleigh = np.empty_like(tau_abs)
    for icol in range(ncol):
        for ilay in range(nlay):
            iatm = 0 if tropo[icol, ilay] else 1
            for gstart, gend in tables.band_lims_gpt:
                iflav = int(tables.gpoint_flavor[iatm, gstart])
                coeff = _rrtmgp_interp2(
                    tables.rayleigh[iatm], int(gstart), int(gend),
                    jtemp[icol, ilay], jeta[:, icol, ilay, iflav],
                    fminor[:, :, icol, ilay, iflav])
                tau_rayleigh[icol, ilay, gstart:gend + 1] = coeff * (
                    col_gas[icol, ilay, idx_h2o] + col_dry[icol, ilay])
    tau = tau_abs + tau_rayleigh
    ssa = tau_rayleigh / np.maximum(3.0 * np.finfo(np.float64).tiny, tau)
    return _RRTMGPGasOptics(tau, ssa, np.zeros_like(tau), col_dry)


class _RRTMGPFluxes:
    def __init__(self, flux_up, flux_dn, flux_dir=None):
        self.flux_up = flux_up
        self.flux_dn = flux_dn
        self.flux_dir = flux_dir


_GAUSS_D = {
    1: np.array([1.0 / 0.6096748751]),
    2: np.array([1.0 / 0.2509907356, 1.0 / 0.7908473988]),
    3: np.array([1.0 / 0.1024922169, 1.0 / 0.4417960320,
                 1.0 / 0.8633751621]),
    4: np.array([1.0 / 0.0454586727, 1.0 / 0.2322334416,
                 1.0 / 0.5740198775, 1.0 / 0.9030775973]),
}
_GAUSS_W = {
    1: np.array([1.0]),
    2: np.array([0.2300253764, 0.7699746236]),
    3: np.array([0.0437820218, 0.3875796738, 0.5686383044]),
    4: np.array([0.0092068785, 0.1285704278, 0.4323381850,
                 0.4298845087]),
}


def np_rrtmgp_lw_rte(tau, lay_source, lev_source, sfc_source, sfc_emis,
                      incident_flux=None, *, top_at_1: bool,
                      n_angles: int = 1):
    """Float64 LW no-scattering source/transport solver.

    Transcription of ``lw_solver_noscat_oneangle`` and its source/transport
    helpers in ``mo_rte_solver_kernels.F90:51-240,620-745``.  The quadrature
    values are the reference frontend defaults (``mo_rte_lw.F90:135-160``).
    """
    tau = np.asarray(tau, dtype=np.float64)
    lay_source = np.asarray(lay_source, dtype=np.float64)
    lev_source = np.asarray(lev_source, dtype=np.float64)
    sfc_source = np.asarray(sfc_source, dtype=np.float64)
    sfc_emis = np.asarray(sfc_emis, dtype=np.float64)
    if tau.ndim != 3:
        raise ValueError("tau must have shape (ncol,nlay,ngpt)")
    ncol, nlay, ngpt = tau.shape
    if lay_source.shape != tau.shape or lev_source.shape != (
            ncol, nlay + 1, ngpt):
        raise ValueError("inconsistent LW source shapes")
    if sfc_source.shape != (ncol, ngpt) or sfc_emis.shape != (ncol, ngpt):
        raise ValueError("inconsistent LW surface source/emissivity shapes")
    if n_angles not in _GAUSS_D:
        raise ValueError("n_angles must be 1, 2, 3, or 4")
    incident = (np.zeros((ncol, ngpt), np.float64)
                if incident_flux is None
                else np.asarray(incident_flux, dtype=np.float64))
    up_total = np.zeros((ncol, nlay + 1), np.float64)
    dn_total = np.zeros_like(up_total)
    top_level = 0 if top_at_1 else nlay
    sfc_level = nlay if top_at_1 else 0
    # Match epsilon(1._wp)^(1/4) using this mirror's float64 working real.
    tau_thresh = np.sqrt(np.sqrt(np.finfo(tau.dtype).eps))

    for d, weight in zip(_GAUSS_D[n_angles], _GAUSS_W[n_angles]):
        for gpt in range(ngpt):
            tau_loc = tau[:, :, gpt] * d
            trans = np.exp(-tau_loc)
            fact = np.where(
                tau_loc > tau_thresh,
                (1.0 - trans) / tau_loc - trans,
                tau_loc * (0.5 + tau_loc * (-1.0 / 3.0
                                             + tau_loc / 8.0)))
            source_inc = ((1.0 - trans) * lev_source[:, 1:, gpt]
                          + 2.0 * fact
                          * (lay_source[:, :, gpt]
                             - lev_source[:, 1:, gpt]))
            source_dec = ((1.0 - trans) * lev_source[:, :-1, gpt]
                          + 2.0 * fact
                          * (lay_source[:, :, gpt]
                             - lev_source[:, :-1, gpt]))
            source_dn = source_inc if top_at_1 else source_dec
            source_up = source_dec if top_at_1 else source_inc
            up = np.empty((ncol, nlay + 1), np.float64)
            dn = np.empty_like(up)
            dn[:, top_level] = incident[:, gpt] / (np.pi * weight)
            if top_at_1:
                for ilev in range(1, nlay + 1):
                    dn[:, ilev] = (trans[:, ilev - 1] * dn[:, ilev - 1]
                                     + source_dn[:, ilev - 1])
            else:
                for ilev in range(nlay - 1, -1, -1):
                    dn[:, ilev] = (trans[:, ilev] * dn[:, ilev + 1]
                                     + source_dn[:, ilev])
            up[:, sfc_level] = (dn[:, sfc_level]
                                * (1.0 - sfc_emis[:, gpt])
                                + sfc_emis[:, gpt] * sfc_source[:, gpt])
            if top_at_1:
                for ilev in range(nlay - 1, -1, -1):
                    up[:, ilev] = (trans[:, ilev] * up[:, ilev + 1]
                                     + source_up[:, ilev])
            else:
                for ilev in range(1, nlay + 1):
                    up[:, ilev] = (trans[:, ilev - 1] * up[:, ilev - 1]
                                     + source_up[:, ilev - 1])
            up_total += np.pi * weight * up
            dn_total += np.pi * weight * dn
    return _RRTMGPFluxes(up_total, dn_total)


def np_rrtmgp_delta_scale(tau, ssa, g):
    """Reference-default delta scaling, ``f=g^2`` (optics lines 76-98)."""
    tau = np.asarray(tau, dtype=np.float64)
    ssa = np.asarray(ssa, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    f = g * g
    wf = ssa * f
    eps = 3.0 * np.finfo(np.float64).tiny
    return ((1.0 - wf) * tau,
            (ssa - wf) / np.maximum(eps, 1.0 - wf),
            (g - f) / np.maximum(eps, 1.0 - f))


def np_rrtmgp_sw_rte(tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif,
                      inc_flux_dir, *, top_at_1: bool):
    """Float64 PIFM two-stream SW solver.

    Transcription of ``sw_solver_2stream``, ``sw_dif_and_source``, and
    ``adding`` in ``mo_rte_solver_kernels.F90:503-609,985-1245``.
    Inputs are already delta-scaled, as in the reference frontend default.
    """
    tau = np.asarray(tau, dtype=np.float64)
    ssa = np.asarray(ssa, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if tau.shape != ssa.shape or tau.shape != g.shape or tau.ndim != 3:
        raise ValueError("tau/ssa/g must share (ncol,nlay,ngpt)")
    ncol, nlay, ngpt = tau.shape
    mu0 = np.asarray(mu0, dtype=np.float64)
    if mu0.shape == (ncol,):
        mu0 = np.broadcast_to(mu0[:, None], (ncol, nlay))
    if mu0.shape != (ncol, nlay):
        raise ValueError("mu0 must have shape (ncol,) or (ncol,nlay)")
    alb_dir = np.asarray(sfc_alb_dir, dtype=np.float64)
    alb_dif = np.asarray(sfc_alb_dif, dtype=np.float64)
    inc = np.asarray(inc_flux_dir, dtype=np.float64)
    if any(x.shape != (ncol, ngpt) for x in (alb_dir, alb_dif, inc)):
        raise ValueError("SW spectral boundary arrays have wrong shape")
    up_total = np.zeros((ncol, nlay + 1), np.float64)
    dn_total = np.zeros_like(up_total)
    dir_total = np.zeros_like(up_total)
    top_level, top_layer = ((0, 0) if top_at_1 else (nlay, nlay - 1))
    # The CUDA port evaluates this solver in FP32.  Preserve the reference
    # epsilon(1._wp) branches at the device working-precision floor while
    # retaining float64 arithmetic for the mirror itself.
    eps = np.finfo(np.float32).eps
    min_k = 1.0e4 * eps
    min_mu0 = np.sqrt(eps)

    for gpt in range(ngpt):
        direct = np.empty((ncol, nlay + 1), np.float64)
        diffuse_dn = np.empty_like(direct)
        diffuse_dn[:, top_level] = 0.0
        direct[:, top_level] = inc[:, gpt] * mu0[:, top_layer]
        rdif = np.empty((ncol, nlay), np.float64)
        tdif = np.empty_like(rdif)
        source_dn = np.empty_like(rdif)
        source_up = np.empty_like(rdif)
        for j in range(nlay):
            lay = j if top_at_1 else nlay - j - 1
            inc_level = lay if top_at_1 else lay + 1
            trans_level = lay + 1 if top_at_1 else lay
            tau_s, w0_s, g_s = (tau[:, lay, gpt], ssa[:, lay, gpt],
                                g[:, lay, gpt])
            gamma1 = (8.0 - w0_s * (5.0 + 3.0 * g_s)) * 0.25
            gamma2 = 3.0 * w0_s * (1.0 - g_s) * 0.25
            k = np.sqrt(np.maximum((gamma1 - gamma2)
                                   * (gamma1 + gamma2), min_k))
            expkt = np.exp(-tau_s * k)
            exp2 = expkt * expkt
            rt = 1.0 / (k * (1.0 + exp2) + gamma1 * (1.0 - exp2))
            rdif[:, lay] = rt * gamma2 * (1.0 - exp2)
            tdif[:, lay] = rt * 2.0 * k * expkt
            mu = np.maximum(min_mu0, mu0[:, lay])
            kmu = k * mu
            denom = 1.0 - kmu * kmu
            denom = np.where(np.abs(denom) >= eps, denom, eps)
            rt = w0_s * rt / denom
            gamma3 = (2.0 - 3.0 * mu * g_s) * 0.25
            gamma4 = 1.0 - gamma3
            alpha1 = gamma1 * gamma4 + gamma2 * gamma3
            alpha2 = gamma1 * gamma3 + gamma2 * gamma4
            kg3, kg4 = k * gamma3, k * gamma4
            tnoscat = np.exp(-tau_s / mu)
            rdir = rt * (
                (1.0 - kmu) * (alpha2 + kg3)
                - (1.0 + kmu) * (alpha2 - kg3) * exp2
                - 2.0 * (kg3 - alpha2 * kmu) * expkt * tnoscat)
            tdir = -rt * (
                (1.0 + kmu) * (alpha1 + kg4) * tnoscat
                - (1.0 - kmu) * (alpha1 - kg4) * exp2 * tnoscat
                - 2.0 * (kg4 + alpha1 * kmu) * expkt)
            rdir = np.maximum(0.0, np.minimum(rdir, 1.0 - tnoscat))
            tdir = np.maximum(0.0,
                              np.minimum(tdir, 1.0 - tnoscat - rdir))
            source_up[:, lay] = rdir * direct[:, inc_level]
            source_dn[:, lay] = tdir * direct[:, inc_level]
            direct[:, trans_level] = tnoscat * direct[:, inc_level]
        sfc_layer = nlay - 1 if top_at_1 else 0
        sfc_level = nlay if top_at_1 else 0
        source_sfc = np.where(mu0[:, sfc_layer] > 0.0,
                              direct[:, sfc_level] * alb_dir[:, gpt], 0.0)
        night = mu0 <= 0.0
        source_up[night] = 0.0
        source_dn[night] = 0.0

        # Diffuse adding method, SH08 equations 9-13.
        albedo = np.empty((ncol, nlay + 1), np.float64)
        source = np.empty_like(albedo)
        denom_add = np.empty((ncol, nlay), np.float64)
        diffuse_up = np.empty_like(albedo)
        if top_at_1:
            albedo[:, nlay] = alb_dif[:, gpt]
            source[:, nlay] = source_sfc
            for lev in range(nlay - 1, -1, -1):
                denom_add[:, lev] = 1.0 / (
                    1.0 - rdif[:, lev] * albedo[:, lev + 1])
                albedo[:, lev] = (rdif[:, lev]
                    + tdif[:, lev] ** 2 * albedo[:, lev + 1]
                    * denom_add[:, lev])
                source[:, lev] = (source_up[:, lev]
                    + tdif[:, lev] * denom_add[:, lev]
                    * (source[:, lev + 1]
                       + albedo[:, lev + 1] * source_dn[:, lev]))
            diffuse_up[:, 0] = diffuse_dn[:, 0] * albedo[:, 0] + source[:, 0]
            for lev in range(1, nlay + 1):
                l = lev - 1
                diffuse_dn[:, lev] = (
                    tdif[:, l] * diffuse_dn[:, lev - 1]
                    + rdif[:, l] * source[:, lev] + source_dn[:, l]
                    ) * denom_add[:, l]
                diffuse_up[:, lev] = (diffuse_dn[:, lev] * albedo[:, lev]
                                       + source[:, lev])
        else:
            albedo[:, 0] = alb_dif[:, gpt]
            source[:, 0] = source_sfc
            for lev in range(nlay):
                denom_add[:, lev] = 1.0 / (
                    1.0 - rdif[:, lev] * albedo[:, lev])
                albedo[:, lev + 1] = (rdif[:, lev]
                    + tdif[:, lev] ** 2 * albedo[:, lev]
                    * denom_add[:, lev])
                source[:, lev + 1] = (source_up[:, lev]
                    + tdif[:, lev] * denom_add[:, lev]
                    * (source[:, lev] + albedo[:, lev] * source_dn[:, lev]))
            diffuse_up[:, nlay] = (diffuse_dn[:, nlay] * albedo[:, nlay]
                                    + source[:, nlay])
            for lev in range(nlay - 1, -1, -1):
                diffuse_dn[:, lev] = (
                    tdif[:, lev] * diffuse_dn[:, lev + 1]
                    + rdif[:, lev] * source[:, lev] + source_dn[:, lev]
                    ) * denom_add[:, lev]
                diffuse_up[:, lev] = (diffuse_dn[:, lev] * albedo[:, lev]
                                       + source[:, lev])
        up_total += diffuse_up
        dn_total += diffuse_dn + direct
        dir_total += direct
    return _RRTMGPFluxes(up_total, dn_total, dir_total)


class _RRTMGPPlanckSources:
    def __init__(self, lay_source, lev_source, sfc_source):
        self.lay_source = lay_source
        self.lev_source = lev_source
        self.sfc_source = sfc_source


def _rrtmgp_planck_band(tables, temperature):
    """``interpolate1D`` (gas-optics kernel lines 715-737)."""
    temperature = np.asarray(temperature, dtype=np.float64)
    delta = ((tables.temp_ref[-1] - tables.temp_ref[0])
             / (tables.totplnk.shape[0] - 1))
    val0 = (temperature - tables.temp_ref[0]) / delta
    frac = val0 - np.trunc(val0)
    idx = np.clip(np.trunc(val0).astype(np.int32),
                  0, tables.totplnk.shape[0] - 2)
    return (tables.totplnk[idx]
            + frac[..., None] * (tables.totplnk[idx + 1]
                                 - tables.totplnk[idx]))


def np_rrtmgp_planck_sources(tables, play, plev, tlay, tlev, tsfc, vmr):
    """Float64 RRTMGP Planck fractions and layer/level/surface sources.

    Transcription of ``compute_Planck_source`` in
    ``mo_gas_optics_rrtmgp_kernels.F90:568-710``.  It shares the exact gas
    interpolation coefficients used by :func:`np_rrtmgp_gas_optics`.
    """
    play = np.asarray(play, dtype=np.float64)
    plev = np.asarray(plev, dtype=np.float64)
    tlay = np.asarray(tlay, dtype=np.float64)
    tlev = np.asarray(tlev, dtype=np.float64)
    tsfc = np.asarray(tsfc, dtype=np.float64)
    vmr = np.asarray(vmr, dtype=np.float64)
    ncol, nlay = play.shape
    if tables.kind != "lw":
        raise ValueError("Planck sources require LW tables")
    if tlev.shape != (ncol, nlay + 1) or tsfc.shape != (ncol,):
        raise ValueError("inconsistent Planck temperature shapes")
    _np_rrtmgp_validate_range("play", play, float(np.min(tables.press_ref)),
                              float(np.max(tables.press_ref)), "Pa")
    _np_rrtmgp_validate_range("plev", plev, 0.0, None, "Pa")
    for name, value in (("tlay", tlay), ("tlev", tlev), ("tsfc", tsfc)):
        _np_rrtmgp_validate_range(
            name, value, float(np.min(tables.temp_ref)),
            float(np.max(tables.temp_ref)), "K")
    idx_h2o = tables.gas_index["h2o"]
    col_dry = np_rrtmgp_col_dry(vmr[:, :, idx_h2o], plev)
    col_gas = vmr * col_dry[:, :, None]
    col_gas[:, :, 0] = col_dry
    (jtemp, jpress, tropo, jeta, _col_mix,
     _fminor, fmajor) = _rrtmgp_interpolation(tables, play, tlay, col_gas)
    pfrac = np.empty((ncol, nlay, tables.ngpt), np.float64)
    for icol in range(ncol):
        for ilay in range(nlay):
            iatm = 0 if tropo[icol, ilay] else 1
            for gstart, gend in tables.band_lims_gpt:
                iflav = int(tables.gpoint_flavor[iatm, gstart])
                pfrac[icol, ilay, gstart:gend + 1] = _rrtmgp_interp3(
                    tables, tables.planck_fraction, gstart, gend,
                    jtemp[icol, ilay], jpress[icol, ilay], iatm,
                    jeta[:, icol, ilay, iflav],
                    fmajor[:, :, :, icol, ilay, iflav], (1.0, 1.0))
    lay_planck = _rrtmgp_planck_band(tables, tlay)
    lev_planck = _rrtmgp_planck_band(tables, tlev)
    sfc_planck = _rrtmgp_planck_band(tables, tsfc)
    lay_source = np.empty_like(pfrac)
    lev_source = np.empty((ncol, nlay + 1, tables.ngpt), np.float64)
    sfc_source = np.empty((ncol, tables.ngpt), np.float64)
    sfc_lay = nlay - 1 if play[0, 0] < play[0, -1] else 0
    for gpt, band in enumerate(tables.gpoint_bands):
        lay_source[:, :, gpt] = pfrac[:, :, gpt] * lay_planck[:, :, band]
        sfc_source[:, gpt] = pfrac[:, sfc_lay, gpt] * sfc_planck[:, band]
        lev_source[:, 0, gpt] = pfrac[:, 0, gpt] * lev_planck[:, 0, band]
        lev_source[:, 1:nlay, gpt] = np.sqrt(
            pfrac[:, :-1, gpt] * pfrac[:, 1:, gpt]) \
            * lev_planck[:, 1:nlay, band]
        lev_source[:, nlay, gpt] = (pfrac[:, nlay - 1, gpt]
                                             * lev_planck[:, nlay, band])
    return _RRTMGPPlanckSources(lay_source, lev_source, sfc_source)


class _RRTMGPCloudOptics:
    def __init__(self, tau, ssa, g):
        self.tau = tau
        self.ssa = ssa
        self.g = g


def np_rrtmgp_cloud_optics(tables, clwp, ciwp, reliq, dgice):
    """Float64 mirror of RRTMGP's liquid/ice cloud-table interpolation.

    This transcribes ``compute_cld_from_table`` and the liquid/ice mixture
    in ``mo_cloud_optics_rrtmgp.F90:344-422``. The medium ice roughness
    selected by the reference all-sky example is used.
    """
    clwp = np.asarray(clwp, dtype=np.float64)
    ciwp = np.asarray(ciwp, dtype=np.float64)
    reliq = np.asarray(reliq, dtype=np.float64)
    dgice = np.asarray(dgice, dtype=np.float64)
    if clwp.ndim != 2 or any(a.shape != clwp.shape
                            for a in (ciwp, reliq, dgice)):
        raise ValueError("cloud inputs must share shape (ncol,nlay)")

    def component(path, size, ext, ssat, asyt, offset, step):
        pos = (size - offset) / step
        index = np.clip(np.floor(pos).astype(np.int32), 0,
                        ext.shape[0] - 2)
        fraction = pos - index
        gather = np.arange(tables.nband)[None, None, :]
        low = index[..., None]
        e = ext[low, gather] + fraction[..., None] * (
            ext[low + 1, gather] - ext[low, gather])
        w = ssat[low, gather] + fraction[..., None] * (
            ssat[low + 1, gather] - ssat[low, gather])
        gg = asyt[low, gather] + fraction[..., None] * (
            asyt[low + 1, gather] - asyt[low, gather])
        tau = np.where(path[..., None] > 0.0, path[..., None] * e, 0.0)
        taussa = tau * w
        return tau, taussa, taussa * gg

    liquid = component(clwp, reliq, tables.extliq, tables.ssaliq,
                       tables.asyliq, tables.radliq_lwr,
                       tables.liq_step_size)
    ice = component(ciwp, dgice, tables.extice[:, :, 1],
                    tables.ssaice[:, :, 1], tables.asyice[:, :, 1],
                    tables.diamice_lwr, tables.ice_step_size)
    tau = liquid[0] + ice[0]
    taussa = liquid[1] + ice[1]
    ssa = taussa / np.maximum(np.finfo(np.float64).eps, tau)
    asym = (liquid[2] + ice[2]) / np.maximum(
        np.finfo(np.float64).eps, taussa)
    return _RRTMGPCloudOptics(tau, ssa, asym)


class _RRTMGPHydrometeorPaths:
    def __init__(self, clwp, ciwp, reliq, dgice):
        self.clwp = clwp
        self.ciwp = ciwp
        self.reliq = reliq
        self.dgice = dgice


def np_rrtmgp_hydrometeor_paths(plev, qc, qr=None, qi=None, qs=None, *,
                                microphysics="kessler", play=None, tlay=None,
                                nc=None, nr=None, ni=None, ns=None,
                                effc=None, effr=None, effi=None, effs=None,
                                cldfra=None):
    """Float64 mirror of the RRTMGP hydrometeor coupling boundary.

    Pressure thickness converts dry-air mixing ratios to condensed water
    paths in g m-2. Per WRF the radiation liquid path is cloud water only
    and ice is cloud ice plus snow; rain never enters the paths
    (module_ra_rrtmg_sw.F:11029-11034, module_ra_rrtmg_lw.F:12488-12493).
    With ``cldfra`` the grid-box paths become in-cloud paths through WRF's
    ``max(0.01, cldfrac)`` division (same lines).  WSM6, Thompson, and NSSL
    consume scheme-native cloud/ice/snow radii in microns and mass-weight
    ice plus snow into the RRTMGP ice diameter.  Morrison liquid size
    is the cloud-droplet gamma radius alone -- rain carries no radiative
    mass, so it contributes no radius either (same driver lines; ``effr``
    is accepted for interface parity with Morrison's diagnostics and
    ignored).  Ice/snow combine by number.  The final size limits are
    the domains of the v1.9 cloud-optics tables.
    """
    plev = np.asarray(plev, dtype=np.float64)
    qc = np.asarray(qc, dtype=np.float64)
    if qc.ndim != 2 or plev.shape != (qc.shape[0], qc.shape[1] + 1):
        raise ValueError("plev/qc must have shapes (ncol,nlay+1)/(ncol,nlay)")

    def field(value, name):
        if value is None:
            return np.zeros_like(qc)
        out = np.asarray(value, dtype=np.float64)
        if out.shape != qc.shape:
            raise ValueError(f"{name} must have shape {qc.shape}, got {out.shape}")
        return out

    qr, qi, qs = field(qr, "qr"), field(qi, "qi"), field(qs, "qs")
    masses = (qc, qr, qi, qs)
    if not all(np.all(np.isfinite(value)) for value in (plev, *masses)):
        raise ValueError("hydrometeor pressure and mass inputs must be finite")
    if any(np.any(value < 0.0) for value in masses):
        raise ValueError("hydrometeor mixing ratios must be non-negative")
    mass_path = np.abs(np.diff(plev, axis=1)) * (1000.0 / 9.80665)
    clwp = qc * mass_path
    ciwp = (qi + qs) * mass_path
    if cldfra is not None:
        cldfra = field(cldfra, "cldfra")
        if not np.all(np.isfinite(cldfra)) or np.any(cldfra < 0.0) \
                or np.any(cldfra > 1.0):
            raise ValueError("cldfra must be finite and within [0, 1]")
        clwp = clwp / np.maximum(0.01, cldfra)
        ciwp = ciwp / np.maximum(0.01, cldfra)
    scheme = str(microphysics).lower()
    if scheme == "kessler":
        return _RRTMGPHydrometeorPaths(
            clwp, ciwp, np.full_like(qc, 10.0), np.full_like(qc, 50.0))
    if scheme in ("wsm6", "thompson", "nssl"):
        if any(x is None for x in (effc, effi, effs)):
            raise ValueError(
                f"{scheme} radii require effc, effi, and effs")
        re_c, re_i, re_s = (
            field(value, name) for value, name in
            ((effc, "effc"), (effi, "effi"), (effs, "effs")))
        radii = (re_c, re_i, re_s)
        if (not all(np.all(np.isfinite(value)) for value in radii)
                or any(np.any(value < 0.0) for value in radii)):
            raise ValueError(
                f"{scheme} effective radii must be finite and non-negative")
        tiny = 1.0e-20
        frozen = qi + qs
        reice = np.where(
            frozen > tiny,
            (qi * re_i + qs * re_s) / np.maximum(frozen, tiny),
            25.0)
        reliq = np.where(qc > tiny, re_c, 10.0)
        return _RRTMGPHydrometeorPaths(
            clwp, ciwp, np.clip(reliq, 2.5, 21.5),
            np.clip(2.0 * reice, 10.0, 180.0))
    if scheme != "morrison":
        raise ValueError(
            "microphysics must be 'kessler', 'wsm6', 'thompson', 'nssl', "
            "or 'morrison'")
    if play is None or tlay is None or any(x is None for x in (nc, nr, ni, ns)):
        raise ValueError("Morrison radii require play, tlay, nc, nr, ni, ns")
    play, tlay = field(play, "play"), field(tlay, "tlay")
    nc, nr = field(nc, "nc"), field(nr, "nr")
    ni, ns = field(ni, "ni"), field(ns, "ns")
    numbers = (nc, nr, ni, ns)
    if not all(np.all(np.isfinite(value)) for value in (play, tlay, *numbers)):
        raise ValueError("Morrison thermodynamic and number inputs must be finite")
    if any(np.any(value < 0.0) for value in numbers):
        raise ValueError("Morrison number concentrations must be non-negative")
    tiny = 1.0e-20

    rho_air = play / (287.15 * tlay)
    shape = 0.0005714 * (nc / 1.0e6 * rho_air) + 0.2714
    pgam = np.clip(1.0 / (shape * shape) - 1.0, 2.0, 10.0)
    gamma_ratio = (pgam + 1.0) * (pgam + 2.0) * (pgam + 3.0)
    lam_c = np.power(np.pi * 997.0 / 6.0 * nc * gamma_ratio
                     / np.maximum(qc, tiny), 1.0 / 3.0)
    re_c = (pgam + 3.0) / np.maximum(2.0 * lam_c, tiny) * 1.0e6

    def exponential_radius(qmass, number, density):
        lam = np.power(np.pi * density * number
                       / np.maximum(qmass, tiny), 1.0 / 3.0)
        return 1.5e6 / np.maximum(lam, tiny)

    supplied_effective = (effc, effr, effi, effs)
    if any(value is not None for value in supplied_effective):
        if any(value is None for value in supplied_effective):
            raise ValueError("Morrison effective radii require effc/effr/"
                             "effi/effs together")
        re_c, re_r, re_i, re_s = (
            field(value, name) for value, name in zip(
                supplied_effective, ("effc", "effr", "effi", "effs")))
        if (not all(np.all(np.isfinite(value))
                    for value in (re_c, re_r, re_i, re_s))
                or any(np.any(value < 0.0)
                       for value in (re_c, re_r, re_i, re_s))):
            raise ValueError("Morrison effective radii must be finite and "
                             "non-negative")
    else:
        re_i = exponential_radius(qi, ni, 500.0)
        re_s = exponential_radius(qs, ns, 100.0)
    wc = np.where((qc > 0.0) & (nc > 0.0), nc, 0.0)
    wi = np.where((qi > 0.0) & (ni > 0.0), ni, 0.0)
    ws = np.where((qs > 0.0) & (ns > 0.0), ns, 0.0)
    reliq = np.where(wc > 0.0, re_c, 10.0)
    reice = np.where(wi + ws > 0.0, (wi * re_i + ws * re_s)
                     / np.maximum(wi + ws, tiny), 25.0)
    return _RRTMGPHydrometeorPaths(
        clwp, ciwp, np.clip(reliq, 2.5, 21.5),
        np.clip(2.0 * reice, 10.0, 180.0))


class _RRTMGPRadiationResult:
    def __init__(self, rthratenlw, rthratensw, swdown, glw):
        self.rthratenlw = rthratenlw
        self.rthratensw = rthratensw
        self.swdown = swdown
        self.glw = glw


def np_rrtmgp_fluxes_to_radiation(lw_up, lw_dn, sw_up, sw_dn, plev,
                                  exner, *, ny, nx):
    """Float64 mirror of the column-driver radiation-slot mapping.

    This is the post-RTE path from bottom-to-top broadband level fluxes
    through pressure-coordinate convergence and Exner conversion to separate
    LW/SW potential-temperature tendencies, SWDOWN, and GLW.
    """
    arrays = [np.asarray(value, dtype=np.float64)
              for value in (lw_up, lw_dn, sw_up, sw_dn, plev, exner)]
    lw_up, lw_dn, sw_up, sw_dn, plev, exner = arrays
    ncol, nlay = exner.shape
    level_shape = (ncol, nlay + 1)
    if ncol != ny * nx or any(value.shape != level_shape for value in
                              (lw_up, lw_dn, sw_up, sw_dn, plev)):
        raise ValueError("flux/pressure columns do not match exner or ny*nx")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("radiation flux and thermodynamic inputs must be finite")
    dp = np.abs(np.diff(plev, axis=1))
    if np.any(dp <= 0.0) or np.any(exner <= 0.0):
        raise ValueError("radiation pressure thickness and Exner must be positive")

    def theta_heating(flux_up, flux_dn):
        net_down = flux_dn - flux_up
        convergence = np.diff(net_down, axis=1)
        packed = (c.G / c.CP) * convergence / dp / exner
        return packed.reshape(ny, nx, nlay).transpose(2, 0, 1)

    return _RRTMGPRadiationResult(
        theta_heating(lw_up, lw_dn), theta_heating(sw_up, sw_dn),
        sw_dn[:, 0].reshape(ny, nx), lw_dn[:, 0].reshape(ny, nx))


def np_cal_cldfra1(qv, qc, qi, qs, t, p, *, f_qc=True, f_qi=True,
                   f_qs=True):
    """Float64 mirror of WRF's icloud=1 Xu-Randall cloud fraction.

    Exact transcription of WRF v4.6.1 ``module_radiation_driver.F``
    ``cal_cldfra1`` (lines 3761-3986): Murray (1966) liquid/ice saturation
    weighted by the condensate ice fraction (lines 3861-3865, 3945),
    then the Xu and Randall (1996) fraction with ALPHA0=100, GAMMA=0.49,
    QCLDMIN=1e-12, PEXP=0.25, RHGRID=1.0 (lines 3806-3807, 3950-3979).

    Supported moisture sets mirror the driver's flag dispatch:
    ``f_qc and f_qi and f_qs`` (Morrison-class, lines 3870-3877, QCLD =
    QI+QC+QS with weight (QI+QS)/QCLD), ``f_qc and f_qi and not f_qs``
    (the arm WRF comments "for P3, mp option 50 or 51", lines 3879-3887,
    QCLD = QI+QC with weight QI/QCLD) and ``f_qc`` alone (Kessler-class,
    lines 3891-3899, QCLD = QC with the 273.15 K phase threshold).
    Rain never enters QCLD (lines 3904-3916).
    """
    qv, qc, qi, qs, t, p = (np.asarray(a, dtype=np.float64)
                            for a in (qv, qc, qi, qs, t, p))
    if not f_qc or (f_qs and not f_qi):
        raise NotImplementedError(
            "cal_cldfra1 port carries WRF's three F_QC arms: qc+qi+qs "
            "(module_radiation_driver.F:3870-3877), qc+qi with no snow "
            "species (the P3 arm, :3879-3887) and qc alone (:3891-3899). "
            f"f_qc={bool(f_qc)}/f_qi={bool(f_qi)}/f_qs={bool(f_qs)} is "
            "WRF's mp=5 arm at :3902-3922, whose weight is the F_ICE_PHY "
            "ice fraction this reference does not carry")
    alpha0, gamma, qcldmin, pexp, rhgrid = 100.0, 0.49, 1.0e-12, 0.25, 1.0
    svp1, svp2, svpi2 = 0.61078, 17.2693882, 21.8745584
    svp3, svpi3, svpt0 = 35.86, 7.66, 273.15
    ep2 = 287.0 / 461.6
    tc = t - svpt0
    esw = 1000.0 * svp1 * np.exp(svp2 * tc / (t - svp3))
    esi = 1000.0 * svp1 * np.exp(svpi2 * tc / (t - svpi3))
    qvsw = ep2 * esw / (p - esw)
    qvsi = ep2 * esi / (p - esi)
    if f_qi and f_qs:
        qcld = qi + qc + qs
        weight = np.where(qcld < qcldmin, 0.0,
                          (qi + qs) / np.maximum(qcld, qcldmin))
    elif f_qi:
        # module_radiation_driver.F:3879-3887, "for P3, mp option 50 or
        # 51": one ice category, no snow species, so QS never enters QCLD.
        qcld = qi + qc
        weight = np.where(qcld < qcldmin, 0.0,
                          qi / np.maximum(qcld, qcldmin))
    else:
        qcld = qc
        weight = np.where(qcld < qcldmin, 0.0,
                          np.where(t > svpt0, 0.0, 1.0))
    qvs_weight = (1.0 - weight) * qvsw + weight * qvsi
    rhum = qv / qvs_weight
    subsat = np.maximum(1.0e-10, rhgrid * qvs_weight - qv)
    arg = np.maximum(-6.9, -alpha0 * qcld / subsat ** gamma)
    fraction = (np.maximum(1.0e-10, rhum) / rhgrid) ** pexp \
        * (1.0 - np.exp(arg))
    fraction = np.where(fraction < 0.01, 0.0, fraction)
    return np.where(qcld < qcldmin, 0.0,
                    np.where(rhum >= rhgrid, 1.0, fraction))


def np_mcica_maxran_masks(play, cldfra, ngpt, permuteseed):
    """Float64/uint32 mirror of WRF RRTMG's McICA maximum-random generator.

    Exact transcription of WRF v4.6.1 ``module_ra_rrtmg_sw.F``:
    kissvec (lines 2008-2040), pmid-fraction seeding advanced by
    ``permuteseed`` draws (lines 1727-1744), the cldmin=1e-20 fraction
    floor (lines 1692, 1717-1725), the icld=2 maximum-random overlap walk
    (lines 1778-1813, WRF Registry default ``cldovrlp=2`` at
    Registry.EM_COMMON:2499), and the subcolumn cloud decision
    ``CDF >= 1 - cldf`` (lines 1941-1944).  One subcolumn per g-point
    (``nsubcsw = ngptsw``, line 1476).  ``play`` is consumed as float32
    exactly like the device kernel so masks are bit-identical.
    """
    play32 = np.ascontiguousarray(np.asarray(play, dtype=np.float32))
    cldf32 = np.ascontiguousarray(np.asarray(cldfra, dtype=np.float32))
    if play32.ndim != 2 or play32.shape[1] < 4:
        raise ValueError("play must have shape (ncol,nlay) with nlay >= 4")
    if cldf32.shape != play32.shape:
        raise ValueError("cldfra must match play's shape")
    pmid = play32.astype(np.float64)
    if np.any(pmid[:, 0] < pmid[:, 1]):
        # module_ra_rrtmg_sw.F:1734-1736 stops unless pmid is supplied
        # bottom-to-top.
        raise ValueError(
            "kissvec seeding requires pmid from the bottom four layers")
    seeds = []
    for n in range(4):
        frac = pmid[:, n] - np.trunc(pmid[:, n])
        seeds.append((frac * 1.0e9).astype(np.int64).astype(np.uint32))
    s1, s2, s3, s4 = seeds

    def kiss():
        nonlocal s1, s2, s3, s4
        s1 = np.uint32(69069) * s1 + np.uint32(1327217885)
        s2 = s2 ^ (s2 << np.uint32(13))
        s2 = s2 ^ (s2 >> np.uint32(17))
        s2 = s2 ^ (s2 << np.uint32(5))
        s3 = np.uint32(18000) * (s3 & np.uint32(65535)) \
            + (s3 >> np.uint32(16))
        s4 = np.uint32(30903) * (s4 & np.uint32(65535)) \
            + (s4 >> np.uint32(16))
        kiss_value = (s1 + s2 + (s3 << np.uint32(16)) + s4).view(np.int32)
        return kiss_value.astype(np.float64) * 2.328306e-10 + 0.5

    for _ in range(int(permuteseed)):
        kiss()
    cldf = cldf32.astype(np.float64)
    cldf = np.where(cldf < 1.0e-20, 0.0, cldf)
    ncol, nlay = cldf.shape
    mask = np.zeros((ncol, nlay, ngpt), dtype=bool)
    for g in range(ngpt):
        draws = [kiss() for _ in range(nlay)]
        cdf = draws[0]
        mask[:, 0, g] = cdf >= 1.0 - cldf[:, 0]
        for k in range(1, nlay):
            keep = cdf > 1.0 - cldf[:, k - 1]
            cdf = np.where(keep, cdf, draws[k] * (1.0 - cldf[:, k - 1]))
            mask[:, k, g] = cdf >= 1.0 - cldf[:, k]
    return mask

# ---------------------------------------------------------------------------
# Kain--Fritsch cumulus: no float64 mirror.  The authority for
# kernels/kf.cu is WRF v4.6.1 itself, through the column oracle in
# tools/kf_wrf461_oracle (tests/test_kf_wrf461_oracle.py, 0 ULP).  The
# mirror that stood here (np_kf_column) encoded the pre-2026-10-07 kernel,
# which re-derived the scheme instead of porting it; it was retired with it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cumulus-driver NCA persistence (Phase 4 Task 6b) -- WRF v4.6.1 DRIVER side:
# module_cu_kfeta.F:406-440 (per-column hold/skip and entry zeroing) plus
# solve_em.F:3558-3571 -> module_physics_addtendc.F:2139-2231 (advance_ppt).
# The KF column scheme itself is graded against WRF by its column oracle;
# this section mirrors only the driver contract that consumes its outputs.
# ---------------------------------------------------------------------------

def np_cumulus_nca_driver_step(held, scheme, *, dt):
    """Advance WRF's per-column KF hold state by one model step (float64).

    ``dt`` is WRF's model-CLOCK step (``cfg.clock_dt`` under the reference case
    compatibility substep integration, ``cfg.dt`` natively); the device
    driver gates this advance to the final internal substep of each clock
    step so every DT here matches WRF's 60 s arithmetic.

    ``held`` carries the persistent driver state: ``nca`` (s), ``pratec``
    (mm s-1), ``raincv`` (mm), ``rainc`` (mm), and the stored uncoupled
    rates ``rthcuten``/``rqvcuten``/``rqccuten``/``rqicuten``/
    ``rqrcuten``/``rqscuten`` shaped
    ``(nz, ny, nx)``.  ``scheme`` is ``None`` on a step where STEPCU is
    not due; on a due step it maps the same rate names plus ``pratec``
    and ``nca_seconds`` to full-grid scheme outputs.  Only columns with
    ``NCA < 0.5*DT`` accept the new values -- WRF skips recomputation for
    every other column (module_cu_kfeta.F:410-412) and zeroes the
    eligible columns' tendencies/RAINCV/PRATEC at entry (414-440), which
    the scheme outputs already encode as zeros for eligible columns that
    do not trigger.  RAINCV = DT*PRATEC (module_cu_kfeta.F:2504-2505).

    Returns ``(new_held, applied)``: ``applied`` maps the rate names to
    the values this step's RK integration uses, while ``new_held``
    reflects ``advance_ppt`` after that RK forcing is captured and is also
    the raw Q*CUTEN state seen by the subsequent microphysics driver
    (module_physics_addtendc.F, KFETASCHEME case): RAINC accumulates
    ``PRATEC*DT`` on every step (line 2141), the stored tendencies are
    zeroed once ``NINT(NCA/DT) <= 1`` with Fortran half-away-from-zero
    rounding (2211-2226), and a positive NCA then decrements by DT
    (2228).  RAINCV/PRATEC survive expiry -- WRF's zeroing at 2216-2217
    is commented out -- so the rain rate keeps accumulating until the
    column's next scheme call replaces it.
    """
    rate_names = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                  "rqrcuten", "rqscuten")
    state = {key: np.array(held[key], dtype=np.float64)
             for key in ("nca", "pratec", "raincv", "rainc", *rate_names)}
    dt = float(dt)
    if scheme is not None:
        eligible = state["nca"] < 0.5 * dt
        for name in rate_names:
            new = np.asarray(scheme[name], dtype=np.float64)
            state[name][:, eligible] = new[:, eligible]
        pratec = np.asarray(scheme["pratec"], dtype=np.float64)
        state["pratec"][eligible] = pratec[eligible]
        state["raincv"][eligible] = dt * pratec[eligible]
        state["nca"][eligible] = np.asarray(
            scheme["nca_seconds"], dtype=np.float64)[eligible]
    applied = {name: state[name].copy() for name in rate_names}
    state["rainc"] += state["pratec"] * dt
    active = state["nca"] > 0.0
    cleared = active & (np.floor(state["nca"] / dt + 0.5) <= 1.0)
    for name in rate_names:
        state[name][:, cleared] = 0.0
    state["nca"][active] -= dt
    return state, applied


# ---------------------------------------------------------------------------
# Radar reflectivity diagnostics (Phase 4 Task 9)
# ---------------------------------------------------------------------------

def _np_refl_maxwellgarnett(vol1, vol2, vol3, m1, m2, m3):
    """Maxwell-Garnett effective refractive index, spheroidal inclusions.

    Mirror of ``m_complex_maxwellgarnett`` (module_mp_radar.F:544-590) on
    the only inclusion string radar_init configures ('spheroidal',
    :106-118, betas :574-576); the volume-closure check is the
    Fortran's 1e-6 guard (:558).
    """
    if abs(vol1 + vol2 + vol3 - 1.0) > 1.0e-6:
        raise ValueError("partial volume fractions must sum to 1")
    m1t = m1 * m1
    m2t = m2 * m2
    m3t = m3 * m3
    beta2 = (2.0 * m1t / (m2t - m1t)
             * (m2t / (m2t - m1t) * np.log(m2t / m1t) - 1.0))
    beta3 = (2.0 * m1t / (m3t - m1t)
             * (m3t / (m3t - m1t) * np.log(m3t / m1t) - 1.0))
    return np.sqrt(((1.0 - vol2 - vol3) * m1t + vol2 * beta2 * m2t
                    + vol3 * beta3 * m3t)
                   / (1.0 - vol2 - vol3 + vol2 * beta2 + vol3 * beta3))


def _np_refl_m_mix_nested(m_a, m_i, m_w, volair, volice, volwater):
    """``get_m_mix_nested`` on radar_init's Morrison string set.

    host='air', matrix='water', inclusion='spheroidal',
    hostmatrix='icewater', hostinclusion='spheroidal'
    (module_mp_radar.F:106-118), which selects exactly two nested
    Maxwell-Garnett evaluations along the host='air' branch (:384-413): the ice-in-water inclusion
    mix, then that mixture as 'ice' inclusions in the air host.
    """
    vol1 = volice / max(volice + volwater, 1.0e-10)     # :391
    vol2 = 1.0 - vol1
    # get_m_mix(..., 0.0, vol1, vol2, 'water', 'spheroidal') resolves to
    # m_complex_maxwellgarnett(volwater, volair, volice, m_w, m_a, m_i)
    # (:517-519).
    mtmp = _np_refl_maxwellgarnett(vol2, 0.0, vol1, m_w, m_a, m_i)
    # hostmatrix='icewater' -> get_m_mix(m_a, mtmp, 2*m_a, volair,
    # 1-volair, 0.0, 'ice', 'spheroidal') (:402-406) resolves to
    # m_complex_maxwellgarnett(volice, volair, volwater, m_i, m_a, m_w)
    # (:514-516) with the mixture as the ice member.
    return _np_refl_maxwellgarnett(1.0 - volair, volair, 0.0,
                                   mtmp, m_a, 2.0 * m_a)


def _np_refl_rayleigh_soak_wetgraupel(x_g, a_geo, b_geo, fmelt,
                                      meltratio_outside, m_w, m_i, rc):
    """Mirror of ``rayleigh_soak_wetgraupel`` (module_mp_radar.F:265-358).

    Backscattering cross section (m2) of one melting snow/graupel
    particle of ice mass ``x_g`` (kg): meltwater fraction ``fmelt`` sits
    90% on the surface (``meltratio_outside``), converging linearly to a
    water drop as fm -> 1; the ice-air-water mixture index comes from the
    nested Maxwell-Garnett path above.  ``rc`` is :func:`radar_init`.
    """
    fm = min(max(fmelt, 0.0), 1.0)                       # :290
    mra = min(max(meltratio_outside, 0.0), 1.0)          # :292
    mra = mra + (1.0 - mra) * fm                         # :299
    x_w = x_g * fm                                       # :301
    d_g = a_geo * x_g ** b_geo                           # :303
    if d_g < 1.0e-12:                                    # :305/:354-356
        return 0.0
    pix = 3.1415926535897932384626434                    # :283
    vg = pix / 6.0 * d_g ** 3                            # :307
    rhog = min(max(x_g / vg, 10.0), 900.0)               # :308
    vg = x_g / rhog                                      # :309
    meltratio_outside_grenz = 1.0 - rhog / 1000.0        # :311
    if mra <= meltratio_outside_grenz:                   # :313-317
        volg = vg * (1.0 - mra * fm)
    else:                                                # :319-331
        fmgrenz = ((900.0 - rhog)
                   / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0))
        if fm <= fmgrenz:
            volg = (1.0 - mra * fm) * vg
        else:
            volg = (x_g - x_w) / 900.0 + x_w / 1000.0
    d_large = (6.0 / pix * volg) ** (1.0 / 3.0)          # :335
    volice = (x_g - x_w) / (volg * 900.0)                # :336
    volwater = x_w / (1000.0 * volg)                     # :337
    volair = 1.0 - volice - volwater                     # :338
    m_core = _np_refl_m_mix_nested(complex(1.0, 0.0), m_i, m_w,
                                   volair, volice, volwater)
    return (abs((m_core ** 2 - 1.0) / (m_core ** 2 + 2.0)) ** 2
            * rc.pi5 * d_large ** 6 / rc.lamda4)         # :351-352


def np_refl10cm_morrison_column(qv, qr, nr, qs, ns, qg, ng, t, p, *,
                                morr_rimed_ice=1):
    """Float64 one-column mirror of Morrison's radar reflectivity.

    ``refl10cm_hm`` (module_mp_morr_two_moment.F:4502-4675) with the
    wrapper's floor ``MAX(-35., dBZ)`` applied (:913-917): Rayleigh
    10 cm equivalent reflectivity (dBZ, shape ``(nz,)``) from rain,
    snow, and graupel using the scheme's own m(D) parameters and
    PROGNOSTIC number concentrations, with the Blahak water-coated
    melting treatment below the highest melting level.  Inputs are
    ``(nz,)`` bottom-to-top columns: mixing ratios kg/kg, numbers kg-1,
    air temperature K, full pressure Pa.  Setup-time constants come from
    :func:`woof.core.refl.radar_init` (WRF computes them once at
    scheme init; the per-column math here is independent of the kernel).
    ``morr_rimed_ice`` selects the same hail/graupel CG used by the scheme.
    """
    from woof.core.refl import (MELT_OUTSIDE_G, MELT_OUTSIDE_S, NRBINS,
                                 radar_init)

    rc = radar_init(morr_rimed_ice)
    names = ("qv", "qr", "nr", "qs", "ns", "qg", "ng", "t", "p")
    a = {name: np.asarray(value, np.float64).copy()
         for name, value in zip(names, (qv, qr, nr, qs, ns, qg, ng, t, p))}
    nz = a["t"].size
    if any(value.shape != (nz,) for value in a.values()):
        raise ValueError("all reflectivity column inputs must be (nz,)")

    # WRF gates activity only on mass before using the number moment in a
    # fractional-power slope (:4544-4584).  Its production scheme supplies
    # positive bounded moments (:1528-1635); explicitly retain non-finite
    # output for invalid active pairs so a corrupt input cannot masquerade
    # as the wrapper's meteorological -35 dBZ clear-air floor.
    invalid_moment = np.zeros(nz, bool)
    for mass, number in (("qr", "nr"), ("qs", "ns"), ("qg", "ng")):
        invalid_moment |= ((a[mass] > 1.0e-9)
                           & ((a[number] <= 0.0)
                              | ~np.isfinite(a[number])))

    temp = a["t"]
    qvc = np.maximum(1.0e-10, a["qv"])                   # :4540
    pres = a["p"]
    # :4542 -- the routine's own moist density (literal 0.622, not EP2).
    rho = 0.622 * pres / (287.0 * temp * (qvc + 0.622))

    rr = np.full(nz, 1.0e-12)
    rs = np.full(nz, 1.0e-12)
    rg = np.full(nz, 1.0e-12)
    ilamr = np.zeros(nz)
    ilams = np.zeros(nz)
    ilamg = np.zeros(nz)
    n0_r = np.zeros(nz)
    n0_s = np.zeros(nz)
    n0_g = np.zeros(nz)
    l_qr = np.zeros(nz, bool)
    l_qs = np.zeros(nz, bool)
    l_qg = np.zeros(nz, bool)
    for k in range(nz):
        if a["qr"][k] > 1.0e-9:                          # :4544-4556
            rr[k] = a["qr"][k] * rho[k]
            nrk = a["nr"][k] * rho[k]
            lamr = (rc.xam_r * rc.xcrg[2] * rc.xorg2
                    * nrk / rr[k]) ** rc.xobmr
            ilamr[k] = 1.0 / lamr
            n0_r[k] = nrk * rc.xorg2 * lamr ** rc.xcre[1]
            l_qr[k] = True
        if a["qs"][k] > 1.0e-9:                          # :4558-4570
            rs[k] = a["qs"][k] * rho[k]
            nsk = a["ns"][k] * rho[k]
            lams = (rc.xam_s * rc.xcsg[2] * rc.xosg2
                    * nsk / rs[k]) ** rc.xobms
            ilams[k] = 1.0 / lams
            n0_s[k] = nsk * rc.xosg2 * lams ** rc.xcse[1]
            l_qs[k] = True
        if a["qg"][k] > 1.0e-9:                          # :4572-4584
            rg[k] = a["qg"][k] * rho[k]
            ngk = a["ng"][k] * rho[k]
            lamg = (rc.xam_g * rc.xcgg[2] * rc.xogg2
                    * ngk / rg[k]) ** rc.xobmg
            ilamg[k] = 1.0 / lamg
            n0_g[k] = ngk * rc.xogg2 * lamg ** rc.xcge[1]
            l_qg[k] = True

    # :4586-4599 -- highest warm rainy level with frozen species above;
    # k_0 is the level just above it.
    melti = False
    k_0 = 0
    for k in range(nz - 2, -1, -1):
        if temp[k] > 273.15 and l_qr[k] and (l_qs[k + 1] or l_qg[k + 1]):
            k_0 = max(k + 1, k_0)
            melti = True
            break

    # :4601-4620 -- Rayleigh integrals; dry snow/graupel carry the
    # (0.176/0.93)*(6/pi)^2*(am/900)^2 density/dielectric adjustment.
    pi_m = 3.1415926535897932384626434
    ze_rain = np.full(nz, 1.0e-22)
    ze_snow = np.full(nz, 1.0e-22)
    ze_graupel = np.full(nz, 1.0e-22)
    for k in range(nz):
        if l_qr[k]:
            ze_rain[k] = n0_r[k] * rc.xcrg[3] * ilamr[k] ** rc.xcre[3]
        if l_qs[k]:
            ze_snow[k] = ((0.176 / 0.93) * (6.0 / pi_m) * (6.0 / pi_m)
                          * (rc.xam_s / 900.0) * (rc.xam_s / 900.0)
                          * n0_s[k] * rc.xcsg[3] * ilams[k] ** rc.xcse[3])
        if l_qg[k]:
            ze_graupel[k] = ((0.176 / 0.93) * (6.0 / pi_m) * (6.0 / pi_m)
                             * (rc.xam_g / 900.0) * (rc.xam_g / 900.0)
                             * n0_g[k] * rc.xcgg[3] * ilamg[k] ** rc.xcge[3])

    # :4622-4671 -- melting snow/graupel: 50-bin Simpson integration of
    # the soaked-particle backscatter, meltwater fraction from the mass
    # ratio against the k_0 reference level.
    if melti and k_0 >= 1:
        for k in range(k_0 - 1, -1, -1):
            if l_qs[k] and l_qs[k_0]:
                fmelt_s = max(0.005, min(1.0 - rs[k] / rs[k_0], 0.99))
                eta = 0.0
                lams = 1.0 / ilams[k]
                for n in range(NRBINS):
                    x = rc.xam_s * rc.xxds[n] ** 3.0     # xbm_s
                    cback = _np_refl_rayleigh_soak_wetgraupel(
                        x, rc.xocms, rc.xobms, fmelt_s, MELT_OUTSIDE_S,
                        rc.m_w_0, rc.m_i_0, rc)
                    f_d = n0_s[k] * np.exp(-lams * rc.xxds[n])  # xmu_s = 0
                    eta += f_d * cback * rc.simpson[n] * rc.xdts[n]
                ze_snow[k] = rc.lamda4 / (rc.pi5 * rc.k_w) * eta
            if l_qg[k] and l_qg[k_0]:
                fmelt_g = max(0.005, min(1.0 - rg[k] / rg[k_0], 0.99))
                eta = 0.0
                lamg = 1.0 / ilamg[k]
                for n in range(NRBINS):
                    x = rc.xam_g * rc.xxdg[n] ** 3.0     # xbm_g
                    cback = _np_refl_rayleigh_soak_wetgraupel(
                        x, rc.xocmg, rc.xobmg, fmelt_g, MELT_OUTSIDE_G,
                        rc.m_w_0, rc.m_i_0, rc)
                    f_d = n0_g[k] * np.exp(-lamg * rc.xxdg[n])  # xmu_g = 0
                    eta += f_d * cback * rc.simpson[n] * rc.xdtg[n]
                ze_graupel[k] = rc.lamda4 / (rc.pi5 * rc.k_w) * eta

    dbz = 10.0 * np.log10((ze_rain + ze_snow + ze_graupel)
                          * 1.0e18)                      # :4670-4672
    out = np.maximum(-35.0, dbz)                         # wrapper :916
    out[invalid_moment] = np.nan
    return out


def np_refl10cm_wsm6_column(qv, qr, qs, qg, t, p, *, hail_opt=0,
                            melting=True):
    """Float64 one-column mirror of WRF v4.6.1 ``refl10cm_wsm6``.

    WSM6 diagnoses fixed-intercept exponential rain/snow/rimed-ice PSDs,
    then applies the same 50-bin Blahak melting-particle calculation used by
    Morrison.  Inputs are bottom-to-top ``(nz,)`` arrays.  ``hail_opt``
    selects WRF's graupel (0) or hail (1) density and intercept.
    ``melting=False`` omits the melting-particle term (rain plus dry snow
    and dry graupel only), the float64 mirror of
    ``launch_refl10cm_wsm6(..., melting=False)``.
    """
    from woof.core.refl import (MELT_OUTSIDE_G, MELT_OUTSIDE_S, NRBINS,
                                 radar_init_wsm6)
    from woof.core.wsm6_constants import rimed_ice_constants

    rc = radar_init_wsm6(hail_opt)
    rimed = rimed_ice_constants(hail_opt)
    names = ("qv", "qr", "qs", "qg", "t", "p")
    a = {name: np.asarray(value, np.float64).copy()
         for name, value in zip(names, (qv, qr, qs, qg, t, p))}
    nz = a["t"].size
    if any(value.shape != (nz,) for value in a.values()):
        raise ValueError("all reflectivity column inputs must be (nz,)")

    temp = a["t"]
    qvc = np.maximum(1.0e-10, a["qv"])
    rho = 0.622 * a["p"] / (287.0 * temp * (qvc + 0.622))
    rr = np.full(nz, 1.0e-12)
    rs = np.full(nz, 1.0e-12)
    rg = np.full(nz, 1.0e-12)
    ilamr = np.zeros(nz)
    ilams = np.zeros(nz)
    ilamg = np.zeros(nz)
    # WSM6's snow intercept increases exponentially below freezing.
    temp_c = np.minimum(-0.001, temp - 273.15)
    n0_s = np.minimum(1.0e11, 2.0e6 * np.exp(-0.12 * temp_c))
    n0_r = 8.0e6
    n0_g = float(rimed.n0g)
    l_qr = a["qr"] > 1.0e-9
    l_qs = a["qs"] > 1.0e-9
    l_qg = a["qg"] > 1.0e-9
    for k in range(nz):
        if l_qr[k]:
            rr[k] = a["qr"][k] * rho[k]
            # Fixed-intercept WSM6 solves the third mass moment, hence
            # lambda exponent 1/(1+xbm)=1/4 (not Morrison's 1/xbm).
            lamr = (rc.xam_r * rc.xcrg[2] * n0_r / rr[k]) ** 0.25
            ilamr[k] = 1.0 / lamr
        if l_qs[k]:
            rs[k] = a["qs"][k] * rho[k]
            lams = (rc.xam_s * rc.xcsg[2] * n0_s[k] / rs[k]) ** 0.25
            ilams[k] = 1.0 / lams
        if l_qg[k]:
            rg[k] = a["qg"][k] * rho[k]
            lamg = (rc.xam_g * rc.xcgg[2] * n0_g / rg[k]) ** 0.25
            ilamg[k] = 1.0 / lamg

    melti = False
    k_0 = 0
    for k in range(nz - 2, -1, -1):
        if temp[k] > 273.15 and l_qr[k] and (l_qs[k + 1] or l_qg[k + 1]):
            k_0 = max(k + 1, k_0)
            melti = True
            break

    pi_m = 3.1415926535897932384626434
    ze_rain = np.full(nz, 1.0e-22)
    ze_snow = np.full(nz, 1.0e-22)
    ze_graupel = np.full(nz, 1.0e-22)
    for k in range(nz):
        if l_qr[k]:
            ze_rain[k] = n0_r * rc.xcrg[3] * ilamr[k] ** rc.xcre[3]
        if l_qs[k]:
            ze_snow[k] = ((0.176 / 0.93) * (6.0 / pi_m) ** 2
                          * (rc.xam_s / 900.0) ** 2 * n0_s[k]
                          * rc.xcsg[3] * ilams[k] ** rc.xcse[3])
        if l_qg[k]:
            ze_graupel[k] = ((0.176 / 0.93) * (6.0 / pi_m) ** 2
                             * (rc.xam_g / 900.0) ** 2 * n0_g
                             * rc.xcgg[3] * ilamg[k] ** rc.xcge[3])

    if melting and melti and k_0 >= 1:
        for k in range(k_0 - 1, -1, -1):
            if l_qs[k] and l_qs[k_0]:
                fmelt = max(0.005, min(1.0 - rs[k] / rs[k_0], 0.99))
                eta = 0.0
                lams = 1.0 / ilams[k]
                for n in range(NRBINS):
                    x = rc.xam_s * rc.xxds[n] ** 3.0
                    cback = _np_refl_rayleigh_soak_wetgraupel(
                        x, rc.xocms, rc.xobms, fmelt, MELT_OUTSIDE_S,
                        rc.m_w_0, rc.m_i_0, rc)
                    eta += (n0_s[k] * np.exp(-lams * rc.xxds[n]) * cback
                            * rc.simpson[n] * rc.xdts[n])
                ze_snow[k] = rc.lamda4 / (rc.pi5 * rc.k_w) * eta
            if l_qg[k] and l_qg[k_0]:
                fmelt = max(0.005, min(1.0 - rg[k] / rg[k_0], 0.99))
                eta = 0.0
                lamg = 1.0 / ilamg[k]
                for n in range(NRBINS):
                    x = rc.xam_g * rc.xxdg[n] ** 3.0
                    cback = _np_refl_rayleigh_soak_wetgraupel(
                        x, rc.xocmg, rc.xobmg, fmelt, MELT_OUTSIDE_G,
                        rc.m_w_0, rc.m_i_0, rc)
                    eta += (n0_g * np.exp(-lamg * rc.xxdg[n]) * cback
                            * rc.simpson[n] * rc.xdtg[n])
                ze_graupel[k] = rc.lamda4 / (rc.pi5 * rc.k_w) * eta

    dbz = 10.0 * np.log10((ze_rain + ze_snow + ze_graupel) * 1.0e18)
    return np.maximum(-35.0, dbz)


def np_refl10cm_kessler_column(qv, qr, t, p):
    """Float64 mirror of the Kessler rain-only reflectivity fallback.

    WRF's Kessler has no refl_10cm diagnostic; this is the Smith et al.
    (1975) rain form for an exponential PSD with the fixed Marshall-Palmer
    intercept N0r = 8e6 m-4 and rho_w = 1000 kg m-3 that Kessler's own
    fall-speed closure assumes (derivation in woof/core/refl.py):
    Ze = Gamma(7)*N0r*lambda**-7 with lambda = (pi*rho_w*N0r/(rho*qr))**0.25.
    Density diagnosis, the 1e-9 kg/kg threshold, the 1e-22 ze floor, and
    the -35 dBZ output floor follow ``refl10cm_hm`` so both schemes'
    outputs share one convention.
    """
    names = ("qv", "qr", "t", "p")
    a = {name: np.asarray(value, np.float64).copy()
         for name, value in zip(names, (qv, qr, t, p))}
    nz = a["t"].size
    if any(value.shape != (nz,) for value in a.values()):
        raise ValueError("all reflectivity column inputs must be (nz,)")
    qvc = np.maximum(1.0e-10, a["qv"])
    rho = 0.622 * a["p"] / (287.0 * a["t"] * (qvc + 0.622))
    pi_m = 3.1415926535897932384626434
    n0r = 8.0e6
    rhow = 1000.0
    ze = np.full(nz, 1.0e-22)
    active = a["qr"] > 1.0e-9
    lam = (pi_m * rhow * n0r / (rho[active] * a["qr"][active])) ** 0.25
    ze[active] = math.gamma(7.0) * n0r * lam ** -7.0
    return np.maximum(-35.0, 10.0 * np.log10(ze * 1.0e18))


# ===========================================================================
# Phase 5 Task 10 (panel lane L3): parent->child nest interpolation mirrors.
#
# FP64 transliterations of WRF v4.6.1 share/sint.F, share/interp_fcn.F and
# dyn_em/nest_init_utils.F.  Per the architecture doc (section D, F6/F7
# amendments) every mirror consumes the SAME FP32-rounded geometry tables
# as the CUDA kernels (``NestRegistration.xig/xjg`` are FP64-built and
# FP32-stored; the FP32 values are widened here) so the oracle floor stays
# discriminating.  The horizontal operator authority is share/sint.F itself
# -- SINT (sint.F:2-198) / SINTB (:203-347), the Smolarkiewicz positive-
# definite monotonic transport with field-dependent DONOR/TR4 statement
# functions and nonlinear limiters -- interp_fcn_sint (interp_fcn.F:874-993)
# and bdy_interp1 (:2423-2626) are only tile/stagger wrappers around it.
# ===========================================================================

#: DATA EP/ 1.E-10/ (sint.F:26): a REAL constant; the mirror widens the
#: FP32 value so kernel and mirror share one limiter epsilon.
_SINT_EP = float(np.float32(1.0e-10))


def np_couple_nest_field(state, field_name, *, dtype=np.float32):
    """Independent WRF REAL transcription of one full-field coupling pass.

    The expression order comes directly from
    ``couple_or_uncouple_em.F:122-144,270-345``: the base and perturbation
    mass terms remain separated as ``(c1*Mub+c2) + (c1*Mu_2)`` and u/v retain
    the literal four-term face order.  The u/v/w weights are divided by their
    map factors before the separately rounded field multiplication, matching
    WRF's stored REAL weight arrays.  It intentionally does not share a helper
    or algebraic formulation with ``lbc_state.cu``, so parity tests can catch
    kernel reassociation.  ``dtype=np.float32`` emulates each WRF REAL
    store/operation boundary; FP64 remains a diagnostic oracle.
    """
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be np.float32 or np.float64")

    def a(name):
        value = getattr(state, name)
        if hasattr(value, "get"):
            value = value.get()
        return np.asarray(value, dtype=dtype)

    def add(left, right):
        return np.asarray(np.asarray(left, dtype=dtype)
                          + np.asarray(right, dtype=dtype), dtype=dtype)

    def mul(left, right):
        return np.asarray(np.asarray(left, dtype=dtype)
                          * np.asarray(right, dtype=dtype), dtype=dtype)

    def div(left, right):
        return np.asarray(np.asarray(left, dtype=dtype)
                          / np.asarray(right, dtype=dtype), dtype=dtype)

    def face_weight(base, perturbation, axis):
        if axis == -1:
            out = np.empty((*base.shape[:-1], base.shape[-1] + 1),
                           dtype=dtype)
            # WRF's generic loop reaches the low physical face.  The mass
            # halo duplicates the adjacent interior cell, but its literal
            # four-term REAL grouping is retained.  Only the high physical
            # face receives WRF's explicit one-sided repair.
            low = add(base[..., 0], base[..., 0])
            low = add(low, perturbation[..., 0])
            low = add(low, perturbation[..., 0])
            out[..., 0] = mul(dtype(0.5), low)
            out[..., -1] = add(base[..., -1], perturbation[..., -1])
            total = add(base[..., 1:], base[..., :-1])
            total = add(total, perturbation[..., 1:])
            total = add(total, perturbation[..., :-1])
            out[..., 1:-1] = mul(dtype(0.5), total)
            return out
        out = np.empty((base.shape[0], base.shape[1] + 1, base.shape[2]),
                       dtype=dtype)
        low = add(base[:, 0], base[:, 0])
        low = add(low, perturbation[:, 0])
        low = add(low, perturbation[:, 0])
        out[:, 0] = mul(dtype(0.5), low)
        out[:, -1] = add(base[:, -1], perturbation[:, -1])
        total = add(base[:, 1:], base[:, :-1])
        total = add(total, perturbation[:, 1:])
        total = add(total, perturbation[:, :-1])
        out[:, 1:-1] = mul(dtype(0.5), total)
        return out

    mup = a("mup")
    mub = a("mub2d")
    if field_name == "mu":
        return np.asarray(mup[None], dtype=dtype).copy()
    c1h, c2h = a("c1h")[:, None, None], a("c2h")[:, None, None]
    c1f, c2f = a("c1f")[:, None, None], a("c2f")[:, None, None]
    has_msf = bool(getattr(state, "has_msf", False))

    base_h = add(mul(c1h, mub[None]), c2h)
    perturbation_h = mul(c1h, mup[None])

    if field_name == "u":
        weight = face_weight(base_h, perturbation_h, -1)
        if has_msf:
            weight = div(weight, a("msfu")[None])
        return mul(weight, a("u"))
    if field_name == "v":
        weight = face_weight(base_h, perturbation_h, -2)
        if has_msf:
            weight = div(weight, a("msfv")[None])
        return mul(weight, a("v"))

    base_f = add(mul(c1f, mub[None]), c2f)
    perturbation_f = mul(c1f, mup[None])
    weight_f = add(base_f, perturbation_f)
    if field_name == "w":
        if has_msf:
            weight_f = div(weight_f, a("msft")[None])
        return mul(weight_f, a("w"))
    if field_name == "ph":
        return mul(weight_f, a("php"))

    weight_h = add(base_h, perturbation_h)
    if field_name == "t":
        thb = a("thb")
        if thb.ndim == 1:
            thb = thb[:, None, None]
        value = add(thb, a("thp"))
        value = add(value, dtype(-300.0))
    else:
        value = a(field_name)
    return mul(weight_h, value)


def np_w_relaxation_current(state, *, source_field=None, dtype=np.float32):
    """WRF nested Davies current-side ``mass_weight(w, mut, c1f, c2f)``.

    Authority: ``module_bc_em.F:327-333,1716-1745``.  Unlike force-time
    donor coupling, this comparison field has no ``msfty`` division.
    """
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be np.float32 or np.float64")

    def host(value):
        if hasattr(value, "get"):
            value = value.get()
        return np.asarray(value, dtype=dtype)

    mub = host(state.mub2d)
    mup = host(state.mup)
    mut = np.asarray(mub + mup, dtype=dtype)
    c1f = host(state.c1f)[:, None, None]
    c2f = host(state.c2f)[:, None, None]
    weight = np.asarray(
        np.asarray(c1f * mut[None], dtype=dtype) + c2f, dtype=dtype)
    w = host(state.w if source_field is None else source_field)
    return np.asarray(weight * w, dtype=dtype)


def np_nest_force(parent_state, child_state, registrations, *,
                  field_names, parent_dt_fp32, parent_interval_ticks=None,
                  spec_zone=1, relax_zone=4, spec_bdy_width=5,
                  dtype=np.float32):
    """REAL-emulation mirror of the complete per-field force table build."""
    tables = {}
    for name in field_names:
        stagger = "x" if name == "u" else "y" if name == "v" else "m"
        tables[name] = np_bdy_interp1(
            np_couple_nest_field(parent_state, name, dtype=dtype),
            np_couple_nest_field(child_state, name, dtype=dtype),
            registrations[stagger], parent_dt_fp32=parent_dt_fp32,
            parent_interval_ticks=parent_interval_ticks,
            spec_zone=spec_zone, relax_zone=relax_zone,
            spec_bdy_width=spec_bdy_width, dtype=dtype)
    return tables


def _sint_donor(y1, y2, a):
    """DONOR statement function, sint.F:37 (upstream donor-cell flux)::

        DONOR(Y1,Y2,A)=(Y1*AMAX1(0.,SIGN(1.,A))-Y2*AMIN1(0.,SIGN(1.,A)))*A

    ``SIGN(1.,A)`` transfers A's sign bit onto 1.0 (signed zero included),
    which ``np.copysign`` reproduces exactly.
    """
    s = np.copysign(1.0, a)
    return (y1 * np.maximum(0.0, s) - y2 * np.minimum(0.0, s)) * a


def _sint_tr4(ym1, y0, yp1, yp2, a):
    """TR4 statement function, sint.F:39-41 (4th-order transport flux)::

        TR4(YM1,Y0,YP1,YP2,A)=A*ONE12*(7.*(YP1+Y0)-(YP2+YM1))
         -A*A*ONE24*(15.*(YP1-Y0)-(YP2-YM1))-A*A*A*ONE12*((YP1+Y0)
         -(YP2+YM1))+A*A*A*A*ONE24*(3.*(YP1-Y0)-(YP2-YM1))

    with ONE12 = 1/12 and ONE24 = 1/24 (sint.F:14).
    """
    one12 = 1.0 / 12.0
    one24 = 1.0 / 24.0
    return (a * one12 * (7.0 * (yp1 + y0) - (yp2 + ym1))
            - a * a * one24 * (15.0 * (yp1 - y0) - (yp2 - ym1))
            - a * a * a * one12 * ((yp1 + y0) - (yp2 + ym1))
            + a * a * a * a * one24 * (3.0 * (yp1 - y0) - (yp2 - ym1)))


def _sint_pass(ym2, ym1, y0, yp1, yp2, a):
    """One 1-D residual-advection pass of SINTB (sint.F:286-310).

    The nonlinear core: donor fluxes FL, low-order update W, 4th-order
    fluxes TR4, antidiffusive fluxes F-FL, and the overshoot/undershoot
    (OV/UN) min/max limiter.  PP(X)=AMAX1(0.,X), PN(X)=AMIN1(0.,X)
    (sint.F:43-44).  Evaluated per field at force time -- NEVER a fixed
    precomputed weighted stencil (F6 amendment): F0/F1 and the OV/UN
    clip factors depend on the field values themselves.
    """
    fl0 = _sint_donor(ym1, y0, a)                        # sint.F:286
    fl1 = _sint_donor(y0, yp1, a)                        # sint.F:287
    w = y0 - (fl1 - fl0)                                 # sint.F:288
    mxm = np.maximum(np.maximum(ym1, y0), np.maximum(yp1, w))   # :289-291
    mn = np.minimum(np.minimum(ym1, y0), np.minimum(yp1, w))    # :292
    f0 = _sint_tr4(ym2, ym1, y0, yp1, a)                 # sint.F:293-295
    f1 = _sint_tr4(ym1, y0, yp1, yp2, a)                 # sint.F:296-298
    f0 = f0 - fl0                                        # sint.F:299
    f1 = f1 - fl1                                        # sint.F:300
    pp0, pn0 = np.maximum(0.0, f0), np.minimum(0.0, f0)
    pp1, pn1 = np.maximum(0.0, f1), np.minimum(0.0, f1)
    ov = (mxm - w) / (-pn1 + pp0 + _SINT_EP)             # sint.F:301-302
    un = (w - mn) / (pp1 - pn0 + _SINT_EP)               # sint.F:303-304
    c0 = pp0 * np.minimum(1.0, ov) + pn0 * np.minimum(1.0, un)  # :305-306
    c1 = pp1 * np.minimum(1.0, un) + pn1 * np.minimum(1.0, ov)  # :307-308
    return w - (c1 - c0)                                 # sint.F:309


def np_sint(cfld, reg, *, dtype=np.float64):
    """Float64 mirror of the ``nest_sint`` kernel (share/sint.F SINT/SINTB).

    Interpolates a parent field onto every child point of ``reg`` (a
    ``woof.core.nest_interp.NestRegistration`` or any object with the
    same ``ci/ip/cj/jp/xig/xjg`` tables).  Geometry semantics
    (interp_fcn.F:971 calling sint over replicated parent planes, child
    pickup ``psca(ci, cj, ip+1+jp*nri)`` at :985):

    - donor cell   ci = ipos + (ni1-1)/nri, ip = mod(ni1-1, nri) with
      ni1 = ni + ioff (interp_fcn.F:976-982 / :2563-2569);
    - offsets      a = XIG(ip+1 + jp*nri), b = XJG(...) (sint.F:54-59),
      consumed here as the FP32-rounded ``reg.xig``/``reg.xjg``;
    - x-pass over the five j-rows J=-2..2, then one y-pass (SINTB
      structure, sint.F:274-341).

    ``cfld`` is ``(ny_p, nx_p)`` or ``(nz, ny_p, nx_p)``; the result has
    the child extent of ``reg`` with the same leading shape.

    PRECISION (shadow-review adjudication, fix round): ``dtype`` selects
    the evaluation precision of the SAME transliteration.

    - ``np.float64`` (default): the ledger's FP64 oracle (architecture
      section D F6; nest_gates N1 ``sint_vs_fp64_mirror``).  Sound for
      the fp32_floor comparator only on fixtures whose outputs stay
      bounded away from sign-changing cancellation: where the
      interpolated value crosses zero, the FP32 kernel pipeline and any
      FP64 evaluation legitimately diverge by orders of magnitude in
      output ULPs (the pre-cancellation terms are O(field) with
      O(eps32*field) representation spread that survives subtraction).
    - ``np.float32``: WRF REAL statement emulation -- per-op
      round-to-nearest FP32, exactly the Fortran REAL arithmetic
      (sint.F computes everything in REAL) and the ``-fmad=false``
      kernel.  This is the comparator's mirror at cancellation-sensitive
      points; one shared formula path, so the two precisions cannot
      drift apart.
    """
    if dtype not in (np.float64, np.float32):
        raise ValueError("dtype must be np.float64 (FP64 oracle) or "
                         "np.float32 (WRF REAL statement emulation)")
    c2 = np.asarray(cfld, dtype=dtype)
    squeeze = c2.ndim == 2
    if squeeze:
        c2 = c2[None]
    nz, nyp, nxp = c2.shape
    ci = np.asarray(reg.ci, dtype=np.int64)
    cj = np.asarray(reg.cj, dtype=np.int64)
    if ci.min() < 2 or ci.max() > nxp - 3 or cj.min() < 2 or cj.max() > nyp - 3:
        raise ValueError("parent field too small for the +-2 SINT stencil "
                         "around the registered donor cells")
    # FP32-rounded offset coefficients (F6: mirror consumes the same FP32
    # geometry as the device tables), widened only in FP64 mode.
    ax = np.asarray(reg.xig, dtype=dtype)[
        np.asarray(reg.ip, dtype=np.int64)][None, None, :]
    ay = np.asarray(reg.xjg, dtype=dtype)[
        np.asarray(reg.jp, dtype=np.int64)][None, :, None]
    ii = ci[None, :]
    jj = cj[:, None]
    z = []
    for joff in range(-2, 3):        # DO 50 J=-IOR,IOR (sint.F:276)
        row = [c2[:, jj + joff, ii + ioff] for ioff in range(-2, 3)]
        z.append(_sint_pass(*row, ax))
    out = _sint_pass(*z, ay)         # y-pass, sint.F:318-341
    return out[0] if squeeze else out


def np_bdy_width(spec_zone, relax_zone, spec_bdy_width):
    """LBC strip width, interp_fcn.F:2517 (verified; = 5 for the bundle)::

        sz = MIN(MAX( spec_zone, relax_zone + 1 ),spec_bdy_width)
    """
    return min(max(int(spec_zone), int(relax_zone) + 1), int(spec_bdy_width))


def np_bdy_interp1(cfld, nfld, reg, *, parent_dt_fp32,
                   parent_interval_ticks=None,
                   spec_zone=1, relax_zone=4, spec_bdy_width=5,
                   dtype=np.float64):
    """Float64 mirror of ``nest_bdy_interp1`` (interp_fcn.F:2423-2626).

    Builds the child's four-side boundary VALUE and TENDENCY tables from
    the coupled parent field at t+dtp (``cfld``) and the child's current
    coupled field (``nfld``):

    - VALUE  = the child's current state (bdy_xs = nfld, :2584);
    - TENDENCY = rdt * (SINT(parent) - nfld) (:2583) where ``rdt`` is
      REAL*8 ``1.D0/cdt`` (:2480, :2500) and **cdt IS THE PARENT/COARSE
      STEP** -- interp_fcn.F:2320 declares ``cdt, ndt`` as "Time step
      size for CG and FG" (dummy decls :2345, :2472).  The F7 amendment
      pins this: the divisor is the parent interval, NEVER the child dt
      (the child dt governs only the Davies application).  The API
      carries ``parent_dt_fp32`` / ``parent_interval_ticks`` explicitly.

    Precision contract (REAL*8-rdt scheme, mirrored EXACTLY -- shadow
    fix round): WRF forms the difference ``psca - nfld`` in REAL (both
    operands are REAL arrays, :2473-2486), promotes the FP32 difference
    to FP64 through the REAL*8 multiply, and stores REAL.  The mirror
    therefore rounds its SINT result and the child state to FP32, forms
    the DIFFERENCE IN FP32 (one REAL rounding, exactly :2583), widens
    it, and multiplies by the FP64 ``1/cdt`` reciprocal; only the final
    FP64->FP32 store rounding is left to the fp32_floor comparator.  A
    previous revision differed here (FP64 difference) -- pinned by
    ``test_bdy_mirror_forms_real_difference``.  ``dtype`` selects the
    SINT evaluation precision exactly as :func:`np_sint` (FP64 oracle
    default; FP32 = WRF REAL emulation for cancellation-sensitive
    fixtures where ``psca`` approaches ``nfld``).

    Table layout follows woof's Phase-4 convention (lateral_bc.py
    ``_field_boundary``): west/east ``(nz, ny_c, sz)``, south/north
    ``(nz, sz, nx_c)``, width index 0 at the domain edge (east/north
    reversed -- WRF's ``bdy_xe(nj,k,nide-ni[+1])`` indexing, :2596-2616).
    Returns ``{"west"|"east"|"south"|"north": (value, tendency)}``,
    float64 arrays carrying FP32-quantized values (WRF's REAL tables).
    """
    if getattr(reg, "wrapper", None) != "bdy":
        raise ValueError("np_bdy_interp1 needs a wrapper='bdy' registration "
                         "(stagger ioff = MAX((nri-1)/2,1), "
                         "interp_fcn.F:2504-2510); an 'interp' registration "
                         "mirrors the wrong stagger geometry")
    if parent_interval_ticks is not None and int(parent_interval_ticks) <= 0:
        raise ValueError("parent_interval_ticks must be positive")
    cdt = np.float32(parent_dt_fp32)
    if not np.isfinite(cdt) or cdt <= 0.0:
        raise ValueError("parent_dt_fp32 must be a positive finite step")
    rdt = np.float64(1.0) / np.float64(cdt)              # :2500, REAL*8
    sz = np_bdy_width(spec_zone, relax_zone, spec_bdy_width)   # :2517
    psca = np.asarray(np_sint(cfld, reg, dtype=dtype))
    if psca.ndim == 2:
        psca = psca[None]
    n2 = np.asarray(nfld, dtype=np.float64)
    if n2.ndim == 2:
        n2 = n2[None]
    if n2.shape != psca.shape:
        raise ValueError("child field shape does not match the registration")
    # The :2583 REAL difference: one FP32 rounding, then the REAL*8
    # promote-and-multiply; the value table is WRF's REAL nfld (:2584).
    diff32 = np.float32(psca) - np.float32(n2)
    tend = rdt * diff32.astype(np.float64)
    val = np.float32(n2).astype(np.float64)
    return {
        # WEST: ni = nids..nids+sz-1, bdy_xs(nj,k,ni) (:2582-2585)
        "west": (val[:, :, :sz].copy(), tend[:, :, :sz].copy()),
        # EAST: width index 0 at the domain edge (:2593-2604)
        "east": (val[:, :, -1:-sz - 1:-1].copy(),
                 tend[:, :, -1:-sz - 1:-1].copy()),
        # SOUTH: nj = njds..njds+sz-1, bdy_ys(ni,k,nj) (:2587-2591)
        "south": (val[:, :sz, :].copy(), tend[:, :sz, :].copy()),
        # NORTH: width index 0 at the domain edge (:2606-2617)
        "north": (val[:, -1:-sz - 1:-1, :].copy(),
                  tend[:, -1:-sz - 1:-1, :].copy()),
    }


def np_blend_terrain(ter_interpolated, ter_input, *,
                     spec_bdy_width=5, blend_width=5):
    """Float64 mirror of ``nest_blend_terrain`` (nest_init_utils.F:712-785).

    Rows/columns <= spec_bdy_width take the parent-interpolated value
    verbatim (:766-769); the next ``blend_width`` frames blend linearly
    with weights ``blend_cell/(blend_width+1)`` on the fine field and
    ``(blend_width+1-blend_cell)/(blend_width+1)`` on the parent field
    (:759-765, r_blend_zones = 1./(blend_width+1) at :755); the interior
    is pure fine.  The descending ``DO blend_cell = blend_width,1,-1``
    overwrite order makes the SMALLEST matching frame win at corners --
    transliterated, not closed-formed.  1-based WRF ``i``/``ide`` map to
    0-based ``i0 = i-1`` with ``ide = nx+1`` (mass points 1..ide-1).

    Accepts 2-D ``(ny, nx)`` (ht, mub) or 3-D ``(nk, ny, nx)`` (phb, k
    inert); returns the blended array.
    """
    coarse = np.asarray(ter_interpolated, dtype=np.float64)
    fine = np.asarray(ter_input, dtype=np.float64)
    if coarse.shape != fine.shape:
        raise ValueError("blend operands must have identical shapes")
    squeeze = fine.ndim == 2
    work_f = fine[None] if squeeze else fine
    work_c = coarse[None] if squeeze else coarse
    nk, ny, nx = work_f.shape
    sbw, bw = int(spec_bdy_width), int(blend_width)
    ide, jde = nx + 1, ny + 1
    i1 = np.arange(1, nx + 1)[None, None, :]    # WRF 1-based i
    j1 = np.arange(1, ny + 1)[None, :, None]    # WRF 1-based j
    r_blend_zones = 1.0 / (bw + 1)                       # :755
    out = work_f.copy()                                  # :742-748
    for blend_cell in range(bw, 0, -1):                  # :759
        hit = ((i1 == sbw + blend_cell) | (j1 == sbw + blend_cell)
               | (i1 == ide - sbw - blend_cell)
               | (j1 == jde - sbw - blend_cell))         # :760-761
        blended = ((blend_cell * work_f
                    + (bw + 1 - blend_cell) * work_c)
                   * r_blend_zones)                      # :762-763
        out = np.where(hit, blended, out)
    spec = ((i1 <= sbw) | (j1 <= sbw)
            | (i1 >= ide - sbw) | (j1 >= jde - sbw))     # :766-767
    out = np.where(spec, work_c, out)                    # :768
    return out[0] if squeeze else out


def np_adjust_tempqv(mub, save_mub, c3, c4, p_top, th, pp, qv, *,
                     use_theta_m=0):
    """Float64 mirror of ``nest_adjust_tempqv`` (nest_init_utils.F:812-890).

    Corrects theta and qv for the MUB change from terrain blending, RH-
    conserving: full pressure before/after the blend from the hybrid
    column ``p = c4(k) + c3(k)*mub + p_top + pp`` (:851, :867 -- ``pp``
    is read, never modified), a two-step dry-adiabatic-lapse temperature
    correction (:874-875, coefficient ``-191.86e-3`` per the :868
    comment ``2*(g/cp-6.5e-3)*R_dry/g``), and RH converted back to qv
    with the WRF-literal Magnus constants 610.78/17.0809/234.175 and the
    0.622 epsilon (:857-858, :882-884).  Both ``use_theta_m`` branches
    (:852-856, :869-880) are transliterated; ``R_v/R_d`` comes from the
    constants module (never hardcoded).  The dummy ``znw`` argument of
    the Fortran interface is unused in its body and is dropped here.

    Shapes: ``mub``/``save_mub`` ``(ny, nx)``; ``c3``/``c4`` ``(nz,)``
    half-level columns; ``th``/``pp``/``qv`` ``(nz, ny, nx)``.  Returns
    ``(th_new, qv_new)``.
    """
    mub = np.asarray(mub, dtype=np.float64)
    save_mub = np.asarray(save_mub, dtype=np.float64)
    c3 = np.asarray(c3, dtype=np.float64)[:, None, None]
    c4 = np.asarray(c4, dtype=np.float64)[:, None, None]
    th = np.asarray(th, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.float64)
    qv = np.asarray(qv, dtype=np.float64)
    p_top = float(p_top)
    rvord = c.RVOVRD                           # module_model_constants rvovrd
    # Pass 1: pre-blend pressure and conserved RH (:848-862).
    p_old = c4 + c3 * save_mub[None] + p_top + pp        # :851
    if use_theta_m == 1:
        tc = ((th + 300.0) * (p_old / 1.0e5) ** (2.0 / 7.0)
              / (1.0 + rvord * qv) - 273.15)             # :853
    else:
        tc = (th + 300.0) * (p_old / 1.0e5) ** (2.0 / 7.0) - 273.15  # :855
    es = 610.78 * np.exp(17.0809 * tc / (234.175 + tc))  # :857
    e = qv * p_old / (0.622 + qv)                        # :858
    rh = e / es                                          # :859
    # Pass 2: post-blend pressure, theta correction, RH -> qv (:864-887).
    p_new = c4 + c3 * mub[None] + p_top + pp             # :867
    if use_theta_m == 1:
        thloc = (th + 300.0) / (1.0 + rvord * qv)        # :870
    else:
        thloc = th + 300.0                               # :872
    dth1 = (-191.86e-3 * thloc / (p_new + p_old)
            * (p_new - p_old))                           # :874
    dth = (-191.86e-3 * (thloc + 0.5 * dth1) / (p_new + p_old)
           * (p_new - p_old))                            # :875
    if use_theta_m == 1:
        th_new = (thloc + dth) * (1.0 + rvord * qv) - 300.0     # :877
    else:
        th_new = (thloc + dth) - 300.0                   # :879
    tc = (thloc + dth) * (p_new / 1.0e5) ** (2.0 / 7.0) - 273.15  # :881
    es = 610.78 * np.exp(17.0809 * tc / (234.175 + tc))  # :882
    e = rh * es                                          # :883
    qv_new = 0.622 * e / (p_new - e)                     # :884
    return th_new, qv_new


def _feedback_range(pos, spec_zone, n_stag_extent, ratio, stag):
    """copy_fcn/copy_fcnm/copy_fcni parent loop bounds (1-based)::

        DO ci = MAX(ipos+spec_zone,cits),
                MIN(ipos+(nide-nids)/nri-istag-spec_zone,cite)

    (interp_fcn.F:1466/:1470 and every sibling branch), with istag = 0
    for the staggered direction, 1 otherwise (:1459-1461).  Serial woof
    covers the whole tile, so cits/cite never bind.
    """
    istag = 0 if stag else 1
    lo = pos + spec_zone
    hi = pos + n_stag_extent // ratio - istag - spec_zone
    return lo, hi


def np_copy_fcn(cfld, nfld, *, i_parent_start, j_parent_start, nri, nrj,
                spec_zone=1, xstag=False, ystag=False):
    """Float64 mirror of ``nest_copy_fcn`` (interp_fcn.F:1397-1742), the
    dormant Phase-5b feedback averaging operator, BOTH parity branches.

    - odd ratio, mass (:1463-1517): parent cell = 1/(nri*nrj) cell
      average of all nri*nrj child points centered on
      ``ni = (ci-ipos)*nri + nri/2 + 1``;
    - odd ratio, x/y stagger (:1519-1562): along-face 1/nri (1/nrj)
      average over the stride-nri ijpoints loop (:1527/:1549);
    - even ratio, mass (:1567-1663): the SAME 1/(nri*nrj) cell average
      (ipoints/jpoints 0..nri-1 from ``ni = (ci-ipos)*nri + istag``);
    - even ratio, u/v (:1667-1737): along-face 1/nri average at the
      coincident face (:1702/:1725).

    The design-spec's old "odd-ratio point sampling" text is stale
    against v4.6.1 (registered deviations list): both branches
    cell-average.  The accumulation weight ``1./REAL(nri*nrj)`` (:1477
    etc.) is a REAL constant: the mirror widens the FP32 value (same
    FP32-rounded-weights contract as the SINT geometry).  ``imask`` and
    ``passes`` are unused by the Fortran body and are dropped.

    ``cfld`` is the parent field ``(nz, ny_p, nx_p)``; ``nfld`` the
    child ``(nz, ny_c, nx_c)`` at matching stagger (mass counts +1 in
    the staggered direction).  Child staggered domain extents
    ``nide-nids``/``njde-njds`` are the child MASS counts.  Returns an
    updated float64 copy of ``cfld``.
    """
    if int(nri) != int(nrj):
        raise ValueError("copy_fcn requires a square refinement ratio "
                         "(interp_fcn.F:1447-1448 aspect-ratio caveat)")
    if xstag and ystag:
        raise ValueError("no doubly staggered fields exist on the C-grid")
    nri, nrj = int(nri), int(nrj)
    ipos, jpos = int(i_parent_start), int(j_parent_start)
    out = np.asarray(cfld, dtype=np.float64).copy()
    nf = np.asarray(nfld, dtype=np.float64)
    if out.ndim == 2:
        out, nf, squeeze = out[None], nf[None], True
    else:
        squeeze = False
    nz, nyc, nxc = nf.shape
    # Child staggered extents (nide-nids, njde-njds) = mass counts.
    nide_span = nxc - 1 if xstag else nxc
    njde_span = nyc - 1 if ystag else nyc
    ci_lo, ci_hi = _feedback_range(ipos, spec_zone, nide_span, nri, xstag)
    cj_lo, cj_hi = _feedback_range(jpos, spec_zone, njde_span, nrj, ystag)
    odd = nrj % 2 != 0                                   # :1463
    w_mass = np.float64(np.float32(1.0) / np.float32(nri * nrj))  # :1477
    w_face = np.float64(np.float32(1.0) / np.float32(nri))        # :1531
    for cj in range(cj_lo, cj_hi + 1):
        for ci in range(ci_lo, ci_hi + 1):
            if odd:
                if not xstag and not ystag:              # :1465-1517
                    ni = (ci - ipos) * nri + nri // 2 + 1
                    nj = (cj - jpos) * nrj + nrj // 2 + 1
                    ijpts = range(1, nri * nrj + 1)      # :1473
                    w = w_mass
                elif xstag:                              # :1519-1539
                    ni = (ci - ipos) * nri + 1
                    nj = (cj - jpos) * nrj + nrj // 2 + 1
                    ijpts = range((nri + 1) // 2,
                                  (nri + 1) // 2 + nri * (nri - 1) + 1,
                                  nri)                   # :1527
                    w = w_face
                else:                                    # :1541-1562
                    ni = (ci - ipos) * nri + nri // 2 + 1
                    nj = (cj - jpos) * nrj + 1
                    start = (nrj * nrj + 1) // 2 - nrj // 2
                    ijpts = range(start, start + nrj)    # :1549
                    w = w_face
                acc = np.float64(0.0)
                for ijpoints in ijpts:
                    ipoints = (ijpoints - 1) % nri + 1 - nri // 2 - 1  # :1474
                    jpoints = (ijpoints - 1) // nri + 1 - nrj // 2 - 1  # :1475
                    acc = acc + w * nf[:, nj + jpoints - 1, ni + ipoints - 1]
            else:
                if not xstag and not ystag:              # :1643-1663
                    ni = (ci - ipos) * nri + 1           # istag = 1 (:1648)
                    nj = (cj - jpos) * nrj + 1
                    ijpts = range(1, nri * nrj + 1)      # :1650
                    w = w_mass
                elif xstag:                              # :1695-1713
                    ni = (ci - ipos) * nri + 1
                    nj = (cj - jpos) * nrj + 1
                    ijpts = range(1, nri * nrj + 1, nri)  # :1702
                    w = w_face
                else:                                    # :1717-1737
                    ni = (ci - ipos) * nri + 1
                    nj = (cj - jpos) * nrj + 1
                    ijpts = range(1, nri + 1)            # :1725
                    w = w_face
                acc = np.float64(0.0)
                for ijpoints in ijpts:
                    ipoints = (ijpoints - 1) % nri       # :1651
                    jpoints = (ijpoints - 1) // nri      # :1652
                    acc = acc + w * nf[:, nj + jpoints - 1, ni + ipoints - 1]
            out[:, cj - 1, ci - 1] = acc
    return out[0] if squeeze else out


def _copy_pick(out, nf, ipos, jpos, nri, nrj, spec_zone, xstag, ystag):
    """Shared single-point pick for copy_fcnm/copy_fcni.

    Odd ratio (:1793-1804 / :1875-1886): ``ni = (ci-ipos)*nri+istag+1``
    (the center child for mass, the coincident face for staggers).
    Even ratio (:1806-1818 / :1888-1900): SW-corner nearest neighbor
    ``ni = (ci-ipos)*nri + 1 + (nri/2 - 1)``.
    """
    nz, nyc, nxc = nf.shape
    nide_span = nxc - 1 if xstag else nxc
    njde_span = nyc - 1 if ystag else nyc
    ci_lo, ci_hi = _feedback_range(ipos, spec_zone, nide_span, nri, xstag)
    cj_lo, cj_hi = _feedback_range(jpos, spec_zone, njde_span, nrj, ystag)
    istag = 0 if xstag else 1
    jstag = 0 if ystag else 1
    odd = nrj % 2 != 0
    for cj in range(cj_lo, cj_hi + 1):
        for ci in range(ci_lo, ci_hi + 1):
            if odd:                                      # :1795-1800
                ni = (ci - ipos) * nri + istag + 1
                nj = (cj - jpos) * nrj + jstag + 1
            else:                                        # :1808-1815
                ni = (ci - ipos) * nri + 1 + (nri // 2 - 1)
                nj = (cj - jpos) * nrj + 1 + (nrj // 2 - 1)
            out[:, cj - 1, ci - 1] = nf[:, nj - 1, ni - 1]
    return out


def np_copy_fcnm(cfld, nfld, *, i_parent_start, j_parent_start, nri, nrj,
                 spec_zone=1, xstag=False, ystag=False):
    """Float64 mirror of ``nest_copy_fcnm`` (interp_fcn.F:1747-1824): the
    1-pt masked-field feedback -- center child on odd ratios, SW-corner
    nearest neighbor on even ratios ("pick nearest neighbor on SW
    corner", :1806).  Dormant Phase-5b machinery, oracled at N1.
    """
    if int(nri) != int(nrj):
        raise ValueError("copy_fcnm requires a square refinement ratio")
    out = np.asarray(cfld, dtype=np.float64).copy()
    nf = np.asarray(nfld, dtype=np.float64)
    squeeze = out.ndim == 2
    if squeeze:
        out, nf = out[None], nf[None]
    out = _copy_pick(out, nf, int(i_parent_start), int(j_parent_start),
                     int(nri), int(nrj), int(spec_zone), xstag, ystag)
    return out[0] if squeeze else out


def np_copy_fcni(cfld, nfld, *, i_parent_start, j_parent_start, nri, nrj,
                 spec_zone=1, xstag=False, ystag=False):
    """Integer mirror of ``nest_copy_fcni`` (interp_fcn.F:1829-1906): the
    1-pt feedback for INTEGER fields, same picks as copy_fcnm (odd
    center :1877-1886, even SW corner :1888-1900).  Dormant Phase-5b.
    """
    if int(nri) != int(nrj):
        raise ValueError("copy_fcni requires a square refinement ratio")
    out = np.asarray(cfld)
    nf = np.asarray(nfld)
    if not (np.issubdtype(out.dtype, np.integer)
            and np.issubdtype(nf.dtype, np.integer)):
        raise TypeError("copy_fcni operates on INTEGER fields")
    out = out.copy()
    squeeze = out.ndim == 2
    if squeeze:
        out, nf = out[None], nf[None]
    out = _copy_pick(out, nf, int(i_parent_start), int(j_parent_start),
                     int(nri), int(nrj), int(spec_zone), xstag, ystag)
    return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# WRF v4.6.1 km_opt=3 (3-D Smagorinsky) float64 mirrors -- periodic grid,
# map factors 1, full total-geopotential metrics.  Pointwise transliterations
# of module_diffusion_em.F: calculate_N2 (:1485-1713), smag_km (:1777-1929),
# vertical_diffusion_s (:4789-4907), and vertical_diffusion_2's prescribed-
# flux surface branches (:4155-4200 vflux CASE(0), :4286-4305 hflux
# CASE(0,2)).  Deliberately loop-based: each value is assembled exactly as
# the Fortran assembles it, independent of the CUDA traversal.
# ---------------------------------------------------------------------------

def _wrf_column_geometry(phi):
    """Closures for the phi-derived vertical metrics on a periodic grid."""
    phi = np.asarray(phi, dtype=np.float64)
    nzp1, ny, nx = phi.shape
    nz = nzp1 - 1
    G = 9.81

    def ph(kw, j, i):
        return phi[min(max(kw, 0), nz), j % ny, i % nx]

    def rdzw(k, j, i):
        kk = min(max(k, 0), nz - 1)
        return G / (ph(kk + 1, j, i) - ph(kk, j, i))

    def rdz(kw, j, i):
        if kw <= 0:
            return 2.0 * G / (ph(1, j, i) - ph(0, j, i))
        if kw >= nz:
            return 2.0 * G / (ph(nz, j, i) - ph(nz - 1, j, i))
        return 2.0 * G / (ph(kw + 1, j, i) - ph(kw - 1, j, i))

    return ph, rdzw, rdz


def np_wrf_calc_n2(thp, thb, p, phi, *, cf1, cf2, cf3,
                   qv=None, qc=None, qi=None, force_dry_branch=False):
    """Mirror of ``wrf_calc_n2`` (WRF ``calculate_N2``).

    ``thp (nz, ny, nx)`` is woof's theta perturbation against ``thb``
    ((nz,) or (nz, ny, nx)); ``p`` the full pressure; ``phi (nz+1, ny, nx)``
    the total geopotential; ``qv/qc/qi`` optional mixing ratios (None =
    absent, the dry reduction).  Returns BN2 ``(nz, ny, nx)`` float64 with
    WRF's three level branches: centered interior (kts+1..ktf-1), the
    MARTA/WCS one-sided surface form, saturated moist-adiabatic branch when
    ``qv >= qvs or qc >= 1e-5``, and the ktf copy of ktf-1.

    ``force_dry_branch`` mirrors the spec 3.3 MUTATION CONTROL --
    ``GPUWM_MUTATE_MOIST_N2_FORCE_DRY`` in ``smag2d.cu``, selected by
    ``woof.core.moist_n2_mutation``.  It suppresses the saturated branch
    **with the moisture still present**, which is not the same reduction as
    passing ``qv=qc=qi=None``: that would also zero ``qtot`` and the
    ``1.61*(qv_p - qv_m)`` term and so would test a different thing.  The
    distinction is the whole point of the control -- it isolates the
    saturated branch from moisture itself.  Off by default; the default
    path is the unmutated mirror.
    """
    G, RD, RV = 9.81, 287.0, 461.6
    CP = 7.0 * RD / 2.0
    RCP = RD / CP
    P0 = 1.0e5
    XLV = 2.5e6
    SVP1, SVP2, SVP3, SVPT0 = 0.6112, 17.67, 29.65, 273.15
    EP2 = RD / RV

    thp = np.asarray(thp, dtype=np.float64)
    thb = np.asarray(thb, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    nz, ny, nx = thp.shape
    theta = thb[:, None, None] + thp if thb.ndim == 1 else thb + thp
    zero = np.zeros_like(thp)
    qv = zero if qv is None else np.asarray(qv, dtype=np.float64)
    qc = zero if qc is None else np.asarray(qc, dtype=np.float64)
    qi = zero if qi is None else np.asarray(qi, dtype=np.float64)
    qtot = qv + qc + qi
    moist = bool(qv.any() or qc.any() or qi.any())

    t = theta * (p / P0) ** RCP
    tc = t - SVPT0
    es = 1000.0 * SVP1 * np.exp(SVP2 * tc / (t - SVP3))
    qvs = EP2 * es / (p - es)
    saturated = ((qv >= qvs) | (qc >= 1.0e-5) if moist
                 else np.zeros(thp.shape, dtype=bool))
    if force_dry_branch:
        # Mutation control (see the docstring): the branch is suppressed,
        # the moisture is not.
        saturated = np.zeros(thp.shape, dtype=bool)

    _, rdzw, rdz = _wrf_column_geometry(phi)
    phi = np.asarray(phi, dtype=np.float64)

    def coefa_at(k, j, i):
        xlvqv = XLV * qv[k, j, i]
        return ((1.0 + xlvqv / RD / t[k, j, i])
                / (1.0 + XLV * xlvqv / CP / RV / t[k, j, i] / t[k, j, i])
                / theta[k, j, i])

    def theta_e(k, j, i):
        return theta[k, j, i] * (
            1.0 + XLV * qvs[k, j, i] / CP / t[k, j, i])

    bn2 = np.zeros_like(thp)
    for j in range(ny):
        for i in range(nx):
            for k in range(1, nz - 1):
                tmpdz = 1.0 / rdz(k, j, i) + 1.0 / rdz(k + 1, j, i)
                if saturated[k, j, i]:
                    bn2[k, j, i] = G * (
                        coefa_at(k, j, i)
                        * (theta_e(k + 1, j, i) - theta_e(k - 1, j, i))
                        / tmpdz
                        - (qtot[k + 1, j, i] - qtot[k - 1, j, i]) / tmpdz)
                else:
                    bn2[k, j, i] = G * (
                        (theta[k + 1, j, i] - theta[k - 1, j, i])
                        / theta[k, j, i] / tmpdz
                        + 1.61 * (qv[k + 1, j, i] - qv[k - 1, j, i]) / tmpdz
                        - (qtot[k + 1, j, i] - qtot[k - 1, j, i]) / tmpdz)
            # Surface level kts.
            k = 0
            qtot_sfc = (cf1 * qtot[0, j, i] + cf2 * qtot[1, j, i]
                        + cf3 * qtot[2, j, i])
            if saturated[k, j, i]:
                tmpdz = 1.0 / rdz(1, j, i) + 0.5 / rdzw(0, j, i)
                z0 = phi[0, j, i] / G
                z1 = 0.5 * (phi[0, j, i] + phi[1, j, i]) / G
                z2 = 0.5 * (phi[1, j, i] + phi[2, j, i]) / G
                w1 = (z0 - z2) / (z1 - z2)
                w2 = 1.0 - w1
                p8w0 = w1 * p[0, j, i] + w2 * p[1, j, i]
                t8w0 = w1 * t[0, j, i] + w2 * t[1, j, i]
                thetasfc = t8w0 / (p8w0 / P0) ** RCP
                qvsfc = (cf1 * qvs[0, j, i] + cf2 * qvs[1, j, i]
                         + cf3 * qvs[2, j, i])
                thetaesfc = thetasfc * (1.0 + XLV * qvsfc / CP / t8w0)
                bn2[k, j, i] = G * (
                    coefa_at(k, j, i)
                    * (theta_e(1, j, i) - thetaesfc) / tmpdz
                    - (qtot[1, j, i] - qtot_sfc) / tmpdz)
            else:
                qvsfc = (cf1 * qv[0, j, i] + cf2 * qv[1, j, i]
                         + cf3 * qv[2, j, i])
                tmpdz = 1.0 / rdzw(0, j, i)
                bn2[k, j, i] = G * (
                    (theta[1, j, i] - theta[0, j, i])
                    / theta[0, j, i] / tmpdz
                    + 1.61 * (qv[1, j, i] - qvsfc) / tmpdz
                    - (qtot[1, j, i] - qtot_sfc) / tmpdz)
            bn2[nz - 1, j, i] = bn2[nz - 2, j, i]
    return bn2


def np_wrf_smag3d_km(u, v, w, phi, bn2, *, dx, dy, dt, c_s,
                     mix_upper_bound, mix_isotropic,
                     fnm, fnp, dn, dnw, cf1, cf2, cf3,
                     prandtl=1.0 / 3.0):
    """Mirror of ``wrf_smag3d_km`` (WRF ``smag_km``), periodic, msf = 1.

    Assembles the FULL deformation invariant -- D11/D22/D12 from
    ``cal_deform_and_div``'s metric-aware forms, D13/D23 with their w-level
    slope terms, D33 = 2*dw/dz -- averages the off-diagonal tensors to mass
    points BEFORE squaring (average-then-square), applies the buoyancy
    reduction ``sqrt(max(0, D^2 - BN2/prandtl))`` and WRF's two
    ``mix_isotropic`` branches with the exact floors and
    ``mix_upper_bound/dt`` caps.  Returns ``(xkmh, xkhh, xkmv, xkhv)``.
    """
    w = np.asarray(w, dtype=np.float64)
    bn2 = np.asarray(bn2, dtype=np.float64)
    nz, ny, nx = bn2.shape
    d = _wrf_deformation_closures(u, v, w, phi, dx=dx, dy=dy,
                                  fnm=fnm, fnp=fnp, dn=dn, dnw=dnw,
                                  cf1=cf1, cf2=cf2, cf3=cf3, shape=bn2.shape)
    rdzw = d["rdzw"]
    defor11, defor22, defor12 = d["defor11"], d["defor22"], d["defor12"]
    defor13, defor23 = d["defor13"], d["defor23"]

    xkmh = np.zeros((nz, ny, nx))
    xkhh = np.zeros_like(xkmh)
    xkmv = np.zeros_like(xkmh)
    xkhv = np.zeros_like(xkmh)
    for j in range(ny):
        for i in range(nx):
            for k in range(nz):
                d11 = defor11(k, j, i)
                d22 = defor22(k, j, i)
                d33 = 2.0 * (w[k + 1, j, i] - w[k, j, i]) * rdzw(k, j, i)
                d12 = 0.25 * (defor12(k, j, i) + defor12(k, j + 1, i)
                              + defor12(k, j, i + 1)
                              + defor12(k, j + 1, i + 1))
                d13 = 0.25 * (defor13(k + 1, j, i) + defor13(k, j, i)
                              + defor13(k + 1, j, i + 1)
                              + defor13(k, j, i + 1))
                d23 = 0.25 * (defor23(k + 1, j, i) + defor23(k, j, i)
                              + defor23(k + 1, j + 1, i)
                              + defor23(k, j + 1, i))
                def2 = (0.5 * (d11 * d11 + d22 * d22 + d33 * d33)
                        + d12 * d12 + d13 * d13 + d23 * d23)
                tmp = np.sqrt(max(
                    0.0, def2 - bn2[k, j, i] / prandtl))
                if mix_isotropic == 0:
                    mlen_h2 = dx * dy
                    mlen_v = 1.0 / rdzw(k, j, i)
                    mlen_v2 = mlen_v * mlen_v
                    kmh = max(c_s * c_s * mlen_h2 * tmp, 1.0e-6 * mlen_h2)
                    kmh = min(kmh, mix_upper_bound * mlen_h2 / dt)
                    kmv = max(c_s * c_s * mlen_v2 * tmp, 1.0e-6 * mlen_v2)
                    kmv = min(kmv, mix_upper_bound * mlen_v2 / dt)
                    khh = min(kmh / prandtl,
                              mix_upper_bound * mlen_h2 / dt)
                    khv = min(kmv / prandtl,
                              mix_upper_bound * mlen_v2 / dt)
                else:
                    rz = rdzw(k, j, i)
                    deltas = (dx * dy / rz) ** 0.33333333
                    deltas2 = deltas * deltas
                    kmh = max(c_s * c_s * deltas2 * tmp, 1.0e-6 * deltas2)
                    kmh = min(kmh, mix_upper_bound * dx * dy / dt)
                    kmv = min(kmh, mix_upper_bound / rz / rz / dt)
                    khh = min(kmh / prandtl,
                              mix_upper_bound * dx * dy / dt)
                    khv = min(kmv / prandtl,
                              mix_upper_bound / rz / rz / dt)
                xkmh[k, j, i] = kmh
                xkhh[k, j, i] = khh
                xkmv[k, j, i] = kmv
                xkhv[k, j, i] = khv
    return xkmh, xkhh, xkmv, xkhv


def np_wrf_vertical_diffusion_s(var, khv, rho, phi, *, fnm, fnp, dnw,
                                thb=None):
    """Mirror of ``wrf_smag_vd_s`` (WRF ``vertical_diffusion_s``,
    doing_tke=.false.): H3 fluxes on w levels with fnm/fnp K and density
    averages, H3 = 0 at surface and top, tendency += g*dH3/dnw.  ``thb``
    (1-D or 3-D), when given, reconstructs the full-theta difference from
    perturbation storage (the T0 offset cancels in the k difference).
    Returns the tendency increment ``(nz, ny, nx)`` float64.
    """
    G = 9.81
    var = np.asarray(var, dtype=np.float64)
    khv = np.asarray(khv, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    fnm = np.asarray(fnm, dtype=np.float64)
    fnp = np.asarray(fnp, dtype=np.float64)
    dnw = np.asarray(dnw, dtype=np.float64)
    nz, ny, nx = var.shape
    if thb is None:
        full = var
    else:
        thb = np.asarray(thb, dtype=np.float64)
        full = var + (thb[:, None, None] if thb.ndim == 1 else thb)
    _, _, rdz = _wrf_column_geometry(phi)

    h3 = np.zeros((nz + 1, ny, nx))
    for kw in range(1, nz):
        xkx = fnm[kw] * khv[kw] + fnp[kw] * khv[kw - 1]
        xkx = xkx * (fnm[kw] * rho[kw] + fnp[kw] * rho[kw - 1])
        for j in range(ny):
            for i in range(nx):
                h3[kw, j, i] = (-xkx[j, i]
                                * (full[kw, j, i] - full[kw - 1, j, i])
                                * rdz(kw, j, i))
    tend = np.zeros_like(var)
    for k in range(nz):
        tend[k] = G * (h3[k + 1] - h3[k]) / dnw[k]
    return tend


def np_wrf_surface_heat_const(heat_flux, rho0, dnw0):
    """Mirror of ``wrf_smag_surface_heat_const`` (hflux CASE(0,2),
    use_theta_m=0): the k=0 coupled-theta increment
    ``-g*heat_flux*rho(kts)/dnw(kts)`` (positive for positive flux since
    WRF's dnw < 0)."""
    return -9.81 * float(heat_flux) * np.asarray(
        rho0, dtype=np.float64) / float(dnw0)


def np_wrf_surface_mom_cd0(u0, v0, rho0, cd0, dnw0):
    """Mirror of the ``vflux CASE(0)`` constant-drag wall stress: the k=0
    ru/rv increments ``g*tao*rho_avg/dnw`` with ``tao = cd0*|V|*u``.

    ``u0 (ny, nx+1)`` / ``v0 (ny+1, nx)`` are the k=0 staggered winds
    (periodic core), ``rho0 (ny, nx)`` the k=0 density.  Returns
    ``(ru0, rv0)`` on the staggered shapes.
    """
    G, eps = 9.81, 1.0e-15
    u0 = np.asarray(u0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    rho0 = np.asarray(rho0, dtype=np.float64)
    ny, nx = rho0.shape
    ru = np.zeros_like(u0)
    rv = np.zeros_like(v0)
    for j in range(ny):
        for i in range(nx + 1):
            ic, im = i % nx, (i - 1) % nx
            vv = 0.25 * (v0[j, ic] + v0[(j + 1) % ny, ic]
                         + v0[j, im] + v0[(j + 1) % ny, im])
            uu = u0[j, min(i, nx)]
            speed = np.sqrt(uu * uu + vv * vv) + eps
            stress = cd0 * speed * uu
            rhoavg = 0.5 * (rho0[j, ic] + rho0[j, im])
            ru[j, i] = G * stress * rhoavg / dnw0
    for j in range(ny + 1):
        for i in range(nx):
            jc, jm = j % ny, (j - 1) % ny
            uu = 0.25 * (u0[jc, i] + u0[jc, (i + 1) % nx]
                         + u0[jm, i] + u0[jm, (i + 1) % nx])
            vv = v0[min(j, ny), i]
            speed = np.sqrt(vv * vv + uu * uu) + eps
            stress = cd0 * speed * vv
            rhoavg = 0.5 * (rho0[jc, i] + rho0[jm, i])
            rv[j, i] = G * stress * rhoavg / dnw0
    return ru, rv


def _wrf_deformation_closures(u, v, w, phi, *, dx, dy,
                              fnm, fnp, dn, dnw, cf1, cf2, cf3, shape=None):
    """Shared float64 closures for the metric-aware WRF deformations on a
    periodic msf=1 grid (cal_deform_and_div transliteration).  Returns a
    dict of pointwise functions; used by the km_opt=2/3 mirrors."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    fnm = np.asarray(fnm, dtype=np.float64)
    fnp = np.asarray(fnp, dtype=np.float64)
    dn = np.asarray(dn, dtype=np.float64)
    dnw = np.asarray(dnw, dtype=np.float64)
    nz = phi.shape[0] - 1
    ny, nx = phi.shape[1:]
    if shape is not None:
        nz, ny, nx = shape
    rdx, rdy = 1.0 / dx, 1.0 / dy
    G = 9.81
    ph, rdzw, rdz = _wrf_column_geometry(phi)

    def uhat(k, j, i):
        return u[k, j % ny, i % nx]

    def vhat(k, j, i):
        return v[k, j % ny, i % nx]

    def what(kw, j, i):
        return w[min(max(kw, 0), nz), j % ny, i % nx]

    def zx(kw, j, iface):
        return rdx * (ph(kw, j, iface) - ph(kw, j, iface - 1)) / G

    def zy(kw, jface, i):
        return rdy * (ph(kw, jface, i) - ph(kw, jface - 1, i)) / G

    def full_weights(kw, pair):
        if kw <= 0:
            return cf1 * pair(0) + cf2 * pair(1) + cf3 * pair(2)
        if kw >= nz:
            cft2 = -0.5 * dnw[nz - 1] / dn[nz - 1]
            return (1.0 - cft2) * pair(nz - 1) + cft2 * pair(nz - 2)
        return fnm[kw] * pair(kw) + fnp[kw] * pair(kw - 1)

    def u_w_xcenter(kw, j, i):
        return 0.5 * full_weights(
            kw, lambda k: uhat(k, j, i) + uhat(k, j, i + 1))

    def v_w_ycenter(kw, j, i):
        return 0.5 * full_weights(
            kw, lambda k: vhat(k, j, i) + vhat(k, j + 1, i))

    def u_w_corner(kw, j, i):
        return 0.5 * full_weights(
            kw, lambda k: uhat(k, j - 1, i) + uhat(k, j, i))

    def v_w_corner(kw, j, i):
        return 0.5 * full_weights(
            kw, lambda k: vhat(k, j, i - 1) + vhat(k, j, i))

    def defor11(k, j, i):
        tmpzx = 0.25 * (zx(k, j, i) + zx(k, j, i + 1)
                        + zx(k + 1, j, i) + zx(k + 1, j, i + 1))
        slope = ((u_w_xcenter(k + 1, j, i) - u_w_xcenter(k, j, i))
                 * tmpzx * rdzw(k, j, i))
        return 2.0 * (rdx * (uhat(k, j, i + 1) - uhat(k, j, i)) - slope)

    def defor22(k, j, i):
        tmpzy = 0.25 * (zy(k, j, i) + zy(k, j + 1, i)
                        + zy(k + 1, j, i) + zy(k + 1, j + 1, i))
        slope = ((v_w_ycenter(k + 1, j, i) - v_w_ycenter(k, j, i))
                 * tmpzy * rdzw(k, j, i))
        return 2.0 * (rdy * (vhat(k, j + 1, i) - vhat(k, j, i)) - slope)

    def defor12(k, j, i):
        rr = (rdzw(k, j, i) + rdzw(k, j, i - 1)
              + rdzw(k, j - 1, i - 1) + rdzw(k, j - 1, i))
        tmpzy = 0.25 * (zy(k, j, i - 1) + zy(k, j, i)
                        + zy(k + 1, j, i - 1) + zy(k + 1, j, i))
        uslope = ((u_w_corner(k + 1, j, i) - u_w_corner(k, j, i))
                  * 0.25 * tmpzy * rr)
        tmpzx = 0.25 * (zx(k, j - 1, i) + zx(k, j, i)
                        + zx(k + 1, j - 1, i) + zx(k + 1, j, i))
        vslope = ((v_w_corner(k + 1, j, i) - v_w_corner(k, j, i))
                  * 0.25 * tmpzx * rr)
        return (rdy * (uhat(k, j, i) - uhat(k, j - 1, i)) - uslope
                + rdx * (vhat(k, j, i) - vhat(k, j, i - 1)) - vslope)

    def w_xavg(k, j, i):
        return 0.25 * (what(k, j, i) + what(k + 1, j, i)
                       + what(k, j, i - 1) + what(k + 1, j, i - 1))

    def w_yavg(k, j, i):
        return 0.25 * (what(k, j, i) + what(k + 1, j, i)
                       + what(k, j - 1, i) + what(k + 1, j - 1, i))

    def defor13(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        rz = 0.5 * (rdz(kw, j, i) + rdz(kw, j, i - 1))
        slope = ((w_xavg(kw, j, i) - w_xavg(kw - 1, j, i))
                 * zx(kw, j, i) * rz)
        dwdx = rdx * (what(kw, j, i) - what(kw, j, i - 1)) - slope
        dudz = (uhat(kw, j, i) - uhat(kw - 1, j, i)) * rz
        return dwdx + dudz

    def defor23(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        rz = 0.5 * (rdz(kw, j, i) + rdz(kw, j - 1, i))
        slope = ((w_yavg(kw, j, i) - w_yavg(kw - 1, j, i))
                 * zy(kw, j, i) * rz)
        dwdy = rdy * (what(kw, j, i) - what(kw, j - 1, i)) - slope
        dvdz = (vhat(kw, j, i) - vhat(kw - 1, j, i)) * rz
        return dwdy + dvdz

    def defor33(k, j, i):
        return 2.0 * (w[k + 1, j % ny, i % nx]
                      - w[k, j % ny, i % nx]) * rdzw(k, j, i)

    return {"defor11": defor11, "defor22": defor22, "defor12": defor12,
            "defor13": defor13, "defor23": defor23, "defor33": defor33,
            "rdzw": rdzw, "rdz": rdz, "ph": ph}


def _np_l_scale(tke_v, bn2_v, deltas):
    """WRF calc_l_scale, one point (module_diffusion_em.F:2341-2406)."""
    l = deltas
    if bn2_v > 1.0e-6:
        tmp = np.sqrt(max(tke_v, 1.0e-6))
        l = 0.76 * tmp / np.sqrt(bn2_v)
        l = min(l, deltas)
        l = max(l, 0.001 * deltas)
    return l


def np_wrf_tke_km(thp, thb, p, tke, bn2, phi, *, dx, dy, dt, c_k,
                  mix_upper_bound, mix_isotropic, tke_seed,
                  cf1, cf2, cf3, prandtl=1.0 / 3.0):
    """Mirror of ``wrf_tke_km`` (WRF ``tke_km``, :2049-2260), periodic,
    msf = 1: both mix_isotropic branches, the dthrdn stability length with
    phy_prep's surface/top p8w-t8w extrapolations, the tke_seed floor, and
    WRF's limiter asymmetry (anisotropic xkhh/xkhv uncapped).  Returns
    ``(xkmh, xkhh, xkmv, xkhv)``.
    """
    G, RD = 9.81, 287.0
    CP = 7.0 * RD / 2.0
    RCP = RD / CP
    P0 = 1.0e5
    thp = np.asarray(thp, dtype=np.float64)
    thb = np.asarray(thb, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    tke = np.asarray(tke, dtype=np.float64)
    bn2 = np.asarray(bn2, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    nz, ny, nx = thp.shape
    theta = thb[:, None, None] + thp if thb.ndim == 1 else thb + thp
    t = theta * (p / P0) ** RCP
    _, rdzw, rdz = _wrf_column_geometry(phi)

    xkmh = np.zeros((nz, ny, nx))
    xkhh = np.zeros_like(xkmh)
    xkmv = np.zeros_like(xkmh)
    xkhv = np.zeros_like(xkmh)
    for j in range(ny):
        for i in range(nx):
            for k in range(nz):
                tmp = np.sqrt(max(tke[k, j, i], tke_seed))
                rz = rdzw(k, j, i)
                if mix_isotropic == 0:
                    if k == 0:
                        tmpdz = (1.0 / rdzw(1, j, i)
                                 + 1.0 / rdzw(0, j, i))
                        z0 = phi[0, j, i] / G
                        z1 = 0.5 * (phi[0, j, i] + phi[1, j, i]) / G
                        z2 = 0.5 * (phi[1, j, i] + phi[2, j, i]) / G
                        w1 = (z0 - z2) / (z1 - z2)
                        w2 = 1.0 - w1
                        p8w0 = w1 * p[0, j, i] + w2 * p[1, j, i]
                        t8w0 = w1 * t[0, j, i] + w2 * t[1, j, i]
                        thetasfc = t8w0 / (p8w0 / P0) ** RCP
                        dthrdn = (theta[1, j, i] - thetasfc) / tmpdz
                    elif k == nz - 1:
                        tmpdz = (1.0 / rdz(nz - 1, j, i)
                                 + 0.5 / rdzw(nz - 1, j, i))
                        z0 = phi[nz, j, i] / G
                        z1 = 0.5 * (phi[nz - 1, j, i]
                                    + phi[nz, j, i]) / G
                        z2 = 0.5 * (phi[nz - 2, j, i]
                                    + phi[nz - 1, j, i]) / G
                        w1 = (z0 - z2) / (z1 - z2)
                        w2 = 1.0 - w1
                        p8wt = np.exp(w1 * np.log(p[nz - 1, j, i])
                                      + w2 * np.log(p[nz - 2, j, i]))
                        t8wt = (w1 * t[nz - 1, j, i]
                                + w2 * t[nz - 2, j, i])
                        thetatop = t8wt / (p8wt / P0) ** RCP
                        dthrdn = ((thetatop - theta[nz - 2, j, i])
                                  / tmpdz)
                    else:
                        tmpdz = (1.0 / rdz(k + 1, j, i)
                                 + 1.0 / rdz(k, j, i))
                        dthrdn = ((theta[k + 1, j, i]
                                   - theta[k - 1, j, i]) / tmpdz)
                    mlen_h = np.sqrt(dx * dy)
                    deltas = 1.0 / rz
                    mlen_v = deltas
                    if dthrdn > 0.0:
                        mlen_s = 0.76 * tmp / np.sqrt(
                            abs(G / theta[k, j, i] * dthrdn))
                        mlen_v = min(mlen_v, mlen_s)
                    kmh = max(c_k * tmp * mlen_h, 1.0e-6 * mlen_h ** 2)
                    kmh = min(kmh, mix_upper_bound * mlen_h ** 2 / dt)
                    kmv = max(c_k * tmp * mlen_v, 1.0e-6 * deltas ** 2)
                    kmv = min(kmv, mix_upper_bound * deltas ** 2 / dt)
                    khh = kmh / prandtl                 # uncapped (WRF)
                    khv = kmv * (1.0 + 2.0 * mlen_v / deltas)
                else:
                    deltas = (dx * dy / rz) ** 0.33333333
                    l = _np_l_scale(tke[k, j, i], bn2[k, j, i], deltas)
                    kmh = min(mix_upper_bound * dx * dy / dt,
                              c_k * tmp * l)
                    kmv = min(mix_upper_bound / rz / rz / dt,
                              c_k * tmp * l)
                    pr_inv = 1.0 + 2.0 * l / deltas
                    khh = min(mix_upper_bound * dx * dy / dt,
                              kmh * pr_inv)
                    khv = min(mix_upper_bound / rz / rz / dt,
                              kmv * pr_inv)
                xkmh[k, j, i] = kmh
                xkhh[k, j, i] = khh
                xkmv[k, j, i] = kmv
                xkhv[k, j, i] = khv
    return xkmh, xkhh, xkmv, xkhv


def np_wrf_tke_rhs(u, v, w, phi, thp, thb, tke, bn2, kmh, kmv, khv,
                   mut, c1h, c2h, *, dx, dy, dt, c_k, isfflx,
                   cd0=0.0, heat_flux=0.0, ustm=None, hfx=None, qv=None,
                   rho0=None, fnm=None, fnp=None, dn=None, dnw=None,
                   cf1=0.0, cf2=0.0, cf3=0.0, return_terms=False):
    """Mirror of ``wrf_tke_rhs`` (WRF tke_shear + tke_buoyancy +
    tke_dissip + the positivity limiter): the once-per-step coupled TKE
    source on a periodic msf=1 grid.  ``return_terms=True`` additionally
    returns the (shear, buoyancy, dissipation) components BEFORE the
    limiter -- the budget-receipt decomposition.
    """
    G, RD = 9.81, 287.0
    CP = 7.0 * RD / 2.0
    d = _wrf_deformation_closures(u, v, w, phi, dx=dx, dy=dy,
                                  fnm=fnm, fnp=fnp, dn=dn, dnw=dnw,
                                  cf1=cf1, cf2=cf2, cf3=cf3)
    thp = np.asarray(thp, dtype=np.float64)
    thb = np.asarray(thb, dtype=np.float64)
    tke = np.asarray(tke, dtype=np.float64)
    bn2 = np.asarray(bn2, dtype=np.float64)
    kmh = np.asarray(kmh, dtype=np.float64)
    kmv = np.asarray(kmv, dtype=np.float64)
    khv = np.asarray(khv, dtype=np.float64)
    mut = np.asarray(mut, dtype=np.float64)
    c1h = np.asarray(c1h, dtype=np.float64)
    c2h = np.asarray(c2h, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nz, ny, nx = thp.shape
    theta = thb[:, None, None] + thp if thb.ndim == 1 else thb + thp
    ce1 = (c_k / 0.10) * 0.19
    ce2 = max(0.0, 0.93 - ce1)

    shear = np.zeros((nz, ny, nx))
    buoy = np.zeros_like(shear)
    dissip = np.zeros_like(shear)
    for j in range(ny):
        for i in range(nx):
            for k in range(nz):
                chm = c1h[k] * mut[j, i] + c2h[k]
                s = 0.0
                d11 = d["defor11"](k, j, i)
                d22 = d["defor22"](k, j, i)
                d33 = d["defor33"](k, j, i)
                s += 0.5 * chm * kmh[k, j, i] * d11 * d11
                s += 0.5 * chm * kmh[k, j, i] * d22 * d22
                s += 0.5 * chm * kmv[k, j, i] * d33 * d33
                s12 = [d["defor12"](k, j, i), d["defor12"](k, j + 1, i),
                       d["defor12"](k, j, i + 1),
                       d["defor12"](k, j + 1, i + 1)]
                s += chm * kmh[k, j, i] * 0.25 * sum(x * x for x in s12)
                s13 = [d["defor13"](k + 1, j, i), d["defor13"](k, j, i),
                       d["defor13"](k + 1, j, i + 1),
                       d["defor13"](k, j, i + 1)]
                s += chm * kmv[k, j, i] * 0.25 * sum(x * x for x in s13)
                s23 = [d["defor23"](k + 1, j, i), d["defor23"](k, j, i),
                       d["defor23"](k + 1, j + 1, i),
                       d["defor23"](k, j + 1, i)]
                s += chm * kmv[k, j, i] * 0.25 * sum(x * x for x in s23)
                if k == 0:
                    usum = u[0, j, i] + u[0, j, (i + 1) % nx]
                    vsum = v[0, j, i] + v[0, (j + 1) % ny, i]
                    absU = 0.5 * np.sqrt(usum * usum + vsum * vsum)
                    if isfflx == 0:
                        Cd = cd0
                    else:
                        absU += 1.0e-15
                        us = 0.0 if ustm is None else float(
                            np.asarray(ustm)[j, i])
                        Cd = us * us / (absU * absU)
                    d13s = 0.5 * (d["defor13"](1, j, i)
                                  + d["defor13"](1, j, i + 1))
                    s += chm * (0.5 * usum * Cd * absU * d13s)
                    d23s = 0.5 * (d["defor23"](1, j, i)
                                  + d["defor23"](1, j + 1, i))
                    s += chm * (0.5 * vsum * Cd * absU * d23s)
                shear[k, j, i] = s

                if k >= 1:
                    buoy[k, j, i] = -chm * khv[k, j, i] * bn2[k, j, i]
                else:
                    if isfflx in (0, 2):
                        hf = heat_flux
                    else:
                        vapor = 0.0 if qv is None else float(
                            np.asarray(qv)[0, j, i])
                        cpm = CP * (1.0 + 0.8 * vapor)
                        hfx_v = 0.0 if hfx is None else float(
                            np.asarray(hfx)[j, i])
                        # WRF CASE(1): heat_flux = (hfx/cpm)/rho with the
                        # vapor-loaded diffusion density at k=kts.
                        rr = 1.0 if rho0 is None else float(
                            np.asarray(rho0)[j, i])
                        hf = hfx_v / cpm / rr
                    buoy[k, j, i] = -chm * (
                        (khv[0, j, i] * bn2[0, j, i])
                        - (G / theta[0, j, i]) * hf) * 0.5
                rz = d["rdzw"](k, j, i)
                deltas = (dx * dy / rz) ** 0.33333333
                l = _np_l_scale(tke[k, j, i], bn2[k, j, i], deltas)
                coefc = 3.9 if (k == 0 or k == nz - 1) else (
                    ce1 + ce2 * l / deltas)
                tketmp = max(tke[k, j, i], 1.0e-6)
                dissip[k, j, i] = -chm * coefc * tketmp ** 1.5 / l
    total = shear + buoy + dissip
    chm3 = c1h[:, None, None] * mut[None] + c2h[:, None, None]
    total = np.maximum(total, -chm3 * np.maximum(0.0, tke) / dt)
    if return_terms:
        return total, shear, buoy, dissip
    return total
