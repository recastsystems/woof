"""WRF's implicit-explicit vertical advection, ``zadvect_implicit = 1``.

WRF 4.3 and later (Wicker and Skamarock 2020, after Shchepetkin 2015),
``dyn_em/module_ieva_em.F`` as WRF 4.7.1's ``rk_tendency`` and
``rk_scalar_tend`` call it (``dyn_em/module_em.F:436-715, 1216-1364``).
On the LAST RK3 substep only (``CHK_IEVA``: ``rk_step == rk_order``; the
2 and 3 alternatives are commented out upstream, so any value above 0
means this) the eta mass flux is split per w point into an explicit part,
which the ordinary upwind operators advect, and an implicit part, which
an upwind column solve advects:

``zadvect_implicit_variant = "wrf_legacy"`` selects the earlier WRF
``module_advect_em`` operator. Its directional horizontal-flow estimate
includes map factors and uses ``alpha_max = 1.0``. Its column solves use
the current stage mass on both sides, and its upper w boundary divides
the complete increment by g. ``"wrf_471"`` retains the existing modern
operator by default. The exact earlier source and native Fortran fixture
are pinned by ``tools/ieva_wrf_oracle/legacy_build.py``. Both variants
keep the declared A179 lower-boundary uncoupling correction below.

* :func:`split_omega` is ``WW_SPLIT``: the explicit share of each w
  point's flux is 1 where its vertical Courant number is under about
  0.73 of the allowed maximum (``alpha_max`` 1.1, less a horizontal-flow
  allowance) and falls off as ``cw_max / cw`` past it;
* :func:`column_mass_new` is ``CALC_MUT_NEW``, the column mass after dt
  the implicit coefficients divide by;
* the ``solve_*`` launchers are ``advect_{s,u,v,ph,w}_implicit``: each
  rewrites the explicit tendency as the solution of a tridiagonal system
  in double precision.

The explicit part of the flux is the only thing the explicit advection,
``rhs_ph``'s vertical term and the scalar PD limiter see.  ``w_damp`` and
the adaptive clock's vertical Courant number still read the FULL flux,
as in WRF (``module_em.F:738-745`` passes ``ww``).  WRF's
``adapt_timestep_em.F`` has no IEVA branch either: the vertical number
is held to the namelist's ``target_cfl`` whether the option is on or
not, and so is woof's adaptive clock.  A run that wants the adaptive
step to pass the explicit vertical limit under the option raises
``target_cfl`` itself, as a WRF user does.

THETA.  WRF transports ``t_2 = theta - t0`` (t0 = 300 K) and woof full
theta (``thb + theta'``), whose 300 K part the acoustic step balances
against the column mass (``kernels/acoustic.cu``: the t0-offset terms
cancel).  The implicit operator does not commute with that constant: run
on full theta it rewrites the constant's mass-flux divergence as an upwind
solve, and a uniform 300 K column under a vertical Courant number of 1.6
gains about 2 K a step.  So the theta solve runs on WRF's own variable:
the explicit advection and the solve take ``(thb - t0) + theta'``
(:func:`theta_minus_t0`, the kernel's ``shift``), exactly WRF's t_tend,
and the constant then rejoins with the FULL mass flux
(:func:`add_theta_offset_flux`), which is what the default path's full
theta tendency carries for it.  A uniform column stays uniform.  Strict
arithmetic (``GPUWM_WRF_EXACT``) is WRF's own convention end to end: the
state holds ``t_2`` (thb = t0), the explicit substeps transport ``thp``
alone and the strict acoustic kernels balance t0 themselves, so there the
constant does not rejoin.  Adding it there too counted t0's flux
divergence twice: 2.6 K of theta and 233 Pa of column mass off WRF after
one step of a real HRRR start, non-finite by step 11.

WHAT DIFFERS FROM WRF, declared.  (1) theta: woof's ``(thb - t0) +
theta'`` is formed from its own two FP32 words where WRF stores ``t_2``
in one, so the inputs differ in their last bits; the operator is WRF's.
(2) woof's stage face masses are ``0.5*(mut(i) + mut(i-1))`` where WRF's
``calc_mu_uv`` sums perturbation and base separately; ``CALC_MUT_NEW``
reads whichever the dynamics coupled its winds with, so it reads woof's.
(3) at a forced domain's outer row the w solve's terrain boundary takes
``set_w_surface``'s one-sided slope where WRF reads its halo; the
acoustic w solve never reads that row.  (4) A179, a correction rather
than a rounding: ``advect_w_implicit``'s two boundary terms are put in the
units of the ``w_old`` terms beside them.  WRF 4.7.1's lower boundary
(module_ieva_em.F:1231-1244) builds the surface w increment from the
COUPLED u/v tendencies, Pa m s-2 where m s-2 belongs, so the term is about
one column mass too large; on the terrain clock's ridge probe (35 m/s,
30 s) it took the crest's w tendency from 4.4e4 to 9.3e7 and the forecast
went NaN in step 3, in WRF's own build as in this port.  Here each of the
three lowest levels' ``ru_t``/``rv_t`` is uncoupled first, times its map
factor and over its stage face mass (WRF's own uncoupling,
module_small_step_em.F:383 and :392).  The upper boundary
(:1248-1253) divides ``(ph_new - ph_old)/dt`` by g as it does the
``ph_tend`` term beside it.  The ridge probe then holds 10 of 10 steps.
tools/ieva_wrf_oracle grades the kernel against WRF's routine with the
same two corrections; unmodified WRF differs only in the columns whose
boundary terms are active.  Everything else is WRF's arithmetic operator
for operator (kernels/ieva.cu).
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np

from woof.core import constants as c
from woof.core.acoustic import _boundary_forced as _forced
from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.core.ieva_constants import IEVA_LEVEL_TIERS, level_tier  # noqa: F401
from woof.core.kernels import get_kernel_int_defines
from woof.core.state import DTYPE, mu_at_u_faces, mu_at_v_faces

#: module_ieva_em.F:7-9 and WW_SPLIT's own PARAMETERs.
ALPHA_MAX = np.float32(1.1)
ALPHA_MIN = np.float32(0.8)
CEPS = np.float32(0.9)
LEGACY_ALPHA_MAX = np.float32(1.0)
LEGACY_CMNX_RATIO = np.float32(ALPHA_MIN / LEGACY_ALPHA_MAX)
LEGACY_CUTOFF = np.float32(np.float32(2.0) - LEGACY_CMNX_RATIO)
LEGACY_R4CMX = np.float32(np.float32(1.0)
    / (np.float32(4.0) - np.float32(4.0) * LEGACY_CMNX_RATIO))
#: gfortran folds these REAL PARAMETER expressions in single precision,
#: one rounding per operation, which is what these float32 operations do.
CMNX_RATIO = np.float32(ALPHA_MIN / ALPHA_MAX)
CUTOFF = np.float32(np.float32(2.0) - CMNX_RATIO)
R4CMX = np.float32(np.float32(1.0)
                   / (np.float32(4.0) - np.float32(4.0) * CMNX_RATIO))

#: WRF's ``t0`` (module_model_constants), the offset of ``t_2 = theta -
#: t0``, the variable WRF's theta column solve reads.
THETA_OFFSET = np.float32(c.T0)

_TPB = 128


def enabled(cfg) -> bool:
    """WRF ``zadvect_implicit > 0``."""
    return int(getattr(cfg, "zadvect_implicit", 0) or 0) > 0


def active_stage(cfg, istage: int, nstages: int = 3) -> bool:
    """WRF ``CHK_IEVA``: on, and this is the last RK substep."""
    return enabled(cfg) and int(istage) == int(nstages) - 1


def legacy(cfg) -> bool:
    """Select the earlier WRF current-mass, directional-flux operator."""
    return getattr(cfg, "zadvect_implicit_variant", "wrf_471") == "wrf_legacy"


def _kernel(nz: int, name: str):
    return get_kernel_int_defines(
        "ieva", name, (("IEVA_KMAX", level_tier(nz)),))


def _blocks(n: int) -> tuple[int]:
    return (max(1, (int(n) + _TPB - 1) // _TPB),)


def _rdx_rdy(cfg) -> tuple[np.float32, np.float32]:
    # WRF solve_em: rdx = 1./dx in single precision.
    return (np.float32(1.0) / np.float32(cfg.dx),
            np.float32(1.0) / np.float32(cfg.dy))


def _periodic_x(cfg) -> bool:
    return not (getattr(cfg, "open_x", False) or _forced(cfg))


def _periodic_y(cfg) -> bool:
    return not (getattr(cfg, "open_y", False) or _forced(cfg))


def split_omega(state, cfg, ww, u, v, mut, dt: float):
    """WRF ``WW_SPLIT`` (IEVA branch): ``(wwE, wwI)`` from ``ww``.

    ``u``/``v`` are the uncoupled winds and ``mut`` the (ny, nx) column
    mass of the call site (the stage's for the dynamics, time n's winds
    and the post-acoustic mass for the scalars), ``dt`` the full step.
    Backed by the scratch slots ``ieva_wwe``/``ieva_wwi``, which the
    scalar split reuses after the dynamics are done with them.
    """
    nz, ny, nx = state.p.shape
    wwE = state.scratch((nz + 1, ny, nx), "ieva_wwe")
    wwI = state.scratch((nz + 1, ny, nx), "ieva_wwi")
    rdx, rdy = _rdx_rdy(cfg)
    old = legacy(cfg)
    params = ((LEGACY_ALPHA_MAX, CEPS, LEGACY_CMNX_RATIO, LEGACY_CUTOFF,
               LEGACY_R4CMX) if old else
              (ALPHA_MAX, CEPS, CMNX_RATIO, CUTOFF, R4CMX))
    _kernel(nz, "ieva_ww_split")(
        _blocks((nz + 1) * ny * nx), (_TPB,),
        (ww, u, v, cp.ascontiguousarray(mut, dtype=DTYPE),
         state.rdnw, state.c1f, state.c2f,
         rdx, rdy, DTYPE(dt), *params,
         DTYPE(0.25), DTYPE(1.0), DTYPE(0.0), wwE, wwI,
         state.msft, np.int32(state.has_msf), np.int32(old),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return wwE, wwI


def stage_face_masses(state, cfg, mu):
    """The stage face column masses ``stage_fluxes`` couples ru/rv with."""
    if WRF_EXACT:
        # calc_mu_uv adds the two perturbations before the two base words.
        mp, mb = state.mup, state.mub2d
        mux0 = DTYPE(0.5) * (((mp + cp.roll(mp, 1, axis=1)) + mb)
                             + cp.roll(mb, 1, axis=1))
        muy0 = DTYPE(0.5) * (((mp + cp.roll(mp, 1, axis=0)) + mb)
                             + cp.roll(mb, 1, axis=0))
        mux = cp.concatenate((mux0, mux0[:, :1]), axis=1)
        muy = cp.concatenate((muy0, muy0[:1]), axis=0)
        if not _periodic_x(cfg):
            for face, column in ((0, 0), (-1, -1)):
                m, b = mp[:, column], mb[:, column]
                mux[:, face] = DTYPE(0.5) * (((m + m) + b) + b)
        if not _periodic_y(cfg):
            for face, row in ((0, 0), (-1, -1)):
                m, b = mp[row], mb[row]
                muy[face] = DTYPE(0.5) * (((m + m) + b) + b)
        return mux, muy
    else:
        mux = mu_at_u_faces(mu)
        muy = mu_at_v_faces(mu)
    if not _periodic_x(cfg):
        mux[:, 0] = mu[:, 0]
        mux[:, -1] = mu[:, -1]
    if not _periodic_y(cfg):
        muy[0, :] = mu[0, :]
        muy[-1, :] = mu[-1, :]
    return mux, muy


def column_mass_new(state, cfg, mut, mut_old, dt: float):
    """WRF ``CALC_MUT_NEW``: ``mut_old - dt * sum_k divv(k)``."""
    nz, ny, nx = state.p.shape
    mux, muy = stage_face_masses(state, cfg, mut)
    out = state.scratch((ny, nx), "ieva_mut_new")
    rdx, rdy = _rdx_rdy(cfg)
    _kernel(nz, "ieva_mut_new")(
        _blocks(ny * nx), (_TPB,),
        (state.u, state.v, cp.ascontiguousarray(mux),
         cp.ascontiguousarray(muy), state.c1h, state.c2h, state.rdnw,
         mut_old, state.msft, state.msfu, state.msfv_inv if WRF_EXACT else state.msfv,
         rdx, rdy, DTYPE(dt), DTYPE(1.0), np.int32(state.has_msf),
         out, np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


@dataclass
class DynamicsSplit:
    """One last-substep IEVA context for ``rk_tendency``'s solves."""

    wwE: cp.ndarray
    wwI: cp.ndarray
    mut: cp.ndarray
    mut_old: cp.ndarray
    mut_new: cp.ndarray
    dt: np.float32


def prepare_dynamics(state, cfg, ww) -> DynamicsSplit:
    """``rk_tendency``'s IEVA preamble, with the selected mass convention."""
    nz, ny, nx = state.p.shape
    dt = DTYPE(cfg.dt)
    mut = state.scratch((ny, nx), "ieva_mut")
    cp.add(state.mub2d, state.mup, out=mut)             # grid%mut
    mut_old = state.scratch((ny, nx), "ieva_mut_old")
    cp.add(state.mub2d, state.mup0, out=mut_old)        # mub + mu_1
    wwE, wwI = split_omega(state, cfg, ww, state.u, state.v, mut, dt)
    # The earlier operator uses the stage mass in both coefficients and
    # old-field terms. Modern WRF estimates a new mass for the coefficients.
    if legacy(cfg):
        mut_old = mut
        mut_new = mut
    else:
        mut_new = column_mass_new(state, cfg, mut, mut_old, dt)
    return DynamicsSplit(wwE, wwI, mut, mut_old, mut_new, dt)


def _base_mode(base) -> int:
    return 1 if base.ndim == 1 else 2


def theta_minus_t0(state) -> cp.ndarray:
    """WRF's ``t_2``: ``(thb - t0) + theta'`` at the stage, a new array.

    ``thb - t0`` first, so the subtraction is exact for any base theta
    between 150 and 600 K (Sterbenz) and only the sum rounds."""
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    out = cp.subtract(thb, THETA_OFFSET, dtype=DTYPE)
    if out.shape != state.thp.shape:
        out = cp.broadcast_to(out, state.thp.shape).copy()
    out += state.thp
    return out


def add_theta_offset_flux(state, cfg, field, ru, rv, ww, launch) -> None:
    """Add the t0 constant's flux divergence with the FULL mass flux to
    ``rth_t``, in place, through the theta operator ``launch`` itself.

    ``field`` is a scratch (nz, ny, nx) array (overwritten).  This is the
    300 K share of the default path's full-theta tendency: what the
    acoustic mass update balances.  With no implicit share anywhere the
    two launches sum to the default path's one up to rounding."""
    field.fill(THETA_OFFSET)
    launch(field, ru, rv, ww, state.rth_t)


def solve_theta(state, cfg, ctx: DynamicsSplit) -> None:
    """``advect_s_implicit`` on theta: s_old is WRF's ``t_1``, woof's
    time-n ``(thb - t0) + thp0``; ``rth_t`` must hold the explicit
    advection of :func:`theta_minus_t0` alone."""
    nz, ny, nx = state.p.shape
    _kernel(nz, "ieva_solve_s")(
        _blocks(ny * nx), (_TPB,),
        (state.rth_t, state.thp0, state.thb, np.int32(_base_mode(state.thb)),
         THETA_OFFSET, ctx.wwI, state.c1h, state.c2h, ctx.mut_old,
         ctx.mut_new, state.rdnw, ctx.dt, DTYPE(1.0), DTYPE(0.0),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def solve_u(state, cfg, ctx: DynamicsSplit) -> None:
    """``advect_u_implicit`` on ``ru_t``."""
    nz, ny, nx = state.p.shape
    periodic = _periodic_x(cfg)
    f_lo, f_hi = (0, nx) if periodic else (1, nx - 1)
    nfaces = f_hi - f_lo + 1
    old = legacy(cfg)
    face = stage_face_masses(state, cfg, ctx.mut)[0] if old else ctx.mut
    _kernel(nz, "ieva_solve_u")(
        _blocks(ny * nfaces), (_TPB,),
        (state.ru_t, state.u0, ctx.wwI, state.c1h, state.c2h,
         ctx.mut_old, ctx.mut_new, state.rdnw, state.msfu, ctx.dt,
         DTYPE(1.0), DTYPE(0.0), DTYPE(0.5), np.int32(state.has_msf),
         np.int32(periodic), np.int32(f_lo), np.int32(f_hi),
         cp.ascontiguousarray(face), np.int32(old),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def solve_v(state, cfg, ctx: DynamicsSplit) -> None:
    """``advect_v_implicit`` on ``rv_t``."""
    nz, ny, nx = state.p.shape
    periodic = _periodic_y(cfg)
    f_lo, f_hi = (0, ny) if periodic else (1, ny - 1)
    nfaces = f_hi - f_lo + 1
    old = legacy(cfg)
    face = stage_face_masses(state, cfg, ctx.mut)[1] if old else ctx.mut
    _kernel(nz, "ieva_solve_v")(
        _blocks(nx * nfaces), (_TPB,),
        (state.rv_t, state.v0, ctx.wwI, state.c1h, state.c2h,
         ctx.mut_old, ctx.mut_new, state.rdnw, state.msfv, ctx.dt,
         DTYPE(1.0), DTYPE(0.0), DTYPE(0.5), np.int32(state.has_msf),
         np.int32(periodic), np.int32(f_lo), np.int32(f_hi),
         cp.ascontiguousarray(face), np.int32(old),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def solve_ph(state, cfg, ctx: DynamicsSplit) -> None:
    """``advect_ph_implicit`` on ``rph_t`` (after rhs_ph with wwE)."""
    nz, ny, nx = state.p.shape
    _kernel(nz, "ieva_solve_ph")(
        _blocks(ny * nx), (_TPB,),
        (state.rph_t, state.php0, state.phb, np.int32(state.phb.ndim == 3),
         ctx.wwI, state.c1f, state.c2f, ctx.mut, state.rdnw, state.msft,
         ctx.dt, DTYPE(1.0), DTYPE(0.0), DTYPE(0.5),
         np.int32(state.has_msf), np.int32(nz), np.int32(ny), np.int32(nx)))


def solve_w(state, cfg, ctx: DynamicsSplit) -> None:
    """``advect_w_implicit`` on ``rw_t`` (after the u, v and ph solves),
    with its two boundary terms in consistent units (A179, declared
    difference (4) above): the lower boundary uncouples ``ru_t``/``rv_t``
    with their map factors and the stage face masses ``stage_fluxes``
    coupled them with."""
    nz, ny, nx = state.p.shape
    rdx, rdy = _rdx_rdy(cfg)
    mux, muy = stage_face_masses(state, cfg, ctx.mut)
    _kernel(nz, "ieva_solve_w")(
        _blocks(ny * nx), (_TPB,),
        (state.rw_t, state.ru_t, state.rv_t,
         cp.ascontiguousarray(mux), cp.ascontiguousarray(muy),
         state.c1h, state.c2h, state.msfu, state.msfv, state.ht, ctx.wwI,
         state.php, state.php0, state.rph_t, state.w0,
         state.c1f, state.c2f,
         DTYPE(state.cf1), DTYPE(state.cf2), DTYPE(state.cf3),
         ctx.mut, ctx.mut_old, ctx.mut_new, state.rdn, state.msft,
         rdx, rdy, ctx.dt, DTYPE(c.G), DTYPE(1.0), DTYPE(0.0), DTYPE(0.5),
         np.int32(state.has_msf), np.int32(not _periodic_x(cfg)),
         np.int32(not _periodic_y(cfg)),
         np.int32(legacy(cfg)),
         np.int32(nz), np.int32(ny), np.int32(nx)))


class ScalarSplit(tuple):
    """Two flux arrays with the selected scalar column-mass convention."""

    def __new__(cls, wwE, wwI, variant="wrf_471"):
        result = super().__new__(cls, (wwE, wwI))
        result.variant = variant
        return result


def split_scalar_omega(state, cfg, ww_m, mut, dt: float):
    """``rk_scalar_tend``'s WW_SPLIT: the acoustic time-averaged ``ww_m``
    split with the time-n winds ``u_1``/``v_1`` and the post-acoustic
    column mass ``muts``. The earlier caller supplies the post-acoustic
    ``u_2``/``v_2`` instead, so that variant reads the current winds."""
    u, v = (state.u, state.v) if legacy(cfg) else (state.u0, state.v0)
    return ScalarSplit(*split_omega(
        state, cfg, ww_m, u, v, mut, dt),
        variant=getattr(cfg, "zadvect_implicit_variant", "wrf_471"))


def solve_scalar(state, tend, q_old, wwI, mu_old, mu_new, dt: float,
                 *, variant="wrf_471") -> None:
    """``advect_s_implicit`` on a transported scalar's advective tendency
    (``rk_scalar_tend``: ``mut_old = mub + mu_1``, ``mut = mut_new =
    muts``). The earlier variant uses ``muts`` in the old-field multiplier
    as well, matching its current-mass formulation."""
    nz, ny, nx = state.p.shape
    if variant == "wrf_legacy":
        mu_old = mu_new
    _kernel(nz, "ieva_solve_s")(
        _blocks(ny * nx), (_TPB,),
        (tend, q_old, q_old, np.int32(0), DTYPE(0.0), wwI, state.c1h,
         state.c2h,
         cp.ascontiguousarray(mu_old, dtype=DTYPE),
         cp.ascontiguousarray(mu_new, dtype=DTYPE),
         state.rdnw, DTYPE(dt), DTYPE(1.0), DTYPE(0.0),
         np.int32(nz), np.int32(ny), np.int32(nx)))
