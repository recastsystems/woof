"""WRF-ARW RK3 split-explicit time-stepping driver.

One ``step`` advances the state by ``cfg.dt`` with the three-stage
Runge-Kutta scheme of Wicker & Skamarock (ARW Tech Note sec. 3.1.2): each
stage recomputes the slow tendencies R^t* from the latest stage estimate
(the acoustic reference state t*), then re-integrates the *time-t* fields
forward with acoustic substeps:

  stage 1:  1 substep,             dtau = dt/3
  stage 2:  ns/2 substeps,         dtau = dt/ns
  stage 3:  ns substeps,           dtau = dt/ns

with ns = ``cfg.time_step_sound``.  The slow forcings are the flux-form
advection (vertical flux = the diagnosed eta mass flux Omega, WRF
``calc_ww_cp``), the horizontal pressure gradient of the t* state (WRF
``horizontal_pressure_gradient``), the buoyancy/vertical-pressure-gradient
term g*(d(p')/d(eta) - mu') (WRF ``pg_buoy_w``) and the advective-form
geopotential RHS with its g*w term (WRF ``rhs_ph``); the perturbation
pressure-gradient and buoyancy terms live inside the acoustic substeps
(woof.core.acoustic, Tech Note eqns 3.4-3.14).

``acoustic=False`` keeps the Phase-1 advection-only transport path (mu' and
phi' frozen) used by the pure-advection verification test.

Time-step rule (WRF guidance): ``dt <= ~6 s per km of dx``; benchmarks run
at CFL ~ 0.3-0.5.
"""

from __future__ import annotations

from functools import lru_cache
import math
import os

from woof.core.state import cp
from woof.forecast_gpu import require_forecast_gpu
import numpy as np

from woof.grid_requirements import FIFTH_ORDER_STENCIL_AXIS, boundary_axis
from woof.config import (RunConfig, constant_k_mixing_active,
                          validate_km_opt, wrf_mixing_package_active)
from woof.wrf_exact import (ENABLED as WRF_EXACT, DIAGNOSTICS_ENABLED,
                            BIGSTEP_ENABLED, DIFFUSION_ENABLED)
from woof.core import constants as c
from woof.core.diff6_edge_workspace import (
    DIFF6_EDGE_HALO, diff6_edge_slope,
    field_values as diff6_edge_field_values,
    phb_values as diff6_edge_phb_values,
    plane_shapes as diff6_edge_plane_shapes,
    planes_values as diff6_edge_planes_values)
from woof.core.state import (DTYPE, DomainState, mu_at_u_faces,
                              mu_at_v_faces)
# CUDA dependencies stay beside the CUDA path; importing step on a CPU-only
# install must reach the named runtime refusal rather than a CuPy import error.
if cp is not None:
    from woof.core.acoustic import (_mass_w_boundary_zone, wrf_acoustic_muts,
                                     prepare_acoustic_coefficients,
                                     prepare_acoustic_substep_launch,
                                     prepare_moist_cq)
    from woof.core.advection import (add_advection_tendencies,
                                      launch_flux_div_scalar, launch_flux_div_u,
                                      launch_flux_div_v, launch_flux_div_w,
                                      vertical_orders)
    from woof.core.diagnostics import update_diagnostics
    from woof.core import ieva
    from woof.core.bandwidth_glue import (
        capture_theta_forcing, prepare_add_arrays,
        total_theta as _glue_total_theta)
    from woof.core.ieva import stage_face_masses
    from woof.core.kernels import get_kernel, get_kernel_int_defines
    from woof.core.smag2d import scalar_index32_fits
    from woof.core.microphysics import apply as apply_microphysics
    from woof.core.moist import (SPECIES, WRF_MOIST_ARRAY_SPECIES,
                                  extra_moist_species, advance_scalars_stage)
    from woof.core.moist_n2_mutation import calc_n2_kernel
    from woof.core.physics import physics_enabled
    from woof.core import tke_budget
    from woof.core.uh_diag import update_up_heli_max
    from woof.ingest.lateral_bc import (apply_state_boundary_values,
                                         apply_state_lateral_boundaries,
                                         capture_folded_relaxation)

_TPB = 128  # threads per block along i (i fastest)

#: Prognostic fields saved to their *0 time-t copies at the start of a step.
_PROGNOSTICS = ("u", "v", "w", "thp", "php", "mup")

#: Slow-tendency accumulators zeroed at the start of every RK stage.
_TENDENCIES = ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t")


def _prepare_bookkeeping(state, pairs, *, zero=False):
    """Cache word-copy descriptors, refreshing when a state buffer changes."""
    if not pairs or len(pairs) > 64:
        return None
    arrays = tuple(a for pair in pairs for a in pair)
    if any(not isinstance(a, cp.ndarray) or a.dtype != np.dtype(np.float32)
           or not a.flags.c_contiguous for a in arrays):
        return None
    if any(src.shape != dst.shape for src, dst in pairs):
        return None
    key = tuple((a.data.ptr, a.size) for a in arrays)
    attr = '_rk_zero_launch' if zero else '_rk_copy_launch'
    cached = getattr(state, attr, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    # Cross-array overlap needs the original ordered assignments.
    ranges = [(a.data.ptr, a.data.ptr + a.nbytes) for a in arrays]
    for i, (_, dst) in enumerate(pairs):
        lo, hi = ranges[2 * i + 1]
        for j, (other_lo, other_hi) in enumerate(ranges):
            if j == 2 * i + 1 or (zero and j == 2 * i):
                continue
            if lo < other_hi and other_lo < hi:
                return None
    rows = np.zeros((64, 3), dtype=np.uint64)
    rows[:len(pairs)] = [(src.data.ptr, dst.data.ptr, dst.size)
                        for src, dst in pairs]
    # A by-value parameter avoids device metadata allocation and upload.
    # Keep a size-one ndarray: CuPy 14 bounds scalar arguments to its
    # small CScalar buffer, while ndarray arguments use the full host bytes.
    table = rows.reshape(-1).view(np.dtype(('V', rows.nbytes)))
    kernel = get_kernel('rk_bookkeeping',
                        'rk_zero_words' if zero else 'rk_copy_words')
    grid = (max(1, (max(dst.size for _, dst in pairs) + 4 * _TPB - 1)
                // (4 * _TPB)),
            len(pairs))
    block = (_TPB,)
    args = (table,)

    def launch():
        kernel(grid, block, args)

    # The table holds only pointers and sizes, all of which are in the key,
    # so a hit rebuilds nothing; no array is retained, because holding the
    # previous buffers would keep them alive across a relocation's swap.
    setattr(state, attr, (key, launch))
    return launch


def _save_time_t(state):
    names = list(_PROGNOSTICS)
    if state.qv is not None:
        names.extend(SPECIES)
    names.extend(extra_moist_species(state))
    if getattr(state, 'tke', None) is not None:
        names.append('tke')
    pairs = tuple((getattr(state, name), getattr(state, name + '0'))
                  for name in names) + _chem_time_pairs(state)
    launch = _prepare_bookkeeping(state, pairs)
    if launch is None:
        for src, dst in pairs:
            dst[...] = src
    else:
        launch()


def _prepare_tendency_zero(state):
    arrays = tuple(getattr(state, name) for name in _TENDENCIES)
    launch = _prepare_bookkeeping(state, tuple((a, a) for a in arrays),
                                  zero=True)
    if launch is not None:
        return launch

    def clear():
        for a in arrays:
            a[...] = 0
    return clear

#: WRF turbulent Prandtl number (share/module_model_constants.F:
#: prandtl = 1./3.0) -- scalars mix with K_h = K_m/prandtl = 3*K_m.
_PRANDTL = 1.0 / 3.0

#: km_opt=4 application kernels by field stagger (smag2d.cu).
_SMAG_HD = {"": "smag_hd_s", "x": "smag_hd_u",
            "y": "smag_hd_v", "z": "smag_hd_w"}

#: WRF v4.6.1 diff_opt=2 stress/scalar kernels used by production.  The
#: legacy ``_SMAG_HD`` entry points remain available only for the frozen flat
#: Phase-2 kernel-oracle tests.
_WRF_SMAG_HD = {"": "wrf_smag_hd_s", "x": "wrf_smag_hd_u",
                "y": "wrf_smag_hd_v", "z": "wrf_smag_hd_w"}


def _b3(arr: cp.ndarray) -> cp.ndarray:
    """Base profile broadcast: 1-D flat column -> (n, 1, 1); 3-D through."""
    return arr if arr.ndim == 3 else arr[:, None, None]


def _boundary_x(cfg: RunConfig) -> bool:
    return cfg.open_x or _boundary_forced(cfg)


def _boundary_y(cfg: RunConfig) -> bool:
    return cfg.open_y or _boundary_forced(cfg)


def _boundary_forced(cfg: RunConfig) -> bool:
    """WRF ``specified_bdy``: external specified OR nested forcing."""
    return bool(getattr(cfg, "specified", False)
                or getattr(cfg, "nested", False))


def _omega_ref(state: DomainState, cfg: RunConfig,
               ru: cp.ndarray, rv: cp.ndarray) -> cp.ndarray:
    """Reference eta mass flux Omega (nz+1, ny, nx) at w levels.

    WRF ``calc_ww_cp`` (dyn_em/module_big_step_utilities_em.F): integrate
    the continuity equation over the column for the (eta-uniform) d(mu)/dt,
    then diagnose Omega level by level from the coupled horizontal mass
    fluxes ``ru``/``rv``, weighting the column-mass tendency by the hybrid
    c1h.  Omega = 0 at the surface and at the model top: WRF sets
    ``ww(kte)`` to zero rather than trusting the column sum (which closes
    only to rounding, since sum(dnw*c1h) = -1), and so does this.

    One thread per column evaluates the routine in WRF's own arithmetic,
    every product and sum boundary where the Fortran has one:

    * the layer divergence is ``divv(k) = (msftx*dnw(k)) * (rdx*(ru(i+1) -
      ru(i)) + rdy*(rv(j+1) - rv(j)))``, the map factor and the layer
      thickness multiplied FIRST and that product applied to the bracket
      (Fortran evaluates ``msftx*dnw*(...)`` left to right);
    * the column total ``dmdt`` is the sequential sum of ``divv`` from the
      surface up, one rounding per layer, not a tree reduction;
    * the recurrence is ``ww(k) = (ww(k-1) - (dnw(k-1)*c1h(k-1))*dmdt) -
      divv(k-1)``: the level product, then the first subtraction, then the
      second, never a pre-added operand ``ww - (a + b)``.

    ``rdx``/``rdy`` are single-precision reciprocals (``rdx = 1./dx`` in
    solve_em), and multiplication commutes exactly, so the order of the two
    factors inside any one product is free; the boundaries are not.  Same
    discipline as :func:`_couple_momentum_kernel`: explicit round-to-nearest
    intrinsics plus ``-fmad=false``.  Without map factors the same kernel
    omits the factor, which is WRF with ``msftx = 1`` exactly (a multiply by
    one is exact).

    Map factors (Task 3): with ru/rv already msf-coupled, the ``msftx``
    weight makes ``ww`` the tech-note Omega = mu*deta/dt / m_y.

    Replaces the ufunc/``sum``/``cumsum`` construction: three full-size
    ufunc passes, a CuPy tree reduction and a CuPy batched scan whose
    summation orders belonged to the library (docs/public/DETERMINISM.md,
    "library-owned reduction order") and whose scan alone cost about 1.2 ms
    per call at 480x384x49 on a 5070 Ti; the per-column scan does the same
    work in about a tenth of that (GitHub issue #5, weiserhase).
    """
    nz, ny, nx = state.p.shape
    ww = state.scratch((nz + 1, ny, nx), "rk_ww")
    rdx = np.float32(1.0) / np.float32(cfg.dx)         # WRF: rdx = 1./dx
    rdy = np.float32(1.0) / np.float32(cfg.dy)
    args = [ru.reshape(-1), rv.reshape(-1), state.dnw, state.c1h]
    if state.has_msf:                                  # WRF calc_ww_cp msftx
        args.append(state.msft.reshape(-1))
    _omega_column_kernel(state.has_msf)(
        *args, rdx, rdy, np.int32(nz), np.int32(ny), np.int32(nx),
        ww.reshape(-1), size=ny * nx)
    return ww


@lru_cache(maxsize=None)
def _omega_column_kernel(has_msf: bool):
    """WRF ``calc_ww_cp`` for one column per thread; see :func:`_omega_ref`.

    Pass one walks the column upward forming ``divv(k)`` in WRF's grouping
    and accumulating ``dmdt`` sequentially; each layer's ``divv`` is parked
    in ``ww(k+1)``, the slot the recurrence reads it back from before
    overwriting it, so no second scratch and no second read of the fluxes.
    Pass two walks the recurrence with its two separate subtractions and
    writes ``ww(0) = 0`` and ``ww(nz) = 0`` explicitly.  Every parameter is
    ``raw`` (the launch size is the column count), the flux and map-factor
    views are contiguous, and index arithmetic is 32-bit for the same reason
    :func:`woof.core.moist._update_scalar_kernel` gives.
    """
    params = ["raw T ru", "raw T rv", "raw T dnw", "raw T c1h"]
    if has_msf:
        params.append("raw T msft")
    params += ["T rdx", "T rdy", "int32 nz", "int32 ny", "int32 nx"]
    weight = "__fmul_rn(msft[col], dnw[k])" if has_msf else "dnw[k]"
    body = [
        "const int col = static_cast<int>(i);",
        "const int ncol = ny * nx;",
        "const int jj = col / nx;",
        "const int ii = col - jj * nx;",
        "T dmdt = T(0);",
        "for (int k = 0; k < nz; ++k) {",
        "  const int u0 = (k * ny + jj) * (nx + 1) + ii;",
        "  const int v0 = (k * (ny + 1) + jj) * nx + ii;",
        "  const T du = __fsub_rn(ru[u0 + 1], ru[u0]);",
        "  const T dv = __fsub_rn(rv[v0 + nx], rv[v0]);",
        "  const T bracket = __fadd_rn(__fmul_rn(rdx, du), __fmul_rn(rdy, dv));",
        f"  const T divv = __fmul_rn({weight}, bracket);",
        "  dmdt = __fadd_rn(dmdt, divv);",
        "  ww[(k + 1) * ncol + col] = divv;",
        "}",
        "T w = T(0);",
        "ww[col] = w;",
        "for (int k = 1; k < nz; ++k) {",
        "  const int at = k * ncol + col;",
        "  const T divv = ww[at];",
        "  w = __fsub_rn(w, __fmul_rn(__fmul_rn(dnw[k - 1], c1h[k - 1]), dmdt));",
        "  w = __fsub_rn(w, divv);",
        "  ww[at] = w;",
        "}",
        "ww[nz * ncol + col] = T(0);",
    ]
    return cp.ElementwiseKernel(", ".join(params), "raw T ww",
                                "\n".join(body), "gpuwm_omega_column_scan",
                                options=("-fmad=false",))


@lru_cache(maxsize=None)
def _couple_momentum_kernel(has_msf: bool, reciprocal: bool = False):
    """WRF ``couple_momentum`` for one staggering, in a single pass.

    Replaces five full-size ufunc launches per component -- the c1h*muface
    multiply, the c2h add, the wind multiply, the copy into the flux scratch
    and the map-factor divide -- along with the two staggered temporaries
    they round through.  Same discipline as
    :func:`woof.core.moist._update_scalar_kernel`: explicit round-to-nearest
    intrinsics plus ``-fmad=false``, because the chain this replaces rounds
    to FP32 at every operator boundary, and the map-factor divide stays a
    separate division AFTER the multiply rather than folding into it.
    """
    params = ["T wind", "raw T c1h", "raw T c2h", "raw T muface"]
    body = ["const int lev = static_cast<int>(i) / ncol;",
            "const int col = static_cast<int>(i) % ncol;",
            "T v = __fmul_rn(__fadd_rn(__fmul_rn(c1h[lev], muface[col]), "
            "c2h[lev]), wind);"]
    if has_msf:
        params.append("raw T msf")
        if reciprocal:
            body.append("v = __fmul_rn(v, msf[col]);")
        else:
            body.append("v = __fdiv_rn(v, msf[col]);")
    params.append("int32 ncol")
    body.append("flux = v;")
    return cp.ElementwiseKernel(", ".join(params), "T flux", "\n".join(body),
                                "gpuwm_couple_momentum",
                                options=("-fmad=false",))


def stage_fluxes(state: DomainState, cfg: RunConfig, *, first_stage: bool = False
                 ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Public RK-stage transport surface (Task 5): ``(ru, rv, ww)``.

    The stage's coupled horizontal mass fluxes ``ru = (c1h*<mu>_x +
    c2h)*u/msfu`` / ``rv = (c1h*<mu>_y + c2h)*v/msfv`` (WRF
    ``couple_momentum``; the msf divisions are identity with the default
    map factors) and the diagnosed eta mass flux Omega ``ww`` (WRF
    ``calc_ww_cp``), all evaluated at the current stage reference t*.
    These are the fluxes that advect theta and momentum in
    ``_add_slow_tendencies``, exactly WRF ``rk_tendency``'s ru/rv/ww.
    The moisture scalars do NOT use them directly: WRF advects scalars
    with the acoustic-substep time-averaged fluxes ru_m/rv_m/ww_m
    (``sumflux``, "needed for consistent mass-conserving scalar
    advection"), which ``step`` accumulates over each stage's substeps as
    mean(u'') + these reference fluxes (solve_em.F:2210-2212).  This
    remains the single sanctioned source of Omega; nothing downstream may
    re-derive it.  Backed by the persistent scratch slots
    ``rk_ru``/``rk_rv``/``rk_ww``: the views stay valid until the next
    ``stage_fluxes`` call on the same state (the acoustic substeps do not
    touch them).
    """
    nz, ny, nx = state.p.shape
    mu = state.total_mu()                              # (ny, nx) t* mass
    # Open boundaries: the boundary-face column mass is the boundary CELL's
    # (WRF's muu/muv under the zero-gradient mu ghost copy), not the
    # periodic wrap average.  One rule, shared with the IEVA column mass
    # (woof.core.ieva.column_mass_new), which must read these same faces.
    mux, muy = stage_face_masses(state, cfg, mu)
    ru = state.scratch((nz, ny, nx + 1), "rk_ru")
    rv = state.scratch((nz, ny + 1, nx), "rk_rv")
    for wind, muface, msf, flux in ((state.u, mux, state.msfu, ru),
                                    (state.v, muy, state.msfv, rv)):
        kernel = _couple_momentum_kernel(
            state.has_msf, WRF_EXACT and wind is state.v)
        args = [wind, state.c1h, state.c2h, muface.reshape(-1)]
        if state.has_msf:                              # U = C(mu)*u/msfu,
            factor = state.msfv_inv if WRF_EXACT and wind is state.v else msf
            args.append(factor.reshape(-1))            # V uses WRF's stored inverse
        kernel(*args, np.int32(muface.size), flux)
    if WRF_EXACT:
        return ru, rv, _omega_ref(state, cfg, *_calc_ww_cp_fluxes(
            state, cfg, first_stage=first_stage))
    return ru, rv, _omega_ref(state, cfg, ru, rv)


#: Strict mode: the outermost ring of mu_2 as WRF's halo holds it at the next
#: step's start -- the acoustic loop's value, copied before spec_bdy_final
#: installs the boundary mass (restart class: carried, woof/io/restart.py).
WRF_MU_HALO_SLOT = "wrf_exact_mu_halo"


def _calc_ww_cp_fluxes(state: DomainState, cfg: RunConfig, *,
                       first_stage: bool = False):
    """Strict mode: the coupled fluxes WRF ``calc_ww_cp`` forms for itself.

    calc_ww_cp does not read ``ru``/``rv``: it builds its own face mass
    ``MUU = 0.5*(MUP(i)+MUB(i)+MUP(i-1)+MUB(i-1))`` (base and perturbation
    of one column first, unlike calc_mu_uv's ``MU(i)+MU(i-1)+MUB(i)+
    MUB(i-1)``), reading the halo column at a non-periodic edge, which WRF
    fills with the edge column, and couples ``(c1h*MUU+c2h)*u/msfuy`` and
    ``(c1h*MUV+c2h)*v*msfvx_inv`` (module_big_step_utilities_em.F).  The
    two face-mass orders agree on the time-t state of the round-3 pair and
    then part: Omega differed in 27748 of 81600 words at the second RK
    stage until it was formed this way.  Each CuPy operation below rounds
    once, in WRF's order.
    """
    nz, ny, nx = state.p.shape
    mup, mub = state.mup, state.mub2d

    def faces(n, periodic):
        idx = cp.arange(n + 1)
        if periodic:
            return idx % n, (idx - 1) % n
        return cp.minimum(idx, n - 1), cp.maximum(idx - 1, 0)

    right, left = faces(nx, not _boundary_x(cfg))
    mup_r, mup_l = mup[:, right], mup[:, left]
    north, south = faces(ny, not _boundary_y(cfg))
    mup_n, mup_s = mup[north], mup[south]
    halo = state.existing_scratch(WRF_MU_HALO_SLOT) if first_stage else None
    if halo is not None:
        # rk_step 1 reads the halo the previous step left: on the outermost
        # faces the ring mass before spec_bdy_final installed its table
        # value.  The two are usually the same word; at step 18 of the
        # round-3 pair they were not, and Omega differed in one column.
        if _boundary_x(cfg):
            mup_l[:, 0] = halo[:, 0]
            mup_r[:, nx] = halo[:, nx - 1]
        if _boundary_y(cfg):
            mup_s[0, :] = halo[0, :]
            mup_n[ny, :] = halo[ny - 1, :]
    muu = DTYPE(0.5) * (((mup_r + mub[:, right]) + mup_l) + mub[:, left])
    muv = DTYPE(0.5) * (((mup_n + mub[north]) + mup_s) + mub[south])
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    fu = state.scratch((nz, ny, nx + 1), "rk_ww_ru")
    fv = state.scratch((nz, ny + 1, nx), "rk_ww_rv")
    fu[...] = ((c1h * muu[None]) + c2h) * state.u
    fv[...] = ((c1h * muv[None]) + c2h) * state.v
    if state.has_msf:
        fu /= state.msfu[None]
        fv *= state.msfv_inv[None]
    return fu, fv


def domain_mass_measure(state: DomainState) -> float:
    """FP64 area-weighted domain dry-mass measure ``sum(mu/msft**2)``.

    The ONE measure both the flat and the mapped boundary-tendency
    branches close against, so a residual cannot be read in one unit and
    scored against the other.  With identity map factors the weight is
    exactly 1.0 and this is the plain column-mass sum.
    """
    return float(cp.sum(state.total_mu().astype(cp.float64)
                        * state.cell_area_weight(), dtype=cp.float64))


def _boundary_mass_tendency_flat(state: DomainState,
                                 cfg: RunConfig) -> cp.ndarray:
    """Unmapped telescoped boundary tendency as a 0-d FP64 device scalar."""
    boundary_x = _boundary_x(cfg)
    boundary_y = _boundary_y(cfg)
    tendency = cp.float64(0.0)
    if not boundary_x and not boundary_y:
        return tendency
    mu = state.total_mu()
    dnw = state.dnw[:, None]
    c1h = state.c1h[:, None]
    c2h = state.c2h[:, None]
    if boundary_x:
        west = (state.u_pp[:, :, 0]
                + (c1h * mu[:, 0][None] + c2h) * state.u[:, :, 0])
        east = (state.u_pp[:, :, -1]
                + (c1h * mu[:, -1][None] + c2h) * state.u[:, :, -1])
        tendency += cp.sum(
            (dnw * DTYPE(1.0 / cfg.dx) * (east - west)).astype(cp.float64),
            dtype=cp.float64)
    if boundary_y:
        south = (state.v_pp[:, 0, :]
                 + (c1h * mu[0, :][None] + c2h) * state.v[:, 0, :])
        north = (state.v_pp[:, -1, :]
                 + (c1h * mu[-1, :][None] + c2h) * state.v[:, -1, :])
        tendency += cp.sum(
            (dnw * DTYPE(1.0 / cfg.dy) * (north - south)).astype(cp.float64),
            dtype=cp.float64)
    return tendency


def _boundary_mass_tendency_mapped(state: DomainState,
                                   cfg: RunConfig) -> cp.ndarray:
    """Mapped telescoped boundary tendency as a 0-d FP64 device scalar.

    Derived from the in-tree mapped acoustic kernel rather than from a
    textbook form.  ``advance_mu_th_msf`` advances the column mass by

        d(mu_c)/dt = m_c**2 * sum_k dnw_k * (rdx*dFx + rdy*dFy) + rmu_c

    with ``m_c**2 = msft*msft`` and the total face flux
    ``F = u'' + (c1h*mu_face + c2h)*u/msfu`` (the reference momentum
    already carries its FACE map factor; ``u''`` carries its own from
    ``small_step_prep``).  Dividing the cell equation by ``m_c**2`` -- the
    weight :meth:`DomainState.cell_area_weight` applies -- removes the map
    factor from the flux term entirely, so the divergence telescopes over
    the domain and only the outermost faces survive.  The ``rmu_t``
    forcing and the specified-zone reset are deliberately NOT folded in
    here: they are separate budget terms, and a flux-only "closure" on a
    forced domain would be a false receipt.

    The mapped face factor is the only arithmetic difference from the flat
    branch, which is why the reduction control on an identity-map state is
    exact rather than approximate.
    """
    boundary_x = _boundary_x(cfg)
    boundary_y = _boundary_y(cfg)
    tendency = cp.float64(0.0)
    if not boundary_x and not boundary_y:
        return tendency
    mu = state.total_mu()
    dnw = state.dnw[:, None]
    c1h = state.c1h[:, None]
    c2h = state.c2h[:, None]
    if boundary_x:
        west = (state.u_pp[:, :, 0]
                + (c1h * mu[:, 0][None] + c2h) * state.u[:, :, 0]
                / state.msfu[:, 0][None])
        east = (state.u_pp[:, :, -1]
                + (c1h * mu[:, -1][None] + c2h) * state.u[:, :, -1]
                / state.msfu[:, -1][None])
        tendency += cp.sum(
            (dnw * DTYPE(1.0 / cfg.dx) * (east - west)).astype(cp.float64),
            dtype=cp.float64)
    if boundary_y:
        south = (state.v_pp[:, 0, :]
                 + (c1h * mu[0, :][None] + c2h) * state.v[:, 0, :]
                 / state.msfv[0, :][None])
        north = (state.v_pp[:, -1, :]
                 + (c1h * mu[-1, :][None] + c2h) * state.v[:, -1, :]
                 / state.msfv[-1, :][None])
        tendency += cp.sum(
            (dnw * DTYPE(1.0 / cfg.dy) * (north - south)).astype(cp.float64),
            dtype=cp.float64)
    return tendency


def _boundary_mass_tendency_forced(state: DomainState,
                                   cfg: RunConfig) -> cp.ndarray:
    """Telescoped flux tendency of a specified or nested grid's active cells.

    On a boundary-forced grid ``advance_mu_th`` and ``advance_mu_th_msf``
    divergence-update only the cells inside the frame
    :func:`woof.core.acoustic._mass_w_boundary_zone` names (WRF's
    ``ids+1..ide-2``); the frame row itself advances by ``rmu_t`` alone.
    The divergence of the active cells therefore telescopes onto the
    faces between the frame and the first active row, not onto the
    domain's outer faces, and the face mass is the mean of the frame cell
    and its active neighbour, as the kernel forms it.  Summing the outer
    faces instead counted fluxes no cell ever receives: a real 720-step
    specified forecast then reported a 4 percent dry-mass residual that
    the model never had.  The frame's ``rmu_t`` and the specified-zone
    reset stay separate budget terms, as for every forced domain.
    """
    z = _mass_w_boundary_zone(cfg)
    ny, nx = state.mup.shape
    rows, cols = slice(z, ny - z), slice(z, nx - z)
    mu = state.total_mu()
    dnw = state.dnw[:, None]
    c1h = state.c1h[:, None]
    c2h = state.c2h[:, None]
    half = DTYPE(0.5)
    faces = []
    for wind, perturbation, mass_a, mass_b, factor in (
            (state.u[:, rows, z], state.u_pp[:, rows, z],
             mu[rows, z - 1], mu[rows, z],
             state.msfu[rows, z] if state.has_msf else None),
            (state.u[:, rows, nx - z], state.u_pp[:, rows, nx - z],
             mu[rows, nx - z - 1], mu[rows, nx - z],
             state.msfu[rows, nx - z] if state.has_msf else None),
            (state.v[:, z, cols], state.v_pp[:, z, cols],
             mu[z - 1, cols], mu[z, cols],
             state.msfv[z, cols] if state.has_msf else None),
            (state.v[:, ny - z, cols], state.v_pp[:, ny - z, cols],
             mu[ny - z - 1, cols], mu[ny - z, cols],
             state.msfv[ny - z, cols] if state.has_msf else None)):
        reference = (c1h * (half * (mass_a + mass_b))[None] + c2h) * wind
        if factor is not None:
            reference = reference / factor[None]
        faces.append(perturbation + reference)
    west, east, south, north = faces
    return (cp.sum((dnw * DTYPE(1.0 / cfg.dx) * (east - west))
                   .astype(cp.float64), dtype=cp.float64)
            + cp.sum((dnw * DTYPE(1.0 / cfg.dy) * (north - south))
                     .astype(cp.float64), dtype=cp.float64))


def boundary_mass_tendency_device(state: DomainState,
                                  cfg: RunConfig) -> cp.ndarray:
    """0-d FP64 device scalar form of :func:`boundary_mass_tendency`.

    Keeping the device scalar unread is what lets the accumulator observer
    run without a per-substep host synchronization.
    """
    if _mass_w_boundary_zone(cfg):
        return _boundary_mass_tendency_forced(state, cfg)
    if state.has_msf:
        return _boundary_mass_tendency_mapped(state, cfg)
    return _boundary_mass_tendency_flat(state, cfg)


def boundary_mass_tendency(state: DomainState, cfg: RunConfig) -> float:
    """FP64 domain-sum dry-mass tendency through the lateral boundary.

    This is the telescoped boundary form of WRF ``advance_mu_t`` evaluated
    after ``advance_uv`` has updated the acoustic perturbation momenta.  The
    total face flux is ``u_pp + (c1h*mu_face+c2h)*u/msfu`` (and analogously
    for v); multiplying its opposing-face difference by ``dnw/dx`` or
    ``dnw/dy`` gives exactly the domain sum of the column-mass equation for
    the measure :func:`domain_mass_measure`.  Internal ``rmu_t`` sources
    are intentionally excluded so a closure residual detects them.  Mapped
    domains take the ARW cell-area weighting
    (:meth:`DomainState.cell_area_weight`); the flat, map-factor-one
    branch keeps the WK82 arithmetic unchanged.  Open grids telescope to
    their outer faces; specified and nested grids to the inner faces of
    the frame the acoustic mass update leaves to ``rmu_t``
    (:func:`_boundary_mass_tendency_forced`).

    Reading the result synchronizes the device; call
    :func:`boundary_mass_tendency_device` from a hot loop instead.
    """
    return float(boundary_mass_tendency_device(state, cfg))


class MassFluxAccumulator:
    """Device-resident FP64 running sum of the substep boundary increments.

    The list-appending ``mass_flux_observer`` reads one device scalar per
    acoustic substep, which is a host synchronization inside the
    innermost loop.  This accumulator adds the same FP64 increments in the
    same order on the device and is read once, at the end of the run, so a
    receipt-enabled real-case integration pays no per-substep sync.  The
    two are mutually exclusive keywords on :func:`step` precisely because
    running both would double-count nothing but would reintroduce the sync
    the accumulator exists to remove.
    """

    __slots__ = ("_total", "count")

    def __init__(self) -> None:
        self._total = cp.zeros((), dtype=cp.float64)
        #: number of accumulated substep increments
        self.count = 0

    def add(self, increment) -> None:
        """Accumulate one FP64 increment without reading it."""
        self._total += increment
        self.count += 1

    def total(self) -> float:
        """Host FP64 total; the one synchronization this observer takes."""
        return float(self._total)

    def reset(self) -> None:
        self._total = cp.zeros((), dtype=cp.float64)
        self.count = 0


def _launch_slow_pgf(state: DomainState, cfg: RunConfig, *, cq=None) -> None:
    """Subtract WRF's moist-cq-scaled large-step horizontal PGF."""
    nz, ny, nx = state.p.shape
    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    cqu, cqv, _cqw, use_cq = cq
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    kernel = get_kernel("dycore", "slow_pgf")
    n = nz * (ny + 1) * (nx + 1)
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (state.ru_t, state.rv_t,
            state.p_perturbation if DIAGNOSTICS_ENABLED else state.p,
            state.pb, state.al, state.alt,
            state.php, state.phb, state.mup, state.mub2d,
            state.c1h, state.c2h, state.rdnw, state.fnm, state.fnp,
            state.cf1, state.cf2, state.cf3,
            np.int32(cfg.top_lid), state.cfn, state.cfn1,
            cqu, cqv, np.int32(use_cq),
            DTYPE(rdx), DTYPE(rdy), DTYPE(0.5 * rdx), DTYPE(0.5 * rdy),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
            np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _launch_slow_buoyancy(state: DomainState, cfg: RunConfig) -> None:
    """Add the fused vertical pressure-gradient and moist buoyancy term."""
    nz, ny, nx = state.p.shape
    dummy = state.p
    if state.qv is None:
        moist_mode = 0
        qv = qc = qr = qi = qs = qg = qh = dummy
    else:
        if getattr(state, "qh", None) is not None:
            moist_mode = 3
        else:
            moist_mode = 2 if getattr(state, "qi", None) is not None else 1
        qv, qc, qr = state.qv, state.qc, state.qr
        if moist_mode >= 2:
            qi = state.qi
            # P3 (mp=50) is the one scheme with qi and NO qs/qg, and
            # q_total's modes are 0/1/2/3 with no "one ice mass" arm.  The
            # absent pair takes the shared zero plane rather than reopening
            # a frozen kernel; moist.absent_mass_plane argues the identity.
            if getattr(state, "qg", None) is None:
                from woof.core.moist import absent_mass_plane
                zero = absent_mass_plane(state)
                qs = state.qs if getattr(state, "qs", None) is not None else zero
                qg = zero
            else:
                qs, qg = state.qs, state.qg
        else:
            qi = qs = qg = dummy
        qh = state.qh if moist_mode == 3 else dummy
    kernel = get_kernel("dycore", "slow_buoyancy")
    n = nz * ny * nx
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (state.rw_t,
            state.p_perturbation if DIAGNOSTICS_ENABLED else state.p,
            state.pb, state.mup, state.mub2d,
            qv, qc, qr, qi, qs, qg, qh, state.rdn, state.rdnw,
            state.c1f, state.c2f,
            state.msft, np.int32(moist_mode), np.int32(state.has_msf),
            np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _validate_geopotential_config(cfg: RunConfig, nx: int, ny: int) -> None:
    """Validate the horizontal geopotential-advection stencil."""
    if cfg.h_sca_adv_order not in (2, 5):
        raise ValueError(
            f"h_sca_adv_order must be 2 or 5, got {cfg.h_sca_adv_order}")
    if cfg.h_sca_adv_order == 5:
        if nx < FIFTH_ORDER_STENCIL_AXIS or ny < FIFTH_ORDER_STENCIL_AXIS:
            raise ValueError(
                f"h_sca_adv_order=5 needs nx, ny >= 7 (7-point stencil), "
                f"got {nx} x {ny}")


def _launch_slow_geopotential(state: DomainState, cfg: RunConfig,
                              ww: cp.ndarray, *, add_vertical: bool) -> None:
    """Apply fused vertical/g*w and horizontal geopotential RHS terms."""
    nz, ny, nx = state.p.shape
    _validate_geopotential_config(cfg, nx, ny)
    if (cfg.open_x or cfg.open_y) and not _boundary_forced(cfg):
        _launch_open_geopotential(state, cfg, ww, add_vertical=add_vertical)
        return
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    kernel = get_kernel("dycore", "slow_geopotential")
    n = nz * ny * nx
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (state.rph_t, ww, state.w, state.u, state.v, state.php, state.phb,
            state.mup, state.mub2d, state.rdnw, state.fnm, state.fnp,
            state.c1f, state.c2f, state.cfn, state.cfn1,
            state.msft, state.msfu, state.msfv,
            DTYPE(0.25 * rdx), DTYPE(0.25 * rdy),
            np.int32(state.has_msf), np.int32(_boundary_x(cfg)),
            np.int32(_boundary_y(cfg)), np.int32(_boundary_forced(cfg)),
            np.int32(cfg.h_sca_adv_order), np.int32(add_vertical),
            np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _launch_slow_geopotential_faces(state: DomainState, cfg: RunConfig,
                                    mux: cp.ndarray,
                                    muy: cp.ndarray) -> None:
    """Apply horizontal geopotential advection with supplied face masses."""
    nz, ny, nx = state.p.shape
    _validate_geopotential_config(cfg, nx, ny)
    if (cfg.open_x or cfg.open_y) and not _boundary_forced(cfg):
        _launch_open_geopotential(state, cfg, state.w, add_vertical=False, mux=mux, muy=muy)
        return
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    kernel = get_kernel("dycore", "slow_geopotential_faces")
    n = nz * ny * nx
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
            (state.rph_t, state.u, state.v, state.php, state.phb,
            mux, muy, state.c1f, state.c2f,
            state.cfn, state.cfn1,
            state.msft, state.msfu, state.msfv,
            DTYPE(0.25 * rdx), DTYPE(0.25 * rdy),
            np.int32(state.has_msf), np.int32(_boundary_x(cfg)),
            np.int32(_boundary_y(cfg)), np.int32(_boundary_forced(cfg)),
            np.int32(cfg.h_sca_adv_order), np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _launch_open_geopotential(state, cfg, ww, *, add_vertical, mux=None, muy=None):
    nz, ny, nx = state.p.shape
    n = nz * ny * nx
    supplied = mux is not None
    get_kernel("dycore", "slow_geopotential_open")(((n + 255) // 256,), (256,),
        (state.rph_t, ww, state.w, state.u, state.v, state.php, state.phb,
         state.mup, state.mub2d, state.rdnw, state.fnm, state.fnp,
         state.c1f, state.c2f, state.cfn, state.cfn1, state.msft, state.msfu, state.msfv,
         mux if supplied else state.mup, muy if supplied else state.mup,
         DTYPE(0.25 / cfg.dx), DTYPE(0.25 / cfg.dy), np.int32(state.has_msf),
         np.int32(cfg.open_x), np.int32(cfg.open_y), np.int32(supplied),
         np.int32(add_vertical), np.int32(cfg.h_sca_adv_order), np.int32(state.phb.ndim == 3),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def _launch_slow_geopotential_vertical(state: DomainState,
                                       ww: cp.ndarray) -> None:
    """Apply only the vertical Omega and g*w terms on a failing config."""
    nz, ny, nx = state.p.shape
    kernel = get_kernel("dycore", "slow_geopotential_vertical")
    n = nz * ny * nx
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (state.rph_t, ww, state.w, state.php, state.phb,
            state.mup, state.mub2d, state.rdnw, state.fnm, state.fnp,
            state.c1f, state.c2f, state.msft,
            np.int32(state.has_msf), np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _add_slow_tendencies(state: DomainState, cfg: RunConfig,
                         ru: cp.ndarray, rv: cp.ndarray,
                         ww: cp.ndarray, *, cq=None, implicit=None) -> None:
    """Accumulate the RK stage forcings R^t* into the coupled tendencies.

    General hybrid/terrain reduction of WRF ``rk_tendency``: flux-form
    advection of u/v/w/theta with the stage transport fluxes from
    :func:`stage_fluxes` (diagnosed Omega as vertical flux, c1/c2-weighted
    coupled momenta), plus the t*-state pressure-gradient/buoyancy terms
    that force the acoustic system, including the alpha'*d(pb)/dx term,
    which survives over terrain where the base pressure varies on eta
    surfaces.  Moist states use WRF's full ``pg_buoy_w`` form for the w
    buoyancy (vapor + hydrometeor loading).  rmu_t stays zero: the
    mass-divergence part of R_mu is computed inside the acoustic
    ``advance_mu_th`` kernel.

    Map factors + Coriolis (Task 3): the working tendencies follow WRF's
    conventions exactly, ru_t/rv_t/rw_t force the msf-coupled momenta
    U = C(mu)u/msfu, V = C(mu)v/msfv, W = C_f(mu)w/msft and rth_t/rph_t
    carry an extra 1/msft (their updates multiply it back), and the
    Coriolis+curvature kernel joins the slot when rotation is enabled.
    With the default map factors every msf branch is skipped and the step
    is bitwise Phase 2 (regression-pinned).

    ``implicit`` (a :class:`woof.core.ieva.DynamicsSplit`, the last RK
    substep with ``zadvect_implicit = 1``) runs WRF's IEVA order instead:
    see :func:`_add_slow_tendencies_ieva`.
    """
    if implicit is not None:
        _add_slow_tendencies_ieva(state, cfg, ru, rv, ww, implicit, cq=cq)
        return
    if WRF_EXACT:
        theta_transport = state.thp
    elif type(state) is DomainState:
        theta_transport = _glue_total_theta(state)
    else:
        theta_transport = state.total_theta()
    vsca, vmom = vertical_orders(cfg)
    launch_flux_div_scalar(theta_transport, ru, rv, ww, state.rth_t,
                           state, cfg.dx, cfg.dy,
                           open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                           msf=state.msft, has_msf=state.has_msf,
                           spec=_boundary_forced(cfg), vorder=vsca)
    launch_flux_div_u(state.u, ru, rv, ww, state.ru_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msfu, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg), vorder=vmom)
    launch_flux_div_v(state.v, ru, rv, ww, state.rv_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msfv, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg), vorder=vmom)
    launch_flux_div_w(state.w, ru, rv, ww, state.rw_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msft, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg), vorder=vsca)

    # Fused WRF horizontal_pressure_gradient.  dycore.cu retains every
    # former eager-CuPy FP32 operator boundary explicitly.
    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    _launch_slow_pgf(state, cfg, cq=cq)

    # Fused WRF pg_buoy_w dry/moist vertical forcing.
    _launch_slow_buoyancy(state, cfg)

    # Preserve rhs_ph's historical failure sequencing: it applied vertical
    # Omega and g*w before rejecting an invalid horizontal stencil.  Valid
    # configurations stay on the single combined hot-path launch.
    try:
        _validate_geopotential_config(cfg, state.p.shape[2], state.p.shape[1])
    except (ValueError, NotImplementedError):
        _launch_slow_geopotential_vertical(state, ww)
        raise

    # Fused WRF rhs_ph vertical, g*w, and horizontal-advection terms.
    _launch_slow_geopotential(state, cfg, ww, add_vertical=True)

    if WRF_EXACT:
        # rk_tendency calls w_damp right after pg_buoy_w and before
        # coriolis/curvature (module_em.F); the default path applies it
        # after the held tendencies, another rounding of rw_tend.
        apply_w_damping(state, cfg, ww)

    # --- Coriolis + curvature (Task 3; WRF rk_tendency's coriolis and
    # curvature calls, kernels/coriolis_map.cu): no-op unless rotation is
    # enabled (set_map_coriolis with nonzero f/e or non-uniform msf).
    if state.rotational:
        add_coriolis_curvature(state, cfg, ru, rv)


def _add_slow_tendencies_ieva(state: DomainState, cfg: RunConfig,
                              ru: cp.ndarray, rv: cp.ndarray,
                              ww: cp.ndarray, ctx, *, cq=None) -> None:
    """WRF ``rk_tendency`` on an IEVA substep (module_em.F:436-745).

    The explicit operators advect with the explicit share of the eta mass
    flux ``ctx.wwE``; then each advective tendency is replaced by its
    column solve against the implicit share ``ctx.wwI``, in WRF's order:
    u and v, theta, ``rhs_ph`` (its vertical term on wwE) and the
    geopotential solve, then w, whose lower boundary reads the u/v
    tendencies after their solves and whose upper boundary reads the
    geopotential tendency after its solve.  Only then do the pressure
    gradient and buoyancy join ru_t/rv_t/rw_t, as they do in WRF, so the
    solves see advection alone.

    Theta is advected and solved as WRF's ``t_2 = theta - t0``; the t0
    constant then rejoins ``rth_t`` with the FULL flux ``ww``, the share
    of the default path's full-theta tendency the acoustic mass update
    balances (:mod:`woof.core.ieva`, THETA).  Every other tendency array
    receives the same sequence of adds it does on the default path.
    """
    wwE = ctx.wwE
    # The explicit share runs WRF's vert_order ladder (the fork's
    # zadvect_implicit = 1 sends only the explicit share through the
    # vert_order fluxes; the implicit share is the first-order solve).
    vsca, vmom = vertical_orders(cfg)

    def theta_flux(field, ru_, rv_, w_, tend):
        launch_flux_div_scalar(field, ru_, rv_, w_, tend,
                               state, cfg.dx, cfg.dy,
                               open_x=_boundary_x(cfg),
                               open_y=_boundary_y(cfg),
                               msf=state.msft, has_msf=state.has_msf,
                               spec=_boundary_forced(cfg), vorder=vsca)

    theta_t = ieva.theta_minus_t0(state)
    theta_flux(theta_t, ru, rv, wwE, state.rth_t)
    launch_flux_div_u(state.u, ru, rv, wwE, state.ru_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msfu, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg), vorder=vmom)
    launch_flux_div_v(state.v, ru, rv, wwE, state.rv_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msfv, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg), vorder=vmom)
    launch_flux_div_w(state.w, ru, rv, wwE, state.rw_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msft, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg), vorder=vsca)
    ieva.solve_u(state, cfg, ctx)
    ieva.solve_v(state, cfg, ctx)
    ieva.solve_theta(state, cfg, ctx)
    if not WRF_EXACT:
        # The default acoustic step transports FULL theta, so the t0
        # share rejoins rth_t here.  Strict arithmetic transports WRF's
        # t_2 = theta - t0 in the acoustic step (wrf_advance_mu_theta reads
        # thp with thb = t0), exactly as the non-IEVA strict branch of
        # _add_slow_tendencies advects thp alone: adding t0 * div(mass
        # flux) there forced every column by 300 K times its mass-flux
        # divergence (stock-WRF door, 20 s step: theta 0.080 K and MU
        # 16 Pa off after one step, non-finite by step 9-11).
        ieva.add_theta_offset_flux(state, cfg, theta_t, ru, rv, ww,
                                   theta_flux)
    del theta_t

    try:
        _validate_geopotential_config(cfg, state.p.shape[2], state.p.shape[1])
    except (ValueError, NotImplementedError):
        _launch_slow_geopotential_vertical(state, wwE)
        raise
    _launch_slow_geopotential(state, cfg, wwE, add_vertical=True)
    ieva.solve_ph(state, cfg, ctx)
    ieva.solve_w(state, cfg, ctx)

    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    _launch_slow_pgf(state, cfg, cq=cq)
    _launch_slow_buoyancy(state, cfg)
    if state.rotational:
        add_coriolis_curvature(state, cfg, ru, rv)


def capture_advective_theta_forcing(state: DomainState, *, cfg=None) -> None:
    """EXPORT the stage's pure advective theta rate as WRF ``RTHFTEN``.

    The exact inverse of :func:`add_h_diabatic_tendency`'s coupling:
    ``rth_t`` forces the WRF-coupled theta and carries an extra ``1/msfty``
    (see :func:`_add_slow_tendencies`), so the uncoupled K s-1 rate a
    cumulus scheme wants is ``rth_t * msfty / (c1h*mut + c2h)`` with the
    stage's own total dry mass ``mut``.  Same units, same one-step lag and
    the same producer/consumer split as ``h_diabatic``.

    CALL SITE IS THE CONTRACT.  This must run in the window after
    :func:`_add_slow_tendencies` returns -- where ``rth_t`` holds the flux
    divergence of theta and nothing else -- and before
    ``physics_tendencies.add_to_slow``, :func:`add_h_diabatic_tendency`,
    :func:`add_fixed_dry_tendencies` and the lateral-boundary fold touch
    it.  WRF's ``module_cumulus_driver.F:867`` pre-folds
    ``RTHRATEN + RTHBLTEN`` into ``RTHFTEN`` for G3SCHEME and
    NTIEDTKESCHEME and NOT for GFSCHEME, which sums the lanes itself
    (``woof/core/kernels/gf.cu:4428``), so an export taken one line later
    makes the scheme integrate the boundary layer and the radiation twice.
    ``tests/test_dycore_advective_forcing_export.py`` is the gate.

    WHAT THE NUMBER IS, precisely, because "advective tendency" is
    ambiguous and the difference is measurable: this is the flux-form
    TRANSPORT tendency of the coupled scalar divided by the stage dry
    mass, which is the quantity ``rth_t`` carries and the quantity the
    h_diabatic coupling inverts.  It differs from the material derivative
    ``-v.grad(theta)`` by the mass-divergence term
    ``theta*(dmu/dt)/mu`` -- order 1e-3 K s-1 against a measured 3e-3
    K s-1 rms on a 12 km CONUS domain, so it is a stated part of the
    export rather than a rounding detail.  Both halves of the pair use
    the same construction and the same reference mass, so the theta and
    qv rates GF sums are consistent with each other.

    A no-op on a state with no advective-forcing consumer, where the
    buffers are ``None``.
    """
    if getattr(state, "rthften", None) is None:
        return
    if WRF_EXACT and getattr(cfg, "cu_physics", 0) == 3:
        # set_tend saves the last RK transport in coupled units. GF's
        # next phy_prep_part2 decouples it with the carried acoustic mass.
        state.rthften[...] = (state.rth_t * state.msft[None]
                             if state.has_msf else state.rth_t)
        return
    if type(state) is DomainState and capture_theta_forcing(state):
        return
    rate = state.rth_t / (state.c1h[:, None, None] * state.total_mu()[None]
                          + state.c2h[:, None, None])
    if state.has_msf:                                  # WRF: the /msfty
        rate *= state.msft[None]                       # rth_t carries
    state.rthften[...] = rate


def add_h_diabatic_tendency(state: DomainState) -> None:
    """Fold held heating with every eager FP32 operation boundary retained."""
    if type(state) is not DomainState:
        _add_h_diabatic_tendency_eager(state)
        return
    arrays = (state.rth_t, state.h_diabatic, state.mub2d, state.mup,
              state.c1h, state.c2h, state.msft)
    if any(not isinstance(a, cp.ndarray) or a.dtype != np.dtype(np.float32)
           or not a.flags.c_contiguous for a in arrays):
        _add_h_diabatic_tendency_eager(state)
        return
    nz, ny, nx = state.rth_t.shape
    if (state.h_diabatic.shape != state.rth_t.shape
            or state.rth_t.size == 0
            or state.mub2d.shape != (ny, nx) or state.mup.shape != (ny, nx)
            or state.msft.shape != (ny, nx)
            or state.c1h.shape != (nz,) or state.c2h.shape != (nz,)):
        _add_h_diabatic_tendency_eager(state)
        return
    lo, hi = state.rth_t.data.ptr, state.rth_t.data.ptr + state.rth_t.nbytes
    if any(lo < a.data.ptr + a.nbytes and a.data.ptr < hi for a in arrays[1:]):
        _add_h_diabatic_tendency_eager(state)
        return
    kernel = get_kernel('held_heating', 'add_held_heating')
    kernel(((state.rth_t.size + _TPB - 1) // _TPB,), (_TPB,),
           (*arrays, np.uint64(state.rth_t.size), np.int32(ny * nx),
            np.int32(state.has_msf)))


def _add_h_diabatic_tendency_eager(state: DomainState) -> None:
    """ADD the retained microphysics heating to the coupled theta tendency.

    WRF ``rk_addtend_dry`` (module_em.F:1076-1080): ``t_tend = t_tend +
    ... + (c1(k)*mut + c2(k))*h_diabatic(i,k,j)/msfty``, executed on EVERY
    RK step with the step's total dry mass ``mut = mub + mu_2``
    (rk_step_prep's ``CALL calculate_full``, module_em.F:143 /
    solve_em.F:652-666, woof's ``total_mu()`` at the stage start).
    ``h_diabatic`` (K/s, uncoupled)
    is the PREVIOUS step's microphysics theta increment per second
    (microphysics.moist_physics_finish); the acoustic substeps integrate
    this tendency (advance_mu_t, module_small_step_em.F:1142), and
    ``_finish_small_steps`` removes the final stage's accumulated
    contribution so the state is heated exactly once, by the direct
    microphysics update.  Float64 mirror:
    ``woof.verify.npref.np_h_diabatic_tendency``.
    """
    hd = (state.c1h[:, None, None] * state.total_mu()[None]
          + state.c2h[:, None, None]) * state.h_diabatic
    if state.has_msf:                                  # WRF: /msfty
        hd /= state.msft[None]
    state.rth_t += hd


def add_rhs_ph_hadv(state: DomainState, cfg: RunConfig,
                    mux: cp.ndarray, muy: cp.ndarray) -> None:
    """SUBTRACT the rhs_ph horizontal phi advection from ``state.rph_t``.

    WRF ``rhs_ph`` advects the FULL geopotential (ph + phb) in advective
    form with the ``h_sca_adv_order`` stencil
    (module_big_step_utilities_em.F:1435 ``advective_order =
    config_flags%h_sca_adv_order``); the advecting face momenta are the
    (c1f*muuf+c2f)-coupled two-half-level sums ``(u(k)+u(k-1))*msfux``
    and the whole term carries 1/msfty at the mass point.  ``mux``/``muy``
    are the t* face masses (WRF muuf/muvf).  Interior full levels 1..nz-1
    only: the k = kte row belongs to the documented rigid-lid deviation.

    ``cfg.h_sca_adv_order == 2`` uses WRF's two-face form (:1516-1584).
    Open and specified outer rows skip the entire normal-direction term,
    including the interior face, as the compiled WRF oracle verifies.

    ``cfg.h_sca_adv_order == 5`` is the reference configuration (Registry
    default, unset in the reference namelist): WRF's <=6 branch
    1/60-weighted 7-point centered stencil on ph and phb
    (:1786-1795 y / :1949-1959 x) applied everywhere when periodic, and
    with WRF's specified-BC narrowing otherwise:

      y rows (mass, 0-based):  0, ny-1 nothing; 1, ny-2 2nd order
        (:1882-1906); 2, ny-3 4th order 1/12 (:1819-1850, gated
        ``open_ys .or. specified``); [3, ny-4] 5th/6th interior
        (:1780-1781 narrowing);
      x cols:  0, nx-1 nothing; 1, nx-2 2nd order (:2023-2048);
        2, nx-3 NOTHING, WRF's 4th-order x pickups are gated on
        ``open_xs``/``open_xe`` ONLY (:1973, :1997), so a specified run
        skips those columns entirely (binding v4.6.1 quirk, transcribed
        as-is); [3, nx-4] 5th/6th interior (:1937-1938).

    Radiative open boundaries with order 5 are not wired (they need WRF's
    boundary-row ph_old upwind terms, :2085-2178); config validation and
    this function both refuse the combination.

    The VERTICAL term of ``rhs_ph`` is independent of every advection
    order: WRF forms ``wdwn = 0.5*(ww(k)+ww(k-1))*rdnw(k-1)*(d(ph+phb))``
    and subtracts ``fnm(k)*wdwn(k+1)+fnp(k)*wdwn(k)`` with no
    ``vert_order`` branch (HRRR fork module_big_step_utilities_em.F
    rhs_ph, ``advective_order = config_flags%h_sca_adv_order`` is the
    only order it reads), so ``v_sca_adv_order`` does not reach it.

    Float64 mirror: ``woof.verify.npref.np_rhs_ph_hadv``.
    """
    _launch_slow_geopotential_faces(state, cfg, mux, muy)


def launch_coriolis_curvature(ru, rv, u, v, w, mut, msft, msfu, msfv, f, e,
                              c1f, c2f, fnm, fnp, dx, dy,
                              ru_t, rv_t, rw_t, *, sina=None, cosa=None,
                              boundary_x=False, boundary_y=False) -> None:
    """ADD the WRF Coriolis + curvature tendencies (coriolis_map.cu).

    Transcribed from WRF v4.6.1 ``module_big_step_utilities_em.F``
    ``coriolis``/``curvature`` (WRF open/specified exclusions; isotropic
    map factors; the full sina/cosa rotation terms: see the kernel header
    for the documented reductions and the WRF line cites).  ``ru``/``rv``
    are the stage's msf-coupled momenta from
    :func:`stage_fluxes`; ``u``/``v``/``w`` the uncoupled stage winds;
    ``mut (ny, nx)`` the total dry mass (the kernel forms the coupled
    ``rw = (c1f*mut + c2f)*w/msft`` inline); ``f``/``e`` the Coriolis
    parameters at mass points; ``sina``/``cosa`` the local map-rotation
    angle at mass points (geo_em SINALPHA/COSALPHA; ``None`` selects WRF's
    unrotated identity sina = 0 / cosa = 1); ``fnm``/``fnp`` the
    half->full weights (WRF passes them as fzm/fzp).  Mirror:
    ``woof.verify.npref.np_coriolis_curvature``.
    """
    nz, ny, nxp1 = ru.shape
    nx = nxp1 - 1
    if sina is None:
        sina = cp.zeros((ny, nx), dtype=DTYPE)
    if cosa is None:
        cosa = cp.ones((ny, nx), dtype=DTYPE)
    kern = get_kernel("coriolis_map", "coriolis_curvature")
    n = nz * (ny + 1) * (nx + 1)
    blocks = (n + 255) // 256
    kern((blocks,), (256,),
         (ru, rv, u, v, w, mut, msft, msfu, msfv, f, e, sina, cosa,
          c1f, c2f, fnm, fnp,
          DTYPE(1.0 / dx), DTYPE(1.0 / dy),
          ru_t, rv_t, rw_t, np.int32(boundary_x), np.int32(boundary_y),
          np.int32(nz), np.int32(ny), np.int32(nx)))


def add_coriolis_curvature(state: DomainState, cfg: RunConfig,
                           ru: cp.ndarray, rv: cp.ndarray) -> None:
    """State-level wrapper: add Coriolis + curvature to the slow tendencies
    (the WRF ``rk_tendency`` coriolis/curvature slot)."""
    launch_coriolis_curvature(
        ru, rv, state.u, state.v, state.w, state.total_mu(),
        state.msft, state.msfu, state.msfv, state.f, state.e,
        state.c1f, state.c2f, state.fnm, state.fnp, cfg.dx, cfg.dy,
        state.ru_t, state.rv_t, state.rw_t,
        sina=state.sina, cosa=state.cosa,
        boundary_x=(cfg.open_x or _boundary_forced(cfg)),
        boundary_y=(cfg.open_y or _boundary_forced(cfg)))


def launch_smag2d_km(u, v, xkmh, xkhh, dx, dy, c_s) -> None:
    """Fill the km_opt=4 eddy viscosities at mass points (smag2d.cu).

    ``u (nz,ny,nx+1)`` / ``v (nz,ny+1,nx)`` device winds; ``xkmh``/``xkhh``
    ``(nz,ny,nx)`` outputs -- momentum K and scalar K = K_m/prandtl.
    Mirror: ``woof.verify.npref.np_smag2d_km``.
    """
    nz, ny, nx = xkmh.shape
    kern = get_kernel("smag2d", "smag2d_km")
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    kern(grid, (_TPB, 1, 1),
         (u, v, xkmh, xkhh, DTYPE(1.0 / dx), DTYPE(1.0 / dy),
          DTYPE(math.sqrt(dx * dy)), DTYPE(c_s), DTYPE(_PRANDTL),
          np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_smag2d_hd(f, xk, mut, c1, c2, dx, dy, tend, stagger="",
                     open_x=False, open_y=False) -> None:
    """ADD the variable-K coupled mixing tendency of one field into ``tend``
    (WRF ``horizontal_diffusion``, coordinate surfaces, smag2d.cu).

    ``stagger`` as in ``launch_add_diff2``: ``""`` mass points (WRF 'm'),
    ``"x"`` u, ``"y"`` v, ``"z"`` w (c1/c2 must then be c1f/c2f; boundary
    levels get no tendency).  ``xk (nz,ny,nx)`` is the mass-point eddy
    viscosity, ``mut (ny,nx)`` the total dry mass.  ``open_x``/``open_y``
    (consumed by the ``"x"``/``"y"`` kernels only) switch the
    boundary-normal face nx-1 / ny-1 to WRF's accurate boundary-datum read
    (field(ide) / field(jde), the stored last column/row) instead of the
    periodic wrap; the caller still zeroes WRF's excluded width-1 strip
    afterwards (``_zero_open_strips``).  Mirror:
    ``woof.verify.npref.np_smag2d_hd``.
    """
    nz, ny, nx = xk.shape
    kern = get_kernel("smag2d", _SMAG_HD[stagger])
    nlev, nys, nxs = f.shape
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    args = [f, xk, mut, c1, c2, DTYPE(1.0 / dx), DTYPE(1.0 / dy), tend,
            np.int32(nz), np.int32(ny), np.int32(nx)]
    if stagger == "x":
        args.append(np.int32(open_x))
    elif stagger == "y":
        args.append(np.int32(open_y))
    kern(grid, (_TPB, 1, 1), tuple(args))


def _wrf_smag_grid_args(state: DomainState, cfg: RunConfig, *,
                        time_t: bool) -> list:
    """Common device arguments for WRF's metric-aware diff_opt=2 kernels."""
    suffix = "0" if time_t else ""
    moist = state.qv is not None
    qv = getattr(state, "qv" + suffix) if moist else state.alt
    return [getattr(state, "u" + suffix), getattr(state, "v" + suffix),
            getattr(state, "w" + suffix), getattr(state, "php" + suffix),
            state.phb, state.alt, qv, state.msft, state.msfu, state.msfv,
            state.fnm, state.fnp, state.dn, state.dnw,
            DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy),
            DTYPE(cfg.dx), DTYPE(cfg.dy),
            DTYPE(state.cf1), DTYPE(state.cf2), DTYPE(state.cf3),
            np.int32(moist)]


def launch_wrf_smag2d_km(state: DomainState, cfg: RunConfig,
                          xkmh, xkhh, *, time_t: bool):
    """WRF v4.6.1 ``cal_deform_and_div`` + ``smag2d_km`` on device.

    Unlike :func:`launch_smag2d_km`'s retained flat-oracle API, this path
    carries total geopotential, map factors, vertical metrics and w.  The
    kernels derive ``rdz/rdzw/zx/zy`` on demand, stage the required
    deformation tensors in dead carrying-buffer prefixes, apply WRF's local
    mixing length and slope limiter, and then preserve WRF's cold-zeroed
    logical outer coefficient row at physical boundaries.  ``phy_bc`` only
    extends that active row into outside ghost cells; it does not copy an
    interior K value into the active boundary.
    """
    nz, ny, nx = xkmh.shape
    n_mass = nz * ny * nx
    # WRF deformation tensors are live only through the u/v stress launches.
    # Borrow the prefixes of the not-yet-built carrying buffers, then replace
    # them with their final tendencies below.  This avoids three additional
    # d04-sized allocations without aliasing any live result.
    d11 = state.scratch((nz + 1, ny, nx), "smag_rw").reshape(-1)[:n_mass]
    d22 = state.scratch((nz, ny, nx + 1), "smag_ru").reshape(-1)[:n_mass]
    d12 = state.scratch((nz, ny + 1, nx), "smag_rv").reshape(-1)[:n_mass]
    d11 = d11.reshape((nz, ny, nx))
    d22 = d22.reshape((nz, ny, nx))
    d12 = d12.reshape((nz, ny, nx))
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    get_kernel("smag2d", "wrf_smag_deform")(
        grid, (_TPB, 1, 1), tuple(common + [d11, d22, d12] + dims))
    tail = [DTYPE(cfg.c_s), DTYPE(_PRANDTL), d11, d22, d12, xkmh, xkhh,
            np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    if cfg.diff_opt == 1:
        get_kernel("diff_opt1", "wrf_diff_opt1_km4")(
            grid, (_TPB, 1, 1),
            (d11, d22, d12, state.msft, DTYPE(cfg.dx), DTYPE(cfg.dy),
             DTYPE(cfg.c_s), DTYPE(_PRANDTL), xkmh, xkhh,
             np.int32(nz), np.int32(ny), np.int32(nx),
             np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))))
    else:
        get_kernel("smag2d", "wrf_smag2d_km")(
            grid, (_TPB, 1, 1), tuple(common + tail))
    if _boundary_x(cfg) or _boundary_y(cfg):
        get_kernel("smag2d", "wrf_smag_km_bc")(
            grid, (_TPB, 1, 1),
            (xkmh, xkhh, np.int32(nz), np.int32(ny), np.int32(nx),
             np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))))
    return d11, d22, d12


def isotropic_km_constants(khdif: float, kvdif: float,
                           prandtl: float = _PRANDTL):
    """WRF ``isotropic_km``'s four coefficients as float32 words.

    module_diffusion_em.F:1756-1767: ``khdif3 = khdif/prandtl`` and
    ``kvdif3 = kvdif/prandtl`` in default REAL with ``prandtl = 1./3.0``
    (share/module_model_constants.F:86), then xkmh = khdif, xkmv = kvdif,
    xkhh = khdif3, xkhv = kvdif3.  The division is one IEEE binary32
    division of the namelist words; ``3*khdif`` (WRF's commented-out
    form, and the old constant-K utility's statement) is a different word
    for some inputs.  ``prandtl`` is :attr:`RunConfig.constant_k_prandtl`
    (WRF's 1/3 by default).  Returns ``(xkmh, xkmv, xkhh, xkhv)``.
    """
    prandtl = np.float32(prandtl)
    kh = np.float32(khdif)
    kv = np.float32(kvdif)
    return kh, kv, np.float32(kh / prandtl), np.float32(kv / prandtl)


def launch_wrf_isotropic_km(state: DomainState, cfg: RunConfig,
                            xkmh, xkhh, *, time_t: bool):
    """WRF km_opt=1: ``cal_deform_and_div`` + ``isotropic_km`` on device.

    The deformation tensors are WRF's (the same ``wrf_smag_deform`` kernel
    as the Smagorinsky launchers), because ``horizontal_diffusion_2`` builds
    the u/v/w stresses from them for every km_opt.  D11 borrows the w
    carrying buffer's prefix as there, but D22 and D12 borrow the x/y face
    workspaces instead of the u/v carrying buffers, so the vertical package
    can fill u and v first (:func:`_compute_wrf_constant_k_tendencies`).  The four exchange coefficients are the namelist
    constants (:func:`isotropic_km_constants`) at EVERY mass point:
    isotropic_km writes its whole tile (its..min(ite,ide-1)), so unlike
    smag2d_km there is no cold-zero outer row to preserve at a physical
    boundary and ``wrf_smag_km_bc`` is not called.  ``xkmv``/``xkhv`` go to
    the ``smag_kmv``/``smag_khv`` slots the km_opt=2/3 launchers fill.
    Returns the (d11, d22, d12) deformation triple.
    """
    nz, ny, nx = xkmh.shape
    n_mass = nz * ny * nx
    d11 = state.scratch((nz + 1, ny, nx), "smag_rw").reshape(-1)[:n_mass]
    d22 = state.scratch((nz, ny, nx + 1), "diff6_x").reshape(-1)[:n_mass]
    d12 = state.scratch((nz, ny + 1, nx), "diff6_y").reshape(-1)[:n_mass]
    d11 = d11.reshape((nz, ny, nx))
    d22 = d22.reshape((nz, ny, nx))
    d12 = d12.reshape((nz, ny, nx))
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    get_kernel("smag2d", "wrf_smag_deform")(
        grid, (_TPB, 1, 1), tuple(common + [d11, d22, d12] + dims))
    kmh, kmv, khh, khv = isotropic_km_constants(
        cfg.khdif, cfg.kvdif, cfg.constant_k_prandtl)
    xkmh.fill(kmh)
    xkhh.fill(khh)
    state.scratch((nz, ny, nx), "smag_kmv").fill(kmv)
    state.scratch((nz, ny, nx), "smag_khv").fill(khv)
    return d11, d22, d12


def launch_wrf_calc_n2(state: DomainState, cfg: RunConfig, bn2, *,
                       time_t: bool) -> None:
    """WRF v4.6.1 ``calculate_N2`` on device (module_diffusion_em.F:
    1485-1713): moist Brunt-Vaisala frequency at mass points into ``bn2``,
    with the saturated moist-adiabatic branch (qv >= qvs or qc >= 1e-5),
    the unsaturated qv/qtot form, the MARTA/WCS one-sided surface level,
    and the ktf copy.  Mirror: ``woof.verify.npref.np_wrf_calc_n2``.

    The kernel is fetched through ``woof.core.moist_n2_mutation`` rather
    than straight from ``get_kernel``, so the LES spec 3.3 mutation control
    -- the saturated branch forced off, for instrument qualification -- can
    select a separately compiled variant under its own cache key.  It is
    off by default and the default path resolves to the same cached
    production kernel as before.
    """
    nz, ny, nx = bn2.shape
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    suffix = "0" if time_t else ""
    thp = getattr(state, "thp" + suffix)
    moist = state.qv is not None
    qc = getattr(state, "qc" + suffix) if moist else None
    qi = (getattr(state, "qi" + suffix)
          if moist and getattr(state, "qi", None) is not None else None)
    dummy = state.alt
    calc_n2_kernel()(
        grid, (_TPB, 1, 1),
        tuple(common + [
            thp, state.thb, np.int32(state.thb.ndim == 3), state.p,
            qc if qc is not None else dummy,
            qi if qi is not None else dummy,
            np.int32(qc is not None), np.int32(qi is not None),
            bn2,
        ] + dims))


def _tke_seed(cfg: RunConfig) -> float:
    """WRF tke_km's seed rule (module_diffusion_em.F:2162-2174): without
    surface drag or a surface heat flux there is no way to generate TKE
    from nothing, so isfflx=0 with both prescribed constants off seeds at
    1e-6; any other isfflx leaves the seed at zero."""
    if cfg.isfflx != 0:
        return 0.0
    if cfg.diff_opt == 2 and cfg.bl_pbl_physics == 0:
        if (cfg.tke_drag_coefficient < 1.0e-10
                and cfg.tke_heat_flux < 1.0e-10):
            return 1.0e-6
        return 0.0
    return 1.0e-6


def launch_wrf_tke_km(state: DomainState, cfg: RunConfig,
                      xkmh, xkhh, *, time_t: bool):
    """WRF v4.6.1 km_opt=2: ``cal_deform_and_div`` + ``calculate_N2`` +
    ``tke_km`` (module_diffusion_em.F:2049-2260) on device, then the
    ``tke_rhs`` forward source (shear + buoyancy + dissipation + the
    positivity limiter) into the ``smag_rtke`` carrying buffer.

    Fills the four exchange coefficients from the time-t prognostic TKE
    (``smag_km``/``smag_kh`` plus the vertical pair ``smag_kmv``/
    ``smag_khv``).  BN2 borrows the ``diff6_x`` prefix exactly as the
    km_opt=3 launcher and stays live through the tke_rhs launch (both
    precede the u/v horizontal staging).  Returns the (d11, d22, d12)
    deformation triple.
    """
    nz, ny, nx = xkmh.shape
    n_mass = nz * ny * nx
    d11 = state.scratch((nz + 1, ny, nx), "smag_rw").reshape(-1)[:n_mass]
    d22 = state.scratch((nz, ny, nx + 1), "smag_ru").reshape(-1)[:n_mass]
    d12 = state.scratch((nz, ny + 1, nx), "smag_rv").reshape(-1)[:n_mass]
    d11 = d11.reshape((nz, ny, nx))
    d22 = d22.reshape((nz, ny, nx))
    d12 = d12.reshape((nz, ny, nx))
    xkmv = state.scratch((nz, ny, nx), "smag_kmv")
    xkhv = state.scratch((nz, ny, nx), "smag_khv")
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    get_kernel("smag2d", "wrf_smag_deform")(
        grid, (_TPB, 1, 1), tuple(common + [d11, d22, d12] + dims))

    bn2 = state.scratch((nz, ny, nx + 1), "diff6_x").reshape(-1)[:n_mass]
    bn2 = bn2.reshape((nz, ny, nx))
    launch_wrf_calc_n2(state, cfg, bn2, time_t=time_t)

    suffix = "0" if time_t else ""
    thp = getattr(state, "thp" + suffix)
    tke = getattr(state, "tke" + suffix)
    get_kernel("smag2d", "wrf_tke_km")(
        grid, (_TPB, 1, 1),
        tuple(common + [
            thp, state.thb, np.int32(state.thb.ndim == 3), state.p,
            tke, bn2,
            DTYPE(cfg.c_k), DTYPE(_PRANDTL), DTYPE(cfg.dt),
            DTYPE(cfg.mix_upper_bound), np.int32(cfg.mix_isotropic),
            DTYPE(_tke_seed(cfg)),
            xkmh, xkhh, xkmv, xkhv,
        ] + dims))
    if _boundary_x(cfg) or _boundary_y(cfg):
        for pair in ((xkmh, xkhh), (xkmv, xkhv)):
            get_kernel("smag2d", "wrf_smag_km_bc")(
                grid, (_TPB, 1, 1),
                (*pair, np.int32(nz), np.int32(ny), np.int32(nx),
                 np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))))

    # WRF first_rk_step_part2.F:904 only calls tke_rhs with diff_opt=2.
    if cfg.diff_opt == 1:
        return d11, d22, d12

    # tke_rhs: the once-per-step forward TKE source, before the borrowed
    # deformation prefixes are replaced by the u/v stress staging.
    fields = (
        state.physics.fields
        if state.physics is not None and hasattr(state.physics, "fields")
        else {}
    )
    dummy = state.mup0
    ustm = fields.get("ustm", dummy)
    hfx = fields.get("hfx", dummy)
    rtke = state.scratch((nz, ny, nx), "smag_rtke")
    mut = state.scratch((ny, nx), "smag_mut")
    mut[...] = state.mub2d + (state.mup0 if time_t else state.mup)
    budget_on = tke_budget.enabled(cfg)
    budget_terms = [tke_budget.term(state, cfg, name)
                    for name in ("shear", "buoyancy", "dissipation",
                                 "limiter")] if budget_on else [rtke] * 4
    if getattr(state, "_moving_frame_velocity", None):
        # MARTA wall production uses ground wind, including prescribed Cd.
        from woof.moving_frame import ground_relative_components
        common[:2] = ground_relative_components(state, *common[:2])
    get_kernel("smag2d", "wrf_tke_rhs")(
        grid, (_TPB, 1, 1),
        tuple(common + [
            thp, state.thb, np.int32(state.thb.ndim == 3),
            tke, bn2, d11, d22, d12,
            xkmh, xkmv, xkhv,
            mut, state.c1h, state.c2h,
            ustm, hfx,
            np.int32("ustm" in fields), np.int32("hfx" in fields),
            DTYPE(cfg.c_k), DTYPE(cfg.dt),
            DTYPE(cfg.tke_drag_coefficient), DTYPE(cfg.tke_heat_flux),
            np.int32(cfg.isfflx),
            rtke,
        ] + budget_terms + [np.int32(budget_on)] + dims))
    return d11, d22, d12


def launch_wrf_smag3d_km(state: DomainState, cfg: RunConfig,
                         xkmh, xkhh, *, time_t: bool):
    """WRF v4.6.1 km_opt=3: ``cal_deform_and_div`` + ``calculate_N2`` +
    ``smag_km`` on device (module_diffusion_em.F:1485-1713, :1777-1929).

    Fills the four exchange-coefficient fields -- ``xkmh``/``xkhh`` in the
    shared ``smag_km``/``smag_kh`` slots plus the km_opt=3-only vertical
    pair in ``smag_kmv``/``smag_khv`` -- from the FULL deformation
    invariant (D13/D23/D33 included, off-diagonal tensors averaged to mass
    points before squaring), the ``sqrt(max(0, D^2 - N^2/Pr))`` buoyancy
    reduction, and WRF's two ``mix_isotropic`` mixing-length branches with
    their exact floors and ``mix_upper_bound/dt`` caps.  BN2 briefly
    borrows the ``diff6_x`` face workspace prefix (dead until the u/v
    horizontal staging that follows the K computation).  Returns the
    (d11, d22, d12) deformation triple exactly as the km_opt=4 launcher.
    """
    nz, ny, nx = xkmh.shape
    n_mass = nz * ny * nx
    d11 = state.scratch((nz + 1, ny, nx), "smag_rw").reshape(-1)[:n_mass]
    d22 = state.scratch((nz, ny, nx + 1), "smag_ru").reshape(-1)[:n_mass]
    d12 = state.scratch((nz, ny + 1, nx), "smag_rv").reshape(-1)[:n_mass]
    d11 = d11.reshape((nz, ny, nx))
    d22 = d22.reshape((nz, ny, nx))
    d12 = d12.reshape((nz, ny, nx))
    xkmv = state.scratch((nz, ny, nx), "smag_kmv")
    xkhv = state.scratch((nz, ny, nx), "smag_khv")
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    get_kernel("smag2d", "wrf_smag_deform")(
        grid, (_TPB, 1, 1), tuple(common + [d11, d22, d12] + dims))

    # calculate_N2 inputs: time-t prognostic theta/moisture, current
    # diagnostic p (refreshed against the time-t state by step() before
    # prepare_fixed_tendencies, exactly as alt in the common args).
    bn2 = state.scratch((nz, ny, nx + 1), "diff6_x").reshape(-1)[:n_mass]
    bn2 = bn2.reshape((nz, ny, nx))
    launch_wrf_calc_n2(state, cfg, bn2, time_t=time_t)

    get_kernel("smag2d", "wrf_smag3d_km")(
        grid, (_TPB, 1, 1),
        tuple(common + [
            DTYPE(cfg.c_s), DTYPE(_PRANDTL), DTYPE(cfg.dt),
            DTYPE(cfg.mix_upper_bound), np.int32(cfg.mix_isotropic),
            d11, d22, d12, bn2, xkmh, xkhh, xkmv, xkhv,
        ] + dims))
    if _boundary_x(cfg) or _boundary_y(cfg):
        for pair in ((xkmh, xkhh), (xkmv, xkhv)):
            get_kernel("smag2d", "wrf_smag_km_bc")(
                grid, (_TPB, 1, 1),
                (*pair, np.int32(nz), np.int32(ny), np.int32(nx),
                 np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))))
    return d11, d22, d12


def _wrf_smag_scalar_kernel(name: str, nz: int, ny: int, nx: int):
    """Select narrow addressing only for the range-checked scalar pair."""
    if name not in ("wrf_smag_flux_s", "wrf_smag_hd_s"):
        raise ValueError(f"not a Smagorinsky scalar kernel: {name!r}")
    if scalar_index32_fits(nz, ny, nx):
        return get_kernel_int_defines(
            "smag2d", name, (("GPUWM_SMAG_INDEX32", 1),))
    return get_kernel("smag2d", name)


def launch_wrf_smag2d_hd(state: DomainState, cfg: RunConfig, f, xk, tend,
                          *, stagger: str, time_t: bool,
                          full_theta: bool = False,
                          deformation=None) -> None:
    """Add WRF's metric-aware tensor-stress/scalar-flux horizontal tendency.

    WRF passes ``grid%t_2 = theta - T0`` to the scalar operator.  For the
    theta row, ``full_theta`` reconstructs that field on demand from woof's
    ``thp = theta - thb`` storage; moisture rows pass their mixing ratios
    unchanged.  Reconstruction inside the flux kernel avoids a full-grid
    temporary and handles both one-dimensional and terrain-following 3-D
    ``thb``.
    """
    if stagger not in _WRF_SMAG_HD:
        raise ValueError(f"unknown Smagorinsky stagger {stagger!r}")
    nz, ny, nx = xk.shape
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    tail = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    nlev, nys, nxs = f.shape
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    if stagger == "":
        flux_x = state.scratch((nz, ny, nx + 1), "diff6_x")
        flux_y = state.scratch((nz, ny + 1, nx), "diff6_y")
        flux_grid = ((nx + 1 + _TPB - 1) // _TPB, ny + 1, nz)
        _wrf_smag_scalar_kernel("wrf_smag_flux_s", nz, ny, nx)(
            flux_grid, (_TPB, 1, 1),
            tuple(common + [f, xk, state.thb, np.int32(full_theta),
                            np.int32(state.thb.ndim == 3),
                            flux_x, flux_y] + tail))
        payload = [flux_x, flux_y, tend]
    elif stagger == "x":
        if deformation is None:
            raise ValueError("u Smagorinsky stress requires deformation")
        d11, _d22, d12 = deformation
        payload = [xk, d11, d12, tend]
    elif stagger == "y":
        if deformation is None:
            raise ValueError("v Smagorinsky stress requires deformation")
        _d11, d22, d12 = deformation
        payload = [xk, d22, d12, tend]
    else:
        # W takes wrf_smag_hd_w in every mode.  449e3d697 made
        # DIFFUSION_ENABLED true whenever WRF_EXACT is off, so the former
        # wrf_smag_w_stress/wrf_smag_hd_w_stress route could no longer run
        # and is retired.
        payload = [xk, tend]
    kernel = (_wrf_smag_scalar_kernel(_WRF_SMAG_HD[stagger], nz, ny, nx)
              if stagger == "" else get_kernel("smag2d", _WRF_SMAG_HD[stagger]))
    kernel(
        grid, (_TPB, 1, 1), tuple(common + payload + tail))


def launch_wrf_smag2d_vertical(
        state: DomainState, cfg: RunConfig, km, *,
        ru, rv, rw, rth, rqv, time_t: bool,
        kmv=None, khv=None, scalar_rows=None, w_km=None,
        surface: bool = True, interior: bool = True) -> None:
    """Add WRF v4.6.1 ``vertical_diffusion_2`` for the PBL-off diff_opt=2
    path (module_first_rk_step_part2.F:1011-1074).

    km_opt=4: ``smag2d_km`` defines ``xkmv=xkmh`` and ``xkhv=0``, so only
    u/v/w have interior vertical stresses (``kmv``/``khv`` omitted).
    km_opt=2/3 passes ``kmv`` (vertical momentum K, consumed by the tau13/
    tau23 operators) and
    ``khv`` plus ``scalar_rows`` -- ``(field, tendency, full_theta)``
    triples mixed by ``vertical_diffusion_s`` with the vertical scalar K.

    Declared divergence: WRF's ``vertical_diffusion_2`` passes ``xkmh``
    to ``vertical_diffusion_w_2`` (module_diffusion_em.F:4145-4155).
    This launcher passes ``xkmv`` unless the caller names ``w_km``: the
    tau33 flux contains the vertical derivative of w and uses the vertical
    momentum coefficient, as do the tau13/tau23 vertical stresses. The
    coefficients coincide for km_opt=4. The compiled WRF leaf fixtures
    compare identical coefficient inputs; the driver fixtures retain WRF's
    original coefficient choice.  km_opt=1 passes ``w_km = xkmh`` and so
    runs WRF's choice (tools/wrf_diffusion_oracle/km1_README.md).

    A ``None`` target skips that row's interior launch, and ``surface =
    False`` skips the surface arms; km_opt=1 uses both to keep WRF's
    accumulation order (:func:`_compute_wrf_constant_k_tendencies`).
    ``interior = False`` launches the surface arms alone, into k = 0
    planes (:func:`_restore_open_ring_surface_fluxes`).

    Surface forcing follows WRF's ``SELECT CASE(isfflx)`` matrix:
    isfflx=0 takes the prescribed ``tke_drag_coefficient`` wall stress and
    ``tke_heat_flux`` heat flux with no moisture flux; isfflx=1 takes
    USTM/HFX/QFX from the surface driver; isfflx=2 takes USTM/QFX from
    the driver but the constant ``tke_heat_flux`` heat.  A surface-layer-
    free run supplies no fields and the ust-based arms stay cold-zero,
    exactly the no-writer WRF state.
    """
    nz, ny, nx = km.shape
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    tail = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    if kmv is None:
        kmv = km
    if w_km is None:
        w_km = kmv
    launches = (
        ("wrf_smag_vd_u", kmv, ru, (nx + 1, ny, nz)),
        ("wrf_smag_vd_v", kmv, rv, (nx, ny + 1, nz)),
        # DIVERGENCE, deliberate, for km_opt=2/3.  WRF hands
        # vertical_diffusion_w_2 xkmh (module_diffusion_em.F:4145-4155);
        # woof hands it xkmv there.  See this function's docstring and the
        # compiled WRF driver fixtures.  km_opt=1 passes w_km = xkmh.
        ("wrf_smag_vd_w", w_km, rw, (nx, ny, nz + 1)),
    )
    for name, xk, tendency, (nxs, nys, nlev) in launches:
        if tendency is None or not interior:
            continue
        grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
        get_kernel("smag2d", name)(
            grid, (_TPB, 1, 1), tuple(common + [xk, tendency] + tail))

    mass_grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    if interior and khv is not None and scalar_rows:
        for field, tendency, full_theta in scalar_rows:
            get_kernel("smag2d", "wrf_smag_vd_s")(
                mass_grid, (_TPB, 1, 1),
                tuple(common + [
                    field, state.thb, np.int32(bool(full_theta)),
                    np.int32(state.thb.ndim == 3), khv, tendency,
                ] + tail))
    if not surface:
        return
    if getattr(state, "_moving_frame_velocity", None):
        # Interior mixing uses relative winds; the lower wall is ground-fixed.
        from woof.moving_frame import ground_relative_components
        common[:2] = ground_relative_components(state, *common[:2])

    fields = (
        state.physics.fields
        if state.physics is not None and hasattr(state.physics, "fields")
        else {}
    )
    active = int(
        cfg.sf_sfclay_physics != 0
        and all(name in fields for name in ("ustm", "hfx", "qfx")))
    isfflx = cfg.isfflx
    dummy = state.mup0
    ustm = fields.get("ustm", dummy)
    hfx = fields.get("hfx", dummy)
    qfx = fields.get("qfx", dummy)
    if isfflx == 0:
        # vflux CASE(0): constant drag coefficient, no surface scheme.
        cd0 = DTYPE(cfg.tke_drag_coefficient)
        for name, tendency, nxs, nys in (
                ("wrf_smag_surface_u_cd0", ru, nx + 1, ny),
                ("wrf_smag_surface_v_cd0", rv, nx, ny + 1)):
            grid = ((nxs + _TPB - 1) // _TPB, nys, 1)
            get_kernel("smag2d", name)(
                grid, (_TPB, 1, 1),
                tuple(common + [cd0, tendency] + tail))
    else:
        # vflux CASE(1,2): ustar from the surface routine (USTM).
        for name, tendency, nxs, nys in (
                ("wrf_smag_surface_u", ru, nx + 1, ny),
                ("wrf_smag_surface_v", rv, nx, ny + 1)):
            grid = ((nxs + _TPB - 1) // _TPB, nys, 1)
            get_kernel("smag2d", name)(
                grid, (_TPB, 1, 1),
                tuple(common + [ustm, np.int32(active), tendency] + tail))
    scalar_grid = ((nx + _TPB - 1) // _TPB, ny, 1)
    if isfflx in (0, 2):
        # hflux CASE(0,2): prescribed constant kinematic heat flux.
        get_kernel("smag2d", "wrf_smag_surface_heat_const")(
            scalar_grid, (_TPB, 1, 1),
            tuple(common + [DTYPE(cfg.tke_heat_flux), hfx,
                            np.int32("hfx" in fields), rth] + tail))
    apply_heat = int(isfflx == 1) and active
    apply_moist = int(isfflx in (1, 2)) and active
    get_kernel("smag2d", "wrf_smag_surface_scalars")(
        scalar_grid, (_TPB, 1, 1),
        tuple(common + [
            hfx, qfx, np.int32(apply_heat), np.int32(apply_moist), rth,
            rqv if rqv is not None else rth,
        ] + tail))


def _horizontal_w_km(state: DomainState, cfg: RunConfig):
    """WRF's ``xkmv`` where ``horizontal_diffusion_w_2`` needs it.

    ``None`` for km_opt=4 (and for the diff6-only path, which consumes no
    K at all): ``smag2d_km`` defines ``xkmv = xkmh``
    (module_diffusion_em.F:2035), so the caller's ``xkmh`` IS ``xkmv``
    there.  km_opt=2/3 fill a separate vertical pair in ``smag_kmv``
    (``tke_km`` :2049-2260 / ``smag_km`` :1890-1908), and on an
    anisotropic grid it is smaller than ``xkmh`` by (dz/dx)^2 -- the whole
    point of ``mix_isotropic = 0``.
    """
    # km_opt=1's xkmv is kvdif (isotropic_km), live only while its package
    # runs; a diff6-only km_opt=1 step must not allocate the slot.
    if cfg.km_opt not in (2, 3) and not constant_k_mixing_active(cfg):
        return None
    return state.scratch(state.p.shape, "smag_kmv")


def _smag2d_specs(state: DomainState, km, kh, *, time_t: bool = False,
                  kmv=None):
    """(field, tend-or-None, K, c1, c2, scratch slot, stagger) rows for the
    km_opt=4 package: momentum takes K_m, scalars (WRF ``theta - T0`` and
    moisture) take K_h = K_m/prandtl; moisture rows carry no state tendency array (their
    increment folds into the scalar update).  ``time_t`` binds the saved
    ``*0`` fields used by WRF's once-per-step forward tendencies.

    ``kmv`` is the w row's horizontal K.  WRF's ``horizontal_diffusion_2``
    hands ``xkmh`` to ``horizontal_diffusion_u_2``/``_v_2`` but ``xkmv`` to
    ``horizontal_diffusion_w_2`` (module_diffusion_em.F:2978-3006; the
    dummy argument is spelled ``xkmv`` at :3524), because that operator is
    the divergence of tau13/tau23 and those stresses take the VERTICAL
    momentum coefficient everywhere else too (``vertical_diffusion_u_2``/
    ``_v_2``, :4128-4147).  ``smag2d_km`` sets ``xkmv = xkmh`` for km_opt=4
    (:2035, "v4.2 and later, this is used for hor. diff. of w"), so the
    default ``None`` -- pass ``xkmh`` -- is that identity, not a shortcut.
    km_opt=2/3 compute a genuinely different vertical pair and must pass
    it."""
    def field(name):
        return getattr(state, name + "0" if time_t else name)
    specs = [(field("u"), state.ru_t, km, state.c1h, state.c2h,
              "smag_ru", "x"),
             (field("v"), state.rv_t, km, state.c1h, state.c2h,
              "smag_rv", "y"),
             (field("w"), state.rw_t, km if kmv is None else kmv,
              state.c1f, state.c2f, "smag_rw", "z"),
             (field("thp"), state.rth_t, kh, state.c1h, state.c2h,
              "smag_rth", "")]
    if state.qv is not None:
        specs += [(field(name), None, kh, state.c1h, state.c2h,
                   "smag_r" + name, "") for name in SPECIES]
    specs += [(field(name), None, kh, state.c1h, state.c2h,
               "smag_r" + name, "")
              for name in extra_moist_species(state)]
    chem = getattr(state, "chem", None)
    if chem is not None:
        # WRF's chem array mixes like the scalar arrays: horizontal_diffusion_s
        # (module_diffusion_em.F:3054-3070), vertical_diffusion_s under PBL
        # off (:4409-4421) and the 6th-order filter in rk_scalar_tend
        # (module_em.F:1421), each held once per step in chem_tend.
        specs += [(getattr(state, row.time_attr if time_t
                           else row.state_attr),
                   None, kh, state.c1h, state.c2h,
                   "smag_r" + row.state_attr, "")
                  for row in chem.transported]
    return specs


def _chem_time_pairs(state) -> tuple:
    """The chem rows' ``(field, chem0_<row>)`` pairs for ``_save_time_t``.

    Saved in the same one bookkeeping launch as the other time-t copies;
    empty on a chem-off state, so its pair table and launch are unchanged.
    """
    chem = getattr(state, "chem", None)
    return () if chem is None else chem.time_pairs(state)


def chem_mix2_exempt_slots(state: DomainState, cfg: RunConfig
                           ) -> frozenset[str]:
    """Chem rows' carrying slots under ``chem_mix2_off`` (horizontal only)."""
    chem = getattr(state, "chem", None)
    if chem is None or not cfg.chem_mix2_off:
        return frozenset()
    return frozenset("smag_r" + row.state_attr for row in chem.transported)


def diff6_exempt_slots(cfg: RunConfig) -> frozenset[str]:
    """Carrying-buffer slots the 6th-order filter must skip this run.

    WRF's ``&dynamics`` filter switches are per Registry ARRAY, and
    ``rk_scalar_tend`` calls ``sixth_order_diffusion`` under
    ``(diff_6th_opt .NE. 0) .and. (.not. mix6_off)`` with the array's own
    switch (dyn_em/module_em.F:1421).  ``moist_mix6_off`` therefore removes
    the moist array's rows and NOTHING else: theta keeps its filter, the
    number/volume tracers keep theirs (they are WRF ``scalar``-package
    fields with their own ``scalar_mix6_off``), and TKE keeps
    ``tke_mix6_off``.

    Returned as slot names because the diff6 row set is addressed by
    carrying buffer, and a name filter cannot accidentally exempt a row
    whose field happens to share a shape with a moist one.
    """

    exempt = frozenset()
    if cfg.moist_mix6_off:
        exempt = frozenset("smag_r" + name for name in WRF_MOIST_ARRAY_SPECIES)
    if getattr(cfg, "chem_mix6_off", False) and getattr(cfg, "chem_sets", ()):
        # WRF's chem_mix6_off (Registry.EM_COMMON:2894) gates the chem array
        # the same way, rk_scalar_tend's mix6_off argument for chem
        # (solve_em.F:2497).
        from woof.chem_table import load as load_chem_table
        exempt |= frozenset("smag_r" + row.state_attr for row in
                            load_chem_table(cfg).transported)
    return exempt


# The four rows WRF filters from ``rk_tendency`` (dyn_em/module_em.F:882,
# :894, :907, :919); every other diff6 row is filtered from
# ``rk_scalar_tend`` (:1425).
_DIFF6_DRY_SLOTS = frozenset(("smag_ru", "smag_rv", "smag_rw", "smag_rth"))


#: The sixth-order filter forms (``RunConfig.diff_6th_form``), by source.
DIFF6_FORMS = ("wrf_461", "noaa_wrf39")

#: The fork's Registry default for ``diff_6th_factor2`` (NOAA-EMC/HRRR
#: v4.1.21 WRFV3.9 Registry/Registry.EM_COMMON:2629).
DIFF6_FACTOR2_FORK_DEFAULT = 0.04

#: Rows whose factor is ``diff_6th_factor`` under every form: the dry rows
#: (``rk_tendency``) and TKE, which the fork's rk_scalar_tend call for
#: tke_2 hands ``grid%diff_6th_factor`` (fork solve_em.F:2587), as it does
#: the tracer array (:2831).  Every other row is a moist (:2418) or scalar
#: (:2974) array row and takes ``diff_6th_factor2`` under the fork.
_DIFF6_FACTOR1_SLOTS = frozenset(("smag_ru", "smag_rv", "smag_rw",
                                  "smag_rth", "smag_rtke"))


def diff6_fork(cfg: RunConfig) -> bool:
    """True when the NOAA WRFV3.9 fork's filter form is selected."""
    return getattr(cfg, "diff_6th_form", "wrf_461") == "noaa_wrf39"


def _diff6_factor(cfg: RunConfig, slot: str) -> float:
    """The clock-scaled filter factor one diff6 row takes.

    ``wrf_461`` hands every row ``diff_6th_factor``.  The fork hands the
    moist and scalar arrays ``diff_6th_factor2`` (solve_em.F:2418, :2974;
    module_em.F:1409-1412 passes it to sixth_order_diffusion) and keeps
    ``diff_6th_factor`` for u, v, w, theta and TKE.
    """
    if not diff6_fork(cfg) or slot in _DIFF6_FACTOR1_SLOTS:
        return _clock_scaled_diff6_factor(cfg)
    if getattr(cfg, "chem_sets", "") and slot.startswith("smag_rchem_"):
        from woof.chem_table import catalog

        row = catalog().species.get(slot[len("smag_rchem_"):])
        if row is not None and row.wrf_array == "tracer":
            # The fork passes factor1 to rk_scalar_tend for its tracer
            # array (solve_em.F:2831), while chem takes factor2. Shared
            # arena storage does not merge those Registry array semantics.
            return _clock_scaled_diff6_factor(cfg)
    factor2 = (DIFF6_FACTOR2_FORK_DEFAULT if cfg.diff_6th_factor2 is None
               else float(cfg.diff_6th_factor2))
    return _clock_scale_factor(cfg, factor2, "diff_6th_factor2")


def refresh_saved_wind_fluxes(state, cfg, ru, rv):
    """Refresh only sumflux's horizontal reference fluxes after limiting.

    The fork passes modified u_save/v_save to sumflux on its last acoustic
    substep (solve_em.F:1743-1746, module_small_step_em.F:1867-1873).
    The stage mass and map factors stay fixed; ww_lin stays unchanged.
    Run this before small_step_finish overwrites the saved stage fields.
    """
    mux, muy = stage_face_masses(state, cfg, state.total_mu())
    for wind, muface, msf, flux in ((state.u, mux, state.msfu, ru),
                                    (state.v, muy, state.msfv, rv)):
        kernel = _couple_momentum_kernel(
            state.has_msf, WRF_EXACT and wind is state.v)
        args = [wind, state.c1h, state.c2h, muface.reshape(-1)]
        if state.has_msf:
            factor = state.msfv_inv if WRF_EXACT and wind is state.v else msf
            args.append(factor.reshape(-1))
        kernel(*args, np.int32(muface.size), flux)


def _diff6_dt(cfg: RunConfig, slot: str) -> float:
    """The ``dt`` WRF hands ``sixth_order_diffusion`` for one row.

    ``diff_6th_coef = diff_6th_factor*0.015625/(2.0*dt)``
    (module_big_step_utilities_em.F:6321), so this argument sets the
    filter's strength and WRF does not use one value for it.
    ``rk_tendency`` is called with ``grid%dt`` (solve_em.F:892) and passes
    it to the u/v/w/theta calls; ``rk_scalar_tend`` is called with
    ``dt_rk`` (solve_em.F:2211, :2380, :2473, :2635, :2777) and passes it
    on as ``dt_step`` for the moist/scalar/tke rows.  Both diff6 blocks sit
    under ``rk_step == 1`` (module_em.F:800, :1378), where
    ``dt_rk = grid%dt/3.`` for ``rk_ord = 3`` (solve_em.F:596-600) -- the
    only order this dycore integrates (namelist_import.py pins it).  WRF's
    scalar filter is therefore three times the strength of its dry filter.

    The NOAA WRFV3.9 fork (``diff_6th_form = "noaa_wrf39"``) builds the
    full step inside rk_scalar_tend, ``dt = dt_rk*(rk_order - rk_step +
    1)`` (fork module_em.F:1216-1217), and hands that to the filter
    (:1409-1412), so every scalar row takes the full step there.  In
    REAL arithmetic that is ``(grid%dt/3.)*3``, formed here in float32 as
    the fork forms it (20 s gives exactly 20.0; another step need not).
    """
    if slot in _DIFF6_DRY_SLOTS:
        return cfg.dt
    if diff6_fork(cfg):
        return float(np.float32(np.float32(cfg.dt) / np.float32(3.0))
                     * np.float32(3.0))
    return cfg.dt / 3.0


def _couple_dry_mixing_map_factor(state: DomainState, specs) -> None:
    """Apply ``rk_addtend_dry``'s ``1/msf`` to the held dry tendencies.

    WRF's dry mixing tendencies reach ``ru_tendf``/``rv_tendf``/
    ``rw_tendf``/``t_tendf`` already carrying the target's map factor --
    ``horizontal_diffusion_u_2`` builds ``mrdx=msfux(i,j)*rdx``
    (module_diffusion_em.F:3304-3312), which ``wrf_smag_hd_u``
    (kernels/smag2d.cu:786-792) transcribes, and the vertical rows carry
    none in either code -- and ``rk_addtend_dry`` divides the sum by it
    once (module_em.F:1043, :1054, :1065, :1078).

    ``sixth_order_diffusion`` also multiplies by the map factor
    (module_big_step_utilities_em.F:6509/:6522/:6531 and :6599/:6605/
    :6614), and kernels/diff6.cu does too since the compiled WRF v4.7.1
    diffusion oracle (83fde6032), so WRF's net diff6 contribution to the
    dry tendencies carries no map factor.  Both packages share one carrying
    buffer here, so the division is taken once over their sum, after diff6
    has accumulated, as WRF takes it.  Until 2.8.2 diff6.cu omitted the
    multiply and this division ran before diff6; the oracle's multiply
    alone would have left the dry diff6 tendency msf times WRF's
    (tests/test_rk_addtend_dry_map_factors.py measures both errors).

    Only the four rows ``rk_addtend_dry`` owns are coupled; the moisture
    and TKE rows have no state tendency and go to ``rk_update_scalar``,
    which adds ``sc_tend`` raw, so their diff6 keeps WRF's map factor.
    """
    if not state.has_msf:
        return
    msf = {"x": state.msfu, "y": state.msfv, "z": state.msft, "": state.msft}
    for f0, tend, _xk, _c1, _c2, slot, stag in specs:
        if tend is not None:
            if BIGSTEP_ENABLED and stag == "y":
                state.scratch(f0.shape, slot)[:] *= state.msfv_inv[None]
            else:
                state.scratch(f0.shape, slot)[:] /= msf[stag][None]


def launch_coordinate_horizontal(field, xk, mut, c1, c2, msft, msfu,
                                 msfv, dx, dy, tendency, *, stagger="",
                                 boundary_x=False, boundary_y=False,
                                 theta_initial=None) -> None:
    """Add WRF's coordinate-surface flux operator with explicit REAL order.

    ``theta_initial`` selects ``horizontal_diffusion_3dmp``. WRF uses
    that initial thermal field for both values of mix_full_fields.
    """
    nz, ny, nx = xk.shape
    nlev, nys, nxs = field.shape
    stag = {"": 0, "x": 1, "y": 2, "z": 3}[stagger]
    get_kernel("diff_opt1", "wrf_diff_opt1_horizontal", (nz, ny, nx))(
        ((nxs + _TPB - 1) // _TPB, nys, nlev), (_TPB, 1, 1),
        (field, xk, mut, c1, c2,
         theta_initial if theta_initial is not None else field,
         msft, msfu, msfv, DTYPE(1.0 / dx), DTYPE(1.0 / dy), tendency,
         np.int32(nz), np.int32(ny), np.int32(nx), np.int32(stag),
         np.int32(boundary_x), np.int32(boundary_y),
         np.int32(theta_initial is not None)))


def initialize_coordinate_reference(state, cfg) -> None:
    """Capture WRF's t_init before the first step or streaming inventory.

    This reference is serialized and carried between streamed slabs. A
    resumed state cannot reconstruct the original field from evolved theta.
    """
    if cfg.diff_opt != 1 or "diff1_theta_initial" in state._scratch:
        return
    if getattr(state, "elapsed_seconds", 0.0) > 0.0:
        raise RuntimeError(
            "diff_opt=1 restart is missing its original thermal reference "
            "diff1_theta_initial; reconstructing it from evolved theta "
            "would change coordinate diffusion")
    initial = state.scratch(state.thp.shape, "diff1_theta_initial")
    initial[...] = state.thp + (state.thb.reshape((-1, 1, 1))
                               if state.thb.ndim == 1 else state.thb)
    initial -= c.T0


def _coordinate_theta_field(state, thp):
    """Keep canonical strict theta words at the coordinate-mixing boundary."""
    if WRF_EXACT:
        # Native exact initialization and the strict RK kernels keep thp in
        # WRF's theta - 300 basis. Adding 300 and subtracting it again loses
        # evolved perturbation bits before horizontal_diffusion_3dmp.
        return thp
    theta = state.scratch(thp.shape, "diff1_theta_work")
    base = state.thb.reshape((-1, 1, 1)) if state.thb.ndim == 1 else state.thb
    theta[...] = thp + base
    theta -= c.T0
    return theta


def _compute_coordinate_tendencies(state, cfg, km, kh, specs, *, time_t):
    """WRF diff_opt=1's forward horizontal mixing, module_em.F:801-840.

    Its coefficient calculation shares module_diffusion_em with diff_opt=2;
    coordinate mixing uses Kh for every scalar, including prognostic TKE.
    WRF does not call tke_rhs or vertical_diffusion_2 in this branch.
    kvdif is zero for these turbulence selections, so the older constant
    vertical operator produces zero without a launch.
    """
    if cfg.km_opt == 2:
        launch_wrf_tke_km(state, cfg, km, kh, time_t=time_t)
    else:
        launch_wrf_smag2d_km(state, cfg, km, kh, time_t=time_t)
    mut = state.scratch(state.mup.shape, "smag_mut")
    mut[...] = state.mub2d + (state.mup0 if time_t else state.mup)
    suffix = "0" if time_t else ""
    thp = getattr(state, "thp" + suffix)
    theta = _coordinate_theta_field(state, thp)
    initialize_coordinate_reference(state, cfg)
    initial = state._scratch["diff1_theta_initial"]
    for f, _tend, _xk, c1, c2, slot, stag in specs:
        buf = state.scratch(f.shape, slot)
        buf[...] = 0
        if WRF_EXACT and time_t and _tend is None:
            # rk_scalar_tend runs after the first acoustic loop, with
            # grid%muts as its mass. The physics slot is folded below.
            continue
        is_theta = slot == "smag_rth"
        launch_coordinate_horizontal(
            theta if is_theta else f, km if stag else kh,
            mut, c1, c2, state.msft, state.msfu, state.msfv,
            cfg.dx, cfg.dy, buf, stagger=stag,
            boundary_x=_boundary_x(cfg), boundary_y=_boundary_y(cfg),
            theta_initial=initial if is_theta else None)
    if (cfg.km_opt == 2 and not getattr(cfg, "tke_mix2_off", False)
            and not (WRF_EXACT and time_t)):
        tke = getattr(state, "tke" + suffix)
        buf = state.scratch(tke.shape, "smag_rtke")
        buf[...] = 0
        launch_coordinate_horizontal(
            tke, kh, mut, state.c1h, state.c2h,
            state.msft, state.msfu, state.msfv, cfg.dx, cfg.dy, buf,
            boundary_x=_boundary_x(cfg), boundary_y=_boundary_y(cfg))
        budget = tke_budget.term(state, cfg, "diffusion_h")
        if budget is not None:
            budget[...] = buf


def add_scalar_coordinate_wrf(state: DomainState, cfg: RunConfig) -> None:
    """Add coordinate scalar mixing at rk_scalar_tend's mass boundary.

    Dry coordinate mixing uses the stage-start mass. Moisture, TKE and
    chem use the first acoustic loop's MUTS, before scalar sixth order
    and lateral relaxation. Their coefficient remains the time-t Kh.
    """
    if not (WRF_EXACT and cfg.diff_opt == 1 and cfg.km_opt in (2, 3, 4)):
        return
    rows = [row for row in _smag2d_specs(state, None, None, time_t=True)
            if row[1] is None]
    exempt = chem_mix2_exempt_slots(state, cfg)
    rows = [row for row in rows if row[5] not in exempt]
    if cfg.km_opt == 2 and not getattr(cfg, "tke_mix2_off", False):
        rows.append((state.tke0, None, None, state.c1h, state.c2h,
                     "smag_rtke", ""))
    if not rows:
        return
    mut = _wrf_muts(state)
    if mut is None:
        raise RuntimeError("strict coordinate scalar mixing needs the acoustic loop's muts")
    kh = state._scratch["smag_kh"]
    for field, _tend, _xk, c1, c2, slot, _stag in rows:
        target = state._scratch[slot]
        launch_coordinate_horizontal(
            field, kh, mut, c1, c2, state.msft, state.msfu, state.msfv,
            cfg.dx, cfg.dy, target,
            boundary_x=_boundary_x(cfg), boundary_y=_boundary_y(cfg))
        if slot == "smag_rtke":
            budget = tke_budget.term(state, cfg, "diffusion_h")
            if budget is not None:
                budget[...] = target


def _compute_wrf_smag_tendencies(state: DomainState, cfg: RunConfig,
                                  km, kh, specs, *, time_t: bool) -> None:
    """Build the once-per-step WRF metric/stress forward tendencies.

    The deformation tensors temporarily occupy mass-sized prefixes of the
    u/v/w carrying buffers.  u and v therefore launch first into the reusable
    diff6 face workspaces; after both have consumed all three tensors, their
    results replace those borrowed buffers.  w and every scalar can then be
    produced normally.  The scalar operator uses the same face workspaces for
    its two explicit metric flux passes.
    """
    if cfg.diff_opt == 1:
        _compute_coordinate_tendencies(state, cfg, km, kh, specs,
                                      time_t=time_t)
        return
    if cfg.km_opt == 1:
        _compute_wrf_constant_k_tendencies(state, cfg, km, kh, specs,
                                           time_t=time_t)
        return
    if cfg.km_opt == 2:
        deformation = launch_wrf_tke_km(
            state, cfg, km, kh, time_t=time_t)
    elif cfg.km_opt == 3:
        deformation = launch_wrf_smag3d_km(
            state, cfg, km, kh, time_t=time_t)
    else:
        deformation = launch_wrf_smag2d_km(
            state, cfg, km, kh, time_t=time_t)

    # u/v must both see the complete deformation set before either borrowed
    # carrying buffer is replaced by its final stress divergence.
    for row, tmp_slot in zip(specs[:2], ("diff6_x", "diff6_y")):
        f, _tend, xk, _c1, _c2, _slot, stag = row
        tmp = state.scratch(f.shape, tmp_slot)
        tmp[...] = 0
        launch_wrf_smag2d_hd(state, cfg, f, xk, tmp, stagger=stag,
                             time_t=time_t, deformation=deformation)
        _zero_open_strips(tmp, cfg, 1)
    for row, tmp_slot in zip(specs[:2], ("diff6_x", "diff6_y")):
        f, _tend, _xk, _c1, _c2, slot, _stag = row
        state.scratch(f.shape, slot)[...] = state.scratch(f.shape, tmp_slot)

    # D11 occupied smag_rw; it is dead after both horizontal-momentum calls.
    # Scalars use fresh H1/H2 fluxes in diff6_x/y for each field.
    # chem_mix2_off drops the chem array's HORIZONTAL mixing only
    # (module_diffusion_em.F:3054); vertical_diffusion_2's chem loop
    # (:4409-4421) has no such gate, so the rows stay in the vertical pass.
    mix2_exempt = chem_mix2_exempt_slots(state, cfg)
    for f, _tend, xk, _c1, _c2, slot, stag in specs[2:]:
        buf = state.scratch(f.shape, slot)
        buf[...] = 0
        if slot in mix2_exempt:
            continue
        launch_wrf_smag2d_hd(state, cfg, f, xk, buf, stagger=stag,
                             time_t=time_t,
                             full_theta=(slot == "smag_rth"))
        _zero_open_strips(buf, cfg, 1)

    if cfg.km_opt == 2:
        # TKE self-diffusion: horizontal with 2*Km_h regardless of the PBL
        # (module_diffusion_em.F:3020-3032, doubling :3988-3996 -- WRF's
        # doing_tke tendency = tmptendf + 2*(tendency - tmptendf)).
        nz_, ny_, nx_ = km.shape
        tke_t = getattr(state, "tke0" if time_t else "tke")
        rtke = state.scratch((nz_, ny_, nx_), "smag_rtke")
        tmp = state.scratch((nz_, ny_, nx_), "smag_tke_tmp")
        if not getattr(cfg, "tke_mix2_off", False):
            tmp[...] = 0
            launch_wrf_smag2d_hd(state, cfg, tke_t, km, tmp, stagger="",
                                 time_t=time_t, full_theta=False)
            _zero_open_strips(tmp, cfg, 1)
            rtke += 2.0 * tmp
            budget_h = tke_budget.term(state, cfg, "diffusion_h")
            if budget_h is not None:
                budget_h[...] = 2.0 * tmp

    if cfg.bl_pbl_physics == 0:
        buffers = {
            slot: state.scratch(f.shape, slot)
            for f, _tend, _xk, _c1, _c2, slot, _stag in specs
        }
        kmv = khv = None
        scalar_rows = None
        if cfg.km_opt in (2, 3):
            # The closure's vertical pair, filled by launch_wrf_tke_km /
            # launch_wrf_smag3d_km above.  Interior vertical scalar mixing
            # covers WRF's rt_tendf row (theta reconstructed from thp+thb)
            # and every moist species (vertical_diffusion_2's moist_loop).
            nz_, ny_, nx_ = km.shape
            kmv = state.scratch((nz_, ny_, nx_), "smag_kmv")
            khv = state.scratch((nz_, ny_, nx_), "smag_khv")
            scalar_rows = [
                (f, buffers[slot], slot == "smag_rth")
                for f, _tend, _xk, _c1, _c2, slot, stag in specs
                if stag == ""
            ]
        launch_wrf_smag2d_vertical(
            state, cfg, km,
            ru=buffers["smag_ru"],
            rv=buffers["smag_rv"],
            rw=buffers["smag_rw"],
            rth=buffers["smag_rth"],
            rqv=buffers.get("smag_rqv"),
            time_t=time_t,
            kmv=kmv, khv=khv, scalar_rows=scalar_rows,
        )
        if cfg.km_opt == 2:
            # Vertical TKE self-diffusion with 2*Km_v (vertical_diffusion_2
            # :4332-4341, doubling :4896-4904), PBL-off only like the rest
            # of vertical_diffusion_2.  Same vertical_diffusion_s operator
            # as the scalars, with Km_v in place of Kh_v and the doubled
            # increment.
            tke_t = getattr(state, "tke0" if time_t else "tke")
            rtke = state.scratch((nz_, ny_, nx_), "smag_rtke")
            tmp = state.scratch((nz_, ny_, nx_), "smag_tke_tmp")
            tmp[...] = 0
            common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
            tail = [np.int32(nz_), np.int32(ny_), np.int32(nx_),
                    np.int32(state.phb.ndim == 3),
                    np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
            mass_grid = ((nx_ + _TPB - 1) // _TPB, ny_, nz_)
            get_kernel("smag2d", "wrf_smag_vd_s")(
                mass_grid, (_TPB, 1, 1),
                tuple(common + [
                    tke_t, state.thb, np.int32(0),
                    np.int32(state.thb.ndim == 3), kmv, tmp,
                ] + tail))
            _zero_open_strips(tmp, cfg, 1)
            rtke += 2.0 * tmp
            budget_v = tke_budget.term(state, cfg, "diffusion_v")
            if budget_v is not None:
                budget_v[...] = 2.0 * tmp
        for f, _tend, _xk, _c1, _c2, slot, _stag in specs:
            _zero_open_strips(
                buffers[slot], cfg, 1)
        _restore_open_ring_surface_fluxes(
            state, cfg, km, ru=buffers["smag_ru"], rv=buffers["smag_rv"],
            rth=buffers["smag_rth"], rqv=buffers.get("smag_rqv"),
            time_t=time_t)


def _add_open_ring(target, plane, cfg: RunConfig) -> None:
    """``target += plane`` on the width-1 outer ring ``_zero_open_strips``
    clears at radiative open boundaries; each ring word once (no corner is
    added twice), in place on views."""
    if cfg.open_x:
        target[:, :1] += plane[:, :1]
        target[:, -1:] += plane[:, -1:]
    if cfg.open_y:
        lo, hi = (1, -1) if cfg.open_x else (None, None)
        target[:1, lo:hi] += plane[:1, lo:hi]
        target[-1:, lo:hi] += plane[-1:, lo:hi]


def _restore_open_ring_surface_fluxes(state: DomainState, cfg: RunConfig,
                                      km, *, ru, rv, rth, rqv,
                                      time_t: bool) -> None:
    """Put WRF's surface-flux words back on the open-boundary ring.

    ``vertical_diffusion_2``'s surface arms (vflux, hflux, qflux) loop
    i = its..min(ite,ide-1) (u faces to ite, v faces to jte) with NO open
    trim (module_diffusion_em.F:4100-4110, :4161-4400), while its interior
    operators and ``horizontal_diffusion_2`` skip the outer row/column at an
    open boundary.  So at an open boundary WRF's lowest-level ``*_tendf`` on
    the outer ring is exactly ``0 + surface flux``, and ``advance_mu_t`` /
    ``rk_update_scalar`` / ``advance_uv`` advance that ring from it (their
    bounds trim only specified/nested domains, module_small_step_em.F:
    1048-1060, module_em.F:1657-1673).  The mixing package clears the ring
    after the vertical pass (:func:`_zero_open_strips`), which dropped the
    surface momentum, heat and moisture fluxes on the outermost ring of
    every PBL-off open-boundary run (km1 column oracle, 2026-10-07).  This
    re-adds them: the surface arms alone into k = 0 planes borrowed from
    the face workspaces (free between the mixing package and diff6), then
    ``+=`` on the cleared ring, which is WRF's ``0 + S`` word for word.
    Specified and nested rings are overwritten by the boundary update in
    both models and are left cleared.
    """
    if _boundary_forced(cfg) or not (cfg.open_x or cfg.open_y):
        return
    nz, ny, nx = km.shape
    flat_x = state.scratch((nz, ny, nx + 1), "diff6_x").reshape(-1)
    flat_y = state.scratch((nz, ny + 1, nx), "diff6_y").reshape(-1)
    nu, nv, nm = ny * (nx + 1), (ny + 1) * nx, ny * nx
    planes = {
        "u": flat_x[:nu].reshape((1, ny, nx + 1)),
        "v": flat_y[:nv].reshape((1, ny + 1, nx)),
        "th": flat_x[nu:nu + nm].reshape((1, ny, nx)),
        "qv": flat_y[nv:nv + nm].reshape((1, ny, nx)),
    }
    for plane in planes.values():
        plane[...] = 0
    launch_wrf_smag2d_vertical(
        state, cfg, km, ru=planes["u"], rv=planes["v"], rw=None,
        rth=planes["th"], rqv=planes["qv"] if rqv is not None else None,
        time_t=time_t, interior=False)
    for target, plane in ((ru, planes["u"]), (rv, planes["v"]),
                          (rth, planes["th"]), (rqv, planes["qv"])):
        if target is None:
            continue
        _add_open_ring(target[0], plane[0], cfg)


def _compute_wrf_constant_k_tendencies(state: DomainState, cfg: RunConfig,
                                       km, kh, specs, *, time_t: bool) -> None:
    """km_opt=1's diff_opt=2 package in module_first_rk_step_part2.F order.

    WRF adds ``vertical_diffusion_2`` (interior stresses and fluxes, then
    the surface arms) into the zeroed ``*_tendf`` before
    ``horizontal_diffusion_2`` adds its one term per point (:1011-1100).
    At the surface level that is a three-term sum, which is not
    associative, so the order is kept rather than commuted: the tensors
    live in the face workspaces (:func:`launch_wrf_isotropic_km`), the
    vertical u/v/scalar package and its surface arms go into the zeroed
    carrying buffers first, and the horizontal operators then add into
    them.  w has no surface arm (vertical + horizontal commute), so its
    buffer, which holds D11 until the u/v stresses are built, is zeroed and
    filled after them.

    The vertical w stress takes WRF's own coefficient here, xkmh = khdif
    (vertical_diffusion_2 hands xkmh to vertical_diffusion_w_2), the same K
    as the tau11/tau22 normal stresses, so the constant-K stress tensor
    stays trace-free when khdif != kvdif.  (km_opt=2/3 keep their declared
    xkmv choice; see :func:`launch_wrf_smag2d_vertical`.)
    """
    deformation = launch_wrf_isotropic_km(state, cfg, km, kh, time_t=time_t)
    buffers = {slot: state.scratch(f.shape, slot)
               for f, _tend, _xk, _c1, _c2, slot, _stag in specs}
    for slot, buf in buffers.items():
        if slot != "smag_rw":                  # D11 lives in its prefix
            buf[...] = 0
    pbl_off = cfg.bl_pbl_physics == 0
    kmv = state.scratch(km.shape, "smag_kmv")
    khv = state.scratch(km.shape, "smag_khv")
    mix2_exempt = chem_mix2_exempt_slots(state, cfg)
    if pbl_off:
        launch_wrf_smag2d_vertical(
            state, cfg, km,
            ru=buffers["smag_ru"], rv=buffers["smag_rv"], rw=None,
            rth=buffers["smag_rth"], rqv=buffers.get("smag_rqv"),
            time_t=time_t, kmv=kmv, khv=khv,
            scalar_rows=[(f, buffers[slot], slot == "smag_rth")
                         for f, _tend, _xk, _c1, _c2, slot, stag in specs
                         if stag == ""])
    for f, _tend, xk, _c1, _c2, slot, stag in specs[:2]:
        launch_wrf_smag2d_hd(state, cfg, f, xk, buffers[slot], stagger=stag,
                             time_t=time_t, deformation=deformation)
    # D11/D22/D12 are dead: the w buffer and the face workspaces are free.
    f, _tend, xk, _c1, _c2, slot, stag = specs[2]
    rw = buffers[slot]
    rw[...] = 0
    launch_wrf_smag2d_hd(state, cfg, f, xk, rw, stagger=stag, time_t=time_t)
    if pbl_off:
        launch_wrf_smag2d_vertical(
            state, cfg, km, ru=None, rv=None, rw=rw, rth=None, rqv=None,
            time_t=time_t, kmv=kmv, w_km=km, surface=False)
    for f, _tend, xk, _c1, _c2, slot, stag in specs[3:]:
        if slot in mix2_exempt:
            continue
        launch_wrf_smag2d_hd(state, cfg, f, xk, buffers[slot], stagger=stag,
                             time_t=time_t, full_theta=(slot == "smag_rth"))
    for buf in buffers.values():
        _zero_open_strips(buf, cfg, 1)
    if pbl_off:
        _restore_open_ring_surface_fluxes(
            state, cfg, km, ru=buffers["smag_ru"], rv=buffers["smag_rv"],
            rth=buffers["smag_rth"], rqv=buffers.get("smag_rqv"),
            time_t=time_t)


def prepare_fixed_tendencies(state: DomainState, cfg: RunConfig,
                             physics_tendencies=None) -> None:
    """Build WRF's time-t ``*_tendf``/``scalar_tends`` once per step.

    ``module_first_rk_step_part2`` computes km_opt=4 mixing and sixth-order
    diffusion from the RK-step-1 (time-t) fields.  ``rk_addtend_dry`` and
    ``rk_update_scalar[_pd]`` then consume the same fixed tendencies on all
    three RK passes.  woof shares the existing ``smag_r*`` carrying
    buffers between both source packages, while one ``diff6_*`` temporary
    per staggering preserves their different open-boundary widths without
    retaining a second full set of per-species buffers.
    """
    nz, ny, nx = state.p.shape
    include_smag = wrf_mixing_package_active(cfg)
    include_diff6 = cfg.diff_6th_opt > 0
    fold = _wrf_relaxation_fold(cfg)
    if not (include_smag or include_diff6 or fold or WRF_EXACT):
        return

    # The K arrays are needed only by Smagorinsky.  Scalar placeholders let
    # _smag2d_specs describe the common carrying buffers without allocating
    # two mass-grid K fields on a diff6-only run.
    if include_smag:
        km = state.scratch((nz, ny, nx), "smag_km")
        kh = state.scratch((nz, ny, nx), "smag_kh")
    else:
        km = kh = None
    specs = _smag2d_specs(state, km, kh, time_t=True,
                          kmv=_horizontal_w_km(state, cfg))
    for f0, _tend, _xk, _c1, _c2, slot, _stag in specs:
        state.scratch(f0.shape, slot)[...] = 0
    if cfg.km_opt == 2:
        # The prognostic-TKE forward tendency (tke_rhs + self-diffusion),
        # consumed by advance_tke_stage on every RK pass.
        state.scratch((nz, ny, nx), "smag_rtke")[...] = 0
        # A term this configuration never produces must read as an accurate
        # zero for the step, not as the previous step's value.
        tke_budget.clear_fields(state, cfg)

    mu_t = state.mub2d + state.mup0
    if include_smag:
        _compute_wrf_smag_tendencies(
            state, cfg, km, kh, specs, time_t=True)
    if WRF_EXACT and physics_tendencies is not None:
        _fold_physics_wrf(state, specs, physics_tendencies)

    if include_diff6:
        to_edge = diff6_to_edge(cfg)
        temp_slot = {"x": "diff6_x", "y": "diff6_y",
                     "z": "diff6_z", "": "diff6_m"}
        exempt = diff6_exempt_slots(cfg)
        # row[5] is the row's carrying slot (see _smag2d_specs).
        diff6_rows = [row for row in specs if row[5] not in exempt]
        if cfg.km_opt == 2:
            # WRF applies the 6th-order filter to tke through
            # rk_scalar_tend unless tke_mix6_off (Registry default
            # .false., Registry.EM_COMMON:2893).
            diff6_rows.append((state.tke0, None, None, state.c1h,
                               state.c2h, "smag_rtke", ""))
        edge = ({"work": _diff6_edge_work(state, cfg)} if to_edge
                else None)
        for f0, _tend, _xk, c1, c2, slot, stag in diff6_rows:
            if _tend is None and _strict_defers_scalar_diff6(cfg):
                continue
            maps = dict(phb=state.phb, msfu=state.msfu, msfv=state.msfv,
                        msft=state.msft)
            if slot == "smag_rtke":
                budget_6 = tke_budget.term(state, cfg, "diffusion_6th")
                if budget_6 is not None:
                    # The budget records the filter's own increment, so
                    # it is formed on a zero tendency of its own.
                    tmp = state.scratch(f0.shape, temp_slot[stag])
                    tmp[...] = 0
                    add_diff6_row(cfg, slot, f0, tmp, mu_t, c1, c2, stag,
                                  edge=edge, **maps)
                    budget_6[...] = tmp
            add_diff6_row(cfg, slot, f0, state.scratch(f0.shape, slot),
                          mu_t, c1, c2, stag, edge=edge, **maps)

    # rk_addtend_dry's 1/msf, taken once over the sum both packages left in
    # the dry slots, after diff6 has accumulated (WRF divides ru_tendf etc.
    # once, module_em.F:1043, :1054, :1065, :1078).  In strict mode on a
    # boundary-forced domain the stage-1 relaxation joins the sum first, so
    # the division waits for fold_lateral_relaxation.
    if not fold:
        _couple_dry_mixing_map_factor(state, specs)


#: Strict mode: WRF's ``grid%muts`` as the last acoustic loop left it,
#: ``MUT + MU''`` (advance_mu_t), which every later calc_p_rho_phi reads
#: until the next acoustic loop (restart class: carried, woof/io/restart.py).
from woof.state_serialization_contract import WRF_MUTS_SLOT


def _wrf_muts(state: DomainState):
    """The carried strict ``muts``, or ``None`` (default mode, first step)."""
    return state.existing_scratch(WRF_MUTS_SLOT) if WRF_EXACT else None


def _init_wrf_acoustic_muts(state: DomainState, istage: int) -> None:
    """small_step_prep's MUTS: ``MUB+MU_2`` on rk_step 1, ``MUB+MU_1`` after.

    MU_1 is the time-t mass (state.mup0); MU_2 the stage reference.
    """
    cp.add(state.mub2d, state.mup if istage == 0 else state.mup0,
           out=wrf_acoustic_muts(state))
    state._wrf_acoustic_muts_initialized = True


def _capture_wrf_muts(state: DomainState, cfg: RunConfig) -> None:
    """WRF's grid%muts after the acoustic loop, for every later diagnosis.

    ``MUT + MU''`` where advance_mu_t integrates; on the specified ring the
    spec_bdyupdate carrier the strict ring kernel advanced.  A domain with
    no ring has no carrier to keep.
    """
    muts = state.scratch(state.mup.shape, WRF_MUTS_SLOT)
    full = cp.add(cp.add(state.mub2d, state.mup), state.mu_pp)
    if _mass_w_boundary_zone(cfg):
        muts[...] = wrf_acoustic_muts(state)
    else:
        muts[...] = full
        wrf_acoustic_muts(state)[...] = full


def _wrf_relaxation_fold(cfg: RunConfig) -> bool:
    """Strict mode folds the lateral relaxation into the held tendency."""
    return WRF_EXACT and bool(cfg.specified or cfg.nested)


#: Held dry slot and the map factor WRF's rk_addtend_dry scales each folded
#: relaxation row by before dividing the sum (module_em.F: u_save*msfuy,
#: v_save*msfvx, w_save*msfty, t_save unscaled).
_FOLD_ROWS = {"u": ("smag_ru", "msfu"), "v": ("smag_rv", "msfv"),
              "w": ("smag_rw", "msft"), "theta": ("smag_rth", None)}


def fold_lateral_relaxation(state: DomainState, cfg: RunConfig) -> None:
    """Strict mode: WRF ``rk_addtend_dry`` on rk_step 1, before any stage adds.

    ``relax_bdy_dry`` writes the stage-1 relaxation into ``u_save`` etc.,
    and ``rk_addtend_dry`` folds it into the held tendency before dividing:
    ``ru_tendf = ru_tendf + u_save*msfuy`` then ``ru_tend + ru_tendf/msfuy``
    on every stage (module_em.F).  The default path adds the relaxation to
    each stage's tendency on its own, a different rounding in the
    relaxation zone (combo-sweep round 2, LOCALIZE.md item 5).  Here the
    held slots, still undivided (:func:`prepare_fixed_tendencies`), take
    the relaxation in WRF's order and then the one division.
    """
    specs = _smag2d_specs(state, None, None, time_t=True)
    shapes = {row[5]: row[0].shape for row in specs}
    for name, relax in capture_folded_relaxation(state, cfg).items():
        slot_name, msf_name = _FOLD_ROWS[name]
        slot = state.scratch(shapes[slot_name], slot_name)
        if msf_name is not None and state.has_msf:
            slot += relax * getattr(state, msf_name)[None]
        else:
            slot += relax
    _couple_dry_mixing_map_factor(state, specs)


def _fold_physics_wrf(state: DomainState, specs, physics_tendencies) -> None:
    """Strict mode: update_phy_ten's terms in the held ``*_tendf``.

    WRF adds the coupled physics tendencies into ``ru_tendf``/``rv_tendf``/
    ``t_tendf``/``moist_tend`` in first_rk_step_part2 (update_phy_ten,
    radiation then PBL then cumulus), and the km_opt mixing then adds onto
    them; rk_addtend_dry divides the whole held sum by the map factor once
    on every stage.  The default path keeps the physics apart, already
    divided, and adds it to each stage's tendency, a different rounding in
    21197 u, 21510 v and 45430 theta words of the first RK stage
    (combo-sweep round 3, A088).  The mixing slot holds the horizontal
    term alone here, so this one add is WRF's (commutative) single add.
    """
    rows = {"smag_ru": physics_tendencies.ru, "smag_rv": physics_tendencies.rv,
            "smag_rth": physics_tendencies.rtheta,
            "smag_rw": physics_tendencies.rw}
    for f0, _tend, _xk, _c1, _c2, slot, _stag in specs:
        if slot in rows:
            add = rows[slot]
        else:
            add = physics_tendencies.scalar_for(slot[len("smag_r"):])
        if add is not None:
            state.scratch(f0.shape, slot)[...] += add


def _strict_defers_scalar_diff6(cfg: RunConfig) -> bool:
    """Strict mode moves the scalar rows' sixth order to RK stage 1's end."""
    return WRF_EXACT and cfg.diff_6th_opt > 0 and not diff6_to_edge(cfg)


def add_scalar_diff6_wrf(state: DomainState, cfg: RunConfig) -> None:
    """Strict mode: ``rk_scalar_tend``'s sixth-order filter, WRF's timing.

    WRF filters moisture, TKE and chem inside ``rk_scalar_tend`` on
    rk_step 1 (module_em.F), which solve_em calls AFTER the first acoustic
    loop with ``grid%muts`` as the routine's ``mut`` (solve_em.F,
    moist_scalar_advance).  So the scalar filter couples its fluxes with
    the post-acoustic mass ``MUT + MU''``, not the time-t mass the dry
    rows use.  Computed with the time-t mass it differed from WRF's held
    qv tendency in 35282 of 80000 words, by about mu''/mu (combo-sweep
    round 3, case p1).  The field is still the time-t scalar.  Each row is
    accumulated into its held slot in WRF's in-place order, after the
    km_opt mixing that first_rk_step_part2 left there.
    """
    if not _strict_defers_scalar_diff6(cfg):
        return
    muts = _wrf_muts(state)
    if muts is None:
        raise RuntimeError(
            "strict scalar sixth order needs the acoustic loop's muts")
    specs = _smag2d_specs(state, None, None, time_t=True)
    exempt = diff6_exempt_slots(cfg)
    rows = [row for row in specs if row[1] is None and row[5] not in exempt]
    if cfg.km_opt == 2:
        rows.append((state.tke0, None, None, state.c1h, state.c2h,
                     "smag_rtke", ""))
    for f0, _tend, _xk, c1, c2, slot, stag in rows:
        tmp = state.scratch(f0.shape, "diff6_m")
        _accumulate_diff6_wrf_order(state, cfg, f0, tmp, muts, c1, c2, slot,
                                    stag)


def _diff6_launch_kwargs(state: DomainState, cfg: RunConfig, slot: str,
                         stag: str) -> dict:
    return dict(stagger=stag, phb=state.phb, msfu=state.msfu, msfv=state.msfv,
                msft=state.msft, slopeopt=cfg.diff_6th_slopeopt,
                thresh=cfg.diff_6th_thresh, dx=cfg.dx, dy=cfg.dy,
                bnd_x=_boundary_x(cfg), bnd_y=_boundary_y(cfg))


def _accumulate_diff6_wrf_order(state: DomainState, cfg: RunConfig, f0, tmp,
                                mu_t, c1, c2, slot: str, stag: str) -> None:
    """Strict mode: add the 6th-order filter into the held tendency in WRF's order.

    WRF ``sixth_order_diffusion`` updates the held tendency in place,
    ``tendency = tendency + tendency_x + tendency_y``
    (module_big_step_utilities_em.F:6624), so the held value enters the
    first addition.  The default path forms ``(0 + tendency_x) +
    tendency_y`` in a temporary and adds that, a different rounding in
    ~3 % of words (combo-sweep round2 LOCALIZE.md, cause 3).  Here the
    temporary starts as a copy of the held tendency, the kernel and the
    seam face accumulate into it, and WRF's skipped open strip keeps the
    held value untouched (WRF's loop bounds never write it).
    """
    target = state.scratch(f0.shape, slot)
    tmp[...] = target
    launch_diff6(f0, tmp, mu_t, c1, c2, _diff6_factor(cfg, slot),
                 _diff6_dt(cfg, slot), cfg.diff_6th_opt, seam_base=target,
                 **_diff6_launch_kwargs(state, cfg, slot, stag))
    _restore_open_strips(tmp, target, cfg, 3)
    target[...] = tmp
    if slot == "smag_rtke":
        budget_6 = tke_budget.term(state, cfg, "diffusion_6th")
        if budget_6 is not None:
            budget_6[...] = 0
            launch_diff6(f0, budget_6, mu_t, c1, c2, _diff6_factor(cfg, slot),
                         _diff6_dt(cfg, slot), cfg.diff_6th_opt,
                         **_diff6_launch_kwargs(state, cfg, slot, stag))
            _zero_open_strips(budget_6, cfg, 3)


def add_fixed_dry_tendencies(state: DomainState, cfg: RunConfig) -> None:
    """Add the held time-t forward tendencies to one RK slow pass.

    WRF ``rk_addtend_dry`` divides each held ``*_tendf`` by the target's
    own map factor here (module_em.F:1043 ``/msfuy``, :1054
    ``*msfvx_inv``, :1065 and :1078 ``/msfty``).  woof takes that
    division once per step, at the end of :func:`prepare_fixed_tendencies`
    -- see :func:`_couple_dry_mixing_map_factor` -- over the sum the
    mixing package and diff6 left in the shared carrying buffer, both of
    which carry WRF's map factor into it.
    """
    if not wrf_mixing_package_active(cfg) and cfg.diff_6th_opt <= 0:
        return
    # K values are not consumed here; the specs provide shapes/targets.
    pairs = tuple((state.scratch(f0.shape, slot), tend)
                  for f0, tend, _xk, _c1, _c2, slot, _stag in _smag2d_specs(
                      state, None, None, time_t=True) if tend is not None)
    launch = prepare_add_arrays(state, pairs)
    if launch is None:
        for src, tend in pairs:
            tend += src
    else:
        launch()


def fixed_scalar_tendencies(state: DomainState, cfg: RunConfig):
    """Return held scalar forward tendencies by Registry field name."""
    if (state.qv is None and getattr(state, "qke_adv", None) is None) or (not wrf_mixing_package_active(cfg)
                            and cfg.diff_6th_opt <= 0):
        return None
    names = [name for name in SPECIES if getattr(state, name, None) is not None]
    names += list(extra_moist_species(state))
    shape = state.p.shape
    return {name: state.scratch(shape, "smag_r" + name) for name in names}


def add_smag2d_tendencies(state: DomainState, cfg: RunConfig,
                          first: bool) -> None:
    """WRF km_opt=4 metric-aware ``diff_opt=2`` Smagorinsky mixing.

    WRF timing semantics (module_first_rk_step_part2 + rk_addtend_dry): the
    deformation, K, and the mixing tendencies are computed ONCE per model
    step on RK stage 1 (``first=True``) from the time-t fields into the
    forward-tendency scratch buffers, then ADDED to every stage's slow
    tendencies.  Momentum mixes with K_m; WRF's ``theta - T0`` field and
    moisture mix with K_h = K_m/prandtl = 3*K_m.  woof reconstructs the
    theta field from ``thp + thb - T0`` inside the scalar-flux kernel.  The
    production kernels carry
    WRF's geopotential metrics, map factors, terrain-coordinate deformation,
    slope-limited K, tensor momentum stress, and metric scalar fluxes.
    """
    nz, ny, nx = state.p.shape
    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    specs = _smag2d_specs(state, km, kh,
                          kmv=_horizontal_w_km(state, cfg))
    if first:
        _compute_wrf_smag_tendencies(
            state, cfg, km, kh, specs, time_t=False)
    pairs = tuple((state.scratch(f.shape, slot), tend)
                  for f, tend, _xk, _c1, _c2, slot, _stag in specs
                  if tend is not None)
    launch = prepare_add_arrays(state, pairs)
    if launch is None:
        for src, tend in pairs:
            tend += src
    else:
        launch()


def apply_smag2d_moisture(state: DomainState, cfg: RunConfig,
                          dt_eff: float) -> None:
    """Fold the stage's km_opt=4 moisture mixing into qv/qc/qr.

    WRF ``rk_update_scalar`` advances scalars with ``advect_tend +
    sc_tend`` in one coupled update; ``advance_scalars_stage`` applied the
    advective part, so adding ``dt_eff*sc_tend/C(mu_new)`` here completes
    the identical sum.  Deviation from WRF on the PD final stage: WRF's
    ``advect_scalar_pd`` folds the accumulated non-advective tendencies --
    this mixing included -- into its provisional low-order state
    ``ph_low``, so WRF's PD limiter DOES see the mixing when it
    renormalizes the fluxes; woof instead adds the mixing after the
    PD-limited advective update, outside the limiter's positivity guard.
    The mixing term itself is identical either way, but any negative
    excursion it produces here is not renormalized away -- the same order
    of deviation ``apply_diff6`` documents for the moisture increment
    (the K_h fluxes are down-gradient, so the excursions stay at
    rounding level in practice; tests/test_smag2d.py gates q >= -1e-6).
    """
    nz, ny, nx = state.p.shape
    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    for name in SPECIES:
        q = getattr(state, name)
        q += dt_eff * state.scratch((nz, ny, nx), "smag_r" + name) / chm
    for name in extra_moist_species(state):
        q = getattr(state, name)
        q += dt_eff * state.scratch((nz, ny, nx), "smag_r" + name) / chm


def _prepare_small_step_init_launch(state: DomainState, cfg: RunConfig,
                                    rk_step: int = 1):
    """Bind the two invariant small-step initialization launches.

    ``cfg`` is consumed for one thing only: whether each horizontal axis is
    periodic.  The uv kernel builds WRF's ``muu``/``muus`` inline and
    ``calc_mu_uv`` picks the boundary face's off-domain neighbour from
    ``periodic_x``/``periodic_y`` (module_big_step_utilities_em.F:59-115),
    so the flag has to reach the kernel; WOOF's spelling of "not periodic"
    is ``_boundary_x``/``_boundary_y`` (open, specified, or nested), the same
    predicate ``slow_pgf`` and the advection launchers already pass.
    """
    nz, ny, nx = state.p.shape
    block = (256,)
    uv_n = nz * (ny + 1) * (nx + 1)
    uv_grid = ((uv_n + 255) // 256,)
    uv_kernel = get_kernel("dycore", "small_step_init_uv")
    uv_args = (
        state.u_pp, state.v_pp, state.u0, state.v0, state.u, state.v,
        state.mup0, state.mup, state.mub2d, state.c1h, state.c2h,
        state.msfu, state.msfv_inv if WRF_EXACT else state.msfv, np.int32(state.has_msf),
        np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
        np.int32(nz), np.int32(ny), np.int32(nx),
    )
    if WRF_EXACT:
        uv_args += (np.int32(rk_step),)

    column_grid = ((ny * nx + 255) // 256,)
    column_kernel = get_kernel("dycore", "small_step_init_column")
    column_args = (
        state.w_pp, state.th_pp, state.ph_pp, state.mu_pp, state.al_pp,
        state.p_pp, state.p_pp_old, state.w0, state.w, state.thp0,
        state.thp, state.php0, state.php, state.p, state.alt, state.mup0,
        state.mup, state.mub2d, state.thb, state.c1h, state.c2h,
        state.c1f, state.c2f, state.rdnw, state.msft,
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(nz), np.int32(ny), np.int32(nx),
    )

    def launch() -> None:
        uv_kernel(uv_grid, block, uv_args)
        column_kernel(column_grid, block, column_args)

    return launch


def _init_small_steps(state: DomainState, cfg: RunConfig) -> None:
    """WRF ``small_step_prep`` + ``calc_p_rho`` (step 0).

    The acoustic perturbations are the deviations of the *time-t* fields
    (the *0 copies) from the current stage reference t*, coupled by the
    matching hybrid column-mass increments (``c1h*mu + c2h`` on half
    levels, ``c1f*mu + c2f`` on full levels); p''/alpha'' are then
    diagnosed from the linearized EOS so the first substep's pressure
    gradient is consistent.  On stage 1 (t* = t) every perturbation is
    exactly zero.
    """
    _prepare_small_step_init_launch(
        state, cfg, int(getattr(cfg, "rk_step", 1)))()


def _prepare_small_step_finish_launch(state: DomainState, cfg: RunConfig,
                                      hdiab_dt: float = 0.0):
    """Bind the two invariant small-step finish launches.

    ``cfg`` carries the same periodic-axis decision the init launcher
    documents: ``small_step_finish`` divides by the very ``muu``/``muus``
    pair ``small_step_prep`` multiplied by (module_small_step_em.F:96-98),
    so the two kernels must agree on the boundary face's neighbour or the
    coupling is not undone.
    """
    nz, ny, nx = state.p.shape
    block = (256,)
    uv_n = nz * (ny + 1) * (nx + 1)
    uv_grid = ((uv_n + 255) // 256,)
    uv_kernel = get_kernel("dycore", "small_step_finish_uv")
    uv_args = (
        state.u, state.v, state.u_pp, state.v_pp, state.mup, state.mu_pp,
        state.mub2d, state.c1h, state.c2h, state.msfu, state.msfv,
        np.int32(state.has_msf),
        np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
        np.int32(nz), np.int32(ny), np.int32(nx),
    ) + ((wrf_acoustic_muts(state),) if WRF_EXACT else ())

    h_diabatic = state.h_diabatic if hdiab_dt else state.p
    column_grid = ((ny * nx + 255) // 256,)
    column_kernel = get_kernel("dycore", "small_step_finish_column")
    column_args = (
        state.w, state.thp, state.php, state.mup, state.w_pp, state.th_pp,
        state.ph_pp, state.mu_pp, state.mub2d, state.thb, state.c1h,
        state.c2h, state.c1f, state.c2f, state.msft, h_diabatic,
        DTYPE(hdiab_dt), np.int32(bool(hdiab_dt)),
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(nz), np.int32(ny), np.int32(nx),
    ) + ((wrf_acoustic_muts(state),) if WRF_EXACT else ())

    def launch() -> None:
        uv_kernel(uv_grid, block, uv_args)
        column_kernel(column_grid, block, column_args)

    return launch


def _finish_small_steps(state: DomainState, cfg: RunConfig,
                        hdiab_dt: float = 0.0) -> None:
    """WRF ``small_step_finish``: fold the acoustic perturbations into the
    uncoupled prognostic fields, the new RK stage estimate.

    ``hdiab_dt`` engages WRF's h_diabatic removal on the FINAL RK step
    (module_small_step_em.F:408-426, the ``rk_step == rk_order`` branch):
    the coupled theta numerator drops
    ``dts*number_of_small_timesteps*(c1h(k)*mut+c2h(k))*h_diabatic``
    (:421), exactly the amount the substeps integrated from the
    ``add_h_diabatic_tendency`` term with the same stage mass ``mu_s``
    (WRF ``mut``), so theta(t+dt) carries NO net h_diabatic contribution
    and the heating enters the state once, in
    ``microphysics.moist_physics_finish``.  Callers pass the final stage's
    length ``nsub*dtau`` (= cfg.dt) there and 0.0 elsewhere (stages 1-2
    keep the heating in their provisional estimates, :408-415).  Float64
    mirror: ``woof.verify.npref.np_small_step_finish_theta``.
    """
    _prepare_small_step_finish_launch(state, cfg, hdiab_dt)()


def _advance_stage(state: DomainState, dt_eff: float) -> None:
    """Advection-only uncoupled prognostic update from the stage-0 fields.

    The tendencies are coupled (mass-weighted with the hybrid increments),
    so each variable q advances through C(mu_new)*q_new = C(mu_old)*q_old
    + dt_eff * r_q_t and is then uncoupled by the new column mass at its
    staggering.  With mu' frozen (acoustic=False) mu_old == mu_new and
    this reduces to q_new = q_old + dt_eff * r_q_t / C(mu).
    """
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    c1f = state.c1f[:, None, None]
    c2f = state.c2f[:, None, None]
    mu0 = state.mub2d + state.mup0                    # (ny, nx) stage-0 mass
    mu = state.mub2d + state.mup                      # (ny, nx) updated mass

    # Map factors (Task 3): the working tendencies are for the msf-coupled
    # variables (U = C*u/msfu, theta as (1/msft)*d(mu*theta)/dt), so
    # uncoupling multiplies each by its msf (identity by default).
    if state.has_msf:
        rth = state.rth_t * state.msft[None]
        rut = state.ru_t * state.msfu[None]
        rvt = state.rv_t * state.msfv[None]
        rwt = state.rw_t * state.msft[None]
    else:
        rth, rut, rvt, rwt = (state.rth_t, state.ru_t, state.rv_t,
                              state.rw_t)

    thb = _b3(state.thb)
    state.thp[...] = (((c1h * mu0[None] + c2h) * (thb + state.thp0)
                       + dt_eff * rth)
                      / (c1h * mu[None] + c2h)) - thb

    mux0, mux = mu_at_u_faces(mu0), mu_at_u_faces(mu)
    state.u[...] = (((c1h * mux0[None] + c2h) * state.u0
                     + dt_eff * rut) / (c1h * mux[None] + c2h))

    muy0, muy = mu_at_v_faces(mu0), mu_at_v_faces(mu)
    state.v[...] = (((c1h * muy0[None] + c2h) * state.v0
                     + dt_eff * rvt) / (c1h * muy[None] + c2h))

    state.w[...] = (((c1f * mu0[None] + c2f) * state.w0
                     + dt_eff * rwt) / (c1f * mu[None] + c2f))


def add_diff6_row(cfg: RunConfig, slot: str, f0, target, mut, c1, c2,
                  stagger: str, *, phb, msfu, msfv, msft,
                  edge: dict | None = None) -> None:
    """Add one field's sixth-order tendency onto its carrying slot, in place.

    This is the production row of :func:`prepare_fixed_tendencies`: the
    row's factor and dt (:func:`_diff6_factor`, :func:`_diff6_dt`), the
    run's slope taper, boundary flags and form.  WRF accumulates onto
    everything its ``*_tendf`` already holds (sixth_order_diffusion :6626,
    ``tendency = tendency + tendency_x + tendency_y``), so ``target`` is
    the slot itself, never a separately formed increment added later: that
    rounds ``T0 + (tx + ty)`` instead of WRF's ``(T0 + tx) + ty``, which the
    WRF 4.6.1 column oracle (tools/wrf_diffusion_oracle/diff6_wrf461_*)
    measured on 127,083 of 1,101,792 words, up to 32,768 ULP away.
    """
    launch = launch_diff6_to_edge if edge is not None else launch_diff6
    launch(f0, target, mut, c1, c2, _diff6_factor(cfg, slot),
           _diff6_dt(cfg, slot), cfg.diff_6th_opt, stagger=stagger,
           phb=phb, msfu=msfu, msfv=msfv, msft=msft,
           slopeopt=cfg.diff_6th_slopeopt, thresh=cfg.diff_6th_thresh,
           dx=cfg.dx, dy=cfg.dy,
           # Boundary-aware reads and WRF's loop bounds: the outermost
           # computed staggered face takes WRF's accurate boundary datum
           # (u ide-3 / v jde-3) and the points WRF's loops skip keep their
           # incoming tendency.  The fork's edge form needs neither.
           bnd_x=_boundary_x(cfg), bnd_y=_boundary_y(cfg),
           **({} if edge is None else edge))


def diff6_loop_bounds(nx: int, ny: int, nxs: int, nys: int,
                      bnd_x: bool, bnd_y: bool) -> tuple[int, int, int, int]:
    """The diff6 main kernel's inclusive stored-index bounds (i_lo, i_hi,
    j_lo, j_hi): WRF sixth_order_diffusion's loop bounds.

    On a periodic axis every stored point (the redundant staggered
    column/row included).  On a forced or open axis WRF computes from
    ids+3 to ide-4 on mass-like positions and to ide-3 on the
    boundary-normal staggered faces (module_big_step_utilities_em.F
    :6327-6440); the main kernel takes 3..n-4 on both, and the staggered
    face n-3, whose stencil reads the true boundary datum field(ide), is
    computed by the seam kernel.  Points outside keep their incoming
    tendency, as the Fortran's loops leave them.

    One deliberate difference: WRF's 'v' branch also trims the y bounds
    when only open_xs is set (``IF ( config_flags%open_xs .or. specified
    )`` at :6378, where the u, w and mass branches test ``specified``
    alone).  That makes an open-x, periodic-y run skip v's three edge rows
    on each side of a PERIODIC axis.  It is a WRF defect and is not copied:
    a periodic axis here is always computed in full.
    """
    i_lo, i_hi = (3, nx - 4) if bnd_x else (0, nxs - 1)
    j_lo, j_hi = (3, ny - 4) if bnd_y else (0, nys - 1)
    return i_lo, i_hi, j_lo, j_hi


def launch_diff6(f, tend, mut, c1, c2, factor: float, dt: float, opt: int,
                 stagger: str = "", *, phb=None, msfu=None, msfv=None,
                 slopeopt: int = 0, thresh: float = 0.10,
                 dx: float = 0.0, dy: float = 0.0,
                 bnd_x: bool = False, bnd_y: bool = False, msft=None,
                 seam_base=None) -> None:
    """ADD the WRF 6th-order horizontal diffusion coupled tendency for one
    field into ``tend`` (kernels/diff6.cu; float64 mirror
    ``woof.verify.npref.np_diff6``).

    ``mut (ny, nx)`` is the total dry column mass the fluxes are coupled
    with (WRF ``mut``); ``c1``/``c2`` the hybrid coefficients at the
    field's levels (``c1h/c2h`` for half-level fields, ``c1f/c2f`` for w).
    ``opt`` is ``diff_6th_opt``: 2 zeroes any up-gradient flux (monotonic),
    any other positive value is the plain operator.  ``stagger`` selects
    the grid position as in ``launch_add_diff2``: ``""`` mass points,
    ``"x"`` u, ``"y"`` v, ``"z"`` w points (BC-pinned boundary levels get
    no tendency).  No dx/dy enters the untapered operator: it removes
    ``factor`` of a 2-D 2dx checkerboard's amplitude per full-``dt``
    integration by construction (``coef = factor/2^6/(2*dt)``, the Fortran
    normalization).

    ``slopeopt >= 1`` with a 3-D ``phb`` engages WRF's terrain-slope taper
    (``diff_6th_slopeopt``, sixth_order_diffusion
    module_big_step_utilities_em.F:6487-6501/6569-6583): each face flux is
    scaled by ``max(1 - dzmax/(thresh*9.81*dx), 0)`` with ``dzmax`` the
    msf-scaled ``phb`` face jump at the field's own level; ``dx``/``dy``
    must then be the physical grid spacings. ``msfu``/``msfv``/``msft``
    default to identity when omitted. The field's own map factor also
    multiplies its tendency, independently of the terrain-slope option.
    Constants and face-mass arithmetic retain compiled WRF REAL rounding.

    ``bnd_x``/``bnd_y`` (callers pass ``_boundary_x(cfg)``/``_boundary_y``:
    open or specified/nested forcing on that axis) apply WRF's loop
    bounds on that axis (:func:`diff6_loop_bounds`): points WRF's loops
    skip keep the incoming ``tend`` untouched, and the outermost computed
    staggered face -- WRF's u(ide-3)/v(jde-3), which the specified/nested
    and open loop bounds INCLUDE -- is computed by
    ``kernels/diff6_seam.cu`` with WRF's accurate read of the stored true
    boundary datum ``field(ide)``/``field(jde)``
    (module_big_step_utilities_em.F:6354-6358/:6381-6385 bounds,
    :6465-6467/:6547-6549 reads) instead of by the periodic-wrap main
    kernel.  ``tend`` may hold any incoming tendency: every computed point
    becomes ``(tend + tendency_x) + tendency_y``, the Fortran's in-place
    association, so production passes the carrying ``*_tendf`` slot
    itself.  The compiled WRF fixtures grade the interior and the two
    seam faces word for word (tests/test_diff6_wrf471_parity.py,
    tests/test_diff6_wrf461_column_oracle.py).
    """
    nlev, nys, nxs = f.shape
    nx = nxs - 1 if stagger == "x" else nxs
    ny = nys - 1 if stagger == "y" else nys
    variant = 1 if stagger == "x" else (2 if stagger == "y" else 0)
    bounds = diff6_loop_bounds(nx, ny, nxs, nys, bnd_x, bnd_y)
    coef = DTYPE(factor) * DTYPE(0.015625) / (DTYPE(2.0) * DTYPE(dt))
    slope = int(slopeopt) >= 1 and phb is not None and phb.ndim == 3
    if slope and (dx <= 0.0 or dy <= 0.0):
        raise ValueError("diff_6th_slopeopt >= 1 needs positive dx/dy")
    phb_arg = phb if slope else mut  # not dereferenced without slopeopt
    msfu_arg = (msfu if msfu is not None
                else cp.ones((ny, nx + 1), dtype=DTYPE))
    msfv_arg = (msfv if msfv is not None
                else cp.ones((ny + 1, nx), dtype=DTYPE))
    msft_arg = (msft if msft is not None
                else cp.ones((ny, nx), dtype=DTYPE))
    # WRF: dzthresh = diff_6th_thresh*9.81*dx (the routine's literal 9.81)
    # Its dx is reconstructed from REAL rdx, rather than kept in binary64.
    dzthr_x = (DTYPE(thresh) * DTYPE(9.81)
               * (DTYPE(1.0) / DTYPE(1.0 / dx))) if slope else DTYPE(0)
    dzthr_y = (DTYPE(thresh) * DTYPE(9.81)
               * (DTYPE(1.0) / DTYPE(1.0 / dy))) if slope else DTYPE(0)
    kern = get_kernel("diff6", "diff6")
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    kern(grid, (_TPB, 1, 1),
         (f, tend, mut, c1, c2, phb_arg, msfu_arg, msfv_arg, msft_arg,
          DTYPE(coef), np.int32(opt), np.int32(1 if slope else 0),
          dzthr_x, dzthr_y,
          np.int32(nlev), np.int32(ny), np.int32(nys),
          np.int32(nx), np.int32(nxs), np.int32(variant),
          np.int32(1 if stagger == "z" else 0),
          *(np.int32(b) for b in bounds)))
    if bnd_x and stagger == "x":
        _launch_diff6_seam("diff6_seam_u", f, tend, mut, c1, c2, phb_arg,
                           msfu_arg, msfv_arg, msft_arg, coef, opt, slope,
                           dzthr_x, dzthr_y, nlev, ny, nx, bnd_y,
                           base=seam_base)
    if bnd_y and stagger == "y":
        _launch_diff6_seam("diff6_seam_v", f, tend, mut, c1, c2, phb_arg,
                           msfu_arg, msfv_arg, msft_arg, coef, opt, slope,
                           dzthr_x, dzthr_y, nlev, ny, nx, bnd_x,
                           base=seam_base)


def _launch_diff6_seam(name, f, tend, mut, c1, c2, phb_arg, msfu_arg,
                       msfv_arg, msft_arg, coef, opt, slope, dzthr_x, dzthr_y,
                       nlev, ny, nx, bnd_cross, base=None) -> None:
    """Compute the WRF-computed high-side staggered face (kernels/
    diff6_seam.cu): u's east column nx-3 / v's north row ny-3, whose
    dflux_p1 reads the stored true boundary datum field(ide)/field(jde).

    The main periodic-wrap kernel would read the OPPOSITE boundary there,
    so its loop bounds (:func:`diff6_loop_bounds`) leave the face out and
    the seam kernel adds WRF's accurate arithmetic onto the incoming
    tendency over WRF's own index range -- the cross-axis range [3, n-4]
    when the cross axis is also forced (``bnd_cross``), the full periodic
    range otherwise.
    """
    seam_u = name == "diff6_seam_u"
    n_along, n_cross = (nx, ny) if seam_u else (ny, nx)
    if n_along < 6:                    # WRF bounds empty: ids+3 > ide-3
        return
    h0, h1 = (3, n_cross - 4) if bnd_cross else (0, n_cross - 1)
    if h1 < h0:
        return
    if not DIFFUSION_ENABLED:
        if seam_u:                         # drop the wrapped-read value
            tend[:, :, nx - 3] = 0 if base is None else base[:, :, nx - 3]
        else:
            tend[:, ny - 3, :] = 0 if base is None else base[:, ny - 3, :]
    kern = get_kernel("diff6_seam", name)
    span = h1 - h0 + 1
    kern(((span + _TPB - 1) // _TPB, 1, nlev), (_TPB, 1, 1),
         (f, tend, mut, c1, c2, phb_arg, msfu_arg, msfv_arg, msft_arg,
          DTYPE(coef), np.int32(opt), np.int32(1 if slope else 0),
          dzthr_x, dzthr_y,
          np.int32(nlev), np.int32(ny), np.int32(nx),
          np.int32(h0), np.int32(h1), np.int32(1 if bnd_cross else 0)))


#: Halo cells the edge-to-edge form pads on each low side; the high side
#: takes one more, so no read of a real point reaches the periodic wrap
#: of the padded core (a staggered high face reads index n + 6 of n + 7).
_DIFF6_EDGE_HALO = DIFF6_EDGE_HALO


def _edge_pad_into(dst: cp.ndarray, src: cp.ndarray,
                   low: int = _DIFF6_EDGE_HALO) -> cp.ndarray:
    """Zero-gradient copies on the two horizontal axes (WRF set_physical_bc3d
    under specified/nested boundaries, share/module_bc.F: dat(ids-1..ids-3)
    = dat(ids), dat(ide..ide+2) = dat(ide-1) on mass points, dat(ide+1..
    ide+3) = dat(ide) on a staggered axis): an exact copy of the edge.

    Writes every element of ``dst`` (shape ``src`` plus ``low`` cells on
    each low side and ``low + 1`` on each high side of the last two axes)
    from ``src`` alone: the value at padded (j, i) is ``src`` at the
    clipped index, which is what ``cp.pad(mode="edge")`` returns, and
    nothing is allocated.
    """
    ny, nx = src.shape[-2:]
    if dst.shape != src.shape[:-2] + (ny + 2 * low + 1, nx + 2 * low + 1):
        raise ValueError(f"edge pad of {src.shape} into {dst.shape}")
    j_lo, j_in, j_hi = slice(0, low), slice(low, low + ny), slice(low + ny, None)
    i_lo, i_in, i_hi = slice(0, low), slice(low, low + nx), slice(low + nx, None)
    dst[..., j_in, i_in] = src
    dst[..., j_in, i_lo] = src[..., :, :1]
    dst[..., j_in, i_hi] = src[..., :, -1:]
    dst[..., j_lo, i_in] = src[..., :1, :]
    dst[..., j_hi, i_in] = src[..., -1:, :]
    dst[..., j_lo, i_lo] = src[..., :1, :1]
    dst[..., j_lo, i_hi] = src[..., :1, -1:]
    dst[..., j_hi, i_lo] = src[..., -1:, :1]
    dst[..., j_hi, i_hi] = src[..., -1:, -1:]
    return dst


def _edge_take(buf: cp.ndarray, shape) -> cp.ndarray:
    """A shaped prefix of one flat edge-form workspace buffer."""
    n = math.prod(shape)
    if n > buf.size:
        raise ValueError(f"diff6 edge workspace holds {buf.size} values, "
                         f"a {shape} row needs {n}")
    return buf.reshape(-1)[:n].reshape(shape)


def _diff6_edge_work(state: DomainState, cfg: RunConfig) -> dict:
    """The edge form's declared workspace: the slots
    :func:`woof.core.diff6_edge_workspace.diff6_edge_slot_shapes` prices,
    drawn through ``DomainState.scratch`` (priced by the preflight's
    registry, listed in its lifetime audit)."""
    nz, ny, nx = state.p.shape
    n_field = diff6_edge_field_values(nz, ny, nx)
    work = {
        "field": state.scratch((n_field,), "diff6_edge_field"),
        "tend": state.scratch((n_field,), "diff6_edge_tend"),
        "planes": state.scratch((diff6_edge_planes_values(ny, nx),),
                                "diff6_edge_planes"),
        "phb": None,
    }
    if diff6_edge_slope(cfg) and state.phb.ndim == 3:
        work["phb"] = state.scratch((diff6_edge_phb_values(nz, ny, nx),),
                                    "diff6_edge_phb")
    return work


def _diff6_edge_transient_work(nlev: int, ny: int, nx: int,
                               slope: bool) -> dict:
    """Unpriced buffers for a direct launch with no ``DomainState``.

    Only the verification callers reach this (tests/test_diff6_fork_form.py
    and tools/wrf_diffusion_oracle/diff6_fork_oracle.py call
    :func:`launch_diff6_to_edge` on bare arrays); both forecast callers
    pass :func:`_diff6_edge_work`.  One flat allocation, partitioned.
    """
    n_field = diff6_edge_field_values(nlev, ny, nx)
    n_planes = diff6_edge_planes_values(ny, nx)
    n_phb = diff6_edge_phb_values(nlev, ny, nx) if slope else 0
    flat = cp.empty(2 * n_field + n_planes + n_phb, dtype=DTYPE)
    return {"field": flat[:n_field], "tend": flat[n_field:2 * n_field],
            "planes": flat[2 * n_field:2 * n_field + n_planes],
            "phb": flat[2 * n_field + n_planes:] if slope else None}


def launch_diff6_to_edge(f, tend, mut, c1, c2, factor, dt, opt, stagger="",
                         phb=None, msfu=None, msfv=None, msft=None,
                         slopeopt=0, thresh=0.10, dx=0.0, dy=0.0,
                         bnd_x=False, bnd_y=False, work=None) -> None:
    """ADD the NOAA WRFV3.9 fork's sixth-order tendency for one field.

    The fork's sixth_order_diffusion (HRRR v4.1.21 WRFV3.9
    module_big_step_utilities_em.F:6596-6632) loops over every point of
    the tile to the domain edge for every field and lets the stencil read
    the halo, which ``set_physical_bc3d`` fills with zero-gradient copies
    of the edge under specified and nested boundaries (share/module_bc.F,
    ``open_xs``/``open_xe`` and the y analogues).  WRF v4.6.1 instead
    stops three points short (:6327-6440), which :func:`launch_diff6`'s
    loop bounds reproduce.

    The arithmetic per point is :func:`launch_diff6`'s unchanged kernel:
    every input is padded with three edge copies on each side (one more
    on the high side), the incoming ``tend`` is copied into the padded
    core, the kernel accumulates there in place, and the real points are
    copied back, so ``tend`` receives the fork's own in-place sum.  No
    real point's stencil reaches the padded core's periodic wrap, so the
    result is the fork's halo read.
    ``bnd_x``/``bnd_y`` are accepted for the common call signature; the
    edge form needs no seam pass.  Callers use this form only on a domain
    whose two axes are both forced (see :func:`diff6_to_edge`).

    ``work`` holds the padded buffers.  The forecast callers pass
    :func:`_diff6_edge_work`, the declared workspace the run preflight
    prices; a bare-array caller (verification only) gets
    :func:`_diff6_edge_transient_work`.  The base geopotential is padded
    only when the slope taper reads it (otherwise the kernel never
    dereferences it), and an omitted map factor is written as identity
    straight into its padded plane.
    """
    del bnd_x, bnd_y
    nlev, nys, nxs = f.shape
    nx = nxs - 1 if stagger == "x" else nxs
    ny = nys - 1 if stagger == "y" else nys
    h = _DIFF6_EDGE_HALO
    pad = 2 * h + 1
    slope = int(slopeopt) >= 1 and phb is not None and phb.ndim == 3
    if work is None:
        work = _diff6_edge_transient_work(
            phb.shape[0] if slope else nlev, ny, nx, slope)
    pshape = (nlev, nys + pad, nxs + pad)
    fp = _edge_pad_into(_edge_take(work["field"], pshape), f)
    tp = _edge_take(work["tend"], pshape)
    tp.fill(0)
    # The incoming tendency enters the padded core, so every real point
    # becomes (tend + tendency_x) + tendency_y, the fork's in-place sum.
    tp[:, h:h + nys, h:h + nxs] = tend
    padded = []
    offset = 0
    for src, shape in zip((mut, msfu, msfv, msft),
                          diff6_edge_plane_shapes(ny, nx)):
        n = shape[0] * shape[1]
        dst = work["planes"][offset:offset + n].reshape(shape)
        offset += n
        if src is None:
            dst.fill(1)
        else:
            _edge_pad_into(dst, src)
        padded.append(dst)
    mutp, msfup, msfvp, msftp = padded
    phbp = None
    if slope:
        phbp = _edge_pad_into(
            _edge_take(work["phb"], (phb.shape[0], ny + pad, nx + pad)), phb)
    launch_diff6(fp, tp, mutp, c1, c2, factor, dt, opt, stagger=stagger,
                 phb=phbp, msfu=msfup, msfv=msfvp, msft=msftp,
                 slopeopt=slopeopt, thresh=thresh, dx=dx, dy=dy,
                 bnd_x=False, bnd_y=False)
    tend[...] = tp[:, h:h + nys, h:h + nxs]


def diff6_to_edge(cfg: RunConfig) -> bool:
    """True when the fork's edge-to-edge loops apply (fork form, and both
    horizontal axes forced: specified or nested).  The fork's loop bounds
    are the same on a periodic axis, where its halo is the periodic copy
    that :func:`launch_diff6` already reads, and an open (radiative) edge
    is not a configuration the fork's operational namelist runs."""
    return diff6_fork(cfg) and _boundary_forced(cfg)


def _clock_scaled_diff6_factor(cfg: RunConfig) -> float:
    """Per-step factor whose clock-interval composition equals WRF's."""
    return _clock_scale_factor(cfg, float(cfg.diff_6th_factor),
                               "diff_6th_factor")


def _clock_scale_factor(cfg: RunConfig, factor: float, name: str) -> float:
    """``factor`` rescaled so its clock-interval composition equals WRF's."""
    clock_dt = cfg.clock_dt if cfg.clock_dt > 0.0 else cfg.dt
    if clock_dt == cfg.dt:
        return factor
    if not 0.0 <= factor <= 1.0:
        raise ValueError(
            f"clock-scaled {name} must lie in [0, 1], got "
            f"{factor}")
    if factor == 1.0:
        return 1.0
    # A 2dx mode retains (1-factor) over one WRF model-clock step.  Taking
    # the matching fractional retention prevents eight 7.5 s compatibility
    # steps from applying the 60 s namelist factor eight times.
    return -math.expm1((cfg.dt / clock_dt) * math.log1p(-factor))


def apply_diff6(state: DomainState, cfg: RunConfig) -> None:
    """Apply one complete diff6 increment (verification utility).

    Production :func:`step` does **not** use this post-update helper.  It
    calls :func:`prepare_fixed_tendencies` once on the time-t fields, adds
    the held dry tendencies to all three RK passes, and passes the held
    scalar tendencies through ``rk_update_scalar[_pd]``.  This helper is
    retained for kernel normalization/clock-composition tests: callers must
    seed the ``*0`` fields, and it applies ``dt*tendf`` directly through the
    post-update hybrid mass.

    At non-periodic lateral boundaries the outermost boundary-normal
    STAGGERED face (WRF's ide-3/jde-3) is computed exactly as WRF
    computes it: the boundary-aware kernel reads the stored true
    boundary datum field(ide)/field(jde) (``launch_diff6`` ``bnd_x``/
    ``bnd_y``), and the width-3 host mask is then precisely WRF's loop
    exclusion on every axis and stagger.

    Row normalization is the production normalization: dry rows use
    ``dt`` and moisture rows use ``dt/3`` through :func:`_diff6_dt`.
    A shared mass-grid scratch slot does not imply a shared coefficient:
    theta and moisture use that same temporary but different WRF callers.

    Applied to u, v, w, theta' and all allocated transported moisture
    scalars.  WRF diffuses theta up to a constant offset, which is
    identical to theta' on flat coordinate surfaces.  Under
    ``moist_mix6_off`` the WRF ``moist``-array rows drop out here exactly as
    they drop out of the production row set (:func:`diff6_exempt_slots`), so
    the helper keeps measuring what production applies.
    """
    opt = cfg.diff_6th_opt
    to_edge = diff6_to_edge(cfg)
    mu_t = state.mub2d + state.mup0                # time-t mass (WRF mut)
    mu = state.total_mu()                          # post-step mass: uncouple
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    c1f = state.c1f[:, None, None]
    c2f = state.c2f[:, None, None]
    chm = c1h * mu[None] + c2h                     # mass-point coupling
    targets = [
        (state.u0, state.u, "x", state.c1h, state.c2h,
         c1h * mu_at_u_faces(mu)[None] + c2h, "diff6_x", "smag_ru"),
        (state.v0, state.v, "y", state.c1h, state.c2h,
         c1h * mu_at_v_faces(mu)[None] + c2h, "diff6_y", "smag_rv"),
        (state.w0, state.w, "z", state.c1f, state.c2f,
         c1f * mu[None] + c2f, "diff6_z", "smag_rw"),
        (state.thp0, state.thp, "", state.c1h, state.c2h, chm,
         "diff6_m", "smag_rth"),
    ]
    if state.qv is not None:
        exempt = diff6_exempt_slots(cfg)
        names = [name for name in SPECIES + tuple(extra_moist_species(state))
                 if "smag_r" + name not in exempt]
        targets += [(getattr(state, name + "0"), getattr(state, name), "",
                     state.c1h, state.c2h, chm, "diff6_m", "smag_r" + name)
                    for name in names]
    edge = {"work": _diff6_edge_work(state, cfg)} if to_edge else {}
    for f0, f, stag, c1, c2, chmf, slot, normalization_slot in targets:
        tendf = state.scratch(f0.shape, slot)
        tendf[...] = 0
        launch = launch_diff6_to_edge if to_edge else launch_diff6
        launch(f0, tendf, mu_t, c1, c2,
               _diff6_factor(cfg, normalization_slot),
               _diff6_dt(cfg, normalization_slot), opt,
               stagger=stag,
               # WRF diff_6th_slopeopt terrain taper (no-op with the
               # default 0 or a flat 1-D phb; the base-state slope
               # and per-face msf enter exactly as the Fortran).
               phb=state.phb, msfu=state.msfu, msfv=state.msfv,
               msft=state.msft,
               slopeopt=cfg.diff_6th_slopeopt,
               thresh=cfg.diff_6th_thresh, dx=cfg.dx, dy=cfg.dy,
               bnd_x=_boundary_x(cfg), bnd_y=_boundary_y(cfg), **edge)
        # tendf entered zeroed; the kernel's own loop bounds leave WRF's
        # non-periodic exclusions at zero (diff6_loop_bounds).
        f += DTYPE(cfg.dt) * tendf / chmf


def _prepare_emdiv_filter_launch(state: DomainState, cfg: RunConfig,
                                  mudf: cp.ndarray,
                                  mu_prev: cp.ndarray | None = None):
    """Bind one stage's invariant external-mode filter launch."""
    nz, ny, nx = state.p.shape
    # The raw union-grid kernel retains the five eager FP32 boundaries in
    # each gx/gy chain.  Saving mu_prev is independent and uses the k=0
    # owner thread for each mass column, avoiding a separate device copy.
    save = mu_prev is not None
    mu_prev_arg = mu_prev if save else mudf
    n = nz * (ny + 1) * (nx + 1)
    grid = ((n + 255) // 256,)
    block = (256,)
    kernel = get_kernel("acoustic", "apply_emdiv")
    args = (
        state.u_pp, state.v_pp, mudf, state.mu_pp, mu_prev_arg, state.c1h,
        state.msfu, state.msfv,
        # Strict mode adds the increment inside advance_uv in WRF's order
        # (acoustic._uv_emdiv_args); this launch then only saves mu''.
        DTYPE(0.0) if WRF_EXACT else DTYPE(-cfg.emdiv * cfg.dx),
        DTYPE(0.0) if WRF_EXACT else DTYPE(-cfg.emdiv * cfg.dy),
        np.int32(state.has_msf),
        np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
        np.int32(_boundary_forced(cfg)), np.int32(cfg.spec_zone),
        np.int32(save), np.int32(nz), np.int32(ny), np.int32(nx),
    )

    def launch() -> None:
        kernel(grid, block, args)

    launch.args = args          # what a joined u/v substep binds by name
    return launch


def apply_emdiv_filter(state: DomainState, cfg: RunConfig,
                       mudf: cp.ndarray,
                       mu_prev: cp.ndarray | None = None) -> None:
    """WRF external-mode divergence damping (module_small_step_em.F).

    Before each acoustic substep the perturbation momenta get WRF
    ``advance_uv``'s ``mudf_xy`` term (lines 809/868, 880/942; map factors
    1): ``u'' += c1h * (-emdiv*dx*(mudf_i - mudf_{i-1}))`` and the y
    analogue, where ``mudf (ny, nx)`` is the PREVIOUS substep's
    column-mass tendency (``advance_mu_t``: dmdt + mu_tend; zeroed by
    ``small_step_prep`` at RK stage 1 ONLY, module_small_step_em.F:128
    guards the reset with ``IF (rk_step == 1)``, so only the very first
    acoustic iteration of the model step adds nothing; stages 2/3 inherit
    the prior stage's final tendency).  This damps the column-integrated (external) mode --
    WRF's stock stabilizer for open lateral boundaries (the em_quarter_ss
    namelist runs emdiv = 0.01).  Boundary-normal faces at open boundaries
    are excluded exactly like the acoustic pressure gradient (the mudf_xy
    loop shares advance_uv's bounds); periodic faces wrap.  Adding the
    increment before the substep kernel is equivalent to WRF's in-kernel
    ordering: both land on u'' before ``advance_mu_t`` consumes it.
    Reference: ``woof.verify.npref.np_emdiv_uv``.
    """
    _prepare_emdiv_filter_launch(state, cfg, mudf, mu_prev)()


def _prepare_emdiv_mudf_launch(state: DomainState, cfg: RunConfig,
                               mudf: cp.ndarray, mu_prev: cp.ndarray,
                               dtau: float):
    """Bind one stage's invariant mudf recurrence launch."""
    ny, nx = state.mup.shape
    grid = ((ny * nx + 255) // 256,)
    block = (256,)
    kernel = get_kernel("acoustic", "update_mudf")
    boundary_forced = np.int32(_boundary_forced(cfg))
    args = (
        mudf, state.mu_pp, mu_prev, DTYPE(dtau), boundary_forced,
        boundary_forced, np.int32(cfg.spec_zone), np.int32(ny), np.int32(nx),
    )

    def launch() -> None:
        kernel(grid, block, args)

    return launch


def _update_emdiv_mudf(state: DomainState, cfg: RunConfig,
                       mudf: cp.ndarray, mu_prev: cp.ndarray,
                       dtau: float) -> None:
    """Finish the exact FP32 mudf recurrence and boundary-strip zeroing."""
    _prepare_emdiv_mudf_launch(state, cfg, mudf, mu_prev, dtau)()


def _prepare_sumflux_launch(name: str, targets: tuple, sources: tuple = (),
                            nsub: int = 0):
    """Bind one invariant batched WRF sumflux launch."""
    sizes = tuple(np.uint64(array.size) for array in targets)
    nmax = max(int(size) for size in sizes)
    grid = ((nmax + 255) // 256,)
    block = (256,)
    args = (*targets, *sources)
    if name == "finish_sumflux":
        args += (DTYPE(nsub),)
    args += (*sizes, np.uint64(nmax))
    kernel = get_kernel("acoustic", name)

    def launch() -> None:
        kernel(grid, block, args)

    return launch


def _sumflux_launch(name: str, targets: tuple, sources: tuple = (),
                    nsub: int = 0) -> None:
    """Batch three independent staggered WRF sumflux array operations."""
    _prepare_sumflux_launch(name, targets, sources, nsub)()


def _zero_open_strips(buf: cp.ndarray, cfg: RunConfig, width: int,
                      stag_high_extra: int = 0) -> None:
    """Zero a coupled mixing tendency over the strip WRF's loop bounds skip
    at open lateral boundaries (no-op when periodic).

    ``width = 1`` mirrors ``horizontal_diffusion``; ``width = 3`` was
    the sixth-order exclusion, which the diff6 kernel now applies as its
    own loop bounds (:func:`diff6_loop_bounds`).  On a non-staggered
    axis the outer ``width`` entries on each side are exactly the points WRF's bounds
    exclude (e.g. mass fields under open_x: WRF computes ids+3..ide-4 of
    the ids..ide-1 cells, so 3 columns go to zero per side); without this
    the wrapped stencils couple the two open boundaries.  On the
    boundary-normal STAGGERED axis (``nx + 1`` u faces under open_x /
    ``ny + 1`` v faces under open_y) WRF's high-side exclusion is
    ``width`` faces counted from the boundary face ide itself, and the
    high side here takes ``width + stag_high_extra``: a caller whose
    stencil cannot reproduce WRF's read of the true boundary datum at the
    outermost computed face may pass ``stag_high_extra = 1`` to zero that
    face as well.  No production caller does any more: the diff6 kernel's
    ``bnd_x``/``bnd_y`` mode and the smag2d u/v kernels both make the
    accurate boundary-datum read themselves (WRF computes u face ide-3
    reading field(i+3) = u(ide); smag2d.cu ``open_x``/``open_y``,
    diff6.cu loop bounds), so ``width = 1`` (smag2d) and the diff6
    bounds are exactly WRF's exclusions for every stagger.  The
    parameter is retained for reconstructing the historical pre-fix mask
    (tests/test_diff6_boundary_face.py's 4d2ce99 capture)."""
    x_hi = width + (stag_high_extra if buf.shape[-1] == cfg.nx + 1 else 0)
    y_hi = width + (stag_high_extra if buf.shape[-2] == cfg.ny + 1 else 0)
    if _boundary_x(cfg):
        buf[..., :width] = 0
        buf[..., -x_hi:] = 0
    if _boundary_y(cfg):
        # Ellipsis keeps this valid for 2-D (ny, nx) buffers too -- the
        # emdiv mudf strip is the live 2-D caller.
        buf[..., :width, :] = 0
        buf[..., -y_hi:, :] = 0


def _restore_open_strips(buf: cp.ndarray, saved: cp.ndarray, cfg: RunConfig,
                         width: int) -> None:
    """Copy ``saved`` back over the strip :func:`_zero_open_strips` zeroes.

    The strict 6th-order path accumulates into a copy of the held tendency;
    WRF's loop bounds never touch the strip, so it keeps the held value.
    """
    if _boundary_x(cfg):
        buf[..., :width] = saved[..., :width]
        buf[..., -width:] = saved[..., -width:]
    if _boundary_y(cfg):
        buf[..., :width, :] = saved[..., :width, :]
        buf[..., -width:, :] = saved[..., -width:, :]


def set_w_surface(state: DomainState, cfg: RunConfig) -> None:
    """Set the kinematic surface value with the eager FP32 operation order."""
    arrays = (state.u, state.v, state.ht, state.w,
              getattr(state, 'msft', None))
    if (any(not isinstance(a, cp.ndarray) or a.dtype != np.dtype(np.float32)
            or not a.flags.c_contiguous for a in arrays)
            or any(not isinstance(value, np.float32)
                   for value in (state.cf1, state.cf2, state.cf3))
            or isinstance(cfg.dx, np.generic) or isinstance(cfg.dy, np.generic)
            or state.u.shape[0] < 3 or state.v.shape[0] < 3
            or min(state.ht.shape) < 2):
        _set_w_surface_eager(state, cfg)
        return
    ny, nx = state.ht.shape
    kernel = get_kernel('surface_w', 'set_surface_w')
    kernel(((ny * nx + _TPB - 1) // _TPB,), (_TPB,),
           (state.u, state.v, state.ht, state.msft, state.w,
            DTYPE(state.cf1), DTYPE(state.cf2), DTYPE(state.cf3),
            DTYPE(0.5 / cfg.dx), DTYPE(0.5 / cfg.dy),
            np.int32(state.has_msf), np.int32(_boundary_x(cfg)),
            np.int32(_boundary_y(cfg)), np.int32(ny), np.int32(nx)))


def _set_w_surface_eager(state: DomainState, cfg: RunConfig) -> None:
    """Kinematic lower boundary condition on the uncoupled w (WRF
    ``set_w_surface``, module_bc_em.F): ``w(sfc) = u.grad(ht)`` with the
    cf1..cf3-weighted three lowest half levels of u/v. Exterior terrain
    donors clamp on nonperiodic axes and wrap on periodic axes.
    Exactly zero over flat terrain.  ``dycore.step`` calls this at the end
    of every step ("reset surface w for consistency", WRF solve_em); case
    builders call it once at init (WRF start_em).
    """
    ht = state.ht
    uc = (state.cf1 * state.u[0] + state.cf2 * state.u[1]
          + state.cf3 * state.u[2])                    # (ny, nx+1)
    vc = (state.cf1 * state.v[0] + state.cf2 * state.v[1]
          + state.cf3 * state.v[2])                    # (ny+1, nx)
    dyn = cp.roll(ht, -1, 0) - ht
    dys = ht - cp.roll(ht, 1, 0)
    dxe = cp.roll(ht, -1, 1) - ht
    dxw = ht - cp.roll(ht, 1, 1)
    if _boundary_y(cfg):
        dys[0, :] = 0.0
        dyn[-1, :] = 0.0
    if _boundary_x(cfg):
        dxw[:, 0] = 0.0
        dxe[:, -1] = 0.0
    sfc = ((0.5 / cfg.dy) * (dyn * vc[1:, :] + dys * vc[:-1, :])
           + (0.5 / cfg.dx) * (dxe * uc[:, 1:] + dxw * uc[:, :-1]))
    if state.has_msf:              # WRF set_w_surface: msfty*(v part) +
        sfc *= state.msft          # msftx*(u part); isotropic single msft
    state.w[0] = sfc


#: Radiation phase speed c* (m/s) of the Klemp-Wilhelmson open lateral BC:
#: WRF's cb = 25 (share/module_model_constants.F:47, consumed by the open
#: radiative blocks in dyn_em/module_advect_em.F), adjudicated over the
#: plan's original 30 (the published KW78 value), the local WRF source is
#: authoritative.  Must match npref.OPEN_CB.
OPEN_CB = 25.0

_BC_THREADS = 256


def apply_open_radiative_bc(state: DomainState, cfg: RunConfig) -> None:
    """Radiative open-BC tendency for the boundary-normal velocities.

    WRF gravity-wave radiative lateral BC (dyn_em/module_advect_em.F
    ``advect_u``/``advect_v`` open blocks, ``tendency = tendency + ...``):
    with ``cfg.open_x``/``open_y`` the one-sided Klemp-Wilhelmson radiative
    term with the outbound-only phase speed ``u_n -/+ c*`` (``OPEN_CB``) is
    ADDED to the coupled slow tendency at the two boundary-normal velocity
    faces (Task 11 prerequisite; Task 9 REPLACED, dropping the terms WRF
    retains there).  The open-aware advection kernels already excluded the
    boundary-normal advection at those faces -- the radiative term stands
    in for it -- and they skip the acoustic pressure gradient
    (woof.core.acoustic) plus the large-step PGF, so over the substeps
    they integrate the radiation equation on top of whatever advection WRF
    keeps (u's vertical advection when only x is open).  No-op with the
    periodic defaults.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    has_msf = bool(getattr(state, "has_msf", False))
    if cfg.open_x:
        kernel = get_kernel("openbc", "open_u_radiative")
        blocks = (nz * ny + _BC_THREADS - 1) // _BC_THREADS
        kernel((blocks,), (_BC_THREADS,),
               (state.ru_t, state.u, state.mup, state.mub2d,
                state.c1h, state.c2h, state.msfu if has_msf else state.mub2d,
                DTYPE(1.0 / cfg.dx), DTYPE(OPEN_CB),
                np.int32(nz), np.int32(ny), np.int32(nx), np.int32(has_msf)))
    if cfg.open_y:
        kernel = get_kernel("openbc", "open_v_radiative")
        blocks = (nz * nx + _BC_THREADS - 1) // _BC_THREADS
        kernel((blocks,), (_BC_THREADS,),
               (state.rv_t, state.v, state.mup, state.mub2d,
                state.c1h, state.c2h, state.msfv if has_msf else state.mub2d,
                DTYPE(1.0 / cfg.dy), DTYPE(OPEN_CB),
                np.int32(nz), np.int32(ny), np.int32(nx), np.int32(has_msf)))


#: WRF ``w_beta`` (share/module_model_constants.F:89): the vertical Courant
#: number where ``w_damp`` starts when ``zadvect_implicit`` is off.
W_DAMP_BETA = 1.0


def w_damp_onset(cfg: RunConfig) -> float:
    """The vertical Courant number above which ``w_damp`` acts.

    WRF 4.7.1 ``w_damp`` (module_big_step_utilities_em.F:2601-2607):
    ``w_damp_on = w_crit_cfl`` when ``zadvect_implicit > 0``, else
    ``w_beta``.  The excess is measured from ``w_crit_cfl`` in both cases.
    """
    if int(getattr(cfg, "zadvect_implicit", 0) or 0) > 0:
        return float(cfg.w_crit_cfl)
    return W_DAMP_BETA


def apply_w_damping(state: DomainState, cfg: RunConfig,
                    ww: cp.ndarray) -> None:
    """WRF ``w_damp`` (module_big_step_utilities_em.F), ``w_damping = 1``.

    Where the vertical Courant number ``|ww/(c1f*mu+c2f)*rdnw*dt|`` of the
    stage's diagnosed eta mass flux ``ww`` exceeds the activation value
    (:func:`w_damp_onset`: w_beta = 1, or ``w_crit_cfl`` under
    ``zadvect_implicit``), the coupled w tendency is pushed against the
    vertical motion by ``w_alpha*(vert_cfl - w_crit_cfl)``, a limiter, not
    physics (WRF adds it for robustness at marginal CFL).  Interior w
    levels only; no-op unless ``cfg.w_damping == 1``.
    """
    if cfg.w_damping != 1:
        return
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    kernel = get_kernel("openbc", "w_damp")
    blocks = ((nz - 1) * ny * nx + _BC_THREADS - 1) // _BC_THREADS
    kernel((blocks,), (_BC_THREADS,),
           (state.rw_t, ww, state.w, state.mup, state.mub2d,
            state.c1f, state.c2f, state.rdnw, DTYPE(cfg.dt),
            DTYPE(w_damp_onset(cfg)), DTYPE(cfg.w_crit_cfl),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _env_flag(name: str) -> bool:
    """An environment switch that reads ``0``/``false``/``off`` as OFF.

    ``bool(os.environ.get(name))`` is true for the string ``"0"``, so a
    user who turns a probe off the obvious way turns it on.  Defined here
    rather than imported because the twin lives in
    :mod:`woof.core.adaptive_clock`, which sits ABOVE this module and
    may not be imported from it.
    """
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


# --- WRF vertical-CFL measurement (off unless GPUWM_WRF_CFL_PROBE=1) ----
#
# health.cu reports a GEOMETRIC vertical Courant number, max |w|/dz over
# EVERY level, and that is not the quantity WRF's target_cfl grades.  WRF
# grades the eta-coordinate form |ww/(c1f*mut+c2f)*rdnw*dt| over
# k = 2..kde-1 -- which apply_w_damping already computes per cell and
# throws away.  The two differ by the vertical velocity used AND by the
# index range: the geometric form includes the thin near-surface layer
# that WRF's loop excludes -- a 26 m first mass level on the trees this
# was measured on -- and that layer dominates it.  Reading a controller
# target against the wrong one of these is how a timestep decision goes
# wrong quietly, so
# this reduces the RIGHT one, from the same ww, at the same stage.
#
# Off by default and free when off: no allocation, no launch.
_WRF_CFL_PROBE = _env_flag("GPUWM_WRF_CFL_PROBE")
#: Force a device->host readback at the end of every model step, which is
#: what a REAL controller must do: it has to know this step's CFL on the
#: host before it can choose the next dt.  The reduction itself is free
#: (measured), but the readback is a synchronisation point, and woof
#: otherwise queues work asynchronously -- so this is the one cost that
#: could make an adaptive dt SLOWER than the steps it saves.  Separate
#: env var precisely so it can be A/B'd against the reduction alone.
_WRF_CFL_PROBE_SYNC = _env_flag("WOOF_WRF_CFL_PROBE_SYNC")
#: Grid primary ring; (grid, card) keys hold additional device rings.
_WRF_CFL_STAT: dict[int | tuple[int, int], object] = {}
_WRF_CFL_LABEL: dict[int, str] = {}
#: Last read-back CFL per domain, populated only under the sync probe.
_WRF_CFL_LAST: dict[int, float] = {}


#: Per-step slots, used as a RING.  One row per model step per domain,
#: so the dump is a TIME SERIES and not just a running max -- a single
#: max says the peak happened, never when, and a controller is designed
#: by how fast the number MOVES.  Written by the same atomics with no
#: host sync during the run.
#:
#: THE RING IS NOT A CONVENIENCE.  Saturating the slot index at the last
#: row instead -- which is what this did -- lands every fold past step
#: 32768 in one row that nothing ever clears, and `atomicMax` then makes
#: the CFL the controller reads a monotonically non-decreasing running
#: maximum.  From that step on the controller can only shrink dt, and a
#: long run (a 24 h nest at dt = 2 s is 43,200 steps; a 72 h root at
#: dt = 8 s is 32,400) collapses toward min_time_step for a reason no
#: diagnostic reports.  The two halves of the fix must go together: the
#: ring keeps the series, and clearing the row at each step's first fold
#: is what makes out[0] THIS step's maximum and stops out[1]/out[2]
#: accumulating without bound in a uint32.
from woof.core.cfl_inventory import (
    WRF_CFL_SLOTS as _WRF_CFL_SLOTS, CFL_HIST_BINS as _CFL_HIST_BINS,
    WRF_CFL_WORDS as _WRF_CFL_WORDS,
)

#: Histogram bins in ``w_cfl_stat``.  MUST match CFL_HIST_BINS and
#: CFL_HIST_SCALE in kernels/openbc.cu -- bin b covers vert_cfl in
#: [b/SCALE, (b+1)/SCALE), and the top bin absorbs >= BINS/SCALE and NaN.
#: There is no way for NVRTC to check that agreement, so
#: :func:`wrf_cfl_histogram_edges` is asserted against the kernel's own
#: arithmetic in tests/test_wrf_cfl_histogram.py rather than trusted.
_CFL_HIST_SCALE = 16.0
#: 4 scalar words + the histogram.  32768 rows x 36 words x 4 B = 4.7 MB
#: per domain, ~14 MB for a three-domain tree.  Against standing rule 3
#: (do not raise VRAM) that is under the 50 MiB bar a change has to earn,
#: and it is diagnostic memory that buys the distribution behind out[0].
_WRF_CFL_CALLS: dict[int, int] = {}



# A tile sweep owns one diagnostic row, just as a resident dycore step does.
# Each kernel reduces its owned mass columns into that same row with atomics.
# No additional device arrays: the existing ring is the accumulator.
_WRF_CFL_DOMAIN_STEP: dict[int, dict] = {}

# The step's fences are re-recorded on events that live as long as the
# recording does, not created per step.  THE BREAKAGE THIS PREVENTS, measured
# 2026-10-03 on a free-threaded 4-card run: the stepping thread created the
# step's "slot is zeroed" events and a slab thread dropped the last reference,
# so the interpreter handed their release back to the stepping thread, where
# two cudaEventDestroy calls waited 10 and 13 ms behind other cards' blocking
# readbacks while finished slabs waited for their halo exchange.  A wait takes
# the event's most recent record at the time of the wait call, so recording
# the same event again next step changes no ordering.
_WRF_CFL_EVENTS: dict[tuple, object] = {}


def _wrf_cfl_event(*key):
    """The reusable event for ``key``, created on the current device."""
    bank = _wrf_cfl_bank("_WRF_CFL_EVENTS")
    event = bank.get(key)
    if event is None:
        event = bank[key] = cp.cuda.Event(disable_timing=True)
    return event


def _wrf_cfl_bank(name):
    """Ordinary reduction banks, private to a scoped ensemble member."""
    from woof.core.cfl_member import current_cfl_member
    owner = current_cfl_member()
    return globals()[name] if owner is None else owner.banks[name]


def _wrf_cfl_recording_enabled():
    from woof.core.cfl_member import current_cfl_member
    owner = current_cfl_member()
    return _WRF_CFL_PROBE if owner is None else owner.enabled


def _wrf_cfl_buffers(grid_id):
    """The primary ring plus additional device rings, in device order."""
    primary = _wrf_cfl_bank("_WRF_CFL_STAT").get(int(grid_id))
    buffers = [] if primary is None else [primary]
    buffers.extend(buf for key, buf in _wrf_cfl_bank("_WRF_CFL_STAT").items()
                   if isinstance(key, tuple) and key[0] == int(grid_id))
    return sorted(buffers, key=lambda buf: int(buf.device.id))


def _wrf_cfl_buffer(cfg):
    grid = int(cfg.grid_id)
    dev = int(cp.cuda.runtime.getDevice())
    primary = _wrf_cfl_bank("_WRF_CFL_STAT").get(grid)
    key = grid if primary is None or int(primary.device.id) == dev else (grid, dev)
    buf = _wrf_cfl_bank("_WRF_CFL_STAT").get(key)
    if buf is None:
        buf = cp.zeros((_WRF_CFL_SLOTS, _WRF_CFL_WORDS), dtype=cp.uint32)
        _wrf_cfl_bank("_WRF_CFL_STAT")[key] = buf
        _wrf_cfl_bank("_WRF_CFL_CALLS").setdefault(grid, 0)
        _wrf_cfl_bank("_WRF_CFL_LABEL")[grid] = f"d{grid:02d} {cfg.nx}x{cfg.ny}x{cfg.nz} dx={cfg.dx:g}"
    return buf


from threading import local as _thread_local
_WRF_CFL_THREAD = _thread_local()


def _wrf_cfl_window(grid_id):
    key = int(grid_id)
    entry = getattr(_WRF_CFL_THREAD, "windows", {}).get(key)
    ctx = _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP").get(key)
    # A thread-local window from a retired step must not satisfy the next
    # step's owned-column contract if a worker forgot to install its window.
    return None if entry is None or entry[0] is not ctx else entry[1]


def begin_wrf_cfl_domain_step(cfg, devices=None) -> None:
    if not _wrf_cfl_recording_enabled():
        return
    key = int(cfg.grid_id)
    if key in _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP"):
        raise RuntimeError(f"d{key:02d} already has an open CFL domain step")
    devices = (cp.cuda.runtime.getDevice(),) if devices is None else tuple(dict.fromkeys(devices))
    for dev in devices:
        with cp.cuda.Device(dev):
            _wrf_cfl_buffer(cfg)
    calls = _wrf_cfl_bank("_WRF_CFL_CALLS")[key]
    if calls % 3:
        raise RuntimeError(f"d{key:02d} CFL step starts inside an RK-stage group")
    slot = (calls // 3) % _WRF_CFL_SLOTS
    ready = {}
    for buf in _wrf_cfl_buffers(key):
        with cp.cuda.Device(buf.device.id):
            buf[slot].fill(0)
            event = _wrf_cfl_event("ready", key, int(buf.device.id))
            event.record()
            ready[int(buf.device.id)] = event
    _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP")[key] = dict(slot=slot, ready=ready, events={})


def set_wrf_cfl_tile_window(grid_id, spec) -> None:
    ctx = _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP").get(int(grid_id))
    if ctx is None:
        return
    if not hasattr(_WRF_CFL_THREAD, "windows"):
        _WRF_CFL_THREAD.windows = {}
    _WRF_CFL_THREAD.windows[int(grid_id)] = (ctx, (
        int(spec.i0 - spec.ci0), int(spec.i1 - spec.ci0),
        int(spec.j0 - spec.cj0), int(spec.j1 - spec.cj0)))
    cp.cuda.get_current_stream().wait_event(ctx["ready"][cp.cuda.runtime.getDevice()])


def finish_wrf_cfl_tile(grid_id) -> None:
    ctx = _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP").get(int(grid_id))
    if ctx is not None:
        stream = cp.cuda.get_current_stream()
        device = cp.cuda.runtime.getDevice()
        event = _wrf_cfl_event("tile", int(grid_id), device, stream.ptr)
        event.record(stream)
        ctx["events"][(device, stream.ptr)] = event


def wrf_cfl_capture_key(grid_id):
    ctx = _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP").get(int(grid_id))
    return None if ctx is None else (ctx["slot"], _wrf_cfl_window(grid_id))


def finish_wrf_cfl_domain_step(grid_id, *, commit=True) -> None:
    key = int(grid_id)
    ctx = _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP").pop(key, None)
    if ctx is None:
        return
    for dev in ctx["ready"]:
        with cp.cuda.Device(dev):
            stream = cp.cuda.get_current_stream()
            for (event_dev, _), event in ctx["events"].items():
                if event_dev == dev:
                    stream.wait_event(event)
    if commit:
        _wrf_cfl_bank("_WRF_CFL_CALLS")[key] += 3
        if _WRF_CFL_PROBE_SYNC:
            _wrf_cfl_bank("_WRF_CFL_LAST")[key] = take_wrf_cfl(key)[0]

def record_wrf_vertical_cfl(state: DomainState, cfg: RunConfig,
                            ww: cp.ndarray) -> None:
    """Fold this stage's WRF vertical CFL into this step's slot."""
    if not _wrf_cfl_recording_enabled():
        return
    # KEYED ON grid_id, not id(state).  Relocation replaces the state
    # object -- measured: a 4-hour run split d02's series into SEVEN
    # segments -- so an id(state) key loses the domain's history on every
    # move.  A controller keyed that way would reset its dt at each
    # relocation, which is invariant (e) of the brief exactly.
    key = int(cfg.grid_id)
    buf = _wrf_cfl_buffer(cfg)
    ctx = _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP").get(key)
    calls = _wrf_cfl_bank("_WRF_CFL_CALLS")[key]
    if ctx is None:
        _wrf_cfl_bank("_WRF_CFL_CALLS")[key] = calls + 1
        slot = (calls // 3) % _WRF_CFL_SLOTS
        if calls % 3 == 0:
            buf[slot].fill(0)
    else:
        slot = ctx["slot"]
        if _wrf_cfl_window(key) is None:
            raise RuntimeError("CFL tile step has no owned-column window")
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    window = None if ctx is None else _wrf_cfl_window(key)
    kernel = get_kernel("openbc", "w_cfl_stat" if window is None
                        else "w_cfl_stat_window")
    columns = ny * nx if window is None else (window[1]-window[0])*(window[3]-window[2])
    blocks = ((nz - 1) * columns + _BC_THREADS - 1) // _BC_THREADS
    args = (ww, state.mup, state.mub2d, state.c1f, state.c2f, state.rdnw,
            state.u, state.v, state.msfu, state.msfv,
            buf[slot], DTYPE(cfg.dt), DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy),
            DTYPE(w_damp_onset(cfg)),
            np.int32(nz), np.int32(ny), np.int32(nx))
    if window is not None:
        args += tuple(np.int32(v) for v in window)
    kernel((blocks,), (_BC_THREADS,), args)
    if ctx is None and _WRF_CFL_PROBE_SYNC and (calls + 1) % 3 == 0:
        # End of a model step: read this step's CFL back the way a
        # controller would have to.  float() on a device scalar is a
        # blocking copy, which is the whole point of measuring it.
        _wrf_cfl_bank("_WRF_CFL_LAST")[key] = float(
            np.uint32(int(buf[slot][0])).view(np.float32))


def take_wrf_cfl(grid_id: int) -> tuple[float, float]:
    """This step's (vertical, horizontal) WRF CFL for one domain.

    The row is cleared by the NEXT step's first fold rather than here, so
    that a reader is never racing the clear against a queued kernel.

    The accumulator is folded on the device every RK stage and read back
    ONCE per model step, here.  That readback is a synchronisation point
    and it is the one cost that could make an adaptive dt slower than the
    steps it saves -- measured at +1.1% on the median against a 2.5% SE,
    i.e. under the measurement host's noise floor
    (docs/ADAPTIVE-TIMESTEP.md section 10).

    Returns ``(0.0, 0.0)`` when the domain has folded nothing yet, which
    is the state before its first solve.  A controller reading that takes
    calc_dt's ``max_cfl < 0.001`` branch and grows by the full increase
    factor -- correct at t=0, where there is no CFL to respect, and
    exactly the overshoot the restart path must avoid (section 5).
    """
    buf = _wrf_cfl_bank("_WRF_CFL_STAT").get(int(grid_id))
    if buf is None:
        return 0.0, 0.0
    calls = _wrf_cfl_bank("_WRF_CFL_CALLS").get(int(grid_id), 0)
    if calls == 0:
        return 0.0, 0.0
    slot = ((calls - 1) // 3) % _WRF_CFL_SLOTS
    from woof.core.cfl_inventory import fold_cfl_words
    rows = []
    for device_buf in _wrf_cfl_buffers(grid_id):
        with cp.cuda.Device(device_buf.device.id):
            rows.append(cp.asnumpy(device_buf[slot]))
    words = rows[0] if len(rows) == 1 else fold_cfl_words(rows)
    vert = float(np.uint32(words[0]).view(np.float32))
    horiz = float(np.uint32(words[3]).view(np.float32))
    return vert, horiz


def enable_wrf_cfl_recording() -> None:
    """Turn the CFL fold on for a run that is not using the env probe.

    The adaptive controller needs the same reduction the probe reads, so
    it switches it on rather than carrying a second copy of the kernel.
    """
    global _WRF_CFL_PROBE
    from woof.core.cfl_member import current_cfl_member
    owner = current_cfl_member()
    if owner is None:
        _WRF_CFL_PROBE = True
    else:
        owner.enabled = True


def reset_wrf_cfl_recording() -> None:
    """Put the fold back where the environment left it and free its rows.

    Prevents a SECOND experiment in the same process inheriting the
    first's: the enable above is a module global and the accumulators are
    module dicts keyed on grid_id, so without this a chained run that
    never asked for the probe kept launching w_cfl_stat every RK stage,
    held 4.72 MB per domain for nothing, and -- because grid_ids repeat
    across experiments -- read the PREVIOUS run's row until its own first
    three folds had landed.
    """
    global _WRF_CFL_PROBE
    from woof.core.cfl_member import current_cfl_member
    owner = current_cfl_member()
    if owner is None:
        _WRF_CFL_PROBE = _env_flag("GPUWM_WRF_CFL_PROBE")
    else:
        owner.enabled = _env_flag("GPUWM_WRF_CFL_PROBE")
    _wrf_cfl_bank("_WRF_CFL_DOMAIN_STEP").clear()
    _wrf_cfl_bank("_WRF_CFL_EVENTS").clear()
    _WRF_CFL_THREAD.windows = {}
    _wrf_cfl_bank("_WRF_CFL_STAT").clear()
    _wrf_cfl_bank("_WRF_CFL_CALLS").clear()
    _wrf_cfl_bank("_WRF_CFL_LABEL").clear()
    _wrf_cfl_bank("_WRF_CFL_LAST").clear()


def wrf_cfl_histogram_edges() -> list[float]:
    """Lower edge of each vert_cfl histogram bin, matching openbc.cu.

    The kernel computes ``bin = (int)(vert_cfl * CFL_HIST_SCALE)``, so bin
    ``b`` starts at ``b / CFL_HIST_SCALE``.  Exposed as a function rather
    than a literal table because the two constants live in a .cu file that
    nothing type-checks against this module.
    """
    return [b / _CFL_HIST_SCALE for b in range(_CFL_HIST_BINS)]


def _hist_quantile(cum: np.ndarray, total: np.ndarray, q: float) -> np.ndarray:
    """Per-step upper bound on the ``q`` quantile of vert_cfl.

    Returns each bin's UPPER edge, which bounds the true quantile from
    above -- the conservative direction for anything that sizes a
    timestep off it.  The top bin is open-ended (it absorbs >= 2.0, and
    NaN, by construction in the kernel), so a quantile landing there is
    reported as ``inf`` rather than as a number the histogram cannot
    support.  Steps that folded nothing give NaN, not a spurious 0.
    """
    target = q * total
    idx = (cum < target[:, None]).sum(axis=1)          # first bin reaching q
    edge = (idx + 1).astype(float) / _CFL_HIST_SCALE
    edge[idx >= _CFL_HIST_BINS - 1] = float("inf")
    edge[total <= 0] = float("nan")
    return edge


def wrf_vertical_cfl_report() -> list[dict]:
    """Per-domain WRF vertical-CFL series, one entry per model step."""
    out = []
    from woof.core.cfl_inventory import fold_cfl_words
    for key in (k for k in _wrf_cfl_bank("_WRF_CFL_STAT") if not isinstance(k, tuple)):
        arrays = []
        for buf in _wrf_cfl_buffers(key):
            with cp.cuda.Device(buf.device.id):
                arrays.append(cp.asnumpy(buf))
        words = arrays[0] if len(arrays) == 1 else fold_cfl_words(arrays)
        steps = max(1, (_wrf_cfl_bank("_WRF_CFL_CALLS").get(key, 0) + 2) // 3)
        steps = min(steps, _WRF_CFL_SLOTS)
        rows = words[:steps]
        series = rows[:, 0].view(np.float32).astype(float).tolist()
        hseries = rows[:, 3].view(np.float32).astype(float).tolist()
        damped = rows[:, 1].astype(np.int64)
        visited = rows[:, 2].astype(np.int64)

        # The distribution behind the max.  out[0] is an order statistic:
        # one cell sets it, and a controller reading it cannot separate
        # "the flow sped up" from "one column is having a moment".  The
        # ratio max/p99.99 is the number that settles whether a quantile
        # controller is worth building -- near 1.0 and the max IS the
        # distribution, so the idea is dead and this says so.
        hist = rows[:, 4:].astype(np.int64)
        cum = np.cumsum(hist, axis=1)
        total = cum[:, -1] if cum.shape[1] else np.zeros(len(rows), np.int64)
        p999 = _hist_quantile(cum, total, 0.999)
        p9999 = _hist_quantile(cum, total, 0.9999)
        vmax = np.asarray(series, dtype=float)
        # np.isfinite on p9999 decides the branch, not defends it.  A quantile
        # landing in the open top bin is +inf, and vmax/inf is 0.0 -- which
        # is FINITE, so it would survive the filter below and drag the
        # median toward zero, reporting "the max is the distribution" for
        # exactly the saturated steps where it is least true.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.isfinite(p9999) & (p9999 > 0),
                             vmax / p9999, np.nan)
        ratio = ratio[np.isfinite(ratio)]
        finite = [v for v in series if math.isfinite(v)]
        mean = (sum(finite) / len(finite)) if finite else float("nan")
        if len(finite) > 1:
            var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
            se = (var / len(finite)) ** 0.5
        else:
            se = 0.0
        out.append({"domain": _wrf_cfl_bank("_WRF_CFL_LABEL").get(key, "?"),
                    "steps": int(steps),
                    "mean_vert_cfl": mean,
                    "se_vert_cfl": se,
                    "max_vert_cfl_wrf": max(finite) if finite else float("nan"),
                    "min_vert_cfl_wrf": min(finite) if finite else float("nan"),
                    "damped_cells": int(damped.sum()),
                    "cells_visited": int(visited.sum()),
                    "steps_with_damping": int((damped > 0).sum()),
                    "max_horiz_cfl_wrf": max(
                        (h for h in hseries if math.isfinite(h)),
                        default=float("nan")),
                    # How far the max sits above the bulk.  A median ratio
                    # near 1 kills the quantile-controller idea outright.
                    "median_max_over_p9999": (
                        float(np.median(ratio)) if ratio.size
                        else float("nan")),
                    "hist_edges": wrf_cfl_histogram_edges(),
                    "hist_total": hist.sum(axis=0).tolist(),
                    "p999_series": p999.tolist(),
                    "p9999_series": p9999.tolist(),
                    "series": series,
                    "horiz_series": hseries})
    return out


if _WRF_CFL_PROBE:                       # dump on exit; probe-only path
    import atexit as _atexit
    import json as _json

    @_atexit.register
    def _dump_wrf_cfl_report() -> None:
        try:
            rows = wrf_vertical_cfl_report()
        except Exception:                # a probe must never fail a run
            return
        if not rows:
            return
        # stdout gets the SUMMARY; the per-step arrays go to the file.
        # "series" was the only name filtered here, which was already
        # letting horiz_series print 900+ floats into a run log, and the
        # quantile series would have made that four such arrays.
        bulky = {"series", "horiz_series", "p999_series", "p9999_series",
                 "hist_edges"}
        text = _json.dumps([{k: v for k, v in row.items() if k not in bulky}
                            for row in rows], indent=2, sort_keys=True)
        path = os.environ.get("WOOF_WRF_CFL_PROBE_OUT")
        if path:
            try:
                io_open = open(path, "w", encoding="utf-8")
            except OSError:
                io_open = None
            if io_open is not None:
                with io_open as handle:      # the FULL series goes to file
                    handle.write(_json.dumps(rows, indent=1, sort_keys=True))
        print("WRF_VERT_CFL_PROBE " + text, flush=True)


def apply_open_zero_gradient(state: DomainState, cfg: RunConfig) -> None:
    """Zero-gradient outbound boundary values at open lateral boundaries.

    The plan's "scalars/theta zero-gradient outbound" clause (WRF
    ``set_physical_bc3d`` open-BC extension, share/module_bc.F, applied to
    the boundary cells themselves since woof carries no ghost strip): at
    every level where the boundary-normal velocity at the boundary face
    points outward, theta', w (interior levels), the tangential velocity,
    and the moisture scalars in the boundary column are copied from the
    first interior neighbour; inflow levels keep their prognostic values.
    phi'/mu' stay prognostic everywhere (their boundary evolution is
    column-local plus the radiated boundary-face divergence).  Called at
    the end of every RK stage; no-op with the periodic defaults.

    The tangential velocity's mask needs one more row/column than the
    boundary-normal mask has, and where that extra one comes from is decided
    by the TANGENTIAL axis's own boundary condition -- see the comment at the
    two ``concatenate`` calls.  ``open_x`` with a periodic y is the case that
    made it matter.
    """
    if cfg.open_x:
        uw = state.u[:, :, 0]                          # (nz, ny) west face
        ue = state.u[:, :, -1]                         # east face
        outw, oute = uw < 0.0, ue > 0.0
        for q in (state.thp, state.qv, state.qc, state.qr):
            if q is None:
                continue
            q[:, :, 0] = cp.where(outw, q[:, :, 1], q[:, :, 0])
            q[:, :, -1] = cp.where(oute, q[:, :, -2], q[:, :, -1])
        if extra_moist_species(state):
            for name in extra_moist_species(state):
                q = getattr(state, name)
                q[:, :, 0] = cp.where(outw, q[:, :, 1], q[:, :, 0])
                q[:, :, -1] = cp.where(oute, q[:, :, -2], q[:, :, -1])
        oww = (uw[1:] + uw[:-1]) < 0.0                 # w levels 1..nz-1
        owe = (ue[1:] + ue[:-1]) > 0.0
        state.w[1:-1, :, 0] = cp.where(oww, state.w[1:-1, :, 1],
                                       state.w[1:-1, :, 0])
        state.w[1:-1, :, -1] = cp.where(owe, state.w[1:-1, :, -2],
                                        state.w[1:-1, :, -1])
        # The mask is (nz, ny) at mass rows and v needs (nz, ny+1) rows, so
        # the tangential axis has to supply one more.  WHICH one is the y
        # boundary condition's business, not a detail: on a NON-periodic y
        # the extra row is a real north face and takes the zero-gradient
        # repeat; on a PERIODIC y row ny is the ALIAS of row 0
        # (tilestream/spec.py's module docstring states the convention, and
        # `harness.make_state` seeds it) and must take row 0's mask, or the
        # two copies of one physical face are given opposite treatments.
        #
        # MEASURED at 96x64x25, open_x with y periodic, ONE resident step:
        # the repeat leaves max|v[ny] - v[0]| = 2.51 m/s, at exactly the two
        # x boundary columns and nowhere else, from a seeded state where it
        # was zero.  The periodic arm of the same probe stays at 0, which is
        # what shows the probe can see the invariant at all.  It is also
        # what made this configuration untileable: a tiled scatter writes the
        # alias FROM row 0 and so cannot reproduce a domain that has broken
        # its own alias.
        y_alias = slice(-1, None) if _boundary_y(cfg) else slice(0, 1)
        mvw = cp.concatenate([outw, outw[:, y_alias]], axis=1)  # v rows 0..ny
        mve = cp.concatenate([oute, oute[:, y_alias]], axis=1)
        state.v[:, :, 0] = cp.where(mvw, state.v[:, :, 1], state.v[:, :, 0])
        state.v[:, :, -1] = cp.where(mve, state.v[:, :, -2],
                                     state.v[:, :, -1])
    if cfg.open_y:
        vs = state.v[:, 0, :]                          # (nz, nx) south face
        vn = state.v[:, -1, :]                         # north face
        outs, outn = vs < 0.0, vn > 0.0
        for q in (state.thp, state.qv, state.qc, state.qr):
            if q is None:
                continue
            q[:, 0, :] = cp.where(outs, q[:, 1, :], q[:, 0, :])
            q[:, -1, :] = cp.where(outn, q[:, -2, :], q[:, -1, :])
        if extra_moist_species(state):
            for name in extra_moist_species(state):
                q = getattr(state, name)
                q[:, 0, :] = cp.where(outs, q[:, 1, :], q[:, 0, :])
                q[:, -1, :] = cp.where(outn, q[:, -2, :], q[:, -1, :])
        ows = (vs[1:] + vs[:-1]) < 0.0
        own = (vn[1:] + vn[:-1]) > 0.0
        state.w[1:-1, 0, :] = cp.where(ows, state.w[1:-1, 1, :],
                                       state.w[1:-1, 0, :])
        state.w[1:-1, -1, :] = cp.where(own, state.w[1:-1, -2, :],
                                        state.w[1:-1, -1, :])
        # The x mirror of the y rule above: with ``open_y`` and a PERIODIC x
        # (``open_x`` off and no specified/nested forcing) u's slot nx is the
        # alias of slot 0 and takes slot 0's mask.
        x_alias = slice(-1, None) if _boundary_x(cfg) else slice(0, 1)
        mus = cp.concatenate([outs, outs[:, x_alias]], axis=1)  # u faces 0..nx
        mun = cp.concatenate([outn, outn[:, x_alias]], axis=1)
        state.u[:, 0, :] = cp.where(mus, state.u[:, 1, :], state.u[:, 0, :])
        state.u[:, -1, :] = cp.where(mun, state.u[:, -2, :],
                                     state.u[:, -1, :])



def close_periodic_alias(state: DomainState, cfg: RunConfig) -> None:
    """On a periodic axis, the extra staggered face IS face 0.  Say so.

    ``u`` has ``nx+1`` faces for ``nx`` mass cells and ``v`` has ``ny+1`` for
    ``ny``.  When the axis wraps, the last of those is not a face of its own:
    it is the SAME FACE as index 0, reached the other way round.
    ``tilestream.spec`` builds every gather, scatter and halo band on exactly
    that identity (spec.py:34-52: "gathers never read the alias slot"; under
    ``periodic=True`` the alias slot is logical face 0), and
    ``TileSpec.scatter`` writes the domain's alias slot FROM face 0 for that
    reason.

    NOTHING WAS MAINTAINING IT.  No periodic stencil writes the slot -- they
    wrap their indices instead -- so it kept whatever the initialiser left,
    which for a window of a larger analysis is the real column one past the
    window: on this case's 128-wide window, ``u[:, :, nx]`` sat 17.4 m/s away
    from ``u[:, :, 0]`` at t=0 and stayed there.

    Consumers read it.  ``dycore._mass_divergence`` (:122-123) differences
    ``ru[:, :, 1:] - ru[:, :, :-1]``, so the last mass column's mass tendency
    came off that stale face, and ``physics._prepare_atmosphere``
    (physics.py:1379-1381) destaggers ``0.5*(u[:-1] + u[1:])``, so the last
    mass column and the top mass row of every surface-layer and PBL carrier
    were driven by a wind that is not part of the solution.  That is wrong on
    ONE GPU.

    It is also the whole of the single-vs-multi physics divergence.  A
    decomposition re-derives the slot from face 0 -- by the contract above,
    and unavoidably: the rank that holds the domain's alias slot receives it
    through the halo exchange from the rank that owns face 0 -- so the two
    arms fed different winds to the same cells and differed after ONE step,
    diffusely and nowhere near a rank seam.  MEASURED, 128x96x49, one step,
    full physics through ``MultiGPUDomain``: 61 of 158 carriers differ
    without this call, 0 with it, at 1x1, 1x2 and 2x2 alike.

    Closing it here, at the end of the step, is inert for a RANK: the halo
    exchange overwrites the tile's outermost faces before the next step, and
    the outermost mass column those faces feed is a halo column that
    ``TileSpec.scatter`` discards.  A NON-PERIODIC axis carries a real
    closing boundary face -- the specified/open lateral boundary owns it --
    and is left alone.
    """
    if not _boundary_x(cfg):
        state.u[..., -1] = state.u[..., 0]
    if not _boundary_y(cfg):
        state.v[..., -1, :] = state.v[..., 0, :]


def step(state: DomainState, cfg: RunConfig, *, acoustic: bool = True,
         mass_flux_observer=None, mass_flux_accumulator=None,
         refl_10cm_due: bool = False, fire_history_due: bool = False) -> None:
    """Advance ``state`` one full RK3 step of length ``cfg.dt``.

    ``acoustic=True`` (default) runs the full ARW split-explicit loop: per
    stage, zero tendencies -> EOS diagnostics at t* -> slow tendencies ->
    initialize the acoustic perturbations from the time-t fields -> acoustic
    substeps -> fold the perturbations into the new stage estimate.
    ``acoustic=False`` is the Phase-1 advection-only path (w, phi', mu'
    frozen).  ``mass_flux_observer``, when supplied, is called with each
    final-RK-stage acoustic substep's boundary-flux mass increment.  Summing
    those increments independently closes the completed step's domain-mass
    change and is otherwise a zero-cost dormant diagnostic.
    ``mass_flux_accumulator`` is the device-resident alternative (a
    :class:`MassFluxAccumulator`): it takes the same increments in the
    same order without reading any of them, so a long integration pays no
    per-substep host synchronization.  The two keywords are mutually
    exclusive and supplying both raises.
    ``refl_10cm_due`` is the history-step flag threaded to microphysics;
    the active scheme computes and stashes radar reflectivity before its
    finish-stage theta writeback and the post-microphysics EOS refresh.
    ``fire_history_due`` is the independent history alarm for firebrand
    landing likelihood and counters, including vapor-only fire cases.

    Config-gated physics (no-ops with the defaults): ``km_opt=1`` with
    ``cfg.khdif/kvdif > 0`` runs WRF's diff_opt=2 mixing package with the
    isotropic constant coefficients (:func:`launch_wrf_isotropic_km`;
    computed once per step from the time-t fields like km_opt 2-4);
    ``cfg.km_opt=4`` adds the WRF 2-D Smagorinsky horizontal mixing
    (:func:`add_smag2d_tendencies` -- computed once per step on stage 1,
    applied every stage; moisture via :func:`apply_smag2d_moisture`); and
    ``damp_opt=3`` engages the Klemp-Dudhia-Hassiotis implicit w-only
    Rayleigh damper inside the acoustic w solve (woof.core.acoustic; no-op
    on the ``acoustic=False`` path, which has no acoustic substeps); and
    ``diff_6th_opt > 0`` computes WRF's 6th-order monotonic horizontal
    forward tendency once from the time-t fields and feeds that held
    tendency to every RK stage (the advection-only test path skips it).

    Moist states (Task 5): each stage's transport fluxes come from the
    public :func:`stage_fluxes` surface, and qv/qc/qr advance right after
    the stage's acoustic loop (``woof.core.moist.advance_scalars_stage``;
    PD limiter on the final stage) so the next stage's EOS sees consistent
    (theta, qv).  The advection-only path stays dry.

    Microphysics (Task 6): with ``cfg.mp_physics != 0`` the configured
    scheme (``woof.core.microphysics.apply``) adjusts theta/qv/qc/qr once
    per step after the RK3 loop -- WRF solve_em's non-timesplit
    microphysics slot -- followed by a diagnostics refresh; the default
    ``mp_physics = 0`` leaves the step bitwise unchanged.  The scheme's
    clamped theta increment is retained as ``state.h_diabatic`` (K/s, WRF
    moist_physics_finish_em) and the NEXT step feeds it to every RK
    stage's theta tendency (:func:`add_h_diabatic_tendency`, WRF
    rk_addtend_dry) while the final stage's fold removes its accumulated
    net contribution (``_finish_small_steps``, WRF small_step_finish) --
    the dynamics see the heating continuously, the state is heated once.

    Open lateral boundaries + w-damping (Task 9 + Task 11 rework, acoustic
    path only): with ``cfg.open_x``/``open_y`` the advection kernels take
    WRF's open-aware bounds, each stage ADDS the radiative term at the
    boundary-normal velocity faces (:func:`apply_open_radiative_bc`), the
    acoustic substeps skip the boundary-face pressure gradient and clamp
    their cross-boundary ghost reads, and each stage ends with the
    zero-gradient-outbound boundary values
    (:func:`apply_open_zero_gradient`).  ``cfg.emdiv > 0`` engages WRF's
    external-mode divergence damping across the acoustic substeps
    (:func:`apply_emdiv_filter`).  ``cfg.w_damping = 1`` adds WRF's
    vertical-velocity limiter per stage (:func:`apply_w_damping`).  All
    no-ops with the defaults; the ``acoustic=False`` Phase-1 test path
    stays periodic.

    Unsupported combinations fail loudly here (and, for the config-only
    parts, in ``woof.config.load_config``): terrain (``terrain_opt != 0``
    or any nonzero ``state.ht``) with radiative-open boundaries raises
    ``NotImplementedError`` because its remaining stencils/bounds are
    periodic-only.  The non-monotonic
    ``diff_6th_opt = 1`` with moisture raises ``ValueError`` (unlimited
    fluxes bypass the PD limiter).  Coriolis/curvature is boundary-aware.
    """
    if cp is None:
        require_forecast_gpu()
    if getattr(cfg, "umove", 0.0) or getattr(cfg, "vmove", 0.0) or getattr(state, "_moving_frame_velocity", None):
        from woof.moving_frame import initialize_moving_frame
        initialize_moving_frame(state, cfg)
    if mass_flux_observer is not None and mass_flux_accumulator is not None:
        raise ValueError(
            "mass_flux_observer and mass_flux_accumulator are mutually "
            "exclusive: the accumulator exists to remove the per-substep "
            "host synchronization the list observer takes, and running "
            "both reinstates it")
    if acoustic and cfg.time_step_sound % 2 != 0:
        raise ValueError(
            f"time_step_sound must be even, got {cfg.time_step_sound}: RK3 "
            "stage 2 runs time_step_sound//2 acoustic substeps of "
            "dt/time_step_sound, which mis-times the dt/2 stage for odd "
            "values."
        )
    # THE fail-closed km_opt decision: this is the site that actually
    # decides whether a horizontal mixing operator runs, so it asks the
    # same shared question the loaders ask rather than restating it (the
    # two used to be separate transcriptions of one rule).
    validate_km_opt(cfg)
    if cfg.mp_physics != 0 and getattr(state, "h_diabatic", None) is None:
        raise ValueError(
            f"mp_physics={cfg.mp_physics} requires cfg.moist=True: the "
            "state carries no h_diabatic array for the retained "
            "microphysics heating")
    # Surface_w and acoustic advance_w_phi receive physical boundary flags;
    # their open-face terrain donors no longer wrap through the opposite side.
    if cfg.diff_6th_opt == 1 and state.qv is not None:
        raise ValueError(
            "diff_6th_opt=1 (non-monotonic) with moisture is not allowed: "
            "the unlimited 6th-order fluxes are applied outside the "
            "PD-limited transport and can drive qv/qc/qr negative; use "
            "the monotonic diff_6th_opt=2")
    if not acoustic and physics_enabled(cfg):
        raise NotImplementedError(
            "non-timesplit physics requires the acoustic RK3 path "
            "(step(acoustic=False) is the Phase-1 dry advection test path)")
    if getattr(state, '_host_setup_state', False):
        from woof.forecast_gpu import ForecastGPURequired

        raise ForecastGPURequired(
            'a NumPy setup state was passed to step; construct a CUDA forecast state')
    require_forecast_gpu()
    _save_time_t(state)
    zero_tendencies = _prepare_tendency_zero(state)

    # WRF solve_em: non-timesplit physics is evaluated once during the
    # first RK pass and held fixed for all three passes.  EOS/phy_prep must
    # see the time-t state.  The default scheme IDs are all zero, so this
    # branch performs no device operation for every frozen Phase-1/2 case.
    physics_tendencies = None
    stochastic_binding = getattr(state, "_ensemble_stochastic", None)
    if stochastic_binding is not None:
        stochastic_binding.before_physics(state, cfg)
    if physics_enabled(cfg):
        if state.physics is None:
            raise RuntimeError(
                "physics is enabled but the state has no PhysicsDriver; "
                "call woof.core.physics.initialize_physics first")
        if not _wrf_reuse_diagnostics(state):
            update_diagnostics(state, cfg.hypsometric_opt,
                               muts=_wrf_muts(state))
        physics_tendencies = state.physics.compute(state, cfg)
        # WRF rk_update_scalar saves scalar_old from the post-physics
        # carrier on stage 1 (module_em.F:1679,1722). Preserve the MYNN solve.
        if getattr(state, "qke_adv", None) is not None:
            state.qke_adv0[...] = state.qke_adv
    elif getattr(getattr(state, "physics", None), "cam_ozone", None) is not None:
        # A nested consumer can require root CAM ozone even with all local
        # schemes disabled. Run its common cadence, adding no tendencies.
        update_diagnostics(state, cfg.hypsometric_opt,
                           muts=_wrf_muts(state))
        state.physics.compute(state, cfg)

    if stochastic_binding is not None:
        physics_tendencies = stochastic_binding.after_nonmicrophysics(
            state, cfg, physics_tendencies)
    if getattr(state, "chem", None) is not None:
        # Native SFIRE emits into tracers before RK transport. The common
        # chem driver owns its source ledger and refreshes affected time-t
        # copies, which native rk_update_scalar copies after fire on RK1.
        from woof.core.chem_driver import chem_pre_transport
        chem_pre_transport(state, cfg, cfg.dt)

    # DA incremental analysis update (woof.da.iau.IauForcing): the analysis
    # increment as held coupled tendencies beside the physics', every RK
    # stage, like WRF's analysis-nudging slot.  A state without an attached
    # forcing takes the same object back and the step is unchanged.
    if getattr(state, "_iau_forcing", None) is not None:
        from woof.da.iau import fold_step_tendencies
        physics_tendencies = fold_step_tendencies(state, cfg,
                                                  physics_tendencies)

    if not acoustic:
        if state.qv is not None:
            raise NotImplementedError(
                "moist transport requires the acoustic dycore path "
                "(step(acoustic=False) is the Phase-1 dry advection test "
                "path)")
        if wrf_mixing_package_active(cfg):
            raise NotImplementedError(
                f"km_opt={cfg.km_opt} mixing requires the acoustic dycore "
                "path (step(acoustic=False) is the Phase-1 dry advection "
                "test path, and WRF's once-per-step mixing package reads "
                "the time-t copies only the acoustic path keeps)")
        for istage, dt_eff in enumerate((cfg.dt / 3.0, cfg.dt / 2.0,
                                         cfg.dt)):
            zero_tendencies()
            update_diagnostics(state, cfg.hypsometric_opt)
            add_advection_tendencies(state, cfg)
            apply_state_lateral_boundaries(state, cfg, rk_stage=istage)
            _advance_stage(state, dt_eff)
        set_w_surface(state, cfg)
        apply_state_boundary_values(state, cfg,
                                    state.elapsed_seconds + cfg.dt)
        state.elapsed_seconds += cfg.dt
        return

    # WRF module_first_rk_step_part2/rk_scalar_tend: compute the forward
    # mixing/diff6 tendencies once from the saved time-t fields.  Every RK
    # pass below consumes these same buffers; the final PD scalar pass folds
    # them before flux renormalization in advance_scalars_stage.
    # Smagorinsky's stresses/fluxes use WRF's dry-air density with vapor
    # loading, rho=(1+qv)/alt (exactly 1/alt for a dry state).  A freshly
    # initialized state has not otherwise run phy_prep/EOS yet, so refresh
    # the time-t diagnostics before evaluating K and its forward tendencies.
    if wrf_mixing_package_active(cfg) and not _wrf_reuse_diagnostics(state):
        update_diagnostics(state, cfg.hypsometric_opt,
                           muts=_wrf_muts(state))
    if cfg.gsd_diagnostics:
        from woof.core.gsd_diagnostics import update_helicity
        update_helicity(state, cfg, history_due=refl_10cm_due)
    prepare_fixed_tendencies(state, cfg, physics_tendencies)
    fixed_scalars = fixed_scalar_tendencies(state, cfg)
    if WRF_EXACT:
        # Folded into the held tendencies (_fold_physics_wrf); nothing adds
        # them again, per stage or per scalar.
        stage_tendencies = getattr(physics_tendencies, "strict_stage_tendencies", None)
        physics_tendencies = stage_tendencies() if stage_tendencies is not None else None

    ns = cfg.time_step_sound
    stages = ((1, cfg.dt / 3.0),
              (max(ns // 2, 1), cfg.dt / ns),
              (ns, cfg.dt / ns))
    emdiv = cfg.emdiv > 0.0
    nzs, nys, nxs = state.p.shape
    launch_small_step_init = _prepare_small_step_init_launch(state, cfg)
    if WRF_EXACT:
        exact_small_step_inits = tuple(
            _prepare_small_step_init_launch(state, cfg, rk_step)
            for rk_step in (1, 2, 3))
    launch_small_step_finish = _prepare_small_step_finish_launch(state, cfg)
    if cfg.mp_physics != 0:
        final_hdiab_dt = stages[-1][0] * stages[-1][1]
        launch_small_step_finish_final = _prepare_small_step_finish_launch(
            state, cfg, final_hdiab_dt)
    else:
        launch_small_step_finish_final = launch_small_step_finish
    small_step_finishes = (
        launch_small_step_finish, launch_small_step_finish,
        launch_small_step_finish_final)
    for istage, (nsub, dtau) in enumerate(stages):
        zero_tendencies()
        if not (istage == 0 and _wrf_reuse_diagnostics(state)):
            update_diagnostics(state, cfg.hypsometric_opt,
                               muts=_wrf_muts(state))  # p, al, alt at t*
        ru, rv, ww = stage_fluxes(state, cfg, first_stage=(istage == 0))
        # WRF calc_cq is fixed at the RK-stage reference state and shared by
        # horizontal_pressure_gradient plus every acoustic substep.
        stage_cq = prepare_moist_cq(state, cfg)
        # zadvect_implicit = 1: WRF CHK_IEVA is true on the last substep.
        if ieva.active_stage(cfg, istage, len(stages)):
            _add_slow_tendencies(state, cfg, ru, rv, ww, cq=stage_cq,
                                 implicit=ieva.prepare_dynamics(
                                     state, cfg, ww))
        else:
            _add_slow_tendencies(state, cfg, ru, rv, ww, cq=stage_cq)
        if istage == 0 or (WRF_EXACT and cfg.cu_physics == 3):
            # WRF RTHFTEN for the cumulus schemes that take it.  THIS LINE
            # is the whole contract: rth_t holds the stage reference
            # fluxes' theta advection and nothing else until the next
            # statement folds physics into it (see the function's
            # docstring for the GFSCHEME double-count trap).  Stage 1
            # matches the once-per-step capture convention h_diabatic and
            # the qv lateral tendency already use.
            capture_advective_theta_forcing(state, cfg=cfg)
        if WRF_EXACT:
            # WRF solve_em order: rk_tendency (w_damp inside, see
            # _add_slow_tendencies), relax_bdy_dry folded into the held
            # *_tendf on rk_step 1, rk_addtend_dry's tend + tendf/msf and
            # then its h_diabatic term in the same statement, spec_bdy_dry.
            implicit_stage = ieva.active_stage(cfg, istage, len(stages))
            if implicit_stage:
                apply_w_damping(state, cfg, ww)
            if istage == 0 and _wrf_relaxation_fold(cfg):
                fold_lateral_relaxation(state, cfg)
            add_fixed_dry_tendencies(state, cfg)
            if physics_tendencies is not None:
                physics_tendencies.add_to_slow(state)
            if cfg.mp_physics != 0:
                add_h_diabatic_tendency(state)
            record_wrf_vertical_cfl(state, cfg, ww)
            apply_state_lateral_boundaries(state, cfg, rk_stage=istage)
        else:
            if physics_tendencies is not None:
                physics_tendencies.add_to_slow(state)
            if cfg.mp_physics != 0:                   # WRF rk_addtend_dry's
                add_h_diabatic_tendency(state)        # h_diabatic slot, every
            add_fixed_dry_tendencies(state, cfg)       # held Smag/diff6 tendf
            apply_w_damping(state, cfg, ww)           # w_damping=1 only
            record_wrf_vertical_cfl(state, cfg, ww)  # probe; off by default
            apply_state_lateral_boundaries(state, cfg, rk_stage=istage)
        apply_open_radiative_bc(state, cfg)           # open_x/open_y only
        if WRF_EXACT:
            exact_small_step_inits[istage]()
            if cfg.cu_physics in (3, 6):
                # Stock leaves reference WW in the ring that advance_mu_th
                # does not overwrite. sumflux still reads that ring.
                state.ww_pp[...] = ww
            _init_wrf_acoustic_muts(state, istage)
        else:
            launch_small_step_init()                 # additive stage seed
        acoustic_coefficients = prepare_acoustic_coefficients(
            state, cfg, dtau, cq=stage_cq)            # fixed for this stage
        mudf = None
        if emdiv:
            mudf = state.scratch(state.mup.shape, "acoustic_mudf")
            launch_emdiv_filter = _prepare_emdiv_filter_launch(
                state, cfg, mudf)
            if istage == 0:
                # WRF small_step_prep zeros MUDF under IF(rk_step==1) ONLY
                # (module_small_step_em.F:128-136); stages 2/3 must inherit
                # the previous stage's final column-mass tendency so emdiv
                # acts on 6 of 7 acoustic iterations, not 4 of 7.
                mudf[...] = 0
        # WRF sumflux (module_small_step_em.F:1473, called every acoustic
        # iteration, solve_em.F:1561): accumulate the small-timestep
        # time-averaged mass fluxes ru_m/rv_m/ww_m -- "needed for
        # consistent mass-conserving scalar advection".  Scalars only;
        # theta/momentum keep the stage fluxes exactly as WRF does.
        moist = state.qv is not None
        # Prognostic TKE (km_opt=2) advects with the same acoustic
        # time-averaged mass fluxes as the moist scalars (WRF
        # rk_scalar_tend for tke, solve_em.F:2362-2399), so a dry TKE run
        # accumulates sumflux too.
        chem = getattr(state, "chem", None)
        # Chem species advect with the same acoustic time-averaged fluxes
        # (WRF rk_scalar_tend for chem, solve_em.F:2476-2500), so a chem run
        # accumulates sumflux even on a dry state.
        scalars = (moist or getattr(state, "tke", None) is not None
                   or chem is not None or getattr(state, "qke_adv", None) is not None)
        if scalars:
            ru_m = state.scratch((nzs, nys, nxs + 1), "rk_ru_m")
            rv_m = state.scratch((nzs, nys + 1, nxs), "rk_rv_m")
            ww_m = state.scratch((nzs + 1, nys, nxs), "rk_ww_m")
        # The joined substep launches (woof/core/acoustic_fusion.py) submit
        # the emdiv increment and the sumflux zero/accumulation themselves
        # where they cover this configuration; ``absorbs`` names which.
        launch_acoustic_substep = prepare_acoustic_substep_launch(
            state, cfg, dtau, acoustic_coefficients, mudf=mudf,
            emdiv_args=launch_emdiv_filter.args if emdiv else None,
            sumflux=(ru_m, rv_m, ww_m) if scalars else None)
        absorbed = getattr(launch_acoustic_substep, "absorbs", frozenset())
        separate_sumflux = scalars and "sumflux" not in absorbed
        if separate_sumflux:
            _sumflux_launch(                          # sumflux iteration==1
                "zero_sumflux", (ru_m, rv_m, ww_m))
            launch_sumflux_accumulation = _prepare_sumflux_launch(
                "accumulate_sumflux", (ru_m, rv_m, ww_m),
                (state.u_pp, state.v_pp, state.ww_pp))
        separate_emdiv = emdiv and "emdiv" not in absorbed
        for i in range(nsub):
            if separate_emdiv:
                launch_emdiv_filter()                 # previous substep's
                                                      # mudf (zero only on
                                                      # step's 1st substep)
            launch_acoustic_substep(first=(i == 0))
            if separate_sumflux:                      # WRF sumflux: post-
                launch_sumflux_accumulation()
            if mass_flux_observer is not None and istage == 2:
                mass_flux_observer(
                    dtau * boundary_mass_tendency(state, cfg))
            elif mass_flux_accumulator is not None and istage == 2:
                mass_flux_accumulator.add(
                    dtau * boundary_mass_tendency_device(state, cfg))
            # advance_mu_th stores WRF MUDF directly before its rounded mass
            # update; reconstructing it from two FP32 states loses sub-ULP
            # column-mass tendencies.
        # WRF small_step_finish: the h_diabatic removal runs on the final RK
        # step only, over dts*number_of_small_timesteps = dt.
        if (scalars and cfg.damp_opt == 3
                and getattr(cfg, "upper_wind_limiter_form", "wrf_461") == "noaa_wrf39"):
            refresh_saved_wind_fluxes(state, cfg, ru, rv)
        if WRF_EXACT:
            _capture_wrf_muts(state, cfg)
        small_step_finishes[istage]()
        if istage == 0:
            add_scalar_coordinate_wrf(state, cfg)     # strict coordinate branch
            add_scalar_diff6_wrf(state, cfg)          # strict mode only
        if scalars:                                   # stage length nsub*dtau
            # WRF sumflux (iteration == number_of_small_timesteps): the
            # substep mean plus the stage-reference coupled fluxes --
            # ru_m = mean(u'') + C(muu)*u_t*/msfuy (F:1584-1592), ww_m =
            # mean(ww'') + ww_1.  Scalars advect with these (solve_em.F:
            # 2210-2212), making the q == const tendency telescope exactly
            # against the acoustic mu update (SK2008 D11).
            _sumflux_launch(
                "finish_sumflux", (ru_m, rv_m, ww_m), (ru, rv, ww), nsub)
            # rk_scalar_tend's IEVA split (module_em.F:1216-1242): ww_m
            # with the time-n winds and the post-acoustic column mass,
            # shared by every transported scalar of this substep.
            # The keyword is passed only when on, so the default call is
            # the call it always was.
            scalar_implicit = {}
            if ieva.active_stage(cfg, istage, len(stages)):
                scalar_implicit["implicit"] = ieva.split_scalar_omega(
                    state, cfg, ww_m, state.mub2d + state.mup, nsub * dtau)
            if moist or getattr(state, "qke_adv", None) is not None:
                advance_scalars_stage(
                    state, cfg, ru_m, rv_m, ww_m, nsub * dtau,
                    final=(istage == len(stages) - 1),
                    apply_relax=(istage == 0),
                    physics_tendencies=physics_tendencies,
                    fixed_tendencies=fixed_scalars,
                    # WRF RQVFTEN, the qv half of the cumulus advective
                    # forcing pair.  Stage 1, where the non-PD branch runs
                    # and the tendency is still pure advection.
                    export_advective_forcing=(istage == 0),
                    **scalar_implicit)
            if getattr(state, "tke", None) is not None:
                from woof.core.moist import advance_tke_stage
                advance_tke_stage(
                    state, cfg, ru_m, rv_m, ww_m, nsub * dtau,
                    final=(istage == len(stages) - 1),
                    fixed_tendency=state.scratch(
                        state.p.shape, "smag_rtke"),
                    **scalar_implicit)
            if chem is not None:
                # WRF chem_scalar_advance (solve_em.F:2466-2616), after the
                # moist and TKE scalars exactly as WRF orders them.
                from woof.core.chem_transport import (
                    advance_chem_stage, chem_fixed_tendencies)
                advance_chem_stage(
                    state, cfg, ru_m, rv_m, ww_m, nsub * dtau,
                    final=(istage == len(stages) - 1),
                    fixed_tendencies=chem_fixed_tendencies(state, cfg),
                    **scalar_implicit)
        apply_open_zero_gradient(state, cfg)          # radiative-open BCs
    # km_opt=2 budget: one device reduction over the completed step, taken
    # here because state.mup is still the mass the final RK scalar update
    # divided by.  Report-only (woof/core/tke_budget.py); a no-op unless
    # cfg.tke_budget is on.
    tke_budget.accumulate(state, cfg)
    # The opt-in analysis operator owns one captured graph. All arithmetic,
    # field masks, cadence and the forcing clock remain on the device.
    nudging_graph = getattr(state, "_spectral_nudging_graph", None)
    if nudging_graph is not None:
        nudging_graph.launch()
    # Whether this step ends on a history time: the streamed/ranked post-step
    # hook builds the chem history fields (the GOCART 550 nm optics among
    # them) only then (woof/core/streaming.py refl_handoff_hook).
    # Held on the chem object, not the state: a state attribute would join
    # the carrier inventory the rank seam plan is built from.
    if getattr(state, "chem", None) is not None:
        state.chem.history_due = bool(refl_10cm_due)
    if WRF_EXACT:
        _wrf_step_epilogue(state, cfg, refl_10cm_due)
    else:
        _step_epilogue(state, cfg, refl_10cm_due)
    fire = getattr(getattr(state, "physics", None), "fire", None)
    spotting = getattr(fire, "spotting", None)
    if spotting is not None:
        # Native solve_em calls spotting after its completed atmosphere and
        # tracer solve/halos; the external chemistry operator follows it.
        spotting.advance_atmosphere(state, cfg, fire, dt=cfg.dt,
                                     history_alarm=fire_history_due)
    if getattr(state, "chem", None) is not None:
        # WRF calls chem_driver after solve_em (share/solve_interface.F),
        # i.e. after the whole dynamics and microphysics step and before
        # the clock advances; tiles and nests reach it through this step.
        from woof.core.chem_driver import chem_step
        chem_step(state, cfg, cfg.dt)
    close_periodic_alias(state, cfg)
    state.elapsed_seconds += cfg.dt


def _step_epilogue(state: DomainState, cfg: RunConfig,
                   refl_10cm_due: bool) -> None:
    """The default step end: boundary values, surface w and diagnosis,
    then the nwp diagnostics and the microphysics adjustment."""
    finish = getattr(state.physics, "finish_history_mass", None)
    if finish is not None:  # stand-in PBL/physics drivers carry no history rows
        finish(state, cfg)
    apply_state_boundary_values(state, cfg,
                                state.elapsed_seconds + cfg.dt)
    set_w_surface(state, cfg)                         # WRF solve_em epilogue
    update_diagnostics(state, cfg.hypsometric_opt)
    if cfg.nwp_diagnostics == 1:
        # WRF's nwp_diagnostics severe-weather lane, UP_HELI_MAX member:
        # fold this completed step's updraft helicity into the running max
        # (u/v/w/ph are final here; the later microphysics adjustment does
        # not touch them).  Reads model state only; writes only the
        # diagnostic's own scratch slots -- inertness is pinned by
        # tests/test_uh_lifecycle.py.
        update_up_heli_max(state, cfg)
    # WRF lightning diagnostics (lightning_option): off touches nothing here.
    lightning_refl_due = False
    if cfg.lightning_option:
        from woof.core.lightning import lightning_due, update_lightning
        lightning_refl_due = (cfg.lightning_option in (1, 2)
                              and lightning_due(state, cfg))
    if cfg.mp_physics != 0:                           # post-RK3 adjustment
        # h_diabatic capture cadence: once per INTERNAL step with cfg.dt.
        # Under the reference case compatibility integrator (dt = clock_dt/8) this
        # deviates from WRF's once-per-model-clock-step cadence, ratified
        # and documented in PROVENANCE.md entry D1 (self-consistent: the
        # capture dt and apply window are the same internal step; the
        # native dt=60 path restores WRF cadence with no code change).
        microphysics_result = apply_microphysics(     # (WRF microphysics
            state, cfg, cfg.dt, refl_10cm_due=(refl_10cm_due or lightning_refl_due))
        if state.physics is not None:
            state.physics.accept_microphysics(
                microphysics_result, dt=cfg.dt)
        update_diagnostics(state, cfg.hypsometric_opt)  # after the RK loop)
    if cfg.lightning_option:
        update_lightning(state, cfg, history_reflectivity_due=refl_10cm_due)
    from woof.core.gsd_diagnostics import reset_interval, update as update_gsd
    reset_interval(state, cfg)
    from woof.core.hail import update as update_hail
    update_hail(state, cfg)
    update_gsd(state, cfg, history_due=refl_10cm_due)
    from woof.core.output_diagnostics import update_precipitation
    update_precipitation(state, cfg)
    from woof.core.climate_diagnostics import update as update_climate
    update_climate(state, cfg)


def _wrf_step_epilogue(state: DomainState, cfg: RunConfig,
                       refl_10cm_due: bool) -> None:
    """Strict mode: WRF solve_em's order after the RK loop.

    calc_p_rho_phi runs at the end of the last RK stage, after its scalar
    update; then the microphysics and its own calc_p_rho_phi; only then
    spec_bdy_final and set_w_surface.  The history P is therefore the
    pre-spec_bdy_final diagnosis on the specified ring, and rk_step 1 of
    the next step reads these p/al without diagnosing again
    (:func:`_wrf_reuse_diagnostics`).  The default order -- boundary
    values first, then the diagnosis -- moved P on the ring (337 words at
    the first step of the round-3 pair).
    """
    finish = getattr(state.physics, "finish_history_mass", None)
    if finish is not None:  # stand-in PBL/physics drivers carry no history rows
        finish(state, cfg, muts=_wrf_muts(state))
    update_diagnostics(state, cfg.hypsometric_opt, muts=_wrf_muts(state))
    # WRF lightning diagnostics (lightning_option): off touches nothing here.
    lightning_refl_due = False
    if cfg.lightning_option:
        from woof.core.lightning import lightning_due, update_lightning
        lightning_refl_due = (cfg.lightning_option in (1, 2)
                              and lightning_due(state, cfg))
    if cfg.mp_physics != 0:
        microphysics_result = apply_microphysics(
            state, cfg, cfg.dt, refl_10cm_due=(refl_10cm_due or lightning_refl_due))
        if state.physics is not None:
            state.physics.accept_microphysics(
                microphysics_result, dt=cfg.dt)
        update_diagnostics(state, cfg.hypsometric_opt,
                           muts=_wrf_muts(state))
    if cfg.specified or cfg.nested:
        state.scratch(state.mup.shape, WRF_MU_HALO_SLOT)[...] = state.mup
    apply_state_boundary_values(state, cfg,
                                state.elapsed_seconds + cfg.dt)
    set_w_surface(state, cfg)
    if cfg.nwp_diagnostics == 1:
        update_up_heli_max(state, cfg)
    # Passive diagnostic (its inertness receipts hash every serialized
    # field): the same place in the step as the default epilogue.
    if cfg.lightning_option:
        update_lightning(state, cfg, history_reflectivity_due=refl_10cm_due)
    from woof.core.gsd_diagnostics import reset_interval, update as update_gsd
    reset_interval(state, cfg)
    from woof.core.hail import update as update_hail
    update_hail(state, cfg)
    update_gsd(state, cfg, history_due=refl_10cm_due)
    from woof.core.output_diagnostics import update_precipitation
    update_precipitation(state, cfg)
    from woof.core.climate_diagnostics import update as update_climate
    update_climate(state, cfg)


def _wrf_reuse_diagnostics(state: DomainState) -> bool:
    """Strict mode: rk_step 1 reads the previous step's p/al as they are.

    True once an acoustic loop has left WRF's muts in this process (the
    previous step's end diagnosed them); a resumed or freshly initialized
    state diagnoses as start_em does.  A WRF-file nest's first step reads
    the p/al its start_domain left, before press_adj moved MU and the first
    force round-tripped its prognostics
    (woof.ingest.wrfinput_nest_initial.finish_strict_nest_start).
    """
    return WRF_EXACT and (_wrf_muts(state) is not None or bool(
        getattr(state, "_wrf_start_diagnostics_carried", False)))


def run_steps(state: DomainState, cfg: RunConfig, n: int, *,
              acoustic: bool = True) -> None:
    """Advance ``state`` by ``n`` full RK3 steps."""
    for _ in range(n):
        step(state, cfg, acoustic=acoustic)


def stability_report(state: DomainState, cfg: RunConfig | None = None,
                     *, boundary_width: int | None = None) -> dict:
    """Runtime health check with one compact device-to-host result readback.

    The two-stage device reduction returns max |u|, |w|, and |theta'| plus
    max ``|w_upper| / dz_cell`` with each upper-face velocity paired with
    its own live geopotential layer thickness.  NaNs propagate through the
    same maxima, so ``"nan"`` checks exactly the same fields as before; bad
    layer geometry makes the CFL non-finite and therefore fails the runner's
    safety gate.  The same traversal returns the flat index of the |w|
    maximum (``w_argmax``, lowest index on a tie) on every call; when
    ``boundary_width`` is supplied it also returns the boundary and
    free-interior |w| maxima used by the reference case integration monitor.
    """
    if state.u.size == 0 or state.w.size == 0 or state.thp.size == 0:
        raise ValueError(
            "zero-size array to reduction operation maximum which has no "
            "identity")
    width = 0 if boundary_width is None else int(boundary_width)
    if boundary_width is not None:
        ny, nx = state.w.shape[1:]
        if width <= 0:
            raise ValueError("boundary_width must be positive")
        if min(ny, nx) < boundary_axis(width, interior_points=1):
            raise ValueError(
                f"boundary_width={width} leaves an empty w interior for "
                f"{ny} x {nx}")
    largest = max(state.u.size, state.w.size, state.thp.size)
    nblocks = min(256, max(1, (largest + 255) // 256))
    partial = state.scratch((nblocks, 9), "integration_health_partial")
    result = state.scratch((8,), "integration_health_result")
    if cfg is None:
        ph = phb = state.w  # ncells=0 below: valid, never dereferenced
        ncells = 0
        phb_full = 0
    else:
        ph = state.php
        phb = state.phb
        ncells = state.thp.size
        phb_full = int(state.phb.ndim == 3)
    kernel = get_kernel("health", "health_partial")
    kernel((nblocks,), (256,),
           (state.u, state.w, state.thp, ph, phb, partial,
            np.uint64(state.u.size), np.uint64(state.w.size),
            np.uint64(state.thp.size), np.uint64(ncells),
            np.int32(phb_full), np.int32(width),
            np.int32(state.w.shape[1]), np.int32(state.w.shape[2]),
            DTYPE(c.G)))
    kernel = get_kernel("health", "health_final")
    kernel((1,), (256,), (partial, result, np.int32(nblocks)))
    host = cp.asnumpy(result)                           # sole health readback
    return decode_stability_record(host, cfg, boundary_width=boundary_width)


def decode_stability_record(host, cfg: RunConfig | None = None, *,
                            boundary_width: int | None = None) -> dict:
    """``health_final``'s eight-word record, as :func:`stability_report`'s dict.

    Factored out so a STREAMED domain, whose record is folded per tile inside
    the sweep (:mod:`woof.core.streaming`), decodes through this exact
    function rather than through a copy of it.  The two paths differ in which
    memory the reduction READS and in nothing else, which is the whole claim
    that fold rests on -- and a duplicated decoder is how such a claim stops
    being true a year later.
    """
    u_max, w_max, th_max = (float(value) for value in host[:3])
    nan = not (math.isfinite(u_max) and math.isfinite(w_max)
               and math.isfinite(th_max))
    cfl = None
    horizontal_cfl = None
    vertical_cfl = None
    if cfg is not None and not nan:
        horizontal_cfl = cfg.dt * u_max / cfg.dx
        vertical_cfl = cfg.dt * float(host[5])
        # NOT max(): CPython seeds the running maximum with the FIRST
        # argument and replaces it only when ``item > maxval``, and
        # ``nan > x`` is False -- so ``max(horizontal, nan)`` returns the
        # HORIZONTAL number and silently discards the NaN.  That NaN is
        # the ONLY channel ``health_final`` has for bad layer geometry
        # (``health.cu`` sets mask bit 32 when a mass cell's live
        # thickness is non-positive or non-finite and writes
        # ``result[5] = nanf("")`` for it), so discarding it left a
        # collapsed or folded model layer -- the classic precursor of an
        # ARW vertical blow-up -- with no observer at all, while every
        # field was still finite and this function's own docstring said
        # the CFL went non-finite.  A non-finite vertical rate has to
        # survive into ``cfl``, which is the number
        # :func:`stability_gate_failed` tests for finiteness.
        cfl = (vertical_cfl if not math.isfinite(vertical_cfl)
               else max(horizontal_cfl, vertical_cfl))
    # WHERE the |w| maximum is, on every record and not only when a
    # boundary split was asked for: both health kernels (``health.cu`` and
    # ``health_tile.cu``) reduce the argmax on every launch, so the index
    # was always in the record and simply not read.  A run that climbs to
    # a blow-up needs it -- the non-finite survey that runs afterwards can
    # say only which box had already gone, and the last finite maximum's
    # place is the nearest thing to where it started.  The index is into
    # w's own (nz+1, ny, nx) grid; with every |w| non-finite it is the
    # kernel's all-ones sentinel, which no reader uses because ``nan`` is
    # then true.
    index_words = np.asarray(host[6:8], dtype=np.float32).view(np.uint32)
    w_argmax = int(index_words[0]) | (int(index_words[1]) << 32)
    report = {"u_max": u_max, "w_max": w_max, "th_max": th_max,
              "cfl": cfl, "horizontal_cfl": horizontal_cfl,
              "vertical_cfl": vertical_cfl, "nan": nan,
              "w_argmax": w_argmax}
    if boundary_width is not None:
        report.update(
            boundary_w_max=float(host[3]), interior_w_max=float(host[4]))
    return report


def stability_gate_failed(report: dict, *, max_cfl: float,
                          max_w_ms: float) -> bool:
    """True when a single-domain history sample crosses a safety limit.

    Equality remains accepted, preserving the established threshold
    convention; the first representable value above either limit fails.
    """

    cfl = report.get("cfl")
    w_max = report.get("w_max")
    return (
        bool(report.get("nan"))
        or cfl is None
        or not math.isfinite(float(cfl))
        or float(cfl) > max_cfl
        or w_max is None
        or not math.isfinite(float(w_max))
        or float(w_max) > max_w_ms
    )


#: The state carriers a non-finite survey reads, each under the name the
#: history file gives it.  The ORDER is a reading order and nothing more:
#: dynamics first, then the moisture species.  It is not the order the
#: fields failed in, which nothing records -- the survey runs once, at the
#: health check that found the record non-finite, so every carrier it
#: lists was already gone by then, and one that failed a hundred steps
#: after another is listed in the same place.  A carrier the configuration
#: never allocated is simply absent from the state and skipped, which is
#: why this is a table rather than a fixed sequence of reads.
NONFINITE_SURVEY_CARRIERS = (
    ("w", "W"), ("u", "U"), ("v", "V"), ("thp", "T"), ("php", "PH"),
    ("mup", "MU"), ("p", "P"), ("al", "AL"), ("alt", "ALT"),
    ("tke", "TKE"),
    ("qv", "QVAPOR"), ("qc", "QCLOUD"), ("qr", "QRAIN"),
    ("qi", "QICE"), ("qs", "QSNOW"), ("qg", "QGRAUP"),
)

#: The axis letters a surveyed carrier's index is reported under, by rank.
#: Three-dimensional carriers are (k, j, i) -- the order a WRF reader
#: already thinks in -- and a two-dimensional one is (j, i) rather than a
#: (k, j, i) with a fabricated level.
NONFINITE_SURVEY_AXES = {3: ("k", "j", "i"), 2: ("j", "i"), 1: ("k",)}

#: The lateral edges a box can reach, in the order they are named, as
#: (axis letter, which end, name).  j runs south to north and i west to
#: east, as in every WRF grid.
NONFINITE_SURVEY_EDGES = (("j", 0, "south"), ("j", 1, "north"),
                          ("i", 0, "west"), ("i", 1, "east"))


def nonfinite_box_edges(bounding_box, shape) -> list[str]:
    """The lateral edges a surveyed box reaches, on its carrier's own grid.

    Judged against the carrier's OWN extents, because a staggered carrier
    is one wider along its stagger: U's last column is ``nx``, V's last
    row is ``ny``, and a box that reaches either is on the domain's edge
    exactly as a mass field's box that reaches ``nx - 1`` is.  A box with
    no j or no i axis (a column-only carrier) reaches no lateral edge.
    """

    box = dict(bounding_box or {})
    shape = [int(value) for value in (shape or ())]
    if len(shape) < 2:
        return []
    extents = {"j": shape[-2], "i": shape[-1]}
    edges = []
    for axis, end, name in NONFINITE_SURVEY_EDGES:
        bounds = box.get(axis)
        if bounds is None:
            continue
        low, high = int(bounds[0]), int(bounds[1])
        if (low == 0) if end == 0 else (high == extents[axis] - 1):
            edges.append(name)
    return edges


def nonfinite_field_survey(state, *, carriers=NONFINITE_SURVEY_CARRIERS) -> dict:
    """WHICH carriers were non-finite, over WHAT box, and how many cells.

    :func:`decode_stability_record` answers "is anything non-finite" with
    one bit, because that is all its eight-word reduction can carry: its
    ``nan`` is a finiteness test on three MAXIMA (u, w, theta') and the
    record holds no field name and no index of a non-finite value.  So a
    run that blew up could say the step it happened on and nothing else
    about it, and "at step 6624" is the one sentence that tells a reader
    to go and re-run the thing to find out more.

    This is the survey taken ONCE, on the failure path, after that bit
    comes back true.  It is a full pass over the allocated carriers and
    it costs a bool temporary per field, which is why it is not on the
    per-step route: a run pays for it exactly when it is already over.

    WHAT IT CANNOT SAY, and therefore does not: where a field went
    non-finite FIRST.  The health check runs every health interval (60
    model seconds, 48 to 144 steps of a downscaled child) and reads only
    the u, w and theta' maxima, so by the time it fires the non-finite set
    has spread for up to that many steps, and a carrier the check does not
    read may have gone long before.  An earlier version reported the
    lowest memory-order index of that set as the "first" cell; in
    (k, j, i) order that is always the set's lowest level and its
    southmost row, so a plume aloft that had spread down its column was
    reported at k=0 on the south edge of its block.  The box and the count
    are what the survey actually measured, so they are what it reports.

    Returns ``{"fields": [...], "surveyed": [names]}`` with one entry per
    non-finite carrier carrying its cell count, the bounding box the
    whole set falls inside, and the lateral edges that box reaches
    (:func:`nonfinite_box_edges`) -- the numbers that distinguish one bad
    cell from a column, a column from a plume, a plume from a field that
    has gone entirely, and an interior blow-up from one at the boundary.
    A carrier with exactly one bad cell also carries that ``cell``, which
    is then a measurement rather than a choice.  A state whose carriers
    are all finite returns an empty ``fields``, which is itself a
    reading: the record said non-finite and the fields do not agree.
    """

    fields = []
    surveyed = []
    for attribute, name in carriers:
        array = getattr(state, attribute, None)
        if array is None or not hasattr(array, "shape"):
            continue
        if getattr(array, "size", 0) == 0 or array.ndim == 0:
            continue
        surveyed.append(name)
        xp = cp.get_array_module(array)
        bad = xp.logical_not(xp.isfinite(array))
        count = int(bad.sum())
        if not count:
            del bad
            continue
        axes = NONFINITE_SURVEY_AXES.get(
            array.ndim, tuple(f"a{rank}" for rank in range(array.ndim)))
        box = {}
        for rank, label in enumerate(axes):
            other = tuple(n for n in range(array.ndim) if n != rank)
            present = bad.any(axis=other) if other else bad
            where = xp.nonzero(present)[0]
            box[label] = [int(where[0]), int(where[-1])]
        shape = [int(value) for value in array.shape]
        entry = {
            "field": name,
            "carrier": attribute,
            "shape": shape,
            "size": int(array.size),
            "count": count,
            "bounding_box": box,
            "edges": nonfinite_box_edges(box, shape),
        }
        if count == 1:
            entry["cell"] = {label: bounds[0]
                             for label, bounds in box.items()}
        fields.append(entry)
        del bad
    return {"fields": fields, "surveyed": surveyed}


def format_survey_cell(cell) -> str:
    """One surveyed index as a reader types it: ``(k=12, j=401, i=388)``."""

    return "(" + ", ".join(f"{label}={value}"
                           for label, value in cell.items()) + ")"
