"""Moisture: positive-definite scalar transport and moist thermodynamics.

Transport (WRF ``module_advect_em.F`` ``advect_scalar_pd``; Skamarock 2006
MWR; ``moist_adv_opt=1``): qv/qc/qr are dry-mass-coupled scalars advected
through all three RK3 stages with WRF's acoustic-substep TIME-AVERAGED
mass fluxes ru_m/rv_m/ww_m (``sumflux``, module_small_step_em.F:1473 --
"needed for consistent mass-conserving scalar advection";
solve_em.F:2210-2212), accumulated by ``dycore.step`` as the substep mean
of the perturbation momenta/Omega'' plus the stage-reference fluxes from
:func:`woof.core.dycore.stage_fluxes`.  Theta and momentum keep the
stage fluxes exactly as WRF's ``rk_tendency`` does; moisture never
re-derives Omega (Task 5 retired the ``rw = -(mu*w)`` placeholder in
advection.py).
Stages 1-2 use the unlimited flux divergence (v_sca_adv_order vertical)
(advection.py/advection.cu); the RK3 FINAL stage only -- exactly as WRF
applies its PD filter -- decomposes every face flux as ``F = F_upwind1 +
F_corr`` and renormalizes the outgoing corrections of any cell they would
overdraw (kernels/pd_advection.cu: ``pd_fluxes`` + ``pd_renorm_apply``;
float64 mirrors ``np_pd_fluxes``/``np_pd_renorm_apply`` in
woof.verify.npref).

The renormalization is positive in exact arithmetic at both vertical
orders.  Order 5 replaces the low-order eta flux above face Courant 1 with
the fork's semi-Lagrangian sum of the upstream cells; below it keeps the
upwind flux.  (The fork's own small-Courant branch takes the downstream
cell, drains empty cells and, through the clamp below, manufactured
scalar mass; only the strict WRF verification build carries it,
kernels/pd_vertical_sl.cu.)  The final
stage update clamps FP32 rounding residuals (order 1 ulp of the
renormalized outflow, in cells the limiter already drove to ~0) at zero,
which perturbs total scalar mass far below the 1e-6 relative conservation
gate.

Thermodynamics (ARW Tech Note ch. 2; dry-theta prognostic, WRF
``use_theta_m=0`` style): the EOS diagnostics use ``theta_m = theta *
(1 + Rv/Rd*qv)`` (diagnostics.cu; the ratio comes from the constants), the
w-equation buoyancy takes WRF's full moist ``pg_buoy_w`` form -- the
``1/(1+q_tot)`` factor on the vertical p' gradient plus the hydrometeor/
vapor loading ``-q_tot/(1+q_tot)*(c1f*mub+c2f)`` with ``q_tot = qv+qc+qr``
(dycore._add_slow_tendencies) -- and the acoustic coefficients pick up
theta_m through the moist pressure in ``c2a = gamma*p/alpha``.  The
linearized-EOS ratios ``(mu*theta)''/theta_ref`` remain invariant under the
per-cell ``(1 + Rv/Rd*qv)`` factor while qv is frozen over the substeps.

The acoustic path also matches WRF ``calc_cq`` each RK stage: ``cqu`` and
``cqv`` scale the horizontal acoustic pressure gradients, while ``cqw``
scales the implicit vertical pressure coefficients and forcing.  The sum
contains every active WRF ``moist`` mass species (qv/qc/qr and, for
mixed-phase schemes, qi/qs/qg), never their number-moment ``scalar`` fields.
RunConfig's
default-ON ``moist_cq`` switch exists only for falsification attribution;
it is not a science-tuning option and production configurations leave it
enabled.  A dry state has no qv allocation and stays on the exact legacy
kernel expressions without cq scratch or launch work.

External aerosol forcing follows the supplied table inventory. WRF
solve_em.F:2803-2839 captures aerosol spec/relax tendencies on RK stage 1
when aer_init_opt>0; :2904-2930 excludes those fields from flow-dependent
zero inflow. The input producer's selected aerosol fields are retained in
coupled snapshots and wrfbdy tables.  The analysed hydrometeor masses a
source publishes, and the number moments the cold start seeds from them,
are specified the same way where the table carries them (WRF
have_bcs_moist / have_bcs_scalar, solve_em.F:2265-2267, :2346, :2812-2815;
woof.boundary_fields).  Unsupplied number moments keep their
flow-dependent boundary treatment.

"""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from woof.config import RunConfig
from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.core import constants as c
from woof.core.advection import launch_flux_div_scalar, vertical_orders
from woof.core.grid import BaseState, VerticalCoord
from woof.grid_requirements import FIFTH_ORDER_STENCIL_AXIS
from woof.core.kernels import get_kernel
from woof.microphysics_schemes import scheme_for_state as _named_scheme_for_state
from woof.core.state import DTYPE, DomainState, init_at_rest
from woof.core.wdm6_constants import WDM6_NUMBER_SPECIES
from woof.core import tke_budget

_TPB = 128  # threads per block along i (i fastest)

#: Frozen Phase-2 moisture species (allocation/operation order in state).
SPECIES = ("qv", "qc", "qr")

#: Ice-category mass species shared by WSM6, Thompson, and Morrison.
ICE_MASS_SPECIES = ("qi", "qs", "qg")

#: Morrison-only prognostic number moments.  ``nc`` is deliberately absent:
#: WRF's
#: default INUM=1 path diagnoses a fixed 250 cm-3 every Morrison call rather
#: than carrying a droplet-number tracer (module_mp_morr...F:116-124,
#: 1493-1496).  Precipitating/ice moments are transported as mixing ratios.
MORRISON_NUMBER_SPECIES = ("nr", "ni", "ns", "ng")
MORRISON_SPECIES = ICE_MASS_SPECIES + MORRISON_NUMBER_SPECIES

# Thompson mp=8 transports rain and ice number only.  The ordering follows
# its WRF Registry scalar package (qni, qnr) after woof's historical
# rain-before-ice state allocation; all loops below discover actual fields so
# they never invent Morrison snow/graupel moments on a Thompson state.
THOMPSON_NUMBER_SPECIES = ("nr", "ni")
TRANSPORTED_NUMBER_SPECIES = MORRISON_NUMBER_SPECIES

# Thompson mp=28 (aerosol-aware) additionally transports the prognostic cloud
# droplet number and the two aerosol number tracers.  WRF registers qnwfa and
# qnifa in the same ``scalar`` package as qnc/qnr/qni
# (Registry.EM_COMMON:3036 binds mp_physics=28 to that package), so all five
# are dry-mass-coupled advected scalars and none of them enters ``calc_cq``.
#
# The ordering keeps ``nr``/``ni`` first so an mp=28 state's leading five
# transported fields are a prefix-compatible superset of mp=8's two, which is
# what lets the generic dycore/moist loops discover them without a scheme
# switch.
#
# NOT added to TRANSPORTED_NUMBER_SPECIES.  That tuple is filtered by
# PRESENCE, and mp_physics=10 already allocates ``state.nc``
# (woof/core/state.py, the Morrison ``number_and_radius`` tuple) while
# DELIBERATELY not transporting it -- WRF's default Morrison INUM=1 diagnoses
# a fixed 250 cm-3 every call (module_mp_morr_two_moment.F:116-124,
# 1493-1496).  A presence-based ``nc`` would start advecting Morrison's
# diagnostic droplet number through every generic dycore consumer and move a
# validated trajectory with nothing raising.
THOMPSON_AERO_NUMBER_SPECIES = ("nr", "ni", "nc", "nwfa", "nifa")

# Native NSSL option 18 carries hail mass plus cloud/rain/ice/snow/graupel/
# hail number, predicted CCN, and graupel/hail particle-volume scalars.  All
# are Registry-transported dry-mass mixing ratios; only the first four ice/
# hail entries contribute water mass and acoustic/buoyancy loading.
NSSL_MASS_SPECIES = ("qi", "qs", "qg", "qh")
NSSL_SCALAR_SPECIES = (
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh")
NSSL_SPECIES = NSSL_MASS_SPECIES + NSSL_SCALAR_SPECIES

# Single-moment hail: WSM7 (mp_physics=24, Registry.EM_COMMON:3034
# ``moist:qv,qc,qr,qi,qs,qg,qh``) carries WSM6's three ice masses and a
# hail mass, and no number moment at all.  Every entry is a ``moist``
# array member, so all four load cq and the buoyancy sum.
SINGLE_MOMENT_HAIL_SPECIES = ICE_MASS_SPECIES + ("qh",)

# Milbrandt-Yau option 9 carries hail mass beside graupel and a number
# moment for EVERY one of the six hydrometeors.  The tuples are defined in
# woof.core.milbrandt2_constants (with the WRF line references) so that
# the offline child's transport inventory can read them without this
# module's CuPy import; they are re-exported here under their names.
from woof.core.milbrandt2_constants import (  # noqa: E402,F401 -- re-exported
    MY2_MASS_SPECIES, MY2_NUMBER_SPECIES, MY2_SPECIES)

# P3 one-category (mp=50) is the first scheme in the tree with ``qi`` and NO
# ``qs``/``qg``, so it cannot reuse ICE_MASS_SPECIES: that tuple would name
# two fields an mp=50 DomainState deliberately does not allocate, and the
# generic dycore loops would raise on ``state.qs0`` at the first RK time-t
# copy.  WRF Registry.EM_COMMON:3038 registers the package as
#     moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib
# and solve_em advects the scalar array beside the moist one, so all four
# scalar entries are transported.  The order here is WRF's own scalar order
# (qni, qnr, qir, qib) after the single moist ice mass, which is also
# woof/core/state.py's mp=50 allocation order -- transport order is
# per-field independent, so the only thing it buys is that the two
# inventories read the same way.
#
# ``qir`` (rime MASS) and ``qib`` (rime VOLUME) must move with ``qi`` or
# they decouple from the ice they describe: qirim/qitot picks the lookup
# table's rime-fraction index and rho_rime = qirim/birim picks its
# rime-density index, so a stationary pair would mis-select every ice fall
# speed and collection rate within a few steps.  Neither is additional
# water: qir is a COMPONENT of qi, and qib is a volume.  They are Registry
# ``scalar`` entries, so they stay out of WRF_MOIST_ARRAY_SPECIES, out of
# ``calc_cq`` and out of the hydrostatic qtot sum, exactly like the number
# moments beside them.
P3_SPECIES = ("qi", "ni", "nr", "qir", "qib")

# Ferrier-Aligo (mp=5) registers moist:qv,qc,qr,qi and scalar:qt
# (Registry.EM_COMMON:3019).  qi is the total ice and is transported; qt is
# rebuilt from qc+qr+qi at every FER_HIRES entry (module_mp_fer_hires.F:388)
# so moving it would change nothing the scheme reads, and the three f_*
# fields are WRF state, never advected.  One ice mass and no qs/qg, so the
# arm sits before the generic filter for P3's reason.
FERRIER_SPECIES = ("qi",)

#: The scratch slot holding a zero plane for a moist-array mass species the
#: active scheme does not allocate.  See :func:`absent_mass_plane`.
ABSENT_MASS_SLOT = "moist_absent_mass"


def absent_mass_plane(state) -> cp.ndarray:
    """A freshly zeroed (nz, ny, nx) stand-in for an unallocated mass field.

    Two fused CUDA kernels sum the WRF ``moist`` array by an integer mode
    rather than by a field list: ``calc_cq`` (kernels/acoustic.cu:80-109,
    mass arms 1/3/6/7) and ``slow_buoyancy``'s ``q_total``
    (kernels/dycore.cu:212-228, modes 0/1/2/3).  Both jump straight from
    "warm rain" to "ice plus snow plus graupel", because until P3 every
    scheme with ``qi`` also had ``qs`` and ``qg``.

    P3 has ONE ice category.  Rather than reopen two FROZEN translation
    units -- tests/test_mp8_frozen.py pins both files' bytes AND their
    compile strings, which is what proves mp=8 has not moved -- the
    callers hand the absent qs/qg slots this shared zero plane and keep
    the six-mass mode.  The sum is exact, not merely close: both kernels
    accumulate with a literal ``add.rn.f32`` (``arn_add``/``rn_add``) over
    non-negative terms starting at +0.0f, so a zero addend is an identity
    on every bit pattern the accumulator can hold.

    Zeroed on every call, immediately before the launch that reads it, so
    the slot keeps its ``write_before_read`` lifetime and no arena
    neighbour can be observed through it.
    """
    # Spelled as a literal, not as ABSENT_MASS_SLOT: the completeness gate
    # (tests/test_preflight.py::test_every_scratch_call_site_is_classified)
    # resolves slot names statically, and a Name-passed slot escapes it.
    plane = state.scratch(state.p.shape, "moist_absent_mass")
    plane[...] = 0.0
    return plane


#: The species WRF v4.6.1 carries in its ``moist`` 4-D array, as opposed to
#: its ``scalar`` array.  Registry/Registry.EM_COMMON declares qv (:453),
#: qc (:455), qr (:457), qi (:459), qs (:465), qg (:467) and qh (:469) with
#: package ``moist``; every number/volume/CCN tracer woof transports --
#: qndrop (:521), qni (:523), qns (:531), qnr (:533), qng (:535), qnc (:541)
#: and the aerosol pair -- is declared with package ``scalar``.
#:
#: The distinction is essential for exactly one thing today and it is
#: worth naming rather than rediscovering: WRF's ``&dynamics`` mixing
#: switches are PER ARRAY.  ``moist_mix2_off``/``moist_mix6_off`` gate the
#: moist array (dyn_em/solve_em.F:2229-2230), ``scalar_mix2_off``/
#: ``scalar_mix6_off`` gate the scalar array (:2795-2796), and a gate that
#: took woof's whole transported set for "moisture" would silently turn off
#: a filter WRF leaves on.
WRF_MOIST_ARRAY_SPECIES = frozenset(
    SPECIES + ICE_MASS_SPECIES + ("qh",))


def _extra_water_species(state: DomainState) -> tuple[str, ...]:
    """Active transported species beyond qv/qc/qr.

    WSM6 owns the three ice mass fields.  Thompson owns those plus rain/ice
    number, while Morrison owns all four precipitating/ice number moments.
    Presence is checked per field so a Thompson ``nr`` cannot accidentally
    imply Morrison-only ``ns``/``ng`` storage.

    Aerosol-aware Thompson (mp=28) is selected on ``state.nwfa``, NOT on
    ``state.nc``.  ``nwfa`` is allocated by exactly one scheme, so it is an
    unambiguous discriminator; ``nc`` is not (mp=10 allocates it and does not
    transport it -- see :data:`THOMPSON_AERO_NUMBER_SPECIES`).  The mp=28 arm
    therefore sits BEFORE the generic presence filter and returns its own
    closed tuple rather than widening that filter.

    Milbrandt-Yau (mp=9) and NSSL (mp=18) BOTH allocate ``qh``, so hail mass
    stopped being a discriminator the moment a second hail scheme landed.
    ``nh`` is the mp=9 one: mp=9 spells its hail number ``nh`` and mp=18
    spells its ``qnh``, and no other scheme allocates either.  The mp=9 test
    therefore sits ahead of the ``qh`` test -- reversing them would hand a
    Milbrandt-Yau state NSSL's nine-scalar package and advect nine fields
    that do not exist.

    P3 (mp=50) is selected on ``state.qir`` for the same reason: rime mass
    is allocated by exactly one scheme.  Its arm must also sit before the
    generic filter, because that filter starts from
    :data:`ICE_MASS_SPECIES` and P3 is the first scheme with ``qi`` and no
    ``qs``/``qg`` -- see :data:`P3_SPECIES`.  It is independent of the mp=9
    arm beside it: no state allocates both ``nh`` and ``qir``.
    """
    if getattr(state,"fast_sbm_bins",None) is not None:
        from woof.core.fast_sbm_contract import EXTRA_SPECIES
        return EXTRA_SPECIES
    if getattr(state, "qi", None) is None:
        return ()
    if getattr(state, "qir", None) is not None:
        return P3_SPECIES
    if getattr(state, "f_rimef", None) is not None:
        # Ferrier-Aligo: f_rimef is allocated by exactly one scheme.
        return FERRIER_SPECIES
    if getattr(state, "nh", None) is not None:
        return MY2_SPECIES
    if getattr(state, "qh", None) is not None:
        # Hail mass with NO NSSL number set is the single-moment hail
        # family (mp=24); NSSL alone allocates ``qnh`` beside ``qh``.
        if getattr(state, "qnh", None) is None:
            return SINGLE_MOMENT_HAIL_SPECIES
        return NSSL_SPECIES
    # Named schemes (woof.microphysics_schemes) answer from their own
    # capability row, selected on their discriminator field; the
    # ``volg`` test sits ahead of the ``nwfa`` test when a scheme also carries
    # the two aerosol tracers.
    named = _named_scheme_for_state(state)
    if named is not None:
        return (named.ice_mass_species + named.moment_species
                + named.aerosol_species)
    if getattr(state, "nwfa", None) is not None:
        return ICE_MASS_SPECIES + THOMPSON_AERO_NUMBER_SPECIES
    # WDM6 alone allocates nn. Its complete Registry scalar package is
    # qnn/qnc/qnr (Registry.EM_COMMON:3031); the generic Morrison presence
    # filter below intentionally excludes diagnostic nc and cannot own it.
    if getattr(state, "nn", None) is not None:
        return ICE_MASS_SPECIES + WDM6_NUMBER_SPECIES
    numbers = tuple(name for name in TRANSPORTED_NUMBER_SPECIES
                    if getattr(state, name, None) is not None)
    return tuple(n for n in ICE_MASS_SPECIES
                 if getattr(state, n, None) is not None) + numbers


def extra_moist_species(state: DomainState) -> tuple[str, ...]:
    """Extra transported fields: the microphysics species, MYNN's optional
    scalar QKE, then every PBL-owned advected scalar the state carries
    (:data:`woof.core.physics_inventory.PBL_SCALAR_NAMES`, one shared
    inventory keyed by PBL selector)."""
    from woof.core.physics_inventory import PBL_SCALAR_NAMES
    carried = tuple(name for name in PBL_SCALAR_NAMES
                    if getattr(state, name, None) is not None)
    return _extra_water_species(state) + (
        ("qke_adv",) if getattr(state, "qke_adv", None) is not None else ()) + carried


def moist_species(state: DomainState) -> tuple[str, ...]:
    """Return the active transported scalar registry for ``state``.

    The conditional preserves the exact qv/qc/qr loop for mp=0/1 while
    exposing the exact prognostic mass/number moments carried by the active
    mixed-phase scheme.
    """
    return tuple(name for name in SPECIES if getattr(state, name, None) is not None) + extra_moist_species(state)


def _mut2d(mu, ny: int, nx: int) -> cp.ndarray:
    """Column mass as a contiguous (ny, nx) float32 device array (accepts a
    scalar or any broadcastable array, mirroring the npref mirrors)."""
    return cp.ascontiguousarray(
        cp.broadcast_to(cp.asarray(mu, dtype=DTYPE), (ny, nx)))


def launch_pd_fluxes(q, q0, ru, rv, rw, mut, coord, dx, dy, dt,
                     fxl, fxc, fyl, fyc, fzl, fzc,
                     msft=None, has_msf=None,
                     open_x=False, open_y=False, vorder=3) -> None:
    """Fill the six PD flux arrays for scalar ``q`` (see pd_advection.cu).

    ``q`` is the RK stage estimate (high-order fluxes), ``q0`` the time-t
    scalar (upwind fluxes), ``mut`` the post-acoustic stage column mass
    (WRF ``muts``; scalar or (ny, nx)).  ``coord`` is anything carrying
    ``rdnw``/``c1h``/``c2h`` (a ``VerticalCoord`` or a ``DomainState``).
    ``msft`` (optional, Task 3) is the mass-point map factor: the upwind
    fluxes/Courant numbers then use the physical face spacings (WRF
    ``advect_scalar_pd``).  ``open_x``/``open_y`` select WRF's
    specified/open boundary treatment along that axis (zero boundary-normal
    faces + degraded near-boundary stencils, no wrap); an open axis needs
    >= 7 cells so the degrade bands cannot overlap.  ``vorder`` is WRF's
    ``v_sca_adv_order`` (3 or 5): the high-order vertical flux the
    limiter corrects against takes that ladder. At order 5 a separate
    kernel replaces the upwind low-order flux with the fork's
    semi-Lagrangian upstream sum on faces above Courant 1; every other face
    keeps the upwind flux (the fork's downstream cell there is confined to
    the strict WRF verification build: it drained empty cells and the
    final clamp turned that into mass). Order 3 never loads that kernel.
    """
    nz, ny, nx = q.shape
    if vorder not in (3, 5):
        raise ValueError(
            f"vorder must be 3 or 5 (WRF vert_order ladders the PD kernel "
            f"carries), got {vorder!r}")
    if open_x and nx < FIFTH_ORDER_STENCIL_AXIS:
        raise ValueError(f"open_x PD advection needs nx >= 7, got {nx}")
    if open_y and ny < FIFTH_ORDER_STENCIL_AXIS:
        raise ValueError(f"open_y PD advection needs ny >= 7, got {ny}")
    if has_msf is None:
        has_msf = msft is not None
    if msft is None:
        msft = cp.ones((ny, nx), dtype=DTYPE)
    kern = get_kernel("pd_advection", "pd_fluxes")
    grid = ((nx + 1 + _TPB - 1) // _TPB, ny + 1, nz + 1)
    args = (q, q0, ru, rv, rw, _mut2d(mut, ny, nx),
          cp.asarray(coord.c1h, dtype=DTYPE),
          cp.asarray(coord.c2h, dtype=DTYPE),
          cp.asarray(coord.rdnw, dtype=DTYPE),
          cp.asarray(coord.fnm, dtype=DTYPE),
          cp.asarray(coord.fnp, dtype=DTYPE), msft,
          DTYPE(dx), DTYPE(dy), DTYPE(dt),
          fxl, fxc, fyl, fyc, fzl, fzc,
          np.int32(nz), np.int32(ny), np.int32(nx), np.int32(has_msf),
          np.int32(open_x), np.int32(open_y), np.int32(vorder))
    kern(grid, (_TPB, 1, 1), args)
    if vorder == 5:
        get_kernel("pd_vertical_sl", "pd_vertical_sl")(
            grid, (_TPB, 1, 1), args)


def launch_pd_renorm_apply(q0, mu_old, fxl, fxc, fyl, fyc, fzl, fzc,
                           tend, coord, dx, dy, dt,
                           msft=None, has_msf=None,
                           open_x=False, open_y=False) -> None:
    """ADD the renormalized PD flux divergence into ``tend``.

    ``mu_old`` is the time-t column mass (WRF ``mu_old``, coupling
    ``ph_low``); flux arrays as filled by :func:`launch_pd_fluxes`.
    ``msft`` (optional, Task 3): ph_low/flux_out weight the horizontal
    divergence by msft^2 (WRF msftx*msfty) and the vertical by msft, and
    the applied tendency weights the horizontal divergence by msft.
    ``open_x``/``open_y`` (must match :func:`launch_pd_fluxes`): the
    limiter skips the outermost cells and the applied horizontal
    divergences skip the boundary cells (WRF specified/open bounds,
    module_advect_em.F:7697-7715, 7817-7821, 7852-7856).
    """
    nz, ny, nx = q0.shape
    if has_msf is None:
        has_msf = msft is not None
    if msft is None:
        msft = cp.ones((ny, nx), dtype=DTYPE)
    kern = get_kernel("pd_advection", "pd_renorm_apply")
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    args = (q0, _mut2d(mu_old, ny, nx), fxl, fxc, fyl, fyc, fzl, fzc,
          cp.asarray(coord.c1h, dtype=DTYPE),
          cp.asarray(coord.c2h, dtype=DTYPE),
          cp.asarray(coord.rdnw, dtype=DTYPE), msft,
          DTYPE(1.0 / dx), DTYPE(1.0 / dy), DTYPE(dt),
          tend, np.int32(nz), np.int32(ny), np.int32(nx),
          np.int32(has_msf), np.int32(open_x), np.int32(open_y))
    from woof.wrf_exact import ADVECTION_ENABLED
    if ADVECTION_ENABLED:
        args += (coord.mub2d, coord.mup0)
    kern(grid, (_TPB, 1, 1), args)


def _pd_fold_sources(state: DomainState, cfg: RunConfig, name: str,
                     q0, chm0, dt_eff: float, physics_tendencies,
                     fixed_tendency=None, lbc_held=None):
    """WRF ``rk_update_scalar_pd`` (module_em.F:1803-1916): on the final
    RK step with PD advection, fold the accumulated physics tendency into
    the TIME-T scalar with the time-t mass (mu_old = mu_new = mu_1 in the
    solve_em.F:1849-1866 call) and zero it, BEFORE the flux limiter -- so
    ``advect_scalar_pd``'s ph_low budget (module_advect_em.F:7733-7737)
    sees the sources ('add in physics tendency first if positive definite
    advection is used', solve_em.F:1839-1841).

    The fold covers the FULL tile INCLUDING the specified zone: the
    ``_spc`` loop bounds are captured at module_em.F:1863-1868 BEFORE the
    specified narrowing at F:1870-1878 (dead code for this routine -- the
    narrowed bounds are never used afterwards), and the fold loop at
    F:1889-1893 runs over the ``_spc`` bounds.  The ring cell's folded
    value is live interior forcing: it is the upwind donor for the first
    interior face and enters the first interior cell's ph_low
    (pinned by tests/test_pd_advection.py::
    test_pd_fold_covers_specified_ring).

    ``fixed_tendency`` carries WRF's stage-1 forward tendency (horizontal
    mixing and/or sixth-order diffusion).  ``lbc_held`` carries qv's held
    lateral-boundary tendency (spec +
    relax, captured once per model step on RK stage 1 exactly like WRF's
    ``moist_tend`` from relax_bdy_scalar/spec_bdy_scalar at
    solve_em.F:2255-2292); WRF's fold covers the full moist_tend, so the
    held LBC part folds here alongside physics.

    Returns ``q0`` itself when there is nothing to fold, else a scratch
    copy with the source applied (the caller must use it for the upwind
    fluxes, ph_low, AND the coupled update, and must NOT re-add the
    physics tendency afterwards).
    """
    physics = (physics_tendencies.scalar_for(name)
               if physics_tendencies is not None else None)
    if physics is None and fixed_tendency is None and lbc_held is None:
        return q0
    nz, ny, nx = q0.shape
    q0_eff = state.scratch((nz, ny, nx), "moist_pd_q0")
    sources = tuple(source for source in (fixed_tendency, physics, lbc_held)
                    if source is not None)
    if _pd_fold_joinable(q0, chm0, sources):
        # One pass for ``q0_eff = q0; q0_eff += dt*S/chm0`` per source:
        # the same FP32 multiply, divide and add per element, in order.
        _pd_fold_kernel(len(sources))(q0, *sources, chm0,
                                      np.float32(dt_eff), q0_eff)
        return q0_eff
    q0_eff[...] = q0
    for source in sources:
        q0_eff += dt_eff * source / chm0
    return q0_eff


def _pd_fold_joinable(q0, chm0, sources) -> bool:
    """The joined fold's element order needs float32 device operands of q0's
    shape.

    The separate expression promotes by NumPy rules; with any other dtype, a
    broadcast operand, or host arrays (a CPU mirror that swaps this module's
    ``cp`` for NumPy, tests/test_pd_advection.py) it keeps the separate
    ufuncs.
    """
    arrays = (q0, chm0) + tuple(sources)
    return all(isinstance(array, _DEVICE_ARRAY) and array.dtype == np.float32
               and array.shape == q0.shape for array in arrays)


#: CuPy's array type as imported, not whatever ``cp`` names at call time.
_DEVICE_ARRAY = cp.ndarray


@lru_cache(maxsize=None)
def _pd_fold_kernel(count: int):
    """``q = q0``, then ``q = q + ((dt*S)/chm0)`` for each of ``count`` sources.

    CuPy compiles this exactly as it compiles its multiply, true_divide and
    add ufuncs (its own options, flush-to-zero appended); the ``_rn``
    intrinsics keep each product, quotient and sum its own rounded FP32
    operation, as the three separate ufunc launches did, so no multiply-add
    contraction forms across them.
    """
    names = [f"s{index}" for index in range(count)]
    body = "float q = q0;\n" + "".join(
        f"q = __fadd_rn(q, __fdiv_rn(__fmul_rn(dt, {name}), chm0));\n"
        for name in names) + "out = q;"
    return cp.ElementwiseKernel(
        "float32 q0, " + "".join(f"float32 {name}, " for name in names)
        + "float32 chm0, float32 dt", "float32 out", body,
        f"gpuwm_pd_fold_sources_{count}")


def _exclude_specified_ring_advection(tend, spec_zone: int) -> None:
    """Clear the specified ring of a PD advective tendency, in place.

    WRF ``rk_update_scalar`` on a specified or nested domain narrows the
    loop that applies ``advect_tend`` to the cells INSIDE the specified
    zone (module_em.F:1671-1678, used at :1695-1700 and :1746-1751); the
    ring itself moves by ``sc_tend`` alone, which on the final PD stage
    has already been folded into the time-t scalar
    (``_pd_fold_sources``, module_em.F:1889-1894).  ``advect_scalar_pd``
    still writes the ring's VERTICAL divergence (module_advect_em.F:
    7787-7791); WRF computes it and never applies it.

    The port applied it.  On the ring the vertical transport of a scalar
    has no horizontal divergence to balance it, so wherever the ring's
    eta mass flux converges into a layer the scalar there grows by a
    fixed fraction every step: a supplied aerosol number compounded in
    the model-top layer of a specified parent's boundary row by about
    5x per half hour, from 5.5e8 to 1.03e15 per kg in 3.9 h, where the
    full-state health gate stopped the forecast (nwfa at k = nz-1,
    j = ny-1).  Every scalar that the end-of-step finalizer or the
    flow-dependent boundary overwrites hid it; a scalar neither of them
    rewrites carried it.

    ``tend`` is the (nz, ny, nx) PD output.  The non-PD stages already
    match WRF: ``flux_div_scalar`` writes nothing on the ring and the held
    boundary tendency takes its place.
    """
    sz = int(spec_zone)
    if sz <= 0:
        return
    tend[:, :sz, :] = 0
    tend[:, -sz:, :] = 0
    tend[:, sz:-sz, :sz] = 0
    tend[:, sz:-sz, -sz:] = 0


#: CuPy's own ``maximum`` ufunc carries exactly this guard, so the clamp arm
#: below has to resolve ``NAN`` the same way it does or a NaN carrier would
#: come out with a different payload.
_NAN_PREAMBLE = """
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif
"""


@lru_cache(maxsize=None)
def _update_scalar_kernel(has_msf: bool, has_physics: bool, has_fixed: bool,
                          clamp: bool):
    """One fused coupled-scalar update kernel per branch combination.

    Every FP32 operation is spelled with an explicit round-to-nearest
    intrinsic AND the kernel is compiled with ``-fmad=false``, because the
    ufunc chain this replaces rounds to FP32 at every operator boundary while
    NVRTC would happily contract ``c1h*mu + c2h`` and ``chm0*q0 + dt*tend``
    into FMAs -- silently moving roughly 3% of the words.  CuPy appends
    ``-ftz=true`` after these options for this kernel and for the ufuncs
    alike, so the subnormal band matches as well; both halves are pinned by
    tests/test_moist.py::test_fused_scalar_update_is_bit_identical.

    Branch specialization rather than runtime flags keeps every launch free of
    dead loads.  A run visits at most a handful of the sixteen combinations
    and each pays one NVRTC compile at warmup.
    """
    params = ["T q0", "T tend", "raw T c1h", "raw T c2h",
              "raw T mu0", "raw T mu"]
    # 32-bit index arithmetic: a field big enough to overflow it would be
    # 8 GB of FP32 on its own, and CuPy's own indexer makes the same call.
    body = ["const int lev = static_cast<int>(i) / ncol;",
            "const int col = static_cast<int>(i) % ncol;",
            "T t = tend;"]
    if has_msf:
        params.append("raw T msft")
        body.append("t = __fmul_rn(t, msft[col]);")
    if has_physics:
        params.append("T physics")
        body.append("t = __fadd_rn(t, physics);")
    if has_fixed:
        params.append("T fixed")
        body.append("t = __fadd_rn(t, fixed);")
    params += ["T dt_eff", "int32 ncol"]
    body += [
        "const T chm0 = __fadd_rn(__fmul_rn(c1h[lev], mu0[col]), c2h[lev]);",
        "const T chm = __fadd_rn(__fmul_rn(c1h[lev], mu[col]), c2h[lev]);",
        "T v = __fmul_rn(chm0, q0);",
        "t = __fmul_rn(dt_eff, t);",
        "v = __fadd_rn(v, t);",
        "v = __fdiv_rn(v, chm);",
    ]
    if clamp:
        body.append("v = isnan(v) ? T(NAN) : T(max(v, T(0)));")
    body.append("q = v;")
    return cp.ElementwiseKernel(
        ", ".join(params), "T q", "\n".join(body),
        "gpuwm_update_scalar", preamble=_NAN_PREAMBLE,
        options=("-fmad=false",))


def _update_scalar_in_place(q, q0, tend, c1h, c2h, mu0, mu, dt_eff: float,
                            *, msft=None, physics=None, fixed=None,
                            clamp: bool) -> None:
    """Apply WRF's coupled scalar update in one pass over the field.

    Fuses what used to be up to eight full-size ufunc launches per species --
    ``tend *= msft``, ``tend += physics``, ``tend += fixed``, then
    ``multiply``/``multiply``/``add``/``divide``/``maximum`` -- plus the two
    (nz, ny, nx) column-mass temporaries ``chm0``/``chm`` the caller used to
    materialize per stage.  The kernel rebuilds both per element from the
    (nz,) coefficient rows and the FLAT (ny*nx,) column masses, deriving the
    level and the column ordinal from its linear index; the operation order
    and every FP32 rounding boundary are those of the ufunc chain, word for
    word.

    ``dt_eff*tend`` is deliberately not written back to ``tend``.  That store
    was dead: every reader of the shared ``moist_rq_t`` scratch -- each
    species iteration here and :func:`advance_tke_stage` -- re-zeros it first.
    """
    kern = _update_scalar_kernel(msft is not None, physics is not None,
                                 fixed is not None, clamp)
    args = [q0, tend, c1h, c2h, mu0, mu]
    for optional in (msft, physics, fixed):
        if optional is not None:
            args.append(optional)
    args += [DTYPE(dt_eff), np.int32(mu.size), q]
    kern(*args)


def _capture_advective_qv_forcing(state: DomainState, tend, mu0, *, cfg=None) -> None:
    """EXPORT the stage's pure advective qv rate as WRF ``RQVFTEN``.

    ``tend`` is the WRF-coupled advective tendency of qv carrying an extra
    ``1/msfty`` (``rk_update_scalar`` multiplies it back), so the
    uncoupled kg kg-1 s-1 rate is ``tend * msfty / (c1h*mu0 + c2h)`` with
    the TIME-T column mass -- the same reference mass
    :func:`woof.core.dycore.capture_advective_theta_forcing` divides by,
    which is what makes the exported pair one rate set rather than two.

    ``mu0`` is the (ny, nx) time-t column mass the caller already formed.
    A no-op on a state with no advective-forcing consumer.
    """
    if getattr(state, "rqvften", None) is None:
        return
    if WRF_EXACT and getattr(cfg, "cu_physics", 0) in (3, 6):
        # WRF set_tend saves each RK stage, including the final PD stage.
        # phy_prep_part2 divides the last coupled value by carried MUTS.
        state.rqvften[...] = (tend * state.msft[None]
                             if state.has_msf else tend)
        return
    rate = tend / (state.c1h[:, None, None] * mu0[None]
                   + state.c2h[:, None, None])
    if state.has_msf:
        rate *= state.msft[None]
    state.rqvften[...] = rate


def _ieva_scalar(state, tend, q_old, implicit, mu0, mu, dt_eff) -> None:
    """``advect_s_implicit`` on one scalar's advective tendency, in place:
    ``mut_old`` the time-t mass, ``mut = mut_new`` the post-acoustic one."""
    from woof.core import ieva
    ieva.solve_scalar(state, tend, q_old, implicit[1], mu0, mu, dt_eff,
                      variant=getattr(implicit, "variant", "wrf_471"))


def _wrf_scalar_source(state: DomainState, name: str, physics_tendencies,
                       fixed_tendencies, out):
    """Strict mode: WRF's ``moist_tend`` for one species before the boundary.

    ``first_rk_step_part2`` starts it at zero and adds the physics
    tendency (update_phy_ten), then the km_opt mixing; rk_scalar_tend adds
    the sixth order on rk_step 1.  woof holds the mixing and the sixth
    order together as the species' fixed tendency.
    """
    out[...] = 0
    physics = (physics_tendencies.scalar_for(name)
               if physics_tendencies is not None else None)
    fixed = (fixed_tendencies.get(name)
             if fixed_tendencies is not None else None)
    if physics is not None:
        out += physics
    if fixed is not None:
        out += fixed
    return out


def _advance_species_wrf(state: DomainState, cfg: RunConfig, name: str, q, q0,
                         tend, held, *, ru, rv, ww, implicit, pd, bufs,
                         mu0, mu, mu0_row, mu_row, chm0, dt_eff: float,
                         final: bool, physics_tendencies, fixed_tendencies,
                         recompute: bool, apply_scalar_lbc,
                         export_advective_forcing: bool,
                         boundary_x: bool, boundary_y: bool) -> None:
    """Strict mode: one species' RK-stage update in WRF's order.

    WRF holds ONE source tendency per species, ``moist_tend`` (physics,
    mixing, sixth order, then on rk_step 1 relax_bdy_scalar adding onto
    that sum and spec_bdy_scalar replacing the specified rows), and
    rk_update_scalar forms ``tendency = advect_tend*msfty + sc_tend``
    (advection only inside the specified ring) before the coupled update.
    On the final positive-definite stage rk_update_scalar_pd first folds
    it into the time-t scalar as ``((c1*muold+c2)*q + dt*sc)/(c1*muold+c2)``
    with the time-t mass on both sides, and the advection then runs on
    that scalar (module_em.F).  The default path adds the parts one by one
    in another grouping, which moved qv in 23719 of 80000 words at the
    end of the first RK stage (combo-sweep round 3, case p1).
    """
    nz, ny, nx = q0.shape
    boundary_forced = cfg.specified or cfg.nested
    if held is not None:
        source = held
    else:
        source = _wrf_scalar_source(state, name, physics_tendencies,
                                    fixed_tendencies,
                                    state.scratch((nz, ny, nx), "moist_sc_t"))
        if (cfg.nested or recompute) and apply_scalar_lbc is not None:
            apply_scalar_lbc(state, cfg, name, source, apply_relax=True,
                             source_field=q0)
    ww_explicit = ww if implicit is None else implicit[0]
    if pd:
        q0_eff = state.scratch((nz, ny, nx), "moist_pd_q0")
        q0_eff[...] = ((chm0 * q0) + (DTYPE(dt_eff) * source)) / chm0
        tend[...] = 0
        # advect_scalar_pd's mut is rk_scalar_tend's grid%muts, the
        # acoustic loop's MUT + MU'' (the ring its own carrier), not
        # mub + mu_2 after small_step_finish: 1674 interior words of the
        # final-stage qv advection differed by that rounding.
        from woof.core.dycore import WRF_MUTS_SLOT
        muts = state.existing_scratch(WRF_MUTS_SLOT)
        launch_pd_fluxes(q, q0_eff, ru, rv, ww_explicit,
                         mu if muts is None else muts, state,
                         cfg.dx, cfg.dy, dt_eff, *bufs,
                         msft=state.msft, has_msf=state.has_msf,
                         open_x=boundary_x, open_y=boundary_y,
                         vorder=vertical_orders(cfg)[0])
        launch_pd_renorm_apply(q0_eff, mu0, *bufs, tend=tend, coord=state,
                               dx=cfg.dx, dy=cfg.dy, dt=dt_eff,
                               msft=state.msft, has_msf=state.has_msf,
                               open_x=boundary_x, open_y=boundary_y)
        if implicit is not None:
            _ieva_scalar(state, tend, q0_eff, implicit, mu0, mu, dt_eff)
        if name == "qv" and WRF_EXACT and cfg.cu_physics in (3, 6):
            _capture_advective_qv_forcing(state, tend, mu0, cfg=cfg)
        if boundary_forced:
            _exclude_specified_ring_advection(tend, cfg.spec_zone)
        _update_scalar_in_place(
            q, q0_eff, tend, state.c1h, state.c2h, mu0_row, mu_row, dt_eff,
            msft=(state.msft.reshape(-1) if state.has_msf else None),
            clamp=False)
        return
    tend[...] = 0
    launch_flux_div_scalar(q, ru, rv, ww_explicit, tend, state,
                           cfg.dx, cfg.dy, open_x=boundary_x,
                           open_y=boundary_y, msf=state.msft,
                           has_msf=state.has_msf, spec=boundary_forced,
                           vorder=vertical_orders(cfg)[0])
    if implicit is not None:
        _ieva_scalar(state, tend, q0, implicit, mu0, mu, dt_eff)
    if name == "qv" and (export_advective_forcing or (WRF_EXACT and cfg.cu_physics in (3, 6))):
        _capture_advective_qv_forcing(state, tend, mu0, cfg=cfg)
    if boundary_forced:
        # flux_div_scalar fills the specified ring; rk_update_scalar's
        # advection loop starts inside it (i_start = ids+spec_zone).
        _exclude_specified_ring_advection(tend, cfg.spec_zone)
    if state.has_msf:
        tend *= state.msft[None]
    tend += source
    # No clamp: rk_update_scalar keeps WRF's float32 residuals, which can
    # be a few 1e-13 below zero after the final stage (one qv word at the
    # first step of the round-3 pair).  The default path clamps them.
    _update_scalar_in_place(q, q0, tend, state.c1h, state.c2h, mu0_row,
                            mu_row, dt_eff, clamp=False)


def advance_scalars_stage(state: DomainState, cfg: RunConfig,
                          ru, rv, ww, dt_eff: float, final: bool,
                          apply_relax: bool = True,
                          physics_tendencies=None,
                          fixed_tendencies=None,
                          export_advective_forcing: bool = False,
                          implicit=None) -> None:
    """Advance qv/qc/qr one RK stage from their time-t copies (``*0``).

    Called by ``dycore.step`` after each stage's acoustic loop with that
    stage's acoustic time-averaged mass fluxes ru_m/rv_m/ww_m (WRF
    ``sumflux``; see the module docstring) as ``ru``/``rv``/``ww``.
    Each scalar advances in coupled form (WRF
    ``rk_update_scalar``): ``C(mu_t)*q_t + dt_eff*tend``, uncoupled by the
    post-acoustic stage mass ``C(mu_new)``.  On the final stage with
    ``cfg.moist_adv_opt == 1`` the tendency is the PD-limited one and the
    update clamps FP32 rounding residuals at zero (module docstring).

    Boundary routing: stages 1-2 use the open-aware ``flux_div_scalar``
    path (Task 11 prerequisite) like theta.  Under SPECIFIED BCs the
    final PD stage runs with WRF's specified bounds in pd_advection.cu
    (advect_scalar_pd, module_advect_em.F:7697-7715) -- the production
    the reference case path.  Radiative-OPEN domains keep the unlimited final stage
    plus clamp: WRF's advect_scalar_pd DOES support open BCs (it carries
    its own non-cb open radiation blocks, module_advect_em.F:7266-7330),
    but woof's PD kernels do not implement those blocks -- an accepted,
    documented woof deviation.  On the PD final stage the
    accumulated stage-1 forward tendencies, physics tendencies, AND qv's
    held lateral spec+relax tendency are folded into the time-t scalar
    BEFORE the limiter (WRF
    rk_update_scalar_pd; ``_pd_fold_sources``) so the renormalization
    budget sees the sources.  The held qv LBC tendency is captured once
    per model step on the first stage from the time-t state (WRF
    moist_tend, solve_em.F:2255-2292) and enters every stage's update --
    relax rows additively, spec rows replacing the advective tendency;
    the spec rows are subsequently overwritten by
    ``apply_state_boundary_values`` exactly as in WRF.

    ``export_advective_forcing`` (the dycore passes ``istage == 0``) writes
    the qv half of the cumulus advective forcing pair, WRF's ``RQVFTEN``,
    from the untouched flux divergence -- see
    :func:`_capture_advective_qv_forcing`.  It is a SECOND hook site from
    theta's, not a duplicate: theta advects with the RK stage reference
    fluxes and qv with the acoustic time-averaged ones, so the two rates
    are captured where their own fluxes are.  The final PD stage cannot
    serve: its tendency is the advection of a source-folded scalar
    (``_pd_fold_sources``), which is no longer pure advection.

    ``implicit`` is ``rk_scalar_tend``'s IEVA split ``(wwE, wwI)`` of
    ``ww`` (``zadvect_implicit = 1``, last substep): every explicit
    operator, the PD limiter included, advects with ``wwE``, and each
    species' advective tendency is then replaced by its column solve
    against ``wwI`` from the time-t scalar the update starts from
    (``advect_s_implicit``, module_em.F:1346-1364), before the msf
    coupling, the lateral fold and the specified-ring exclusion.
    """
    ww_explicit = ww if implicit is None else implicit[0]
    nz, ny, nx = state.p.shape
    mu0 = state.mub2d + state.mup0                     # time-t column mass
    mu = state.mub2d + state.mup                       # post-acoustic mass
    # Flat column rows for the fused update, which addresses a column by the
    # ordinal it derives from its linear index.
    mu0_row = mu0.reshape(-1)
    mu_row = mu.reshape(-1)
    msft_row = state.msft.reshape(-1)
    tend = state.scratch((nz, ny, nx), "moist_rq_t")
    boundary_forced = cfg.specified or cfg.nested
    boundary_x = cfg.open_x or boundary_forced
    boundary_y = cfg.open_y or boundary_forced
    # WRF advect_scalar_pd fully supports specified domains (dedicated
    # limiter bounds, module_advect_em.F:7697-7715), and the ratified
    # the reference case namelist runs moist_adv_opt=1 -- so the PD final stage is
    # ENABLED under specified BCs (the pre-fix routing forced the
    # unlimited-plus-clamp path there, manufacturing scalar mass).
    # Radiative-open domains keep the non-PD routing.  WRF's
    # advect_scalar_pd DOES support open BCs -- it carries its own non-cb
    # open radiation blocks (module_advect_em.F:7266-7330) -- but woof's
    # PD kernels do not implement those blocks, so open domains run the
    # unlimited final stage plus clamp instead: an accepted, documented
    # woof deviation (open-BC benchmark pins).
    pd = (final and cfg.moist_adv_opt == 1
          and not (cfg.open_x or cfg.open_y))
    scalar_pd = (final and getattr(state, "qke_adv", None) is not None
                 and cfg.scalar_adv_opt == 1
                 and not (cfg.open_x or cfg.open_y))
    # The coupled update rebuilds both column masses per element, so the only
    # consumer left of a materialized ``chm0`` is the PD source fold; ``chm``
    # has none at all.
    chm0 = (state.c1h[:, None, None] * mu0[None] + state.c2h[:, None, None]
            if pd or scalar_pd else None)
    # WRF captures qv's lateral spec+relax tendency ONCE per model step,
    # on RK step 1 from the time-t scalar (relax_bdy_scalar +
    # spec_bdy_scalar into moist_tend, solve_em.F:2255-2292, gated
    # im == P_QV), and moist_tend persists into EVERY stage's
    # rk_update_scalar and the final rk_update_scalar_pd fold
    # (module_em.F:1889-1893).  ``apply_relax`` marks the capture stage
    # (dycore passes istage == 0, where state.qv still holds time t).
    # Final-review MAJOR: the relax part previously reached only the
    # discarded stage-1 provisional estimate.
    held_lbc = {}
    # The supplied hydrometeor masses and their seeded numbers
    # (woof.boundary_fields, WRF have_bcs_moist / have_bcs_scalar).  Their
    # spec+relax tendency is WRF's same rk_step-1 capture, recomputed on
    # every stage from the time-t copy at the time-t mass (the nested
    # path's form), so no full-domain held array per species is kept.
    recomputed_lbc = frozenset()
    apply_scalar_lbc = None
    if boundary_forced and state.lateral_boundaries is not None:
        from woof.ingest.lateral_bc import (
            apply_state_scalar_lateral_boundary as apply_scalar_lbc,
        )
        if cfg.specified:
            # Capture each supplied external scalar once at RK stage 1.
            # Scalar tables have the same mass coupling as qv. WRF retains
            # scalar_tend through all three stages (solve_em.F:2803-2868).
            from woof.boundary_fields import potential_external_scalar_fields
            from woof.ingest.lateral_bc import _active_device_interval
            supplied = _active_device_interval(state, cfg)[0].fields
            for name in potential_external_scalar_fields(cfg):
                if name not in supplied:
                    continue
                # Keep the finite slot inventory explicit for the allocation
                # census; each slot is priced and classified independently.
                if name == "qv":
                    held = state.scratch((nz, ny, nx), "lbc_qv_held")
                elif name == "nwfa":
                    held = state.scratch((nz, ny, nx), "lbc_nwfa_held")
                elif name == "nifa":
                    held = state.scratch((nz, ny, nx), "lbc_nifa_held")
                else:
                    raise KeyError(f"unregistered external scalar hold: {name}")
                held_lbc[name] = held
                if apply_relax:
                    if WRF_EXACT:
                        # WRF relaxes onto moist_tend, which already holds
                        # the species' physics and mixing, with grid%mut,
                        # the stage-start (time-t) mass: the capture runs
                        # after small_step_finish has moved state.mup.
                        _wrf_scalar_source(state, name, physics_tendencies,
                                           fixed_tendencies, held)
                        apply_scalar_lbc(state, cfg, name, held,
                                         apply_relax=True,
                                         source_field=getattr(state,
                                                              name + "0"))
                    else:
                        held[...] = 0
                        apply_scalar_lbc(state, cfg, name, held,
                                         apply_relax=True)
            from woof.boundary_fields import HELD_BOUNDARY_FIELDS
            from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
            recomputed_lbc = frozenset(
                name for name in supplied
                if name in COUPLED_SCALAR_STATE_FIELDS
                and name not in HELD_BOUNDARY_FIELDS)
    if pd or scalar_pd:
        bufs = (state.scratch((nz, ny, nx + 1), "pd_fxl"),
                state.scratch((nz, ny, nx + 1), "pd_fxc"),
                state.scratch((nz, ny + 1, nx), "pd_fyl"),
                state.scratch((nz, ny + 1, nx), "pd_fyc"),
                state.scratch((nz + 1, ny, nx), "pd_fzl"),
                state.scratch((nz + 1, ny, nx), "pd_fzc"))
    for name in (name for name in SPECIES if getattr(state, name, None) is not None):
        q = getattr(state, name)
        q0 = getattr(state, name + "0")
        if WRF_EXACT:
            _advance_species_wrf(
                state, cfg, name, q, q0, tend, held_lbc.get(name), ru=ru,
                rv=rv, ww=ww, implicit=implicit, pd=pd,
                bufs=(bufs if pd else None), mu0=mu0, mu=mu,
                mu0_row=mu0_row, mu_row=mu_row, chm0=chm0, dt_eff=dt_eff,
                final=final, physics_tendencies=physics_tendencies,
                fixed_tendencies=fixed_tendencies,
                recompute=name in recomputed_lbc,
                apply_scalar_lbc=apply_scalar_lbc,
                export_advective_forcing=export_advective_forcing,
                boundary_x=boundary_x, boundary_y=boundary_y)
            continue
        # The zeroing lives on each consuming branch rather than the loop
        # prologue: on the positive-definite path the buffer is re-zeroed
        # below before any kernel reads it, so a prologue memset was ten
        # dead full-field writes per step on non-nested lanes.  Only the
        # nested lateral fold reads ``tend`` before that second zeroing.
        recompute = name in recomputed_lbc
        if pd:
            nested_held = None
            if (cfg.nested or recompute) and apply_scalar_lbc is not None:
                tend[...] = 0
                apply_scalar_lbc(state, cfg, name, tend, apply_relax=True,
                                 source_field=q0)
                nested_held = tend
            q0_eff = _pd_fold_sources(
                state, cfg, name, q0, chm0, dt_eff, physics_tendencies,
                fixed_tendency=(fixed_tendencies.get(name)
                                if fixed_tendencies is not None else None),
                lbc_held=(nested_held if nested_held is not None
                          else held_lbc.get(name)))
            # WRF module_em.F:1889-1894 folds sc_tend into the time-t
            # scalar and then clears it before advect_scalar_pd accumulates
            # the advective tendency.  Nested forcing aliases ``tend`` here,
            # so retaining it would consume B once in q0_eff and again as
            # msft*B in the final update.
            tend[...] = 0
            launch_pd_fluxes(q, q0_eff, ru, rv, ww_explicit, mu, state,
                             cfg.dx, cfg.dy, dt_eff, *bufs,
                             msft=state.msft, has_msf=state.has_msf,
                             open_x=boundary_x, open_y=boundary_y,
                             vorder=vertical_orders(cfg)[0])
            launch_pd_renorm_apply(q0_eff, mu0, *bufs, tend=tend,
                                   coord=state,
                                   dx=cfg.dx, dy=cfg.dy, dt=dt_eff,
                                   msft=state.msft, has_msf=state.has_msf,
                                   open_x=boundary_x, open_y=boundary_y)
            if implicit is not None:
                _ieva_scalar(state, tend, q0_eff, implicit, mu0, mu, dt_eff)
            if boundary_forced:
                _exclude_specified_ring_advection(tend, cfg.spec_zone)
            # ``msft`` carries WRF rk_update_scalar's
            # tendency = advect_tend*msfty, now inside the fused update.
            _update_scalar_in_place(
                q, q0_eff, tend, state.c1h, state.c2h, mu0_row, mu_row,
                dt_eff, msft=(msft_row if state.has_msf else None),
                clamp=True)
        else:
            # flux_div_scalar accumulates into ``tend`` and deliberately
            # writes nothing at specified-boundary cells, so this branch
            # must start from a clean buffer.
            tend[...] = 0
            launch_flux_div_scalar(q, ru, rv, ww_explicit, tend, state,
                                   cfg.dx, cfg.dy,
                                   open_x=boundary_x, open_y=boundary_y,
                                   msf=state.msft, has_msf=state.has_msf,
                                   spec=boundary_forced,
                                   vorder=vertical_orders(cfg)[0])
            if implicit is not None:
                _ieva_scalar(state, tend, q0, implicit, mu0, mu, dt_eff)
            if export_advective_forcing and name == "qv":
                # WRF RQVFTEN.  The window is exactly here: ``tend`` holds
                # the acoustic time-averaged flux divergence of qv and
                # nothing else -- the msft multiply, the lateral spec/relax
                # fold and the physics/fixed sources inside the fused
                # update all come after, and the shared ``moist_rq_t``
                # scratch is overwritten by qc on the very next iteration.
                # Uncoupled by the same time-t column mass the theta
                # export uses (dycore.capture_advective_theta_forcing), so
                # the pair is one consistent rate set.
                _capture_advective_qv_forcing(state, tend, mu0)
            held = held_lbc.get(name)
            # A lateral tendency landing in ``tend`` below would be scaled a
            # second time if the msf coupling waited for the fused update, so
            # those rows keep the standalone multiply.
            lbc_after_msf = (held is not None
                             or ((cfg.nested or recompute)
                                 and apply_scalar_lbc is not None))
            if state.has_msf and lbc_after_msf:  # WRF rk_update_scalar:
                tend *= state.msft[None]         # tendency=advect_tend*msfty
            if (cfg.nested or recompute) and apply_scalar_lbc is not None:
                # Relax rows add; spec rows REPLACE the advective tendency
                # (the kernel's spec_bdy_scalar), as the held path does.
                apply_scalar_lbc(state, cfg, name, tend, apply_relax=True,
                                 source_field=q0)
            if held is not None:
                # The held spec+relax tendency enters EVERY stage (WRF
                # rk_update_scalar adds moist_tend on every rk step over
                # the _spc bounds).  Relax rows add; spec rows REPLACE
                # the advective tendency (spec_bdy_scalar semantics --
                # the capture wrote the pure boundary tendency there).
                sz = cfg.spec_zone
                tend += held
                tend[:, :sz, :] = held[:, :sz, :]
                tend[:, ny - sz:, :] = held[:, ny - sz:, :]
                tend[:, sz:ny - sz, :sz] = held[:, sz:ny - sz, :sz]
                tend[:, sz:ny - sz, nx - sz:] = (
                    held[:, sz:ny - sz, nx - sz:])
            _update_scalar_in_place(
                q, q0, tend, state.c1h, state.c2h, mu0_row, mu_row, dt_eff,
                msft=(msft_row if state.has_msf and not lbc_after_msf
                      else None),
                physics=(physics_tendencies.scalar_for(name)
                         if physics_tendencies is not None else None),
                fixed=(fixed_tendencies.get(name)
                       if fixed_tendencies is not None else None),
                clamp=final)

    if extra_moist_species(state):
        for name in extra_moist_species(state):
            # QKE follows scalar_adv_opt. Preserve every pre-existing
            # field's selector when the new carrier is absent.
            pd = (scalar_pd if name == "qke_adv" else
                  (final and cfg.moist_adv_opt == 1
                   and not (cfg.open_x or cfg.open_y)))
            q = getattr(state, name)
            q0 = getattr(state, name + "0")
            if WRF_EXACT:
                _advance_species_wrf(
                    state, cfg, name, q, q0, tend, held_lbc.get(name), ru=ru,
                    rv=rv, ww=ww, implicit=implicit, pd=pd,
                    bufs=(bufs if pd else None), mu0=mu0, mu=mu,
                    mu0_row=mu0_row, mu_row=mu_row, chm0=chm0, dt_eff=dt_eff,
                    final=final, physics_tendencies=physics_tendencies,
                    fixed_tendencies=fixed_tendencies,
                    recompute=name in recomputed_lbc,
                    apply_scalar_lbc=apply_scalar_lbc,
                    export_advective_forcing=export_advective_forcing,
                    boundary_x=boundary_x, boundary_y=boundary_y)
                continue
            # See the qv/qc/qr loop: zeroing moved onto the branches that
            # consume it, dropping the dead prologue memset on the pd path.
            recompute = name in recomputed_lbc
            if pd:
                nested_held = None
                if (cfg.nested or recompute) and apply_scalar_lbc is not None:
                    tend[...] = 0
                    apply_scalar_lbc(state, cfg, name, tend,
                                     apply_relax=True, source_field=q0)
                    nested_held = tend
                q0_eff = _pd_fold_sources(
                    state, cfg, name, q0, chm0, dt_eff,
                    physics_tendencies,
                    fixed_tendency=(fixed_tendencies.get(name)
                                    if fixed_tendencies is not None else None),
                    lbc_held=(nested_held if nested_held is not None
                              else held_lbc.get(name)))
                # See the qv/qc/qr loop above: WRF clears sc_tend after the
                # positive-definite source fold and before advection.
                tend[...] = 0
                launch_pd_fluxes(q, q0_eff, ru, rv, ww_explicit, mu, state,
                                 cfg.dx, cfg.dy, dt_eff, *bufs,
                                 msft=state.msft, has_msf=state.has_msf,
                                 open_x=boundary_x, open_y=boundary_y,
                                 vorder=vertical_orders(cfg)[0])
                launch_pd_renorm_apply(q0_eff, mu0, *bufs, tend=tend,
                                       coord=state,
                                       dx=cfg.dx, dy=cfg.dy, dt=dt_eff,
                                       msft=state.msft,
                                       has_msf=state.has_msf,
                                       open_x=boundary_x, open_y=boundary_y)
                if implicit is not None:
                    _ieva_scalar(state, tend, q0_eff, implicit, mu0, mu,
                                 dt_eff)
                if boundary_forced:
                    _exclude_specified_ring_advection(tend, cfg.spec_zone)
                _update_scalar_in_place(
                    q, q0_eff, tend, state.c1h, state.c2h, mu0_row, mu_row,
                    dt_eff,
                    msft=(msft_row if state.has_msf else None), clamp=True)
            else:
                # flux_div_scalar accumulates; start from a clean buffer.
                tend[...] = 0
                launch_flux_div_scalar(q, ru, rv, ww_explicit, tend, state,
                                       cfg.dx, cfg.dy,
                                       open_x=boundary_x, open_y=boundary_y,
                                       msf=state.msft,
                                       has_msf=state.has_msf,
                                       spec=boundary_forced,
                                       vorder=vertical_orders(cfg)[0])
                if implicit is not None:
                    _ieva_scalar(state, tend, q0, implicit, mu0, mu, dt_eff)
                held = held_lbc.get(name)
                lbc_after_msf = (
                    held is not None
                    or ((cfg.nested or recompute)
                        and apply_scalar_lbc is not None))
                if state.has_msf and lbc_after_msf:
                    tend *= state.msft[None]
                if (cfg.nested or recompute) and apply_scalar_lbc is not None:
                    apply_scalar_lbc(state, cfg, name, tend,
                                     apply_relax=True, source_field=q0)
                if held is not None:
                    sz = cfg.spec_zone
                    tend += held
                    tend[:, :sz, :] = held[:, :sz, :]
                    tend[:, ny - sz:, :] = held[:, ny - sz:, :]
                    tend[:, sz:ny - sz, :sz] = held[:, sz:ny - sz, :sz]
                    tend[:, sz:ny - sz, nx - sz:] = (
                        held[:, sz:ny - sz, nx - sz:])
                _update_scalar_in_place(
                    q, q0, tend, state.c1h, state.c2h, mu0_row, mu_row,
                    dt_eff,
                    msft=(msft_row if state.has_msf and not lbc_after_msf
                          else None),
                    physics=(physics_tendencies.scalar_for(name)
                             if physics_tendencies is not None else None),
                    fixed=(fixed_tendencies.get(name)
                           if fixed_tendencies is not None else None),
                    clamp=final)
    if cfg.specified:
        _apply_specified_scalar_flow_boundaries(
            state, cfg, ru, rv, {*held_lbc, *recomputed_lbc})


def _apply_specified_scalar_flow_boundaries(state, cfg, ru, rv, held_lbc):
    """WRF solve_em.F:2893-2930: QNN precedes the ordinary scalar branch.

    ``held_lbc`` names every scalar the boundary table specifies; those
    take no flow-dependent boundary (solve_em.F:2346 under
    have_bcs_moist, :2912-2917 under have_bcs_scalar).
    """
    from woof.core.microphysics_transition import NSSL2_BACKGROUND_CCN_PER_KG
    from woof.ingest.lateral_bc import apply_flow_dependent_boundaries

    species = moist_species(state)
    qnn_name = "nn" if cfg.mp_physics == 16 else "qnn"
    if qnn_name in species:
        # start_em.F:1750-1760 leaves WDM6's namelist ccn_conc intact but
        # replaces NSSL's grid value with nssl_cccn/1.225. NSSL's admitted
        # parameter identity pins that result to this FP32 concentration.
        inflow = (cfg.wdm6_ccn_conc if cfg.mp_physics == 16
                  else NSSL2_BACKGROUND_CCN_PER_KG)
        apply_flow_dependent_boundaries(
            (getattr(state, qnn_name),), ru, rv, cfg.spec_zone,
            inflow_value=inflow)
    fields = tuple(getattr(state, name) for name in species
                   if name not in ("qv", qnn_name) and name not in held_lbc)
    # The CUDA ABI carries nine independent fields per launch. NSSL's
    # full scalar package is larger; retain adjacent Registry-order batches.
    for start in range(0, len(fields), 9):
        apply_flow_dependent_boundaries(
            fields[start:start + 9], ru, rv, cfg.spec_zone)


def init_moist_balanced(cfg: RunConfig, coord: VerticalCoord,
                        base: BaseState, qv_func,
                        thp_func=None) -> DomainState:
    """Discretely balanced moist at-rest state (+ optional theta bubble).

    Transcribed from WRF v4.6.1 ``module_initialize_ideal.F`` (the
    quarter_ss full-state construction, lines 1026-1063): with the DRY base
    state and mu' = 0 (vapor rides as a scalar on the unchanged dry-mass
    column), integrate the perturbation pressure DOWN the column by
    inverting the discrete moist w-equation buoyancy (``pg_buoy_w`` == 0
    exactly, row by row), invert the moist EOS ``alt = (Rd/p0) * theta *
    (1 + Rv/Rd*qv) * (p/p0)^(-cv/cp)`` for the dry specific volume, then
    integrate the perturbation geopotential UP with the discrete
    hydrostatic recurrence.  The theta perturbation does not enter the
    pressure construction (WRF adds its bubble without recomputing p), so
    one builder serves the at-rest and bubble cases; ``qv_func == 0``
    reduces to the dry rebalance exactly.

    ``qv_func(z) -> (nz,) or (nz, ny, nx)`` maps the base-state half-level
    heights to vapor mixing ratio; ``thp_func(x, z)`` is the
    ``init_theta_perturbation`` contract.  Flat base states only (the
    terrain moist init is a later-task concern); float64 throughout;
    qc = qr = 0.
    """
    from woof.core.diagnostics import update_diagnostics

    if not cfg.moist:
        raise ValueError("init_moist_balanced requires cfg.moist=True "
                         "(the state carries no qv arrays otherwise)")
    if base.terrain_z is not None:
        raise NotImplementedError(
            "init_moist_balanced supports flat base states only")
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    s = init_at_rest(cfg, coord, base)

    z = s.height_half()                                # (nz,)
    x = (np.arange(nx) + 0.5) * cfg.dx - 0.5 * nx * cfg.dx
    qv = np.asarray(qv_func(z), dtype=np.float64)
    qv3 = np.broadcast_to(qv[:, None, None] if qv.ndim == 1 else qv,
                          (nz, ny, nx))
    thp = (np.zeros((nz, ny, nx))
           if thp_func is None else np.asarray(thp_func(x, z), np.float64))
    th = np.asarray(base.thb, dtype=np.float64)[:, None, None] + thp

    mub = float(base.mub)
    mup = 0.0                                          # dry mass unchanged
    c1f, c2f = coord.c1f, coord.c2f
    c1h, c2h = coord.c1h, coord.c2h

    # Perturbation pressure down from the top: each row inverts the
    # discrete pg_buoy_w balance at the w level above (WRF qvf1/qvf2).
    p = np.zeros((nz, ny, nx))
    qvf1 = qv3[nz - 1]
    qvf2 = 1.0 / (1.0 + qvf1)
    qvf1 = qvf1 * qvf2
    p[nz - 1] = (-0.5 * (c1f[nz] * mup + qvf1 * (c1f[nz] * mub + c2f[nz]))
                 / coord.rdnw[nz - 1] / qvf2)
    for kk in range(nz - 2, -1, -1):
        kw = kk + 1                                    # w level above cell kk
        qvf1 = 0.5 * (qv3[kk] + qv3[kk + 1])
        qvf2 = 1.0 / (1.0 + qvf1)
        qvf1 = qvf1 * qvf2
        p[kk] = p[kk + 1] - (c1f[kw] * mup
                             + qvf1 * (c1f[kw] * mub + c2f[kw])) \
            / qvf2 / coord.rdn[kw]

    # Moist EOS inverted for the dry specific volume; then the discrete
    # hydrostatic recurrence for the perturbation geopotential.
    qvf = 1.0 + c.RVOVRD * qv3
    pb3 = np.asarray(base.pb, dtype=np.float64)[:, None, None]
    alt = (c.RD / c.P0) * th * qvf * ((p + pb3) / c.P0) ** (-c.CV / c.CP)
    al = alt - np.asarray(base.alb, dtype=np.float64)[:, None, None]
    php = np.zeros((nz + 1, ny, nx))
    for k in range(nz):
        php[k + 1] = php[k] - coord.dnw[k] * (
            ((c1h[k] * mub + c2h[k]) + c1h[k] * mup) * al[k]
            + (c1h[k] * mup) * base.alb[k])

    s.thp[...] = cp.asarray(thp, dtype=DTYPE)
    s.php[...] = cp.asarray(php, dtype=DTYPE)
    s.qv[...] = cp.asarray(qv3, dtype=DTYPE)
    # Idealized cases always use hypsometric_opt=1, regardless of namelist
    # setting (WRF share/input_wrf.F:1038); the php recurrence above is the
    # opt-1 discrete hydrostatic inversion, so the default key matches.
    update_diagnostics(s)
    return s


def advance_tke_stage(state: DomainState, cfg: RunConfig,
                      ru, rv, ww, dt_eff: float, final: bool,
                      fixed_tendency=None, implicit=None) -> None:
    """Advance the km_opt=2 prognostic TKE one RK stage from ``tke0``.

    WRF advects tke through ``rk_scalar_tend``/``rk_update_scalar[_pd]``
    exactly like a moist scalar (solve_em.F:2362-2399, PD update at
    :2100-2119 under tke_adv_opt=1), with the once-per-step forward
    source (tke_rhs + self-diffusion, held in ``smag_rtke``) folded on
    every stage and into the PD budget on the final stage.

    Two WRF steps follow every ``rk_update_scalar`` pass, in this order
    (solve_em.F:2432-2451):

    * ``bound_tke`` (module_em.F:2490-2520) clamps the carrier into
      ``[0, tke_upper_bound]`` over the whole mass grid -- on EVERY domain
      including doubly periodic ones, and on EVERY RK stage, not just the
      final one.  It replaces the plain zero clamp the moist rows use.
    * ``flow_dep_bdy`` (module_bc.F:2335-2456) fills the specified zone of
      a ``specified``/``nested`` domain from the flow direction: zero on
      inflow, zero-gradient from the first interior row/column on outflow.
      That is the WHOLE lateral-boundary treatment WRF gives TKE -- the
      Registry row (EM_COMMON:312) has no ``b`` flag, so no wrfbdy stream
      carries TKE and there is no spec/relax forcing to apply, and no
      ``i``/``f`` flag, so nests neither inherit nor feed it back.
      Radiative-open domains get no extra arm in WRF either: the open
      radiation lives in the advection operator and in ``set_physical_bc3d``,
      both already routed through ``launch_flux_div_scalar``'s ``open_x``/
      ``open_y`` and the diffusion tendencies' open-strip zeroing.

    ``implicit`` is the substep's IEVA split, as in
    :func:`advance_scalars_stage` (WRF's tke call of ``rk_scalar_tend``).
    """
    if implicit is not None:
        ww = implicit[0]
    nz, ny, nx = state.p.shape
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    mu0 = state.mub2d + state.mup0
    mu = state.mub2d + state.mup
    chm0 = c1h * mu0[None] + c2h
    chm = c1h * mu[None] + c2h
    tend = state.scratch((nz, ny, nx), "moist_rq_t")
    boundary_x = cfg.open_x or cfg.specified or cfg.nested
    boundary_y = cfg.open_y or cfg.specified or cfg.nested
    q = state.tke
    q0 = state.tke0
    pd = final and not (cfg.open_x or cfg.open_y)
    tend[...] = 0
    if pd:
        bufs = (state.scratch((nz, ny, nx + 1), "pd_fxl"),
                state.scratch((nz, ny, nx + 1), "pd_fxc"),
                state.scratch((nz, ny + 1, nx), "pd_fyl"),
                state.scratch((nz, ny + 1, nx), "pd_fyc"),
                state.scratch((nz + 1, ny, nx), "pd_fzl"),
                state.scratch((nz + 1, ny, nx), "pd_fzc"))
        q0_eff = _pd_fold_sources(
            state, cfg, "tke", q0, chm0, dt_eff, None,
            fixed_tendency=fixed_tendency)
        launch_pd_fluxes(q, q0_eff, ru, rv, ww, mu, state,
                         cfg.dx, cfg.dy, dt_eff, *bufs,
                         msft=state.msft, has_msf=state.has_msf,
                         open_x=boundary_x, open_y=boundary_y,
                         vorder=vertical_orders(cfg)[0])
        launch_pd_renorm_apply(q0_eff, mu0, *bufs, tend=tend,
                               coord=state,
                               dx=cfg.dx, dy=cfg.dy, dt=dt_eff,
                               msft=state.msft, has_msf=state.has_msf,
                               open_x=boundary_x, open_y=boundary_y)
        if implicit is not None:
            _ieva_scalar(state, tend, q0_eff, implicit, mu0, mu, dt_eff)
        if state.has_msf:
            tend *= state.msft[None]
        # WRF's PD fold moved the sources into q0_eff, so ``tend`` here is
        # the pure advective flux divergence -- the budget's transport term.
        _tke_budget_transport(state, cfg, tend, final=final)
        q[...] = (chm0 * q0_eff + dt_eff * tend) / chm
    else:
        launch_flux_div_scalar(q, ru, rv, ww, tend, state,
                               cfg.dx, cfg.dy,
                               open_x=boundary_x, open_y=boundary_y,
                               msf=state.msft, has_msf=state.has_msf,
                               spec=(cfg.specified or cfg.nested),
                               vorder=vertical_orders(cfg)[0])
        if implicit is not None:
            _ieva_scalar(state, tend, q0, implicit, mu0, mu, dt_eff)
        if state.has_msf:
            tend *= state.msft[None]
        _tke_budget_transport(state, cfg, tend, final=final)
        if fixed_tendency is not None:
            tend += fixed_tendency
        q[...] = (chm0 * q0 + dt_eff * tend) / chm

    # WRF bound_tke (module_em.F:2490-2520), called after EVERY
    # rk_update_scalar pass on EVERY domain (solve_em.F:2434-2440) over the
    # whole mass grid -- kts..kte-1 with kte = kpe = kde, i.e. every mass
    # level.  It subsumes the zero clamp the moist rows apply on the final
    # stage and adds the ceiling those rows have no analogue for.
    if final:
        raw = tke_budget.raw_carrier(state, cfg)
        if raw is not None:
            raw[...] = q
    # Resolved from module globals at call time, so a mutation control can
    # replace exactly this step (tests/test_tke_budget.py).
    bound_tke(q, cfg.tke_upper_bound)

    # WRF flow_dep_bdy (module_bc.F:2335-2456): the entire lateral-boundary
    # treatment TKE gets, applied every stage after the bound.  Nothing
    # forces it from a parent or a wrfbdy file -- the Registry row
    # (EM_COMMON:312) carries neither a ``b`` nor an ``i`` flag.
    if cfg.specified or cfg.nested:
        from woof.ingest import lateral_bc
        lateral_bc.apply_flow_dependent_boundaries(
            (q,), ru, rv, cfg.spec_zone)


def bound_tke(tke, upper_bound: float) -> None:
    """WRF ``bound_tke`` (dyn_em/module_em.F:2490-2520), in place.

    ``tke = min(tke_upper_bound, max(tke, 0))`` over the whole mass grid.
    Its own named function so a mutation control can remove exactly this
    step and nothing else.
    """
    cp.clip(tke, 0.0, DTYPE(upper_bound), out=tke)


def _tke_budget_transport(state, cfg, tend, *, final: bool) -> None:
    """File the final stage's advective tendency as the transport term.

    Only the final RK stage advances the carrier from ``tke0`` to the new
    step value, so it is the only stage whose tendency belongs in a
    per-step budget; stages 1-2 produce discarded provisional estimates.
    """
    if not final:
        return
    slot = tke_budget.term(state, cfg, "transport")
    if slot is not None:
        slot[...] = tend
