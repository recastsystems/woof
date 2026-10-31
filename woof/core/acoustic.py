"""Launchers for the split-explicit acoustic substeps (ARW sec. 3.1.2).

``acoustic_substep`` drives one complete substep: the explicit forward
steps of the perturbation horizontal momenta (``advance_uv``) and of
mu''/Omega''/(mu*theta)'' (``advance_mu_th``), the vertically implicit
Crank-Nicolson w''-phi'' solve (``calc_coefs`` + ``advance_w_phi``, which
also applies the terrain kinematic lower BC and the damp_opt=3 implicit
w damper), with the linearized-EOS p''/alpha'' diagnosis fused into the
column solve and specified-frame handling.
All kernels take the general hybrid/terrain arguments (2-D ``mub2d``,
c1/c2 coefficient arrays, 1-D or 3-D base profiles via ``base3d``).
"""

from __future__ import annotations

import numpy as np

from woof.config import RunConfig
from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.core.kernels import (get_kernel, get_kernel_int_defines,
                                module_source, module_source_int_defines)
from woof.core.state import DTYPE, DomainState
from woof.microphysics_schemes import mass_loading_species as _mass_loading_species

_THREADS = 256
# A column solve does not share work between threads.  Smaller fast mapped
# blocks expose enough column blocks to fill the device on nested domains.
_W_PHI_MSF_THREADS = 64


def prepare_upper_wind_limiter(state, cfg, dtau):
    """Bind the fork's advance_w limiter on the saved physical stage winds.

    state.u/v are u_save/v_save during the acoustic loop, consumed by
    small_step_finish. state.php is ph_save. The fork's column recurrence
    runs after advance_mu_t and before small_step_finish on each substep.
    """
    if (getattr(cfg, "upper_wind_limiter_form", "wrf_461") != "noaa_wrf39"
            or cfg.damp_opt != 3):
        return None
    if cfg.nz < 3:
        raise ValueError("the upper-wind limiter reads three top mass levels; nz must be at least 3")
    kernel = get_kernel("upper_wind_limiter", "upper_wind_limiter")
    args = (state.u, state.v, state.php, state.phb, DTYPE(dtau),
            DTYPE(cfg.zdamp), _base3d(state),
            np.int32(_mass_w_boundary_zone(cfg)), np.int32(cfg.nz),
            np.int32(cfg.ny), np.int32(cfg.nx))
    grid = ((cfg.ny * cfg.nx + _THREADS - 1) // _THREADS,)
    return lambda: kernel(grid, (_THREADS,), args)


#: Strict mode: WRF's grid%muts through one acoustic loop (see
#: advance_exact_frame_mu_t).  Rebuilt at every stage start.
WRF_ACOUSTIC_MUTS_SLOT = "acoustic_wrf_muts"


def wrf_acoustic_muts(state: DomainState):
    return state.scratch(state.mup.shape[-2:], WRF_ACOUSTIC_MUTS_SLOT)


def init_wrf_acoustic_muts(state: DomainState, reference=None):
    """Seed the stage carrier before the first standalone substep."""
    import cupy as cp
    cp.add(state.mub2d, state.mup if reference is None else reference,
           out=wrf_acoustic_muts(state))
    state._wrf_acoustic_muts_initialized = True


def _uv_emdiv_args(state: DomainState, cfg: RunConfig, mudf) -> tuple:
    """Strict mode: advance_uv's in-statement external-mode damping.

    WRF adds ``c1h*mudf_xy`` inside the u/v statement, after the pressure
    gradient, with ``mudf_xy = -emdiv*dx*(MUDF(i)-MUDF(i-1))/msfuy`` in
    REAL; the strict kernel takes MUDF and the REAL scale here, and the
    separate emdiv launch then only saves mu'' (dycore).  Empty outside
    strict mode, where the kernel has no such parameters.
    """
    if not WRF_EXACT:
        return ()
    use = mudf is not None and cfg.emdiv > 0.0
    return (mudf if use else state.mup, state.msfu, state.msfv_inv,
            DTYPE(DTYPE(-cfg.emdiv) * DTYPE(cfg.dx)),
            DTYPE(DTYPE(-cfg.emdiv) * DTYPE(cfg.dy)),
            np.int32(use), np.int32(state.has_msf))


def damp_magnitude(cfg: RunConfig, dtau: float) -> np.float32:
    """The damp_opt=3 coefficient ``dampmag`` of WRF ``advance_w``.

    WRF 4.6.1 module_small_step_em.F:1284 forms ``dampmag =
    dts*config_flags%dampcoef`` as ONE float32 product of two float32
    operands, in every build.  Forming ``dtau*dampcoef`` in Python doubles
    and rounding the result once gives a different word whenever the double
    product is not already a float32 (dtau 4.5 s with dampcoef 0.2: WRF
    0.900000036, the double route 0.899999976), and the whole damping layer
    then differs from WRF by up to 2,401 ULP
    (tools/upper_damping_wrf461_oracle).  2.8.7 took the double route in
    the default build and the float32 product only under the strict build;
    both builds now take WRF's.  0 when the implicit damper is off.
    """
    if cfg.damp_opt != 3:
        return DTYPE(0.0)
    return DTYPE(DTYPE(dtau) * DTYPE(cfg.dampcoef))


def _base3d(state: DomainState) -> np.int32:
    """1 when the base profiles are per-column 3-D fields (terrain)."""
    return np.int32(state.thb.ndim == 3)


def _boundary_forced(cfg: RunConfig) -> bool:
    """True for specified or parent-forced nested lateral frames."""
    return bool(getattr(cfg, "specified", False)
                or getattr(cfg, "nested", False))


def _boundary_x(cfg: RunConfig) -> bool:
    return cfg.open_x or _boundary_forced(cfg)


def _boundary_y(cfg: RunConfig) -> bool:
    return cfg.open_y or _boundary_forced(cfg)


def _spec_zone(cfg: RunConfig) -> int:
    return int(cfg.spec_zone) if _boundary_forced(cfg) else 0


def _frame_takes_table_w(cfg: RunConfig) -> bool:
    """Whether the specified/nested frame integrates ``w`` from its table.

    A nest always does (solve_em.F:1602-1611, ``spec_bdyupdate(w_2,
    rw_tend)``); a specified domain does when it relaxes w
    (``RunConfig.relax_w``).  Every other specified domain takes WRF's
    root-domain zero-gradient copy.  The phi update and the pressure
    diagnosis on the frame are the same either way, which is why the two
    frame kernels differ in their ``w`` line alone.
    """
    from woof.ingest.lateral_bc import specified_relaxes_w

    return bool(getattr(cfg, "nested", False)) or specified_relaxes_w(cfg)


def _mass_w_boundary_zone(cfg: RunConfig) -> int:
    """WRF advance_mu_t/advance_w omit only the physical outer row.

    ``advance_uv`` deliberately narrows by the full configured specified
    zone, but WRF's mass/theta and implicit w/phi loops use ids+1/ide-2 and
    jds+1/jde-2 regardless of ``spec_zone``.  The wider frame is overwritten
    by ``spec_bdyupdate`` only after those provisional dynamics (and MUDF)
    have been formed.
    """
    return 1 if _boundary_forced(cfg) else 0


def prepare_moist_cq(state: DomainState, cfg: RunConfig) -> tuple:
    """Build WRF ``cqu/cqv/cqw`` once from the RK-stage moisture state.

    WRF ``calc_cq`` sums the active ``moist`` Registry package selected by
    ``mp_physics``: qv only for 0, qv/qc/qr for Kessler, and additionally
    qi/qs/qg for Morrison.  Number moments live in WRF's separate ``scalar``
    registry and are not mass loading.  The returned face arrays are
    scratch-arena views.  A dry state or the falsification-only
    ``moist_cq=False`` path returns harmless existing pointers plus a false
    flag, allocating and launching nothing.
    """
    use_cq = bool(getattr(cfg, "moist_cq", True)
                  and getattr(state, "qv", None) is not None)
    if not use_cq:
        return state.p, state.p, state.p, False

    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    cqu = state.scratch((nz, ny, nx + 1), "acoustic_cqu")
    cqv = state.scratch((nz, ny + 1, nx), "acoustic_cqv")
    cqw = state.scratch((nz + 1, ny, nx), "acoustic_cqw")
    if cfg.mp_physics == 0:
        qi = qs = qg = state.qv              # placeholders are never read
        n_mass = 1
    elif cfg.mp_physics in (1, 3):
        # Kessler (Registry.EM_COMMON:3015) and WSM3 (:3017) both declare
        # moist:qv,qc,qr; WSM3's cloud ice and snow live inside qc and qr.
        qi = qs = qg = state.qv              # ice placeholders are never read
        n_mass = 3
    elif getattr(state, "qi", None) is not None \
            and getattr(state, "qg", None) is None:
        # P3 one-category (Registry.EM_COMMON:3038): moist:qv,qc,qr,qi --
        # ONE ice mass, no snow, no graupel.  calc_cq's mass arms are
        # 1/3/6/7 and 4 is not one of them, so the absent qs/qg take the
        # shared zero plane at n_mass=6; see moist.absent_mass_plane for
        # why that is exact and why the kernel is not reopened.
        #
        # Selected on PRESENCE, matching the identical guard in
        # dycore._launch_slow_buoyancy and the predicate
        # preflight.scratch_slot_registry prices the plane under.  mp=50 is
        # the only scheme here today; a cfg-keyed arm would send the next
        # one-ice-category port back to the AttributeError this replaced.
        #
        # qir/qib stay out for the same reason the number moments do --
        # Registry ``scalar``, not ``moist`` -- and qir would be double
        # counting besides: it is a COMPONENT of qi, not extra mass.
        from woof.core.moist import absent_mass_plane
        qi = state.qi
        zero = absent_mass_plane(state)
        qs = state.qs if getattr(state, "qs", None) is not None else zero
        qg = zero
        n_mass = 6
    elif cfg.mp_physics in (2, 6, 8, 9, 10, 16, 18, 24, 28, 30, 106):
        # mp=28 is numerically IDENTICAL to mp=8 here.  WRF's calc_cq sums
        # the Registry ``moist`` package only, and Registry.EM_COMMON:3036
        # gives aerosol-aware Thompson
        #     moist:qv,qc,qr,qi,qs,qg
        #     scalar:qni,qnr,qnc,qnwfa,qnifa,qnbca
        # -- six masses, exactly mp=8's, with every number moment in the
        # separate ``scalar`` package.  They must therefore stay out of cq
        # AND out of the w-equation buoyancy loading; a droplet number of
        # order 1e8 entering q_tot would be a catastrophic, not a subtle,
        # error.
        #
        # mp=9 (Milbrandt-Yau) carries hail in the MOIST package exactly
        # like NSSL: Registry.EM_COMMON:469 declares qh with package
        # ``moist`` and the milbrandt2mom package binds
        # moist:qv,qc,qr,qi,qs,qg,qh with all six number moments in
        # ``scalar``, so its cq is NSSL's seven-mass sum (1.9.1 D1's
        # route: this arm was cfg-keyed and mp=9 was missing, so an
        # ACCEPTED Milbrandt-Yau real case refused at its first RK stage
        # -- unlike slow_buoyancy, whose presence-keyed dispatch already
        # ran mode 3).
        qi = state.qi
        qs, qg = state.qs, state.qg
        n_mass = 7 if cfg.mp_physics in (9, 18, 24) else 6
    elif _mass_loading_species(cfg.mp_physics) == ("qi", "qs", "qg"):
        # A NAMED scheme whose capability row declares the six-mass moist
        # package: the same calc_cq call as the Thompson family.
        qi = state.qi
        qs, qg = state.qs, state.qg
        n_mass = 6
    else:
        raise ValueError(f"unsupported mp_physics={cfg.mp_physics} for cq")
    qh = state.qh if cfg.mp_physics in (9, 18, 24) else state.qv
    n = (nz + 1) * (ny + 1) * (nx + 1)
    blocks = (n + _THREADS - 1) // _THREADS
    get_kernel("acoustic", "calc_cq")(
        (blocks,), (_THREADS,),
        (state.qv, state.qc, state.qr, qi, qs, qg, qh, cqu, cqv, cqw,
         np.int32(n_mass), np.int32(nz), np.int32(ny), np.int32(nx)))
    return cqu, cqv, cqw, True


def acoustic_substep_explicit(state: DomainState, cfg: RunConfig,
                              dtau: float, first: bool, cq=None,
                              mudf=None) -> None:
    """Explicit part of one acoustic substep of length ``dtau``.

    Advances ``u_pp``/``v_pp`` with the damped pressure gradient (the
    gradient sees ``p_pp + smdiv*(p_pp - p_pp_old)``; on the ``first``
    substep there is no history and the damping is dropped), saves
    the pre-update ``p_pp``/``mu_pp``/``th_pp`` histories while traversing
    those fields, then advances ``mu_pp``, ``ww_pp`` and ``th_pp`` from the
    updated momenta.

    Open lateral boundaries (Task 9): with ``cfg.open_x``/``open_y`` the
    boundary-normal momentum at the two boundary faces skips the pressure
    gradient and advances by the large-step tendency alone, WRF
    ``advance_uv``'s ``i_start_up``/``j_start_vp`` loop-bound exclusions
    (module_small_step_em.F).  The kernel computes the periodic update
    everywhere (the seam gradient is meaningless there) and the boundary
    faces are then rebuilt from their saved pre-substep values plus
    ``dtau * ru_t`` (the radiative tendency installed by
    ``dycore.apply_open_radiative_bc``).
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    smdiv = 0.0 if first else cfg.smdiv
    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    cqu, cqv, _cqw, use_cq = cq
    mudf_arg = state.mup if mudf is None else mudf
    write_mudf = np.int32(mudf is not None)
    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")

    if WRF_EXACT and (first or not getattr(state, "_wrf_acoustic_muts_initialized", False)):
        init_wrf_acoustic_muts(state)

    radiative_x = cfg.open_x and not _boundary_forced(cfg)
    radiative_y = cfg.open_y and not _boundary_forced(cfg)
    if radiative_x:
        sx = state.scratch((nz, ny, 2), "openbc_upp_faces")
        sx[..., 0] = state.u_pp[:, :, 0]
        sx[..., 1] = state.u_pp[:, :, -1]
    if radiative_y:
        sy = state.scratch((nz, 2, nx), "openbc_vpp_faces")
        sy[:, 0, :] = state.v_pp[:, 0, :]
        sy[:, 1, :] = state.v_pp[:, -1, :]

    kernel = get_kernel("acoustic", "advance_uv")
    n = nz * (ny + 1) * (nx + 1)
    blocks = (n + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (state.u_pp, state.v_pp, state.ru_t, state.rv_t,
            state.p_pp, state.p_pp_old, state.ph_pp, state.php, state.phb,
            state.alt, state.al_pp, state.pb,
             state.mup, state.mu_pp, state.mub2d,
             state.c1h, state.c2h, state.fnm, state.fnp, state.rdnw,
             cqu, cqv, np.int32(use_cq),
             state.cf1, state.cf2, state.cf3,
            np.int32(cfg.top_lid),
            DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy),
            DTYPE(dtau), DTYPE(smdiv),
            np.int32(_spec_zone(cfg)), _base3d(state),
            np.int32(nz), np.int32(ny), np.int32(nx))
           + _uv_emdiv_args(state, cfg, mudf))

    if radiative_x:
        state.u_pp[:, :, 0] = sx[..., 0] + DTYPE(dtau) * state.ru_t[:, :, 0]
        state.u_pp[:, :, -1] = (sx[..., 1]
                                + DTYPE(dtau) * state.ru_t[:, :, -1])
    if radiative_y:
        state.v_pp[:, 0, :] = sy[:, 0, :] + DTYPE(dtau) * state.rv_t[:, 0, :]
        state.v_pp[:, -1, :] = (sy[:, 1, :]
                                + DTYPE(dtau) * state.rv_t[:, -1, :])
    # Open boundaries: the kernel's cross-boundary neighbour reads clamp
    # (WRF's zero-gradient ghosts) instead of wrapping.  Map factors
    # (Task 3) select the _msf kernel variant: the msf==1 kernel stays
    # byte-identical to Phase 2 so its codegen cannot drift off the pinned
    # bitwise regression.
    blocks = (ny * nx + _THREADS - 1) // _THREADS
    if state.has_msf:
        kernel = get_kernel("acoustic", "advance_mu_th_msf", (nz, ny, nx))
        kernel((blocks,), (_THREADS,),
               (state.u_pp, state.v_pp, state.u, state.v,
                 state.mup, state.mu_pp, mu_old, state.rmu_t,
                 mudf_arg, write_mudf,
                 state.thp, state.thb, state.th_pp, th_old, state.rth_t,
                 state.ww_pp,
                 *((state.scratch((nz + 1, ny, nx), "rk_ww"),) if WRF_EXACT else ()),
                 state.p_pp, state.p_pp_old,
                 state.dnw, state.rdnw, state.fnm, state.fnp,
                state.c1h, state.c2h, state.mub2d,
                state.msft, state.msfu, state.msfv_inv if WRF_EXACT else state.msfv,
                DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy), DTYPE(dtau),
                _base3d(state), np.int32(nz), np.int32(ny), np.int32(nx),
                np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
                np.int32(_mass_w_boundary_zone(cfg))))

    else:
        kernel = get_kernel("acoustic", "advance_mu_th")
        kernel((blocks,), (_THREADS,),
               (state.u_pp, state.v_pp, state.u, state.v,
                 state.mup, state.mu_pp, mu_old, state.rmu_t,
                 mudf_arg, write_mudf,
                 state.thp, state.thb, state.th_pp, th_old, state.rth_t,
                 state.ww_pp,
                 *((state.scratch((nz + 1, ny, nx), "rk_ww"),) if WRF_EXACT else ()),
                 state.p_pp, state.p_pp_old,
                 state.dnw, state.rdnw, state.fnm, state.fnp,
                state.c1h, state.c2h, state.mub2d,
                DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy), DTYPE(dtau),
                _base3d(state), np.int32(nz), np.int32(ny), np.int32(nx),
                np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
                np.int32(_mass_w_boundary_zone(cfg))))

    if WRF_EXACT and _mass_w_boundary_zone(cfg):
        get_kernel("acoustic", "advance_exact_frame_mu_t")(
            (blocks,), (_THREADS,),
            (state.mu_pp, state.th_pp, state.rmu_t, state.rth_t,
             DTYPE(dtau), np.int32(_mass_w_boundary_zone(cfg)),
             np.int32(not _boundary_x(cfg)), np.int32(nz),
             np.int32(ny), np.int32(nx),
             wrf_acoustic_muts(state), state.mub2d, state.mup))


#: ``WPHI_MAX_LEV`` tiers ``kernels/acoustic.cu`` is compiled at, ascending.
#: The bound sizes the one per-thread RHS column in ``advance_w_phi`` and
#: ``advance_w_phi_msf`` and nothing else, so a tier costs 4 B of per-thread
#: local frame per admitted full level and moves no arithmetic.
#:
#: 129 is the literal the source keeps when nothing overrides it.  Every
#: ``nz <= 128`` configuration therefore compiles the UNSPECIALIZED module
#: -- :func:`woof.core.kernels.get_kernel`, no define injected, the exact
#: string NVRTC received before this ladder existed -- which is what makes
#: the below-ceiling kernel identical by construction rather than by
#: measurement.  Higher tiers exist only to admit deeper columns; they are
#: coarse on purpose, so a sweep over nz does not compile a module per nz.
WPHI_LEVEL_TIERS = (129, 193, 257)

#: The tier compiled with no define injected (the literal in acoustic.cu).
UNSPECIALIZED_WPHI_LEVEL_TIER = WPHI_LEVEL_TIERS[0]

#: Deepest ``nz`` the in-thread tridiagonal solve admits, from the top tier.
#: Mirrored by ``woof.physics_vertical_contract`` for front doors that must
#: refuse a grid without importing CuPy.
MAX_ACOUSTIC_LEVELS = WPHI_LEVEL_TIERS[-1] - 1


def wphi_level_tier(nz: int) -> int:
    """The ``WPHI_MAX_LEV`` this ``nz`` compiles the w''-phi'' solve at.

    The solve indexes full levels ``0..nz``, so a configuration needs
    ``nz + 1`` entries and takes the smallest tier that holds them.
    """
    nz = int(nz)
    if nz < 1:
        raise ValueError(f"nz must be positive, got {nz}")
    for tier in WPHI_LEVEL_TIERS:
        if nz + 1 <= tier:
            return tier
    raise ValueError(f"nz={nz} exceeds the in-thread solve limit "
                     f"({MAX_ACOUSTIC_LEVELS} half levels)")


def wphi_module_defines(nz: int) -> tuple[tuple[str, int], ...]:
    """Integer defines the w''-phi'' module compiles with at this ``nz``.

    EMPTY at the shipped tier.  That emptiness is the whole mechanism: it
    routes the launcher to the unspecialized loader, which assembles the
    same string it assembled before the ladder existed, so no ``nz <= 128``
    configuration can see a different translation unit.  Both the launcher
    and :func:`wphi_kernel_source` branch here and nowhere else, so the
    source a test digests is the source the driver compiles.
    """
    tier = wphi_level_tier(nz)
    if tier == UNSPECIALIZED_WPHI_LEVEL_TIER:
        return ()
    return (("WPHI_MAX_LEV", tier),)


def wphi_kernel_source(nz: int, ny: int = 0, nx: int = 0, *,
                       name: str = "advance_w_phi_msf") -> str:
    """The exact string NVRTC receives for the w''-phi'' solve at ``nz``.

    CPU-only mirror of what :func:`_w_phi_kernel` compiles, so the
    below-ceiling identity is testable on a host with no device.
    """
    defines = wphi_module_defines(nz)
    if ny and nx:
        from woof.core.kernel_shape import shape_defines
        defines += shape_defines("acoustic", name, (nz, ny, nx))
    if not defines:
        return module_source("acoustic")
    return module_source_int_defines("acoustic", defines)


def _w_phi_kernel(name: str, nz: int, ny: int = 0, nx: int = 0):
    """The w''-phi'' solve compiled at this configuration's tier.

    Registered mapped kernels bind supplied horizontal dimensions as well
    as the level count. Omitting dimensions keeps the generic source;
    larger vertical tiers retain their existing storage-size definitions.
    """
    defines = wphi_module_defines(nz)
    if not defines:
        return get_kernel("acoustic", name, (nz, ny, nx) if ny and nx else ())
    return get_kernel_int_defines("acoustic", name, defines, (nz, ny, nx) if ny and nx else ())


def prepare_acoustic_coefficients(state: DomainState, cfg: RunConfig,
                                  dtau: float, cq=None) -> tuple:
    """Prepare the fixed-reference implicit coefficients for one RK stage.

    ``p``, ``alt`` and ``mup`` are the stage-reference diagnostics/state;
    acoustic substeps update only the perturbation ``*_pp`` fields.  The
    returned persistent buffers therefore remain valid until the stage is
    folded back by ``dycore._finish_small_steps``.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    wphi_level_tier(nz)   # refuse a too-deep column before allocating
    c2a = state.scratch((nz, ny, nx), "acoustic_c2a")
    a = state.scratch((nz + 1, ny, nx), "acoustic_a")
    alpha = state.scratch((nz + 1, ny, nx), "acoustic_alpha")
    gam = state.scratch((nz + 1, ny, nx), "acoustic_gamma")
    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    cqu, cqv, cqw, use_cq = cq
    blocks = (ny * nx + _THREADS - 1) // _THREADS
    kernel = get_kernel("acoustic", "calc_coefs")
    kernel((blocks,), (_THREADS,),
           (state.p, state.alt, state.mup, c2a, a, alpha, gam,
            state.rdn, state.rdnw,
            state.c1h, state.c2h, state.c1f, state.c2f, state.mub2d,
            cqw, np.int32(use_cq),
            DTYPE(dtau), DTYPE(cfg.epssm), np.int32(cfg.top_lid),
            np.int32(nz), np.int32(ny), np.int32(nx)))
    base = (c2a, a, alpha, gam)
    return base + cq if use_cq else base


def prepare_acoustic_substep_launch(state: DomainState, cfg: RunConfig,
                                    dtau: float, coefficients: tuple, *,
                                    mudf=None, emdiv_args=None, sumflux=None):
    """Bind one stage's invariant acoustic launch metadata once.

    The returned callable submits the same arithmetic as
    :func:`acoustic_substep`.  Device arrays are retained by reference and
    continue to expose their in-place updates; only the immutable Python
    launch containers and scalar wrappers are reused.

    In the default build apply_emdiv and advance_uv run as one joined launch
    (:mod:`woof.core.acoustic_fusion`); a joined column launch (mu''/theta''/
    Omega'', the w''-phi'' solve, the specified frame and the sumflux
    accumulation) is selectable there and off by default, as measured.  ``emdiv_args`` is the argument tuple the
    caller would launch ``apply_emdiv`` with before this substep and
    ``sumflux`` the (ru_m, rv_m, ww_m) the caller would zero before the
    stage's first substep and accumulate after each one.  The returned
    callable's ``absorbs`` names which of those it now submits itself
    ("emdiv", "sumflux"); the caller launches the rest as before.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    upper_wind_limiter = prepare_upper_wind_limiter(state, cfg, dtau)
    wphi_level_tier(nz)   # refuse a too-deep column before allocating

    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")
    c2a, a, alpha, gam = coefficients[:4]
    if len(coefficients) == 4:
        cq = (state.p, state.p, state.p, False)
    else:
        cq = coefficients[4:]
    cqu, cqv, cqw, use_cq = cq
    mudf_arg = state.mup if mudf is None else mudf
    write_mudf = np.int32(mudf is not None)

    block = (_THREADS,)
    uv_n = nz * (ny + 1) * (nx + 1)
    uv_grid = ((uv_n + _THREADS - 1) // _THREADS,)
    column_grid = ((ny * nx + _THREADS - 1) // _THREADS,)
    rdx = DTYPE(1.0 / cfg.dx)
    rdy = DTYPE(1.0 / cfg.dy)
    dtau_arg = DTYPE(dtau)
    base3d = _base3d(state)
    nz_arg, ny_arg, nx_arg = np.int32(nz), np.int32(ny), np.int32(nx)
    spec_zone = np.int32(_spec_zone(cfg))
    mass_w_zone = np.int32(_mass_w_boundary_zone(cfg))
    boundary_x = np.int32(_boundary_x(cfg))
    boundary_y = np.int32(_boundary_y(cfg))

    uv_kernel = get_kernel("acoustic", "advance_uv")
    uv_prefix = (
        state.u_pp, state.v_pp, state.ru_t, state.rv_t,
        state.p_pp, state.p_pp_old, state.ph_pp, state.php, state.phb,
        state.alt, state.al_pp, state.pb,
        state.mup, state.mu_pp, state.mub2d,
        state.c1h, state.c2h, state.fnm, state.fnp, state.rdnw,
        cqu, cqv, np.int32(use_cq),
        state.cf1, state.cf2, state.cf3, np.int32(cfg.top_lid),
        rdx, rdy, dtau_arg,
    )
    uv_suffix = ((spec_zone, base3d, nz_arg, ny_arg, nx_arg)
                 + _uv_emdiv_args(state, cfg, mudf))
    uv_first_args = uv_prefix + (DTYPE(0.0),) + uv_suffix
    uv_later_args = uv_prefix + (DTYPE(cfg.smdiv),) + uv_suffix

    if state.has_msf:
        mu_name = "advance_mu_th_msf"
        mu_map_args = (state.msft, state.msfu, state.msfv_inv if WRF_EXACT else state.msfv)
    else:
        mu_name = "advance_mu_th"
        mu_map_args = ()
    mu_kernel = get_kernel("acoustic", mu_name, (nz, ny, nx))
    mu_args = (
        state.u_pp, state.v_pp, state.u, state.v,
        state.mup, state.mu_pp, mu_old, state.rmu_t,
        mudf_arg, write_mudf,
        state.thp, state.thb, state.th_pp, th_old, state.rth_t,
        state.ww_pp,
                 *((state.scratch((nz + 1, ny, nx), "rk_ww"),) if WRF_EXACT else ()),
                 state.p_pp, state.p_pp_old,
        state.dnw, state.rdnw, state.fnm, state.fnp,
        state.c1h, state.c2h, state.mub2d,
    ) + mu_map_args + (
        rdx, rdy, dtau_arg, base3d, nz_arg, ny_arg, nx_arg,
        boundary_x, boundary_y, mass_w_zone,
    )

    exact_frame_kernel = None
    exact_frame_args = None
    if WRF_EXACT and int(mass_w_zone):
        exact_frame_kernel = get_kernel("acoustic", "advance_exact_frame_mu_t")
        exact_frame_args = (
            state.mu_pp, state.th_pp, state.rmu_t, state.rth_t,
            dtau_arg, mass_w_zone, np.int32(not int(boundary_x)),
            nz_arg, ny_arg, nx_arg,
            wrf_acoustic_muts(state), state.mub2d, state.mup,
        )

    dampmag = damp_magnitude(cfg, dtau)
    w_name = "advance_w_phi_msf" if state.has_msf else "advance_w_phi"
    w_kernel = _w_phi_kernel(w_name, nz, ny, nx)
    w_threads = _W_PHI_MSF_THREADS if state.has_msf and not WRF_EXACT else _THREADS
    w_grid = ((ny * nx + w_threads - 1) // w_threads,)
    w_block = (w_threads,)
    w_map_args = (state.msft,) if state.has_msf else ()
    w_args = (
        state.w_pp, state.ph_pp, state.rw_t, state.rph_t, state.ww_pp,
        state.mu_pp, mu_old, state.th_pp, th_old,
        state.thp, state.thb, state.php, state.phb, state.alt, c2a,
        a, alpha, gam, state.mup,
        state.u_pp, state.v_pp, state.w, state.ht,
        state.rdn, state.rdnw, state.fnm, state.fnp,
        state.c1h, state.c2h, state.c1f, state.c2f, state.mub2d,
        cqw, np.int32(use_cq),
    ) + w_map_args + (
        state.p_pp, state.al_pp, state.cf1, state.cf2, state.cf3,
        rdx, rdy, dtau_arg, DTYPE(cfg.epssm), DTYPE(dampmag),
        DTYPE(cfg.zdamp), boundary_x, boundary_y, mass_w_zone, base3d,
        np.int32(cfg.top_lid), nz_arg, ny_arg, nx_arg,
    )

    frame_kernel = None
    frame_args = None
    if cfg.specified and not _frame_takes_table_w(cfg):
        frame_kernel = get_kernel("acoustic", "advance_specified_phi_w")
        frame_args = (
            state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t,
            state.th_pp, state.mup, state.mu_pp, state.mub2d, state.rmu_t,
            state.php, state.thp, state.thb, state.alt, c2a, state.rdnw,
            state.c1h, state.c2h, state.c1f, state.c2f, dtau_arg,
            np.int32(cfg.spec_zone), base3d, nz_arg, ny_arg, nx_arg,
        ) + ((wrf_acoustic_muts(state),) if WRF_EXACT else ())
    elif _frame_takes_table_w(cfg):
        frame_kernel = get_kernel("acoustic", "advance_nested_phi_w", (nz, ny, nx))
        frame_args = (
            state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t,
            state.rw_t, state.th_pp, state.mup, state.mu_pp, state.mub2d,
            state.rmu_t, state.php, state.thp, state.thb, state.alt, c2a,
            state.rdnw, state.c1h, state.c2h, state.c1f, state.c2f,
            dtau_arg, np.int32(cfg.spec_zone), base3d,
            nz_arg, ny_arg, nx_arg,
        ) + ((wrf_acoustic_muts(state),) if WRF_EXACT else ())

    radiative_x = cfg.open_x and not _boundary_forced(cfg)
    radiative_y = cfg.open_y and not _boundary_forced(cfg)
    sx = (state.scratch((nz, ny, 2), "openbc_upp_faces")
          if radiative_x else None)
    sy = (state.scratch((nz, 2, nx), "openbc_vpp_faces")
          if radiative_y else None)

    joined_uv = joined_column = None
    joined_written = {}
    frame_after_column = False
    if _fusion_enabled(nz):
        from woof.core.acoustic_fusion import FusionUnsupported
        try:
            (joined_uv, joined_column, joined_written,
             frame_after_column) = _prepare_joined(
                state, cfg, mu_name, mu_args, w_name, w_args,
                frame_kernel is not None and not _frame_takes_table_w(cfg),
                frame_args, int(mass_w_zone), uv_first_args, uv_later_args,
                emdiv_args if not (radiative_x or radiative_y) else None,
                sumflux, upper_wind_limiter is None
                and exact_frame_kernel is None
                and (frame_kernel is None or not _frame_takes_table_w(cfg)))
        except FusionUnsupported as refusal:
            # The separate kernels are the reference the joined text is
            # proven against, so they are always a correct substep; say
            # once why this process runs them.
            _warn_separate_substep(str(refusal))

    def launch(*, first: bool) -> None:
        if radiative_x:
            sx[..., 0] = state.u_pp[:, :, 0]
            sx[..., 1] = state.u_pp[:, :, -1]
        if radiative_y:
            sy[:, 0, :] = state.v_pp[:, 0, :]
            sy[:, 1, :] = state.v_pp[:, -1, :]

        if joined_uv is not None:
            joined_uv[0](uv_grid, block, joined_uv[1] if first else joined_uv[2])
        else:
            uv_kernel(uv_grid, block,
                      uv_first_args if first else uv_later_args)

        if radiative_x:
            state.u_pp[:, :, 0] = (sx[..., 0]
                                   + dtau_arg * state.ru_t[:, :, 0])
            state.u_pp[:, :, -1] = (sx[..., 1]
                                    + dtau_arg * state.ru_t[:, :, -1])
        if radiative_y:
            state.v_pp[:, 0, :] = (sy[:, 0, :]
                                   + dtau_arg * state.rv_t[:, 0, :])
            state.v_pp[:, -1, :] = (sy[:, 1, :]
                                    + dtau_arg * state.rv_t[:, -1, :])

        if joined_column is not None:
            joined_column[0](column_grid, block,
                             joined_column[1] if first else joined_column[2])
            if frame_after_column:
                frame_kernel(column_grid, block, frame_args)
            return
        mu_kernel(column_grid, block, mu_args)
        if exact_frame_kernel is not None:
            exact_frame_kernel(column_grid, block, exact_frame_args)
        w_kernel(w_grid, w_block, w_args)
        if upper_wind_limiter is not None:
            upper_wind_limiter()
        if frame_kernel is not None:
            frame_kernel(column_grid, block, frame_args)

    launch.absorbs = frozenset(
        (("emdiv",) if joined_uv is not None else ())
        + (("sumflux",) if joined_written.get("ru_m") is not None else ()))
    launch.joined = (joined_uv is not None, joined_column is not None)
    launch.joined_written = joined_written
    return launch


_SEPARATE_SUBSTEP_WARNED: set = set()


def _warn_separate_substep(reason: str) -> None:
    if reason in _SEPARATE_SUBSTEP_WARNED:
        return
    _SEPARATE_SUBSTEP_WARNED.add(reason)
    import warnings
    warnings.warn("acoustic substep: the joined launches refused this kernel "
                  f"text ({reason}); running the separate kernels",
                  RuntimeWarning, stacklevel=3)


def _fusion_enabled(nz: int) -> bool:
    """Joined substep launches: default arithmetic build, shipped w tier, and
    the loader's own kernels.

    The breakage the last condition prevents: verification harnesses (the
    WRF 4.7.1 small-step oracles, woof/verify/smallstep_*_oracle.py) swap
    ``get_kernel``/``_w_phi_kernel`` here for probe kernels and measure what
    the substep launches.  A joined launch would bypass the probe and the
    harness would grade the wrong kernel (measured: 7 setup errors and one
    word gate that no longer rejected its defective probe).
    """
    from woof.core import acoustic_fusion
    return (acoustic_fusion.ENABLED and not WRF_EXACT
            and not wphi_module_defines(nz)
            and get_kernel is _LOADER_GET_KERNEL
            and _w_phi_kernel is _LOADER_W_PHI_KERNEL)


def _prepare_joined(state, cfg, mu_name, mu_args, w_name, w_args,
                    specified_frame, frame_args, mass_w_zone,
                    uv_first_args, uv_later_args, emdiv_args, sumflux,
                    column_ok):
    """The joined u/v and column launches as (kernel, first args, later args).

    Returns those two (each None where the separate kernels stay), the
    arrays the joined launches may write, by parameter name, and whether the
    specified-frame kernel still runs as its own launch after the joined
    column (the separate order: mu, w, frame).  Each joined
    entry's arguments are bound by name from the argument tuples
    the separate kernels are launched with, so every parameter receives the
    very array or value its original kernel received.  A configuration the
    joined text does not cover keeps the separate kernel (None).
    """
    from woof.core import acoustic_fusion as fusion

    source = fusion.acoustic_source()
    options = fusion._audit_options()
    params = lambda entry: fusion.original_parameters(source, entry, options)
    # The frame's w copy is turned around (the source column writes it), which
    # needs the frame to be exactly the columns the w''-phi'' solve skips.
    # Which phases join the column is a measured choice (fusion.JOIN_*).
    frame = (frame_args is not None and specified_frame and fusion.JOIN_COLUMN_FRAME
             and int(cfg.spec_zone) == mass_w_zone)
    frame_after = frame_args is not None and specified_frame and not frame
    column = column_ok and fusion.JOIN_COLUMN and (frame_args is None or frame or frame_after)
    sumflux = sumflux if fusion.JOIN_COLUMN_SUMFLUX else None
    if emdiv_args is None and not column:
        return None, None, {}, False
    unit, uv_names, column_names, receipt = fusion.joined_unit(
        source, bool(state.has_msf), frame, sumflux is not None, options)
    joined_uv = joined_column = None
    if emdiv_args is not None:
        emdiv = (params("apply_emdiv"), emdiv_args, {"spec_zone": "emdiv_spec_zone"})
        joined_uv = (fusion.kernel(unit, fusion.UV_ENTRY),
                     fusion.bind(uv_names, emdiv, (params("advance_uv"), uv_first_args, {})),
                     fusion.bind(uv_names, emdiv, (params("advance_uv"), uv_later_args, {})))
    if column:
        named = [(params(mu_name), mu_args, {}), (params(w_name), w_args, {})]
        if frame:
            named.append((params("advance_specified_phi_w"), frame_args,
                          {"spec_zone": "frame_spec_zone"}))
        if sumflux is not None:
            named.append((("ru_m", "rv_m", "ww_m"), tuple(sumflux), {}))
        joined_column = (
            fusion.kernel(unit, fusion.COLUMN_ENTRY),
            fusion.bind(column_names, *named, (("sumflux_first",), (np.int32(1),), {})),
            fusion.bind(column_names, *named, (("sumflux_first",), (np.int32(0),), {})))
    # Every array the whole substep may write -- joined launch or separate
    # kernel -- for tools/acoustic_fusion_replay.py, which must restore all
    # of them between the two paths (a separate column run twice on one
    # state is two substeps).
    written = {}
    originals = [("advance_uv", uv_first_args, {}), (mu_name, mu_args, {}),
                 (w_name, w_args, {})]
    if emdiv_args is not None:
        originals.append(("apply_emdiv", emdiv_args, {"spec_zone": "emdiv_spec_zone"}))
    if frame_args is not None and specified_frame:
        originals.append(("advance_specified_phi_w", frame_args, {}))
    for entry, args, _ in originals:
        for name, value in zip(params(entry), args):
            if name in fusion.writable_parameters(source, entry, options):
                written.setdefault(f"{entry}.{name}", value)
    if joined_column is not None and sumflux is not None:
        written.update(zip(("ru_m", "rv_m", "ww_m"), sumflux))
    return joined_uv, joined_column, written, column and frame_after


def acoustic_substep(state: DomainState, cfg: RunConfig,
                     dtau: float, first: bool, coefficients=None,
                     mudf=None) -> None:
    """One complete acoustic substep of length ``dtau``.

    Explicit part (Task 10), then the vertically implicit w''-phi'' solve
    off-centered by ``cfg.epssm`` (WRF ``calc_coef_w``/``advance_w``,
    kinematic terrain bottom BC, and WRF's configurable open/rigid top), then
    the linearized-EOS p''/alpha'' diagnosis.  The pre-substep
    mu''/(mu*theta)'' are saved first: the implicit solve's buoyancy needs
    their epssm averages (WRF ``muave``/``t_2ave``).
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    wphi_level_tier(nz)   # refuse a too-deep column before allocating

    if coefficients is None:
        cq = prepare_moist_cq(state, cfg)
    elif len(coefficients) == 4:
        cq = (state.p, state.p, state.p, False)
    else:
        cq = coefficients[4:]
    acoustic_substep_explicit(state, cfg, dtau, first, cq=cq, mudf=mudf)
    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")

    # Standalone callers may omit a prepared tuple; dycore prepares exactly
    # once per stage and supplies it to every substep.
    if coefficients is None:
        coefficients = prepare_acoustic_coefficients(state, cfg, dtau, cq=cq)
    c2a, a, alpha, gam = coefficients[:4]
    cqu, cqv, cqw, use_cq = cq
    w_threads = _W_PHI_MSF_THREADS if state.has_msf and not WRF_EXACT else _THREADS
    blocks = (ny * nx + w_threads - 1) // w_threads

    dampmag = damp_magnitude(cfg, dtau)
    # Map factors (Task 3): the _msf variant carries WRF advance_w's msfty
    # factors; the msf==1 kernel stays byte-identical to Phase 2.
    name = "advance_w_phi_msf" if state.has_msf else "advance_w_phi"
    kernel = _w_phi_kernel(name, nz, ny, nx)
    args = [state.w_pp, state.ph_pp, state.rw_t, state.rph_t, state.ww_pp,
            state.mu_pp, mu_old, state.th_pp, th_old,
             state.thp, state.thb, state.php, state.phb, state.alt, c2a,
             a, alpha, gam, state.mup,
            state.u_pp, state.v_pp, state.w, state.ht,
            state.rdn, state.rdnw, state.fnm, state.fnp,
             state.c1h, state.c2h, state.c1f, state.c2f, state.mub2d,
             cqw, np.int32(use_cq)]
    if state.has_msf:
        args.append(state.msft)
    args += [state.p_pp, state.al_pp,
             state.cf1, state.cf2, state.cf3,
             DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy),
              DTYPE(dtau), DTYPE(cfg.epssm),
              DTYPE(dampmag), DTYPE(cfg.zdamp),
              np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
              np.int32(_mass_w_boundary_zone(cfg)), _base3d(state),
              np.int32(cfg.top_lid),
              np.int32(nz), np.int32(ny), np.int32(nx)]
    kernel((blocks,), (w_threads,), tuple(args))

    upper_wind_limiter = prepare_upper_wind_limiter(state, cfg, dtau)
    if upper_wind_limiter is not None:
        upper_wind_limiter()

    if cfg.specified and not _frame_takes_table_w(cfg):
        # The raw frame kernel preserves every eager FP32 round point in
        # spec_bdyupdate_ph and combines its independent zero-gradient w copy.
        n = ny * nx
        blocks = (n + _THREADS - 1) // _THREADS
        get_kernel("acoustic", "advance_specified_phi_w")(
            (blocks,), (_THREADS,),
            (state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t,
             state.th_pp, state.mup, state.mu_pp, state.mub2d, state.rmu_t,
             state.php, state.thp, state.thb, state.alt, c2a, state.rdnw,
             state.c1h, state.c2h, state.c1f, state.c2f, DTYPE(dtau),
             np.int32(cfg.spec_zone), _base3d(state), np.int32(nz),
             np.int32(ny), np.int32(nx))
            + ((wrf_acoustic_muts(state),) if WRF_EXACT else ()))
    elif _frame_takes_table_w(cfg):
        # solve_em.F:1577-1611 ELSE: spec_bdyupdate_ph followed by
        # spec_bdyupdate(w_2, rw_tend, dts_rk).  A specified domain that
        # relaxes w (relax_w) takes this same frame.  This child-only raw kernel
        # also repeats dyn's fused pressure diagnosis on the changed frame.
        n = ny * nx
        blocks = (n + _THREADS - 1) // _THREADS
        get_kernel("acoustic", "advance_nested_phi_w", (nz, ny, nx))(
            (blocks,), (_THREADS,),
            (state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t,
             state.rw_t, state.th_pp, state.mup, state.mu_pp, state.mub2d,
             state.rmu_t, state.php, state.thp, state.thb, state.alt, c2a,
             state.rdnw, state.c1h, state.c2h, state.c1f, state.c2f,
             DTYPE(dtau), np.int32(cfg.spec_zone), _base3d(state),
             np.int32(nz), np.int32(ny), np.int32(nx))
            + ((wrf_acoustic_muts(state),) if WRF_EXACT else ()))


def run_acoustic_only(state: DomainState, cfg: RunConfig,
                      dtau: float, n: int) -> None:
    """Drive ``n`` acoustic substeps around the frozen reference state.

    With zero large-step tendencies this integrates the pure (linearized)
    acoustic system: used by the sound-wave-speed test.
    """
    if n <= 0:
        return
    coefficients = prepare_acoustic_coefficients(state, cfg, dtau)
    launch_substep = prepare_acoustic_substep_launch(
        state, cfg, dtau, coefficients)
    for i in range(n):
        launch_substep(first=(i == 0))


#: The loader's kernel accessors as this module bound them at import; a
#: harness that swaps either keeps the separate kernels (_fusion_enabled).
_LOADER_GET_KERNEL = get_kernel
_LOADER_W_PHI_KERNEL = _w_phi_kernel
