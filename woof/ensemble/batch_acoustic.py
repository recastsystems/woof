"""Explicit member-batched acoustic stage components using existing CUDA bodies.

The coefficient and substep metadata mirror ``core.acoustic``. All batch
pointers are bound through an admitted BatchedDomainState inventory, including
shared or member-owned reference fields. Numerical launches cover every member;
there is no member stepping loop. This component does not implement a forecast
executor, boundary-table preparation, RK bookkeeping or history identity.
"""

from __future__ import annotations

import numpy as np

from woof.core import acoustic as original
from woof.ensemble.batch_kernel import KernelSpec, PointerSpec, prepare_batch_kernel_launch
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported
from woof.microphysics_schemes import mass_loading_species as _mass_loading_species

_THREADS = 256
_F32 = np.float32
_I32 = np.int32


class _ScalarState:
    """Scalar-shaped views for the unchanged N=1 helpers."""

    def __init__(self, state):
        self.state = state

    def __getattr__(self, name):
        if name in self.state.storage.specs:
            return self.state.member_view(name, 0)
        return getattr(self.state, name)

    def scratch(self, shape, slot, dtype=None):
        self.state.scratch(shape, slot, dtype)
        return self.state.scratch_member_view(slot, 0)

    def existing_scratch(self, slot):
        value = self.state.existing_scratch(slot)
        return None if value is None else self.state.scratch_member_view(slot, 0)


def _scalar_state(state):
    return _ScalarState(state) if isinstance(state, BatchedDomainState) else state


def _scalar_array(state, value):
    if isinstance(state, BatchedDomainState):
        for name, array in state.storage.arrays.items():
            if value is array:
                return state.storage.member_view(name, 0)
    return value


def _single(state):
    return not isinstance(state, BatchedDomainState) or state.members == 1


def _fields(state, names):
    return tuple(getattr(state, name) for name in names.split())


class _Bindings:
    def __init__(self, state, cfg):
        if not isinstance(state, BatchedDomainState):
            raise TypeError("member acoustic launches need a BatchedDomainState inventory")
        expected = (cfg.nz, cfg.ny, cfg.nx)
        if state.storage.specs["p"].shape != expected:
            raise ValueError("acoustic grid differs from the admitted member pressure shape")
        self.state, self.cfg = state, cfg
        self.inventory = {id(array): (name, spec)
                          for name, spec in state.storage.specs.items()
                          for array in (state.storage.arrays[name],)}
        intervals = []
        for name, spec in state.storage.specs.items():
            array = state.storage.arrays[name]
            if (not hasattr(array, "__cuda_array_interface__")
                    or array.dtype != np.dtype(spec.dtype) or not array.flags.c_contiguous
                    or tuple(array.shape) != spec.allocation_shape(state.members)):
                raise ValueError(f"{name} differs from its admitted CUDA shape/dtype backing")
            start = int(array.__cuda_array_interface__["data"][0])
            intervals.append((start, start + array.nbytes, name))
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"admitted acoustic backings {previous[2]} and {current[2]} overlap; "
                    "parallel writes would change another field's input")

    def owner(self, value):
        found = self.inventory.get(id(value))
        if found is None:
            raise BatchStateUnsupported(
                "acoustic pointer is outside the admitted state/scratch "
                "inventory; its member stride and allocation lifetime are unknown")
        name, spec = found
        if self.state.storage.arrays[name] is not value:
            raise ValueError("acoustic backing identity changed after admission")
        return name, spec

    def bind(self, entry, args, pointers, grid, *, defines=()):
        rows, strides = [], {}
        for name, value in pointers:
            owner, spec = self.owner(value)
            rows.append(PointerSpec(name, spec.ownership, dtype=spec.dtype))
            strides[name] = self.state.storage.pointer_stride_bytes(owner)
        spec = KernelSpec("acoustic", entry, tuple(rows))
        return prepare_batch_kernel_launch(
            spec, self.state.members, grid, (_THREADS,), args,
            pointer_strides=strides, defines=defines)

    @property
    def base3d(self):
        return _I32(len(self.state.storage.specs["thb"].shape) == 3)


def _pairs(names, arrays):
    return tuple(zip(names.split(), arrays, strict=True))


def prepare_moist_cq(state, cfg):
    """Return the original dry dummy tuple or batch the active mass-loading sum."""
    if _single(state):
        return original.prepare_moist_cq(_scalar_state(state), cfg)
    if not bool(getattr(cfg, "moist_cq", True) and getattr(state, "qv", None) is not None):
        return state.p, state.p, state.p, False
    bindings = _Bindings(state, cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    cqu = state.scratch((nz, ny, nx + 1), "acoustic_cqu")
    cqv = state.scratch((nz, ny + 1, nx), "acoustic_cqv")
    cqw = state.scratch((nz + 1, ny, nx), "acoustic_cqw")
    if cfg.mp_physics == 0:
        qi = qs = qg = state.qv
        n_mass = 1
    elif cfg.mp_physics in (1, 3):
        qi = qs = qg = state.qv
        n_mass = 3
    elif getattr(state, "qi", None) is not None and getattr(state, "qs", None) is None:
        qi = state.qi
        qs = qg = state.scratch((nz, ny, nx), "moist_absent_mass")
        qs[...] = 0.0
        n_mass = 6
    elif cfg.mp_physics in (2, 6, 8, 9, 10, 16, 18, 24, 28, 30, 106):
        qi, qs, qg = state.qi, state.qs, state.qg
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
    arrays = (state.qv, state.qc, state.qr, qi, qs, qg, qh, cqu, cqv, cqw)
    args = arrays + (_I32(n_mass), _I32(nz), _I32(ny), _I32(nx))
    n = (nz + 1) * (ny + 1) * (nx + 1)
    bindings.bind("calc_cq", args, _pairs("qv qc qr qi qs qg qh cqu cqv cqw", arrays),
                  ((n + _THREADS - 1) // _THREADS,))()
    return cqu, cqv, cqw, True


def prepare_acoustic_coefficients(state, cfg, dtau, cq=None):
    """Prepare stage-fixed coefficients with the original tier/scalar rules."""
    if _single(state):
        if cq is not None:
            cq = tuple(_scalar_array(state, value) for value in cq)
        return original.prepare_acoustic_coefficients(_scalar_state(state), cfg, dtau, cq=cq)
    bindings = _Bindings(state, cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    original.wphi_level_tier(nz)
    c2a = state.scratch((nz, ny, nx), "acoustic_c2a")
    a = state.scratch((nz + 1, ny, nx), "acoustic_a")
    alpha = state.scratch((nz + 1, ny, nx), "acoustic_alpha")
    gam = state.scratch((nz + 1, ny, nx), "acoustic_gamma")
    cq = prepare_moist_cq(state, cfg) if cq is None else cq
    _cqu, _cqv, cqw, use_cq = cq
    arrays = (state.p, state.alt, state.mup, c2a, a, alpha, gam) + _fields(
        state, "rdn rdnw c1h c2h c1f c2f mub2d") + (cqw,)
    args = arrays + (_I32(use_cq), _F32(dtau), _F32(cfg.epssm), _I32(cfg.top_lid),
                     _I32(nz), _I32(ny), _I32(nx))
    bindings.bind("calc_coefs", args, _pairs(
        "p alt mup c2a a alpha gam rdn rdnw c1h c2h c1f c2f mub2d cqw", arrays),
        ((ny * nx + _THREADS - 1) // _THREADS,))()
    base = (c2a, a, alpha, gam)
    return base + tuple(cq) if use_cq else base


def prepare_acoustic_substep_launch(state, cfg, dtau, coefficients, *, mudf=None):
    """Bind the original acoustic ordering, one batch launch per raw operation."""
    if _single(state):
        return original.prepare_acoustic_substep_launch(
            _scalar_state(state), cfg, dtau,
            tuple(_scalar_array(state, value) for value in coefficients),
            mudf=_scalar_array(state, mudf))
    if (getattr(cfg, "upper_wind_limiter_form", "wrf_461") == "noaa_wrf39"
            and cfg.damp_opt == 3):
        raise BatchStateUnsupported(
            "the member-batched acoustic graph does not bind the fork's "
            "saved-wind limiter; upper_wind_limiter_form = noaa_wrf39 "
            "runs on the ordinary door")
    bindings = _Bindings(state, cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    original.wphi_level_tier(nz)
    if len(coefficients) not in (4, 8):
        raise ValueError("acoustic coefficients need the original four or eight entries")
    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")
    c2a, a, alpha, gam = coefficients[:4]
    cq = (state.p, state.p, state.p, False) if len(coefficients) == 4 else coefficients[4:]
    cqu, cqv, cqw, use_cq = cq
    mudf_arg = state.mup if mudf is None else mudf
    write_mudf = _I32(mudf is not None)
    uv_grid = ((nz * (ny + 1) * (nx + 1) + _THREADS - 1) // _THREADS,)
    columns = ((ny * nx + _THREADS - 1) // _THREADS,)
    rdx, rdy, dt = _F32(1.0 / cfg.dx), _F32(1.0 / cfg.dy), _F32(dtau)
    nz_i, ny_i, nx_i = _I32(nz), _I32(ny), _I32(nx)
    base3d = bindings.base3d
    spec_zone = _I32(original._spec_zone(cfg))
    mass_w_zone = _I32(original._mass_w_boundary_zone(cfg))
    bx, by = _I32(original._boundary_x(cfg)), _I32(original._boundary_y(cfg))

    uv_arrays = _fields(state, "u_pp v_pp ru_t rv_t p_pp p_pp_old ph_pp php phb alt al_pp pb mup mu_pp mub2d c1h c2h fnm fnp rdnw") + (cqu, cqv)
    uv_prefix = uv_arrays + (_I32(use_cq), state.cf1, state.cf2, state.cf3,
                             _I32(cfg.top_lid), rdx, rdy, dt)
    uv_suffix = ((spec_zone, base3d, nz_i, ny_i, nx_i)
                 + original._uv_emdiv_args(state, cfg, mudf))
    uv_pointers = _pairs("u_pp v_pp ru_t rv_t p_pp p_pp_old ph_pp php phb alt al_pp pb mup mu_pp mub2d c1h c2h fnm fnp rdnw cqu cqv", uv_arrays)
    if original.WRF_EXACT:
        uv_pointers += _pairs("mudf msfu msfv", uv_suffix[5:8])
    uv_first = bindings.bind("advance_uv", uv_prefix + (_F32(0.0),) + uv_suffix,
                             uv_pointers, uv_grid)
    uv_later = bindings.bind("advance_uv", uv_prefix + (_F32(cfg.smdiv),) + uv_suffix,
                             uv_pointers, uv_grid)

    mu_head = _fields(state, "u_pp v_pp u v mup mu_pp") + (mu_old, state.rmu_t, mudf_arg)
    # The strict source reads the stage's full eta mass flux after ww_pp.
    # This is an admitted member scratch carrier, exactly as the scalar bind.
    ww_ref = (state.scratch((nz + 1, ny, nx), "rk_ww"),) if original.WRF_EXACT else ()
    mu_tail_arrays = (state.thp, state.thb, state.th_pp, th_old, state.rth_t,
                      state.ww_pp) + ww_ref + (state.p_pp, state.p_pp_old) + _fields(
                          state, "dnw rdnw fnm fnp c1h c2h mub2d")
    mu_maps = ((state.msft, state.msfu, state.msfv_inv if original.WRF_EXACT else state.msfv)
               if state.has_msf else ())
    mu_arrays = mu_head + mu_tail_arrays + mu_maps
    mu_args = mu_head + (write_mudf,) + mu_tail_arrays + mu_maps + (
        rdx, rdy, dt, base3d, nz_i, ny_i, nx_i, bx, by, mass_w_zone)
    mu_names = "u_pp v_pp u v mup mu_pp mu_pp_old rmu_t mudf thp thb th_pp th_pp_old rth_t ww_pp"
    if original.WRF_EXACT:
        mu_names += " ww_ref"
    mu_names += " p_pp p_pp_old dnw rdnw fnm fnp c1h c2h mub2d"
    if state.has_msf:
        mu_names += " msft msfu msfv"
    mu = bindings.bind("advance_mu_th_msf" if state.has_msf else "advance_mu_th",
                       mu_args, _pairs(mu_names, mu_arrays), columns)

    exact_frame = None
    if original.WRF_EXACT and int(mass_w_zone):
        arrays = _fields(state, "mu_pp th_pp rmu_t rth_t")
        carrier = (original.wrf_acoustic_muts(state), state.mub2d, state.mup)
        exact_frame = bindings.bind("advance_exact_frame_mu_t", arrays + (
            dt, mass_w_zone, _I32(not int(bx)), nz_i, ny_i, nx_i) + carrier,
            _pairs("mu_pp th_pp rmu_t rth_t muts mub2d mup", arrays + carrier), columns)

    w_head = _fields(state, "w_pp ph_pp rw_t rph_t ww_pp mu_pp") + (
        mu_old, state.th_pp, th_old) + _fields(state, "thp thb php phb alt") + (
        c2a, a, alpha, gam) + _fields(state, "mup u_pp v_pp w ht rdn rdnw fnm fnp c1h c2h c1f c2f mub2d") + (cqw,)
    w_maps = (state.msft,) if state.has_msf else ()
    w_outputs = (state.p_pp, state.al_pp)
    dampmag = original.damp_magnitude(cfg, dtau)
    w_args = w_head + (_I32(use_cq),) + w_maps + w_outputs + (
        state.cf1, state.cf2, state.cf3, rdx, rdy, dt, _F32(cfg.epssm),
        _F32(dampmag), _F32(cfg.zdamp), bx, by, mass_w_zone, base3d,
        _I32(cfg.top_lid), nz_i, ny_i, nx_i)
    w_names = "w_pp ph_pp rw_t rph_t ww_pp mu_pp mu_pp_old th_pp th_pp_old thp thb php phb alt c2a a alpha gam mup u_pp v_pp w_ref ht rdn rdnw fnm fnp c1h c2h c1f c2f mub2d cqw"
    if state.has_msf:
        w_names += " msft"
    w_names += " p_pp al_pp"
    w = bindings.bind("advance_w_phi_msf" if state.has_msf else "advance_w_phi",
                      w_args, _pairs(w_names, w_head + w_maps + w_outputs), columns,
                      defines=original.wphi_module_defines(nz))

    frame = None
    carrier = (original.wrf_acoustic_muts(state),) if original.WRF_EXACT else ()
    carrier_pointers = _pairs("muts_wrf", carrier) if carrier else ()
    if cfg.specified and not original._frame_takes_table_w(cfg):
        arrays = _fields(state, "ph_pp w_pp p_pp al_pp rph_t th_pp mup mu_pp mub2d rmu_t php thp thb alt") + (c2a,) + _fields(state, "rdnw c1h c2h c1f c2f")
        frame = bindings.bind("advance_specified_phi_w", arrays + (
            dt, _I32(cfg.spec_zone), base3d, nz_i, ny_i, nx_i) + carrier,
            _pairs("ph_pp w_pp p_pp al_pp rph_t th_pp mup mu_pp mub2d rmu_t php thp thb alt c2a rdnw c1h c2h c1f c2f", arrays) + carrier_pointers, columns)
    elif original._frame_takes_table_w(cfg):
        arrays = _fields(state, "ph_pp w_pp p_pp al_pp rph_t rw_t th_pp mup mu_pp mub2d rmu_t php thp thb alt") + (c2a,) + _fields(state, "rdnw c1h c2h c1f c2f")
        frame = bindings.bind("advance_nested_phi_w", arrays + (
            dt, _I32(cfg.spec_zone), base3d, nz_i, ny_i, nx_i) + carrier,
            _pairs("ph_pp w_pp p_pp al_pp rph_t rw_t th_pp mup mu_pp mub2d rmu_t php thp thb alt c2a rdnw c1h c2h c1f c2f", arrays) + carrier_pointers, columns)

    radiative_x = cfg.open_x and not original._boundary_forced(cfg)
    radiative_y = cfg.open_y and not original._boundary_forced(cfg)
    sx = state.scratch((nz, ny, 2), "openbc_upp_faces") if radiative_x else None
    sy = state.scratch((nz, 2, nx), "openbc_vpp_faces") if radiative_y else None

    def launch(*, first):
        import cupy as cp
        if original.WRF_EXACT and first:
            original.init_wrf_acoustic_muts(state)
        if radiative_x:
            sx[..., 0] = state.u_pp[:, :, :, 0]
            sx[..., 1] = state.u_pp[:, :, :, -1]
        if radiative_y:
            sy[:, :, 0, :] = state.v_pp[:, :, 0, :]
            sy[:, :, 1, :] = state.v_pp[:, :, -1, :]
        (uv_first if first else uv_later)()
        # Restore each face with the original multiply then saved-face add.
        # The UV result on this face is overwritten, so it is the temporary.
        if radiative_x:
            for face, saved in ((0, 0), (-1, 1)):
                out = state.u_pp[:, :, :, face]
                cp.multiply(dt, state.ru_t[:, :, :, face], out=out)
                cp.add(sx[..., saved], out, out=out)
        if radiative_y:
            for face, saved in ((0, 0), (-1, 1)):
                out = state.v_pp[:, :, face, :]
                cp.multiply(dt, state.rv_t[:, :, face, :], out=out)
                cp.add(sy[:, :, saved, :], out, out=out)
        mu()
        if exact_frame is not None:
            exact_frame()
        w()
        if frame is not None:
            frame()

    return launch


def prepare_emdiv_filter_launch(state, cfg, mudf, mu_prev=None):
    """Bind the original external-mode filter across all members."""
    if _single(state):
        from woof.core.dycore import _prepare_emdiv_filter_launch
        return _prepare_emdiv_filter_launch(_scalar_state(state), cfg,
                                             _scalar_array(state, mudf),
                                             _scalar_array(state, mu_prev))
    bindings = _Bindings(state, cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    save = mu_prev is not None
    mu_prev_arg = mu_prev if save else mudf
    arrays = (state.u_pp, state.v_pp, mudf, state.mu_pp, mu_prev_arg,
              state.c1h, state.msfu, state.msfv)
    # Strict damping has already entered advance_uv's statement. This
    # launch only saves the previous mass, matching the scalar launcher.
    args = arrays + (_F32(0.0) if original.WRF_EXACT else _F32(-cfg.emdiv * cfg.dx),
                     _F32(0.0) if original.WRF_EXACT else _F32(-cfg.emdiv * cfg.dy),
                     _I32(state.has_msf), _I32(original._boundary_x(cfg)),
                     _I32(original._boundary_y(cfg)), _I32(original._boundary_forced(cfg)),
                     _I32(cfg.spec_zone), _I32(save), _I32(nz), _I32(ny), _I32(nx))
    return bindings.bind("apply_emdiv", args,
                         _pairs("u_pp v_pp mudf mu_pp mu_prev c1h msfu msfv", arrays),
                         ((nz * (ny + 1) * (nx + 1) + 255) // 256,))


def prepare_emdiv_mudf_launch(state, cfg, mudf, mu_prev, dtau):
    """Bind the original MUDF recurrence without an additional member copy."""
    if _single(state):
        from woof.core.dycore import _prepare_emdiv_mudf_launch
        return _prepare_emdiv_mudf_launch(_scalar_state(state), cfg,
                                           _scalar_array(state, mudf),
                                           _scalar_array(state, mu_prev), dtau)
    bindings = _Bindings(state, cfg)
    arrays = (mudf, state.mu_pp, mu_prev)
    forced = _I32(original._boundary_forced(cfg))
    args = arrays + (_F32(dtau), forced, forced, _I32(cfg.spec_zone),
                     _I32(cfg.ny), _I32(cfg.nx))
    return bindings.bind("update_mudf", args, _pairs("mudf mu_pp mu_prev", arrays),
                         ((cfg.ny * cfg.nx + 255) // 256,))
