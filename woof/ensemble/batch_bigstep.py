"""Dry big-step raw launches over inventoried member state and scratch.

These are partial primitives, not RK integration, history, forcing or physics
drivers. They reuse existing CUDA sources and scalar argument trees. N=1 uses
the unchanged scalar helpers on original-shaped views. N>1 launches each
numerical entry once for all members, with explicit pointer ownership/strides.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from woof.core.device_inventory import state_array_shapes
from woof.ensemble.batch_kernel import KernelSpec, PointerSpec, prepare_batch_kernel_launch
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported, _exact_key


# Parameter name -> existing allocation name. Roles are resolved from storage
# specs, including byte-verified shared candidates and member-owned scratch.
POINTER_FIELDS = {
    "slow_pgf": (
        ("ru_t", "ru_t"), ("rv_t", "rv_t"), ("p", "p"), ("pb", "pb"),
        ("al", "al"), ("alt", "alt"), ("php", "php"), ("phb", "phb"),
        ("mup", "mup"), ("mub2d", "mub2d"), ("c1h", "c1h"),
        ("c2h", "c2h"), ("rdnw", "rdnw"), ("fnm", "fnm"),
        ("fnp", "fnp"), ("cqu", "p"), ("cqv", "p")),
    "slow_buoyancy": (
        ("rw_t", "rw_t"), ("p", "p"), ("pb", "pb"), ("mup", "mup"),
        ("mub2d", "mub2d"), ("qv", "p"), ("qc", "p"), ("qr", "p"),
        ("qi", "p"), ("qs", "p"), ("qg", "p"), ("qh", "p"),
        ("rdn", "rdn"), ("rdnw", "rdnw"), ("c1f", "c1f"),
        ("c2f", "c2f"), ("msft", "msft")),
    "slow_geopotential": (
        ("rph_t", "rph_t"), ("ww", "scratch:rk_ww"), ("w", "w"),
        ("u", "u"), ("v", "v"), ("php", "php"), ("phb", "phb"),
        ("mup", "mup"), ("mub2d", "mub2d"), ("rdnw", "rdnw"),
        ("fnm", "fnm"), ("fnp", "fnp"), ("c1f", "c1f"),
        ("c2f", "c2f"), ("msft", "msft"), ("msfu", "msfu"),
        ("msfv", "msfv")),
    # Radiative open boundaries without specified forcing take WRF's native
    # rhs_ph (890e2523a); without face masses both face pointers read mup.
    "slow_geopotential_open": (
        ("rph_t", "rph_t"), ("ww", "scratch:rk_ww"), ("w", "w"),
        ("u", "u"), ("v", "v"), ("php", "php"), ("phb", "phb"),
        ("mup", "mup"), ("mub", "mub2d"), ("rdnw", "rdnw"),
        ("fnm", "fnm"), ("fnp", "fnp"), ("c1f", "c1f"),
        ("c2f", "c2f"), ("msft", "msft"), ("msfu", "msfu"),
        ("msfv", "msfv"), ("mux", "mup"), ("muy", "mup")),
    "small_step_init_uv": tuple((name, name) for name in (
        "u_pp", "v_pp", "u0", "v0", "u", "v", "mup0", "mup", "mub2d",
        "c1h", "c2h", "msfu", "msfv")),
    "small_step_init_column": tuple((name, name) for name in (
        "w_pp", "th_pp", "ph_pp", "mu_pp", "al_pp", "p_pp", "p_pp_old",
        "w0", "w", "thp0", "thp", "php0", "php", "p", "alt", "mup0",
        "mup", "mub2d", "thb", "c1h", "c2h", "c1f", "c2f", "rdnw", "msft")),
    "small_step_finish_uv": tuple((name, name) for name in (
        "u", "v", "u_pp", "v_pp", "mup", "mu_pp", "mub2d", "c1h", "c2h",
        "msfu", "msfv")),
    "small_step_finish_column": tuple((name, name) for name in (
        "w", "thp", "php", "mup", "w_pp", "th_pp", "ph_pp", "mu_pp",
        "mub2d", "thb", "c1h", "c2h", "c1f", "c2f", "msft")) + (("h_diabatic", "p"),),
    "set_surface_w": tuple((name, name) for name in ("u", "v", "ht", "msft", "w")),
    "coriolis_curvature": (
        ("ru", "scratch:rk_ru"), ("rv", "scratch:rk_rv"), ("u", "u"),
        ("v", "v"), ("w", "w"), ("mut", "scratch:smag_mut"),
        ("msft", "msft"), ("msfu", "msfu"), ("msfv", "msfv"),
        ("f", "f"), ("e", "e"), ("sina", "sina"), ("cosa", "cosa"),
        ("c1f", "c1f"), ("c2f", "c2f"), ("fzm", "fnm"), ("fzp", "fnp"),
        ("ru_t", "ru_t"), ("rv_t", "rv_t"), ("rw_t", "rw_t")),
}
_MODULES = {"set_surface_w": "surface_w", "coriolis_curvature": "coriolis_map"}
_WRITE_PARAMETERS = {
    "slow_pgf": ("ru_t", "rv_t"), "slow_buoyancy": ("rw_t",),
    "slow_geopotential": ("rph_t",), "slow_geopotential_open": ("rph_t",),
    "set_surface_w": ("w",),
    "coriolis_curvature": ("ru_t", "rv_t", "rw_t"),
    "small_step_init_uv": ("u_pp", "v_pp"),
    "small_step_init_column": ("w_pp", "th_pp", "ph_pp", "mu_pp", "al_pp", "p_pp", "p_pp_old"),
    "small_step_finish_uv": ("u", "v"),
    "small_step_finish_column": ("w", "thp", "php", "mup"),
}


def kernel_spec_for(batch, entry, *, bindings=None):
    """Audited roles from the state's exact allocation inventory, without CUDA."""
    if not isinstance(batch, BatchedDomainState):
        raise TypeError("a batch kernel specification needs BatchedDomainState storage")
    replacements = dict(bindings or {})
    allowed = {parameter for parameter, _ in POINTER_FIELDS[entry]}
    if replacements.keys() - allowed:
        raise ValueError("pointer overrides must name an audited entry parameter")
    pointers = []
    for parameter, default in POINTER_FIELDS[entry]:
        allocation = replacements.get(parameter, default)
        if allocation not in batch.storage.specs:
            raise BatchStateUnsupported(f"{entry} pointer {parameter} needs planned allocation {allocation!r}; an implicit temporary would bypass admission")
        spec = batch.storage.specs[allocation]
        if np.dtype(spec.dtype) != np.dtype(np.float32):
            raise TypeError(f"{entry} pointer {parameter} requires float32 backing")
        if parameter in _WRITE_PARAMETERS[entry] and spec.ownership != "member":
            raise BatchStateUnsupported(f"{entry} output {parameter} cannot write shared member state")
        pointers.append(PointerSpec(parameter, spec.ownership, spec.dtype))
    return KernelSpec(_MODULES.get(entry, "dycore"), entry, tuple(pointers))


def _context(state, cfg):
    batch = isinstance(state, BatchedDomainState)
    if cfg is None:
        if not batch:
            raise TypeError("an existing scalar state needs its explicit RunConfig")
        cfg = state.cfg
    if batch and _exact_key(cfg) != _exact_key(state.cfg):
        raise BatchStateUnsupported("big-step configuration differs from the prepared state; flags/grid would bind another numerical problem")
    return cfg, state.members if batch else 1


def prepare_w_damping(state, *, ww='scratch:rk_ww'):
    """Use the original cell-local limiter with explicit member pointers."""
    from woof.core import dycore
    cfg=state.cfg
    if cfg.w_damping != 1:
        return lambda: None
    if state.members == 1:
        from woof.ensemble.batch_dycore import scalar_domain_view
        scalar=scalar_domain_view(state)
        return lambda: dycore.apply_w_damping(scalar,cfg,state.member_view(ww,0))
    rows=(('rw_t','rw_t'),('ww',ww),('w','w'),('mup','mup'),('mub2d','mub2d'),
          ('c1f','c1f'),('c2f','c2f'),('rdnw','rdnw'))
    spec=KernelSpec('openbc','w_damp',tuple(PointerSpec(parameter,state.storage.specs[name].ownership)
                                          for parameter,name in rows))
    count=(cfg.nz-1)*cfg.ny*cfg.nx
    args=tuple(state.storage.arrays[name] for _,name in rows)+(np.float32(cfg.dt),
          np.float32(dycore.w_damp_onset(cfg)),np.float32(cfg.w_crit_cfl),
          np.int32(cfg.nz),np.int32(cfg.ny),np.int32(cfg.nx))
    return prepare_batch_kernel_launch(spec,state.members,((count+255)//256,),(256,),args,
          pointer_strides={parameter:state.storage.pointer_stride_bytes(name) for parameter,name in rows})


def _single_view(state):
    if not isinstance(state, BatchedDomainState):
        return state
    values = dict(state.scalars)
    values.update({name: state.member_view(name, 0)
                   for name in state.storage.specs if not name.startswith("scratch:")})
    values.setdefault("qv", None)
    values["physics"] = None
    values["_scratch"] = {slot: state.scratch_member_view(slot, 0) for slot in state._scratch}
    return SimpleNamespace(**values)


def _allocation(state, name, *, single=False):
    if isinstance(state, BatchedDomainState):
        if name not in state.storage.specs:
            raise BatchStateUnsupported(f"required allocation {name!r} is not planned")
        return state.storage.member_view(name, 0) if single else state.storage.arrays[name]
    if name.startswith("scratch:"):
        result = state.existing_scratch(name[len("scratch:"):])
        if result is None:
            raise BatchStateUnsupported(f"required scratch {name!r} is not prepared")
        return result
    return getattr(state, name)


def _shape(state, name):
    if isinstance(state, BatchedDomainState) and name not in state.storage.specs:
        raise BatchStateUnsupported(f"required allocation {name!r} is not planned")
    return (state.storage.specs[name].shape if isinstance(state, BatchedDomainState)
            else tuple(_allocation(state, name).shape))


def _bound_launch(state, entry, args, grid, *, bindings=None, threads=256):
    replacements = dict(bindings or {})
    spec = kernel_spec_for(state, entry, bindings=replacements)
    strides = {parameter: state.storage.pointer_stride_bytes(replacements.get(parameter, name))
               for parameter, name in POINTER_FIELDS[entry]}
    launch = prepare_batch_kernel_launch(
        spec, state.members, grid, (threads,), tuple(args), pointer_strides=strides)
    launch.numerical_entries = (entry,)
    return launch


def _single_launch(function, entries):
    function.numerical_entries = tuple(entries)
    return function


def _pressure_field(state):
    from woof.wrf_exact import DIAGNOSTICS_ENABLED
    from woof.core import dycore
    if bool(dycore.DIAGNOSTICS_ENABLED) != bool(DIAGNOSTICS_ENABLED):
        raise BatchStateUnsupported("pressure compiler mode changed after the scalar helper was imported; a fresh process is needed for one consistent pressure ABI")
    if not DIAGNOSTICS_ENABLED:
        return "p"
    if isinstance(state, BatchedDomainState):
        spec = state.storage.specs.get("p_perturbation")
        if spec is None:
            raise BatchStateUnsupported("strict big-step pressure needs planned p_perturbation member backing; subtracting from rounded total pressure would change its single-run words")
        if (spec.ownership != "member" or spec.shape != state.storage.specs["p"].shape
                or np.dtype(spec.dtype) != np.dtype(np.float32)):
            raise ValueError("strict perturbation pressure must retain the member pressure shape/dtype and independent ownership")
    else:
        pressure = getattr(state, "p_perturbation", None)
        if pressure is None:
            raise BatchStateUnsupported("strict scalar big-step pressure needs its prepared p_perturbation field; rounded total pressure is not an equivalent carrier")
        if pressure.shape != state.p.shape or pressure.dtype != np.dtype(np.float32):
            raise ValueError("strict scalar perturbation pressure differs from the original pressure shape/dtype")
    return "p_perturbation"


def prepare_slow_pgf(state, cfg=None, *, cq=None):
    """One original horizontal pressure-gradient launch for all members."""
    cfg, members = _context(state, cfg)
    pressure = _pressure_field(state)
    from woof.core import dycore
    if members == 1:
        single = _single_view(state)
        if cq is not None:
            from woof.ensemble.batch_acoustic import _scalar_array
            cq = tuple(_scalar_array(state, value) for value in cq)
        return _single_launch(lambda: dycore._launch_slow_pgf(single, cfg, cq=cq), ("slow_pgf",))
    from woof.ensemble.batch_acoustic import prepare_moist_cq, _Bindings
    cqu, cqv, _cqw, use_cq = prepare_moist_cq(state, cfg) if cq is None else cq
    owners = _Bindings(state, cfg)
    replacements = {"p": pressure, "cqu": owners.owner(cqu)[0], "cqv": owners.owner(cqv)[0]}
    a = lambda name: _allocation(state, name)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    args = (a("ru_t"), a("rv_t"), a(pressure), a("pb"), a("al"), a("alt"),
            a("php"), a("phb"), a("mup"), a("mub2d"), a("c1h"), a("c2h"),
            a("rdnw"), a("fnm"), a("fnp"), state.cf1, state.cf2, state.cf3,
            np.int32(cfg.top_lid), state.cfn, state.cfn1, cqu, cqv, np.int32(use_cq),
            np.float32(rdx), np.float32(rdy), np.float32(0.5 * rdx), np.float32(0.5 * rdy),
            np.int32(dycore._boundary_x(cfg)), np.int32(dycore._boundary_y(cfg)),
            np.int32(len(_shape(state, "phb")) == 3), np.int32(nz), np.int32(ny), np.int32(nx))
    return _bound_launch(state, "slow_pgf", args, ((nz * (ny + 1) * (nx + 1) + 255) // 256,), bindings=replacements)


def prepare_slow_buoyancy(state, cfg=None):
    """One vertical pressure-gradient/dry-buoyancy launch for all members."""
    cfg, members = _context(state, cfg)
    pressure = _pressure_field(state)
    from woof.core import dycore
    if members == 1:
        single = _single_view(state)
        return _single_launch(lambda: dycore._launch_slow_buoyancy(single, cfg), ("slow_buoyancy",))
    a = lambda name: _allocation(state, name)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    mode = 0
    moisture = {name: "p" for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")}
    if cfg.moist:
        mode = 3 if getattr(state, "qh", None) is not None else (2 if getattr(state, "qi", None) is not None else 1)
        moisture.update({name: name for name in ("qv", "qc", "qr")})
        if mode >= 2:
            moisture["qi"] = "qi"
            if getattr(state, "qs", None) is None:
                absent = state.existing_scratch("moist_absent_mass")
                if absent is None:
                    raise BatchStateUnsupported("absent snow/graupel mass needs its admitted zero plane before moist buoyancy")
                absent.fill(np.float32(0))
                moisture.update(qs="scratch:moist_absent_mass", qg="scratch:moist_absent_mass")
            else:
                moisture.update(qs="qs", qg="qg")
        if mode == 3:
            moisture["qh"] = "qh"
    args = (a("rw_t"), a(pressure), a("pb"), a("mup"), a("mub2d"),
            *(a(moisture[name]) for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")),
            a("rdn"), a("rdnw"), a("c1f"), a("c2f"), a("msft"),
            np.int32(mode), np.int32(state.has_msf), np.int32(len(_shape(state, "phb")) == 3),
            np.int32(nz), np.int32(ny), np.int32(nx))
    return _bound_launch(state, "slow_buoyancy", args, ((nz * ny * nx + 255) // 256,), bindings={"p": pressure, **moisture})


def prepare_slow_geopotential(state, cfg=None, *, ww="scratch:rk_ww", add_vertical=True):
    """One existing fused geopotential RHS launch, with explicit Omega backing."""
    cfg, members = _context(state, cfg)
    from woof.core import dycore
    dycore._validate_geopotential_config(cfg, cfg.nx, cfg.ny)
    if _shape(state, ww) != state_array_shapes(cfg)["w"]:
        raise ValueError("geopotential Omega backing must have the original full-level shape")
    if isinstance(state, BatchedDomainState) and state.storage.specs[ww].ownership != "member":
        raise BatchStateUnsupported("geopotential Omega is a mutable stage flux and needs independent member backing")
    if members == 1:
        single, omega = _single_view(state), _allocation(state, ww, single=True)
        return _single_launch(lambda: dycore._launch_slow_geopotential(single, cfg, omega, add_vertical=add_vertical), ("slow_geopotential",))
    a = lambda name: _allocation(state, name)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    if (cfg.open_x or cfg.open_y) and not dycore._boundary_forced(cfg):
        # The scalar helper sends this configuration to the native open
        # kernel (890e2523a); the members must take the same entry or their
        # words differ from a single run's.
        args = (a("rph_t"), a(ww), a("w"), a("u"), a("v"), a("php"), a("phb"),
                a("mup"), a("mub2d"), a("rdnw"), a("fnm"), a("fnp"), a("c1f"), a("c2f"),
                state.cfn, state.cfn1, a("msft"), a("msfu"), a("msfv"), a("mup"), a("mup"),
                np.float32(0.25 / cfg.dx), np.float32(0.25 / cfg.dy), np.int32(state.has_msf),
                np.int32(cfg.open_x), np.int32(cfg.open_y), np.int32(0),
                np.int32(add_vertical), np.int32(cfg.h_sca_adv_order),
                np.int32(len(_shape(state, "phb")) == 3),
                np.int32(nz), np.int32(ny), np.int32(nx))
        return _bound_launch(state, "slow_geopotential_open", args, ((nz * ny * nx + 255) // 256,),
                             bindings={"ww": ww})
    args = (a("rph_t"), a(ww), a("w"), a("u"), a("v"), a("php"), a("phb"),
            a("mup"), a("mub2d"), a("rdnw"), a("fnm"), a("fnp"), a("c1f"), a("c2f"),
            state.cfn, state.cfn1, a("msft"), a("msfu"), a("msfv"),
            np.float32(0.25 * rdx), np.float32(0.25 * rdy), np.int32(state.has_msf),
            np.int32(dycore._boundary_x(cfg)), np.int32(dycore._boundary_y(cfg)),
            np.int32(dycore._boundary_forced(cfg)), np.int32(cfg.h_sca_adv_order),
            np.int32(add_vertical), np.int32(len(_shape(state, "phb")) == 3),
            np.int32(nz), np.int32(ny), np.int32(nx))
    return _bound_launch(state, "slow_geopotential", args, ((nz * ny * nx + 255) // 256,), bindings={"ww": ww})


def prepare_small_step_init(state, cfg=None, *, rk_step=1):
    """The original UV and column initialization entries, each batched once."""
    cfg, members = _context(state, cfg)
    from woof.core import dycore
    if members == 1:
        return _single_launch(dycore._prepare_small_step_init_launch(_single_view(state), cfg, rk_step),
                              ("small_step_init_uv", "small_step_init_column"))
    a = lambda name: _allocation(state, name)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    uv = tuple(a(name) for name in ("u_pp", "v_pp", "u0", "v0", "u", "v", "mup0", "mup", "mub2d", "c1h", "c2h", "msfu", "msfv"))
    if dycore.WRF_EXACT:
        uv = uv[:-1] + (a("msfv_inv"),)
    uv += (np.int32(state.has_msf), np.int32(dycore._boundary_x(cfg)), np.int32(dycore._boundary_y(cfg)),
           np.int32(nz), np.int32(ny), np.int32(nx))
    if dycore.WRF_EXACT:
        uv += (np.int32(rk_step),)
    column = tuple(a(name) for name in ("w_pp", "th_pp", "ph_pp", "mu_pp", "al_pp", "p_pp", "p_pp_old", "w0", "w", "thp0", "thp", "php0", "php", "p", "alt", "mup0", "mup", "mub2d", "thb", "c1h", "c2h", "c1f", "c2f", "rdnw", "msft"))
    column += (np.int32(state.has_msf), np.int32(len(_shape(state, "thb")) == 3), np.int32(nz), np.int32(ny), np.int32(nx))
    launches = (_bound_launch(state, "small_step_init_uv", uv, ((nz * (ny + 1) * (nx + 1) + 255) // 256,),
                                 bindings={"msfv": "msfv_inv"} if dycore.WRF_EXACT else None),
                _bound_launch(state, "small_step_init_column", column, ((ny * nx + 255) // 256,)))

    def launch():
        launches[0]()
        launches[1]()

    launch.numerical_entries = ("small_step_init_uv", "small_step_init_column")
    return launch


def prepare_small_step_finish(state, cfg=None, *, hdiab_dt=0.0):
    """The original dry UV and column finish entries, each batched once."""
    cfg, members = _context(state, cfg)
    if hdiab_dt and getattr(state, "h_diabatic", None) is None:
        raise BatchStateUnsupported("a nonzero heating removal interval needs its admitted member h_diabatic field")
    from woof.core import dycore
    if members == 1:
        return _single_launch(dycore._prepare_small_step_finish_launch(_single_view(state), cfg, hdiab_dt),
                              ("small_step_finish_uv", "small_step_finish_column"))
    a = lambda name: _allocation(state, name)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    uv = tuple(a(name) for name in ("u", "v", "u_pp", "v_pp", "mup", "mu_pp", "mub2d", "c1h", "c2h", "msfu", "msfv"))
    uv += (np.int32(state.has_msf), np.int32(dycore._boundary_x(cfg)), np.int32(dycore._boundary_y(cfg)), np.int32(nz), np.int32(ny), np.int32(nx))
    heating = "h_diabatic" if hdiab_dt else "p"
    column = tuple(a(name) for name in ("w", "thp", "php", "mup", "w_pp", "th_pp", "ph_pp", "mu_pp", "mub2d", "thb", "c1h", "c2h", "c1f", "c2f", "msft", heating))
    column += (np.float32(hdiab_dt), np.int32(bool(hdiab_dt)), np.int32(state.has_msf),
               np.int32(len(_shape(state, "thb")) == 3), np.int32(nz), np.int32(ny), np.int32(nx))
    launches = (_bound_launch(state, "small_step_finish_uv", uv, ((nz * (ny + 1) * (nx + 1) + 255) // 256,)),
                _bound_launch(state, "small_step_finish_column", column, ((ny * nx + 255) // 256,), bindings={"h_diabatic": heating}))

    def launch():
        launches[0]()
        launches[1]()

    launch.numerical_entries = ("small_step_finish_uv", "small_step_finish_column")
    return launch


def prepare_w_surface(state, cfg=None):
    """The original fused terrain surface-w entry, without an eager substitution."""
    cfg, members = _context(state, cfg)
    from woof.core import dycore
    if members == 1:
        single = _single_view(state)
        return _single_launch(lambda: dycore.set_w_surface(single, cfg), ("set_surface_w",))
    if (cfg.nz < 3 or min(cfg.ny, cfg.nx) < 2
            or isinstance(cfg.dx, np.generic) or isinstance(cfg.dy, np.generic)
            or any(not isinstance(value, np.float32) for value in (state.cf1, state.cf2, state.cf3))):
        raise BatchStateUnsupported("this surface-w configuration takes the scalar eager path; the fused batch entry would change its arithmetic or read missing lower levels")
    a = lambda name: _allocation(state, name)
    args = (a("u"), a("v"), a("ht"), a("msft"), a("w"),
            np.float32(state.cf1), np.float32(state.cf2), np.float32(state.cf3),
            np.float32(0.5 / cfg.dx), np.float32(0.5 / cfg.dy),
            np.int32(state.has_msf), np.int32(dycore._boundary_x(cfg)), np.int32(dycore._boundary_y(cfg)),
            np.int32(cfg.ny), np.int32(cfg.nx))
    return _bound_launch(state, "set_surface_w", args, ((cfg.ny * cfg.nx + 127) // 128,), threads=128)


def prepare_coriolis_curvature(state, cfg=None, *, ru="scratch:rk_ru", rv="scratch:rk_rv", mut="scratch:smag_mut"):
    """One existing Coriolis/curvature entry with already prepared total mass.

    Coupled momenta and total mass must have planned member-owned backing.
    This primitive does not calculate or allocate them, and does not choose
    whether a full RK driver should invoke the rotational tendency slot.
    """
    cfg, members = _context(state, cfg)
    from woof.core import dycore
    shapes = state_array_shapes(cfg)
    for name, expected in ((ru, shapes["u"]), (rv, shapes["v"]), (mut, shapes["mup"])):
        if _shape(state, name) != expected:
            raise ValueError("Coriolis coupled momentum/total mass shape differs from its original staggering")
        if isinstance(state, BatchedDomainState) and state.storage.specs[name].ownership != "member":
            raise BatchStateUnsupported("Coriolis coupled momentum and total mass require independent member backings")
    if members == 1:
        single = _single_view(state)
        r_u, r_v, mass = (_allocation(state, name, single=True) for name in (ru, rv, mut))

        def launch():
            dycore.launch_coriolis_curvature(
                r_u, r_v, single.u, single.v, single.w, mass,
                single.msft, single.msfu, single.msfv, single.f, single.e,
                single.c1f, single.c2f, single.fnm, single.fnp, cfg.dx, cfg.dy,
                single.ru_t, single.rv_t, single.rw_t, sina=single.sina, cosa=single.cosa,
                boundary_x=dycore._boundary_x(cfg), boundary_y=dycore._boundary_y(cfg))

        return _single_launch(launch, ("coriolis_curvature",))
    a = lambda name: _allocation(state, name)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    args = (a(ru), a(rv), a("u"), a("v"), a("w"), a(mut),
            a("msft"), a("msfu"), a("msfv"), a("f"), a("e"), a("sina"), a("cosa"),
            a("c1f"), a("c2f"), a("fnm"), a("fnp"), np.float32(1.0 / cfg.dx), np.float32(1.0 / cfg.dy),
            a("ru_t"), a("rv_t"), a("rw_t"), np.int32(dycore._boundary_x(cfg)), np.int32(dycore._boundary_y(cfg)),
            np.int32(nz), np.int32(ny), np.int32(nx))
    return _bound_launch(state, "coriolis_curvature", args, ((nz * (ny + 1) * (nx + 1) + 255) // 256,), bindings={"ru": ru, "rv": rv, "mut": mut})
