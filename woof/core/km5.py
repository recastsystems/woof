"""WRF 4.6.1 SMS-3DTKE call order and device workspaces.

The closure, implicit column solves and horizontal stresses execute on the
device. Native capture replay is in tools/km5_wrf461_capture.
"""
import numpy as np

from woof.core.kernels import get_kernel, get_kernel_int_defines


def compute(state, cfg, km, kh, specs, *, time_t, physics_tendencies=None):
    from woof.core import dycore as d
    nz, ny, nx = state.p.shape
    mass = (nz, ny, nx)
    full = (nz + 1, ny, nx)
    f, integer = np.float32, np.int32
    bx, by = d._boundary_x(cfg), d._boundary_y(cfg)
    suffix = "0" if time_t else ""
    u, v, w, thp, tke = (getattr(state, x + suffix)
                          for x in ("u", "v", "w", "thp", "tke"))
    a = {}

    def work(name, shape=full, dtype=None):
        a[name] = state.scratch(shape, "km5_" + name, dtype=dtype)
        return a[name]

    for name in ("theta", "rho", "up", "vp", "rdz", "rdzw", "d33",
                 "d13", "d23", "p8w", "t8w", "zfull", "grad", "solveq", "solved"):
        work(name)
    work("zx", (nz + 1, ny, nx + 1))
    work("zy", (nz + 1, ny + 1, nx))
    for name in ("d11", "d22", "d12", "bn2"):
        work(name, mass)
    for name in ("nlflux", "dlk", "elmin", "l_diss", "xkmv_meso"):
        a[name] = getattr(state, name)
    # Native nonlocal_flux leaves the extra top plane cold-zero. Shared
    # domain workspaces must restore it before tke_rhs reads k + 1.
    a["nlflux"][-1].fill(0)
    a["gamu"], a["gamv"] = state.gamu, state.gamv
    fields = state.physics.fields if state.physics is not None else {}
    for name in ("ust", "hfx", "qfx", "br", "rmol", "wspd"):
        a[name] = fields.get(name)
        if a[name] is None:
            # module_physics_init.F cold-starts friction velocity at 1.e-4.
            work(name, (ny, nx)).fill(1.0e-4 if name == "ust" else 0)
    # USTM is WRF's surface friction velocity at the diffusion boundary.
    a["ustm"] = fields.get("ustm", a["ust"])
    a["hpbl"] = fields.get("pblh")
    if a["hpbl"] is None:
        work("hpbl", (ny, nx))
    a["kpbl"] = fields.get("kpbl")
    if a["kpbl"] is None:
        work("kpbl", (ny, nx), np.int32)
    qv = getattr(state, "qv" + suffix)
    a["qv"] = qv if qv is not None else work("dry_qv", mass)
    if qv is None:
        a["qv"].fill(0)
    # The arena can be shared by domains. A zero allocated on first use
    # is not a constant after another domain has borrowed its backing.
    base = work("zero_profile", (nz,))
    base.fill(0)
    a.update(tke=tke, mu=work("mu", (ny, nx)), kmh=km, khh=kh,
             kmv=state.scratch(mass, "smag_kmv"),
             khv=state.scratch(mass, "smag_khv"))
    a["mu"][...] = state.mub2d + getattr(state, "mup" + suffix)
    common = d._wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [integer(nz), integer(ny), integer(nx),
            integer(state.phb.ndim == 3), integer(bx), integer(by)]
    grid = ((nx + 127) // 128, ny, nz)
    geometry = (get_kernel_int_defines(
        "km5_geometry", "km5_geometry", (("GPUWM_USE_THETA_M", 1),))
        if cfg.use_theta_m else get_kernel("km5_geometry", "km5_geometry"))
    geometry(
        (grid[0], ny, nz + 1), (128, 1, 1),
        tuple(common + [thp, state.thb, integer(state.thb.ndim == 3), state.p]
              + [a[x] for x in ("theta", "rho", "up", "vp", "rdz", "rdzw",
                                "zx", "zy", "d33", "d13", "d23", "p8w", "t8w")]
              + dims))
    get_kernel("smag2d", "wrf_smag_deform")(
        grid, (128, 1, 1), tuple(common + [a[x] for x in ("d11", "d22", "d12")] + dims))
    d.launch_wrf_calc_n2(state, cfg, a["bn2"], time_t=time_t)

    # WRF seeds the held tendencies with physics before implicit mixing.
    seed = {}
    if d.WRF_EXACT and physics_tendencies is not None:
        d._fold_physics_wrf(state, specs, physics_tendencies)
        seed = {"smag_ru": physics_tendencies.ru,
                "smag_rv": physics_tendencies.rv,
                "smag_rw": physics_tendencies.rw,
                "smag_rth": physics_tendencies.rtheta}
        for row in specs:
            slot = row[5]
            if slot not in seed:
                seed[slot] = physics_tendencies.scalar_for(slot[len("smag_r"):])

    def call(name, args, *, cells=False, stagger=""):
        bounds = [nz, ny, nx, int(bx), nx - int(bx and stagger != "x"),
                  int(by), ny - int(by and stagger != "y")]
        get_kernel("km5", name)((grid[0], ny, nz if cells else 1), (128, 1, 1),
            tuple(a[x] if isinstance(x, str) else x for x in args)
            + tuple(integer(x) for x in bounds))

    maps = [state.msft, state.msft]
    call("km5_heights_gradient", ["rdzw", "rdz", "theta", "p8w", "t8w", "zfull", "grad"])
    call("km5_nonlocal", ["zfull", "theta", "up", "vp", "qv", "rho", "ust", "hfx", "br"]
         + maps + ["wspd", "nlflux", "gamu", "gamv", "hpbl", "kpbl", f(cfg.dx), f(cfg.dy),
                   f(461.6) / f(287.0) - f(1), f(0.4), integer(cfg.isfflx), f(cfg.tke_heat_flux)])
    call("km5_free_length", ["zfull", "rdzw", "theta", "tke", "elmin"], cells=True)
    call("km5_meso_length", ["zfull", "rdzw", "grad", "tke", "qv", "rho", "rmol", "hpbl",
         "elmin", "hfx", "dlk", integer(cfg.isfflx), f(cfg.tke_heat_flux), f(0.4)])
    call("km5_coefficients", ["theta", "tke", "grad", "rdzw", "d11", "d22", "d12", "zx", "zy"]
         + maps + ["hpbl", "dlk", "kmh", "khh", "kmv", "khv", "xkmv_meso", f(cfg.dx), f(cfg.dy),
                   f(cfg.c_s), f(cfg.c_k), f(1 / 3), f(d._tke_seed(cfg)),
                   integer(nx + 1), integer(ny), integer(nx), integer(ny + 1)], cells=True)
    if bx or by:
        # xkmv_meso has the same unwritten, cold-zero active ring as Km.
        # Its implicit momentum stencil reads that ring at i-1 / j-1.
        for pair in ((km, kh), (a["kmv"], a["khv"]),
                     (a["xkmv_meso"], a["xkmv_meso"])):
            get_kernel("smag2d", "wrf_smag_km_bc")(grid, (128, 1, 1),
                pair + tuple(integer(x) for x in (nz, ny, nx, bx, by)))
    rtke = state.scratch(mass, "smag_rtke")
    strides = [integer(u.shape[2]), integer(u.shape[1]), integer(v.shape[2]), integer(v.shape[1])]
    call("km5_tke_source", ["theta", "tke", "bn2", "d11", "d22", "d33", "d12", "d13", "d23",
         u, v, "kmh", "kmv", "khv", "mu", state.c1h, state.c2h, "rdzw"] + maps
         + ["ustm", "hfx", "qv", "rho", "nlflux", "hpbl", "dlk", "l_diss", rtke,
            f(cfg.dx), f(cfg.dy), f(cfg.dt), f(cfg.c_k), f(cfg.tke_drag_coefficient),
            f(cfg.tke_heat_flux), integer(cfg.isfflx)] + strides, cells=True)
    for component, slot, stag in ((0, "smag_ru", "x"), (1, "smag_rv", "y")):
        call("km5_implicit_momentum", [u, v, state._scratch[slot], "kmv", "xkmv_meso", "rho",
             "rdz", "mu", "ustm", "gamu" if component == 0 else "gamv", state.fnm, state.fnp,
             state.dnw, state.c1h, state.c2h, "solveq", "solved", f(cfg.dt), integer(component)]
             + strides, stagger=stag)
    call("km5_implicit_w", [w, state._scratch["smag_rw"], "kmh", "rho", "rdzw", state.dn,
         "mu", state.c1f, state.c2f, "solveq", "solved", f(cfg.dt)])
    scalar_rows = [(row[0], state._scratch[row[5]],
                    2 if row[5] == "smag_rth" else 1 if row[5] == "smag_rqv" else 0,
                    row[5]) for row in specs if row[6] == ""
                   and row[5] not in {"smag_r" + name for name in
                       ("nr", "ns", "ng", "nh", "qnr", "qns", "qng", "qnh", "qt", "qvolg")} ]
    scalar_rows.append((tke, rtke, 3, "smag_rtke"))
    for var, target, kind, slot in scalar_rows:
        # Theta's additive constant cancels in the column solve. The
        # existing reference basis is used for non-full-field diffusion.
        if kind == 2 and not d.WRF_EXACT:
            var = work("theta_mix", mass)
            if cfg.use_theta_m:
                # Geometry uses dry theta; diffusion transports theta_m.
                thermal_base = (state.thb.reshape((-1, 1, 1))
                                if state.thb.ndim == 1 else state.thb)
                var[...] = thp + thermal_base
                var -= f(300)
            else:
                var[...] = a["theta"][:nz] - f(300)
        # The shared native-input gate verifies the consumed WRF base
        # profiles are zero. This is independent of thb's EOS reference.
        subtract = not cfg.mix_full_fields and kind in (1, 2)
        call("km5_implicit_scalar", [var, target, base, "kmv" if kind == 3 else "khv", "rho", "mu",
             state.c1h, state.c2h, state.fnm, state.fnp, state.dnw, "rdz", "nlflux", "l_diss",
             "qv", "theta", "hfx", "qfx", "solveq", "solved", f(cfg.dt), f(461.6) / f(287.0),
             integer(kind), integer(subtract), integer(cfg.use_theta_m)])
    deformation = tuple(a[x] for x in ("d11", "d22", "d12"))
    exempt = d.chem_mix2_exempt_slots(state, cfg)
    for var, _, diffusivity, _, _, slot, stag in specs:
        if slot not in exempt:
            d.launch_wrf_smag2d_hd(state, cfg, var, diffusivity, state._scratch[slot],
                stagger=stag, time_t=time_t, full_theta=slot == "smag_rth", deformation=deformation)
        if seed.get(slot) is None:
            d._zero_open_strips(state._scratch[slot], cfg, 1)
        else:
            d._restore_open_strips(state._scratch[slot], seed[slot], cfg, 1)
    if cfg.km_opt == 5:  # WRF km=5 ignores tke_mix2_off (:3020).
        tmp = state.scratch(mass, "smag_tke_tmp")
        if d.WRF_EXACT:
            # horizontal_diffusion_s retains the original tendency, adds
            # diffusion, then doubles that rounded difference (:3988).
            tmp[...] = rtke
            d.launch_wrf_smag2d_hd(state, cfg, tke, km, rtke,
                                  stagger="", time_t=time_t)
            rtke[...] = tmp + f(2) * (rtke - tmp)
        else:
            tmp[...] = 0
            d.launch_wrf_smag2d_hd(state, cfg, tke, km, tmp,
                                  stagger="", time_t=time_t)
            d._zero_open_strips(tmp, cfg, 1)
            rtke += 2 * tmp
