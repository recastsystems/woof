"""Device-resident Noah-MP surface classification and parameter selection.

The launch capacity is the grid size, never a device count. Each current land
cell owns its original row. Inactive rows borrow the first active column for
safe intermediate arithmetic and never scatter. Thus an in-place land/ice or
category change takes effect on the next call without a host decision.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

# Pinned MPTABLE USGS/MODIS NVEG=27/20, SOILPARM STAS=19.
# Price the complete rectangle, including categories the native routine skips.
MAX_VEGETATION_CLASSES = 27
MAX_SOIL_CLASSES = 19


def selected_options(cfg):
    return int(getattr(cfg,"noahmp_output",1))==3 or float(getattr(cfg,"soiltstep",0.0)) > 0 or any(int(getattr(cfg, name, default)) != default
               for name, default in (("opt_gla",1), ("opt_alb", 2), ("opt_stc", 1),
                                     ("opt_rsf", 1), ("opt_btr", 1), ("opt_snf", 1),
                                     ("dveg", 4), ("opt_inf", 1), ("opt_run", 3),
                                     ("opt_rad", 3), ("opt_frz", 1), ("opt_tbot", 2), ("opt_crs", 1), ("opt_sfc", 1), ("opt_soil", 1), ("opt_tdrn",0),("opt_irr",0),("opt_irrm",0),("opt_crop",0)))


def parameter_bytes_per_row(cfg):
    from woof.core.noahmp_column_slab import PARAMETER_INTS, slab_parameter_widths
    return 4 * (sum(slab_parameter_widths(opt_alb=cfg.opt_alb,
                                         opt_rsf=cfg.opt_rsf, opt_run=cfg.opt_run, opt_rad=cfg.opt_rad, opt_crs=cfg.opt_crs, opt_sfc=cfg.opt_sfc,dveg=cfg.dveg,opt_tdrn=cfg.opt_tdrn,opt_irr=cfg.opt_irr,opt_crop=cfg.opt_crop).values())
                + len(PARAMETER_INTS))


_MASK = ContextVar("noahmp_surface_mask", default=None)


@contextmanager
def active_columns(mask):
    token = _MASK.set(mask)
    try:
        yield
    finally:
        _MASK.reset(token)


def column_mask():
    return _MASK.get()


def active_probe(flag):
    mask = _MASK.get()
    return flag if mask is None else flag & mask


def parameter_pool(params, nsoil):
    """Upload the finite table identity space once, before model stepping.

    Uses the existing pinned TRANSFER_MP_PARAMETERS for table setup. Runtime
    selection, including changes to either category grid, is a device gather.
    Soil 14's unassigned FRZX is never invented: an active such row is refused.
    """
    import cupy as cp
    from woof.core.noahmp import SOILPARM_SECTION
    from woof.core.noahmp_column_slab import parameter_fields

    key = (cp.cuda.runtime.getDevice(), int(nsoil),
           params.land_options["opt_alb"], params.land_options["opt_rsf"],
           params.land_options.get("opt_run", 3), params.land_options.get("opt_rad", 3), params.land_options.get("opt_crs", 1), params.land_options.get("opt_sfc", 1),params.land_options.get("opt_soil",1),params.land_options.get("dveg",4),params.land_options.get("opt_tdrn",0),params.land_options.get("opt_irr",0),params.land_options.get("opt_crop",0))
    pools = params.__dict__.setdefault("_device_parameter_pools", {})
    if key in pools:
        return pools[key]
    ns = len(params.bundle.soil[SOILPARM_SECTION].rows)
    nv = params.n_vegetation_classes
    if nv > MAX_VEGETATION_CLASSES or ns > MAX_SOIL_CLASSES:
        raise ValueError("Noah-MP parameter tables exceed their admission bound")
    depth_soils=params.land_options.get("opt_soil",1)==2
    rows = [params.slab_row(vegtyp=v, soiltyp=((7,14,14,14) if s==14 else (s,) * nsoil),
                           slopetype=1, soilcolor=4, nsoil=nsoil)
            for v in range(1, nv + 1) for s in range(1, ns + 1)
            if (s != 14 or depth_soils) and params._veg.values["NROOT"][v-1] > 0]
    lookup = np.full((nv + 1, ns + 1), -1, dtype=np.int64)
    index = 0
    for v in range(1, nv + 1):
        for s in range(1, ns + 1):
            if (s != 14 or depth_soils) and params._veg.values["NROOT"][v-1] > 0:
                lookup[v, s] = index
                index += 1
    pool = parameter_fields(rows, np.arange(len(rows), dtype=np.int32),
                            opt_alb=key[2], opt_rsf=key[3], opt_run=key[4], opt_rad=key[5], opt_crs=key[6], opt_sfc=key[7],dveg=key[9],opt_tdrn=key[10],opt_irr=key[11],opt_crop=key[12])
    pools[key] = (pool, cp.asarray(lookup), nv, ns)
    return pools[key]


_CLASSIFIERS = {}


def _classify(params, fields, lookup, nv, ns, urban_on):
    """One launch for the current masks and native urban/soil remapping."""
    import cupy as cp
    identity = params.land_use
    key = (identity.isice, identity.isurban, identity.natural,
           tuple(identity.lcz), nv, ns, bool(urban_on))
    kernel = _CLASSIFIERS.get(key)
    if kernel is None:
        urban_test = " || ".join(f"rawveg == {int(v)}" for v in
                                  (identity.isurban, *identity.lcz))
        mapped_urban = identity.natural if urban_on else identity.isurban
        body = f"""
        urban = {urban_test};
        veg = urban ? {int(mapped_urban)} : (int)rawveg;
        soil = rawsoil == 14 && xice == 0.0f ? 7 : (int)rawsoil;
        sea_ice = xice >= threshold;
        water = (xland - 1.5f >= 0.0f) && !sea_ice;
        const bool land = !(sea_ice || water);
        glacier = land && veg == {int(identity.isice)};
        active = land && !glacier;
        const int cv = min(max(veg,1),{nv});
        const int cs = min(max(soil,1),{ns});
        mapped = lookup[cv * {ns+1} + cs];
        invalid = active && (veg < 1 || veg > {nv} || soil < 1 ||
                             soil > {ns} || mapped < 0);
        """
        kernel = cp.ElementwiseKernel(
            "float32 xland, float32 xice, V rawveg, S rawsoil, "
            "raw int64 lookup, float32 threshold",
            "bool sea_ice, bool water, bool glacier, bool active, "
            "int32 veg, int32 soil, int64 mapped, bool invalid, bool urban",
            body, "noahmp_surface_classify", options=("-std=c++17",))
        _CLASSIFIERS[key] = kernel
    return kernel(fields["xland"], fields["xice"], fields["ivgtyp"],
                  fields["isltyp"], lookup, np.float32(params.xice_threshold))


def layout(params, fields, *, nsoil, sf_urban_physics=0):
    import cupy as cp
    from woof.core import health_ledger

    pool, lookup, nv, ns = parameter_pool(params, nsoil)
    sea_ice, water, glacier, active, veg, soil, mapped, invalid, urban = _classify(
        params, fields, lookup, nv, ns, bool(sf_urban_physics))
    flags = (invalid.any().astype(cp.uint32)
             | ((glacier.any() & (not params.glacier_path)).astype(cp.uint32) << 1)).reshape(1)

    def refuse(bits):
        if bits & 1:
            raise ValueError("Noah-MP active column has an out-of-table category "
                             "or a native-unassigned parameter row")
        if bits & 2:
            from woof.core.noahmp_runtime import NoahmpGlacierColumnError
            raise NoahmpGlacierColumnError(
                "Noah-MP glacier path is disabled (glacier_path=False); "
                "the land solver cannot substitute for it")

    ledger = health_ledger.active()
    if ledger is None:
        refuse(int(flags.item()))
    else:
        ledger.record("noahmp.surface.device_identity", flags, refuse)
    n = int(active.size)
    linear = cp.arange(n, dtype=cp.int64)
    first = active.reshape(-1).argmax().astype(cp.int64)
    gather = cp.where(active.reshape(-1), linear, first)
    nx = int(active.shape[1])
    gather_j, gather_i = gather // nx, gather % nx
    row = cp.maximum(mapped, 0).reshape(-1)
    row = row[gather]
    return dict(sea_ice=sea_ice, glacier=glacier, vegtyp=veg, soiltyp=soil, urban=urban,
                land_j=linear // nx, land_i=linear % nx,
                gather_j=gather_j, gather_i=gather_i, active=active.reshape(-1),
                pool=pool, row=row,
                census=dict(land=active.sum(), water=water.sum(),
                            sea_ice=sea_ice.sum(), glacier=glacier.sum()))
