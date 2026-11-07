"""Resident native Noah-MP glacier driver; fixed grid launches and deferred errors.

Source: WRF-lineage v4 noahmp 848f54ad, module_sf_noahmpdrv.F. The
column is the already retained glacier transcription. No v5 source is used.
"""
import numpy as np

PACK_FIELDS = ('temperature', 'p_interface', 'u', 'v', 'qv', 'dz', 'cosz', 'swdown', 'rainbl', 'glw', 'tmn', 'qsnowxy', 'sneqvoxy', 'alboldxy', 'cmxy', 'chxy', 'snow', 'snowh', 'tgxy', 'taussxy', 'qsfc', 'snicexy', 'snliqxy', 'smois', 'sh2o', 'zsnsoxy', 'tsnoxy', 'tslb')
OUTPUT_FIELDS = ('sagxy', 'irgxy', 'shgxy', 'evgxy', 'ghvxy', 'pahgxy', 'irbxy', 'shbxy', 'evbxy', 'ghbxy', 'pahbxy', 'savxy', 'ircxy', 'shcxy', 'evcxy', 'trxy', 'pahvxy', 'qintsxy', 'qintrxy', 'qdripsxy', 'qdriprxy', 'qthrosxy', 'qthrorxy', 'qsnsubxy', 'qsnfroxy', 'qsubcxy', 'qfrocxy', 'qevacxy', 'qdewcxy', 'qfrzcxy', 'qmeltcxy', 'rainlsm', 'snowlsm')
WRITE_FIELDS = ('tsk', 'hfx', 'grdflx', 'snow', 'snowh', 'emiss', 'qsfc', 'tgxy', 'cmxy', 'chxy', 'sneqvoxy', 'alboldxy', 'qsnowxy', 'taussxy', 't2mbxy', 'tradxy', 'edirxy', 'fsaxy', 'firaxy', 'chbxy', 'fpicexy', 'qsnbotxy', 'qmeltxy', 'eflxbxy', 'qfx', 'lh', 'smstav', 'smstot', 'canwat', 'canliqxy', 'canicexy', 'fvegxy', 'ecanxy', 'etranxy', 'rs', 'canhsxy', 'tvxy', 'eahxy', 'tahxy', 'fwetxy', 'wslakexy', 'waxy', 'lai', 'xsaixy', 't2mvxy', 'rssunxy', 'rsshaxy', 'chvxy', 'sfcrunoff', 'udrunoff', 'albedo', 'snowc', 'smois', 'sh2o', 'tslb', 'tsnoxy', 'zsnsoxy', 'snicexy', 'snliqxy', 'acsnow', 'acsnom', 'pondingxy', 'z0', 'znt', 'q2mvxy', 'q2mbxy', 'runsfxy', 'runsbxy', 'soldrain', 'soilenergy', 'snowenergy')


def run_glacier_slab(fields, atmosphere, active, *, coszen, zsoil, dt,
                     opt_alb=2, opt_gla=1, opt_stc=1, opt_tbot=2, snicar_options=None, noahmp_snow_age="wrf_470"):
    import cupy as cp
    from woof.core import health_ledger
    from woof.core.noahmp_glacier_gpu import glacier_module
    from woof.core.noahmp_glacier import GlacierBalanceError
    n = int(active.size)
    shape = fields["tsk"].shape
    if active.dtype != cp.bool_ or active.shape != shape:
        raise ValueError("glacier active mask must match the resident surface grid")
    source = dict(fields)
    source.update({name: atmosphere[name] for name in
                   ("temperature", "p_interface", "u", "v", "qv", "dz")})
    source["cosz"] = coszen
    # Layers, including the first two interfaces, are contiguous planes.
    arrays = tuple(source[name] for name in PACK_FIELDS)
    writes = tuple(fields[name] if name != "soldrain" or name in fields
                   else fields["runsbxy"] for name in WRITE_FIELDS)
    if (any(value.dtype != cp.float32 or not value.flags.c_contiguous
            or value.shape[-2:] != shape for value in (*arrays, *writes))
        or fields["isnowxy"].dtype != cp.int32
        or not fields["isnowxy"].flags.c_contiguous):
        raise ValueError("glacier driver requires contiguous resident f32 planes and i32 ISNOW")
    # No packed grid buffers or per-field scatter launches are needed.
    # The current native glacier mask gates the fused thread's stack rows.
    snow = None
    if opt_alb == 3:
        from woof.core.snow_papers import evaluate, _module
        # Layered snow over a refracting, scattering glacier-ice stack.
        packed = cp.empty((n, 56), dtype=cp.float32)
        from woof.core.snow_papers import STATE_FIELD, STATE_WORDS
        layers = cp.empty((n,3,4),dtype=cp.float32)
        state = cp.empty((n,STATE_WORDS),dtype=cp.float32)
        ice_layers = cp.empty((n,4,3),dtype=cp.float32)
        if n:
            _module(cp.cuda.Device().id).get_function("snow_glacier_pack")(
                ((n+255)//256,), (256,),
                (active, coszen, *(fields[name] for name in
                    ("tgxy", "snowh", "sneqvoxy", "snow", "qsnowxy", "alboldxy",
                     "taussxy", "swdown", "snicexy", "snliqxy", "zsnsoxy", "tsnoxy", STATE_FIELD)),
                 zsoil, packed, layers, state, ice_layers, np.int32(n), np.float32(dt),
                 np.float32((snicar_options or {}).get("snicar_ice_density",900.)),
                 np.float32((snicar_options or {}).get("snicar_bubble_radius_um",1000.))))
        snow, carried, absorption = evaluate(packed, cp.ones(n,dtype=cp.int32),
            layers=layers, state=state, options=snicar_options,
            glacier=active, ice_layers=ice_layers,
            air_temperature=atmosphere["temperature"][0].ravel(), return_fluxes=True)
        heating = cp.ascontiguousarray(cp.sum(absorption*packed[:,None,52:56],axis=2))
    errors = cp.empty(n, dtype=cp.int32)
    source.update({name: fields[name] if name != "soldrain" or name in fields
                   else fields["runsbxy"] for name in WRITE_FIELDS})
    names = tuple(dict.fromkeys((*PACK_FIELDS, *WRITE_FIELDS)))
    module = glacier_module(noahmp_snow_age=noahmp_snow_age, selected_options=True)
    module.get_function("noahmp_glacier_driver_resident")(((n + 63)//64,), (64,),
        (*(source[name] for name in names), fields["isnowxy"], active, zsoil,
         np.float32(dt), np.int32(n), np.int32("soldrain" in fields),
         np.int32(opt_alb), np.int32(opt_gla), np.int32(opt_stc), np.int32(opt_tbot),
         *(fields.get(name,fields["tsk"]) for name in OUTPUT_FIELDS),
         np.int32("acsagv" in fields),
         snow if snow is not None else np.uint64(0),
         heating if snow is not None else np.uint64(0),
         carried if snow is not None else np.uint64(0), errors))
    if opt_alb == 3:
        fields[STATE_FIELD][...] = cp.where(active[None,:,:],
            carried.T.reshape(STATE_WORDS,*shape), fields[STATE_FIELD])
    error = cp.max(errors).astype(cp.uint32).reshape(1)
    def check(code):
        if code:
            raise GlacierBalanceError(f"NOAHMP_GLACIER native balance gate code {code}")
    ledger = health_ledger.active()
    if ledger is None:
        check(int(error.item()))
    else:
        ledger.record("noahmp.glacier.balance", error, check)
