"""Independent radiation-spectrum construction and composition.

Existing paired adapters retain their construction and arithmetic. Mixed
selections execute only each selected spectrum and merge its own carriers.
"""
from __future__ import annotations


def transported_smoke_attr(cfg):
    """The state attribute of the transported chem row the legacy RRTMG
    shortwave reads as smoke, or None.

    Only ``simple_dir_fdb = 1`` (the HRRR v4 fork's direct smoke feedback)
    asks for it; the row is the one whose ``radiation.shortwave`` is
    the chem table's smoke layer-AOD coupling
    (woof.chem_table.SpeciesRow.feeds_shortwave_aod, the smoke set's row).  Every run without the switch
    builds the adapter it always did.  validate_chem_config has already
    refused zero or several such rows and every double count.
    """
    if not int(getattr(cfg, "simple_dir_fdb", 0) or 0):
        return None
    from woof.chem_table import load
    table = load(cfg)
    rows = [] if table is None else [
        row for row in table.transported if row.feeds_shortwave_aod]
    if len(rows) != 1:
        raise ValueError(
            "simple_dir_fdb=1 needs exactly one transported chem row feeding "
            f"the shortwave as smoke; found {[r.name for r in rows]}")
    return rows[0].state_attr


def radiation_adapters(adapter):
    """Yield distinct leaf adapters in execution order."""
    seen = set()
    def visit(current):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        components = getattr(current, "spectrum_adapters", None)
        if components is not None:
            for component in components:
                yield from visit(component)
        else:
            yield current
    return tuple(visit(adapter))


def modern_radiation_adapters(adapter):
    return tuple(item for item in radiation_adapters(adapter)
                 if type(item).__module__ == "woof.core.rrtmgp"
                 and type(item).__name__ == "RRTMGPRadiation")


def attach_modern_workspace(adapter, workspace):
    """Attach a shared CUDA workspace only to the engines that consume it."""
    if workspace is not None:
        for engine in modern_radiation_adapters(adapter):
            engine.column_chunk = workspace.column_chunk
            engine.chunk_workspace = workspace


def legacy_radiation_adapter(adapter):
    return next((item for item in radiation_adapters(adapter)
                 if type(item).__module__ == "woof.core.rrtmg_legacy"
                 and type(item).__name__ == "RRTMGLegacyRadiation"), None)


class ComposedRadiation:
    """Keep each spectrum's own heating, flux and restart authority."""
    @property
    def spectrum_adapters(self):
        return (self.longwave_adapter, self.shortwave_adapter)

    def __init__(self, start_time, latitude_deg, longitude_deg, *,
                 longwave_adapter=None, shortwave_adapter=None):
        if longwave_adapter is None and shortwave_adapter is None:
            raise ValueError("radiation composition needs an active spectrum")
        self.start_time = start_time
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.longwave_adapter = longwave_adapter
        self.shortwave_adapter = shortwave_adapter
        self.publishes_olr = bool(getattr(longwave_adapter, "publishes_olr", False))
        self.publishes_cldfra = any(
            bool(getattr(item, "publishes_cldfra", False))
            for item in (longwave_adapter, shortwave_adapter))
        self.glw_provenance = "scheme" if longwave_adapter is not None else "declared"

    @property
    def update_count(self):
        return max(getattr(item, "update_count", 0)
                   for item in radiation_adapters(self))

    @property
    def _o33d_grid(self):
        adapter = legacy_radiation_adapter(self)
        return None if adapter is None else adapter._o33d_grid

    @_o33d_grid.setter
    def _o33d_grid(self, value):
        adapter = legacy_radiation_adapter(self)
        if adapter is None:
            raise ValueError("this radiation composition has no retained ozone provider")
        adapter._o33d_grid = value

    @property
    def column_chunk(self):
        adapters = modern_radiation_adapters(self)
        return adapters[0].column_chunk if adapters else None

    @column_chunk.setter
    def column_chunk(self, value):
        for adapter in modern_radiation_adapters(self):
            adapter.column_chunk = value

    @property
    def chunk_workspace(self):
        adapters = modern_radiation_adapters(self)
        return getattr(adapters[0], "chunk_workspace", None) if adapters else None

    @chunk_workspace.setter
    def chunk_workspace(self, value):
        for adapter in modern_radiation_adapters(self):
            adapter.chunk_workspace = value

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from woof.core.physics import RadiationResult
        arguments = dict(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        lw = (self.longwave_adapter(**arguments)
              if self.longwave_adapter is not None else None)
        sw = (self.shortwave_adapter(**arguments)
              if self.shortwave_adapter is not None else None)
        zero = None
        if lw is None or sw is None:
            zero = cp.zeros_like(atmosphere["pressure"])
        if sw is None:
            # No shortwave spectrum: the driver still writes the radiation
            # call's COSZEN (radconst + calc_coszen at xtime + radt*0.5):
            # the solar generation's geometry, or under GPUWM_WRF_EXACT=1
            # with wrf_461 WRF 4.6.1's word for word (the namelist radt raw).
            from woof.core.wrf_radiation_clock import wrf_radiation_coszen
            minutes = cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes
            generation = getattr(cfg, "solar_generation", "wrf_480")
            from woof.wrf_exact import ENABLED as _strict
            if _strict and generation == "wrf_461":
                coszen = wrf_radiation_coszen(
                    self.start_time, float(state.elapsed_seconds), minutes,
                    self.latitude_deg, self.longitude_deg)
            else:
                from datetime import timedelta
                from woof.core.dudhia import wrf_solar_geometry
                from woof.core.physics import _physics_period_seconds
                valid_time = self.start_time + timedelta(seconds=float(state.elapsed_seconds))
                interval = _physics_period_seconds(minutes, cfg)
                from woof.core.solar import solar_period_seconds
                interval = solar_period_seconds(cfg, interval)
                coszen, _ = wrf_solar_geometry(valid_time, self.latitude_deg,
                    self.longitude_deg, hour_offset_seconds=0.5 * interval, generation=generation,
                    start_time=self.start_time, elapsed_seconds=state.elapsed_seconds)
            swdown = cp.zeros_like(fields["glw"])
            gsw = swdown
        else:
            swdown, gsw, coszen = sw.swdown, sw.gsw, sw.coszen
        return RadiationResult(
            boundary_fluxes={name:value for item in (lw,sw) if item is not None
                             for name,value in (getattr(item,"boundary_fluxes",None) or {}).items()},
            rthratenlw=lw.rthratenlw if lw is not None else zero,
            rthratensw=sw.rthratensw if sw is not None else zero,
            swdown=swdown, glw=lw.glw if lw is not None else fields["glw"],
            gsw=gsw, coszen=coszen, olr=lw.olr if lw is not None else None,
            # History consumes the native beam even without interpolation.
            swddir=getattr(sw, "swddir", None) if sw is not None else None,
            swddif=getattr(sw, "swddif", None) if sw is not None else None,
            aod5503d=getattr(sw,"aod5503d",None),
            aerosol2d=getattr(sw,"aerosol2d",None),
            aodtot=getattr(sw, "aodtot", None) if sw is not None else None,
            # Both spectra build the same driver CLDFRA where both compute
            # one; the longwave's is taken first, the shortwave's when the
            # longwave half computes none.
            cldfra=next((item.cldfra for item in (lw, sw)
                         if item is not None
                         and getattr(item, "cldfra", None) is not None),
                        None))

    def restart_identity(self):
        from woof.io.restart import _array_setup_identity
        return {
            "start_time": self.start_time.isoformat(),
            "latitude": _array_setup_identity(self.latitude_deg),
            "longitude": _array_setup_identity(self.longitude_deg),
            "algorithm": "independent-radiation-spectra-v1",
            "above_atmosphere_policy": "each selected spectrum retains its own column policy",
            "longwave": adapter_restart_identity(self.longwave_adapter),
            "shortwave": adapter_restart_identity(self.shortwave_adapter),
        }


def adapter_restart_identity(adapter):
    """Bind selected engines' setup without treating nested state as disposable."""
    if adapter is None:
        return None
    from woof.io.restart import (_array_setup_identity, _resolved_object_setup_identity,
                                  _rrtmgp_workspace_identity)
    identity = {
        "class": type(adapter).__module__ + "." + type(adapter).__qualname__,
        "start_time": adapter.start_time.isoformat(),
        "latitude": _array_setup_identity(adapter.latitude_deg),
        "longitude": _array_setup_identity(adapter.longitude_deg),
    }
    declared = getattr(adapter, "restart_identity", None)
    if declared is not None:
        identity["implementation"] = declared() if callable(declared) else declared
    elif modern_radiation_adapters(adapter):
        identity.update({
            "column_chunk": adapter.column_chunk,
            "validation_mode": adapter.validation_mode,
            "trace_gas_overrides": adapter.trace_gas_overrides,
            "trace_vmr": adapter.trace_vmr,
            "ozone_log_pressure": _array_setup_identity(adapter._ozone_logp),
            "ozone_vmr": _array_setup_identity(adapter._ozone_vmr),
            "tables": {name: _resolved_object_setup_identity(getattr(adapter, name), name)
                       for name in ("lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables")},
            "chunk_workspace": _rrtmgp_workspace_identity(getattr(adapter, "chunk_workspace", None)),
        })
    elif (type(adapter).__module__ == "woof.core.analytic_radiation"
          and type(adapter).__name__ == "AnalyticClearSkyRadiation"):
        from woof.core.analytic_radiation import (
            CLEAR_SKY_TRANSMISSIVITY, SOLAR_CONSTANT_WM2, STEFAN_BOLTZMANN)
        identity["proxy_constants"] = [CLEAR_SKY_TRANSMISSIVITY,
                                       SOLAR_CONSTANT_WM2, STEFAN_BOLTZMANN]
    else:
        raise ValueError("a custom radiation spectrum must declare restart_identity")
    if hasattr(adapter, "p_top"):
        identity["p_top"] = None if adapter.p_top is None else float(adapter.p_top)
    if isinstance(getattr(adapter, "longwave", None), bool):
        identity["spectra"] = {"lw": bool(adapter.longwave), "sw": bool(adapter.shortwave)}
    return identity


def trace_gas_override_consumption(cfg, overrides):
    """Selected absorption spectra per declared gas; an empty tuple is inactive.

    Legacy SW accepts an N2O profile for WRF interface compatibility but its
    optical-depth operator does not consume it. Keep that distinction visible
    without changing the requested value or inventing a coefficient consumer.
    """
    from woof.config import radiation_scheme_ids
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    from woof.core.trace_gases import (
        CLASSIC_GASES, LEGACY_LW_GASES, LEGACY_SW_GASES, LEGACY_SW_ABSORPTION_GASES,
        validate_trace_gas_overrides)
    declared = validate_trace_gas_overrides(overrides)
    if not declared:
        return {}
    lw, sw = radiation_scheme_ids(cfg)
    supported = set(CLASSIC_GASES) if lw == 1 else set()
    consumers = {"lw": set(supported), "sw": set()}
    if 4 in (lw, sw):
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            if lw == 4:
                supported.update(LEGACY_LW_GASES)
                consumers["lw"].update(LEGACY_LW_GASES)
            if sw == 4:
                supported.update(LEGACY_SW_GASES)
                consumers["sw"].update(LEGACY_SW_ABSORPTION_GASES)
        else:
            from woof.core.rrtmgp import coefficient_gas_names
            if lw == 4:
                supported.update(coefficient_gas_names("lw"))
                consumers["lw"].update(coefficient_gas_names("lw"))
            if sw == 4:
                supported.update(coefficient_gas_names("sw"))
                consumers["sw"].update(coefficient_gas_names("sw"))
    if not supported:
        return {gas: () for gas in declared}
    validate_trace_gas_overrides(declared, supported=supported,
                                 consumer="selected radiation spectra")
    return {gas: tuple(spectrum for spectrum, gases in consumers.items() if gas in gases)
            for gas in declared}


def trace_gas_override_status(cfg, overrides):
    """Summarize actual consumption without labelling inactive operands applied."""
    consumption = trace_gas_override_consumption(cfg, overrides)
    if not consumption:
        return "not_declared"
    applied = sum(bool(spectra) for spectra in consumption.values())
    return ("applied" if applied == len(consumption) else
            "partially_applied" if applied else "inactive")


def make_radiation(cfg, start_time, latitude_deg, longitude_deg, *,
                   p_top=None, column_chunk=None, trace_gas_overrides=None,
                   ozone_parent=None, ozone_routing=None):
    """Construct the requested spectra without substituting another selector.

    ``ozone_routing`` is the constructing route's own name for how a legacy
    RRTMG adapter obtains ozone (``woof.core.cam_ozone.OZONE_ROUTINGS``);
    ``None`` derives it from the construction.  Selectors other than legacy
    RRTMG carry no such routing and ignore it.
    """
    from woof.config import radiation_scheme_ids
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    lw, sw = radiation_scheme_ids(cfg)
    trace_gas_override_status(cfg, trace_gas_overrides)
    from woof.core.trace_gases import (
        CLASSIC_GASES, LEGACY_LW_GASES, LEGACY_SW_GASES, trace_gas_subset)
    if not (lw or sw):
        return None
    args = (start_time, latitude_deg, longitude_deg)
    def engine(selector, longwave, shortwave):
        if selector == 0:
            return None
        if selector == 1 and longwave:
            from woof.core.rrtm_lw import RRTMLongwaveRadiation
            return RRTMLongwaveRadiation(*args, p_top=p_top, icloud=cfg.icloud,
                                        column_chunk=column_chunk,
                                        ghg_input=int(getattr(cfg, "ghg_input", 0)),
                                        trace_gas_overrides=trace_gas_subset(
                                            trace_gas_overrides, CLASSIC_GASES) or None)
        if selector == 1:
            from woof.core.dudhia import DudhiaShortwaveRadiation
            return DudhiaShortwaveRadiation(*args, swrad_scat=cfg.swrad_scat, icloud=cfg.icloud)
        if selector == 5:
            return ColumnTableRadiation(*args, selector=selector, column_chunk=column_chunk)
        if selector == 90:
            from woof.core.analytic_radiation import AnalyticClearSkyRadiation
            return AnalyticClearSkyRadiation(*args)
        if selector != 4:
            raise ValueError(f"no radiation engine for selector {selector}")
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
            smoke_options = {}
            manifest = getattr(cfg, "rrtmg_smoke_manifest", "")
            if shortwave and manifest:
                from woof.core.rrtmg_smoke_manifest import (
                    BoundSmokeManifest, validate_smoke_manifest)
                descriptor = validate_smoke_manifest(manifest, start_time, cfg.nz)
                descriptor.require_vertical(cfg.eta_levels, cfg.hybrid_opt, cfg.etac, cfg.p_top)
                if (descriptor.times[-1] - descriptor.start_time).total_seconds() < cfg.run_seconds:
                    raise ValueError("prescribed smoke does not cover the complete requested run")
                provider = BoundSmokeManifest(manifest, start_time, latitude_deg, longitude_deg, cfg.nz)
                provider.require_vertical(cfg.eta_levels, cfg.hybrid_opt, cfg.etac, cfg.p_top)
                provider.require_coverage(cfg.run_seconds)
                smoke_options["smoke_provider"] = provider
            transported = transported_smoke_attr(cfg) if shortwave else None
            if transported is not None:
                if manifest:
                    raise ValueError(
                        "rrtmg_smoke_manifest with simple_dir_fdb=1: the "
                        "shortwave would count the plume twice (a prescribed "
                        "profile and the forecast's own smoke); drop one")
                smoke_options["transported_smoke"] = transported
            return RRTMGLegacyRadiation(*args, p_top=p_top, o3input=cfg.o3input,
                ghg_input=int(getattr(cfg, "ghg_input", 0)),
                ozone_parent=ozone_parent, ozone_routing=ozone_routing,
                longwave=longwave, shortwave=shortwave,
                aer_opt=int(getattr(cfg, "aer_opt", 0)) if shortwave else 0,
                aer3_type_source=(getattr(cfg, "rrtmg_aer3_type_source", "rural")
                                  if shortwave else "rural"),
                trace_gas_overrides=trace_gas_subset(trace_gas_overrides,
                    (LEGACY_LW_GASES if longwave else frozenset())
                    | (LEGACY_SW_GASES if shortwave else frozenset())) or None,
                **smoke_options)
        from woof.core.rrtmgp import RRTMGPRadiation, coefficient_gas_names
        selected_gases = trace_gas_overrides
        if int(getattr(cfg, "ghg_input", 0)) == 1:
            # RTE+RRTMGP holds its well-mixed gases for the run (its own
            # date policy, woof.core.rrtmgp.trace_gases); ghg_input=1 gives
            # it WRF's CAMtr values at the start time instead.  Explicit
            # overrides still win.  Not WRF's per-call refresh: that is the
            # legacy RRTMG arm, the one held to wrf.exe.
            from woof.core.camtr_gases import camtr_trace_gases
            day = start_time.timetuple().tm_yday - 1 + (
                start_time.hour + start_time.minute / 60.0
                + start_time.second / 3600.0) / 24.0
            camtr = {name: value for name, value in
                     camtr_trace_gases(start_time.year, day).items() if value > 0.0}
            selected_gases = {**camtr, **(trace_gas_overrides or {})}
        if selected_gases:
            supported = set()
            if longwave:
                supported.update(coefficient_gas_names("lw"))
            if shortwave:
                supported.update(coefficient_gas_names("sw"))
            selected_gases = trace_gas_subset(selected_gases, supported) or None
        options = dict(trace_gas_overrides=selected_gases,
                       longwave=longwave, shortwave=shortwave)
        if column_chunk is not None:
            options["column_chunk"] = column_chunk
        return RRTMGPRadiation(*args, **options)
    if (lw, sw) == (1, 1):
        from woof.core.rrtm_lw import RRTMDudhiaRadiation
        return RRTMDudhiaRadiation(*args, p_top=p_top,
                                   icloud=cfg.icloud, swrad_scat=cfg.swrad_scat,
                                   column_chunk=column_chunk,
                                   ghg_input=int(getattr(cfg, "ghg_input", 0)),
                                   trace_gas_overrides=trace_gas_overrides)
    if lw == sw:
        return engine(lw, True, True)
    if (lw, sw) == (0, 1):
        return engine(sw, False, True)
    # One device geography pair is shared by both CUDA leaves. Legacy
    # leaves retain their existing host geography conversion.
    import cupy as cp
    args = (start_time,
            cp.ascontiguousarray(cp.asarray(latitude_deg, dtype=cp.float32)),
            cp.ascontiguousarray(cp.asarray(longitude_deg, dtype=cp.float32)))
    return ComposedRadiation(*args, longwave_adapter=engine(lw, True, False),
                             shortwave_adapter=engine(sw, False, True))


class ColumnTableRadiation:
    """Run the registered paired column solver through the shared CUDA loader."""
    publishes_olr = True
    publishes_cldfra = True
    glw_provenance = 'scheme'
    diagnostics = ('SWDDNI','SWDDIR','SWDDIF','TLWDN','TLWUP','SLWDN','SLWUP',
                   'TSWDN','TSWUP','SSWDN','SSWUP')

    def __init__(self, start_time, latitude_deg, longitude_deg, *, selector=5,
                 column_chunk=None):
        import cupy as cp
        if selector != 5:
            raise ValueError(f'no table column solver for radiation {selector}')
        self.start_time = start_time
        self.latitude_deg = cp.asarray(latitude_deg, dtype=cp.float32)
        self.longitude_deg = cp.asarray(longitude_deg, dtype=cp.float32)
        self.column_chunk = max(1, int(column_chunk or 128))
        self.update_count = 0

    def restart_identity(self):
        return {'algorithm': 'wrf461-goddard-defined-visible-depth-v1',
                'above_atmosphere_policy': 'model-column-top-interface-zero-hpa',
                'radii_policy': 'gsfc-microns-or-wrf-use-mp-re-meter-conversion-otherwise-zero',
                'cloud_table': 'ad4bb1dd8446a2b00559ac425e21255a01b79a4cac7b5e3e693a3c98c222a1c8'}

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        import numpy as np
        from woof.core.kernels import get_kernel_int_defines
        from woof.core.radiation_tables import radiation_table_words
        from woof.core.physics import RadiationResult
        from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags, radconst, calc_coszen
        from woof.core.rrtmgp import cal_cldfra1
        from woof.core.swint import calendar_scalars
        nz, ny, nx = atmosphere['pressure'].shape
        if nz < 2:
            raise ValueError('interface-temperature extrapolation needs at least two layers')
        ncol=ny*nx
        grid=lambda x: cp.ascontiguousarray(cp.asarray(x,dtype=cp.float32)).reshape(-1,ncol)
        zeros=cp.zeros((nz,ncol),cp.float32)
        moist={name: grid(getattr(state,name)) if getattr(state,name,None) is not None else zeros
               for name in ('qv','qc','qr','qi','qs','qg','qh')}
        t,p=grid(atmosphere['temperature']),grid(atmosphere['pressure'])
        fqi,fqs=((True,True) if int(cfg.mp_physics)==7 else
                 legacy_cloud_fraction_flags(int(cfg.mp_physics)))
        cf=(cal_cldfra1(moist['qv'],moist['qc'],moist['qi'],moist['qs'],t,p,
                       f_qc=True,f_qi=fqi,f_qs=fqs) if cfg.icloud else zeros)
        from woof.core.mynn_radiation import merge_mynn_bl_clouds, mynn_bl_cloud_active, wrf_itimestep
        active_bl=mynn_bl_cloud_active(getattr(cfg,'bl_pbl_physics',0),getattr(cfg,'icloud_bl',0))
        if active_bl:
            moist['qc']=moist['qc'].copy();moist['qi']=moist['qi'].copy()
        moist['qc'],moist['qi'],cf=merge_mynn_bl_clouds(moist['qc'],moist['qi'],cf,
            qc_bl=grid(fields['qc_bl']) if active_bl else None,
            qi_bl=grid(fields['qi_bl']) if active_bl else None,
            cldfra_bl=grid(fields['cldfra_bl']) if active_bl else None,
            bl_pbl_physics=getattr(cfg,'bl_pbl_physics',0),icloud_bl=getattr(cfg,'icloud_bl',0),
            itimestep=wrf_itimestep(state.elapsed_seconds,cfg.dt) if active_bl else 1,
            bl_mynn_version=getattr(cfg,'bl_mynn_version','wrf_461'),
            temperature=t if active_bl else None)
        values,direct_gsfc=radiation_radius_provider_fields(
            state,int(cfg.mp_physics),getattr(cfg,'use_mp_re',1))
        radii=[zeros if value is None else grid(value) if direct_gsfc else
               grid(value)*np.float32(1.e-6)*np.float32(1.e6) for value in values]
        julian,xtime,gmt=calendar_scalars(self.start_time,state.elapsed_seconds)
        declin,solcon=radconst(julian)
        minutes=np.float32(cfg.radt if cfg.radt>0 else cfg.radt_minutes)
        cosine=cp.asarray(calc_coszen(julian,xtime+minutes*np.float32(.5),gmt,
            cp.asnumpy(self.latitude_deg).reshape(-1),cp.asnumpy(self.longitude_deg).reshape(-1),declin))
        defs=(('GDD_MAX_LEVELS',nz),)
        kernel=lambda name:get_kernel_int_defines('goddard',name,defs)
        tw=cp.empty((nz+1,ncol),cp.float32)
        count=(nz+1)*ncol
        kernel('goddard_interface_temperature')(((count+127)//128,),(128,),
            (t,grid(atmosphere['z_interface']),cp.asarray(state.fnm,dtype=cp.float32),
             cp.asarray(state.fnp,dtype=cp.float32),tw,np.int32(nz),np.int32(ncol)))
        heat_lw=cp.empty((nz,ncol),cp.float32);heat_sw=cp.empty_like(heat_lw)
        surface=cp.empty((15,ncol),cp.float32)
        table=radiation_table_words('goddard-cloud')
        for first in range(0,ncol,self.column_chunk):
            last=min(first+self.column_chunk,ncol);nc=last-first
            pack=lambda value:cp.ascontiguousarray(value[...,first:last]).reshape(-1)
            # WRF phy_prep passes (1+qv)/ALT as rho_phy to goddardrad.
            rho=grid(atmosphere["rho"])
            raw=cp.concatenate([pack(grid(atmosphere['p_interface'])),pack(t),pack(p),
                pack(moist['qv']),pack(cf),pack(grid(atmosphere['dz'])),pack(rho),
                *[pack(moist['q'+s]) for s in ('i','c','r','s','g','h')],
                *[pack(r) for r in radii],pack(self.latitude_deg.reshape(1,-1)),
                pack(cosine.reshape(1,-1)),pack(tw[:1]),
                *[pack(grid(fields[name])) for name in ('tsk','emiss','albedo')],
                pack(grid(atmosphere['exner']))])
            band=cp.empty(((5*nz+15)*nc,),cp.float32)
            flags=cp.empty((3*nc,),cp.int32);dz=cp.empty((nz*nc,),cp.float32)
            q=cp.empty((6*nz*nc,),cp.float32);re=cp.empty_like(q)
            launch=(((nc+31)//32,),(32,))
            kernel('goddard_prepare_columns')(*launch,(raw,band,flags,dz,q,re,
                np.int32(nz),np.int32(nc),np.int32(int(julian)+1)))
            optics=cp.empty((63*nz*nc,),cp.float32)
            kernel('goddard_cloud_columns')(((nz*nc+127)//128,),(128,),
                (dz,q,re,table,optics,np.int32(nz),np.int32(nc)))
            flux=cp.empty(((6*(nz+1)+4)*nc,),cp.float32)
            kernel('goddard_bands_columns')(*launch,(band,flags,optics,flux,np.int32(nz),np.int32(nc)))
            result=cp.empty(((2*nz+15)*nc,),cp.float32)
            kernel('goddard_publish_columns')(*launch,(band,flux,raw,result,
                np.int32(nz),np.int32(nc),np.float32(solcon)))
            result=result.reshape(2*nz+15,nc)
            heat_lw[:,first:last]=result[:nz];heat_sw[:,first:last]=result[nz:2*nz]
            surface[:,first:last]=result[2*nz:]
        self.update_count+=1
        diagnostics={name:surface[index].reshape(ny,nx) for name,index in
            zip(self.diagnostics,(4,5,6,7,8,9,10,11,12,13,14))}
        return RadiationResult(rthratenlw=heat_lw.reshape(nz,ny,nx),
            rthratensw=heat_sw.reshape(nz,ny,nx),swdown=surface[0].reshape(ny,nx),
            glw=surface[1].reshape(ny,nx),gsw=surface[2].reshape(ny,nx),
            coszen=cosine.reshape(ny,nx),olr=surface[3].reshape(ny,nx),
            swddir=diagnostics['SWDDIR'],swddif=diagnostics['SWDDIF'],
            cldfra=cf.reshape(nz,ny,nx),flux_diagnostics=diagnostics)


def radiation_radius_provider_fields(state, mp_physics, use_mp_re=1):
    """Resolve WRF's six GSFC radius carriers in ice/cloud/rain/snow/graupel/hail order."""
    if int(mp_physics)==7:
        names=tuple('re_'+species+'_gsfc' for species in
                    ('ice','cloud','rain','snow','graupel','hail'))
        values=tuple(getattr(state,name,None) for name in names)
        for name,value in zip(names,values):
            if value is None:
                raise ValueError(f'mp=7 radiation requires state.{name}; missing cloud optics radius')
        return values,True
    from woof.core.rrtmg_legacy import legacy_scheme_has_req
    hc,hi,hs=legacy_scheme_has_req(int(mp_physics),use_mp_re)
    names=(('effi',hi),('effc',hc),(None,0),('effs',hs),(None,0),(None,0))
    values=[]
    for name,declared in names:
        value=getattr(state,name,None) if name and declared else None
        if name and declared and value is None:
            raise ValueError(f'microphysics declares {name} but supplies no effective radius')
        values.append(value)
    # With no WRF radius provider, the GSFC Registry carriers stay zero.
    # The cloud LUT owns its documented lower-radius clipping, including zero.
    return tuple(values),False
