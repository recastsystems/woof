"""GPU column interface to public WRF 4.7.1 Farms_driver.

All arrays remain on the device. The last model level is excluded from
the path integrals, exactly as Integrate_1var/Integrate_2var do in WRF.
"""
from __future__ import annotations
import numpy as np

FARMS_FIELDS = ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2", "swddnic2")


def farms_surface_columns(p8w, rho, dz, albedo, ssa, asymmetry, aod550,
                          angstrom, coszen, qv, qi, qs, qc,
                          re_cloud, re_ice, re_snow, cldfra, *,
                          julian, has_reqc=0, has_reqi=0, has_reqs=0, out=None):
    """Six surface fluxes for (nlev,ncol) columns, radii in metres.

    Inputs must be contiguous float32 CuPy buffers. p8w can include the
    final interface; its surface row is the only pressure row FARMS reads.
    The output can be a persistent driver allocation of shape (6,ncol).
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    if len(qv.shape) != 2 or qv.shape[0] < 2:
        raise ValueError("FARMS needs (nlev,ncol), at least two levels")
    nz, nc = qv.shape
    volume = (rho, dz, qv, qi, qs, qc, re_cloud, re_ice, re_snow, cldfra)
    surface = (albedo, ssa, asymmetry, aod550, angstrom, coszen)
    for value in (p8w, *volume, *surface):
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                or not value.flags.c_contiguous):
            raise TypeError("FARMS inputs must be contiguous float32 device arrays")
    if any(value.shape != (nz, nc) for value in volume):
        raise ValueError("FARMS volume buffers must match qv.shape")
    if p8w.shape not in ((nz, nc), (nz + 1, nc)):
        raise ValueError("FARMS interface pressure has the wrong shape")
    if any(value.shape != (nc,) for value in surface):
        raise ValueError("FARMS surface buffers must have shape (ncol,)")
    flags = (has_reqc, has_reqi, has_reqs)
    if any(value not in (0, 1) for value in flags):
        raise ValueError("FARMS has_re* flags must be 0 or 1")
    if out is None:
        out = cp.empty((6, nc), dtype=cp.float32)
    if (not isinstance(out, cp.ndarray) or out.shape != (6, nc)
            or out.dtype != cp.float32 or not out.flags.c_contiguous):
        raise TypeError("FARMS output must be contiguous float32 (6,ncol)")
    from woof.wrf_exact import ENABLED
    entry = "farms_column" if ENABLED else "farms_column_fast"
    grid = ((nc + 63) // 64,) if ENABLED else ((nc + 31) // 32,)
    block = (64,) if ENABLED else (32, 8)
    get_kernel("farms", entry)(
        grid, block,
        (np.int32(nc), np.int32(nz - 1), *(np.int32(value) for value in flags),
         np.float32(julian), p8w, rho, dz, albedo, ssa, asymmetry, aod550,
         angstrom, coszen, qv, qi, qs, qc, re_cloud, re_ice, re_snow, cldfra, out))
    return out


from woof.core.swint import ShortwaveInterpolation


class FastSurfaceShortwave(ShortwaveInterpolation):
    """FARMS on every Solar_step, with this domain's transported geography.

    Fluxes live in driver.fields for history, checkpoint, nest transport
    and each ensemble member. Only cosine scratch is rebuilt on restart.
    """
    def every_step(self, driver, atmosphere, state, cfg):
        import cupy as cp
        from woof.core.rrtmg_legacy import legacy_scheme_has_req, legacy_cloud_fraction_flags
        from woof.core.swint import calendar_scalars
        nz, ny, nx = atmosphere['qv'].shape
        nc = ny * nx
        column = lambda value: cp.ascontiguousarray(value, dtype=cp.float32).reshape(-1, nc)
        surface = lambda value: cp.ascontiguousarray(value, dtype=cp.float32).reshape(nc)
        cf = cp.zeros((nz,ny,nx), cp.float32)
        qc, qi = atmosphere['qc'], atmosphere['qi']
        diagnosed = False
        if int(cfg.icloud) == 3 or cfg.cu_rad_feedback:
            from woof.core.cloud_fraction import prepare_radiation_clouds
            view = prepare_radiation_clouds(atmosphere, driver.fields, state, cfg)
            atmosphere = view.atmosphere
            qc, qi = atmosphere['qc'], atmosphere['qi']
            if view.fraction is not None:
                cf, diagnosed = view.fraction, True
        if not diagnosed and int(cfg.icloud) == 2:
            from woof.core.rrtmgp import cal_cldfra2
            fqi, _ = legacy_cloud_fraction_flags(int(cfg.mp_physics))
            cf = cal_cldfra2(column(qc), column(qi),
                f_qc=int(cfg.mp_physics)!=0, f_qi=fqi).reshape(nz,ny,nx)
        if int(cfg.icloud) == 1:
            from woof.core.uwpbl import uwpbl_radiation_cloud_fraction
            from woof.core.mynn_radiation import merge_mynn_bl_clouds, mynn_bl_cloud_active, wrf_itimestep
            if not diagnosed:
                if int(cfg.mp_physics) == 95:
                    # Eta carries its saturation ice weight as F_ICE_PHY.
                    uwpbl_radiation_cloud_fraction(atmosphere, cfg, cf, state=state)
                else:
                    from woof.core.kernels import get_kernel
                    fqi, fqs = legacy_cloud_fraction_flags(int(cfg.mp_physics))
                    values = tuple(cp.ascontiguousarray(atmosphere[key], dtype=cp.float32)
                                   for key in ('qv','qc','qi','qs','temperature','pressure'))
                    get_kernel('farms','farms_cloud_fraction')(
                        ((cf.size+127)//128,), (128,),
                        (*values, cf, np.int32(cf.size),
                         np.int32(bool(cfg.moist) and int(cfg.mp_physics)!=0),
                         np.int32(fqi), np.int32(fqs), np.int32(int(cfg.mp_physics) in (5,15))))
            if mynn_bl_cloud_active(cfg.bl_pbl_physics, cfg.icloud_bl):
                qc, qi, cf = merge_mynn_bl_clouds(qc.copy(), qi.copy(), cf,
                    qc_bl=driver.fields.get('qc_bl'), qi_bl=driver.fields.get('qi_bl'),
                    cldfra_bl=driver.fields.get('cldfra_bl'),
                    bl_pbl_physics=cfg.bl_pbl_physics, icloud_bl=cfg.icloud_bl,
                    itimestep=wrf_itimestep(state.elapsed_seconds,cfg.dt),
                    bl_mynn_version=cfg.bl_mynn_version,
                    temperature=atmosphere['temperature'])
        if int(cfg.shcu_physics) == 5:
            from woof.core.column_cumulus import merge_column_cumulus_clouds
            qc, qi, cf = merge_column_cumulus_clouds(
                qc, qi, cf, fields=driver.fields, cfg=cfg, columns=lambda value: value)
        # WRF Solar_step updates CLDFRA on every FARMS step, including
        # steps between full radiation calls. Publish that same fraction
        # for history and consumers holding the radiation cloud field.
        if driver.cldfra is not None:
            driver.cldfra[...] = cf
        if 'uw_cldfra' in driver.fields:
            driver.fields['uw_cldfra'][...] = cf
        flags = legacy_scheme_has_req(cfg.mp_physics, cfg.use_mp_re)
        zero = cp.zeros((nz,nc), cp.float32)
        radii = []
        for name, flag in zip(('effc','effi','effs'), flags):
            value = getattr(state, name, None)
            if flag and value is None:
                raise ValueError(f'FARMS microphysics radius state.{name} is missing')
            radii.append(column(value) * np.float32(1.e-6) if flag else zero)
        if cfg.aer_opt == 0:
            aer = cp.zeros(nc, cp.float32)
            ssa, asy, aod, ang = aer, aer, aer, aer
        elif cfg.aer_opt == 1:
            if driver.aodtot is None:
                raise ValueError('FARMS aer_opt=1 needs the radiation AODTOT producer')
            aod = surface(driver.aodtot)
            ssa, asy, ang = (cp.full(nc,np.float32(x),cp.float32) for x in
                             (cfg.aer_ssa_val,cfg.aer_asy_val,cfg.aer_angexp_val))
        elif cfg.aer_opt == 2 or (cfg.aer_opt == 3 and cfg.rrtmg_aer3_source == "wrf_461"):
            ssa, asy, aod, ang = (surface(driver.aerosol2d[name]) for name in
                                  ('AERSSA2D','AERASY2D','AOD5502D','ANGEXP2D'))
        else:
            raise ValueError('FARMS aer_opt=3 needs stock WRF gt_aod diagnostics')
        julian, _, _ = calendar_scalars(self.start_time,state.elapsed_seconds)
        fluxes = farms_surface_columns(
            column(atmosphere['p_interface']), column(atmosphere['rho']),
            column(atmosphere['dz']), surface(driver.fields['albedo']),
            ssa, asy, aod, ang, self.coszen_loc(state.elapsed_seconds),
            column(atmosphere['qv']), column(qi), column(atmosphere['qs']),
            column(qc), *radii, column(cf), julian=julian,
            has_reqc=int(flags[0]), has_reqi=int(flags[1]), has_reqs=int(flags[2]))
        for name, value in zip(FARMS_FIELDS,fluxes):
            driver.fields[name][...] = value.reshape(ny,nx)
        if cfg.couple_farms:
            driver.fields['swdown'][...] = driver.fields['swdown2']
            from woof.core.radiation_carriers import CARRIER_SOURCE_RADIATION_SCHEME
            driver.carriers.declare('swdown', source=CARRIER_SOURCE_RADIATION_SCHEME,
                                    model_time=float(state.elapsed_seconds))
