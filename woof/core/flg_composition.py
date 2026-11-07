"""WRF's exceptional LW-7 call and empty SW-7 case, in native order."""
from dataclasses import replace
from woof.core.radiation_composition import ComposedRadiation


class NativeFLGComposition(ComposedRadiation):
    def __init__(self, *args, flg_longwave, **kwargs):
        super().__init__(*args, **kwargs)
        self.flg_longwave = flg_longwave
        self.diagnostics = (self.longwave_adapter.diagnostics if flg_longwave else ())

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        import numpy as np
        if not self.flg_longwave:
            # SW-7 is an empty CASE. The radiation driver has zeroed GSW,
            # SWDOWN and beam fluxes before executing the longwave call.
            return super().__call__(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
        arguments = dict(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
        lw = self.longwave_adapter(**arguments)
        sw = self.shortwave_adapter(**arguments)
        from woof.core.kernels import load_module
        kernel = load_module('flg_driver').get_function('flg_mix_driver')
        # Packed total is column-major; the returned heating is model order.
        nz,ny,nx = atmosphere['pressure'].shape
        total = cp.ascontiguousarray(self.longwave_adapter.heat_total.T).reshape(nz,ny,nx)
        heating = cp.empty_like(lw.rthratenlw)
        kernel(((heating.size+127)//128,),(128,),
               (np.int32(heating.size),total,cp.ascontiguousarray(sw.rthratensw),
                cp.ascontiguousarray(lw.rthratenlw),heating))
        diagnostics = dict(lw.flux_diagnostics)
        diagnostics.update({name:value for name,value in (sw.flux_diagnostics or {}).items()
                            if name.startswith(('SW','TSW','SSW'))})
        for name in ('SWDDIR','SWDDIF'):
            value = getattr(sw,name.lower(),None)
            if value is not None:diagnostics[name] = value
        if sw.swddir is not None and sw.coszen is not None:
            diagnostics['SWDDNI'] = cp.where(sw.coszen>0.,
                sw.swddir/cp.maximum(sw.coszen,cp.float32(1.e-4)),cp.float32(0.))
        elif int(cfg.ra_sw_physics) == 1:
            # Native radiation_driver:2884-2914 applies this fallback after
            # Dudhia. At the low-sun columns it leaves the LW-7 beam intact.
            for name in ('SWDDNI','SWDDIR','SWDDIF'):
                diagnostics[name] = diagnostics[name].copy()
            n = ny*nx
            terrain = fields.get('hgt',fields.get('ht'))
            if terrain is None:
                terrain = atmosphere['z_interface'][0]
            load_module('flg_driver').get_function('flg_mix_beams')(
                (((n+127)//128),),(128,),(np.int32(n),
                    cp.ascontiguousarray(sw.coszen),cp.ascontiguousarray(terrain),
                    cp.ascontiguousarray(sw.swdown),self.longwave_adapter.solar,
                    diagnostics['SWDDNI'],diagnostics['SWDDIR'],diagnostics['SWDDIF']))
        return replace(lw,rthratensw=heating,swdown=sw.swdown,gsw=sw.gsw,coszen=sw.coszen,
                       swddir=diagnostics['SWDDIR'],swddif=diagnostics['SWDDIF'],
                       boundary_fluxes=sw.boundary_fluxes,flux_diagnostics=diagnostics)

    def restart_identity(self):
        identity = super().restart_identity()
        identity['algorithm'] = 'wrf461-flg-native-mixed-sequence-v1'
        identity['flg_longwave'] = self.flg_longwave
        return identity
