"""Bind actual model state to WRF's DFI carrier layout on the device.

Native names and package order: WRF v4.6.1 Registry.EM_COMMON:452-580,
source and public-domain notice in tools/dfi_wrf461_oracle. Number moments
belong to SCALAR; only mass species belong to MOIST. Their dummy slot is
kept because native inactive QC/QI package indices refer to that slot.
"""
from __future__ import annotations

import numpy as np

from woof.core.dfi import (DeviceDfiState, DfiModelDiagnostics, NativeLayout,
                            RADII, SURFACE_WATER, SURFACE_SOIL, RUC_SURFACE, COORDS)

# Registry order, not alphabetical order or the order of package membership.
MOIST_NAMES = ('QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QICE2', 'QICE3',
               'QSNOW', 'QGRAUP', 'QHAIL')
SCALAR_NAMES = ('QNDROP', 'QNICE', 'QNICE2', 'QNICE3', 'CWM', 'QNSNOW',
    'QNRAIN', 'QNGRAUPEL', 'QNHAIL', 'QNCCN', 'QNCLOUD', 'QVGRAUPEL', 'QVHAIL',
    'QZRAIN', 'QZGRAUPEL', 'QZHAIL', 'QRIMEF', 'QIR', 'QIB', 'QIR2', 'QIB2',
    'QZI', 'QVOLI', 'QAOLI', 'QVOLI2', 'QAOLI2', 'QVOLI3', 'QAOLI3',
    'QNWFA', 'QNIFA', 'QNBCA')


def backward_configuration(cfg, *, dt, moist_adv_opt=0):
    """WRF dfi_bck_init's model selectors; construction is setup only."""
    from dataclasses import fields
    from woof.config import _DfiBackwardRunConfig, validate_km_opt
    if dt >= 0:
        raise ValueError('backward DFI requires negative dt')
    values = {f.name:getattr(cfg,f.name) for f in fields(cfg) if f.init}
    values.update(dt=float(dt),clock_dt=float(dt),mp_physics=0,ra_physics=0,
        ra_lw_physics=0,ra_sw_physics=0,sf_surface_physics=0,
        sf_sfclay_physics=0,sf_urban_physics=0,bl_pbl_physics=0,cu_physics=0,
        damp_opt=0,gwd_opt=0,diff_6th_opt=0,km_opt=1,diff_opt=0,
        sf_ocean_physics=0,fractional_seaice=0,cu_diag=0,
        use_adaptive_time_step=False,moist_adv_opt=moist_adv_opt)
    result = _DfiBackwardRunConfig(**values)
    validate_km_opt(result)
    return result


class DfiPhysicsInitialization:
    """Retain setup inputs and the fixed-terrain base for start_domain.

    WRF start_em:554-636 reconstructs the same base from fixed geography.
    Its forward/forecast phy_init receives the current surface and soil,
    rather than rereading the analysis's original soil. The existing
    scheme initializers own this reinitialization; DFI supplies their
    actual live inputs at the phase boundary.
    """

    _LIVE_INPUTS = {'tsk':'tsk','soil_temperature':'tslb',
        'soil_moisture':'smois','liquid_moisture':'sh2o','snow':'snow',
        'snow_depth':'snowh','canwat':'canwat','xice':'xice','sst':'sst',
        'swdown':'swdown','glw':'glw','pblh':'pblh','mavail':'mavail'}

    def __init__(self, state, inputs):
        self.state, self.inputs = state, inputs
        self.driver = state.physics
        self.base = {name:getattr(state,name).copy() for name in
                     ('phb','pb','alb','mub2d','thb','dphb_resid')}
        self.radii = {name:getattr(state,name).copy() for name in
                      ('effc','effi','effs') if getattr(state,name,None) is not None}

    def restore_base(self, cfg):
        from woof.core.dycore import update_diagnostics
        for name,value in self.base.items():
            getattr(self.state,name)[...] = value
        update_diagnostics(self.state,cfg.hypsometric_opt)

    def reinitialize(self, cfg, *, epoch):
        from woof.core.physics import initialize_physics
        state = self.state
        surface = (state.physics or self.driver).fields
        inputs = dict(self.inputs)
        for argument,field in self._LIVE_INPUTS.items():
            if field in surface:
                inputs[argument] = surface[field]
        # A fresh driver owns fresh held tendencies and scheme clocks.
        # Reuse declared radiation geometry; the clock remains relative
        # to the original analysis, so solar time is not frozen/reset.
        inputs['cumulus'] = None
        self.restore_base(cfg)
        state.domain_start_offset = float(epoch)
        state._dfi_physics_epoch = float(epoch)
        remember = getattr(state,'_dfi_remember_physics',False)
        state._dfi_remember_physics = False
        try:
            driver = initialize_physics(state,cfg,**inputs)
        finally:
            state._dfi_remember_physics = remember
        # Native phy_init leaves these existing grid diagnostics intact.
        # A fresh driver otherwise replaces them with allocation defaults,
        # losing the input/last-forward-pass fields in initial history.
        for name in ('t2','q2','th2','u10','v10','psfc'):
            if name in surface and name in driver.fields:
                driver.fields[name][...] = surface[name]
        for name,value in self.radii.items():
            getattr(state,name)[...] = value
        self.driver = driver
        return driver


class ModelDfiState:
    """Own packed native fields and a DeviceDfiState for one actual domain.

    Missing ACTIVE state is an error. Zero allocations below are only
    native inactive Registry slots, never substitutes for active species,
    surface fields, WW, PHP or PH0. Those diagnostics have their own model
    calculation points in dycore.step.
    """

    def __init__(self, state, cfg, *, save_hydrometeors):
        import cupy as cp
        from woof.core.kernels import get_kernel
        from woof.ingest.wrfinput import active_moisture_map
        from woof.wrf_exact import ENABLED, DIAGNOSTICS_ENABLED

        self.state, self.cfg = state, cfg
        if state.phb.ndim != 3:
            raise ValueError('DFI model binding needs native per-column base fields')
        if state.physics is None:
            raise ValueError('bind DFI after the native physics fields are initialized')
        if not hasattr(state, '_dfi_diagnostics'):
            state._dfi_diagnostics = DfiModelDiagnostics(state)
        self.diagnostics = state._dfi_diagnostics
        active = dict(active_moisture_map(cfg))
        unknown = active.keys() - set(MOIST_NAMES+SCALAR_NAMES)
        if unknown:
            raise ValueError('DFI native package inventory is missing '+', '.join(sorted(unknown)))
        self.moist_names = tuple(n for n in MOIST_NAMES if n in active)
        self.scalar_names = tuple(n for n in SCALAR_NAMES if n in active)
        if not self.moist_names or self.moist_names[0] != 'QVAPOR':
            raise ValueError('DFI native rebalance needs its active vapor carrier')
        shape = state.p.shape
        moist = cp.zeros((1+len(self.moist_names), *shape), cp.float32)
        scalar = cp.zeros((1+len(self.scalar_names), *shape), cp.float32)
        self.pack_pairs = []
        for names, target in ((self.moist_names, moist), (self.scalar_names, scalar)):
            for i, name in enumerate(names, 1):
                source = getattr(state, active[name], None)
                if source is None or source.shape != shape:
                    raise ValueError(f'DFI lacks active native {name} model carrier {active[name]}')
                self.pack_pairs.append((source, target[i]))
        self.copy_t, self.copy_p = not ENABLED, not DIAGNOSTICS_ENABLED
        fields = dict(mu_2=state.mup, u_2=state.u, v_2=state.v, w_2=state.w,
            ww=self.diagnostics.ww, t_2=cp.empty_like(state.thp) if self.copy_t else state.thp,
            phb=state.phb, ph0=self.diagnostics.ph0, php=self.diagnostics.php,
            p=cp.empty_like(state.p) if self.copy_p else state.p_perturbation,
            ph_2=state.php, tke_2=(state.tke if state.tke is not None else cp.zeros_like(state.p)),
            al=state.al, alt=state.alt, pb=state.pb, moist=moist, scalar=scalar,
            alb=state.alb, mub=state.mub2d)
        surface = state.physics.fields
        needed = SURFACE_WATER+SURFACE_SOIL+('psfc', 'p_hyd')
        if cfg.sf_surface_physics == 3:
            needed += RUC_SURFACE
        for name in needed:
            if name not in surface:
                raise ValueError(f'DFI lacks active native surface/physics field {name}')
            fields[name] = surface[name]
        fields.update({name: getattr(state, name) for name in COORDS})
        self.radii_pairs = []
        for native, model in zip(RADII, ('effc', 'effi', 'effs')):
            fields[native] = cp.zeros_like(state.p)
            source = getattr(state, model, None)
            if source is not None:
                self.radii_pairs.append((source, fields[native]))
        slots = {name: i for i, name in enumerate(self.moist_names, 1)}
        layout = NativeLayout(columns=state.mup.size, levels=shape[0],
            moist_species=moist.shape[0], qv=slots['QVAPOR'], qc=slots.get('QCLOUD', 0),
            qi=slots.get('QICE', 0), first_scalar=1,
            theta_m=bool(getattr(cfg, 'use_theta_m', 0)),
            hypsometric=cfg.hypsometric_opt, p_top=float(state.p_top),
            surface_physics=cfg.sf_surface_physics, microphysics_dfi=cfg.mp_physics,
            save_hydrometeors=bool(save_hydrometeors))
        self.native = DeviceDfiState(fields, layout)
        self._kernels = {n: get_kernel('dfi', 'dfi_'+n) for n in
                         ('export_thermodynamics', 'import_thermodynamics', 'convert_radii')}
        self.pack = self.native._rows(cp, [(a,b,0) for a,b in self.pack_pairs])
        self.unpack = self.native._rows(cp, [(b,a,0) for a,b in self.pack_pairs])
        self.radius_export = self._radius_rows(cp, self.radii_pairs)
        self.radius_import = self._radius_rows(cp, [(b,a) for a,b in self.radii_pairs])
        self.export()

    @staticmethod
    def _radius_rows(cp, pairs):
        if not pairs:
            return None
        rows = cp.asarray(np.asarray([(a.data.ptr,b.data.ptr,a.size) for a,b in pairs], np.uint64))
        return rows, ((max(a.size for a,b in pairs)+255)//256, len(pairs))

    def _thermodynamics(self, name, *extra):
        state, fields = self.state, self.native.fields
        self._kernels[name](((state.p.size+255)//256,), (256,),
            (state.thp,state.thb,state.p,state.pb,fields['t_2'],fields['p'],
             np.uint64(state.p.size),np.uint32(state.mup.size),
             np.int32(state.thb.ndim==3),*extra))

    def _radii(self, rows, scale):
        if rows is not None:
            descriptor, grid = rows
            self._kernels['convert_radii'](grid,(256,),(descriptor,np.float32(scale)))

    def export(self):
        """Snapshot native representations before each captured DFI sample."""
        self._thermodynamics('export_thermodynamics', np.int32(self.copy_t), np.int32(self.copy_p))
        self.native._launch_rows('snapshot_fields', self.pack)
        self._radii(self.radius_export, 1.e-6)

    def import_filtered(self):
        """Install the filtered native state at a phase boundary, on the GPU."""
        self._thermodynamics('import_thermodynamics', np.int32(self.copy_t))
        self.native._launch_rows('snapshot_fields', self.unpack)
        self._radii(self.radius_import, 1.e6)

    def accumulate(self, weights):
        self.export()
        self.native.accumulate(weights, advance_cursor=False)

    def rebind_physics(self):
        """Rebind phase-initialized surfaces without losing saved DFI arrays."""
        import cupy as cp
        surface = self.state.physics.fields
        names = SURFACE_WATER+self.native.soil+('psfc','p_hyd')
        for name in names:
            if name not in surface or surface[name].shape != self.native.fields[name].shape:
                raise ValueError('DFI phase initialization changed surface inventory: '+name)
            self.native.fields[name] = surface[name]
        self.native._setup_descriptors(cp)

