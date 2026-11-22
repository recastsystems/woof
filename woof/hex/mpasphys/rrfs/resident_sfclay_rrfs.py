"""Persistent MYNN-SFC glue for the RRFS resident framework facade.

Replaces sf_mynnsfc_v600.driver's allocations and field assignments with one
CopyPlan pack and one unpack. The driver and column kernels retain their
arithmetic and order, including the first call's pblh passed to the sea call.
Native-library calls continue through the existing driver. Every pack row has
one writer: zero-fill only fields the original pack did not assign.
"""
from __future__ import annotations

import numpy as np

from . import sf_mynnsfc_v600 as sfc
from .glue_plan import CopyPlan

# The measurement harness alone disables fusion for the previous-glue arm.
FUSED = True
_PROFILES = ('u3d', 'v3d', 't3d', 'qv3d', 'p3d', 'dz8w', 'th3d', 'rho3d')


def resident_slot(provider, call, scheme=None):
    """Select only the resident facade; ordinary and native pools keep their path.

    ``scheme`` is the slot's MYNN-SFC module (sf_mynnsfc_v600 when None; sf_mynnsfc_v610 under a suite version).
    """
    from .resident_rrfs import ResidentFrameworkFacade
    if not FUSED or not isinstance(getattr(provider, 'native', None), ResidentFrameworkFacade):
        return None
    scheme = sfc if scheme is None else scheme
    name = '_rrfs_sfclay_resident' if scheme is sfc else '_rrfs_sfclay_resident_' + scheme.__name__.rsplit('.', 1)[-1]
    slot = getattr(provider, name, None)
    if slot is None:
        slot = ResidentSfclaySlot(provider, call, scheme)
        setattr(provider, name, slot)
    return slot


class ResidentSfclaySlot:
    """Scheme-layout buffers whose lifetime is the provider's resident session."""

    def __init__(self, provider, call, scheme=sfc):
        self.provider = provider
        self.sfc = scheme
        cp = provider.cp
        config = provider['configs']
        self.frac = bool(call.options['config_frac_seaice'])
        self.restart = bool(config['config_do_restart'])
        self.cycling = bool(config['config_do_DAcycling'])
        self.len_disp = np.float32(config['config_len_disp'])
        # A framework host scalar, read once at setup, never a device field read.
        self.threshold = np.float32(provider['scalar_state']['xice_threshold'])
        self.n = provider.session.layout.dimensions['nCellsSolve']
        self.packed = cp.empty((len(self.sfc.FIELDS), self.n), cp.float32)
        self.sea = cp.empty_like(self.packed) if self.frac else self.packed
        self.vegetation = cp.ones(self.n, cp.int32)
        self.errors = cp.empty(self.n, cp.int32)
        lsm = {'sf_noaa': 1, 'sf_noahmp': 2, 'sf_ruc': 3}.get(call.options['config_lsm_scheme'], 4)
        values = dict(ivegsrc=0, redrag=0, sfc_z0_type=0, spp_pbl=0,
                      sf_mynn_sfcflux_water=call.options['config_mynn_sfcflux_water'],
                      sf_mynn_sfcflux_land=call.options['config_mynn_sfcflux_land'],
                      shalwater_z0=0, isfflx=1, restart=int(self.restart), cycling=int(self.cycling),
                      initflag=0, flag_iter=1, flagc_lsm=lsm, itimestep=0)
        self.options = cp.asarray([values[name] for name in self.sfc.OPTIONS], cp.int32)
        # Time-varying integers travel in the column launch's arguments. There
        # is no per-call options allocation, host read, or options-table upload.
        self._signature = None
        self._from = self.sfc.module().get_function('mynnsfc_from_mpas')
        self._to = self.sfc.module().get_function('mynnsfc_to_mpas')
        register = getattr(provider, 'on_prepared_release', None)
        if register is not None:
            # Drop held prepared views and plans when they are released; rebind on the next call
            # (NativeDevicePoolViews.on_prepared_release says why).
            register(self._drop_prepared)

    def _drop_prepared(self):
        self._signature = None
        self.plans = None

    def _resolve(self):
        p = self.provider
        diag, surface, profiles = p['diag_physics'], p['sfc_input'], p['atm_input']
        inputs = {name: diag[name] for name in self.sfc._DIAG_INPUT}
        inputs['pblh'] = diag['hpbl']
        inputs.update((name, surface[key]) for name, key in self.sfc._SURFACE.items())
        inputs['psfcpa'] = profiles['psfcpa']
        for name in _PROFILES:
            value = profiles[name]
            if (value.dtype != p.cp.float32 or value.ndim != 2 or value.shape[0] < 2
                    or value.shape[1] != self.n or not value.flags.c_contiguous):
                raise ValueError('MYNN-SFC profile ' + name +
                                 ' requires contiguous float32 (levels>=2, nCells)')
            inputs[name] = value[0]
            if name in ('u3d', 'v3d', 'dz8w'):
                inputs[name + '2'] = value[1]
        outputs = {name: diag[self.sfc._DIAG_NAMES.get(name, name)] for name in self.sfc._WRITE}
        handoff = ({name: p['prepared'][name + '_sea'] for name in self.sfc._SEA_HANDOFF}
                   if self.frac else {})
        self.density = p['mesh']['meshDensity']
        self.ice = surface['xice']
        self.sst = surface['sst'] if self.frac else self.packed[0]
        arrays = (*inputs.values(), *outputs.values(), *handoff.values(), self.density, self.ice, self.sst)
        signature = tuple((a.data.ptr, a.shape, a.strides, a.dtype.str) for a in arrays)
        if signature != self._signature:
            pack = CopyPlan('rrfs sfclay pack')
            for i, name in enumerate(self.sfc.FIELDS):
                if name in inputs:
                    pack.copy(self.packed[i], inputs[name])
                else:
                    pack.fill(self.packed[i], np.float32(0))
            unpack = CopyPlan('rrfs sfclay unpack')
            for name, dst in outputs.items():
                unpack.copy(dst, self.packed[self.sfc.FIELDS.index(name)])
            for name, dst in handoff.items():
                unpack.copy(dst, self.sea[self.sfc.FIELDS.index(name)])
            self.plans = pack.build(), unpack.build()
            self._signature = signature

    def run(self, call):
        self._resolve()
        pack, unpack = self.plans
        pack()
        n = self.n
        grid, block = ((n + 127) // 128,), (128,)
        if n:
            self._from(grid, block, (self.packed, self.sea, self.density, self.sst, self.ice,
                                    self.len_disp, self.threshold, np.int32(self.frac), np.int32(n)))
        kwargs = dict(vegtype=self.vegetation, errors=self.errors, xice_threshold=self.threshold,
                      itimestep=call.itimestep,
                      initflag=int(not (self.restart or self.cycling or call.itimestep > 1)),
                      corrections=getattr(call, 'corrections', ()))
        from ...wave_surface import feedback
        kwargs['ocean_z0'] = feedback(self.provider)
        self.sfc.run_packed(self.packed, self.options, **kwargs)
        if self.frac:
            # This dependency sits between the two column calls, not in pack.
            index = self.sfc.FIELDS.index('pblh')
            self.sea[index] = self.packed[index]
            self.sfc.run_packed(self.sea, self.options, **kwargs)
            if n:
                self._to(grid, block, (self.packed, self.sea, self.ice, self.threshold, np.int32(n)))
        unpack()
        return self.provider['diag_physics']
