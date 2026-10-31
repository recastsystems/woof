"""Persistent RRFS radiation boundary buffers and fused word copies.

The numerical kernels remain those of radiation_rrfs and the shared RRTMG
wrappers. Each boundary uses one pack and one unpack launch. Pack plans
are rebound when a provider changes storage, including after an A/B restore.
The shared wrappers still own their dynamic solver workspaces and selection
of layer groups and sunlit columns.
"""
from __future__ import annotations

import numpy as np

from .glue_plan import CopyPlan
from .driver_rrfs import StrictPhysicsKernelCache

# Solver output allocations change between calls. Passing their pointers as
# kernel arguments avoids rebuilding/uploading a CopyPlan table after the
# solver has been queued. Only integer address arithmetic and word moves.
_UNPACK_OPS = 23
UNPACK_SOURCE = r"""
#define ARGS(N) unsigned int* d##N, const unsigned int* s##N, \
    long long a##N, long long b##N, long long c##N, long long f##N, \
    long long n##N, long long w##N, unsigned int v##N
#define MOVE(N) case N: if(e<n##N) { \
    long long row=e/w##N, col=e%w##N; \
    d##N[row*a##N+col*b##N] = s##N ? s##N[row*c##N+col*f##N] : v##N; \
    } break;
extern "C" __global__ void rrfs_radiation_unpack(
""" + ','.join(f'ARGS({i})' for i in range(_UNPACK_OPS)) + r""") {
    long long e=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    switch(blockIdx.y) {
""" + ''.join(f'MOVE({i})' for i in range(_UNPACK_OPS)) + r"""
    }
}
"""

FUSED = True  # Disabled only by the real-input comparison launcher.


def resident_slot(provider, call):
    from .resident_rrfs import ResidentFrameworkFacade
    if (not FUSED or call.slot not in ('radiation_sw', 'radiation_lw') or
            not isinstance(getattr(provider, 'native', None), ResidentFrameworkFacade)):
        return None
    slots = getattr(provider, '_rrfs_radiation_slots', None)
    if slots is None:
        slots = provider._rrfs_radiation_slots = {}
    if call.slot not in slots:
        slots[call.slot] = ResidentRadiationSlot(provider, call.slot)
    return slots[call.slot]


class ResidentRadiationSlot:
    """One slot's buffers, with fresh authority views on every call."""

    def __init__(self, provider, kind):
        self.provider, self.cp, self.kind = provider, provider.cp, kind
        dims = provider.session.layout.dimensions
        self.nc, self.nz = int(dims['nCellsSolve']), int(dims['nVertLevels'])
        self.buffers, self._plans, self._signatures = {}, {}, {}
        self._unpack_copies, self._unpack_fills = (), ()
        # One strict kernel cache per provider, reused by both slots: every new
        # StrictPhysicsKernelCache opens the AOT pack and fingerprints NVRTC
        # (~0.5 s each on an RTX PRO 6000 box), which made the first radiation
        # step ~1 s slower. Prefer the framework's own cache built at init.
        cache = getattr(provider, '_rrfs_glue_kernels', None)
        if cache is None:
            cache = getattr(provider.native, '_kernels', None)
            if not isinstance(cache, StrictPhysicsKernelCache):
                cache = StrictPhysicsKernelCache()
            provider._rrfs_glue_kernels = cache
        self._unpack_kernel = cache.raw_kernel(
            'rrfs_radiation_unpack', UNPACK_SOURCE, module_key='rrfs.radiation.unpack')
        register = getattr(provider, 'on_prepared_release', None)
        if register is not None:
            # Drop held prepared views and plans when they are released; rebind on the next call
            # (NativeDevicePoolViews.on_prepared_release says why).
            register(self._drop_prepared)

    def _drop_prepared(self):
        self._plans, self._signatures = {}, {}
        self._unpack_copies, self._unpack_fills = (), ()

    @property
    def plans(self):
        # Proof only: use the existing CopyPlan overlap checker for exactly
        # the same views passed to the argument-based unpack kernel.
        unpack = CopyPlan(self.kind + '.unpack')
        for dst, src in self._unpack_copies:
            unpack.copy(dst, src)
        for dst, value in self._unpack_fills:
            unpack.fill(dst, value)
        return (*self._plans.values(), unpack)

    def buffer(self, name, shape):
        value = self.buffers.get(name)
        if value is None or value.shape != shape:
            value = self.buffers[name] = self.cp.empty(shape, self.cp.float32)
        return value

    def move(self, name, copies, fills=()):
        if name == 'unpack':
            return self.unpack(copies, fills)
        def signature(a):
            return (a.data.ptr, a.shape, a.strides, a.dtype.str)
        key = (tuple((signature(dst), signature(src)) for dst, src in copies),
               tuple((signature(dst), float(value)) for dst, value in fills))
        if self._signatures.get(name) != key:
            plan = CopyPlan(self.kind + '.' + name)
            for dst, src in copies:
                plan.copy(dst, src)
            for dst, value in fills:
                plan.fill(dst, value)
            self._plans[name] = plan.build()
            self._signatures[name] = key
        self._plans[name]()

    def unpack(self, copies, fills):
        self._unpack_copies, self._unpack_fills = copies, fills
        operations = [(dst, src, 0) for dst, src in copies]
        operations.extend((dst, None, value) for dst, value in fills)
        if len(operations) > _UNPACK_OPS:
            raise ValueError('radiation unpack exceeds its argument roster')
        args, largest = [], 0
        for dst, src, value in operations:
            if dst.dtype != self.cp.float32 or dst.ndim not in (1, 2):
                raise TypeError('radiation unpack expects rank 1/2 float32 views')
            if src is not None and (src.shape != dst.shape or src.dtype != dst.dtype):
                raise ValueError('radiation unpack source and destination must agree')
            def strides(array):
                if array is None:
                    return (0, 0)
                if any(s % 4 for s in array.strides):
                    raise ValueError('radiation unpack needs whole-word strides')
                return ((0, array.strides[0] // 4) if array.ndim == 1 else
                        tuple(s // 4 for s in array.strides))
            ds, ss = strides(dst), strides(src)
            word = np.asarray(value, dtype=np.float32).view(np.uint32)[()]
            args.extend((dst, np.uint64(0) if src is None else src,
                         *(np.int64(v) for v in (*ds, *ss, dst.size, dst.shape[-1])), word))
            largest = max(largest, dst.size)
        padding = (np.uint64(0), np.uint64(0), *(np.int64(0) for _ in range(6)), np.uint32(0))
        args.extend(padding * (_UNPACK_OPS - len(operations)))
        if largest:
            self._unpack_kernel(((largest + 255) // 256, len(operations)), (256,), tuple(args))

    def prepared(self, name, levels=None):
        return self.provider.prepared(name + '_p')[:self.nz if levels is None else levels, :self.nc]

    def degrees(self, name, source, degrad):
        from .radiation_rrfs import _launch
        out = self.buffer(name, (self.nc,))
        _launch('degrees', self.nc, (np.int32(self.nc), np.float32(degrad), source, out))
        return out

    def run(self, call):
        if self.kind == 'radiation_sw':
            return self.shortwave(call)
        return self.longwave(call)

    def shortwave(self, call):
        from . import radiation_rrfs as rad
        from ..rrtmg_sw_v842 import FIELD_NAMES, SCALAR_OUTPUTS, PROFILE_OUTPUTS
        p, nc, nz = self.provider, self.nc, self.nz
        engine, config = p.rrtmg_sw_rrfs_resources, call.options
        diag, sfc, mesh = (p[n] for n in ('diag_physics', 'sfc_input', 'mesh'))
        constants, flags = p['constants'], p['scalar_state']
        # The shared controls helper reads its four device scalars back.
        # Keep it before the boundary work, as in the original binder, so
        # that read does not also wait for the aerosol kernels queued here.
        calendar = {n: p.native.get_table('mpas_atmphys_manager.' + n)[0][()]
                    for n in ('curr_julday', 'gmt', 'julday')}
        controls = engine.driver_controls(config, xtime_s=call.time_seconds,
            dt_radtsw=flags['dt_radtsw'], curr_julday=calendar['curr_julday'],
            degrad=constants['degrad'], dpd=constants['dpd'])
        scheme = config['config_microp_scheme'].strip()
        diagnosed = bool(config['config_microp_re']) and config['config_microp_scheme'] in rad.RADIUS_SCHEMES
        aware = scheme == 'mp_thompson_aerosols' or (scheme == 'mp_tempo' and config['config_tempo_aerosolaware'])
        aliases = dict(p3d='pres_hyd', t3d='t', dz8w='dz', pi3d='pi',
                       qv3d='qvrad', qc3d='qcrad', qi3d='qirad', qs3d='qsrad')
        fields = {n: self.buffer(n, (nc, nz)) for n in FIELD_NAMES}
        copies, fills = [], []
        for name, dst in fields.items():
            if name in aliases:
                copies.append((dst, self.prepared(aliases[name]).T))
            elif name.startswith('re_') and not diagnosed:
                fills.append((dst, 0))
            else:
                copies.append((dst, diag['cldfrac' if name == 'cldfra3d' else name][:nz, :nc].T))
        p8w, t8w = (self.buffer(n, (nc, nz + 1)) for n in ('p8w', 't8w'))
        copies.extend(((p8w, self.prepared('pres2_hyd', nz + 1).T),
                       (t8w, self.prepared('t2', nz + 1).T)))
        surface = dict(albedo=diag['sfc_albedo'][:nc], tsk=sfc['skintemp'][:nc])
        surface.update({n: sfc[n][:nc] for n in ('snow', 'xice', 'xland')})
        surface.update(xlat=self.degrees('xlat', mesh['latCell'][:nc], constants['degrad']),
                       xlong=self.degrees('xlong', mesh['lonCell'][:nc], constants['degrad']))
        pin = rad._ozone_pool(p, call)['pin']
        ozone = self.buffer('ozone', (nc, int(pin.size)))
        if config['config_o3climatology']:
            copies.append((ozone, diag['o3clim'][:, :nc].T))
        aerosols = {}
        if aware:
            for name, native in (('nwfa', 'nwfa'), ('nifa', 'nifa'), ('aerosol_qv', 'qv')):
                aerosols[name] = self.buffer(name, (nc, nz))
                copies.append((aerosols[name], self.prepared(native).T))
        self.move('pack', copies, fills)
        aer = None
        if aware:
            aer = engine.thompson_aerosols(fields['p3d'], fields['t3d'], aerosols['aerosol_qv'],
                fields['dz8w'], aerosols['nwfa'], aerosols['nifa'], surface['xland'], mesh['zgrid'][0, :nc])
        controls.update(gmt=calendar['gmt'], julday=calendar['julday'], icloud=1,
                        pin=pin, o3clim=ozone, aer_opt=3 if aware else 0, aerosol=aer)
        controls.update({n: flags[n] for n in ('has_reqc', 'has_reqi', 'has_reqs')})
        out = engine.rrtmg_swrad(fields, surface, p8w, t8w, **controls)
        copies = [(diag[n][:nc], out[n]) for n in SCALAR_OUTPUTS]
        copies.append((p['tend_physics']['rthratensw'][:, :nc], out['rthratensw'].T))
        # Preserve the public result's contiguous level-major layout as well
        # as the resident pools, with the same single unpack launch.
        for name in ('rthratensw', *PROFILE_OUTPUTS):
            public = self.buffer('out.' + name, out[name].T.shape)
            copies.append((public, out[name].T))
            out[name] = public
        if aer is not None:
            public = self.buffer('out.taod5503d', aer['taod5503d'].T.shape)
            copies.extend(((diag['taod5502d'][:nc], aer['taod5502d']),
                           (diag['taod5503d'][:, :nc], aer['taod5503d'].T),
                           (public, aer['taod5503d'].T)))
            out.update(taod5502d=aer['taod5502d'], taod5503d=public)
        self.move('unpack', copies)
        return out

    def longwave(self, call):
        from . import radiation_rrfs as rad
        from .. import rrtmg_lw_v842 as lw
        from ..radshared_v842 import OVERLAP
        p, nc, nz = self.provider, self.nc, self.nz
        diag, sfc, config = p['diag_physics'], p['sfc_input'], call.options
        flags = p['scalar_state']
        radius_scheme = config['config_microp_scheme'].strip() in rad.RADIUS_SCHEMES
        enabled = bool(config['config_microp_re']) and radius_scheme
        model = self.buffer('model', (nc, len(lw.MODEL_FIELDS), nz))
        interfaces = self.buffer('interfaces', (nc, 2, nz + 1))
        surface = self.buffer('surface', (nc, len(lw.SURFACE_FIELDS)))
        aliases = dict(p3d='pres_hyd', t3d='t', dz8w='dz', pi3d='pi',
                       qv3d='qvrad', qc3d='qcrad', qi3d='qirad', qs3d='qsrad')
        copies, fills = [], []
        for j, name in enumerate(lw.MODEL_FIELDS):
            dst = model[:, j, :]
            if name in aliases:
                copies.append((dst, self.prepared(aliases[name]).T))
            elif name.startswith('re_') and not enabled:
                fills.append((dst, 0))
            else:
                copies.append((dst, diag['cldfrac' if name == 'cldfra3d' else name][:nz, :nc].T))
        copies.extend(((interfaces[:, 0], self.prepared('pres2_hyd', nz + 1).T),
                       (interfaces[:, 1], self.prepared('t2', nz + 1).T)))
        xlat = self.degrees('xlat', p['mesh']['latCell'][:nc], p['constants']['degrad'])
        sources = dict(tsk=sfc['skintemp'][:nc], emiss=diag['sfc_emiss'][:nc], xlat=xlat,
                       **{n: sfc[n][:nc] for n in ('xland', 'xice', 'snow')})
        copies.extend((surface[:, j], sources[n]) for j, n in enumerate(lw.SURFACE_FIELDS))
        o3input = 2 if config['config_o3climatology'] else 0
        pin, ozone, o3vmr = None, None, None
        if o3input == 2:
            pin = rad._ozone_pool(p, call)['pin']
            ozone = self.buffer('ozone', (nc, int(pin.size)))
            pressure = self.buffer('pressure', (nc, nz))
            copies.extend(((ozone, diag['o3clim'][:, :nc].T),
                           (pressure, self.prepared('pres_hyd').T)))
        self.move('pack', copies, fills)
        if o3input == 2:
            mb = self.buffer('pressure_mb', (nc, nz))
            lw._gpu_kernel('mpas_pressure_mb')(((nc*nz+127)//128,), (128,), (np.int32(nc*nz), pressure, mb))
            o3vmr = lw.vinterp_ozn_device(mb, pin, ozone)
        overlap = OVERLAP[config['config_radt_cld_overlap'].strip()]
        idcor = int(overlap >= 4 and config['config_radt_cld_dcorrlen'].strip() == 'latitude_varying')
        result = lw.rrtmg_lwrad_device(model, interfaces, surface, p.rrtmg_lw_rrfs_resources,
            icloud=1, julday=int(p.native.get_table('mpas_atmphys_manager.julday')[0][()]),
            cldovrlp=overlap, idcor=idcor,
            has_reqc=int(flags['has_reqc']), has_reqi=int(flags['has_reqi']), has_reqs=int(flags['has_reqs']),
            o3input=o3input, pin=pin, ozone=ozone, retain_profiles=False)
        copies = [(diag['olrtoa' if n == 'olr' else n][:nc], result[n]) for n in lw.FLUX_FIELDS]
        copies.append((p['tend_physics']['rthratenlw'][:, :nc], result['rthratenlw'].T))
        if o3vmr is not None:
            copies.append((diag['o3vmr'][:, :nc], o3vmr.T))
        fills = []
        if radius_scheme:
            for n in ('rre_cloud', 'rre_ice', 'rre_snow'):
                if enabled:
                    copies.append((diag[n][:, :nc], result[n].T))
                else:
                    fills.append((diag[n][:, :nc], 0))
        self.move('unpack', copies, fills)
        return result
