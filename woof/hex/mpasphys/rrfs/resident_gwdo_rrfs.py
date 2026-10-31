"""Persistent RRFS orographic UGWP boundary, with one pack and one unpack.

The assignments are ugwp_oro_v842.bind_native_pools/run_slot's transaction.
All arithmetic uses its unchanged dx, drag and rounded-difference kernels.
CopyPlan moves words only. Prepared top storage and pool halos are untouched.
NGW remains on its existing route: the hrrrv5 suite defaults it off. Embedded
NGW callbacks use the original boundary, which preserves their full contract.
"""
from __future__ import annotations

import numpy as np

from .. import ugwp_oro_v842 as oro
from .glue_plan import CopyPlan

# Measurement harness switch; production always uses the resident boundary.
FUSED = True


def resident_slot(provider, call):
    from .resident_rrfs import ResidentFrameworkFacade
    if (not FUSED or not isinstance(getattr(provider, 'native', None), ResidentFrameworkFacade)
            or call.options.get('config_ngw_scheme', False)):
        return None
    slot = getattr(provider, '_rrfs_gwdo_resident', None)
    if slot is None:
        slot = provider._rrfs_gwdo_resident = ResidentGwdoSlot(provider)
    return slot


def _descriptor(array):
    return array.data.ptr, array.shape, array.strides, array.dtype.str


class ResidentGwdoSlot:
    """Run-scoped work buffers; rebuild plans if any bound storage changes."""

    def __init__(self, provider):
        self.provider = provider
        self.cp = provider.cp
        self._signature = None
        self._shape = None
        register = getattr(provider, 'on_prepared_release', None)
        if register is not None:
            # Drop held prepared views and plans when they are released; rebind on the next call
            # (NativeDevicePoolViews.on_prepared_release says why).
            register(self._drop_prepared)

    def _drop_prepared(self):
        self._signature = None
        self.plans = self.outputs = self.before = self.prsi = self.kpbl = self.density = None

    def _build(self, packet, diags):
        cp = self.cp
        fields = packet['fields']
        nz, nc = fields['u_p'].shape
        if self._shape != (nz, nc):
            self.layers = cp.empty((len(oro.LAYER_INPUTS), nz, nc), dtype=cp.float32)
            self.surface = cp.empty((len(oro.SURFACE_INPUTS), nc), dtype=cp.float32)
            self.out = cp.empty((len(oro.LAYER_OUTPUTS), nz, nc), dtype=cp.float32)
            self.sout = cp.empty((len(oro.SURFACE_OUTPUTS), nc), dtype=cp.float32)
            self.diff = cp.empty((2, nz, nc), dtype=cp.float32)
            self.status = cp.empty(1, dtype=cp.int32)
            self._shape = nz, nc
        if nz < 2 or nc < 1:
            raise ValueError('UGWP requires at least two levels and one column')
        # Same input mapping as gwdo_from_MPAS, without its allocations or launches.
        dynamic = dict(uproj='u_p', vproj='v_p', t1='t_p', q1='qv_p', prsl='pres_hydd_p',
                       prslk='pi_p', zl='zmid_p', dz='dz_p', rublten='rublten', rvblten='rvblten',
                       hpbl='hpbl', br1='br', xland1='xland')
        columns = {key: fields[value] for key, value in dynamic.items()}
        for scale, suffix in (('ls', ''), ('ss', 'ss')):
            columns['var' + suffix] = fields['var2d' + scale]
            columns['oc1' + suffix] = fields['con' + scale]
            for direction in range(1, 5):
                for statistic in ('oa', 'ol'):
                    columns[f'{statistic}2d{direction}{suffix}'] = fields[f'{statistic}{direction}{scale}']
        pack = CopyPlan('rrfs ugwp oro pack')
        for index, name in enumerate(oro.LAYER_INPUTS):
            oro._resident(name, columns[name], (nz, nc), 'float32')
            pack.copy(self.layers[index], columns[name])
        for index, name in enumerate(oro.SURFACE_INPUTS):
            if name in ('sina', 'cosa'):
                pack.fill(self.surface[index], int(name == 'cosa'))
            elif name != 'dxmeter':
                oro._resident(name, columns[name], (nc,), 'float32')
                pack.copy(self.surface[index], columns[name])
        pack.fill(self.status, 0)
        self.density = oro._resident('meshDensity', fields['meshDensity'], (nc,), 'float32')
        self.prsi = oro._resident('prsi', fields['pres2_hydd_p'], (nz + 1, nc), 'float32')
        self.kpbl = oro._resident('kpbl', fields['kpbl'], (nc,), 'int32')
        self.dx = self.surface[oro.SURFACE_INPUTS.index('dxmeter')]
        outputs = {name: self.out[i] for i, name in enumerate(oro.LAYER_OUTPUTS[:12 if diags else 4])}
        outputs.update({name: self.sout[i] for i, name in enumerate(oro.SURFACE_OUTPUTS[:10 if diags else 2])})
        outputs.update(rubldiff=self.diff[0], rvbldiff=self.diff[1], rthblten=fields['rthblten'])
        unpack = CopyPlan('rrfs ugwp oro unpack')

        def copy(dst, src):
            # rthblten passes through: its pool target is already its source.
            if _descriptor(dst) != _descriptor(src):
                unpack.copy(dst, src)

        for name, target in packet['from_targets'].items():
            # These prepared tendencies are overwritten by the final output below.
            if name not in packet['prepared_targets']:
                value = fields[name]
                copy(target[:value.shape[0]], value)
        for name, target in zip(('sina', 'cosa', 'dxmeter'), packet['orientation']):
            copy(target, self.surface[oro.SURFACE_INPUTS.index(name)])
        for name, target in packet['targets'].items():
            copy(target, outputs[name])
        for name, target in packet['prepared_targets'].items():
            value = outputs[name]
            copy(target[:value.shape[0]], value)
        self.outputs = outputs
        self.before = fields['rublten'], fields['rvblten']
        self.plans = pack.build(), unpack.build()
        module = oro._module(cp.cuda.runtime.getDevice(), nz)
        self.dx_kernel = module.get_function('ugwp_oro_dx')
        self.run_kernel = module.get_function('ugwp_oro_run')
        self.diff_kernel = module.get_function('ugwp_oro_diff')

    def run(self, call):
        # Re-resolve provider-owned arrays each call, including after an output boundary.
        # Binding has no copies once the fields are resident and constants are host metadata.
        packet = oro._bind_native_pools(self.provider, call)['ugwp_oro_v842']
        diags = bool(call.options['config_ugwp_diags'])
        signature = (diags, *(tuple((name, _descriptor(a)) for name, a in packet[key].items())
                              for key in ('fields', 'targets', 'prepared_targets', 'from_targets')),
                     tuple(_descriptor(a) for a in packet['orientation']))
        if signature != self._signature:
            self._build(packet, diags)
            self._signature = signature
        constants = packet['constants']
        nz, nc = self._shape
        pack, unpack = self.plans
        pack()
        self.dx_kernel(((nc + 127) // 128,), (128,),
                       (self.density, self.dx, np.float32(constants['len_disp']), np.int32(nc)))
        self.run_kernel(((nc + 63) // 64,), (64,),
                        (self.layers, self.prsi, self.surface, self.kpbl, self.out, self.sout, self.status,
                         *(np.float32(constants[k]) for k in ('g', 'cp', 'rd', 'rv', 'fv', 'pi')),
                         np.float32(call.dt), np.int32(nz), np.int32(nc), np.int32(diags)))
        from ..deferred_checks_v842 import defer
        # defer captures a fresh reduced word before the next pack clears the status.
        defer(self.status, 'UGWP_INVALID_KPBL: native small-scale path would read beyond layer storage',
              error=ValueError)
        for i, before in enumerate(self.before):
            self.diff_kernel(((nz * nc + 127) // 128,), (128,),
                             (before, self.out[i], self.diff[i], np.int32(nz * nc)))
        unpack()
        return self.outputs
