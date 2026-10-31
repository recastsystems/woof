"""Resident MYNN-EDMF slot glue (RRFS hrrrv5 route): persistent Fortran-order buffers, fused pack and unpack.

Authority: the transaction of bl_mynnedmf_c942418.bind_native_pools + run_slot on the resident framework facade
(resident_rrfs.ResidentFrameworkFacade), fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F
461-1283 and 1535-1663 at 1d6e370e, in the same order:
  1. every bound module array as the facade's whole allocation (the provider's k,i core, plus the allocation tail
     for the arrays the provider does not cover);
  2. driver_rrfs.pbl_from_mpas's tile writes (the bl_mynnedmf branch, no chemistry);
  3. the translated wrapper (MynnEdmfCuda.run) on those whole allocations in Fortran order;
  4. driver_rrfs.pbl_to_mpas's pool writes and the facade's put_variable of every bound array.

THE COST THIS REMOVES: steps 1-4 ran as about 830 GPU operations per call (a whole C-order copy and a transposed
core copy per module array on read, a copy of every array in pbl_from_mpas, a Fortran-order copy of every argument
and its copy back in MynnEdmfCuda.run, a copy of every bound pool field in pbl_to_mpas, a transposed write per array
and a whole write per pool field on put): 741 kernels, 85 device copies and 8 fills per call, 8.5 ms of host per call
in the SPEED-2 profile p2 of c2efafcf (hrrrv5 x1.2562, RTX 5090, 2026-10-28). Here the whole allocations persist
across calls in Fortran order (so MynnEdmfCuda.run copies nothing), and steps 1, 2 and 4 are three fused word-copy
launches (glue_plan.CopyPlan) around the two driver kernels: about ten GPU operations per call.

Every word moves exactly as before (no arithmetic and no dtype conversion in the glue; the two driver kernels and the
wrapper are the same code on the same values), so the slot's outputs are the same words: the proof is the
in-process A/B of old and new slot on real hrrrv5 inputs (SPEED-2/glue.md). Arrays the provider does not cover
(allocation tails, host-authority module arrays) keep the facade's own get_variable/put_variable per call.
"""
from __future__ import annotations

from ...native_powf import CUDA_NATIVE_POWF
from . import bl_mynnedmf_c942418 as mynn
from . import driver_rrfs as driver
from .glue_plan import CopyPlan
# Module-level, not only inside ResidentMynnEdmfSlot: the AOT kernel inventory (cuda_backend/aot_build.inventory) evaluates
# each raw_kernel declaration in this module namespace.  THE BREAKAGE THIS PREVENTS: with the names imported only inside
# the method, the inventory raised "unresolved kernel declaration: name CUDA_NATIVE_POWF is not defined" and no kernel
# pack (make-pack, the wheel kernel set) could be built for any tree carrying this module (V5090, 2026-10-28).
from . import driver_rrfs as driver
from ...native_powf import CUDA_NATIVE_POWF

#: Off only in the A/B measurement harness, which runs the previous glue on the same captured state.
FUSED = True

_DIMS = ('ids', 'ide', 'jds', 'jde', 'kds', 'kde', 'ims', 'ime', 'jms', 'jme',
         'kms', 'kme', 'jts', 'jte', 'kts', 'kte')


class _Unsupported(Exception):
    """A selection this resident slot does not cover; the previous glue runs it."""


def resident_slot(provider, call, module=None):
    """The persistent slot for the resident facade route, or None (native library route, or unsupported options)."""
    from .resident_rrfs import ResidentFrameworkFacade
    if not FUSED or not isinstance(getattr(provider, 'native', None), ResidentFrameworkFacade):
        return None
    module = mynn if module is None else module
    slot = getattr(provider, '_rrfs_mynnedmf_resident', None)
    if slot is None:
        try:
            slot = ResidentMynnEdmfSlot(provider, call, module)
        except _Unsupported:
            slot = False
        provider._rrfs_mynnedmf_resident = slot
    return slot or None


class ResidentMynnEdmfSlot:
    """One MYNN-EDMF call per step on persistent whole allocations."""

    def __init__(self, provider, call, module):
        self.module = module
        self.extra_surface_fields = tuple(getattr(module, 'EXTRA_SURFACE_FIELDS', ()))
        cp = provider.cp
        facade = provider.native
        self.provider, self.facade, self.cp = provider, facade, cp
        # Rank-0 module scalars: tile bounds, species flags and manager intervals the host framework set at
        # physics_init (bind_native_pools reads the same words each call; none changes after initialization).
        metadata = {}
        for name in self.module.ARGUMENTS:
            if name.startswith('f_') or name in _DIMS:
                value = facade.get_variable(name)
                if value is None:
                    raise RuntimeError('Native MYNN scalar is absent: ' + name)
                metadata[name] = value[0].item()
        for name in ('dt_pbl', 'do_chemistry', 'num_chem'):
            value = facade.get_variable(name)
            if value is None:
                raise RuntimeError('Native MYNN manager scalar is absent: ' + name)
            metadata[name] = value[0].item()
        metadata.update(its=provider.session.arguments.its, ite=provider.session.arguments.ite)
        if any(metadata[n] != 1 for n in ('ims', 'kms', 'jms')):
            raise RuntimeError('r4 common PBL conversion requires native one-based memory origins')
        if metadata['jts'] != metadata['jte']:
            raise _Unsupported('a multi-j tile')
        options = self._options(call)
        if options['config_pbl_scheme'].strip() != 'bl_mynnedmf':
            raise ValueError('MYNN-EDMF callback received a different native selector')
        self.metadata = metadata
        if not hasattr(provider, '_rrfs_mynnedmf_backend'):
            image = getattr(provider, '_rrfs_mynnedmf_image', None)
            if image is None:
                raise RuntimeError('MYNN native callback requires its strict precompiled CUDA resource')
            provider._rrfs_mynnedmf_backend = self.module.MynnEdmfCuda(compiled_image=image)
            provider._rrfs_mynnedmf_common_kernels = driver.StrictPhysicsKernelCache()
        self.backend = provider._rrfs_mynnedmf_backend
        kernels = provider._rrfs_mynnedmf_common_kernels
        self._radiation = kernels.raw_kernel('rrfs_pbl_radiation_sum', CUDA_NATIVE_POWF + driver._SURFACE_CUDA,
                                             module_key='rrfs.driver.pbl.radiation')
        self._spacing = kernels.raw_kernel('rrfs_surface_spacing', CUDA_NATIVE_POWF + driver._SURFACE_CUDA,
                                           module_key='rrfs.driver.surface.spacing')
        # The bound module arrays, in bind_native_pools order; whole allocations persist in Fortran order.  A covering
        # array (its whole allocation is the provider's k,i core) is the provider's own words seen in Fortran order
        # (_alias), so it needs no allocation and no pack/unpack copy: the kernel reads and writes the words the
        # copies moved.  THE BREAKAGE THIS PREVENTS: a second whole copy of every covering array held for the run,
        # about 21 KB per cell on hrrrv5 (x1.40962, 2026-10-28), on top of the provider's own.
        self.names, self.lower, self.whole = [], {}, {}
        self.covered, self.generic, self.aliased = [], [], set()
        for name in dict.fromkeys((*self.module.NATIVE_ARRAY_BINDINGS.values(), 'kzq_p')):
            stored = facade._host.variable_storage(name)
            if stored is None:
                continue
            host, lower = stored
            self.names.append(name)
            self.lower[name] = lower
            _, packed = facade._view(name)
            if packed is not None and host.ndim and facade._covers(host):
                self.covered.append(name)
                if self._alias(packed, host) is not None:
                    self.aliased.add(name)
                    continue
            else:
                self.generic.append(name)
            self.whole[name] = cp.empty(host.shape, dtype=host.dtype, order='F')
        its, ite = metadata['its'], metadata['ite']
        kts, kte = metadata['kts'], metadata['kte']
        nc, nz = ite - its + 1, kte - kts + 1
        self.cells, self.levels = slice(its - 1, ite), slice(kts - 1, kte)
        self.tiles, self.j = slice(metadata['jts'] - 1, metadata['jte']), metadata['jte'] - 1
        self.nc, self.nz = nc, nz
        lw = self._pool('tend_physics', 'rthratenlw')
        density = self._pool('mesh', 'meshDensity')
        self.lw = cp.empty((nc, nz), dtype=lw.dtype)
        self.sw = cp.empty((nc, nz), dtype=lw.dtype)
        self.radiation = cp.empty((nc, nz), dtype=cp.float32)
        self.density = cp.empty(nc, dtype=density.dtype)
        self.dx = cp.empty(nc, dtype=cp.float32)
        self.pool_fields = (_pool_fields(metadata, extra_surface_fields=self.extra_surface_fields)
                            + _chemistry_pool_fields(metadata, options))
        self._signature = None
        register = getattr(provider, 'on_prepared_release', None)
        if register is not None:
            # Drop held prepared views and plans when they are released; rebind on the next call
            # (NativeDevicePoolViews.on_prepared_release says why).
            register(self._drop_prepared)

    def _drop_prepared(self):
        self._signature = None
        self.views = self.plans = None
        for name in self.aliased:
            self.whole.pop(name, None)

    def _alias(self, view, host):
        """The provider's k,i view as this array's Fortran-order whole allocation, or None if the layouts differ."""
        import math
        cp = self.cp
        if (view.dtype != host.dtype or not view.flags.c_contiguous or view.size != math.prod(host.shape)
                or view.size == 0):
            return None
        strides, step = [], view.itemsize
        for extent in host.shape:
            strides.append(step)
            step *= extent
        whole = cp.ndarray(host.shape, dtype=host.dtype, memptr=view.data, strides=tuple(strides))
        core = self.facade._core(whole, host)
        if core.shape != view.shape or core.strides != view.strides or core.data.ptr != view.data.ptr:
            return None
        return whole

    # ------------------------------------------------------------------ authorities
    def _options(self, call):
        options = dict(self.provider.session.layout.configs)
        options.update(call.options)
        return options

    def _pool(self, pool, name):
        return self.facade._authority(pool, name, 1)

    def _resolve(self):
        """The device authorities this call reads and writes; a rebuild follows any change of storage."""
        self.views = {name: self.provider.prepared(name) for name in self.covered}
        signature = tuple(self.views[name].data.ptr for name in self.covered)
        self.pools = {}
        for pool, name in self.pool_fields:
            self.pools[pool, name] = self._pool(pool, name)
        signature += tuple((key, array.data.ptr, array.shape) for key, array in self.pools.items())
        return signature

    def _build(self, options):
        cp = self.cp
        facade, W = self.facade, self.whole
        cells, levels, tiles, j = self.cells, self.levels, self.tiles, self.j
        diag = lambda name: self.pools['diag_physics', name]
        sfc = lambda name: self.pools['sfc_input', name]
        tend = lambda name: self.pools['tend_physics', name]
        meta = self.metadata
        # 1. whole allocations from the provider's k,i cores (covering arrays), and the driver kernels' inputs
        pack = CopyPlan('rrfs mynnedmf pack')
        for name in self.covered:
            host = facade._host.variable_storage(name)[0]
            if name in self.aliased:
                W[name] = self._alias(self.views[name], host)
                if W[name] is None:
                    raise RuntimeError(f'MYNN-EDMF {name}: the provider view no longer covers its whole allocation')
                continue
            pack.copy(facade._core(W[name], host), self.views[name])
        pack.copy(self.lw, tend('rthratenlw')[levels, cells].T)
        pack.copy(self.sw, tend('rthratensw')[levels, cells].T)
        pack.copy(self.density, self._pool('mesh', 'meshDensity')[cells])
        # 2. pbl_from_mpas tile writes, bl_mynnedmf branch (driver_rrfs.pbl_from_mpas, same statements in order)
        tile = CopyPlan('rrfs mynnedmf from_mpas')
        targets = []

        def surface(name, value):
            targets.append(name)
            dst = W[name][cells, tiles]
            if isinstance(value, cp.ndarray):
                tile.copy(dst, value[:, None])
            else:
                tile.fill(dst, value)

        def volume(name, value):
            targets.append(name)
            dst = W[name][cells, levels, tiles]
            if isinstance(value, cp.ndarray):
                tile.copy(dst, value[:, :, None])
            else:
                tile.fill(dst, value)

        for field in ('hfx', 'hpbl', 'qfx', 'ust', 'wspd', 'znt'):
            surface(field + '_p', diag(field)[cells])
        surface('xland_p', sfc('xland')[cells])
        surface('kpbl_p', cp.int32(1))
        for name in ('uoce_p', 'voce_p'):
            surface(name, cp.float32(0))
        volume('rthraten_p', self.radiation)
        surface('dx_p', self.dx)
        for field in ('ch', 'qsfc'):
            surface(field + '_p', diag(field)[cells])
        surface('tsk_p', sfc('skintemp')[cells])
        from .driver_rrfs import _PBL_VOLUME, _PBL_COMMON_TEND, _PBL_DIAG_TEND, _PBL_NUMBER_TEND
        for name, field in _PBL_VOLUME.items():
            volume(name, diag(field)[levels, cells].T)
        for name in ('excessh_p', 'excessq_p', *(n + '_p' for n in self.extra_surface_fields)):
            surface(name, cp.float32(0))
        for field in (*_PBL_DIAG_TEND, *_PBL_NUMBER_TEND):
            volume(field + '_p', cp.float32(0))
        volume('pattern_spp_pbl', cp.float32(0))
        for name in ('maxwidthbl_p', 'maxmfbl_p', 'zbl_plume_p'):
            surface(name, cp.float32(0))
        for field in (*_PBL_COMMON_TEND, 'kzh', 'kzm', 'kzq'):
            volume(field + '_p', cp.float32(0))
        if meta['do_chemistry']:
            # driver_rrfs.pbl_from_mpas chemistry branch (driver_pbl.F:1007-1016), same statements in order.
            nchem = int(meta['num_chem'])
            start = int(self.provider.session.layout.indices['state', 'chemistry_start']) - 1
            chemistry = self.pools['state', 'scalars'][start:start + nchem, levels, cells]
            targets.append('chem_p')
            tile.copy(W['chem_p'][cells, levels, tiles, :nchem], chemistry.transpose(2, 1, 0)[:, :, None, :])
            if options['config_mynn_enh_mix']:
                if options['config_smoke_scheme'].strip() == 'on':
                    surface('frp_out_p', diag('frp_out')[cells])
                else:
                    surface('frp_out_p', cp.float32(0))
                if (options['config_anthro_scheme'].strip() != 'off'
                        or options['config_rwc_scheme'].strip() != 'off'):
                    surface('aero_emis_for_enhmix_p', diag('aero_emis_for_enhmix')[cells])
                else:
                    surface('aero_emis_for_enhmix_p', cp.float32(0))
            targets += ['ddvel_p', 'tend_chem_settle_p']
            if int(options['drydep_opt']) != 0:
                tile.copy(W['ddvel_p'][cells, tiles, :nchem], diag('ddvel')[:nchem, cells].T[:, None, :])
                tile.copy(W['tend_chem_settle_p'][cells, levels, tiles, :nchem],
                          tend('tend_chem_settle')[:nchem, levels, cells].transpose(2, 1, 0)[:, :, None, :])
            else:
                # Native ddvel assignment spans the complete species allocation.
                if nchem:
                    tile.fill(W['ddvel_p'][cells, tiles, :], cp.float32(0))
                tile.fill(W['tend_chem_settle_p'][cells, levels, tiles, :nchem], cp.float32(0))
        if len(set(targets)) != len(targets):
            raise RuntimeError('pbl_from_mpas writes one array twice; the fused tile plan needs distinct targets')
        # 4. pbl_to_mpas pool writes (single j) and put_variable of every covering array
        unpack = CopyPlan('rrfs mynnedmf to_mpas')

        def out_surface(field, name=None):
            unpack.copy(diag(field)[cells], W[name or field + '_p'][cells, j])

        def out_volume(pool, field, name=None):
            unpack.copy(self.pools[pool, field][levels, cells], W[name or field + '_p'][cells, levels, j].T)

        for field in ('hpbl', 'kpbl'):
            out_surface(field)
        if meta['do_chemistry']:
            # driver_rrfs.pbl_to_mpas: the chemistry species back into scalars (driver_pbl.F:1007-1016).
            nchem = int(meta['num_chem'])
            start = int(self.provider.session.layout.indices['state', 'chemistry_start']) - 1
            unpack.copy(self.pools['state', 'scalars'][start:start + nchem, levels, cells],
                        W['chem_p'][cells, levels, j, :nchem].transpose(2, 1, 0))
        for field in _PBL_COMMON_TEND:
            out_volume('tend_physics', field)
        for field, name in (('kzh', 'kzh_p'), ('kzm', 'kzm_p'), ('kzq', 'kzh_p')):
            out_volume('diag_physics', field, name)
        for name, field in _PBL_VOLUME.items():
            out_volume('diag_physics', field, name)
        for field in _PBL_DIAG_TEND:
            out_volume('diag_physics', field)
        out_volume('tend_physics', 'rqsblten')
        if meta['f_ni']:
            out_volume('tend_physics', 'rniblten')
        if meta['f_nc'] and meta['f_nifa'] and meta['f_nwfa']:
            for field in ('rncblten', 'rnifablten', 'rnwfablten'):
                out_volume('tend_physics', field)
        for field, name in (('maxmf', 'maxmfbl_p'), ('maxwidth', 'maxwidthbl_p'), ('ztop_plume', 'zbl_plume_p'),
                            ('excess_h', 'excessh_p'), ('excess_q', 'excessq_p'),
                            *((n, n + '_p') for n in self.extra_surface_fields)):
            out_surface(field, name)
        for name in self.covered:
            if name in self.aliased:
                continue
            host = facade._host.variable_storage(name)[0]
            unpack.copy(self.views[name], facade._core(W[name], host))
        self.plans = (pack.build(), tile.build(), unpack.build())

    # ------------------------------------------------------------------ one call
    def run(self, call):
        import numpy as np
        cp = self.cp
        options = self._options(call)
        signature = self._resolve()
        if signature != self._signature:
            self._build(options)
            self._signature = signature
        pack, tile, unpack = self.plans
        W, facade = self.whole, self.facade
        # Arrays outside the provider's cores: the facade's own whole allocation (tail or host authority).
        for name in self.generic:
            whole, _ = facade.get_variable(name)
            W[name][...] = cp.asarray(whole)
        pack()
        nc, nz = self.nc, self.nz
        self._radiation(((nc * nz + 127) // 128,), (128,), (np.int32(nc * nz), self.lw, self.sw, self.radiation))
        self._spacing(((nc + 127) // 128,), (128,),
                      (np.int32(nc), self.density, np.float32(options['config_len_disp']), self.dx))
        tile()
        arguments = self.module.native_driver_arguments(W, self.metadata, options, itimestep=call.itimestep)
        result = self.backend.run(arguments)
        # Read off the step path (deferred_checks_v842), as run_slot does.
        from ..deferred_checks_v842 import defer
        defer((result['errflg'] != 0).astype(cp.int32), 'MYNN-EDMF CUDA wrapper returned a native error flag')
        unpack()
        for name in self.generic:
            facade.put_variable(name, W[name], lower=self.lower[name])
        return result


def _chemistry_pool_fields(flags, options):
    """The pool fields the chemistry branch of pbl_from_mpas/pbl_to_mpas reads or writes (bind_native_pools)."""
    if not flags['do_chemistry']:
        return ()
    rows = [('state', 'scalars')]
    if int(options['drydep_opt']) != 0:
        rows += [('diag_physics', 'ddvel'), ('tend_physics', 'tend_chem_settle')]
    if options['config_mynn_enh_mix']:
        if options['config_smoke_scheme'].strip() == 'on':
            rows.append(('diag_physics', 'frp_out'))
        if options['config_anthro_scheme'].strip() != 'off' or options['config_rwc_scheme'].strip() != 'off':
            rows.append(('diag_physics', 'aero_emis_for_enhmix'))
    return tuple(rows)


# Pool fields pbl_from_mpas and pbl_to_mpas read or write for bl_mynnedmf without chemistry (bind_native_pools binds
# these and the same flag-dependent number tendencies).
def _pool_fields(flags, extra_surface_fields=()):
    from .driver_rrfs import _PBL_VOLUME, _PBL_COMMON_TEND, _PBL_DIAG_TEND
    diag = ['hfx', 'hpbl', 'qfx', 'ust', 'wspd', 'znt', 'ch', 'qsfc', 'kpbl', 'kzh', 'kzm', 'kzq',
            *_PBL_VOLUME.values(), *_PBL_DIAG_TEND, 'maxmf', 'maxwidth', 'ztop_plume', 'excess_h', 'excess_q',
            *extra_surface_fields]
    tend = ['rthratenlw', 'rthratensw', *_PBL_COMMON_TEND, 'rqsblten']
    if flags['f_ni']:
        tend.append('rniblten')
    if flags['f_nc'] and flags['f_nifa'] and flags['f_nwfa']:
        tend += ['rncblten', 'rnifablten', 'rnwfablten']
    return tuple(dict.fromkeys([('diag_physics', n) for n in diag] + [('tend_physics', n) for n in tend]
                               + [('sfc_input', 'xland'), ('sfc_input', 'skintemp'), ('mesh', 'meshDensity')]))
