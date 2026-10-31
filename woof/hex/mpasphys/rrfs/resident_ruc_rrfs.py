"""Resident RUC slot glue (RRFS hrrrv5 route): land, sea ice and surface diagnostics on persistent work arrays.

Authority: the transaction of ruc_lsm_8c9f1a8._bind_native_ruc + execute_native_context (ResidentRucProvider.
prepare_ruc / launch_boundary / finish_ruc / synchronize, then put_field and put_variable of every bound field) on the
resident framework facade (resident_rrfs.ResidentFrameworkFacade). Fork sources at 1d6e370e:
mpas_atmphys_driver_lsm.F:388-1036, mpas_atmphys_driver_seaice.F:253-817, mpas_atmphys_sfc_diagnostics.F:32-145.

THE COST THIS REMOVES: each RUC call rebuilt everything: a copy of every bound pool field and module array, the 955
table words re-read from the framework and uploaded, a fresh 168 MB workspace, every per-cell scalar argument
(cp.full), every work array, a pointer table and presence flags uploaded per boundary launch, two blocking reads
(`isice`/`iswater` .item() from device words), and a write-back of every bound field: 292-323 kernels and 234-246
copies (27-30 host-to-device, 2 device-to-host) per land or sea-ice call, 6.7-7.9 ms of host per call (SPEED-2 profile
p2 of c2efafcf, hrrrv5 x1.2562, RTX 5090, 2026-10-28). Here the constants, tables, pointer tables and work arrays are
made once, and each call is one fused gather launch, the same driver boundary kernels, the scheme kernel, one fused
scatter launch and one deferred error check.

The scheme reads and writes the same words: work arrays receive the same pool and module words (copies, broadcasts and
fills only, no conversion), the boundary kernels and the scheme run unchanged on them, and the scatter writes the same
outputs to the same pool words in the same last-writer order. Pool fields and module arrays are written in place
instead of through copies that were written back whole, which leaves every unchanged word as it was. Proof: the
in-process A/B of old and new slot on real hrrrv5 inputs (SPEED-2/glue.md).
"""
from __future__ import annotations

import numpy as np

from . import ruc_lsm_8c9f1a8 as ruc
from .glue_plan import CopyPlan

#: Off only in the A/B measurement harness, which runs the previous glue on the same captured state.
FUSED = True

_LAND, _ICE, _DIAG = 'land_ruc_land', 'ice_ruc_ice', 'diags_sfcdiags_ruclsm'


class _Unsupported(Exception):
    """A binding this resident slot does not cover; the previous glue runs it."""


def resident_slot(provider, call, entry):
    """The persistent slot for the resident facade route, or None (native library route, or unsupported)."""
    from .resident_rrfs import ResidentFrameworkFacade
    if not FUSED or not isinstance(getattr(provider, 'native', None), ResidentFrameworkFacade):
        return None
    slots = provider.__dict__.setdefault('_rrfs_ruc_resident', {})
    slot = slots.get(entry)
    if slot is None:
        try:
            slot = ResidentRucSlot(provider, call, entry)
        except _Unsupported:
            slot = False
        slots[entry] = slot
    return slot or None


def _scalars(facade, provider, call, entry):
    """_bind_native_ruc's scalar block: options, framework constants and module integers (host words)."""
    options = dict(provider.session.layout.configs)
    options.update(call.options)
    scalars = {}
    landuse = None
    if entry in (_LAND, _ICE):
        landuse = facade.get_text_field('sfc_input', 'mminlu')
        scalars.update(myjpbl=0, frpcpn=1, rdlai2d=1,
                       micro_on=options['config_microp_scheme'].strip() != 'off',
                       conv_on=options['config_convection_scheme'].strip() != 'off',
                       restart=options['config_do_restart'], cycling=options['config_do_DAcycling'],
                       mminlu={'USGS': 1, 'MODIS': 2, 'MODIFIED_IGBP_MODIS_NOAH': 3}[landuse])
        if entry == _LAND:
            scalars['lakemodel'] = options['config_lakemodel']
        else:
            scalars.update(config_frac_seaice=options['config_frac_seaice'],
                           mynn=options['config_sfclayer_scheme'].strip() == 'sf_mynnsfclay')
            # Parameter in MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_vars.F:824, not a runtime export.
            scalars['seaice_albedo_default'] = np.float32('0.65')
    else:
        scalars['mynn'] = options['config_sfclayer_scheme'].strip() == 'sf_mynnsfclay'
    for field in ruc.ARGUMENTS[entry]:
        name = field['name']
        if field['dim'] or name in scalars or name in ('spp_lsm', 'mosaic_lu', 'mosaic_soil',
                                                         'dt', 'ktau', 'frpcpn', 'myjpbl', 'rdlai2d'):
            continue
        scalars[name] = ruc._native_scalar(provider, name)
    scalars.update(its=provider.session.arguments.its, ite=provider.session.arguments.ite)
    return scalars, landuse


class _Boundary:
    """One native entry launch on persistent arguments: launch_boundary's pointer table, built once."""

    def __init__(self, entry, arguments, tables, workspace, errors, present=None, scalars=None):
        import cupy as cp
        layout = ruc.ARGUMENTS[entry]
        n = arguments[layout[0]['name']].shape[-1]
        pointers = []
        for field in layout:
            a = arguments[field['name']]
            dtype = np.dtype('float32' if field['type'] == 'float' else 'int32')
            if not isinstance(a, cp.ndarray) or a.dtype != dtype or not a.flags.c_contiguous or a.shape[-1] != n:
                raise ValueError(f"{field['name']} must be a contiguous CUDA {dtype} array with {n} cells")
            pointers.append(a.data.ptr)
        self.presence = []
        for field in layout:
            if field['optional']:
                flag = cp.full(n, np.int32((present or {}).get(field['name'], True)), dtype=cp.int32)
                pointers.append(flag.data.ptr)
                self.presence.append(flag)
        if not isinstance(tables, cp.ndarray) or tables.dtype != cp.float32 or tables.size != ruc.TABLE_WORDS:
            raise ValueError('RUC parameter tables must be 955 resident float32 words')
        rows, bound = ruc.arena_rows(entry, arguments, scalars)
        self.rows = rows
        if workspace is not None:
            self._check(workspace, n)
        # Arena offsets are int32 (launch_boundary): past 2**31 words a cell addresses another cell's scratch.
        if n * (ruc.WORK_WORDS if bound is None else bound) >= 2**31:
            raise ValueError(f'RUC launch of {n} cells x {ruc.WORK_WORDS if bound is None else bound} arena words '
                             'overflows the kernel int32 arena offsets; run this tile on more ranks')
        self.tile_stop = None
        if entry == _LAND:
            # Native L501 exits the whole tile; L505-582 mutates preceding cells (launch_boundary).
            self.tile_stop = cp.empty(2, dtype=cp.int32)
            pointers.append(self.tile_stop.data.ptr)
        if entry in (_LAND, _ICE):
            pointers += [0, 0]   # no native-capture entry storage, no entry-call trace on the forecast path
        self.entry, self.n = entry, n
        self.arguments, self.tables, self.workspace, self.errors = arguments, tables, workspace, errors
        self.ptrs = cp.asarray(pointers, dtype=cp.uint64)
        module = ruc._module(cp.cuda.Device().id, False)
        self.preflight = module.get_function('ruc_land_tile_preflight') if entry == _LAND else None
        self.kernel = module.get_function('oracle_' + entry)

    def _check(self, workspace, n):
        import cupy as cp
        if (workspace.ndim != 2 or workspace.shape[1] != n or workspace.shape[0] < self.rows
                or workspace.dtype != cp.float32 or not workspace.flags.c_contiguous):
            raise ValueError(f'RUC workspace must have shape (>= {self.rows} arena words, {n} cells), float32 and '
                             'contiguous storage')

    def __call__(self, workspace=None):
        n = self.n
        if n == 0:
            return
        if workspace is not None:
            self._check(workspace, n)
        else:
            workspace = self.workspace
        if self.preflight is not None:
            self.preflight((1,), (1,), (self.ptrs, self.tile_stop, np.int32(n)))
        # One warp per block: the land/ice kernels hold 254-255 registers per thread and one independent column per
        # thread (no shared memory, barriers or warp intrinsics, so the block size cannot change a word). In 64-thread
        # blocks the x1.2562 tile (2,562 cells) sat on 41 of the RTX 5090's 170 SMs, two warps each; one warp per
        # block spreads it over 81 SMs: land 2.685 -> 2.416 ms, sea ice 2.205 -> 1.998 ms per call (CUDA events,
        # hrrrv5 6 h fast, quiet card, SPEED-2 glue lane 2026-10-28).
        self.kernel(((n + 31) // 32,), (32,), (self.ptrs, self.tables, workspace, self.errors, np.int32(n)))


class ResidentRucSlot:
    """One RUC entry (land, sea ice or surface diagnostics) per call on persistent device storage."""

    def __init__(self, provider, call, entry):
        facade = provider.native
        self.provider, self.facade, self.cp, self.entry = provider, facade, provider.cp, entry
        self.scalars, landuse = _scalars(facade, provider, call, entry)
        self.dt = ruc._native_interval(provider) if entry != _DIAG else None
        cp = self.cp
        tables = provider.__dict__.setdefault('_rrfs_ruc_tables', {})
        if landuse is None:
            # Surface diagnostics take the common pointer but never dereference the tables (_bind_native_ruc).
            self.tables = tables.get(None)
            if self.tables is None:
                self.tables = tables[None] = cp.empty(ruc.TABLE_WORDS, cp.float32)
        else:
            self.tables = tables.get(landuse)
            if self.tables is None:
                self.tables = tables[landuse] = ruc._native_ruc_tables(facade, cp, landuse)
        self._signature = None
        self._built = None
        self._init_keys = frozenset(self.__dict__) | {'_init_keys'}
        register = getattr(provider, 'on_prepared_release', None)
        if register is not None:
            # Drop held prepared views and plans when they are released; rebind on the next call
            # (NativeDevicePoolViews.on_prepared_release says why).
            register(self._drop_prepared)

    def _drop_prepared(self):
        # Everything _resolve/_build made binds the prepared views; the next call rebuilds it.
        for name in [k for k in self.__dict__ if k not in self._init_keys]:
            del self.__dict__[name]
        self._signature = None

    # ------------------------------------------------------------------ authorities
    def _pool(self, pool, name):
        if pool == 'physics':
            try:
                view = self.provider.prepared(name)
            except ValueError as exc:
                raise _Unsupported(name) from exc
            # _bind_native_ruc's layout: Fortran i,k,j -> k,i with singleton axes dropped, cells last.
            return view.reshape(tuple(d for d in view.shape[:-1] if d != 1) + (view.shape[-1],))
        return self.facade._authority(pool, name, 1)

    def _cells(self):
        return slice(self.scalars['its'] - 1, self.scalars['ite'])

    def _resolve(self):
        """Current device authorities of every bound field; storage changes trigger a rebuild."""
        self.pools = {key: self._pool(*key) for key in self._fields()}
        return tuple((key, a.data.ptr, a.shape) for key, a in self.pools.items())

    def _fields(self):
        """The pool fields and module arrays this entry reads or writes (_bind_native_ruc's required sets)."""
        entry, scalars = self.entry, self.scalars
        required = [('sfc_input', 'skintemp')]
        for field in ruc.ARGUMENTS[entry]:
            name = field['name']
            if not field['dim']:
                if name in ('spp_lsm', 'mosaic_lu', 'mosaic_soil'):
                    required.append(('diag_physics', 'spp_lsm_loc' if name == 'spp_lsm' else name))
            elif name in ruc.ATM_BINDINGS:
                required.append(('physics', ruc.ATM_BINDINGS[name]))
            elif name == 'globalcells':
                required.append(('mesh', 'indexToCellID'))
            elif entry == _DIAG and name == 'psfc2d':
                required.append(('diag', 'surface_pressure'))
            elif name in ruc.SFC_BINDINGS:
                required.append(('sfc_input', ruc.SFC_BINDINGS[name]))
            elif name not in ('pattern_spp_lsm', 'field_sf', 'lakemask', 'rainbl', 'frzfrac'):
                required.append(('diag_physics', 'qcg' if entry == _LAND and name == 'qsg'
                                 else ruc.DIAG_BINDINGS.get(name, name)))
        for field in ruc.ARGUMENTS[entry]:
            name = field['name']
            if not field['dim'] or field['intent'] == 'in' or name in ('pattern_spp_lsm', 'field_sf'):
                continue
            if entry == _ICE and name not in ruc.ICE_COPYBACK:
                continue
            required.append(('sfc_input', ruc.SFC_BINDINGS[name]) if name in ruc.SFC_BINDINGS
                            else ('diag_physics', ruc.DIAG_BINDINGS.get(name, name)))
        if entry in (_LAND, _ICE):
            required += [('sfc_input', 'zs'), ('diag_physics', 'gsw'), ('diag_physics', 'sfc_albedo')]
            if scalars['micro_on']:
                required += [('diag_physics', 'sr'), ('diag_physics', 'rainncv')]
            if scalars['conv_on']:
                required.append(('diag_physics', 'raincv'))
        if entry == _DIAG:
            required += [('diag_physics', 'cqs'), ('diag_physics', 'chs')]
        if entry == _ICE:
            required += [('diag_physics', name) for name in ruc.ICE_FINISH_FIELDS if name != 'tsk']
            required += [('physics', name + '_sea') for name in ruc.ICE_FINISH_FIELDS]
            required += [('sfc_input', 'sfc_albbck'), ('diag_physics', 'sfc_emiss')]
        return tuple(dict.fromkeys(required))

    # ------------------------------------------------------------------ build (once per storage layout)
    def _build(self):
        cp, entry, scalars, P = self.cp, self.entry, self.scalars, self.pools
        cells = self._cells()
        sfc = lambda name: P['sfc_input', name]
        diag = lambda name: P['diag_physics', name]
        physics = lambda name: P['physics', name]
        n = sfc('skintemp')[..., cells].shape[-1]
        self.n = n
        # Arena scratch is taken from CuPy's pool for each call (run), as launch_boundary did, sized to the arena
        # words the entry can address (ruc.arena_words), not WORK_WORDS.  THE BREAKAGE THIS PREVENTS: one persistent
        # (WORK_WORDS, cells) workspace held 64 KiB per cell for the whole run, so every other phase (MYNN-EDMF,
        # radiation, dynamics) peaked on top of it; between RUC calls the pool hands the same words to them.
        workspace = None
        gather = CopyPlan('rrfs ruc gather ' + entry)
        scatter = CopyPlan('rrfs ruc scatter ' + entry)
        errors = cp.empty(5, dtype=cp.int32)   # one fatal word per boundary launch of this entry
        gather.fill(errors, 0)
        self.per_call = []   # (array, callable) per-call fills the gather table cannot hold (ktau)
        self.dt_words = []
        launches = []

        def scalar(name, value, kind):
            array = cp.empty(n, dtype='float32' if kind == 'float' else 'int32')
            gather.fill(array, value)
            return array

        # ---- prepare_ruc
        result = {}
        for field in ruc.ARGUMENTS[entry]:
            name = field['name']
            if name in ruc.ATM_BINDINGS:
                source = physics(ruc.ATM_BINDINGS[name])[..., cells]
            elif entry == _DIAG and name == 'psfc2d':
                source = P['diag', 'surface_pressure'][..., cells]
            elif name == 'globalcells':
                source = P['mesh', 'indexToCellID'][..., cells]
            elif name in ('pattern_spp_lsm', 'field_sf'):
                result[name] = cp.empty((scalars['kme'] - scalars['kms'] + 1, n), cp.float32)
                gather.fill(result[name], 0)
                continue
            elif name == 'lakemask':
                result[name] = cp.empty(n, cp.float32)
                gather.fill(result[name], 0)
                continue
            elif name in ('rainbl', 'frzfrac'):
                result[name] = cp.empty(n, cp.float32)
                continue
            elif name in ruc.SFC_BINDINGS:
                if entry == _ICE and name == 'albbck':
                    continue   # _prepare_ice replaces this work array with the background albedo below
                source = sfc(ruc.SFC_BINDINGS[name])[..., cells]
                if name == 'zs' and entry in (_LAND, _ICE):
                    # Native zs work profile is taken from the first tile cell.
                    result[name] = cp.empty(source.shape, source.dtype)
                    gather.copy(result[name], cp.broadcast_to(sfc('zs')[:, cells][:, :1], source.shape))
                    continue
            elif field['dim']:
                source = diag('qcg' if entry == _LAND and name == 'qsg' else ruc.DIAG_BINDINGS.get(name, name))[
                    ..., cells]
            else:
                if name in ('spp_lsm', 'mosaic_lu', 'mosaic_soil'):
                    flag = diag('spp_lsm_loc' if name == 'spp_lsm' else name)
                    if flag.ndim:
                        flag = flag[..., cells]
                    if flag.dtype != cp.int32:
                        raise _Unsupported(f'{name} flag dtype {flag.dtype}')
                    result[name] = cp.empty(n, cp.int32)
                    gather.copy(result[name], cp.broadcast_to(flag, (n,)))
                    continue
                kind = field['type']
                if name == 'ktau':
                    result[name] = cp.empty(n, dtype='float32' if kind == 'float' else 'int32')
                    self.per_call.append(result[name])
                    continue
                if name == 'dt':
                    result[name] = scalar(name, self._dt, kind)
                    continue
                value = 1 if name in ruc.COLUMN_EXTENTS else scalars[name]
                if entry in (_LAND, _ICE) and name in ('myjpbl', 'frpcpn', 'rdlai2d'):
                    value = {'myjpbl': 0, 'frpcpn': 1, 'rdlai2d': 1}[name]   # prepare_ruc's fixed driver flags
                result[name] = scalar(name, value, kind)
                continue
            # Atmospheric profiles are intent(in) in all three entries and
            # no preparation boundary writes them. Bind the resident words
            # directly when the column slice satisfies the kernel layout.
            # Partial regional slices retain the packed work array.
            if (name in ruc.ATM_BINDINGS and field['intent'] == 'in'
                    and source.flags.c_contiguous):
                result[name] = source
                continue
            # Other driver arguments retain independent work arrays.
            result[name] = cp.empty(source.shape, source.dtype)
            gather.copy(result[name], source)
        if entry == _DIAG and not scalars['mynn']:
            self.pre_copy = (diag('cqs'), diag('chs'))
        else:
            self.pre_copy = None
        self.background = None
        if entry in (_LAND, _ICE):
            forcing = {'gsw': diag('gsw')[..., cells], 'ncells': scalar('ncells', 1, 'int')}
            for name, flag in (('sr', 'micro_on'), ('rainncv', 'micro_on'), ('raincv', 'conv_on')):
                if scalars[flag]:
                    forcing[name] = diag(name)[..., cells]
                else:
                    forcing[name] = cp.empty(n, cp.float32)
                    gather.fill(forcing[name], 0)
            forcing['sfc_albedo'] = diag('sfc_albedo')[..., cells]
            forcing.update(sr_p=result['frzfrac'], rainbl_p=result['rainbl'], swdown_p=cp.empty(n, cp.float32))
            for flag in ('micro_on', 'conv_on'):
                forcing[flag] = scalar(flag, scalars[flag], 'int')
            launches.append(_Boundary('drivers_driver_forcing', forcing, self.tables, workspace, errors[0:1],
                                      scalars=scalars))
            if entry == _ICE:
                # _prepare_ice (mpas_atmphys_driver_seaice.F:373-458)
                background = cp.empty(n, cp.float32)
                gather.fill(background, scalars['seaice_albedo_default'])
                self.background = (background, sfc('sfc_albbck')[..., cells], result['xice'],
                                   result['xice_threshold'])
                result['albbck'] = background
                prep = {k: result[k] for k in ('xice_threshold', 'xice')}
                prep['ncells'] = scalar('ncells', 1, 'int')
                prep.update(sfc_albedo=diag('sfc_albedo')[..., cells], sfc_emiss=diag('sfc_emiss')[..., cells],
                            sfc_albedo_p=result['alb'], sfc_emiss_p=result['emiss'])
                launches.append(_Boundary('drivers_seaice_prepare', prep, self.tables, workspace, errors[1:2],
                                          scalars=scalars))
                correction = {k: result[k] for k in ('ims', 'ime', 'jms', 'jme', 'its', 'ite', 'jts', 'jte', 'xice')}
                correction.update(xice_thresh=result['xice_threshold'], tsk=result['soilt'],
                                  tsk_sea=physics('tsk_sea')[..., cells], tsk_ice=cp.empty(n, cp.float32))
                launches.append(_Boundary('drivers_correct_tsk_over_seaice', correction, self.tables, workspace,
                                          errors[2:3]))
                self.tsk_ice = (result['soilt'], correction['tsk_ice'])
                for name in ruc.ICE_FINISH_FIELDS:
                    if name not in ('tsk',) and name not in result:
                        result[name] = cp.empty_like(diag(name)[..., cells])
                        gather.copy(result[name], diag(name)[..., cells])
        main_error = errors[3:4]
        self.main = _Boundary(entry, result, self.tables, workspace, main_error, scalars=scalars,
                              present={'pattern_spp_lsm': entry != _ICE, 'field_sf': entry != _ICE})
        # ---- finish_ruc
        writes = {}

        def write(pool, name, source):
            target = P[pool, name][..., cells]
            writes.pop((pool, name), None)   # a later copyto of the same words wins
            writes[pool, name] = (target, source)

        self.blend = None
        if entry == _ICE:
            blend = {'ncells': scalar('ncells', 1, 'int'), 'xice_p': result['xice'],
                     'xice_threshold': result['xice_threshold'],
                     'sfc_albedo_p': result['alb'], 'sfc_emiss_p': result['emiss']}
            for flag in ('config_frac_seaice', 'mynn'):
                blend[flag] = scalar(flag, scalars[flag], 'int')
            for name in ruc.ICE_FINISH_FIELDS:
                blend[name + '_p'] = result['soilt' if name == 'tsk' else name]
                blend[name + '_sea'] = physics(name + '_sea')[..., cells]
            self.blend = _Boundary('drivers_seaice_finish', blend, self.tables, workspace, errors[4:5],
                                   scalars=scalars)
            for name in ('cqs', 'cpm', 'chs2', 'cqs2', 'qgh'):
                write('diag_physics', name, result[name])
        for field in ruc.ARGUMENTS[entry]:
            name = field['name']
            if not field['dim']:
                continue
            if entry == _ICE:
                if name not in ruc.ICE_COPYBACK:
                    continue
            elif field['intent'] == 'in':
                continue
            if name in ('pattern_spp_lsm', 'field_sf'):
                continue
            if name in ruc.SFC_BINDINGS:
                write('sfc_input', ruc.SFC_BINDINGS[name], result[name])
            else:
                write('diag_physics', ruc.DIAG_BINDINGS.get(name, name), result[name])
        if entry == _ICE:
            write('diag_physics', 'flqc', result['flhc'])
            write('sfc_input', 'sfc_albbck', result['albbck'])
        if entry == _LAND and scalars['micro_on']:
            write('diag_physics', 'sr', result['frzfrac'])
        for target, source in writes.values():
            scatter.copy(target, source)
        self.result, self.launches, self.errors = result, launches, errors
        self.rows = max(b.rows for b in (*launches, self.main, *((self.blend,) if self.blend is not None else ())))
        self.gather, self.scatter = gather.build(), scatter.build()

    @property
    def _dt(self):
        return self._call_dt

    # ------------------------------------------------------------------ one call
    def run(self, call):
        cp = self.cp
        dt = self.dt if self.dt is not None else call.dt
        signature = (self._resolve(), dt)
        if signature != self._signature:
            self._call_dt = dt
            self._build()
            self._signature = signature
        if self.pre_copy is not None:
            cp.copyto(*self.pre_copy)
        self.gather()
        for array in self.per_call:
            array.fill(call.itimestep)
        work = cp.empty((self.rows, self.n), cp.float32)
        boundaries = iter(self.launches)
        if self.entry in (_LAND, _ICE):
            next(boundaries)(work)                 # drivers_driver_forcing
        if self.background is not None:
            background, albbck, xice, threshold = self.background
            cp.copyto(background, albbck, where=xice < threshold)
            next(boundaries)(work)                 # drivers_seaice_prepare
            next(boundaries)(work)                 # drivers_correct_tsk_over_seaice
            cp.copyto(*self.tsk_ice)
        self.main(work)
        if self.blend is not None:
            self.blend(work)
        self.scatter()
        # Every word is set only at a source-native fatal-error call site; read off the step path.
        from ..deferred_checks_v842 import defer
        defer(self.errors, 'RUC reached a pinned native fatal-error branch')
        return {'errors': self.errors, 'arguments': self.result}
