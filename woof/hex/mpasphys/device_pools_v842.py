"""Prepared CUDA slot views; mpas_atmphys_driver.F:185-384.

The native driver allocates/prepares these pools before a slot and consumes its
changed words after it. This adapter copies ABI storage and owns solve/halo
extents; schemes retain all arithmetic and from/to conversions.
"""
from __future__ import annotations

from woof.hex.device_memory_release import shared_card_memory
from collections.abc import Mapping
from dataclasses import replace
import importlib
import json
from pathlib import Path
from typing import Any

from ..device_memory_release import phase_release
from ..physics_suites import load_table
from .driver_v842 import Backend


class NativeDevicePoolViews(Mapping):
    """Lazy solve-cell views of native ABI storage; mpas_atmphys_interface.F:284-425."""

    # MPAS driver allocates the native prepared arrays before slots: mpas_atmphys_driver.F:185-209.
    def __init__(self, session, device):
        self.session, self.native, self.cp = session, session.native, device
        self.buffers = {}
        self.values = {}
        self.level = 1
        self.resident = False
        self.resident_buffers = {}
        self._field_rows = {}
        self.transfers = {'initial_upload_bytes': 0, 'publication_download_bytes': 0}
        # Prepared work arrays given back between physics calls (release_prepared): key -> (host, shape, dtype,
        # lower).  _prepared_nonzero names the ones whose first upload was not all zeros; they stay resident.
        self._released_prepared = {}
        self._prepared_nonzero = set()
        # release_prepared bumps the epoch and runs these hooks: holders of prepared views drop them, so the arrays
        # really go back to the pool, and rebind to the new arrays on their next call (on_prepared_release).
        self.prepared_epoch = 0
        self._release_hooks = []

    def __iter__(self):
        """Named pools follow Registry.xml:1360-4020."""
        return iter((*self.session.layout.pools,'configs','constants','scalar_state','prepared'))

    def __len__(self):
        """Registry pool enumeration; Registry.xml:1360-4020."""
        return len(self.session.layout.pools)+4

    def __getitem__(self, name):
        """Expose native fields or prepared globals; mpas_atmphys_vars.F:68-955."""
        if name in self.values:return self.values[name]
        if name=='configs':return self.session.layout.configs
        if name=='constants':return _ConstantViews(self.native)
        if name=='scalar_state':return _ScalarViews(self.native)
        if name=='prepared':return _PreparedViews(self)
        if name not in self.session.layout.pools:raise KeyError(name)
        if name not in self.values:self.values[name]=_PoolViews(self,name)
        return self.values[name]

    def field(self, pool, name, *, resident=False):
        """Pack solve cells or full storage; reject mixed cached extents.

        mpas_atmphys_interface.F:304-305. Both layouts must be contiguous:
        slicing a full multidimensional allocation cannot supply the packed
        solve layout while halo/dummy columns remain in its row stride.
        """
        if name in self.session.layout.dimensions:
            return self.session.layout.dimensions[name]
        row=self._field_rows.get((pool,name))
        if row is None:
            row=next((r for r in self.session.layout.fields
                      if r['pool']==pool and name in (r['name'],r['pool_name'])),None)
            if row is None:raise KeyError(f'{pool}.{name}')
            for alias in (row['name'],row['pool_name']):
                self._field_rows[pool,alias]=row
        key=pool,row['pool_name'],self.level if pool=='state' else 1
        entity=next((d for d in row['dimensions'] if d in ('nCells','nEdges','nVertices')),None)
        selection=(...,slice(0,self.session.layout.dimensions[entity+'Solve'])) if entity and not resident else (...,)
        cached=self.buffers.get(key)
        if cached is None and self.resident:
            cached=self.resident_buffers.get(key)
        if cached is not None:
            host,packed,cached_selection=cached
            requested_shape=host[selection].shape
            cached_shape=host[cached_selection].shape
            if cached_shape != requested_shape:
                raise ValueError(f'cached extent conflict for {pool}.{row["pool_name"]} level {key[2]}: '
                                 f'requested {requested_shape}, cached {cached_shape}; '
                                 'owned and full layouts cannot share a packed allocation')
            self.buffers[key]=cached
            return packed
        # The framework's live storage, not a copy: the views publish into it in place (field_storage).  The
        # native referee library has no such accessor and keeps its copies.
        storage=getattr(self.native,'field_storage',None)
        host=(storage(pool,row['pool_name'],level=key[2]) if storage is not None
              else self.native.get_field(pool,row['pool_name'],level=key[2]))
        selected=host[selection]
        self._before_resident_allocation()
        # ascontiguousarray promotes rank-zero arrays to length one. Registry
        # scalars must keep rank zero through publication and scheme bindings.
        packed=self.cp.ascontiguousarray(self.cp.asarray(selected)).reshape(selected.shape)
        self.buffers[key]=(host,packed,selection)
        if self.resident:
            self.resident_buffers[key] = self.buffers[key]
            self.transfers['initial_upload_bytes'] += packed.nbytes
        return packed

    def prepared(self, name):
        """Native i,k,j globals become k,i views; mpas_atmphys_vars.F:179-217; mpas_atmphys_vars.F:934-942."""
        key=('prepared',name)
        if key in self.buffers:return self.buffers[key][1]
        if self.resident and key in self.resident_buffers:
            self.buffers[key] = self.resident_buffers[key]
            return self.buffers[key][1]
        released = self._released_prepared.pop(key, None) if self.resident else None
        if released is not None:
            # Native allocates this array afresh per physics call (allocate_forall_physics and the per-driver
            # allocate_* routines, deallocated after the call); its first upload was all zeros, so a zeroed
            # device array is the same words as re-uploading it, without the host copy.
            host, shape, dtype, lower = released
            packed = self.cp.zeros(shape, dtype=dtype)
            self.buffers[key] = (host, packed, lower)
            self.resident_buffers[key] = self.buffers[key]
            return packed
        storage=getattr(self.native,'variable_storage',None)
        value=storage(name) if storage is not None else self.native.get_variable(name)
        if value is None:raise RuntimeError(f'native prepared array {name} is not allocated before CUDA slot')
        host,lower=value
        if host.ndim not in (2,3,4) or host.shape[2 if host.ndim==4 else -1]!=1 or lower[0]!=1:
            raise ValueError(f'native prepared {name} must have its declared i[,k],one-j bounds')
        count=self.session.layout.dimensions['nCellsSolve']
        # Surface globals are i,j; profiles are i,k,j. MPAS vars.F:934-942.
        view=host[:count,:,0,:].transpose(2,1,0) if host.ndim==4 else host[:count,:,0].T if host.ndim==3 else host[:count,0]
        self._before_resident_allocation()
        packed=self.cp.ascontiguousarray(self.cp.asarray(view))
        self.buffers[key]=(host,packed,lower)
        if self.resident:
            self.resident_buffers[key] = self.buffers[key]
            self.transfers['initial_upload_bytes'] += packed.nbytes
            if packed.size and bool(packed.any()):
                self._prepared_nonzero.add(key)
        return packed

    def on_prepared_release(self, hook):
        """Run ``hook()`` every time release_prepared gives the prepared work arrays back.

        THE BREAKAGE THIS PREVENTS (rrfs/c3-8card-speed 826f1d6, mv3km hrrrv5 1 card, 2026-10-29): the resident glue
        (driver preparation, surface layer, GWDO, radiation, MYNN-EDMF, RUC) holds views and word-copy plans of the
        prepared arrays across calls.  After a release the driver preparation, bound once, kept writing the old
        arrays while the next slots read the new zeroed ones (RUC took its pinned native fatal-error branch at step
        2), and every holder kept the old memory alive through the dycore, so the release gave little back.  Each
        holder registers a hook that drops what it holds; it rebinds to the current arrays on its next call.
        """
        self._release_hooks.append(hook)

    def _prepared_release_due(self):
        """Whether the prepared work arrays go back at this hand-off: by default always (release_prepared says why).

        1960569 gated the release on < 15 % free memory because the glue kept stale pointers after it; the glue now
        rebinds (on_prepared_release), so that gate is retired as the default.  The earlier gate, kept selectable:

        THE BREAKAGE THIS PREVENTS (rrfs/c3-8card-speed 826f1d6, mv3km hrrrv5 1 card, 2026-10-29): the speed3
        glue slots (surface layer, GWDO, radiation, driver) bind device pointers of prepared arrays once and
        reuse them every call; releasing the arrays at every physics-to-dycore hand-off gave them new addresses
        at the next call, the bound plans kept the old ones, and RUC took its pinned native fatal-error branch
        at step 2. The release exists for memory-tight shapes (NOAA conus3km on 4 x 96 GB cards, rank 1 out of
        memory in step 1), so it now runs only there: when the card has less than 15 % of its memory free
        (device free + this process's idle pool blocks). WOOF_HEX_RELEASE_PREPARED=always restores the
        release at every hand-off; =never keeps them resident. Memory-tight shapes still need the glue plans
        to rebind after a release (named follow-up).
        """
        import os
        mode = os.environ.get('WOOF_HEX_RELEASE_PREPARED', 'always')
        if mode == 'always':
            return True
        if mode == 'never':
            return False
        free, total = self.cp.cuda.runtime.memGetInfo()
        idle = self.cp.get_default_memory_pool().free_bytes()
        return free + idle < 0.15 * total

    def release_prepared(self):
        """Give the prepared physics work arrays back to the device pool between physics calls.

        Native allocates them per physics call and frees them after it (mpas_atmphys_driver.F:185-209
        allocate_forall_physics and deallocate_forall_physics, and the per-scheme allocate and deallocate
        routines); the resident route held all of them for the whole run.  THE BREAKAGE THIS PREVENTS (vram
        census 2026-10-28, RESULT.md: 86 physics work arrays held for the run, 21.4 KB per column): on NOAA
        conus3km over 4 x 96 GB cards (602k local columns per rank) the held arrays (12.9 GB per card) sat through
        every acoustic substep, the binding phase, and rank 1 ran out of memory in its first step.  The pool
        reuses the memory in the dycore; prepared() makes a zeroed array on the next request.  Arrays whose
        first upload was not all zeros stay resident.  Returns the bytes given back.
        """
        if not self.resident:
            return 0
        if not self._prepared_release_due():
            return 0
        import os
        audit = [] if os.environ.get('WOOF_HEX_RELEASE_PREPARED_AUDIT') else None
        freed = 0
        packed = None
        for key in [k for k in self.resident_buffers if k[0] == 'prepared' and k not in self._prepared_nonzero]:
            host, packed, lower = self.resident_buffers.pop(key)
            self.buffers.pop(key, None)
            self._released_prepared[key] = (host, packed.shape, packed.dtype, lower)
            freed += packed.nbytes
            if audit is not None:
                audit.append((key, packed))
        del packed
        if freed:
            self.prepared_epoch += 1
            for hook in self._release_hooks:
                hook()
        if audit:
            # Measurement only: a released array still referenced after the hooks is held by a binder that would
            # keep writing it (the RUC fatal) and keeps its memory from the dycore.
            import gc, sys
            seen = self.__dict__.setdefault('_audit_reported', set())
            for item in audit:
                key, array = item
                holders = sys.getrefcount(array) - 3
                if holders > 0 and key not in seen:
                    seen.add(key)
                    kinds = sorted({type(r).__name__ for r in gc.get_referrers(array)
                                    if r is not item and r is not audit})
                    print(f'[release-audit] prepared {key[1]} still referenced by {holders} holder(s): {kinds}',
                          file=sys.stderr, flush=True)
            del item, array
        return freed

    def _before_resident_allocation(self):
        """Keep persistent fields out of large, idle scheme scratch blocks.

        A first-use 36 MiB field can split a free GiB workspace and pin its
        remaining fragments for the whole forecast. At the measured dycore
        failure 1.19 GiB was idle but could not be returned by the pool. Drain
        available blocks before creating a new persistent field. This runs
        only on first allocation, never on a cached per-step lookup.
        """
        if not shared_card_memory():
            return
        factory = getattr(self.cp, 'get_default_memory_pool', None)
        if self.resident and factory is not None:
            factory().free_all_blocks()

    def write_field(self, pool, name, value, *, level=1):
        """Update the authoritative device allocation without a host round trip."""
        old = self.level
        self.level = level
        try:
            target = self.field(pool, name)
            source = self.cp.asarray(value)
            if source.shape == target.shape:
                target[...] = source
            elif (source.shape[:-1] == target.shape[:-1]
                  and (source.shape[-1] == target.shape[-1] + 1
                       or (getattr(self.session.layout, 'samehost_partition', False)
                           and source.shape[-1] > target.shape[-1]))):
                # The native dummy column, or on a same-host rank the halo
                # columns: owned entities come first in a rank's local order.
                target[...] = source[..., :target.shape[-1]]
            else:
                raise ValueError(f'resident {pool}.{name} extent differs: {source.shape} != {target.shape}')
        finally:
            self.level = old

    def publish(self):
        """Copy persistent pool words at an explicit output/checkpoint boundary.

        Prepared scheme scratch is not Registry output. Its native allocation
        is retained solely for ABI shape discovery and eventual finalization.
        """
        for key, (host, packed, selection) in self.resident_buffers.items():
            if key[0] == 'prepared':
                continue
            if packed.dtype in (self.cp.float32, self.cp.int32):
                value = packed.view(self.cp.uint32).get(order='F').view(packed.dtype)
            else:
                value = packed.get()
            host[selection] = value
            self.native.put_field(key[0], key[1], host, level=key[2],
                                  nlevels=self.native._fields[key[0], key[1]][2])
            self.transfers['publication_download_bytes'] += packed.nbytes

    def resident_pool(self, name):
        """Resident rings; RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710."""
        return _ResidentPoolViews(self,name)

    def exchange_halo_group(self, name):
        """Native group transaction; RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1336-1367."""
        members=self.halo_groups[name]
        self.cp.cuda.get_current_stream().synchronize()
        for field in members:
            pool,key=self.native._registered[field]
            host,packed,selection=self.buffers[pool,key,1]
            host[selection]=packed.get()
            self.native.put_field(pool,key,host,level=1,nlevels=self.native._fields[pool,key][2])
        self.native.exchange_halo_group(name)
        for field in members:
            pool,key=self.native._registered[field]
            host,packed,selection=self.buffers[pool,key,1]
            updated=self.native.get_field(pool,key,level=1)
            host[...]=updated
            packed[...] = self.cp.asarray(updated[selection])

    def begin(self, call, descriptor):
        """Bind a slot after native preparation; mpas_atmphys_driver.F:209-365."""
        self.buffers.clear();self.values.clear()
        self.halo_groups={g['name']:tuple(g['fields']) for g in descriptor.get('halo_groups',())}
        for group in descriptor.get('halo_groups',()):
            if group['name'] not in self.native._groups:
                self.native.exchange_group(group['name'],group['fields'],level=group['time_level'],
                                           halo_layers=group['halo_layers'])
        self.level=2 if call.slot=='microphysics' else 1
        self.values['constants']=_ConstantViews(self.native,descriptor.get('constant_aliases',{}))
        for pool, aliases in descriptor.get("profile_pools", {}).items():
            self.values[pool]=_ProfileViews(self,aliases)
        profiles=descriptor.get('profiles',{})
        if profiles:
            self.values['atm_input']=_ProfileViews(self,profiles)
        binder=descriptor.get('bind')
        if binder:
            module=importlib.import_module('.'+descriptor['module'],__package__)
            return getattr(module,binder)(self,call)
        return self

    def commit(self):
        """Return changed solve words without touching halos; mpas_atmphys_driver.F:209-384."""
        if self.resident:
            # Scheme adapters may retire input-only transaction entries. The
            # persistent authority retains those allocations independently.
            self.resident_buffers.update(self.buffers)
            return
        for key,(host,packed,selection) in self.buffers.items():
            if (key[0] != 'prepared' and isinstance(packed, getattr(self.cp, 'ndarray', ()))
                    and packed.dtype in (self.cp.float32, self.cp.int32)):
                # Ordinary pools retain Fortran order. Pack integer words on
                # the GPU before the existing transfer to preserve NaN payloads
                # and avoid a strided CPU assignment. Prepared globals below
                # use different transposed layouts and retain their old path.
                value=packed.view(self.cp.uint32).get(order='F').view(packed.dtype)
            else:
                value=packed.get() if hasattr(packed,'get') else packed
            if key[0]=='prepared':
                if host.ndim==4:
                    host[:value.shape[2],:,0,:]=value.transpose(2,1,0)
                elif host.ndim==3:
                    host[:value.shape[1],:,0]=value.T
                else:
                    host[:value.shape[0],0]=value
                self.native.put_variable(key[1],host,lower=selection)
            else:
                host[selection]=value
                self.native.put_field(key[0],key[1],host,level=key[2],
                                      nlevels=self.native._fields[key[0],key[1]][2])
        self.buffers.clear()


class _PoolViews(Mapping):
    """Native pool keys, Registry.xml:1360-4020."""
    def __init__(self, owner, pool):
        """Pool allocation ownership; mpas_pool_routines.F:2110-3143."""
        self.owner,self.pool=owner,pool
    def __iter__(self):
        """Enumerate actual Registry names; Registry.xml:1360-4020."""
        return iter(r['pool_name'] for r in self.owner.session.layout.fields if r['pool']==self.pool)
    def __len__(self):
        """Registry field count; Registry.xml:1360-4020."""
        return sum(1 for _ in self)
    def __getitem__(self,name):
        """Access solve-cell storage; mpas_atmphys_interface.F:304-305."""
        return self.owner.field(self.pool,name)


class _PreparedViews(Mapping):
    """Prepared global names; mpas_atmphys_vars.F:68-955."""
    def __init__(self,owner):
        """Globals are allocated by native glue; mpas_atmphys_vars.F:179-217; mpas_atmphys_vars.F:934-942."""
        self.owner=owner
    def __iter__(self):
        """Declared names; mpas_atmphys_vars.F:68-955."""
        return iter(self.owner.native.abi['variables'])
    def __len__(self):
        """Native declaration count; mpas_atmphys_vars.F:68-955."""
        return len(self.owner.native.abi['variables'])
    def __getitem__(self,name):
        """Contiguous k,i view; mpas_atmphys_vars.F:179-217; mpas_atmphys_vars.F:934-942."""
        return self.owner.prepared(name)


class _ResidentPoolViews(_PoolViews):
    """Full resident entity arrays for a halo-aware slot boundary."""
    def __getitem__(self,name):
        """Resident getters; RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710."""
        return self.owner.field(self.pool,name,resident=True)


class _ProfileViews(_PreparedViews):
    """Driver argument aliases; mpas_atmphys_driver_sfclayer.F:1016-1047."""
    def __init__(self,owner,aliases):
        """Native argument spelling; mpas_atmphys_driver_sfclayer.F:1016-1047."""
        super().__init__(owner);self.aliases=aliases
    def __iter__(self):
        """Declared profile aliases; mpas_atmphys_driver_sfclayer.F:1016-1047."""
        return iter(self.aliases)
    def __len__(self):
        """Profile alias count; mpas_atmphys_driver_sfclayer.F:1016-1047."""
        return len(self.aliases)
    def __getitem__(self,name):
        """Reuse prepared native words; mpas_atmphys_driver_sfclayer.F:1016-1047."""
        return self.owner.prepared(self.aliases[name])


class _ScalarViews(Mapping):
    """Native manager scalar values; mpas_atmphys_vars.F:68-955."""
    def __init__(self,native):
        """Bind native manager globals; mpas_atmphys_vars.F:68-955."""
        self.native=native
    def __iter__(self):
        """Enumerate scalar globals; mpas_atmphys_vars.F:68-955."""
        return iter(n for n,r in self.native.abi['variables'].items() if r['rank']==0)
    def __len__(self):
        """Scalar declaration count; mpas_atmphys_vars.F:68-955."""
        return sum(1 for _ in self)
    def __getitem__(self,name):
        """Copy a native scalar, preserving its dtype; mpas_atmphys_vars.F:68-955."""
        value=self.native.get_variable(name)
        if value is None or value[0].ndim:raise KeyError(name)
        return value[0][()]


class _ConstantViews(Mapping):
    """Native resolved constants; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
    def __init__(self,native,aliases=None):
        """No host re-evaluation of constant expressions; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
        self.native=native
        self.aliases=aliases or {}
    def __iter__(self):
        """Provider's native constant names; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
        return iter(dict.fromkeys((*self.native.abi.get('constants',{}),*self.aliases)))
    def __len__(self):
        """Provider constant count; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
        return sum(1 for _ in self)
    def __getitem__(self,name):
        """Read compiled native words; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
        if not callable(getattr(self.native,'get_constant',None)):
            raise RuntimeError('native constant export is absent: CUDA slot constants must match compiled MPAS words')
        return self.native.get_constant(self.aliases.get(name,name))


def _session_landuse(provider):
    """The hydrated static's land-use table name, for Noah's packet; mpas_atmphys_landuse.F:177-227."""
    try:
        value=provider.native.get_text_field('sfc_input','mminlu').strip()
    except (AttributeError,KeyError,RuntimeError,ValueError):
        return None
    return value or None


def register_cuda_slots(registry,config,provider,resources=None):
    """Register only table-selected CUDA callbacks; mpas_atmphys_driver.F:209-384."""
    rows=load_table()[getattr(config,'table_key','mpasphys_v842')]['slots']
    for slot,scheme in config.selectors.items():
        if not config.active(slot) or config.backends[slot]!=Backend.HEX_CUDA:continue
        descriptor=rows[slot].get('cuda_entries',{}).get(scheme)
        if descriptor is None:
            raise RuntimeError(f'MPAS CUDA slot {slot}/{scheme} has no published callback descriptor')
        try:
            module=importlib.import_module('.'+descriptor['module'],__package__)
        except ModuleNotFoundError as exc:
            raise RuntimeError(f'MPAS CUDA slot module {descriptor["module"]} is not in this merged tree') from exc
        callback=getattr(module,descriptor['call'])
        resource=descriptor.get('resources')
        if resource:
            if resources is None:
                # First use fetches (or reuses) the certified prebuilt packet.
                # Framework and scheme tables need no native compiler here.
                from .resources_build import ensure_packet
                needs_mynnedmf=any(selected=='bl_mynnedmf' and config.active(name)
                                  and config.backends[name]==Backend.HEX_CUDA
                                  for name,selected in config.selectors.items())
                resources=ensure_packet(landuse=_session_landuse(provider),
                                        include_mynnedmf=needs_mynnedmf)
            from .resources_build import load_packet
            packets=load_packet(resources)
            provider.resources_packet = resources
            packet_key = resource.get('packet_key', scheme)
            if scheme == 'bl_mynnedmf' and packet_key not in packets:
                # THE BREAKAGE THIS PREVENTS: the fetched physics-resources packet is DATA only, so it has no
                # bl_mynnedmf entry (the RRFS MYNN-EDMF image is compiled for the run's device), and every hrrrv5
                # run on a plain install refused at the PBL slot. The forecast's first use builds the image into
                # the resources cache (resources_build.build_mynnedmf_image, cached by source, options and arch).
                from .resources_build import build_mynnedmf_image, default_cache
                packets = dict(packets)
                packets[packet_key] = build_mynnedmf_image(default_cache(), module=resource['module'])
            if packet_key not in packets:
                raise RuntimeError(f'MPAS CUDA slot {slot}/{scheme}: physics-resources packet {resources} has no '
                                   f'{scheme} entry (woof hex physics-resources build names what it does not build)')
            loader=importlib.import_module('.'+resource['module'],__package__)
            from . import moving_resident_v842
            import os
            if (moving_resident_v842.enabled()
                    and os.environ.get('WOOF_HEX_MOVING_RESIDENT_TABLES','1') != '0'):
                from .moving_tables_v842 import load
                loaded = load(loader, resource['call'], packets[packet_key], provider.cp,
                              arguments=resource.get('arguments', {}))
            else:
                loaded = getattr(loader,resource['call'])(packets[packet_key],provider.cp,
                                                         **resource.get('arguments', {}))
            setattr(provider,resource['attribute'],loaded)
        def invoke(call,callback=callback,descriptor=descriptor,slot=slot):
            """Slot copy transaction; mpas_atmphys_driver.F:209-384."""
            # Device memory the slot does not use is given back at its two
            # boundaries (woof.hex.device_memory_release; knobs
            # WOOF_HEX_POOL_RELEASE, GPUWM_HEX_LMEM_RELEASE).
            with phase_release('slot:'+slot):
                prepared=provider.begin(call,descriptor)
                result=callback(replace(call,pools=prepared))
                provider.commit()
            return result
        registry.register(slot,scheme,Backend.HEX_CUDA,invoke,
                          corrections=descriptor.get('corrections',()))
        diagnostics=descriptor.get('post_lsm_diagnostics')
        if diagnostics:
            # MPAS mpas_atmphys_driver.F:313-325; diagnostics after LSM/seaice.
            diagnostic_callback=getattr(module,diagnostics)
            provider.session.hooks.external['atmphys_sfc_diagnostics'] = (
                lambda call,callback=diagnostic_callback: callback(provider,call))
