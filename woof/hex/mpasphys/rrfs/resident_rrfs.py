"""Resident GPU glue for the RRFS-MPAS fork route (RRFSx/MPAS-Model 1d6e370e, table-admitted selections).

Authority (fork source at 1d6e370e): mpas_atmphys_driver.F:127-419 (the per-step order, run by
driver_rrfs.PhysicsDriver), mpas_atmphys_interface.F:284-652 (MPAS_to_physics), mpas_atmphys_todynamics.F:98-493
(physics_get_tend), mpas_atmphys_driver_convection.F:1477-1544 (update_convection_step1).

This is the v8.4.2 resident path (resident_v842.ResidentPhysicsGlue) on the fork's pools: after activation the
mutable pool fields and module arrays belong to the GPU until an explicit publication boundary (output, restart),
and no operation of the step runs in a native library.  The glue binds the fork's replay-proven CUDA bodies
(driver_rrfs.prepare_mpas_to_physics; the shared coupling with driver_rrfs.coupling_selectors) as the route's
operations.  The fork's full-allocation scheme bindings (MYNN-EDMF, RUC land/sea ice/surface diagnostics) read the
framework's storage API (get_field/get_variable/put_field/put_variable) on whole native allocations;
:class:`ResidentFrameworkFacade` serves that API from the resident device pools, so those bindings stay byte for
byte as replayed and touch no host words on the step path.
"""
from __future__ import annotations

import numpy as np

from ..resident_v842 import (native_physics_geometry, regional_physics_geometry,
                             _solved_edges_reference_dummy)
from ..resident_v842 import ResidentPhysicsGlue as _GlueV842

# Allocation sets the fork driver allocates and deallocates around its slots (mpas_atmphys_driver.F:196-410).
# Resident runs allocate them once after physics_init and keep one persistent lifetime, as v8.4.2's does.
STEMS = ('forall_physics', 'cloudiness', 'radiation_sw', 'radiation_lw', 'sfclayer', 'lsm', 'pbl',
         'gwdo', 'convection', 'microphysics', 'seaice', 'smoke')


def _stems(config):
    """The allocation sets this selection's driver uses (mpas_atmphys_driver.F:196-410, fork)."""
    active = config.active
    stems = ['forall_physics']
    if active('radiation_lw') or active('radiation_sw'):
        stems.append('cloudiness')
    stems += [s for s in ('radiation_sw', 'radiation_lw') if active(s)]
    if active('surface_layer'):
        stems.append('sfclayer')
    if active('land_surface') and config.selectors['land_surface'] in ('sf_noah', 'sf_ruc'):
        stems.append('lsm')
    if active('land_surface') and active('seaice'):
        stems.append('seaice')
    if active('smoke'):
        stems.append('smoke')
    if active('pbl') and active('surface_layer'):
        stems.append('pbl')
    for slot, stem in (('gwdo', 'gwdo'), ('convection', 'convection'), ('microphysics', 'microphysics')):
        if active(slot):
            stems.append(stem)
    return stems


class ResidentFrameworkFacade:
    """The ported framework's storage API, served from the resident device pools.

    The fork's MYNN-EDMF and RUC bindings take whole native allocations: pool fields with their dummy column and
    module arrays in native i,k,j order with every allocation tail.  The resident authority is the device pool
    provider's (resident_v842 layout: owned-extent pool fields, k,i module views).  This facade assembles a whole
    allocation on the device from that authority plus a device copy of the words outside it (the dummy column,
    unused allocation tails), and writes a binding's result back the same way.  Host storage is the authority only
    for what the provider holds no view of: rank-0 module scalars (calendar flags, tile bounds, manager intervals,
    written by the host framework) and module arrays that are not per-cell profiles (``host_variables`` names
    them) until a binding writes one from the device; from then on its device copy is the authority
    (``device_owned``), so no binding's per-step write crosses to the host.  Every other attribute is the host framework's: the provider's own storage accessors,
    tables, constants, configs and the publication path (``put_field`` of a host array) stay host operations.
    """

    def __init__(self, provider, host):
        self._provider, self._host, self._cp = provider, host, provider.cp
        self._field_rest = {}     # (pool, name, level) -> device words beyond the provider's owned extent
        self._variable_whole = {}  # name -> device whole allocation, kept only when the provider view covers less
        self._count = int(provider.session.layout.dimensions['nCellsSolve'])
        self.host_variables = set()
        self.device_owned = {}     # non-profile module arrays a binding wrote from the device

    def __getattr__(self, name):
        return getattr(self._host, name)

    # ------------------------------------------------------------------ pool fields
    def _authority(self, pool, name, level):
        provider = self._provider
        old, provider.level = provider.level, level
        try:
            return provider.field(pool, name)
        finally:
            provider.level = old

    def get_field(self, pool, name, *, level=1, out=None):
        if out is not None:
            raise ValueError('the resident framework facade returns new device arrays, never into out=')
        dtype, shape, _ = self._host._fields[pool, name]
        packed = self._authority(pool, name, level)
        if packed.shape == shape:
            return packed.copy()
        whole = self._cp.empty(shape, dtype=dtype)
        n = packed.shape[-1]
        if packed.shape[:-1] != shape[:-1] or n > shape[-1]:
            raise ValueError(f'resident {pool}.{name} extent {packed.shape} does not fit its allocation {shape}')
        whole[..., :n] = packed
        rest = self._field_rest.get((pool, name, level))
        if rest is None:
            rest = self._field_rest[pool, name, level] = self._cp.asarray(
                np.ascontiguousarray(self._host.field_storage(pool, name, level=level)[..., n:]))
        whole[..., n:] = rest
        return whole

    def put_field(self, pool, name, value, *, level=1, nlevels=1):
        if not isinstance(value, self._cp.ndarray):
            # Host words: the provider's publication (or a host writer); the host framework owns them.
            return self._host.put_field(pool, name, value, level=level, nlevels=nlevels)
        dtype, shape, _ = self._host._fields[pool, name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f'resident {pool}.{name} write {value.shape}/{value.dtype} differs from {shape}/{dtype}')
        packed = self._authority(pool, name, level)
        if packed.shape == shape:
            packed[...] = value
            return None
        n = packed.shape[-1]
        packed[...] = value[..., :n]
        self._field_rest[pool, name, level] = value[..., n:].copy()
        return None

    # ------------------------------------------------------------------ module arrays
    def _view(self, name):
        """The provider's k,i device authority, or None when the provider keeps no view of this array."""
        stored = self._host.variable_storage(name)
        if stored is None:
            return None, None
        host, lower = stored
        if host.ndim == 0:
            return host, None
        try:
            return host, self._provider.prepared(name)
        except ValueError:
            # Not a per-cell i[,k][,s],j profile (resident_v842 layout): host words stay its authority.
            self.host_variables.add(name)
            return host, None

    def _core(self, whole, host):
        """The slice of a whole allocation the provider's view holds, in the provider's layout."""
        c = self._count
        if host.ndim == 4:
            return whole[:c, :, 0, :].transpose(2, 1, 0)
        if host.ndim == 3:
            return whole[:c, :, 0].T
        return whole[:c, 0]

    def _covers(self, host):
        return host.shape[0] == self._count and host.shape[2 if host.ndim == 4 else -1] == 1

    def get_variable(self, name):
        host, packed = self._view(name)
        if host is None:
            return None
        lower = self._host.variable_storage(name)[1]
        if packed is None:
            owned = self.device_owned.get(name)
            return (host.copy(order='F') if owned is None else owned.copy()), lower
        whole = self._variable_whole.get(name)
        if whole is None:
            whole = self._cp.empty(host.shape, dtype=host.dtype)
            if not self._covers(host):
                whole[...] = self._cp.asarray(np.ascontiguousarray(host))
                self._variable_whole[name] = whole
        whole = whole.copy()
        self._core(whole, host)[...] = packed
        return whole, lower

    def put_variable(self, name, value, *, lower=None):
        if not isinstance(value, self._cp.ndarray):
            # A host writer takes the array back (and any device copy a binding left is stale).
            self.device_owned.pop(name, None)
            return self._host.put_variable(name, value, lower=lower)
        host, packed = self._view(name)
        if host is None:
            raise RuntimeError(f'resident module array {name} is not allocated')
        if lower is not None and tuple(lower) != tuple(self._host.variable_storage(name)[1]):
            raise ValueError(f'resident module array {name} lower bounds differ')
        if value.shape != host.shape:
            raise ValueError(f'resident module array {name} shape {value.shape} differs from {host.shape}')
        if packed is None:
            # THE COST THIS REMOVES: MYNN-EDMF writes ddvel_p (rank 3, no per-cell view) every PBL call; through
            # host words that was a device-to-host copy and a host sync per step (one-off count, hrrrv5 x1.2562).
            self.device_owned[name] = value.astype(host.dtype, copy=True)
            return None
        packed[...] = self._core(value.astype(host.dtype, copy=False), host)
        if not self._covers(host):
            self._variable_whole[name] = value.astype(host.dtype, copy=True)
        return None

    def call(self, name, **arguments):
        # One persistent allocation lifetime (allocated by the glue after physics_init).
        if name.startswith(('allocate_', 'deallocate_')):
            return None
        return self._host.call(name, **arguments)


class ResidentPhysicsGlue:
    """GPU field authority and the fork's driver glue for the admitted RRFS selection (route_table resident row)."""

    def __init__(self, session, coupling, mesh, *, halo=None):
        from ...cuda_physics_v841 import CudaMpasCouplingV842
        from ..route_table import service_type
        from .driver_rrfs import coupling_selectors
        self.session, self.coupling = session, coupling
        self.halo = halo
        self.provider = p = session.device_pools
        self.cp, self.cache = p.cp, coupling.physics_cache
        self.dimensions = d = session.layout.dimensions
        self.nlev, self.ncells = d['nVertLevels'], d['nCellsSolve']
        self.names = tuple(next(row['components'] for row in session.layout.fields
                                if row['pool'] == 'state' and row['name'] == 'scalars'))
        if self.names[0] != 'qv':
            raise ValueError('resident RRFS preparation requires native qv as the first scalar')
        config = session.config
        host = session.native
        if not getattr(host, 'framework_only', False):
            raise ValueError('the RRFS resident glue runs on the ported fork framework, never on a native library')
        # Allocate once after physics_init established tile bounds and module integers; allocate_* only allocates
        # module arrays (framework_backend_v842._allocate), with no field arithmetic.
        self.stems = _stems(config)
        for stem in self.stems:
            host.call('allocate_' + stem)
        p.buffers.clear()
        p.values.clear()
        p.resident = True
        p.resident_buffers.clear()
        self.facade = ResidentFrameworkFacade(p, host)
        p.native = self.facade
        # Resident glue slots compile their word-copy kernels through this
        # strict cache; a new StrictPhysicsKernelCache opens the AOT pack and
        # fingerprints NVRTC (~0.5 s each), which slowed the first step.
        p._rrfs_glue_kernels = self.cache
        if config.active('smoke'):
            from .smoke_resident import install
            install(p)
        self.invalid = self.cp.zeros(1, dtype=self.cp.int32)
        regional = bool(session.layout.configs.get('config_apply_lbcs', False))
        # A same-host rank (samehost_v842), as resident_v842's: native's full-extent momentum tendencies (nCells),
        # owned columns from the schemes and halo columns from their owners, before the edge projection reads
        # both cells of every owned edge (tend_toEdges after exchange_halo_group 'physics:blten').
        self.local_rates = {}
        local_cells = None
        if halo is not None:
            local_cells = d['nCells']
            for group in ('physics:blten', 'physics:cuten'):
                for name in host._groups.get(group, ((),))[0]:
                    self.local_rates[name] = self.cp.zeros((self.nlev, local_cells), dtype=self.cp.float32)
        dummy_edges = (not regional and halo is None
                       and _solved_edges_reference_dummy(host, d['nCellsSolve'], d['nEdgesSolve']))
        # A limited-area rank: the garbage column follows its local cells (regional_physics_geometry local_cells).
        geometry = (regional_physics_geometry(p, local_cells=local_cells) if regional or dummy_edges
                    else native_physics_geometry(p, local_cells=local_cells))
        self.options = coupling_selectors(config.execution_options)
        self.assembler = CudaMpasCouplingV842(regional_garbage=regional or dummy_edges,
            selectors=self.options, scalar_names=self.names, geometry=geometry, kernel_cache=self.cache,
            exchange_halo_group=(lambda group: None) if halo is None else self.exchange_group,
            pbl_edge=p.field('tend_physics', 'rublten_Edge'),
            convection_edge=self.cp.zeros((self.nlev, d['nEdgesSolve']), dtype=self.cp.float32),
            tend_u_phys=p.field('diag', 'tend_u_phys'))
        if session.surface_service is None:
            NativeSurfaceService = service_type(config, 'surface_update',
                                                'woof.hex.mpasphys.surface_update_v842:NativeSurfaceService')
            session.surface_service = NativeSurfaceService(session, self.cp)
        operations = dict(session.surface_service.callbacks())
        operations.update(MPAS_to_physics=self.prepare, physics_get_tend=self.get_tend,
                          post_transport_prepare=session.hooks.external['post_transport_prepare'],
                          update_convection_step1=self.convection_step1,
                          snapshot_gwdo_tendencies=session.snapshot_gwdo)
        if config.active('land_surface') and 'atmphys_sfc_diagnostics' not in session.hooks.external:
            raise ValueError("the RRFS resident glue needs the land scheme's CUDA surface diagnostics "
                             "(its table row's post_lsm_diagnostics)")
        for stem in STEMS:
            operations['allocate_' + stem] = self.noop
            operations['deallocate_' + stem] = self.noop
        if session.precipitation_window is not None:
            session.precipitation_window.device_pools = p
        self.operations = frozenset((*operations, 'atmphys_sfc_diagnostics',
                                     *(('o3climatology_from_MPAS',) if 'o3climatology_from_MPAS'
                                       in session.hooks.external else ())))
        session.hooks.external.update(operations)
        session.hooks.resident_operations = self.operations
        session.resident_glue = self
        # Resolve the resident driver helpers once. Older driver modules keep
        # the existing path, allowing the separately owned changes to be folded
        # in either order. The shared runtime's MPAS-A route never binds this hook.
        from . import driver_rrfs
        self._resident_prepare = getattr(driver_rrfs, 'resident_prepare', None)
        self._resident_get_tend = getattr(driver_rrfs, 'resident_get_tend', None)
        coupling._rrfs_resident_prepare_transport = getattr(driver_rrfs, 'resident_prepare_transport', None)

    # exchange_halo_group(domain, group) on a same-host rank, todynamics.F:380-412: the MPAS-A route's.
    exchange_group = _GlueV842.exchange_group

    @staticmethod
    def noop(call):
        """Persistent allocation lifetime."""
        return None

    def convection_step1(self, call):
        """mpas_atmphys_driver_convection.F:1477-1544 (fork): only cu_kain_fritsch changes words here."""
        if call.options['config_convection_scheme'].strip() == 'cu_kain_fritsch':
            raise RuntimeError('update_convection_step1 for cu_kain_fritsch has no resident GPU operation; '
                               'the RRFS resident row does not admit it')
        return None

    def prepare(self, call):
        """MPAS_to_physics, fork mpas_atmphys_interface.F:284-652 (driver_rrfs.prepare_mpas_to_physics)."""
        helper = getattr(self, '_resident_prepare', None)
        if helper is not None:
            return helper(self, call)
        from .driver_rrfs import prepare_mpas_to_physics
        p, nz = self.provider, self.nlev
        p.level = 1
        field, prepared = p.field, p.prepared
        fields = {name: field('state', name) for name in ('rho_zz', 'theta_m', 'w')}
        fields.update({name: field('mesh', name) for name in ('zgrid', 'zz')})
        fields.update({name: field('diag', name) for name in ('exner', 'pressure_base', 'pressure_p',
                                                              'uReconstructZonal', 'uReconstructMeridional')})
        fields.update({name: prepared(name)[:nz + 1] for name in ('fzm_p', 'fzp_p')})
        scalars = field('state', 'scalars')
        species = {name: self.cp.ascontiguousarray(scalars[index])
                   for index, name in enumerate(self.names)
                   if name in ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'nc', 'ni', 'nifa', 'nwfa')}
        flags = {name: bool(p['scalar_state']['f_' + name]) for name in ('nc', 'ni', 'nifa', 'nwfa')}
        options = dict(call.options)
        result = prepare_mpas_to_physics(fields=fields, species=species, options=options,
                                         scalar_flags=flags, kernel_cache=self.cache)
        for name, value in result.items():
            if name == 'plrad':
                field('diag_physics', 'plrad')[...] = value
                continue
            target = prepared(name)
            if value.ndim == 2:
                target[:value.shape[0]] = value
            else:
                target[...] = value
        # Native writes diag.surface_pressure and psfc_p to identical words.
        field('diag', 'surface_pressure')[...] = result['psfc_p']
        p.commit()

    def get_tend(self, call):
        """physics_get_tend, fork mpas_atmphys_todynamics.F:98-493, once per step."""
        helper = getattr(self, '_resident_get_tend', None)
        if helper is not None:
            return helper(self, call)
        p = self.provider
        p.level = 2
        mass = p.field('state', 'rho_zz')
        p.level = 1
        rates = {row['pool_name']: p.field('tend_physics', row['pool_name'])
                 for row in self.session.layout.fields if row['pool'] == 'tend_physics'
                 and (row['pool_name'].endswith('ten')
                      or row['pool_name'] in ('rthratenlw', 'rthratensw'))}
        rates.update(self.local_rates)   # a rank's local-extent momentum rates, as resident_v842.get_tend
        # 44c0849 time_integration.F:2150-2165: perturb process tendencies
        # before conversion, then accumulated prognostic tendencies afterwards.
        # The optional implementation is supplied by the selected route owner.
        apply_sppt = None
        if call.options.get('do_sppt', False):
            apply_sppt = self.session.hooks.external.get('sppt_apply')
            if apply_sppt is None:
                raise RuntimeError('do_sppt requires the selected route sppt_apply hook')
            apply_sppt('phys', {name: rates[name] for name in
                       ('rucuten', 'rvcuten', 'rublten', 'rvblten')}, call=call, provider=p)
        out = self.assembler.get_tend(mass=mass, mass_edge=p.field('diag', 'rho_edge'),
            theta_m=p.field('state', 'theta_m'), scalars=p.field('state', 'scalars'),
            tendencies=rates, n_cells_solve=self.ncells,
            n_edges_solve=self.dimensions['nEdgesSolve'], rk_step=call.pools['rk_step'],
            dynamics_substep=call.pools['dynamics_substep'], invalid_flag=self.invalid)
        if apply_sppt is not None:
            apply_sppt('prog', {'tend_rtheta_physics': out.rho_theta,
                                'tend_ru_physics': out.rho_u}, call=call, provider=p)
        p.field('tend', 'scalars_tend')[...] = out.scalars
        p.commit()
        return out
