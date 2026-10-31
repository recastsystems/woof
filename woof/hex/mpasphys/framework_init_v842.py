# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""MPAS v8.4.2 framework initialization for the resident CUDA physics suite.

Authorities: mpas_atmphys_control.F:335-440, manager.F:376-766,
init.F:83-592, init_microphysics.F:50-92 and lsm_noahinit.F:76-207.
Scheme tables are the existing authenticated Rust/CUDA resources. They are
built once by register_cuda_slots, rather than built again by native MPAS.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
from dataclasses import replace

import numpy as np

from .driver_v842 import interval_seconds
from .math_v842 import bind_common_carrier_math  # the AOT inventory reads the kernel declaration below


def initialize_directions(latitude, longitude):
    """init_dirs_forphys and r3_normalize, init.F:538-592, RKIND operations.

    GNU's scalar SIN/COS use libm sinf/cosf. Calling those C functions avoids
    NumPy's platform-dependent vector transcendental implementation.
    """
    libm = ct.CDLL(ctypes.util.find_library('m') or None)
    sin, cos = libm.sinf, libm.cosf
    for function in (sin, cos):
        function.argtypes, function.restype = [ct.c_float], ct.c_float
    f = np.float32
    # Scalar libm supplies the exact transcendental words. Bulk NumPy
    # arithmetic then keeps each RKIND operation separate, without a Python
    # loop over the normalization of a global mesh.
    slon = np.fromiter(map(sin, longitude), dtype=f, count=longitude.size)
    clon = np.fromiter(map(cos, longitude), dtype=f, count=longitude.size)
    slat = np.fromiter(map(sin, latitude), dtype=f, count=latitude.size)
    clat = np.fromiter(map(cos, latitude), dtype=f, count=latitude.size)
    east = np.asfortranarray(np.stack((-slon, clon, np.zeros_like(slon))))
    north = np.asfortranarray(np.stack((-clon*slat, -slon*slat, clat)))
    for out in (east, north):
        x, y, z = out
        inverse = f(1) / np.sqrt((x*x + y*y) + z*z)
        out *= inverse[None, :]
    return east, north


def initialize_aerosol_emissions(nwfa, rho_zz, zz, zgrid, area):
    """MPAS-A v8.4.2 init_microphysics.F:239-245, cold-start solve cells.

    Preserve REAL assignment boundaries and left-associated products. The
    supplied climatology owns the aerosol profile; no profile is invented.
    """
    f = np.float32
    airmass = f(rho_zz * zz)
    airmass = f(f(airmass * f(zgrid[1] - zgrid[0])) * area)
    return f(f(f(nwfa * f(0.000196)) * airmass) * f(0.5e-10))


class FrameworkInitialization:
    def __init__(self, session):
        self.session, self.native = session, session.native
        self.dimensions = session.layout.dimensions
        self.configs = session.layout.configs

    def operations(self):
        operations = {name: getattr(self, name) for name in ('physics_tables_init',
            'physics_registry_init', 'physics_run_init', 'physics_init', 'atmphys_finalize')}
        if self.configs['config_o3climatology']:
            operations['o3climatology_from_MPAS'] = self.o3climatology_from_mpas
        if not any(self.session.config.active(slot) for slot in
                   ('pbl', 'convection', 'radiation_lw', 'radiation_sw')):
            operations['physics_get_tend'] = self.inactive_tendencies
        return operations

    def o3climatology_from_mpas(self, call):
        """mpas_atmphys_o3climatology.F:234-327: shared radiation time update.

        Bind the existing CUDA interpolation independently of resident-suite
        admission. Nonresident sessions commit solve columns back to their
        framework pools; resident sessions retain device authority.
        """
        from .radshared_v842 import bind_native_ozone_pools, ozone_slot
        provider = self.session.device_pools
        provider.begin(call, {})
        result = ozone_slot(replace(call, pools=bind_native_ozone_pools(provider, call)))
        provider.commit()
        return result

    def inactive_tendencies(self, call):
        """Suite none still assembles RK sources, todynamics.F:98-498.

        Use the existing CUDA coupling, including the modified-theta conversion
        for exceptional input words, rather than replacing its arithmetic with
        a constant-zero source. The resident suite overrides this callback.
        """
        from ..cuda_physics_v841 import couple_mpas_physics_tendencies_v842, CudaMpasStageSourcesV842
        from .compile_v842 import MpasPhysicsKernelCache
        p = self.session.device_pools
        if not hasattr(self, '_inactive_cache'):
            self._inactive_cache = MpasPhysicsKernelCache()
        p.level = 2
        mass = p.field('state', 'rho_zz')
        p.level = 1
        edge = p.field('diag', 'rho_edge')
        names = next(row['components'] for row in self.session.layout.fields
                     if row['pool'] == 'state' and row['name'] == 'scalars')
        out = couple_mpas_physics_tendencies_v842(mass=mass, mass_edge=edge,
            theta_m=p.field('state', 'theta_m'), scalars=p.field('state', 'scalars'),
            scalar_names=names, tendencies={}, selectors=self.session.config.execution_options,
            pbl_edge=p.cp.zeros_like(edge), convection_edge=p.cp.zeros_like(edge),
            tend_u_phys=p.field('diag', 'tend_u_phys'),
            n_cells_solve=self.dimensions['nCellsSolve'], n_edges_solve=self.dimensions['nEdgesSolve'],
            rk_step=call.pools['rk_step'], dynamics_substep=call.pools['dynamics_substep'],
            kernel_cache=self._inactive_cache)
        p.field('tend', 'scalars_tend')[...] = out['scalars_tend']
        p.commit()
        return CudaMpasStageSourcesV842(*(out[name] for name in
            ('tend_rho_physics', 'tend_ru_physics', 'tend_rtheta_physics', 'scalars_tend')))


    def scalar(self, name, value):
        kind = self.native.abi['variables'][name]['kind']
        self.native.put_variable(name, np.array(value, np.float32 if kind == 'real' else np.int32))

    def field(self, pool, name):
        return self.native.get_field(pool, name)

    def write(self, pool, name, values, *, count=None):
        result = self.field(pool, name)
        if count is None:
            result[...] = values
        else:
            result[..., :count] = values
        self.native.put_field(pool, name, result, nlevels=self.native._fields[pool, name][2])

    def physics_tables_init(self, call):
        """control.F:393-440 checks Thompson availability, without building it."""
        provider = self.session.device_pools
        from ..physics_suites import load_table
        rows = load_table()['mpasphys_v842']['slots']
        state = {}
        for slot, scheme in self.session.config.selectors.items():
            if not self.session.config.active(slot):
                continue
            descriptor = rows[slot]['cuda_entries'][scheme]
            resource = descriptor.get('resources')
            if resource:
                state[scheme] = getattr(provider, resource['attribute'])
        self.native.scheme_state = state
        self.scalar('l_mp_tables', False)

    def physics_registry_init(self, call):
        """control.F:335-390: cold Noah layer thickness, including all cells."""
        if not self.configs['config_do_restart'] and self.configs['config_lsm_scheme'] in ('sf_noah', 'sf_noahmp'):
            dzs = self.field('sfc_input', 'dzs')
            dzs[:, :self.dimensions['nCells']] = np.array([.10, .30, .60, 1.00], np.float32)[:, None]
            self.write('sfc_input', 'dzs', dzs)

    def physics_run_init(self, call):
        """manager.F:376-766. PhysicsManager owns the recurring alarms."""
        d, c, f = self.dimensions, self.configs, np.float32
        dt = f(float(self.native.clock_state.dt))
        values = dict(ims=1, ime=d['nCellsSolve'], jms=1, jme=1, kms=1,
            kme=d['nVertLevels']+1, ids=1, ide=d['nCellsSolve']+1, jds=1, jde=2,
            kds=1, kde=d['nVertLevels']+1, its=1, ite=d['nCellsSolve'],
            jts=1, jte=1, kts=1, kte=d['nVertLevels'],
            num_months=d['nMonths'], num_soils=d['nSoilLevels'],
            num_oznlevels=d['nOznLevels'], dt_dyn=dt, dt_microp=dt, n_microp=1,
            l_conv=False, l_radtlw=False, l_radtsw=False, l_diags=False,
            l_camlw=False, l_acrain=False, l_acradt=False,
            xice_threshold=.02 if c['config_frac_seaice'] else .5)
        for variable, key in (('dt_radtlw', 'config_radtlw_interval'),
                              ('dt_radtsw', 'config_radtsw_interval'),
                              ('dt_cu', 'config_conv_interval'), ('dt_pbl', 'config_pbl_interval')):
            seconds = interval_seconds(c[key])
            values[variable] = dt if seconds is None else f(float(seconds))
        if c['config_lsm_scheme'] == 'sf_noah':
            values['sf_surface_physics'] = 2
        if c['config_microp_scheme'] in ('mp_thompson', 'mp_thompson_aerosols'):
            values['n_microp'] = max(int(np.floor(f(f(dt/f(90)) + f(.5)))), 1)
            values['dt_microp'] = f(dt/f(values['n_microp']))
        values['n_cu'] = max(int(np.floor(f(f(values['dt_cu']/dt) + f(.5)))), 1)
        re = (c['config_microp_re'] and c['config_microp_scheme'] in
              ('mp_thompson', 'mp_thompson_aerosols', 'mp_wsm6') and
              c['config_radt_lw_scheme'] == 'rrtmg_lw' and c['config_radt_sw_scheme'] == 'rrtmg_sw')
        values.update(has_reqc=int(re), has_reqi=int(re), has_reqs=int(re))
        for name, value in values.items():
            self.scalar(name, value)
        time = self.native.clock_state.start_time
        h, rem = divmod(int(time.seconds), 3600)
        minute, second = divmod(rem, 60)
        # manager.F:457-463 retains start GMT and initial zero-based whole day.
        gmt = f(f(f(h) + f(f(minute)/f(60))) + f(f(second)/f(3600)))
        for name, value in dict(year=np.int32(time.year), julday=np.int32(time.day_of_year),
                                gmt=gmt, curr_julday=f(time.day_of_year-1)).items():
            self.native._tables['mpas_atmphys_manager.' + name] = np.array(value), ()

    def physics_init(self, call):
        d, c, f = self.dimensions, self.configs, np.float32
        nc, ns = d['nCells'], d['nCellsSolve']
        east, north = initialize_directions(self.field('mesh', 'latCell')[:nc],
                                            self.field('mesh', 'lonCell')[:nc])
        self.write('mesh', 'east', east, count=nc)
        self.write('mesh', 'north', north, count=nc)
        for species in ('qc', 'qr', 'qi', 'qs', 'qg', 'nc', 'ni', 'nifa', 'nwfa'):
            self.scalar('f_' + species,
                        self.native._dimension_values['state', 'index_' + species] > -1)
        for name in ('f_qoz', 'f_nbca'):
            self.scalar(name, False)
        height = f(f(.5) / self.field('mesh', 'rdzw')[0])
        top = int(height <= f(c['config_microphysics_top']))
        dzu = self.field('mesh', 'dzu')
        for k in range(1, d['nVertLevels']):
            height = f(height + dzu[k])
            if height <= f(c['config_microphysics_top']):
                top = k+1
        self.write('tend_physics', 'mp_top_level', top)
        if not c['config_do_restart']:
            from .surface_update_v842 import RADIATION_FIELDS
            for name in RADIATION_FIELDS:
                self.write('diag_physics', 'ac' + name, 0, count=ns)
                self.write('diag_physics', 'i_ac' + name, 0, count=ns)
            for selector, name in (('config_convection_scheme', 'i_rainc'), ('config_microp_scheme', 'i_rainnc')):
                if c[selector] != 'off':
                    self.write('diag_physics', name, 0, count=ns)
            for name, source in (('xicem', 'xice'), ('sstsk', 'sst'), ('tyear_mean', 'tmn'), ('tlag', 'tmn')):
                self.write('diag_physics', name, self.field('sfc_input', source)[:ns], count=ns)
            for name in ('nsteps_accum', 'ndays_accum', 'tday_accum', 'tyear_accum'):
                self.write('diag_physics', name, 0, count=ns)
            if c['config_oml1d']:
                sst = self.field('sfc_input', 'sst')[:ns]
                for name in ('t_oml', 't_oml_initial'):
                    self.write('diag_physics', name, sst, count=ns)
                depth = f(c['config_oml_hml0'])
                if depth > 0:
                    for name in ('h_oml', 'h_oml_initial'):
                        self.write('diag_physics', name, depth, count=ns)
                else:
                    self.write('diag_physics', 'h_oml', self.field('diag_physics', 'h_oml_initial')[:ns], count=ns)
                if depth >= 0:
                    for name in ('hu_oml', 'hv_oml'):
                        self.write('diag_physics', name, 0, count=ns)
                    self.write('diag_physics', 't_oml_200m_initial', sst-f(5), count=ns)
        if c['config_o3climatology']:
            self.initialize_ozone(call)
        self.initialize_landuse()
        if c['config_convection_scheme'] == 'cu_kain_fritsch' and not c['config_do_restart']:
            # driver_convection.F:388-396 includes every local mesh cell.
            # Its immutable LUTAB is built by the existing CUDA KF loader.
            self.write('diag_physics', 'nca', f(-100), count=nc)
        if c['config_microp_scheme'] in ('mp_thompson', 'mp_thompson_aerosols'):
            land = self.field('sfc_input', 'landmask')[:ns]
            nt = self.field('diag_physics', 'nt_c')[:ns]
            nt[land == 1], nt[land == 0] = f(300e6), f(100e6)
            self.write('diag_physics', 'nt_c', nt, count=ns)
            self.write('diag_physics', 'mu_c', np.minimum(f(15), f(f(1000e6)/nt)+f(2)), count=ns)
        if c['config_microp_scheme'] == 'mp_thompson_aerosols' and not c['config_do_restart']:
            # init_microphysics.F:298-301; table_ccnAct is the slot's certified resource.
            self.initialize_thompson_aerosols()
        if c['config_lsm_scheme'] == 'sf_noah':
            self.initialize_noah()
        # MYNN and SF_MYNN initialization constants are already bound in their
        # CUDA modules; RRTMG's resources include its CUDA initialization pass.
        self.native._initialized = True

    def initialize_thompson_aerosols(self):
        """init_thompson_aerosols_forMPAS, init_microphysics.F:110-234, after its restart return.

        THE BREAKAGE THIS PREVENTS: without the native library nothing formed the
        Thompson-aerosol surface emission (nwfa2d, nifa2d) or the height profiles of
        an all-zero nwfa/nifa, so mp_thompson_aerosols could not run on the ported
        framework.  One CUDA thread per solve column, native's RKIND order with no
        contraction; cos/exp are the shared GNU carrier maths.
        """
        from .math_v842 import bind_common_carrier_math
        from .compile_v842 import MpasPhysicsKernelCache
        import cupy as cp
        d, f = self.dimensions, np.float32
        nz, ns = d['nVertLevels'], d['nCellsSolve']
        state = self.native._storage['state', 'scalars'][0]
        index = {name: int(self.native._dimension_values['state', 'index_' + name]) - 1
                 for name in ('nwfa', 'nifa')}
        zgrid = cp.asarray(np.ascontiguousarray(self.field('mesh', 'zgrid')[:nz+1, :ns].T, dtype=f))
        zz = cp.asarray(np.ascontiguousarray(self.field('mesh', 'zz')[0, :ns], dtype=f))
        rho = cp.asarray(np.ascontiguousarray(self.field('state', 'rho_zz')[0, :ns], dtype=f))
        area = cp.asarray(np.ascontiguousarray(self.field('mesh', 'areaCell')[:ns], dtype=f))
        # Parameter-only ALOG folds at compile time (correctly rounded), module_mp_thompson.F:119-122.
        def folded_log(numerator, denominator):
            return f(np.log(np.float64(f(f(numerator) / f(denominator)))))
        profiles = {}
        for name, (n1, n0) in (('nwfa', (50.0e6, 300.0e6)), ('nifa', (0.5e6, 1.5e6))):
            column = state[index[name], :nz, :ns]
            # mpas_dmpar_min/max over the solve columns: one process holds them all.
            regenerate = bool(column.min() == 0 and column.max() == 0)
            profiles[name] = (regenerate, f(f(-1.0) * folded_log(n1, n0)), f(n0), f(n1))
        out_nwfa = cp.asarray(np.ascontiguousarray(state[index['nwfa'], :nz, :ns].T, dtype=f))
        out_nifa = cp.asarray(np.ascontiguousarray(state[index['nifa'], :nz, :ns].T, dtype=f))
        nwfa2d = cp.empty(ns, dtype=f)
        if not hasattr(self, '_aerosol_cache'):
            self._aerosol_cache = MpasPhysicsKernelCache()
        kernel = self._aerosol_cache.raw_kernel('thompson_aerosols_init_v842',
            bind_common_carrier_math(_THOMPSON_AEROSOL_INIT),
            module_key='hexcore.mpasphys.framework_init_v842.thompson_aerosols')
        (w_on, w_log, w0, w1), (i_on, i_log, i0, i1) = profiles['nwfa'], profiles['nifa']
        kernel(((ns + 127) // 128,), (128,),
               (np.int32(nz), np.int32(ns), zgrid, zz, rho, area,
                np.int32(w_on), w_log, w0, w1, np.int32(i_on), i_log, i0, i1,
                out_nwfa, out_nifa, nwfa2d))
        state[index['nwfa'], :nz, :ns] = out_nwfa.get().T
        state[index['nifa'], :nz, :ns] = out_nifa.get().T
        self.write('diag_physics', 'nwfa2d', nwfa2d.get(), count=ns)
        self.write('diag_physics', 'nifa2d', f(0), count=ns)

    def initialize_ozone(self, call):
        from . import radshared_v842 as ozone
        from .framework_resources_v842 import ozone_resources
        p = self.session.device_pools
        p.radshared_v842_tables = ozone_resources(p.resources_packet,
            self.dimensions['nOznLevels'], p.cp)
        pools = ozone.bind_native_ozone_initial_pools(p, call)
        ozone.init_ozone_slot(replace(call, pools=pools))
        p.commit()

    def initialize_landuse(self):
        from . import noah_v842 as noah
        p, c, nc = self.session.device_pools, self.configs, self.dimensions['nCells']
        cp = p.cp
        classification = self.native.get_text_field('sfc_input', 'mminlu')
        resources = getattr(p, 'noah_v842_resources', None)
        if resources is None:
            # landuse_init_forMPAS runs even when all scheme slots are off.
            # Load its Rust tables without constructing a native framework.
            from .resources_build import ensure_packet, load_packet
            packet = getattr(p, 'resources_packet', self.native.resources)
            if packet is None:
                packet = ensure_packet(landuse=classification)
            p.resources_packet = packet
            resources = noah.load_native_resources(load_packet(packet)['sf_noah'], cp)
            p.noah_v842_resources = resources
        if classification != resources['landuse']:
            raise ValueError('landuse resource classification differs from mminlu')
        self.native.landuse_state = dict(mminlu=classification,
            isurban=noah.LANDUSE_URBAN_INDEX.get(classification, 0),
            iswater=int(self.field('sfc_input', 'iswater')),
            isice=int(self.field('sfc_input', 'isice')))
        floats, integers = np.zeros((nc, 14), np.float32), np.zeros((nc, 9), np.int32)
        fields = (('mesh', 'latCell'), ('sfc_input', 'snoalb'), ('sfc_input', 'snowc'),
                  ('sfc_input', 'xice'), ('sfc_input', 'sfc_albbck'),
                  ('diag_physics', 'z0'), ('diag_physics', 'znt'),
                  ('diag_physics', 'sfc_emibck'), ('diag_physics', 'sfc_albedo'),
                  ('diag_physics', 'sfc_emiss'), ('diag_physics', 'thc'),
                  ('diag_physics', 'mavail'), ('diag_physics', 'ust'))
        for i, key in enumerate(fields):
            if key in self.native._fields:
                floats[:, i] = self.field(*key)[:nc]
        floats[:, 13] = self.native.get_variable('xice_threshold')[0]
        integers[:, 0] = self.field('sfc_input', 'ivgtyp')[:nc]
        integers[:, 1:] = [self.native.clock_state.current_time.day_of_year,
            self.native.landuse_state['iswater'], self.native.landuse_state['isice'],
            int(c['config_do_restart']), int(c['config_sfc_albedo']),
            int(c['config_frac_seaice']), int(fields[11] in self.native._fields),
            int(fields[12] in self.native._fields)]
        packed = cp.asarray(floats)
        status = noah.initialize_landuse(packed, cp.asarray(integers), resources['landuse_tables'].view(cp.float32).reshape(7, 12, 100))
        if bool(cp.any(status)):
            raise RuntimeError('MPAS landuse initialization failed: ' + str(status.get().tolist()))
        values = packed.get()
        for i, key in enumerate(fields):
            if i >= 4 and key in self.native._fields:
                self.write(*key, values[:, i], count=nc)

    def initialize_noah(self):
        from . import noah_v842 as noah
        p, c, nc = self.session.device_pools, self.configs, self.dimensions['nCells']
        cp = p.cp
        fields = ('tslb', 'smois', 'sh2o', 'snoalb', 'snow', 'snowh')
        values = np.concatenate([self.field('sfc_input', name)[..., :nc].reshape(-1, nc).T
                                 for name in fields], axis=1)
        integers = np.empty((nc, 4), np.int32)
        integers[:, 0] = self.field('sfc_input', 'ivgtyp')[:nc]
        integers[:, 1] = self.field('sfc_input', 'isltyp')[:nc]
        integers[:, 2:] = int(c['config_do_restart']), int(c['config_sfc_snowalbedo'])
        packed = cp.ascontiguousarray(cp.asarray(values))
        status = noah.replay_columns('init', packed, cp.asarray(integers), p.noah_v842_resources['tables'])
        if bool(cp.any(status)):
            raise RuntimeError('MPAS Noah initialization failed: ' + str(status.get().tolist()))
        result = packed.get()
        offset = 0
        for name in fields:
            width = 4 if name in ('tslb', 'smois', 'sh2o') else 1
            value = result[:, offset:offset+width].T
            self.write('sfc_input', name, value if width > 1 else value[0], count=nc)
            offset += width

    def atmphys_finalize(self, call):
        """Non-stream state is owned by FrameworkPhysics.close, not scheme SAVE."""
        self.native._initialized = False
        self.native.scheme_state.clear()


_THOMPSON_AEROSOL_INIT = r'''
__device__ __forceinline__ float aerosol_h01(float h1) {
  if (h1 <= 1000.0f) return 0.8f;
  if (h1 >= 2500.0f) return 0.01f;
  return __fmul_rn(0.8f, cosf(__fsub_rn(__fmul_rn(h1, 0.001f), 1.0f)));
}
__device__ __forceinline__ float aerosol_hgt(const float *z, int k) {
  return __fmul_rn(0.5f, __fadd_rn(z[k], z[k+1]));
}
__device__ void aerosol_profile(int nz, const float *z, float minus_log, float n0, float n1, float *out) {
  float h1 = aerosol_hgt(z, 0);
  float scale = __fdiv_rn(minus_log, aerosol_h01(h1));
  // init_microphysics.F: k=1 uses hgt(2)-hgt(1), every k>=2 its own height.
  out[0] = __fadd_rn(n1, __fmul_rn(n0, expf(-__fmul_rn(__fdiv_rn(__fsub_rn(aerosol_hgt(z, 1), h1), 1000.0f), scale))));
  for (int k = 1; k < nz; ++k)
    out[k] = __fadd_rn(n1, __fmul_rn(n0, expf(-__fmul_rn(__fdiv_rn(__fsub_rn(aerosol_hgt(z, k), h1), 1000.0f), scale))));
}
extern "C" __global__ void thompson_aerosols_init_v842(int nz, int ns, const float *zgrid, const float *zz,
    const float *rho_zz, const float *area, int nwfa_on, float nwfa_log, float nwfa0, float nwfa1,
    int nifa_on, float nifa_log, float nifa0, float nifa1, float *nwfa, float *nifa, float *nwfa2d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ns) return;
  const float *z = zgrid + (size_t)i * (nz + 1);
  if (nwfa_on) aerosol_profile(nz, z, nwfa_log, nwfa0, nwfa1, nwfa + (size_t)i * nz);
  if (nifa_on) aerosol_profile(nz, z, nifa_log, nifa0, nifa1, nifa + (size_t)i * nz);
  float airmass = __fmul_rn(rho_zz[i], zz[i]);
  airmass = __fmul_rn(__fmul_rn(airmass, __fsub_rn(z[1], z[0])), area[i]);
  nwfa2d[i] = __fmul_rn(__fmul_rn(__fmul_rn(nwfa[(size_t)i * nz], 0.000196f), airmass), 0.5e-10f);
}
'''
