"""RRFS-MPAS framework and scheme initialization with no native library.

The fork's mpas_atmphys_init.F:83-481 (physics_init), mpas_atmphys_control.F:385-503
(physics_registry_init, physics_tables_init) and mpas_atmphys_manager.F:478-826 (physics_run_init) at
1d6e370e, on the ported FrameworkPhysicsRRFS. Shared with the plain MPAS route where the fork source is
identical: the ozone climatology (mpas_atmphys_o3climatology.F is byte-identical to v8.4.2), the landuse
kernel (fork landuse.F adds only the RUC mosaic scalars, the 15 s urban index and the DA-cycling return), the
RRTMG tables (already CUDA resources). The driver-owned numerical preparation (directions, species flags,
accumulators, OML, tlag) is driver_rrfs.initialize_physics_pools, proven bitwise by r4's init replay.

What is not ported yet refuses by name before a forecast could run on missing state (see ``GAPS``).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..math_v842 import bind_common_carrier_math
from ..driver_v842 import interval_seconds
from ..framework_init_v842 import FrameworkInitialization

FORK = 'MPAS-Model-rrfs 1d6e370e'

# Scheme initializers this module does not port yet. Each names the state a forecast would otherwise run without.
GAPS = {
    'ruc_tables': 'RUC RUCLSM_SOILVEGPARM (mpas_atmphys_lsm_rucinit.F:148-174) fills module_ruc_land tables '
                  '(VEGPARM.TBL/SOILPARM.TBL) that ruc_lsm_8c9f1a8 reads by name; ruclsminit and every RUC call '
                  'need them',
    'convection_init': 'init_convection for a convection scheme the fork control (mpas_atmphys_control.F:205-215) '
                       'does not list has no ported path',
    'microphysics_init': 'init_microphysics for Thompson/WSM6 on the RRFS route has no ported path',
}


# TEMPO 9adf9ef tempo_init without a native library (FrameworkInitializationRRFS.initialize_tempo_tables):
# tempo_table_init's inputs and outputs, in its order (rust/physics-tables/src/bin/tempo_table_init.rs).
S = ()
TEMPO_CONSTANT_SCALARS = ('av_r', 'bv_r', 'av_s', 'bv_s', 'bv_c', 'bv_i', 'av_g_old', 'bv_g_old', 'av_g_new',
                          'bv_g_new', 'am_s', 'bm_s', 'bm_g', 'bm_i', 'bm_r', 'rho_i', 'xm0i', 'd0c', 'd0r', 'd0s',
                          'd0g', 'c_cube', 'c_sqrd', 'mu_r', 'mu_s', 'mu_g', 'mu_i', 'rho_w', 'sc', 'pi', 'lsub',
                          'lvap0', 'rv', 'rdry', 't0')
TEMPO_CONSTANT_ARRAYS = ('rho_g', 'av_g', 'bv_g', 'r_c', 'r_i', 'r_r', 'r_s', 'r_g', 'nt_i', 'n0r_exp', 'n0g_exp',
                         'nt_in')
F, I, D = 'float32', 'int32', 'float64'
TEMPO_DERIVED = (
    [('av_g', F, (9,)), ('bv_g', F, (9,)), ('dim_nrhg', I, S), ('am_i', F, S), ('am_r', F, S), ('am_g', F, (9,))]
    + [(n, F, S) for n in ('ar_volume', 'lfus', 'olfus', 'orv', 'rho_not', 'rho_not0', 'sc3', 'd0i', 'xm0s', 'xm0g',
                           'obmi', 'obmr', 'oams', 'obms', 'ocms', 'obmg')]
    + [('oamg', F, (9,)), ('ocmg', F, (9,)), ('cce', F, (5, 15)), ('ccg', F, (5, 15)), ('ocg1', F, (15,)),
       ('ocg2', F, (15,)), ('cie', F, (7,)), ('cig', F, (7,)), ('oig1', F, S), ('oig2', F, S), ('cre', F, (13,)),
       ('crg', F, (13,)), ('ore1', F, S), ('org1', F, S), ('org2', F, S), ('org3', F, S), ('cse', F, (17,)),
       ('csg', F, (17,)), ('cge', F, (12, 9)), ('cgg', F, (12, 9))]
    + [(n, F, S) for n in ('oge1', 'ogg1', 'ogg2', 'ogg3', 't1_qr_qc', 't1_qr_qi', 't2_qr_qi', 't1_qs_qc', 't1_qs_qi',
                           't1_qr_ev', 't2_qr_ev', 't1_qs_sd', 't2_qs_sd', 't1_qs_me', 't2_qs_me', 't1_qg_sd',
                           't1_qg_me')]
    + [(n, I, S) for n in ('nic2', 'nii2', 'nir2', 'nis2', 'nig2', 'nii3', 'nir3', 'nig3', 'niin2')]
    + [(n, D, (100,)) for n in ('dc', 'dtc', 'di', 'dti', 'dr', 'dtr', 'ds', 'dts', 'dg', 'dtg', 't_nc')]
    + [('nic1', D, S)])
TEMPO_HAIL_BINS = (('hbins', D, (50,)), ('dhbins', D, (50,)))
TEMPO_RADAR_BINS = (('sbins_radar', D, (50,)), ('dsbins_radar', D, (50,)), ('gbins_radar', D, (50,)),
                    ('dgbins_radar', D, (50,)))


def tempo_constant_names():
    """TEMPO module words that are PARAMETERs or declaration values: everything tempo_init does not write."""
    from .tempo_v303 import TABLE_LAYOUT
    written = {name for name, _, _ in (*TEMPO_DERIVED, *TEMPO_HAIL_BINS, *TEMPO_RADAR_BINS)} - {'av_g', 'bv_g'}
    written |= set(FrameworkInitializationRRFS.TEMPO_FILE_TABLES)
    written |= {name for name, _ in FrameworkInitializationRRFS.TEMPO_COMPUTED_TABLES}
    return [name for name, _ in TABLE_LAYOUT if name not in written]


def gap(name):
    return RuntimeError(f'RRFS ported initialization: {GAPS[name]} ({FORK}); not ported yet, so the ported '
                        'framework refuses rather than run a forecast on uninitialized scheme state')


# MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init_tempo.F:40-185 at 1d6e370e.
# Parameters from TEMPO 9adf9ef module_mp_tempo_params.F90:97-103 (wp = RKIND single precision).
_TEMPO_AEROSOLS_CUDA = r"""
extern "C" __global__ void rrfs_tempo_aerosol_profile(
    const int cells, const int levels, const int zstride, const int sstride,
    const float *zgrid, float *field, const float base, const float scale, const float ratio_log) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=cells) return;
    const float *z=zgrid+(long)c*zstride;
    float *q=field+(long)c*sstride;
    float h1=__fmul_rn(0.5f,__fadd_rn(z[0],z[1]));
    float h2=__fmul_rn(0.5f,__fadd_rn(z[1],z[2]));
    float h01;
    if(h1<=1000.0f) h01=0.8f;
    else if(h1>=2500.0f) h01=0.01f;
    else h01=__fmul_rn(0.8f,cosf(__fsub_rn(__fmul_rn(h1,0.001f),1.0f)));
    float n3=__fdiv_rn(__fmul_rn(-1.0f,ratio_log),h01);
    q[0]=__fadd_rn(base,__fmul_rn(scale,expf(__fmul_rn(-__fdiv_rn(__fsub_rn(h2,h1),1000.0f),n3))));
    for(int k=1;k<levels;k++){
        float hk=__fmul_rn(0.5f,__fadd_rn(z[k],z[k+1]));
        q[k]=__fadd_rn(base,__fmul_rn(scale,expf(__fmul_rn(-__fdiv_rn(__fsub_rn(hk,h1),1000.0f),n3))));
    }
}
extern "C" __global__ void rrfs_tempo_aerosol_surface(
    const int cells, const int zstride, const int sstride, const float *zgrid, const float *nwfa,
    const float *area, float *nwfa2d, float *nifa2d, const float lower, const float upper) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=cells) return;
    float dz=__fsub_rn(zgrid[(long)c*zstride+1],zgrid[(long)c*zstride]);
    if(dz<9.0f) dz=9.0f;
    float v=fmaxf(lower,fminf(upper,nwfa[(long)c*sstride]));
    nwfa2d[c]=__fmul_rn(__fmul_rn(__fmul_rn(v,0.000196f),__fdiv_rn(5.0f,dz)),__fdiv_rn(area[c],9.e6f));
    nifa2d[c]=0.0f;
}
"""


# TEMPO lookup words read ahead (FrameworkInitializationRRFS.prefetch_tables), keyed by the calling thread, the
# files' layout and their resolved paths, until physics_init on that thread takes them (read_tempo_lookup_files).
# Per thread: every same-host rank thread reads its own words, as each read them itself before.
_LOOKUP_PREFETCH: dict = {}


def tempo_lookup_path_in(named, filename, *, resource='mpas-source-rrfs', pin_set='rrfs'):
    """One NOAA lookup file: the last named table directory that holds it, else the fetched fork source's
    (FrameworkInitializationRRFS.tempo_lookup_path, for a ``named`` directory list known before the backend)."""
    from pathlib import Path
    if named is not None:
        roots = [named] if isinstance(named, (str, Path)) else list(named)
        for root in reversed(roots):
            if (Path(root) / filename).is_file():
                return Path(root) / filename
    from ..resources_build import ensure_route_resources
    source = ensure_route_resources((resource,), pin_set=pin_set)[resource]
    return Path(source) / 'src/core_atmosphere/physics/physics_wrf/files' / filename


def _log_word(ratio):
    """ALOG of a constant-folded parameter ratio, correctly rounded like gfortran's MPFR folding."""
    return np.float32(math.log(float(np.float32(ratio))))


class FrameworkInitializationRRFS(FrameworkInitialization):
    """The fork's four initialization entries on FrameworkPhysicsRRFS (mpas_atm_core.F:589-596)."""

    # Proof tools only (tests/rrfs_framework_init_replay.py --allow-gaps): skip a named gap and record it, so
    # a word comparison shows exactly which words the gap owns. A forecast never sets it.
    allow_gaps = False

    def _gap(self, name):
        if not self.allow_gaps:
            raise gap(name)
        self.skipped_gaps = getattr(self, 'skipped_gaps', set()) | {name}

    def operations(self):
        operations = {name: getattr(self, name) for name in ('physics_tables_init',
            'physics_registry_init', 'physics_run_init', 'physics_init', 'atmphys_finalize')}
        if self.configs['config_o3climatology']:
            operations['o3climatology_from_MPAS'] = self.o3climatology_from_mpas
        return operations

    def physics_tables_init(self, call):
        """control.F:480-503: l_mp_tables is set true; the Thompson file check returns at once.

        Its `/= mp_thompson .or. /= mp_thompson_aerosols` guard is always true, so no file is inquired.
        Scheme resources are the CUDA providers' (the RRFS table row names them).
        """
        from ...physics_suites import load_table
        provider = getattr(self.session, 'device_pools', None)
        rows = load_table()[getattr(self.session.config, 'table_key', 'mpasphys_rrfs')]['slots']
        state = {}
        for slot, scheme in self.session.config.selectors.items():
            if not self.session.config.active(slot) or provider is None:
                continue
            descriptor = rows.get(slot, {}).get('cuda_entries', {}).get(scheme) or {}
            resource = descriptor.get('resources')
            if resource and hasattr(provider, resource['attribute']):
                state[scheme] = getattr(provider, resource['attribute'])
        self.native.scheme_state = state
        self.scalar('l_mp_tables', True)

    def physics_registry_init(self, call):
        """control.F:385-453: cold-start soil layer thickness for Noah (4) and RUC (9), every local cell."""
        if self.configs['config_do_restart']:
            return
        scheme = self.configs['config_lsm_scheme'].strip()
        if scheme in ('sf_noah', 'sf_noahmp'):
            values = (.10, .30, .60, 1.00)
        elif scheme == 'sf_ruc':
            values = (.005, .025, .045, .13, .25, .35, .50, 1.00, .70)
        else:
            return
        dzs = self.field('sfc_input', 'dzs')
        dzs[:len(values), :self.dimensions['nCells']] = np.array(values, np.float32)[:, None]
        self.write('sfc_input', 'dzs', dzs)

    def physics_run_init(self, call):
        """manager.F:478-826 (fork): module scalars, intervals and the start calendar."""
        d, c, f = self.dimensions, self.configs, np.float32
        dt = f(float(self.native.clock_state.dt))
        dim = lambda name: int(d.get(name, 0))
        values = dict(ims=1, ime=d['nCellsSolve'], jms=1, jme=1, kms=1,
            kme=d['nVertLevels']+1, ids=1, ide=d['nCellsSolve']+1, jds=1, jde=2,
            kds=1, kde=d['nVertLevels']+1, its=1, ite=d['nCellsSolve'],
            jts=1, jte=1, kts=1, kte=d['nVertLevels'],
            num_months=d['nMonths'], num_soils=d['nSoilLevels'],
            num_landcat=dim('nlcat'), num_soilcat=dim('nscat'), num_chem=dim('nChem'),
            kanthro=dim('nkanthro'), kwildfire=dim('nkwildfire'), kbiogenic=dim('nkbiogenic'),
            kvolcanic=dim('nkvolcanic'), kreswoodcomb=dim('nkreswoodcomb'),
            # manager.F:478-483 gets Registry num_e_* dimensions. n_e_*
            # are its local pointer names, not keys in the Registry layout.
            num_e_bb_in=dim('num_e_bb_in'), num_e_ant_in=dim('num_e_ant_in'),
            num_e_bb_out=dim('num_e_bb_out'), num_e_dust_out=dim('num_e_dust_out'),
            num_e_ant_out=dim('num_e_ant_out'),
            num_oznlevels=d['nOznLevels'], dt_dyn=dt,
            l_conv=False, l_radtlw=False, l_radtsw=False, l_diags=False,
            l_camlw=False, l_acrain=False, l_acradt=False,
            xice_threshold=.02 if c['config_frac_seaice'] else .5)
        chemistry = (c.get('config_smoke_scheme', 'off').strip() == 'on'
                     or c.get('config_dust_scheme', 'off').strip() == 'on'
                     or c.get('config_anthro_scheme', 'off').strip() != 'off')
        values['do_chemistry'] = chemistry
        for variable, key in (('dt_radtlw', 'config_radtlw_interval'),
                              ('dt_radtsw', 'config_radtsw_interval'),
                              ('dt_cu', 'config_conv_interval'), ('dt_pbl', 'config_pbl_interval')):
            seconds = interval_seconds(c[key])
            values[variable] = dt if seconds is None else f(float(seconds))
        if c['config_lsm_scheme'].strip() == 'sf_noah':
            values['sf_surface_physics'] = 2
        micro = c['config_microp_scheme'].strip()
        n_microp, dt_microp = 1, dt
        if micro in ('mp_thompson', 'mp_tempo', 'mp_thompson_aerosols'):
            # NINT of a positive RKIND quotient, then dt_dyn / n_microp in RKIND.
            n_microp = max(int(np.floor(f(f(dt/f(90)) + f(.5)))), 1)
            dt_microp = f(dt/f(n_microp))
            if dt <= dt_microp:
                dt_microp = dt
        if micro == 'mp_nssl2m' and dt > f(f(1.25001) * f(60)):
            n_microp = max(int(np.floor(f(f(dt/f(60)) + f(.5)))), 2)
            dt_microp = f(dt/f(n_microp))
        values.update(n_microp=n_microp, dt_microp=dt_microp)
        values['n_cu'] = max(int(np.floor(f(f(values['dt_cu']/dt) + f(.5)))), 1)
        re = (c['config_microp_re'] and micro in
              ('mp_thompson', 'mp_tempo', 'mp_thompson_aerosols', 'mp_wsm6', 'mp_nssl2m') and
              c['config_radt_lw_scheme'].strip() == 'rrtmg_lw' and c['config_radt_sw_scheme'].strip() == 'rrtmg_sw')
        values.update(has_reqc=int(re), has_reqi=int(re), has_reqs=int(re))
        if 'cam' in (c['config_radt_lw_scheme'].strip()[:3], c['config_radt_sw_scheme'].strip()[:3]):
            values.update(doabsems=False, cam_abs_dim1=dim('cam_dim1'), cam_abs_dim2=d['nVertLevels']+1,
                          num_aerosols=dim('nAerosols'), num_aerlevels=dim('nAerLevels'))
        for name, value in values.items():
            if name in self.native.abi['variables']:
                self.scalar(name, value)
        time = self.native.clock_state.start_time
        h, rem = divmod(int(time.seconds), 3600)
        minute, second = divmod(rem, 60)
        gmt = f(f(f(h) + f(f(minute)/f(60))) + f(f(second)/f(3600)))
        for name, value in dict(year=np.int32(time.year), julday=np.int32(time.day_of_year),
                                gmt=gmt, curr_julday=f(time.day_of_year-1)).items():
            self.native._tables['mpas_atmphys_manager.' + name] = np.array(value), ()

    # ---------------------------------------------------------------- physics_init

    def physics_init(self, call):
        """init.F:83-481 (fork): driver preparation, then the delegated initializers in native order."""
        c = self.configs
        julday = self.native.clock_state.current_time.day_of_year
        p = self.session.device_pools
        if getattr(p, 'resources_packet', None) is None and self.native.resources is not None:
            # The route's authenticated resource packet (ozone and landuse tables), as the v8.4.2 framework.
            p.resources_packet = self.native.resources
        self.prepare_pools()
        if c['config_o3climatology']:
            self.initialize_ozone(call)
        self.initialize_landuse_rrfs(julday)
        if c['config_convection_scheme'].strip() != 'off':
            self.initialize_convection()
        if c['config_microp_scheme'].strip() != 'off':
            self.initialize_microphysics()
        # init_pbl: only bl_mynn has an initializer (driver_pbl.F:1286-1313); bl_mynnedmf reads module
        # scalars at call time. init_sfclayer: sf_mynnsfclay has no initializer (driver_sfclayer.F:1083-1100);
        # the CUDA surface-layer modules bind their own constants.
        if c['config_pbl_scheme'].strip() == 'bl_mynn':
            pass  # bl_mynn_init constants are bound in the shared CUDA MYNN module.
        if c['config_lsm_scheme'].strip() == 'sf_ruc':
            self.initialize_ruc()
        if c['config_lsm_scheme'].strip() == 'sf_noah':
            self.initialize_noah()
        if c.get('config_gwdo_scheme', 'off').strip() == 'bl_ugwp_gwdo':
            self.check_orographic_statistics()
            if c.get('config_ngw_scheme', False):
                self.initialize_ngw()
        # init_smoke is empty in the pinned fork (driver_smoke.F:998-1005).
        # The first mpas_smoke_driver call initializes scalar indices, dry
        # deposition and radiation tables (mpas_smoke_wrapper.F90:270-315).
        self.native._initialized = True

    _ACCUMULATED = ('swdnb', 'swdnbc', 'swdnt', 'swdntc', 'swupb', 'swupbc', 'swupt', 'swuptc',
                    'lwdnb', 'lwdnbc', 'lwdnt', 'lwdntc', 'lwupb', 'lwupbc', 'lwupt', 'lwuptc')
    PREPARED_FIELDS = {
        'mesh': ('latCell', 'lonCell', 'east', 'north'),
        'diag_physics': ('i_rainc', 'i_rainnc', *('ac' + x for x in _ACCUMULATED), *('i_ac' + x for x in _ACCUMULATED),
                         'xicem', 'sstsk', 't_oml', 't_oml_initial', 'h_oml', 'h_oml_initial', 'hu_oml', 'hv_oml',
                         't_oml_200m_initial', 'nsteps_accum', 'ndays_accum', 'tday_accum', 'tyear_accum',
                         'tyear_mean', 'tlag'),
        'sfc_input': ('xice', 'sst', 'tmn'),
    }

    def prepare_pools(self):
        """init.F:84-396: directions, species flags and cold-start pool words (driver_rrfs, CUDA)."""
        from .driver_rrfs import StrictPhysicsKernelCache, initialize_physics_pools
        p = self.session.device_pools
        cp, n = p.cp, self.native
        # Only the words driver_rrfs.initialize_physics_pools reads or writes (init.F:84-396), not whole pools.
        pools = {}
        prepared_fields = dict(self.PREPARED_FIELDS)
        if 'config_microphysics_top' in self.configs:
            prepared_fields['mesh'] = (*prepared_fields['mesh'], 'rdzw', 'dzu')
            prepared_fields['tend_physics'] = ('mp_top_level',)
        for pool, names in prepared_fields.items():
            pools[pool] = {name: cp.ascontiguousarray(cp.asarray(n.field_storage(pool, name)))
                           for name in names if (pool, name) in n._fields}
        indices = {name: int(value) for (pool, name), value in n._dimension_values.items()
                   if pool == 'state' and name.startswith('index_')}
        options = {key: self.configs.get(key) for key in (
            'config_do_restart', 'config_do_DAcycling', 'config_convection_scheme',
            'config_microp_scheme', 'config_oml1d', 'config_oml_hml0')}
        if 'config_microphysics_top' in self.configs:
            options['config_microphysics_top'] = self.configs['config_microphysics_top']
        options['config_do_DAcycling'] = bool(options['config_do_DAcycling'])
        dims = {name: self.dimensions[name] for name in ('nCells', 'nCellsSolve', 'nLags')}
        if 'config_microphysics_top' in self.configs:
            dims['nVertLevels'] = self.dimensions['nVertLevels']
        if not hasattr(self, '_kernels'):
            self._kernels = StrictPhysicsKernelCache()
        output, flags = initialize_physics_pools(pools=pools, options=options, dimensions=dims,
            scalar_indices=indices, kernel_cache=self._kernels)
        for pool, fields in output.items():
            for name, value in fields.items():
                if pool != 'sfc_input':   # sfc_input is read only here
                    # Contiguous device packing promotes native scalar fields
                    # to length-one arrays. Restore the Registry rank on writeback.
                    metadata = n._fields[pool, name]
                    n.put_field(pool, name, cp.asnumpy(value).reshape(metadata[1]), nlevels=metadata[2])
        for name, value in flags.items():
            if name in n.abi['variables']:
                self.scalar(name, value)

    def initialize_landuse_rrfs(self, julday):
        """landuse.F:96-330 (fork): v8.4.2's kernel plus the fork's RUC scalars, 15 s urban index, DA return."""
        c = self.configs
        if c['config_lsm_scheme'].strip() == 'sf_ruc':
            for name, value in (('spp_lsm_loc', 0), ('mosaic_lu', 1), ('mosaic_soil', 1)):
                if ('diag_physics', name) in self.native._fields:
                    self.write('diag_physics', name, np.int32(value))
        if c.get('config_do_DAcycling', False) and not c['config_do_restart']:
            # landuse.F:275-280: tables read, then return before the per-cell loop.
            self._landuse_state()
            return
        self.initialize_landuse()
        self._landuse_state()

    def _landuse_state(self):
        from .. import noah_v842 as noah
        classification = self.native.get_text_field('sfc_input', 'mminlu')
        self.native.landuse_state = dict(mminlu=classification,
            isurban=13 if classification == 'MODIFIED_IGBP_MODIS_NOAH_15s'
                    else noah.LANDUSE_URBAN_INDEX.get(classification, 0),
            iswater=int(self.field('sfc_input', 'iswater')), isice=int(self.field('sfc_input', 'isice')))

    # rucinit.F:158-166: the RUC dataset names native maps; any other classification leaves MMINLURUC unset and
    # native stops in RUCLSM_SOILVEGPARM ("land use dataset ... not found in VEGPARM.TBL").
    RUC_LANDUSE = {'USGS': 'USGS-RUC', 'MODIS': 'MODI-RUC', 'MODIFIED_IGBP_MODIS_NOAH': 'MODI-RUC'}

    def table_directory(self):
        """The pinned MPAS-Data tables (c57dbc7, the fork's own pin) the route's resource packet carries."""
        from pathlib import Path
        from ..resources_build import load_packet
        packet = getattr(self.session.device_pools, 'resources_packet', None) or self.native.resources
        if packet is None:
            raise RuntimeError('RRFS ported initialization needs the route resource packet (VEGPARM/SOILPARM/'
                               'GENPARM for RUC): none is bound')
        return Path(load_packet(packet)['sf_noah']['table_directory'])

    @staticmethod
    def ruc_table_words(directory, landuse, soil='STAS-RUC'):
        """RUCLSM_SOILVEGPARM (rucinit.F:248-520) through the Rust decoder: 955 module_ruc_land words."""
        import subprocess
        from ..resources_build import default_cache, locate_decoders
        decoder = locate_decoders(default_cache())['ruc_table_decode']
        raw = subprocess.run([str(decoder), str(directory), landuse, soil], check=True,
                             capture_output=True).stdout
        words = np.frombuffer(raw, dtype='<u4')
        from .ruc_lsm_8c9f1a8 import TABLE_WORDS
        if words.size != TABLE_WORDS:
            raise ValueError(f'RUC table decoder wrote {words.size} words, the module holds {TABLE_WORDS}')
        return words

    def initialize_ruc(self):
        """ruc_init_forMPAS (rucinit.F:58-245): module_ruc_land tables, then ruclsminit on CUDA.

        The tables are the Rust decoder's words, held as module_ruc_land.* (native kinds and extents), the
        names ruc_lsm_8c9f1a8 reads. ruclsminit's numerics are the RUC port's own certified CUDA entry
        (init_rucinit, ruc_lsm_8c9f1a8.launch_boundary), over every local cell (1:nCells).
        """
        from . import ruc_lsm_8c9f1a8 as ruc
        n, c = self.native, self.configs
        mminlu = n.get_text_field('sfc_input', 'mminlu')
        landuse = self.RUC_LANDUSE.get(mminlu)
        if landuse is None:
            raise RuntimeError(f'RUC LSM: land use classification {mminlu!r} has no RUC table (rucinit.F:158-166 '
                               'maps USGS, MODIS and MODIFIED_IGBP_MODIS_NOAH only); native stops with '
                               '"land use dataset not found in VEGPARM.TBL"')
        p = self.session.device_pools
        cp = p.cp
        words = self.ruc_table_words(self.table_directory(), landuse)
        for name, spec in ruc.TABLE_LAYOUT.items():
            if spec['character']:
                continue
            value = words[spec['offset']:spec['offset'] + spec['words']].view(
                np.int32 if spec['type'] == 'int' else np.float32)
            n.put_table('module_ruc_land.' + name, value.copy() if spec['dim'] else value[0].copy(),
                        lower=(1,) * len(spec['dim']))
        tables = cp.asarray(words.view(np.float32))
        self.native.scheme_state['ruc_table_words'] = tables
        cells, levels = self.dimensions['nCells'], self.dimensions['nSoilLevels']
        full = lambda value: cp.full(cells, value, dtype=cp.int32)
        def column(pool, name):
            return cp.asarray(np.ascontiguousarray(n.get_field(pool, name)[..., :cells]))
        arguments = dict(restart=full(int(c['config_do_restart'])), cycling=full(int(c['config_do_DAcycling'])),
                         # Each CUDA thread is one native column: the per-cell extent is 1 (COLUMN_EXTENTS).
                         ncells=full(1), nsoillevels=full(levels))
        for pool, names in (('sfc_input', ('ivgtyp', 'isltyp', 'tslb', 'smois', 'xice', 'sh2o', 'snow', 'snowh')),
                            ('diag_physics', ('smfr3d', 'mavail', 'znt'))):
            for name in names:
                arguments[name] = column(pool, name)
        storage = ruc.launch_boundary('init_rucinit', arguments, tables)
        cp.cuda.get_current_stream().synchronize()
        if int(storage['errors'].get()[0]):
            raise RuntimeError('module_ruc_land.F: lsminit: out of range value of ISLTYP. Is this field in the input?')
        for pool, names in (('sfc_input', ('sh2o', 'snowh')), ('diag_physics', ('smfr3d', 'mavail', 'znt'))):
            for name in names:
                value = n.get_field(pool, name)
                value[..., :cells] = cp.asnumpy(arguments[name])
                n.put_field(pool, name, value, nlevels=n._fields[pool, name][2])

    def initialize_ngw(self):
        """Fork init.F:452-466: initialize NGW spectra, profiles and latitude interpolation on CUDA."""
        from ..ugwp_ngw_v842 import initialize
        cp = self.session.device_pools.cp
        cells, levels = self.dimensions['nCells'], self.dimensions['nVertLevels']
        _, indices, weights = initialize(cp.asarray(self.field('mesh', 'latCell')[:cells]),
            cp.asarray(self.field('ngw_input', 'LATS')),
            cp.asarray(self.field('mesh', 'rdzw')[:levels]),
            cp.asarray(self.field('mesh', 'dzu')[:levels]))
        for name, value in (('jindx1_tau', indices[0]), ('jindx2_tau', indices[1]),
                            ('ddy_j1tau', weights[0]), ('ddy_j2tau', weights[1])):
            self.write('ngw_input', name, cp.asnumpy(value), count=cells)

    def check_orographic_statistics(self):
        """init.F:448-451: bl_ugwp_gwdo reads the ugwp_oro_data_in stream before the first step.

        On the ported framework those words arrive with the pools: the route passes the orographic statistics
        file to the Rust pool reader (forecast_v842 sources, --orographic-statistics), which records each
        field's origin. Native would read the stream's file; a field that came from the Registry's initial
        value instead would run the drag on zero var2d/oa/ol statistics, so that is refused by name.
        """
        import json
        from pathlib import Path
        stream = self.session.layout.root.find("./streams/stream[@name='ugwp_oro_data_in']")
        names = [] if stream is None else [v.get('name') for v in stream.findall('var')]
        manifest = json.loads((Path(self.session.hydration.payload) / 'manifest.json').read_text())
        origins = {r['name']: r.get('origins') or [] for r in manifest['fields']
                   if r['pool'] in ('sfc_input', 'mesh')}
        missing = [name for name in names if not origins.get(name)
                   or all(str(o).startswith('registry-initial-value') for o in origins[name])]
        if missing:
            raise RuntimeError('bl_ugwp_gwdo needs the orographic statistics of the ugwp_oro_data_in stream '
                               f'(ugwp_oro_data.nc); the pools hold Registry initial values for {missing[:4]}... '
                               'Pass --orographic-statistics so the pool reader reads them')

    def initialize_convection(self):
        c = self.configs
        scheme = c['config_convection_scheme'].strip()
        if scheme in ('cu_ntiedtke', 'cu_tiedtke', 'cu_grell_freitas', 'cu_grell_freitas_li'):
            # Fork driver_convection.F:495-541 (1d6e370e) and 484-530 (44c0849): init_convection's select has a
            # case for cu_kain_fritsch only, so these selectors initialize nothing. The CUDA wrappers bind their
            # constants on the first call (GF-Li: rrfs.cu_gfl_bf6a4b0). THE BREAKAGE THIS RETIRES: hrrrv5 +
            # cu_grell_freitas_li (zoom globe, 2026-10-31) refused here, a gap the fork source does not have.
            return
        if scheme == 'cu_kain_fritsch':
            if not c['config_do_restart']:
                self.write('diag_physics', 'nca', np.float32(-100), count=self.dimensions['nCells'])
            return
        self._gap('convection_init')

    def initialize_microphysics(self):
        """Fork init_microphysics: the row initializer, or TEMPO tables and aerosols."""
        c = self.configs
        from importlib import import_module
        from ...physics_suites import load_table
        rows = load_table()[getattr(self.session.config, 'table_key', 'mpasphys_rrfs')]['slots']
        descriptor = rows['microphysics'].get('cuda_entries', {}).get(c['config_microp_scheme'].strip(), {})
        if initializer := descriptor.get('initialize'):
            module = import_module('woof.hex.mpasphys.' + descriptor['module'])
            getattr(module, initializer)(self.session.device_pools)
            return
        if c['config_microp_scheme'].strip() == 'mp_wsm6':
            # Fork driver_microphysics.F:294-300 initializes scheme constants only;
            # the shared CUDA WSM6 loader performs that initialization.
            return
        if c['config_microp_scheme'].strip() == 'mp_thompson':
            # Fork driver_microphysics.F:276-290, identical to v8.4.2.
            ns, f = self.dimensions['nCellsSolve'], np.float32
            land = self.field('sfc_input', 'landmask')[:ns]
            nt = self.field('diag_physics', 'nt_c')[:ns]
            nt[land == 1], nt[land == 0] = f(300e6), f(100e6)
            self.write('diag_physics', 'nt_c', nt, count=ns)
            self.write('diag_physics', 'mu_c', np.minimum(f(15), f(f(1000e6)/nt)+f(2)), count=ns)
            return
        if c['config_microp_scheme'].strip() != 'mp_tempo':
            self._gap('microphysics_init')
            return
        if self.native.get_table('module_mp_tempo_params.tcg_racg') is None:
            self.initialize_tempo_tables()
        if c.get('config_tempo_aerosolaware', False):
            self.initialize_tempo_aerosols()

    # module_mp_tempo_driver.F90:457-560: the arrays of the four NOAA lookup files, in read order.
    TEMPO_FILE_TABLES = ('tpi_qrfz', 'tni_qrfz', 'tpg_qrfz', 'tnr_qrfz', 'tpi_qcfz', 'tni_qcfz',
                         'tcs_racs1', 'tmr_racs1', 'tcs_racs2', 'tmr_racs2', 'tcr_sacr1', 'tms_sacr1',
                         'tcr_sacr2', 'tms_sacr2', 'tnr_racs1', 'tnr_racs2', 'tnr_sacr1', 'tnr_sacr2',
                         'tcg_racg', 'tmr_racg', 'tcr_gacr', 'tnr_racg', 'tnr_gacr', 'tnccn_act')

    def tempo_lookup_path(self, filename):
        """One NOAA lookup file: the last named table directory that holds it, else the fetched fork source's.

        Named directories are the run's --native-physics-tables, the same override a native run stages by basename
        (tools/native_physics_reference/rrfs/run_forecast.py --tables). THE BREAKAGE THIS PREVENTS: the ported
        route ignored that option and always read the source tree's copies, so a run could not name the tables a
        native reference ran with (the pinned gnu-strict reference uses its own strict-rebuilt
        qr_acr_qg/qr_acr_qs/freeze_water files; the source copies differ in bytes, and TEMPO then differed from
        native in every column with rain-graupel or rain-snow collection or freezing from step 1).
        """
        return tempo_lookup_path_in(getattr(self.native, 'tables', None), filename,
                                    resource=getattr(self.native, 'resource', 'mpas-source-rrfs'),
                                    pin_set=getattr(self.native, 'pin_set', 'rrfs'))

    # utils.F90 compute_drop_evap, compute_efrw, compute_efsw, qi_aut_qs: computed by tempo_table_compute (Rust,
    # all cores), output in this order, binary64 Fortran order.
    TEMPO_COMPUTED_TABLES = (('tpc_wev', (100, 37, 100)), ('tnc_wev', (100, 37, 100)), ('t_efrw', (100, 100)),
                             ('t_efsw', (100, 100)), ('tps_iaus', (64, 55)), ('tni_iaus', (64, 55)),
                             ('tpi_ide', (64, 55)))
    # Its inputs, in its input order: binary32 scalars, binary32 arrays, binary64 arrays.
    TEMPO_COMPUTE_INPUTS = (('am_r', 'bm_r', 'obmr', 'nu_c_scale', 'rho_w', 'pi', 'av_s', 'bv_s', 'fv_s', 'am_s',
                             'bm_s', 'd0s', 'am_i', 'oig1', 'obmi', 'bm_i', 'mu_i', 'd0i'),
                            ('r_c', 'cce', 'ccg', 'ocg1', 'cie', 'cig', 'nt_i', 'r_i'),
                            ('dc', 'dtc', 't_nc', 'dr', 'ds', 'di', 'dti'))

    def compute_tempo_tables(self):
        """The large tables tempo_init computes, from the parameters and bins already held (Rust, all cores)."""
        import subprocess
        from ..resources_build import default_cache, locate_decoders
        n = self.native
        get = lambda name: n.get_table('module_mp_tempo_params.' + name)[0]
        scalars, f32, f64 = self.TEMPO_COMPUTE_INPUTS
        parts = [np.float32(get(name)).tobytes() for name in scalars]
        parts += [np.asarray(get(name), np.float32).tobytes(order='F') for name in f32]
        parts += [np.asarray(get(name), np.float64).tobytes(order='F') for name in f64]
        tool = locate_decoders(default_cache())['tempo_table_compute']
        raw = subprocess.run([str(tool)], input=b''.join(parts), check=True, capture_output=True).stdout
        offset = 0
        for name, shape in self.TEMPO_COMPUTED_TABLES:
            count = int(np.prod(shape))
            value = np.frombuffer(raw, dtype='<f8', count=count, offset=offset).reshape(shape, order='F')
            offset += count * 8
            n.put_table('module_mp_tempo_params.' + name, value, lower=(1,) * len(shape), copy=False)
        if offset != len(raw):
            raise ValueError('tempo_table_compute wrote words the tables do not hold')

    def initialize_tempo_tables(self):
        """tempo_init (module_mp_tempo_driver.F90:42-182) without a native library.

        1. The PARAMETERs and declaration values tempo_init starts from: words captured from the pinned TEMPO
           source by tools/framework_metadata_rrfs.py (metadata.json tempo_constants), as MPAS's constants.
        2. initialize_graupel_vars / initialize_parameters / the bins: tempo_table_init (Rust), for this run's
           hail-aware and melting-reflectivity flags (hail-size bins: tempo_cfgs default on).
        3. compute_efrw / compute_efsw / compute_drop_evap / qi_aut_qs: tempo_table_compute (Rust, all cores).
        4. The four NOAA lookup files: the certified Rust lookup decoder (lookup_decode), one per thread.
        Each is held as module_mp_tempo_params.<name> in its native kind, the names tempo_v303 reads, and
        tests/rrfs_framework_init_replay.py compares every one against a native session.
        """
        import math
        import subprocess
        from .tempo_v303 import TABLE_LAYOUT
        from ..resources_build import default_cache, locate_decoders
        n, c = self.native, self.configs
        tools = locate_decoders(default_cache())
        constants = n.abi['tempo_constants']
        for name, row in constants.items():
            words = np.array(row['words'], np.uint64 if row['dtype'] == 'float64' else np.uint32)
            value = words.view(np.dtype(row['dtype']).newbyteorder('<')).astype(row['dtype'])
            if row['shape']:
                n.put_table('module_mp_tempo_params.' + name, value.reshape(row['shape']), lower=(1,))
            else:
                n.put_table('module_mp_tempo_params.' + name, value.reshape(()), lower=())
        melt = bool(c.get('config_tempo_refl10cm_from_melting', False))
        flags = np.array([bool(c.get('config_tempo_hailaware', True)), melt, True], '<i4')
        constant = lambda name: n.get_table('module_mp_tempo_params.' + name)[0]
        parts = [flags.tobytes()]
        parts += [np.float32(constant(name)).tobytes() for name in TEMPO_CONSTANT_SCALARS]
        parts += [np.asarray(constant(name), np.float32).tobytes(order='F') for name in TEMPO_CONSTANT_ARRAYS]
        raw = subprocess.run([str(tools['tempo_table_init'])], input=b''.join(parts), check=True,
                             capture_output=True).stdout
        offset = 0
        for name, dtype, shape in (*TEMPO_DERIVED, *TEMPO_HAIL_BINS, *(TEMPO_RADAR_BINS if melt else ())):
            count = math.prod(shape) if shape else 1
            value = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
            offset += count * value.itemsize
            n.put_table('module_mp_tempo_params.' + name,
                        value.reshape(shape, order='F') if shape else value.reshape(()).copy(),
                        lower=(1,) * len(shape), copy=False)
        if offset != len(raw):
            raise ValueError('tempo_table_init wrote words the tables do not hold')
        self.compute_tempo_tables()
        self.read_tempo_lookup_files()

    # module_mp_tempo_driver.F90:457-560: each NOAA lookup file and the arrays it holds, in read order.
    # Stream files (access='stream') carry no record markers; ccn_activate.bin is one sequential record.
    TEMPO_LOOKUP_FILES = (('freeze_water_data_tempo_v3', 0, 6, False), ('qr_acr_qs_data_tempo_v3', 6, 18, False),
                          ('qr_acr_qg_data_tempo_v3', 18, 23, False), ('ccn_activate.bin', 23, 24, True))

    @classmethod
    def _tempo_lookup_words(cls, paths):
        """The four big-endian NOAA files through the certified Rust lookup decoder (lookup_decode cdylib),
        one file per thread (its SHA-256, then its arrays): ({filename: sha256}, [arrays of each file in order])."""
        import hashlib
        from concurrent.futures import ThreadPoolExecutor
        from .tempo_v303 import TABLE_LAYOUT
        from ..resources_build import default_cache, locate_decoders
        from ..table_resources_v842 import RustLookupRecords
        library = locate_decoders(default_cache())['lookup_decode']
        with open(library, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        decoder = RustLookupRecords({'path': str(library), 'sha256': digest})
        shapes = {name: tuple(d for d in shape if d != 1) for name, shape in TABLE_LAYOUT}
        def read(row, path):
            filename, first, last, records = row
            with open(path, 'rb') as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            return sha, decoder.arrays(path, cls.TEMPO_FILE_TABLES[first:last], shapes,
                                       records=records, little=False)
        with ThreadPoolExecutor(len(cls.TEMPO_LOOKUP_FILES)) as pool:
            results = list(pool.map(read, cls.TEMPO_LOOKUP_FILES, paths))
        # The receipt names the exact lookup bytes this run read (native records its staged tables the same way).
        return ({row[0]: sha for row, (sha, _) in zip(cls.TEMPO_LOOKUP_FILES, results)},
                [arrays for _, arrays in results])

    @classmethod
    def prefetch_tables(cls, configs, named, *, resource='mpas-source-rrfs', pin_set='rrfs'):
        """Start reading the TEMPO lookup files on a background thread, before the dycore inputs.

        THE COST THIS REMOVES (RRFS hrrrv5, mv3km, V6000, 2026-10-28): physics_init read and hashed the four
        NOAA lookup files (~1 GB) at ~2.1 s of start-up, after the dycore was built, while the card waited.  They
        depend on nothing the run computes, so they are read beside the dycore inputs and build (the Rust
        decoder and hashlib release the GIL) and read_tempo_lookup_files takes the same words.  Any failure here
        is left to read_tempo_lookup_files, which reads them again itself and raises its own error.

        ``resource``/``pin_set``: the fork source the backend resolves its tables from (the row's bound backend
        arguments; tempo_lookup_path).  THE BREAKAGE THIS PREVENTS (RRFS hrrrv5 on the 44c0849 default row,
        mv3km, V8X6000, 2026-10-29): the read-ahead always resolved the 1d6e370e fork's files, so on the default
        row it never matched, read_tempo_lookup_files read the 44c0849 files again on the main thread, and the
        unused read-ahead (~0.93 GB) stayed held for the whole run in every rank process.
        """
        import threading
        from concurrent.futures import Future
        try:
            if str(configs.get('config_microp_scheme', '')).strip() != 'mp_tempo':
                return None
            paths = tuple(tempo_lookup_path_in(named, row[0], resource=resource, pin_set=pin_set)
                          for row in cls.TEMPO_LOOKUP_FILES)
        except Exception:  # noqa: BLE001 - read_tempo_lookup_files resolves and raises by itself
            return None
        key = (threading.get_ident(), cls.TEMPO_LOOKUP_FILES, cls.TEMPO_FILE_TABLES, paths)
        if key in _LOOKUP_PREFETCH:
            return _LOOKUP_PREFETCH[key]
        future = Future()

        def run():
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(cls._tempo_lookup_words(paths))
            except BaseException as error:  # noqa: BLE001 - surfaced as a miss in read_tempo_lookup_files
                future.set_exception(error)
        _LOOKUP_PREFETCH[key] = future
        threading.Thread(target=run, name='hex-tempo-lookup-prefetch', daemon=True).start()
        return future

    def read_tempo_lookup_files(self):
        """The four NOAA lookup files' arrays, straight into the held tables (about 1 GB): the words
        prefetch_tables read ahead for these paths, else read here (_tempo_lookup_words)."""
        paths = tuple(self.tempo_lookup_path(row[0]) for row in self.TEMPO_LOOKUP_FILES)
        import threading
        ident = threading.get_ident()
        future = _LOOKUP_PREFETCH.pop((ident, self.TEMPO_LOOKUP_FILES, self.TEMPO_FILE_TABLES, paths), None)
        # A read-ahead on this thread for other paths is never taken: drop it, so its ~1 GB of words is not held
        # for the rest of the run (prefetch_tables names the breakage).
        for key in [key for key in list(_LOOKUP_PREFETCH) if key[0] == ident]:
            _LOOKUP_PREFETCH.pop(key, None)
        words = None
        if future is not None:
            try:
                words = future.result()
            except Exception:  # noqa: BLE001 - read again here; this read raises its own error
                words = None
        lookup_sha256, files = words if words is not None else self._tempo_lookup_words(paths)
        self.native.lookup_sha256 = dict(lookup_sha256)
        for arrays in files:
            for name, value in arrays.items():
                self.native.put_table('module_mp_tempo_params.' + name, value,
                                      lower=(1,) * value.ndim, copy=False)

    def initialize_tempo_aerosols(self):
        """init_tempo.F:40-185: climatological profiles where the input has none, then nwfa2d/nifa2d."""
        if self.configs['config_do_restart']:
            return
        p, n = self.session.device_pools, self.native
        cp, ns, nz = p.cp, self.dimensions['nCellsSolve'], self.dimensions['nVertLevels']
        if not hasattr(self, '_kernels'):
            from .driver_rrfs import StrictPhysicsKernelCache
            self._kernels = StrictPhysicsKernelCache()
        kernel = lambda name: self._kernels.raw_kernel(name, bind_common_carrier_math(_TEMPO_AEROSOLS_CUDA),
                                                       module_key='rrfs-tempo-aerosol-initialization')
        level = 1
        scalars = n.get_field('state', 'scalars', level=level)          # (num_scalars, nVertLevels, nCells+1)
        zgrid = cp.asarray(np.ascontiguousarray(n.get_field('mesh', 'zgrid').T))   # (nCells+1, nVertLevels+1)
        area = cp.asarray(np.ascontiguousarray(n.get_field('mesh', 'areaCell')))
        f = np.float32
        index = {name: n._dimension_values['state', 'index_' + name] - 1 for name in ('nwfa', 'nifa')}
        columns = {}
        # init_tempo.F:100-105,131-136: the test reads GLOBAL extrema (mpas_dmpar_min_real/max_real); a rank with
        # no owned cells contributes Fortran's empty minval/maxval (+huge/-huge). One card has no reducer.
        reduce = getattr(self.session, 'rank_reduce', None)
        for name, (base, scale) in (('nwfa', (f(50.0e6), f(300.0e6))), ('nifa', (f(0.5e6), f(1.5e6)))):
            words = scalars[index[name], :, :ns]
            column = cp.asarray(np.ascontiguousarray(scalars[index[name]].T))  # (nCells+1, nVertLevels)
            low, high = (float(words.min()), float(words.max())) if ns else (np.inf, -np.inf)
            if reduce is not None:
                low, high = reduce(low, 'min'), reduce(high, 'max')
            if not (low == 0 and high == 0):
                columns[name] = column
                continue
            ratio = _log_word(f(base / scale))
            kernel('rrfs_tempo_aerosol_profile')(((ns+127)//128,), (128,), (np.int32(ns), np.int32(nz),
                np.int32(zgrid.shape[1]), np.int32(column.shape[1]), zgrid, column, base, scale, ratio))
            scalars[index[name], :, :ns] = cp.asnumpy(column).T[:, :ns]
            columns[name] = column
        n.put_field('state', 'scalars', scalars, level=level, nlevels=n._fields['state', 'scalars'][2])
        nwfa2d = cp.asarray(n.get_field('diag_physics', 'nwfa2d'))
        nifa2d = cp.asarray(n.get_field('diag_physics', 'nifa2d'))
        kernel('rrfs_tempo_aerosol_surface')(((ns+127)//128,), (128,), (np.int32(ns),
            np.int32(zgrid.shape[1]), np.int32(columns['nwfa'].shape[1]), zgrid, columns['nwfa'], area,
            nwfa2d, nifa2d, f(11.1e6), f(9999.e6)))
        self.write('diag_physics', 'nwfa2d', cp.asnumpy(nwfa2d))
        self.write('diag_physics', 'nifa2d', cp.asnumpy(nifa2d))

    def atmphys_finalize(self, call):
        self.native._initialized = False
        self.native.scheme_state.clear()


# The name the RRFS row's framework entry binds (route_table: ...framework_init_rrfs:FrameworkInitialization).
FrameworkInitialization = FrameworkInitializationRRFS
