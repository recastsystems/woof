"""GPU resident WRF DFI carriers and graph operations.

The owner supplies actual native carriers in [species, level, y, x] layout.
This module does not substitute zeros for missing model state, infer fields
from names, or admit an unwired forecast selector. In particular, supplying
these operators is separate from supplying an advancing signed model graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

DYNAMICS = {
    'mu_2': 'dfi_mu', 'u_2': 'dfi_u', 'v_2': 'dfi_v', 'w_2': 'dfi_w',
    'ww': 'dfi_ww', 't_2': 'dfi_t', 'phb': 'dfi_phb', 'ph0': 'dfi_ph0',
    'php': 'dfi_php', 'p': 'dfi_p', 'ph_2': 'dfi_ph', 'tke_2': 'dfi_tke',
    'al': 'dfi_al', 'alt': 'dfi_alt', 'pb': 'dfi_pb',
}
RADII = ('re_cloud', 're_ice', 're_snow')
GSFC_RADII = tuple('re_'+name+'_gsfc' for name in
                  ('cloud', 'rain', 'ice', 'snow', 'graupel', 'hail'))
SURFACE_WATER = ('snow', 'snowh', 'snowc', 'canwat')
SURFACE_SOIL = ('tsk', 'tslb', 'smois')
RUC_SURFACE = ('qvg', 'tsnav', 'soilt1', 'smfr3d', 'keepfr3dflag')
COORDS = ('c1f', 'c2f', 'c1h', 'c2h', 'c3f', 'c4f', 'c3h', 'c4h',
          'rdnw', 'rdn', 'dnw')


class DfiModelDiagnostics:
    """Keep WRF's WW, PHP and PH0 at their native calculation points.

    Attach as ``state._dfi_diagnostics`` before a DFI pass. These buffers
    survive both acoustic scratch reuse and phase transitions. They do not
    alias ``state.php`` (PH perturbation) or ``state.ww_pp`` (WW acoustic
    perturbation). Native cold allocations start WW, PHP and PH0 at zero.
    PH0 is assigned by rebalance, not by ordinary cold start_domain_em.
    The real cold WSM6/Noah capture witnesses all three zero allocations.
    """

    def __init__(self, state):
        import cupy as cp
        from woof.core.kernels import get_kernel

        self.state = state
        self.ww = cp.zeros_like(state.w)
        self.php = cp.zeros_like(state.p)
        self.ph0 = cp.zeros_like(state.php)
        self._kernels = {name: get_kernel('dfi', 'dfi_capture_'+name)
                         for name in ('ww', 'php', 'ph0')}

    def _geopotential(self, name, output):
        state = self.state
        self._kernels[name](((output.size+255)//256,), (256,),
            (state.php, state.phb, output, np.uint64(output.size),
             np.uint32(state.mup.size), np.int32(state.phb.ndim == 3)))

    def initialize_ph0(self):
        self._geopotential('ph0', self.ph0)

    def prepare_stage(self):
        self._geopotential('php', self.php)

    def finish_stage(self, reference):
        self._kernels['ww'](((self.ww.size+255)//256,), (256,),
            (self.state.ww_pp, reference, self.ww, np.uint64(self.ww.size)))


@dataclass(frozen=True)
class NativeLayout:
    columns: int
    levels: int
    moist_species: int
    qv: int
    qc: int
    qi: int
    first_scalar: int
    theta_m: bool
    hypsometric: int
    p_top: float
    surface_physics: int
    microphysics_dfi: int
    save_hydrometeors: bool

    def __post_init__(self):
        if self.columns <= 0 or self.levels < 2:
            raise ValueError('DFI needs positive columns and at least two mass levels')
        if self.hypsometric not in (1, 2):
            raise ValueError('WRF DFI hypsometric_opt must be 1 or 2')
        if not (0 <= self.qv < self.moist_species and
                self.first_scalar == self.qv):
            raise ValueError('DFI moisture inventory does not contain native QV')
        if not (
                0 <= self.qc < self.moist_species and
                0 <= self.qi < self.moist_species):
            raise ValueError('native backward DFI RH needs its QC and QI slots, including inactive package slots')


class DeviceDfiState:
    """Own native DFI work arrays; keep all graph resources alive.

    Construction, descriptor allocation, graph capture and final receipts are
    setup/boundary work. A pass is submitted by the Rust runtime, with CUDA
    graphs for these operations and for the actual signed model advance.
    """

    def __init__(self, fields: Mapping, layout: NativeLayout, *, strict=None):
        import cupy as cp
        from woof.core.kernels import compile_module_preserving_subnormals, module_source
        from woof.numerics import is_strict
        process_strict = is_strict()
        if strict is None:
            strict = process_strict
        if strict and not process_strict:
            raise ValueError('DFI strict proof requires GPUWM_NUMERICS=strict before process startup')
        self.layout = layout
        self.fields = dict(fields)
        self.radii = GSFC_RADII if layout.microphysics_dfi == 7 else RADII
        self.soil = SURFACE_SOIL + (RUC_SURFACE if layout.surface_physics == 3 else ())
        required = set(DYNAMICS) | {'moist', 'scalar'} | set(self.radii)
        required |= set(SURFACE_WATER+self.soil)
        required |= {'alb', 'p_hyd', 'mub', 'psfc'} | set(COORDS)
        missing = sorted(required - self.fields.keys())
        if missing:
            raise ValueError('native DFI carrier inventory is incomplete: '+', '.join(missing))
        for name in required:
            a = self.fields[name]
            if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or not a.flags.c_contiguous:
                raise TypeError(f'DFI {name} must be a contiguous GPU float32 carrier')
        volume = layout.levels*layout.columns
        if self.fields['moist'].size != layout.moist_species*volume:
            raise ValueError('DFI moisture package size differs from the native inventory')
        for n in ('t_2', 'p', 'pb', 'alb', 'alt', 'al', 'p_hyd'):
            if self.fields[n].size != volume:
                raise ValueError(f'DFI {n} must contain every mass-level column')
        for n in ('ph_2', 'phb', 'ph0'):
            if self.fields[n].size != (layout.levels+1)*layout.columns:
                raise ValueError(f'DFI {n} must contain every interface-level column')
        for n in SURFACE_WATER+('tsk', 'mu_2', 'mub', 'psfc'):
            if self.fields[n].size != layout.columns:
                raise ValueError(f'DFI {n} surface extent differs from its columns')
        for n in COORDS:
            needed=layout.levels+int(n in ('c1f','c2f','c3f','c4f'))
            if self.fields[n].size < needed:
                raise ValueError(f'DFI {n} is missing native vertical-coordinate entries')
        readonly=set(COORDS)|{'mub','alb'}
        extents=sorted((a.data.ptr,a.data.ptr+a.nbytes,n)
                       for n,a in self.fields.items() if n in required)
        for i,(lo,hi,n) in enumerate(extents):
            for other_lo,other_hi,other in extents[i+1:]:
                if other_lo>=hi:
                    break
                if n not in readonly or other not in readonly:
                    raise ValueError(f'DFI native carriers {n} and {other} overlap')
        options = ('-std=c++17', '--ftz=false')
        if strict:
            options += ('-DGPUWM_WRF_EXACT=1', '--fmad=false', '--prec-div=true', '--prec-sqrt=true')
        self.modules = {n: compile_module_preserving_subnormals(n, module_source(n), options)
                        for n in ('dfi', 'dfi_reset')}
        self.kernels = {}
        for name in ('clear_fields', 'accumulate_fields', 'advance_coefficient',
                     'add_coefficient', 'advance_cursor', 'reset_fields'):
            self.kernels[name] = self.modules['dfi'].get_function('dfi_'+name)
        for name in ('snapshot_fields', 'reset_surface_water', 'reset_surface_soil',
                     'save_relative_humidity', 'restore_saturated_vapour', 'rebalance_columns'):
            self.kernels[name] = self.modules['dfi_reset'].get_function('dfi_'+name)
        self.saved = {b: cp.zeros_like(self.fields[a]) for a,b in DYNAMICS.items()}
        for name in ('moist', 'scalar')+self.radii+SURFACE_WATER+self.soil:
            self.saved['dfi_'+name] = cp.zeros_like(self.fields[name])
        self.saved['dfi_rh'] = cp.zeros_like(self.fields['p'])
        self.cursor = cp.zeros(1, cp.uint32)
        self.total = cp.zeros(1, cp.float32)
        self.error = cp.zeros(1, cp.uint32)
        self.moist = self.fields['moist'].reshape(layout.moist_species, volume)
        self.saved_moist = self.saved['dfi_moist'].reshape(layout.moist_species, volume)
        self._owners = []
        self._setup_descriptors(cp)

    def _rows(self, cp, pairs):
        if not pairs:
            return None
        rows = cp.asarray(np.asarray([[a.data.ptr,b.data.ptr,a.size,flag]
                                      for a,b,flag in pairs], dtype=np.uint64))
        self._owners.extend([rows, *[a for a,b,f in pairs], *[b for a,b,f in pairs]])
        return rows, ((max(a.size for a,b,f in pairs)+255)//256, len(pairs))

    def _setup_descriptors(self, cp):
        l=self.layout;f=self.fields;s=self.saved
        dynamics=[(f[a],s[b],0) for a,b in DYNAMICS.items()]
        if l.save_hydrometeors:
            filtered=[(self.moist[l.qv], self.saved_moist[l.qv], 1)]
        else:
            filtered=[(f[n],s['dfi_'+n],1) for n in ('moist','scalar')+self.radii]
        self.filtered=self._rows(cp,dynamics+filtered)
        # Native dfi_clear_accumulation clears LIVE effective radii and
        # leaves their accumulation arrays untouched across passes.
        cleared = dynamics + (filtered if l.save_hydrometeors else filtered[:2])
        self.clear_accumulators=self._rows(cp,cleared)
        self.clear_radii=(None if l.save_hydrometeors else
            self._rows(cp,[(s['dfi_'+n],f[n],0) for n in self.radii]))
        self.water=self._rows(cp,[(f[n],s['dfi_'+n],0) for n in SURFACE_WATER])
        self.soils=self._rows(cp,[(f[n],s['dfi_'+n],0) for n in self.soil])
        snapshots=[(f[n],s['dfi_'+n],0) for n in SURFACE_WATER+self.soil]
        restore=[(s['dfi_scalar'],f['scalar'],0)]
        hydro=[(self.saved_moist[l.first_scalar+1:],self.moist[l.first_scalar+1:],0)]
        if l.save_hydrometeors:
            snapshots += [(self.moist[l.first_scalar+1:],self.saved_moist[l.first_scalar+1:],1),
                          (f['scalar'],s['dfi_scalar'],1)]
        self.snapshot=self._rows(cp,snapshots)
        self.restore_backward=self._rows(cp,restore)
        self.restore_forward=self._rows(cp,restore+hydro)
        names=('mu_2','mub','t_2','p','pb','alb','alt','al','p_hyd','ph_2','phb','ph0','psfc')
        moist=self.moist[l.first_scalar:].reshape(-1)
        arrays=[f[n] for n in names]+[moist]+[f[n] for n in COORDS]
        self.rebalance_pointers=cp.asarray(np.asarray([a.data.ptr for a in arrays],dtype=np.uint64))
        self._owners.extend(arrays+[self.rebalance_pointers])

    def _launch_rows(self, name, descriptor, *extra):
        if descriptor is not None:
            rows, grid=descriptor
            self.kernels[name](grid,(256,),(rows,*extra))

    def save(self, *, backward):
        l=self.layout;f=self.fields
        self._launch_rows('snapshot_fields',self.snapshot)
        if backward:
            words=l.levels*l.columns
            self.kernels['save_relative_humidity'](((words+255)//256,),(256,),
                (f['t_2'],f['p'],f['pb'],self.moist[l.qv],self.moist[l.qc],
                 self.moist[l.qi],self.saved['dfi_rh'],np.uint64(words),np.int32(l.theta_m)))

    def clear(self):
        self._launch_rows('clear_fields',self.clear_accumulators)
        self._launch_rows('clear_fields',self.clear_radii)
        self.cursor.fill(0);self.total.fill(0);self.error.fill(0)

    def accumulate(self, weights, *, advance_cursor=True):
        self._launch_rows('accumulate_fields',self.filtered,weights,self.cursor,
                          np.uint32(weights.size),self.error)
        self.kernels['advance_coefficient' if advance_cursor else 'add_coefficient']((1,),(1,),
            (weights,self.cursor,np.uint32(weights.size),self.total,self.error))

    def advance(self):
        """Advance the DFI coefficient index once per actual model solve."""
        self.kernels['advance_cursor']((1,),(1,),(self.cursor,))

    def rebalance(self):
        l=self.layout
        self.kernels['rebalance_columns'](((l.columns+127)//128,),(128,),
            (self.rebalance_pointers,np.uint32(l.columns),np.uint32(l.levels),
             np.uint32(l.moist_species-l.first_scalar),np.int32(l.theta_m),
             np.int32(l.hypsometric),np.float32(l.p_top)))

    def restore_hydrometeors(self):
        if self.layout.save_hydrometeors:
            self._launch_rows('snapshot_fields',self.restore_forward)

    def reset(self, *, backward):
        l=self.layout;f=self.fields
        self._launch_rows('reset_fields',self.filtered,self.total,self.error)
        self._launch_rows('reset_surface_water',self.water,np.int32(l.save_hydrometeors))
        self._launch_rows('reset_surface_soil',self.soils,f['snowc'],np.uint32(l.columns),
                          np.int32(l.save_hydrometeors))
        if l.save_hydrometeors:
            self._launch_rows('snapshot_fields',self.restore_backward if backward else self.restore_forward)
            if not backward:
                words=l.levels*l.columns
                self.kernels['restore_saturated_vapour'](((words+255)//256,),(256,),
                    (f['t_2'],f['p'],f['pb'],self.moist[l.qv],self.saved['dfi_rh'],
                     np.uint64(words),np.int32(l.theta_m)))
            self.rebalance()

    def capture_graph(self, operation, *, stream):
        """Capture after setup/compilation, retaining graph and all buffers."""
        with stream:
            stream.begin_capture()
            try:
                operation()
                graph=stream.end_capture()
            except BaseException:
                # End an invalid capture so the stream has a defined state.
                try:
                    stream.end_capture()
                except BaseException:
                    pass
                raise
        self._owners.append(graph)
        return graph
