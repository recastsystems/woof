"""RRFS-MPAS fork surface services on resident RKIND arrays, with no native library.

Authority: MPAS-Model-rrfs@1d6e370e src/core_atmosphere/physics/mpas_atmphys_update_surface.F:54-144 and
mpas_atmphys_manager.F:235-241.  Against MPAS-Model v8.4.2 the fork changes only ``physics_update_surface``:

* the greenness update (vegfra and its monthly min/max) runs only when ``config_gvf_update`` is true;
* with ``config_dust_scheme`` not ``off`` it interpolates ``feff_m_in`` -> ``feff`` and ``albedo_drag_m_in`` ->
  ``albedo_drag`` (diag_physics, package mpas_dust_in); at RRFSx 44c0849 (update_surface.F:81-121) the one drag
  partition ``rdrag_m_in`` -> ``rdrag`` replaces both (DUST_MONTHLY: the pairs the route's Registry declares);
* when ``lai12m`` is allocated (package lsm_ruc_in, the RUC land surface) it interpolates ``lai12m`` ->
  ``lai`` (diag_physics), the monthly MODIS leaf area index RUC reads.

The background-albedo update, ``physics_update_sst``, ``physics_update_sstskin``, ``physics_update_deepsoiltemp``,
the radiation diagnostics and the convective precipitation buckets are byte-identical to v8.4.2's
(mpas_atmphys_update_surface.F:145-568, mpas_atmphys_update.F, mpas_atmphys_date_time.F are unchanged in
the fork), so they run the v8.4.2 service's kernels.  monthly_interp_to_date's arithmetic
(mpas_atmphys_date_time.F) is the v8.4.2 ``monthly`` kernel's: (f(m2)*w2 + f(m1)*w1) / span, each operation
rounded to nearest, with no contraction.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from functools import lru_cache

import numpy as np

from ..surface_update_v842 import CUDA_OPTIONS, NativeSurfaceService, _array

# MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_update_surface.F:110-142;
# mpas_atmphys_date_time.F monthly_interp_to_date, monthly_min_max.
_CUDA = r'''
__device__ float add(float a,float b){return __fadd_rn(a,b);}
__device__ float mul(float a,float b){return __fmul_rn(a,b);}
__device__ float divr(float a,float b){return __fdiv_rn(a,b);}
// monthly_interp_to_date: the v8.4.2 monthly kernel's expression.
extern "C" __global__ void fork_interp(int n,int m1,int m2,int w1,int w2,int span,
 const float* f,float* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 out[i]=divr(add(mul(f[m2+12*i],float(w2)),mul(f[m1+12*i],float(w1))),float(span));
}
// update_surface.F:112-120: sfc_albbck = interp / 100., 0.08 over water.
extern "C" __global__ void fork_albedo(int n,int m1,int m2,int w1,int w2,int span,
 const float* alb,const int* land,float* bg){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float v=divr(add(mul(alb[m2+12*i],float(w2)),mul(alb[m1+12*i],float(w1))),float(span));
 bg[i]=land[i]==0?0.08f:divr(v,100.f);
}
// monthly_min_max over the twelve months.
extern "C" __global__ void fork_minmax(int n,const float* f,float* lo,float* hi){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float mn=f[12*i],mx=mn;
 for(int k=1;k<12;k++){float v=f[k+12*i];if(v<mn)mn=v;if(v>mx)mx=v;}
 lo[i]=mn;hi[i]=mx;
}
'''


def _module():
    """The fork surface module loaded on the calling card (one handle per card, as surface_update_v842)."""
    import cupy as cp
    return _module_on(int(cp.cuda.Device().id))


@lru_cache(maxsize=None)
def _module_on(device: int):
    import cupy as cp
    from cupy.cuda import compiler
    # Direct NVRTC keeps --ftz=false (CuPy's RawModule cache appends forced FTZ).
    binary, _ = compiler.compile_using_nvrtc(
        _CUDA, options=tuple(_hex_numerics.options(CUDA_OPTIONS, keep_ftz=True)),
        arch=cp.cuda.Device(device).compute_capability)
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(binary)
    return module


def _launch(name, n, args):
    if n:
        _module().get_function(name)(((n + 127) // 128,), (128,), (np.int32(n), *args))


# FENGSHA monthly drag inputs (monthly, current-month target), in native order: 1d6e370e update_surface.F:124-125
# (feff, albedo_drag); 44c0849 update_surface.F:120 (rdrag). A route uses the pairs its Registry declares.
DUST_MONTHLY = (("feff_m_in", "feff"), ("albedo_drag_m_in", "albedo_drag"), ("rdrag_m_in", "rdrag"))


def physics_update_surface(current_date, *, config_sfc_albedo, config_gvf_update, config_dust_scheme,
                           mesh, sfc_input, diag_physics, calendar, lai_allocated, dust_monthly=DUST_MONTHLY[:2]):
    """mpas_atmphys_update_surface.F:54-144 (fork).  Returns the (pool, field) words it changed."""
    n = mesh["nCellsSolve"]
    capacity = mesh.get("nCells", n)
    weights = tuple(np.int32(v) for v in calendar.weights(current_date))
    changed = []
    if config_sfc_albedo:
        alb = _array(sfc_input, "albedo12m", (12, capacity), order="F")
        land = _array(sfc_input, "landmask", (capacity,), "int32")
        _launch("fork_albedo", n, (*weights, alb, land, _array(sfc_input, "sfc_albbck", (capacity,))))
        changed.append(("sfc_input", "sfc_albbck"))
    if str(config_dust_scheme).strip() != "off":
        for monthly, out in dust_monthly:
            _launch("fork_interp", n, (*weights, _array(diag_physics, monthly, (12, capacity), order="F"),
                                       _array(diag_physics, out, (capacity,))))
            changed.append(("diag_physics", out))
    if config_gvf_update:
        green = _array(sfc_input, "greenfrac", (12, capacity), order="F")
        _launch("fork_interp", n, (*weights, green, _array(sfc_input, "vegfra", (capacity,))))
        _launch("fork_minmax", n, (green, _array(sfc_input, "shdmin", (capacity,)),
                                   _array(sfc_input, "shdmax", (capacity,))))
        changed += [("sfc_input", "vegfra"), ("sfc_input", "shdmin"), ("sfc_input", "shdmax")]
    if lai_allocated:
        _launch("fork_interp", n, (*weights, _array(sfc_input, "lai12m", (12, capacity), order="F"),
                                   _array(diag_physics, "lai", (capacity,))))
        changed.append(("diag_physics", "lai"))
    return tuple(changed)


class ForkSurfaceService(NativeSurfaceService):
    """The fork's driver-boundary surface service: v8.4.2's, with the fork's physics_update_surface."""

    def invoke(self, call):
        """mpas_atmphys_manager.F:235-241 (fork); every other operation is v8.4.2's."""
        if call.slot != 'physics_update_surface':
            return super().invoke(call)
        from .driver_rrfs import package_selection
        options = call.options
        packages = package_selection(self.session.config)
        lai = bool(packages.get('lsm_ruc_in', False))
        dust = str(options.get('config_dust_scheme', 'off')).strip() != 'off'
        sfc_fields = ['landmask', 'albedo12m', 'sfc_albbck', 'greenfrac', 'vegfra', 'shdmin', 'shdmax']
        diag_fields = []
        if lai:
            sfc_fields.append('lai12m')
            diag_fields.append('lai')
        pairs = tuple(pair for pair in DUST_MONTHLY
                      if all(('diag_physics', name) in self.native._fields for name in pair)) if dust else ()
        if dust and not pairs:
            raise KeyError('dust is on but the route Registry declares no FENGSHA monthly drag input')
        for pair in pairs:
            diag_fields += list(pair)
        sfc = self._pool('sfc_input', tuple(sfc_fields))
        diag = self._pool('diag_physics', tuple(diag_fields)) if diag_fields else {}
        mesh = dict(self.session.layout.dimensions)
        mesh['nCells'] = sfc['landmask'].size
        changed = physics_update_surface(
            self.session.arguments.calendar['current_date'],
            config_sfc_albedo=bool(options['config_sfc_albedo']),
            config_gvf_update=bool(options.get('config_gvf_update', True)),
            config_dust_scheme=options.get('config_dust_scheme', 'off'),
            mesh=mesh, sfc_input=sfc, diag_physics=diag, calendar=self.calendar, lai_allocated=lai,
            dust_monthly=pairs)
        for pool, field in changed:
            self._commit(pool, sfc if pool == 'sfc_input' else diag, (field,))
        self.executions['physics_update_surface'] += 1
        return self.receipt()
