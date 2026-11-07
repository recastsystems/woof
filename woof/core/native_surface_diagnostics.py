"""Device-array ABI for public WRF RASM and solar diagnostics.

Production callers bind arrays from the checked allocation inventory once;
standalone callers validate every argument by default. The caller supplies native producers, including the native top memory plane and radii in metres. This ABI does not admit a namelist selector on its
own. Native absent-radius branches preserve the incoming output words.
"""
from __future__ import annotations
import numpy as np
from woof.core.level_diagnostics import _array

SURFACE = tuple('psfc tsk t2 th2 q2 u10 v10 hfx lh swdnb glw lwupb swupb swupt swdnt lwupt lwdnt'.split())
MEANS = ('psfc','tsk','pmsl',*SURFACE[2:])
SOLAR_PROFILES = tuple('rho dz8w ph phb cldfrac3d qv qc qi qs qc_tot qi_tot re_cloud re_ice re_snow'.split())
SOLAR_PLANES = ('coszen','swdnb','swdnt')
SOLAR_OUTPUTS = tuple('clrnidx sza ghi_accum cldfrac2d wvp2d lwp2d iwp2d swp2d wp2d_sum lwp2d_tot iwp2d_tot wp2d_tot_sum re_cloud_path re_ice_path re_snow_path re_cloud_path_tot re_ice_path_tot tau_qc tau_qi tau_qs tau_qc_tot tau_qi_tot cbase ctop cbase_tot ctop_tot'.split())
def _launch(name, count, args):
    from woof.core.kernels import get_kernel
    get_kernel('native_surface_diagnostics',name)(((count+127)//128,),(128,),args)

def solar(profiles, planes, output, *, dt, first, last, present=(True,True,True,True), radii=(1,1,1), _validated=False):
    """Native zero-based inclusive kts/kte; integration ends at kte-1.

    ``present`` is F_QV followed by the three conjunctions of F_Qx and
    P_Qx>PARAM_FIRST_SCALAR. ``radii`` is native HAS_REQC/I/S, not truthiness.
    Every profile includes the native top memory plane.
    """
    nz,ny,nx=profiles['rho'].shape
    if not 0<=first<last<nz-1:
        raise ValueError('solar requires 0 <= kts < kte < memory_top for cloud face reads')
    if not _validated:
        for key in SOLAR_PROFILES: _array(profiles[key],(nz,ny,nx),key)
        for key in SOLAR_PLANES: _array(planes[key],(ny,nx),key)
        for key in SOLAR_OUTPUTS: _array(output[key],(ny,nx),key)
    _launch('solar_diag',nx*ny,(*(profiles[k] for k in SOLAR_PROFILES),
        *(planes[k] for k in SOLAR_PLANES),*(output[k] for k in SOLAR_OUTPUTS),
        np.int32(nz),np.int32(nx*ny),np.int32(first),np.int32(last),np.float32(dt),
        *(np.int32(x) for x in (*present,*radii))))

def rasm_mean(surface, atmosphere, output, counter, *, reset, finish, _validated=False):
    """Calendar decisions are host clock metadata, column work is CUDA."""
    ny,nx=surface['psfc'].shape
    if not _validated:
        for k in SURFACE: _array(surface[k],(ny,nx),k)
        for k in ('t','p','pb','moist','ht'): _array(atmosphere[k],(ny,nx),k)
        for k in MEANS: _array(output[k],(ny,nx),k)
        if counter.dtype!=np.dtype('int32') or counter.shape!=(ny,nx):
            raise ValueError('RASM counter must be an int32 column plane')
    _launch('rasm_mean',nx*ny,(*(surface[k] for k in SURFACE),
        *(atmosphere[k] for k in ('t','p','pb','moist','ht')),*(output[k] for k in MEANS),
        counter,np.int32(nx*ny),np.int32(reset),np.int32(finish)))

def rasm_diurnal(surface, temporary, diurnal, counter, days, *, reset, reset_month, finish, finish_month, cycle, _validated=False):
    ny,nx=surface['psfc'].shape
    if not 1<=cycle<=8: raise ValueError('native diurnal cycle is 1..8')
    if not _validated:
        for k in SURFACE:
            _array(surface[k],(ny,nx),k); _array(temporary[k],(ny,nx),k)
            _array(diurnal[k],(8,ny,nx),k)
        for a in (counter,days):
            if a.dtype!=np.dtype('int32') or a.shape!=(ny,nx):
                raise ValueError('RASM counters must be int32 column planes')
    _launch('rasm_diurnal',nx*ny,(*(surface[k] for k in SURFACE),
        *(temporary[k] for k in SURFACE),*(diurnal[k] for k in SURFACE),counter,days,
        np.int32(nx*ny),*(np.int32(x) for x in (reset,reset_month,finish,finish_month,cycle))))
