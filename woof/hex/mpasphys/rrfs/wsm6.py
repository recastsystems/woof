"""RRFS WSM6: shared MMM numerics, full-column writeback and fork radar.

MPAS-Model-rrfs 1d6e370e interface.F:1223,1291,1528 writes through kte.
driver_microphysics.F:1198 leaves refl10cm_1km_max unchanged in WSM6.
The MMM a4baf7f WSM6 scheme is byte-identical to MPAS-A v8.4.2.
"""
from functools import lru_cache
from .. import wsm6_v842 as shared


def bind_native(provider, call):
    packet = shared.bind_native(provider, call,
        writeback_levels=provider.session.layout.dimensions['nVertLevels'])
    if packet['options']['l_diags']:
        packet['options']['relhum'] = provider['diag']['relhum']
    return packet


def relative_humidity_source():
    # Fork driver_microphysics.F:1383-1435 uses the same RSLF/RSIF
    # polynomials as the already ported v842 diagnostic. WSM6 v[0] is
    # post-scheme potential temperature, v[1] vapor and v[8] pressure.
    from ..kessler_v842 import CUDA_RELHUM_SOURCE
    helpers, marker, _ = CUDA_RELHUM_SOURCE.partition('extern "C" __global__')
    assert marker
    return r'''
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define MAX(a,b) fmaxf((a),(b))
#define MIN(a,b) fminf((a),(b))
''' + helpers + r'''
extern "C" __global__ void wsm6_relhum(const float *v,const float *pi,
 float *rh,int nz,int nc,int ns) {
 int x=blockIdx.x*blockDim.x+threadIdx.x;
 if(x>=nz*nc || x%nc>=ns) return;
 float t=M(v[x],pi[x]), p=v[8*nz*nc+x];
 float qvs=mpas_rslf(p,t);
 if(S(t,273.16f)<=0.f) qvs=mpas_rsif(p,t);
 rh[x]=M(D(v[nz*nc+x],qvs),100.f);
}
'''


@lru_cache(None)
def relative_humidity_kernel(device):
    import cupy as cp
    with cp.cuda.Device(device):
        return shared._compile_cuda(relative_humidity_source(), ('wsm6_relhum',))[0]


def radar_source(header, nz, numerics=None):
    source = shared.cuda_source(header, max_levels=max(2, nz), numerics=numerics)
    line = 'SF(refl10cm_1km_max)=mp_max(SF(refl10cm_1km_max),SF(refl10cm_1km));'
    assert source.count(line) == 1
    source = source.replace(line, '')
    return source


@lru_cache(None)
def radar_kernel(device, header, nz):
    import cupy as cp
    with cp.cuda.Device(device):
        return shared._compile_cuda(radar_source(header, nz), ('wsm6_radar_to_mpas',))[0]


def cuda_entry(call):
    from ...rkind_libm import wsm6_device_math_header
    p = call.pools
    header = wsm6_device_math_header()
    options = dict(p['options'])
    if options['l_diags']:
        theta = p['fields']['theta_m']
        options['radar_kernel'] = radar_kernel(theta.device.id, header, theta.shape[0])
        options['relhum_kernel'] = relative_humidity_kernel(theta.device.id)
    return shared.step(p['fields'], p['constants'], device_math_header=header, **options)
