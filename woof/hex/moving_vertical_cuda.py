"""Ordered physical-height mass integrals and rate interpolation on one CUDA device.

The default on CUDA hosts; an explicit CPU reference remains in moving_vertical.
CPU-only remapping with hidden devices never imports CuPy.
Each target/interface thread visits CSR donors in their stored order, matching
NumPy bincount's additions. No atomics, reassociation, or fused multiply-adds.
The geometry belongs to one immutable HeightConservativeMap; its device copy
is retained for that map's lifetime. Dynamic products are uploaded per call.
"""
from __future__ import annotations

from functools import lru_cache
import os
import numpy as np
from .moving_cuda_work import independent_stream, shared_geometry


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    extern "C" __global__ void cumulative_integral(
        const long long *ptr, const long long *cols, const double *area,
        const unsigned char *layer, const double *sample,
        const double *source_z, const double *centres,
        const double *means, const double *slopes, const double *cumulative,
        const double *ratio, const double *below,
        double *out, const int ns, const int nt, const int levels,
        const long long nnz, const int aligned, const int extend) {
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= (long long)(levels+1)*nt) return;
        const int l = i / nt, row = i % nt;
        if(aligned && l == 0) {out[i]=0.;return;}
        double sum = 0.;
        for(long long j=ptr[row]; j<ptr[row+1]; ++j) {
            const long long c=cols[j];
            double carried;
            if(l==levels) carried=cumulative[(long long)levels*ns+c];
            else {
                const long long q=(long long)l*nnz+j;
                const long long k=layer[q], source=k*ns+c;
                const double z=sample[q], lower=source_z[source], centre=centres[source];
                const double dz=z-centre, dl=lower-centre;
                carried=(cumulative[source]+means[source]*(z-lower))
                    +(.5*slopes[source])*(dz*dz-dl*dl);
                if(extend && z<source_z[c]) carried=-ratio[c]*below[q];
            }
            sum += area[j]*carried;
        }
        out[i]=sum;
    }
    ''', 'cumulative_integral', options=('--fmad=false',))


@shared_geometry('_cuda_cumulative_geometry')
def _cumulative_geometry(weights):
    """One ordered donor geometry shared by the mass and interface kernels."""
    import cupy as cp
    ns, nt = len(weights.source_area), len(weights.target_area)
    cache = getattr(weights, '_cuda_cumulative_geometry', None)
    device = cp.cuda.runtime.getDevice()
    if cache is None or cache['device'] != device:
        matrix = weights.overlap.tocsr()
        # COO traversal used by the host must be identical to CSR order.
        if not (np.array_equal(matrix.indices, weights.cols) and
                np.array_equal(np.repeat(np.arange(nt), np.diff(matrix.indptr)), weights.rows)):
            raise ValueError('CUDA cumulative remap requires ordered CSR donors')
        arrays = (matrix.indptr.astype(np.int64), weights.cols.astype(np.int64),
                  weights.areas, weights.sample_layer, weights.sample_height,
                  weights.source_zgrid, weights.source_centres)
        cache = dict(device=device, arrays=tuple(cp.asarray(np.ascontiguousarray(x)) for x in arrays))
        weights._cuda_cumulative_geometry = cache
    return cache


@independent_stream
def cumulative_integral(weights, means, slopes, cumulative, values, density, *, keep_device=False):
    """The target cumulative integral; ``keep_device``: return it on the card (device_layer_result)."""
    import cupy as cp
    # The class's source/target nominal level counts are equal in the moving
    # forecast. General remapping with unequal counts retains the CPU path.
    ns, nt = len(weights.source_area), len(weights.target_area)
    levels = len(weights.target_dzw)
    cache = _cumulative_geometry(weights)
    extend = density is not None and weights.below_ground is not None
    if extend:
        # The below-ground extension is prepared from the current atmosphere;
        # it can change between remaps even when geometry is reused.
        ratio = cp.asarray(values[0]/density[0])
        below = _below_ground_values(weights)['values']
    else:
        ratio = below = cp.empty(0, dtype=cp.float64)
    # Device profiles (device_layer_result) arrive on the card already; host profiles are uploaded.
    products = tuple(x if isinstance(x, cp.ndarray) else cp.asarray(np.ascontiguousarray(x))
                     for x in (means, slopes, cumulative))
    out = cp.empty((levels+1, nt), dtype=cp.float64)
    _kernel()(((out.size+127)//128,), (128,), (*cache['arrays'], *products, ratio, below, out,
        np.int32(ns), np.int32(nt), np.int32(levels), np.int64(len(weights.areas)),
        np.int32(weights.endpoint_alignment), np.int32(extend)))
    if keep_device:
        return out
    return cp.asnumpy(out)  # Required host result boundary; no extra synchronization.


@shared_geometry('_cuda_below_ground')
def _below_ground_values(weights):
    """One carried extension shared by all mass-product streams this move."""
    import cupy as cp
    device=cp.cuda.runtime.getDevice()
    cache=getattr(weights,'_cuda_below_ground',None)
    if (cache is None or cache['device']!=device or cache['source'] is not weights.below_ground):
        cache=dict(device=device,source=weights.below_ground,
                   values=cp.asarray(np.ascontiguousarray(weights.below_ground,dtype=np.float64)))
        weights._cuda_below_ground=cache
    return cache


# Fast numerics rewrites kernel options, so explicit rounded intrinsics keep the
# host's multiply, add and donor accumulation separate even with --fmad=true.
INTERFACE_OPTIONS = ('--fmad=false',)


@lru_cache(maxsize=1)
def _interface_kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    extern "C" __global__ void interface_integral(
        const long long *ptr, const long long *cols, const double *area,
        const unsigned char *layer, const double *sample, const double *source_z,
        const double *target_area, const double *values, double *out,
        const int ns, const int nt, const int levels, const long long nnz) {
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i >= (long long)(levels+1)*nt) return;
        const int l=i/nt, row=i%nt;
        double sum=0.;
        for(long long j=ptr[row];j<ptr[row+1];++j) {
            const long long c=cols[j], q=(long long)l*nnz+j;
            const long long low=(long long)layer[q]*ns+c, high=low+ns;
            const double dz=__dsub_rn(source_z[high],source_z[low]);
            const double raw=__ddiv_rn(__dsub_rn(sample[q],source_z[low]),dz);
            const double fraction=raw>0. ? (raw<1. ? raw : 1.) : 0.;
            const double delta=__dsub_rn(values[high],values[low]);
            const double carried=__dadd_rn(values[low],__dmul_rn(fraction,delta));
            sum=__dadd_rn(sum,__dmul_rn(area[j],carried));
        }
        out[i]=__ddiv_rn(sum,target_area[row]);
    }
    ''', 'interface_integral', options=INTERFACE_OPTIONS)


@independent_stream
def interface_integral(weights, values):
    """Physical-height interface interpolation in the host's CSR donor order.

    The CPU reference loops over every target level, builds donor temporaries,
    and bincounts them. One device thread now owns each target interface/cell;
    geometry stays resident with the mass remap. Only the result crosses back
    to the current host balance/physics boundary.
    """
    import cupy as cp
    geometry = _cumulative_geometry(weights)
    # The final cumulative-geometry array (source centres) is unused here.
    arrays = geometry['arrays'][:6]
    area = geometry.get('target_area')
    if area is None:
        area = geometry['target_area'] = cp.asarray(weights.target_area)
    source = cp.asarray(np.ascontiguousarray(values, dtype=np.float64))
    out = cp.empty(weights.target_zgrid.shape, dtype=cp.float64)
    ns, nt = len(weights.source_area), len(weights.target_area)
    levels = len(weights.target_dzw)
    _interface_kernel()(((out.size+127)//128,), (128,), (*arrays, area, source, out,
        np.int32(ns), np.int32(nt), np.int32(levels), np.int64(len(weights.areas))))
    return cp.asnumpy(out)


RATE_OPTIONS = ('--fmad=false',)


@lru_cache(maxsize=1)
def _layer_kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    extern "C" __global__ void layer_integral(
        const long long *ptr, const long long *cols, const double *area,
        const double *sample, const double *centres, const double *target_area,
        const double *values, double *out, const int ns, const int nt,
        const int source_levels, const int target_levels, const int middle,
        const int searches, const long long nnz) {
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= (long long)target_levels*middle*nt) return;
        const int row=i%nt, channel=(i/nt)%middle, level=i/((long long)nt*middle);
        double sum=0.;
        for(long long donor=ptr[row]; donor<ptr[row+1]; ++donor) {
            const long long cell=cols[donor];
            const double height=.5*(sample[(long long)level*nnz+donor]
                                 +sample[(long long)(level+1)*nnz+donor]);
            int lo=0, hi=source_levels-1;
            for(int search=0;search<searches;++search) {
                const int mid=(lo+hi+1)/2;
                if(centres[(long long)mid*ns+cell]<=height) lo=mid;
                else hi=mid-1;
            }
            const int last=source_levels>1 ? source_levels-2 : 0;
            const int k=lo<last ? lo : last;
            const int upper=k+1<source_levels ? k+1 : source_levels-1;
            const double lower=centres[(long long)k*ns+cell];
            const double span=centres[(long long)upper*ns+cell]-lower;
            double fraction=(height-lower)/(span>0. ? span : 1.);
            fraction=fraction<0. ? 0. : (fraction>1. ? 1. : fraction);
            const double a=values[((long long)k*middle+channel)*ns+cell];
            const double b=values[((long long)upper*middle+channel)*ns+cell];
            const double carried=a+fraction*(b-a);
            sum+=area[donor]*carried;
        }
        out[i]=sum/target_area[row];
    }
    ''', 'layer_integral', options=RATE_OPTIONS)


@lru_cache(maxsize=1)
def _rate_kernels():
    import cupy as cp
    source = r'''
    extern "C" __global__ void rate_geometry(
        const long long *cols, const double *sample, const double *centres,
        int *lower_index, double *fractions, const int ns, const int source_levels,
        const int levels, const int searches, const long long nnz) {
        const long long q=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(q>=(long long)levels*nnz) return;
        const long long cell=cols[q%nnz];
        const double height=.5*(sample[q]+sample[q+nnz]);
        int lo=0, hi=source_levels-1;
        for(int search=0;search<searches;++search) {
            const int mid=(lo+hi+1)/2;
            if(centres[(long long)mid*ns+cell]<=height) lo=mid;
            else hi=mid-1;
        }
        const int last=source_levels>1 ? source_levels-2 : 0;
        const int k=lo<last ? lo : last;
        const int upper=k+1<source_levels ? k+1 : source_levels-1;
        const double lower=centres[(long long)k*ns+cell];
        const double span=centres[(long long)upper*ns+cell]-lower;
        double fraction=(height-lower)/(span>0. ? span : 1.);
        fraction=fraction<0. ? 0. : (fraction>1. ? 1. : fraction);
        lower_index[q]=k;
        fractions[q]=fraction;
    }
    extern "C" __global__ void rate_apply(
        const long long *ptr, const long long *cols, const double *area,
        const int *lower_index, const double *fractions, const double *target_area,
        const double *values, double *out, const int ns, const int nt,
        const int source_levels, const int levels, const int middle, const long long nnz) {
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=(long long)levels*middle*nt) return;
        const int row=i%nt, channel=(i/nt)%middle, level=i/((long long)nt*middle);
        double sum=0.;
        for(long long donor=ptr[row];donor<ptr[row+1];++donor) {
            const long long cell=cols[donor], q=(long long)level*nnz+donor;
            const int k=lower_index[q], upper=k+1<source_levels ? k+1 : source_levels-1;
            const double fraction=fractions[q];
            const double a=values[((long long)k*middle+channel)*ns+cell];
            const double b=values[((long long)upper*middle+channel)*ns+cell];
            const double carried=a+fraction*(b-a);
            sum+=area[donor]*carried;
        }
        out[i]=sum/target_area[row];
    }
    '''
    return tuple(cp.RawKernel(source, name, options=RATE_OPTIONS)
                 for name in ('rate_geometry', 'rate_apply'))


@shared_geometry('_cuda_rate_geometry')
def _rate_geometry(weights):
    import cupy as cp
    ns, nt = len(weights.source_area), len(weights.target_area)
    device = cp.cuda.runtime.getDevice()
    cache = getattr(weights, '_cuda_rate_geometry', None)
    if cache is None or cache['device'] != device:
        base = _cumulative_geometry(weights)
        arrays = base['arrays']
        area = base.get('target_area')
        if area is None:
            area = base['target_area'] = cp.asarray(weights.target_area)
        cache = dict(device=device, arrays=(*arrays[:3], arrays[4], arrays[6], area))
        weights._cuda_rate_geometry = cache
    return cache


@shared_geometry('_cuda_rate_search')
def _rate_search(weights):
    """Height searches and fractions depend on geometry, never on a carried field."""
    import cupy as cp
    device = cp.cuda.runtime.getDevice()
    cache = getattr(weights, '_cuda_rate_search', None)
    if cache is None or cache['device'] != device:
        geometry = _rate_geometry(weights)['arrays']
        ns, source_levels = len(weights.source_area), len(weights.source_dzw)
        levels, nnz = len(weights.target_dzw), len(weights.areas)
        lower = cp.empty((levels, nnz), dtype=cp.int32)
        fraction = cp.empty((levels, nnz), dtype=cp.float64)
        if lower.size:
            _rate_kernels()[0](((lower.size+127)//128,), (128,),
                (geometry[1], geometry[3], geometry[4], lower, fraction,
                 np.int32(ns), np.int32(source_levels), np.int32(levels),
                 np.int32(source_levels.bit_length()), np.int64(nnz)))
        cache = dict(device=device, arrays=(lower, fraction))
        weights._cuda_rate_search = cache
    return cache


@independent_stream
def prepare_geometry(weights):
    """Upload static remap geometry beside the old segment, within its lease budget."""
    import cupy as cp
    limit = int(os.environ.get('WOOF_HEX_MOVING_PREPARE_BUDGET_MIB', '20000')) * 2**20
    reserve = int(os.environ.get('WOOF_HEX_MOVING_TRANSIENT_MIB', '4096')) * 2**20
    from .class_work_arena import _ARENAS
    live = cp.get_default_memory_pool().used_bytes() + sum(a.large.used_bytes() for a in _ARENAS)
    # Conservative upload plus rate-search storage; no dynamic state is staged.
    arrays = (weights.sample_layer, weights.sample_height, weights.source_zgrid,
              weights.source_centres, weights.areas, weights.cols, weights.rows)
    estimate = sum(a.nbytes for a in arrays) + len(weights.target_dzw)*len(weights.areas)*12 + (64 << 20)
    if live + estimate + reserve > limit:
        return dict(prepared=False, live_bytes=live, estimated_bytes=estimate)
    _cumulative_geometry(weights)
    _profile_geometry(weights)
    _rate_geometry(weights)
    if os.environ.get('WOOF_HEX_MOVING_RATE_GEOMETRY','1') != '0':
        _rate_search(weights)
    return dict(prepared=True, live_bytes=live, estimated_bytes=estimate)


@independent_stream
def layer_integral(weights, values):
    """Interpolate current rates in height, then sum donors in CPU order."""
    import cupy as cp
    source = cp.asarray(np.ascontiguousarray(values))
    out = layer_integral_device(weights, source)
    return cp.asnumpy(out).reshape((out.shape[0], *values.shape[1:-1], out.shape[-1]))


def layer_integral_device(weights, source):
    """Apply the same ordered rate kernel to an already resident FP64 batch."""
    import cupy as cp
    ns, nt = len(weights.source_area), len(weights.target_area)
    source_levels, target_levels = len(weights.source_dzw), len(weights.target_dzw)
    middle = int(np.prod(source.shape[1:-1], dtype=np.int64))
    cache = _rate_geometry(weights)
    out = cp.empty((target_levels, middle, nt), dtype=cp.float64)
    if os.environ.get('WOOF_HEX_MOVING_RATE_GEOMETRY','1') == '0':
        _layer_kernel()(((out.size+127)//128,), (128,), (*cache['arrays'], source, out,
            np.int32(ns), np.int32(nt), np.int32(source_levels), np.int32(target_levels),
            np.int32(middle), np.int32(source_levels.bit_length()), np.int64(len(weights.areas))))
    else:
        search = _rate_search(weights)
        arrays = (*cache['arrays'][:3], *search['arrays'], cache['arrays'][-1])
        _rate_kernels()[1](((out.size+127)//128,), (128,), (*arrays, source, out,
            np.int32(ns), np.int32(nt), np.int32(source_levels), np.int32(target_levels),
            np.int32(middle), np.int64(len(weights.areas))))
    return out


# ----------------------------------------------------------------------------------------------- device profiles
# THE COST THIS REMOVES (RTX 5090, 2026-10-29, the 90,320-cell storm-following region): every product of a
# move's atmosphere remap (19 per convection_permitting move) built its donor profiles on the host in column
# blocks, uploaded three level-by-cell FP64 arrays, downloaded the target cumulative integral, then rounded the
# positive columns and differenced them on the host again.  The profiles, the positive roundoff and the layer
# difference now stay on the card beside the integral: one upload of the product, one download of the result.
#
# WHAT IS EXACT ABOUT IT.  One thread owns one column and walks its levels in the host's order with the host's
# operations, compiled without FMA contraction: the mean ``(value*dzw)/dz``, the one-sided and centred slopes,
# the monotone limiter (NumPy's ``minimum``: ``a < b ? a : b``), the positive clip (NumPy's clip:
# ``min(max(x, lo), hi)`` with ``max = a > b ? a : b``), the sequential ``cumsum`` (its first word is the first
# product, never ``0 + p``), the roundoff's clip and running maximum (``maximum.accumulate``: ``prev > x ? prev :
# x``) and ``np.diff(...)/(dzw*area)``.  Signed zeros follow NumPy 2.5's rules, measured on an RTX 5090 host.  The global
# totals (einsum) and the domain rescale stay on the host, on the downloaded result.
# The products and sums use ``__dmul_rn``/``__dadd_rn``, which are never contracted: a fast-numerics run rewrites
# every unit's ``--fmad=false`` to ``--fmad=true`` (woof.hex.numerics, the engine's compile wrapper), and the
# cumulative sum ``run + value*dzw`` became one FMA there.  THE BREAKAGE THAT PREVENTS (RTX 5090, 2026-10-29): a
# resident run's moved region differed from the process-per-segment route in 292 M of 1.24 G words, while the
# same remap replayed outside the runtime (no rewrite) matched it word for word.
# ``WOOF_HEX_MOVING_DEVICE_PROFILES=0`` keeps the host profiles (a comparison knob).

PROFILES_ENV = "WOOF_HEX_MOVING_DEVICE_PROFILES"
#: The kernel's own options; a fast-numerics run rewrites --fmad=false to --fmad=true (the arithmetic above holds
#: either way: tests/test_moving_vertical_cuda.py compiles it both ways).
PROFILE_OPTIONS: tuple[str, ...] = ('--fmad=false',)


def device_profiles_enabled() -> bool:
    import os
    return os.environ.get(PROFILES_ENV, "1") != "0"


@lru_cache(maxsize=1)
def _profile_kernels():
    import cupy as cp
    source = r'''
    __device__ __forceinline__ double np_min(double a, double b) { return a < b ? a : b; }
    __device__ __forceinline__ double np_max(double a, double b) { return a > b ? a : b; }
    __device__ __forceinline__ double np_clip(double x, double lo, double hi) { return np_min(np_max(x, lo), hi); }

    extern "C" __global__ void source_profiles(
        const double *values, const double *dz, const double *centres, const double *dzw,
        const double *reach, const double *lowest,
        double *means, double *slopes, double *cumulative,
        const int levels, const long long ns, const int aligned, const int positive, const int has_lowest) {
        const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if(j >= ns) return;
        for(int k = 0; k < levels; ++k) {
            const long long i = (long long)k*ns + j;
            means[i] = __dmul_rn(values[i], dzw[k])/dz[i];
            slopes[i] = 0.;
        }
        if(levels > 1 && !aligned) {
            slopes[j] = (means[ns+j]-means[j])/(centres[ns+j]-centres[j]);
            const long long a = (long long)(levels-1)*ns + j, b = (long long)(levels-2)*ns + j;
            slopes[a] = (means[a]-means[b])/(centres[a]-centres[b]);
        }
        for(int k = 1; k + 1 < levels; ++k) {
            const long long lo = (long long)(k-1)*ns + j, mid = (long long)k*ns + j, hi = (long long)(k+1)*ns + j;
            const double left_lower = (means[mid]-means[lo])/(centres[mid]-centres[lo]);
            const double left_upper = (means[hi]-means[mid])/(centres[hi]-centres[mid]);
            const double central = (means[hi]-means[lo])/(centres[hi]-centres[lo]);
            const double a = 2*left_lower, b = central, c = 2*left_upper;
            const bool same = (signbit(a) == signbit(b)) && (signbit(b) == signbit(c));
            slopes[mid] = same ? copysign(np_min(np_min(fabs(a), fabs(b)), fabs(c)), b) : 0.;
        }
        if(positive) {
            for(int k = 0; k < levels; ++k) {
                const long long i = (long long)k*ns + j;
                const double bound = __dmul_rn(2., means[i])/dz[i];
                slopes[i] = np_clip(slopes[i], -bound, bound);
            }
            if(!aligned) slopes[j] = np_min(slopes[j], means[j]/reach[j]);
        }
        if(has_lowest) slopes[j] = lowest[j];
        cumulative[j] = 0.;
        double run = 0.;
        for(int k = 0; k < levels; ++k) {
            const long long i = (long long)k*ns + j;
            const double product = __dmul_rn(values[i], dzw[k]);
            run = k == 0 ? product : __dadd_rn(run, product);
            cumulative[(long long)(k+1)*ns + j] = run;
        }
    }

    extern "C" __global__ void finish_layers(
        double *cumulative, const double *denominator, double *result,
        int *flags, long long *changed, double *largest, double *relative,
        const int levels, const long long nt, const int positive, const int absolute_height) {
        const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if(j >= nt) return;
        if(positive) {
            const double bottom = cumulative[j], top = cumulative[(long long)levels*nt + j];
            int flag = 0;
            if(top < bottom || (!absolute_height && (top < 0 || bottom != 0))) flag |= 1;
            const double scale = np_max(fabs(bottom), fabs(top));
            const double allowance = __dmul_rn(128.0*2.220446049250313e-16, scale);
            long long count = 0;
            double big = 0., rel = 0.;
            double run = 0.;
            for(int k = 0; k <= levels; ++k) {
                const long long i = (long long)k*nt + j;
                const double x = cumulative[i];
                const double clipped = np_clip(x, bottom, top);
                run = k == 0 ? clipped : np_max(run, clipped);
                const double change = fabs(run - x);
                if(change > allowance) flag |= 2;
                if(change != 0) ++count;
                big = np_max(big, change);
                rel = np_max(rel, scale != 0 ? change/scale : 0.);
                cumulative[i] = run;
            }
            flags[j] = flag; changed[j] = count; largest[j] = big; relative[j] = rel;
        }
        for(int k = 0; k < levels; ++k) {
            const long long i = (long long)k*nt + j;
            result[i] = (cumulative[i + nt] - cumulative[i])/denominator[i];
        }
    }
    '''
    module = cp.RawModule(code=source, options=PROFILE_OPTIONS)
    return module.get_function('source_profiles'), module.get_function('finish_layers')


@shared_geometry('_cuda_profile_geometry')
def _profile_geometry(weights):
    import cupy as cp
    device = cp.cuda.runtime.getDevice()
    cache = getattr(weights, '_cuda_profile_geometry', None)
    if cache is None or cache['device'] != device:
        reach = getattr(weights, 'source_bottom_reach', None)
        arrays = dict(dz=weights.source_dz, centres=weights.source_centres, dzw=weights.source_dzw,
                      reach=np.zeros(len(weights.source_area)) if reach is None else reach,
                      # The host's denominator array, word for word (NumPy's outer product).
                      denominator=weights.target_dzw[:, None]*weights.target_area)
        cache = dict(device=device, **{k: cp.asarray(np.ascontiguousarray(v, dtype=np.float64))
                                        for k, v in arrays.items()})
        weights._cuda_profile_geometry = cache
    return cache


@independent_stream
def device_layer_result(weights, values, density, positive):
    """``(layer result on the host, positivity receipt)`` of one product, profiles and roundoff on the card.

    The words are the host path's (``_source_profiles`` -> ``cumulative_integral`` ->
    ``positive_cumulative_roundoff`` -> ``np.diff(...)/(dzw*area)``); see the block comment above.
    """
    import cupy as cp
    profiles, finish = _profile_kernels()
    geometry = _profile_geometry(weights)
    levels, ns = values.shape
    nt = len(weights.target_area)
    device_values = cp.asarray(np.ascontiguousarray(values, dtype=np.float64))
    has_lowest = density is not None
    lowest = (cp.asarray(np.ascontiguousarray(values[0]/density[0]*weights._lowest_density_slope(density)))
              if has_lowest else cp.empty(1, dtype=cp.float64))
    means = cp.empty((levels, ns), dtype=cp.float64)
    slopes = cp.empty((levels, ns), dtype=cp.float64)
    cumulative = cp.empty((levels + 1, ns), dtype=cp.float64)
    threads = 128
    profiles(((ns + threads - 1)//threads,), (threads,), (
        device_values, geometry['dz'], geometry['centres'], geometry['dzw'], geometry['reach'], lowest,
        means, slopes, cumulative, np.int32(levels), np.int64(ns), np.int32(weights.endpoint_alignment),
        np.int32(bool(positive)), np.int32(has_lowest)))
    target = cumulative_integral(weights, means, slopes, cumulative, values, density, keep_device=True)
    del means, slopes, cumulative, device_values, lowest
    result = cp.empty((levels, nt), dtype=cp.float64)
    flags = cp.zeros(nt, dtype=cp.int32)
    changed = cp.zeros(nt, dtype=cp.int64)
    largest = cp.zeros(nt, dtype=cp.float64)
    relative = cp.zeros(nt, dtype=cp.float64)
    finish(((nt + threads - 1)//threads,), (threads,), (
        target, geometry['denominator'], result, flags, changed, largest, relative,
        np.int32(levels), np.int64(nt), np.int32(bool(positive)), np.int32(not weights.endpoint_alignment)))
    positivity = None
    if positive:
        # Keep the original reductions, but cross the host boundary once per
        # product instead of synchronizing five times while other products run.
        checked = cp.asnumpy(cp.stack((((flags & 1) != 0).any(),
            ((flags & 2) != 0).any(), changed.sum(), largest.max(), relative.max())))
        if bool(checked[0]):
            raise ValueError('positive cumulative endpoints are invalid')
        if bool(checked[1]):
            raise ValueError('positive cumulative reversal exceeds FP64 arithmetic allowance')
        positivity = dict(changed_interfaces=int(checked[2]),
                          maximum_integrated_adjustment=float(checked[3]),
                          maximum_column_relative_adjustment=float(checked[4]),
                          column_endpoints_preserved=True)
    return cp.asnumpy(result), positivity
