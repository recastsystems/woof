"""Bounded batches of independent carried physics rates on one height map.

The channel axis is not reduced: each field still visits the same donors in
the same order. Source/target vector rotations keep their NumPy expressions.
Only the shared height interpolation and transfer are batched.
"""
import os
import time
from functools import lru_cache
from functools import wraps
import threading
import numpy as np

_PINNED_LOCK=threading.RLock()
_PINNED_MEMORY=None
_PINNED_CAPACITY=0


def _pinned_memory(cp,size):
    global _PINNED_MEMORY,_PINNED_CAPACITY
    if size>_PINNED_CAPACITY:
        _PINNED_MEMORY=cp.cuda.alloc_pinned_memory(size)
        _PINNED_CAPACITY=size
    return _PINNED_MEMORY


def prepare_uploads(weights):
    """Pin one bounded source slab beside the old segment, never per move."""
    import cupy as cp
    size=len(weights.source_area)*len(weights.source_dzw)*8
    limit=max(size,int(os.environ.get('WOOF_HEX_MOVING_RATE_BATCH_MIB','256'))*2**20)
    with _PINNED_LOCK:
        _pinned_memory(cp,max(limit,2*size))
        return _PINNED_CAPACITY


def _pinned_sources(function):
    @wraps(function)
    def call(*args,**kwargs):
        with _PINNED_LOCK:
            try:
                return function(*args,**kwargs)
            except BaseException:
                import cupy as cp
                cp.cuda.get_current_stream().synchronize()
                raise
    return call


@lru_cache(maxsize=2)
def _pack_kernel(itemsize):
    import cupy as cp
    scalar = 'float' if itemsize == 4 else 'double'
    source = r'''
    extern "C" __global__ void pack_rate(
        const char *u,const char *v,const double *east,const double *north,
        double *out,const long long sk,const long long sc,
        const long long vk,const long long vc,const long long ns,
        const long long n,const int width,const int channel,const int axis,const int vector) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
        long long k=i/ns,c=i%ns;
        double x=(double)(*(const TYPE*)(u+k*sk+c*sc));
        if(vector){
            double y=(double)(*(const TYPE*)(v+k*vk+c*vc));
            x=__dadd_rn(__dmul_rn(x,east[3*c+axis]),__dmul_rn(y,north[3*c+axis]));
        }
        out[(k*width+channel)*ns+c]=x;
    }
    '''.replace('TYPE',scalar)
    return cp.RawKernel(source,'pack_rate',options=('--fmad=false',))


def _pack_rate(cp, target, channel, u, v=None, east=None, north=None, axis=0):
    vector = v is not None
    if u.dtype.itemsize not in (4,8):
        u = u.astype(cp.float64)
        if vector:
            v = v.astype(cp.float64)
    if not vector:
        v, east, north = u, u, u  # Unread pointers on the scalar branch.
    _pack_kernel(u.dtype.itemsize)(((u.size+255)//256,), (256,), (
        u,v,east,north,target,np.int64(u.strides[0]),np.int64(u.strides[1]),
        np.int64(v.strides[0]),np.int64(v.strides[1]),np.int64(u.shape[1]),
        np.int64(u.size),np.int32(target.shape[1]),np.int32(channel),np.int32(axis),np.int32(vector)))


def _upload_inputs(cp,pools,keys,ns):
    """One pinned integer-word upload per dtype in this bounded rate batch."""
    from concurrent.futures import ThreadPoolExecutor
    from ..host_parallel import contiguous_copy
    groups={}
    for key in dict.fromkeys(keys):
        source=np.asarray(pools[key][...,:ns])
        if not (source.flags.c_contiguous or source.flags.f_contiguous):
            source=contiguous_copy(source,dtype=source.dtype)
        groups.setdefault(source.dtype.str,[]).append((key,source))
    uploaded,staging={},[]
    needed=0
    for fields in groups.values():
        needed=(needed+7)//8*8+sum(source.nbytes for _,source in fields)
    memory=_pinned_memory(cp,needed)
    offset=0
    for dtype,fields in groups.items():
        size=fields[0][1].size
        unsigned=np.dtype('u'+str(np.dtype(dtype).itemsize))
        offset=(offset+7)//8*8
        host=np.ndarray((len(fields),size),dtype=unsigned,buffer=memory,offset=offset)
        offset+=host.nbytes
        def copy(index):
            np.copyto(host[index],fields[index][1].view(unsigned).ravel(order='K'))
        with ThreadPoolExecutor(max_workers=min(4,len(fields)),thread_name_prefix='hex-rate-upload') as pool:
            list(pool.map(copy,range(len(fields))))
        device=cp.asarray(host)
        for index,(key,source) in enumerate(fields):
            order='F' if source.flags.f_contiguous and not source.flags.c_contiguous else 'C'
            uploaded[key]=device[index].view(source.dtype).reshape(source.shape,order=order)
        # The asynchronous transfer owns these pinned words until this batch's
        # result download finishes. Never recycle the buffer at upload return.
        staging.append(host)
    return uploaded,staging


def remap(weights, pools, scalar_keys, vector_pairs, source_mesh, target_mesh):
    if (getattr(weights, 'backend', 'cpu') == 'cuda'
            and os.environ.get('WOOF_HEX_MOVING_RATE_DEVICE_BATCH', '1') != '0'):
        return _device_remap(weights, pools, scalar_keys, vector_pairs, source_mesh, target_mesh)
    from ..moving_physics import local_vector_bases
    ns, levels = len(weights.source_area), len(weights.source_dzw)
    jobs, vectors = [], []
    if vector_pairs:
        east, north = local_vector_bases(source_mesh)
        target_east, target_north = local_vector_bases(target_mesh)
    for u_key, v_key in vector_pairs:
        u, v = (pools[key][..., :ns] for key in (u_key, v_key))
        if u.shape != (levels, ns) or v.shape != u.shape:
            continue  # Other staggers retain carry_horizontal_vector's path.
        if u.dtype != v.dtype:
            raise ValueError('physics vector components have incompatible dtypes')
        vectors.append((u_key, v_key, u.dtype))
        for axis in range(3):
            key = (u_key, v_key, axis)
            jobs.append((key, lambda u=u, v=v, axis=axis: u*east[:, axis]+v*north[:, axis]))
    vector_keys = {key for u, v, _ in vectors for key in (u, v)}
    scalar = []
    for key in scalar_keys:
        if key in vector_keys:
            continue
        cells = pools[key][..., :ns]
        if (cells.shape == (levels, ns) and cells.dtype.kind == 'f'
                and np.isfinite(cells).all()):
            scalar.append(key)
            jobs.append((key, lambda cells=cells: cells))
    size = levels*ns*8
    limit = max(size, int(os.environ.get('WOOF_HEX_MOVING_RATE_BATCH_MIB','256')) * 2**20)
    width = max(1, limit//max(size, 1))
    mapped, batches = {}, 0
    for first in range(0, len(jobs), width):
        group = jobs[first:first+width]
        packed = np.empty((levels, len(group), ns), dtype=np.float64)
        for channel, (_, make) in enumerate(group):
            packed[:, channel, :] = make()
        result = weights.layer_intensive(packed)
        for channel, (key, _) in enumerate(group):
            mapped[key] = result[:, channel, :]
        batches += 1
    completed = {key:(True, mapped.pop(key), 'physical-height layers') for key in scalar}
    for u_key, v_key, dtype in vectors:
        components = [mapped.pop((u_key, v_key, axis)) for axis in range(3)]
        u = sum(value*target_east[:, axis] for axis, value in enumerate(components))
        v = sum(value*target_north[:, axis] for axis, value in enumerate(components))
        for key, value in ((u_key, u), (v_key, v)):
            completed[key] = (True, value.astype(dtype), 'earth-Cartesian vector, physical height')
    return completed, dict(fields=len(scalar)+2*len(vectors), channels=len(jobs),
                           batches=batches, maximum_input_bytes=min(width, len(jobs))*size)


def _device_remap(weights, pools, scalar_keys, vector_pairs, source_mesh, target_mesh):
    """Pack, rotate and project independent rates on the card, before one download.

    B3V's physics restore costs 5.5-7.3 s beside the atmosphere remap on an
    RTX 5090 host. The host batch builds and rotates several GB of FP64 temporaries.
    Keep those words on the card; each ufunc below remains a separate rounded
    operation, including Python sum's initial addition of positive zero.
    """
    from ..moving_cuda_work import independent_stream

    @independent_stream
    @_pinned_sources
    def run():
        import cupy as cp
        from ..moving_physics import local_vector_bases
        from ..moving_vertical_cuda import layer_integral_device
        ns, levels = len(weights.source_area), len(weights.source_dzw)
        vectors, jobs, scalar = [], [], []
        for u_key, v_key in vector_pairs:
            u, v = (pools[key][..., :ns] for key in (u_key, v_key))
            if u.shape != (levels, ns) or v.shape != u.shape:
                continue
            if u.dtype != v.dtype:
                raise ValueError('physics vector components have incompatible dtypes')
            if not (np.isfinite(u).all() and np.isfinite(v).all()):
                raise ValueError('height map layer field is not finite')
            vectors.append((u_key, v_key, u.dtype))
            jobs.extend(((u_key, v_key, axis) for axis in range(3)))
        vector_keys = {key for u, v, _ in vectors for key in (u, v)}
        for key in scalar_keys:
            if key in vector_keys:
                continue
            cells = pools[key][..., :ns]
            if (cells.shape == (levels, ns) and cells.dtype.kind == 'f'
                    and np.isfinite(cells).all()):
                scalar.append(key)
                jobs.append(key)
        if vectors:
            east, north = (cp.asarray(x) for x in local_vector_bases(source_mesh))
            target_east, target_north = (cp.asarray(x) for x in local_vector_bases(target_mesh))
        size = levels*ns*8
        limit = max(size, int(os.environ.get('WOOF_HEX_MOVING_RATE_BATCH_MIB', '256'))*2**20)
        width = max(1, limit//max(size, 1))
        completed, mapped, batches, downloads, uploads = {}, {}, 0, 0, 0
        seconds = dict(pack=0., interpolate=0., project=0.)
        for first in range(0, len(jobs), width):
            began = time.perf_counter()
            group = jobs[first:first+width]
            packed = cp.empty((levels, len(group), ns), dtype=cp.float64)
            source_keys=[k for key in group for k in (key[:2] if isinstance(key,tuple) else (key,))]
            device_sources,staging = _upload_inputs(cp,pools,source_keys,ns)
            uploads += len(staging)
            for channel, key in enumerate(group):
                if isinstance(key, tuple):
                    u_key, v_key, axis = key
                    u, v = device_sources[u_key],device_sources[v_key]
                    _pack_rate(cp,packed,channel,u,v,east,north,axis)
                else:
                    source = device_sources[key]
                    _pack_rate(cp,packed,channel,source)
            seconds['pack'] += time.perf_counter()-began
            began = time.perf_counter()
            result = layer_integral_device(weights, packed)
            seconds['interpolate'] += time.perf_counter()-began
            began = time.perf_counter()
            scalar_channels = []
            for channel, key in enumerate(group):
                if isinstance(key, tuple):
                    # A view would retain every unrelated channel in this batch.
                    mapped[key] = result[:, channel, :].copy()
                else:
                    scalar_channels.append((channel,key))
            if scalar_channels:
                # Pack integer words once and cross the host boundary once for
                # this bounded batch, without serial per-field synchronizes.
                selected = cp.asarray([channel for channel,_ in scalar_channels])
                host = cp.asnumpy(cp.take(result.view(cp.uint64),selected,axis=1)).view(np.float64)
                downloads += 1
                for channel, (_,key) in enumerate(scalar_channels):
                    completed[key] = (True,host[:,channel,:],'physical-height layers')
            seconds['project'] += time.perf_counter()-began
            batches += 1
            if not scalar_channels:
                # A vector-only batch has not downloaded a result yet. Its
                # pinned source must survive the queued H2D transfer before
                # the pool can hand those pages to the next host packer.
                cp.cuda.get_current_stream().synchronize()
            del device_sources,staging,packed,result
        began = time.perf_counter()
        for u_key, v_key, dtype in vectors:
            components = [mapped.pop((u_key, v_key, axis)) for axis in range(3)]
            u = sum(value*target_east[:, axis] for axis, value in enumerate(components))
            v = sum(value*target_north[:, axis] for axis, value in enumerate(components))
            wordtype = cp.uint32 if dtype.itemsize == 4 else cp.uint64
            host = cp.asnumpy(cp.stack([value.astype(dtype).view(wordtype) for value in (u,v)])).view(dtype)
            downloads += 1
            for channel, key in enumerate((u_key,v_key)):
                completed[key] = (True, host[channel],
                                  'earth-Cartesian vector, physical height')
        seconds['project'] += time.perf_counter()-began
        return completed, dict(fields=len(scalar)+2*len(vectors), channels=len(jobs), batches=batches,
                               maximum_input_bytes=min(width, len(jobs))*size,
                               device_pack=True,host_uploads=uploads,host_downloads=downloads,host_wall_seconds=seconds)
    return run()
