"""Project carried native words onto LTS classes with integer GPU gathers."""
import numpy as np
from functools import lru_cache
from .moving_cuda_work import independent_stream


@independent_stream
def warm():
    """Resolve the dimension-independent uint32 gather code before a move."""
    _kernel().compile()


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    extern "C" __global__ void project_native_words(
        const unsigned int *source,unsigned int *out,const long long *metadata,
        const int ndim,const long long count){
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=count)return;
        long long rest=i,global=0;
        for(int axis=0;axis<ndim;++axis){
            const long long extent=metadata[3*axis],stride=metadata[3*axis+1];
            const long long pointer=metadata[3*axis+2];
            long long coordinate=rest%extent;rest/=extent;
            if(pointer)coordinate=((const long long *)pointer)[coordinate];
            global+=coordinate*stride;
        }
        out[i]=source[global];
    }
    ''', 'project_native_words')


@independent_stream
def prepare(maps, layout):
    import cupy as cp
    device = []
    for mapping in maps:
        local = {}
        for entity, (ids, owned, _) in mapping.items():
            ids = np.asarray(ids, dtype=np.int64)
            local[entity] = {
                entity: cp.asarray(np.concatenate((ids, [layout.dimensions[entity]]))),
                entity+'All': cp.asarray(ids), entity+'Solve': cp.asarray(ids[:owned])}
        limits={}
        for entity,(ids,owned,_) in mapping.items():
            limits[entity]=layout.dimensions[entity]
            limits[entity+'All']=int(np.max(ids)) if len(ids) else -1
            limits[entity+'Solve']=int(np.max(ids[:owned])) if owned else -1
        device.append((local,limits))
    return dict(device_id=cp.cuda.runtime.getDevice(), maps=device)


def project(buffer, row, layout, maps, prepared):
    import cupy as cp
    # Mesh connectivity keeps the CPU's one-based reference transformation.
    # A moving physics carrier normally contains no mesh field at all.
    if row['pool'] == 'mesh' or np.asarray(buffer).size == 0:
        from .mpasphys.partition_v842 import _project_buffer
        return [_project_buffer(buffer, row, layout, mapping) for mapping in maps]
    with cp.cuda.Device(prepared['device_id']):
        return _project(buffer, row, layout, prepared)


@independent_stream
def _project(buffer, row, layout, prepared):
    import cupy as cp
    value = np.asarray(buffer)
    expected = tuple(row['shape'])
    dtype = np.dtype('<f4' if row['kind'] == 'real' else '<i4')
    if value.shape != expected:
        raise ValueError(f"{row['pool']}.{row['name']} shape {value.shape} != {expected}")
    if row['kind'] not in ('real','integer') or value.dtype != dtype:
        raise ValueError(f"{row['pool']}.{row['name']} is not a native word buffer")
    # No floating operation is permitted: uint32 transports signaling NaNs,
    # signed zero and integer padding without FTZ or NaN canonicalization.
    source = cp.asarray(np.asfortranarray(value).view(np.uint32))
    result = []
    for mapping,limits in prepared['maps']:
        shape=list(value.shape)
        metadata=[]
        stride=1
        for axis, dimension in enumerate(row['dimensions']):
            indices=None
            for entity, indices in mapping.items():
                if dimension in indices:
                    indices=indices[dimension]
                    if limits[dimension]>=value.shape[axis]:
                        raise IndexError('native class projection index exceeds source extent')
                    break
            else:
                indices=None
            if indices is not None:
                shape[axis]=indices.size
            metadata.extend((shape[axis],stride,0 if indices is None else indices.data.ptr))
            stride*=value.shape[axis]
        words=cp.empty(shape,dtype=cp.uint32,order='F')
        if words.size:
            description=cp.asarray(np.array(metadata,dtype=np.int64))
            _kernel()(((words.size+127)//128,), (128,),
                      (source,words,description,np.int32(value.ndim),np.int64(words.size)))
        host = cp.asnumpy(words,order='F').view(dtype)
        result.append(host)
    return result
