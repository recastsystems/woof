#!/usr/bin/env python3
"""Replay observation-only native MP_NTU captures on the GPU.

The generated block must be made from the pinned, unmodified source using
f2cu.py. This tool does not apply the recovered prototype's source changes.
It compares every output word, including signed zero and nonfinite words.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from capture import FIELDS, OUTPUT, SURFACE
from generate import EXPECTED

HERE = Path(__file__).resolve().parent
ENGINE_NAMES = {"TH": "theta", "PII": "pii", "AS": "as_", "GRAPNC": "graupelnc", "GRAPNCV": "graupelncv"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_capture(path: Path, *, output=False):
    with path.open('rb') as f:
        header = np.fromfile(f, dtype='<i4', count=8)
        if header.size != 8 or header[0] != 0x4E545532:
            raise ValueError(f'{path}: invalid native capture header')
        _, nx, ny, nz, step, its, jts, ccnty = (int(v) for v in header)
        dt = np.fromfile(f, dtype='<f4', count=1)[0]
        names = OUTPUT if output else FIELDS
        fields = {name: np.fromfile(f, dtype='<f4', count=nz*ny*nx).reshape(nz,ny,nx) for name in names}
        if not output:
            fields['W'] = np.fromfile(f, dtype='<f4', count=(nz+1)*ny*nx).reshape(nz+1,ny,nx)
        surface = {name: np.fromfile(f, dtype='<f4', count=ny*nx).reshape(ny,nx) for name in SURFACE}
        if f.read(1):
            raise ValueError(f'{path}: trailing native capture bytes')
    return dict(nx=nx,ny=ny,nz=nz,step=step,its=its,jts=jts,ccnty=ccnty,dt=float(dt)), fields, surface


def replay(pin: Path, expected: Path, generated: Path, receipt: Path, *, debug=False, fast_compile=None):
    import cupy as cp
    from woof.core import ntu
    from woof.nvrtc_ptx_cache import compile_using_nvrtc
    from woof import numerics

    if numerics.mode() != 'strict':
        raise ValueError('Native word replay requires GPUWM_NUMERICS=strict')
    meta, fields, surface = read_capture(pin)
    outmeta, reference, reference_surface = read_capture(expected, output=True)
    if meta != outmeta:
        raise ValueError('native input/output capture metadata differ')
    kmax = ntu.kmax_for(meta['nz'])
    source = ntu.module_source(kmax)
    before, rest = source.split('// ---- BEGIN transcription (f2cu.py) ----\n', 1)
    shipping, after = rest.split('// ---- END transcription ----\n', 1)
    if digest(generated) != EXPECTED or shipping.strip() != generated.read_text().strip():
        raise ValueError('replay must use the pinned transcription shipped in ntu.cu')
    source = before + '// ---- BEGIN transcription (f2cu.py) ----\n' + generated.read_text() + '\n// ---- END transcription ----\n' + after
    options = ('-std=c++17','--ftz=false','--fmad=false','--prec-div=true','--prec-sqrt=true')
    if debug:
        options += ('--device-debug',)
    if fast_compile:
        options += (f'--Ofast-compile={fast_compile}',)
    started = time.monotonic()
    image, _ = compile_using_nvrtc(source, options, filename='ntu-native471.cu')
    module = cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image,str) else image)
    compile_seconds = time.monotonic()-started
    device_fields = {ENGINE_NAMES.get(n,n.lower()):cp.asarray(a) for n,a in fields.items()}
    device_surface = {ENGINE_NAMES.get(n,n.lower()):cp.asarray(a) for n,a in surface.items()}
    f3 = cp.asarray(np.array([device_fields[n].data.ptr for n in ntu._FIELD3],dtype=np.uint64))
    f2 = cp.asarray(np.array([device_surface[n].data.ptr for n in ntu._FIELD2],dtype=np.uint64))
    constants = ntu.aero_constants(cp.cuda.Device().id,meta['ccnty'])
    status = cp.zeros(1,dtype=cp.uint32)
    if os.environ.get('NTU_NATIVE_CONSTANTS'):
        raw=Path(os.environ['NTU_NATIVE_CONSTANTS']).read_bytes()
        if raw != cp.asnumpy(constants).tobytes():
            raise ValueError('diagnostic native table differs from the shipping table')
    start, end = cp.cuda.Event(), cp.cuda.Event()
    ncol = meta['nx']*meta['ny']
    kernel = module.get_function('ntu_level_pre_kernel')
    workspace=cp.empty((ntu.WORK_VOLUMES,*device_fields['theta'].shape),dtype=cp.float32)
    def call():
        ntu.launch_stages(module,constants,f3,f2,workspace,ncol,meta['nz'],meta['dt'])
    attrs={name:int(cp.cuda.driver.funcGetAttribute(getattr(cp.cuda.driver,name),kernel.ptr)) for name in ('CU_FUNC_ATTRIBUTE_NUM_REGS','CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES','CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES')}
    receipt.with_suffix('.compile.json').write_text(json.dumps(dict(compile_seconds=compile_seconds,attributes=attrs,device_memory=cp.cuda.runtime.memGetInfo(),options=options),indent=2)+'\n')
    start.record()
    call()
    end.record()
    end.synchronize()
    rows={}
    for name, truth in {**reference,**reference_surface}.items():
        key = ENGINE_NAMES.get(name,name.lower())
        actual = cp.asnumpy((device_fields if name in reference else device_surface)[key])
        different = actual.view(np.uint32) != truth.view(np.uint32)
        finite = np.isfinite(actual)&np.isfinite(truth)
        rows[name] = dict(words=int(actual.size),different_words=int(different.sum()),
                          native_nonfinite=int((~np.isfinite(truth)).sum()),gpu_nonfinite=int((~np.isfinite(actual)).sum()),
                          max_absolute_error=float(np.max(np.abs(actual[finite].astype('float64')-truth[finite]))) if finite.any() else None,
                          native_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),gpu_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
        if different.any():
            ids=np.flatnonzero(different)[:12]
            rows[name]['first_differences']=[dict(index=int(i),native_word=int(truth.view(np.uint32).ravel()[i]),gpu_word=int(actual.view(np.uint32).ravel()[i])) for i in ids]
    result = dict(input=str(pin),native_output=str(expected),input_sha256=digest(pin),native_output_sha256=digest(expected),
                  metadata=meta,generated_sha256=digest(generated),assembled_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  compile_options=options,compile_seconds=compile_seconds,device_name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  gpu_stage_ms=float(cp.cuda.get_elapsed_time(start,end)),
                  kernel_attributes={name:int(cp.cuda.driver.funcGetAttribute(getattr(cp.cuda.driver,name),kernel.ptr)) for name in ('CU_FUNC_ATTRIBUTE_NUM_REGS','CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES','CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES')},
                  status_word=int(cp.asnumpy(status)[0]),fields=rows,words=sum(r['words'] for r in rows.values()),
                  different_words=sum(r['different_words'] for r in rows.values()))
    result['passed'] = result['different_words']==0 and result['status_word']==0
    # Restore exactly the captured operands and time a resident launch, after
    # first-use driver/JIT/stack setup. No physics input is manufactured here.
    for name,array in fields.items():
        device_fields[ENGINE_NAMES.get(name,name.lower())].set(array)
    for name,array in surface.items():
        device_surface[ENGINE_NAMES.get(name,name.lower())].set(array)
    status.fill(0)
    cp.cuda.get_current_stream().synchronize()
    start.record()
    call()
    end.record()
    end.synchronize()
    result['resident_gpu_stage_ms']=float(cp.cuda.get_elapsed_time(start,end))
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('words','different_words','gpu_stage_ms','status_word','passed')},indent=2))
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('input','native_output','generated','receipt'):
        ap.add_argument(name,type=Path)
    ap.add_argument('--debug',action='store_true')
    ap.add_argument('--fast-compile',choices=('min','mid','max'))
    args=ap.parse_args()
    result=replay(args.input,args.native_output,args.generated,args.receipt,debug=args.debug,fast_compile=args.fast_compile)
    raise SystemExit(0 if result['passed'] else 1)
