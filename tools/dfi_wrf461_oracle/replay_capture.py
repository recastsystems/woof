#!/usr/bin/env python3
"""Replay observation-only captures from instrument_native.py on the GPU.

Inputs are captured native weather states. The graph loads that trajectory;
it is not a forecast graph. All inputs are uploaded before one Rust submission
per pass. There is no Python callback or readback between captured samples.
"""
from pathlib import Path
import argparse
import ctypes as C
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DYNAMICS = ['mu_2', 'u_2', 'v_2', 'w_2', 'ww', 't_2', 'phb', 'ph0',
            'php', 'p', 'ph_2', 'tke_2', 'al', 'alt', 'pb']
RADII = ['re_cloud', 're_ice', 're_snow']
GSFC = ['re_' + f + '_gsfc' for f in ('cloud', 'rain', 'ice', 'snow', 'graupel', 'hail')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_capture(path, np):
    data = path.read_bytes()
    endian = '<' if int.from_bytes(data[:4], 'little') == 1145456945 else '>'
    header = np.frombuffer(data, dtype=endian + 'i4', count=10).astype(np.int32)
    assert header[0] == 1145456945 and header[6] == 0, 'capture requires native savehydmeteors=0'
    names = DYNAMICS + ['moist', 'scalar'] + (GSFC if header[7] == 7 else RADII)
    coeff, total = np.frombuffer(data, dtype=endian + 'f4', count=2, offset=40).astype(np.float32)
    pos = 48
    inputs, accumulated = [], []
    for name in names:
        n = int(np.frombuffer(data, dtype=endian + 'i4', count=1, offset=pos)[0])
        assert 0 < n < 1000000, (path, name, n)
        pos += 4
        inputs.append(np.frombuffer(data, dtype=endian + 'f4', count=n, offset=pos).astype(np.float32))
        pos += 4*n
        accumulated.append(np.frombuffer(data, dtype=endian + 'f4', count=n, offset=pos).astype(np.float32))
        pos += 4*n
    assert pos == len(data), (path, pos, len(data))
    return dict(header=header, names=names, coefficient=coeff, total=total,
                inputs=inputs, accumulated=accumulated, path=path.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dt', type=float, required=True, help='absolute time_step in the captured native namelist')
    p.add_argument('--filter', type=int, required=True)
    p.add_argument('--library', type=Path,
                   help='whole-built DFI crate artifact; otherwise build the whole crate with Cargo')
    p.add_argument('--domain-dt', action='append', default=[], metavar='ID=SECONDS',
                   help='explicit timestep of a captured nested domain; root uses --dt')
    p.add_argument('--cutoff', type=float, required=True)
    args = p.parse_args()
    import numpy as np
    import cupy as cp
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    native = sorted(args.capture.glob('lane-dfi-*.bin'))
    assert native
    captures = [read_capture(x, np) for x in native]
    # Sequence-numbered captures preserve native repeated child endpoints.
    # Group by domain/option/stage, never by timestep alone: native nested
    # integrate() can accumulate the same index at two recursive boundaries.
    passes, pending, normalizations = {}, {}, {}
    for x in captures:
        h=x['header'];key=tuple(int(h[j]) for j in (1,2,3));mode=int(h[5])
        if mode==0:
            assert key not in pending, ('missing native post sample',key)
            pending[key]=x
        elif mode==1:
            pre=pending.pop(key)
            assert pre['names']==x['names']
            assert pre['header'][4]==h[4]
            for a,b in zip(pre['inputs'],x['inputs']):
                assert np.array_equal(a.view(np.uint32),b.view(np.uint32)), 'capture changed an input'
            passes.setdefault(key,[]).append((pre,x))
        elif mode in (2,3):
            assert mode not in normalizations.setdefault(key,{})
            normalizations[key][mode]=x
    assert not pending and passes, 'incomplete native sample pairs'
    domain_dt={1:args.dt}
    for spec in args.domain_dt:
        grid_id,value=spec.split('=',1)
        grid_id=int(grid_id);value=float(value)
        assert grid_id>0 and np.isfinite(value) and value>0
        domain_dt[grid_id]=value
    assert set(k[0] for k in passes)<=domain_dt.keys(), 'supply --domain-dt for every nest'
    lib_path = out / 'libarwen_dfi.so'
    if args.library is None:
        command = ['cargo', 'build', '--release', '--locked', '--offline', '--jobs', '4',
                   '--manifest-path', str(ROOT / 'tools/dfi-runtime/Cargo.toml')]
        subprocess.run(command, check=True)
        artifact = ROOT / 'tools/dfi-runtime/target/release/libarwen_dfi.so'
    else:
        artifact = args.library.resolve()
        command = ['use-prebuilt-whole-crate', str(artifact)]
    if artifact.resolve() != lib_path.resolve():
        shutil.copy2(artifact, lib_path)
    lib = C.CDLL(str(lib_path))
    weights = lib.aw_dfi_weights
    weights.argtypes = [C.c_int, C.c_float, C.c_float, C.c_int,
                        C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    weights.restype = C.c_int
    driver = C.CDLL('libcuda.so.1')
    submit = lib.aw_dfi_submit_pass
    submit.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                       C.c_void_p, C.POINTER(C.c_void_p), C.c_uint, C.c_uint,
                       C.c_uint, C.c_void_p, C.POINTER(C.c_void_p)]
    submit.restype = C.c_int
    source = (ROOT / 'woof/core/kernels/dfi.cu').read_text()
    source += r'''
// The replay input trajectory is fully resident, and is not model stepping.
extern "C" __global__ void captured_step(const unsigned long long* rows,
    const unsigned long long* trajectories, const unsigned long long* results,
    const unsigned int* cursor) {
 const unsigned long long* row=rows+4ull*blockIdx.y;
 unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 unsigned int sample=*cursor;
 if(i < row[2]) {
   float* state=reinterpret_cast<float*>(row[0]);
   const float* accum=reinterpret_cast<const float*>(row[1]);
   const float* input=reinterpret_cast<const float*>(trajectories[blockIdx.y]);
   float* result=reinterpret_cast<float*>(results[blockIdx.y]);
   if(sample) result[(sample-1)*row[2]+i]=accum[i];
   state[i]=input[sample*row[2]+i];
 }
}
extern "C" __global__ void captured_final(const unsigned long long* rows,
    const unsigned long long* results, unsigned int samples) {
 const unsigned long long* row=rows+4ull*blockIdx.y;
 unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i < row[2]) {
   float* result=reinterpret_cast<float*>(results[blockIdx.y]);
   const float* accum=reinterpret_cast<const float*>(row[1]);
   result[(samples-1)*row[2]+i]=accum[i];
 }
}
'''
    receipt = dict(scope='real native DFI tile accumulation and array normalization replay',
                   whole_forecast_qualified=False, captures=[dict(name=x.name, bytes=x.stat().st_size, sha256=sha(x)) for x in native],
                   rust_command=command, rust_library_sha256=sha(lib_path),
                   kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                   cupy_version=cp.__version__, nvrtc_version=list(cp.cuda.nvrtc.getVersion()),
                   cases=[], strict_words=0, strict_differing_words=0)
    receipt['coefficient_config'] = dict(dt=args.dt, filter=args.filter, cutoff=args.cutoff)
    for strict in (True, False):
        options = ['--std=c++11']
        if strict:
            options += ['-DGPUWM_WRF_EXACT=1', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true']
        image, _ = cp.cuda.compiler.compile_using_nvrtc(source, options=tuple(options), arch=cp.cuda.Device().compute_capability)
        binary = out / ('strict.cubin' if strict else 'fast.cubin')
        binary.write_bytes(image)
        module = cp.RawModule(path=str(binary))
        accum = module.get_function('dfi_accumulate_fields')
        advance = module.get_function('dfi_advance_coefficient')
        reset = module.get_function('dfi_reset_fields')
        step = module.get_function('captured_step')
        final = module.get_function('captured_final')
        for key, samples in passes.items():
            n = len(samples)
            names = samples[0][0]['names']
            counts = [len(x) for x in samples[0][0]['inputs']]
            native_weights = np.asarray([s[0]['coefficient'] for s in samples], dtype=np.float32)
            step_indices=np.asarray([int(x[0]['header'][4]) for x in samples],dtype=np.int64)
            assert step_indices[0]==0
            unique_count=int(step_indices[-1])+1
            rust_base=np.empty(unique_count,dtype=np.float32)
            written=C.c_size_t()
            dt=(-1 if key[2]==1 else 1)*domain_dt[key[0]]
            status=weights(round((unique_count-1)*abs(dt)),dt,args.cutoff,args.filter,
                rust_base.ctypes.data,unique_count,C.byref(written))
            assert status==0 and written.value==unique_count, 'native capture/namelist sample count disagreement'
            rust_weights=rust_base[step_indices]
            coeff_bad=int(np.count_nonzero(rust_weights.view(np.uint32)!=native_weights.view(np.uint32)))
            assert coeff_bad==0, ('real-window weights differ',key,coeff_bad)
            for i in range(1,n):
                assert step_indices[i]-step_indices[i-1] in (0,1)
                assert samples[i][0]['total'].view(np.uint32)==samples[i-1][1]['total'].view(np.uint32)
                for a,b in zip(samples[i][0]['accumulated'],samples[i-1][1]['accumulated']):
                    assert np.array_equal(a.view(np.uint32),b.view(np.uint32)), 'native pass is not continuous'
            states = [cp.asarray(a) for a in samples[0][0]['inputs']]
            sums = [cp.asarray(a) for a in samples[0][0]['accumulated']]
            inputs = [cp.asarray(np.stack([s[0]['inputs'][i] for s in samples])) for i in range(len(names))]
            outputs = [cp.empty((n, count), dtype=cp.float32) for count in counts]
            rows = cp.asarray(np.asarray([[a.data.ptr, b.data.ptr, a.size, int(name not in DYNAMICS)]
                                         for name, a, b in zip(names, states, sums)], dtype=np.uint64))
            input_ptrs = cp.asarray(np.asarray([x.data.ptr for x in inputs], dtype=np.uint64))
            output_ptrs = cp.asarray(np.asarray([x.data.ptr for x in outputs], dtype=np.uint64))
            h = cp.asarray(rust_weights)
            cursor = cp.zeros(1, dtype=cp.uint32)
            total = cp.asarray(np.asarray([samples[0][0]['total']], dtype=np.float32))
            error = cp.zeros(1, dtype=cp.uint32)
            blocks = (max(counts)+255)//256
            grid = (blocks, len(names))
            step.compile()
            cp.cuda.get_current_stream().synchronize()
            stream = cp.cuda.Stream(non_blocking=True)
            acc_args = [C.c_void_p(x.data.ptr) for x in (rows, h, cursor)] + [C.c_uint(n), C.c_void_p(error.data.ptr)]
            cur_args = [C.c_void_p(h.data.ptr), C.c_void_p(cursor.data.ptr), C.c_uint(n), C.c_void_p(total.data.ptr), C.c_void_p(error.data.ptr)]
            acc_params = (C.c_void_p*len(acc_args))(*[C.addressof(x) for x in acc_args])
            cur_params = (C.c_void_p*len(cur_args))(*[C.addressof(x) for x in cur_args])
            with stream:
                stream.begin_capture()
                step(grid, (256,), (rows, input_ptrs, output_ptrs, cursor))
                graph = stream.end_capture()
                accum(grid, (256,), (rows, h, cursor, np.uint32(n), error))
                advance((1,), (1,), (h, cursor, np.uint32(n), total, error))
                start, stop = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                status = submit(C.cast(driver.cuGraphLaunch, C.c_void_p), C.cast(driver.cuLaunchKernel, C.c_void_p),
                                graph.graphExec, stream.ptr, accum.kernel.ptr, acc_params, blocks, len(names), n-1,
                                advance.kernel.ptr, cur_params)
                assert status == 0, status
                stop.record()
                final(grid, (256,), (rows, output_ptrs, np.uint32(n)))
                if key in normalizations:
                    reset(grid, (256,), (rows, total, error))
            stream.synchronize()
            assert int(error.get()[0]) == 0 and int(cursor.get()[0]) == n
            words, differing, max_abs = 1, 0, 0.0
            field_receipts = []
            for i, name in enumerate(names):
                expected = np.stack([s[1]['accumulated'][i] for s in samples])
                actual = outputs[i].get()
                bad = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
                assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
                max_abs = max(max_abs, float(np.max(np.abs(actual-expected))))
                words += actual.size
                differing += bad
                field_receipts.append(dict(field=name, words=int(actual.size), differing_words=bad))
            differing += int(total.get().view(np.uint32)[0] != samples[-1][1]['total'].view(np.uint32))
            reset_words, reset_bad = 0, 0
            reset_modes = normalizations.get(key)
            if reset_modes is None:
                assert key[1:]==(2,1), 'only native DDFI backward accumulation skips reset'
            else:
                assert 2 in reset_modes and 3 in reset_modes, 'normalization capture missing'
                assert reset_modes[2]['total'].view(np.uint32) == samples[-1][1]['total'].view(np.uint32)
                for before, after in zip(reset_modes[2]['accumulated'], samples[-1][1]['accumulated']):
                    assert np.array_equal(before.view(np.uint32), after.view(np.uint32)), 'normalization uses a different accumulation'
                for i, name in enumerate(names):
                    expected = reset_modes[3]['inputs'][i]
                    actual = states[i].get()
                    reset_words += actual.size
                    reset_bad += int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
            if strict:
                receipt['strict_words'] += n + words + reset_words
                receipt['strict_differing_words'] += differing + reset_bad
            receipt['cases'].append(dict(domain=key[0],dfi_opt=key[1],stage=key[2],strict=strict,
                unique_coefficient_indices=unique_count,repeated_endpoint_samples=n-unique_count,
                sample_indices=step_indices.tolist(),
                nvrtc_options=options, samples=n, active_fields=names,
                coefficient_words=n, coefficient_differing_words=coeff_bad,
                words=words, differing_words=differing, fields=field_receipts,
                normalization_words=reset_words, normalization_differing_words=reset_bad,
                max_accumulation_abs_difference=max_abs,
                replay_ms_per_sample=cp.cuda.get_elapsed_time(start, stop)/max(n-1, 1)))
            # Save the receipt even on a strict failure, then stop with evidence.
            (out/'real-capture-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
            if strict:
                assert differing + reset_bad == 0, (key, differing, reset_bad)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('captures', 'cases')}, indent=2))


if __name__ == '__main__':
    main()
