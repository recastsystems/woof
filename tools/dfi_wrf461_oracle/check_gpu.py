"""Bounded synthetic array check using original WRF accumulation statements.

This does not capture or qualify a real forecast, nests, surface restoration,
savehydmeteors=1, or a production model graph. The graph advances a synthetic
fixture, not weather. Run check.py first to build the Rust library and weights.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DYNAMICS = ['mu_2', 'u_2', 'v_2', 'w_2', 'ww', 't_2', 'phb', 'ph0',
            'php', 'p', 'ph_2', 'tke_2', 'al', 'alt', 'pb']
ORDINARY_RADII = ['re_cloud', 're_ice', 're_snow']
GSFC_RADII = ['re_cloud_gsfc', 're_rain_gsfc', 're_ice_gsfc',
              're_snow_gsfc', 're_graupel_gsfc', 're_hail_gsfc']
FIELDS = DYNAMICS + ['moist', 'scalar'] + ORDINARY_RADII + GSFC_RADII


def accumulation_field(field):
    return 'dfi_' + (field[:-2] if field.endswith('_2') else field)


def original_driver(source, out):
    start = source.index('   SUBROUTINE dfi_accumulate(')
    end = source.index('   END SUBROUTINE dfi_accumulate', start)
    accumulate = source[start:end + len('   END SUBROUTINE dfi_accumulate')]
    # Only the unmodified dynamics/moisture normalization statements.
    # Surface and hydrometeor restoration are deliberately not claimed.
    start = source.index('      grid%MU_2(:,:)    = grid%dfi_mu')
    end = source.index('     if ( grid%sf_surface_physics', start)
    reset = source[start:end]
    ranks = {'mu_2': 2, 'moist': 4, 'scalar': 4}
    declarations, allocations, initial, update, write = [], [], [], [], []
    for field in FIELDS:
        rank = ranks.get(field, 3)
        shape = ','.join(['n'] + ['1'] * (rank - 2) + ['2' if rank == 4 else '1'])
        colons = ','.join([':'] * rank)
        dfi = accumulation_field(field)
        declarations += [f' real, allocatable :: {field}({colons}), {dfi}({colons}), d_{field}({colons})']
        allocations += [f' allocate(grid%{field}({shape}),grid%{dfi}({shape}),grid%d_{field}({shape}))']
        initial += [f' read(11) grid%{field}', f' read(11) grid%d_{field}', f' grid%{dfi}=0.']
        update += [f' grid%{field}=grid%{field}+grid%d_{field}']
        write += [f' write(12) grid%{dfi}', f' write(13) grid%{field}']
    driver = '''module module_domain
 type domain
 integer :: dfi_opt=1, dfi_stage=1, dfi_savehydmeteors=0, mp_physics_dfi=0, itimestep=0
 real :: hcoeff(60), hcoeff_tot=0.
''' + '\n'.join(declarations) + '''
 end type
end module
module module_driver_constants
end module
module module_machine
end module
module module_model_constants
end module
module module_state_description
 integer, parameter :: DFI_NODFI=0, DFI_FST=3, DFI_BCK=1, DFI_FWD=2
 ! Registry.EM_COMMON:3082, package nuwrf4icescheme_dfi mp_physics_dfi==7.
 integer, parameter :: P_QV=1, NUWRF4ICESCHEME_DFI=7
end module
''' + accumulate + '''
subroutine normalize(grid)
 use module_domain
 use module_state_description
 type(domain), pointer :: grid
''' + reset + '''
end subroutine
program array_driver
 use module_domain
 implicit none
 integer, parameter :: n=1031
 type(domain), pointer :: grid
 character(1024) :: input, weights, accum, normalized, arg
 integer :: i
 interface
 subroutine dfi_accumulate(grid)
 use module_domain
 type(domain), pointer :: grid
 end subroutine
 subroutine normalize(grid)
 use module_domain
 type(domain), pointer :: grid
 end subroutine
 end interface
 allocate(grid)
 call get_command_argument(1,input)
 call get_command_argument(2,weights)
 call get_command_argument(3,accum)
 call get_command_argument(4,normalized)
 call get_command_argument(5,arg)
 read(arg,*) grid%mp_physics_dfi
''' + '\n'.join(allocations) + '''
 open(11,file=trim(input),access='stream',form='unformatted',status='old')
''' + '\n'.join(initial) + '''
 close(11)
 open(11,file=trim(weights),access='stream',form='unformatted',status='old')
 read(11) grid%hcoeff
 close(11)
 do i=0,59
 grid%itimestep=i
 call dfi_accumulate(grid)
 if(i<59) then
''' + '\n'.join(update) + '''
 endif
 enddo
 call normalize(grid)
 open(12,file=trim(accum),access='stream',form='unformatted',status='replace')
 open(13,file=trim(normalized),access='stream',form='unformatted',status='replace')
''' + '\n'.join(write) + '''
 write(12) grid%hcoeff_tot
 close(12)
 close(13)
end program
'''
    target = out / 'original_array.F90'
    target.write_text(driver)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    import numpy as np
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.read_bytes()
    pin = json.loads((HERE / 'SOURCE.json').read_text())['files']['share/dfi.F']
    assert hashlib.sha256(source).hexdigest() == pin
    native = original_driver(source.decode(), out)
    flags = ['-cpp', '-DEM_CORE=1', '-ffree-form', '-ffree-line-length-none',
             '-O2', '-fno-fast-math', '-ffp-contract=off', '-fcheck=bounds']
    with (out / 'gpu-commands.log').open('w') as log:
        command = ['gfortran', *flags, str(native), '-o', str(out / 'array_driver')]
        result = subprocess.run(command, capture_output=True, text=True, cwd=out)
        log.write('$ ' + ' '.join(command) + '\n' + result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        lib = C.CDLL(str(out / 'libarwen_dfi.so'))
        weights = np.zeros(60, dtype=np.float32)
        written = C.c_size_t()
        fn = lib.aw_dfi_weights
        fn.argtypes = [C.c_int, C.c_float, C.c_float, C.c_int, C.c_void_p,
                       C.c_size_t, C.POINTER(C.c_size_t)]
        assert fn(1770, 30, 3600, 4, weights.ctypes.data, 60, C.byref(written)) == 0
        assert written.value == 60
        weights.tofile(out / 'weights.bin')
        rng = np.random.default_rng(471)
        initial, delta, counts = [], [], []
        with (out / 'fixture.bin').open('wb') as fixture:
            for field in FIELDS:
                count = 1031 * (2 if field in ('moist', 'scalar') else 1)
                a = rng.uniform(-100, 100, count).astype(np.float32)
                d = rng.uniform(-0.125, 0.125, count).astype(np.float32)
                # Exercise strict gradual underflow as well as signed data.
                tiny = np.finfo(np.float32).tiny
                sub = np.nextafter(np.float32(0), np.float32(1))
                a[:6] = [0.0, -0.0, tiny/2, -tiny/2, sub, -sub]
                d[:6] = 0.0
                fixture.write(a.tobytes())
                fixture.write(d.tobytes())
                initial.append(a)
                delta.append(d)
                counts.append(count)
        code = (ROOT / 'woof/core/kernels/dfi.cu').read_text()
        # Synthetic live-state graph. This is expressly not a weather model.
        code += r'''
extern "C" __global__ void fixture_step(const unsigned long long* rows,
                                       const unsigned long long* changes) {
 const unsigned long long* row=rows+4ull*blockIdx.y;
 unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<row[2]) {
   float* state=reinterpret_cast<float*>(row[0]);
   const float* d=reinterpret_cast<const float*>(changes[blockIdx.y]);
   state[i]=__fadd_rn(state[i],d[i]);
 }
}
'''
        driver = C.CDLL('libcuda.so.1')
        graph_launch = driver.cuGraphLaunch
        kernel_launch = driver.cuLaunchKernel
        submit = lib.aw_dfi_submit_pass
        submit.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                           C.c_void_p, C.POINTER(C.c_void_p), C.c_uint,
                           C.c_uint, C.c_uint, C.c_void_p, C.POINTER(C.c_void_p)]
        submit.restype = C.c_int
        receipt = {'scope': 'synthetic arrays and synthetic advancing CUDA graph',
                   'real_forecast_capture': False, 'source_sha256': pin,
                   'kernel_sha256': hashlib.sha256(code.encode()).hexdigest(),
                   'rust_library_sha256': hashlib.sha256((out / 'libarwen_dfi.so').read_bytes()).hexdigest(),
                   'fixture_sha256': hashlib.sha256((out / 'fixture.bin').read_bytes()).hexdigest(),
                   'original_array_sha256': hashlib.sha256(native.read_bytes()).hexdigest(),
                   'cupy_version': cp.__version__,
                   'nvrtc_version': list(cp.cuda.nvrtc.getVersion()),
                   'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                   'cases': [], 'differing_words': 0, 'words': 0}
        for strict in (True, False):
            options = ['--std=c++11']
            if strict:
                options += ['-DGPUWM_WRF_EXACT=1', '--fmad=false', '--ftz=false',
                            '--prec-div=true', '--prec-sqrt=true']
            # RawModule(code=...) appends -ftz=true AFTER caller options in
            # CuPy 14. Compile at the direct NVRTC boundary, then load that
            # image, so the strict flags recorded here are effective.
            image, _ = cp.cuda.compiler.compile_using_nvrtc(
                code, options=tuple(options), arch=cp.cuda.Device().compute_capability)
            image_path = out / ('strict.cubin' if strict else 'fast.cubin')
            image_path.write_bytes(image)
            module = cp.RawModule(path=str(image_path))
            accumulate = module.get_function('dfi_accumulate_fields')
            advance = module.get_function('dfi_advance_coefficient')
            reset = module.get_function('dfi_reset_fields')
            step = module.get_function('fixture_step')
            log.write('NVRTC options: ' + repr(options) + '\n')
            for mp in (0, 7):
                command = [str(out / 'array_driver'), str(out / 'fixture.bin'),
                           str(out / 'weights.bin'), str(out / 'accum.bin'),
                           str(out / 'reset.bin'), str(mp)]
                result = subprocess.run(command, capture_output=True, text=True, cwd=out)
                log.write('$ ' + ' '.join(command) + '\n' + result.stdout + result.stderr)
                assert result.returncode == 0, result.stderr
                expected_accum = np.fromfile(out / 'accum.bin', dtype=np.float32)
                expected_reset = np.fromfile(out / 'reset.bin', dtype=np.float32)
                active = DYNAMICS + ['moist', 'scalar'] + (GSFC_RADII if mp else ORDINARY_RADII)
                indices = [FIELDS.index(x) for x in active]
                states = [cp.asarray(initial[i]) for i in indices]
                changes = [cp.asarray(delta[i]) for i in indices]
                accumulated = [cp.zeros_like(x) for x in states]
                rows = cp.asarray(np.asarray([
                    [a.data.ptr, b.data.ptr, a.size, int(field not in DYNAMICS)]
                    for field, a, b in zip(active, states, accumulated)], dtype=np.uint64))
                change_ptrs = cp.asarray(np.asarray([x.data.ptr for x in changes], dtype=np.uint64))
                h = cp.asarray(weights)
                cursor = cp.zeros(1, dtype=cp.uint32)
                total = cp.zeros(1, dtype=cp.float32)
                error = cp.zeros(1, dtype=cp.uint32)
                blocks = (max(x.size for x in states) + 255) // 256
                grid = (blocks, len(active))
                # Force compilation before stream capture.
                step.compile()
                stream = cp.cuda.Stream(non_blocking=True)
                cp.cuda.get_current_stream().synchronize()
                with stream:
                    stream.begin_capture()
                    step(grid, (256,), (rows, change_ptrs))
                    graph = stream.end_capture()
                    accumulate(grid, (256,), (rows, h, cursor, np.uint32(60), error))
                    advance((1,), (1,), (h, cursor, np.uint32(60), total, error))
                    # CUDA driver copies argument values during submission;
                    # ctypes owners below remain live through completion.
                    args_acc = [C.c_void_p(x.data.ptr) for x in (rows, h, cursor)]
                    args_acc += [C.c_uint(60), C.c_void_p(error.data.ptr)]
                    params_acc = (C.c_void_p * len(args_acc))(*[C.addressof(x) for x in args_acc])
                    args_cursor = [C.c_void_p(h.data.ptr), C.c_void_p(cursor.data.ptr),
                                   C.c_uint(60), C.c_void_p(total.data.ptr), C.c_void_p(error.data.ptr)]
                    params_cursor = (C.c_void_p * len(args_cursor))(*[C.addressof(x) for x in args_cursor])
                    start, stop = cp.cuda.Event(), cp.cuda.Event()
                    start.record()
                    status = submit(C.cast(graph_launch, C.c_void_p), C.cast(kernel_launch, C.c_void_p),
                                    graph.graphExec, stream.ptr, accumulate.kernel.ptr,
                                    params_acc, blocks, len(active), 59,
                                    advance.kernel.ptr, params_cursor)
                    assert status == 0, f'CUDA submission failed: {status}'
                    stop.record()
                    reset(grid, (256,), (rows, total, error))
                stream.synchronize()
                assert int(error.get()[0]) == 0
                assert int(cursor.get()[0]) == 60
                differing = 0
                word_count = 1
                offset = 0
                max_abs = 0.0
                for i, field in enumerate(FIELDS):
                    count = counts[i]
                    if field in active:
                        j = active.index(field)
                        for actual, expected in ((accumulated[j].get(), expected_accum[offset:offset+count]),
                                                 (states[j].get(), expected_reset[offset:offset+count])):
                            assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
                            differing += int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
                            max_abs = max(max_abs, float(np.max(np.abs(actual-expected))))
                            word_count += count
                    offset += count
                differing += int(total.get().view(np.uint32)[0] != expected_accum[-1:].view(np.uint32)[0])
                if strict:
                    assert differing == 0, f'strict differs in {differing} words'
                    receipt['differing_words'] += differing
                    receipt['words'] += word_count
                receipt['cases'].append({'strict': strict, 'mp_physics_dfi': mp,
                    'nvrtc_options': options,
                    'samples': 60, 'active_fields': len(active), 'words': word_count,
                    'differing_words': differing, 'max_abs_difference': max_abs,
                    'synthetic_graph_ms_per_step': cp.cuda.get_elapsed_time(start, stop)/59})
        (out / 'gpu-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
