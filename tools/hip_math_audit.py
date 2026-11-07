#!/usr/bin/env python3
"""Enumerate every binary32 input word through strict device math.

Run CUDA with --output cuda.json; run HIP with --authority cuda.json.
SHA-256 binds every output word in ordered chunks, including NaN payloads.
Double functions receive the exact binary64 conversion of each float input;
this does not exhaust the binary64 input space. No math implementation is
supplied by this diagnostic. It exercises the selected production compiler.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

UNARY = ('exp exp2 expm1 log log2 log10 log1p sin cos tan asin acos atan '
         'sinh cosh tanh asinh acosh atanh erf erfc tgamma lgamma cbrt').split()
FUNCTIONS = [name + 'f' for name in UNARY] + UNARY + ['powf_eos']
OPTIONS = ('--std=c++17', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true')


def source(name):
    double = name in UNARY
    expression = 'powf(x, 1.4f)' if name == 'powf_eos' else f'{name}(x)'
    cast = '__double_as_longlong' if double else '__float_as_uint'
    word = 'unsigned long long' if double else 'unsigned'
    value = 'double' if double else 'float'
    return f'''extern "C" __global__ void audit(unsigned long long first,
        unsigned count, {word} *out) {{
        unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=count)return;
        {value} x=({value})__uint_as_float((unsigned)(first+i));
        out[i]={cast}({expression});
    }}'''


def digest(array):
    return hashlib.sha256(memoryview(array)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--functions', nargs='+', choices=FUNCTIONS, default=FUNCTIONS)
    parser.add_argument('--chunk-words', type=int, default=2**24)
    parser.add_argument('--limit', type=int, default=2**32,
                        help='smaller values are partial diagnostics, never exhaustive proof')
    parser.add_argument('--hash-workers', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.limit <= 2**32 or not 0 < args.chunk_words <= 2**30:
        parser.error('invalid enumeration size')
    if os.environ.get('GPUWM_NUMERICS') != 'strict':
        parser.error('set GPUWM_NUMERICS=strict')

    import woof
    import cupy as cp
    from woof.hip_backend import install, device_info
    install()
    from cupy.cuda import compiler, function
    from woof.numerics import preserve_final_options
    authority = json.loads(args.authority.read_text()) if args.authority else None
    if authority and (authority['limit'] != args.limit or authority['chunk_words'] != args.chunk_words):
        parser.error('authority enumeration differs')
    result = dict(device=device_info(), backend=os.environ.get('WOOF_BACKEND'),
                  complete=False, requested_functions=args.functions,
                  limit=args.limit, chunk_words=args.chunk_words,
                  exhaustive_binary32_inputs=args.limit == 2**32,
                  double_domain='all binary32 inputs converted exactly to binary64',
                  powf_domain='all binary32 bases with the EOS exponent 1.4f only',
                  compilation_route='direct; no CuPy RawModule implicit FTZ append',
                  options=OPTIONS, functions={}, exact=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    with ThreadPoolExecutor(max_workers=args.hash_workers) as workers:
        for name in args.functions:
            text = source(name)
            with preserve_final_options():
                image, _ = compiler._compile_using_nvrtc_no_warning(text, OPTIONS)
            module = function.Module()
            module.load(image)
            kernel = module.get_function('audit')
            dtype = cp.uint64 if name in UNARY else cp.uint32
            out = cp.empty(min(args.chunk_words, args.limit), dtype)
            pending = []
            chunks = []
            start = time.monotonic()
            for first in range(0, args.limit, args.chunk_words):
                count = min(args.chunk_words, args.limit - first)
                kernel(((count + 255) // 256,), (256,), (np.uint64(first), np.uint32(count), out))
                # get() waits for the selected stream before a worker hashes.
                pending.append((first, count, workers.submit(digest, out[:count].get())))
                if len(pending) >= args.hash_workers + 1:
                    begin, words, future = pending.pop(0)
                    chunks.append(dict(first=begin, words=words, sha256=future.result()))
            for begin, words, future in pending:
                chunks.append(dict(first=begin, words=words, sha256=future.result()))
            row = dict(source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       enumerated_inputs=sum(chunk['words'] for chunk in chunks),
                       output_word_bytes=np.dtype(dtype).itemsize, chunks=chunks,
                       seconds=time.monotonic() - start)
            if authority:
                expected = authority['functions'][name]
                if row['source_sha256'] != expected['source_sha256']:
                    raise ValueError(f'authority expression differs: {name}')
                row['different_chunks'] = sum(a != b for a, b in zip(chunks, expected['chunks']))
                row['exact'] = chunks == expected['chunks']
            result['functions'][name] = row
            result['exact'] = (all(r['exact'] for r in result['functions'].values()) if authority else None)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps(dict(function=name, inputs=args.limit, seconds=row['seconds'],
                                  exact=row.get('exact'))), flush=True)
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2))
    return 1 if result['exact'] is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
