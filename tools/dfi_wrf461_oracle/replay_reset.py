#!/usr/bin/env python3
"""Bitwise GPU replay of full-column native DFI save/reset/rebalance captures."""
import argparse
import hashlib
import json
from pathlib import Path

from instrument_native import FIELDS, RADII, GSFC
from instrument_reset import WATER, SOIL, RUC, COORDS

ROOT = Path(__file__).resolve().parents[2]


def read(path, np):
    data = path.read_bytes()
    endian = '<' if int.from_bytes(data[:4], 'little') == 1145456946 else '>'
    h = np.frombuffer(data, dtype=endian+'i4', count=18).astype(np.int32)
    assert h[0] == 1145456946
    total, p_top = np.frombuffer(data, dtype=endian+'f4', count=2, offset=72).astype(np.float32)
    pos, fields = 80, {}
    while pos < len(data):
        name = data[pos:pos+32].decode().strip(); pos += 32
        *dims, n = np.frombuffer(data, dtype=endian+'i4', count=5, offset=pos).astype(int)
        pos += 20
        assert n == np.prod(dims) and n < 1000000, (name, dims, n)
        a = np.frombuffer(data, dtype=endian+'f4', count=n, offset=pos).astype(np.float32)
        pos += 4*n
        # Native is Fortran (x,k,y,species); GPU is (species,k,y,x).
        fields[name] = a.reshape(tuple(reversed(dims))).transpose(0, 2, 1, 3).copy().reshape(-1)
    assert pos == len(data)
    return dict(path=path.name, h=h, total=total, p_top=p_top, fields=fields)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import cupy as cp
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.capture.glob('lane-reset-*.bin'))
    captures = [read(x, np) for x in paths]
    assert captures
    source = '#ifdef GPUWM_WRF_EXACT\n' + (ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text() + '\n#endif\n'
    source += (ROOT/'woof/core/kernels/dfi.cu').read_text() + '\n'
    source += (ROOT/'woof/core/kernels/dfi_reset.cu').read_text()
    receipt = dict(scope='real native complete DFI save/reset/rebalance operators',
                   forecast_qualified=False, strict_words=0, strict_differing_words=0,
                   kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   captures=[dict(name=x.name, bytes=x.stat().st_size,
                                  sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in paths],
                   gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                   cases=[])
    for strict in (True, False):
        options = ['--std=c++17', '-I'+str(ROOT/'woof/core/kernels')]
        if strict:
            options += ['-DGPUWM_WRF_EXACT=1', '--fmad=false', '--ftz=false',
                        '--prec-div=true', '--prec-sqrt=true']
        image, _ = cp.cuda.compiler.compile_using_nvrtc(source, options=tuple(options),
                         arch=cp.cuda.Device().compute_capability)
        binary = args.output/('strict.cubin' if strict else 'fast.cubin')
        binary.write_bytes(image)
        module = cp.RawModule(path=str(binary))
        kernels = {name: module.get_function('dfi_'+name) for name in (
            'reset_fields', 'snapshot_fields', 'reset_surface_water',
            'reset_surface_soil', 'save_relative_humidity',
            'restore_saturated_vapour', 'rebalance_columns')}

        def descriptors(pairs):
            return cp.asarray(np.asarray([[a.data.ptr, b.data.ptr, a.size, flag]
                                   for a, b, flag in pairs], dtype=np.uint64))

        def rows_launch(name, pairs, extra=()):
            if not pairs:
                return None
            rows = descriptors(pairs)
            kernels[name](((max(a.size for a,b,f in pairs)+255)//256, len(pairs)),
                          (256,), (rows, *extra))
            return rows  # caller retains descriptors until completion

        def rebalance(d, cap, hold):
            h = cap['h']; cols = int(h[16]*h[17]); nz = int(h[12])
            # Native moisture slot 1 precedes PARAM_FIRST_SCALAR. Do not
            # include it in the recurrence's ordered species reduction.
            first = int(h[15])-1
            moist = d['moist'].reshape(int(h[7]), nz, cols)[first:].reshape(-1)
            names = ['mu_2', 'mub', 't_2', 'p', 'pb', 'alb', 'alt', 'al',
                     'p_hyd', 'ph_2', 'phb', 'ph0', 'psfc']
            arrays = [d[n] for n in names]+[moist]+[d[n] for n in COORDS]
            ptrs = cp.asarray(np.asarray([a.data.ptr for a in arrays], dtype=np.uint64))
            hold.extend([moist, ptrs])
            kernels['rebalance_columns'](((cols+127)//128,), (128,),
                (ptrs, np.uint32(cols), np.uint32(nz), np.uint32(int(h[7])-first),
                 np.int32(h[10]), np.int32(h[11]), np.float32(cap['p_top'])))

        def compare(kind, pre, post, live, names, elapsed_ms):
            result = dict(kind=kind, strict=strict, pre=pre['path'], post=post['path'],
                          stage=int(pre['h'][2]), words=0, differing_words=0,
                          elapsed_ms=elapsed_ms, fields=[])
            for name in names:
                got = cp.asnumpy(live[name]); expected = post['fields'][name]
                bad = np.flatnonzero(got.view(np.uint32) != expected.view(np.uint32))
                finite = np.isfinite(expected) & np.isfinite(got)
                f = dict(name=name, words=got.size, differing_words=bad.size,
                         max_abs=float(np.max(np.abs(got[finite]-expected[finite]), initial=0)))
                if bad.size:
                    at = int(bad[0]); f['first'] = dict(index=at, actual=float(got[at]),
                          expected=float(expected[at]), actual_word=int(got.view(np.uint32)[at]),
                          expected_word=int(expected.view(np.uint32)[at]))
                result['fields'].append(f)
                result['words'] += got.size; result['differing_words'] += bad.size
            if strict:
                receipt['strict_words'] += result['words']
                receipt['strict_differing_words'] += result['differing_words']
            receipt['cases'].append(result)

        for i, cap in enumerate(captures):
            mode = int(cap['h'][4]); h=cap['h']; cols=int(h[16]*h[17]); nz=int(h[12])
            if mode not in (4, 5, 7):
                continue
            d = {n: cp.asarray(a) for n,a in cap['fields'].items()}
            hold=[]
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            if mode == 4:
                names=WATER+SOIL+(RUC if int(h[9])==3 else [])
                pairs=[(d[n], d['dfi_'+n], 0) for n in names]
                expected_names=['dfi_'+n for n in names]
                if h[5]:
                    for name in ('moist', 'scalar'):
                        live, saved=d[name],d['dfi_'+name]
                        if name=='moist':
                            # Save all hydrometeors after water vapour.
                            first=int(h[15]); live=live[first*nz*cols:]; saved=saved[first*nz*cols:]
                        pairs.append((live,saved,1))
                        expected_names.append('dfi_'+name)
                hold.append(rows_launch('snapshot_fields',pairs))
                if h[2]==1:
                    m=d['moist'].reshape(int(h[7]),nz*cols)
                    # WRF's QV, QC, QI package indices for this captured run.
                    qv=m[int(h[14])-1]; qc=m[int(h[14])]; qi=m[int(h[14])+2]
                    hold.extend([m,qv,qc,qi])
                    kernels['save_relative_humidity'](((nz*cols+255)//256,), (256,),
                        (d['t_2'],d['p'],d['pb'],qv,qc,qi,d['dfi_rh'],
                         np.uint64(nz*cols),np.int32(h[10])))
                    expected_names.append('dfi_rh')
                stop.record(); stop.synchronize()
                compare('save',cap,cap,d,expected_names,cp.cuda.get_elapsed_time(start,stop))
                continue
            if mode==7:
                # A complete-reset capture is also immediately after the
                # rebalance when the optional internal mode 8 is absent.
                assert i+1<len(captures) and captures[i+1]['h'][4] in (6,8)
                rebalance(d,cap,hold)
                names=['p','alt','al','p_hyd','ph_2','ph0','psfc']
                post=captures[i+1]
            else:
                post=next(x for x in captures[i+1:] if x['h'][4]==6)
                assert np.array_equal(h[[1,2,3,5]],post['h'][[1,2,3,5]])
                pairs=[(d[a],d[b],0) for a,b,dim in FIELDS if a not in ('moist','scalar')]
                if h[5]:
                    v=(int(h[14])-1)*nz*cols
                    pairs.append((d['moist'][v:v+nz*cols],d['dfi_moist'][v:v+nz*cols],1))
                else:
                    for n in ['moist','scalar']+(GSFC if h[6]==7 else RADII):
                        pairs.append((d[n],d['dfi_'+n],1))
                total=cp.asarray(np.asarray([cap['total']],dtype=np.float32))
                error=cp.zeros(1,dtype=np.uint32);hold.extend([total,error])
                hold.append(rows_launch('reset_fields',pairs,(total,error)))
                hold.append(rows_launch('reset_surface_water',[(d[n],d['dfi_'+n],0) for n in WATER],(np.int32(h[5]),)))
                names=SOIL+(RUC if h[9]==3 else [])
                hold.append(rows_launch('reset_surface_soil',[(d[n],d['dfi_'+n],0) for n in names],
                                       (d['snowc'],np.uint32(cols),np.int32(h[5]))))
                if h[5]:
                    restore=[(d['dfi_scalar'],d['scalar'],0)]
                    if h[2]==2:
                        v=int(h[15])*nz*cols
                        restore.append((d['dfi_moist'][v:],d['moist'][v:],0))
                    hold.append(rows_launch('snapshot_fields',restore))
                    if h[2]==2:
                        v=(int(h[14])-1)*nz*cols; qv=d['moist'][v:v+nz*cols];hold.append(qv)
                        kernels['restore_saturated_vapour'](((nz*cols+255)//256,), (256,),
                            (d['t_2'],d['p'],d['pb'],qv,d['dfi_rh'],np.uint64(nz*cols),np.int32(h[10])))
                    rebalance(d,cap,hold)
                names=[n for n in d if not n.startswith('dfi_') and n not in COORDS]
            stop.record();stop.synchronize()
            compare('rebalance' if mode==7 else 'complete_reset',cap,post,d,names,
                    cp.cuda.get_elapsed_time(start,stop))
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('strict_words','strict_differing_words')}))
    if receipt['strict_differing_words']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
