"""Replay stream observations made inside a strict native GSFCSWRAD call.

The bounded recorder and complete run command are retained with the lane's
evidence. This reader preserves every native float32 input word. It never
constructs atmospheric profiles or replaces nonfinite native outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from woof.core.gsfcsw import packed_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.captures.glob('gsfc-real-v2-*.bin')):
        raw = path.read_bytes()
        nl, nc, warm, qi, qn, day = np.frombuffer(raw[:24], dtype='>i4')
        solar, latitude, heat_capacity, gravity = np.frombuffer(raw[24:40], dtype='>f4')
        assert heat_capacity == np.float32(1004.5) and gravity == np.float32(9.81)
        n = int(nl)+1
        native = np.frombuffer(raw[40:], dtype='>f4').astype('f4').reshape(int(nc),13*n+4)
        source = cp.asarray(native[:,:9*n+2])
        incoming = cp.asarray(native[:,9*n+2:10*n+1])
        expected = native[:,10*n+2:]
        settings = dict(warm_rain=bool(warm), f_qi=bool(qi), f_qndrop=bool(qn),
                        julday=int(day), solcon=float(solar), center_lat=float(latitude))
        actual = cp.asnumpy(packed_columns(source, int(nl), initial_heating=incoming, **settings))
        fields = {}
        for name, region in (('RTHRATEN',slice(0,int(nl))),
                             ('TAUCLDI',slice(n,2*n-1)),
                             ('TAUCLDC',slice(2*n,3*n-1)),
                             ('GSW',slice(3*n,3*n+1)),
                             ('RSWTOA',slice(3*n+1,3*n+2))):
            got = np.ascontiguousarray(actual[:,region])
            want = np.ascontiguousarray(expected[:,region])
            different = got.view('u4') != want.view('u4')
            fields[name] = dict(words=int(want.size),
                different_words=int(np.count_nonzero(different)),
                first_differences=[{'column':int(i), 'level':int(k),
                    'native_word':int(want[i,k].view('u4')), 'gpu_word':int(got[i,k].view('u4')),
                    'native':float(want[i,k]), 'gpu':float(got[i,k])}
                    for i,k in np.argwhere(different)[:8]],
                native_nonfinite=int(np.count_nonzero(~np.isfinite(want))),
                gpu_nonfinite=int(np.count_nonzero(~np.isfinite(got))),
                native_sha256=hashlib.sha256(want.tobytes()).hexdigest(),
                gpu_sha256=hashlib.sha256(got.tobytes()).hexdigest())
        rows.append(dict(capture=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                         bytes=len(raw), layers=int(nl), columns=int(nc),
                         settings=settings, coszen_min=float(native[:,9*n].min()),
                         coszen_max=float(native[:,9*n].max()),
                         fields=fields))
    assert rows, 'no native module observations found'
    result = dict(schema='gsfcsw-native-real-replay-v2', numerics=__import__('woof.numerics',fromlist=['mode']).mode(),
                  captures=rows, words=sum(f['words'] for r in rows for f in r['fields'].values()),
                  different_words=sum(f['different_words'] for r in rows for f in r['fields'].values()))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='captures'}))
    return int(result['different_words'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
