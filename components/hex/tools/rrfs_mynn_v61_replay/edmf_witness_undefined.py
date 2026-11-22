"""Witness for the MYNN-EDMF v6.1.1 cycled first step: native reads an undefined driver local there.

At initflag = 1 with restart false the v6.1.1 driver stages none of the turbulence state (src/
module_bl_mynnedmf_driver.F90:646-670 runs only for initflag == 0 .or. restart), and with cycling true the core
then reads maxval(qke1(:)) of that never-written automatic array (src/module_bl_mynnedmf.F90:462-467) to decide
initialize_qke. So the native answer is whatever the stack held.

    1. the standalone referee, fresh process, on the cycled call: POST A;
    2. the same referee, same inputs, after one other real call has run in the process (a different stack): POST B;
       A != B shows the native words depend on undefined memory;
    3. the referee with one labelled witness edit (qke1(kts:kte) = zero at the staging point, the value hex's zeroed
       arena holds) against the GPU module: every word.
A referee tool: native builds run on the CPU here only, never on a run path.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]


def post_of(ref, values, np):
    status = ref.call(values)
    arrays = {a['name']: values[a['name']] for a in ref.arguments
              if a['dimensions'] and a['intent'] in ('out', 'inout') and values.get(a['name']) is not None}
    arrays['errflg'] = np.array([status['errflg']], dtype=np.int32)
    return arrays


def digest(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode()); h.update(arrays[k].tobytes(order='F'))
    return h.hexdigest()


def main(a):
    import numpy as np
    from edmf_referee import Referee
    from edmf_replay import records, load, calls, interface_pint, v6_arguments, compare
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c6d08709 as pinned
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_v611 as v611
    codec = records()

    def arguments(capture, interface, index):
        call = calls(capture)[index]
        pre = load(capture, call['prefix'], 'pre', list(pinned.ARGUMENTS), codec)
        values = v6_arguments(pinned, pre, interface_pint(interface, index, codec), v611)
        values = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in values.items()}
        values['errflg'] = 0
        return values

    ref = Referee(a.referee)
    if a.dirty:
        post_of(ref, arguments(a.dirty_capture, a.dirty_interface, a.dirty_index), np)
    arrays = post_of(ref, arguments(a.capture, a.interface, a.index), np)
    np.savez(a.out, **arrays)
    row = dict(referee=ref.receipt['library_sha256'], source_edits=ref.receipt.get('source_edits', []),
               dirty_first=bool(a.dirty), digest=digest(arrays), post=str(a.out))
    if a.against:
        other = dict(np.load(a.against))
        fields = compare(other, arrays, [k for k in arrays if k in other])
        row.update(against=str(a.against), words=sum(f['words'] for f in fields),
                   different=sum(f['different'] for f in fields))
    print(json.dumps(row))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--referee', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--interface', type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--dirty', action='store_true')
    p.add_argument('--dirty-capture', type=Path)
    p.add_argument('--dirty-interface', type=Path)
    p.add_argument('--dirty-index', type=int, default=1)
    p.add_argument('--against', type=Path)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
