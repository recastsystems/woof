"""Module replay of MYNN-EDMF v6.1.1 (bl_mynnedmf_v611) on real 44c0849 calls, against the standalone native tag.

Inputs are captures of native RRFS-MPAS 44c0849 (strict) at its mynnedmf_driver call (the c6d08709 ABI, PRE and POST)
and at MPAS_to_physics (pres2_hyd_p, the interface pressure the v6 driver takes as pint), one model step each.

    witness   standalone pinned referee (c6d08709) on PRE vs the captured native POST: every word.
    referee   standalone v6.1.1 referee on the v6 arguments (the module's own native_driver_arguments mapping):
              writes its POST words (scratch, deleted after the receipt).
    gpu       bl_mynnedmf_v611 on the GPU (strict image), same arguments, vs the referee POST, at lane widths.

A referee tool: the native builds run on the CPU here only, never on a run path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent
sys.path[:0] = [str(HERE), str(TREE / 'tests'), str(TREE / 'src')]

#: The capture record codec (the Rust bridge.so the A3 capture tooling builds); MYNN_CAPTURE_CODEC names it.
CODEC = Path(os.environ.get('MYNN_CAPTURE_CODEC', 'capture-codec/bridge.so'))
#: Outputs of the pinned c6d08709 driver that native leaves undefined with config_mynn_tkebudget = 0: it copies its
#: never-written automatic TKE-budget arrays dqke1..qdiss1 out (MPAS/module_bl_mynnedmf_driver.F90 at c6d08709;
#: the A3 hidden-state capture, bl_mynnedmf_c6d08709.MynnEdmfCuda.run). Stack contents, so the standalone build
#: cannot reproduce them; the v6.1.1 driver allocates those arrays only when tke_budget == 1 and leaves the outputs
#: untouched otherwise.
PINNED_UNDEFINED = ('dqke', 'qwt', 'qshear', 'qbuoy', 'qdiss')


def records():
    from rrfs_driver_dump_replay import Records
    return Records(CODEC)


def calls(capture: Path):
    rows = [json.loads(line) for line in (capture / 'rank0-calls.jsonl').read_text().splitlines()]
    return [r for r in rows if r['phase'] == 'pre']


def load(capture: Path, prefix: str, phase: str, names, codec):
    out = {}
    for name in names:
        path = capture / f'{prefix}.{phase}.{name}.bin'
        if not path.exists():
            out[name] = None
            continue
        value, _ = codec.load(path)
        out[name] = value
    return out


def interface_pint(capture: Path, step_index: int, codec):
    path = capture / f'boundary__mpas_to_physics.s0000000000.e{step_index:010d}.r000000.post.vars.pres2_hyd_p.bin'
    value, _ = codec.load(path)
    return value


def v6_arguments(pinned, pre, pint, v611):
    """The v6 driver arguments the route builds from the same host state (bl_mynnedmf_v611.native_driver_arguments).

    The captured call is the 44c0849 host's (c6d08709 dummies); its host arrays are named by the pinned module's
    NATIVE_ARRAY_BINDINGS, its options by OPTION_ARGUMENTS, so the host state is reconstructed exactly."""
    import numpy as np
    prepared = {host: pre[dummy] for dummy, host in pinned.NATIVE_ARRAY_BINDINGS.items() if pre.get(dummy) is not None}
    prepared['pres2_hyd_p'] = pint
    scalar = lambda n: pre[n].reshape(-1)[0].item()
    metadata = {n: scalar(n) for n in v611.HOST_SCALARS}
    metadata.update(its=scalar('its'), ite=scalar('ite'), dt_pbl=scalar('delt'), do_chemistry=bool(scalar('mix_chem')),
                    num_chem=scalar('nchem'))
    config = {key: scalar(arg) for key, (arg, _) in pinned.OPTION_ARGUMENTS.items()}
    config.update(config_do_restart=bool(scalar('do_restart')), config_do_DAcycling=bool(scalar('do_dacycling')))
    initflag = scalar('initflag')
    values = v611.native_driver_arguments(prepared, metadata, config, itimestep=1 if initflag else 2)
    assert values['initflag'] == initflag, (values['initflag'], initflag)
    # Fresh private copies: each run writes its own.
    return {k: (np.array(v, order='F', copy=True) if isinstance(v, np.ndarray) else v) for k, v in values.items()}


def compare(expected, actual, names):
    import numpy as np
    rows = []
    for name in names:
        e, a = expected.get(name), actual.get(name)
        if e is None and a is None:
            continue
        e, a = np.asarray(e), np.asarray(a)
        assert e.shape == a.shape and e.dtype == a.dtype, (name, e.shape, a.shape, e.dtype, a.dtype)
        ev = np.ascontiguousarray(e).reshape(-1).view('V' + str(e.dtype.itemsize))
        av = np.ascontiguousarray(a).reshape(-1).view('V' + str(a.dtype.itemsize))
        rows.append(dict(name=name, words=int(ev.size), different=int(np.count_nonzero(ev != av))))
    return rows


def witness(args):
    import numpy as np
    from edmf_referee import Referee
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c6d08709 as pinned
    codec = records()
    ref = Referee(args.referee)
    names = [a['name'] for a in ref.arguments]
    out = []
    for call in calls(args.capture):
        pre = load(args.capture, call['prefix'], 'pre', names, codec)
        post = load(args.capture, call['prefix'], 'post', names, codec)
        values = {}
        for k, v in pre.items():
            if k == 'errmsg' or v is None:
                continue
            values[k] = (np.array(v, order='F', copy=True) if pinned.ARGUMENTS[k]['dimensions']
                         else v.reshape(-1)[0].item())
        status = ref.call(values)
        arrays = [a['name'] for a in ref.arguments if a['dimensions'] and a['intent'] in ('out', 'inout')]
        actual = {k: values.get(k) for k in arrays}
        actual['errflg'] = np.array(status['errflg'], dtype=np.int32).reshape(post['errflg'].shape)
        rows = compare(post, actual, arrays + ['errflg'])
        declared = [r for r in rows if r['name'] in PINNED_UNDEFINED]
        counted = [r for r in rows if r['name'] not in PINNED_UNDEFINED]
        out.append(dict(call=call['call'], prefix=call['prefix'], fields=rows,
                        words=sum(r['words'] for r in counted), different=sum(r['different'] for r in counted),
                        declared_undefined=declared))
        print(json.dumps({k: out[-1][k] for k in ('call', 'words', 'different')}), flush=True)
    receipt = dict(schema='mynn-v61-harness-witness-v1', referee=ref.receipt['library_sha256'],
                   capture=str(args.capture), rows=out, words=sum(r['words'] for r in out),
                   different=sum(r['different'] for r in out), declared_undefined_fields=list(PINNED_UNDEFINED))
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return int(receipt['different'] != 0)


def referee(args):
    import numpy as np
    from edmf_referee import Referee
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c6d08709 as pinned
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_v611 as v611
    codec = records()
    ref = Referee(args.referee)
    c6d_names = list(pinned.ARGUMENTS)
    args.out.mkdir(parents=True, exist_ok=True)
    out = []
    for index, call in enumerate(calls(args.capture)):
        pre = load(args.capture, call['prefix'], 'pre', c6d_names, codec)
        pint = interface_pint(args.interface, index, codec)
        values = v6_arguments(pinned, pre, pint, v611)
        values = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in values.items()}
        values['errflg'] = 0
        status = ref.call(values)
        arrays = {a['name']: values[a['name']] for a in ref.arguments
                  if a['dimensions'] and a['intent'] in ('out', 'inout') and values.get(a['name']) is not None}
        arrays['errflg'] = np.array([status['errflg']], dtype=np.int32)
        path = args.out / f'call{index}.npz'
        np.savez(path, **arrays)
        out.append(dict(call=call['call'], prefix=call['prefix'], post=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), errflg=status['errflg'],
                        errmsg=status['errmsg'], initflag=values['initflag'],
                        restart=values['restart'], cycling=values['cycling']))
        print(json.dumps(out[-1]), flush=True)
    receipt = dict(schema='mynn-v61-referee-post-v1', referee=ref.receipt, capture=str(args.capture),
                   interface=str(args.interface), rows=out)
    (args.out / 'referee-post.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


def gpu(args):
    import numpy as np
    from woof.hex import numerics
    numerics.set_mode(args.numerics, source='mynn-v61-replay')
    import cupy as cp
    from types import SimpleNamespace
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c6d08709 as pinned
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_v611 as v611
    codec = records()
    packet = json.loads(args.packet.read_text())['packet']
    image = v611.load_cuda_packet(packet, SimpleNamespace(cuda=cp.cuda))
    backend = v611.MynnEdmfCuda(compiled_image=image)
    post_receipt = json.loads((args.post / 'referee-post.json').read_text())
    rows = []
    for index, call in enumerate(calls(args.capture)):
        pre = load(args.capture, call['prefix'], 'pre', list(pinned.ARGUMENTS), codec)
        pint = interface_pint(args.interface, index, codec)
        values = v6_arguments(pinned, pre, pint, v611)
        device = {k: (cp.asarray(v, order='F') if isinstance(v, np.ndarray) else v) for k, v in values.items()}
        cp.cuda.Stream.null.synchronize()
        result = backend.run(device)
        cp.cuda.Stream.null.synchronize()
        expected = dict(np.load(post_receipt['rows'][index]['post']))
        actual = {k: cp.asnumpy(device[k]) for k in expected if k != 'errflg'}
        errors = cp.asnumpy(result['errflg'])
        assert np.all(errors == errors[0]), 'per-column error flags differ'
        actual['errflg'] = errors[:1].astype(np.int32)
        fields = compare(expected, actual, list(expected))
        rows.append(dict(call=call['call'], lane_width=backend.last_lane_width, batch_columns=backend.last_batch_columns,
                         arena_bytes_per_column=v611.workspace_bytes(device), fields=fields,
                         words=sum(f['words'] for f in fields), different=sum(f['different'] for f in fields)))
        print(json.dumps({k: rows[-1][k] for k in ('call', 'lane_width', 'words', 'different')}), flush=True)
        if args.debug and rows[-1]['different']:
            print(json.dumps([f for f in fields if f['different']]), flush=True)
    receipt = dict(schema='mynn-v61-module-replay-v1', numerics=args.numerics, rows=rows,
                   words=sum(r['words'] for r in rows), different=sum(r['different'] for r in rows),
                   lane_width_forced=os.environ.get('WOOF_HEX_MYNNEDMF_LANES'), packet=packet,
                   device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return int(receipt['different'] != 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    w = sub.add_parser('witness')
    w.add_argument('--referee', type=Path, required=True)
    w.add_argument('--capture', type=Path, required=True)
    w.add_argument('--receipt', type=Path, required=True)
    r = sub.add_parser('referee')
    r.add_argument('--referee', type=Path, required=True)
    r.add_argument('--capture', type=Path, required=True)
    r.add_argument('--interface', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    g = sub.add_parser('gpu')
    g.add_argument('--packet', type=Path, required=True)
    g.add_argument('--capture', type=Path, required=True)
    g.add_argument('--interface', type=Path, required=True)
    g.add_argument('--post', type=Path, required=True)
    g.add_argument('--receipt', type=Path, required=True)
    g.add_argument('--numerics', default='strict')
    g.add_argument('--debug', action='store_true')
    a = p.parse_args()
    raise SystemExit({'witness': witness, 'referee': referee, 'gpu': gpu}[a.action](a))
