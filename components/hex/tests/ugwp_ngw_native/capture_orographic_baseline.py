"""Replay f1's unchanged native parent with its embedded NGW logical false.

module_bl_ugwp_gwdo.F:361-376 gates NGW; the parent first applies orographic
drag. All original argument/state words are retained except that scalar logical.
This provides the actual additive baselines for separate NGW column replay.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
from compare_real import Transport, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dump','parent-replay','transport','out','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    transport = Transport(args.transport)
    fn = transport.lib.bridge_disable_ngw
    fn.argtypes = [C.c_char_p,C.c_char_p]; fn.restype = C.c_int
    inputs = args.out/'inputs'; inputs.mkdir()
    captures = sorted(args.dump.glob('gwdo_ugwp*.pre.ngw_scheme.bin'))
    if not captures: raise ValueError('no embedded NGW call capture')
    records = []
    for selector in captures:
        prefix = selector.name.removesuffix('.pre.ngw_scheme.bin')
        outputs = args.out/prefix
        hashes = {}
        for path in args.dump.glob(prefix+'.pre.*.bin'):
            hashes[path.name] = sha(path)
            destination = inputs/path.name
            if path == selector:
                if fn(str(path.resolve()).encode(),str(destination.resolve()).encode()):
                    raise ValueError(transport.lib.bridge_error().decode())
            else:
                destination.symlink_to(path.resolve())
        with (args.out/(prefix+'.log')).open('w') as log:
            env = dict(os.environ,MPAS_ORACLE_DIR=str(args.out.resolve()),MPAS_ORACLE_STEPS='0,1',MPAS_ORACLE_KEYS='gwdo_ugwp__driver_gwdo__L781')
            subprocess.run([str(args.parent_replay.resolve()),str(inputs/prefix),str(outputs.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        payloads = sorted(args.out.glob(prefix+'.post.*.bin'))
        if not payloads: raise ValueError('native parent replay wrote no outputs')
        records.append(dict(capture=prefix,inputs_sha256=hashes,disabled_selector_sha256=sha(inputs/selector.name),outputs_sha256={p.name:sha(p) for p in payloads}))
    transport.close()
    args.receipt.write_text(json.dumps(dict(schema='p15.native-orographic-baseline/v1',acceptance=False,parent_replay_sha256=sha(args.parent_replay),generator_sha256=sha(__file__),transport_sha256=sha(args.transport),cases=records),indent=2)+'\n')


if __name__ == '__main__': main()
