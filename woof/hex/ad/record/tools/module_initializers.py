#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1 (ad/phys): list the zero-argument kernels the ordinary forward launched BEFORE the first replayed step, in
modules the plan loads. Their only possible effect is module-scope device state (`__device__` globals), which the
recorder's step-1 initial snapshot (live driver allocations only) never holds. The runner launches them, in
recorded order, right after loading the modules (plan field module_initializers).

Breakage this prevents: RUC's ruc_tbq_init fills the module global ruc_tbq_table once per loaded module at
physics_init (step 0). The replay loaded the RUC module from its image and never ran it, so sea ice
(oracle_ice_ruc_ice, not held) read a zero saturation table: 38 allocations wrong at step 1, the same way in every
replay (sumtrace localisation, 2026-11-04).

Refuses a pre-step launch in a plan module that takes no parameters but passes them through `extra`, and any
zero-argument launch whose function or module cannot be resolved from the trace.
Usage: module_initializers.py DRIVER_TRACE.jsonl.gz PLAN.json  -> JSON list on stdout
"""
import gzip
import json
import sys


def _lines(trace_path):
    import shutil
    import subprocess
    if shutil.which('pigz'):
        proc = subprocess.Popen(['pigz', '-dc', str(trace_path)], stdout=subprocess.PIPE, bufsize=1 << 22)
        try:
            yield from proc.stdout
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()
        return
    with gzip.open(trace_path, 'rb') as stream:
        yield from stream


def initializers(trace_path, plan):
    modules = {m['id'] for m in plan['modules']}
    image = {}       # module handle -> image sha256
    function = {}    # function handle -> (module handle, name)
    found = []
    # G2: inflate in pigz beside the parse and json-decode only the four record types read here. THE BREAKAGE THIS
    # PREVENTS: gzip text mode + json.loads on every pre-step trace line ran at 100 % of one core at the start of
    # p1fix_package (part of its 15-min pre-pool stretch, a recording host 2026-11-04 20:15Z) while the card waited.
    wanted = (b'"ModuleImage"', b'"ModuleFunction"', b'"LaunchKernel"', b'"Boundary"')
    for line in _lines(trace_path):
            if not any(w in line for w in wanted):
                continue
            d = json.loads(line)
            t = d.get('type')
            if t == 'Boundary' and d.get('a') == 1:
                break
            if t == 'ModuleImage' and d.get('module'):
                image[d['module']] = d['sha256']
            elif t == 'ModuleFunction':
                function[d['function']] = (d['module'], bytes.fromhex(d['name_hex']).decode())
            elif t == 'LaunchKernel':
                if d.get('parameters') or d.get('extra'):
                    continue
                handle = int(d['function'], 16)
                if handle not in function or function[handle][0] not in image:
                    sys.exit(f'refusing: unresolved zero-argument pre-step launch at sequence {d["sequence"]}')
                module_handle, name = function[handle]
                if image[module_handle] not in modules:
                    continue
                if d.get('shared_bytes', 0) < 0:
                    sys.exit('refusing: invalid shared bytes')
                found.append({'module': image[module_handle], 'function': name, 'grid': d['grid'],
                              'block': d['block'], 'shared_bytes': d.get('shared_bytes', 0),
                              'recorded_sequence': d['sequence']})
    return found


if __name__ == '__main__':
    plan = json.load(open(sys.argv[2]))
    print(json.dumps(initializers(sys.argv[1], plan)))
