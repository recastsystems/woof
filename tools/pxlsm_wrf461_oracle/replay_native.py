"""Compare every GPU output word against bounded native PXLSM snapshots.

This reads captures from instrument_native.py, not analytic fixtures.
Inputs and expected arrays stay in scratch; the report retains their hashes
and every field's differing-word count. No defect mask or tolerance is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from woof import wrf_exact
from pxlsm_oracle_support import OUTPUT_FIELDS, read_case, run_port, ulp_distance


def replay_live_inputs(case, path, start_time):
    """Replay the production Rust record reader and captured device clock."""
    from types import SimpleNamespace
    from woof.spectral_nudging.px_inputs import PxInputs
    from woof.core.pxlsm import launch_pxlsm
    from pxlsm_oracle_support import device_state, port_words
    fields, atm, landusef, soilcbot = device_state(case)
    state = SimpleNamespace(physics=SimpleNamespace(fields=fields),
                            elapsed_seconds=case.curr_secs0)
    cfg = SimpleNamespace(nx=case.nx,ny=case.ny,dt=case.dt,
                          sgfdda_interval_m=case.anal_interval/60.,
                          use_adaptive_time_step=False)
    owner = PxInputs(state,cfg,path,start_time,case.curr_secs0+case.dt)
    owner.before_step()
    # Compare every streamed endpoint before asking the physics to use it.
    for name in owner.names:
        np.testing.assert_array_equal(fields[name].get().view(np.uint32),
                                      case.a2[name].view(np.uint32),err_msg=name)
    launch_pxlsm(fields,atm,landusef=landusef,soilcbot=soilcbot,
        dt=case.dt,dtbl=case.dtbl,itimestep=999,curr_secs=9999.,
        smois_init=case.smois_init,soil_nudge=case.soil_nudge,
        modis_veg=case.modis_veg,anal_interval=case.anal_interval,
        domain_shape=(case.ny,case.nx),dzs=case.dzs,device_clock=owner.current)
    yield port_words(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--auxinput9',type=Path,help='also qualify the production Rust reader and CUDA record clock')
    parser.add_argument('--start-time',default='2000-01-24T12:00:00')
    args = parser.parse_args()
    if not wrf_exact.ENABLED:
        parser.error('set GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 before startup')
    import cupy as cp
    cases = sorted(args.captures.glob('case-native-step*.bin'))
    steps = (1, 61, 121)
    expected_names = [f'case-native-step{step:06}.bin' for step in steps]
    if [path.name for path in cases] != expected_names:
        parser.error(f'expected exactly the native captures {expected_names}')
    report = dict(strict=True, compared_words=0, differing_words=0,
                  excluded_words=0, cupy=cp.__version__, numpy=np.__version__,
                  device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  cases=[])
    for source, itimestep in zip(cases, steps):
        name = source.stem.removeprefix('case-')
        expected_path = args.captures / f'out-{name}.bin'
        case = read_case(name, directory=args.captures)
        if case.nsteps != 1 or case.itimestep0 != itimestep:
            parser.error(f'{source}: header does not identify the requested single native call')
        entry = dict(name=name, columns=case.nx * case.ny,
                     itimestep=case.itimestep0, curr_secs=case.curr_secs0,
                     soil_nudge=case.soil_nudge, smois_init=case.smois_init,
                     input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     native_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest(),
                     fields=[])
        if args.auxinput9:
            from datetime import datetime
            results = replay_live_inputs(case,args.auxinput9,datetime.fromisoformat(args.start_time))
            entry['analysis_input_words'] = 6*case.nx*case.ny
            entry['analysis_input_differing_words'] = 0
        else:
            results = run_port(case)
        for step, result in enumerate(results):
            for field in OUTPUT_FIELDS:
                got, want = result[field], case.expected[step][field]
                differing = int(np.count_nonzero(got != want))
                record = dict(step=step, field=field, words=int(want.size),
                              differing_words=differing,
                              gpu_sha256=hashlib.sha256(got.tobytes()).hexdigest(),
                              native_sha256=hashlib.sha256(want.tobytes()).hexdigest())
                if field != 'isltyp':
                    record['max_ulp'] = int(ulp_distance(got, want).max(initial=0))
                    record['native_nonfinite_words'] = int(np.count_nonzero(
                        ~np.isfinite(want.view(np.float32))))
                if differing:
                    index = tuple(np.argwhere(got != want)[0])
                    record['first_index'] = list(map(int, index))
                    record['first_gpu_bits'] = int(got[index])
                    record['first_native_bits'] = int(want[index])
                report['compared_words'] += record['words']
                report['differing_words'] += differing
                entry['fields'].append(record)
        report['cases'].append(entry)
    from woof.certify.kernel_manifest import kernel_manifest
    report['kernel_manifest'] = kernel_manifest()
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))
    return int(report['differing_words'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
