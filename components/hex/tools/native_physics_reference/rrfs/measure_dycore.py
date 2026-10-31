#!/usr/bin/env python3
"""Run unchanged strict executables on one byte-identical dry deck and input.

This is a source-version measurement, not a parity gate. Reuses f0 hashing,
completion recognition and the Rust full-stream word comparator; never decodes
or transforms weather arrays in Python. Every argument comes from request data.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_runs import compare, digest
from native_completion import completed


def completion_witness(text):
    """Record the fork's nonfatal interval lookup without accepting I/O errors.

    At 1d6e370e, diagnostics/mpas_convective_diagnostics.F:371 queries the
    diagnostics stream even when no such stream was requested. Framework
    mpas_stream_manager.F:1750 logs this exact message. Preserve it in the
    measurement receipt; every other ERROR/CRITICAL/FATAL still fails the
    shared completion guard. This applies only to source-version measurements.
    """
    message = "ERROR: Stream 'diagnostics' does not exist. Cannot retrieve stream input/output intervals"
    lines = text.splitlines()
    known = sum(line.strip() == message for line in lines)
    remaining = '\n'.join(line for line in lines if line.strip() != message)
    return dict(completed=completed(remaining, 'atmosphere'),
        missing_diagnostics_interval_messages=known,
        other_error_messages=[line for line in lines
            if re.match(r'^\s*(?:ERROR|CRITICAL|FATAL)(?::|\b)', line) and line.strip() != message],
        measurement_only=True)


def run(build, deck, inputs, out, mpi, ranks):
    if ranks < 1:
        raise ValueError('positive MPI rank count required')
    build, deck = Path(build).resolve(strict=True), Path(deck).resolve(strict=True)
    manifest = json.loads((build / 'MANIFEST.json').read_text())
    audit = json.loads((build / 'flag-audit.json').read_text())
    binary = build / 'atmosphere_model'
    if (not audit['all_ok'] or audit['recipe'] != 'gnu-strict'
            or manifest['core'] != 'atmosphere' or manifest['recipe'] != 'gnu-strict'
            or digest(binary) != manifest['binary_sha256']
            or digest(build / 'flag-audit.json') != manifest.get('flags_sha256', manifest.get('files', {}).get('flag-audit.json'))
            or digest(build / 'toolchain.txt') != manifest['toolchain_sha256']):
        raise ValueError('dycore measurement requires a verified strict atmosphere build')
    indexed = json.loads((deck / 'MANIFEST.json').read_text())
    for name, expected in indexed['files'].items():
        if digest(deck / name) != expected:
            raise ValueError('common deck bytes changed: ' + name)
    for name, row in inputs.items():
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError('common weather input bytes changed: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    for name in indexed['files']:
        shutil.copy2(deck / name, out / name)
    for name, row in inputs.items():
        if Path(name).name != name or (out / name).exists():
            raise ValueError('unsafe common input filename')
        (out / name).symlink_to(Path(row['path']).resolve(strict=True))
    argv = [str(Path(mpi).resolve(strict=True)), '-np', str(ranks), str(binary)]
    with (out / 'native.stdout.log').open('w') as log:
        status = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')).returncode
    logfile = out / 'log.atmosphere.0000.out'
    # Scratch retirement takes the same lock. Persist native exit independently
    # before output hashing so an orchestration failure cannot erase its status.
    with (out/'native-output-receipt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        native_exit = dict(schema='mpas.native-process-exit/v1', exit_status=status,
            argv=argv, host=os.uname().nodename, binary_sha256=digest(binary),
            native_log_sha256=digest(logfile) if logfile.is_file() else None)
        (out/'NATIVE-EXIT.json').write_text(json.dumps(native_exit,indent=2)+'\n')
        outputs = {p.name: digest(p) for p in sorted(out.glob('*.nc')) if not p.is_symlink()}
    witness = completion_witness(logfile.read_text()) if logfile.is_file() else None
    success = status == 0 and bool(outputs) and witness is not None and witness['completed']
    receipt = dict(schema='mpas.rrfs-dry-dycore-run/v1', build=str(build),
        source_pin=manifest.get('pin_set', {}).get('fork', manifest.get('upstream')),
        binary_sha256=digest(binary), audit_sha256=digest(build / 'flag-audit.json'),
        toolchain_sha256=digest(build / 'toolchain.txt'), common_deck_sha256=digest(deck / 'MANIFEST.json'),
        inputs={name: row['sha256'] for name, row in inputs.items()}, outputs=outputs,
        ranks=ranks, argv=argv, host=os.uname().nodename, cpu_affinity=sorted(os.sched_getaffinity(0)),
        exit_status=status, completed=success, completion_witness=witness,
        native_exit_receipt_sha256=digest(out/'NATIVE-EXIT.json'),
        native_log_sha256=digest(logfile) if logfile.is_file() else None)
    (out / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def measurement(reference, other, comparator, start, dt):
    if dt <= 0:
        raise ValueError('positive integration timestep required')
    manifests = [json.loads((p / 'MANIFEST.json').read_text()) for p in (reference, other)]
    if any(not m['completed'] for m in manifests):
        raise ValueError('dycore comparison requires two completed runs')
    for key in ('inputs', 'common_deck_sha256', 'ranks', 'host'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('common dycore run contract differs: ' + key)
    for path, manifest in zip((reference, other), manifests):
        if {p.name:digest(p) for p in path.glob('*.nc') if not p.is_symlink()} != manifest['outputs']:
            raise ValueError('dycore output bytes changed after native execution')
    result = compare(reference, other, comparator, '*.nc')
    first = None
    for name, entry in result['files'].items():
        timestamp = datetime.strptime(name.removeprefix('tracer.').removesuffix('.nc'), '%Y-%m-%d_%H.%M.%S')
        elapsed = (timestamp - datetime.strptime(start, '%Y-%m-%d_%H:%M:%S')).total_seconds()
        if elapsed < 0 or elapsed % dt:
            raise ValueError('dycore output time is not an integration step')
        for field in entry['fields']:
            if field['differing_words']:
                first = dict(file=name, field=field['field'], word=field['first_difference'],
                             step=int(elapsed / dt), elapsed_seconds=elapsed)
                break
        if first is not None:
            break
    result.update(schema='mpas.rrfs-dry-dycore-measurement/v1', first_difference=first,
        measurement_only=True, ranks=manifests[0]['ranks'], host=manifests[0]['host'],
        source_pins=[m['source_pin'] for m in manifests],
        run_manifest_sha256=[digest(p / 'MANIFEST.json') for p in (reference, other)],
        comparator_sha256=digest(comparator), dt=dt, start=start)
    return result


def recover_measurement(run_directory, out, retired_receipt, output_names):
    """Recover a completed measurement's wrapper grade without editing evidence.

    Keep the original receipt, exact source log and recorded retired hashes. Only
    explicit retained timestamps are linked into the measurement directory; all
    fields at those timestamps remain inside the Rust comparison. This never
    promotes a failed native process or a forecast to reference acceptance.
    """
    original_path=run_directory/'MANIFEST.json'
    original=json.loads(original_path.read_text())
    log=run_directory/'log.atmosphere.0000.out'
    witness=completion_witness(log.read_text())
    if original['exit_status'] != 0 or not witness['completed']:
        raise ValueError('native measurement did not finish or has an unrecognized error')
    if 'native_exit_receipt_sha256' in original:
        exit_path=run_directory/'NATIVE-EXIT.json'
        native_exit=json.loads(exit_path.read_text())
        if (digest(exit_path)!=original['native_exit_receipt_sha256']
                or native_exit['exit_status']!=0
                or native_exit['argv']!=original['argv']
                or native_exit['binary_sha256']!=original['binary_sha256']
                or native_exit['native_log_sha256']!=digest(log)):
            raise ValueError('native process exit witness differs')
    build=Path(original['build']); native=json.loads((build/'MANIFEST.json').read_text())
    audit=json.loads((build/'flag-audit.json').read_text())
    if (not audit['all_ok'] or audit['recipe']!='gnu-strict'
            or native['core']!='atmosphere' or native['recipe']!='gnu-strict'
            or native.get('pin_set',{}).get('fork',native.get('upstream'))!=original['source_pin']
            or digest(build/'atmosphere_model')!=original['binary_sha256']
            or original['binary_sha256']!=native['binary_sha256']
            or digest(build/'flag-audit.json')!=original['audit_sha256']
            or digest(build/'toolchain.txt')!=original['toolchain_sha256']):
        raise ValueError('measurement native source/build receipt changed')
    retired=json.loads(retired_receipt.read_text())['retired']
    for name, expected in original['outputs'].items():
        path=run_directory/name
        if path.is_file():
            if digest(path)!=expected:raise ValueError('native output changed: '+name)
        elif retired.get(run_directory.name+'/'+name,{}).get('sha256')!=expected:
            raise ValueError('missing output lacks matching retired-byte witness: '+name)
    if not output_names or len(set(output_names))!=len(output_names):
        raise ValueError('explicit unique retained output timestamps required')
    outputs={}
    for name in output_names:
        if Path(name).name!=name or name not in original['outputs']:
            raise ValueError('retained output is absent from original native receipt')
        path=run_directory/name
        if path.is_symlink() or digest(path)!=original['outputs'][name]:
            raise ValueError('retained native output bytes changed')
        outputs[name]=original['outputs'][name]
    out.mkdir(parents=True,exist_ok=False)
    for name in output_names:os.link(run_directory/name,out/name)
    shutil.copyfile(original_path,out/'ORIGINAL-MANIFEST.json')
    shutil.copyfile(log,out/log.name)
    if 'native_exit_receipt_sha256' in original:
        shutil.copyfile(run_directory/'NATIVE-EXIT.json',out/'NATIVE-EXIT.json')
    receipt=dict(original, outputs=outputs, completed=True, completion_witness=witness,
        measurement_only=True, native_log_sha256=digest(log),
        original_manifest_sha256=digest(original_path),
        retired_output_receipt_sha256=digest(retired_receipt),
        retained_output_scope='Explicit timestamps, every field; source-version measurement only, not full-output identity')
    (out/'MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--request', type=Path, required=True)
    execute.add_argument('--build', type=Path, required=True)
    execute.add_argument('--out', type=Path, required=True)
    diff = sub.add_parser('compare')
    for key in ('reference', 'other', 'comparator', 'out'):
        diff.add_argument('--' + key, type=Path, required=True)
    diff.add_argument('--start', required=True)
    diff.add_argument('--dt', type=float, required=True)
    recover=sub.add_parser('recover-measurement')
    for key in ('run-directory','out','retired-receipt','output-names'):
        recover.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'run':
        request = json.loads(args.request.read_text())
        receipt = run(args.build, request['deck'], request['inputs'], args.out, request['mpi'], request['ranks'])
        print(json.dumps({key:receipt[key] for key in ('source_pin', 'completed', 'exit_status')}))
        sys.exit(0 if receipt['completed'] else 1)
    elif args.action == 'recover-measurement':
        result=recover_measurement(args.run_directory,args.out,args.retired_receipt,json.loads(args.output_names.read_text()))
        print(json.dumps({key:result[key] for key in ('completed','measurement_only','completion_witness')}))
    else:
        result = measurement(args.reference, args.other, args.comparator, args.start, args.dt)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key:result[key] for key in ('all_identical', 'first_difference')}))
