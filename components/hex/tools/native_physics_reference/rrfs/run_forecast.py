#!/usr/bin/env python3
"""Run an actual fork deck with nested stream lists and explicit input mappings.

Extends f0 run_forecast.py. Success and native failure both receive a receipt;
every emitted file, including done markers, is hashed without field transforms.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_forecast import sha


def table_file(path):
    return path.is_file() and not os.access(path, os.X_OK) and (path.suffix.lower() in ('.tbl', '.dbl', '.bin', '.dat')
                              or path.name.endswith(('_DATA', '_tempo_v3'))
                              or path.name == 'ugwp_limb_tau.nc')


def stage_inputs(files, materialized, out, ingress):
    """Keep native-writable case inputs private without changing their bytes."""
    for name, source in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError('input/deck collision ' + name)
        if name in materialized:
            shutil.copy2(source, target)
            # Immutable source permissions must not prevent native DA output.
            target.chmod(target.stat().st_mode | 0o200)
            if sha(target) != ingress[name]:
                raise ValueError('case bytes changed during input copy: ' + name)
        else:
            target.symlink_to(source)


def forecast(build, deck, case, tables, out, mpi, ranks):
    if ranks < 1:
        raise ValueError('positive MPI rank count required')
    build, deck = build.resolve(strict=True), deck.resolve(strict=True)
    audit = json.loads((build / 'flag-audit.json').read_text())
    if not audit['all_ok'] or audit['core'] != 'atmosphere':
        raise ValueError('forecast requires an audited native atmosphere executable')
    pins = json.loads((build / 'SOURCE-PINS.json').read_text())
    manifest = json.loads((build / 'MANIFEST.json').read_text())
    binary = build / 'atmosphere_model'
    from rrfs.ruled_reference import input_authorities, validate_authority
    init_authorities = input_authorities(case)
    corrected = manifest.get('source_correction')
    if corrected:
        validate_authority(corrected,sha(binary))
        if corrected['kind'] != 'gfl' or pins.get('source_correction') != corrected:
            raise ValueError('atmosphere source-corrected authority differs')
    if manifest['binary_sha256'] != sha(binary) or manifest['flags_sha256'] != sha(build / 'flag-audit.json'):
        raise ValueError('binary or audit differs from published build receipt')
    index_path = deck.parent / 'INDEX.json'
    index = json.loads(index_path.read_text())
    resolved = next(row for row in index['decks'] if row['deck'] == deck.name)
    for name, digest in resolved['files'].items():
        if sha(deck / name) != digest:
            raise ValueError('deck differs from generated index: ' + name)
    files = {}
    for path in build.rglob('*'):
        if table_file(path):
            if path.name in files and sha(path) != sha(files[path.name]):
                raise ValueError('ambiguous built table ' + path.name)
            files[path.name] = path.resolve()
    table_overrides = {}
    for directory in tables:
        for path in sorted(directory.iterdir()):
            if not table_file(path):
                continue
            if path.name in files and sha(files[path.name]) != sha(path):
                table_overrides[path.name] = dict(previous_sha256=sha(files[path.name]), selected_sha256=sha(path))
            files[path.name] = path.resolve()
    for name, source in case['files'].items():
        if name in files:
            raise ValueError('case/table collision ' + name)
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe input filename')
        files[name] = Path(source).resolve(strict=True)
    # Native DA/restart streams can open ingress files for writing. Case bytes
    # therefore belong to this run; table files remain read-only linked inputs.
    materialized = set(case['files'])
    archive_module = None
    if case.get('readonly_inputs') or case.get('output_archive'):
        spec = importlib.util.spec_from_file_location('rrfs_archive_outputs', Path(__file__).with_name('archive_outputs.py'))
        archive_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(archive_module)
    if case.get('readonly_inputs'):
        selected = set(case['readonly_inputs'])
        if not selected.issubset(materialized):
            raise ValueError('read-only selection is not a case input')
        materialized -= archive_module.readonly_inputs(ET.parse(deck/'streams.atmosphere').getroot(), selected)
    ingress = {name: sha(source) for name, source in files.items()}
    out.mkdir(parents=True, exist_ok=False)
    for name in resolved['files']:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(deck / name, target)
    stage_inputs(files, materialized, out, ingress)
    staged_stat = {name: ((out / name).stat().st_ino, (out / name).stat().st_mtime_ns)
                   for name in materialized}
    argv = [str(mpi.resolve(strict=True)), '-np', str(ranks), str(binary)]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    archiver = None
    if case.get('output_archive'):
        request = case['output_archive']
        # This is a storage choice, not an alternate scientific runtime.
        spec = importlib.util.spec_from_file_location('rrfs_archive_deck', Path(__file__).with_name('generate_decks.py'))
        generator = importlib.util.module_from_spec(spec); spec.loader.exec_module(generator)
        groups = generator.read_namelist((out/'namelist.atmosphere').read_text())
        flat = {k:v for values in groups.values() for k,v in values.items()}
        runtime = dict(start=flat['config_start_time'].strip("'\""),
                       duration=flat['config_run_duration'].strip("'\""))
        archiver = archive_module.ArchiveOutputs(out, runtime, request, materialized)
    with (out / 'native.stdout.log').open('w') as log:
        if archiver is None:
            status = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        else:
            process = subprocess.Popen(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                while process.poll() is None:
                    archiver.collect()
                    time.sleep(2)
                status = process.returncode
                archiver.collect(finished=True)
            except BaseException:
                process.terminate(); process.wait()
                raise
    logfile = out / 'log.atmosphere.0000.out'
    finished = logfile.is_file() and 'Finished running the atmosphere core' in logfile.read_text()
    local_after = {name: sha(out / name) for name in materialized}
    native_wrote = {name: (local_after[name] != ingress[name] or
        ((out / name).stat().st_ino, (out / name).stat().st_mtime_ns) != staged_stat[name])
        for name in materialized}
    source_after = {name: sha(path) for name, path in files.items()}
    outputs = {}
    if archiver:
        outputs.update({name: row['sha256'] for name,row in archiver.files.items()})
    for path in sorted(out.rglob('*')):
        name = str(path.relative_to(out))
        if not path.is_file() or path.is_symlink() or name in resolved['files']:
            continue
        if name in materialized and not native_wrote[name]:
            continue
        outputs[name] = sha(path)
    input_witness = {name: dict(ingress_sha256=ingress[name],
        local_after_sha256=local_after[name], source_after_sha256=source_after[name],
        native_changed=local_after[name] != ingress[name], native_wrote=native_wrote[name])
        for name in sorted(materialized)}
    sources_unchanged = source_after == ingress
    receipt = dict(schema='mpas.rrfs-native-forecast/v1', pin_set_hash=manifest['pin_set_hash'],
                   fork=pins['fork'], recipe=audit['recipe'], binary_sha256=sha(binary),
                   toolchain_sha256=sha(build / 'toolchain.txt'),
                   argv=argv, ranks=ranks, host=os.uname().nodename, cpu_affinity=sorted(os.sched_getaffinity(0)),
                   deck=deck.name, deck_index_sha256=sha(index_path), resolved_configuration=resolved,
                   runtime=index['runtime'], case=case, inputs=ingress,
                   materialized_inputs=input_witness, source_inputs_after=source_after,
                   source_inputs_unchanged=sources_unchanged,
                   selected_table_overrides=table_overrides, outputs=outputs,
                   exit_status=status, completed=bool(status == 0 and finished and sources_unchanged))
    (out / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    receipt['init_derivative_authorities'] = init_authorities
    if corrected:
        receipt['source_correction'] = corrected
    receipt['reference_labels'] = [a['label'] for a in init_authorities] + ([corrected['label']] if corrected else [])
    (out / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('build', 'deck', 'case', 'out', 'mpi'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--tables', type=Path, action='append', default=[])
    p.add_argument('--ranks', type=int, required=True)
    a = p.parse_args()
    receipt = forecast(a.build, a.deck, json.loads(a.case.read_text()), a.tables, a.out, a.mpi, a.ranks)
    print(json.dumps({key: receipt[key] for key in ('completed', 'exit_status', 'recipe', 'deck', 'ranks')}))
    sys.exit(0 if receipt['completed'] else 1)
