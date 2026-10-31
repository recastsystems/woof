#!/usr/bin/env python3
"""Run fork static/invariant/case7/case9 through the pinned workflow templates.

Extends f0's native case orchestration; all decode and interpolation stays in
ungrib and init_atmosphere_model. Explicit rows select stages, inputs and clocks.
"""
import argparse
import json
from pathlib import Path
import shutil
import os
import re
import subprocess
from string import Template
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cache_cases import link, run, sha, stamp
from publish_cache import publish


def decode(row, out, ungrib, vtable):
    out.mkdir(parents=True, exist_ok=False)
    link(vtable, out / 'Vtable')
    for i, parts in enumerate(row['gribs']):
        suffix = ''.join(chr(65 + (i // (26 ** p)) % 26) for p in (2, 1, 0))
        # Workflow exrrfs_ungrib.sh:60-80 concatenates pgrb2 and pgrb2b.
        with (out / ('GRIBFILE.' + suffix)).open('wb') as target:
            for source in parts:
                with Path(source).open('rb') as stream:
                    shutil.copyfileobj(stream, target, length=1 << 20)
    (out / 'namelist.wps').write_text('&share\n wrf_core = \'ARW\'\n max_dom = 1\n'
        f" start_date = '{row['start']}'\n end_date = '{row['stop']}'\n interval_seconds = {row['interval_seconds']}\n/\n"
        "&ungrib\n out_format = 'WPS'\n prefix = 'GFS'\n/\n")
    run([str(ungrib)], out, 'ungrib.log')
    expected = int((stamp(row['stop']) - stamp(row['start'])).total_seconds() / row['interval_seconds']) + 1
    files = sorted(out.glob('GFS:*'))
    if len(files) != expected:
        raise ValueError('ungrib valid-time count differs from requested interval')
    (out / 'MANIFEST.json').write_text(json.dumps(dict(row=row, ungrib_sha256=sha(ungrib),
        vtable_sha256=sha(vtable), inputs={str(Path(p).resolve()): sha(Path(p)) for parts in row['gribs'] for p in parts},
        intermediates={p.name: sha(p) for p in files}), indent=2) + '\n')
    # Retain the original downloaded inputs and hashed intermediates; duplicate
    # concatenated GRIBFILE bytes are disposable immediately after decode.
    for path in out.glob('GRIBFILE.*'):
        path.unlink()
    return out


def configuration(row, workflow):
    """Render the exact workflow configuration for execution and cache reuse."""
    variables = row['variables']
    namelist_template = workflow / row['namelist_template']
    templates = [namelist_template, *[workflow / name for name in row.get('namelist_companion_templates', [])]]
    text = Template('\n'.join(path.read_text() for path in templates)).substitute(variables)
    for key, value in row.get('namelist_replacements', {}).items():
        text = text.replace('@' + key + '@', str(value))
    for replacement in row.get('text_replacements', []):
        if text.count(replacement['old']) != 1 or not replacement.get('authority'):
            raise ValueError('configuration replacement lacks a unique source/authority')
        text = text.replace(replacement['old'], replacement['new'])
    if '@' in text or '${' in text:
        raise ValueError('unresolved init namelist template')
    streams_template = workflow / row['streams_template']
    streams = streams_template.read_text()
    for key, value in row.get('stream_replacements', {}).items():
        streams = streams.replace('@' + key + '@', str(value))
    if '@' in streams:
        raise ValueError('unresolved init stream template')
    ET.fromstring(streams)
    return {'namelist.init_atmosphere': text, 'streams.init_atmosphere': streams}


def verify_intermediates(directory):
    receipt = json.loads((directory / 'MANIFEST.json').read_text())
    if not receipt['intermediates'] or any(sha(directory / name) != digest
                                         for name, digest in receipt['intermediates'].items()):
        raise ValueError('native decoded intermediate bytes changed')


def decoded_directory(row, rendered):
    """Require native weather inputs unless the deck explicitly disables them."""
    if row.get('decoded'):
        directory = Path(row['decoded'])
        verify_intermediates(directory)
        return directory
    text = rendered['namelist.init_atmosphere']
    # mpas_init_atm_cases.F:4468-4478 at 1d6e370e reads terrain from
    # the native intermediate even when meteorological interpolation is off.
    for option in ('config_met_interp', 'config_input_sst', 'config_blend_bdy_terrain'):
        values = re.findall(r'^\s*' + option + r'\s*=\s*(\.?[A-Za-z]+\.?)(?=\s|,|$)', text, re.M | re.I)
        if len(values) != 1 or values[0].strip('.').lower() != 'false':
            raise ValueError('missing decoded inputs without an explicit disabled weather stage: ' + option)
    return None


def prepare(row, out, workflow, build, mpi):
    build = build.resolve(strict=True)
    audit = json.loads((build / 'flag-audit.json').read_text())
    source_pins = json.loads((build / 'SOURCE-PINS.json').read_text())
    approved = source_pins.get('source_correction')
    if approved:
        from rrfs.ruled_reference import validate_authority
        validate_authority(approved,sha(build/'init_atmosphere_model'))
        if approved['kind'] != 'init' or row['variables']['init_case'] not in (7,9):
            raise ValueError('approved initializer is limited to canonical case 7/9')
    if not audit['all_ok'] or audit['recipe'] != 'gnu-strict' or audit['core'] != 'init_atmosphere':
        raise ValueError('cases require audited strict init')
    rendered = configuration(row, workflow)
    if approved:
        from rrfs.ruled_reference import validate_init_configuration
        validate_init_configuration(rendered['namelist.init_atmosphere'])
    decoded = decoded_directory(row, rendered)
    geography_manifest = Path(row['geography_manifest']) if 'geography_manifest' in row else None
    if geography_manifest is not None and sha(geography_manifest) != row['geography_source_manifest_sha256']:
        raise ValueError('native geography source manifest changed')
    out.mkdir(parents=True, exist_ok=False)
    if geography_manifest is not None:
        shutil.copy2(geography_manifest, out / 'GEOGRAPHY-MANIFEST.json')
    for name, source in row['inputs'].items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        link(source, target)
    for intermediate in decoded.glob('GFS:*') if decoded is not None else ():
        link(intermediate, out / intermediate.name)
    for name, source in row.get('runtime_files', {}).items():
        link(source, out / name)
    for name, content in rendered.items():
        (out / name).write_text(content)
    binary = build / 'init_atmosphere_model'
    argv = [str(mpi), '-np', str(row['ranks']), str(binary)]
    with (out / 'native.stdout.log').open('w') as native_log:
        status = subprocess.run(argv, cwd=out, stdout=native_log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='1')).returncode
    log = out / 'log.init_atmosphere.0000.out'
    finished = log.is_file() and 'Finished running the init_atmosphere core' in log.read_text()
    outputs = sorted(p for p in out.glob('*.nc') if not p.is_symlink())
    complete = status == 0 and finished and bool(outputs) and set(row['expected_outputs']).issubset({p.name for p in outputs})
    receipt = dict(schema='mpas.rrfs-native-input/v1', row=row, binary_sha256=sha(binary),
                   source_pins=source_pins, host=os.uname().nodename,
                   audit_sha256=sha(build / 'flag-audit.json'),
                   workflow_templates=template_hashes(row, workflow),
                   configuration={name: sha(out / name) for name in rendered},
                   inputs={name: sha(Path(source)) for name, source in row['inputs'].items()},
                   runtime_files={name: sha(Path(source)) for name, source in row.get('runtime_files', {}).items()},
                   decoded_manifest_sha256=sha(decoded / 'MANIFEST.json') if decoded is not None else None,
                   outputs={p.name: sha(p) for p in outputs}, argv=argv, ranks=row['ranks'],
                   cpu_affinity=sorted(os.sched_getaffinity(0)), exit_status=status, completed=bool(complete))
    (out / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if approved:
        receipt['source_correction'] = approved
        receipt['reference_label'] = approved['label']
        (out / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not complete:
        raise ValueError('fork initialization failed; receipt: ' + str(out / 'MANIFEST.json'))
    return receipt


def template_hashes(row, workflow):
    return {name: sha(workflow / name) for name in
            [row['namelist_template'], row['streams_template'], *row.get('namelist_companion_templates', [])]}


def verified_case(row, out, workflow, build, mpi):
    """Reuse only a completed case with unchanged native inputs and configuration."""
    if not out.exists():
        return prepare(row, out, workflow, build, mpi)
    receipt = json.loads((out / 'MANIFEST.json').read_text())
    if not receipt['completed'] or receipt['exit_status']:
        raise ValueError('existing native case did not complete; use a new attempt directory')
    previous = receipt['row']
    keys = ('variables', 'ranks', 'namelist_template', 'streams_template', 'namelist_companion_templates', 'namelist_replacements',
            'text_replacements', 'stream_replacements', 'expected_outputs', 'geography_source_manifest_sha256')
    if any(previous.get(key) != row.get(key) for key in keys):
        raise ValueError('completed native case configuration differs')
    rendered = configuration(row, workflow)
    decoded = decoded_directory(row, rendered)
    if any((out / name).read_text() != content for name, content in rendered.items()):
        raise ValueError('completed native case deck bytes changed')
    expected = dict(binary_sha256=sha(build / 'init_atmosphere_model'),
        audit_sha256=sha(build / 'flag-audit.json'),
        workflow_templates=template_hashes(row, workflow),
        inputs={name:sha(Path(source)) for name, source in row['inputs'].items()},
        runtime_files={name:sha(Path(source)) for name, source in row.get('runtime_files', {}).items()},
        decoded_manifest_sha256=sha(decoded / 'MANIFEST.json') if decoded is not None else None)
    if 'source_pins' in receipt:
        expected['source_pins'] = json.loads((build / 'SOURCE-PINS.json').read_text())
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError('completed native case input or toolchain bytes differ')
    if any(sha(out / name) != value for name, value in receipt['outputs'].items()):
        raise ValueError('completed native case output bytes changed')
    if 'geography_manifest' in row and (
            sha(Path(row['geography_manifest'])) != row['geography_source_manifest_sha256']
            or sha(out / 'GEOGRAPHY-MANIFEST.json') != row['geography_source_manifest_sha256']):
        raise ValueError('completed native case geography source manifest changed')
    return receipt


def publish_case(out, destination, ready, description):
    """Publish native outputs and receipts; input bytes remain in their caches."""
    receipt = json.loads((out / 'MANIFEST.json').read_text())
    from rrfs.ruled_reference import validate_init_case
    approved = validate_init_case(receipt)
    if approved and approved['label'] not in description:
        raise ValueError('corrected initializer case publication must name its ruling label')
    if approved:
        from rrfs.ruled_reference import validate_init_configuration
        validate_init_configuration((out / 'namelist.init_atmosphere').read_text())
    if receipt.get('source_pins', {}).get('diagnostic'):
        raise ValueError('diagnostic initializer outputs cannot become the primary reference')
    if not receipt['completed'] or receipt['exit_status']:
        raise ValueError('cannot publish an incomplete native case')
    for name, expected in receipt['outputs'].items():
        if sha(out / name) != expected:
            raise ValueError('native case output differs from receipt')
    if not receipt.get('configuration') or any(sha(out / name) != digest
                                             for name, digest in receipt['configuration'].items()):
        raise ValueError('native case configuration differs from receipt')
    geography = receipt['row'].get('geography_source_manifest_sha256')
    if geography is not None and sha(out / 'GEOGRAPHY-MANIFEST.json') != geography:
        raise ValueError('native case geography source manifest differs')
    if destination.exists():
        publication = json.loads((destination / 'PUBLICATION.json').read_text())
        if sha(out / 'MANIFEST.json') != sha(destination / 'MANIFEST.json'):
            raise ValueError('existing immutable case publication differs')
        if any(sha(destination / name) != value for name, value in publication['files'].items()):
            raise ValueError('immutable case publication bytes changed')
        return dict(path=str(destination), publication_sha256=sha(destination / 'PUBLICATION.json'),
                    files=len(publication['files']))
    patterns = [*receipt['outputs'], 'MANIFEST.json', '*.log', '*.out', 'namelist.*', 'streams.*']
    if geography is not None:
        patterns.append('GEOGRAPHY-MANIFEST.json')
    # Preserve all original bytes while avoiding dense allocation of native
    # zero-filled work fields. Selection/sealing remain f0's implementation.
    import importlib.util
    spec = importlib.util.spec_from_file_location('rrfs_case_sparse_publication',
                                                  Path(__file__).with_name('publish_sparse.py'))
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    return storage.publish(out, destination, patterns, ready, description)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('index', type=Path)
    p.add_argument('--work', type=Path, required=True)
    for key in ('workflow', 'build', 'mpi', 'ready'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--ungrib', type=Path)
    p.add_argument('--vtable', type=Path)
    a = p.parse_args()
    index = json.loads(a.index.read_text())
    for row in index.get('decode', []):
        if not a.ungrib or not a.vtable:
            raise ValueError('decode requires explicit native ungrib and Vtable')
        print(decode(row, a.work / row['name'], a.ungrib.resolve(), a.vtable.resolve()), flush=True)
    for row in index.get('cases', []):
        out = a.work / row['name']
        verified_case(row, out, a.workflow.resolve(), a.build.resolve(), a.mpi.resolve())
        if 'publish' in row:
            print(json.dumps(publish_case(out, Path(row['publish']), a.ready, row['description'])), flush=True)
