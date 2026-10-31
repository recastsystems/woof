"""Replay complete strict native files through the RRFS Rust I/O sibling.

This is native-field transport evidence. It does not execute a Hex forecast.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
from datetime import datetime
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from woof.hex.output import HistoryField
from woof.hex.restart_rrfs import read_stream, write_stream, write_done_marker, done_marker_enabled
from woof.hex.streams_rrfs import _all_packages, parse_streams


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native_packages(native_root, manifest, streams):
    """Read witnessed physics flags; apply core setup at fork interface:137-258.

    Physics flags are printed by pristine mpas_atmphys_packages.F:137-294.
    This harness reads their actual values, rather than enabling all schemes.
    """
    log = native_root / 'log.atmosphere.0000.out'
    if sha(log) != manifest['outputs'][log.name]:
        raise ValueError('native package log changed')
    text = log.read_text()
    flags = dict(re.findall(r'^\s*([a-zA-Z0-9_]+)\s*[=:]\s*([TF])\s*$', text, re.M))
    active = {name for name, value in flags.items()
              if value == 'T' and name in _all_packages()}
    source = Path(os.environ['RRFS_MPAS_REF'])
    defaults = {}
    for tag in re.findall(r'<nml_option\b[^>]*>',
            (source / 'src/core_atmosphere/Registry.xml').read_text()):
        option = ET.fromstring(tag if tag.endswith('/>') else tag[:-1] + '/>')
        defaults[option.attrib['name']] = option.attrib['default_value']
    settings = dict(defaults, **manifest['resolved_configuration']['resolved_settings'])
    def enabled(key):
        value = settings[key]
        return value is True or str(value).lower() in ('true', '.true.', 't')
    core = {
        'iau': str(settings['config_IAU_option']).lower() != 'off',
        'limited_area': enabled('config_apply_lbcs'),
        'jedi_da': enabled('config_jedi_da'),
        'no_invariant_stream': ('invariant' not in streams or
                               streams['invariant'].interval('input') != 'initial_only'),
        'ugwp_orog_stream': settings['config_gwdo_scheme'] == 'bl_ugwp_gwdo',
        'ugwp_ngw_stream': enabled('config_ngw_scheme'),
        'ugwp_diags_stream': enabled('config_ugwp_diags'),
    }
    active.update(name for name, value in core.items() if value)
    return frozenset(active), dict(log_sha256=sha(log), flags=flags, core=core,
                                  registry_sha256=sha(source / 'src/core_atmosphere/Registry.xml'))


def replay(args):
    native_root = args.native_run.resolve(strict=True)
    manifest_path = native_root / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'mpas.rrfs-native-forecast/v1' or
            manifest.get('fork') != '1d6e370e7b43a332074809f170532a216c2212fb' or
            manifest.get('recipe') != 'gnu-strict' or
            manifest.get('completed') is not True or manifest.get('ranks') != 1):
        raise ValueError('a completed pinned strict fork run at one rank is required')
    streams = parse_streams(native_root / 'streams.atmosphere')
    active_packages, package_receipt = native_packages(native_root, manifest, streams)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    markers = []
    for filename, expected_hash in manifest['outputs'].items():
        if Path(filename).suffix != '.nc':
            continue
        source = native_root / filename
        if sha(source) != expected_hash:
            raise ValueError('native output changed: ' + filename)
        with read_stream(source, scratch=args.output, select=('xtime', 'initial_time')) as clock:
            stamps = clock.variables['xtime'].text
            initial = datetime.strptime(clock.variables['initial_time'].text[0], '%Y-%m-%d_%H:%M:%S')
        first = datetime.strptime(stamps[0], '%Y-%m-%d_%H:%M:%S')
        candidates = [s for s in streams.values() if 'output' in s.options['type'].split(';')
                      and s.filename(first, initial) == filename]
        if len(candidates) != 1:
            raise ValueError('native output must bind to exactly one declared stream: ' + filename)
        stream = candidates[0]
        output = args.output / filename
        for record, stamp in enumerate(stamps):
            with read_stream(source, scratch=args.output, record=record) as state:
                fields = {name: HistoryField(var.values, var.dimensions, var.attrs)
                          for name, var in state.variables.items()}
                mesh = SimpleNamespace(dimensions=state.dimensions, arrays={}, attrs={})
                # Native metadata and words are the authority here. This test
                # deliberately does not claim to recreate forecast state.
                from dataclasses import replace
                effective = replace(stream, options=dict(stream.options,
                    clobber_mode='truncate' if record == 0 else 'append'))
                write_stream(output, effective, mesh, fields, stamp, initial_time=initial,
                    active_packages=active_packages, streams=streams,
                    global_attrs=state.global_attrs, scratch=args.output, record_index=record)
        compared = subprocess.run([str(args.comparator.resolve(strict=True)), str(source), str(output)],
                                  text=True, capture_output=True)
        comparison = json.loads(compared.stdout) if compared.stdout.strip() else {}
        with read_stream(source, scratch=args.output) as native, read_stream(output, scratch=args.output) as encoded:
            attributes_identical = (native.global_attrs == encoded.global_attrs and
                {name: var.attrs for name, var in native.variables.items()} ==
                {name: var.attrs for name, var in encoded.variables.items()})
        passed = (compared.returncode == 0 and comparison.get('all_identical') is True and
                  comparison.get('words', 0) > 0 and comparison.get('differing_words') == 0 and
                  not comparison.get('errors') and attributes_identical)
        rows.append(dict(file=filename, stream=stream.name, records=len(stamps),
            native_sha256=expected_hash, encoded_sha256=sha(output),
            comparison=comparison, comparator_exit=compared.returncode,
            comparator_stderr=compared.stderr, attributes_identical=attributes_identical, passed=passed))
        marker_name = filename + '.done'
        native_marker = native_root / marker_name
        encoded_marker = args.output / marker_name
        expected_marker = done_marker_enabled(stream)
        marker_passed = (native_marker.exists() == encoded_marker.exists() == expected_marker)
        marker_row = dict(file=marker_name, stream=stream.name,
                          expected=expected_marker, passed=marker_passed)
        if expected_marker:
            if sha(native_marker) != manifest['outputs'].get(marker_name):
                raise ValueError('native done marker changed: ' + marker_name)
            native_text = native_marker.read_text()
            native_time = datetime.strptime(native_text, '%Y-%m-%d %H:%M:%S\n')
            encoded_time = datetime.strptime(encoded_marker.read_text(), '%Y-%m-%d %H:%M:%S\n')
            # Wall clocks differ across executions. Compare exact serialization
            # at the witnessed native instant, without claiming equal run times.
            write_done_marker(output, now=native_time)
            exact = native_marker.read_bytes() == encoded_marker.read_bytes()
            marker_row.update(native_sha256=sha(native_marker),
                encoded_sha256=sha(encoded_marker), native_wall_clock=native_time.isoformat(),
                replay_wall_clock=encoded_time.isoformat(),
                same_instant_bytes_identical=exact,
                passed=marker_passed and exact)
        markers.append(marker_row)
    passed = bool(rows) and all(row['passed'] for row in rows) and all(m['passed'] for m in markers)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(dict(schema='rrfs-native-file-transport-replay/v1',
        native_manifest_sha256=sha(manifest_path), run_host=os.uname().nodename,
        process_count=1, all_words_and_attributes_identical=passed,
        active_packages=sorted(active_packages), package_authority=package_receipt,
        scope='Rust transport of actual native forecast fields; no Hex forecast executed',
        whole_model_proof=False, files=rows, done_markers=markers,
        marker_scope='Native presence and exact timestamp serialization at the same witnessed instant'), indent=2) + '\n')
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('native-run', 'output', 'comparator', 'receipt'):
        parser.add_argument('--' + key, type=Path, required=True)
    raise SystemExit(replay(parser.parse_args()))
