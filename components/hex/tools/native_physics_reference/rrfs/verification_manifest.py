#!/usr/bin/env python3
"""Bind a sealed RRFS forecast to the existing verify_mpas native-run contract.

The fork's original publication remains unchanged. Rust describes every stored
NetCDF record; the existing verifier binds its times to the native stream clock.
This creates comparison metadata, not a forecast, replay or acceptance claim.
"""
import argparse
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from woof.hex.verify_mpas import stream_filename_times
from rrfs.verify_diagnostic_contract import publication
from rrfs.ruled_reference import input_authorities, validate_authority
from publish_cache import sha


def generate(request, comparator):
    pinned = request['forecast']
    source = Path(pinned['path'])
    pub = publication(source/'PUBLICATION.json', pinned['publication_sha256'])
    native = json.loads((source/'MANIFEST.json').read_text())
    if native.get('schema') != 'mpas.rrfs-native-forecast/v1' or (
            not native['completed'] or native['exit_status'] != 0):
        raise ValueError('verification requires a completed native RRFS forecast')
    if not any('Finished running the atmosphere core' in (source/n).read_text()
               for n in pub['files'] if n.startswith('log.atmosphere.') and n.endswith('.out')):
        raise ValueError('native forecast final banner missing')
    authorities = input_authorities(native['case'])
    if authorities != native.get('init_derivative_authorities', []):
        raise ValueError('native initializer authority differs')
    if native.get('source_correction'):
        validate_authority(native['source_correction'], native['binary_sha256'])
    inputs, auxiliary = request['inputs'], request['auxiliary_inputs']
    if set(inputs) != {'namelist','streams','init'}:
        raise ValueError('verification requires exactly namelist, streams and init')
    for key, ref in inputs.items():
        path = Path(ref['path'])
        expected = native['inputs'].get(path.name) if key == 'init' else (
            native['resolved_configuration']['files'].get(key+'.atmosphere'))
        if sha(path) != ref['sha256'] or ref['sha256'] != expected:
            raise ValueError('verification primary input differs: '+key)
    for name, ref in auxiliary.items():
        expected = native['inputs'].get(name) or native['resolved_configuration']['files'].get(name)
        if sha(Path(ref['path'])) != ref['sha256'] or ref['sha256'] != expected:
            raise ValueError('verification auxiliary input differs: '+name)
    streams = ET.parse(inputs['streams']['path']).getroot()
    if any(f.get('name') not in auxiliary for f in streams.iter('file')):
        raise ValueError('verification stream list missing from auxiliary inputs')
    patterns = []
    for stream in streams:
        if 'output' not in stream.get('type',''):
            continue
        pattern = re.escape(stream.get('filename_template',''))
        for token, width in [('Y',4),('M',2),('D',2),('h',2),('m',2),('s',2)]:
            pattern = pattern.replace(re.escape('$'+token), '(?P<'+token+'>\\d{'+str(width)+'})')
        patterns.append((stream,re.compile(pattern)))
    start = datetime.strptime(native['runtime']['start'],'%Y-%m-%d_%H:%M:%S')
    dt = Decimal(str(native['runtime']['dt']))
    if dt <= 0:
        raise ValueError('native timestep must be positive')
    outputs = []
    for name, digest in native['outputs'].items():
        if not name.endswith('.nc'):
            continue
        matches = [(s,p.fullmatch(name)) for s,p in patterns if p.fullmatch(name)]
        if len(matches) != 1 or pub['files'].get(name) != digest:
            raise ValueError('native output lacks one sealed stream binding: '+name)
        stream, match = matches[0]
        result = subprocess.run([str(comparator),'--describe',str(source/name)],
                                text=True,capture_output=True,check=True)
        metadata = json.loads(result.stdout)
        filename_clock = metadata['times'] is None
        times = stream_filename_times(stream,match.groupdict(),metadata['records']) if filename_clock else metadata['times']
        steps = []
        for stamp in times:
            elapsed = datetime.strptime(stamp,'%Y-%m-%d_%H:%M:%S')-start
            seconds = Decimal(elapsed.days*86400+elapsed.seconds)
            if seconds < 0 or seconds % dt:
                raise ValueError('native record lies off timestep clock')
            steps.append(int(seconds/dt))
        if not steps or steps != sorted(set(steps)):
            raise ValueError('native record times absent, repeated or reversed')
        output = dict(stream=stream.get('name'),path=name,sha256=digest,times=times,steps=steps,
                      records=metadata['records'],time_source='STREAM-FILENAME+RECORD-COUNT' if filename_clock else metadata['time_source'])
        if filename_clock:
            output['clock_witness'] = {'filename_template': stream.get('filename_template')}
        outputs.append(output)
    if not outputs:
        raise ValueError('native forecast has no NetCDF output')
    return dict(schema='mpas-native-run-v1',pin_set=native['pin_set_hash'],process_count=native['ranks'],
        inputs={k:v['sha256'] for k,v in inputs.items()},
        auxiliary_inputs={k:v['sha256'] for k,v in auxiliary.items()},complete=True,
        output_root=str(source),outputs=outputs,source_publication=pinned,
        cache_manifest_sha256=pub['files']['MANIFEST.json'],native_input_inventory=native['inputs'],
        init_derivative_authorities=authorities,reference_labels=native['reference_labels'],
        source_correction=native.get('source_correction'),comparator_sha256=sha(comparator))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--comparator',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise ValueError('verification metadata destination already exists')
    result=generate(json.loads(a.request.read_text()),a.comparator)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(native_manifest=str(a.out),sha256=sha(a.out),outputs=len(result['outputs']))))
