#!/usr/bin/env python
"""Collect a native UGWP driver boundary from a published initialization.

This is orchestration only: MPAS reads its own netCDF and runs the whole native
model. No weather values are decoded or generated in Python. The capture build
must be an already audited f1 build; the native reference is never rebuilt.
MPAS authority: mpas_atmphys_driver_gwdo.F:779-829.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

SITE = 'gwdo_ugwp__driver_gwdo__L781'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'deck', 'case', 'out', 'tables', 'runner', 'inventory', 'pins', 'words', 'column-door'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mpi', required=True)
    p.add_argument('--duration', required=True)
    p.add_argument('--set', action='append', default=[], metavar='CONFIG_SCHEME=VALUE',
                   help='explicit native companion option; recorded in the capture receipt')
    p.add_argument('--time-metadata', type=Path, help='explicit Rust metadata preparation for a native static stream lacking xtime')
    a = p.parse_args()
    root = a.out.resolve(); root.mkdir(parents=True, exist_ok=False)
    deck = root/'deck'; shutil.copytree(a.deck, deck, copy_function=shutil.copyfile)
    deck.chmod(0o755)
    namelist = deck/'namelist.atmosphere'
    content, count = re.subn(r"config_run_duration\s*=\s*'[^']*'", "config_run_duration = '"+a.duration+"'", namelist.read_text())
    if count != 1: raise ValueError('deck must have one duration')
    overrides = {}
    for item in a.set:
        if not re.fullmatch(r'config_[a-z_]+_scheme=[a-z0-9_]+', item):
            raise ValueError('requires a literal native scheme assignment')
        key, value = item.split('=', 1)
        if key in overrides: raise ValueError('duplicate scheme override')
        overrides[key] = value
        assignment = f" {key} = '{value}'"
        content, replacements = re.subn(r'\b'+key+r"\s*=\s*'[^']*'", assignment, content)
        if replacements > 1: raise ValueError('duplicate namelist scheme')
        if replacements == 0:
            content, groups = re.subn(r'(&physics\s*\n)', lambda m: m[1]+assignment+'\n', content)
            if groups != 1: raise ValueError('requires one physics namelist group')
    namelist.write_text(content)
    streams = ET.parse(deck/'streams.atmosphere')
    for stream in streams.getroot():
        if 'output' in stream.get('type', '').split(';'):
            stream.set('output_interval', a.duration)
    streams.write(deck/'streams.atmosphere', encoding='unicode')
    case = a.case.resolve()
    if a.time_metadata:
        # Only owned copies receive the required seek coordinate. All existing
        # science payloads are verified unchanged by the Rust executable.
        view = root/'case'; view.mkdir()
        oro = next(stream.get('filename_template') for stream in streams.getroot()
                   if stream.get('name') == 'ugwp_oro_data_in')
        for src in case.iterdir():
            if src.is_file() and src.name != oro: (view/src.name).symlink_to(src)
        stamp = re.search(r"config_start_time\s*=\s*'([^']*)'", content).group(1)
        proc = subprocess.run([str(a.time_metadata.resolve()), str(case/oro), str(view/oro), stamp],
                              capture_output=True, text=True, check=True)
        (root/'time-metadata.json').write_text(proc.stdout)
        case = view
    dumps = root/'dumps'; dumps.mkdir()
    env = dict(os.environ, MPAS_ORACLE_DIR=str(dumps), MPAS_ORACLE_STEPS='0,1', MPAS_ORACLE_KEYS=SITE)
    cmd = [os.sys.executable, str(a.runner.resolve()), '--build', str(a.build.resolve()), '--deck', str(deck),
           '--case', str(case), '--out', str(root/'forecast'), '--tables', str(a.tables.resolve()),
           '--mpi', a.mpi, '--ranks', '1', '--pin-set-hash', sha(a.pins)]
    with (root/'capture.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([os.sys.executable, str(a.column_door.resolve()), '--scheme', 'gwdo_ugwp', '--seal',
                    '--dump', str(dumps), '--words', str(a.words.resolve()), '--inventory', str(a.inventory.resolve()),
                    '--pins', str(a.pins.resolve()), '--recipe', 'GNU-strict f1 frozen whole-model boundary'], check=True)
    manifest = json.loads((dumps/'manifest.json').read_text())
    if not manifest['captures']: raise ValueError('native run did not reach UGWP')
    publication = a.case/'PUBLICATION.json'
    result = {'scope':'real native driver boundary; not a hex forecast acceptance',
              'case_publication':str(publication), 'case_publication_sha256':sha(publication),
              'capture_binary_sha256':sha(a.build/'atmosphere_model'), 'captures':manifest['captures'],
              'dump_root_sha256':manifest['dump_root_sha256'], 'duration':a.duration,
              'runner_sha256':sha(a.runner), 'column_door_sha256':sha(a.column_door),
              'native_output_manifest_sha256':sha(root/'forecast/MANIFEST.json'),
              'time_metadata_preparation':bool(a.time_metadata), 'explicit_companions':overrides}
    (root/'capture.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
