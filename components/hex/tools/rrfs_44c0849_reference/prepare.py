#!/usr/bin/env python3
"""Apply the published strict build plumbing with the new source pin data."""
import json
import difflib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'native_physics_reference'))
from rrfs import prepare_build

reference, destination = map(Path, sys.argv[1:3])
pins = json.loads((reference / 'SOURCE-PINS.json').read_text())
prepare_build.PINS = dict(prepare_build.PINS, fork=pins['model_commit'], workflow=pins['workflow_commit'],
                         submodules={r['path']: r['commit'] for r in pins['submodules']})
old = Path(__import__('os').environ['MPAS_LANES']) / 'ref/rrfs/native/51879f7aa26b3292f1e8464691e308ca1dfbae36de8c88fdf467b775cdd947d0'
prepare_build.prepare(reference / 'src/MPAS-Model', destination, 'gnu-strict',
                      old / 'src/core_atmosphere/physics/physics_wrf/files')
if os.environ.get('INIT_POINTER_WITNESS') == '1':
    file = destination / 'src/core_init_atmosphere/mpas_init_atm_cases.F'
    before = file.read_text()
    after = before.replace("      call mpas_pool_get_array(fg, 'p', p_fg)\n! Chemistry", "      call mpas_pool_get_array(fg, 'p', p_fg)\n      nullify(smoke_fine_fg, dust_fine_fg, dust_coarse_fg)\n! Chemistry")
    after = after.replace("      call mpas_pool_get_array(fg, 'qnr', qnr_fg)\n! Chemistry", "      call mpas_pool_get_array(fg, 'qnr', qnr_fg)\n      nullify(smoke_fine_fg, dust_fine_fg, dust_coarse_fg)\n! Chemistry")
    assert after.count('nullify(smoke_fine_fg, dust_fine_fg, dust_coarse_fg)') == 2
    file.write_text(after)
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/src/core_init_atmosphere/mpas_init_atm_cases.F', tofile='b/src/core_init_atmosphere/mpas_init_atm_cases.F'))
    (destination / 'source-correction.diff').write_text(patch)
    manifest = destination / 'SOURCE-PINS.json'
    data = json.loads(manifest.read_text())
    data['source_correction'] = 'Optional smoke/dust pointer nullify witness. Not pristine.'
    manifest.write_text(json.dumps(data, indent=2) + '\n')
