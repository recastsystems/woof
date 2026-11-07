#!/usr/bin/env python3
"""Apply the published strict build plumbing with the new source pin data.

Referee (``RRFS_44C0849_REFEREE``, default ``pristine``):

- ``pristine``: the pinned 44c0849 source, unchanged.
- ``patched``: the REFERENCE OF RECORD for RRFS 44c0849 claims, always labelled patched, never pristine. It is the
  pinned source plus exactly four owner-approved lines, each where native reads a value it never set:
  1. two ``nullify(smoke_fine_fg, dust_fine_fg, dust_coarse_fg)`` lines in ``mpas_init_atm_cases.F`` (owner ruling
     2026-10-14 for fork 1d6e370e, carried to 44c0849 2026-10-29): native's case-7/9 init tests optional smoke/dust
     pointers it never nullified and crashes;
  2. two ``ilnb = 0`` lines in RUC, before ``call snowsoil`` (``module_ruc_land.F90``) and before ``call snowseaice``
     (``module_ruc_ice.F90``) (owner ruling 2026-11-06): for snow thinner than ``snth`` (about 10 mm water
     equivalent) native reads the uninitialised local ``ilnb`` to pick the ``tsnav`` formula, and ``tsnav`` feeds
     the next step's snow compaction. Witness: pristine native against native with these two lines reproduces the
     hex-vs-pristine-native divergence word for word (127,564 vs 127,565 words at step 8, x1.40962 hrrrv5 strict),
     native with them is 0 different run to run, and hex (which sets ``ilnb = 0``) matches it.
  The build writes the exact change to ``source-correction.diff`` and labels ``SOURCE-PINS.json``
  (``referee``, ``source_correction``, ``patches``), so every binary says which referee it is.

``INIT_POINTER_WITNESS=1`` (older recipe, kept as it was) applies only the two NULLIFY lines.
"""
import json
import difflib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'native_physics_reference'))
from rrfs import prepare_build

NULLIFY = '      nullify(smoke_fine_fg, dust_fine_fg, dust_coarse_fg)\n'
ILNB = '      ilnb = 0  ! PATCHED REFEREE (owner ruling 2026-11-06): native leaves ilnb undefined for snow thinner than snth\n'
PATCHED_LABEL = ('RRFS 44c0849 strict, PATCHED referee (not pristine): two NULLIFY lines (init, owner ruling 2026-10-14, '
                 'carried 2026-10-29) + two ilnb = 0 lines (RUC, owner ruling 2026-11-06)')
PATCHES = [
    dict(file='src/core_init_atmosphere/mpas_init_atm_cases.F', line=NULLIFY.strip(), count=2, ruling='2026-10-14',
         reason='native case-7/9 init tests optional smoke/dust pointers it never nullified and crashes'),
    dict(file='src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90', line='ilnb = 0',
         before='call snowsoil (', count=1, ruling='2026-11-06',
         reason='native reads the uninitialised local ilnb for snow thinner than snth to choose the tsnav formula'),
    dict(file='src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_ice.F90', line='ilnb = 0',
         before='call snowseaice', count=1, ruling='2026-11-06',
         reason='native reads the uninitialised local ilnb for snow thinner than snth to choose the tsnav formula'),
]

referee = os.environ.get('RRFS_44C0849_REFEREE', 'pristine')
if referee not in ('pristine', 'patched'):
    raise SystemExit(f'RRFS_44C0849_REFEREE must be pristine or patched, not {referee!r}')
init_witness = os.environ.get('INIT_POINTER_WITNESS') == '1'

reference, destination = map(Path, sys.argv[1:3])
pins = json.loads((reference / 'SOURCE-PINS.json').read_text())
prepare_build.PINS = dict(prepare_build.PINS, fork=pins['model_commit'], workflow=pins['workflow_commit'],
                         submodules={r['path']: r['commit'] for r in pins['submodules']})
old = Path(__import__('os').environ['MPAS_LANES']) / 'ref/rrfs/native/51879f7aa26b3292f1e8464691e308ca1dfbae36de8c88fdf467b775cdd947d0'
prepare_build.prepare(reference / 'src/MPAS-Model', destination, 'gnu-strict',
                      old / 'src/core_atmosphere/physics/physics_wrf/files')


def nullify(text: str) -> str:
    after = text.replace("      call mpas_pool_get_array(fg, 'p', p_fg)\n! Chemistry", "      call mpas_pool_get_array(fg, 'p', p_fg)\n" + NULLIFY + "! Chemistry")
    after = after.replace("      call mpas_pool_get_array(fg, 'qnr', qnr_fg)\n! Chemistry", "      call mpas_pool_get_array(fg, 'qnr', qnr_fg)\n" + NULLIFY + "! Chemistry")
    assert after.count(NULLIFY) == 2, 'the two NULLIFY locations changed: refuse rather than build an unlabelled referee'
    return after


def ilnb(text: str, call: str) -> str:
    lines = text.splitlines(True)
    hits = [i for i, line in enumerate(lines) if call in line and not line.lstrip().startswith('!')]
    assert len(hits) == 1, f'expected one {call!r}, found {len(hits)}: refuse rather than build an unlabelled referee'
    lines.insert(hits[0], ILNB)
    return ''.join(lines)


changes = []
if referee == 'patched' or init_witness:
    changes.append((PATCHES[0]['file'], nullify))
if referee == 'patched':
    changes += [(PATCHES[1]['file'], lambda t: ilnb(t, PATCHES[1]['before'])),
                (PATCHES[2]['file'], lambda t: ilnb(t, PATCHES[2]['before']))]
diffs = []
for relative, edit in changes:
    file = destination / relative
    before = file.read_text()
    after = edit(before)
    file.write_text(after)
    diffs.append(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                              fromfile='a/' + relative, tofile='b/' + relative)))
if changes:
    (destination / 'source-correction.diff').write_text(''.join(diffs))
    manifest = destination / 'SOURCE-PINS.json'
    data = json.loads(manifest.read_text())
    if referee == 'patched':
        data.update(referee='patched', pristine=False, source_correction=PATCHED_LABEL, patches=PATCHES)
    else:
        data['source_correction'] = 'Optional smoke/dust pointer nullify witness. Not pristine.'
    manifest.write_text(json.dumps(data, indent=2) + '\n')
