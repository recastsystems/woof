#!/usr/bin/env python3
"""Render NOAA's hrrrv5 deck for one global reference case from rrfs-workflow at the pin in SOURCE-PINS.json.

Usage: make_deck.py DECK [--mesh x1.2562] [--start 2023-03-31_12:00:00] [--hours 6] [--dt 720]

Fetches parm/hrrrv5/namelist.atmosphere, parm/streams.atmosphere and fix/stream_list/hrrrv5/* from
NOAA-EMC/rrfs-workflow at the pinned commit (nothing of it ships with hex), fills the workflow's template
variables, and makes the two changes a global one-process reference run needs, each printed as it is made:
no lateral boundaries (config_apply_lbcs = false) and the partition prefix without the workflow's graphinfo/
folder. History, diagnostics, restart and analysis-state streams are written every hour. For the x1.2562
six-hour case the namelist's sha256 begins c5b84d079d3f72aa, the deck of the reference run of record.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAM_LISTS = ('analysis', 'background', 'control', 'da_state', 'diagnostics', 'ensemble', 'output', 'surface')

p = argparse.ArgumentParser()
p.add_argument('deck', type=Path)
p.add_argument('--mesh', default='x1.2562')
p.add_argument('--start', default='2023-03-31_12:00:00')
p.add_argument('--hours', type=int, default=6)
p.add_argument('--dt', type=float, default=720.0)
p.add_argument('--substeps', type=int, default=4)
a = p.parse_args()

pins = json.loads((HERE / 'SOURCE-PINS.json').read_text())
raw = 'https://raw.githubusercontent.com/' + pins['workflow_url'].removeprefix('https://github.com/') + '/' + pins['workflow_commit'] + '/'


def get(path):
    with urllib.request.urlopen(raw + path, timeout=60) as r:
        return r.read().decode()


variables = dict(dt=repr(a.dt), start_time=a.start, run_duration=f'{a.hours:02d}:00:00', substeps=str(a.substeps),
                 pio_num_iotasks='0', pio_stride='1', MESH_NAME=a.mesh, do_DAcycling='false', radt='60',
                 lsm_scheme='sf_ruc', nsoillevels='9', jedi_da='false', do_sppt='false')
namelist = get('parm/hrrrv5/namelist.atmosphere')
for key, value in variables.items():
    namelist = namelist.replace('${' + key + '}', value)
assert '${' not in namelist, 'an unfilled workflow variable is left in the namelist'


def change(text, old, new, why):
    assert text.count(old) == 1, f'the workflow template changed at {old!r}'
    print(f'deck change: {old.strip()} -> {new.strip()} ({why})')
    return text.replace(old, new)


namelist = change(namelist, 'config_apply_lbcs = true', 'config_apply_lbcs = false', 'global mesh, no lateral boundaries')
namelist = change(namelist, f"'graphinfo/{a.mesh}.graph.info.part.'", f"'{a.mesh}.graph.info.part.'", 'partition files beside the run')
streams = get('parm/streams.atmosphere')
for key in ('mpasout_interval', 'restart_interval', 'history_interval', 'diag_interval'):
    streams = streams.replace('@' + key + '@', '01:00:00')
streams = streams.replace('@lbc_interval@', '1')
assert '@' not in re.sub(r'<!--.*?-->', '', streams, flags=re.S), 'an unfilled workflow variable is left in the streams'
a.deck.mkdir(parents=True, exist_ok=False)
(a.deck / 'stream_list').mkdir()
(a.deck / 'namelist.atmosphere').write_text(namelist)
(a.deck / 'streams.atmosphere').write_text(streams)
for name in STREAM_LISTS:
    (a.deck / 'stream_list' / f'stream_list.atmosphere.{name}').write_text(get(f'fix/stream_list/hrrrv5/stream_list.atmosphere.{name}'))
digest = hashlib.sha256(namelist.encode()).hexdigest()
print(f'deck ready: {a.deck} (rrfs-workflow {pins["workflow_commit"][:9]}, {a.mesh}, {a.start}, {a.hours} h, dt {a.dt:g} s; '
      f'namelist sha256 {digest[:16]})')
