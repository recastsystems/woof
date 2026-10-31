#!/usr/bin/env python3
"""Stage the requested dry dycore decks without physics-dependent diagnostics."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

ref = Path(sys.argv[1]).resolve()
source = ref / 'src/MPAS-Model/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F'
requested = set(re.findall(r"mpas_stream_inclusion_count\('([^']+)'", source.read_text()))
for suffix in ('step1', '6h'):
    original = ref / f'runs/x1.2562-dry-{suffix}'
    name = f'x1.2562-dry-no-convective-stats-{suffix}'
    run = ref / 'runs' / name
    deck = ref / 'decks' / name
    run.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ref / f'decks/x1.2562-dry-{suffix}', deck, dirs_exist_ok=True)
    streams = ET.parse(deck / 'streams.atmosphere')
    root = streams.getroot()
    root.remove(next(node for node in root if node.get('name') == 'ugwp_oro_data_in'))
    ET.indent(root)
    streams.write(deck / 'streams.atmosphere', encoding='unicode')
    removed = {}
    for file in (deck / 'stream_list').iterdir():
        lines = file.read_text().splitlines()
        gone = [line for line in lines if line.strip() in requested]
        if gone:
            removed[file.name] = gone
            file.write_text('\n'.join(line for line in lines if line.strip() not in requested) + '\n')
    for file in original.iterdir():
        if file.is_symlink() and file.name not in ('stream_list', 'namelist.atmosphere', 'streams.atmosphere'):
            if not (run / file.name).exists():
                (run / file.name).symlink_to(file.resolve())
    for file in deck.iterdir():
        if not (run / file.name).exists():
            (run / file.name).symlink_to(file)
    receipt = json.loads((original / 'INPUTS.json').read_text())
    receipt['label'] = 'Physics-off diagnostic deck with convective maxima/statistics omitted; pristine native binary'
    receipt['adaptation_authority'] = 'GUIDE INBOX 2026-10-28T22:09:14Z'
    receipt['removed_fields'] = removed
    receipt['removed_input_streams'] = {'ugwp_oro_data_in': 'Gravity-wave drag is off in this dycore-only deck; otherwise the unused immutable stream requires xtime absent in the static UGWP file.'}
    receipt['deck_files'] = {str(p.relative_to(deck)): hashlib.sha256(p.read_bytes()).hexdigest() for p in deck.rglob('*') if p.is_file()}
    (run / 'INPUTS.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(run, sum(map(len, removed.values())))
