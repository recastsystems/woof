#!/usr/bin/env python3
"""List semantic deck changes and the workflow commits that introduced keys."""
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ref = Path(sys.argv[1]).resolve()
wf = ref / 'wf/rrfs-workflow'
old = Path(__import__('os').environ['MPAS_LANES']) / 'ref/rrfs/decks/x1.2562-20230331-generated-v1/suite-hrrrv5'
new = ref / 'decks/x1.2562-6h'
def nml(file):
    out = {}
    for line in file.read_text().splitlines():
        match = re.match(r'\s*(\w+)\s*=\s*(.*?)\s*(?:!.*)?$', line)
        if match:
            key, value = match.groups()
            value = value.strip().strip(',').lower()
            if value in ('.true.', '.false.'):
                value = value.strip('.')
            try:
                value = str(float(value))
            except ValueError:
                pass
            out[key] = value
    return out
def streams(file):
    out = {}
    for node in ET.parse(file).getroot():
        for key, value in node.attrib.items():
            out[node.get('name') + '.' + key] = value
        for child in node:
            out[node.get('name') + '.' + child.tag + '.' + child.get('name')] = 'present'
    return out
def introduced(key, path):
    text = subprocess.check_output(['git', '-C', str(wf), 'log', '--reverse', '--format=%H', '-S', key, '--', path], text=True, timeout=30).strip()
    return text.splitlines()[0] if text else 'not a workflow template key; generated-deck adaptation'
lines = ['Deck comparison against the published 1d6e370e generated hrrrv5 deck.',
         'Target workflow 570c81fec5adaefe966b9dd3eec3272da150fc9e. Source init uses the carried two-NULLIFY derivative.',
         'Commands: git log --reverse --format=%H -S KEY -- PATH; git diff 08eee4ceee301898d3cb837e03699a8da0d1cb32 570c81f -- parm/hrrrv5 parm/streams.atmosphere fix/stream_list/hrrrv5.',
         'A template introduction commit identifies the key. Duration, filename, cadence and global/lam changes below are local reference adaptations, not upstream science changes.', '',
         '| File/key | Old | New | Introduction / reason |', '|---|---|---|---|']
for filename, parser, template in [('namelist.atmosphere', nml, 'parm/hrrrv5/namelist.atmosphere'), ('streams.atmosphere', streams, 'parm/streams.atmosphere')]:
    a, b = parser(old / filename), parser(new / filename)
    for key in sorted(set(a) | set(b)):
        if a.get(key) == b.get(key):
            continue
        origin = introduced(key if filename.startswith('namelist') else key.split('.')[0], template)
        if key == 'config_mynn_ess':
            origin = '9c958777cf2f040a510a495fc9cd24ec10749e17; MPAS Registry key introduced by cae8a92f375770a3be613cf754083ca7a70c0a11'
        if key not in b and filename.startswith('namelist'):
            origin += '; old generated explicit scheme override omitted by new template, native hrrrv5 suite resolves it'
        lines.append(f'| {filename}:{key} | {a.get(key,"absent")} | {b.get(key,"absent")} | {origin} |')
for filename in sorted(set(p.name for p in (old / 'stream_list').iterdir()) | set(p.name for p in (new / 'stream_list').iterdir())):
    a = set((old / 'stream_list' / filename).read_text().split()) if (old / 'stream_list' / filename).exists() else set()
    b = set((new / 'stream_list' / filename).read_text().split()) if (new / 'stream_list' / filename).exists() else set()
    for key in sorted(a ^ b):
        lines.append(f'| {filename}:{key} | {"present" if key in a else "absent"} | {"present" if key in b else "absent"} | {introduced(key, "fix/stream_list/hrrrv5/" + filename)} |')
old_init = Path(__import__('os').environ['MPAS_LANES']) / 'ref/rrfs/cases/approved-canonical-global-ruc9-v1/x1.2562-soil9/2023-03-31_12.00.00/case7/namelist.init_atmosphere'
new_init = ref / 'cases/x1.2562/case7-pointer-witness/namelist.init_atmosphere'
assert nml(old_init) == nml(new_init)
lines += ['', 'Init namelist: 0 added, removed or changed assignments against the old canonical case7 namelist. Target workflow template substituted with the old canonical variables (case7, GFS58, RUC9, levels59, L60 vertical file). Case9 changes init_case=9 and end_time=13Z, and input stream reads case7-init.nc instead of static.nc. All other template keys are preserved.',
          'Per-mesh/run adaptations: x1.2562 applies no LBC, dt720; lam7k applies LBC, dt20, invariant stream omitted because its case7 file carries invariant fields. Both use four substeps. Runs use 6h global / 1h lam duration, hourly history/diag/restart; step1 variants use one dt and one-dt output cadence. Deterministic do_sppt=false. Relative filename substitutions are init.nc, invariant.nc, ugwp_oro_data.nc.',
          'The optional dry diagnostic request sets physics suite none and gwdo/lsm/pbl off, and adds tracer_oracle with state/base/scalar fields at initial time and endpoint. The original full diagnostic streams cause a native convective-diagnostic SIGSEGV. Separate dry-no-convective-stats decks omit 27 physics-dependent convective fields per GUIDE 22:09:14Z, plus unused ugwp_oro_data_in whose static file lacks xtime with GWDO off. INPUTS.json lists every omission. These are diagnostic dycore controls, not full-physics forecast proof; readiness is recorded separately in READY.md.']
(ref / 'decks/DECK-DIFF.md').write_text('\n'.join(lines) + '\n')
patch = subprocess.check_output(['git', '-C', str(wf), 'diff', '08eee4ceee301898d3cb837e03699a8da0d1cb32', '570c81f', '--', 'parm/hrrrv5', 'parm/streams.atmosphere', 'fix/stream_list/hrrrv5'], timeout=60)
(ref / 'decks/workflow-deck.diff').write_bytes(patch)
with (ref / 'READY.md').open('a') as f:
    path = ref / 'decks/DECK-DIFF.md'
    f.write(f'workflow decks | {path} | sha256 {hashlib.sha256(path.read_bytes()).hexdigest()} | per-key semantic changes, introduction commits and local substitutions; runs bind deck hashes.\n')
print(ref / 'decks/DECK-DIFF.md')
