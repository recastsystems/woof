#!/usr/bin/env python3
"""Render the pinned workflow decks with explicit mesh/run substitutions."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

ref = Path(sys.argv[1]).resolve()
wf = ref / 'wf/rrfs-workflow'
home = Path.home()
lanes = Path(os.environ['MPAS_LANES'])
old = lanes / 'ref/rrfs/cases/x1.2562/2023-03-31_12.00.00/soil9'
lam = Path(os.environ['RRFS_LAM7K_CASE'])
jobs = [('x1.2562-6h', 'x1.2562', 21600, False), ('x1.2562-6h-repeat', 'x1.2562', 21600, False),
        ('lam7k-1h', 'lam7k', 3600, False), ('x1.2562-step1', 'x1.2562', 720, False),
        ('lam7k-step1', 'lam7k', 20, False), ('x1.2562-dry-step1', 'x1.2562', 720, True),
        ('x1.2562-dry-6h', 'x1.2562', 21600, True)]
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def clock(seconds):
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'
for name, mesh, duration, dry in jobs:
    dt = 720 if mesh == 'x1.2562' else 20
    interval = clock(dt) if name.endswith('step1') else '01:00:00'
    deck = ref / 'decks' / name
    run = ref / 'runs' / name
    deck.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    variables = dict(dt=float(dt), start_time='2023-03-31_12:00:00', run_duration=clock(duration), substeps=4,
                     pio_num_iotasks=0, pio_stride=1, MESH_NAME=mesh, do_DAcycling='false', radt='60',
                     lsm_scheme='sf_ruc', nsoillevels=9, jedi_da='false', do_sppt='false')
    text = (wf / 'parm/hrrrv5/namelist.atmosphere').read_text()
    for key, value in variables.items():
        text = text.replace('${' + key + '}', str(value))
    assert '${' not in text
    text = text.replace('graphinfo/', '')
    if mesh == 'x1.2562':
        text = text.replace('config_apply_lbcs = true', 'config_apply_lbcs = false')
    if dry:
        text = text.replace("config_physics_suite = 'hrrrv5'", "config_physics_suite = 'none'")
        # Explicit physics-off control requested by the dycore lane.
        for key in ('gwdo', 'lsm', 'pbl'):
            text = re.sub(r"(config_" + key + r"_scheme\s*=\s*)'[^']*'", r"\1'off'", text)
    (deck / 'namelist.atmosphere').write_text(text)
    streams = (wf / 'parm/streams.atmosphere').read_text()
    for key in ('mpasout_interval', 'restart_interval', 'history_interval', 'diag_interval'):
        streams = streams.replace('@' + key + '@', interval)
    streams = streams.replace('@lbc_interval@', '1')
    root = ET.fromstring(streams)
    if dry:
        tracer = ET.SubElement(root, 'stream', name='tracer_oracle', type='output',
            filename_template='tracer.$Y-$M-$D_$h.$m.$s.nc', output_interval=clock(duration),
            clobber_mode='replace_files', io_type='pnetcdf,cdf5')
        for variable in ('rho_zz', 'theta_m', 'ru', 'rw', 'u', 'w', 'rho_base', 'theta_base',
                         'rtheta_base', 'rho_p', 'rtheta_p', 'exner', 'exner_base', 'pressure_p', 'pressure_base'):
            ET.SubElement(tracer, 'var', name=variable)
        ET.SubElement(tracer, 'var_array', name='scalars')
    if mesh == 'lam7k':
        # The cut mesh case7 already carries its invariant fields, as the prior lam deck did.
        root.remove(next(node for node in root if node.get('name') == 'invariant'))
    ET.indent(root)
    (deck / 'streams.atmosphere').write_text(ET.tostring(root, encoding='unicode') + '\n')
    shutil.copytree(wf / 'fix/stream_list/hrrrv5', deck / 'stream_list', dirs_exist_ok=True)
    links = {'init.nc': ref / 'cases' / mesh / 'case7-pointer-witness/init.nc',
             'ugwp_oro_data.nc': old / 'static/ugwp_oro_data.nc' if mesh == 'x1.2562' else lam / 'lam7k.ugwp_oro_data.nc'}
    if mesh == 'x1.2562':
        links['invariant.nc'] = old / 'invariant/invariant.nc'
    else:
        links.update({p.name: p for p in (ref / 'cases/lam7k/case9-pointer-witness').glob('lbc.*.nc')})
    for folder in (ref / 'native/atmosphere', ref / 'tables'):
        for p in folder.rglob('*'):
            if p.is_file() and not os.access(p, os.X_OK) and (p.suffix.lower() in ('.tbl', '.dbl', '.bin', '.dat') or p.name.endswith(('_DATA', '_tempo_v3')) or p.name == 'ugwp_limb_tau.nc'):
                links[p.name] = p.resolve()
    for p in deck.iterdir():
        links[p.name] = p
    for name_, source in links.items():
        assert source.exists(), source
        target = run / name_
        if not target.exists():
            target.symlink_to(source)
    receipt = dict(workflow='570c81fec5adaefe966b9dd3eec3272da150fc9e', mesh=mesh, dt=dt, seconds=duration,
                   physics_off=dry, variables=variables, init_label='two-NULLIFY derivative; confirmation pending',
                   deck_files={str(p.relative_to(deck)): sha(p) for p in deck.rglob('*') if p.is_file()},
                   inputs={key: dict(path=str(p), sha256=sha(p)) for key, p in links.items() if p.is_file() and p.suffix == '.nc'},
                   binary_sha256=sha(ref / 'native/atmosphere/atmosphere_model'))
    (run / 'INPUTS.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(run)
