#!/usr/bin/env python3
"""Stage workflow case 7 and case 9 against the existing public intermediates."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ref = Path(sys.argv[1]).resolve()
home = Path.home()
lanes = Path(os.environ['MPAS_LANES'])
old = lanes / 'ref/rrfs/cases/approved-canonical-global-ruc9-v1/x1.2562-soil9/2023-03-31_12.00.00/case7'
row = json.loads((old / 'MANIFEST.json').read_text())['row']
wf = ref / 'wf/rrfs-workflow'
public = lanes / 'ref/rrfs/cases/workflow-public-tables-v1'
lam = Path(os.environ['RRFS_LAM7K_CASE'])
for mesh, case in [('x1.2562', 7), ('lam7k', 7), ('lam7k', 9)]:
    dst = ref / 'cases' / mesh / f'case{case}'
    dst.mkdir(parents=True, exist_ok=True)
    variables = dict(row['variables'], init_case=case, decomp_file_prefix=mesh + '.graph.info.part.')
    if case == 9:
        variables['end_time'] = '2023-03-31_13:00:00'
    text = (wf / 'parm/hrrrv5/namelist.init_atmosphere').read_text()
    for key, value in variables.items():
        text = text.replace('${' + key + '}', str(value))
    assert '${' not in text
    (dst / 'namelist.init_atmosphere').write_text(text)
    text = (wf / 'parm/streams.init_atmosphere').read_text()
    for key, value in dict(input_stream='case7-init.nc' if case == 9 else 'static.nc', output_stream='init.nc', lbc_interval=1).items():
        text = text.replace('@' + key + '@', str(value))
    assert '@' not in text
    (dst / 'streams.init_atmosphere').write_text(text)
    files = {'static.nc': lanes / 'ref/rrfs/cases/x1.2562/2023-03-31_12.00.00/soil9/static/static.nc' if mesh == 'x1.2562' else lam / 'lam7k.static.nc',
             'ZETA_LEVELS.txt': wf / 'fix/vert_levels/L60.txt'}
    if case == 9:
        files['case7-init.nc'] = ref / 'cases' / mesh / 'case7-pointer-witness/init.nc'
    for p in public.iterdir():
        if p.suffix.lower() in ('.tbl', '.dat', '.bin', '.dbl'):
            files[p.name] = p
    for hour in range(2 if case == 9 else 1):
        name = f'GFS:2023-03-31_{12 + hour:02d}'
        files[name] = lanes / f'ref/rrfs/cases/hourly-gfs-native-ungrib-20230331-v1/hour-{hour:02d}' / name
    manifest = {}
    for name, source in files.items():
        assert source.is_file(), source
        path = dst / name
        if not path.exists():
            path.symlink_to(source)
        manifest[name] = dict(source=str(source), bytes=source.stat().st_size)
    (dst / 'INPUTS.json').write_text(json.dumps(dict(workflow='570c81fec5adaefe966b9dd3eec3272da150fc9e', variables=variables, files=manifest), indent=2) + '\n')
    print(dst)
