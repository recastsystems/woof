#!/usr/bin/env python3
"""Carry the prior RUC integer-start witness method onto the new strict build."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

original, work = map(lambda p: Path(p).resolve(), sys.argv[1:3])
value = int(sys.argv[3])
assert not work.exists()
subprocess.run(['cp', '--reflink=auto', '-a', str(original), str(work)], check=True, timeout=120)
opt = Path(__import__('os').environ['RRFS_FORK_CPU_OPT'])
compiler = str(opt / 'mpich/bin/mpif90')
rows = [json.loads(line) for line in (original / 'compile-commands.jsonl').read_text().splitlines()]
commands = []
for name in ('module_ruc_land.F90', 'module_ruc_ice.F90'):
    row = next(r for r in reversed(rows) if r['compiler'] == 'mpif90' and name in r['argv'])
    cwd = Path(row['cwd'].replace(str(original), str(work)))
    argv = [compiler, f'-finit-integer={value}', *[a.replace(str(original), str(work)) for a in row['argv']]]
    subprocess.run(argv, cwd=cwd, check=True, timeout=180)
    obj = cwd / name.replace('.F90', '.o')
    subprocess.run(['ar', '-ru', str(work / 'src/core_atmosphere/physics/libphys.a'), str(obj)], check=True, timeout=30)
    shutil.copy2(obj, work / 'src/core_atmosphere/libphys' / obj.name)
    subprocess.run(['ar', '-ru', 'libdycore.a', 'libphys/' + obj.name], cwd=work / 'src/core_atmosphere', check=True, timeout=30)
    commands.append(dict(cwd=str(cwd), argv=argv))
argv = [compiler, '-O3', '-o', str(work / 'atmosphere_model'), *[str(p) for p in sorted((work / 'src/driver').glob('*.o'))],
        '-L.', '-ldycore', '-lops', '-lframework', '-L./external/SMIOL', '-lsmiolf', '-lsmiol', '-L/usr', '-lnetcdff', '-lnetcdf',
        '-L' + str(opt / 'pnetcdf/lib'), '-lpnetcdf', '-I./external/esmf_time_f90', '-L./external/esmf_time_f90', '-lesmf_time']
subprocess.run(argv, cwd=work / 'src', check=True, timeout=60)
commands.append(dict(cwd=str(work / 'src'), argv=argv))
binary = work / 'atmosphere_model'
(work / 'WITNESS.json').write_text(json.dumps(dict(label='RUC integer-start diagnostic witness, not pristine',
    start_value=value, source_changes=0, changed_compile_units=['module_ruc_land.F90', 'module_ruc_ice.F90'],
    original=str(original), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), commands=commands), indent=2) + '\n')
print(binary)
