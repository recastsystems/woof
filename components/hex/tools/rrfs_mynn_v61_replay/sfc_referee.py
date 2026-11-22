#!/usr/bin/env python3
"""Build a standalone native MYNN-SFC referee (CPU, referee tool only, never a run path).

v600: MYNN-SFC v6.0.0 b66187f exactly as the RRFS 44c0849 referee compiles it (compile-commands.jsonl of
      native/4eb8e25...; that host's MYNN-SFC make receives no COREDEF, so no -Dmpas).
v610: MYNN-SFC v6.1.0 a038a9e with the same lines plus -Dmpas, which the v6.1.x host passes
      (ufs-community/MPAS-Model noaa/develop de016e0 src/core_atmosphere/physics/Makefile:4,77 COREDEF=-Dmpas).
Both link mpas_kind_types, mpas_constants and mpas_atmphys_constants from the 44c0849 tree, compiled with
their own recorded lines. The only added flag is -fPIC (shared object for the ctypes bridge).
Usage: sfc_referee.py v600|v610 OUT   with RRFS_44C0849_REFERENCE and MYNN_SFC_CLONE set
"""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

version, out = sys.argv[1], Path(sys.argv[2])
assert version in ('v600', 'v610')
here = Path(__file__).resolve().parent
ref = Path(os.environ['RRFS_44C0849_REFERENCE'])   # the rebuildable 44c0849 referee (tools/rrfs_44c0849_reference)
mpas = ref/'src/MPAS-Model'
commands = ref/'native/4eb8e25bebd2aee3e2d6fa913beb4e2ed1a8b98249e75a016eef0b9b4fc0945b/compile-commands.jsonl'
upstream = Path(os.environ['MYNN_SFC_CLONE'])      # a clone of NCAR/MYNN-SFC holding a038a9e
out.mkdir(parents=True, exist_ok=False)
lines = {}
for line in commands.read_text().splitlines():
    d = json.loads(line)
    if d['compiler'] != 'gfortran':
        continue
    src = [a for a in d['argv'] if a.endswith(('.F', '.F90'))]
    if len(src) == 1:
        lines.setdefault((d['cwd'].split('build-atmosphere/')[-1], src[0]), [a for a in d['argv'] if not a.startswith('-I')])
def recorded(directory, name):
    argv = lines[(directory, name)]
    assert argv[-2:] == ['-c', name], argv[-3:]
    return argv[:-2]
deps = [('src/framework', 'mpas_kind_types.F'), ('src/framework', 'mpas_constants.F'),
        ('src/core_atmosphere/physics', 'mpas_atmphys_constants.F')]
sfc_dir = 'src/core_atmosphere/physics/physics_noaa/MYNN-SFC'
order = ['module_sf_mynnsfc_common.F90', 'module_sf_mynnsfc_land.F90', 'module_sf_mynnsfc_water.F90',
         'module_sf_mynnsfc_ice.F90', 'module_sf_mynnsfc_driver.F90']
sources = {}
for directory, name in deps:
    shutil.copyfile(mpas/directory/name, out/name); sources[name] = str(mpas/directory/name)
if version == 'v600':
    pin = 'b66187f54a0c169e36592552634867474e603ac0'
    for name in order:
        path = mpas/sfc_dir/('MPAS/'+name if name in ('module_sf_mynnsfc_common.F90', 'module_sf_mynnsfc_driver.F90') else name)
        shutil.copyfile(path, out/name); sources[name] = str(path)
    bridge = here.parents[1]/'tests/mpas_rrfs_mynnsfc/native_mynnsfc_bridge.F90'
else:
    pin = 'a038a9ea35a12f0bf879fac8b64c99a77ffdae1c'
    for name in order:
        rel = 'src/MPAS/'+name if name == 'module_sf_mynnsfc_common.F90' else 'src/'+name
        text = subprocess.run(['git', '-C', str(upstream), 'show', pin+':'+rel], check=True, capture_output=True).stdout
        (out/name).write_bytes(text); sources[name] = 'NCAR/MYNN-SFC@'+pin+':'+rel
    bridge = here/'native_mynnsfc_bridge_v610.F90'
shutil.copyfile(bridge, out/'bridge.F90')
compiled = []
for directory, name in deps:
    argv = ['gfortran', *recorded(directory, name), '-fPIC', '-c', name]
    subprocess.run(argv, cwd=out, check=True); compiled.append(argv)
for name in order:
    argv = ['gfortran', *recorded(sfc_dir, name)]
    if version == 'v610':
        argv.append('-Dmpas')
    argv += ['-fPIC', '-c', name]
    subprocess.run(argv, cwd=out, check=True); compiled.append(argv)
bridge_argv = ['gfortran', *recorded(sfc_dir, 'module_sf_mynnsfc_driver.F90'), '-fPIC', '-c', 'bridge.F90']
subprocess.run(bridge_argv, cwd=out, check=True); compiled.append(bridge_argv)
objects = [Path(n).with_suffix('.o').name for _, n in deps] + [Path(n).with_suffix('.o').name for n in order] + ['bridge.o']
link = ['gfortran', '-shared', '-o', 'libmynnsfc.so', *objects]
subprocess.run(link, cwd=out, check=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
version_text = subprocess.run(['gfortran', '--version'], capture_output=True, text=True).stdout.splitlines()[0]
receipt = dict(schema='mynnsfc-standalone-referee-v1', version=version, mynn_sfc_commit=pin,
    label=('MYNN-SFC v6.0.0 b66187f (RRFS 44c0849 pin), 44c0849 strict compile lines' if version == 'v600'
           else 'MYNN-SFC v6.1.0 a038a9e, 44c0849 strict compile lines + -Dmpas (v6.1.x host COREDEF)'),
    gfortran=version_text, compile=compiled, link=link, sources=sources,
    source_sha256={n: sha(out/n) for n in [*(n for _, n in deps), *order, 'bridge.F90']},
    library_sha256=sha(out/'libmynnsfc.so'), referee_tool_only=True)
(out/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k: receipt[k] for k in ('version', 'mynn_sfc_commit', 'gfortran', 'library_sha256')}))
