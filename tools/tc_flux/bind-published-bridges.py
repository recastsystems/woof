"""Bind installed Rust helpers without changing shared installations."""
from pathlib import Path
import os
import json, shlex
from woof.bridge_assets import BUNDLED_ARTIFACTS
from woof.hex.engines import ENGINES
from woof.hex.restart_v842 import _STREAM_ENGINE
shared=next(Path('/workspace/atlas/_env/venv/lib').glob('python*/site-packages'))
roots=[shared/'woof/hex/libexec',shared/'woof/libexec/bridges']
env={}
for artifact in BUNDLED_ARTIFACTS:
 for root in roots[::-1]:
  paths=[root/name for name in (artifact.name,'lib'+artifact.name+'.so',artifact.name+'.so')]
  hit=next((p for p in paths if p.is_file()),None)
  if hit:
   env[artifact.env_var]=str(hit);break
for spec in (*ENGINES, _STREAM_ENGINE):
 hit=next((root/spec.name for root in roots if (root/spec.name).is_file()),None)
 if hit:
  for name in (*spec.env_names,spec.gpuwm_env):
   if name:env[name]=str(hit)
local_renderer=Path('runs/bin/rw_wrfbatch').resolve()
if local_renderer.is_file():
 for name in ('WOOF_RW_WRFBATCH','WOOF_HEX_RW_WRFBATCH','MPAS_PORT_RW_WRFBATCH'):
  env[name]=str(local_renderer)
env['PATH']=os.pathsep.join(str(root) for root in roots)+os.pathsep+os.environ['PATH']
Path('runs/bridge-env.json').write_text(json.dumps(env,indent=2)+'\n')
Path('runs/bridge-env.sh').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items()))
