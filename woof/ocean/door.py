"""Table-driven ocean CLI and atmosphere attachment. Numeric preparation is Rust."""
from __future__ import annotations
from pathlib import Path
import json, os, math, importlib, subprocess, mmap
from functools import lru_cache
from .interface import OceanGrid, Bathymetry, validate_fields

@lru_cache(maxsize=1)
def rows():
    return json.loads(Path(__file__).with_name('rows.json').read_text())

def read_config(path):
    import tomllib
    path=Path(path).resolve()
    doc=tomllib.loads(path.read_text())
    cfg=doc.get('ocean',doc)
    unknown=set(cfg)-{'solvers','prepared','interval_seconds','factories','options'}
    if unknown:raise ValueError(f"unknown ocean config keys: {sorted(unknown)}")
    cfg.setdefault('solvers',['waves','surge']);cfg.setdefault('interval_seconds',60.)
    if not isinstance(cfg['solvers'],list) or not cfg['solvers'] or len(set(cfg['solvers']))!=len(cfg['solvers']):raise ValueError("ocean solvers must be a nonempty unique list")
    if set(cfg['solvers'])-rows()['solvers'].keys():raise ValueError("unknown ocean solver row")
    if not math.isfinite(cfg['interval_seconds']) or cfg['interval_seconds']<=0:raise ValueError("ocean interval_seconds must be positive and finite")
    cfg['prepared']=str((path.parent/cfg['prepared']).resolve())
    return cfg

def upload(path,shape):
    """Raw Rust-produced FP32 bytes go straight through the CUDA driver."""
    import cupy as cp
    import ctypes
    path=Path(path);expected=4*math.prod(shape)
    if path.stat().st_size!=expected:raise ValueError(f"prepared bytes differ for {path}")
    out=cp.empty(tuple(shape),dtype=cp.float32)
    with path.open('rb') as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_COPY) as view:
            ptr=ctypes.addressof(ctypes.c_char.from_buffer(view))
            cp.cuda.runtime.memcpy(out.data.ptr,ptr,expected,cp.cuda.runtime.memcpyHostToDevice)
    return out

def download(array,path):
    """Output-cadence CUDA byte transfer. No Python decode or field transform."""
    import cupy as cp
    import ctypes
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w+b') as f:
        f.truncate(array.nbytes)
        with mmap.mmap(f.fileno(),array.nbytes,access=mmap.ACCESS_WRITE) as view:
            ptr=ctypes.addressof(ctypes.c_char.from_buffer(view))
            cp.cuda.runtime.memcpy(ptr,array.data.ptr,array.nbytes,cp.cuda.runtime.memcpyDeviceToHost)


def load_prepared(path):
    import hashlib
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m['schema']!='gpuwm.ocean.prepared/v1':raise ValueError("wrong ocean preparation schema")
    arrays={}
    for name,row in m['arrays'].items():
        source=path/row['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(f"ocean input hash mismatch: {name}")
        arrays[name]=upload(source,row['shape'])
    grid=OceanGrid(m['nx'],m['ny'],m['dx'],m['dy'],*(arrays[n] for n in ('latitude','longitude','cosalpha','sinalpha')))
    bathy=Bathymetry(*(arrays[n] for n in ('elevation','depth','wet','coastal')),m['datum'],str(path/'manifest.json'))
    grid.validate();validate_fields(bathy,grid)
    return grid,bathy,m

def prepare_coupler(cfg):
    from .coupling import DeviceCoupler
    grid,bathy,manifest=load_prepared(cfg['prepared'])
    selected={}
    for name in cfg['solvers']:
        spec=cfg.get('factories',{}).get(name,rows()['solvers'][name]['factory'])
        module,sep,factory=spec.partition(':')
        if not sep or not module.startswith('woof.ocean.'):
            raise ValueError("ocean factories must name woof.ocean.module:callable")
        try:fn=getattr(importlib.import_module(module),factory)
        except (ModuleNotFoundError,AttributeError) as exc:
            raise RuntimeError(f"real {name} solver factory {spec} is unavailable; deliver {rows()['solvers'][name]['report_lane']} before this coupled run") from exc
        selected[name]=fn(grid,bathy,config=cfg.get('options',{}).get(name,{}))
        validate_fields(selected[name].outputs,grid)
    return DeviceCoupler(grid,bathy,**selected,interval=cfg['interval_seconds'])

def attach_atmosphere(driver,cfg,*,latitude=None,longitude=None):
    if not getattr(cfg,"ocean_config",""):return
    config=read_config(cfg.ocean_config)
    driver.ocean=prepare_coupler(config)
    if driver.fields['psfc'].shape!=driver.ocean.grid.shape:raise ValueError("ocean and atmosphere must share the prepared mass grid")
    import cupy as cp
    if latitude is None or longitude is None:raise ValueError("ocean coupling needs atmosphere latitude and longitude to verify the shared grid")
    for name,source,target in (("latitude",latitude,driver.ocean.grid.latitude),("longitude",longitude,driver.ocean.grid.longitude)):
        a=cp.asarray(source,dtype=cp.float32).reshape(target.shape)
        if not bool(cp.allclose(a,target,atol=1.e-5,rtol=0.)):raise ValueError(f"ocean {name} differs from atmosphere grid")
    # The existing driver currently accepts only common-grid coupling. A
    # remapped ocean grid requires a Rust conservative map, never reshaping.

def native_binary():
    from woof import bridges
    name=bridges.executable_name('rw_ocean')
    def find():return bridges.find_artifact('GPUWM_RW_OCEAN',name,crate_relative='tools/ocean_prepare')
    found=find()
    if found is None and bridges.fetch_on_demand('rw_ocean'):found=find()
    if found is not None:return found
    raise FileNotFoundError('rw_ocean is required; build tools/ocean_prepare with cargo build --release --locked, or run woof fetch-bridges')

def native_main(args):
    subprocess.run([str(native_binary()),str(args.request.resolve())],check=True)
    return 0

def render_ocean(request):
    from woof import rustwx
    # Same resolver and environment as every production renderer door.
    binary=rustwx.find_renderer()
    if binary is None:raise FileNotFoundError(rustwx.renderer_remedy())
    subprocess.run([str(binary),'--ocean-products',str(Path(request).resolve())],env=rustwx.renderer_env(),check=True)

def render_main(args):render_ocean(args.request);return 0

def register_cli(subparsers):
    p=subparsers.add_parser('ocean',help='prepare and run coupled atmosphere, waves and surge')
    commands=p.add_subparsers(dest='ocean_command',required=True)
    from .waves.data import register_cli as register_wave_data_cli
    register_wave_data_cli(commands)
    for name in ('prepare','extract'):
        s=commands.add_parser(name,help='Rust bathymetry or atmosphere field preparation')
        s.add_argument('request',type=Path);s.set_defaults(func=native_main)
    s=commands.add_parser('render',help='render ocean products through rw_wrfbatch')
    s.add_argument('request',type=Path);s.set_defaults(func=render_main)
    for name in ('run','prove-forcing'):
        s=commands.add_parser(name,help='run the real solvers' if name=='run' else 'replay real atmosphere forcing through the device hand-off')
        s.add_argument('config',type=Path);s.add_argument('--forcing',type=Path,required=True)
        s.add_argument('--outdir',type=Path,required=True);s.add_argument('--render',action='store_true')
        s.add_argument('--research-data-licence',default='',help='explicit opt-in to the named non-default data licence')
        s.set_defaults(func=replay_main,proof=name=='prove-forcing')

def replay_main(args):
    from .replay import run
    run(args.config,args.forcing,args.outdir,proof=args.proof,render=args.render,research_data_licence=args.research_data_licence)
    return 0
