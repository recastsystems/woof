#!/usr/bin/env python3
"""Compile the production translation units with a real HIP toolchain.

No GPU context is used. Each successful row records source and ELF digests;
failed rows retain the complete compiler diagnostics. A failed or omitted
translation unit cannot make this command succeed.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
from woof.hip_backend import compile_aot


def production_loader_unit(name):
    """Capture the complete source/options passed by a device-free loader."""
    import contextlib,sys,types
    from unittest.mock import patch
    # Import numerical dependencies before installing the launch sink. CuPy
    # imports NumPy too; replacing it during NumPy's first import is unsafe.
    if name=='snow_papers':
        from woof.core import snow_papers,ieee_module
    elif name=='ocean_pwp':
        from woof.hex.mpasphys import ocean_pwp_v842
    else:raise ValueError('Unknown production loader '+name)
    rows=[]
    def capture(source,*args,options=(),**kwargs):
        rows.append((source,tuple(options)))
    fake=types.SimpleNamespace(RawKernel=capture,
        cuda=types.SimpleNamespace(Device=lambda device:contextlib.nullcontext()))
    with patch.dict(sys.modules,cupy=fake):
        if name=='snow_papers':
            def module(source,options,*args,**kwargs):capture(source,options=options)
            with patch.object(ieee_module,'compile_module',module):
                snow_papers._module.__wrapped__(0)
        elif name=='ocean_pwp':
            ocean_pwp_v842._kernel()
    if len(rows)!=1:raise ValueError('Expected one production unit for '+name)
    return rows[0]


def generated_units():
    """Record real source generators on host arrays, without executing kernels.

    The launch sink only inventories source. These calls are not numerical
    tests; numerical qualification uses the separate whole-program runners.
    """
    import inspect,sys,types
    import numpy as np
    from woof.core import noahmp_kernel_sources as compiler
    from woof.core import noahmp_slab_libm as slab
    from woof.core import noahmp_column_slab as column
    from woof.core import noahmp_energy_slab as energy
    from woof.core import mynn_pbl_gpu,ruc_gpu
    rows=[]
    def capture(source,name=None,options=(),**kwargs):
        rows.append(dict(name='generated/'+(name or str(len(rows))),source=source,
                         options=tuple(options) or ('-std=c++17','--fmad=false')))
        return lambda *args,**kwargs:None
    fake=types.ModuleType('cupy')
    fake.__dict__.update({k:getattr(np,k) for k in dir(np) if not k.startswith('__')})
    fake.asnumpy=np.asarray;fake.ascontiguousarray=np.ascontiguousarray
    fake.cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(getDevice=lambda:0))
    fake.RawKernel=capture
    old=sys.modules.get('cupy');old_compile=compiler.compile_generated_slab_kernel
    old_slab_compile=slab.compile_generated_slab_kernel
    old_mynn,old_ruc=mynn_pbl_gpu.cp,ruc_gpu.cp
    try:
        sys.modules['cupy']=fake;compiler.compile_generated_slab_kernel=capture
        slab.compile_generated_slab_kernel=capture
        mynn_pbl_gpu.cp=ruc_gpu.cp=fake
        scalar=np.ones(3,np.float32);layers=np.ones((3,4),np.float32)
        slab.copy_slab_slots(np.empty((3,8),np.float32),[(0,scalar),(1,layers)])
        slab.split_slab_slots(np.empty((3,8),np.float32),{'a':(0,1),'b':(1,4)})
        j=np.array([0,1,1],np.int64);i=np.array([0,0,1],np.int64)
        slab.scatter_slab_fields(j,i,[(np.empty((2,2),np.float32),scalar),
                                    (np.empty((4,2,2),np.float32),layers)])
        slab.gather_slab_fields(j,i,{'a':np.ones((2,2),np.float32),
                                    'b':np.ones((4,2,2),np.float32)},(2,2))
        slab.take_slab_rows(np.array([0,2],np.int64),{'a':scalar,'b':layers},3)
        column._vege_spread_kernels()
        nsnow,nsoil=energy.NSNOW,energy.NSOIL
        out={k:np.ones((3,width),np.int32 if k=='imelt' else np.float32)
             for k,width in [('hcpct',nsnow+nsoil),('imelt',nsnow+nsoil),
                             ('snicev',nsnow),('snliqv',nsnow),('epore',nsnow)]}
        out.update({k:scalar.copy() for k in ('fsrv','fsrg','bgap','wgap','eflxb')})
        energy.energy_seg4(dict(isnow=np.zeros(3,np.int32),nroot=np.ones(3,np.int32),
                               cosz=scalar),{},dict(btrani=layers,dt_soil=np.float32(1)),out)
        for predicate in mynn_pbl_gpu._VALIDATION_PREDICATES:
            inspect.unwrap(mynn_pbl_gpu._validation_batch_kernel)(predicate,3)
            rows[-1]['name']+='_'+predicate
        inspect.unwrap(ruc_gpu._validation_scan_kernel)(3)
    finally:
        if old is None:sys.modules.pop('cupy',None)
        else:sys.modules['cupy']=old
        compiler.compile_generated_slab_kernel=old_compile
        slab.compile_generated_slab_kernel=old_slab_compile
        mynn_pbl_gpu.cp,ruc_gpu.cp=old_mynn,old_ruc
    from woof.ensemble.batch_ruc import packed_ruc_source
    for soil in (6,9):
        rows.append(dict(name='generated/packed_ruc_'+str(soil),source=packed_ruc_source(soil),options=('-std=c++17',)))
    return rows


def generated_literal(tree,literal):
    import ast,collections
    function=next(f for f in ast.walk(tree) if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))
                  and any(n is literal for n in ast.walk(f)))
    defaults=collections.defaultdict(lambda:0)
    env=dict(nsoil=4,nsnow=3,NSOIL=4,NSNOW=3,NLAY=7,n=1,nx=1,
             s=defaults,g=defaults,e=defaults,v=defaults,b=defaults)
    pure={'str','range','enumerate','zip','len','tuple','list','dict','sorted','int','float'}
    methods={'join','replace','items','values','keys'}
    def safe(expr):
        return all((isinstance(n.func,ast.Name) and n.func.id in pure) or
                   (isinstance(n.func,ast.Attribute) and n.func.attr in methods)
                   for n in ast.walk(expr) if isinstance(n,ast.Call))
    final=None
    nodes=sorted((n for n in ast.walk(function) if isinstance(n,(ast.Assign,ast.AugAssign))),key=lambda n:(n.lineno,n.col_offset))
    for statement in nodes:
        if not safe(statement.value):continue
        try:
            exec(compile(ast.Module(body=[statement],type_ignores=[]),'<production-generator>','exec'),env)
        except (NameError,KeyError,TypeError,AttributeError,ValueError):continue
        for key in ('source','code'):
            value=env.get(key)
            if isinstance(value,str) and literal.value[:40] in value:final=value
    if final is None or any(token in final for token in ('ARGS','BODY','NSOIL','NSNOW','NLAY')):
        raise ValueError('Unresolved production generator '+function.name)
    return final


def hex_declared_units(recipes):
    """Production declarations plus complete kernels outside forecast recordings."""
    import ast,importlib,re
    from woof.hex.cuda_backend.aot_build import inventory as declarations
    rows=[]
    for mode in ('strict','fast'):
        for i,row in enumerate(declarations(numerics=mode,catalogs=False)):
            rows.append(dict(name='hex-extra/declared/'+mode+'/'+str(i),
                             source=row['source'],options=tuple(row.get('options',()))))
    known='\n'.join(row['source'] for row in [*rows,*json.loads(Path(recipes).read_text())])
    root=Path(recipes).parent.parent
    global_pattern=r'__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)\s*\('
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
    from woof.hex.mynn_glibc_math import mynn_glibc_source
    for path in sorted(root.rglob('*')):
        if path.suffix=='.py':
            tree=ast.parse(path.read_text())
            values=[(n.lineno,n.value) for n in ast.walk(tree)
                    if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        elif path.suffix=='.cu':values=[(1,path.read_text())]
        else:continue
        relative=str(path.relative_to(root)); module=None
        for line,source in values:
            options=('-std=c++17','--fmad=false')
            names=re.findall(global_pattern,source)
            if not names or all(n in known for n in names):continue
            if source.count('{')!=source.count('}') or not re.search(global_pattern+r'[^;]*?\{',source,re.S):continue
            if path.suffix=='.py':
                module=module or importlib.import_module('hexcore.'+'.'.join(path.relative_to(root).with_suffix('').parts))
                if relative in ('cuda_acoustic_rrfs.py','iau_v842.py'):source=module._CUDA_SOURCE
                elif relative=='cuda_vertical_mixing_rrfs.py':source=module._SOURCE
                elif relative=='moving_balance_cuda.py':source=source.replace('LEVELS','55')
                elif relative=='moving_momentum_cuda.py':source=source.replace('ORDER',','.join(map(str,module._projection_order() or (0,1,2))))
                elif relative=='moving_math_cuda.py':
                    from woof.hex.mpasphys.diagnostics_v842 import native_diagnostic_source
                    source=native_diagnostic_source()+source
                elif relative=='mixing_v842_les.py':source=module._cuda_source()
                elif relative=='mpasphys/kain_fritsch_v842.py':source=module.shared_math_source()+module.CUDA_SOURCE
                elif relative=='mpasphys/wsm6_refl_start_v842.py':source=module.source(module.device_math_header(),55,numerics='strict')
                elif relative=='mpasphys/rrfs/myj.py':source=module.cuda_source(55)
                elif relative=='mpasphys/rrfs/ruc_lsm_8c9f1a8.py':source=module.shared_math_source()+module.CUDA_SOURCE
                elif relative=='mpasphys/rrfs/sf_mynn_rrfs.py':source=module.cuda_source()
                elif relative=='mpasdiag/convective_rrfs.py':source=module.device_source()
                elif relative=='mpasphys/oml_v842.py':
                    source=(Path(__import__('woof').__file__).parent/'core/kernels/glibc_trig_flt32.cuh').read_text()+'\n'+source
                elif relative=='mpasphys/ocean_pwp_v842.py':
                    source,options=production_loader_unit('ocean_pwp')
                elif relative=='mpasphys/moving_rates_batch_v842.py':
                    # Enumerate both public scalar widths used by this builder.
                    rows.append(dict(name='hex-extra/generated/pack_rate_float',source=source.replace('TYPE','float'),options=('--fmad=false',)))
                    source=source.replace('TYPE','double')
                elif relative=='mpasphys/rrfs/tempo_v303.py':
                    from woof.hex.mpasphys import resources_build as resources
                    source=module.cuda_source(resources.tempo_maths(resources._vendored_maths()),max_levels=60,numerics='strict')
                elif relative in ('mpasphys/rrfs/bl_mynnedmf_c942418.py','mpasphys/rrfs/bl_mynnedmf_c6d08709.py'):
                    source=CUDA_FTZ_HELPERS+mynn_glibc_source()+source
                elif relative=='mpasphys/rrfs/mp_nssl2m.py':
                    from woof.hex.cuda_backend.aot_build import physics_inventory
                    # The production composer lowers/splits this entry. Its
                    # complete compiled variants are in the declarations.
                    continue
            elif relative=='init_gpu/physics.cu':
                kernel_root=Path(__import__('woof').__file__).parent/'core/kernels'
                source='\n'.join((kernel_root/name).read_text() for name in
                                 ('glibc_flt32.cuh','glibc_trig_flt32.cuh'))+'\n'+source
            rows.append(dict(name='hex-extra/literal/'+relative+':'+str(line),source=source,
                             options=options))
            known+='\n'+source
    # This generic validator has both IEEE widths; the recorded dry forecast
    # only used float32. Capture its real generator without a device context.
    from woof.hex.cuda_backend import containers
    from types import SimpleNamespace
    old=containers._cp
    try:
        containers._cp=lambda:SimpleNamespace(RawKernel=lambda code,name,**kw:rows.append(
            dict(name='hex-extra/generated/'+name,source=code,options=('-std=c++17',))))
        containers._finite_word_kernel.__wrapped__(8)
    finally:containers._cp=old
    unique={}
    for row in rows:
        key=hashlib.sha256(json.dumps([row['source'],row['options']]).encode()).hexdigest()
        unique.setdefault(key,row)
    return list(unique.values())

def rust_units(package):
    """The six canonical Rust helper translation units, including local headers."""
    import re
    project=package.parent
    post=project/'tools/rustwx/crates/rw-post/kernels'
    paths=(
        'tools/hydro_world/src/cuda/hydrography.cu',
        'tools/rustwx/crates/rw-post/kernels/woof_post.cu',
        'tools/rustwx/crates/rw-grib2export/kernels/rrfs_upp.cu',
        'tools/rustwx/crates/fire-indices/src/nelson.cu',
        'tools/rustwx/crates/rustwx-render/src/rasterize_structured.cu',
        'tools/rustwx/vendor/bowecho/crates/simradar_cuda/kernels/p3_lut_segments.cu',
    )
    def expanded(path,stack=()):
        if path in stack:raise ValueError('Recursive kernel include '+str(path))
        def header(match):
            name=match.group(1)
            candidate=path.parent/name
            if not candidate.is_file():candidate=post/name
            if not candidate.is_file():raise ValueError('Unbound Rust kernel include '+name)
            text=expanded(candidate,(*stack,path))
            if '#pragma once' in text:
                guard='GPUWM_EMBEDDED_'+re.sub(r'[^A-Za-z0-9]','_',name).upper()
                text='\n#ifndef '+guard+'\n#define '+guard+'\n'+text.replace('#pragma once','')+'\n#endif\n'
            return text
        return re.sub(r'^\s*#include\s+"([^"]+)"',header,path.read_text(),flags=re.M)
    return [dict(name='rust/'+str(Path(name).relative_to('tools')),source=expanded(project/name),
                 options=('-std=c++17','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'))
            for name in paths]


def inventory(hex_recipes=None):
    # These are AMD compiler jobs even when this host imports CUDA CuPy.
    # Select the canonical AMD source-composer branch (not CUDA-only CCCL
    # barrier optimizations) without initializing a device or an array backend.
    previous = os.environ.get('WOOF_BACKEND')
    os.environ['WOOF_BACKEND'] = 'hip'
    try:
        return _inventory(hex_recipes)
    finally:
        if previous is None:
            os.environ.pop('WOOF_BACKEND', None)
        else:
            os.environ['WOOF_BACKEND'] = previous


def _inventory(hex_recipes=None):
    from woof.core import kernels
    root=Path(kernels.__file__).parent
    rows=[]
    from woof.core.noahmp_kernel_sources import NOAHMP_TRANSLATION_UNITS, runtime_unit
    from woof.core import p3_device,rrtmg_lw,rrtmg_sw,terrain_drag,urban_bem,gbmpbl,eeps_pbl,acm2pbl
    from woof.hydro import reservoirs
    from woof.core import sfire_atm,sfire_core,sfire_moisture,sfire_phys,sfire_spotting,sfire_wind
    composed={name:(lambda name=name:runtime_unit(name).source) for name in NOAHMP_TRANSLATION_UNITS}
    composed.update(acm2pbl=acm2pbl.module_source,gbmpbl=gbmpbl.compile_source,eeps=lambda:eeps_pbl.module_source(59),
        p3=p3_device.p3_source,p3_zdiag=p3_device.p3_reflectivity_source,
        rrtmg_lw=rrtmg_lw._gpu_source,terrain_drag=terrain_drag.module_source,
        urban_bep_bem=urban_bem.module_source,nextgen_reservoirs=reservoirs.kernel_source,
        sfire_atm=sfire_atm.module_source,sfire_core=sfire_core.source,
        sfire_moisture=sfire_moisture.module_source,sfire_phys=sfire_phys.module_source,
        sfire_spotting=sfire_spotting.module_source,sfire_wind=sfire_wind.module_source,
        ysu_default=lambda:kernels.module_source('ysu'),
        rrtmgp_rte_default=lambda:kernels.module_source('rrtmgp_rte'))
    def sw():
        _,defines=rrtmg_sw._pack_cuda_tables(rrtmg_sw.load_sw_tables())
        return defines+(root/'rrtmg_sw.cu').read_text()
    composed['rrtmg_sw']=sw
    lw_fragments={p.stem for p in root.glob('rrtmg_lw_*.cu')}
    # Units whose own loaders compose source and options differently from
    # kernels.module_source/module_options: use the production recipe.
    def _etanew():
        from woof.core import etanew
        return etanew.module_source(),etanew.module_options()
    def _no_ftz(name,defines):
        from woof.core.kernels.no_ftz import module_recipe_no_ftz
        return module_recipe_no_ftz(name,defines)
    recipes=dict(etanew=_etanew,
        morrison_aero=lambda:_no_ftz('morrison_aero',(('MA_KMAX',50),('MA_T0_MILLIKELVIN',300000))),
        morrison_aero_refl=lambda:_no_ftz('morrison_aero_refl',(('REFL_KMAX',50),)),
        vortex_init=lambda:((root/'vortex_init.cu').read_text(),('--std=c++17',)))
    for path in sorted(root.glob('*.cu')):
        if path.stem in lw_fragments:continue # Included by the production LW composer above.
        if path.stem in recipes:
            try:
                source,options=recipes[path.stem]()
                rows.append(dict(name='arwen/'+path.stem,source=source,options=tuple(options)))
            except Exception as e:
                rows.append(dict(name='arwen/'+path.stem,error=str(e)))
            continue
        try:
            if path.stem=='snow_papers':
                source,options=production_loader_unit('snow_papers')
                rows.append(dict(name='arwen/'+path.stem,source=source,options=options))
                continue
            source=composed[path.stem]() if path.stem in composed else kernels.module_source(path.stem)
            rows.append(dict(name='arwen/'+path.stem,source=source,
                             options=acm2pbl.MODULE_OPTIONS if path.stem=='acm2pbl' else kernels.module_options(path.stem)))
        except Exception as e:
            rows.append(dict(name='arwen/'+path.stem,error=str(e)))
    # Non-core package kernels have self-contained entry points.
    package=root.parent.parent
    for path in sorted(package.rglob('*.cu')):
        if root in path.parents: continue
        source=path.read_text()
        relative=str(path.relative_to(package))
        if relative=='fddaobs/forecast.cu':
            # The ordinary forecast loader prepends the ABI Frame definition.
            source=path.with_name('kernels.cu').read_text()+'\n'+source
        if relative in ('ocean/waves/physics.cu','ocean/waves/diagnostics.cu','ocean/waves/budget.cu'):
            if relative!='ocean/waves/physics.cu':continue
            source='\n'.join(path.with_name(n).read_text() for n in
                             ('kernels.cu','physics.cu','diagnostics.cu','budget.cu'))
        if relative=='hydro/kernels/routed_replay.cu':
            from woof.hydro.routed_init import replay_source
            source=replay_source(True)
        if relative=='hydro/pet.cu':
            source=(root/'glibc_flt64.cuh').read_text()+'\n#define PET_STRICT 1\n'+source
            for scalar in ('float','double'):
                rows.append(dict(name='arwen/'+relative+'/'+scalar,source=source,
                    options=('-std=c++17','--fmad=false','-DFORCING_TYPE='+scalar)))
            continue
        if relative=='data/mynn471/model.cu':
            # Compiled only inside mynn471_gpu.source(nz): the shared MYNN prefix
            # and math helpers are prepended, with the image route's controls.
            from woof.core.mynn471_gpu import source as mynn471_source
            from woof.numerics import is_strict
            exact=is_strict()
            rows.append(dict(name='arwen/'+relative,source=mynn471_source(50),
                options=('-std=c++17','--ftz=false','--fmad='+('false' if exact else 'true'),
                         '--prec-div=true','--prec-sqrt=true')+(('-DGPUWM_WRF_EXACT=1',) if exact else ())))
            continue
        if str(path.relative_to(package))=='hydro/noahowp/physics.cu':
            from woof.hydro.noahowp.math_source import preamble
            source=preamble('strict')+source
        rows.append(dict(name='arwen/'+str(path.relative_to(package)),source=source,options=('-std=c++17',)))
    # Python-embedded complete kernels are canonical source too. Fragments
    # assembled by generators are covered by recorded production recipes.
    import ast
    for path in sorted(package.rglob('*.py')):
        if any(part in ('backend','verify') for part in path.relative_to(package).parts):continue
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
            code=node.value
            if '__global__' not in code or len(code)<150:continue
            if not code.count('{') or code.count('{')!=code.count('}'):continue
            if code.lstrip().startswith('import '):continue
            if not __import__('re').search(r'__global__.*?void\s+\w+\s*\(',code,__import__('re').S):continue
            relative=str(path.relative_to(package))
            # Use production composers where the literal borrows its prelude.
            if relative=='core/noah_sfcdiags.py':
                from woof.core.noah_sfcdiags import _source
                code=_source()
            elif relative=='core/tempo_runtime.py':
                from woof.core.tempo_runtime import composed_source
                code=composed_source()
            elif relative in ('da/hydrometeor_analysis.py','da/radar_tten.py'):
                code=(root/'glibc_flt64.cuh').read_text()+'\n'+code
            elif relative=='core/storm_tracking.py':
                code=(root/'portable_libm64.cuh').read_text()+'\n'+code
            elif relative=='da/letkf_device.py':code=code.replace('RR','8')
            elif relative=='core/rrtmg_lw.py':code='#define NGPTLW 140\n'+code
            elif relative=='hydro/output.py':code=code.replace('REAL','double')
            elif relative=='core/spp_kernel_sources.py':
                from woof.core.spp_kernel_sources import specialized_source
                code=specialized_source('mynn_surface')
            elif relative=='core/noahmp_slab_libm.py':
                if 'ARGS' in code or 'NSOIL' in code:
                    code=generated_literal(tree,node)
                elif 'noahmp_tanhf_glibc243' in code:
                    code=(root/'sfire_libm.cuh').read_text()+'\n'+code
            elif relative=='core/nssl1_coefficients.py':
                # This literal is appended to the one-moment source by its
                # real loader. Capture that loader, including lookup tables.
                from woof.core import nssl1
                import inspect,types
                captured=[];original=kernels._load_module_preserving_subnormals
                def capture(name,source,options):
                    captured.append((source,options))
                    return types.SimpleNamespace(get_function=lambda name:lambda *args:None)
                try:
                    kernels._load_module_preserving_subnormals=capture
                    inspect.unwrap(nssl1.kernels)(59,True,shapes=(0.0,0.5))
                finally:kernels._load_module_preserving_subnormals=original
                assert len(captured)==1
                code,options=captured[0]
                rows.append(dict(name='generated/nssl1_coefficients',source=code,options=options))
                continue
            rows.append(dict(name='inline/'+relative+':'+str(node.lineno),source=code,options=('-std=c++17','--fmad=false')))

    rows.extend(generated_units())
    rows.extend(rust_units(package))
    from woof.numerics import effective_options
    for row in rows:
        if 'options' in row:row['options']=effective_options(row['options'])
    if hex_recipes:
        rows.extend(hex_declared_units(hex_recipes))
        seen=set()
        for row in json.loads(Path(hex_recipes).read_text()):
            # The CUDA include roots are wheel-local and have no meaning for
            # AMD. CuPy's architecture-independent headers are retained.
            import cupy
            site=str(Path(cupy.__file__).parent.parent)
            opts=tuple(o.replace('@SITE@',site) for o in row.get('options',())
                       if not ('nvidia/' in o or '/_cccl/' in o))
            key=hashlib.sha256(json.dumps([row['source'],opts]).encode()).hexdigest()
            if key in seen:continue
            seen.add(key)
            rows.append(dict(name='hex/'+key,source=row['source'],options=opts))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hex-recipes',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arch',action='append',default=[])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--only')
    p.add_argument('--cupy-include',type=Path,help='Use the target wheel headers when cross-building off-card')
    p.add_argument('--rawmodule-variants',action='store_true',help='Also compile the CuPy CUDA RawModule FTZ boundary policy')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=inventory(a.hex_recipes)
    # Declarations use the same portable include placeholders as recordings.
    # CUDA CCCL/toolkit roots are not HIP dependencies; retain CuPy headers.
    import cupy
    site=str(Path(cupy.__file__).resolve().parent.parent)
    for row in rows:
        row['options']=tuple(o.replace('@SITE@',site) for o in row.get('options',())
                             if not ('nvidia/' in o or '/_cccl/' in o))
        if '#include <cupy/' in row['source'] or '#include "cupy/' in row['source']:
            row['options']+=('-I'+site+'/cupy/_core/include',)
        if a.cupy_include:
            row['options']=tuple(o.replace(site+'/cupy/_core/include',str(a.cupy_include.resolve()))
                                 for o in row['options'])
    if a.rawmodule_variants:
        rows += [dict(row,name=row['name']+'/rawmodule',options=(*row.get('options',()),'-ftz=true'))
                 for row in rows if 'error' not in row]
    if a.only:rows=[r for r in rows if a.only in r['name']]
    jobs=[(r,t) for r in rows for t in (a.arch or ['gfx942','gfx1100'])]
    def compile(job):
        row,target=job;name=row['name'].replace('/','_')+'-'+target
        start=time.monotonic()
        try:
            if 'error' in row:raise ValueError(row['error'])
            # Inventory rows already bind their strict/fast compiler options.
            # Never reinterpret a fast declaration with this process's policy.
            blob,receipt=compile_aot(row['source'],row['options'],target,output=a.output/(name+'.hsaco'))
            result=dict(name=row['name'],arch=target,ok=True,bytes=len(blob),**{k:receipt[k] for k in ['source_sha256','translated_sha256','image_sha256','options']})
        except Exception as e:
            log=a.output/(name+'.log');log.write_text(str(e))
            result=dict(name=row['name'],arch=target,ok=False,error=str(e)[:150],log=str(log))
        result['seconds']=time.monotonic()-start
        print(json.dumps(result),flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:results=list(pool.map(compile,jobs))
    report=dict(units=len(rows),jobs=len(jobs),passed=sum(r['ok'] for r in results),failed=sum(not r['ok'] for r in results),results=results)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    return int(bool(report['failed']))

if __name__=='__main__':raise SystemExit(main())
