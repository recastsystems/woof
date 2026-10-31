#!/usr/bin/env python3
"""Real whole microphysics handoff; fork mpas_atmphys_interface.F:643-1625 at 1d6e370e."""
import hashlib,json
import numpy as np


def replay(root,prefix,target,library,manifest_path):
    import cupy as cp
    from rrfs_driver_dump_replay import authenticate,Records
    from woof.hex.mpasphys.rrfs.driver_rrfs import StrictPhysicsKernelCache,microphysics_from_mpas,microphysics_to_mpas
    source='src/core_atmosphere/physics/mpas_atmphys_interface.F'
    manifest_sha=authenticate(root,prefix,manifest_path,source=source)
    m=json.loads(manifest_path.read_text());p=json.loads((manifest_path.parent/'passivity.json').read_text());producer=json.loads((manifest_path.parent/'producer.json').read_text())
    if (not m['observer_passivity_proven'] or not p['bitwise_equal'] or p['different_words']!=0
        or any(hashlib.sha256((manifest_path.parent/name).read_bytes()).hexdigest()!=m[key] for name,key in
            [('passivity.json','passivity_receipt_sha256'),('producer.json','producer_attempt_sha256'),('input-request.json','input_request_sha256')])
        or producer['native_sha256']!=m['native_binary_sha256'] or producer['process_count']!=1
        or producer['observer_sha256']!=m['observer_sha256'] or producer['source_sha256']!=m['observer_source_sha256']
        or producer['input_request_sha256']!=m['input_request_sha256'] or any(r['exit']!=0 for r in producer['runs'].values())):
        raise ValueError('Microphysics whole native producer/passivity changed')
    pre={p.name[len(prefix+'.pre.'):-4]:p for p in root.glob(prefix+'.pre.*.bin')}
    post={p.name[len(prefix+'.post.'):-4]:p for p in root.glob(prefix+'.post.*.bin')}
    if not pre or set(pre)!=set(post):raise ValueError('Whole microphysics PRE/POST inventory differs')
    codec=Records(library);loaded={k:codec.load(p)[0] for k,p in pre.items()};nc=int(loaded['argument.ite']);nz=int(loaded['vars.kte']);tl=int(loaded['argument.time_lev'])
    if any(int(loaded[k])!=1 for k in ('argument.its','vars.kts','vars.jts','vars.jte')):
        raise ValueError('This real packet requires one complete native vertical tile')
    def key(pool,name):return pool+'.'+name+'.t'+str(tl if pool=='state' else 1)
    def device(name):return cp.asarray(np.ascontiguousarray(loaded[name][...,:nc]))
    fields={n:device(key('state',n)) for n in ('theta_m','rho_zz')}
    fields.update({n:device(key('mesh',n)) for n in ('zz','zgrid')})
    fields.update({n:device(key('diag',n)) for n in ('exner','pressure_base','pressure_p')})
    scalar_key=key('state','scalars');scalars=device(scalar_key)
    indices={name.removeprefix('state.index_'):int(value)-1 for name,value in loaded.items() if name.startswith('state.index_')}
    species={name:cp.ascontiguousarray(scalars[index]) for name,index in indices.items()}
    options={name.removeprefix('block.configs.'):value for name,value in loaded.items() if name.startswith('block.configs.')}
    for n in list(options):
        if n.startswith('config_tempo_'):options[n]=bool(options[n])
    prepared={name.removeprefix('vars.'):cp.asarray(np.ascontiguousarray(value[:nc,:nz,0].T if value.ndim==3 else value[:nc,0]))
        for name,value in loaded.items() if name.startswith('vars.') and isinstance(value,np.ndarray) and value.ndim>=2}
    tendencies={name.split('.')[1]:device(name) for name in loaded if name.startswith('tend_physics.') and '.t' in name and loaded[name].ndim}
    replacement={};cache=StrictPhysicsKernelCache()
    def replace(name,value):
        old=cp.asarray(np.ascontiguousarray(loaded[name]))
        if name.startswith('vars.'):
            if old.ndim==3:old[:nc,:nz,0]=value.T
            elif old.ndim==2:old[:nc,0]=value
            else:raise ValueError('module scratch rank differs')
        else:old[...,:nc]=value
        replacement[name]=cp.asnumpy(old)
    if prefix.startswith('boundary__microphysics_from_mpas.'):
        fields['w']=device(key('state','w'))
        diagnostics={name.split('.')[1]:device(name) for name in loaded if name.startswith('diag_physics.') and '.t' in name}
        outputs,rates=microphysics_from_mpas(fields=fields,species=species,diagnostics=diagnostics,options=options,kernel_cache=cache)
        for name,value in outputs.items():replace('vars.'+name,value)
        for name,value in rates.items():replace(key('tend_physics',name),value)
    elif prefix.startswith('boundary__microphysics_to_mpas.'):
        fields.update({n:device(key('diag',n)) for n in ('exner_base','rtheta_base','rtheta_p','surface_pressure')})
        outputs,new_species,diagnostics,rates=microphysics_to_mpas(fields=fields,species=species,prepared=prepared,
            previous_tendencies=tendencies,options=options,dt_dyn=float(loaded['vars.dt_dyn']),kernel_cache=cache,
            mp_top_level=int(loaded[key('tend_physics','mp_top_level')]) if key('tend_physics','mp_top_level') in loaded else None,
            previous_diagnostics={name.split('.')[1]:device(name) for name in loaded if name.startswith('diag_physics.') and '.t' in name})
        for name,value in outputs.items():
            pool='state' if name in ('theta_m','rho_zz') else 'mesh' if name=='zz' else 'tend' if name in ('rt_diabatic_tend','tend_sfc_pressure') else 'diag'
            replace(key(pool,name),value)
        new_scalars=cp.asarray(np.ascontiguousarray(loaded[scalar_key]))
        for name,value in new_species.items():new_scalars[indices[name],:nz,:nc]=value
        replacement[scalar_key]=cp.asnumpy(new_scalars)
        for name,value in diagnostics.items():replace(key('diag_physics',name),value)
        for name,value in rates.items():replace(key('tend_physics',name),value)
    else:raise ValueError('Unknown complete microphysics boundary')
    target.mkdir(parents=True,exist_ok=True);comparisons={}
    for name,path in pre.items():
        actual=target/post[name].name;codec.post(path,actual,replacement.get(name));comparisons[name]=codec.compare(post[name],actual)
    return {'schema':'r4.real-complete-microphysics-interface/v1','scope':'both complete handoffs; all captured allocated scratch and direct pool getters; one stated real TEMPO aerosol/hail-aware case; other selector branches not real-replayed',
        'bitwise_equal':all(c['bitwise_equal'] for c in comparisons.values()),'words':sum(c['words'] for c in comparisons.values()),
        'different_words':sum(c['different_words'] for c in comparisons.values()),'output_records':len(comparisons),
        'manifest_sha256':manifest_sha,'replaced_records':sorted(replacement),'comparisons':comparisons}
