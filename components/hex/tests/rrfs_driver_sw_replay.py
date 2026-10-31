#!/usr/bin/env python3
"""Real whole shortwave handoff; fork mpas_atmphys_driver_radiation_sw.F:370-879 at 1d6e370e."""
import hashlib,json
import numpy as np


def replay(root,prefix,target,library,manifest_path):
    import cupy as cp
    from rrfs_driver_dump_replay import authenticate,Records
    from woof.hex.mpasphys.rrfs.driver_rrfs import StrictPhysicsKernelCache,radiation_sw_from_mpas,radiation_sw_to_mpas
    source='src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F'
    manifest_sha=authenticate(root,prefix,manifest_path,source=source)
    m=json.loads(manifest_path.read_text());p=json.loads((manifest_path.parent/'passivity.json').read_text());producer=json.loads((manifest_path.parent/'producer.json').read_text())
    if (not m['observer_passivity_proven'] or not p['bitwise_equal'] or p['different_words']!=0
        or any(hashlib.sha256((manifest_path.parent/name).read_bytes()).hexdigest()!=m[key] for name,key in
            [('passivity.json','passivity_receipt_sha256'),('producer.json','producer_attempt_sha256'),('input-request.json','input_request_sha256')])
        or producer['native_sha256']!=m['native_binary_sha256'] or producer['process_count']!=1
        or producer['observer_sha256']!=m['observer_sha256'] or producer['source_sha256']!=m['observer_source_sha256']
        or producer['input_request_sha256']!=m['input_request_sha256'] or any(r['exit']!=0 for r in producer['runs'].values())):
        raise ValueError('shortwave whole native producer/passivity changed')
    pre={p.name[len(prefix+'.pre.'):-4]:p for p in root.glob(prefix+'.pre.*.bin')}
    post={p.name[len(prefix+'.post.'):-4]:p for p in root.glob(prefix+'.post.*.bin')}
    if not pre or set(pre)!=set(post):raise ValueError('Whole shortwave PRE/POST inventory differs')
    codec=Records(library);loaded={k:codec.load(p)[0] for k,p in pre.items()}
    options={name.removeprefix('block.configs.'):value for name,value in loaded.items() if name.startswith('block.configs.')}
    pools={pool:{name.split('.')[1]:cp.ascontiguousarray(cp.asarray(value)) for name,value in loaded.items()
                 if name.startswith(pool+'.') and '.t' in name}
           for pool in ('mesh','state','diag_physics','atm_input','sfc_input','tend_physics')}
    prepared={name.removeprefix('vars.'):cp.ascontiguousarray(cp.asarray(value)) for name,value in loaded.items()
              if name.startswith('vars.') and isinstance(value,np.ndarray) and value.ndim>=1}
    bounds={name:int(loaded['argument.'+name]) for name in ('its','ite')}
    bounds.update({name:int(loaded['vars.'+name]) for name in ('kts','kte','jts','jte')})
    dimensions={name:int(loaded['vars.'+name]) for name in ('num_oznlevels','num_aerlevels','num_aerosols','num_months','cam_abs_dim1','cam_abs_dim2')}
    common=dict(pools=pools,prepared=prepared,rrtmg_only=m.get("table_key")=="mpasphys_rrfs_44c0849",**bounds)
    replacement={}
    if prefix.startswith('boundary__radiation_sw_from_mpas.'):
        result,output,aer_opt=radiation_sw_from_mpas(**common,options=options,dimensions=dimensions,
            xtime_s=float(loaded['argument.xtime_s']),kernel_cache=StrictPhysicsKernelCache(),constants={name.removeprefix("constants."):value for name,value in loaded.items() if name.startswith("constants.")})
        replacement={'vars.'+name:cp.asnumpy(value) for name,value in result.items()}
        replacement['vars.aer_opt']=np.int32(aer_opt)
    elif prefix.startswith('boundary__radiation_sw_to_mpas.'):
        output=radiation_sw_to_mpas(**common,options=options)
    else:raise ValueError('Unknown complete shortwave boundary')
    for name in pre:
        parts=name.split('.')
        if len(parts)==3 and parts[0] in output and parts[1] in output[parts[0]]:
            replacement[name]=cp.asnumpy(output[parts[0]][parts[1]])
    target.mkdir(parents=True,exist_ok=True);comparisons={}
    for name,path in pre.items():
        actual=target/post[name].name;codec.post(path,actual,replacement.get(name));comparisons[name]=codec.compare(post[name],actual)
    return {'schema':'r4.real-complete-sw-interface/v1','scope':'both complete handoffs; all captured allocated scratch and direct pool getters; one stated real RRTMG case; other selector branches not real-replayed',
        'bitwise_equal':all(c['bitwise_equal'] for c in comparisons.values()),'words':sum(c['words'] for c in comparisons.values()),
        'different_words':sum(c['different_words'] for c in comparisons.values()),'output_records':len(comparisons),
        'manifest_sha256':manifest_sha,'native_capture_host':producer['host'],'resolved_options':{k:(v.item() if isinstance(v,np.ndarray) and v.size==1 else v) for k,v in options.items()},'tile_bounds':bounds,'dimensions':dimensions,'replaced_records':sorted(replacement),'comparisons':comparisons}
