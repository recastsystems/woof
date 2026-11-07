"""Consume native &fdda observation controls without touching grid FDDA."""
from dataclasses import fields
from woof.config import RunConfig

NATIVE_KEYS = tuple(f.name for f in fields(RunConfig)
                    if f.name.startswith('obs_') and f.name not in (
                        'obs_inputfile','obs_domain_id','obs_parent_grid_ratio','obs_levelidn',
                        'obs_truelat1','obs_truelat2','obs_stand_lon')) + ('max_obs','fdda_start','fdda_end')


def consume(section, count):
    from woof.wrf_namelist_registry import wrf_namelist_keys
    registry=wrf_namelist_keys()
    columns={}
    for key in NATIVE_KEYS:
        spec=registry[('fdda',key)]
        kind=spec['type']
        if kind=='logical':
            default=str(spec['default']).lower() in ('.true.','true')
        else:
            default=(int if kind=='integer' else float)(spec['default'])
        values=section.registry_col(key,count,default) if spec['nentries']=='max_domains' else [section.scalar(key,default)]*count
        for value in values:
            if kind=='logical' and type(value) is not bool or kind=='integer' and type(value) is not int or kind=='real' and (isinstance(value,bool) or not isinstance(value,(float,int))):
                raise ValueError(f'&fdda/{key} must contain {kind} values')
        columns[key]=values
    # fddaobs_init: an omitted nested window takes the coarse-grid value.
    for n in range(1,count):
        if columns['obs_twindo'][n]==0:
            columns['obs_twindo'][n]=columns['obs_twindo'][0]
    from types import SimpleNamespace
    from .config import validate
    for n in range(count):
        validate(SimpleNamespace(**{key:values[n] for key,values in columns.items()},
            obs_domain_id=n+1,obs_parent_grid_ratio=1,map_proj=1,
            use_adaptive_time_step=False))
    return columns
