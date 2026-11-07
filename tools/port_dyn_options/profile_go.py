"""Interleave default/option transport launches inside a real go forecast."""
import atexit
import functools
import json
from pathlib import Path
import runpy
import sys


def main():
    import cupy as cp
    import numpy as np
    from woof.core import advection, moist, dycore
    from dataclasses import replace
    import copy
    output = Path(sys.argv.pop(1))
    events, calls = {}, {}
    def wrap(module, name):
        original = getattr(module, name)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            if not kwargs.get('weno', False) and kwargs.get('vorder', 3) == 3 and kwargs.get('horder', 5) == 5:
                return original(*args, **kwargs)
            count = calls.get(name, 0)
            calls[name] = count + 1
            alternate = list(args)
            if name == 'launch_pd_fluxes':
                for i in range(10,16): alternate[i] = args[i].copy()
            else: alternate[4] = args[4].copy()
            default = dict(kwargs, vorder=3)
            if name != 'launch_pd_fluxes': default['weno'] = False
            if name == 'launch_flux_div_scalar': default.pop('weno',None)
            if 'horder' in kwargs: default['horder'] = 5
            for label in (('default', 'option') if count % 2 else ('option', 'default')):
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                if label == 'option': original(*args, **kwargs)
                else: original(*alternate, **default)
                end.record()
                events.setdefault(name + '.' + label, []).append((start, end))
        setattr(module, name, observed)
    for name in ('launch_flux_div_scalar','launch_flux_div_u', 'launch_flux_div_v', 'launch_flux_div_w'):
        wrap(advection,name)
    # moist imports the scalar launcher by value before the wrappers exist.
    moist.launch_flux_div_scalar=advection.launch_flux_div_scalar
    wrap(moist,'launch_pd_fluxes')
    def wrap_phi(name):
        original = getattr(dycore, name)
        @functools.wraps(original)
        def observed(state, cfg, *args, **kwargs):
            if cfg.h_sca_adv_order == 5 or getattr(cfg, '_legacy_h_sca', False):
                return original(state, cfg, *args, **kwargs)
            count = calls.get(name, 0); calls[name] = count + 1
            alternate = copy.copy(state); alternate.rph_t = state.rph_t.copy()
            default = replace(cfg, h_sca_adv_order=5, _legacy_h_sca=False)
            for label in (('default','option') if count % 2 else ('option','default')):
                start,end = cp.cuda.Event(),cp.cuda.Event(); start.record()
                if label == 'option': original(state,cfg,*args,**kwargs)
                else: original(alternate,default,*args,**kwargs)
                end.record(); events.setdefault(name+'.'+label,[]).append((start,end))
        setattr(dycore,name,observed)
    for name in ('_launch_slow_geopotential','_launch_slow_geopotential_faces'):
        wrap_phi(name)
    def finish():
        if not events: return
        cp.cuda.get_current_stream().synchronize()
        records = {}
        for key, pairs in events.items():
            values = [float(cp.cuda.get_elapsed_time(a,b)) for a,b in pairs]
            warm = values[3:] or values
            records[key] = dict(calls=len(values), median_ms=float(np.median(warm)),
                                mean_ms=float(np.mean(warm)), all_ms=values)
        output.write_text(json.dumps(dict(method='CUDA events; interleaved launch order; '
            'default arm writes a separate tendency copy; first three calls excluded',
            stages=records), indent=2) + '\n')
    atexit.register(finish)
    sys.argv = ['woof', 'go', *sys.argv[1:]]
    runpy.run_module('woof', run_name='__main__')

if __name__ == '__main__':
    main()
