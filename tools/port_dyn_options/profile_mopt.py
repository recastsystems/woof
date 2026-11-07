"""Interleave the optional stress output with its default mixing stage."""
from pathlib import Path
from dataclasses import replace
import atexit, functools, json, runpy, sys

def main():
    import cupy as cp
    import numpy as np
    from woof.core import dycore
    output=Path(sys.argv.pop(1)); events={'option':[], 'default':[]}; calls=0
    original=dycore.prepare_fixed_tendencies
    @functools.wraps(original)
    def observed(state,cfg,*args,**kwargs):
        nonlocal calls
        if cfg.m_opt != 1:return original(state,cfg,*args,**kwargs)
        if cfg.km_opt != 4 or cfg.diff_6th_opt or cfg.obs_nudge_opt:
            raise RuntimeError('This interleaving receipt is for the default km_opt=4 real case')
        calls+=1
        for label in (('option','default') if calls%2 else ('default','option')):
            start,end=cp.cuda.Event(),cp.cuda.Event();start.record()
            original(state,cfg if label=='option' else replace(cfg,m_opt=0),*args,**kwargs)
            end.record();events[label].append((start,end))
    dycore.prepare_fixed_tendencies=observed
    def finish():
        cp.cuda.get_current_stream().synchronize()
        stages={}
        for key,pairs in events.items():
            values=[float(cp.cuda.get_elapsed_time(a,b)) for a,b in pairs]
            if values:stages[key]=dict(calls=len(values),median_ms=float(np.median(values[3:])),all_ms=values)
        output.write_text(json.dumps(dict(method='CUDA events over prepare_fixed_tendencies; interleaved m_opt=0/1 on identical time-t fields; first three excluded; no prognostic mutation in this km_opt=4 stage',stages=stages),indent=2)+'\n')
    atexit.register(finish);sys.argv=['woof','go',*sys.argv[1:]];runpy.run_module('woof',run_name='__main__')
if __name__=='__main__':main()
