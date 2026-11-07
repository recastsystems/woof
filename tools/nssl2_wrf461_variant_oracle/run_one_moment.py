"""Native one-moment replay through the same production CUDA entry."""
import hashlib
import numpy as np
import cases
from run_oracle import OUTPUT_FIELDS,SURFACE_FIELDS


def run_mode(mode,resolved,nsteps,dt,it0,*,input_case=None,surface_initial=None,diagnostic_initial=None):
    import cupy as cp
    from woof.core import nssl1
    state,env=cases.build_case(mode) if input_case is None else input_case
    nx,nz=state['theta'].shape
    cell=lambda a:cp.asarray(np.ascontiguousarray(a.T[:,None,:]))
    device={n:cell(state[n]) for n in nssl1.STATE_FIELDS}
    environment={n:cell(env[n]) for n in nssl1.ENV_FIELDS}
    diagnostic={n:cp.full((nz,1,nx),-999.,cp.float32) if diagnostic_initial is None else cell(diagnostic_initial[n]) for n in nssl1.OUTPUT_FIELDS[20:]}
    surface={n:cp.zeros((1,nx),cp.float32) if surface_initial is None else cp.asarray(surface_initial[n][None,:]) for n in SURFACE_FIELDS}
    cells=cp.empty((27 if resolved.hail else 22,nz,1,nx),cp.float32)
    from woof.core.nssl2_shape import resolved_shapes
    shapes=resolved_shapes(mode.alphah,mode.alphahl,mode.icdx,mode.icdxhl)
    compiled=(nssl1.kernels(nz,resolved.hail) if mode.cccn==0.5e9 and shapes==(0.,1.) else nssl1.kernels(nz,resolved.hail,abs(mode.cccn),shapes))
    trajectory={n:[] for n in (*OUTPUT_FIELDS,*SURFACE_FIELDS)}
    for step in range(nsteps):
        nssl1.launch(device,environment,diagnostic,surface,cells,dt,hail=resolved.hail,compiled=compiled)
        cp.cuda.get_current_stream().synchronize()
        for n in OUTPUT_FIELDS:
            value=diagnostic[n] if n in diagnostic else device[n]
            trajectory[n].append(cp.asnumpy(value)[:,0,:].T.copy())
        for n in SURFACE_FIELDS:trajectory[n].append(cp.asnumpy(surface[n])[0].copy())
    out={n:np.stack(v) for n,v in trajectory.items()}
    for n in cases.STATE_FIELDS:
        previous=np.concatenate((state[n][None,...],out[n][:-1]),axis=0)
        out['tendency_'+n]=(out[n]-previous)/np.float32(dt)
    return out,{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in {**state,**env}.items()}
