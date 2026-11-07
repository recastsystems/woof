"""Complete periodic moist RK state and Rust histories against singles."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("use_theta_m", [0, 1])
@pytest.mark.parametrize('count', (1, 4, 10))
@pytest.mark.parametrize('terrain,mapped', ((False, False), (False, True), (True, False), (True, True)))
def test_two_moist_rk_steps_and_native_histories_match_original(count, terrain, mapped, tmp_path, use_theta_m):
    import cupy as cp
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from woof.core import dycore
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble.batch_state import PreparedHostMember, BatchedDomainState, SHARED_STATE_CANDIDATES
    from woof.ensemble.batch_moist_dycore import workspace_specs, required_scratch_slots, prepare_moist_step
    from woof.ensemble.batch_dycore import member_domain_view
    from woof.io.wrfout import WrfoutWriter, state_frame
    from woof.io import nc_writer_bridge
    assert nc_writer_bridge.unavailable_reason() is None, 'Rust history output is required'
    component, references, source_cfg, _ = _pack_physical(count, moist=True, terrain=terrain, mapped=mapped)
    cfg = replace(source_cfg, use_theta_m=use_theta_m, km_opt=1, diff_opt=2, khdif=0., kvdif=0., diff_6th_opt=0)
    shapes, extras = state_array_shapes(cfg), workspace_specs(cfg)
    inputs=[]
    for scalar in references:
        if use_theta_m:
            from woof.core.constants import RVOVRD
            base = scalar.thb if scalar.thb.ndim == 3 else scalar.thb[:, None, None]
            scalar.thp[...] = (base + scalar.thp) * (np.float32(1) + np.float32(RVOVRD) * scalar.qv) - base
            scalar.thp0[...] = scalar.thp
            scalar.use_theta_m = use_theta_m
        arrays={name: cp.asnumpy(getattr(scalar,name)) for name in shapes}
        arrays.update({spec.name:np.zeros(spec.shape,spec.dtype) for spec in extras})
        controls={'physics','lateral_boundaries','_scratch','_scratch_arena','_host_setup_state','_phb_host'}
        scalars={name:value for name,value in vars(scalar).items() if name not in shapes and name not in controls}
        snapshot={'ticks':0,'step_ticks':3,'tick_den':1,'run_ticks':30,'step_count':0,
                  'dt_fp32':np.float32(cfg.dt),'dtbc_fp32':np.float32(0)}
        inputs.append(PreparedHostMember(cfg, arrays, scalars, snapshot,
            scratch={name:cp.asnumpy(value) for name,value in scalar._scratch.items()},phb_host=scalar._phb_host))
    del component
    shared=tuple(sorted(SHARED_STATE_CANDIDATES & shapes.keys()))
    state=BatchedDomainState.from_prepared(inputs,array_module=cp,available_bytes=2**30,
        shared_fields=shared,extra_specs=extras,scratch_slots=required_scratch_slots(cfg))
    advance=prepare_moist_step(state)
    attrs={'START_DATE':'2026-10-02_00:00:00','DT':np.float32(cfg.dt)}
    def write(path,view,valid):
        fields=state_frame(view,include_diagnostic_pressure=True)
        with WrfoutWriter(path,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,dx=cfg.dx,dy=cfg.dy,
                         global_attrs=attrs,field_schema=fields,engine='rust') as writer:
            writer.write_frame(valid,fields)
    for index in range(2):
        advance()
        cp.cuda.get_current_stream().synchronize()
        for member,scalar in enumerate(references):
            dycore.step(scalar,cfg,acoustic=True)
            for name in shapes:
                actual=cp.asnumpy(state.member_view(name,member)).view(np.uint32)
                expected=cp.asnumpy(getattr(scalar,name)).view(np.uint32)
                assert actual.tobytes()==expected.tobytes(),(index,member,name)
            assert state.elapsed_seconds==scalar.elapsed_seconds
            valid=('2026-10-02_00:00:03','2026-10-02_00:00:06')[index]
            batch_path=tmp_path/f'batch-{index}-{member}.nc'
            scalar_path=tmp_path/f'single-{index}-{member}.nc'
            write(batch_path,member_domain_view(state,member),valid)
            write(scalar_path,scalar,valid)
            assert batch_path.read_bytes()==scalar_path.read_bytes(),(index,member,'history')
        assert state.clock['step_count']==index+1
        assert state.clock['ticks']==(index+1)*state.clock['step_ticks']
