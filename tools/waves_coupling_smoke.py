"""One bounded pre-case device smoke, not forecast-skill evidence."""
from __future__ import annotations
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import cupy as cp
from woof.ocean.interface import OceanGrid,Bathymetry
from woof.ocean.coupling import DeviceCoupler
from woof.ocean.exchange import BlockExchange
from woof.ocean.checkpoint import snapshot,restore
from woof.ocean.waves import create_solver


def main():
    wall=time.perf_counter();shape=(24,32);fine=(48,64)
    table=json.loads((Path(__file__).parents[1]/'woof/ocean/waves/partition_shapes.v1.json').read_text())
    lat=cp.broadcast_to(cp.linspace(24,26,shape[0],dtype=cp.float32)[:,None],shape).copy()
    lon=cp.broadcast_to(cp.linspace(-90,-87,shape[1],dtype=cp.float32)[None,:],shape).copy()
    grid=OceanGrid(shape[1],shape[0],8000.,8000.,lat,lon,cp.ones(shape,cp.float32),cp.zeros(shape,cp.float32))
    wet=cp.ones(shape,cp.float32);wet[:,0]=0
    depth=wet*100.;bathy=Bathymetry(-depth,depth,wet,cp.zeros_like(wet),'EGM2008','synthetic-device-smoke')
    fields={key:cp.full(fine,value,cp.float32) for key,value in
            (('u10',18.),('v10',5.),('psfc',100000.),('ust',.7),('znt',.002),('xland',2.),('xice',0.))}
    atmosphere={key:cp.full((1,*fine),value,cp.float32) for key,value in
                (('temperature',298.),('qv',.015),('p_interface',100000.))}
    before={key:value.view(cp.uint32).copy() for key,value in fields.items()}
    def make():
        solver=create_solver(grid,bathy,config={'backend':'curvilinear','map_factor':cp.ones(shape,cp.float32),
            'spectral':{'frequencies':36,'directions':36},'source_pack':'st4'})
        partitions=cp.zeros((shape[0]*shape[1],4,4),cp.float32)
        partitions[:,0,0]=2.;partitions[:,0,1]=8.;partitions[:,0,2]=250.
        partitions[:,1,0]=1.;partitions[:,1,1]=14.;partitions[:,1,2]=180.;partitions[:,1,3]=1.
        solver.initialize_partitions(partitions,valid_time=0.,shape_table=table)
        indices=solver.backend.wet_to_full[solver.backend.boundary_cells]
        boundary=cp.ascontiguousarray(partitions[indices])
        solver.set_boundary_partitions(boundary,boundary,time0=0.,time1=10800.,shape_table=table)
        solver.set_stations(cp.asarray([1,shape[0]*shape[1]-1],dtype=cp.int32))
        c=DeviceCoupler(grid,bathy,waves=solver,interval=300.,roughness_feedback=False)
        c.exchange=BlockExchange(fine,grid,factor=2,cosalpha=cp.ones(fine,cp.float32),
            sinalpha=cp.zeros(fine,cp.float32),area_weight=cp.ones(fine,cp.float32),xland=fields['xland'])
        c.handoff(u10=fields['u10'],v10=fields['v10'],psfc=fields['psfc'],ustar=fields['ust'],
                  temperature=atmosphere['temperature'][0],qv=atmosphere['qv'][0],time_seconds=0.,grid_relative=True)
        solver.initialize_surface(c.forcing)
        c.station_metadata=[{'id':'synthetic-west','lat':24.,'lon':-90.+3./31.,'cell':1},
                            {'id':'synthetic-east','lat':26.,'lon':-87.,'cell':shape[0]*shape[1]-1}]
        c.output_interval=600.;c.run_end=1200.
        return c
    original=make()
    output=os.environ.get('WOOF_WAVES_SMOKE_OUTPUT')
    if output:
        from woof.ocean.products import WaveOutput
        original.output=WaveOutput(Path(output)/'cf-frames',original)
        original.output.collect()
    for step in range(20):
        original.surface_roughness(fields['znt'],fields['ust'],ice=fields['xice'])
        original.atmosphere_step(fields,atmosphere,step*30.,30.)
    original.waves.collect()
    saved=snapshot(original)
    stored={key:cp.asnumpy(value) for key,value in saved['arrays'].items()}
    resumed=make();restore(resumed,saved['metadata'],stored,cp.asarray)
    immediate={key:int(cp.count_nonzero(value.view(cp.uint8)!=snapshot(resumed)['arrays'][key].view(cp.uint8)))
               for key,value in snapshot(original)['arrays'].items()}
    print(json.dumps({'restart_immediate_differing_bytes':immediate}),flush=True)
    assert not any(immediate.values()),immediate
    for step in range(20,40):
        for c in (original,resumed):c.atmosphere_step(fields,atmosphere,step*30.,30.)
    a=original.waves.collect();b=resumed.waves.collect()
    arrays_a=snapshot(original)['arrays'];arrays_b=snapshot(resumed)['arrays']
    differences={key:int(cp.count_nonzero(value.view(cp.uint8)!=arrays_b[key].view(cp.uint8))) for key,value in arrays_a.items()}
    magnitudes={key:float(cp.max(cp.abs(value-arrays_b[key]))) for key,value in arrays_a.items() if value.dtype.kind=='f'}
    print(json.dumps({'restart_continuation_differing_bytes':differences,'max_absolute_difference':magnitudes}),flush=True)
    atmosphere_differences={key:int(cp.count_nonzero(value.view(cp.uint32)!=before[key])) for key,value in fields.items()}
    # The core DIA and partition diagnostics use floating atomic sums. The
    # checkpoint must restore every word exactly; subsequent reductions are
    # compared at 64 FP32 eps times each array's maximum magnitude. This is
    # a short continuation check, not a claim of bitwise wave trajectories.
    relative={key:magnitudes[key]/max(float(cp.max(cp.abs(value))),1e-12)
              for key,value in arrays_a.items() if key in magnitudes}
    assert all(v<=64*float(cp.finfo(cp.float32).eps) for v in relative.values()),relative
    assert all(differences[key]==0 for key,value in arrays_a.items() if value.dtype.kind!='f')
    assert not any(atmosphere_differences.values()),atmosphere_differences
    assert original.steps==4 and resumed.steps==4
    assert original.feedback_z0 is None
    original.roughness_feedback=True
    original.buffers['u10'].fill(60.);original.buffers['v10'].fill(0.)
    original.surface_roughness(fields['znt'],fields['ust'],ice=fields['xice'])
    cp.cuda.get_current_stream().synchronize()
    assert bool(cp.all(cp.isfinite(original.z0)))
    zmax=float(cp.max(original.z0))
    # Table 2's extreme-wind lower limiting Cd=0.7e-3 maps to ~2.7 micrometres.
    assert zmax<1e-5,zmax
    products=None
    if output:
        from woof.ocean.products import publish
        original.output.collect()
        products=publish(SimpleNamespace(ocean=original),outdir=Path(output),
                         source_label='synthetic device coupling smoke; no forecast')
    result={'schema':'gpuwm.waves.coupling-smoke/v1','status':'PASS','scope':'synthetic device exchange and wave restart only',
            'atmosphere_steps':40,'coupling_steps':original.steps,'spectral_bins':[36,36],
            'restart_loaded_differing_bytes':immediate,'continuation_differing_bytes':differences,
            'continuation_relative_max':relative,'continuation_tolerance_fp32_eps':64,
            'input_field_differing_words':atmosphere_differences,
            'high_wind_z0_max_m':zmax,'solver':a,'resumed_solver':b,'products':products,'wall_seconds':time.perf_counter()-wall}
    if output:(Path(output)/'smoke-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
