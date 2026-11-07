"""Bounded actual surface-kernel checks with the shipping fast numerics."""
import json
from pathlib import Path
import runpy
import time

checks=runpy.run_path(str(Path(__file__).parents[1]/'tests/test_ocean_device_gpu.py'))
results=[]
for name,args in (
    ('test_rotation_density_and_stress_remain_device_resident',()),
    ('test_mynn_consumes_wave_roughness_before_fluxes',(0,)),
    ('test_mynn_consumes_wave_roughness_before_fluxes',(1,)),
    ('test_wave_age_roughness_uses_depth_and_preserves_land_and_ice',()),
    ('test_myj_consumes_wave_roughness_inside_flux_iterations',()),
    ('test_mm5_consumes_incoming_wave_momentum_roughness',(1,)),
    ('test_mm5_consumes_incoming_wave_momentum_roughness',(91,)),
):
    started=time.perf_counter();checks[name](*args)
    row={'check':name,'arguments':args,'status':'PASS','wall_seconds':time.perf_counter()-started}
    results.append(row);print(json.dumps(row),flush=True)
print(json.dumps({'schema':'gpuwm.waves.surface-smoke/v1','status':'PASS',
                  'numerics':'fast','checks':results}),flush=True)
