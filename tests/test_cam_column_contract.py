"""The CAM3 port must pass the real WRF column fields and the mutation control."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

_MODULE = Path(__file__).resolve().parents[1] / 'tools/cam_wrf461_oracle/validate.py'
_spec = importlib.util.spec_from_file_location('cam_column_validate', _MODULE)
_validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validator)


def test_one_bit_output_mutation_is_detected():
    reference = {('lw',1,'output:qrl'): np.array([1.0,-0.2,0.0],dtype=np.float64)}
    candidate = {key:value.copy() for key,value in reference.items()}
    candidate[('lw',1,'output:qrl')].view(np.uint64)[0] ^= np.uint64(1)
    result = _validator.compare(reference,candidate)
    assert not result['pass']
    assert result['mismatches'][0]['max_ulp'] == 1
    assert result['mismatches'][0]['changed_words'] == 1


def test_missing_output_field_is_refused():
    with pytest.raises(ValueError,match='inventory'):
        _validator.compare({('sw',64,'output:qrs'):np.array([0.0])},{})


def test_nan_is_not_a_parity_pass():
    field = {('sw',64,'output:qrs'):np.array([float('nan')])}
    with pytest.raises(ValueError,match='nonfinite'):
        _validator.compare(field,field)


def test_cam_column_gpu_entry_is_available():
    assert importlib.util.find_spec('woof.core.cam3') is not None, 'CAM3 column GPU entry is absent; selectors 3/3 must remain unadmitted'


def test_cam_workspace_is_bounded_while_retained_state_scales_with_grid():
    from woof.core.cam3_memory import allocation_shapes, worker_count
    small = allocation_shapes(48, 8, 8)
    large = allocation_shapes(48, 64, 64)
    assert worker_count(1) == 32
    assert worker_count(4096) == 4096
    assert small['cam3/state'][1] == (21962220,)
    assert small['cam3/workspace'][1] == (64, 454528)
    assert large['cam3/workspace'][1] == (4096, 454528)
    for name in ('abstot_3d', 'absnxt_3d', 'emstot_3d'):
        before = small['cam3/retained/' + name][1]
        after = large['cam3/retained/' + name][1]
        assert after[0] == 64 * before[0]
        assert after[1] == before[1]


def test_preflight_prices_cam_owned_state_only_when_selected():
    from dataclasses import replace
    from datetime import datetime
    from woof.config import RunConfig
    from woof.experiment import experiment_from_run_config
    from woof.core.preflight import estimate_domain
    cfg = RunConfig(nx=12, ny=12, nz=48, dx=3000., dy=3000.,
                    ztop=20000., dt=10., run_seconds=60.,
                    ra_lw_physics=3, ra_sw_physics=3)
    cam = estimate_domain(experiment_from_run_config(cfg, datetime(2024, 5, 21)).root)
    off = estimate_domain(experiment_from_run_config(
        replace(cfg, ra_lw_physics=0, ra_sw_physics=0), datetime(2024, 5, 21)).root)
    items = {item.name: item for item in cam.items if item.name.startswith('cam3/')}
    assert items['cam3/state'].nbytes == 21962220
    assert items['cam3/workspace'].nbytes == 160 * 454528
    assert len([name for name in items if name.startswith('cam3/retained/')]) == 3
    assert not any(item.name.startswith('cam3/') for item in off.items)


def test_cam_local_frames_are_priced_for_the_deepest_selected_domain():
    from dataclasses import replace
    from datetime import datetime
    from woof.config import RunConfig
    from woof.experiment import experiment_from_run_config
    from woof.core.preflight import kernel_local_frame_bytes, physics_kernel_modules
    from woof.core.cam3_memory import local_frame_bound
    from woof.core.kernel_frame_recordings import assumed_frame_bound
    cfg = RunConfig(nx=12, ny=12, nz=48, dx=3000., dy=3000.,
                    ztop=20000., dt=10., run_seconds=60.,
                    ra_lw_physics=3, ra_sw_physics=3)
    exp = experiment_from_run_config(cfg, datetime(2024, 5, 21))
    shallow = replace(exp.root, grid_id=2, run=replace(cfg, nz=16))
    nested = replace(exp, domains=(exp.root, shallow))
    frames = kernel_local_frame_bytes(nested)
    assert frames['cam3_emissivity'] == max(assumed_frame_bound(),
                                          local_frame_bound(48),local_frame_bound(16))
    assert frames['cam3_prepare'] == 0
    off = experiment_from_run_config(replace(cfg, ra_lw_physics=0,
                                            ra_sw_physics=0), exp.start_time)
    assert not {'cam3_emissivity', 'cam3_prepare'} & physics_kernel_modules(off)
