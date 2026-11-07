from pathlib import Path
import pytest
from woof.config import RunConfig, validate_run_config

@pytest.mark.parametrize('shortwave', [0, 1, 4, 5, 90])
def test_selector_31_preserves_independent_shortwave_selection(shortwave):
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=10000., dy=10000., ztop=20000., dt=10., run_seconds=20., ra_physics=0, ra_lw_physics=31, ra_sw_physics=shortwave)
    validate_run_config(cfg)

def test_namelist_selector_and_restart_row():
    from woof.namelist_import import _RA_LW_MAP
    from woof.physics_registry import stock_callable_class
    assert _RA_LW_MAP[31][0] == 31
    assert stock_callable_class('radiation', {'ra_lw_physics':31,'ra_sw_physics':0}) == 'woof.core.radiation_composition.NewtonianRelaxationRadiation'

def test_runtime_forcing_uses_shared_loader_and_rk_placement():
    from woof.core.kernels import module_source, module_options
    src = module_source('idealized_forcing')
    assert '__fdiv_rn' in src
    assert '--use_fast_math' not in module_options('idealized_forcing')
    dynamics = (Path(__file__).parents[1]/'woof/core/dycore.py').read_text()
    assert 'apply_idealized_momentum_damping(state, cfg, ru, rv)' in dynamics

def test_non31_damping_does_not_touch_the_state():
    # Dispatch returns before inspecting a state or importing a kernel.
    from woof.core.dycore import apply_idealized_momentum_damping
    cfg = RunConfig(nx=8,ny=8,nz=8,dx=10000.,dy=10000.,ztop=20000.,dt=10.,run_seconds=20.)
    apply_idealized_momentum_damping(None,cfg,None,None)

def test_hex_column_plan_admits_complete_forcing_selection():
    from woof.core.mpas_column_suite import MPAS_DEFAULT_PHYSICS, column_physics_plan
    physics = dict(MPAS_DEFAULT_PHYSICS, ra_physics=0,
                   ra_lw_physics=31, ra_sw_physics=0)
    plan = column_physics_plan(physics)
    assert plan.config.ra_lw_physics == 31
    assert plan.identity["rk_momentum"] == "held-suarez-edge-v1"
    # Actual construction still needs an RK binding; planning allocates none.

def test_hex_column_construction_requires_rk_momentum():
    from test_mpas_column_physics import _kwargs
    from woof.core.mpas_column_suite import MPAS_DEFAULT_PHYSICS, MpasColumnPhysics
    physics = dict(MPAS_DEFAULT_PHYSICS, ra_physics=0,
                   ra_lw_physics=31, ra_sw_physics=0)
    with pytest.raises(ValueError, match='requires RK-stage edge momentum damping'):
        MpasColumnPhysics(**_kwargs(physics))

@pytest.mark.parametrize("shortwave", [0, 1])
def test_selector_31_through_the_real_cli(tmp_path, shortwave):
    import tomllib
    from test_namelist_import import _pair, INPUT_TEXT
    from woof import cli
    from woof.experiment import load_experiment
    inp = INPUT_TEXT.replace('ra_lw_physics = 4, 4','ra_lw_physics = 31, 31').replace('ra_sw_physics = 4, 4',f'ra_sw_physics = {shortwave}, {shortwave}')
    wps, namelist = _pair(tmp_path, inp=inp)
    out = tmp_path/'resolved.toml'
    assert cli.main(['import-namelist',str(wps),str(namelist),'--output',str(out)]) == 0
    doc=tomllib.loads(out.read_text())
    assert doc['shared']['ra_lw_physics']==31
    assert doc['shared']['ra_sw_physics']==shortwave
    cfg=load_experiment(out)
    assert all(d.run.ra_lw_physics==31 and d.run.ra_sw_physics==shortwave
               for d in cfg.domains)


from conftest import requires_gpu

@requires_gpu
@pytest.mark.gpu
def test_source_pinned_column_words():
    from tools.relaxation_wrf461_oracle.validate import run
    assert run(Path(__file__).parent/'data/relaxation31.hex')['passed']

@requires_gpu
@pytest.mark.gpu
def test_one_bit_rate_change_is_detected():
    from tools.relaxation_wrf461_oracle.validate import run
    assert run(Path(__file__).parent/'data/relaxation31.hex', mutate=True)['passed']


@requires_gpu
@pytest.mark.gpu
def test_pressure_halo_preserves_native_components_without_stage_allocation():
    import cupy as cp
    import numpy as np
    from types import SimpleNamespace
    from woof.core.dycore import apply_idealized_momentum_damping
    from woof.core.kernels import load_module_no_ftz
    # The pressure split is observable in WRF's left-associated four-term
    # face sums. Test both the saved perturbation and total-only state,
    # including a horizontally varying base and all boundary corner reads.
    nz,ny,nx=3,4,5
    cfg=RunConfig(nx=nx,ny=ny,nz=nz,dx=10000.,dy=10000.,ztop=16000.,
                  dt=10.,run_seconds=10.,ra_lw_physics=31,ra_sw_physics=0)
    cells=np.arange(nz*ny*nx,dtype=np.float32).reshape(nz,ny,nx)
    for edge,one_dimensional,saved in ((False,True,False),(True,False,True)):
        from dataclasses import replace
        cfg=replace(cfg,open_x=edge,open_y=edge)
        base=(90000.-np.arange(nz,dtype=np.float32)*30000. if one_dimensional
              else 90000.-cells*1000.)
        base3=np.broadcast_to(base[:,None,None] if one_dimensional else base,cells.shape)
        perturb=(cells-20.)*.03125
        total=base3+perturb
        original=perturb if saved else total-base3
        storage={}
        def scratch(shape,key):
            if key not in storage:storage[key]=cp.empty(shape,dtype=cp.float32)
            return storage[key]
        state=SimpleNamespace(p=cp.asarray(total),pb=cp.asarray(base),scratch=scratch,
                              ru_t=cp.zeros((nz,ny,nx+1),cp.float32),
                              rv_t=cp.zeros((nz,ny+1,nx),cp.float32))
        if saved:state.p_perturbation=cp.asarray(perturb)
        ru=cp.ones_like(state.ru_t);rv=cp.ones_like(state.rv_t)
        apply_idealized_momentum_damping(state,cfg,ru,rv)
        pointers={name:value.data.ptr for name,value in storage.items()}
        for name,expected in (('p',original),('pb',base3)):
            expected=np.pad(expected,((0,0),(1,1),(1,1)),mode='edge' if edge else 'wrap')
            actual=cp.asnumpy(storage['relaxation31_'+name+'_halo'])
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        # Compare the complete damping result with the native-word kernel
        # fed explicitly prepared pressure halos, not just the pack itself.
        for axis,wind,actual in ((0,ru,state.ru_t),(1,rv,state.rv_t)):
            expected=cp.zeros_like(actual)
            load_module_no_ftz('idealized_forcing', no_fmad=True).get_function('relax_momentum')(
                ((wind.size+255)//256,), (256,), (
                    np.int32(nx),np.int32(ny),np.int32(nz),np.int32(axis),
                    np.int32(nx+1 if axis==0 else ny),
                    storage['relaxation31_p_halo'],storage['relaxation31_pb_halo'],wind,expected))
            np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),cp.asnumpy(expected).view(np.uint32))
        apply_idealized_momentum_damping(state,cfg,ru,rv)
        assert pointers=={name:value.data.ptr for name,value in storage.items()}


def test_global_inspection_names_unbound_diffusion_after_forcing_admission(tmp_path, capsys):
    import json
    from pathlib import Path
    from woof import cli
    path = tmp_path / "namelist.input"
    text = (Path(__file__).parent / "data/global_heldsuarez.input").read_text()
    text = text.replace("diff_opt                            = 1,", "diff_opt                            = 2,")
    assert text != (Path(__file__).parent / "data/global_heldsuarez.input").read_text()
    path.write_text(text)
    assert cli.main(["ideal-plan", str(path), "--case", "em_heldsuarez", "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert not result["imports"] and not result["runs"]
    refusal = result["refusals"][0]
    assert refusal["code"] == "global_admission"
    assert "diff_opt" in refusal["breakage"]
    assert "global latitude/longitude state builder or pole topology" not in refusal["breakage"]
