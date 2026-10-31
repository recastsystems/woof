"""Bitwise replay against the scalar operators profiled at d620bf0c."""
import ast
from pathlib import Path
import numpy as np
import pytest
from woof.hex import acoustic, acoustic_v841, diagnostics, driver, engines, terrain, transport
from woof.hex import moving_jolt_check as check


def references(module):
    source = ast.parse((Path(__file__).parent / ('data/moving_round2_' + module.__name__.split('.')[-1] + '.py')).read_text())
    namespace = dict(module.__dict__)
    names = []
    for node in source.body:
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
            node.returns = None
            for arg in (*node.args.args, *node.args.kwonlyargs):
                arg.annotation = None
    exec(compile(source, 'scalar-round2-reference', 'exec'), namespace)
    return {name: namespace[name] for name in names}


def bits(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    else:
        for field in expected.__slots__:
            bits(getattr(actual, field), getattr(expected, field))


def test_all_profiled_operators_on_actual_moving_step(tmp_path, monkeypatch):
    for spec in (engines.MESH, engines.GEOMETRY):
        if engines.locate(spec)[0] is None:
            pytest.skip(f'{spec.name} needed for actual moving fixture')
    family = check.build_family(tmp_path)
    mesh, cfg, coeff, winds = check.setup(family['meshes'][0])
    state, saved, prep = check.fixture(mesh, check.terrain_for(mesh), amp=1e-5)
    integ = check._Integrator(device='cpu', lts=False, cfg=cfg, coeff=coeff, winds=winds, rate_shift=0)
    live = integ.driver(state, saved, prep)
    calls = {}
    for module in (acoustic_v841, terrain, acoustic, diagnostics, transport):
        for name, scalar in references(module).items():
            vector = getattr(module, name)
            def compare(*args, _scalar=scalar, _vector=vector, _name=name, **kwargs):
                expected = _scalar(*args, **kwargs)
                actual = _vector(*args, **kwargs)
                bits(actual, expected)
                calls[_name] = calls.get(_name, 0) + 1
                return actual
            monkeypatch.setattr(module, name, compare)
            if hasattr(driver, name):
                monkeypatch.setattr(driver, name, compare)
    result, _ = integ.advance(live, state, saved)
    assert len(calls) == 6 and all(count > 0 for count in calls.values())
    assert np.isfinite(result.rho).all()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize('monotonic', [False, True])
@pytest.mark.parametrize('regional', [False, True])
@pytest.mark.parametrize('width', [6, 10])
def test_horizontal_branches_signed_zero(dtype, order, monotonic, regional, width):
    rng = np.random.default_rng(127)
    noise = lambda shape: rng.normal(size=shape).astype(dtype)
    tracers, velocity = noise((3, 7, 19)), noise((7, 24))
    velocity[0, :4] = [0., -0., 1., -1.]
    tracers[0, 0, :2] = [0., -0.]
    args = (tracers, velocity, noise(24), rng.integers(0, 19, (24, 2)),
            noise((24, width)), noise((24, width)), np.arange(24) % (width + 1),
            rng.integers(0, 19, (24, width)), order, .25)
    kwargs = dict(monotonic_source_flux=monotonic, config_apply_lbcs=regional,
                  bdy_mask_edge=np.arange(24) % 8)
    scalar = references(transport)['_atmosphere_horizontal_edge_values']
    bits(transport._atmosphere_horizontal_edge_values(*args, **kwargs), scalar(*args, **kwargs))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('regional', [False, True])
@pytest.mark.parametrize('small_step', [1, 2])
def test_acoustic_columns_and_regional_masks(dtype, regional, small_step):
    from test_contract_hostprep import bundle
    from woof.hex.offcentering_v841 import build_v841_acoustic_offcentering
    b, _, _ = bundle(7, seed=14)
    rng = np.random.default_rng(74)
    noise = lambda shape: rng.normal(0, .01, shape).astype(dtype)
    cells, edges, rows = (7,b.ncells),(7,b.nedges),(8,b.ncells)
    state = acoustic.AcousticStepState(ru_p=noise(edges),ru_avg=noise(edges),rw_p=noise(rows),
        ww_avg=noise(rows),rho_pp=noise(cells),rtheta_pp=noise(cells),rtheta_pp_old=noise(cells))
    coef = acoustic_v841.VerticalImplicitCoefficientsV841(cofwr=noise(cells),cofwz=noise(cells),
        coftz=noise(rows),cofwt=noise(cells),cofrz=noise((7,)),a_tri=noise(cells),
        b_tri=noise(cells),c_tri=noise(cells),alpha_tri=1+noise(cells),gamma_tri=noise(cells))
    forcing = acoustic.AcousticStepForcing(rho_zz=1+abs(noise(cells)),theta_m=noise(cells),
        zz=1+abs(noise(cells)),exner=noise(cells),cqu=noise(edges),
        zxu=noise(edges),dss=abs(noise(rows)),tend_ru=noise(edges),tend_rho=noise(cells),
        tend_rt=noise(cells),tend_rw=noise(rows),w=noise(rows),rw=noise(rows),rw_save=noise(rows))
    rdzw = b.vertical.rdzw.astype(dtype)
    off = build_v841_acoustic_offcentering(rdzw, minimum=.1, maximum=1., transition_bottom_z=10000, transition_top_z=30000)
    kwargs = dict(dts=4.,small_step=small_step,offcentering=off,
        fzm=b.vertical.fzm.astype(dtype),fzp=b.vertical.fzp.astype(dtype),rdzw=rdzw,
        specified_zone_edge=(b.masks.bdy_mask_edge>5).astype(dtype) if regional else None,
        specified_zone_cell=(b.masks.bdy_mask_cell>5).astype(dtype) if regional else None)
    scalar = references(acoustic_v841)['advance_acoustic_step_v841']
    bits(acoustic_v841.advance_acoustic_step_v841(b.mesh,state,forcing,coef,**kwargs), scalar(b.mesh,state,forcing,coef,**kwargs))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('apvm', [0., .1])
@pytest.mark.parametrize('rk_step', [None, 2, 3])
def test_diagnostics_all_paths(dtype, inverse, apvm, rk_step):
    from test_mesh_validation import graded_sphere_mesh
    mesh = graded_sphere_mesh(level=1, zoom=2.)
    rng = np.random.default_rng(443)
    u = rng.normal(size=(7, mesh.dcEdge.size)).astype(dtype)
    h = (1 + rng.random((7, mesh.areaCell.size))).astype(dtype)
    kwargs = dict(dt=4., apvm_upwinding=apvm, rk_step=rk_step, cached_tangential_velocity=u/2)
    if inverse:
        for name, field in [('inv_area_cell','areaCell'), ('inv_area_triangle','areaTriangle'),
                            ('inv_dc_edge','dcEdge'), ('inv_dv_edge','dvEdge')]:
            kwargs[name] = 1 / getattr(mesh, field).astype(dtype)
    scalar = references(diagnostics)['compute_solve_diagnostics']
    bits(diagnostics.compute_solve_diagnostics(mesh,u,h,**kwargs), scalar(mesh,u,h,**kwargs))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('regional', [False, True])
def test_omega_conversion_all_cells(dtype, regional):
    from test_mesh_validation import graded_sphere_mesh
    mesh = graded_sphere_mesh(level=1, zoom=2.)
    nc, ne, width = mesh.areaCell.size, mesh.dcEdge.size, mesh.edgesOnCell.shape[1]
    rng = np.random.default_rng(39)
    noise = lambda shape: rng.normal(size=shape).astype(dtype)
    w, u = noise((8, nc)), noise((7, ne))
    u[0, :2] = [0., -0.]
    kwargs = dict(fzm=noise(7), fzp=noise(7), zz=noise((7,nc)),
                  zb_cell=noise((8,nc,width)), zb3_cell=noise((8,nc,width)),
                  boundary_mask_cell=np.arange(nc)%8 if regional else None, relaxation_zone=5)
    scalar = references(acoustic)['convert_w_tendency_to_omega']
    bits(acoustic.convert_w_tendency_to_omega(mesh,w,u,**kwargs),scalar(mesh,w,u,**kwargs))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['logical', 'netcdf'])
def test_velocity_recovery_terrain_and_variable_counts(dtype, layout):
    from types import SimpleNamespace
    rng = np.random.default_rng(4599)
    nc, nl = 17, 8
    cells = np.array([[c, (c+1)%nc] for c in range(nc)])
    edges = np.array([[(c-1)%nc, c] for c in range(nc)])
    counts = np.full(nc, 2)
    counts[::3] = 1
    mesh = SimpleNamespace(arrays=dict(nEdgesOnCell=counts,edgesOnCell=edges,cellsOnEdge=cells))
    noise = lambda shape: rng.normal(size=shape).astype(dtype)
    coupling = terrain.TerrainCoupling(counts, edges, cells,
        acoustic.edge_signs_on_cells(mesh).astype(dtype),noise((nl+1,nc,2)),noise((nl+1,nc,2)),.25)
    kwargs = dict(rho_zz=rng.uniform(.1,2,(nl,nc)).astype(dtype),ru=noise((nl,nc)),rw=noise((nl+1,nc)),
        zz=rng.uniform(.5,1.5,(nl,nc)).astype(dtype),fzm=np.full(nl,.42,dtype),fzp=np.full(nl,.58,dtype),
        cf1=1.2,cf2=-.3,cf3=.1,coupling=coupling,array_layout=layout)
    kwargs['ru'][0, :2] = [0., -0.]
    if layout == 'netcdf':
        for name in ('rho_zz','ru','rw','zz'):
            kwargs[name] = kwargs[name].T
    scalar = references(terrain)['recover_velocities']
    bits(terrain.recover_velocities(**kwargs), scalar(**kwargs))
