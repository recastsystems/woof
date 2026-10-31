"""Limited-area runs on the MPAS suite route (parity run group 1, #3 and #4).

Defect: ``--lbc-dir`` never reached the suite route, so every region, point, country, swath and cycle fine forecast
with ``--physics-suite convection_permitting`` built the global dycore on a cull and stopped in its terrain coupling
("cells_on_edge contains a missing or out-of-range cell").  These tests fail on rel/hex-0.4.4 222d6e2c.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.forecast_v842 import bind_limited_area, namelist_option


def _arguments(lbc_dir=None, **extra):
    return SimpleNamespace(lbc_dir=lbc_dir, **extra)


def test_lbc_dir_sets_mpas_config_apply_lbcs(tmp_path):
    text = bind_limited_area(_arguments(tmp_path), "&physics\nconfig_physics_suite='convection_permitting'\n/\n")
    assert namelist_option(text, 'limited_area', 'config_apply_lbcs') is True
    assert namelist_option(text, 'physics', 'config_physics_suite') == 'convection_permitting'


def test_no_lbc_dir_leaves_a_global_run_unchanged():
    text = "&physics\nconfig_physics_suite='convection_permitting'\n/\n"
    assert bind_limited_area(_arguments(), text) == text


def test_namelist_limited_area_without_series_is_refused():
    with pytest.raises(ValueError, match='needs its lateral-boundary series'):
        bind_limited_area(_arguments(), "&limited_area\nconfig_apply_lbcs=.true.\n/\n")


@pytest.mark.parametrize('extra, named', [
    (dict(iau_window_seconds=3600.0, samehost_devices='0,1'), 'IAU on more than one card'),
    (dict(stochastic='{"sppt":true}'), 'stochastic'),
    (dict(chemistry_config='chem.json'), 'chemistry'),
])
def test_combinations_the_regional_dycore_does_not_carry_are_refused_by_name(tmp_path, extra, named):
    with pytest.raises(ValueError, match=named):
        bind_limited_area(_arguments(tmp_path, **extra), '')


def test_namelist_iau_on_several_cards_is_refused_at_the_door(tmp_path):
    # A namelist &iau run on several cards passed the door and every rank refused it in runtime_v842 after the
    # whole launch; one card still takes it.
    text = "&iau\nconfig_IAU_option='on'\nconfig_IAU_window_length_s=3600.0\n/\n"
    with pytest.raises(ValueError, match='IAU on more than one card'):
        bind_limited_area(_arguments(tmp_path, samehost_devices='0,1'), text)
    assert namelist_option(bind_limited_area(_arguments(tmp_path), text), 'limited_area', 'config_apply_lbcs') is True


@pytest.mark.parametrize('extra', [dict(jedi_background=True), dict(da_analysis='analysis.nc'),
                                   dict(iau_window_seconds=3600.0)])
def test_limited_area_takes_the_da_hand_offs(tmp_path, extra):
    # fix/hex-lam-da-v842: the JEDI background/analysis and IAU on one card open the regional dycore on the
    # analysed state at the analysis time; the door no longer refuses them.
    text = bind_limited_area(_arguments(tmp_path, **extra), '')
    assert namelist_option(text, 'limited_area', 'config_apply_lbcs') is True


def test_unchanged_analysis_keeps_the_carried_words():
    # fix/hex-lam-da-v842: an analysis equal to its background inserts nothing; a changed cell takes the recoupled
    # words and so do the edges beside it.
    import numpy as np
    from types import SimpleNamespace as NS
    from woof.hex.driver import DrySavedDiagnostics
    from woof.hex.state import PrognosticState
    from woof.hex.mpasphys.cycling_v842 import _changed_elements, keep_unchanged
    rng = np.random.default_rng(0)
    cells, edges, levels = 4, 5, 3
    mesh = NS(arrays={'cellsOnEdge': np.array([[0, 1], [1, 2], [2, 3], [3, -1], [0, 3]])})
    background = {'rho': rng.random((levels, cells), np.float32), 'theta': rng.random((levels, cells), np.float32),
                  'w': rng.random((levels + 1, cells), np.float32), 'qv': rng.random((levels, cells), np.float32),
                  'u': rng.random((levels, edges), np.float32)}

    def build(seed):
        r = np.random.default_rng(seed)
        state = PrognosticState(r.random((levels, cells), np.float32), r.random((levels, cells), np.float32),
                                r.random((levels, edges), np.float32), r.random((levels + 1, cells), np.float32),
                                r.random((1, levels, cells), np.float32), time_seconds=3600.0)
        saved = DrySavedDiagnostics(*(r.random(shape, np.float32) for shape in
                                      [(levels, cells)] * 5 + [(levels, edges), (levels + 1, cells)]))
        return state, saved
    carried, recoupled = build(1), build(2)
    same = {k: v.copy() for k, v in background.items()}
    cell_mask, edge_mask = _changed_elements(same, background, ('qv',), mesh, cells, edges)
    assert not cell_mask.any() and not edge_mask.any()
    state, saved = keep_unchanged(*carried, *build(2), cell_mask, edge_mask)
    for name in ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars'):
        assert np.array_equal(getattr(state, name), getattr(carried[0], name))
    for name in ('theta_m', 'exner', 'normal_velocity', 'vertical_velocity'):
        assert np.array_equal(getattr(saved, name), getattr(carried[1], name))
    same['theta'][1, 2] += 1.0
    cell_mask, edge_mask = _changed_elements(same, background, ('qv',), mesh, cells, edges)
    assert cell_mask.tolist() == [False, False, True, False]
    assert edge_mask.tolist() == [False, True, True, False, False]
    state, _ = keep_unchanged(*carried, *recoupled, cell_mask, edge_mask)
    assert np.array_equal(state.rho[:, 2], recoupled[0].rho[:, 2])
    assert np.array_equal(state.rho[:, 0], carried[0].rho[:, 0])
    assert np.array_equal(state.rho_u[:, 1], recoupled[0].rho_u[:, 1])
    assert np.array_equal(state.rho_u[:, 4], carried[0].rho_u[:, 4])


def test_limited_area_runs_on_several_cards(tmp_path):
    # Each rank builds its regional dycore on its own cut (samehost_v842); the door no longer refuses the cards.
    text = bind_limited_area(_arguments(tmp_path, samehost_devices='0,1'), '')
    assert namelist_option(text, 'limited_area', 'config_apply_lbcs') is True


def test_regional_suite_runs_load_every_moist_species():
    # The regional qv-only coefficients stay for the dycore's own one-species regional runs; a suite run (which
    # passes its Registry mass-loading count) takes the global atm_compute_moist_coefficients path.
    import inspect
    from woof.hex.cuda_driver import CudaDryDycoreDriver
    source = inspect.getsource(CudaDryDycoreDriver._step_device_v841)
    assert 'regional_suite_loading' in source


def test_projection_geometry_names_the_garbage_cell():
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip('no CUDA device')
    from woof.hex.mpasphys.resident_v842 import regional_physics_geometry
    n_cells, n_edges = 3, 4
    # Native one-based cellsOnEdge (2, nEdges); the last edge is a ring-7 edge whose second endpoint is the garbage
    # cell nCells+1.
    cells_on_edge = np.array([[1, 2, 3, 3], [2, 3, 1, n_cells + 1]], dtype=np.int32)
    vectors = np.arange(3 * n_cells, dtype=np.float32).reshape(3, n_cells)
    fields = {'cellsOnEdge': cells_on_edge, 'east': vectors, 'north': vectors + 1,
              'edgeNormalVectors': np.ones((3, n_edges), np.float32)}
    provider = SimpleNamespace(cp=cp, field=lambda pool, name: cp.asarray(fields[name]),
                               session=SimpleNamespace(layout=SimpleNamespace(
                                   dimensions={'nCellsSolve': n_cells, 'nEdgesSolve': n_edges})))
    geometry = regional_physics_geometry(provider)
    assert geometry.n_cells == n_cells + 1 and geometry.regional_garbage_cell == n_cells
    assert cp.asnumpy(geometry.cells_on_edge)[-1].tolist() == [2, n_cells]
    assert not cp.asnumpy(geometry.east_cell)[n_cells].any()


def test_lb1_boundary_forcing_columns_follow_the_mode(monkeypatch):
    from woof.hex.mpasphys.runtime_v842 import ResidentNativeCoupling
    coupling = ResidentNativeCoupling.__new__(ResidentNativeCoupling)
    coupling.cp = np
    coupling.regional = SimpleNamespace(bdy_mask_cell=np.array([0, 0, 3, 7, 7], np.int32))
    coupling.session = SimpleNamespace(config=SimpleNamespace(mode='correct'))
    assert coupling.boundary_forcing_columns().tolist() == [False, False, True, True, True]
    for mode, knob in (('exact', '1'), ('correct', '0')):
        fresh = ResidentNativeCoupling.__new__(ResidentNativeCoupling)
        fresh.cp, fresh.regional = np, coupling.regional
        fresh.session = SimpleNamespace(config=SimpleNamespace(mode=mode))
        monkeypatch.setenv('WOOF_HEX_MPAS_LB1', knob)
        assert fresh.boundary_forcing_columns() is None
    monkeypatch.delenv('WOOF_HEX_MPAS_LB1')
    closed = ResidentNativeCoupling.__new__(ResidentNativeCoupling)
    closed.session = coupling.session
    assert closed.boundary_forcing_columns() is None


def test_stored_zero_neighbours_become_the_garbage_element():
    from woof.hex.mpasphys.runtime_v842 import remap_limited_area_sentinels
    store = {('mesh', 'cellsOnEdge'): np.array([[1, 2, 0], [2, 0, 0]], np.int32),
             ('mesh', 'edgesOnVertex'): np.array([[1, 0], [2, 3]], np.int32)}
    native = SimpleNamespace(_fields={k: (v.dtype, v.shape, 1) for k, v in store.items()},
                             get_field=lambda pool, name: store[pool, name].copy(),
                             put_field=lambda pool, name, value, nlevels=1: store.__setitem__((pool, name), value))
    layout = SimpleNamespace(dimensions={'nCells': 2, 'nEdges': 3, 'nVertices': 1},
                             fields=[{'pool': 'mesh', 'name': n, 'pool_name': n}
                                     for n in ('cellsOnEdge', 'edgesOnVertex')])
    counts = remap_limited_area_sentinels(native, layout)
    assert counts == {'cellsOnEdge': 3, 'edgesOnVertex': 1}
    assert store['mesh', 'cellsOnEdge'].tolist() == [[1, 2, 3], [2, 3, 3]]
    assert store['mesh', 'edgesOnVertex'].tolist() == [[1, 4], [2, 3]]


def test_regional_iau_door_keeps_the_boundary_driver(tmp_path):
    text = bind_limited_area(_arguments(tmp_path, iau_window_seconds=600.0), '')
    assert namelist_option(text, 'limited_area', 'config_apply_lbcs') is True


@pytest.mark.parametrize('regional', [False, True])
def test_suite_iau_payload_uses_real_solve_extent_and_zero_padding(tmp_path, monkeypatch, regional):
    import hashlib
    import json
    from woof.hex.mpasphys.runtime_v842 import SuiteIau
    import woof.hex.iau_v842 as iau
    ns, nl, nc, ne = 2, 3, 4, 5
    names = {'u_amb': (nl, ne), 'rho_amb': (nl, nc), 'theta_amb': (nl, nc),
             'scalars_amb': (ns, nl, nc)}
    fields = []
    originals = {}
    for name, shape in names.items():
        a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
        originals[name] = a
        allocated = np.full(shape[:-1] + (shape[-1] + 1,), 999.0, np.float32)
        allocated[..., :-1] = a
        raw = allocated.tobytes(order='F')
        (tmp_path / name).write_bytes(raw)
        fields.append(dict(pool='tend_iau', name=name, payload=name, shape=allocated.shape,
                           sha256=hashlib.sha256(raw).hexdigest(), origins=['test-stream']))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(fields=fields)))
    layout = SimpleNamespace(configs=dict(config_IAU_option='on', config_IAU_window_length_s=600.0),
        dimensions=dict(nCells=nc, nCellsSolve=nc, nEdgesSolve=ne),
        indices={('state', 'moist_start'): 1, ('state', 'moist_end'): ns, ('state', 'index_qv'): 1})
    pad = int(regional)
    dycore = SimpleNamespace(cp=np, atmosphere=SimpleNamespace(vertical=SimpleNamespace(zz=np.ones((nl,nc+pad), np.float32))))
    if regional:
        dycore.regional_v841 = SimpleNamespace()
    mesh = SimpleNamespace(arrays=dict(cellsOnEdge=np.array([[0,1],[1,2],[2,3],[3,-1],[0,3]])))
    forcing = SuiteIau(layout=layout, payload=tmp_path, config=SimpleNamespace(mode='correct'), mesh=mesh, dycore=dycore, dt=18)
    for key, name in {'u':'u_amb','rho':'rho_amb','theta':'theta_amb','scalars':'scalars_amb'}.items():
        actual = forcing.increments[key]
        assert actual.shape[-1] == originals[name].shape[-1] + pad
        np.testing.assert_array_equal(actual[..., :originals[name].shape[-1]], originals[name])
        if regional:
            assert not actual[..., -1].any()
    assert forcing.cells_on_edge.shape == (ne+pad,2)
    assert forcing.cells_on_edge[3,1] == -1
    if regional:
        assert forcing.cells_on_edge[-1].tolist() == [-1,-1]
    calls = []
    monkeypatch.setattr(iau, 'atm_add_tend_anal_incr', lambda config, structs, *args, **kwargs: calls.append(structs))
    state = SimpleNamespace(rho=np.ones((nl,nc+pad)), scalars=np.zeros((ns,nl,nc+pad)))
    saved = SimpleNamespace(theta_m=state.rho, normal_velocity=np.zeros((nl,ne+pad)))
    sources = SimpleNamespace(rho=state.rho.copy(), rho_theta=state.rho.copy(), rho_u=saved.normal_velocity.copy(), scalars=state.scalars.copy())
    forcing.apply(itimestep=1, sources=sources, state=state, saved=saved,
                  diagnostics=SimpleNamespace(h_edge=saved.normal_velocity), cache=None)
    assert calls[0]['mesh']['nCells'] == nc+pad
    assert calls[0]['mesh']['nCellsSolve'] == nc
    assert calls[0]['mesh']['nEdgesSolve'] == ne
