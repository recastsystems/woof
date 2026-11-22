"""Every port compares integer views against the unchanged NumPy authority."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from woof.hex.acoustic import AcousticStepState, AcousticStepForcing, edge_signs_on_cells
from woof.hex.acoustic_v841 import VerticalImplicitCoefficientsV841, advance_acoustic_step_v841
from woof.hex.drivers import contract_hostprep as rust
from woof.hex.drivers import run_cuda_regional_contract as deck
from woof.hex.offcentering_v841 import build_v841_acoustic_offcentering
from woof.hex.transport import (AdvectionCoefficients, _atmosphere_horizontal_edge_values,
    _atmosphere_vertical_flux, advance_scalars)
import site_support
from test_mesh_validation import graded_sphere_mesh


@pytest.fixture
def hostprep():
    """The bitwise tests below run the real helper; a checkout without it skips by name."""
    return site_support.contract_hostprep_or_skip()


def bits(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype == np.dtype("float32")
    np.testing.assert_array_equal(a.view("u4"), b.view("u4"))


def bundle(nl=7, nt=1, seed=214):
    mesh = graded_sphere_mesh(level=1, zoom=2.0)
    nc = mesh.dimensions["nCells"]
    ne = mesh.dimensions["nEdges"]
    rng = np.random.default_rng(seed)
    noise = lambda shape: rng.normal(0, 0.01, shape).astype("f4")
    stage = noise((nt,nl,nc))
    stage[0,0,:2] = [0.0,-0.0]
    velocity = noise((nl,ne))
    velocity[0,:4] = [0.0,-0.0,1.0,-1.0]
    b = NS(mesh=mesh, ncells=nc, nedges=ne, nlev=nl,
        max_edges=mesh.edgesOnCell.shape[1], cells_on_edge=mesh.cellsOnEdge.copy(),
        n_edges_on_cell=mesh.nEdgesOnCell, edges_on_cell=mesh.edgesOnCell,
        edge_sign_on_cell=edge_signs_on_cells(mesh).astype("f4"),
        dv_edge=mesh.dvEdge.astype("f4"), inv_dc_edge=np.reciprocal(mesh.dcEdge.astype("f4")),
        inv_area_cell=np.reciprocal(mesh.areaCell.astype("f4")),
        state=NS(rho=(1.0+abs(noise((nl,nc)))).astype("f4"),
                 rho_u=velocity, scalars=stage.copy()), scalars_stage=stage,
        saved=NS(vertical_velocity=noise((nl+1,nc))),
        vertical=NS(fzm=np.full(nl,0.4,"f4"), fzp=np.full(nl,0.6,"f4"),
                    rdzw=np.full(nl,0.01,"f4")),
        masks=NS(bdy_mask_cell=np.arange(nc)%8, bdy_mask_edge=np.arange(ne)%8))
    width = 6
    coef = AdvectionCoefficients(noise((ne,width)),noise((ne,width)),
        rng.integers(0,width+1,ne,dtype="i8"),rng.integers(0,nc,(ne,width),dtype="i8"),3)
    return b, coef, noise


@pytest.mark.parametrize("nl,nt,seed", [(2,1,12),(7,1,214),(19,3,729)])
def test_horizontal_all_rings_and_signed_zero(nl, nt, seed, hostprep):
    b, co, _ = bundle(nl,nt,seed)
    # A specified edge is skipped before its sentinel cells can be read.
    edge = int(np.flatnonzero(b.masks.bdy_mask_edge > 5)[0])
    b.cells_on_edge[edge,1] = b.ncells
    expected = _atmosphere_horizontal_edge_values(b.scalars_stage,b.state.rho_u,
        b.dv_edge,b.cells_on_edge,co.adv_coefs,co.adv_coefs_3rd,
        co.n_adv_cells_for_edge,co.adv_cells_for_edge,3,0.25,
        config_apply_lbcs=True,bdy_mask_edge=b.masks.bdy_mask_edge)
    bits(rust.horizontal(b,co),expected)


@pytest.mark.parametrize("nl", [2,3,7,19])
def test_vertical_endpoints_and_interior(nl, hostprep):
    b, _, _ = bundle(nl)
    expected = _atmosphere_vertical_flux(b.scalars_stage[0],b.saved.vertical_velocity,
        b.vertical.fzm,b.vertical.fzp,3,0.25)
    bits(rust.vertical(b), expected)


@pytest.mark.parametrize("nt", [1,3])
def test_finish_including_specified_cells(nt, hostprep):
    b,co,_ = bundle(nt=nt)
    edges = rust.horizontal(b,co)
    flux = np.stack([_atmosphere_vertical_flux(stage,b.saved.vertical_velocity,
        b.vertical.fzm,b.vertical.fzp,3,0.25) for stage in b.scalars_stage])
    expected = advance_scalars(b.mesh,b.state.scalars,b.scalars_stage,b.state.rho,
        b.state.rho*np.float32(1.000001),b.state.rho_u,b.saved.vertical_velocity,120,
        coefficients=co,fzm=b.vertical.fzm,fzp=b.vertical.fzp,rdzw=b.vertical.rdzw,
        config_apply_lbcs=True,bdy_mask_cell=b.masks.bdy_mask_cell,
        bdy_mask_edge=b.masks.bdy_mask_edge,advance_density=True,inv_area_cell=b.inv_area_cell)
    actual = rust.finish(b,edges,flux)
    bits(actual.scalars,expected.scalars)
    bits(actual.density,expected.density)


@pytest.mark.parametrize("nl,regional,seed", [(2,False,1),(7,False,13),(7,True,14),(19,True,20)])
def test_acoustic_all_output_fields(nl,regional,seed,hostprep):
    b,_,noise = bundle(nl,seed=seed)
    cells, edges, rows = (nl,b.ncells),(nl,b.nedges),(nl+1,b.ncells)
    b.masks.spec_zone_mask_cell = (b.masks.bdy_mask_cell>5).astype("f4") if regional else np.zeros(b.ncells,"f4")
    b.masks.spec_zone_mask_edge = (b.masks.bdy_mask_edge>5).astype("f4") if regional else np.zeros(b.nedges,"f4")
    state = AcousticStepState(ru_p=noise(edges),ru_avg=noise(edges),rw_p=noise(rows),
        ww_avg=noise(rows),rho_pp=noise(cells),rtheta_pp=noise(cells),rtheta_pp_old=noise(cells))
    coef = VerticalImplicitCoefficientsV841(cofwr=noise(cells),cofwz=noise(cells),
        coftz=noise(rows),cofwt=noise(cells),cofrz=noise((nl,)),a_tri=noise(cells),
        b_tri=noise(cells),c_tri=noise(cells),alpha_tri=(1+noise(cells)).astype("f4"),gamma_tri=noise(cells))
    forcing = AcousticStepForcing(rho_zz=b.state.rho,theta_m=noise(cells),
        zz=(1+abs(noise(cells))).astype("f4"),exner=noise(cells),cqu=noise(edges),
        zxu=noise(edges),dss=abs(noise(rows)),tend_ru=noise(edges),tend_rho=noise(cells),
        tend_rt=noise(cells),tend_rw=noise(rows),w=noise(rows),rw=noise(rows),rw_save=noise(rows))
    off = build_v841_acoustic_offcentering(b.vertical.rdzw,minimum=0.1,maximum=1.,
        transition_bottom_z=10000,transition_top_z=30000)
    expected = advance_acoustic_step_v841(b.mesh,state,forcing,coef,dts=deck._DTS,
        small_step=2,offcentering=off,fzm=b.vertical.fzm,fzp=b.vertical.fzp,rdzw=b.vertical.rdzw,
        specified_zone_edge=b.masks.spec_zone_mask_edge,specified_zone_cell=b.masks.spec_zone_mask_cell)
    actual = rust.acoustic(b,(off,coef,forcing,state))
    for name in state.__slots__:
        bits(getattr(actual,name),getattr(expected,name))


def test_vertical_subnormal_and_signed_zero_bits(hostprep):
    b, _, _ = bundle(7)
    b.scalars_stage[:] *= np.float32(1.e-36)
    expected = _atmosphere_vertical_flux(b.scalars_stage[0],b.saved.vertical_velocity,
        b.vertical.fzm,b.vertical.fzp,3,0.25)
    bits(rust.vertical(b),expected)


def _append_pair(path,name):
    import os
    os.environ["WOOF_TIME_LEDGER"] = path
    attrs = {"deck":name,"kernels":[]}
    deck._emit("prep.deck","begin",attrs)
    deck._emit("prep.deck","end",attrs)


def test_processes_append_one_pair_each(tmp_path):
    import json
    import multiprocessing
    path = str(tmp_path/"timeline.jsonl")
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_append_pair,args=(path,item.name)) for item in deck.DECKS]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    events = [json.loads(line) for line in open(path)]
    assert len(events) == 16
    for item in deck.DECKS:
        assert [row["phase"] for row in events if row["attrs"]["deck"] == item.name] == ["begin","end"]


def test_reference_once_and_distinct_flux_inputs():
    b = object.__new__(deck.Bundle)
    b._references = {}
    produced = []
    def make():
        produced.append(True)
        return object()
    a = b.reference(("transport-finish","first"),make)
    assert b.reference(("transport-finish","first"),make) is a
    assert b.reference(("transport-finish","other"),make) is not a
    assert len(produced) == 2


def test_append_schema_and_unwritable_ledger(monkeypatch,tmp_path):
    import json
    path = tmp_path/"timeline.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER",str(path))
    attrs = {"deck":"acoustic-regional","kernels":["acoustic_ru_regional_v841"]}
    deck._emit("prep.deck","begin",attrs)

    deck._emit("prep.deck","end",attrs)
    events = [json.loads(row) for row in path.read_text().splitlines()]
    assert [event["phase"] for event in events] == ["begin","end"]
    assert all(event["v"] == 1 and event["attrs"] == attrs for event in events)
    assert set(events[0]) == {"v","t_utc","t_mono","host","rental_id","run_id",
        "product","stage","phase","card","pid","attrs"}
    monkeypatch.setenv("WOOF_TIME_LEDGER",str(tmp_path))
    deck._emit("prep.deck","begin",attrs)


@pytest.mark.parametrize("physical,visible,expected", [
    ("3", "0", 3), (None, "2,4", 2), (None, "GPU-opaque", None),
    (None, "", None),
])
def test_ledger_card_is_a_c1_integer_or_null(monkeypatch,tmp_path,physical,visible,expected):
    import json
    monkeypatch.delenv("CARD_QUEUE_CARDS", raising=False)
    monkeypatch.setenv("WOOF_CARD_QUEUE_PREFIX", "")
    if physical is None:
        monkeypatch.delenv("WOOF_CARD", raising=False)
    else:
        monkeypatch.setenv("WOOF_CARD", physical)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    path = tmp_path / "timeline.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(path))
    deck._emit("prep.deck", "begin", {"deck": "acoustic-regional", "kernels": []})
    card = json.loads(path.read_text())["card"]
    assert card == expected
    assert card is None or type(card) is int


def test_mutex_card_index_survives_uuid_visibility(monkeypatch,tmp_path):
    import json
    monkeypatch.delenv("WOOF_CARD", raising=False)
    monkeypatch.setenv("CARD_QUEUE_CARDS", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-opaque")
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(path))
    deck._emit("prep.deck", "begin", {"deck": "acoustic-regional", "kernels": []})
    assert json.loads(path.read_text())["card"] == 4


def test_missing_explicit_engine_does_not_select_other_binary(monkeypatch,tmp_path):
    monkeypatch.setenv("WOOF_HEX_CONTRACT_HOSTPREP",str(tmp_path/"missing"))
    alternate = tmp_path/"alternate"
    alternate.write_bytes(b"unused")
    monkeypatch.setattr(rust,"gpuwm_candidates",lambda spec: (alternate,))
    rust.engine.cache_clear()
    try:
        with pytest.raises(rust.MpasPortError,match="float32 host references"):
            rust.engine()
    finally:
        rust.engine.cache_clear()


def test_installed_default_resolves_the_shipped_native_reference(monkeypatch, tmp_path):
    package = tmp_path / "installed/hexcore"
    module = package / "drivers/contract_hostprep.py"
    module.parent.mkdir(parents=True)
    executable = package / "libexec/hex_contract_hostprep"
    executable.parent.mkdir()
    executable.write_bytes(b"installed native reference")
    monkeypatch.delenv("WOOF_HEX_CONTRACT_HOSTPREP", raising=False)
    monkeypatch.setattr(rust, "__file__", str(module))
    monkeypatch.setattr(rust, "gpuwm_candidates", lambda spec: ())
    from woof.hex import helpers  # (WOOF) the wheel's libexec is read through woof.hex.helpers
    monkeypatch.setattr(helpers, "PACKAGED_DIR", package / "libexec")
    rust.engine.cache_clear()
    try:
        assert rust.engine() == executable
    finally:
        rust.engine.cache_clear()
