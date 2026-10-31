"""[hex_nests] dycore sources (w, cape, updraft helicity) on every physics route.

THE BREAKAGE (WOOF owner canary run_9948508bd7cfdee3, 2026-10-29): a 'wmax >= -1000' table was accepted and
quoted, then refused at arm time on the box: "[hex_nests] field(s) [wmax] read [w], which this physics route
does not form on the card (it forms [refl10cm])".  Both routes now form w, cape and updraft_helicity_max from
the dycore.  The CPU tier runs with numpy standing in for CuPy; CAPE (a CUDA kernel) runs on the card tier.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import nest_look_dycore, nest_triggers


def _mesh():
    # Two owned cells, three vertices; one-based connectivity as an MPAS file stores it.
    return SimpleNamespace(
        verticesOnCell=np.array([[1, 2, 3], [2, 3, 1]]),
        nEdgesOnCell=np.array([3, 2]),
        cellsOnVertex=np.array([[1, 2, 3], [1, 2, 3], [2, 1, 3]]),
        kiteAreasOnVertex=np.array([[1.0, 2.0, 0.5], [3.0, 1.5, 0.5], [0.25, 0.75, 0.5]]),
        areaCell=np.array([2.0, 4.0, 1.0]),
    )


def test_helicity_weights_pick_each_vertex_kite_of_the_cell_over_its_area():
    vertex, weight = nest_look_dycore.helicity_weights(_mesh(), np.array([0, 1]), owned_cells=2)
    assert vertex.tolist() == [[0, 1, 2], [1, 2, 0]]
    # cell 0: v0 slot 0 (1.0), v1 slot 0 (3.0), v2 slot 1 (0.75); over area 2
    assert np.allclose(weight[0], [0.5, 1.5, 0.375])
    # cell 1: v1 slot 1 (1.5), v2 slot 0 (0.25); third slot past nEdgesOnCell: 0; over area 4
    assert np.allclose(weight[1], [0.375, 0.0625, 0.0])


def _driver(nl=4, cells=2, vertices=3):
    rng = np.random.default_rng(7)
    w = rng.normal(0.0, 5.0, (nl + 1, cells)).astype(np.float32)
    zgrid = np.array([[100.0, 0.0], [1500.0, 1400.0], [3000.0, 2900.0], [4500.0, 4400.0], [7000.0, 6900.0]],
                     np.float32)
    vort = rng.normal(0.0, 0.01, (nl, vertices)).astype(np.float32)
    atmosphere = SimpleNamespace(
        saved=SimpleNamespace(vertical_velocity=w, normal_velocity=None),
        state=SimpleNamespace(rho=None, rho_u=None),
        vertical=SimpleNamespace(zgrid=zgrid))
    horizontal = SimpleNamespace(solve_vorticity=lambda rho, rho_u, normal_velocity=None: vort)
    return SimpleNamespace(atmosphere=atmosphere, horizontal=horizontal), w, zgrid, vort


def test_w_is_the_resident_array_and_uh_is_mpas_formula():
    driver, w, zgrid, vort = _driver()
    sources = nest_look_dycore.DycoreLookSources(np, needed=["w", "updraft_helicity_max"],
                                                 cells=np.array([0, 1]), owned_cells=2, qv_index=None,
                                                 mesh=_mesh())
    got = sources.form(driver)
    assert np.shares_memory(got["w"], w) and got["w"].shape == (5, 2)
    vertex, weight = nest_look_dycore.helicity_weights(_mesh(), np.array([0, 1]), owned_cells=2)
    for c in range(2):
        expected = 0.0
        for k in range(4):
            zeta = sum(weight[c, j] * vort[k, vertex[c, j]] for j in range(3))
            h = max(0.5 * (w[k, c] + w[k + 1, c]), 0.0) * max(zeta, 0.0)
            zb = max(zgrid[k, c] - zgrid[0, c], 2000.0)
            zt = min(zgrid[k + 1, c] - zgrid[0, c], 5000.0)
            expected += h * max(0.0, zt - zb)
        assert got["updraft_helicity_max"][c] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_cape_without_water_vapour_is_refused_at_arm():
    with pytest.raises(ValueError, match="qv"):
        nest_look_dycore.DycoreLookSources(np, needed=["cape"], cells=np.array([0]), owned_cells=1,
                                           qv_index=None)


def _table(field):
    return {"hex_nests": {"area": {"circle": {"lat": 0.0, "lon": 0.0, "radius_km": 3000}},
                          "nest": {"dx_km": 3, "size_km": 400, "hours": 3},
                          "when": [{"all": [{"field": field, "op": ">=", "value": -1000}]}]}}


@pytest.mark.parametrize("field", ["wmax", "mucape", "uh25"])
def test_every_table_row_arms_on_the_sources_both_routes_now_form(field, tmp_path):
    from woof.hex.mpasphys.nest_look_v842 import route_sources

    table = nest_triggers.resolve(_table(field)["hex_nests"])
    geometry = nest_triggers.area_geometry(
        table["area"], lat_rad=np.zeros(2), lon_rad=np.array([0.0, 0.01]), area_cell=np.ones(2),
        cells_on_cell=np.array([[2], [1]]), n_edges_on_cell=np.array([1, 1]))
    mpas_sources = route_sources(SimpleNamespace(selectors={"microphysics": "mp_wsm6"}, backends={}),
                                 ported_framework=True)
    arwen_sources = ["refl10cm", *nest_look_dycore.SOURCES]
    for sources in (mpas_sources, arwen_sources):
        trigger = nest_triggers.LoopTrigger(table, run_dir=tmp_path, run_start="2025-05-19_12:00:00", dt_s=20.0,
                                            geometry=geometry, xp=np, available_sources=sources)
        assert set(trigger.needed_sources()) <= set(sources)
