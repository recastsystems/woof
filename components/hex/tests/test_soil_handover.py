"""CPU contracts for soil selection, depth remapping and the worker boundary."""
from pathlib import Path

import numpy as np
import pytest

from woof.hex import hrrr_intermediate as hi
from woof.hex.soil_handover import (RUC_DEPTHS_M, RUC_NAMES, node_bounds,
                                   prepare_soil_met, remap_column)
from woof.hex.wps_intermediate import WpsIntermediateReader


def write_met(path, temperature, moisture, *, nodes=True, legacy=True, extra_layers=False,
              source="NCEP HRRR Model", endian=">"):
    """Small WPS column fixture; this tests soil handling, not horizontal remap."""
    target = hi.LatLonTarget(south=38., west=-98., dlat=.025, dlon=.025,
                            nx=temperature.shape[1], ny=temperature.shape[2])
    def record(name, values, units):
        return hi.wps_record_bytes(hdate="2022-03-05_15:00:00", xfcst=0., map_source=source,
                                  field_name=name, units=units, description="soil handover fixture",
                                  level=200100., target=target, values=values.T)
    atmospheric = record("TT", np.full(temperature.shape[1:], 295., dtype=np.float32), "K")
    records = [atmospheric]
    depths = np.rint(RUC_DEPTHS_M*100).astype(int)
    layers = [(0, 10), (10, 40), (40, 100), (100, 200)]
    if nodes:
        for k, cm in enumerate(depths):
            tn, mn = (f"ST{cm:03d}{cm:03d}", f"SM{cm:03d}{cm:03d}") if legacy else RUC_NAMES[k]
            records += [record(tn, temperature[k], "K"), record(mn, moisture[k], "m3 m-3")]
    if not nodes or extra_layers:
        for k, (a, b) in enumerate(layers):
            records += [record(f"ST{a:03d}{b:03d}", temperature[k], "K"),
                        record(f"SM{a:03d}{b:03d}", moisture[k], "m3 m-3")]
    path.write_bytes(b"".join(records))
    return atmospheric


def read_soil(path):
    with WpsIntermediateReader(path) as reader:
        f = {v.field: v for v in reader.iter_fields()}
    return np.stack([f[a].values for a, _ in RUC_NAMES]), np.stack([f[b].values for _, b in RUC_NAMES]), f


def test_identity_is_exact_and_does_not_alias():
    rng = np.random.default_rng(29)
    t = rng.uniform(260, 300, (9, 12, 2)).astype(np.float32)
    q = rng.uniform(.05, .5, t.shape).astype(np.float32)
    ot, oq, r = remap_column(t, q, RUC_DEPTHS_M, node_bounds(RUC_DEPTHS_M))
    assert np.array_equal(t, ot) and np.array_equal(q, oq)
    assert not np.shares_memory(t, ot) and r["path"] == "native-nodes"


def test_preserved_nodes_override_four_layers_and_header_count(tmp_path):
    t = np.arange(9*6, dtype=np.float32).reshape(9, 3, 2)+260
    q = t/1000
    original = tmp_path/"old.met"
    atmosphere = write_met(original, t, q, extra_layers=True)
    fixed, count, r = prepare_soil_met(original, tmp_path/"fixed.met")
    ot, oq, fields = read_soil(fixed)
    assert np.array_equal(t, ot) and np.array_equal(q, oq)
    assert fixed.read_bytes().startswith(atmosphere)
    assert count == 9 and len(fields) == 19 and len(r["discarded_fields"]) == 8
    assert r["source_map_source"] == "NCEP HRRR Model"
    assert r["max_water_error_after_float32_m"] == 0


def test_already_canonical_is_zero_copy(tmp_path):
    t, q = np.full((9, 2, 2), 280., np.float32), np.full((9, 2, 2), .2, np.float32)
    original = tmp_path/"nodes.met"
    write_met(original, t, q, legacy=False)
    fixed, count, r = prepare_soil_met(original, tmp_path/"unused.met")
    assert fixed == original and count == 9 and not r["rewritten"]
    assert not (tmp_path/"unused.met").exists()


def test_four_layer_source_conserves_water(tmp_path):
    rng = np.random.default_rng(84)
    t = rng.uniform(265, 290, (4, 3, 2)).astype(np.float32)
    q = rng.uniform(.05, .5, t.shape).astype(np.float32)
    original = tmp_path/"four.met"
    write_met(original, t, q, nodes=False, source="NCEP GFS")
    fixed, count, r = prepare_soil_met(original, tmp_path/"ruc.met")
    ot, oq, fields = read_soil(fixed)
    before = np.array([.1, .3, .6, 1.]) @ q.reshape(4, -1)
    after = np.diff(node_bounds(RUC_DEPTHS_M)) @ oq.reshape(9, -1)
    assert np.max(abs(before-after)) < 3e-8
    assert count == 9 and r["path"] == "conservative-depth-remap"
    assert (ot > 0).all() and (oq >= 0).all()
    assert r["source_map_source"] == "NCEP GFS"  # never lose donor provenance
    assert fields["TT"].map_source == "NCEP GFS"
    assert fields["SOILT001"].map_source == r["soil_transport_map_source"] == "NCEP RAP Model"


def test_same_number_but_different_depths_is_not_identity():
    z = RUC_DEPTHS_M*1.2
    t = np.linspace(270, 290, 9)[:, None]
    q = np.linspace(.1, .4, 9)[:, None]
    ot, oq, r = remap_column(t, q, z, node_bounds(z))
    assert r["path"] == "conservative-depth-remap"
    assert not np.array_equal(ot, t)
    assert np.max(abs(np.diff(node_bounds(z))@q-np.diff(node_bounds(RUC_DEPTHS_M))@oq)) < 1e-15


def test_zero_water_and_invalid_depths():
    b = np.array([0., .1, .4, 1., 2.]); z = .5*(b[:-1]+b[1:])
    t = np.full((4, 2), 280.)
    _, q, _ = remap_column(t, np.zeros_like(t), z, b)
    assert np.count_nonzero(q) == 0
    with pytest.raises(ValueError):
        remap_column(t, t*.001, z[::-1], b)
    with pytest.raises(ValueError):
        remap_column(t, -t*.001, z, b)


def test_partial_native_column_does_not_mix_with_layers(tmp_path):
    t, q = np.full((9, 2, 2), 280., np.float32), np.full((9, 2, 2), .2, np.float32)
    source = tmp_path/"partial.met"
    write_met(source, t, q, legacy=False)
    # Missing moisture at one node must not silently become a zero first guess.
    data = source.read_bytes().replace(b"SOILM300 ", b"MISSING  ")
    source.write_bytes(data)
    with pytest.raises(ValueError, match="no complete"):
        prepare_soil_met(source, tmp_path/"fixed.met")


def test_target_depths_are_the_producer_contract():
    from woof.hex.mpasphys.rrfs.cold_start_rrfs import load_producers
    for row in load_producers()["producers"].values():
        assert np.array_equal(row["soil"]["node_depths_m"], RUC_DEPTHS_M)


def test_only_ruc_nodes_are_handed_over(tmp_path):
    from woof.hex.soil_handover import carries_ruc_nodes, handover_or_fork
    nodes, layers = tmp_path/"hrrr.met", tmp_path/"gfs.met"
    write_met(nodes, np.full((9, 2, 2), 280., np.float32), np.full((9, 2, 2), .2, np.float32), extra_layers=True)
    write_met(layers, np.full((4, 2, 2), 280., np.float32), np.full((4, 2, 2), .2, np.float32),
              nodes=False, source="NCEP GFS")
    assert carries_ruc_nodes(nodes) and not carries_ruc_nodes(layers)
    met, count, r = handover_or_fork(layers, tmp_path/"x.met", target_depths=list(RUC_DEPTHS_M), requested=4)
    assert met == layers and count == 4 and r["path"] == "fork-interpolation" and not (tmp_path/"x.met").exists()
    met, count, r = handover_or_fork(nodes, tmp_path/"y.met", target_depths=list(RUC_DEPTHS_M), requested=4)
    assert count == 9 and r["path"] == "native-nodes" and r["requested_nfgsoillevels"] == 4
