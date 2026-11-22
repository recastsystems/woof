"""A hex export from a run on a RUC land surface prepares a WOOF-ArWen child (the hex-history RUC nine-node variant).

THE BREAKAGE THIS PREVENTS (WOOF run_0368311be596aa52, 2026-11-10): a WOOF-ArWen downscale of a hex run on the RRFS v2 suite
was refused at prep, "soil_temperature selector layer_value=10.0 matches 0 entries of 'soil_layers_stag' ([0.0, 0.5,
3.0, 7.5, 20.5, 45.5, 80.5, 130.5, 230.5])": the only hex-history profile named Noah's four layers.
"""
import json

import netCDF4
import numpy as np
import pytest

from woof import source_authorities as authorities

BASE = "hex-history-netcdf-v1"
RUC9 = "hex-history-ruc9-netcdf-v1"
# What rw_mpas_convert writes for each: the running sum of the init's dzs, in cm.
RUC9_LABELS = [0.0, 0.5, 3.0, 7.5, 20.5, 45.5, 80.5, 130.5, 230.5]
NOAH_LABELS = [0.0, 10.0, 40.0, 100.0]


def _frame(path, labels):
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("soil_layers_stag", len(labels))
        d.createVariable("soil_layers_stag", "f8", ("soil_layers_stag",))[:] = np.array(labels)
    return path


def test_the_variant_ships_pinned_and_is_in_the_hex_family():
    assert authorities.packaged_profile_family(BASE) == (BASE, RUC9)
    assert set(authorities.packaged_authorities(RUC9)) == {"mapping", "composition", "provenance"}
    assert authorities.packaged_profile(RUC9)["data_role"] == authorities.packaged_profile(BASE)["data_role"]


def test_the_variant_reads_the_nine_ruc_nodes_at_their_depths():
    composition = json.loads(authorities.packaged_authorities(RUC9)["composition"].read_text())
    nodes = composition["soil_layers"]["source_nodes"]
    assert [n["depth"] for n in nodes] == [0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0]
    assert [n["selectors"]["soil_temperature"]["layer_value"] for n in nodes] == RUC9_LABELS
    assert composition["soil_layers"]["remap"]["kind"] == "linear_node_samples"
    mapping = json.loads(authorities.packaged_authorities(RUC9)["mapping"].read_text())
    for field in ("soil_temperature", "volumetric_soil_moisture"):
        assert [s["layer_value"] for s in mapping["fields"][field]["selectors"]] == RUC9_LABELS
    # Everything but the soil column is the Noah profile's.
    base = json.loads(authorities.packaged_authorities(BASE)["mapping"].read_text())
    assert {k: v for k, v in mapping["fields"].items() if "soil" not in k} == {
        k: v for k, v in base["fields"].items() if "soil" not in k}


def test_the_frames_choose_the_profile(tmp_path):
    ruc = _frame(tmp_path / "ruc.nc", RUC9_LABELS)
    noah = _frame(tmp_path / "noah.nc", NOAH_LABELS)
    other = _frame(tmp_path / "other.nc", [0.0, 5.0])
    assert authorities.packaged_variant_for_inputs(BASE, [ruc, noah]) == RUC9
    assert authorities.packaged_variant_for_inputs(BASE, [noah]) == BASE
    assert authorities.packaged_variant_for_inputs(BASE, [other]) == BASE  # its own decode names the coordinate
    assert authorities.packaged_variant_for_inputs(BASE, [tmp_path / "not-yet.nc"]) == BASE
    assert authorities.packaged_variant_for_inputs("hrrr-prs-grib2-v1", [ruc]) == "hrrr-prs-grib2-v1"


def test_a_preparation_is_held_to_the_variant_it_bound():
    pins = authorities.packaged_authority_sha256(RUC9)
    assert authorities.packaged_variant_for_composition(BASE, pins["composition"]) == RUC9
    assert authorities.packaged_variant_for_composition(
        BASE, authorities.packaged_authority_sha256(BASE)["composition"]) == BASE
    assert authorities.packaged_variant_for_composition(BASE, "0" * 64) == BASE
