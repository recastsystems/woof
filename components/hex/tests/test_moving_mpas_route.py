"""The MPAS suite route's moving hooks (woof.hex.mpasphys.moving_v842), host-only.

The whole-forecast evidence is a moving run on the GPU (docs/moving-nest.md); these pin the pieces that
decide what a move carries and how: the Registry-derived field set, the clock and restart flag of a
resumed segment, the surface-role donor, the sealed carrier and the route admission.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from woof.hex.mpasphys import moving_v842 as mv

REGISTRY = """
<registry>
 <var_struct name="state" time_levs="2"><var name="theta_m" dimensions="nVertLevels nCells Time"/></var_struct>
 <var_struct name="diag" time_levs="1"><var name="pressure_p" dimensions="nVertLevels nCells Time"/></var_struct>
 <var_struct name="tend" time_levs="1">
  <var name="rt_diabatic_tend" dimensions="nVertLevels nCells Time"/>
  <var name="tend_theta" dimensions="nVertLevels nCells Time"/>
 </var_struct>
 <var_struct name="sfc_input" time_levs="1">
  <var name="xland" dimensions="nCells Time"/>
  <var name="tslb" dimensions="nSoilLevels nCells Time"/>
  <var name="ivgtyp" dimensions="nCells"/>
 </var_struct>
 <var_struct name="diag_physics" time_levs="1">
  <var name="qke" dimensions="nVertLevels nCells Time"/>
  <var name="refl10cm" dimensions="nVertLevels nCells Time"/>
  <var name="refl10cm_1km_max" dimensions="nCells Time"/>
  <var name="nt_c" dimensions="nCells"/>
  <var name="cldfrac_off" dimensions="nVertLevels nCells Time" packages="off_package"/>
 </var_struct>
 <streams>
  <stream name="restart" type="input;output">
   <var name="theta_m"/><var name="pressure_p"/><var name="rt_diabatic_tend"/><var name="xland"/>
   <var name="tslb"/><var name="ivgtyp"/><var name="qke"/><var name="nt_c"/>
  </stream>
 </streams>
</registry>
"""


def _layout():
    root = ET.fromstring(REGISTRY)
    fields = []
    for struct in root.findall("./var_struct"):
        for var in struct:
            if var.get("packages"):
                continue
            dims = [d for d in var.get("dimensions").split() if d != "Time"]
            fields.append(dict(pool=struct.get("name"), name=var.get("name"), pool_name=var.get("name"),
                               kind="integer" if var.get("name") == "ivgtyp" else "real", dimensions=dims,
                               shape=[4] * len(dims), components=[], nlevels=1))
    enabled = lambda element: not element.get("packages")
    return SimpleNamespace(root=root, fields=fields, enabled=enabled, configs={}, dimensions={"nCells": 3})


def test_carried_set_is_read_from_the_registry():
    names = {(r["pool"], r["name"]) for r in mv.carried_rows(_layout())}
    # restart-stream physics state and single-level time-dependent diagnostics are carried ...
    assert {("diag_physics", "qke"), ("diag_physics", "nt_c"), ("sfc_input", "tslb"),
            ("diag_physics", "refl10cm_1km_max"), ("tend", "rt_diabatic_tend")} <= names
    # ... the dycore's own fields cross through the atmospheric remap, geography comes from the new member,
    # and a vertical diagnostic outside the restart stream is recomputed by its scheme.
    for excluded in (("state", "theta_m"), ("diag", "pressure_p"), ("tend", "tend_theta"), ("sfc_input", "xland"),
                     ("sfc_input", "ivgtyp"), ("diag_physics", "refl10cm")):
        assert excluded not in names


def test_start_time_names_the_mpas_start_and_refuses_two():
    from woof.hex.mpasphys.forecast_v842 import apply_start_time, namelist_option

    text = "&physics\nconfig_physics_suite='convection_permitting'\n/\n"
    assert apply_start_time(text, None) == text
    set_text = apply_start_time(text, "2025-11-08_12:00:00")
    assert namelist_option(set_text, "nhyd_model", "config_start_time") == "2025-11-08_12:00:00"
    assert namelist_option(set_text, "physics", "config_physics_suite") == "convection_permitting"
    named = "&nhyd_model\nconfig_start_time='2025-11-08_12:00:00'\n/\n"
    assert apply_start_time(named, "2025-11-08_12:00:00") == named
    with pytest.raises(ValueError, match="two starts"):
        apply_start_time(named, "2025-11-09_00:00:00")


def test_a_resumed_segment_is_a_restart_at_the_carrier_time():
    from woof.hex.mpasphys.forecast_v842 import namelist_option

    route = mv.MovingRoute(carrier=dict(valid_time="2025-11-08_13:00:00", time_seconds=3600.0),
                           origin_time="2025-11-08_12:00:00", offset_seconds=3600.0)
    text = route.resume_namelist("&nhyd_model\nconfig_start_time='2025-11-08_12:00:00'\nconfig_dt=225\n/\n")
    assert namelist_option(text, "nhyd_model", "config_start_time") == "2025-11-08_13:00:00"
    assert namelist_option(text, "nhyd_model", "config_dt") == 225
    assert namelist_option(text, "restart", "config_do_restart") is None   # set after the pools are read
    layout = SimpleNamespace(configs={"config_do_restart": False})
    route.mark_restart(layout)
    assert layout.configs["config_do_restart"] is True
    with pytest.raises(ValueError, match="started 2025-11-08_12:00:00"):
        route.resume_namelist("&nhyd_model\nconfig_start_time='2025-11-09_00:00:00'\n/\n")
    route.frame_seconds = 1800.0
    route.begin(None)
    route.offset_seconds = 900.0
    with pytest.raises(ValueError, match="history frame"):
        route.begin(None)


def test_surface_donor_keeps_the_role_of_the_new_cell():
    # Three source cells (land, water, land with another soil); two target cells (water, land).
    overlap = csr_matrix(np.array([[0.7, 0.3, 0.0], [0.0, 0.6, 0.4]]))
    weights = SimpleNamespace(overlap=overlap)
    source = mv._roles(dict(xland=[1, 2, 1], xice=[0, 0, 0], isltyp=[3, 14, 5], ivgtyp=[7, 17, 7]), 0.5, 15)
    target = mv._roles(dict(xland=[2, 1], xice=[0, 0], isltyp=[14, 5], ivgtyp=[17, 7]), 0.5, 15)
    xyz = np.eye(3)
    donors, receipt = mv.coherent_donors(weights, source, target, xyz, xyz[:2])
    # the largest overlap of target 0 is land: the water cell is taken; target 1 keeps its land soil class
    assert donors.tolist() == [1, 2]
    assert receipt["role_mismatch_cells"] == 2


def test_surface_donor_falls_back_to_the_nearest_cell_of_the_same_role():
    overlap = csr_matrix(np.array([[1.0, 0.0, 0.0]]))
    source = mv._roles(dict(xland=[1, 1, 2], xice=[0, 0, 0], isltyp=[3, 3, 14], ivgtyp=[7, 7, 17]), 0.5, 15)
    target = mv._roles(dict(xland=[2], xice=[0], isltyp=[14], ivgtyp=[17]), 0.5, 15)
    xyz = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    donors, receipt = mv.coherent_donors(SimpleNamespace(overlap=overlap), source, target, xyz, xyz[:1])
    assert donors.tolist() == [2] and receipt["nearest_role_donors"] == 1


def test_fields_map_by_their_vertical_axis():
    class Map:
        overlap = csr_matrix(np.ones((1, 2)) * 0.5)
        source_area = np.ones(2)

        def intensive(self, values):
            return np.asarray(values, float).mean(axis=-1, keepdims=True)

        def layer_intensive(self, values):
            return np.asarray(values, float)[:, :1] * 10

        def interface_intensive(self, values):
            return np.asarray(values, float)[:, 1:] * 100

    values = np.array([[1.0, 3.0], [5.0, 7.0]])
    assert mv.map_field(values, ["nVertLevels", "nCells"], Map()).tolist() == [[10.0], [50.0]]
    assert mv.map_field(values, ["nVertLevelsP1", "nCells"], Map()).tolist() == [[300.0], [700.0]]
    assert mv.map_field(values, ["nSoilLevels", "nCells"], Map()).tolist() == [[2.0], [6.0]]
    assert mv.map_field(values, ["nCells"], Map(), donors=np.array([1])).tolist() == [[3.0], [7.0]]


def test_the_carrier_is_sealed_and_route_bound(tmp_path):
    path = tmp_path / "boundary.pkl"
    payload = dict(schema=mv.SCHEMA, time_seconds=3600.0, valid_time="2025-11-08_13:00:00", origin_time="x")
    receipt = mv._write_sealed(path, payload, {})
    assert json.loads(path.with_suffix(".pkl.json").read_text())["sha256"] == receipt["sha256"]
    assert mv.load_carrier(path)["time_seconds"] == 3600.0
    other = tmp_path / "arwen.pkl"
    mv._write_sealed(other, dict(payload, schema="gpuwm-hex.moving-boundary/v1"), {})
    with pytest.raises(ValueError, match="WOOF route"):
        mv.load_carrier(other)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sealed receipt"):
        mv.load_carrier(path)


def test_zone_carry_is_the_named_gap_and_stops_the_controller(tmp_path):
    from woof.hex.fullstep_lts_admission import MOVING_ROUTE_REFUSAL
    from woof.hex.moving_run import route_refusal

    with pytest.raises(ValueError) as caught:
        mv.MovingRoute.from_environ(SimpleNamespace(history_every_minutes=60),
                                    {"WOOF_HEX_MOVING_ZONE": "/z.zone.npz"})
    text = str(caught.value)
    assert text.startswith(MOVING_ROUTE_REFUSAL) and "zone carry" in text
    (tmp_path / "forecast.log").write_text("woof hex: " + text + "\n")
    assert route_refusal(tmp_path) == text


def test_the_controller_reads_the_mpas_route_receipt(tmp_path):
    from woof.hex.moving_run import Controller

    (tmp_path / "fc").mkdir()
    control = {"stopped_on_request": {"stop_at_seconds": 3600.0}}
    remap = {"conservation": {"mass": 1e-12}, "atmosphere_remap_seconds": 2.0,
             "physics": {"surface_donors": {"role_mismatch_cells": 4}}}
    (tmp_path / "fc" / "native-physics-receipt.json").write_text(json.dumps(
        {"moving_control": control, "moving_remap": remap, "moving_boundary_export": {"bytes": 1},
         "steps_completed": 16}))
    out = Controller.segment_receipt(None, tmp_path)
    assert out["control"] == control and out["route"] == "mpas-suite"
    assert out["conservation"] == {"mass": 1e-12} and out["physics_carry"] == {"role_mismatch_cells": 4}


@pytest.mark.parametrize("name", ["GPUWM_HEX_MOVING_ANALYSIS", "GPUWM_HEX_MOVING_BACKGROUND"])
def test_mpas_route_never_silently_ignores_moving_analysis(name):
    from woof.hex.fullstep_lts_admission import MOVING_ROUTE_REFUSAL
    with pytest.raises(ValueError, match='analysis insertion') as error:
        mv.MovingRoute.from_environ(SimpleNamespace(), {name:'request.json'})
    assert str(error.value).startswith(MOVING_ROUTE_REFUSAL)
