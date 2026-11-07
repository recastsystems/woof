"""Native adaptive-habit admission and the full transported registry package."""
import ast
import os
import subprocess
from pathlib import Path
import numpy as np
from woof.config import RunConfig, MP_PHYSICS_ACCEPTED
from woof.physics_registry import consumer_row_for_selector

ROOT=Path(__file__).resolve().parents[1]

def test_namelist_55_keeps_native_selector():
    if os.environ.get("WOOF_ISHMAEL_BASELINE") == "1":
        source=subprocess.check_output(["git","show","integrate/2.8.7:woof/namelist_import.py"],cwd=ROOT,text=True)
        tree=ast.parse(source)
        assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_MP_MAP" for t in n.targets))
        mapped=ast.literal_eval(assignment.value)[55][0]
    else:
        from woof.namelist_import import _MP_MAP
        mapped=_MP_MAP[55][0]
    assert mapped == 55, "mp_physics=55 still imports a different microphysics scheme"

def test_native_state_allocates_every_registry_moment():
    from woof.core.state import DomainState
    from woof.core.device_inventory import state_array_shapes
    from woof.io.restart import classify_state_attr
    cfg=RunConfig(nx=12,ny=12,nz=48,dx=3000,dy=3000,ztop=16000,dt=10,run_seconds=60,moist=True,mp_physics=55)
    state=DomainState(cfg,array_module=np)
    row=consumer_row_for_selector("microphysics","native_column",55)
    assert 55 in MP_PHYSICS_ACCEPTED
    assert len(row["transport"]) == 16
    for name in row["transport"]:
        assert getattr(state,name).shape == (48,12,12)
        assert getattr(state,name+"0").shape == (48,12,12)
        assert classify_state_attr(name) == "serialize"
        assert classify_state_attr(name+"0") == "rebuild"
    assert tuple(row["mass_slots"]) == ("qi", "qi2", "qi3")
    assert getattr(state,"qs",None) is None and getattr(state,"qg",None) is None
    shapes=state_array_shapes(cfg)
    for name in row["state_fields"]:
        assert shapes[name] == (48,12,12)

def test_all_habit_moments_have_advection_and_nest_inventory():
    from woof.core.nest_fields import nest_field_kinds
    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    from woof.core.preflight import scratch_slot_registry, scratch_slot_lifetime
    cfg=RunConfig(nx=12,ny=12,nz=48,dx=3000,dy=3000,ztop=16000,dt=10,run_seconds=60,moist=True,mp_physics=55,km_opt=4)
    row=consumer_row_for_selector("microphysics","native_column",55)
    assert set(row["transport"]) <= set(nest_field_kinds(cfg))
    assert set(row["transport"]) <= COUPLED_SCALAR_STATE_FIELDS
    slots=scratch_slot_registry(cfg)
    for name in row["transport"]:
        slot="smag_r"+name
        assert slots[slot] == (48,12,12)
        assert scratch_slot_lifetime(slot) is not None
    for slot in list(row["diagnostics"].values()) + list(row["scratch"].values()):
        assert slots[slot] == (48,12,12)
        assert scratch_slot_lifetime(slot) is not None


def test_wrfinput_door_requires_every_native_prognostic():
    from woof.ingest.wrfinput import active_moisture_inventory, active_moisture_map, SUPPORTED_MICROPHYSICS, UNSUPPORTED_MICROPHYSICS
    cfg=RunConfig(nx=12,ny=12,nz=48,dx=3000,dy=3000,ztop=16000,dt=10,run_seconds=60,moist=True,mp_physics=55)
    required, allowed=active_moisture_inventory(cfg)
    assert required == allowed
    assert len(required) == 16
    assert {"QVOLI", "QAOLI", "QVOLI2", "QAOLI2", "QVOLI3", "QAOLI3", "QICE2", "QICE3"} <= required
    assert set(active_moisture_map(cfg).values()) == set(consumer_row_for_selector("microphysics","native_column",55)["transport"])
    assert 55 in SUPPORTED_MICROPHYSICS and 55 not in UNSUPPORTED_MICROPHYSICS


def test_wrf_restart_reads_each_habit_instead_of_accepting_only_zero_cold_starts():
    from woof.ingest.wrfrst import WRFRST_DISPOSITIONS
    from woof.ingest.wrfinput import active_moisture_map
    cfg = RunConfig(nx=12, ny=12, nz=48, dx=3000, dy=3000, ztop=16000,
                    dt=10, run_seconds=60, moist=True, mp_physics=55)
    for name in active_moisture_map(cfg):
        assert WRFRST_DISPOSITIONS[name].kind == "read", name


def test_native_wrfinput_geometry_maps_all_registered_fields():
    from woof.ingest.wrfinput import WRFINPUT_DIMENSIONS, active_moisture_map
    cfg=RunConfig(nx=12,ny=12,nz=48,dx=3000,dy=3000,ztop=16000,dt=10,run_seconds=60,moist=True,mp_physics=55)
    for name in active_moisture_map(cfg):
        assert WRFINPUT_DIMENSIONS[name] == ("bottom_top","south_north","west_east")


def test_native_workspace_cannot_break_the_final_digest():
    from woof.io.restart import classify_scratch_slot
    row=consumer_row_for_selector("microphysics","native_column",55)
    for slot in list(row["diagnostics"].values())+list(row["scratch"].values())+["native_column_radiation_ice"]:
        assert classify_scratch_slot(slot) == "rebuild"


def test_native_history_preserves_all_ice_moments_and_hybrid_mass():
    from woof.core.state import DomainState
    from woof.io.history_layout import live_state_history_fields
    cfg=RunConfig(nx=12,ny=12,nz=48,dx=3000,dy=3000,ztop=16000,dt=10,run_seconds=60,moist=True,mp_physics=55)
    state=DomainState(cfg,array_module=np)
    fields=live_state_history_fields(state)
    for name in ("QICE","QICE2","QICE3","QVOLI","QAOLI","QVOLI2","QAOLI2","QVOLI3","QAOLI3","C1H","C2H"):
        assert name in fields
    assert fields["C1H"] is state.c1h and fields["C2H"] is state.c2h


def test_default_cumulus_uses_wrfs_no_snow_feedback_branch():
    from woof.core.kf import kf_phase_mode_for_microphysics, KFPhaseMode
    row=consumer_row_for_selector("microphysics","native_column",55)
    assert "qi" in row["transport"] and "qs" not in row["transport"]
    assert kf_phase_mode_for_microphysics(55) == KFPhaseMode.NO_SEPARATE_SNOW
