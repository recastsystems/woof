from pathlib import Path
from woof.config import RunConfig
from woof.namelist_import import PHYSICS_PINS

def test_skin_admission_and_kernel():
    assert "sst_skin" not in {row[0] for row in PHYSICS_PINS}
    assert RunConfig(nx=4, ny=4, nz=16, dx=1000., dy=1000., ztop=8000., dt=10., run_seconds=0.).sst_skin == 0
    assert Path("woof/core/kernels/sstskin.cu").is_file()

def test_output_schema_and_restart_off_identity():
    from woof.io.wrf_output_schema import SSTSKIN_OUTPUT_FIELDS
    assert [(r.netcdf_name, r.units) for r in SSTSKIN_OUTPUT_FIELDS.values()] == [("DTW","C")]
    from woof.io.restart import _drop_inert_terrain_drag
    values = {"sst_skin":0}
    _drop_inert_terrain_drag(values)
    assert "sst_skin" not in values
    values = {"sst_skin":1}
    _drop_inert_terrain_drag(values)
    assert values["sst_skin"] == 1

def test_skin_allocations_and_history_are_declared():
    from woof.core.preflight import physics_array_shapes
    from woof.io.history_layout import produced_history_shapes
    from dataclasses import replace
    cfg = RunConfig(nx=4,ny=4,nz=16,dx=1000.,dy=1000.,ztop=8000.,dt=10.,run_seconds=0.,sf_sfclay_physics=1)
    assert "fields/dtw" not in physics_array_shapes(cfg)
    off_history = produced_history_shapes(cfg)
    assert off_history["SSTSK"] == (4,4)
    assert "DTW" not in off_history
    skin = replace(cfg,sst_skin=1)
    shapes = physics_array_shapes(skin)
    for key in ("sst","sstsk","dtw","gsw"):
        assert shapes[f"fields/{key}"] == (4,4)
    for key in ("SST","SSTSK","DTW"):
        assert produced_history_shapes(skin)[key] == (4,4)


def test_skin_selector_appends_without_changing_old_positional_fields():
    from dataclasses import fields
    names = [f.name for f in fields(RunConfig)]
    start = names.index("sst_skin")
    assert names[start:start+5] == ["sst_skin", "sst_update", "surface_analysis", "tmn_update", "lagday"]
    assert all(RunConfig.__dataclass_fields__[name].kw_only for name in names[start:])
    assert RunConfig.__dataclass_fields__["sst_skin"].kw_only
    assert RunConfig.__dataclass_fields__["sst_skin"].default == 0


def test_skin_off_preserves_public_documents_and_on_binds_identity():
    from dataclasses import replace
    from woof.experiment import domain_config_document, experiment_config_document
    from woof.core.model import restart_identity_payload
    from test_radiation_driver_identity import _experiment
    off = _experiment()
    assert "sst_skin" not in domain_config_document(off.root)["run"]
    assert "sst_skin" not in experiment_config_document(off)["domains"][0]["run"]
    on = replace(off, domains=(replace(off.root, run=replace(off.root.run, sst_skin=1)),))
    assert domain_config_document(on.root)["run"]["sst_skin"] == 1
    assert experiment_config_document(on)["domains"][0]["run"]["sst_skin"] == 1
    assert restart_identity_payload(on)["domains"][0]["run"]["sst_skin"] == 1


def test_skin_kernel_is_priced_only_when_selected():
    from test_radiation_driver_identity import _experiment
    from woof.core.preflight import domain_kernel_modules, KERNEL_MAX_LOCAL_SIZE_BYTES
    from woof.core.kernel_frame_recordings import frame_ceiling
    assert "sstskin" not in domain_kernel_modules(_experiment().root, prices_refl=False)
    assert "sstskin" in domain_kernel_modules(_experiment(sst_skin=1, sf_sfclay_physics=91).root, prices_refl=False)
    assert KERNEL_MAX_LOCAL_SIZE_BYTES["sstskin"] == frame_ceiling()["sstskin"] == 0
