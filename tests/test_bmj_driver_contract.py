"""BMJ admission and persistent diagnostic contract."""
import pytest
from woof.config import RunConfig, validate_run_config
from woof.core.column_cumulus import column_kernel_spec


def _config(**kwargs):
    return RunConfig(nx=12, ny=8, nz=45, dx=12000., dy=12000.,
                     ztop=14000., dt=30., run_seconds=0., **kwargs)


def test_bmj_default_cadence_is_admitted():
    cfg = validate_run_config(_config(cu_physics=2, moist=True))
    assert cfg.cudt_minutes == 5.0


def test_bmj_cloud_and_precipitation_outputs_are_persistent():
    spec = column_kernel_spec(2)
    assert set(spec.persist_outputs) == {
        "qcconv", "qiconv", "ccldfra", "convcld", "htop", "hbot",
        "raincv", "pratec"}
    assert spec.feedback["parameter"] == "bmj_rad_feedback"


def test_bmj_feedback_has_a_real_radiation_consumer():
    cfg = validate_run_config(_config(
        cu_physics=2, moist=True, ra_physics=4, bmj_rad_feedback=True))
    assert cfg.bmj_rad_feedback


@pytest.mark.parametrize('pair', [(0, 0), (1, 0), (4, 1), (5, 5), (5, 4), (4, 5)])
def test_bmj_feedback_accepts_physical_mixed_and_disabled_streams(pair):
    cfg = validate_run_config(_config(
        cu_physics=2, moist=True, ra_physics=0,
        ra_lw_physics=pair[0], ra_sw_physics=pair[1], bmj_rad_feedback=True))
    assert cfg.bmj_rad_feedback


def test_bmj_namelist_keeps_each_domain_cadence_and_feedback(tmp_path):
    import tomllib
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text=INPUT_TEXT.replace(' cu_physics = 1, 0,',' cu_physics = 2, 2,')
    text=text.replace(' cudt = 5, 0,',' cudt = 1.5, 3,\n bmj_rad_feedback = .true., .false.,')
    toml,_=import_namelists(*_pair(tmp_path,inp=text))
    data=tomllib.loads(toml)
    assert [row['cudt_minutes'] for row in data['domain']] == [1.5,3.]
    assert [row['bmj_rad_feedback'] for row in data['domain']] == [True,False]


def test_feedback_rejects_a_radiation_adapter_without_a_cloud_merge():
    import pytest
    with pytest.raises(ValueError,match='does not merge the carried convective cloud'):
        validate_run_config(_config(cu_physics=2,moist=True,ra_physics=90,bmj_rad_feedback=True))


def test_preflight_prices_the_restart_carried_bmj_fields():
    from woof.core.preflight import physics_array_shapes
    cfg=_config(cu_physics=2,moist=True)
    shapes=physics_array_shapes(cfg)
    assert shapes['fields/cldefi'] == (8,12)
    assert shapes['fields/qcconv'] == (45,8,12)
    assert shapes['fields/qiconv'] == (45,8,12)
    assert shapes['fields/ccldfra'] == (45,8,12)
    assert shapes['fields/convcld'] == (8,12)


def test_bmj_history_contains_carried_state_without_changing_other_schemes():
    from dataclasses import replace
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import BMJ_OUTPUT_FIELDS
    cfg = _config(cu_physics=2, moist=True)
    shapes = produced_history_shapes(cfg)
    assert all(row.netcdf_name in shapes for row in BMJ_OUTPUT_FIELDS.values())
    assert shapes["CLDEFI"] == (cfg.ny, cfg.nx)
    assert shapes["CCLDFRA"] == (cfg.nz, cfg.ny, cfg.nx)
    off = produced_history_shapes(replace(cfg, cu_physics=0))
    assert not ({row.netcdf_name for row in BMJ_OUTPUT_FIELDS.values()} & off.keys())
