"""Noah mosaic selector law and default-off checkpoint compatibility."""
from dataclasses import asdict, fields, replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from woof.config import RunConfig, load_config, validate_noah_mosaic_config, validate_run_config
from woof.io.restart import _configuration_digest_values, _mosaic_checkpoint_config, _require_config_match


def _cfg(**kw):
    return RunConfig(nx=12, ny=12, nz=8, dx=3000.0, dy=3000.0,
                     ztop=12000.0, dt=6.0, run_seconds=60.0, **kw)



@pytest.mark.parametrize("option", [True, False, 1.0, "1", -1, 2])
def test_selector_refuses_skipped_noah(option):
    with pytest.raises(ValueError, match="neither Noah arm.*land surface would not be integrated"):
        validate_noah_mosaic_config(_cfg(sf_surface_mosaic=option))


@pytest.mark.parametrize("lsm", [0, 3, 4])
def test_non_noah_refuses_ignored_switch(lsm):
    with pytest.raises(ValueError, match="every other LSM silently ignores"):
        validate_noah_mosaic_config(_cfg(sf_surface_mosaic=1, sf_surface_physics=lsm))


@pytest.mark.parametrize("count", [0, -1, True, False, 1.5, "3"])
def test_bad_count_refuses_no_tile(count):
    with pytest.raises(ValueError, match="Noah has no tile to integrate"):
        validate_noah_mosaic_config(_cfg(sf_surface_mosaic=1, sf_surface_physics=2, mosaic_cat=count))


@pytest.mark.parametrize("urban", [2, 3])
def test_urban_refuses_wrfs_unsupported_pairing(urban):
    cfg = SimpleNamespace(sf_surface_mosaic=1, sf_surface_physics=2, mosaic_cat=3, sf_urban_physics=urban)
    with pytest.raises(ValueError, match="mosaic option cannot work with urban options 2 and 3"):
        validate_noah_mosaic_config(cfg)


def test_ucm_pairing_admitted():
    validate_noah_mosaic_config(_cfg(sf_surface_mosaic=1, sf_surface_physics=2, sf_urban_physics=1))


def test_off_does_not_read_count_or_change_checkpoint_identity():
    cfg = _cfg()
    validate_run_config(cfg)
    # Pin every preexisting default through the baseline's last field.
    legacy = {}
    for field in fields(cfg):
        legacy[field.name] = getattr(cfg, field.name)
        if field.name == "min_time_step_sound":
            break
    # e13fa45c0 / 59f7e280f enable moist_cq by default. The pre-mosaic
    # anchor was recorded with it disabled; unwind only that later default.
    assert legacy["moist_cq"] is True
    historical = dict(legacy, moist_cq=False)
    def digest(values):
        return hashlib.sha256(json.dumps(
            values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    assert digest(legacy) != digest(historical)
    assert digest(historical) == "feb433beb07ab5f7056b98fd684ddea7ef7eaef00dd164d4bd2c93f5ac7519fb"
    before = asdict(cfg)
    # This pre-G3 header has no inactive cumulus-only controls.
    before.pop("cugd_avedx")
    before.pop("maxiens")
    before.pop("maxens")
    before.pop("maxens2")
    before.pop("maxens3")
    before.pop("ensdim")
    before.pop("cu_diag")
    before.pop("convtrans_avglen_m")
    before.pop("sf_surface_mosaic")
    before.pop("mosaic_cat")
    before.pop("mosaic_urban_canopy")
    # A checkpoint written before the mosaic trio also predates the two
    # diffusion selectors appended after it (lane/282-namelist-tolerance),
    # whose defaults the same echo drops for the same reason.
    before.pop("diff_opt")
    before.pop("mix_full_fields")
    # The CLM lake quartet appended later is dropped by the same echo when off.
    for name in ("sf_lake_physics", "use_lakedepth", "lakedepth_default", "lake_min_elev"):
        before.pop(name)
    # So is the ocean mixed layer's quartet (2.9.1) when off.
    for name in ("sf_ocean_physics", "oml_hml0", "oml_gamma", "oml_relaxation_time"):
        before.pop(name)
    # lane/cap-slab-ocean's 3-D PWP controls drop with the ocean off too.
    for name in ("ocean_levels", "ocean_analysis", "ocean_analysis_source",
                 "ocean_mld_climatology", "omdt", "ocean_z", "ocean_t",
                 "ocean_s"):
        before.pop(name)
    # The Noah-MP snow-aging selector (2.9.0) is read only by
    # sf_surface_physics = 4; this Noah run's comparison drops it.
    before.pop("noahmp_snow_age")
    # The Fitch wind-farm controls (labs/windfarm, 2.9.11) drop with farm
    # physics off (woof.wind_table.drop_inert_config).
    for name in ("wind_farm_scheme", "wind_farms", "wind_farm_table",
                 "windfarm_tke_factor", "wind_farm_max_dx_m"):
        before.pop(name)
    assert _configuration_digest_values(asdict(cfg)) == _configuration_digest_values(before)
    assert _mosaic_checkpoint_config(asdict(cfg)) == before
    _require_config_match(before, cfg, "old checkpoint")
    validate_run_config(replace(cfg, mosaic_cat=object()))
    assert (cfg.sf_surface_mosaic, cfg.mosaic_cat) == (0, 3)


def test_on_digest_binds_count_and_refuses_resume_change():
    cfg = _cfg(sf_surface_mosaic=1, sf_surface_physics=2)
    assert _configuration_digest_values(asdict(cfg))["mosaic_cat"] == 3
    from woof.io.restart import RestartMismatchError
    with pytest.raises(RestartMismatchError, match="mosaic_cat"):
        _require_config_match(asdict(cfg), replace(cfg, mosaic_cat=5), "checkpoint")


def test_real_toml_loader(tmp_path):
    path = tmp_path / "mosaic.toml"
    path.write_text("[grid]\nnx = 12\nny = 12\nnz = 8\ndx = 3000.0\ndy = 3000.0\nztop = 12000.0\n[run]\ndt = 6.0\nrun_seconds = 60.0\nsf_surface_physics = 2\nsf_sfclay_physics = 1\nbl_pbl_physics = 1\nsf_surface_mosaic = 1\nmosaic_cat = 5\n")
    cfg = load_config(path)
    assert (cfg.sf_surface_mosaic, cfg.mosaic_cat) == (1, 5)


# ---------------------------------------------------------------------------
# mosaic_urban_canopy: WRF's dominant-urban rule by default, the town rule as
# a named option (woof/config.py MOSAIC_URBAN_CANOPY_RULES)
# ---------------------------------------------------------------------------

def _urban_mosaic(**kw):
    return _cfg(sf_surface_mosaic=1, sf_surface_physics=2, sf_urban_physics=1,
                **kw)


def test_the_two_rules_are_a_table_and_wrfs_is_the_default():
    from woof.config import (MOSAIC_URBAN_CANOPY_DEFAULT,
                              MOSAIC_URBAN_CANOPY_RULES)
    assert set(MOSAIC_URBAN_CANOPY_RULES) == {"dominant", "every_tile"}
    assert MOSAIC_URBAN_CANOPY_DEFAULT == "dominant"
    assert RunConfig.__dataclass_fields__["mosaic_urban_canopy"].default == "dominant"
    assert "WRF v4.7.1" in MOSAIC_URBAN_CANOPY_RULES["dominant"]
    validate_noah_mosaic_config(_urban_mosaic())
    validate_noah_mosaic_config(_urban_mosaic(mosaic_urban_canopy="every_tile"))


@pytest.mark.parametrize("rule", ["town", "", "DOMINANT", "every-tile", None, 1, True])
def test_an_unknown_rule_is_refused_by_name(rule):
    with pytest.raises(ValueError, match="mosaic_urban_canopy must be one of.*dominant.*every_tile"):
        validate_noah_mosaic_config(_urban_mosaic(mosaic_urban_canopy=rule))


def test_the_town_rule_without_tiles_is_refused():
    with pytest.raises(ValueError, match="needs sf_surface_mosaic = 1.*no urban tile"):
        validate_noah_mosaic_config(_cfg(sf_surface_physics=2, sf_urban_physics=1,
                                         mosaic_urban_canopy="every_tile"))


@pytest.mark.parametrize("urban", [0, 2, 3])
def test_the_town_rule_without_the_single_layer_canopy_is_refused(urban):
    cfg = SimpleNamespace(sf_surface_mosaic=1, sf_surface_physics=2, mosaic_cat=3,
                          sf_urban_physics=urban, mosaic_urban_canopy="every_tile")
    with pytest.raises(ValueError, match="needs sf_urban_physics = 1"):
        validate_noah_mosaic_config(cfg)


def test_the_default_rule_is_accepted_everywhere_the_town_rule_is_not():
    # "dominant" is what every configuration ran before the key existed, so
    # it is legal with mosaic off, urban off and every LSM.
    validate_run_config(_cfg())
    for kw in ({}, {"sf_surface_physics": 4}, {"sf_surface_physics": 0},
               {"sf_urban_physics": 1}, {"sf_urban_physics": 2}):
        validate_noah_mosaic_config(_cfg(**kw))


def test_the_default_rule_moves_no_identity_and_the_town_rule_binds():
    from woof.io.restart import RestartMismatchError
    for cfg in (_cfg(), _urban_mosaic()):
        values = asdict(cfg)
        assert "mosaic_urban_canopy" not in _mosaic_checkpoint_config(values)
        assert "mosaic_urban_canopy" not in _configuration_digest_values(values)
    town = _urban_mosaic(mosaic_urban_canopy="every_tile")
    assert _mosaic_checkpoint_config(asdict(town))["mosaic_urban_canopy"] == "every_tile"
    assert _configuration_digest_values(asdict(town))["mosaic_urban_canopy"] == "every_tile"
    with pytest.raises(RestartMismatchError, match="mosaic_urban_canopy"):
        _require_config_match(asdict(town), _urban_mosaic(), "checkpoint")
    with pytest.raises(RestartMismatchError, match="mosaic_urban_canopy"):
        _require_config_match(asdict(_urban_mosaic()), town, "checkpoint")
    _require_config_match(asdict(_urban_mosaic()), _urban_mosaic(), "checkpoint")


def test_the_experiment_fingerprint_drops_mosaic_off_and_wrfs_rule():
    from woof.core.model import restart_identity_payload
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold
    exp = load_scaffold()
    for domain in restart_identity_payload(exp)["domains"]:
        for name in ("sf_surface_mosaic", "mosaic_cat", "mosaic_urban_canopy"):
            assert name not in domain["run"]

    def with_run(**kw):
        return replace(exp, domains=tuple(
            replace(domain, run=replace(domain.run, **kw)) for domain in exp.domains))
    on = restart_identity_payload(with_run(sf_surface_mosaic=1, sf_surface_physics=2,
                                           sf_urban_physics=1))
    assert all(d["run"]["sf_surface_mosaic"] == 1 and "mosaic_urban_canopy" not in d["run"]
               for d in on["domains"])
    town = restart_identity_payload(with_run(sf_surface_mosaic=1, sf_surface_physics=2,
                                             sf_urban_physics=1,
                                             mosaic_urban_canopy="every_tile"))
    assert all(d["run"]["mosaic_urban_canopy"] == "every_tile" for d in town["domains"])


def test_preparation_never_reads_the_mosaic_keys():
    # One prepared cache serves mosaic off, WRF's rule and the town rule:
    # the three keys are preparation-inert, so only the switch differs
    # between the arms of a comparison.
    from woof.ingest.prepared_cache import (PREPARATION_INERT_RUN_FIELDS,
                                             effective_prepared_domain_config)
    for name in ("sf_surface_mosaic", "mosaic_cat", "mosaic_urban_canopy"):
        assert f"run.{name}" in PREPARATION_INERT_RUN_FIELDS
    base = {"run": {"sf_surface_mosaic": 0, "mosaic_cat": 3,
                    "mosaic_urban_canopy": "dominant", "mp_physics": 8}}
    town = {"run": {"sf_surface_mosaic": 1, "mosaic_cat": 5,
                    "mosaic_urban_canopy": "every_tile", "mp_physics": 8}}
    assert effective_prepared_domain_config(base) == effective_prepared_domain_config(town)


def test_real_toml_loader_reads_the_rule(tmp_path):
    lines = ["[grid]", "nx = 12", "ny = 12", "nz = 8", "dx = 3000.0", "dy = 3000.0",
             "ztop = 12000.0", "[run]", "dt = 6.0", "run_seconds = 60.0",
             "sf_surface_physics = 2", "sf_sfclay_physics = 1", "bl_pbl_physics = 1",
             "sf_surface_mosaic = 1", "sf_urban_physics = 1",
             'mosaic_urban_canopy = "every_tile"']
    path = tmp_path / "town.toml"
    path.write_text("".join(line + chr(10) for line in lines))
    assert load_config(path).mosaic_urban_canopy == "every_tile"
    path.write_text("".join(line + chr(10) for line in lines
                            if line != "sf_surface_mosaic = 1"))
    with pytest.raises(ValueError, match="needs sf_surface_mosaic = 1"):
        load_config(path)
