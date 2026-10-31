"""LSMRUC's cold start of the ground vapour, by lineage.

Public WRF (v4.6.1 ``phys/module_sf_ruclsm.F:505-514``) starts an invalid
QVG from saturation at the skin times moisture availability and an invalid
QCG from the lowest-level condensate.  The operational RAP/HRRR branch
(``:479-483`` there) starts an invalid QVG from the lowest-level vapour and
sets QCG to zero, with no separate QCG check.  ``ruc_qvg_cold_start``
names the form; the default is ``wrf`` everywhere.
"""

from dataclasses import asdict, replace

import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config
from woof.core.ruc_tier import (RUC_QVG_COLD_START_DEFAULT,
                                 RUC_QVG_COLD_START_FORMS,
                                 ruc_qvg_cold_start_form)

f32 = np.float32


def test_the_default_is_the_public_form():
    assert RUC_QVG_COLD_START_FORMS == ("air", "wrf")
    assert RUC_QVG_COLD_START_DEFAULT == "wrf"
    assert RunConfig.__dataclass_fields__["ruc_qvg_cold_start"].default == "wrf"
    assert ruc_qvg_cold_start_form("air") == 1
    assert ruc_qvg_cold_start_form("wrf") == 0


@pytest.mark.parametrize("bad", ["typo", "", 0, None, "AIR", "wrf_461"])
def test_an_unknown_form_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="ruc_qvg_cold_start"):
        ruc_qvg_cold_start_form(bad)


def _first_call(form, *, qvg, qcg):
    """One cold LSMRUC call on one oracle column with QVG and QCG preset."""
    from woof.core.ruc import ruc_land_surface_step
    from ruc_mosaic_fixture import driver_calls

    _label, values, keywords, _expected = next(driver_calls())
    values = {name: np.ascontiguousarray(value[..., :1])
              for name, value in values.items()}
    for name in ("ivgtyp", "isltyp"):
        keywords[name] = np.ascontiguousarray(keywords[name][:1])
    for name in ("landusef", "soilctop"):
        keywords[name] = np.ascontiguousarray(keywords[name][:, :1])
    keywords["ktau"] = 1
    keywords["qvg_cold_start"] = form
    values["qvg"][...] = f32(qvg)
    values["qcg"][...] = f32(qcg)
    return values, ruc_land_surface_step(dict(values), **keywords)


def test_the_two_forms_start_different_ground_vapour():
    values, air = _first_call("air", qvg=0.0, qcg=0.0)
    _, wrf = _first_call("wrf", qvg=0.0, qcg=0.0)
    # Both start, then SFCTMP advances QVG one step, so compare the start
    # through what it changes: the two forms give different surface states.
    assert not np.array_equal(np.asarray(air.qvg), np.asarray(wrf.qvg))
    assert float(values["qv3d"][0]) > 0.0


def test_a_valid_ground_vapour_is_kept_by_both_forms():
    _, air = _first_call("air", qvg=0.01, qcg=0.0)
    _, wrf = _first_call("wrf", qvg=0.01, qcg=0.0)
    for name in ("qvg", "qsfc", "soilt", "hfx", "qfx"):
        np.testing.assert_array_equal(np.asarray(getattr(air, name)),
                                      np.asarray(getattr(wrf, name)),
                                      err_msg=name)


def test_the_transcription_default_is_the_oracle_form():
    import inspect

    from woof.core.ruc import ruc_land_surface_step
    default = inspect.signature(ruc_land_surface_step).parameters[
        "qvg_cold_start"].default
    assert default == "wrf"


def _cfg(**kwargs):
    base = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                dt=6.0, run_seconds=60.0)
    base.update(kwargs)
    return RunConfig(**base)


def test_the_configuration_refuses_an_unknown_form():
    validate_run_config(_cfg(ruc_qvg_cold_start="air"))
    with pytest.raises(ValueError, match="ruc_qvg_cold_start"):
        validate_run_config(_cfg(ruc_qvg_cold_start="saturated"))


def test_restart_flip_refuses_and_the_old_header_reads_as_the_default():
    from woof.io.restart import _require_config_match, configuration_echo
    cfg = _cfg()
    stored = asdict(cfg)
    stored.pop("ruc_qvg_cold_start")
    _require_config_match(stored, cfg, "checkpoint")
    assert "ruc_qvg_cold_start" not in configuration_echo(cfg)
    legacy = replace(cfg, ruc_qvg_cold_start="air")
    assert configuration_echo(legacy)["ruc_qvg_cold_start"] == "air"
    with pytest.raises(ValueError, match="ruc_qvg_cold_start"):
        _require_config_match(stored, legacy, "checkpoint")


def test_the_form_is_preparation_inert():
    from woof.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS
    assert "run.ruc_qvg_cold_start" in PREPARATION_INERT_RUN_FIELDS


# ------------------------------------------------- the authoring doors
#
# No woof start carries QVG, so every RUC forecast cold-starts it.  The
# public-WRF form dropped the land skin 3.5-4 K on the first RUC call over
# the Los Angeles basin's dry soil (WOOF LA 750 m, f01 2 m T -2.00 F against
# ASOS; -0.09 F with "air").  The HRRR recipes wrote "air"; a RUC suite
# authored from any other source, or put on a file by the physics catalog
# (sensitivity '-ruc-' members, the marine-fog preset), did not.

RUC_SUITE = "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1"


def _authored(profile, source="gfs"):
    import tomllib
    from datetime import datetime

    from woof import domain_wizard as wizard

    text = wizard.render_config(
        name="ruc cold start", start_time=datetime(2026, 10, 28, 12), hours=2,
        projection=wizard._projection_entries(34.0, -118.3, "auto"),
        dims=[(64, 64)], ratios=(), root_dx_m=3000.0,
        fetch_hints=wizard._candidate_fetch_hints(source), case_data=None,
        profile=profile)
    return text, tomllib.loads(text)["shared"]


@pytest.mark.parametrize("source", ["gfs", "era5", "hrrr"])
def test_gpuwm_domain_writes_the_air_cold_start_for_a_ruc_suite(source):
    from woof.domain_wizard import experiment_from_text

    text, shared = _authored(RUC_SUITE, source)
    assert shared["sf_surface_physics"] == 3
    assert shared["ruc_qvg_cold_start"] == "air"
    run = experiment_from_text(text, source="ruc.toml").domains[0].run
    assert run.ruc_qvg_cold_start == "air"


def test_gpuwm_domain_writes_no_ruc_key_for_another_land_model():
    from woof.domain_wizard import DEFAULT_PHYSICS_PROFILE

    _, shared = _authored(DEFAULT_PHYSICS_PROFILE)
    assert shared["sf_surface_physics"] != 3
    assert "ruc_qvg_cold_start" not in shared


def _mixed(text, request, tmp_path):
    import tomllib

    from woof import physics_catalog as pc
    from woof.experiment import load_experiment

    out = pc.apply_to_experiment(text, request)
    path = tmp_path / "mix.toml"
    path.write_text(out, encoding="utf-8")
    return tomllib.loads(out), load_experiment(path).domains[0].run


def test_a_mix_that_moves_a_file_onto_ruc_writes_the_air_cold_start(tmp_path):
    from woof.domain_wizard import DEFAULT_PHYSICS_PROFILE

    text, shared = _authored(DEFAULT_PHYSICS_PROFILE)
    assert shared["sf_surface_physics"] != 3
    for request in ({"suite": RUC_SUITE, "source": "gfs"},
                    {"preset": "coastal-fog-stratus", "source": "gfs"},
                    {"choices": {"land_surface": "ruc-lsm"}, "source": "gfs"}):
        document, run = _mixed(text, request, tmp_path)
        assert run.sf_surface_physics == 3, request
        assert document["shared"]["ruc_qvg_cold_start"] == "air", request
        assert run.ruc_qvg_cold_start == "air", request
        assert all("ruc_qvg_cold_start" not in table for table in document["domain"])


def test_a_mix_keeps_the_cold_start_of_a_file_already_on_ruc(tmp_path):
    # An imported WRF namelist (the parity path) or any RUC file that
    # states no form runs public WRF's; a mix that does not move it off RUC
    # does not move its cold start either.
    text, _ = _authored(RUC_SUITE)
    text = text.replace('ruc_qvg_cold_start = "air"\n', "")
    document, run = _mixed(text, {"suite": RUC_SUITE, "choices": {"microphysics": "wsm6-mp6"},
                                  "source": "gfs"}, tmp_path)
    assert run.sf_surface_physics == 3 and run.mp_physics == 6
    assert "ruc_qvg_cold_start" not in document["shared"]
    assert run.ruc_qvg_cold_start == "wrf"


def test_a_stated_cold_start_is_kept_through_a_mix(tmp_path):
    from woof.domain_wizard import DEFAULT_PHYSICS_PROFILE

    text, _ = _authored(DEFAULT_PHYSICS_PROFILE)
    text = text.replace("[shared]\n", '[shared]\nruc_qvg_cold_start = "wrf"\n', 1)
    _, run = _mixed(text, {"suite": RUC_SUITE, "source": "gfs"}, tmp_path)
    assert run.sf_surface_physics == 3 and run.ruc_qvg_cold_start == "wrf"
    _, run = _mixed(_authored(DEFAULT_PHYSICS_PROFILE)[0],
                    {"suite": RUC_SUITE, "source": "gfs",
                     "settings": {"ruc_qvg_cold_start": "wrf"}}, tmp_path)
    assert run.ruc_qvg_cold_start == "wrf"


def test_the_run_document_round_trip_keeps_each_form():
    # The run document drops the default "wrf" (woof.experiment); with the
    # form written only at authoring, an absent key always loads as "wrf".
    from woof.domain_wizard import experiment_from_text
    from woof.experiment import experiment_config_document

    text, _ = _authored(RUC_SUITE)
    for written, expected in ((text, "air"),
                              (text.replace('ruc_qvg_cold_start = "air"\n', ""), None)):
        run_document = experiment_config_document(experiment_from_text(written, source="ruc.toml"))["domains"][0]["run"]
        assert run_document.get("ruc_qvg_cold_start") == expected


def test_a_step_down_onto_ruc_land_starts_from_the_air():
    # The default child suite under 4 km is the HRRR-shape RUC suite; a
    # child of a Noah parent took the parent's inert "wrf".
    from woof.downscale import child_physics_config

    noah_parent = {"sf_surface_physics": 2, "num_soil_layers": 4, "sf_sfclay_physics": 1, "epssm": 0.1}
    merged, profile, _ = child_physics_config(noah_parent, child_dx_m=1000.0, requested=None)
    assert profile and merged["sf_surface_physics"] == 3
    assert merged["ruc_qvg_cold_start"] == "air"
    ruc_parent = {**merged}
    ruc_parent.pop("ruc_qvg_cold_start")
    merged, _, _ = child_physics_config(ruc_parent, child_dx_m=1000.0, requested=None)
    assert "ruc_qvg_cold_start" not in merged
    merged, _, _ = child_physics_config(noah_parent, child_dx_m=1000.0, requested="parent")
    assert "ruc_qvg_cold_start" not in merged
