"""TEMPO starts from HRRR on the HRRR route.

Named breakage: ``[fetch] source = "hrrr"`` with ``mp_physics = "tempo"``
was refused before any fetch -- "&physics mp_physics = 900: no ratified
woof mapping" -- because the route writes its own WRF namelists from the
TOML and reads them back through the importer, which had no row for
woof's TEMPO id.  Behind that, the importer refused aer_opt = 3 for any
scheme outside the Thompson family (TEMPO carries nwfa/nifa), and the
round trip compared the WIF selector pair that is dead state under TEMPO.

The fixture is the L1-tempo-arwen Iowa case (the 3 km HRRR recipe with
mp = tempo) with an HRRR start.  CPU-only: nothing is fetched.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "tempo_hrrr" / "tempo_hrrr_iowa.toml"


def test_the_importer_maps_gpuwms_tempo_id():
    from woof.namelist_import import _MP_MAP

    assert _MP_MAP[900] == (900, "none (woof TEMPO id; no WRF selector)",
                            "TEMPO")


def test_aer_opt_3_asks_one_predicate():
    from woof.config import carries_aerosol_numbers

    assert carries_aerosol_numbers(900)      # TEMPO: nwfa and nifa
    assert carries_aerosol_numbers(28)
    assert carries_aerosol_numbers(38)
    assert not carries_aerosol_numbers(8)    # classic Thompson: none
    assert not carries_aerosol_numbers(11)   # CAMMGMP: named, no nwfa/nifa
    assert not carries_aerosol_numbers("x")


def test_the_wif_pair_is_dead_state_outside_the_thompson_family():
    from woof.ingest.prepared_cache import effective_prepared_domain_config

    def run(mp):
        return {"run": {"mp_physics": mp, "aer_init_opt": 1,
                        "wif_input_opt": 1, "cu_physics": 1}}

    tempo = effective_prepared_domain_config(run(900))["run"]
    assert (tempo["aer_init_opt"], tempo["wif_input_opt"]) == (0, 0)
    thompson = effective_prepared_domain_config(run(28))["run"]
    assert (thompson["aer_init_opt"], thompson["wif_input_opt"]) == (1, 1)


def test_a_tempo_configuration_writes_and_round_trips_its_route_namelists(tmp_path):
    from woof.domain_wizard import experiment_from_text
    from woof.hrrr_route_inputs import run_route_inputs

    config = tmp_path / FIXTURE.name
    text = FIXTURE.read_text(encoding="utf-8")
    config.write_text(text, encoding="utf-8")
    raw = tomllib.loads(text)
    assert raw["shared"]["mp_physics"] == "tempo"
    assert raw["fetch"]["source"] == "hrrr"
    exp = experiment_from_text(text, source=str(config))
    paths = run_route_inputs(config, exp, raw=raw, into=tmp_path / "route")
    namelist = paths["namelist_input"].read_text(encoding="utf-8")
    assert "mp_physics                          = 900," in namelist


def test_hrrr_analysed_numbers_reach_tempos_state():
    """The start-state door: an analysis carrying QN* fields under TEMPO.

    Named breakage: SUPPORTED_MICROPHYSICS had no 900, so the analysed
    number targets (the hrrr and hrrr-native routes read HRRR's QNCLOUD,
    QNRAIN, QNICE, QNWFA, QNIFA) refused "mp_physics=900 is not a WRF
    microphysics package this door recognises".
    """
    from woof.domain_wizard import experiment_from_text
    from woof.ingest.analyzed_numbers import metgrid_number_targets
    from woof.ingest.wrfinput import active_moisture_inventory

    text = FIXTURE.read_text(encoding="utf-8")
    cfg = experiment_from_text(text, source=str(FIXTURE)).domains[0].run
    assert cfg.mp_physics == 900
    required, allowed = active_moisture_inventory(cfg)
    assert required == allowed == frozenset((
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNCLOUD",
        "QNRAIN", "QNICE", "QNGRAUPEL", "QNWFA", "QNIFA"))
    assert metgrid_number_targets(cfg) == {
        "QNC": "nc", "QNI": "ni", "QNR": "nr", "QNG": "ng",
        "QNWFA": "nwfa", "QNIFA": "nifa"}


def _hrrr_recipe_switches(**overrides):
    """The fixture with the HRRR recipe's three switches that were refused
    under TEMPO: gsd_diagnostics = 1, mp_zero_out_all = 1 and
    use_rap_aero_icbc = true."""
    from dataclasses import replace
    from woof.domain_wizard import experiment_from_text

    text = FIXTURE.read_text(encoding="utf-8")
    cfg = experiment_from_text(text, source=str(FIXTURE)).domains[0].run
    return replace(cfg, **{"gsd_diagnostics": 1, "mp_zero_out_all": 1,
                           "use_rap_aero_icbc": True, **overrides})


def test_the_hrrr_recipe_switches_validate_under_tempo():
    from woof.config import validate_run_config

    validate_run_config(_hrrr_recipe_switches())


def test_mp_zero_out_all_names_tempos_first_scalar():
    from woof.core.microphysics import WRF_FIRST_SCALAR_SPECIES

    # WRF 4.8.0 Registry.EM_COMMON: qni (:526) is TEMPO's first declared scalar.
    assert WRF_FIRST_SCALAR_SPECIES[900] == "ni"


def test_gsd_hail_window_is_tempos_own_psd():
    from woof.core.hail import scratch_shapes, tempo_feeds_gsd_hail

    cfg = _hrrr_recipe_switches()
    assert tempo_feeds_gsd_hail(cfg)
    shapes = scratch_shapes(cfg)
    assert {"hail_maxk1", "hail_max2d"} <= set(shapes)
    off = _hrrr_recipe_switches(gsd_diagnostics=0)
    assert not tempo_feeds_gsd_hail(off)


def test_analyzed_aerosol_rides_tempos_boundaries_only_when_asked():
    from woof.boundary_fields import (admissible_boundary_inventory,
                                       external_scalar_fields)

    on = _hrrr_recipe_switches()
    assert external_scalar_fields(on)[-2:] == ("nwfa", "nifa")
    assert admissible_boundary_inventory(on, ("qv", "nwfa", "nifa"))
    off = _hrrr_recipe_switches(use_rap_aero_icbc=False)
    assert "nwfa" not in external_scalar_fields(off)
    assert not admissible_boundary_inventory(off, ("qv", "nwfa", "nifa"))


def test_a_resume_keeps_the_checkpoints_tempo_generation(tmp_path, monkeypatch):
    """Named breakage: an old (v3.0.3) TEMPO checkpoint was refused under
    its own unchanged configuration once v3.2.5 became the default."""
    from woof.core import tempo
    from woof.domain_wizard import experiment_from_text
    import woof.io.restart as restart

    exp = experiment_from_text(FIXTURE.read_text(encoding="utf-8"),
                               source=str(FIXTURE))
    assert exp.root.run.tempo_version == "v3.2.5"
    old = {"config": {"mp_physics": 900}}            # a header from before the key
    monkeypatch.setattr(restart, "read_restart_header", lambda path: old)
    kept = tempo.adopt_checkpoint_generation(exp, tmp_path / "ckpt.npz")
    assert kept.root.run.tempo_version == "v3.0.3"
    assert kept.root.run.tempo_hailhyperaware is True
    new = {"config": {"mp_physics": 900, "tempo_version": "v3.2.5"}}
    monkeypatch.setattr(restart, "read_restart_header", lambda path: new)
    assert tempo.adopt_checkpoint_generation(
        exp, tmp_path / "c.npz").root.run.tempo_version == "v3.2.5"
    assert tempo.adopt_checkpoint_generation(exp, None) is exp
