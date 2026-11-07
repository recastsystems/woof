"""Native source preparation must carry actual physics, independent of presets."""
from dataclasses import asdict, replace
from datetime import datetime
from types import SimpleNamespace
import copy
import tomllib

import numpy as np
import pytest

from woof.experiment import VerticalConfig, build_experiment
from woof.experiment_document import render_experiment_document, publish_experiment_document
from woof.hrrr_configuration import resolve_root_experiment
from woof.hrrr_route_inputs import render_namelist_input, target_domain
from woof.hrrr_prepared_bundle import render_wps_namelist
from woof.ingest.hrrr_target import HrrrTargetDomain
from woof.ingest.microphysics_cold_start import cold_start_contract, source_absent_microphysics
from woof.physics_compat import WSM6_PROFILE_ID, MYNN_RUC_PROFILE_ID, single_domain_runtime_switches
from tools import hrrr_single_domain_benchmark as benchmark


def _native_tables():
    """Build a fresh native configuration for each route scenario."""
    vertical = VerticalConfig(eta_levels=tuple(float(x) for x in np.linspace(1, 0, 13)),
                              p_top=5000., hybrid_opt=2, etac=.2)
    target = replace(HrrrTargetDomain.legacy_500x500(), nx=50, ny=50, nz=12)
    raw, _ = benchmark._experiment_tables(vertical, run_seconds=3600, target=target,
                                          physics_profile=WSM6_PROFILE_ID)
    return raw, target


def _case(tmp_path, changes=None):
    raw, target = _native_tables()
    if changes:
        raw["shared"].update(changes)
    exp = build_experiment(copy.deepcopy(raw), source="configured native control")
    config = tmp_path / "experiment.toml"
    config.write_text(render_experiment_document(raw), encoding="utf-8")
    namelist = tmp_path / "namelist.input"
    namelist.write_text(render_namelist_input(exp), encoding="utf-8")
    wps = tmp_path / "namelist.wps"
    wps.write_text(render_wps_namelist(exp), encoding="utf-8")
    return exp, target, config, namelist, wps


@pytest.mark.parametrize("changes", [{}, {"sf_sfclay_physics": 1, "epssm": .17},
    {"bl_pbl_physics": 5, "sf_surface_physics": 3, "num_soil_layers": 6, "bldt": 0.},
    {"cu_physics": 1, "cudt_minutes": 7.5}])
def test_actual_configuration_reaches_published_root_without_preset_reconstruction(tmp_path, changes):
    exp, target, config, namelist, wps = _case(tmp_path, changes)
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert asdict(actual.root.run) == asdict(exp.root.run)
    publish_experiment_document(tmp_path / "published.toml", raw, actual)
    receipt = benchmark._configured_physics_receipt(actual.root.run)
    assert receipt["profile"] is None
    for key, value in changes.items():
        assert receipt["resolved"][key] == value


@pytest.mark.parametrize("supplied_wps", [False, True])
def test_namelist_only_uses_common_importer_and_keeps_selected_surface(tmp_path, supplied_wps):
    exp, target, config, namelist, wps = _case(tmp_path, {"sf_sfclay_physics": 1})
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        wps_namelist=wps if supplied_wps else None,
        acknowledgements=tuple(exp.acknowledgements))
    assert actual.root.run.sf_sfclay_physics == 1
    assert actual.root.run.mp_physics == exp.root.run.mp_physics
    assert actual.root.run.cu_physics == exp.root.run.cu_physics


def test_named_profile_remains_an_explicit_equality_assertion(tmp_path):
    exp, target, config, namelist, wps = _case(tmp_path, {"sf_sfclay_physics": 1})
    with pytest.raises(ValueError, match="differs from profile"):
        resolve_root_experiment(target=target, vertical=exp.vertical,
            namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
            experiment_config=config, physics_profile=WSM6_PROFILE_ID)


@pytest.mark.parametrize("root_pbl,nest_pbl", [(1, 900), (900, 1)],
                         ids=["ysu-root", "sase-root"])
def test_the_root_of_a_mixed_sase_tree_prepares_with_a_shared_sase_selector(
        tmp_path, root_pbl, nest_pbl):
    """The HRRR root preparation keeps d01 alone and rebuilds it.

    On a tree that mixes SASE with YSU the loader applies a [shared] SASE
    selector to the SASE domains only.  Cut down to a YSU root, the
    selector reached no kept domain, and the rebuild refused the root for
    "sase_moist_n2=False requires bl_pbl_physics=900", so a tree whose
    whole-tree load succeeds stopped at prepare.  The root now prepares
    with the values it runs in the tree; a SASE root keeps the selector.
    """
    raw, target = _native_tables()
    raw["shared"]["sase_moist_n2"] = False
    raw["domain"][0].update(bl_pbl_physics=root_pbl, km_opt=0 if root_pbl == 900 else 4)
    raw["domain"].append({
        "grid_id": 2, "parent_id": 1, "i_parent_start": 18, "j_parent_start": 18,
        "parent_grid_ratio": 3, "parent_time_step_ratio": 3, "nx": 30, "ny": 30,
        "history_interval_s": 300.0, "specified": False, "nested": True,
        "bl_pbl_physics": nest_pbl, "km_opt": 0 if nest_pbl == 900 else 4})
    tree = build_experiment(copy.deepcopy(raw), source="mixed SASE tree")
    by_pbl = {d.run.bl_pbl_physics: d.run for d in tree.domains}
    assert by_pbl[900].sase_moist_n2 is False and by_pbl[1].sase_moist_n2 is True
    config = tmp_path / "experiment.toml"
    config.write_text(render_experiment_document(raw), encoding="utf-8")
    actual, _ = resolve_root_experiment(target=target, vertical=tree.vertical,
        namelist_input=tmp_path / "namelist.input", start_time=tree.start_time,
        run_seconds=tree.run_seconds, experiment_config=config)
    assert actual.root.run.bl_pbl_physics == root_pbl
    assert actual.root.run.sase_moist_n2 is tree.root.run.sase_moist_n2


def _grell_tree_tables(*, root_cu, child_cu, shared=None, root=None,
                       child=None):
    """A route tree whose root and nest run the given cumulus schemes."""
    raw, target = _native_tables()
    raw["shared"].update(cu_physics=0, cudt_minutes=0.0, **(shared or {}))
    raw["domain"][0].update(cu_physics=root_cu, **(root or {}))
    raw["domain"].append({
        "grid_id": 2, "parent_id": 1, "i_parent_start": 18, "j_parent_start": 18,
        "parent_grid_ratio": 3, "parent_time_step_ratio": 3, "nx": 30, "ny": 30,
        "history_interval_s": 300.0, "specified": False, "nested": True,
        # The root's radiation cadence, which the namelists state per domain.
        "radt": raw["domain"][0]["radt"],
        "radt_minutes": raw["domain"][0]["radt_minutes"],
        "cu_physics": child_cu, **(child or {})})
    return raw, target


def test_the_route_namelists_carry_the_grell_freitas_closure(tmp_path):
    """Both namelists state clos_choice and ishallow, and the route runs them.

    Omitted, the stock-WRF arm ran the Registry default closure while the
    woof arm ran the configured one, and the hierarchy read 0 on the
    Grell-Freitas root: it stopped after the fetch and the root
    preparation.  The pair now carries the configured values, the raw
    runtime contract admits them, the hierarchy imports them on the
    Grell-Freitas root only, and the drift check takes the cumulus-off
    nest's 0.
    """
    from woof.hrrr_hierarchy_direct import _require_raw_stock_delta
    from woof.hrrr_route_inputs import write_hrrr_route_inputs
    from woof.namelist_import import import_namelists, parse_namelist
    import tomllib

    raw, _target = _grell_tree_tables(
        root_cu=3, child_cu=0, shared={"clos_choice": 1, "ishallow": 1})
    exp = build_experiment(copy.deepcopy(raw), source="grell tree")
    assert [(d.run.clos_choice, d.run.ishallow) for d in exp.domains] == [
        (1, 1), (0, 0)]
    config = tmp_path / "experiment.toml"
    config.write_text(render_experiment_document(raw), encoding="utf-8")
    wps, target, native, stock = write_hrrr_route_inputs(
        config, exp, wps_text=render_wps_namelist(exp),
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))
    for path in (native, stock):
        physics = parse_namelist(path)["physics"]
        assert physics["clos_choice"] == [1]
        assert physics["ishallow"] == [1]
    _require_raw_stock_delta(native, stock)
    text, _ = import_namelists(wps, native, name=exp.name,
                               acknowledgements=tuple(exp.acknowledgements))
    imported = build_experiment(tomllib.loads(text), source="imported tree")
    assert [(d.run.cu_physics, d.run.clos_choice, d.run.ishallow)
            for d in imported.domains] == [(3, 1, 1), (0, 0, 0)]


def test_grell_freitas_domains_with_different_closures_are_refused_on_the_route(
        tmp_path):
    """WRF reads clos_choice once for the whole run, so the pair cannot
    carry two; the refusal says which domains and the way out."""
    from woof.hrrr_route_inputs import HrrrRouteInputError

    raw, _target = _grell_tree_tables(
        root_cu=3, child_cu=3, root={"clos_choice": 1},
        child={"clos_choice": 5})
    exp = build_experiment(copy.deepcopy(raw), source="two closures")
    with pytest.raises(HrrrRouteInputError,
                       match="set different closures") as caught:
        render_namelist_input(exp)
    assert "d01 clos_choice = 1" in str(caught.value)
    assert "d02 clos_choice = 5" in str(caught.value)
    assert "[shared]" in str(caught.value)


def test_the_root_under_a_grell_freitas_nest_prepares_with_a_shared_closure(
        tmp_path):
    """The HRRR root preparation keeps d01 alone and rebuilds it.

    A [shared] clos_choice on a cumulus-off root under a Grell-Freitas
    nest reached only the nest; cut down to the root, the key would have
    refused the rebuild.  The root prepares with the values it runs in
    the tree, and a Grell-Freitas root keeps them.
    """
    for root_cu, child_cu in ((0, 3), (3, 0)):
        raw, target = _grell_tree_tables(
            root_cu=root_cu, child_cu=child_cu,
            shared={"clos_choice": 1, "ishallow": 1})
        tree = build_experiment(copy.deepcopy(raw), source="grell tree")
        config = tmp_path / f"experiment-{root_cu}.toml"
        config.write_text(render_experiment_document(raw), encoding="utf-8")
        actual, _ = resolve_root_experiment(target=target, vertical=tree.vertical,
            namelist_input=tmp_path / "namelist.input", start_time=tree.start_time,
            run_seconds=tree.run_seconds, experiment_config=config)
        assert actual.root.run.cu_physics == root_cu
        assert actual.root.run.clos_choice == tree.root.run.clos_choice
        assert actual.root.run.ishallow == tree.root.run.ishallow


def _storm_following_tables(mover_source):
    """A route tree with a moving child, as a storm-following layout writes it.

    ``mover_source`` picks what moves the child: the weather tracker
    (``follow``), a scheduled itinerary (``move``), or the tracker on a
    grandchild whose parent slides to keep it contained
    (``containment``).
    """
    raw, target = _native_tables()
    root = raw["domain"][0]
    raw["domain"].append({
        "grid_id": 2, "parent_id": 1, "i_parent_start": 14, "j_parent_start": 14,
        "parent_grid_ratio": 3, "parent_time_step_ratio": 3, "nx": 60, "ny": 60,
        "history_interval_s": 300.0, "specified": False, "nested": True,
        "radt": root["radt"], "radt_minutes": root["radt_minutes"]})
    follow = {"field": "uh", "threshold": 40.0, "fallback_threshold": 40.0,
              "search_margin_cells": 8, "min_shift_cells": 2,
              "max_shift_cells": 6, "cooldown_seconds": 600.0}
    relocation = {"enabled": True, "grid_id": 2, "cadence_seconds": 600.0,
                  "max_move_parent_cells": 6, "min_overlap_fraction": 0.25}
    if mover_source == "follow":
        relocation["follow"] = follow
    elif mover_source == "move":
        relocation["move"] = [{"at_seconds": 1200.0, "di_parent_cells": 2,
                               "dj_parent_cells": 1}]
    else:
        raw["domain"].append({
            "grid_id": 3, "parent_id": 2, "i_parent_start": 20, "j_parent_start": 20,
            "parent_grid_ratio": 3, "parent_time_step_ratio": 3, "nx": 45, "ny": 45,
            "history_interval_s": 300.0, "specified": False, "nested": True,
            "radt": root["radt"], "radt_minutes": root["radt_minutes"]})
        relocation.update(grid_id=3, follow=follow,
                          containment={"grid_id": 2, "deadband_cells": 4})
    raw["relocation"] = relocation
    return raw, target


@pytest.mark.parametrize("mover_source", ["follow", "move", "containment"])
def test_the_root_of_a_storm_following_tree_prepares(tmp_path, mover_source):
    """The HRRR root preparation keeps d01 alone and rebuilds it.

    The [relocation] table names the child that moves, never the root,
    so a root cut down from a storm-following tree kept a table naming a
    domain it no longer had, and the rebuild refused it: "grid_id = 2 in
    [relocation] ... is not a domain of this experiment (have [1])".
    Every storm-following layout stopped at the root preparation that
    way.  The root now prepares as it runs in the tree, without the
    mover's table, and the tree itself still moves its child.
    """
    raw, target = _storm_following_tables(mover_source)
    tree = build_experiment(copy.deepcopy(raw), source="storm-following tree")
    assert tree.relocation.enabled
    from woof.toml_document import emit_experiment_toml
    config = tmp_path / "experiment.toml"
    config.write_text(emit_experiment_toml(raw), encoding="utf-8")
    actual, root_tables = resolve_root_experiment(target=target, vertical=tree.vertical,
        namelist_input=tmp_path / "namelist.input", start_time=tree.start_time,
        run_seconds=tree.run_seconds, experiment_config=config)
    assert [d.grid_id for d in actual.domains] == [1]
    assert not actual.relocation.enabled
    assert "relocation" not in root_tables
    assert asdict(actual.root.run) == asdict(tree.root.run)
    publish_experiment_document(tmp_path / "published.toml", root_tables, actual)
    # The operator's configuration is untouched: the tree keeps its mover.
    import tomllib
    reread = build_experiment(tomllib.loads(config.read_text(encoding="utf-8")),
                              source="storm-following tree, reread")
    assert reread.relocation == tree.relocation


def test_a_cut_that_keeps_the_mover_keeps_its_relocation():
    """Only a cut that drops the mover drops its table."""
    from woof.experiment import drop_unreached_relocation

    raw, _target = _storm_following_tables("containment")
    kept = copy.deepcopy(raw)
    drop_unreached_relocation(kept, [1, 2, 3])
    assert kept["relocation"] == raw["relocation"]
    cut = copy.deepcopy(raw)
    drop_unreached_relocation(cut, [1, 2])
    assert "relocation" not in cut
    disabled = {"relocation": {"enabled": False}}
    drop_unreached_relocation(disabled, [1])
    assert disabled == {"relocation": {"enabled": False}}


def test_configuration_geometry_mismatch_still_refuses_before_source_consumption(tmp_path):
    exp, target, config, namelist, _ = _case(tmp_path)
    with pytest.raises(ValueError, match="configured d01 differs"):
        resolve_root_experiment(target=replace(target, nx=51), vertical=exp.vertical,
            namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
            experiment_config=config)


@pytest.mark.parametrize("mp", [1, 6, 8, 9, 10, 16, 18, 28, 50])
def test_source_absent_contract_depends_on_microphysics_not_surface_or_profile(mp):
    baseline = SimpleNamespace(mp_physics=mp, wdm6_ccn_conc=1.23e8,
                               sf_surface_physics=2, bl_pbl_physics=1)
    varied = SimpleNamespace(**(vars(baseline) | {"sf_surface_physics": 3, "bl_pbl_physics": 5}))
    assert cold_start_contract(baseline) == cold_start_contract(varied)
    if mp == 16:
        assert cold_start_contract(baseline)[1]["nn"][0] == float(np.float32(1.23e8))


def test_omitted_cli_window_and_history_use_actual_configuration(tmp_path):
    from tools.prepare_hrrr_wrf import _configured_defaults
    exp, _, config, _, _ = _case(tmp_path)
    args = SimpleNamespace(experiment_config=config, run_seconds=None,
                           history_interval_seconds=None)
    _configured_defaults(args)
    assert args.run_seconds == exp.run_seconds
    assert args.history_interval_seconds == exp.root.history_interval_s
    args.run_seconds, args.history_interval_seconds = 1800., 30.
    _configured_defaults(args)
    assert (args.run_seconds, args.history_interval_seconds) == (1800., 30.)


def test_the_route_door_and_the_preparation_door_agree_about_mp28():
    """R-044's capability, measured across the two doors it crosses.

    THE FAILURE THIS CLOSES, which is the one the route module's own
    docstring names: "a wizard that reports PASS, a fetch, a root
    preparation, and only then a refusal naming a switch the wizard had
    already chosen".  R-044 opened the route door -- mp=28 left
    ``route_physics_problems`` and ``SUPPORTED_MICROPHYSICS`` became the
    ported set -- and left a RAISE standing behind it in
    ``ingest/microphysics_cold_start.py``, reached from
    ``tools/prepare_hrrr_wrf.py`` after the PASS.  Two doors, one answer,
    measured here rather than asserted about.

    The dataset precondition is a different question and is not softened
    by this: it is measured for every source by
    woof.config.mp28_aerosol_lateral_forcing_precondition, raised at the
    run door and reported at plan review, and its own gates are in
    tests/test_authority_agreement.py and below.
    """
    from woof.config import RunConfig
    from woof.hrrr_route_inputs import (
        ROUTE_GATED_SWITCHES, SUPPORTED_MICROPHYSICS, route_physics_problems)

    cfg = RunConfig(nx=16, ny=16, nz=12, dx=3000., dy=3000., ztop=20000.,
                    dt=5., run_seconds=30., moist=True, mp_physics=28)
    assert 28 in SUPPORTED_MICROPHYSICS
    assert route_physics_problems(
        {switch: getattr(cfg, switch)
         for switch in ROUTE_GATED_SWITCHES}) == []

    fields, expected = cold_start_contract(cfg)
    assert fields == ("QNCLOUD", "QNRAIN", "QNICE")
    assert set(expected) == {"nc", "nr", "ni"}
    assert all(bits == 0 for _value, bits in expected.values()), expected
    # nwfa/nifa are absent on purpose: their initial condition belongs to
    # mp28_aerosol_source, not to the cold-start contract, and
    # woof/ingest/real.py refuses a nonzero value it did not write.
    assert not [name for name in expected if name.endswith("fa")]


@pytest.mark.parametrize("mp", [1, 6, 8, 9, 10, 16, 18, 28, 50])
def test_cold_start_contract_matches_actual_host_state_allocation(mp):
    from woof.config import RunConfig
    from woof.core.state import DomainState
    cfg = RunConfig(nx=12, ny=12, nz=12, dx=3000., dy=3000., ztop=20000.,
                    dt=5., run_seconds=30., moist=True, mp_physics=mp,
                    wdm6_ccn_conc=1.23e8)
    state = DomainState(cfg, array_module=np)
    _, expected = cold_start_contract(cfg)
    for name, (value, bits) in expected.items():
        field = getattr(state, name)
        assert field.dtype == np.float32
        assert np.all(field.view(np.uint32) == bits), name


def test_namelist_only_defaults_keep_producing_clock_and_cli_override(tmp_path):
    import json
    from tools.prepare_hrrr_wrf import _configured_defaults
    exp, target, _, namelist, wps = _case(tmp_path)
    domain = tmp_path / "target.json"
    domain.write_text(json.dumps(target.to_payload()))
    args = SimpleNamespace(experiment_config=None, namelist_input=namelist,
        domain_spec=domain, wps_namelist=wps, physics_profile=None,
        ack=exp.acknowledgements, run_seconds=None, history_interval_seconds=None)
    _configured_defaults(args)
    assert args.run_seconds == exp.run_seconds == 3600.
    assert args.history_interval_seconds == exp.root.history_interval_s
    args.run_seconds, args.history_interval_seconds = 1800., 30.
    _configured_defaults(args)
    assert (args.run_seconds, args.history_interval_seconds) == (1800., 30.)


def test_prepared_authority_excludes_execution_tiles_but_preserves_physics(tmp_path):
    import tomllib
    exp, target, config, namelist, wps = _case(tmp_path)
    original = config.read_text() + '\n[tiles]\nmode = "auto"\n[fetch]\nsource = "hrrr"\n'
    config.write_text(original)
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    path = publish_experiment_document(tmp_path / "prepared.toml", raw, actual)
    assert "tiles" not in tomllib.loads(path.read_text())
    assert asdict(actual.root.run) == asdict(exp.root.run)
    assert config.read_text() == original



@pytest.mark.parametrize("downscale", [True, False])
def test_native_companion_tables_survive_resolution_and_publication(tmp_path, downscale):
    import tomllib
    from woof.branch import emit_experiment_toml
    from woof.experiment import load_experiment

    exp, target, config, namelist, wps = _case(tmp_path)
    companions = {
        "case_data": {"forcing": "not-fetched.grib", "vtable": "Vtable",
            "wps_namelist": "namelist.wps", "geog_root": "geog",
            "sfcp_to_sfcp": True, "output_title": "declared settings",
            "co2_vmr": 0.000731, "source_orography": {"d01": "terrain.nc"},
            "source_orography_variable": "z"},
        "static": {"highres": {"enabled": False, "cache_root": "static-cache"}},
        "ingest": {"soil_texture_downscale": downscale},
    }
    original = config.read_text() + emit_experiment_toml(companions)
    config.write_text(original)
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert asdict(actual.root.run) == asdict(exp.root.run)
    expected_companions = {name: dict(value) for name, value in companions.items()}
    from woof.case_data import resolved_case_data_paths
    expected_companions["case_data"] = resolved_case_data_paths(
        companions["case_data"], base_dir=tmp_path, source=str(config))
    expected_companions["static"] = {"source": "hrrr-conus-v4", "highres": {
        **companions["static"]["highres"], "cache_root": str(tmp_path / "static-cache")}}
    assert {name: raw[name] for name in companions} == expected_companions
    published = publish_experiment_document(tmp_path / "prepared.toml", raw, actual)
    assert {name: tomllib.loads(published.read_text())[name]
            for name in companions} == expected_companions
    assert load_experiment(published) == actual
    assert config.read_text() == original


@pytest.mark.parametrize("extra,match", [
    ('[ingest]\nsoil_texture_downscale = "off"\n', 'must be true or false'),
    ('[static.highres]\nenabled = true\ncache_root = "c"\nunowned = 1\n',
     'does not have a key'),
    ('[case_data]\nco2_vmr = 0.0007\n', 'missing'),
])
def test_native_companion_owner_validation_still_refuses_invalid_data(tmp_path, extra, match):
    exp, target, config, namelist, wps = _case(tmp_path)
    config.write_text(config.read_text() + extra)
    with pytest.raises(ValueError, match=match):
        resolve_root_experiment(target=target, vertical=exp.vertical,
            namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
            experiment_config=config, wps_namelist=wps)


@pytest.mark.parametrize("changes", [
    {"mp_physics": 9},
    {"mp_physics": 16, "wdm6_ccn_conc": 1.23e8, "wdm6_hail_opt": 1},
    {"bl_pbl_physics": 5, "sf_surface_physics": 3, "num_soil_layers": 6, "bldt": 0.},
])
def test_full_native_emission_and_prepared_guard_keep_configured_physics(tmp_path, changes):
    from woof.hrrr_route_inputs import write_hrrr_route_inputs
    from woof.namelist_import import parse_namelist

    exp, target, config, namelist, wps = _case(tmp_path, changes)
    paths = write_hrrr_route_inputs(config, exp, wps_text=render_wps_namelist(exp),
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))
    if exp.root.run.mp_physics == 16:
        for path in (paths[2], paths[3]):
            physics = parse_namelist(path)["physics"]
            assert physics["hail_opt"] == [1]
            assert physics["ccn_conc"] == [1.23e8]
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert asdict(actual.root.run) == asdict(exp.root.run)
    publish_experiment_document(tmp_path / "published.toml", raw, actual)
    receipt = benchmark._configured_physics_receipt(actual.root.run,
        acknowledgements=tuple(actual.acknowledgements))
    assert receipt["profile"] is None
    benchmark._validate_resolved_hrrr_profile(actual, receipt)
    assert all(receipt["resolved"][key] == value for key, value in changes.items())
    receipt["resolved"]["mp_physics"] = 14
    with pytest.raises(RuntimeError, match="physics differs"):
        benchmark._validate_resolved_hrrr_profile(actual, receipt)


@pytest.mark.parametrize("changes,match", [
    ({"mp_physics": 16, "wdm6_ccn_conc": 1.e7}, "outside the WDM family"),
    ({"mp_physics": 16, "wdm6_hail_opt": 2}, "wdm6_hail_opt"),
    ({"sf_surface_physics": 3, "num_soil_layers": 5}, "soil"),
    # 15 and 51 are ported now; 95 (old Eta Ferrier) and 27 (WRF v4.7
    # UDM) are WRF selectors with no woof package.
    ({"mp_physics": 95}, "not ported"),
    ({"mp_physics": 27}, "not ported"),
    # THERE IS NO mp=28 ROW HERE, and the absence is the measurement.
    # This route used to subtract 28 on its own feet, naming aerosol
    # boundary species absent from the native analyzed stream; that
    # premise stopped being true when the WIF climatology ingest landed
    # and nwfa/nifa joined the coupled boundary fields on every route.
    # What is actually conditional is the DATASET -- a question about the
    # machine, not about these selections -- so it is asked at the run
    # door and measured in the test below rather than by this parser
    # battery, which is also the namelist importer's.
])
def test_native_emission_still_refuses_invalid_and_unported_selections(
        tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        _case(tmp_path, changes)


def test_the_native_route_admits_mp28_once_the_dataset_question_is_answered(
        tmp_path, monkeypatch):
    """The other half of the row above, and the capability it restored.

    The route's admitted set is DERIVED from the nest-transition
    resolver's ported selectors and no longer subtracts anything, so a
    scheme the engine implements is reachable here as soon as its own
    preconditions are met.  Taking the way out the refusal names -- the
    deliberate synthetic aerosol source -- is enough, which is what makes
    that sentence a refusal with a way out rather than a wall.

    AND WHICH DOOR ASKS.  The configuration parser admits mp=28 with no
    dataset, because a configuration is portable and whether a 225 MB
    file is installed is a property of the machine; the RUN door refuses
    it, in the sentence the registry publishes, before step 0.  Both are
    measured here so neither can quietly move to the other.
    """

    from woof.config import validate_run_preparation
    from woof.hrrr_route_inputs import SUPPORTED_MICROPHYSICS
    from woof.ingest import wif_climatology, wif_dataset

    assert 28 in SUPPORTED_MICROPHYSICS
    # The MACHINE must not decide this: every WIF search rung -- the two
    # environment overrides, the staged root and the working directory --
    # is pointed somewhere empty, which is the state a user without the
    # climatology is in.
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    empty = tmp_path / "no-staged-wif"
    empty.mkdir()
    monkeypatch.setenv(wif_dataset.WIF_DATASET_ROOT_ENV, str(empty))
    monkeypatch.chdir(empty)
    exp, _target, _config, _namelist, _wps = _case(
        tmp_path, {"mp_physics": 28, "mp28_aerosol_source": "synthetic"})
    assert exp.root.run.mp_physics == 28
    assert exp.root.run.mp28_aerosol_source == "synthetic"
    validate_run_preparation(exp.root.run)

    # The default source, same machine: the parser still admits it and the
    # run door names the dataset and both ways out.
    auto, _t, _c, _n, _w = _case(tmp_path, {"mp_physics": 28})
    assert auto.root.run.mp28_aerosol_source == "auto"
    assert auto.root.run.specified is True
    with pytest.raises(ValueError,
                       match="QNWFA_QNIFA_SIGMA_MONTHLY.dat") as refusal:
        validate_run_preparation(auto.root.run)
    assert "mp28_aerosol_source='synthetic'" in str(refusal.value)


@pytest.mark.parametrize("feedback,smooth", [(0, 0), (1, 0), (1, 2)])
def test_route_namelists_carry_the_configured_feedback(tmp_path, feedback, smooth):
    """The route's namelists say what the config says about feedback.

    The renderer used to refuse feedback=1 and, below that refusal,
    spelled feedback and smooth_option as literal zeros; the hierarchy
    stage rebuilds the experiment from these bytes, so lifting only the
    refusal would have prepared every two-way tree as one-way.
    """
    from woof.namelist_import import parse_namelist_text
    exp, *_ = _case(tmp_path)
    exp = replace(exp, feedback=feedback, smooth_option=smooth)
    for stock in (False, True):
        domains = parse_namelist_text(
            render_namelist_input(exp, stock=stock))["domains"]
        assert domains["feedback"] == [feedback]
        assert domains["smooth_option"] == [smooth]


#: Every adaptive-clock field the route carries, scope-1 keys first.
_ADAPTIVE_FIELDS = (
    "use_adaptive_time_step", "step_to_output_time", "adaptation_domain",
    "target_cfl", "target_hcfl", "max_step_increase_pct",
    "starting_time_step", "starting_time_step_den",
    "max_time_step", "max_time_step_den",
    "min_time_step", "min_time_step_den")


def test_a_two_way_adaptive_tree_goes_through_the_route(tmp_path):
    """A parent and nest on two-way feedback and an adaptive clock with per-domain clamps.

    The shape of a lean 1 km / 500 m layout: feedback 1, the adaptive
    clock on, the nest growing by 51 % a step against its parent's 5 %
    and clamped to its own bounds.  Four refusals stood in turn on the
    route: the renderer's one-way refusal before anything was fetched;
    behind it, namelists that carried no adaptive key, so the round trip
    refused the clock; then the importer, which read each max_domains
    clamp as one value for the tree and refused the nest's own; then the
    hierarchy stage's drift check, which took those preparation-inert
    clamps for a trajectory difference after the root was prepared.  The
    pair now carries all of it from the config, the importer rebuilds the
    same tree from the bytes, and the hierarchy gate admits it.
    """
    from woof.hrrr_forecast import hrrr_forcing_end_hour
    from woof.hrrr_hierarchy_direct import (_require_raw_stock_delta,
                                             _supported_hierarchy_slice)
    from woof.hrrr_route_inputs import write_hrrr_route_inputs
    from woof.ingest.hrrr_target import load_hrrr_target_domain
    from woof.namelist_import import import_namelists, parse_namelist
    import tomllib

    raw, _target = _grell_tree_tables(
        root_cu=0, child_cu=0,
        shared={"use_adaptive_time_step": True, "step_to_output_time": True},
        root={"max_step_increase_pct": 5, "starting_time_step": 5,
              "min_time_step": 2, "max_time_step": 8},
        child={"max_step_increase_pct": 51, "starting_time_step": 5,
               "starting_time_step_den": 3, "min_time_step": 1,
               "min_time_step_den": 2, "max_time_step": 3})
    raw["experiment"].update(feedback=1, smooth_option=0)
    exp = build_experiment(copy.deepcopy(raw), source="two-way adaptive tree")
    expected = [[getattr(d.run, name) for name in _ADAPTIVE_FIELDS]
                for d in exp.domains]
    assert expected[0] != expected[1]
    config = tmp_path / "experiment.toml"
    config.write_text(render_experiment_document(raw), encoding="utf-8")
    wps, target, native, stock = write_hrrr_route_inputs(
        config, exp, wps_text=render_wps_namelist(exp),
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))

    for path in (native, stock):
        domains = parse_namelist(path)["domains"]
        assert domains["feedback"] == [1]
        assert domains["use_adaptive_time_step"] == [True]
        assert domains["max_step_increase_pct"] == [5, 51]
        assert domains["starting_time_step_den"] == [0, 3]
    _require_raw_stock_delta(native, stock)

    text, _ = import_namelists(wps, native, name=exp.name,
                               acknowledgements=tuple(exp.acknowledgements))
    imported = build_experiment(tomllib.loads(text), source="imported tree")
    assert (imported.feedback, imported.smooth_option) == (1, 0)
    assert [[getattr(d.run, name) for name in _ADAPTIVE_FIELDS]
            for d in imported.domains] == expected
    _supported_hierarchy_slice(
        imported, load_hrrr_target_domain(target),
        forcing_hours=tuple(range(
            hrrr_forcing_end_hour(imported.run_seconds) + 1)))


_NOISE_BUBBLE = {"bubbles": [{
    "center_lat": 38.5, "center_lon": -99.5, "center_height_m": 1500.0,
    "radius_km": 10.0, "depth_m": 1500.0, "amplitude_k": 0.01}]}


def _perturbed_config(tmp_path, *, tree):
    """A route configuration carrying a 0.01 K [perturbation] block."""
    raw, target = _native_tables()
    raw["experiment"]["name"] = target.name
    if tree:
        raw["domain"].append({
            "grid_id": 2, "parent_id": 1, "i_parent_start": 18, "j_parent_start": 18,
            "parent_grid_ratio": 3, "parent_time_step_ratio": 3, "nx": 30, "ny": 30,
            "history_interval_s": 300.0, "specified": False, "nested": True})
    raw["perturbation"] = copy.deepcopy(_NOISE_BUBBLE)
    exp = build_experiment(copy.deepcopy(raw), source="perturbed native control")
    config = tmp_path / "experiment.toml"
    config.write_text(render_experiment_document(raw), encoding="utf-8")
    return exp, target, config


def test_a_native_root_of_a_perturbed_tree_prepares_and_records_the_deferral(
        tmp_path, capsys):
    """The native HRRR route reads the configuration only at its root.

    Its hierarchy stage builds the children from namelists, so the root
    preparation is where the tree's [perturbation] block is seen: it
    prepares, and the deferral the root seals (and the hierarchy relays)
    is the one receipt every source's tree preparation writes.
    """
    from woof.experiment import deferred_initial_perturbation
    from woof.hrrr_configuration import root_perturbation_deferral

    exp, target, config = _perturbed_config(tmp_path, tree=True)
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=tmp_path / "namelist.input", start_time=exp.start_time,
        run_seconds=exp.run_seconds, experiment_config=config)
    assert len(actual.domains) == 1
    assert capsys.readouterr().err == ""
    # The root publishes its d01 slice as the bundle's authority, the
    # block included: dropped, the root alone would run unperturbed
    # under the bubbles' name, and unwritable it stopped the preparation.
    from woof.experiment import build_experiment_from_config_tables

    published = publish_experiment_document(tmp_path / "published.toml", raw, actual)
    reloaded = build_experiment_from_config_tables(
        tomllib.loads(published.read_text(encoding="utf-8")),
        source=str(published), base_dir=tmp_path)
    assert reloaded.perturbation == exp.perturbation
    deferred = root_perturbation_deferral(config)
    assert deferred == deferred_initial_perturbation(
        exp, "any route", announce=False)
    assert deferred["config"] == exp.perturbation.receipt()
    assert "deferred to prepared forecast initialization" in (
        capsys.readouterr().err)
    assert root_perturbation_deferral(None) is None


def test_a_native_single_domain_perturbation_is_deferred_to_its_runner(tmp_path):
    """The prepared single-domain runner applies the bubbles to its restored
    state (fix/single-domain-perturbation), so the root preparer defers the
    block, naming that runner, instead of refusing it."""
    from woof.hrrr_configuration import root_perturbation_deferral

    exp, target, config = _perturbed_config(tmp_path, tree=False)
    actual, _raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=tmp_path / "namelist.input", start_time=exp.start_time,
        run_seconds=exp.run_seconds, experiment_config=config)
    assert actual.perturbation == exp.perturbation
    deferred = root_perturbation_deferral(config, announce=False)
    assert deferred["application_route"] == (
        "woof.prepared_single_domain_forecast")


@pytest.mark.parametrize("source", ["hrrr", "rap-native"])
def test_the_root_preparation_fills_the_source_generations_before_splitting_fetch(
        tmp_path, source):
    """A bare operational-fork configuration prepares with the fork pair.

    The root preparation splits ``[fetch]`` off before it loads, so the
    load-time fill (woof.physics_source_defaults.GENERATION_SELECTORS)
    never saw the source: a configuration naming an hrrr or rap source
    under mp_physics = 28 with no generation key written resolved, and
    recorded in its physics receipt, WRF v4.6.1's surface layer and
    Thompson with the blend snow fall, while ``woof sim`` integrated the
    fork pair from the same file.  The fill now runs before the split.
    """
    exp, target, config, namelist, wps = _case(tmp_path, {"mp_physics": 28})
    assert exp.root.run.thompson_version == "wrf_461"
    assert exp.root.run.thompson_fork_snow_fall == "blend"
    # (The fixture authors its surface layer through the hrrr recipe, so
    # only the two Thompson keys are bare here; the fill writes all three.)
    config.write_text(config.read_text(encoding="utf-8")
                      + f'\n[fetch]\nsource = "{source}"\n', encoding="utf-8")
    actual, _ = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert actual.root.run.mp_physics == 28
    assert actual.root.run.thompson_version == "wrf_39_noaa"
    assert actual.root.run.thompson_fork_snow_fall == "wrf_39_noaa"
    assert actual.root.run.mynn_sfclay_variant == "gsl_wrf39"
    receipt = benchmark._configured_physics_receipt(actual.root.run)
    assert receipt["resolved"]["thompson_version"] == "wrf_39_noaa"
    assert receipt["resolved"]["thompson_fork_snow_fall"] == "wrf_39_noaa"
    assert receipt["resolved"]["mynn_sfclay_variant"] == "gsl_wrf39"


def test_the_root_preparation_keeps_a_written_generation_and_the_generic_default(tmp_path):
    """A written ``thompson_version = "wrf_461"`` under a fork source is
    kept (and the fork snow fall is then not filled, since woof.config
    refuses it under wrf_461); a configuration with no [fetch] source
    resolves the generic defaults as before."""
    exp, target, config, namelist, wps = _case(
        tmp_path, {"mp_physics": 28, "thompson_version": "wrf_461"})
    config.write_text(config.read_text(encoding="utf-8")
                      + '\n[fetch]\nsource = "hrrr"\n', encoding="utf-8")
    actual, _ = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert actual.root.run.thompson_version == "wrf_461"
    assert actual.root.run.thompson_fork_snow_fall == "blend"
    assert actual.root.run.mynn_sfclay_variant == "gsl_wrf39"
    exp, target, config, namelist, wps = _case(tmp_path, {"mp_physics": 28})
    actual, _ = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert asdict(actual.root.run) == asdict(exp.root.run)
