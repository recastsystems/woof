"""Thompson with MYNN surface layer and PBL over RUC, as named suites.

The MYNN + RUC pair shipped only with WSM6, so the operational class
that puts Thompson microphysics on that surface and boundary layer had
no name, and no menu, ``--physics-profile`` list or preset could offer
it.  Two rows close it, each its WSM6 row with the microphysics moved.
This file pins what they are for:

* each composes: the registry resolves it, its runtime product differs
  from its WSM6 row in the microphysics alone, and the run door accepts it;
* each is declared on exactly the routes and sources its WSM6 row is,
  immediately after it, because the RUC soil ingest is what limits both;
* the catalog, its preset and the wizard menu list it, and the engine's
  fit check sizes a plan that names it;
* the radiation-bearing row loads over a night window with no
  declaration, while the Dudhia row is refused on the SAME window.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from woof.config import RunConfig, validate_run_config
from woof.domain_wizard import experiment_from_text, render_config
from woof.experiment import build_experiment
from woof.physics_compat import (
    ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    first_local_night_time,
    identify_single_domain_profile,
    single_domain_runtime_switches,
)
from woof.physics_registry import physics_registry

#: (Thompson row, the WSM6 row it moves the microphysics of).
PAIRS = (
    (THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, MYNN_RUC_RTE_RRTMGP_PROFILE_ID),
    (THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID, MYNN_RUC_PROFILE_ID),
)
NEW = tuple(new for new, _base in PAIRS)

#: The same reference geometry and window tests/test_mynn_radiation_profiles.py
#: measures the MYNN family on: local night falls inside the window.
_PROJECTION = {
    "map_proj": "lambert", "ref_lat": 33.8, "ref_lon": -87.29,
    "truelat1": 23.8, "truelat2": 43.8, "stand_lon": -87.29,
}
_NIGHT_START = datetime(2011, 4, 26, 12)
_NIGHT_HOURS = 48


#: A window that stays in daylight at the same point: 15Z to 18Z is
#: 10:00 to 13:00 local, where a Dudhia suite loads undeclared.
_DAY_START = datetime(2011, 4, 26, 15)
_DAY_HOURS = 3


def _emitted(profile, *, start=_NIGHT_START, hours=_NIGHT_HOURS):
    return render_config(
        name="thompsonmynnruc", start_time=start, hours=hours,
        projection=dict(_PROJECTION), dims=[(120, 100)], ratios=(),
        fetch_hints={"source": "era5"}, case_data=None, profile=profile)


def _raw(text):
    import tomllib

    raw = tomllib.loads(text)
    raw.pop("fetch", None)
    raw.pop("case_data", None)
    return raw


# ---------------------------------------------------------------------------
# The suite composes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile, base", PAIRS)
def test_the_suite_is_its_wsm6_row_with_thompson_in_it(profile, base):
    registry = physics_registry()
    template = registry["templates"][profile]
    base_template = registry["templates"][base]
    assert template["components"] == {
        **base_template["components"], "microphysics": "thompson-mp8"}
    assert template["components"]["pbl"] == "mynn"
    assert template["components"]["surface_layer"] == "mynn"
    assert template["components"]["land_surface"] == "ruc-lsm"
    # The maturity the siblings carry, derived rather than asserted.
    assert template["maturity"] == base_template["maturity"] \
        == "implemented-unverified"
    assert template["parameters"] == base_template["parameters"]
    assert template["warnings"]


@pytest.mark.parametrize("profile, base", PAIRS)
def test_no_warning_names_a_wsm6_row_but_the_one_pairing_sentence(
        profile, base):
    """A warning copied from the WSM6 base named that base's own partner as
    this row's only difference; the only sentence that may name a WSM6 row
    is the one saying which row this suite differs from."""

    warnings = physics_registry()["templates"][profile]["warnings"]
    naming = [warning for warning in warnings if "wsm6-" in warning]
    assert naming == [
        warning for warning in warnings
        if warning.startswith("This template differs from " + base)]


@pytest.mark.parametrize("profile, base", PAIRS)
def test_the_runtime_product_moves_the_microphysics_and_nothing_else(
        profile, base):
    """A paired run against the WSM6 row isolates the microphysics.

    ``moist_cq`` moves with it because it is the Thompson option's own
    required setting, the value the Thompson validation suite carries too.
    """

    new = single_domain_runtime_switches(profile)
    old = single_domain_runtime_switches(base)
    moved = {name for name in set(new) | set(old)
             if new.get(name) != old.get(name)}
    assert moved <= {"mp_physics", "moist_cq"}
    assert (old["mp_physics"], new["mp_physics"]) == (6, 8)
    assert new["moist_cq"] == single_domain_runtime_switches(
        THOMPSON_PROFILE_ID)["moist_cq"]
    assert (int(new["bl_pbl_physics"]), int(new["sf_sfclay_physics"]),
            int(new["sf_surface_physics"]), int(new["num_soil_layers"])) == (
                5, 5, 3, 9)


@pytest.mark.parametrize("profile", NEW)
def test_the_run_door_accepts_the_suite(profile):
    switches = single_domain_runtime_switches(profile)
    cfg = RunConfig(nx=41, ny=41, nz=40, dx=1000.0, dy=1000.0,
                    ztop=20000.0, dt=5.0, run_seconds=60.0,
                    time_step_sound=4, **switches)
    validate_run_config(cfg)
    assert identify_single_domain_profile(cfg) == profile


# ---------------------------------------------------------------------------
# Declared where the WSM6 row is, and only there.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile, base", PAIRS)
def test_the_suite_follows_its_wsm6_row_on_every_route_and_source(
        profile, base):
    routes = physics_registry()["runner_routes"]
    declared_somewhere = False
    for route_id, route in routes.items():
        for group in ("source_template_ids", "expert_template_ids"):
            for source_id, declared in (route.get(group) or {}).items():
                assert (profile in declared) == (base in declared), (
                    route_id, group, source_id)
                if profile in declared:
                    declared_somewhere = True
                    assert declared.index(profile) == \
                        declared.index(base) + 1, (route_id, source_id)
    assert declared_somewhere
    assert profile in SINGLE_DOMAIN_PHYSICS_PROFILES


# ---------------------------------------------------------------------------
# The catalog, its preset and the wizard offer it.
# ---------------------------------------------------------------------------

def test_the_wizard_menu_offers_both_and_ranks_the_night_valid_one_first():
    from woof.domain_wizard import WIZARD_PHYSICS_PROFILES

    menu = list(WIZARD_PHYSICS_PROFILES)
    assert set(NEW) <= set(menu)
    # The radiation-bearing row sits in the nocturnally valid block: ahead
    # of every Dudhia suite, beside its WSM6 sibling.
    assert menu.index(THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID) == \
        menu.index(MYNN_RUC_RTE_RRTMGP_PROFILE_ID) + 1
    assert menu.index(THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID) == \
        menu.index(MYNN_RUC_PROFILE_ID) + 1
    assert menu.index(THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID) < \
        menu.index(THOMPSON_PROFILE_ID)


@pytest.mark.parametrize("source", ("hrrr", "era5"))
def test_the_catalog_lists_the_suite_and_the_preset_runs(source):
    from woof import physics_catalog as pc

    document = pc.catalog(source=source)
    offered = {row["id"] for row in document["suites"]
               if row["on_create_page"]}
    assert set(NEW) <= offered
    preset = pc.preset("coastal-fog-stratus")
    assert preset["suite"] == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    verdict = pc.check({"preset": "coastal-fog-stratus", "source": source})
    assert verdict["valid"], verdict.get("refusal")
    assert verdict["named_suite"] == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID


@pytest.mark.parametrize("profile", NEW)
def test_the_fit_check_sizes_a_plan_that_names_the_suite(tmp_path, profile):
    """The engine's fit check is ``run-plan --resolve`` on the plan intent.

    The same resolve the Create page's Fit runs: the wizard emits the
    config for the named suite, the loader accepts it, and the resolved
    domain carries the suite's switches.  The window is 12Z to 18Z at a
    central-US point, all daylight, so the Dudhia row needs no declaration.
    """

    import json

    from woof.runplan import PLAN_SCHEMA, load_plan, resolve_plan

    intent = {"point": "35.2,-97.4", "source": "hrrr", "root_dx_km": 3,
              "cycle": "2024-05-03T12", "hours": 6, "vram_gib": 24,
              "physics_profile": profile}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "thompson-mynn-ruc-plan",
        "route": "prepared", "config": {"intent": intent},
        "output_root": str(tmp_path / "run")}), encoding="utf-8")
    _resolution, exp, _data = resolve_plan(load_plan(path),
                                           require_inputs=False)
    run = exp.domains[0].run
    assert (run.mp_physics, run.bl_pbl_physics, run.sf_sfclay_physics,
            run.sf_surface_physics, run.num_soil_layers) == (8, 5, 5, 3, 9)


# ---------------------------------------------------------------------------
# Night: the radiation-bearing row loads, the Dudhia row is refused.
# ---------------------------------------------------------------------------

def test_the_window_contains_local_night():
    night = first_local_night_time(
        _NIGHT_START, _NIGHT_HOURS * 3600.0,
        ref_lat=_PROJECTION["ref_lat"], ref_lon=_PROJECTION["ref_lon"])
    assert night is not None


def test_the_radiation_bearing_row_loads_over_night_undeclared():
    raw = _raw(_emitted(THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID))
    raw["experiment"].pop("acknowledgements", None)
    experiment = build_experiment(raw, source="<thompson-mynn-ruc-night>")
    assert experiment.acknowledgements == ()
    run = experiment.root.run
    assert run.mp_physics == 8
    assert (run.ra_lw_physics, run.ra_sw_physics) == (4, 4)


def test_the_dudhia_row_is_refused_over_the_same_night_undeclared():
    text = _emitted(THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID)
    raw = _raw(text)
    raw["experiment"].pop("acknowledgements", None)
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<thompson-mynn-ruc-dudhia-night>")
    message = str(caught.value)
    assert "local night" in message
    assert "ra_lw_physics 0" in message
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in message


@pytest.mark.parametrize("profile", NEW)
def test_the_emitted_config_loads_through_the_shared_front_door(profile):
    """Over daylight, where both rows are valid; night is pinned above."""

    night = first_local_night_time(
        _DAY_START, _DAY_HOURS * 3600.0,
        ref_lat=_PROJECTION["ref_lat"], ref_lon=_PROJECTION["ref_lon"])
    assert night is None
    experiment = experiment_from_text(
        _emitted(profile, start=_DAY_START, hours=_DAY_HOURS),
        source="<thompson-mynn-ruc>")
    run = experiment.root.run
    expected = single_domain_runtime_switches(profile)
    assert {name: getattr(run, name) for name in expected} == expected
    assert identify_single_domain_profile(run) == profile


# ---------------------------------------------------------------------------
# The default below 1 km
# ---------------------------------------------------------------------------


def _door(tmp_path, *extra, source="hrrr", root_dx="2.25", chain="3"):
    """``woof domain`` with no --physics-profile, the custom ladder form."""

    import argparse

    from woof import domain_wizard as wizard

    parser = argparse.ArgumentParser()
    wizard.register_cli(parser.add_subparsers())
    out = tmp_path / f"{source}-{root_dx}-{chain}.toml"
    argv = ["domain", "--source", source, "--cycle", "2026-09-20T18",
            "--hours", "6", "--root-dx", root_dx, "--point", "37.62,-122.2",
            "--point-extent-km", "200", "--vram-gib", "16", "--tiles", "off",
            "--out", str(out), *extra]
    if chain:
        argv[argv.index("--point"):argv.index("--point")] = ["--chain", chain]
    args = parser.parse_args(argv)
    args.explain = False
    assert wizard.domain_main(args) == 0
    text = out.read_text()
    return text, experiment_from_text(text, source=str(out))


@pytest.mark.parametrize("source,root_dx,chain", [
    ("hrrr-prs", "2.25", "3"), ("era5", "2.25", "3"), ("gfs", "2.25", "3"),
    # A single native HRRR domain, and a native HRRR tree whose hierarchy
    # stage pins the soil column its land surface runs.
    ("hrrr", "0.75", ""), ("hrrr", "2.25", "3"),
])
def test_a_custom_sub_km_domain_with_no_profile_takes_the_fog_suite_and_the_adaptive_clock(
        tmp_path, source, root_dx, chain):
    """The site's custom route: 2.25 km with a 750 m nest, no suite named.

    On 2.8.0 the default keyed on the source alone, so this domain got
    YSU and Noah (Morrison on hrrr-prs, Thompson + YSU + MM5 + Noah on
    hrrr), the pair the fog screen showed losing stratus.  The spacing
    row binds now wherever the source's route admits the suite, and the
    auto clock is adaptive at these spacings (500 m to 12 km).
    """

    text, exp = _door(tmp_path, source=source, root_dx=root_dx, chain=chain)
    assert f"# PHYSICS: {THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID}" in text
    switches = single_domain_runtime_switches(
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    for domain in exp.domains:
        run = domain.run
        assert (run.mp_physics, run.bl_pbl_physics, run.sf_sfclay_physics,
                run.sf_surface_physics, run.num_soil_layers) == (
            switches["mp_physics"], switches["bl_pbl_physics"],
            switches["sf_sfclay_physics"], switches["sf_surface_physics"],
            switches["num_soil_layers"]) == (8, 5, 5, 3, 9)
        assert run.use_adaptive_time_step is True
    assert "use_adaptive_time_step = true" in text


@pytest.mark.parametrize("source", ["hrrr-prs", "era5", "gfs"])
def test_the_sub_km_default_tree_never_damps_the_nest_harder_than_its_parent(
        tmp_path, source):
    """2.25 km with a 750 m nest, no suite named: sixth-order damping.

    The fog suite pins diff_6th_factor = 0.08 on the root, and the nest
    used to take the certified ladder's second rung, 0.10, so the default
    sub-km tree carried a child damped harder than its parent.  The nest
    now takes the smaller of its parent's value and the ladder's.
    """

    text, exp = _door(tmp_path, source=source)
    assert f"# PHYSICS: {THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID}" in text
    root = single_domain_runtime_switches(
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)["diff_6th_factor"]
    factors = [float(domain.run.diff_6th_factor) for domain in exp.domains]
    assert factors == [root, root] == [0.08, 0.08]


def test_a_coarser_grid_keeps_the_sources_own_default(tmp_path):
    """The control: the same door at 3 km with no nest is unchanged."""

    from woof.physics_menu import default_profile_for

    text, exp = _door(tmp_path, source="gfs", root_dx="3", chain="")
    assert f"# PHYSICS: {default_profile_for('gfs')}" in text
    assert default_profile_for("gfs") != THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    assert exp.root.run.bl_pbl_physics == 1


def test_a_nested_hrrr_domain_takes_the_sub_km_default_and_the_suite_named(tmp_path):
    """The nested HRRR route's hierarchy stage pins the soil its land surface runs.

    Its certified raw runtime contract held &physics/num_soil_layers = 4
    whatever the land surface, so a --source hrrr tree kept the route's
    own YSU and Noah default below 1 km and a RUC suite named there was
    refused.  The pin follows the land surface now (nine layers for RUC),
    so the tree takes the sub-km default like every other source, and
    naming the suite is admitted too.
    """

    from woof.physics_menu import profile_route_blocker

    text, exp = _door(tmp_path, source="hrrr")
    assert f"# PHYSICS: {THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID}" in text
    assert [domain.run.num_soil_layers for domain in exp.domains] == [9, 9]
    for domains in (1, 2, 4):
        assert profile_route_blocker(
            THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, "hrrr", domains=domains) is None
    named, _ = _door(tmp_path / "named", "--physics-profile",
                     THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, source="hrrr")
    assert f"# PHYSICS: {THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID}" in named


# The deep-soil donor replaces the former shallow-soil refusal.
def test_a_source_with_the_deep_soil_donor_gets_the_spacing_default_below_1_km():
    """The authenticated deep-soil donor supplies GEM soil below its shallow layer."""

    from woof.physics_menu import (default_basis, default_profile_for,
                                    spacing_default_rows)

    row = spacing_default_rows("gem-gdps")[0]
    assert row["admitted"] is True
    assert row["why_not"] is None
    assert default_profile_for("gem-gdps", 750.0) == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    for source in ("hrrr", "hrrr-prs", "era5", "gfs"):
        # One domain; a nested hrrr tree is the test above.
        assert default_profile_for(source, 750.0) == \
            THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
        assert default_profile_for(source, 999.0) == \
            THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
        # The bound is exclusive: a 1 km grid is not sub-km.
        assert default_profile_for(source, 1000.0) == default_profile_for(source)
        assert "fog" in default_basis(source, 750.0)


def test_under_500_m_the_suite_binds_and_the_clock_stays_fixed_by_name(tmp_path):
    """Below the terrain clock's measured spacings auto keeps one step.

    The terrain clock's stability map was measured from 500 m to 12 km,
    and it is what caps the adaptive step over steep ground at launch, so
    a 250 m nest gets the fog suite and a fixed step, and the door says
    which spacing kept it fixed.
    """

    text, exp = _door(tmp_path, "--clock", "auto", source="gfs",
                      root_dx="1", chain="4")
    assert f"# PHYSICS: {THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID}" in text
    assert "use_adaptive_time_step" not in text
    assert all(domain.run.use_adaptive_time_step is False
               for domain in exp.domains)
    from fractions import Fraction

    from woof.domain_wizard import clock_decision

    root = exp.root
    step = Fraction(root.time_step) + Fraction(
        root.time_step_fract_num, root.time_step_fract_den)
    adaptive, why = clock_decision("auto", time_step=step, root_dx_m=1000.0,
                                   ratios=(4,))
    assert adaptive is False
    assert "250 m lies outside the 500..12000 m spacings" in why


def test_new_forecast_checks_the_suite_the_plan_will_run():
    """New forecast reads the same row the wizard binds, at the draft's grid."""

    from woof.gui.api import draft_default_suite, draft_finest_dx_m
    from woof.physics_menu import default_profile_for

    base = {"source": "gfs", "ladder": None, "chain": None, "dx_km": None}
    assert draft_finest_dx_m(base) == 12000.0
    assert draft_default_suite(base) == default_profile_for("gfs")
    sub_km = {**base, "dx_km": 2.25, "chain": "3"}
    assert draft_finest_dx_m(sub_km) == pytest.approx(750.0)
    assert draft_default_suite(sub_km) == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    ladder = {**base, "ladder": "12-3-1-0.5"}
    assert draft_finest_dx_m(ladder) == 500.0
    assert draft_default_suite(ladder) == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    assert draft_finest_dx_m({**base, "ladder": "auto"}) is None
    # A nested native HRRR draft takes it too.
    assert draft_default_suite({**sub_km, "source": "hrrr"}) == \
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID


#: Grids below and above 1 km, one domain and trees: (root km, nest ratios).
_CHECK_GRIDS = [(0.75, ()), (0.999, ()), (1.0, ()), (3.0, ()), (12.0, ()),
                (2.25, (3,)), (3.0, (3,)), (9.0, (3,)), (12.0, (4, 3)), (12.0, (4, 3, 2))]


def _check_request(root_km, ratios):
    from woof.domain_wizard import finest_spacing_m

    request = {"dx_km": root_km}
    if ratios:
        request.update(finest_dx_km=finest_spacing_m(root_km * 1000.0, ratios) / 1000.0,
                       domains=len(ratios) + 1)
    return request


@pytest.mark.parametrize("root_km,ratios", _CHECK_GRIDS)
def test_the_physics_check_names_the_suite_the_run_binds_for_every_source(root_km, ratios):
    """A check naming no suite and `woof domain` naming none answer from one table row.

    The check read the source's own default at every spacing, so `woof physics-catalog --check
    '{"source": "gfs", "dx_km": 0.75}'`, and New forecast's Physics step with it, named Morrison with YSU and
    Noah on a grid whose run binds the sub-km row.  Every registered source, below and above 1 km, on one
    domain and on a tree: the check's base and default are the suite the wizard binds, the one it writes.
    """

    from woof import physics_catalog as pc
    from woof.domain_wizard import finest_spacing_m, resolved_physics_profile
    from woof.physics_menu import registered_sources, spacing_default

    finest_m = finest_spacing_m(root_km * 1000.0, ratios)
    spacing_selected = []
    for source in registered_sources():
        run = resolved_physics_profile(source, None, finest_dx_m=finest_m, domains=len(ratios) + 1)
        verdict = pc.check({**_check_request(root_km, ratios), "source": source})
        assert (verdict["base_suite"], verdict["default_suite"]) == (run, run), source
        assert verdict["finest_dx_km"] == pytest.approx(finest_m / 1000.0)
        assert verdict["domains"] == len(ratios) + 1
        if verdict["valid"]:
            assert verdict["named_suite"] == run, source
        grid_row = spacing_default(source, finest_m, len(ratios) + 1)
        if grid_row is not None:
            assert run == grid_row["profile_id"] \
                == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, source
        spacing_selected.append(grid_row is not None)
    # The spacing row binds below 1 km and nowhere else. A source's
    # separate recommendation may select that same suite at coarser grids.
    assert any(spacing_selected) == (finest_m < 1000.0)


@pytest.mark.parametrize("source,root_dx,chain", [
    ("gfs", "0.75", ""), ("gfs", "3", ""), ("gfs", "2.25", "3"),
    ("hrrr", "0.75", ""), ("hrrr", "2.25", "3"),
])
def test_the_physics_check_and_the_written_file_carry_one_suite(tmp_path, source, root_dx, chain):
    """The same agreement read from the file `woof domain` writes, and from the check of that file's mix."""

    from woof import physics_catalog as pc

    ratios = tuple(int(value) for value in chain.split(",") if value)
    text, exp = _door(tmp_path, source=source, root_dx=root_dx, chain=chain)
    verdict = pc.check({**_check_request(float(root_dx), ratios), "source": source})
    assert f"# PHYSICS: {verdict['base_suite']}" in text
    assert pc.experiment_grid(text) == pytest.approx(
        {"dx_km": float(root_dx), "finest_dx_km": verdict["finest_dx_km"], "domains": len(exp.domains)})
    # A mix naming no suite changes that same suite when written into the file.
    assert pc.request_default_suite({**pc.experiment_grid(text), "source": source}) == verdict["base_suite"]


def test_schemes_picked_with_no_suite_change_the_default_the_grid_runs(tmp_path):
    """The check, `woof domain --physics-choices` and the run manifest take one base for a mix.

    The check reads the default at the finest grid, so the wizard writes picks naming no suite over that same
    suite and the manifest records it as their base: a pick of microphysics alone on a 750 m nest keeps MYNN and
    RUC, the boundary layer and land surface the Physics step showed running.
    """

    from woof import physics_catalog as pc
    from woof.physics_menu import default_profile_for
    from woof.runplan import build_plan, manifest_physics

    choices = {"microphysics": "nssl2-mp18"}
    verdict = pc.check({**_check_request(2.25, (3,)), "source": "gfs", "choices": choices})
    assert verdict["valid"] and verdict["base_suite"] == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    assert "physics_profile" not in verdict["plan_intent"]
    text, exp = _door(tmp_path, "--physics-choices", '{"microphysics": "nssl2-mp18"}', source="gfs")
    assert f"schemes picked over {THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID}" in text
    for domain in exp.domains:
        run = domain.run
        assert (run.mp_physics, run.bl_pbl_physics, run.sf_surface_physics) == (18, 5, 3)
    for root_dx_km, chain, base in ((2.25, "3", THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID),
                                    (3, None, default_profile_for("gfs"))):
        intent = {"point": "37.62,-122.2", "source": "gfs", "cycle": "2026-09-20T18", "hours": 1,
                  "card": "16gb", "root_dx_km": root_dx_km, "physics_choices": choices,
                  **({"chain": chain} if chain else {})}
        plan = build_plan({"schema": "gpuwm.run-plan.v1", "name": "p", "route": "prepared",
                           "config": {"intent": intent}, "output_root": str(tmp_path / f"out{root_dx_km}")},
                          source="test", base_dir=tmp_path, sha256="0" * 64)
        recorded = manifest_physics(plan)
        assert recorded["base_suite"] == base and recorded["choices"] == choices, recorded


def test_new_forecasts_physics_step_sends_the_drafts_nests_to_the_check():
    """The step's check carries the ladder or chain's finest grid, so its default is the plan's."""

    import json
    from types import SimpleNamespace

    from woof.gui.api import ApiError, PhysicsMixin, draft_check_grid

    def sent(payload):
        reply = PhysicsMixin.physics_check(SimpleNamespace(), payload, True)
        return json.loads(reply.body["argv"][reply.body["argv"].index("--check") + 1])

    assert "finest_dx_km" not in sent({"source": "gfs", "dx_km": 0.75})
    assert sent({"source": "gfs", "dx_km": 2.25, "chain": "3"})["finest_dx_km"] == pytest.approx(0.75)
    ladder = sent({"source": "gfs", "ladder": "12-3-1-0.5"})
    assert (ladder["finest_dx_km"], ladder["domains"]) == (0.5, 4)
    # A ladder's root is the wizard's 12 km root, where the check reads the root's cumulus.
    assert ladder["dx_km"] == 12.0
    assert "finest_dx_km" not in sent({"source": "gfs", "ladder": "auto"})
    assert sent({"source": "gfs", "ladder": "auto"})["dx_km"] == 12.0
    assert draft_check_grid({"ladder": None, "chain": None, "dx_km": 0.75}) == {}
    with pytest.raises(ApiError):
        sent({"source": "gfs", "ladder": "12-6"})


def test_the_menus_answer_the_spacing_row_for_one_domain_and_for_a_tree():
    """A one-domain answer alone read admitted for hrrr, whose trees keep their own."""

    from woof.physics_catalog import catalog
    from woof.physics_menu import source_menu

    for document in (source_menu("hrrr"), catalog(source="hrrr")):
        row = document["spacing_defaults"][0]
        assert row["profile_id"] == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
        assert row["admitted"] is True and row["why_not"] is None
        assert row["admitted_nested"] is True and row["why_not_nested"] is None
    for source in ("gfs", "era5", "hrrr-prs"):
        row = source_menu(source)["spacing_defaults"][0]
        assert row["admitted"] is True and row["admitted_nested"] is True
    row = source_menu("gem-gdps")["spacing_defaults"][0]
    assert row["admitted"] is True and row["admitted_nested"] is True


def test_the_assistant_marks_the_suite_the_ladder_runs_unnamed():
    """Below 1 km the ladder's default is the spacing row's where admitted."""

    from woof.gui.assistant.plan import ladder_default_profile
    from woof.physics_menu import source_menu

    def row_for(source):
        menu = source_menu(source)
        return {"default_profile": menu["default_profile_id"],
                "spacing_defaults": menu["spacing_defaults"]}

    gfs = row_for("gfs")
    assert ladder_default_profile(gfs, "12-3-1-0.5") == \
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    for ladder in ("12", "12-3", "12-3-1", "auto"):
        assert ladder_default_profile(gfs, ladder) == gfs["default_profile"]
    # A nested hrrr ladder takes it too.
    hrrr = row_for("hrrr")
    assert ladder_default_profile(hrrr, "12-3-1-0.5") == \
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    # A menu with no spacing rows (an older engine) reads as before.
    assert ladder_default_profile({"default_profile": "p1"}, "12-3-1-0.5") == "p1"


#: The grids of the deepest preset ladder, as a fit of the auto ladder describes them (gpuwm.gui.api.describe_fit).
_AUTO_SUB_KM_FIT = {"domains": [{"dx_km": 12.0}, {"dx_km": 3.0}, {"dx_km": 1.0}, {"dx_km": 0.5}]}
_AUTO_KM_FIT = {"domains": [{"dx_km": 12.0}, {"dx_km": 3.0}]}


def test_the_assistant_marks_the_default_of_the_ladder_auto_fits():
    """On auto the default marked is the one of the ladder its fit lands on.

    ladder_default_profile(row, "auto") returned the source's own default, so a plan that chose it left the profile
    unnamed and said "Morrison, the source's default" while the fitted 500 m ladder ran the sub-km suite.
    """

    from woof.gui.assistant.plan import ladder_default_profile
    from woof.physics_menu import source_menu

    menu = source_menu("gfs")
    gfs = {"default_profile": menu["default_profile_id"], "spacing_defaults": menu["spacing_defaults"]}
    assert ladder_default_profile(gfs, "auto", _AUTO_SUB_KM_FIT) == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    assert ladder_default_profile(gfs, "auto", _AUTO_KM_FIT) == gfs["default_profile"]
    # A named ladder is its own grid, whatever a fit says.
    assert ladder_default_profile(gfs, "12-3", _AUTO_SUB_KM_FIT) == gfs["default_profile"]


def test_the_assistant_fits_the_auto_ladder_before_it_marks_the_physics():
    """The plan asks the fit where auto lands, then marks and leaves unnamed that ladder's default only."""

    from datetime import timezone
    from types import SimpleNamespace

    from woof.gui.assistant.plan import Planner
    from woof.physics_menu import source_menu

    menu = source_menu("gfs")
    own = menu["default_profile_id"]
    row = {"id": "gfs", "name": "GFS", "coverage": None, "horizon_hours": 384, "step_hours": 1,
           "default_profile": own, "spacing_defaults": menu["spacing_defaults"],
           "profiles": [{"id": own, "summary": "Morrison"},
                        {"id": THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, "summary": "Thompson, MYNN, RUC"}]}
    fits = []

    def fit(payload, dry):
        fits.append(dict(payload))
        return SimpleNamespace(body={"fit": {**_AUTO_SUB_KM_FIT, "words": "4 domains."}})

    api = SimpleNamespace(sources=lambda: {"default_cycle": "2026-09-20T18", "sources": [row],
                                           "ladders": ["12", "12-3", "12-3-1", "12-3-1-0.5", "auto"]},
                          system=lambda: {"card": "24gb", "devices": [{"name": "card"}]}, fit=fit)
    for picked, named in ((own, own), (THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, None)):
        asked = {}
        picks = {"day": "now", "box_size": "600", "ladder": "auto", "source": "gfs", "physics": picked,
                 "machine": "this-computer"}

        def decide(question, state):
            asked[question.id] = dict(question.options)
            return {"question": question.id, "choice": picks[question.id], "reason": "",
                    "options": dict(question.options), "probability": 1.0}

        fits.clear()
        planner = Planner(None, decide, lambda name, args, fn: fn(), api,
                          now=datetime(2026, 9, 20, 20, tzinfo=timezone.utc))
        plan = planner.plan("storms", place={"lat": 37.6, "lon": -122.2, "place": "the bay", "finest_km": None})
        # Fitted before the physics question, with no physics named: the run the field left alone makes.
        assert fits[0]["ladder"] == "auto" and "profile" not in fits[0]
        options = asked["physics"]
        assert options[THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID].endswith("(the default on this ladder)")
        assert "default" not in options[own]
        # The source's own default is not what the fitted ladder runs unnamed, so choosing it names it.
        assert plan["fields"]["profile"] == named and plan["fields"]["ladder"] == "auto"


def test_new_forecast_reads_the_auto_ladders_fitted_grid():
    """The Physics step, Start and the night check read the grid the auto ladder's fit landed on.

    draft_check_grid returned {} for auto, so the check was asked about {"source": "gfs"} and named Morrison with
    YSU and Noah, and New forecast tagged them default, while the fitted 500 m ladder runs Thompson, MYNN and RUC.
    """

    import json
    from types import SimpleNamespace

    from woof.gui.api import ApiError, PhysicsMixin, draft_default_suite, fit_grid
    from woof.physics_menu import default_profile_for

    def sent(payload):
        reply = PhysicsMixin.physics_check(SimpleNamespace(), payload, True)
        asked = json.loads(reply.body["argv"][reply.body["argv"].index("--check") + 1])
        return asked.get("finest_dx_km"), asked.get("domains")

    grid = fit_grid(_AUTO_SUB_KM_FIT)
    assert grid == {"finest_dx_km": 0.5, "domains": 4}
    assert sent({"source": "gfs", "ladder": "auto", **grid}) == (0.5, 4)
    assert sent({"source": "gfs", "ladder": "auto", **fit_grid(_AUTO_KM_FIT)}) == (3.0, 2)
    # A fitted grid is the auto ladder's alone, both keys together, on a grid the auto ladder reaches.
    for payload in ({"ladder": "12-3", **grid}, {"dx_km": 3.0, **grid}, {"ladder": "auto", "finest_dx_km": 0.5},
                    {"ladder": "auto", "finest_dx_km": 0.75, "domains": 4}):
        with pytest.raises(ApiError):
            sent({"source": "gfs", **payload})
    assert draft_default_suite({"source": "gfs", "ladder": "auto", "fitted": grid}) == \
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    assert draft_default_suite({"source": "gfs", "ladder": "auto", "fitted": None}) == default_profile_for("gfs")


def test_a_start_on_the_auto_ladder_checks_its_picks_on_the_fitted_grid():
    """Picks on auto ride into the plan as they are and are checked where the fit lands.

    A Start that arrives without the fitted grid is fitted first.  Before this the check read the source's own
    default, so nssl2-mp18 picked on a fitted 500 m ladder was recorded as a YSU, Noah and KF suite.
    """

    import json

    from woof import physics_catalog as pc
    from woof.gui.api import CreateMixin, Reply

    class Runner:
        def __init__(self):
            self.checks = []

        def query(self, argv, **_):
            body = json.loads(argv[argv.index("--check") + 1])
            self.checks.append(body)
            return pc.check(body)

    class Door(CreateMixin):
        def __init__(self):
            self.runner = Runner()
            self.fits = []

        def fit(self, payload, dry):
            self.fits.append(dict(payload))
            return Reply(200, {"fit": _AUTO_SUB_KM_FIT})

    choices = {"microphysics": "nssl2-mp18"}
    payload = {"name": "x", "physics_choices": choices, "profile": "a-set-the-page-named"}
    base = {"source": "gfs", "ladder": "auto", "fitted": None, "profile": "a-set-the-page-named", "dx_km": None,
            "nz": None, "cycle": "2026-09-20T18", "hours": 1, "lat": 37.62, "lon": -122.2, "following": False}
    door = Door()
    # The fit itself: its depth is not known yet, so the picks go to the wizard as they are, over no named set.
    fitting = dict(base)
    assert door.composed_physics(payload, fitting) is None
    assert (fitting["physics_choices"], fitting["profile"], door.runner.checks) == (choices, None, [])
    # A Start without the fitted grid is fitted once, with the same plan, and checked on the grid it lands on.
    start = dict(base)
    door.bind_auto_grid(payload, start)
    assert start["fitted"] == {"finest_dx_km": 0.5, "domains": 4} and len(door.fits) == 1
    record = door.composed_physics(payload, start)
    asked = door.runner.checks[-1]
    assert (asked["finest_dx_km"], asked["domains"]) == (0.5, 4)
    assert record["base_suite"] == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    switches = pc.check(asked)["resolved"]
    assert record["resolved"] == switches
    assert (start["physics_choices"], start["profile"]) == (choices, None)
    # A Start that carries the grid is not fitted again.
    door.bind_auto_grid(payload, {**base, "fitted": {"finest_dx_km": 0.5, "domains": 4}})
    assert len(door.fits) == 1


def test_an_auto_ladder_mix_records_the_base_its_fitted_file_carries(tmp_path):
    """The run manifest reads the grid the auto ladder's fit lands on.

    Measured on 2.8.1 before this: an intent {ladder: auto, card: 24gb, physics_choices: {microphysics:
    nssl2-mp18}} resolved to a file headed "schemes picked over thompson-mp8-mynn-mynn-ruc..." while the manifest
    recorded base_suite morrison-mp10 and named a YSU, Noah and KF suite the run does not carry.
    """

    import json

    from woof import physics_catalog as pc
    from woof.physics_menu import default_profile_for
    from woof.runplan import build_plan, manifest_physics, refresh_manifest_physics, resolve_plan

    choices = {"microphysics": "nssl2-mp18"}
    intent = {"point": "37.62,-122.2", "source": "gfs", "cycle": "2026-09-20T18", "hours": 1,
              "card": "24gb", "ladder": "auto", "physics_choices": choices}
    plan = build_plan({"schema": "gpuwm.run-plan.v1", "name": "p", "route": "prepared",
                       "config": {"intent": intent}, "output_root": str(tmp_path / "out")},
                      source="test", base_dir=tmp_path, sha256="0" * 64)
    pending = manifest_physics(plan)
    assert pending["base_suite"] is None and pending["suite"] is None and "auto ladder" in pending["unresolved"]
    resolution, _exp, _data = resolve_plan(plan, generate_into=tmp_path / "gen", require_inputs=False)
    text = resolution["generated_config"]
    grid = pc.experiment_grid(text)
    # The fit reached below 1 km, the grid the defect was measured on.
    assert grid["finest_dx_km"] < 1.0, grid
    written = default_profile_for("gfs", grid["finest_dx_km"] * 1000.0, grid["domains"])
    assert written == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    assert f"schemes picked over {written}" in text
    recorded = manifest_physics(plan, generated_config=text)
    assert recorded["base_suite"] == written and recorded["choices"] == choices and "unresolved" not in recorded
    # The published record is rewritten with it once the plan resolves.
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(json.dumps({"schema": "m", "physics": pending}), encoding="utf-8")
    refresh_manifest_physics(manifest, plan, text)
    assert json.loads(manifest.read_text(encoding="utf-8")) == {"schema": "m", "physics": recorded}
