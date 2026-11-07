"""The machine-facing front door: plan in, typed events out.

Every test here is CPU-only.  The one thing this module cannot exercise
without a card is the integration itself, so the GPU half is replaced at
the seam the front door actually uses -- ``runtime.run_experiment``,
driving the REAL :class:`woof.runplan.RunObserver` through the REAL
progress protocol.  What that buys is that the observer's contract is
under test rather than mocked away: a stub that called ``preparing`` with
a phase the mapping table does not know, or emitted progress before the
forecast stage opened, would fail here the same way the live pipeline
would.

The config is ``tests/test_case_data.py``'s fixture pair, reused rather
than copied: it is the smallest TOML that loads through the same
``load_experiment_case_bytes`` seam ``woof run`` uses, it names no case,
and reusing it means a schema change breaks one fixture instead of two.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conftest import requires_cupy

from woof import explain
from woof.runplan import (EVENT_SCHEMA, EVENT_TAGS, EVENTS_FILENAME,
                           MANIFEST_FILENAME, MANIFEST_SCHEMA, PLAN_SCHEMA,
                           STAGES, EventStream, PlanError, RunObserver,
                           build_plan, execute_plan, load_plan,
                           probe_environment, read_events, resolve_plan,
                           run_plan_main)
from test_case_data import make_case_toml


#: A fetched-source receipt in the real fetch's shape.  Since abf04433ed8
#: (parallel cache reuse) the chain snapshots it (launch_supplements.
#: hrrr_source_manifest) right after the fetch, so a placeholder byte is a
#: malformed manifest that stops the chain before the preparation stage.
_FETCHED_SUMS = "0" * 64 + "  hrrr.t12z.wrfprsf00.grib2\n"


def _plan_document(config_path: Path, run_dir: Path, **overrides):
    document = {
        "schema": PLAN_SCHEMA,
        "name": "front-door-fixture",
        "route": "experiment",
        "config": {"path": str(config_path)},
        "output_root": str(run_dir),
    }
    document.update(overrides)
    return document


def _write_plan(tmp_path, config_path, run_dir, **overrides) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(_plan_document(config_path, run_dir, **overrides)),
        encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The plan document
# ---------------------------------------------------------------------------


def test_a_plan_round_trips_through_the_loader_with_paths_made_absolute(
        tmp_path):
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan = load_plan(_write_plan(tmp_path, config, run_dir))

    assert plan.name == "front-door-fixture"
    assert plan.route == "experiment"
    assert plan.config_path == config
    assert plan.run_dir == run_dir
    assert len(plan.sha256) == 64
    # Every run option the route declares is resolved, present or not.
    assert set(plan.run_options) == {
        "device", "dry_run", "restart", "health_debug", "grib2", "verify_visuals",
        "geog_root", "render_products", "render_section", "keep_checkpoints", "input_cycle",
        # Where the hourly products are published (woof.product_publish).
        "publish_root", "publish_cmd",
        # The ensemble option every route carries; absent means one forecast.
        "ensemble"}
    assert plan.run_options["ensemble"] is None
    assert plan.run_options["geog_root"] is None
    assert plan.run_options["render_products"] is None
    assert plan.run_options["render_section"] is None
    assert plan.run_options["dry_run"] is False
    assert plan.run_options["verify_visuals"] is True


def test_observation_verification_is_outside_the_config_identity(tmp_path):
    config = make_case_toml(tmp_path)
    default = load_plan(_write_plan(tmp_path, config, tmp_path / "run"))
    off = load_plan(_write_plan(tmp_path, config, tmp_path / "run", run_options={"verify_visuals": False}))
    assert default.config_bytes() == off.config_bytes() == config.read_bytes()
    assert default.run_options["verify_visuals"] is True
    assert off.run_options["verify_visuals"] is False
    from dataclasses import fields
    from woof.config import RunConfig
    assert "verify_visuals" not in {field.name for field in fields(RunConfig)}


def test_a_section_line_is_a_run_option_and_a_file_resolves_beside_the_plan(
        tmp_path):
    """``render_section`` is ``woof render --section``'s own value.

    A line is kept as written; a file is resolved against the plan's own
    directory and read as the renderer reads it; an ``xsec:`` term with
    no line is refused when the plan is built, before anything runs.
    """

    config = make_case_toml(tmp_path)
    spec = "composite_reflectivity,xsec:QCLOUD=0.01,0.1/wa"
    line = "38.3,-99.0,38.3,-98.4"
    plan = load_plan(_write_plan(tmp_path, config, tmp_path / "run",
                                 run_options={"render_products": spec,
                                              "render_section": line}))
    assert plan.run_options["render_section"] == line
    (tmp_path / "cut.json").write_text(json.dumps(
        {"points": [[38.3, -99.0], [38.4, -98.7], [38.3, -98.4]],
         "extend_km": 10}), encoding="utf-8")
    plan = load_plan(_write_plan(tmp_path, config, tmp_path / "run",
                                 run_options={"render_products": spec,
                                              "render_section": "cut.json"}))
    assert plan.run_options["render_section"] == str(tmp_path / "cut.json")
    for options, expected in (
            ({"render_products": spec}, "names no line"),
            ({"render_products": "xsec:wa"}, "names no line"),
            ({"render_products": spec, "render_section": "missing.json"},
             "neither 'lat,lon,lat,lon' nor a readable JSON file")):
        with pytest.raises(PlanError, match=expected):
            load_plan(_write_plan(tmp_path, config, tmp_path / "run",
                                  run_options=options))
    # No section term, no line needed; `none` draws nothing.
    for products in ("composite_reflectivity", "none"):
        assert load_plan(_write_plan(
            tmp_path, config, tmp_path / "run",
            run_options={"render_products": products})
        ).run_options["render_section"] is None


def test_relative_paths_resolve_against_the_plans_own_directory(tmp_path):
    nested = tmp_path / "plans"
    nested.mkdir()
    make_case_toml(tmp_path)
    (tmp_path / "case.toml").rename(nested / "case.toml")
    path = nested / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "relative", "route": "experiment",
        "config": {"path": "case.toml"}, "output_root": "runs/one",
    }), encoding="utf-8")

    plan = load_plan(path)
    assert plan.config_path == nested / "case.toml"
    assert plan.run_dir == (nested / "runs" / "one").resolve()


def test_an_unknown_top_level_key_is_refused_and_names_the_known_ones(
        tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"inline": "x = 1"}, "outputroot": "typo"},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "'outputroot'" in text
    assert "output_root" in text
    # The consequence, not just the fact -- the repo's refusal voice.
    assert "no key is ignored" in text


def test_an_unknown_schema_id_is_refused_naming_both_ids(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": "gpuwm.run-plan.v2", "name": "x",
                    "route": "experiment", "config": {"inline": "x = 1"}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert PLAN_SCHEMA in text and "gpuwm.run-plan.v2" in text


def test_an_unknown_route_is_refused_and_lists_the_routes_this_build_has(
        tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "teleport",
                    "config": {"inline": "x = 1"}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert "experiment" in str(refusal.value)


def test_config_must_carry_exactly_one_of_the_three_spellings(tmp_path):
    for config, expected in (
            ({}, "[]"),
            ({"path": "a.toml", "inline": "x = 1"}, "['inline', 'path']"),
            ({"inline": "x = 1", "intent": {}}, "['inline', 'intent']")):
        with pytest.raises(PlanError) as refusal:
            build_plan({"schema": PLAN_SCHEMA, "name": "x",
                        "route": "experiment", "config": config},
                       source="probe.json", base_dir=tmp_path,
                       sha256="0" * 64)
        text = str(refusal.value)
        # The refusal names exactly which spellings it found, so the
        # reader does not have to work out which two collided.
        assert expected in text
        assert "'path'" in text and "'inline'" in text and "'intent'" in text


def test_a_run_option_the_route_does_not_declare_is_refused(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"inline": "x = 1"},
                    "run_options": {"overclock": True}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert "overclock" in str(refusal.value)


def test_a_fetch_argument_list_is_checked_against_gpuwm_fetchs_own_parser(
        tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"inline": "x = 1"},
                    "fetch": {"args": ["--not-a-fetch-flag"]}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert "fetch.args" in str(refusal.value)


def test_defaulted_plan_keys_are_reported_never_applied_silently(tmp_path):
    plan = build_plan(
        {"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
         "config": {"inline": "x = 1"}},
        source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    keys = {(entry["scope"], entry["key"])
            for entry in plan.automatic_resolutions}
    assert ("plan", "output_root") in keys
    assert ("run_options", "dry_run") in keys
    assert ("run_options", "device") in keys


# ---------------------------------------------------------------------------
# Resolution through the real config seam
# ---------------------------------------------------------------------------


def test_the_envelope_resolves_through_the_real_config_loader(tmp_path):
    config = make_case_toml(tmp_path)
    plan = load_plan(_write_plan(tmp_path, config, tmp_path / "run"))

    resolution, exp, data = resolve_plan(plan)

    # The snapshot is the objects the model will run, not a re-read.
    assert resolution["configuration"]["experiment"]["name"] == exp.name
    assert resolution["configuration"]["case_data"]["output_title"] == (
        data.output_title)
    assert resolution["plan"]["config_sha256"] == (
        resolution["plan"]["config_sha256"])
    # It is JSON, all the way down -- no consumer needs a Python type.
    json.dumps(resolution)


def test_an_inline_config_takes_the_same_route_as_a_config_on_disk(tmp_path):
    config = make_case_toml(tmp_path)
    text = config.read_text(encoding="utf-8")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "inline", "route": "experiment",
        "config": {"inline": text}, "output_root": str(tmp_path / "run"),
    }), encoding="utf-8")

    inline_resolution, inline_exp, _ = resolve_plan(load_plan(path))
    file_resolution, file_exp, _ = resolve_plan(
        load_plan(_write_plan(tmp_path, config, tmp_path / "run")))

    assert inline_exp == file_exp
    assert (inline_resolution["configuration"]
            == file_resolution["configuration"])


def test_the_derived_timestep_of_every_domain_is_reported_out_loud(tmp_path):
    config = make_case_toml(tmp_path)
    resolution, exp, _ = resolve_plan(
        load_plan(_write_plan(tmp_path, config, tmp_path / "run")))

    steps = [entry for entry in resolution["automatic_resolutions"]
             if entry["key"] == "dt"]
    assert len(steps) == len(exp.domains)
    root = steps[0]
    assert root["basis"] == "declared_time_step"
    assert root["value"] == pytest.approx(float(exp.domains[0].run.dt))
    # The exact rational is carried beside the binary64 image, because
    # the two are genuinely different numbers.
    assert root["exact"]


def test_a_schema_default_the_config_did_not_spell_is_reported(tmp_path):
    config = make_case_toml(tmp_path)
    resolution, _exp, _ = resolve_plan(
        load_plan(_write_plan(tmp_path, config, tmp_path / "run")))
    defaults = {entry["key"] for entry in resolution["automatic_resolutions"]
                if entry.get("basis") == "schema_default"
                and entry["scope"] == "experiment"}
    # The fixture spells none of these; all three change the model.
    assert {"feedback", "blend_width", "spec_bdy_width"} <= defaults


_PROJECTION_TOML = """
[projection]
map_proj = "lambert"
ref_lat = 35.0
ref_lon = -97.0
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -97.0
"""


def test_a_field_authored_as_its_own_table_is_not_a_schema_default(tmp_path):
    """The author wrote it; ``--resolve`` must not say the schema did.

    ``projection``, ``relocation`` and ``perturbation`` are
    ``ExperimentConfig`` fields written as TOP-LEVEL tables, and the
    spelled-key check read only inside ``[experiment]``.  A moving-nest
    plan was therefore told ``relocation`` was a schema default and handed
    the schema's value -- ``enabled = false`` -- for a nest that follows a
    storm.  Both directions are asserted, because a check that reported
    nothing would also pass the first half.
    """

    from test_case_data import _EXPERIMENT_TOML

    def _defaults(experiment_toml):
        config = make_case_toml(tmp_path, experiment=experiment_toml)
        resolution, _exp, _ = resolve_plan(
            load_plan(_write_plan(tmp_path, config, tmp_path / "run")))
        return {entry["key"]
                for entry in resolution["automatic_resolutions"]
                if entry.get("basis") == "schema_default"
                and entry["scope"] == "experiment"}

    assert "projection" in _defaults(_EXPERIMENT_TOML)
    spelled = _defaults(_EXPERIMENT_TOML + _PROJECTION_TOML)
    assert "projection" not in spelled
    # Still reported for the tables this config genuinely does not carry.
    assert {"relocation", "perturbation"} <= spelled


def test_the_spelled_check_covers_every_table_a_field_can_be_written_as():
    """Directly, on the one function, for the tables the fixture cannot carry.

    ``[relocation]`` needs a validated two-domain tree to load, so the
    end-to-end fixture above cannot spell it; the mechanism is the same
    one, so it is asserted here where the document can be written by hand.
    """

    from woof.runplan import _schema_default_resolutions

    experiment = {"name": "x", "start_time": None, "run_seconds": 1.0,
                  "restart_interval_s": 0.0}
    document = {"experiment": experiment}
    bare = {entry["key"]
            for entry in _schema_default_resolutions(document)}
    assert {"relocation", "projection", "perturbation"} <= bare

    authored = {entry["key"] for entry in _schema_default_resolutions({
        "experiment": experiment,
        "relocation": {"enabled": True, "grid_id": 2},
        "projection": {"map_proj": "lambert"},
        "perturbation": {"bubbles": []},
    })}
    assert not ({"relocation", "projection", "perturbation"} & authored)
    # An [experiment] key still resolves the same way it always did.
    assert "blend_width" in authored


def test_a_library_warning_reaches_the_stream_as_fields(tmp_path):
    captured: list[dict[str, str]] = []
    from woof.runplan import collect_warnings

    with collect_warnings(captured):
        explain.warn("something worth saying", "and why it is worth it")
    assert captured == [{"action": "something worth saying",
                         "why": "and why it is worth it"}]

    # Detached again: the sink stops receiving, and the observer list is
    # not left holding a reference to a dead test's list.
    explain.warn("after", "detach")
    assert len(captured) == 1


def test_a_warning_observer_that_raises_cannot_fail_the_run():
    def hostile(record):
        raise RuntimeError("observer exploded")

    explain.add_warning_observer(hostile)
    try:
        explain.warn("the run continues")  # must not raise
    finally:
        explain.remove_warning_observer(hostile)


# ---------------------------------------------------------------------------
# The intent route
# ---------------------------------------------------------------------------


_INTENT = {"point": "39,-98", "source": "era5", "cycle": "2024-05-03T12",
           "hours": 1, "vram_gib": 24}


def _intent_plan(tmp_path, run_dir, **intent):
    document = {
        "schema": PLAN_SCHEMA, "name": "intent-fixture",
        "route": "experiment", "config": {"intent": {**_INTENT, **intent}},
        "output_root": str(run_dir),
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_an_intent_block_becomes_the_wizards_own_argv(tmp_path):
    from woof.runplan import intent_arguments

    arguments = intent_arguments(_INTENT, out=tmp_path / "c.toml")
    assert "--point" in arguments and "39,-98" in arguments
    assert "--cycle" in arguments and "2024-05-03T12" in arguments
    assert "--vram-gib" in arguments and "24" in arguments
    # run-plan owns where the config lands; a plan cannot redirect it.
    assert arguments[-2:] == ["--out", str(tmp_path / "c.toml")]


def test_an_intent_key_the_wizard_has_no_flag_for_is_refused(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"intent": {**_INTENT, "nx": 400}}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "'nx'" in text
    # And it names the keys that DO exist, so a front end can correct.
    assert "point" in text and "ladder" in text


def test_an_intent_sets_the_vertical_level_count_through_the_wizards_own_flag(tmp_path):
    # A front end asking for more eta levels spells the wizard's --nz, and the resolved grid and the generated
    # config both carry that count, resampled from the wizard's own ladder (a user asked for more levels, nz).
    from woof.runplan import intent_arguments

    arguments = intent_arguments({**_INTENT, "nz": 80}, out=tmp_path / "c.toml")
    assert arguments[arguments.index("--nz") + 1] == "80"
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run", nz=80))
    resolution, exp, _data = resolve_plan(plan, require_inputs=False)
    assert {domain.run.nz for domain in exp.domains} == {80}
    assert "nz = 80" in resolution["generated_config"]


def test_an_intent_carries_tiles_and_acknowledgements_into_the_config(
        tmp_path):
    """--tiles and --ack were settable only through an experiment document.

    A front end building an intent could not ask for streaming or declare
    a governed experiment: ``config.intent`` had no such keys and refused
    them.  They are wizard flags delivered through the generated config,
    and ``ack`` is an append flag, so a list is spelled one ``--ack`` per
    id (``--ack A B`` is refused by the wizard's parser).
    """

    import tomllib

    from woof.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      THOMPSON_PROFILE_ID)
    from woof.runplan import intent_arguments

    acks = [ASYMMETRIC_RADIATION_NOCTURNAL_ACK, "second-id"]
    arguments = intent_arguments({**_INTENT, "ack": acks, "tiles": "auto"},
                                 out=tmp_path / "c.toml")
    assert [arguments[i + 1] for i, token in enumerate(arguments)
            if token == "--ack"] == acks
    assert arguments[arguments.index("--tiles") + 1] == "auto"

    # 33.8 N in late April: an 18 h window from 12Z runs through local
    # night, which a longwave-off suite may run only when declared.
    plan = load_plan(_intent_plan(
        tmp_path, tmp_path / "run", point="33.8,-87.29",
        cycle="2011-04-26T12", hours=18, physics_profile=THOMPSON_PROFILE_ID,
        tiles="auto", ack=[ASYMMETRIC_RADIATION_NOCTURNAL_ACK]))
    resolution, exp, _data = resolve_plan(plan, require_inputs=False)
    generated = tomllib.loads(resolution["generated_config"])
    assert generated["tiles"] == {"mode": "auto"}
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in (
        generated["experiment"]["acknowledgements"])
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in exp.acknowledgements
    assert exp.tiles.mode == "auto"


def test_an_intent_sizes_a_point_to_its_own_extent(tmp_path):
    from woof.runplan import intent_arguments

    arguments = intent_arguments({**_INTENT, "point_extent_km": 900},
                                 out=tmp_path / "c.toml")
    assert arguments[arguments.index("--point-extent-km") + 1] == "900"
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run",
                                  point_extent_km=900))
    _resolution, exp, _data = resolve_plan(plan, require_inputs=False)
    root = exp.domains[0].run
    assert max(root.nx, root.ny) * root.dx / 1000.0 <= 900.0


def test_an_intent_without_a_place_is_refused(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"intent": {"cycle": "latest"}}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert "no default place" in str(refusal.value)


def test_an_intent_source_the_route_cannot_run_is_refused_with_the_reason(
        tmp_path):
    """gfs emissions carry no [case_data]; say so, and name the route
    that DOES drive them, so the remedy is one edited field."""

    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"intent": {**_INTENT, "source": "gfs"}}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "case_data" in text
    assert 'route = "prepared"' in text


def test_the_wizard_writes_the_config_and_resolution_carries_it_verbatim(
        tmp_path):
    run_dir = tmp_path / "run"
    plan = load_plan(_intent_plan(tmp_path, run_dir))
    resolution, exp, data = resolve_plan(plan, require_inputs=False)

    assert resolution["plan"]["config_kind"] == "intent"
    # The generated text is carried whole: the caller never typed it.
    assert "Emitted by `woof domain`" in resolution["generated_config"]
    assert "[case_data]" in resolution["generated_config"]
    assert exp.name and data.output_title
    # A query-mode resolution generates into a throwaway directory.
    assert not run_dir.exists()


def test_the_generation_itself_is_an_automatic_resolution(tmp_path):
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run"))
    resolution, _exp, _data = resolve_plan(plan, require_inputs=False)
    entries = {(e["scope"], e["key"]) for e
               in resolution["automatic_resolutions"]}
    assert ("config", "generated_by") in entries
    assert ("config", "generated_config") in entries
    # Domain size is FITTED, never typed -- said out loud, per domain.
    fitted = [e for e in resolution["automatic_resolutions"]
              if e["key"] == "nx_ny"]
    assert fitted and fitted[0]["basis"] == "fitted_to_vram_budget"


def test_resolve_and_estimate_both_work_on_an_intent_plan(tmp_path, capsys):
    """Studio's live estimate strip runs on intent, before any config."""

    from woof.cli import build_parser

    plan_path = _intent_plan(tmp_path, tmp_path / "run")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--resolve", str(plan_path)])) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["generated_config"]

    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["vram"]["estimate_bytes"] > 0
    assert estimate["disk"]["total_frames"] > 0


def test_resolve_reports_the_engines_own_minimum_domain_size(tmp_path):
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run"))
    resolution, _exp, _data = resolve_plan(plan, require_inputs=False)
    floor = resolution["domain_size_floor"]
    assert floor["root_mass_points"] == {"nx": 60, "ny": 48}
    assert floor["nest_span_mass_points"] == 12
    assert floor["clearance_rows"] == 10
    assert "FITTED" in floor["basis"]


def test_the_floor_is_derived_from_the_wizard_not_transcribed(monkeypatch):
    """Move the wizard's bracket; the reported floor must move with it."""

    from woof import domain_wizard
    from woof.runplan import domain_size_floor

    monkeypatch.setattr(domain_wizard, "_MIN_SCALE", 1.0)
    assert domain_size_floor()["root_mass_points"] == {"nx": 110, "ny": 88}


def test_a_shape_that_cannot_fit_refuses_with_the_engines_words_and_the_floor(
        tmp_path):
    """The fit refusal a front end most needs the numbers from."""

    plan = load_plan(_intent_plan(
        tmp_path, tmp_path / "run", ladder="12-3-1-0.5", vram_gib=4))
    with pytest.raises(PlanError) as refusal:
        resolve_plan(plan, require_inputs=False)
    text = str(refusal.value)
    assert "does not fit" in text
    # The wizard's own sentence, whichever of its two refusals fired.  On
    # a 4 GiB card it is now the harder one: since task 206 the wizard
    # asks whether the suite's GRID-INDEPENDENT envelope alone is the
    # whole card, instead of inferring that from a budget that no longer
    # contains it, and on this card it is -- so "minimum layout" would
    # point the reader at a grid lever that cannot help.
    assert ("minimum layout" in text
            or "no budget for ladder" in text)
    assert "grid-independent" in text
    assert "root_mass_points" in text      # the structured floor beside it


# ---------------------------------------------------------------------------
# The prepared route (the credential-free golden path)
# ---------------------------------------------------------------------------


_GFS_INTENT = {"point": "35.2,-97.4", "source": "gfs", "root_dx_km": 3,
               "cycle": "2024-05-03T12", "hours": 6, "vram_gib": 24}


def _prepared_plan(tmp_path, run_dir, **intent):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "golden-path", "route": "prepared",
        "config": {"intent": {**_GFS_INTENT, **intent}},
        "output_root": str(run_dir),
    }), encoding="utf-8")
    return path


def test_a_gfs_intent_is_accepted_on_the_prepared_route(tmp_path):
    """The refusal on the experiment route is not a global one."""

    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    assert plan.route == "prepared"
    assert plan.config_intent["source"] == "gfs"


def test_the_prepared_route_resolves_a_config_with_no_case_data(tmp_path):
    """GFS emissions carry no [case_data]; the route says so, not fails."""

    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    resolution, exp, data = resolve_plan(plan, require_inputs=False)

    assert data is None
    assert resolution["configuration"]["case_data"] is None
    assert resolution["configuration"]["experiment"]["name"]
    assert resolution["declared_inputs"] == []
    # And the generated config really is the gfs one.
    assert "[fetch]" in resolution["generated_config"]
    assert 'source = "gfs"' in resolution["generated_config"]


def test_estimate_works_on_a_prepared_route_plan(tmp_path, capsys):
    from woof.cli import build_parser

    plan_path = _prepared_plan(tmp_path, tmp_path / "run")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["vram"]["estimate_bytes"] > 0
    assert document["disk"]["total_frames"] > 0


def test_the_prepared_route_declares_the_data_dir_option(tmp_path):
    """It downloads its own inputs, so it takes where they land."""

    from woof.runplan import ROUTES

    assert "data_dir" in ROUTES["prepared"].run_options
    # The experiment route declares its inputs in [case_data] instead.
    assert "data_dir" not in ROUTES["experiment"].run_options
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    assert plan.run_options["data_dir"] is None


def test_go_stage_labels_all_map_onto_a_run_plan_stage():
    """A label go can emit that this front door cannot place is a hole."""

    from woof.runplan import STAGES, _GO_STAGES

    assert set(_GO_STAGES.values()) <= set(STAGES)
    # Every label go actually uses is covered.
    assert set(_GO_STAGES) == {"authority", "fetch", "manifest", "prepare",
                               "forecast", "render"}


def test_the_chain_observer_renders_go_stages_as_run_plan_stages(tmp_path):
    """Drive the observer with go's own hook vocabulary."""

    from woof.runplan import _GoObserver

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        chain = _GoObserver(observer)
        chain.stage_begin(label="authority", command=["x"])
        chain.stage_end(label="authority", exit_code=0, ok=True,
                        elapsed_seconds=1.0, progress=None)
        chain.stage_begin(label="fetch", command=["x"])
        chain.stage_begin(label="forecast", command=["x"])
        chain.stage_heartbeat(
            label="forecast", elapsed_seconds=40.0,
            progress={"status": "RUNNING", "model_elapsed_seconds": 1200.0})
        observer.finish_stage()

    records = read_events(tmp_path / EVENTS_FILENAME)
    stages = [r["stage"] for r in records if r["event"] == "stage_started"]
    # prepare opens, closes for the download, and opens again: go's real
    # order, not a flattened one.
    assert stages == ["prepare", "fetch", "forecast"]
    progress = next(r for r in records if r["event"] == "model_progress")
    assert progress["model_seconds"] == 1200.0
    assert progress["speed_x"] == 30.0
    assert progress["step_ms"] is None
    # A polled sample says so, so a consumer never mistakes it for a
    # per-step one.
    assert progress["source"] == "stage_progress_file"


def test_a_failed_chain_stage_is_named_in_a_warning(tmp_path):
    from woof.runplan import _GoObserver

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        chain = _GoObserver(RunObserver(events))
        chain.stage_begin(label="prepare", command=["x"])
        chain.stage_end(label="prepare", exit_code=2, ok=False,
                        elapsed_seconds=3.0, progress=None)
    warning = next(r for r in read_events(tmp_path / EVENTS_FILENAME)
                   if r["event"] == "warning")
    assert warning["code"] == "chain_stage_failed"
    assert warning["stage"] == "prepare"
    assert warning["exit_code"] == 2


@pytest.mark.parametrize("channel", ["stderr", "stdout", "empty"])
def test_failed_preparation_carries_subprocess_diagnostic_into_final_event(
        tmp_path, monkeypatch, channel):
    """A remote client reading only job.error still receives the stage's cause."""
    import sys
    import woof.go_cli as go_cli
    from woof import capabilities

    # The fake chain stops in a CPU subprocess before any model execution.
    monkeypatch.setattr(capabilities, "require", lambda *args, **kwargs: None)

    diagnostic = 'GFS Rust bridge failed: Section 5 declares 0 data points'
    script = "import sys; "
    if channel != "empty":
        script += f"print({diagnostic!r}, file=sys.{channel}); "
    script += "sys.exit(2)"

    def fake_go_main(args, *, observer=None, **_):
        try:
            go_cli.run_stage("prepare", [sys.executable, "-c", script],
                             explain=False, observer=observer)
        except go_cli.GoStageFailed:
            return 2
        pytest.fail("The failed preparation must stop the chain")

    monkeypatch.setattr(go_cli, "go_main", fake_go_main)
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        assert execute_plan(plan, events=events) == 1
    records = read_events(plan.run_dir / EVENTS_FILENAME)
    failed = records[-1]
    assert failed["event"] == "failed"
    assert failed["stage"] == "prepare"
    assert "The prepare stage failed (exit 2)." in failed["message"]
    assert "No later stage ran." in failed["message"]
    assert (diagnostic in failed["message"]) is (channel != "empty")
    assert not any(row["event"] == "stage_started" and row["stage"] == "forecast"
                   for row in records)


def test_go_runs_the_forecast_in_process_only_for_an_observer(monkeypatch,
                                                              tmp_path):
    """The subprocess default is what keeps a CUDA failure in one stage."""

    import woof.go_cli as go_cli

    plan = {"runner": "woof.prepared_single_domain_forecast",
            "source": "gfs", "prepared": tmp_path / "prep",
            "authority": tmp_path / "auth", "run": tmp_path / "run",
            "profile": None}
    digests = {"proof": "a", "source_manifest": "b",
               "prepared_content": "c"}
    spawned = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: spawned.append(label))

    go_cli._run_forecast(plan, digests, explain=False, observer=None)
    assert spawned == ["forecast"]

    # With an observer it imports the runner and calls main(argv,
    # observer=...) instead -- same command, no second argument set.
    seen = {}

    class Runner:
        @staticmethod
        def main(argv, *, observer):
            seen["argv"] = argv
            seen["observer"] = observer
            return 0

    monkeypatch.setattr("importlib.import_module", lambda name: Runner)
    sentinel = object()
    go_cli._run_forecast(plan, digests, explain=False, observer=sentinel)
    assert spawned == ["forecast"]          # nothing more was spawned
    assert seen["observer"] is sentinel
    assert seen["argv"][:2] == ["--source", "gfs"]
    assert "--proof-sha256" in seen["argv"]

    # ... and an observer that says it does NOT host keeps the
    # subprocess.  `woof go` now always carries a stage-event
    # observer, and telemetry must not be what makes a bare chain give
    # up the process isolation this test exists to protect.
    class Telemetry:
        hosts_forecast = False

    go_cli._run_forecast(plan, digests, explain=False, observer=Telemetry())
    assert spawned == ["forecast", "forecast"]
    assert seen["observer"] is sentinel      # the host was not re-entered


def test_go_notifications_are_off_without_an_observer():
    """Every existing woof go caller must be unaffected."""

    from woof.go_cli import _notify

    # No observer: nothing happens, and nothing raises.
    _notify(None, "stage_begin", label="x")

    class Hostile:
        def stage_begin(self, **_):
            raise RuntimeError("observer exploded")

    # A hostile observer cannot take a stage down either.
    _notify(Hostile(), "stage_begin", label="x")


@pytest.mark.parametrize("runner", ["woof.prepared_single_domain_forecast",
                                    "woof.prepared_domain_tree_forecast"])
def test_both_runner_mains_accept_an_observer(runner):
    import importlib
    import inspect

    module = importlib.import_module(runner)
    parameter = inspect.signature(module.main).parameters["observer"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


# NEEDS CUPY INSTALLED, and opens no device.  The eighteen tests marked
# below run the real `woof go` / `woof run-plan` chain with its stages
# stubbed; without the array library the door refuses before the first
# stage (`this command needs cupy ... Refusing here, before the fetch
# stage`) and returns 2, so the events, the handoffs and the exit codes
# these tests hold are never produced.  Measured on the Linux release
# node: every one red in a venv without cupy, every one green in the
# same tree with cupy-cuda13x installed (proof/node-reds-276).
@requires_cupy
def test_a_passing_prepared_run_ends_with_completed_not_failed(
        tmp_path, monkeypatch):
    """The severity-one regression: every good run announced failure.

    The route reported ``completed_seconds: None``; that reached
    ``heartbeat.complete``, whose ``float()`` raised INSIDE the arm that
    emits ``failed`` -- after the chain had already printed its validity
    PASS.  A consumer that trusts the contract marked every good run
    failed.
    """

    import woof.go_cli as go_cli

    chain = tmp_path / "run" / "chain"
    forecast = chain / "run"

    def fake_go_main(args, *, observer=None, **_):
        forecast.mkdir(parents=True, exist_ok=True)
        # The real filename function: WRF spells the valid time with
        # colons, which Windows will not accept in a path, so the
        # product substitutes underscores and a fixture must not
        # invent its own spelling.
        from datetime import datetime as _dt

        from woof.io.wrfout import wrfout_filename
        (forecast / wrfout_filename(_dt(2024, 5, 3, 12), 1)).write_bytes(
            b"f")
        (forecast / "progress.json").write_text(json.dumps({
            "schema": "gpuwm-prepared-single-domain-progress-v1",
            "status": "PASS", "model_elapsed_seconds": 21600.0,
            "frame_count": 13}), encoding="utf-8")
        (forecast / "report.json").write_text(
            json.dumps({"status": "PASS"}), encoding="utf-8")
        return 0

    monkeypatch.setattr(go_cli, "go_main", fake_go_main)
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        code = execute_plan(plan, events=events)

    events = read_events(run_dir / EVENTS_FILENAME)
    assert code == 0, [r for r in events if r["event"] == "failed"]
    assert events[-1]["event"] == "completed"
    summary = events[-1]["summary"]
    assert summary["completed_seconds"] == 21600.0
    assert summary["status"] == "PASS"
    assert summary["nan_free"] is True
    assert summary["wrfout_count"] == 1


def test_a_summary_with_no_completion_number_cannot_crash_the_boundary():
    """Defence in depth for the same class of bug on a future route."""

    from woof.runplan import _finite_seconds

    assert _finite_seconds(None) == 0.0
    assert _finite_seconds("21600") == 0.0
    assert _finite_seconds(True) == 0.0
    assert _finite_seconds(float("nan")) == 0.0
    assert _finite_seconds(float("inf")) == 0.0
    assert _finite_seconds(-5.0) == 0.0
    assert _finite_seconds(21600) == 21600.0


def test_every_intent_key_declares_how_it_reaches_the_chain():
    """The audit, made permanent.

    geog_root, data_dir, forcing and vtable were all accepted at intent
    level and then dropped on the prepared route -- the wizard writes
    the last three into [case_data], which it does not emit for gfs.  A
    plan naming a non-default geography tree ran against the default
    one, silently.
    """

    from woof.runplan import _INTENT_DELIVERY, _INTENT_FLAGS

    assert set(_INTENT_FLAGS) == set(_INTENT_DELIVERY)
    for key, delivery in _INTENT_DELIVERY.items():
        assert delivery in ("config", "case_data") or \
            delivery.startswith("go:"), (key, delivery)


@requires_cupy
def test_go_delivered_intent_keys_are_forwarded_to_the_chain(
        tmp_path, monkeypatch):
    import woof.go_cli as go_cli

    seen = {}

    def fake_go_main(args, *, observer=None, **_):
        seen["geog_root"] = getattr(args, "geog_root", None)
        seen["data_dir"] = getattr(args, "data_dir", None)
        return 1          # stop before the summary; forwarding is the point

    monkeypatch.setattr(go_cli, "go_main", fake_go_main)
    geog = tmp_path / "MY_GEOG"
    geog.mkdir()
    data = tmp_path / "MY_DATA"
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run",
                                    geog_root=str(geog),
                                    data_dir=str(data)))
    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)

    assert Path(seen["geog_root"]) == geog
    assert Path(seen["data_dir"]) == data


@requires_cupy
def test_a_run_option_beats_the_intents_copy_of_the_same_key(
        tmp_path, monkeypatch):
    import woof.go_cli as go_cli

    seen = {}
    monkeypatch.setattr(go_cli, "go_main",
                        lambda args, *, observer=None, **_: seen.update(
                            geog_root=getattr(args, "geog_root", None)) or 1)
    intent_geog = tmp_path / "FROM_INTENT"
    intent_geog.mkdir()
    option_geog = tmp_path / "FROM_OPTION"
    option_geog.mkdir()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "p", "route": "prepared",
        "config": {"intent": {**_GFS_INTENT,
                              "geog_root": str(intent_geog)}},
        "run_options": {"geog_root": str(option_geog)},
        "output_root": str(tmp_path / "run"),
    }), encoding="utf-8")
    plan = load_plan(path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)
    assert Path(seen["geog_root"]) == option_geog


def test_an_intent_key_the_prepared_route_cannot_deliver_is_refused(
        tmp_path):
    """Accepted-then-dropped is the failure mode; refuse instead."""

    for key, value in (("forcing", ["a.grib"]), ("vtable", "V.tbl")):
        with pytest.raises(PlanError) as refusal:
            build_plan({"schema": PLAN_SCHEMA, "name": "x",
                        "route": "prepared",
                        "config": {"intent": {**_GFS_INTENT, key: value}}},
                       source="probe.json", base_dir=tmp_path,
                       sha256="0" * 64)
        text = str(refusal.value)
        assert key in text
        assert "silently dropped" in text


# ---------------------------------------------------------------------------
# HRRR on the prepared route
# ---------------------------------------------------------------------------


_HRRR_INTENT = {"point": "35.2,-97.4", "source": "hrrr", "root_dx_km": 3,
                "cycle": "2024-05-03T12", "hours": 6, "vram_gib": 24}


def _hrrr_plan(tmp_path, run_dir, **overrides):
    intent = {**_HRRR_INTENT, **overrides.pop("intent", {})}
    document = {
        "schema": PLAN_SCHEMA, "name": "hrrr-plan", "route": "prepared",
        "config": {"intent": intent}, "output_root": str(run_dir),
        **overrides,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_an_hrrr_intent_resolves_through_the_wizard(tmp_path):
    plan = load_plan(_hrrr_plan(tmp_path, tmp_path / "run"))
    resolution, exp, data = resolve_plan(plan, require_inputs=False)

    assert data is None
    assert len(exp.domains) == 1
    assert exp.domains[0].run.dx == 3000.0
    assert 'source = "hrrr"' in resolution["generated_config"]


def test_an_hrrr_intent_estimates(tmp_path, capsys):
    from woof.cli import build_parser

    plan_path = _hrrr_plan(tmp_path, tmp_path / "run")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["vram"]["estimate_bytes"] > 0
    assert document["disk"]["total_frames"] > 0


def test_the_wizard_writes_every_file_the_hrrr_route_reads(tmp_path):
    """Four files beside the config; the HRRR tools read them, not the TOML."""

    from woof.hrrr_route_inputs import route_input_paths
    from woof.runplan import generate_intent_config

    plan = load_plan(_hrrr_plan(tmp_path, tmp_path / "run"))
    generated, _ = generate_intent_config(plan, destination=tmp_path / "gen")
    for role, path in route_input_paths(generated).items():
        assert path.is_file(), role


def test_an_hrrr_intent_resolves_a_p3_suite_through_the_wizard(tmp_path):
    """mp=50 rides the wizard-emitted-config leg like any other suite.

    The reachability proof for the P3 door on the plan route, no network
    and no GPU: the intent names the registered P3 profile
    (p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1), the wizard emits the config,
    and the same emission writes every file the HRRR route reads -- so
    the route-inputs physics gate (which used to refuse mp_physics=50 by
    prose) has demonstrably admitted it.
    """

    from woof.hrrr_route_inputs import route_input_paths
    from woof.runplan import generate_intent_config

    p3_profile = "p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1"
    plan = load_plan(_hrrr_plan(
        tmp_path, tmp_path / "run",
        intent={"physics_profile": p3_profile}))
    resolution, exp, data = resolve_plan(plan, require_inputs=False)
    assert data is None
    assert exp.domains[0].run.mp_physics == 50
    assert (exp.domains[0].run.ra_lw_physics,
            exp.domains[0].run.ra_sw_physics) == (4, 4)
    assert exp.domains[0].run.ra_rrtmg_variant == "rrtmg_legacy"

    generated, _ = generate_intent_config(plan, destination=tmp_path / "gen")
    for role, path in route_input_paths(generated).items():
        assert path.is_file(), role


def test_the_prepared_route_now_drives_a_multi_domain_hrrr_plan(tmp_path):
    """This asserted a refusal until the tree chain was wired.

    It named the two things it could not drive -- the hierarchy stage
    and the tree runner -- so the accurate replacement is that both are
    now on the path.  The chain itself is covered end to end in
    tests/test_runplan_hrrr_tree.py; this is the negative-control side:
    a nested HRRR plan no longer stops at a sentence.
    """

    import inspect

    from woof.runplan import _hrrr_chain

    source = inspect.getsource(_hrrr_chain)
    assert "single-domain only" not in source
    # The branch is taken on domain count, after the shared root
    # preparation rather than before it.
    assert "if len(exp.domains) > 1:" in source
    assert "_hrrr_hierarchy_stage(" in source
    assert "_hrrr_tree_forecast(" in source


def test_a_source_the_prepared_route_cannot_drive_is_refused(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "prepared",
                    "config": {"intent": {**_HRRR_INTENT,
                                          "source": "era5"}}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "era5" in text and "experiment" in text


# ---------------------------------------------------------------------------
# The staged chain: packaged mapped sources on the prepared route
# ---------------------------------------------------------------------------


_STAGED_INTENT = {"point": "50.1,8.7", "source": "icon-eu",
                  "cycle": "2026-08-18T06", "hours": 3, "vram_gib": 16}


def _staged_plan(tmp_path, run_dir, **overrides):
    intent = {**_STAGED_INTENT, **overrides.pop("intent", {})}
    document = {
        "schema": PLAN_SCHEMA, "name": "staged-plan", "route": "prepared",
        "config": {"intent": intent}, "output_root": str(run_dir),
        **overrides,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_staged_source_intent_resolves_through_the_wizard(tmp_path):
    """A packaged mapped source's intent reaches the real wizard."""

    plan = load_plan(_staged_plan(tmp_path, tmp_path / "run"))
    resolution, exp, data = resolve_plan(plan, require_inputs=False)

    assert data is None
    assert 'source = "icon-eu"' in resolution["generated_config"]
    assert resolution["configuration"]["case_data"] is None


def test_a_staged_source_intent_estimates(tmp_path, capsys):
    from woof.cli import build_parser

    plan_path = _staged_plan(tmp_path, tmp_path / "run")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["vram"]["estimate_bytes"] > 0
    assert document["disk"]["total_frames"] > 0


def _executed_staged_chain(tmp_path, monkeypatch, *, prep=None,
                           forecast=None, sim=None):
    """Drive the staged chain end to end with every stage observed.

    The stages themselves are the fetch route's, rw-wps's and the
    runner's own programs, each covered by its own suite; what THIS
    chain owns -- and what these mocks pin -- is the composition: the
    fetch driven from the config's own hints, the preparation composed
    from the fetch's published handoff plus exactly the four
    caller-owned flags, and the forecast bound off the bundle.
    """

    import woof.go_cli as go_cli
    import woof.runplan as runplan_module
    import woof.stage_cli as stage_cli

    staged = []
    handoff_argv = ["--source", "icon-eu",
                    "--input-list", str(tmp_path / "inputs.txt"),
                    "--supplement", f"surface={tmp_path / 'invariant.grib2'}",
                    "--author-input-manifest", str(tmp_path / "inputs.json")]

    def fake_fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        from woof import fetch_routes
        (out / fetch_routes.PREP_ARGUMENTS_NAME).write_text(json.dumps({
            "schema": fetch_routes.PREP_ARGUMENTS_SCHEMA,
            "source": "icon-eu", "prep_source": "icon-eu",
            "argv": handoff_argv, "unbound_supplement_roles": [],
            "member": None, "member_set": None,
        }), encoding="utf-8")
        staged.append(("fetch", list(arguments)))
        return {}

    def fake_prep(arguments):
        staged.append(("prepare", [str(a) for a in arguments]))
        prep_root = Path(arguments[arguments.index("--output-root") + 1])
        prep_root.mkdir(parents=True, exist_ok=True)
        (prep_root / "proof.json").write_text("{}", encoding="utf-8")

    def fake_resolve_bundle(prepared_root):
        return {"document": Path(prepared_root) / "proof.json",
                "schema": "probe", "source": "icon-eu",
                "layout": "single", "domains": 1, "payload": {}}

    def fake_sim_command(bundle, **kw):
        staged.append(("sim_command", dict(kw)))
        return ["python", "-m", "runner", "--source", bundle["source"],
                "--outdir", str(kw["outdir"])]

    def fake_forecast(argv, *, layout, observer):
        staged.append(("forecast", list(argv), layout))

    monkeypatch.setattr(runplan_module, "_run_fetch", fake_fetch)
    monkeypatch.setattr(runplan_module, "_run_prep", prep or fake_prep)
    monkeypatch.setattr(runplan_module, "_staged_forecast",
                        forecast or fake_forecast)
    monkeypatch.setattr(stage_cli, "resolve_bundle", fake_resolve_bundle)
    monkeypatch.setattr(stage_cli, "sim_command", sim or fake_sim_command)
    monkeypatch.setattr(
        go_cli, "_render_stage",
        lambda plan, **kw: staged.append(("render", dict(plan))))

    geog = tmp_path / "GEOG"
    geog.mkdir(exist_ok=True)
    plan = load_plan(_staged_plan(
        tmp_path, tmp_path / "run",
        run_options={"geog_root": str(geog)}))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)
    return (staged, handoff_argv,
            read_events(plan.run_dir / EVENTS_FILENAME), plan)


@requires_cupy
def test_the_staged_chain_composes_prep_from_the_fetch_handoff(
        tmp_path, monkeypatch):
    """The preparation argv IS the fetch route's published binding, plus
    exactly the four flags the handoff declares are the caller's.  The
    one token it replaces is the authored input manifest, which the chain
    writes beside its own preparation so the fetch folder's manifest stays
    the standalone prep command's."""

    staged, handoff_argv, _events, plan = _executed_staged_chain(
        tmp_path, monkeypatch)

    fetch = next(cmd for label, *cmd in staged if label == "fetch")[0]
    assert "--source" in fetch and "icon-eu" in fetch
    assert "--cycle" in fetch and "2026-08-18T06" in fetch

    prepare = next(cmd for label, *cmd in staged if label == "prepare")[0]
    expected = list(handoff_argv)
    expected[expected.index("--author-input-manifest") + 1] = str(
        plan.run_dir / "chain" / "inputs.json")
    assert prepare[:len(handoff_argv)] == expected
    appended = prepare[len(handoff_argv):]
    assert appended[::2] == ["--wps-namelist", "--experiment-config",
                             "--geog-root", "--output-root"]
    namelist = Path(appended[1])
    assert namelist.name.endswith(".namelist.wps") and namelist.is_file()
    config = Path(appended[3])
    assert config.name == "intent-config.toml" and config.is_file()
    assert appended[5] == str(tmp_path / "GEOG")
    assert Path(appended[7]) == plan.run_dir / "chain" / "prep"


@requires_cupy
def test_the_staged_chain_binds_the_forecast_off_the_bundle(
        tmp_path, monkeypatch):
    staged, _argv, events, plan = _executed_staged_chain(
        tmp_path, monkeypatch)

    sim = next(cmd for label, *cmd in staged if label == "sim_command")[0]
    assert Path(str(sim["experiment_config"])).name == "intent-config.toml"
    assert Path(str(sim["wps_namelist"])).name.endswith(".namelist.wps")
    assert Path(str(sim["outdir"])) == plan.run_dir / "chain" / "run"

    forecast = next(entry for entry in staged if entry[0] == "forecast")
    assert forecast[2] == "single"
    # The runner argv is the composed command minus the interpreter
    # prefix, exactly as `woof sim` itself strips it.
    assert forecast[1][0] == "--source"

    stages = [record["stage"] for record in events
              if record["event"] == "stage_started"]
    assert stages == ["fetch", "prepare", "forecast", "finalize"]
    render = next(entry for entry in staged if entry[0] == "render")
    assert Path(str(render[1]["render"])) == plan.run_dir / "chain" / "png"


@requires_cupy
def test_the_staged_chain_starts_the_forecast_at_the_prepared_head(
        tmp_path, monkeypatch):
    """Chained preparation on the staged route: the forecast is bound to the
    prepared HEAD and starts while the preparation is still running; the
    seal is reported after it."""

    import threading
    from datetime import datetime, timezone

    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")

    import woof.runplan as runplan_module
    import woof.stage_cli as stage_cli
    from woof.ingest.boundary_stream import HEAD_SCHEMA, head_sha256

    started = threading.Event()
    bound = {}

    def chained_prep(arguments):
        prep_root = Path(arguments[arguments.index("--output-root") + 1])
        bound["root"] = prep_root
        stream = prep_root / "boundary-stream"
        stream.mkdir(parents=True)
        head = {"schema": HEAD_SCHEMA,
                "basis": {"schema": HEAD_SCHEMA, "cache": {},
                          "proof_head": {"schema": "probe"},
                          "input_manifest_sha256": "0" * 64},
                "decision": {"chained": True},
                # A chain binds only a head written after it started.
                "created_utc": datetime.now(timezone.utc).isoformat()}
        head["head_sha256"] = head_sha256(head)
        bound["head"] = head["head_sha256"]
        (stream / "head.json").write_text(json.dumps(head), encoding="utf-8")
        assert started.wait(10), "the forecast did not start at the head"
        (prep_root / "proof.json").write_text("{}", encoding="utf-8")

    def head_bundle(prepared_root, head):
        return {"document": Path(prepared_root) / "proof.json",
                "root": Path(prepared_root), "schema": "probe",
                "source": "icon-eu", "layout": "single", "domains": 1,
                "payload": {}, "head_sha256": head,
                "source_manifest_sha256": "0" * 64}

    def recording_sim(bundle, **kw):
        bound["sim_head"] = bundle.get("head_sha256")
        return ["python", "-m", "runner", "--outdir", str(kw["outdir"])]

    def forecast(argv, *, layout, observer):
        bound["sealed_at_start"] = (
            Path(bound["root"]) / "proof.json").exists()
        started.set()
        # The forecast outlives the preparation, as it does on a real run:
        # the seal is reported when it happens, not when the forecast ends.
        import time
        deadline = time.monotonic() + 10
        while not (Path(bound["root"]) / "proof.json").exists():
            assert time.monotonic() < deadline, "the preparation never sealed"
            time.sleep(0.01)
        time.sleep(0.5)

    monkeypatch.setattr(stage_cli, "resolve_head_bundle", head_bundle)
    _staged, _argv, events, _plan = _executed_staged_chain(
        tmp_path, monkeypatch, prep=chained_prep, forecast=forecast,
        sim=recording_sim)
    assert bound["sim_head"] == bound["head"]
    assert bound["sealed_at_start"] is False
    names = [record["event"] for record in events]
    assert names.index("prepare_head_ready") < names.index("prepare_sealed")
    assert names.count("prepare_sealed") == 1
    sealed_ms = next(record["emitted_unix_ms"] for record in events
                     if record["event"] == "prepare_sealed")
    forecast_end_ms = next(record["emitted_unix_ms"] for record in events
                           if record["event"] == "stage_finished"
                           and record["stage"] == "forecast")
    assert forecast_end_ms - sealed_ms >= 300
    stages = [record["stage"] for record in events
              if record["event"] == "stage_started"]
    assert stages == ["fetch", "prepare", "forecast", "finalize"]


@requires_cupy
def test_the_staged_chain_says_the_seal_before_the_stage_bookkeeping_ends(
        tmp_path, monkeypatch):
    """``prepare_sealed`` lands when the preparer seals, not when the stage's
    reuse binding (git provenance and argument digests) is written: here that
    binding cannot finish until the forecast has ended, and the forecast must
    still see the seal on the stream while it runs."""

    import threading
    import time
    from datetime import datetime, timezone

    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")

    import woof.stage_cli as stage_cli
    import woof.stage_reuse as stage_reuse
    from woof.ingest.boundary_stream import HEAD_SCHEMA, head_sha256

    started = threading.Event()
    forecast_done = threading.Event()
    bound = {}

    def chained_prep(arguments):
        prep_root = Path(arguments[arguments.index("--output-root") + 1])
        bound["root"] = prep_root
        stream = prep_root / "boundary-stream"
        stream.mkdir(parents=True)
        head = {"schema": HEAD_SCHEMA,
                "basis": {"schema": HEAD_SCHEMA, "cache": {},
                          "proof_head": {"schema": "probe"},
                          "input_manifest_sha256": "0" * 64},
                "decision": {"chained": True},
                "created_utc": datetime.now(timezone.utc).isoformat()}
        head["head_sha256"] = head_sha256(head)
        (stream / "head.json").write_text(json.dumps(head), encoding="utf-8")
        assert started.wait(10), "the forecast did not start at the head"
        (prep_root / "proof.json").write_text("{}", encoding="utf-8")

    real_binding = stage_reuse.write_binding

    def slow_binding(*args, **kwargs):
        assert forecast_done.wait(20), "the forecast never ended"
        return real_binding(*args, **kwargs)

    def head_bundle(prepared_root, head):
        return {"document": Path(prepared_root) / "proof.json",
                "root": Path(prepared_root), "schema": "probe",
                "source": "icon-eu", "layout": "single", "domains": 1,
                "payload": {}, "head_sha256": head,
                "source_manifest_sha256": "0" * 64}

    def forecast(argv, *, layout, observer):
        started.set()
        deadline = time.monotonic() + 10
        seen = False
        while not seen and time.monotonic() < deadline:
            seen = any(record["event"] == "prepare_sealed"
                       for record in read_events(observer.events.path,
                                                 allow_partial_tail=True))
            time.sleep(0.01)
        bound["seal_seen_while_forecasting"] = seen
        forecast_done.set()

    monkeypatch.setattr(stage_reuse, "write_binding", slow_binding)
    monkeypatch.setattr(stage_cli, "resolve_head_bundle", head_bundle)
    _staged, _argv, events, _plan = _executed_staged_chain(
        tmp_path, monkeypatch, prep=chained_prep, forecast=forecast,
        sim=lambda bundle, **kw: ["python", "-m", "runner",
                                  "--outdir", str(kw["outdir"])])
    assert bound["seal_seen_while_forecasting"] is True
    names = [record["event"] for record in events]
    assert names.index("prepare_head_ready") < names.index("prepare_sealed")
    assert names.count("prepare_sealed") == 1
    assert (bound["root"] / stage_reuse.BINDING_NAME).is_file()


@requires_cupy
def test_a_failed_staged_forecast_stops_its_preparation_at_once(
        tmp_path, monkeypatch):
    """The failure is said at once and the preparation is asked to stop
    (``stop.json``); the run never sits in ``waiting:preparation`` while the
    producer builds every later forcing time (test host run 2a, test host r1b/r1d,
    2026-10-21: a failed forecast hid behind 19-24 min of that)."""

    import threading
    import time
    from datetime import datetime, timezone

    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")

    import woof.stage_cli as stage_cli
    from woof.ingest.boundary_stream import HEAD_SCHEMA, STOP_NAME, head_sha256
    from woof.supervisor import HEARTBEAT_NAME, read_heartbeat

    started = threading.Event()
    seen = {"statuses": set()}

    def chained_prep(arguments):
        prep_root = Path(arguments[arguments.index("--output-root") + 1])
        stream = prep_root / "boundary-stream"
        stream.mkdir(parents=True)
        head = {"schema": HEAD_SCHEMA,
                "basis": {"schema": HEAD_SCHEMA, "cache": {},
                          "proof_head": {"schema": "probe"},
                          "input_manifest_sha256": "0" * 64},
                "decision": {"chained": True},
                "created_utc": datetime.now(timezone.utc).isoformat()}
        head["head_sha256"] = head_sha256(head)
        (stream / "head.json").write_text(json.dumps(head), encoding="utf-8")
        assert started.wait(10), "the forecast did not start at the head"
        progress = prep_root.parent.parent / HEARTBEAT_NAME
        deadline = time.monotonic() + 10
        # A producer reads stop.json between forcing times; this one polls it.
        while not (stream / STOP_NAME).is_file() and time.monotonic() < deadline:
            try:
                beat = read_heartbeat(progress)
            except (OSError, ValueError):
                beat = None
            if beat is not None:
                seen["statuses"].add(beat.status)
            time.sleep(0.01)
        seen["stop"] = json.loads((stream / STOP_NAME).read_text(encoding="utf-8"))
        raise RuntimeError("stopped by its forecast")

    def head_bundle(prepared_root, head):
        return {"document": Path(prepared_root) / "proof.json",
                "root": Path(prepared_root), "schema": "probe",
                "source": "icon-eu", "layout": "single", "domains": 1,
                "payload": {}, "head_sha256": head,
                "source_manifest_sha256": "0" * 64}

    def forecast(argv, *, layout, observer):
        started.set()
        raise RuntimeError("the model went unstable")

    monkeypatch.setattr(stage_cli, "resolve_head_bundle", head_bundle)
    _staged, _argv, events, plan = _executed_staged_chain(
        tmp_path, monkeypatch, prep=chained_prep, forecast=forecast,
        sim=lambda bundle, **kw: ["python", "-m", "runner",
                                  "--outdir", str(kw["outdir"])])
    assert seen["stop"]["reason"].startswith("the forecast failed: RuntimeError")
    assert "waiting:preparation" not in seen["statuses"]
    failed = next(record for record in events if record["event"] == "failed")
    assert "unstable" in failed["message"]
    assert read_heartbeat(plan.run_dir / HEARTBEAT_NAME).status == "failed"


# NEEDS CUPY INSTALLED, and opens no device: this test asserts the staged
# chain's own refusal names prep-arguments.json; without cupy the run-plan
# door refuses first and names the missing wheel instead.
@requires_cupy
def test_the_staged_chain_refuses_a_fetch_with_no_handoff(
        tmp_path, monkeypatch):
    """A legacy-shaped fetch directory cannot feed the staged prep."""

    import woof.runplan as runplan_module

    def bare_fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(runplan_module, "_run_fetch", bare_fetch)
    geog = tmp_path / "GEOG"
    geog.mkdir(exist_ok=True)
    plan = load_plan(_staged_plan(
        tmp_path, tmp_path / "run",
        run_options={"geog_root": str(geog)}))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        assert execute_plan(plan, events=events) == 1
    failed = next(record for record
                  in read_events(plan.run_dir / EVENTS_FILENAME)
                  if record["event"] == "failed")
    assert "prep-arguments.json" in failed["message"]


def _staged_hrrr_chain(tmp_path, monkeypatch, **plan_overrides):
    """Assemble the HRRR chain without executing any stage."""

    import woof.go_cli as go_cli
    import woof.runplan as runplan_module

    staged = []
    monkeypatch.setattr(
        go_cli, "run_stage",
        lambda label, command, **kw: staged.append((label, list(command))))

    def fake_fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text(_FETCHED_SUMS, encoding="utf-8")
        staged.append(("fetch", list(arguments)))
        return {}

    monkeypatch.setattr(runplan_module, "_run_fetch", fake_fetch)
    geog = tmp_path / "GEOG"
    geog.mkdir(exist_ok=True)
    plan_overrides.setdefault("run_options", {"geog_root": str(geog)})
    plan = load_plan(_hrrr_plan(tmp_path, tmp_path / "run",
                                **plan_overrides))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)
    return staged, read_events(plan.run_dir / EVENTS_FILENAME)


@requires_cupy
def test_the_hrrr_chain_passes_the_wps_namelist_to_the_preparer(
        tmp_path, monkeypatch):
    """The whole point of this increment.

    The runner's HRRR manifest role inventory requires a wps_namelist
    role, and the preparer only records it if handed the file.  The
    wizard's printed chain never passed it, so the bundle could not be
    read by the single-domain runner at all.
    """

    staged, _events = _staged_hrrr_chain(tmp_path, monkeypatch)
    prepare = next(cmd for label, cmd in staged if label == "prepare")

    assert "--wps-namelist" in prepare
    namelist = Path(prepare[prepare.index("--wps-namelist") + 1])
    assert namelist.name.endswith(".namelist.wps")
    assert namelist.is_file()


@requires_cupy
def test_the_hrrr_prepare_command_is_the_wizards_printed_one(
        tmp_path, monkeypatch):
    """Every flag the documented chain passes, and the cycle it spells."""

    staged, _events = _staged_hrrr_chain(tmp_path, monkeypatch)
    prepare = next(cmd for label, cmd in staged if label == "prepare")

    assert prepare[1:3] == ["-m", "tools.prepare_hrrr_wrf"]
    for flag in ("--source-root", "--source-manifest",
                 "--source-manifest-sha256", "--domain-spec",
                 "--namelist-input", "--geog-root", "--cycle",
                 "--run-seconds", "--history-interval-seconds",
                 "--skip-stock-wrf-export", "--output-root"):
        assert flag in prepare, flag
    # [fetch] spells the cycle YYYY-MM-DDTHH; the preparer takes
    # YYYY-MM-DD_HH:MM:SS.  Converted, not sliced.
    assert prepare[prepare.index("--cycle") + 1] == "2024-05-03_12:00:00"
    assert prepare[prepare.index("--run-seconds") + 1] == "21600"
    # The digest is the real one, of the file the fetch actually wrote.
    import hashlib

    manifest = Path(prepare[prepare.index("--source-manifest") + 1])
    assert prepare[prepare.index("--source-manifest-sha256") + 1] == \
        hashlib.sha256(manifest.read_bytes()).hexdigest()


@requires_cupy
def test_the_hrrr_fetch_argv_comes_from_the_configs_own_fetch_hints(
        tmp_path, monkeypatch):
    staged, _events = _staged_hrrr_chain(tmp_path, monkeypatch)
    fetch = next(cmd for label, cmd in staged if label == "fetch")

    assert fetch[fetch.index("--source") + 1] == "hrrr"
    assert fetch[fetch.index("--cycle") + 1] == "2024-05-03T12"
    assert fetch[fetch.index("--hours") + 1] == "6"
    assert "--area" in fetch          # the wizard sized it
    # And it is a valid `woof fetch` argv, by that parser's own reckoning.
    from woof.cli import build_parser

    build_parser().parse_args(["fetch", *fetch])


@requires_cupy
def test_the_hrrr_chain_emits_the_stages_in_the_documented_order(
        tmp_path, monkeypatch):
    _staged, events = _staged_hrrr_chain(tmp_path, monkeypatch)
    started = [r["stage"] for r in events if r["event"] == "stage_started"]
    assert started[:2] == ["fetch", "prepare"]


@requires_cupy
def test_the_physics_profile_is_passed_only_when_the_plan_states_it(
        tmp_path, monkeypatch):
    """The route owns its physics gate; this layer must not invent one."""

    staged, _ = _staged_hrrr_chain(tmp_path, monkeypatch)
    prepare = next(cmd for label, cmd in staged if label == "prepare")
    assert "--physics-profile" not in prepare

    second = tmp_path / "stated"
    second.mkdir()
    staged, _ = _staged_hrrr_chain(
        second, monkeypatch,
        run_options={"geog_root": str(second / "GEOG"),
                     "physics_profile": "wsm6-ysu-mm5-noah-no-radiation-v1"})
    prepare = next(cmd for label, cmd in staged if label == "prepare")
    assert prepare[prepare.index("--physics-profile") + 1] == \
        "wsm6-ysu-mm5-noah-no-radiation-v1"


# ---------------------------------------------------------------------------
# Selective rendering
# ---------------------------------------------------------------------------


def _go_plan(tmp_path, **extra):
    return {"run": tmp_path / "run", "render": tmp_path / "png", **extra}


def test_render_products_reaches_the_render_command_verbatim(tmp_path):
    """The renderer owns the vocabulary; this passes the spec through."""

    from woof.go_cli import render_command

    frames = [tmp_path / "wrfout_d01_0001"]
    plain = render_command(_go_plan(tmp_path), frames)
    assert "--products" not in plain          # default set, unchanged

    filtered = render_command(
        _go_plan(tmp_path, render_products="sbcape,srh_0_1km"), frames)
    assert filtered[filtered.index("--products") + 1] == "sbcape,srh_0_1km"
    # Everything else about the command is identical.
    assert filtered[:len(plain)] == plain


def test_render_products_none_skips_the_stage_without_running_it(
        tmp_path, monkeypatch, capsys):
    import woof.go_cli as go_cli

    ran = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append(label))

    assert go_cli._render_stage(
        _go_plan(tmp_path, render_products="none"), explain=False) is False
    assert ran == []
    assert "render skipped" in capsys.readouterr().out

    # Spelled any way a person would spell it.
    assert go_cli._render_stage(
        _go_plan(tmp_path, render_products=" NONE "), explain=False) is False
    assert ran == []


def test_the_default_render_stage_is_untouched_when_no_filter_is_given(
        tmp_path, monkeypatch):
    """`woof go` itself sets nothing, so its behaviour cannot move."""

    import woof.go_cli as go_cli

    ran = []
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "wrfout_frames",
                        lambda plan: [tmp_path / "wrfout_d01_0001"])
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: ran.append((label, list(command))))

    assert go_cli._render_stage(_go_plan(tmp_path), explain=False) is True
    assert ran[0][0] == "render"
    assert "--products" not in ran[0][1]


def test_render_products_is_a_run_option_on_the_prepared_route(tmp_path):
    from woof.runplan import ROUTES

    assert "render_products" in ROUTES["prepared"].run_options
    # Not an intent key: intent mirrors `woof domain` flags one for
    # one, and the wizard writes configs, not pictures.
    from woof.runplan import _INTENT_FLAGS

    assert "render_products" not in _INTENT_FLAGS

    plan = load_plan(_prepared_plan(
        tmp_path, tmp_path / "run",
        ))
    assert plan.run_options["render_products"] is None


@requires_cupy
def test_the_run_option_is_stamped_onto_the_namespace_go_reads(
        tmp_path, monkeypatch):
    import woof.go_cli as go_cli

    seen = {}
    monkeypatch.setattr(
        go_cli, "go_main",
        lambda args, *, observer=None, **_: seen.update(
            products=getattr(args, "render_products", None)) or 1)

    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "p", "route": "prepared",
        "config": {"intent": dict(_GFS_INTENT)},
        "run_options": {"render_products": "refl,t2"},
        "output_root": str(tmp_path / "run"),
    }), encoding="utf-8")
    plan = load_plan(path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)
    assert seen["products"] == "refl,t2"


def test_both_chains_honour_the_same_render_filter(tmp_path, monkeypatch):
    """The HRRR chain renders too, so the option cannot mean two things."""

    import woof.go_cli as go_cli
    import woof.runplan as runplan_module

    rendered = []
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "wrfout_frames",
                        lambda plan: [tmp_path / "wrfout_d01_0001"])
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: rendered.append(list(command)))

    # The HRRR arm builds the same render plan go does, from its own
    # forecast directory.
    stage = runplan_module._GoObserver
    assert stage is not None
    go_cli._render_stage(
        {"run": tmp_path / "run", "render": tmp_path / "png",
         "render_products": "composite_reflectivity"},
        explain=False)
    assert rendered[0][rendered[0].index("--products") + 1] ==         "composite_reflectivity"


def test_the_catalog_query_returns_the_renderers_real_list(capsys):
    """Asked, never transcribed -- and the parse is checked."""

    from woof.cli import build_parser
    from woof.runplan import CATALOG_SCHEMA

    assert run_plan_main(
        build_parser().parse_args(["run-plan", "--catalog"])) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["schema"] == CATALOG_SCHEMA
    assert document["skip_token"] == "none"
    if document["engine"] is None:
        # No usable rw_wrfbatch on this box.  `auto` refuses rather than
        # answering from a second engine's catalog (render law, audit
        # F7), and the document carries the refusal so a picker can say
        # what is wrong instead of offering products that do not exist.
        assert document["products"] is None
        assert "fetch-bridges" in document["error"]
        return
    assert document["engine"] == "rust"
    products = document["products"]
    assert products and all(entry["name"] for entry in products)
    # No header or footer line leaked in as a product.
    names = [entry["name"] for entry in products]
    assert not any(" " in name for name in names)
    assert not any(name.startswith(("group keywords", "selectable_slugs"))
                   for name in names)
    # A disagreement with the renderer's own count is reported, not hidden.
    assert "parse_warning" not in document


def test_the_catalog_needs_no_plan_and_writes_nothing(tmp_path):
    from woof.runplan import render_catalog

    document = render_catalog()
    assert document["spec"]
    assert list(tmp_path.iterdir()) == []


def test_a_box_with_no_renderer_answers_with_the_refusal(monkeypatch):
    """A named "no answer" beats an answer from a different catalog.

    This used to hand the picker the matplotlib engine's five products.
    They are not the products ``woof render`` would draw on that box --
    ``--engine auto`` refuses there now, because weather fields come
    from ``rw_wrfbatch`` (render law, 2026-08-06; audit
    F7) -- so offering them was offering a menu nothing serves.  The
    document says ``engine: null``, ``products: null`` and why.
    """

    from woof import rustwx
    from woof.runplan import render_catalog

    monkeypatch.setattr(rustwx, "find_renderer", lambda: None)
    document = render_catalog()
    assert document["engine"] is None
    assert document["products"] is None
    assert "rw_wrfbatch" in document["error"]
    assert "fetch-bridges" in document["error"]
    # The refusal reaches the JSON as ONE readable block, not with the
    # `[[explain]]` sentinel a terminal layer is supposed to strip.
    assert "[[explain]]" not in document["error"]


def _sources_document(capsys):
    """Run the real front door and return its one printed document."""

    from woof.cli import build_parser

    assert run_plan_main(
        build_parser().parse_args(["run-plan", "--sources"])) == 0
    captured = capsys.readouterr().out
    # stdout is the machine channel and carries the document alone: the
    # registry imports print, and the redirect is what keeps their
    # chatter off the channel a consumer calls json.loads on.
    return json.loads(captured), captured


def test_the_sources_query_answers_with_the_registry_schema(capsys):
    from woof.runplan import SOURCES_SCHEMA

    document, raw = _sources_document(capsys)
    assert document["schema"] == SOURCES_SCHEMA
    assert raw.lstrip().startswith("{")
    assert raw.count("\n{\n") == 0
    assert document["registry_schema"] == "gpuwm-native-source-adapters-v1"
    assert document["gpuwm_version"]
    assert document["readiness_rule"] and document["certification_rule"]
    assert set(document["routes"]) == {"experiment", "prepared"}


def test_the_sources_query_emits_every_registered_row_in_registry_order(capsys):
    """THE arbitrary-test gate.

    The list and its order are the registry's own.  This cell fails the
    moment the reply is built from anything but ``_ADAPTERS`` -- a
    per-model dict, a curated trio, a sort someone thought looked
    tidier.
    """

    from woof.source_adapters import source_adapters

    document, _raw = _sources_document(capsys)
    registered = [adapter.source_id for adapter in source_adapters()]
    assert document["source_count"] == len(registered)
    assert [row["source_id"] for row in document["sources"]] == registered
    assert document["runnable_source_count"] == sum(
        adapter.runnable for adapter in source_adapters())


def test_a_new_registry_row_appears_with_no_code_change(capsys, monkeypatch):
    """Adding a model is table work: a row appears, nothing is edited.

    The proof the arbitrary acceptance test asks for.  A row grafted
    onto the registry -- and onto nothing else -- must reach the reply
    with its declared facts intact, in its registry position, and with
    a truthful "no fetch route / no intent route" verdict rather than a
    traceback.
    """

    import dataclasses

    from woof import source_adapters as registry

    grafted = dataclasses.replace(
        next(row for row in registry.source_adapters() if row.runnable),
        source_id="probe-arbitrary-model",
        aliases=("probe-arbitrary-alias",), display_name=None)
    monkeypatch.setattr(
        registry, "_ADAPTERS", (*registry.source_adapters(), grafted))

    document, _raw = _sources_document(capsys)
    ids = [row["source_id"] for row in document["sources"]]
    assert ids[-1] == "probe-arbitrary-model"
    assert document["source_count"] == len(ids)
    row = document["sources"][-1]
    # A row that fills no display-name column reads back as its id --
    # the fallback, never `null` and never a traceback.
    assert row["title"] == "probe-arbitrary-model"
    assert row["display_name"] == "probe-arbitrary-model"
    assert row["aliases"] == ["probe-arbitrary-alias"]
    assert row["source_kind"] == grafted.source_kind.value
    assert row["maturity"]["status"] == grafted.status.value
    assert row["fetch"]["kind"] == "none"
    assert row["fetch"]["route_id"] is None
    assert row["fetch"]["refusal"]
    # The grafted row copies a drivable row's runner but registers no
    # acquisition route under its own id, so the DERIVED verdict is a
    # truthful "no fetch route" -- not silence, not "unknown source".
    assert row["run_plan"]["intent_routes"] == []
    assert row["run_plan"]["intent_supported"] is False
    assert "no acquisition route" in row["run_plan"]["intent_refusal"]


def test_every_sources_row_declares_its_fetch_kind_from_the_route_tables(
        capsys):
    """The census is the registry's owners and the three tables', never a literal.

    A row whose registry entry names an acquisition owner (a model-table
    family that runs the model to make its bytes) is fetched by that
    owner, ahead of the three transport tables.
    """

    from woof import fetch_routes
    from woof.source_adapters import source_adapters

    document, _raw = _sources_document(capsys)
    owners = {adapter.source_id: adapter.acquisition_owner
              for adapter in source_adapters() if adapter.acquisition_owner}
    table = set(fetch_routes.route_ids())
    legacy = set(fetch_routes.LEGACY_ROUTE_SOURCES)
    refused = set(fetch_routes.refusal_ids())

    census = {"acquisition_owner": 0, "table_route": 0, "legacy_transport": 0,
              "refused": 0, "none": 0}
    for row in document["sources"]:
        source_id = row["source_id"]
        if source_id in owners:
            assert row["fetch"]["kind"] == "acquisition_owner", source_id
            assert row["fetch"]["acquisition_owner"] == owners[source_id]
            assert row["fetch"]["route_id"] is None
            census["acquisition_owner"] += 1
            continue
        if source_id in table:
            expected = "table_route"
        elif source_id in legacy:
            expected = "legacy_transport"
        elif source_id in refused:
            expected = "refused"
        else:
            expected = "none"
        assert row["fetch"]["kind"] == expected, source_id
        census[expected] += 1
        if expected == "table_route":
            assert row["fetch"]["route_id"] == source_id
            assert row["fetch"]["refusal"] is None
        else:
            assert row["fetch"]["route_id"] is None
        if expected == "legacy_transport":
            # `route_for` raises "has its own transport" here.  That is
            # not a refusal, and reporting it as one would send a reader
            # hunting for a fetch route that exists.
            assert row["fetch"]["refusal"] is None
        if expected in {"refused", "none"}:
            assert row["fetch"]["refusal"]
            assert "[[explain]]" not in row["fetch"]["refusal"]
    assert sum(census.values()) == document["source_count"]
    assert census["table_route"] == len((table - set(owners)) & set(
        row["source_id"] for row in document["sources"]))


def test_every_sources_row_carries_the_registry_display_name(capsys):
    """The reply names a source the way a person would say it.

    ``title`` used to repeat ``source_id``, so every consumer showed
    ``gem-gdps`` and ``aigefs`` and the only way to a human name was a
    per-model lookup in a front end.  The name is a registry COLUMN now
    and this cell fails the moment the reply stops copying it.
    """

    from woof.source_adapters import source_adapters

    document, _raw = _sources_document(capsys)
    rows = {row["source_id"]: row for row in document["sources"]}
    for adapter in source_adapters():
        row = rows[adapter.source_id]
        assert row["display_name"] == adapter.display_title, adapter.source_id
        # `title` stays in the reply for consumers already reading it,
        # and now carries the name rather than the id.
        assert row["title"] == adapter.display_title
        assert row["display_name"] != row["source_id"]


def test_every_sources_row_carries_its_credential_facts(capsys):
    """What a row needs configured, from the row -- not a GUI's table.

    A front end used to hardcode "this one source needs a key".  Every
    row answers now: the ones that need nothing say the table declares
    nothing, and the ones that need something name it, name where it is
    configured, and name what breaks without it.
    """

    from woof.source_adapters import source_adapters
    from woof.source_credentials import credential_facts

    document, _raw = _sources_document(capsys)
    rows = {row["source_id"]: row for row in document["sources"]}
    required = []
    for adapter in source_adapters():
        block = rows[adapter.source_id]["credentials"]
        assert block["summary"], adapter.source_id
        assert block["required"] is bool(adapter.credentials)
        assert len(block["items"]) == len(adapter.credentials)
        for item, credential in zip(block["items"], adapter.credentials):
            assert item == credential_facts(credential)
            assert item["breakage"] in item["absent_message"]
            assert item["status_message"] in (
                item["present_message"], item["absent_message"])
        if block["required"]:
            required.append(adapter.source_id)
    # The registry's own census, not a literal: whichever rows declare a
    # credential are the rows that report one.
    assert required == [adapter.source_id for adapter in source_adapters()
                        if adapter.credentials]


def test_a_new_registry_row_reaches_the_reply_with_its_name_and_credential(
        capsys, monkeypatch):
    """THE arbitrary acceptance test for both new columns.

    A source registered tomorrow -- one row, no code edited anywhere --
    comes out of the real ``--sources`` door with its human name and its
    credential facts, and the presence verdict follows the declared
    location rather than anything that knows the source's name.
    """

    import dataclasses

    from woof import source_adapters as registry
    from woof.source_credentials import CredentialLocation, SourceCredential

    credential = SourceCredential(
        credential_id="probe-arbitrary-credential",
        display_name="Probe Arbitrary API key",
        location_kind=CredentialLocation.ENV_VAR,
        location="WOOF_PROBE_ARBITRARY_KEY",
        needed_for="acquisition",
        breakage=("the acquisition step cannot authenticate and the "
                  "download is rejected by the provider"),
        obtain_url="https://example.invalid/keys")
    grafted = dataclasses.replace(
        next(row for row in registry.source_adapters() if row.runnable),
        source_id="probe-arbitrary-model",
        display_name="Probe Arbitrary Model 9000", aliases=(),
        credentials=(credential,))
    monkeypatch.setattr(
        registry, "_ADAPTERS", (*registry.source_adapters(), grafted))
    monkeypatch.delenv(credential.location, raising=False)

    document, _raw = _sources_document(capsys)
    row = document["sources"][-1]
    assert row["source_id"] == "probe-arbitrary-model"
    assert row["display_name"] == "Probe Arbitrary Model 9000"
    assert row["title"] == "Probe Arbitrary Model 9000"
    block = row["credentials"]
    assert block["required"] is True
    item, = block["items"]
    assert item["display_name"] == "Probe Arbitrary API key"
    assert item["present"] is False
    assert item["breakage"] in item["status_message"]
    assert "https://example.invalid/keys" in item["status_message"]

    monkeypatch.setenv(credential.location, "probe-secret")
    document, raw = _sources_document(capsys)
    item, = document["sources"][-1]["credentials"]["items"]
    assert item["present"] is True
    # Existence, never the value: a key must not reach a JSON front door.
    assert "probe-secret" not in raw


def test_sources_coverage_repeats_the_registry_window_or_says_global(capsys):
    from woof.source_adapters import source_adapters

    document, _raw = _sources_document(capsys)
    rows = {row["source_id"]: row for row in document["sources"]}
    for adapter in source_adapters():
        coverage = rows[adapter.source_id]["coverage"]
        window = adapter.coverage_window
        if window is None:
            assert coverage is None, adapter.source_id
            continue
        assert coverage["kind"] == type(window).__name__
        south, west, north, east = window.envelope()
        assert (coverage["south"], coverage["west"], coverage["north"],
                coverage["east"]) == (south, west, north, east)
        assert (coverage["centre_lat"], coverage["centre_lon"]) == tuple(
            window.centre())
        assert coverage["describe"] == window.describe()
        assert coverage["grid"]


def test_sources_names_the_run_plan_intent_reach_accurately(capsys):
    """Registered rows and intent-drivable ones are separate truths.

    The picker's accuracy depends on this field: the registry decodes
    more than the run-plan INTENT door can drive, and a picker that hid
    the difference would offer launches that refuse.  The field is the
    DERIVED verdict -- registry facts through
    :func:`woof.runplan.intent_drivability` -- never a hand-kept list,
    and an undrivable row carries the derived refusal naming its
    missing fact.
    """

    from woof.runplan import intent_drivability

    document, _raw = _sources_document(capsys)
    drivability = intent_drivability()
    for row in document["sources"]:
        verdict = drivability[row["source_id"]]
        assert row["run_plan"]["intent_routes"] == sorted(
            verdict["routes"]), row["source_id"]
        assert row["run_plan"]["intent_supported"] is bool(
            verdict["routes"])
        assert row["run_plan"]["intent_chain"] == verdict["chain"]
        if row["run_plan"]["intent_supported"]:
            assert row["run_plan"]["intent_refusal"] is None
        else:
            assert row["run_plan"]["intent_refusal"], row["source_id"]
            assert "unknown source" not in row["run_plan"]["intent_refusal"]
    supported = {row["source_id"] for row in document["sources"]
                 if row["run_plan"]["intent_supported"]}
    # The widening's floor: the old trio, plus the receipt-proven staged
    # source, are all drivable.  The exact set is the derivation's.
    assert {"gfs", "hrrr", "era5", "icon-eu", "rap"} <= supported


def test_intent_drivability_is_derived_from_registry_facts():
    """Each class of row lands where its declared facts put it.

    Not a mirror of the implementation: every assertion here pairs a
    fact the registry, the route table or the packaged profile DECLARES
    -- a member set, a missing acquisition route, no cadence, a
    non-runnable status, a composed profile -- with the verdict that
    fact must produce.  The chain a row lands on is asserted against its
    RUNNER, written here as the fixed mapping a reader can check by eye,
    because asking the dispatcher for it would only prove the verdict
    equals itself.
    """

    from woof import fetch_routes
    from woof.runplan import intent_drivability
    from woof.source_adapters import source_adapters

    #: runner id -> the chain that runner's rows must land on.  A new
    #: runner is a deliberate edit here, which is the point: this is the
    #: independent statement of the fork the planner derives.
    chain_of_runner = {
        "era5_combined_grib1_v1": "experiment",
        "gfs_pgrb2_0p25_v1": "prepared:go",
        "hrrr_f00_f12_v1": "prepared:hrrr",
        "mapped_composition_v1": "prepared:staged",
        "twentycrv3_member_grib2_v1": "prepared:staged",
    }

    drivability = intent_drivability()
    adapters = {a.source_id: a for a in source_adapters()}
    table = set(fetch_routes.route_ids())
    downloadable = set(fetch_routes.all_fetchable_sources())

    for source, verdict in drivability.items():
        adapter = adapters[source]
        if not adapter.runnable:
            assert verdict["routes"] == []
            assert adapter.status.value in verdict["refusal"]
            continue
        if adapter.forcing_interval_seconds is None:
            assert verdict["routes"] == []
            assert "forcing_interval_seconds" in verdict["refusal"]
            continue
        if adapter.runner not in chain_of_runner:
            assert verdict["routes"] == []
            assert adapter.runner in verdict["refusal"]
            continue
        if verdict["routes"]:
            # A drivable row lands on the chain its RUNNER names.
            assert verdict["chain"] == chain_of_runner[adapter.runner]
            # A member axis is the ROUTE's declaration, never the
            # planner's: a row that declares a member set is drivable
            # only where its acquisition route declares one too.
            if adapter.member_set is not None:
                assert source in table
                assert fetch_routes.route_for(source).members is not None
            # Nothing is drivable without a way to its bytes: either an
            # acquisition route, or a declared local input contract whose
            # root review will demand.
            if verdict["chain"] != "experiment" and source not in downloadable:
                assert verdict["requires_source_root"] is True
                assert verdict["source_root_reason"] == (
                    fetch_routes.acquisition_refusal_reason(source))
            else:
                assert not verdict.get("requires_source_root")
            # The mapped chain composes a packaged profile, so a row
            # without one cannot reach it.
            if verdict["chain"] == "prepared:staged":
                assert adapter.packaged_profile is not None
        else:
            assert verdict["refusal"]

    # The fork is the registry row's, not a name list's: these four are
    # spelled out because each is a different declared fact.
    assert drivability["gfs"]["chain"] == "prepared:go"
    assert drivability["hrrr"]["chain"] == "prepared:hrrr"
    assert drivability["era5"]["chain"] == "experiment"
    staged = {source for source, verdict in drivability.items()
              if verdict["chain"] == "prepared:staged"}
    for source in staged:
        assert adapters[source].runner in {
            "mapped_composition_v1", "twentycrv3_member_grib2_v1"}
        assert adapters[source].packaged_profile is not None
    assert {"icon-eu", "rap"} <= staged
    assert {source for source in staged
            if drivability[source].get("requires_source_root")}


def test_a_new_registry_row_becomes_intent_drivable_with_zero_code_change(
        monkeypatch, capsys):
    """THE arbitrary acceptance test for the widened door.

    A row grafted onto the registry with the declared facts of a
    packaged staged source -- runnable, a forcing cadence, a composed
    packaged profile, a table acquisition route, no member set -- must
    come out intent-drivable, with no edit anywhere in run-plan.  The
    profile and fetch route reuse a real staged source's table entries,
    because that is exactly what adding a model is: table work.
    """

    import dataclasses

    from woof import fetch_routes
    from woof import source_adapters as registry
    from woof.runplan import intent_drivability

    donor = next(
        adapter for adapter in registry.source_adapters()
        if adapter.runner == "mapped_composition_v1"
        and adapter.packaged_profile is not None
        and adapter.member_set is None
        and adapter.source_id in set(fetch_routes.route_ids()))
    grafted = dataclasses.replace(
        donor, source_id="probe-arbitrary-model", aliases=())
    monkeypatch.setattr(
        registry, "_ADAPTERS", (*registry.source_adapters(), grafted))
    # A real row addition rebuilds the alias index at import; the graft
    # mirrors that, exactly as adding the row to the source file would.
    monkeypatch.setattr(
        registry, "_ALIASES",
        {**registry._ALIASES,  # noqa: SLF001 - the graft IS the test
         "probe-arbitrary-model": grafted})
    monkeypatch.setattr(
        fetch_routes, "_ROUTES",
        {**fetch_routes._ROUTES,  # noqa: SLF001 - the graft IS the test
         "probe-arbitrary-model":
             fetch_routes._ROUTES[donor.source_id]})  # noqa: SLF001

    verdict = intent_drivability()["probe-arbitrary-model"]
    assert verdict == {"routes": ["prepared"], "chain": "prepared:staged",
                       "refusal": None}

    # And the shape gate accepts an intent naming it, on the route the
    # derivation names -- no hardcoded list left to bounce off.
    from woof.runplan import _build_intent

    accepted = _build_intent(
        {"point": "50.1,8.7", "cycle": "2026-08-18T06",
         "source": "probe-arbitrary-model"}, route="prepared")
    assert accepted["source"] == "probe-arbitrary-model"

    # The picker document says the same thing through the front door.
    document, _raw = _sources_document(capsys)
    row = next(row for row in document["sources"]
               if row["source_id"] == "probe-arbitrary-model")
    assert row["run_plan"]["intent_supported"] is True
    assert row["run_plan"]["intent_routes"] == ["prepared"]


def test_an_undrivable_intent_source_refusal_names_the_missing_fact(
        tmp_path):
    """Never "unknown source": the refusal is the registry's own fact."""

    cases = {
        # A row with no runnable implementation route.
        "nam": ("no runnable implementation route", "status"),
        # The generic mapped adapter declares no per-source cadence.
        "mapped": ("forcing_interval_seconds", "woof domain"),
    }
    for source, needles in cases.items():
        with pytest.raises(PlanError) as refusal:
            build_plan(
                {"schema": PLAN_SCHEMA, "name": "x", "route": "prepared",
                 "config": {"intent": {"point": "39,-98",
                                       "cycle": "2026-08-18T06",
                                       "source": source}}},
                source="probe.json", base_dir=tmp_path, sha256="0" * 64)
        text = str(refusal.value)
        assert "unknown source" not in text.lower(), source
        for needle in needles:
            assert needle in text, (source, needle, text)


def test_an_ensemble_intent_source_is_drivable_on_the_prepared_route(tmp_path):
    """The other half of the retired guard: the ensemble intent plans.

    It used to be refused for carrying a member set at all, on the
    reasoning that an intent has no member axis.  The route's grammar
    supplies the default member and the staged chain verifies the bytes
    that arrive against it, so the fact the refusal was built on is gone
    and the plan is built instead.
    """

    from woof.runplan import intent_drivability

    plan = build_plan(
        {"schema": PLAN_SCHEMA, "name": "x", "route": "prepared",
         "config": {"intent": {"point": "39,-98", "cycle": "2026-08-18T06",
                               "source": "gefs"}}},
        source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert plan is not None
    verdict = intent_drivability()["gefs"]
    assert verdict["routes"] == ["prepared"]
    assert verdict["chain"] == "prepared:staged"
    assert verdict["refusal"] is None


def test_a_source_the_registry_does_not_hold_points_at_the_sources_door(
        tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan(
            {"schema": PLAN_SCHEMA, "name": "x", "route": "prepared",
             "config": {"intent": {"point": "39,-98",
                                   "cycle": "2026-08-18T06",
                                   "source": "not-a-model"}}},
            source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "not in the source registry" in text
    assert "run-plan --sources" in text


def test_a_staged_source_intent_is_accepted_on_the_prepared_route(tmp_path):
    """The gap this task closes: a packaged source the wizard already
    plans, accepted by the door that used to bounce it off a trio."""

    plan = load_plan(_prepared_plan(
        tmp_path, tmp_path / "run",
        **{"point": "50.1,8.7", "source": "icon-eu",
           "cycle": "2026-08-18T06", "hours": 3, "vram_gib": 16}))
    assert plan.config_intent["source"] == "icon-eu"


def test_a_staged_source_intent_on_the_experiment_route_names_prepared(
        tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan(
            {"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
             "config": {"intent": {"point": "50.1,8.7",
                                   "cycle": "2026-08-18T06",
                                   "source": "icon-eu"}}},
            source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "prepared" in text and "[case_data]" in text


def test_the_sources_query_needs_no_plan_and_writes_nothing(tmp_path):
    from woof.runplan import source_inventory

    document = source_inventory()
    assert document["sources"]
    assert list(tmp_path.iterdir()) == []


def test_the_no_plan_refusal_names_the_sources_door():
    """A typo must not get a refusal that hides a door it could use."""

    from woof.cli import build_parser

    with pytest.raises(PlanError) as refusal:
        run_plan_main(build_parser().parse_args(["run-plan"]))
    assert "--sources" in str(refusal.value)


def test_the_probe_schema_inventory_advertises_the_sources_document():
    from woof.runplan import SOURCES_SCHEMA, probe_environment

    document = probe_environment(readiness=False)
    assert document["schemas"]["sources"] == SOURCES_SCHEMA


def test_an_unknown_run_plan_flag_is_still_refused_by_argparse():
    from woof.cli import build_parser

    with pytest.raises(SystemExit) as exit_code:
        build_parser().parse_args(["run-plan", "--sourcez"])
    assert exit_code.value.code == 2


def test_the_sources_reply_survives_a_registry_that_raises(monkeypatch):
    """A query mode reports; it never raises into a caller's parser."""

    from woof import source_adapters as registry
    from woof.runplan import SOURCES_SCHEMA, source_inventory

    def _broken() -> dict:
        raise RuntimeError("registry manifest unavailable")

    monkeypatch.setattr(registry, "source_capability_manifest", _broken)
    document = source_inventory()
    assert document["schema"] == SOURCES_SCHEMA
    assert document["source_count"] is None
    assert "registry manifest unavailable" in document["error"]
    assert document["sources"]


# ---------------------------------------------------------------------------
# Output cadence
# ---------------------------------------------------------------------------


def _emit(tmp_path, extra):
    """Run the real wizard and return its emitted history intervals."""

    import contextlib
    import io as _io
    import tomllib as _tomllib

    from woof.cli import build_parser
    from woof.domain_wizard import domain_main

    out = tmp_path / "c.toml"
    args = build_parser().parse_args([
        "domain", "--point", "39,-98", "--source", "era5", "--cycle",
        "2024-05-03T12", "--hours", "1", "--vram-gib", "24",
        "--ladder", "12-3", "--out", str(out), *extra])
    args.interactive = False
    with contextlib.redirect_stdout(_io.StringIO()):
        assert domain_main(args) == 0
    domains = _tomllib.load(_io.BytesIO(out.read_bytes()))["domain"]
    return [d["history_interval_s"] for d in domains]


def test_the_wizards_default_cadence_is_unchanged(tmp_path):
    """The flag is a default, not a behaviour change."""

    from woof.domain_wizard import (DEFAULT_NEST_HISTORY_INTERVAL_S,
                                     DEFAULT_ROOT_HISTORY_INTERVAL_S)

    assert _emit(tmp_path, []) == [DEFAULT_ROOT_HISTORY_INTERVAL_S,
                                   DEFAULT_NEST_HISTORY_INTERVAL_S]
    assert (DEFAULT_ROOT_HISTORY_INTERVAL_S,
            DEFAULT_NEST_HISTORY_INTERVAL_S) == (3600.0, 900.0)


def test_the_cadence_flags_reach_the_emitted_config(tmp_path):
    assert _emit(tmp_path, ["--history-interval", "600",
                            "--nest-history-interval", "300"]) == [600.0, 300.0]


def test_a_root_cadence_alone_leaves_the_nest_on_its_default(tmp_path):
    assert _emit(tmp_path, ["--history-interval", "1800"]) == [1800.0, 900.0]


def test_requested_wizard_cadence_derives_a_compatible_clock_before_emission(
        tmp_path):
    """An omitted clock can move; the requested output interval cannot."""

    import contextlib
    import io as _io

    from woof.cli import build_parser
    from woof.domain_wizard import domain_main
    from woof.experiment import load_experiment

    out = tmp_path / "c.toml"
    args = build_parser().parse_args([
        "domain", "--point", "39,-98", "--source", "era5", "--cycle",
        "2024-05-03T12", "--hours", "1", "--vram-gib", "24",
        "--out", str(out), "--history-interval", "7"])
    args.interactive = False
    assert not out.exists()
    with contextlib.redirect_stdout(_io.StringIO()):
        assert domain_main(args) == 0
    # Round-trip the actual public artifact through the shared clock checks.
    # The old spacing-only 60 s default could not represent this interval.
    exp = load_experiment(out)
    assert exp.root.run.dt == 1.0
    assert exp.root.history_interval_s == 7.0
    assert exp.root.run.dx == 12000.0
    assert exp.run_seconds == 3600.0
    assert "derived root time step adjusted 60 -> 1 s" in out.read_text(
        encoding="utf-8")


def test_cadence_is_an_intent_key_on_both_routes(tmp_path):
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run",
                                  history_interval_s=1800))
    resolution, exp, _ = resolve_plan(plan, require_inputs=False)
    assert exp.domains[0].history_interval_s == 1800.0
    assert "history_interval_s = 1800.0" in resolution["generated_config"]


def test_intent_cadence_preserves_requested_physics_with_a_derived_clock(tmp_path):
    from woof.domain_wizard import DEFAULT_PHYSICS_PROFILE, profile_switches

    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run",
        history_interval_s=7, physics_profile=DEFAULT_PHYSICS_PROFILE))
    resolution, exp, _ = resolve_plan(plan, require_inputs=False)
    assert exp.root.run.dt == 1.0
    assert exp.root.history_interval_s == 7.0
    assert exp.run_seconds == 3600.0
    # A named suite is caller authority. Its physics periods/selectors must
    # not be rewritten to accommodate an incompatible author-chosen clock.
    declared = profile_switches(DEFAULT_PHYSICS_PROFILE)
    for key in ("mp_physics", "cu_physics", "bl_pbl_physics",
                "sf_sfclay_physics", "sf_surface_physics",
                "ra_lw_physics", "ra_sw_physics", "radt", "cudt_minutes"):
        assert getattr(exp.root.run, key) == declared[key], key
    assert "history_interval_s = 7.0" in resolution["generated_config"]
    assert "derived root time step adjusted 60 -> 1 s" in resolution["generated_config"]
    assert not plan.run_dir.exists()  # resolution still does not launch


# ---------------------------------------------------------------------------
# Cycle "latest"
# ---------------------------------------------------------------------------


def test_latest_is_resolved_to_a_concrete_cycle_before_the_fetch_runs(
        monkeypatch):
    from datetime import datetime as _dt

    import woof.fetch as fetch
    from woof.runplan import resolve_fetch_cycle

    monkeypatch.setattr(fetch, "resolve_latest_cycle",
                        lambda source, last_hour: _dt(2026, 8, 7, 12))
    # The whole-cycle rule's resolver call; the as-posted default is
    # pinned in tests/test_readiness.py.
    arguments, resolutions, warnings = resolve_fetch_cycle(
        ["--source", "gfs", "--cycle", "latest", "--hours", "6",
         "--whole-cycle", "--out", "data"])

    assert "latest" not in arguments
    assert arguments[arguments.index("--cycle") + 1] == "2026-08-07T12"
    assert resolutions[0] == {
        "scope": "fetch", "key": "cycle", "value": "2026-08-07T12",
        "basis": "resolved_latest", "note": resolutions[0]["note"]}
    assert "complete by construction" in resolutions[0]["note"]


def test_an_explicit_cycle_is_left_exactly_as_written(monkeypatch):
    import woof.fetch as fetch
    from woof.runplan import resolve_fetch_cycle

    def refuse(*a, **k):
        raise AssertionError("an explicit cycle must not probe the mirrors")

    monkeypatch.setattr(fetch, "resolve_latest_cycle", refuse)
    arguments, resolutions, warnings = resolve_fetch_cycle(
        ["--source", "gfs", "--cycle", "2026-08-07T00", "--hours", "6"])
    assert arguments[arguments.index("--cycle") + 1] == "2026-08-07T00"
    assert resolutions == [] and warnings == []


def test_latest_is_matched_case_insensitively(monkeypatch):
    from datetime import datetime as _dt

    import woof.fetch as fetch
    from woof.runplan import resolve_fetch_cycle

    monkeypatch.setattr(fetch, "resolve_latest_cycle",
                        lambda source, last_hour: _dt(2026, 8, 7, 12))
    arguments, resolutions, _ = resolve_fetch_cycle(
        ["--source", "hrrr", "--cycle", "Latest", "--hours", "3",
         "--whole-cycle"])
    assert arguments[arguments.index("--cycle") + 1] == "2026-08-07T12"
    assert resolutions


def test_a_stale_latest_cycle_is_a_warning_never_a_refusal(monkeypatch):
    """Newer cycles publishing = the run starts older than you think."""

    from datetime import datetime as _dt, timedelta as _td

    import woof.fetch as fetch
    import woof.runplan as runplan_module
    from woof.runplan import resolve_fetch_cycle

    stale = _dt(2026, 8, 7, 0)
    monkeypatch.setattr(fetch, "resolve_latest_cycle",
                        lambda source, last_hour: stale)
    monkeypatch.setattr(
        runplan_module, "datetime",
        type("C", (_dt,), {"now": classmethod(
            lambda cls, tz=None: stale + _td(hours=20))}))

    arguments, resolutions, warnings = resolve_fetch_cycle(
        ["--source", "gfs", "--cycle", "latest", "--hours", "6",
         "--whole-cycle"])

    assert arguments[arguments.index("--cycle") + 1] == "2026-08-07T00"
    assert warnings and warnings[0]["code"] == "latest_cycle_is_not_the_newest"
    assert warnings[0]["age_hours"] == 20
    assert "not yet published" in warnings[0]["message"]


def test_the_fetch_hours_include_the_forecast_start_lead(monkeypatch):
    """`latest` must cover the END of the window, lead included."""

    from datetime import datetime as _dt

    import woof.fetch as fetch
    from woof.runplan import resolve_fetch_cycle

    seen = {}

    def record(source, last_hour):
        seen["last_hour"] = last_hour
        return _dt(2026, 8, 7, 0)

    monkeypatch.setattr(fetch, "resolve_latest_cycle", record)
    resolve_fetch_cycle(["--source", "gfs", "--cycle", "latest",
                         "--hours", "12", "--forecast-start-hour", "6",
                         "--whole-cycle"])
    assert seen["last_hour"] == 18


# ---------------------------------------------------------------------------
# The event stream
# ---------------------------------------------------------------------------


def test_the_stream_writes_one_envelope_per_line_with_a_dense_sequence(
        tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="a")
        events.emit("warning", code="c", message="m")
        events.emit("completed", dry_run=True)

    records = read_events(path)
    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert {record["schema_version"] for record in records} == {EVENT_SCHEMA}
    assert [record["event"] for record in records] == [
        "plan_accepted", "warning", "completed"]
    # Event-specific fields are flattened alongside the envelope, not
    # nested under a payload key.
    assert records[1]["code"] == "c"
    assert all(isinstance(record["emitted_unix_ms"], int)
               for record in records)


def test_an_unknown_event_tag_is_refused_rather_than_written(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        with pytest.raises(PlanError):
            events.emit("almost_completed")
    assert read_events(path) == []


def test_an_event_may_not_shadow_an_envelope_key(tmp_path):
    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        with pytest.raises(PlanError) as refusal:
            events.emit("warning", sequence=99, code="c", message="m")
    assert "sequence" in str(refusal.value)


def test_a_replayed_stream_is_identical_to_what_was_emitted(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    emitted = []
    with EventStream(path, mirror=None) as events:
        for index in range(20):
            emitted.append(events.emit("model_progress", domain=1,
                                       model_seconds=float(index)))
    assert read_events(path) == emitted


def test_a_second_stream_into_the_same_file_continues_the_sequence(tmp_path):
    """Append must not restart at 1, or the whole file becomes unreadable.

    The file is opened for append, so a resume -- or any caller that
    reuses a run directory -- writes after records that already exist.
    A restarted counter puts a 1 after a 7 and read_events refuses the
    lot as reordered.
    """

    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="first")
        events.emit("completed", dry_run=True)
    with EventStream(path, mirror=None) as events:
        assert events.sequence == 2
        events.emit("plan_accepted", name="second")

    records = read_events(path)
    assert [r["sequence"] for r in records] == [1, 2, 3]
    assert records[-1]["name"] == "second"


def test_a_torn_final_line_is_refused_unless_the_caller_says_otherwise(
        tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="a")
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"schema_version": "gpuwm.run-p')

    with pytest.raises(PlanError):
        read_events(path)
    assert len(read_events(path, allow_partial_tail=True)) == 1


def test_a_sequence_gap_is_refused_because_it_means_a_lost_line(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    path.write_text("\n".join(json.dumps({
        "schema_version": EVENT_SCHEMA, "sequence": sequence,
        "emitted_unix_ms": 0, "event": "warning", "code": "c",
        "message": "m"}) for sequence in (1, 3)) + "\n", encoding="utf-8")
    with pytest.raises(PlanError) as refusal:
        read_events(path)
    assert "sequence" in str(refusal.value)


# ---------------------------------------------------------------------------
# A run, with the GPU integrate step replaced at the observer seam
# ---------------------------------------------------------------------------


class _StubSummary:
    """What ``run_experiment`` returns, in the shape the route reads."""

    def __init__(self, paths):
        self.wrfout_paths = tuple(paths)
        self.completed_seconds = 3600.0
        self.nan_free = True


def _stub_run_experiment(fail_at=None, frames=2):
    """A run that drives the real observer through the real protocol.

    The phases and the keyword set are the pipeline's own -- copied from
    ``runtime._preparation_progress``'s call sites and the per-step
    ``progress_callback`` -- so this stub is wrong in exactly the ways
    the live pipeline would be wrong, and no others.
    """

    def run_experiment(exp, data, outdir, *, restart=None,
                       progress_callback=None, health_debug=False):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for phase in ("quarantine-wrfout", "resolve-schedule",
                      "prepare-case"):
            progress_callback.preparing(phase)
        if fail_at == "prepare":
            raise RuntimeError("preparation refused")
        for phase in ("initialize-health-validator", "cold-start-wrfout",
                      "initial-health-gate"):
            progress_callback.preparing(phase)
        for step in range(1, frames + 1):
            valid = exp.start_time + timedelta(seconds=step * 60.0)
            path = outdir / f"wrfout_d01_{step:03d}"
            path.write_bytes(b"frame")
            paths.append(path)
            progress_callback.output_committed(
                domain=1, valid_time=valid, path=path)
            progress_callback(
                model_elapsed_seconds=float(step * 60),
                outer_step=step, last_durable_wrfout=path,
                last_checkpoint=None, phase="post-d01-sync",
                step_wall_seconds=0.004)
            if fail_at == "forecast" and step == 1:
                raise RuntimeError("the dycore refused")
        return _StubSummary(paths)

    return run_experiment


@pytest.fixture()
def stubbed_runtime(monkeypatch):
    def install(**kwargs):
        from woof import runtime
        monkeypatch.setattr(runtime, "run_experiment",
                            _stub_run_experiment(**kwargs))
    return install


def _run(tmp_path, stubbed_runtime, *, plan_overrides=None, **stub):
    stubbed_runtime(**stub)
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan = load_plan(_write_plan(tmp_path, config, run_dir,
                                 **(plan_overrides or {})))
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        code = execute_plan(plan, events=events)
    return code, run_dir, read_events(run_dir / EVENTS_FILENAME)


@requires_cupy
def test_a_completed_run_emits_its_events_in_order_with_a_dense_sequence(
        tmp_path, stubbed_runtime):
    code, run_dir, events = _run(tmp_path, stubbed_runtime)

    assert code == 0
    assert [record["sequence"] for record in events] == list(
        range(1, len(events) + 1))
    tags = [record["event"] for record in events]
    assert tags[0] == "plan_accepted"
    assert tags[1] == "resolved_plan"
    assert tags[-1] == "completed"
    assert set(tags) <= set(EVENT_TAGS)

    # The stages the pipeline's own phases opened, in order, each one
    # started before it finished.
    started = [record["stage"] for record in events
               if record["event"] == "stage_started"]
    finished = [record["stage"] for record in events
                if record["event"] == "stage_finished"]
    assert started == ["prepare", "initialize", "forecast", "finalize"]
    assert finished == started
    assert all(stage in STAGES for stage in started)

    # Every output that landed is announced with its domain, valid time
    # and path -- no consumer re-derives them from a filename.
    committed = [record for record in events
                 if record["event"] == "output_committed"]
    assert len(committed) == 2
    assert committed[0]["domain"] == 1
    assert Path(committed[0]["path"]).is_file()
    datetime.fromisoformat(committed[0]["valid_time"])

    progress = [record for record in events
                if record["event"] == "model_progress"]
    assert [record["outer_step"] for record in progress] == [1, 2]
    assert progress[0]["step_ms"] == pytest.approx(4.0)
    assert progress[-1]["model_seconds"] == 120.0
    assert progress[-1]["wall_seconds"] >= 0.0

    completed = events[-1]
    assert completed["summary"]["wrfout_count"] == 2
    assert completed["outputs_committed"] == 2


@requires_cupy
def test_an_output_is_announced_only_after_the_progress_that_precedes_it_is(
        tmp_path, stubbed_runtime):
    """Ordering is the contract; a consumer builds a timeline from it."""

    _code, _run_dir, events = _run(tmp_path, stubbed_runtime)
    ordered = [(record["sequence"], record["event"]) for record in events
               if record["event"] in ("output_committed", "model_progress")]
    assert [tag for _, tag in ordered] == [
        "output_committed", "model_progress"] * 2
    assert [sequence for sequence, _ in ordered] == sorted(
        sequence for sequence, _ in ordered)


def test_the_manifest_is_written_before_any_work_and_names_every_stream(
        tmp_path, stubbed_runtime):
    from woof.supervisor import (FAILURE_CAPSULE_SCHEMA, HEARTBEAT_NAME,
                                  HEARTBEAT_SCHEMA)

    _code, run_dir, events = _run(tmp_path, stubbed_runtime)
    manifest = json.loads(
        (run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["pid"] > 0
    assert len(manifest["plan_sha256"]) == 64
    assert Path(manifest["events_path"]) == run_dir / EVENTS_FILENAME
    assert manifest["events_schema"] == EVENT_SCHEMA
    # It points at the schemas this module does NOT own, so a front end
    # never has to know their filenames.
    assert Path(manifest["progress_path"]) == run_dir / HEARTBEAT_NAME
    assert manifest["progress_schema"] == HEARTBEAT_SCHEMA
    assert manifest["failure_capsule_schema"] == FAILURE_CAPSULE_SCHEMA
    assert "heartbeat" in manifest["reattach"]

    # The manifest exists by the time the first event says it does.
    assert events[0]["manifest_path"] == str(run_dir / MANIFEST_FILENAME)


def test_reattach_replay_yields_exactly_the_event_list_that_was_emitted(
        tmp_path, stubbed_runtime):
    _code, run_dir, events = _run(tmp_path, stubbed_runtime)
    manifest = json.loads(
        (run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    # A consumer attaching afterwards has only the manifest, and reaches
    # the same history from it.
    replayed = read_events(manifest["events_path"])
    assert replayed == events
    # Twice, because a replay that is not idempotent is not a replay.
    assert read_events(manifest["events_path"]) == replayed


@requires_cupy
def test_the_supervisors_own_heartbeat_is_the_one_that_gets_written(
        tmp_path, stubbed_runtime):
    """No second progress writer: this front door composes with theirs."""

    from woof.supervisor import HEARTBEAT_NAME, HEARTBEAT_SCHEMA, read_heartbeat

    _code, run_dir, _events = _run(tmp_path, stubbed_runtime)
    heartbeat = read_heartbeat(run_dir / HEARTBEAT_NAME)
    assert heartbeat.schema == HEARTBEAT_SCHEMA
    assert heartbeat.status == "complete"
    assert heartbeat.outer_step == 2
    assert heartbeat.model_elapsed_seconds == 3600.0


@requires_cupy
def test_a_failed_run_emits_failed_last_and_exits_nonzero(
        tmp_path, stubbed_runtime):
    code, _run_dir, events = _run(tmp_path, stubbed_runtime,
                                  fail_at="forecast")

    assert code != 0
    assert events[-1]["event"] == "failed"
    assert events[-1]["stage"] == "forecast"
    assert events[-1]["error_class"] == "RuntimeError"
    assert "dycore refused" in events[-1]["message"]
    # The stage that was open is closed before the failure, so a
    # consumer's stage timeline has no stage left hanging.
    closing = [record for record in events
               if record["event"] == "stage_finished"]
    assert closing[-1]["stage"] == "forecast"
    assert closing[-1]["outcome"] == "failed"


@requires_cupy
def test_a_failure_during_preparation_names_the_stage_it_failed_in(
        tmp_path, stubbed_runtime):
    code, _run_dir, events = _run(tmp_path, stubbed_runtime,
                                  fail_at="prepare")
    assert code != 0
    assert events[-1]["event"] == "failed"
    assert events[-1]["stage"] == "prepare"


def test_a_dry_run_resolves_the_plan_and_stops_before_any_device_work(
        tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run must not reach the runtime")

    from woof import runtime
    monkeypatch.setattr(runtime, "run_experiment", refuse)

    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan = load_plan(_write_plan(tmp_path, config, run_dir,
                                 run_options={"dry_run": True}))
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        code = execute_plan(plan, events=events)

    events = read_events(run_dir / EVENTS_FILENAME)
    assert code == 0
    assert events[-1]["event"] == "completed"
    assert events[-1]["dry_run"] is True
    # It still resolved: the whole point of a dry run is the snapshot.
    assert any(record["event"] == "resolved_plan" for record in events)


@requires_cupy
def test_the_resolved_plan_event_carries_the_snapshot_and_the_resolutions(
        tmp_path, stubbed_runtime):
    _code, _run_dir, events = _run(tmp_path, stubbed_runtime)
    resolved = next(record for record in events
                    if record["event"] == "resolved_plan")
    assert resolved["configuration"]["experiment"]["name"]
    assert resolved["automatic_resolutions"]
    assert any(entry["key"] == "execution_mode"
               for entry in resolved["automatic_resolutions"])


# NEEDS CUPY INSTALLED, and opens no device: this test runs the execution
# road and reads its stdout; without cupy the door refuses ahead of it and
# there is no stream to mirror.
@requires_cupy
def test_the_stream_is_mirrored_to_stdout_line_for_line(
        tmp_path, stubbed_runtime, capsys):
    stubbed_runtime()
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan_path = _write_plan(tmp_path, config, run_dir)

    from woof.cli import build_parser
    args = build_parser().parse_args(["run-plan", str(plan_path)])
    code = run_plan_main(args)

    assert code == 0
    printed = [json.loads(line) for line
               in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert printed == read_events(run_dir / EVENTS_FILENAME)


# ---------------------------------------------------------------------------
# The observer, on its own
# ---------------------------------------------------------------------------


def test_an_unmapped_preparation_phase_is_reported_not_mis_filed(tmp_path):
    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        observer.enter_stage("prepare")
        observer.preparing("a-phase-nobody-mapped")
        observer.finish_stage()
    records = read_events(tmp_path / EVENTS_FILENAME)
    warning = next(record for record in records
                   if record["event"] == "warning")
    assert warning["code"] == "unmapped_pipeline_phase"
    assert warning["phase"] == "a-phase-nobody-mapped"
    # And it still lands in the open stage's phase list rather than
    # vanishing.
    finished = next(record for record in records
                    if record["event"] == "stage_finished")
    assert "a-phase-nobody-mapped" in finished["phases"]


def _phases_the_pipeline_emits() -> set[str]:
    """Every phase literal a ``_preparation_progress`` call site passes."""

    import re

    call = re.compile(r"_preparation_progress\(\s*[A-Za-z_][A-Za-z0-9_.]*\s*,"
                      r"\s*\"([a-z0-9-]+)\"\s*\)")
    package = Path(__file__).resolve().parents[1] / "woof"
    found: set[str] = set()
    for path in package.rglob("*.py"):
        found.update(call.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_phase_the_pipeline_emits_has_a_stage():
    """A155: every prepared `woof go` printed two "no stage for" warnings.

    The prepared runners report validate-prepared-inputs and
    restore-prepared-domain(-tree), and a tree restart
    restore-nest-lifecycle, and none of the four was in the table, so the
    warning meant for a phase the pipeline GREW fired on every run.  This
    holds the table to the pipeline's own call sites, so a new phase fails
    here instead of warning on every run; the test above keeps the warning
    for a phase nobody mapped.
    """

    from woof import runplan

    emitted = _phases_the_pipeline_emits()
    assert {"validate-prepared-inputs", "restore-prepared-domain",
            "restore-prepared-domain-tree", "prepare-case"} <= emitted, (
        f"the call-site scan found too little to mean anything: {emitted}")
    unmapped = sorted(emitted - set(runplan._PHASE_STAGES))
    assert unmapped == [], (
        f"the pipeline reports {unmapped} and run-plan maps them to no "
        "stage, so every run reaching them prints a warning")
    assert set(runplan._PHASE_STAGES.values()) <= set(STAGES)


def test_a_prepared_runner_s_phases_stay_in_the_forecast_stage(tmp_path):
    """The prepared chains open `forecast` before the runner starts; the
    runner's check and load of its bundle land there, with no warning and
    no extra stage."""

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        observer.enter_stage("forecast", phase="forecast")
        observer.preparing("validate-prepared-inputs")
        observer.preparing("restore-prepared-domain")
        observer(model_elapsed_seconds=60.0, outer_step=1)
        observer.finish_stage()
    records = read_events(tmp_path / EVENTS_FILENAME)
    assert [record for record in records if record["event"] == "warning"] == []
    finished = [record for record in records
                if record["event"] == "stage_finished"]
    assert [(record["stage"], record["phases"]) for record in finished] == [
        ("forecast", ["forecast", "validate-prepared-inputs",
                      "restore-prepared-domain"])]


def test_speed_x_is_null_rather_than_an_infinity_before_any_wall_elapses(
        tmp_path):
    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        observer(model_elapsed_seconds=0.0, outer_step=0,
                 last_durable_wrfout=None, last_checkpoint=None,
                 phase="initialized-or-restored", step_wall_seconds=0.0)
    progress = next(record for record in read_events(
        tmp_path / EVENTS_FILENAME) if record["event"] == "model_progress")
    assert progress["speed_x"] is None
    assert progress["step_ms"] is None


def test_the_landing_hook_is_optional_on_a_hand_built_writer_shell():
    """The seam must not require the ``__init__`` path to have run.

    ``tests/test_wrfout.py`` stands this writer up field-by-field
    through ``object.__new__`` because a CPU-only harness cannot
    allocate a CuPy stream.  The first version of the landing hook set
    its attributes in ``__init__`` only, so every such shell raised
    AttributeError on the worker thread -- four tests, none of them
    about this feature.  The defaults live on the class for that reason.
    """

    from woof.io.wrfout import AsyncDomainWrfoutWriter

    shell = object.__new__(AsyncDomainWrfoutWriter)
    assert shell.landing_observer is None
    assert shell.grid_id is None


def test_the_landing_hook_can_be_bound_after_the_writers_are_built():
    """Both prepared runners build their closure OVER the writers.

    Their progress closure reports ``writers.paths``, so it cannot exist
    before the object it reads -- a real cycle, which reordering does
    not break.  ``attach_progress_callback`` is the late-binding half,
    and these two nodes pin it: it binds, and it refuses once a frame
    has gone by.
    """

    from woof.io.wrfout import AsyncDomainWrfoutWriter, PerDomainWrfoutWriters

    writers = object.__new__(PerDomainWrfoutWriters)
    shell = object.__new__(AsyncDomainWrfoutWriter)
    shell.paths = []
    shell._pending = 0
    shell._condition = __import__("threading").Condition()
    writers._writers = {1: shell}

    def observed(**_kwargs):
        pass

    carrier = type("Carrier", (), {"output_committed": staticmethod(observed)})()
    writers.attach_progress_callback(carrier)
    assert shell.landing_observer is observed

    # A progress object without the hook binds nothing, rather than
    # binding something that cannot be called.
    writers.attach_progress_callback(lambda **_: None)
    assert shell.landing_observer is None


def test_binding_the_landing_hook_late_is_refused_once_a_frame_has_landed():
    from woof.io.wrfout import AsyncDomainWrfoutWriter, PerDomainWrfoutWriters

    writers = object.__new__(PerDomainWrfoutWriters)
    shell = object.__new__(AsyncDomainWrfoutWriter)
    shell.paths = [Path("wrfout_d01_0001")]
    shell._pending = 0
    shell._condition = __import__("threading").Condition()
    writers._writers = {1: shell}

    with pytest.raises(RuntimeError) as refusal:
        writers.attach_progress_callback(
            type("C", (), {"output_committed": staticmethod(lambda **_: None)})())
    assert "silently miss" in str(refusal.value)


@pytest.mark.parametrize("runner", ["woof.prepared_single_domain_forecast",
                                    "woof.prepared_domain_tree_forecast"])
def test_both_prepared_runners_accept_an_external_observer(runner):
    """The parameter exists and defaults to off, on both sibling routes.

    This is the plumbing a future run-plan route needs; the routes
    themselves are not registered yet.  Signature-level, because the
    runner bodies need a card.
    """

    import importlib
    import inspect

    module = importlib.import_module(runner)
    entry = (module.run_prepared_forecast
             if hasattr(module, "run_prepared_forecast")
             else module.run_prepared_tree)
    parameter = inspect.signature(entry).parameters["observer"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_observer_works_with_no_heartbeat_at_all(tmp_path):
    """The heartbeat is composed, not required: this must not crash."""

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events, heartbeat=None)
        observer.starting()
        observer.preparing("prepare-case")
        observer.complete(1.0)
        observer.failed()
    assert read_events(tmp_path / EVENTS_FILENAME)


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------


def test_resolve_prints_one_json_document_and_runs_nothing(
        tmp_path, capsys, monkeypatch):
    from woof import runtime
    monkeypatch.setattr(runtime, "run_experiment", lambda *a, **k: 1 / 0)

    config = make_case_toml(tmp_path)
    plan_path = _write_plan(tmp_path, config, tmp_path / "run")
    from woof.cli import build_parser
    args = build_parser().parse_args(["run-plan", "--resolve", str(plan_path)])

    assert run_plan_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "gpuwm.run-plan.resolved.v1"
    assert document["configuration"]["experiment"]["name"]
    assert document["automatic_resolutions"]
    # Nothing was created: --resolve answers a question, it does not
    # claim a directory.
    assert not (tmp_path / "run" / EVENTS_FILENAME).exists()


def test_estimate_reports_measured_numbers_and_nulls_the_unmeasured_ones(
        tmp_path, capsys):
    # The disk figure is measured now (history, checkpoints and pictures
    # from bytes per cell; the download and preparation from the sizes in
    # woof/data/download-bytes.v1.json), so it is a number; the wall
    # time is still unmeasured for an arbitrary configuration.
    # This is a pre-download estimate. Present inputs are now inventoried
    # for retained boundary counts, so placeholder bytes are not GRIB data.
    config = make_case_toml(tmp_path, files=False)
    plan_path = _write_plan(tmp_path, config, tmp_path / "run")
    from woof.cli import build_parser
    args = build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])

    assert run_plan_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "gpuwm.run-plan.estimate.v1"
    assert document["vram"]["estimate_bytes"] > 0
    assert document["disk"]["total_frames"] > 0
    assert document["disk"]["bytes"] == (
        document["disk"]["history_bytes"] + document["disk"]["checkpoint_bytes"]
        + document["disk"]["picture_bytes"] + document["disk"]["download_bytes"]
        + document["disk"]["preparation_bytes"])
    # The accurate null, with its basis stated rather than a number
    # this package never measured.
    assert document["wall_time"]["seconds"] is None
    assert document["wall_time"]["basis"]
    # This config names no [fetch] table: nothing is downloaded, and the
    # basis says so rather than leaving the reader a null.
    assert document["download"]["bytes"] == 0
    assert "downloads nothing" in document["download"]["basis"]


def test_probe_answers_without_a_plan_and_without_touching_the_card(
        capsys, monkeypatch):
    import woof.core.preflight as preflight
    import woof.supervisor as supervisor
    from woof.supervisor import GPUIdentity

    monkeypatch.setattr(supervisor, "query_gpus",
                        lambda: (GPUIdentity("GPU-fixture", "999.00",
                                             "Fixture Card", 0),))
    monkeypatch.setattr(preflight, "device_physical_total_bytes",
                        lambda: 32 * 1024 ** 3)
    monkeypatch.setattr(preflight, "device_wide_used_bytes",
                        lambda: 2 * 1024 ** 3)

    from woof.cli import build_parser
    args = build_parser().parse_args(
        ["run-plan", "--probe", "--no-readiness"])

    assert run_plan_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "gpuwm.run-plan.probe.v1"
    assert document["devices"][0]["name"] == "Fixture Card"
    assert document["devices"][0]["memory_free_bytes"] == 30 * 1024 ** 3
    assert document["readiness"]["collected"] is False
    assert "experiment" in document["routes"]
    assert document["schemas"]["event"] == EVENT_SCHEMA


def test_a_probe_survives_a_machine_with_no_nvidia_smi_at_all(monkeypatch):
    import woof.supervisor as supervisor

    def absent():
        raise supervisor.GPUPreflightError("nvidia-smi unavailable")

    monkeypatch.setattr(supervisor, "query_gpus", absent)
    document = probe_environment(readiness=False)
    assert document["devices"] == []
    assert "nvidia-smi" in document["device_query_error"]


def test_the_module_entry_and_the_subcommand_take_the_same_flags():
    """One parser, two spellings; they cannot drift."""

    from woof.cli import build_parser

    subcommand = build_parser().parse_args(["run-plan", "--probe"])
    assert subcommand.probe is True
    assert subcommand.func.__name__ == "run_plan_main"


# ---------------------------------------------------------------------------
# The real artifact, in a real subprocess
# ---------------------------------------------------------------------------
#
# Everything above drives the front door in-process, which is where the
# contract lives.  These three run the actual command, because two of
# its promises are only true of a process: the exit code, and what
# reaches the terminal.  The first version of `python -m woof.runplan`
# passed every in-process test above and still printed the [[explain]]
# sentinel to stderr on a layered refusal -- it had grown its own
# refusal boundary instead of using the one boundary. Only running it
# showed that.


def _cli(*tokens, cwd, offline=False):
    """Run the real command in a fresh interpreter, pinned to this tree.

    ``offline`` points every HTTP client at a closed local port, for a
    test whose premise is that the forcing cannot be fetched: ERA5 has a
    keyless provider, so on a box with internet access the plan would
    otherwise fetch it and run the whole forecast.
    """

    import os
    import subprocess
    import sys as _sys

    repo = str(Path(__file__).resolve().parents[1])
    environment = dict(os.environ)
    # The pin is essential: without it a subprocess resolves woof
    # through whatever editable install this interpreter carries, and
    # the test silently exercises a different checkout.
    environment["PYTHONPATH"] = repo + os.pathsep + environment.get(
        "PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if offline:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                     "http_proxy", "https_proxy", "all_proxy"):
            environment[name] = "http://127.0.0.1:9"
        for name in ("NO_PROXY", "no_proxy"):
            environment.pop(name, None)
    return subprocess.run(
        [_sys.executable, "-m", "woof.runplan", *tokens],
        capture_output=True, text=True, cwd=str(cwd), env=environment,
        timeout=300)


def test_the_real_command_exits_zero_and_prints_only_the_event_stream(
        tmp_path):
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan_path = _write_plan(tmp_path, config, run_dir,
                            run_options={"dry_run": True})

    result = _cli(str(plan_path), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    printed = [json.loads(line) for line in lines]
    assert [record["event"] for record in printed] == [
        "plan_accepted", "resolved_plan", "completed"]
    # stdout is the machine channel and carries nothing else at all.
    assert len(printed) == len(lines)
    assert printed == read_events(run_dir / EVENTS_FILENAME)
    assert (run_dir / MANIFEST_FILENAME).is_file()


@requires_cupy
def test_the_real_command_keeps_stdout_pure_through_a_talking_pipeline(
        tmp_path):
    """The regression the dry-run test could not see.

    A real run reaches code that prints for a person: the pipeline's
    resolved-config report, the feedback advisory, and (on an intent
    plan) the whole wizard.  All of it used to land in the middle of the
    JSONL a consumer is calling json.loads on line by line.  The
    dry-run subprocess test never reaches any of it and passed happily.

    An intent plan is the sharpest version -- the wizard is the
    chattiest thing in the package -- and it only has to get as far as
    resolution to prove the point, so this stays CPU-only.
    """

    run_dir = tmp_path / "run"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "talker", "route": "experiment",
        "config": {"intent": {"point": "39,-98", "source": "era5",
                              "cycle": "2024-05-03T12", "hours": 1,
                              "vram_gib": 24}},
        "output_root": str(run_dir),
    }), encoding="utf-8")

    result = _cli(str(plan_path), cwd=tmp_path, offline=True)

    # It fails -- the forcing was never fetched -- and that is fine:
    # what is under test is which channel each half went down.
    assert result.returncode == 1
    for line in result.stdout.splitlines():
        if line.strip():
            json.loads(line)   # every stdout line, without exception
    events = [json.loads(line) for line in result.stdout.splitlines()
              if line.strip()]
    assert events[-1]["event"] == "failed"
    # The wizard really did run and really did talk -- to stderr.
    assert "woof domain" in result.stderr
    assert "sizing:" in result.stderr


@pytest.mark.parametrize("failure", [
    "ImportError('No module named cupy')",
    "RuntimeError('CUDA driver version is insufficient')",
    "OSError('cannot load nvrtc64_120_0.dll')",
])
def test_probe_works_on_a_box_whose_cupy_will_not_load(tmp_path, failure):
    """--probe exists to preflight an install, including a broken one.

    An ABSENT CuPy raises ImportError and was always survived.  An
    INSTALLED BUT UNLOADABLE one -- a cupy-cuda12x wheel on a CUDA-13
    box, a missing nvrtc DLL -- raises RuntimeError or OSError from
    inside the import, and woof.cli reaches that import at module
    scope (cli -> downscale -> offline_child -> core.state).  So the
    whole command line died on exactly the installs --probe is for.

    A subprocess, because the guard runs once at import and cannot be
    re-armed in a live interpreter.
    """

    import os
    import subprocess
    import sys as _sys

    repo = str(Path(__file__).resolve().parents[1])
    script = tmp_path / "probe_without_cupy.py"
    script.write_text(
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('cupy', 'cupy_backends'):\n"
        f"            raise {failure}\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from woof.runplan import _module_entry\n"
        "sys.exit(_module_entry(['--probe', '--no-readiness']))\n",
        encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = repo + os.pathsep + environment.get(
        "PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [_sys.executable, str(script)], capture_output=True, text=True,
        cwd=str(tmp_path), env=environment, timeout=300)

    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["schema"] == "gpuwm.run-plan.probe.v1"
    assert document["readiness"]["collected"] is False


def test_an_unloadable_cupy_still_fails_loudly_where_it_is_needed(tmp_path):
    """Deferred, not swallowed: the use site names the real cause."""

    import os
    import subprocess
    import sys as _sys

    repo = str(Path(__file__).resolve().parents[1])
    script = tmp_path / "require_cupy.py"
    script.write_text(
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('cupy', 'cupy_backends'):\n"
        "            raise RuntimeError('CUDA driver version is insufficient')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from woof.core.state import _require_cupy\n"
        "try:\n"
        "    _require_cupy()\n"
        "except RuntimeError as error:\n"
        "    print(error)\n",
        encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = repo + os.pathsep + environment.get(
        "PYTHONPATH", "")
    result = subprocess.run(
        [_sys.executable, str(script)], capture_output=True, text=True,
        cwd=str(tmp_path), env=environment, timeout=300)

    assert result.returncode == 0, result.stderr
    # Installed-and-broken is a different problem from absent, with a
    # different fix, so it does not get the "install it" sentence alone.
    assert "CuPy IS installed here but failed to load" in result.stdout
    assert "CUDA driver version is insufficient" in result.stdout
    assert "woof doctor" in result.stdout


def test_the_real_command_refuses_a_bad_plan_at_exit_2_in_one_layer(tmp_path):
    from woof.explain import EXPLAIN_MARK

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": "gpuwm.run-plan.v9", "name": "x", "route": "experiment",
        "config": {"inline": "x = 1"}}), encoding="utf-8")

    result = _cli(str(plan_path), cwd=tmp_path)

    assert result.returncode == 2
    assert PLAN_SCHEMA in result.stderr
    # ONE layer reaches the terminal, and the sentinel never does.
    assert EXPLAIN_MARK.strip() not in result.stderr
    assert "--explain" in result.stderr
    assert result.stdout == ""


def test_the_real_command_exits_nonzero_with_failed_as_its_last_line(
        tmp_path):
    config = make_case_toml(tmp_path)
    # A declared input that is not there: it loads far enough to be
    # accepted, then fails inside resolution -- so the failure arrives
    # as an event on an already-open stream, which is the case a
    # consumer has to handle.
    (tmp_path / "Vtable.ERA5").unlink()
    run_dir = tmp_path / "run"

    result = _cli(str(_write_plan(tmp_path, config, run_dir)), cwd=tmp_path)

    assert result.returncode == 1
    events = [json.loads(line) for line in result.stdout.splitlines()
              if line.strip()]
    assert events[-1]["event"] == "failed"
    assert events[-1]["error_class"]
    assert events[-1]["message"]
    assert read_events(run_dir / EVENTS_FILENAME) == events


# ---------------------------------------------------------------------------
# A config.path that is not on disk
# ---------------------------------------------------------------------------


def _absent_config_plan(tmp_path) -> tuple[Path, Path]:
    """A plan naming a config that was never written, and that path."""

    absent = tmp_path / "nowhere" / "case.toml"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "absent-config",
        "route": "experiment", "config": {"path": str(absent)},
        "output_root": str(tmp_path / "run")}), encoding="utf-8")
    return path, absent


def test_an_unreadable_config_path_refuses_at_the_read_seam(tmp_path):
    """PlanError is a ValueError, which is what exits 2 rather than crashing.

    The class is the mechanism the two subprocess tests below measure;
    asserting it here is what points a failure at this seam instead of
    at the front door.
    """

    plan_path, _absent = _absent_config_plan(tmp_path)
    with pytest.raises(PlanError) as refusal:
        load_plan(plan_path).config_bytes()
    assert "config.path" in str(refusal.value)


def test_the_query_doors_refuse_an_absent_config_at_exit_2(tmp_path):
    """The estimate door's exit code, and the door beside it.

    ``config.path`` is read before anything else and the read had
    nothing between it and the terminal: ``--estimate`` and
    ``--resolve`` printed a bare FileNotFoundError traceback at exit 1,
    where every other plan defect on this door is one sentence at exit
    2.  A front end driving the estimate strip got a crash it could not
    tell apart from a broken install.
    """

    from woof.explain import EXPLAIN_MARK

    plan_path, absent = _absent_config_plan(tmp_path)
    for mode in ("--estimate", "--resolve"):
        result = _cli(mode, str(plan_path), cwd=tmp_path)

        assert result.returncode == 2, result.stderr
        assert "Traceback" not in result.stderr
        # The missing path, the plan key that named it, and the remedy.
        assert str(absent) in result.stderr
        assert "config.path" in result.stderr
        assert "config.inline" in result.stderr
        assert EXPLAIN_MARK.strip() not in result.stderr
        # stdout is the machine channel: a refused query prints no
        # document rather than a partial one.
        assert result.stdout == ""


# NEEDS CUPY INSTALLED, and opens no device: this test asserts the road
# classifies an absent configuration as a PlanError; without cupy it meets
# CapabilityMissing first.
@requires_cupy
def test_the_execution_road_calls_an_absent_config_a_plan_defect(tmp_path):
    """Same input, the road that runs.

    This one never crashed -- the event boundary catches everything --
    but it classed the failure ``FileNotFoundError``, whose remedy sends
    the reader to `woof check CONFIG` for a declared [case_data] input.
    The file that is absent is the config itself, and that remedy has
    nothing to say about it.
    """

    plan_path, absent = _absent_config_plan(tmp_path)
    result = _cli(str(plan_path), cwd=tmp_path)

    assert result.returncode == 1
    events = [json.loads(line) for line in result.stdout.splitlines()
              if line.strip()]
    assert events[-1]["event"] == "failed"
    assert events[-1]["error_class"] == "PlanError"
    assert "fix the plan document" in events[-1]["remedy"]
    assert str(absent) in events[-1]["message"]


# ---------------------------------------------------------------------------
# `--cycle latest` through the intent door
# ---------------------------------------------------------------------------


def test_an_intent_resolves_latest_for_a_reanalysis_through_the_real_door(
        tmp_path):
    """Studio drives this exact shape, and it used to be a refusal.

    ``run-plan --resolve`` on an era5 intent carrying ``cycle:
    "latest"`` exited 2 with "--cycle latest is only meaningful for
    gfs/gdas/hrrr; ERA5 is a reanalysis published with a delay of
    several days" -- a sentence about a delay, offered instead of the
    time that delay defines.  No network: the CDS publishes no object to
    probe, so the answer comes from the declared publication delay.
    """

    from datetime import datetime, timedelta, timezone

    plan_path = _intent_plan(tmp_path, tmp_path / "run", cycle="latest")
    result = _cli("--resolve", str(plan_path), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    resolved = [entry for entry in document["automatic_resolutions"]
                if entry["key"] == "cycle"]
    assert resolved, "the resolved cycle is not reported"
    assert resolved[0]["basis"] == "resolved_latest"
    # The literal query is never what runs: the concrete cycle is.
    assert resolved[0]["value"] != "latest"
    cycle = datetime.strptime(resolved[0]["value"], "%Y-%m-%dT%H")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    # Behind real time by the delay the registry row declares, and on
    # the analysis grid it declares, rather than at an arbitrary hour.
    assert cycle < now - timedelta(days=4)
    assert cycle.hour in (0, 6, 12, 18)
    # The note says which mechanism answered, because a reader told
    # "probed the mirrors" would go looking for a network step that
    # never happened.
    assert "declared publication delay" in resolved[0]["note"]
    # And the plan the wizard wrote starts AT that cycle.
    assert cycle.strftime("%Y-%m-%dT%H") in document["generated_config"]


def _mp28_experiment_with_no_dataset_anywhere(tmp_path, monkeypatch) -> Path:
    """An mp=28 config on a machine with no WIF climatology at all.

    Every rung of the resolver's ladder is pointed at a directory that
    does not exist, including the working-directory rung, which is WRF's
    own ``constants_name`` rule.
    """

    from test_case_data import _EXPERIMENT_TOML

    from woof.ingest import wif_climatology

    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    monkeypatch.setenv("WOOF_WIF_DATA_ROOT", str(tmp_path / "no-staged-wif"))
    monkeypatch.setenv("HOME", str(tmp_path / "no-staged-wif"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "no-staged-wif"))

    case = tmp_path / "case"
    case.mkdir()
    monkeypatch.chdir(case)
    return make_case_toml(
        case, experiment=_EXPERIMENT_TOML + "moist = true\nmp_physics = 28\n")


def test_a_sealed_bundle_is_not_refused_for_a_dataset_it_never_opens(
        tmp_path, monkeypatch):
    """THE DEFECT: prepared:existing was asked what only a preparation needs.

    The WIF climatology is read while a forecast is PREPARED.  A sealed
    bundle is already prepared: ``_existing_prepared_forecast`` hands it
    to the runner, which verifies the payload and opens no dataset.  Plan
    review asked the preparation preconditions of every chain anyway, so
    an mp=28 bundle prepared on a machine that had the 225 MB file and
    carried to one that does not -- which is the entire point of a sealed
    bundle -- was refused for an input its run never opens.

    Both arms are driven, because "it was admitted" alone is also what a
    machine that HAS the dataset would say: the same config with no
    bundle is still refused in the same environment, which is the
    positive evidence that the dataset really is absent here.
    """

    from test_stage_seams import _tree_bundle

    config = _mp28_experiment_with_no_dataset_anywhere(tmp_path, monkeypatch)
    prepared = _tree_bundle(tmp_path / "prepared", domains=1)

    def _plan(**options):
        return build_plan({
            "schema": PLAN_SCHEMA, "name": "sealed bundle, no dataset",
            "route": "prepared" if options else "experiment",
            "config": {"path": str(config)},
            "output_root": str(tmp_path / ("run" + str(len(options)))),
            "run_options": {"render_products": "none", **options},
        }, source="test plan", base_dir=config.parent, sha256="a" * 64)

    # 1. THE CHAIN THAT PREPARES is still refused, in one sentence.
    with pytest.raises(PlanError) as refusal:
        resolve_plan(_plan(), require_inputs=False)
    assert "QNWFA_QNIFA_SIGMA_MONTHLY.dat" in str(refusal.value)

    # 2. THE CHAIN THAT CONSUMES A SEALED BUNDLE is admitted.
    resolution, exp, _data = resolve_plan(
        _plan(prepared_root=str(prepared)), require_inputs=False)
    assert exp.root.run.mp_physics == 28
    assert exp.root.run.specified is True
    assert not (tmp_path / "run1").exists()
    assert ("execution", "prepared_root") in {
        (entry["scope"], entry["key"])
        for entry in resolution["automatic_resolutions"]}

@pytest.mark.parametrize("stop, code", [
    ("go_interrupted", 130),   # Ctrl-C landed while waiting on the stage
    ("stage_sigint", 130),     # the stage answered the same SIGINT first
    ("stage_exit_130", 130),   # ... as a Python stage does, with 130
    ("stage_failed", 1),       # a real failure is still a failure
    ("go_returned_130", 130),  # the in-process chain answered the stop itself
])
def test_a_stop_during_a_stage_exits_130_like_a_stop_between_stages(
        tmp_path, monkeypatch, stop, code):
    """The desktop reads the worker's exit code back from its receipts.

    Its saved-run reader calls 130 "stopped" and every other nonzero
    code "failed", and offers downscaling only from a completed or
    stopped run.  ``run_stage`` raises ``GoInterrupted`` (not a
    ``KeyboardInterrupt``) when the Ctrl-C lands while it waits on a
    stage subprocess, which is where a stop during fetch, prepare,
    forecast or render lands, so the run used to exit 1 and lose its
    downscale door the next time the desktop opened.
    """
    import woof.go_cli as go_cli
    from woof import capabilities
    from woof.runplan import StageExitError

    monkeypatch.setattr(capabilities, "require", lambda *args, **kwargs: None)
    raised = {
        "go_interrupted": go_cli.GoInterrupted("render", 4242),
        "stage_sigint": StageExitError("render", -2),
        "stage_exit_130": StageExitError("render", 130),
        "stage_failed": StageExitError("render", 2),
    }.get(stop)

    def fake_go_main(args, *, observer=None, **_):
        if stop == "go_returned_130":
            # What `woof go` does with its own interrupt: names the
            # stage, returns 130.
            observer.stage_begin(label="render", command=["rw_wrfbatch"])
            return 130
        raise raised

    monkeypatch.setattr(go_cli, "go_main", fake_go_main)
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        assert execute_plan(plan, events=events) == code
    failed = read_events(plan.run_dir / EVENTS_FILENAME)[-1]
    assert failed["event"] == "failed"
    assert failed["interrupted"] is (code == 130)
    assert failed["exit_code"] == (130 if code == 130 else 2)
    if stop == "go_interrupted":
        assert failed["error_class"] == "GoInterrupted"
        assert failed["message"] == "interrupted during render"
    if stop == "go_returned_130":
        assert failed["error_class"] == "ChainInterrupted"
        assert failed["message"] == "interrupted during render"


# ---------------------------------------------------------------------------
# The warning vocabulary
# ---------------------------------------------------------------------------
#
# WHAT BREAKAGE THIS PINS (gate law).  ``warning`` is ONE tag, and what a
# reader switches on is the ``code`` inside it.  That code was a free
# string documented nowhere: a route could invent one, and the record it
# wrote was a line every other reader of the stream dropped on the floor.
# The tags have been a closed tuple since this module existed; the codes
# now have the same guarantee, checked against the package's own source
# so a code added to a route and not to the table fails here.


def _emitted_warning_codes() -> dict[str, set[str]]:
    """Every literal ``warn``/``emit`` code in the package, by module."""

    import re

    import woof

    package = Path(woof.__file__).parent
    # The two spellings a code is written in: an observer's
    # ``warn("code", ...)`` -- or the private ``_warn`` a publisher holds
    # -- and the direct ``emit("warning", code="code", ...)``.  A code
    # ASSEMBLED at runtime matches neither, which is why the one family
    # that does that is spelled as a prefix in the table.
    pattern = re.compile(
        r'(?:\b_?warn\(\s*"([a-z0-9_]+)"'
        r'|emit\(\s*"warning"\s*,\s*code\s*=\s*"([a-z0-9_]+)")')
    found: dict[str, set[str]] = {}
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for first, second in pattern.findall(text):
            found.setdefault(first or second, set()).add(
                str(path.relative_to(package).as_posix()))
    return found


def test_every_warning_code_this_package_emits_is_in_the_vocabulary():
    from woof.runplan import WARNING_CODES, WARNING_CODE_PREFIXES

    undocumented = {
        code: sorted(modules)
        for code, modules in _emitted_warning_codes().items()
        if code not in WARNING_CODES
        and not code.startswith(WARNING_CODE_PREFIXES)}
    assert undocumented == {}, (
        "a warning code reached the event stream without a line in "
        "woof.runplan.WARNING_CODES saying what it means")


def test_every_documented_warning_code_says_what_it_means():
    from woof.runplan import WARNING_CODES

    assert WARNING_CODES
    for code, meaning in WARNING_CODES.items():
        assert code == code.lower() and " " not in code, code
        assert isinstance(meaning, str) and len(meaning) > 20, code


def test_a_run_that_did_not_finish_has_a_code_for_its_kept_pictures():
    """The code this lane added, and the fields a reader keys on: how
    many pictures are on disk and where the banner beside them is."""

    from woof.runplan import WARNING_CODES

    meaning = WARNING_CODES["early_render_kept"]
    assert "KEPT" in meaning
    assert "banner" in meaning


_LATE_LEAD = {"source": "gefs", "cycle": "2026-09-30T12", "lead": 30,
              "valid_time": "2026-10-01T18:00:00Z",
              "expected_at": "2026-09-30T15:53:00Z",
              "late_at": "2026-09-30T16:53:00Z", "late_after_minutes": 60,
              "last_answer": "not_posted",
              "model_elapsed_seconds": 97200.0,
              "model_valid_time": "2026-10-01T15:00:00Z",
              "frames_kept": 28, "checkpoint": None}


def test_every_posting_and_wait_tag_is_allowlisted_with_its_fields():
    """A136: a consumer switching on ``event`` can be exhaustive."""

    from woof.runplan import POSTING_EVENT_FIELDS

    for tag in ("posting_schedule", "lead_posted", "lead_ready",
                "source_wait_started", "source_wait_progress",
                "source_wait_finished", "boundary_wait_started",
                "boundary_wait_finished", "source_behind"):
        assert tag in EVENT_TAGS
        assert POSTING_EVENT_FIELDS[tag]
    # Model time on every wait, and the cause on the preparation waits.
    for tag in ("source_wait_started", "source_wait_progress",
                "source_wait_finished", "boundary_wait_started",
                "boundary_wait_finished", "source_behind"):
        assert {"model_elapsed_seconds", "model_valid_time"} <= set(
            POSTING_EVENT_FIELDS[tag])
    assert "cause" in POSTING_EVENT_FIELDS["boundary_wait_started"]


@pytest.mark.parametrize("how", ["observer", "error"])
def test_a_source_that_falls_behind_exits_75_naming_the_lead(
        tmp_path, monkeypatch, how):
    """``source_behind`` then ``failed`` (``SourceBehind``), and exit 75.

    Before this, run-plan returned 1 for every failure, so a site could
    not tell a lead later than its budget (launch again once it posts)
    from a broken run.
    """

    import woof.go_cli as go_cli
    from woof import capabilities
    from woof.ingest.boundary_stream import SourceBehind

    monkeypatch.setattr(capabilities, "require", lambda *args, **kwargs: None)

    def fake_go_main(args, *, observer=None, **_):
        observer.stage_begin(label="forecast", command=["runner"])
        if how == "observer":
            # The hosted forecast tells its observer, then the chain stops
            # at the stage with the runner's exit code.
            observer.source_behind(dict(_LATE_LEAD))
            raise go_cli.GoStageFailed(75)
        raise SourceBehind(_LATE_LEAD)

    monkeypatch.setattr(go_cli, "go_main", fake_go_main)
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        assert execute_plan(plan, events=events) == 75
    records = read_events(plan.run_dir / EVENTS_FILENAME)
    behind, failed = records[-2], records[-1]
    assert behind["event"] == "source_behind"
    assert behind["lead"] == 30 and behind["late_after_minutes"] == 60
    assert behind["frames_kept"] == 28
    assert failed["event"] == "failed"
    assert failed["error_class"] == "SourceBehind"
    assert failed["exit_code"] == 75
    assert failed["message"].startswith(
        "gefs f030 of the 2026-09-30T12 cycle has not posted by 16:53Z")
    assert "The forecast stopped at 27:00" in failed["message"]


def test_a_staged_chain_render_plan_carries_the_config_data_source(tmp_path):
    """THE BREAKAGE: `woof go` renders an HRRR run through the staged chain,
    whose render plan named no data source, so finalize could not draw the
    run beside HRRR (measured on a real 1 h HRRR run on a 4090, 2026-10-05)."""
    from woof.runplan import _chain_render_plan

    def plan_for(config_text):
        return build_plan(
            {"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
             "config": {"inline": config_text}},
            source="probe.json", base_dir=tmp_path, sha256="0" * 64)

    forecast = tmp_path / "chain" / "run"
    sourced = _chain_render_plan(plan_for('[fetch]\nsource = "hrrr"\n'),
                                 forecast_dir=forecast, run_dir=tmp_path)
    assert sourced["source"] == "hrrr"
    assert sourced["render"] == forecast.parent / "png"
    unsourced = _chain_render_plan(plan_for("x = 1\n"),
                                   forecast_dir=forecast, run_dir=tmp_path)
    assert "source" not in unsourced
