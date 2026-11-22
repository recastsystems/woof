"""Research recommendations must remain connected to actual native capabilities.

These CPU checks qualify catalog structure and configuration semantics only.
They do not qualify GPU fit, preparation, forecast skill or physical outcomes.
The renderer-owned inventory is deliberately independent of family presets.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import hashlib
import math
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "woof/data/tui/research-workspaces.json").read_text(encoding="utf-8"))
CONFIGS = {row["id"]: row for row in CATALOG["configurations"]}
FAMILIES = {row["id"]: row for row in CATALOG["families"]}
SUBMODES = {row["id"]: row for row in CATALOG["submodes"]}
LEAVES = {row["id"]: row for row in CATALOG["leaves"]}
INVENTORY = json.loads((ROOT / "tools/rustwx/crates/rustwx-products/tests/fixtures/"
                       "product_catalog_inventory_v1.json").read_text(encoding="utf-8"))
NATIVE_PRODUCTS = {name for lane in INVENTORY["lanes"].values() for name in lane}
CAPABILITIES = json.loads((ROOT / "woof/data/tui/research-diagnostics.json").read_text(encoding="utf-8"))
RESEARCH_PRODUCTS = set(CAPABILITIES["products"])
METHODS = {"regional", "nested", "archived_downscale", "moving_nest", "controlled_scenario"}


def _family(recipe):
    return SUBMODES[LEAVES[recipe["leaf_id"]]["submode_id"]]["family_id"]


def _scientific_fingerprint(recipe):
    geometry = recipe["geometry"]
    return json.dumps({
        "method": recipe["method"],
        "geometry": {key: geometry[key] for key in (
            "root_dx_km", "nest_ratios", "forecast_hours", "history_interval_s")},
        "tracker": recipe["tracker"], "scenario": recipe["scenario"],
    }, sort_keys=True)


def _native_follow(tracker):
    from woof.core.storm_tracking import build_follow_config
    values = {"field": tracker["field"], "threshold": tracker["threshold"],
              "search_margin_cells": 10, "min_shift_cells": 1,
              "max_shift_cells": 4, "cooldown_seconds": 300.0}
    for key in ("level_hpa", "fallback_threshold"):
        if tracker[key] is not None:
            values[key] = tracker[key]
    return build_follow_config(values, "research-catalog-check")


def _assert_tracker_units(tracker):
    field = tracker["field"]
    expected = "m2/s2" if field == "uh" else "dBZ" if field == "reflectivity" else (
        "hPa" if tracker["level_hpa"] == 0 else "m")
    assert tracker["units"] == expected


def test_complete_hierarchy_matches_tui_families_and_has_no_orphans():
    assert CATALOG["schema_version"] == 1
    assert (len(FAMILIES), len(LEAVES), len(CONFIGS)) == (12, 38, 114)
    for name in ("families", "submodes", "leaves", "configurations", "citations"):
        rows = CATALOG[name]
        assert len({row["id"] for row in rows}) == len(rows), name
    modes = (ROOT / "tools/arwen-tui/src/workflows.rs").read_text(encoding="utf-8")
    assert set(re.findall(r'Mode\s*\{\s*id:\s*"([^"]+)"', modes)) == set(FAMILIES)
    for family in FAMILIES.values():
        assert set(family["submode_ids"]) == {
            row["id"] for row in SUBMODES.values() if row["family_id"] == family["id"]}
    for submode in SUBMODES.values():
        assert submode["family_id"] in FAMILIES
        assert set(submode["leaf_ids"]) == {
            row["id"] for row in LEAVES.values() if row["submode_id"] == submode["id"]}
    for leaf in LEAVES.values():
        assert leaf["submode_id"] in SUBMODES
        assert set(leaf["configuration_ids"]) == {
            row["id"] for row in CONFIGS.values() if row["leaf_id"] == leaf["id"]}
    assert all(recipe["leaf_id"] in LEAVES for recipe in CONFIGS.values())


def test_parent_recommendations_stay_within_their_own_research_branch():
    for family in FAMILIES.values():
        ids = family["recommended_config_ids"]
        assert len(ids) >= 3 and len(ids) == len(set(ids))
        assert all(_family(CONFIGS[key]) == family["id"] for key in ids)
    for submode in SUBMODES.values():
        ids = submode["recommended_config_ids"]
        assert len(ids) >= 3 and len(ids) == len(set(ids))
        assert all(LEAVES[CONFIGS[key]["leaf_id"]]["submode_id"] == submode["id"] for key in ids)
    for leaf in LEAVES.values():
        ids = leaf["configuration_ids"]
        assert len(ids) == len(set(ids)) == 3
        assert len({_scientific_fingerprint(CONFIGS[key]) for key in ids}) == 3, leaf["id"]
        assert len({CONFIGS[key]["research_question"] for key in ids}) == 3


@pytest.mark.parametrize("recipe", CATALOG["configurations"], ids=lambda row: row["id"])
def test_recipe_uses_native_products_and_valid_scientific_geometry(recipe):
    from woof.research_workspaces import validate_recipe
    known = RESEARCH_PRODUCTS
    # Do not reduce this to a family preset: valid custom diagnostics may
    # supplement the preset, and the renderer is the selector authority.
    validate_recipe(recipe, known_products=known)
    assert len(recipe["diagnostics"]) == len(set(recipe["diagnostics"]))
    assert recipe["method"] in METHODS
    geometry = recipe["geometry"]
    assert math.isfinite(geometry["minimum_root_span_km"]) and geometry["minimum_root_span_km"] > 0
    finest = geometry["root_dx_km"] / math.prod(geometry["nest_ratios"])
    assert finest == pytest.approx(geometry["preferred_finest_dx_km"])
    assert geometry["forecast_hours"] * 3600 >= geometry["history_interval_s"]
    assert len(geometry["nest_ratios"]) <= 2
    assert all(ratio in (2, 3, 4) for ratio in geometry["nest_ratios"])
    assert recipe["qualification_status"] == "unqualified"
    assert recipe["validation_status"] == "unvalidated"  # legacy export alias
    assert recipe["comparison"] and recipe["limitations"] and recipe["input_requirements"]
    if recipe["method"] == "archived_downscale":
        assert recipe["requires_existing_state"]


@pytest.mark.parametrize("field,value,canonical,legacy", [
    ("qualification_status", "unqualified", "unqualified", "unvalidated"),
    ("validation_status", "unvalidated", "unqualified", "unvalidated"),
    ("qualification_status", "unqualified-candidate-policy",
     "unqualified-candidate-policy", "unvalidated-candidate-policy"),
    ("validation_status", "unvalidated-candidate-policy",
     "unqualified-candidate-policy", "unvalidated-candidate-policy"),
    ("validation_status", "catalog recommendations are not science validation",
     "unqualified", "catalog recommendations are not science validation"),
])
def test_catalog_accepts_canonical_and_legacy_qualification_rows(
        tmp_path, monkeypatch, field, value, canonical, legacy):
    from woof import research_workspaces as research
    document = deepcopy(CATALOG)
    for row in document["configurations"]:
        row.pop("qualification_status", None)
        row.pop("validation_status", None)
        row[field] = value
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(research, "CATALOG_PATH", path)
    restored = research.catalog_document()
    assert len(restored["configurations"]) == len(CONFIGS)
    for row in restored["configurations"]:
        assert row["qualification_status"] == canonical
        assert row["validation_status"] == legacy
        assert _scientific_fingerprint(row) == _scientific_fingerprint(CONFIGS[row["id"]])


def test_conflicting_qualification_aliases_are_refused_before_recipe_use():
    from woof.research_workspaces import validate_recipe
    recipe = deepcopy(next(iter(CONFIGS.values())))
    recipe["qualification_status"] = "qualified"
    recipe["validation_status"] = "unvalidated"
    with pytest.raises(ValueError, match="qualification_status and legacy validation_status disagree"):
        validate_recipe(recipe)


def test_tracker_fields_units_and_fallbacks_match_the_native_contract():
    trackers = [row["tracker"] for row in CONFIGS.values() if row["tracker"]]
    assert {row["field"] for row in trackers} == {"pressure", "uh", "reflectivity"}
    for recipe in CONFIGS.values():
        tracker = recipe["tracker"]
        assert (tracker is not None) == (recipe["method"] == "moving_nest")
        if tracker is None:
            continue
        _assert_tracker_units(tracker)
        native = _native_follow(tracker)
        assert native.field == tracker["field"]
        assert native.threshold == tracker["threshold"]
        if native.field == "pressure":
            assert native.level_hpa == (850.0,)
            assert tracker["units"] == "m"
            assert native.fallback_threshold is None
        elif native.field == "uh":
            assert native.fallback_threshold == tracker["fallback_threshold"]
        else:
            assert native.level_hpa is None and native.fallback_threshold is None
        assert Fraction(str(tracker["cadence_seconds"])) % Fraction(
            str(recipe["geometry"]["history_interval_s"])) == 0


def test_tracker_negative_controls_reject_new_signals_and_changed_units():
    from woof.research_workspaces import validate_recipe
    tracker = deepcopy(CONFIGS["rotation.follow"]["tracker"])
    tracker["field"] = "hail"
    with pytest.raises(ValueError):
        _native_follow(tracker)
    tracker = deepcopy(CONFIGS["rotation.follow"]["tracker"])
    tracker["fallback_threshold"] = None
    with pytest.raises(ValueError):
        _native_follow(tracker)
    tracker = deepcopy(CONFIGS["tropical-track.follow"]["tracker"])
    tracker["level_hpa"] = 0
    with pytest.raises(ValueError):
        _native_follow(tracker)  # 20 metres cannot become a 20 hPa ceiling.
    tracker = deepcopy(CONFIGS["tropical-track.follow"]["tracker"])
    tracker["units"] = "hPa"
    with pytest.raises(AssertionError):
        _assert_tracker_units(tracker)
    recipe = deepcopy(CONFIGS["tropical-track.follow"])
    recipe["tracker"] = tracker
    with pytest.raises(ValueError, match="units"):
        validate_recipe(recipe, known_products=RESEARCH_PRODUCTS)


def test_warm_bubble_comparisons_use_only_native_fields_and_isolate_changes():
    from woof.experiment import BubbleConfig
    scenarios = [row for row in CONFIGS.values() if row["scenario"]]
    assert len(scenarios) == 3
    native = []
    for recipe in scenarios:
        declaration = recipe["scenario"]
        assert recipe["method"] == "controlled_scenario"
        assert declaration["operation"] == "warm_bubble"
        assert declaration["placement"] == "user-selected domain centre"
        values = {key: value for key, value in declaration.items() if key not in ("operation", "placement")}
        native.append(BubbleConfig(center_lat=35.3, center_lon=-97.5, **values).receipt())
    reference = native[0]
    assert {key for key in reference if reference[key] != native[1][key]} == {"amplitude_k"}
    assert {key for key in reference if reference[key] != native[2][key]} == {"rh_preserve"}
    assert native[1]["amplitude_k"] == 2 * reference["amplitude_k"]
    for recipe in CONFIGS.values():
        if recipe["leaf_id"] == "scenario-tropical-input":
            assert recipe["requires_existing_state"] and recipe["scenario"] is None
            assert "No hurricane insertion" in " ".join(recipe["limitations"])


def test_hardware_profiles_preserve_scientific_window_and_extent_constraints():
    from woof.research_workspaces import effective_profile
    for recipe in CONFIGS.values():
        for hardware in ("8", "12", "16", "24", "32"):
            profile = effective_profile(recipe, hardware)
            original, actual = recipe["geometry"], profile["geometry"]
            for key in ("forecast_hours", "history_interval_s", "minimum_root_span_km", "extent_intent"):
                assert actual[key] == original[key], (recipe["id"], hardware, key)
            assert profile["effective_finest_dx_km"] == pytest.approx(
                actual["root_dx_km"] / math.prod(actual["nest_ratios"]))
            assert profile["qualification_status"] == "unqualified-candidate-policy"
            assert profile["validation_status"] == "unvalidated-candidate-policy"
            assert profile["tradeoff"] and profile["resolution_tradeoff"]
            if recipe["scenario"]:
                assert actual == original and profile["nz"] == 49
            elif recipe["requires_existing_state"]:
                assert actual == original and profile["nz"] is None
                assert profile["parent_dependent"]


def test_sources_are_resolvable_and_public_reference_covers_every_recipe():
    citations = {row["id"]: row for row in CATALOG["citations"]}
    public = (ROOT / "docs/public/RESEARCH-WORKSPACES.md").read_text(encoding="utf-8")
    for recipe in CONFIGS.values():
        assert recipe["id"] in public
        assert recipe["citation_ids"]
        for key in recipe["citation_ids"]:
            source = citations[key]
            assert source["title"] and source["publisher"] and source["supports"]
            if not source["url"].startswith("https://"):
                # the catalog is a data record and keeps the engine's package path (gpuwm/...);
                # in this tree the package is woof/
                url = source["url"]
                if url.startswith("gpuwm/") and not (ROOT / url).exists():
                    url = "woof/" + url[len("gpuwm/"):]
                from _woof_snapshot import left_out
                if left_out(url):
                    continue  # a cited source this tree leaves out (the terminal UI's own sources)
                assert (ROOT / url).is_file(), source["url"]


def test_research_capability_manifest_is_bound_to_the_actual_import_and_window_contracts():
    assert CAPABILITIES["schema"] == "arwen.research.diagnostics.v1"
    assert CAPABILITIES["authority_normalization"] == "UTF-8 with LF line endings"
    for path, expected in CAPABILITIES["authority"].items():
        assert hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest() == expected, (
            f"{path}: review the import/window contracts, then run "
            "python -m tools.battery.record_research_authority")
    assert {name for name in RESEARCH_PRODUCTS if not name.startswith("var:")} <= NATIVE_PRODUCTS
    assert {name for name in RESEARCH_PRODUCTS if name.startswith("var:")} == {
        "var:wrf_lapse_rate_0_3km", "var:wrf_lapse_rate_700_500"}
    assert not RESEARCH_PRODUCTS & set(CAPABILITIES["unavailable"])
    for recipe in CONFIGS.values():
        for name in recipe["diagnostics"]:
            assert recipe["geometry"]["forecast_hours"] >= CAPABILITIES["products"][name]["minimum_hours"]
        if any(name.startswith("var:wrf_lapse_rate") for name in recipe["diagnostics"]):
            assert "plain temperature" in " ".join(recipe["further_analysis"])
            assert "virtual-temperature" in " ".join(recipe["further_analysis"])


@pytest.mark.parametrize("product", sorted(CAPABILITIES["unavailable"]))
def test_all_source_vocabulary_cannot_admit_an_unserved_history_quantity(product):
    from woof.research_workspaces import validate_recipe
    recipe = deepcopy(CONFIGS["regional-evolution.reference"])
    recipe["diagnostics"] = [product]
    assert product in NATIVE_PRODUCTS
    with pytest.raises(ValueError, match="not supported by the WOOF-ArWen history renderer"):
        validate_recipe(recipe, known_products=NATIVE_PRODUCTS)


def test_six_hour_study_cannot_promise_a_daily_extremum():
    from woof.research_workspaces import validate_recipe
    recipe = deepcopy(CONFIGS["radiation-frost.dawn"])
    assert recipe["geometry"]["forecast_hours"] == 6
    assert "2m_temp_0_24h_min" not in recipe["diagnostics"]
    recipe["diagnostics"].append("2m_temp_0_24h_min")
    with pytest.raises(ValueError, match="needs at least 24 hours"):
        validate_recipe(recipe)


@pytest.mark.parametrize("target,damage", [
    ("CATALOG_PATH", "missing"), ("CATALOG_PATH", "truncated"),
    ("CATALOG_PATH", "utf8"), ("CATALOG_PATH", "root"),
    ("CATALOG_PATH", "dangling"), ("CATALOG_PATH", "duplicate"),
    ("HARDWARE_PATH", "missing"), ("HARDWARE_PATH", "schema"),
    ("HARDWARE_PATH", "classes"), ("HARDWARE_PATH", "numeric"),
    ("HARDWARE_PATH", "duplicate_key"),
    ("DIAGNOSTICS_PATH", "missing"), ("DIAGNOSTICS_PATH", "schema"),
])
def test_damaged_packaged_metadata_has_a_named_public_refusal(tmp_path, monkeypatch, capsys, target, damage):
    from woof import research_workspaces as research
    from woof.cli import main
    path = tmp_path / getattr(research, target).name
    document = json.loads(getattr(research, target).read_bytes())
    if damage == "schema": document["schema"] = "unknown-v999"
    if damage == "classes": document["profiles"].pop("8")
    if damage == "numeric": document["profiles"]["8"]["storm"]["root_dx_km"] = "four"
    if damage == "dangling": document["configurations"][0]["leaf_id"] = "missing-leaf"
    if damage == "duplicate": document["configurations"].append(deepcopy(document["configurations"][0]))
    if damage != "missing":
        payload = (b"{" if damage == "truncated" else b"\xff" if damage == "utf8" else b"[]" if damage == "root" else
                   b'{"schema":"a","schema":"b"}' if damage == "duplicate_key" else json.dumps(document).encode())
        path.write_bytes(payload)
    monkeypatch.setattr(research, target, path)
    assert main(["research", "catalog", "--json"]) == 2
    captured = capsys.readouterr()
    assert str(path) in captured.err
    assert "Restore" in captured.err or "restore" in captured.err
    assert "Traceback" not in captured.err + captured.out


@pytest.mark.parametrize("companion", [False, True])
def test_public_research_collision_is_plain_and_precedes_sizing(tmp_path, monkeypatch, capsys, companion):
    from woof import domain_wizard as wizard
    from woof.cli import main
    output = tmp_path / "study.toml"
    existing = output.with_name(output.name + ".arwen-plots.json") if companion else output
    existing.write_bytes(b"do not replace")
    monkeypatch.setattr(wizard, "resolve_sizing_budget", lambda *a, **k: pytest.fail("collision reached sizing"))
    assert main(["research", "create", "regional-evolution.reference", "--point=35.3,-97.5",
                 "--cycle=2026-09-05T18", "--vram-gib=8", "--out=" + str(output)]) == 2
    captured = capsys.readouterr()
    assert "Choose a new output filename" in captured.err
    assert str(existing) in captured.err and "Traceback" not in captured.err
    assert existing.read_bytes() == b"do not replace"
    assert set(tmp_path.iterdir()) == {existing}


def test_creation_does_not_need_a_renderer_and_binds_the_metadata_actually_read(tmp_path, monkeypatch):
    from woof import domain_wizard as wizard, research_workspaces as research, tui_products
    from woof.cli import build_parser
    catalog = tmp_path / "catalog.json"; hardware = tmp_path / "hardware.json"
    catalog.write_bytes(research.CATALOG_PATH.read_bytes()); hardware.write_bytes(research.HARDWARE_PATH.read_bytes())
    original_hashes = {"catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
                       "hardware_policy_sha256": hashlib.sha256(hardware.read_bytes()).hexdigest()}
    monkeypatch.setattr(research, "CATALOG_PATH", catalog); monkeypatch.setattr(research, "HARDWARE_PATH", hardware)
    monkeypatch.setattr(tui_products, "catalog_document", lambda: pytest.fail("creation launched the renderer catalog"))
    original = wizard.domain_main
    def change_metadata_after_compile(*args, **kwargs):
        result = original(*args, **kwargs)
        document = json.loads(catalog.read_bytes()); document["configurations"][0]["title"] = "Changed after compile"
        catalog.write_text(json.dumps(document), encoding="utf-8")
        document = json.loads(hardware.read_bytes()); document["profiles"]["8"]["nz"] = 64
        hardware.write_text(json.dumps(document), encoding="utf-8")
        return result
    monkeypatch.setattr(wizard, "domain_main", change_metadata_after_compile)
    output = tmp_path / "study.toml"
    args = build_parser().parse_args(["research", "create", "regional-evolution.reference", "--point=35.3,-97.5",
        "--cycle=2026-09-05T18", "--hardware-class=8", "--vram-gib=8", "--out=" + str(output)])
    receipt = research.create_workspace(args)
    assert output.is_file() and receipt["admission"]["forecast_started"] is False
    assert all(receipt[key] == value for key, value in original_hashes.items())
    assert receipt["recipe"]["title"] == CONFIGS["regional-evolution.reference"]["title"]
    assert receipt["effective_profile"]["nz"] == 49
    assert receipt["catalog_sha256"] != hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert receipt["hardware_policy_sha256"] != hashlib.sha256(hardware.read_bytes()).hexdigest()


def _unrecorded_native_product() -> str:
    """A product the renderer's catalog carries and the record does not.

    Chosen from the two tables rather than hard-coded, so completing the
    record's isobaric rows cannot turn this check green by accident.
    """
    unlisted = sorted(NATIVE_PRODUCTS - RESEARCH_PRODUCTS
                      - set(CAPABILITIES["unavailable"]))
    assert unlisted, "every catalog product is recorded; this check is moot"
    return unlisted[0]


def test_a_stored_isobaric_chart_is_not_refused_for_being_unlisted(capsys):
    """The record is not the vocabulary; its absence refuses nothing."""
    from woof.research_workspaces import validate_recipe

    recipe = deepcopy(CONFIGS["regional-evolution.reference"])
    recipe["diagnostics"] = ["500mb_temperature_height_winds"]
    assert "500mb_temperature_height_winds" in NATIVE_PRODUCTS
    validate_recipe(recipe)

    # And the mechanism itself, over a product nothing recorded at all:
    # it runs, priced from the most conservative recorded basis, with
    # that basis stated and one warning.
    unrecorded = _unrecorded_native_product()
    recipe["diagnostics"] = [unrecorded]
    capsys.readouterr()
    validate_recipe(recipe)
    err = capsys.readouterr().err
    assert "warning:" in err and unrecorded in err
    assert "no recorded window requirement" in err


def test_the_isobaric_rows_of_the_record_are_complete():
    """Table work: the stored isobaric charts carry their own row."""
    for level in ("200mb", "250mb", "300mb", "500mb", "700mb", "850mb"):
        assert f"{level}_height_winds" in RESEARCH_PRODUCTS, level
    for slug in ("500mb_temperature_height_winds", "300mb_rh_height_winds",
                 "850mb_dewpoint_height_winds"):
        assert CAPABILITIES["products"][slug]["minimum_hours"] == 0, slug


def test_a_recorded_refusal_names_the_product_and_its_reason():
    """The bare sentence named neither, so nobody could act on it."""
    from woof.research_workspaces import validate_recipe

    recipe = deepcopy(CONFIGS["regional-evolution.reference"])
    recipe["diagnostics"] = ["dcape"]
    with pytest.raises(ValueError, match="dcape.*downdraft energy"):
        validate_recipe(recipe)


def test_a_slug_outside_the_renderers_catalog_is_still_refused():
    """When a caller HAS the catalog, a name outside it is missing."""
    from woof.research_workspaces import validate_recipe

    recipe = deepcopy(CONFIGS["regional-evolution.reference"])
    recipe["diagnostics"] = ["not_a_product_any_engine_carries"]
    with pytest.raises(ValueError, match="not supported by the WOOF-ArWen history renderer"):
        validate_recipe(recipe, known_products=NATIVE_PRODUCTS)


def test_both_doors_read_one_lane_record():
    """The recipe validator and the preset picker, one function."""
    from woof import research_workspaces, tui_products

    assert (research_workspaces.diagnostic_capabilities()
            == tui_products.lane_capabilities(research_workspaces.DIAGNOSTICS_PATH))
    assert research_workspaces.DIAGNOSTICS_PATH == tui_products.DIAGNOSTICS_PATH
