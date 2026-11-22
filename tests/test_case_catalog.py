"""Case data cannot replace scientific settings or overwrite an existing study."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
import zipfile

import pytest

from woof import case_catalog as catalog


EXAMPLES = Path(catalog.__file__).parent / "data" / "case-catalog"
NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)


@pytest.fixture
def document():
    return json.loads((EXAMPLES / "example.json").read_text(encoding="utf-8"))


def write_catalog(tmp_path, document, name="cases.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_json_and_toml_examples_are_the_same_data_and_keep_original_bytes():
    left = catalog.load_catalog(EXAMPLES / "example.json")
    right = catalog.load_catalog(EXAMPLES / "example.toml")
    assert left.document == right.document
    assert left.original == (EXAMPLES / "example.json").read_bytes()
    assert right.original == (EXAMPLES / "example.toml").read_bytes()
    assert left.sha256 == hashlib.sha256(left.original).hexdigest()
    assert left.sha256 != right.sha256


def test_bundled_historical_catalog_preserves_all_earlier_cases_and_newer_records():
    from tools.build_builtin_case_catalog import _json_bytes
    path = catalog.builtin_catalog_path()
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"catalog.json", "merge-provenance.json"}
        payload = archive.read("catalog.json")
        document = json.loads(payload)
        provenance = json.loads(archive.read("merge-provenance.json"))
    assert provenance["merged_catalog_sha256"] == hashlib.sha256(payload).hexdigest()
    assert document["schema"] == "arwen.case-catalog/v2"
    assert len(document["cases"]) == provenance["case_count"] == 300
    ids = {row["id"] for row in document["cases"]}
    assert len(ids) == 300
    assert len(provenance["older_case_ids_retained"]) == 200
    assert set(provenance["older_case_ids_retained"]) <= ids
    assert len(provenance["new_case_ids_added"]) == 100
    assert provenance["older_case_ids_missing"] == []
    assert [row["sha256"] for row in provenance["inputs"]] == [
        "4a746e5098aa5a641e1b66a948c35423a680a4825013ba20b9288732bb32d828",
        "818004eaa37973edd4eacd0dcf7608f490717f8fc0131d735dd8fa375b500bf8"]
    selected = {row["id"]: row["sha256"] for row in provenance["selected_records"]}
    assert selected == {row["id"]: hashlib.sha256(_json_bytes(row)).hexdigest()
                        for row in document["cases"]}
    assert document["metadata"]["bundled_merge"]["inputs"] == provenance["inputs"]


def test_builtin_merge_selects_whole_newer_records_and_is_reproducible(tmp_path):
    from tools.build_builtin_case_catalog import build_catalog
    def archive(name, cases):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as output:
            output.writestr("catalog.json", json.dumps({"schema": "arwen.case-catalog/v2",
                "title": name, "cases": cases, "metadata": {"preserved": "source metadata"}}))
        return path
    old = archive("old.zip", [{"id": "one", "category": "tornado", "old_only": 1,
                              "physics": {"mp": 8}}])
    new_cases = [{"id": "one", "category": "tornado", "physics": {"mp": 10}},
                 {"id": "two", "category": "synoptic", "note": "literal data"}]
    new = archive("new.zip", new_cases)
    payload, provenance = build_catalog(old, new)
    assert payload == build_catalog(old, new)[0]
    import io
    with zipfile.ZipFile(io.BytesIO(payload)) as output:
        merged = json.loads(output.read("catalog.json"))
    assert merged["cases"] == new_cases
    assert merged["metadata"]["preserved"] == "source metadata"
    assert provenance["duplicate_case_ids"] == ["one"]
    incomplete = archive("missing.zip", [new_cases[1]])
    with pytest.raises(ValueError, match="omits earlier case IDs"):
        build_catalog(old, incomplete)


def test_cli_uses_installed_catalog_default_from_an_unrelated_folder(tmp_path, monkeypatch, capsys):
    from woof.cli import main
    monkeypatch.chdir(tmp_path)
    assert main(["case-catalog", "default", "--json"]) == 0
    default = json.loads(capsys.readouterr().out)
    assert default["case_count"] == 300
    assert Path(default["path"]) == catalog.builtin_catalog_path()
    assert main(["case-catalog", "list", "--limit=1", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total"] == 300 and len(result["cases"]) == 1
    assert result["provenance"]["source"] == default["path"]
    assert result["cases"][0]["id"] == "tornado-2013-05-31-el-reno-oklahoma"
    assert main(["case-catalog", "list", "--catalog", str(EXAMPLES / "example.json"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 2
    assert not list(tmp_path.iterdir())


def test_missing_builtin_catalog_names_the_installation_remedy(tmp_path, monkeypatch, capsys):
    from woof.cli import main
    monkeypatch.setattr(catalog, "BUILTIN_CATALOG_PATH", tmp_path / "missing.zip")
    assert main(["case-catalog", "default", "--json"]) == 2
    assert "reinstall WOOF-ArWen" in json.loads(capsys.readouterr().out)["error"]


def test_validation_keeps_caller_data_and_accepts_exactly_the_same_text_controls(document):
    document["cases"][0]["event"]["start_utc"] = "2026-09-01T01:00:00+01:00"
    document["cases"][0]["metadata"] = {"nested": [{"text": "café 日本\tline\nreturn\r"}]}
    original = deepcopy(document)
    validated = catalog.validate_catalog(document)
    assert document == original
    assert validated["cases"][0]["event"]["start_utc"] == "2026-09-01T00:00:00Z"
    validated["cases"][0]["metadata"]["nested"][0]["text"] = "changed"
    assert document == original
    for codepoint in range(256):
        value = "left" + chr(codepoint) + "right"
        if codepoint < 32 and chr(codepoint) not in "\n\t\r":
            with pytest.raises(catalog.CatalogError, match="control character"):
                catalog._text(value, "test")
        else:
            assert catalog._text(value, "test") == value


def test_empty_and_metadata_search_preserve_case_order_and_filters(tmp_path, document):
    document["cases"][0]["metadata"] = {"nested": {"prose": "rare-metadata-token 日本"}}
    loaded = catalog.load_catalog(write_catalog(tmp_path, document))
    empty = catalog.list_cases(loaded)
    assert empty == catalog.list_cases(loaded, query=" \t\n")
    assert [row["id"] for row in empty["cases"]] == [row["id"] for row in document["cases"]]
    found = catalog.list_cases(loaded, query="rare-METADATA-token 日本")
    assert [row["id"] for row in found["cases"]] == [document["cases"][0]["id"]]
    assert catalog.list_cases(loaded, query="rare-metadata-token", event_kind="not-an-event")["total"] == 0


def test_catalog_session_detects_changed_bytes_even_with_the_same_size_and_timestamp(tmp_path, document, monkeypatch):
    import os
    path = write_catalog(tmp_path, document)
    session = catalog._CatalogSession()
    loads = []
    original_loader = catalog._load_catalog_bytes
    def record(path, raw):
        loads.append(hashlib.sha256(raw).hexdigest())
        return original_loader(path, raw)
    monkeypatch.setattr(catalog, "_load_catalog_bytes", record)
    first = session.load(path)
    assert session.load(path) is first and len(loads) == 1
    stamp = path.stat()
    before = path.read_bytes()
    title = document["cases"][0]["title"]
    replacement = ("X" if title[0] != "X" else "Y") + title[1:]
    changed = before.replace(json.dumps(title).encode(), json.dumps(replacement).encode(), 1)
    assert len(changed) == len(before) and changed != before
    path.write_bytes(changed)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = session.load(path)
    assert len(loads) == 2 and second.sha256 != first.sha256
    assert second.original == changed and second.document["cases"][0]["title"] == replacement
    detail = catalog.case_detail(second, document["cases"][0]["id"])
    detail["case"]["title"] = "A caller's edit"
    assert session.load(path).document["cases"][0]["title"] == replacement
    path.write_text("not a JSON catalog", encoding="utf-8")
    with pytest.raises(catalog.CatalogError):
        session.load(path)
    path.write_bytes(changed)
    assert session.load(path) is second


def test_catalog_worker_is_read_only_and_recovers_request_framing(tmp_path, document):
    import io
    path = write_catalog(tmp_path, document)
    malformed = proposal_document()
    malformed["cases"] = [None]
    malformed_path = write_catalog(tmp_path, malformed, "malformed-proposal.json")
    target = tmp_path / "must-not-create.toml"
    requests = [
        b"not json\n",
        b"[\"list\", false]\n",
        b"\xff\n",
        (json.dumps(["create", document["cases"][0]["id"], "--out", str(target)]) + "\n").encode(),
        b"[\"export\", \"--out\", \"ignored.json\"]\n",
        b"[\"list\", \"--help\"]\n",
        b"[\"list\", \"--unknown\"]\n",
        b"x" * (64 * 1024 + 10) + b"\n",
        b"[" * 2000 + b"]" * 2000 + b"\n",
        (json.dumps(["list", "--catalog", str(malformed_path)]) + "\n").encode(),
        (json.dumps(["list", "--catalog", str(path), "--json"]) + "\n").encode(),
    ]
    output = io.StringIO()
    assert catalog._tui_server(io.BytesIO(b"".join(requests)), output) == 0
    results = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(results) == len(requests)
    assert all(row["schema"] == "arwen.case-error.v1" and row["created"] is False
               for row in results[:-1])
    assert results[-1]["schema"] == "arwen.case-list.v1" and results[-1]["total"] == 2
    assert set(tmp_path.iterdir()) == {path, malformed_path}


def test_catalog_worker_reuses_validation_but_recomputes_preview(tmp_path, document, monkeypatch):
    import io
    from woof import source_availability
    path = write_catalog(tmp_path, document)
    requests = ["list", "show", "preview", "preview"]
    ident = document["cases"][0]["id"]
    original_loader = catalog._load_catalog_bytes
    original_availability = source_availability.availability
    loads, previews = [], []
    def load(path, raw):
        loads.append(raw)
        return original_loader(path, raw)
    def available(*args, **kwargs):
        previews.append(True)
        kwargs["now"] = NOW
        result = original_availability(*args, **kwargs)
        return result | {"test_observation": len(previews)}
    monkeypatch.setattr(catalog, "_load_catalog_bytes", load)
    monkeypatch.setattr(source_availability, "availability", available)
    encoded = b"".join((json.dumps([command, *([] if command == "list" else [ident]),
        "--catalog", str(path), "--json"]) + "\n").encode() for command in requests)
    output = io.StringIO()
    assert catalog._tui_server(io.BytesIO(encoded), output) == 0
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(loads) == 1 and len(previews) == 2
    assert [row["schema"] for row in rows] == ["arwen.case-list.v1", "arwen.case-detail.v1",
                                             "arwen.case-preview.v1", "arwen.case-preview.v1"]
    assert rows[2]["source_availability"]["test_observation"] == 1
    assert rows[3]["source_availability"]["test_observation"] == 2
    assert {row["provenance"]["original_sha256"] for row in rows} == {hashlib.sha256(path.read_bytes()).hexdigest()}


def test_standalone_catalog_entrypoint_has_normal_results_and_hides_private_worker(capsys):
    assert catalog.main(["list", "--catalog", str(EXAMPLES / "example.json"), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == catalog.list_cases(EXAMPLES / "example.json")
    with pytest.raises(SystemExit) as stopped:
        catalog.main(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "preview" in help_text and "--tui-server" not in help_text


def test_distributed_schema_validates_both_authoring_examples():
    import jsonschema
    schema = json.loads(catalog.SCHEMA_PATH.read_text(encoding="utf-8"))
    for name in ("example.json", "example.toml"):
        jsonschema.Draft202012Validator(schema).validate(catalog.load_catalog(EXAMPLES / name).document)


def test_catalog_prose_never_runs_a_command(tmp_path, document, monkeypatch):
    document["cases"][0]["metadata"] = {"command": "touch DO-NOT-CREATE", "environment": {"SECRET": "not a setting"}}
    document["cases"][0]["recommendations"].append({"topic": "unknown-science", "text": "$(run something) is only prose"})
    path = write_catalog(tmp_path, document)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("catalog load executed a process"))
    loaded = catalog.load_catalog(path)
    assert loaded.document["cases"][0]["metadata"]["command"] == "touch DO-NOT-CREATE"
    assert not (tmp_path / "DO-NOT-CREATE").exists()


def test_source_aliases_and_offset_times_normalize_without_changing_original(tmp_path, document):
    option = document["cases"][0]["source_options"][0]
    option.update(source="GFS", cycle_utc="2026-09-01T03:00:00+03:00")
    path = write_catalog(tmp_path, document)
    loaded = catalog.load_catalog(path)
    normalized = loaded.document["cases"][0]["source_options"][0]
    assert normalized["source"] == "gfs"
    assert normalized["cycle_utc"] == "2026-09-01T00:00:00Z"
    assert b"+03:00" in loaded.original


@pytest.mark.parametrize("identity", ["../escape", "C:drive", "has/slash", "UPPER", "con", "a" * 129])
def test_case_ids_cannot_be_paths_or_reserved_filenames(document, identity):
    document["cases"][0]["id"] = identity
    with pytest.raises(catalog.CatalogError, match="safe lowercase ID"):
        catalog.validate_catalog(document)


def test_duplicate_case_ids_and_duplicate_json_choices_are_refused(tmp_path, document):
    document["cases"][1]["id"] = document["cases"][0]["id"]
    with pytest.raises(catalog.CatalogError, match="Duplicate case ID"):
        catalog.validate_catalog(document)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(catalog.CatalogError, match="Duplicate JSON field"):
        catalog.load_catalog(path)


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row["tiers"]["lower"]["bounds_degrees"].update(south=40), "nonzero area"),
    (lambda row: row["tiers"]["lower"].update(root_dx_km=float("nan")), "finite"),
    (lambda row: row["tiers"]["lower"].update(run_hours=True), "finite integer"),
    (lambda row: row["tiers"]["lower"].update(root_dx_m=12000), "Unknown"),
    (lambda row: row["tiers"]["lower"].update(run_hours=24), "must not decrease"),
    (lambda row: row["tiers"]["lower"].update(root_dx_km=0.1), "coarser to finer"),
    (lambda row: row["tiers"]["lower"]["bounds_degrees"].update(west=-110), "footprint"),
    (lambda row: row["source_options"][0].update(cycle_utc="2026-09-01T00:30:00Z"), "exact UTC hour"),
])
def test_invalid_units_ranges_and_tier_order_never_reach_creation(document, mutation, match):
    mutation(document["cases"][0])
    with pytest.raises(catalog.CatalogError, match=match):
        catalog.validate_catalog(document)


def test_antimeridian_tiers_use_eastward_containment(document):
    row = document["cases"][0]
    for tier, west, east in (("lower", 178, -178), ("recommended", 170, -170), ("upper", 160, -160)):
        row["tiers"][tier]["bounds_degrees"].update(west=west, east=east)
    assert catalog.validate_catalog(document)["cases"][0]["tiers"]["lower"]["bounds_degrees"]["west"] == 178


@pytest.mark.parametrize("key", ["command", "environment", "output_dir", "api_token", "wif_climatology_path", "km_opt_zero_acknowledgement"])
def test_only_scientific_native_override_fields_are_admitted(key):
    with pytest.raises(catalog.CatalogError, match="not a recognized native scientific setting"):
        catalog.validate_native_overrides({"shared": {key: "do not execute"}})


def test_native_parameter_types_and_values_use_the_actual_registry():
    catalog.validate_native_overrides({"shared": {"num_soil_layers": 4}, "domains": [{"grid_id": 1, "settings": {"diff_6th_factor": 0.1}}]})
    for value in (True, "0.1", -1.0, 2.0):
        with pytest.raises(catalog.CatalogError):
            catalog.validate_native_overrides({"domains": [{"grid_id": 1, "settings": {"diff_6th_factor": value}}]})
    with pytest.raises(catalog.CatalogError, match="scope"):
        catalog.validate_native_overrides({"domains": [{"grid_id": 1, "settings": {"num_soil_layers": 4}}]})


def test_search_is_paged_and_filters_by_case_kind_and_source():
    result = catalog.list_cases(EXAMPLES / "example.json", query="synthetic", source="GFS", event_kind="synthetic", limit=1)
    assert result["total"] == 2
    assert len(result["cases"]) == 1
    assert catalog.list_cases(EXAMPLES / "example.json", query="typed")["total"] == 1
    assert catalog.list_cases(EXAMPLES / "example.json", source="era5")["total"] == 0


def test_unknown_future_source_remains_data_but_cannot_be_selected(tmp_path, document):
    document["cases"][0]["source_options"][0]["source"] = "future-model"
    path = write_catalog(tmp_path, document)
    assert catalog.list_cases(path, source="future-model")["total"] == 1
    with pytest.raises(catalog.CatalogError, match="Source option"):
        catalog.preview_case(path, document["cases"][0]["id"], now=NOW)


def test_preview_exposes_selected_settings_and_does_not_create_files(tmp_path, document):
    path = write_catalog(tmp_path, document)
    before = list(tmp_path.iterdir())
    result = catalog.preview_case(path, "synthetic-overrides-example", tier="lower", now=NOW,
                                  native_overrides={"domains": [{"grid_id": 1, "settings": {"epssm": 0.3}}]})
    assert result["cycle"] == "2026-09-01T00"
    assert result["native_overrides"]["domains"][0]["settings"]["epssm"] == 0.3
    assert result["forecast_started"] is False
    assert list(tmp_path.iterdir()) == before


def test_a_domain_override_cannot_be_silently_dropped_by_a_smaller_tier(tmp_path, document):
    path = write_catalog(tmp_path, document)
    with pytest.raises(catalog.CatalogError, match="creates only 1 domain"):
        catalog.preview_case(path, document["cases"][0]["id"], tier="lower", now=NOW,
                             native_overrides={"domains": [{"grid_id": 2, "settings": {"epssm": 0.3}}]})


def test_export_preserves_original_toml_and_refuses_overwrite(tmp_path):
    source = EXAMPLES / "example.toml"
    target = tmp_path / "original.toml"
    result = catalog.export_catalog(source, target, original=True)
    assert target.read_bytes() == source.read_bytes()
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        catalog.export_catalog(source, target)
    assert target.read_bytes() == source.read_bytes()


def test_catalog_changed_since_preview_cannot_create_different_science(tmp_path, document):
    path = write_catalog(tmp_path, document)
    preview = catalog.preview_case(path, document["cases"][0]["id"], tier="lower", now=NOW)
    document["cases"][0]["tiers"]["lower"]["run_hours"] = 2
    write_catalog(tmp_path, document)
    with pytest.raises(catalog.CatalogError, match="changed after the preview"):
        catalog.create_case(path, document["cases"][0]["id"], out=tmp_path / "changed.toml", tier="lower", now=NOW,
                            expected_catalog_sha256=preview["provenance"]["original_sha256"])
    assert not (tmp_path / "changed.toml").exists()


def test_native_creation_applies_science_and_preserves_original_and_receipt(tmp_path):
    source = EXAMPLES / "example.json"
    out = tmp_path / "study.toml"
    result = catalog.create_case(source, "synthetic-overrides-example", out=out,
                                 tier="lower", vram_gib=32, now=NOW)
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["shared"]["num_soil_layers"] == 4
    assert raw["domain"][0]["epssm"] == 0.4
    assert raw["domain"][0]["diff_6th_factor"] == 0.1
    assert Path(result["original_catalog"]).read_bytes() == source.read_bytes()
    receipt = json.loads(Path(str(out) + ".arwen-case.json").read_text())
    assert receipt["original_catalog_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["config_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt["admission"]["forecast_started"] is False
    assert receipt["files"] and result["domains"]
    preserved = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        catalog.create_case(source, "synthetic-profile-example", out=out,
                            tier="lower", vram_gib=32, now=NOW)
    assert preserved == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_invalid_native_combination_leaves_no_published_configuration(tmp_path):
    out = tmp_path / "invalid.toml"
    with pytest.raises(ValueError, match="sf_surface_physics"):
        catalog.create_case(EXAMPLES / "example.json", "synthetic-profile-example", out=out,
                            tier="lower", vram_gib=32, now=NOW,
                            native_overrides={"shared": {"sf_surface_physics": 999}})
    assert not out.exists()
    assert not list(tmp_path.glob("invalid*"))


def test_catalog_memory_refusal_keeps_its_own_retry_route_and_publishes_nothing(tmp_path, monkeypatch):
    from woof import domain_wizard as wizard, research_workspaces as research
    admission = research._admission

    def final_budget_refusal(text, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(wizard, "sizing_budget_bytes", lambda *args, **options: 1)
            return admission(text, **kwargs)

    monkeypatch.setattr(research, "_admission", final_budget_refusal)
    out = tmp_path / "refused.toml"
    with pytest.raises(ValueError, match="smaller tier from this catalog") as failed:
        catalog.create_case(EXAMPLES / "example.json", "synthetic-profile-example", out=out,
                            tier="lower", vram_gib=32, now=NOW)
    assert "--hardware-class" not in str(failed.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["polygon", "era5", "hrrr"])
def test_geometry_only_opens_native_domains_without_any_gpu_sizing(tmp_path, monkeypatch, kind):
    from woof import domain_wizard as wizard, research_workspaces as research

    def forbidden(*args, **kwargs):
        raise AssertionError("Opening a case must not resolve or admit a GPU memory budget")

    for name in ("resolve_sizing_budget", "domain_main", "_sizing_phases", "sizing_budget_bytes"):
        monkeypatch.setattr(wizard, name, forbidden)
    monkeypatch.setattr(research, "_admission", forbidden)
    if kind == "polygon":
        source, case_id, options = EXAMPLES / "example.json", "synthetic-overrides-example", {}
    else:
        source = proposal_zip(tmp_path, document=worldwide_proposal_document())
        case_id, options = "synthetic-import", {"source_option": kind}
    out = tmp_path / "opened.toml"
    receipt = catalog.create_case(source, case_id, out=out, tier="lower", now=NOW,
                                  geometry_only=True, **options)
    from woof.experiment import load_experiment
    experiment = load_experiment(out)
    assert experiment.domains
    if kind == "hrrr":
        assert all(domain.run.moist_cq for domain in experiment.domains)
    assert receipt["admission"]["status"] == "geometry-validated"
    assert receipt["admission"]["memory_admission"] == "deferred-to-review"
    assert receipt["admission"]["forecast_started"] is False
    assert "envelope_budget_bytes" not in receipt["admission"]
    assert receipt["config_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert Path(receipt["original_catalog"]).read_bytes() == source.read_bytes()
    if kind != "polygon":
        assert [d.run.dx / 1000 for d in experiment.domains] == ([12, 3, 1] if kind == "era5" else [3, 1])
        assert [d.run.nx for d in experiment.domains] == ([50, 72, 72] if kind == "era5" else [72, 72])
    if kind == "polygon":
        assert experiment.domains[0].run.epssm == .4
    preserved = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        catalog.create_case(source, case_id, out=out, tier="lower", now=NOW,
                            geometry_only=True, **options)
    assert preserved == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


@pytest.mark.parametrize("failure", ["science", "coverage", "capacity"])
def test_geometry_only_retains_native_validation_and_never_publishes_invalid_case(tmp_path, failure):
    options = {"geometry_only": True}
    source, case_id = EXAMPLES / "example.json", "synthetic-profile-example"
    pattern = "sf_surface_physics"
    if failure == "science":
        options["native_overrides"] = {"shared": {"sf_surface_physics": 999}}
    elif failure == "capacity":
        options["vram_gib"] = 32
        pattern = "defers GPU memory admission"
    else:
        document = worldwide_proposal_document()
        for preset in document["cases"][0]["presets"].values():
            for domain in preset["source_domain_recipes"]["regional_3_1"]["domains"]:
                domain["center_lat"], domain["center_lon"] = 35., 140.
        source, case_id = proposal_zip(tmp_path, document=document), "synthetic-import"
        options["source_option"] = "hrrr"
        pattern = "(?i)(coverage|outside|hrrr)"
    out = tmp_path / "invalid.toml"
    with pytest.raises(ValueError, match=pattern):
        catalog.create_case(source, case_id, out=out, tier="lower", now=NOW, **options)
    assert not list(tmp_path.glob("invalid*"))


def test_cli_geometry_only_returns_deferred_admission_receipt(tmp_path, monkeypatch, capsys):
    from woof.cli import main
    from woof import domain_wizard as wizard
    monkeypatch.setattr(wizard, "resolve_sizing_budget", lambda *a, **k: pytest.fail("GPU probe on Open Case"))
    out = tmp_path / "cli-open.toml"
    assert main(["case-catalog", "create", "synthetic-overrides-example", "--catalog",
                 str(EXAMPLES / "example.json"), "--tier", "lower", "--out", str(out),
                 "--geometry-only", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["admission"]["memory_admission"] == "deferred-to-review"
    assert result["forecast_started"] is False


def test_cli_dispatch_lists_cases_as_compact_json(capsys):
    from woof.cli import main
    assert main(["case-catalog", "list", "--catalog", str(EXAMPLES / "example.json"), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == "arwen.case-list.v1"
    assert output["total"] == 2


def proposal_document():
    """Synthetic data in the uploaded proposal's shape; no historical claims."""
    selectors = {"moist": True, "mp_physics": 8, "bl_pbl_physics": 1,
                 "sf_sfclay_physics": 1, "sf_surface_physics": 2,
                 "ra_lw_physics": 4, "ra_sw_physics": 4,
                 "ra_rrtmg_variant": "rrtmg_legacy", "cu_physics": 0,
                 "diff_opt": 2, "km_opt": 4}
    presets, schedules = {}, {}
    for tier, cycle, hours in (("minimum", "2013-05-31T14:00:00Z", 11),
                               ("preferred", "2013-05-31T12:00:00Z", 13),
                               ("large", "2013-05-31T00:00:00Z", 25)):
        domains = []
        for index, dx, width in ((1, 12, 600), (2, 3, 216), (3, 1, 72)):
            choice = dict(selectors, cu_physics=1 if index == 1 else 0)
            domains.append({"id": f"d{index:02d}", "parent_id": f"d{index-1:02d}" if index > 1 else None,
                            "center_lat": 35.5, "center_lon": -97.95, "dx_km": dx,
                            "width_km": width, "height_km": width, "selectors": choice})
        presets[tier] = {"domains": domains, "vertical_grid": {"levels_intent": 49}}
        schedules[tier] = {"start_utc": cycle, "total_integration_hours": hours,
                           "source_files_verified": False}
    return {"schema": "arwen.case-catalog/v1", "catalog_version": "1.0.0-research-proposal",
            "title": "Synthetic proposal import", "metadata": {"readiness": "planning only"},
            "initialization_sources": {"era5": {"engine_source_id": "era5", "time_cadence_hours": 1}},
            "physics_profiles": {"synthetic-control": {"soil_layers_required": 4}},
            "cases": [{"id": "synthetic-import", "name": "Synthetic import", "category": "synthetic",
                       "initialization": {"default_source_id": "era5", "alternatives": [
                           {"source_id": "era5", "schedules": schedules}]},
                       "physics": {"recommended_profile_id": "synthetic-control"}, "presets": presets}]}


def proposal_zip(tmp_path, *, extra_member=None, document=None):
    path = tmp_path / "proposal.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("proposal/catalog.json", json.dumps(document or proposal_document()))
        archive.writestr("proposal/authoring.py", "raise RuntimeError('never execute catalog scripts')")
        if extra_member:
            # ZipInfo normally normalizes Windows separators at construction;
            # retain the deliberately malformed on-wire name for this control.
            info = zipfile.ZipInfo("fixture")
            info.filename = extra_member
            info.orig_filename = extra_member
            archive.writestr(info, "untrusted")
    return path


def test_proposal_zip_preserves_original_and_maps_source_specific_tiers(tmp_path):
    path = proposal_zip(tmp_path)
    loaded = catalog.load_catalog(path)
    assert loaded.original == path.read_bytes()
    assert loaded.format == "zip"
    mapping = loaded.document["catalog"]["metadata"]["import_mapping"]
    assert mapping["archive"]["scripts_executed"] is False
    assert mapping["archive"]["files_extracted"] == 0
    assert not mapping["issues"]
    lower = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    preferred = catalog.preview_case(loaded, "synthetic-import", tier="recommended", now=NOW)
    assert (lower["cycle"], lower["geometry"]["run_hours"]) == ("2013-05-31T14", 11)
    assert (preferred["cycle"], preferred["geometry"]["run_hours"]) == ("2013-05-31T12", 13)
    assert lower["source_option"]["cadence_hours"] == 1
    assert lower["native_overrides"]["shared"]["mp_physics"] == 8
    assert not list(tmp_path.glob("*.py"))


@pytest.mark.parametrize("member", ["../escape.json", "/absolute.json", "C:/outside.json", "folder\\escape.json"])
def test_zip_paths_cannot_escape_even_when_the_member_is_not_selected(tmp_path, member):
    with pytest.raises(catalog.CatalogError, match="Unsafe catalog ZIP member"):
        catalog.load_catalog(proposal_zip(tmp_path, extra_member=member))


def test_unsupported_imported_selector_is_visible_and_blocks_creation(tmp_path):
    document = proposal_document()
    document["cases"][0]["presets"]["minimum"]["domains"][0]["selectors"]["command"] = "do not run"
    path = proposal_zip(tmp_path, document=document)
    loaded = catalog.load_catalog(path)
    assert catalog.list_cases(loaded)["total"] == 1
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    assert any("command" in issue for issue in preview["import_issues"])
    with pytest.raises(catalog.CatalogError, match="unsupported settings"):
        catalog.create_case(loaded, "synthetic-import", out=tmp_path / "blocked.toml", tier="lower", now=NOW)
    assert not (tmp_path / "blocked.toml").exists()


def test_imported_model_coordinate_diffusion_is_preserved(tmp_path):
    document = proposal_document()
    for preset in document["cases"][0]["presets"].values():
        for domain in preset["domains"]:
            domain["selectors"].update(diff_opt=1, mix_full_fields=False)
    loaded = catalog.load_catalog(proposal_zip(tmp_path, document=document))
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    assert not preview["import_issues"]
    assert preview["native_overrides"]["shared"]["diff_opt"] == 1
    assert preview["native_overrides"]["shared"]["mix_full_fields"] is False


def test_imported_model_coordinate_diffusion_can_be_per_domain(tmp_path):
    document = proposal_document()
    domain = document["cases"][0]["presets"]["minimum"]["domains"][1]
    domain["selectors"].update(diff_opt=1, mix_full_fields=False)
    loaded = catalog.load_catalog(proposal_zip(tmp_path, document=document))
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    assert not preview["import_issues"]
    overrides = preview["native_overrides"]["domains"]
    assert any(row["grid_id"] == 2 and row["settings"]["diff_opt"] == 1
               and row["settings"]["mix_full_fields"] is False for row in overrides)


@pytest.mark.parametrize("value", [False, 0, 3])
def test_imported_diffusion_requires_a_supported_selector(tmp_path, value):
    document = proposal_document()
    domain = document["cases"][0]["presets"]["minimum"]["domains"][0]
    domain["selectors"]["diff_opt"] = value
    loaded = catalog.load_catalog(proposal_zip(tmp_path, document=document))
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    assert any("diff_opt" in issue for issue in preview["import_issues"])


@pytest.mark.parametrize("provider,artifact", [("cds", "era5-combined.grib"),
                                               ("arco", "era5-combined.nc")])
def test_imported_native_creation_honors_hourly_cadence_and_unique_vtables(
        tmp_path, monkeypatch, provider, artifact):
    # The keyless default publishes NetCDF; a present CDS key selects GRIB.
    from woof import fetch
    monkeypatch.setattr(fetch, "cds_credentials_present", lambda: provider == "cds")
    path = proposal_zip(tmp_path)
    for name in ("one", "two"):
        out = tmp_path / (name + ".toml")
        receipt = catalog.create_case(path, "synthetic-import", out=out, tier="lower", vram_gib=32, now=NOW)
        raw = tomllib.loads(out.read_text(encoding="utf-8"))
        assert raw["fetch"]["cycle"] == "2013-05-31T14"
        assert raw["fetch"]["cadence"] == 1 and raw["fetch"]["hours"] == 11
        assert raw["case_data"]["forcing_interval_s"] == 3600
        assert raw["fetch"].get("era5_provider", "cds") == provider
        assert [Path(value).resolve() for value in raw["case_data"]["forcing"]] == [(tmp_path / "data" / name / artifact).resolve()]
        assert raw["case_data"]["geog_root"] == "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG"
        assert raw["case_data"]["vtable"].startswith(name + ".")
        assert (tmp_path / raw["case_data"]["vtable"]).is_file()
        assert "interval_seconds = 3600" in out.with_suffix(".namelist.wps").read_text()
        assert Path(receipt["original_catalog"]).read_bytes() == path.read_bytes()
        assert receipt["admission"]["forcing_interval_seconds"] == 3600
        assert [d["dx_km"] for d in receipt["domains"]] == [12, 3, 1]


def test_an_imported_case_defaults_to_the_modern_radiation_arm(tmp_path):
    """The importer's stamped profile follows the product default.

    woof/case_catalog.py binds that field when the caller names no
    profile, so it is a user-reachable default and moves with the route
    default rather than staying on the legacy twin it replaced.
    """
    from woof.physics_compat import THOMPSON_RTE_RRTMGP_PROFILE_ID
    loaded = catalog.load_catalog(proposal_zip(tmp_path))
    case = loaded.document["cases"][0]
    assert case["physics_profile"] == THOMPSON_RTE_RRTMGP_PROFILE_ID
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    assert preview["physics_profile"] == THOMPSON_RTE_RRTMGP_PROFILE_ID


@pytest.mark.parametrize("named_profile,typed_variant,expected_token", [
    (None, "rrtmg_legacy", "wrf-rrtmg-4-4-legacy-v1"),
    ("thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1", "rte-rrtmgp", "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"),
])
def test_a_typed_radiation_variant_carries_its_own_compatibility_token(
        tmp_path, named_profile, typed_variant, expected_token):
    """A proposal that types the engine by switch still composes.

    The wizard writes wrf_rrtmg_compatibility from the profile; the
    proposal's typed ra_rrtmg_variant lands as a shared override on top,
    and woof.config refuses the pair when the token records the other
    implementation. Under the modern default a proposal typing the
    legacy engine met that refusal; under the legacy profile a proposal
    typing the modern engine meets its mirror. The token now follows the
    typed variant in both directions, and the typed variant is what runs.
    """
    document = proposal_document()
    for preset in document["cases"][0]["presets"].values():
        for domain in preset["domains"]:
            domain["selectors"]["ra_rrtmg_variant"] = typed_variant
    path = proposal_zip(tmp_path, document=document)
    out = tmp_path / "typed.toml"
    catalog.create_case(path, "synthetic-import", out=out, tier="lower", now=NOW,
                        geometry_only=True, physics_profile=named_profile)
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["shared"]["ra_rrtmg_variant"] == typed_variant
    assert raw["shared"]["wrf_rrtmg_compatibility"] == expected_token


def test_a_typed_compatibility_token_is_left_as_typed(tmp_path):
    """Typing both keys is the proposal's own statement, kept verbatim."""
    document = proposal_document()
    for preset in document["cases"][0]["presets"].values():
        for domain in preset["domains"]:
            domain["selectors"]["ra_rrtmg_variant"] = "rrtmg_legacy"
            domain["selectors"]["wrf_rrtmg_compatibility"] = "none"
    path = proposal_zip(tmp_path, document=document)
    out = tmp_path / "typed-both.toml"
    catalog.create_case(path, "synthetic-import", out=out, tier="lower", now=NOW,
                        geometry_only=True)
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["shared"]["ra_rrtmg_variant"] == "rrtmg_legacy"
    assert raw["shared"]["wrf_rrtmg_compatibility"] == "none"


def worldwide_proposal_document():
    from copy import deepcopy
    document = proposal_document()
    document.update(schema="arwen.case-catalog/v2", catalog_version="2.0.0-worldwide-research-proposal")
    document["initialization_sources"]["era5"] = {"boundary_interval_hours": 6}
    document["initialization_sources"]["hrrr"] = {"boundary_interval_hours": 1}
    case = document["cases"][0]
    primary = case["initialization"].pop("alternatives")[0]
    primary["domain_recipe"] = "standard_12_3_1"
    regional = deepcopy(primary)
    regional.update(source_id="hrrr", domain_recipe="regional_3_1", status_by_tier={})
    for tier in ("minimum", "preferred", "large"):
        primary["schedules"][tier]["boundary_interval_hours"] = 6
        regional["schedules"][tier] = {"start_utc": "2026-09-01T18:00:00Z",
            "total_integration_hours": 6, "boundary_interval_hours": 1}
        regional["status_by_tier"][tier] = {"blocking_reasons": [], "actual_files_verified": False}
        # The recipe is this case's own d02/d03 subtree, kept under the ids the
        # catalog gave it; the importer resolves the tree from parent_id.
        domains = deepcopy(case["presets"][tier]["domains"][1:])
        domains[0]["parent_id"] = None
        case["presets"][tier]["source_domain_recipes"] = {"regional_3_1": {"domains": domains}}
    case["initialization"]["source_options"] = {"era5": primary, "hrrr": regional}
    return document


@pytest.mark.parametrize("source,spacing,cadence", [("era5", [12,3,1], 6), ("hrrr", [3,1], 1)])
def test_worldwide_source_geometry_and_cadence_reach_native_creation(tmp_path, source, spacing, cadence):
    path = proposal_zip(tmp_path, document=worldwide_proposal_document())
    loaded = catalog.load_catalog(path)
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", source_option=source, now=NOW)
    assert [row["dx_km"] for row in preview["geometry"]["domain_intents"]] == spacing
    assert preview["source_option"]["cadence_hours"] == cadence
    assert preview["provenance"]["catalog"]["provenance"][0]["source_schema"] == "arwen.case-catalog/v2"
    assert preview["source_availability"]["hours"] == (12 if source == "era5" else 6)
    output = tmp_path / f"{source}.toml"
    receipt = catalog.create_case(loaded, "synthetic-import", out=output, tier="lower",
                                  source_option=source, vram_gib=32, now=NOW)
    raw = tomllib.loads(output.read_text(encoding="utf-8"))
    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import route_input_paths
    experiment = load_experiment(output)
    assert [row.run.dx / 1000 for row in experiment.domains] == spacing
    if source == "hrrr":
        assert all(domain.run.moist_cq for domain in experiment.domains)
    # A created case carries every file its own input route reads.
    for role, companion in route_input_paths(output).items():
        assert companion.is_file() == (source == "hrrr" or role == "wps_namelist"), role
    assert raw["fetch"].get("cadence", 1) == cadence
    assert raw["experiment"]["run_seconds"] == preview["geometry"]["run_hours"] * 3600
    assert receipt["admission"]["forecast_started"] is False
    assert Path(receipt["original_catalog"]).read_bytes() == path.read_bytes()


def test_a_regional_case_the_route_cannot_state_is_refused_whole(tmp_path):
    """The route reads namelists, and the pair it writes has no row for this one.

    The HRRR route prepares the tree, and its mirrored WRF arm runs, from
    the namelist pair it emits beside the TOML.  Its renderer writes no
    swrad_scat row, so a configuration stating swrad_scat = 1.5 would be
    read back as WRF's 1.0: a different model from the TOML beside it.
    Publication refuses the mismatch and creates nothing.

    (moist_cq = false was this example until 6ce05efd2 made the forecast
    read it from the TOML (ROUTE_FORECAST_TOML_FIELDS), so it no longer
    refuses.  relax_w, the example 574743358 moved the bare-configuration
    test to, is not a catalog override.  swrad_scat is one of the
    catalog-allowed settings the route's renderer does not write, measured
    2026-10-07 at the 2.8.7 gate; when the renderer learns to carry it,
    this example moves again.)
    """
    from woof.hrrr_route_inputs import HrrrRouteInputError

    loaded = catalog.load_catalog(proposal_zip(tmp_path, document=worldwide_proposal_document()))
    output = tmp_path / "unstateable.toml"
    with pytest.raises(HrrrRouteInputError, match="swrad_scat"):
        catalog.create_case(loaded, "synthetic-import", out=output, tier="lower",
                            source_option="hrrr", vram_gib=32, now=NOW,
                            native_overrides={"shared": {"swrad_scat": 1.5}})
    assert not output.exists()
    assert not list(tmp_path.glob("unstateable*"))


def test_worldwide_source_blocking_reasons_remain_visible_and_prevent_creation(tmp_path):
    document = worldwide_proposal_document()
    document["cases"][0]["initialization"]["source_options"]["hrrr"]["status_by_tier"]["minimum"]["blocking_reasons"] = ["outside_declared_source_coverage"]
    loaded = catalog.load_catalog(proposal_zip(tmp_path, document=document))
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", source_option="hrrr", now=NOW)
    assert "outside_declared_source_coverage" in preview["import_issues"]
    with pytest.raises(catalog.CatalogError, match="outside_declared_source_coverage"):
        catalog.create_case(loaded, "synthetic-import", out=tmp_path / "blocked.toml", tier="lower", source_option="hrrr", vram_gib=32, now=NOW)
    assert not (tmp_path / "blocked.toml").exists()


def subtree_proposal_document():
    """The same proposal with its outermost domain dropped from every preset."""
    document = proposal_document()
    for preset in document["cases"][0]["presets"].values():
        kept = preset["domains"][1:]
        kept[0]["parent_id"] = None
        preset["domains"] = kept
    return document


def test_a_catalog_subtree_keeps_its_own_domain_ids(tmp_path):
    from woof.case_catalog_import import convert_proposal_v1
    document = subtree_proposal_document()
    assert [(row["id"], row["parent_id"]) for row in document["cases"][0]["presets"]["minimum"]["domains"]] == [
        ("d02", None), ("d03", "d02")]
    lower = convert_proposal_v1(deepcopy(document))["cases"][0]["tiers"]["lower"]
    assert lower["nest_ratios"] == [3]
    assert lower["root_dx_km"] == 3
    assert [row["dx_km"] for row in lower["domain_intents"]] == [3, 1]
    assert [row["grid_id"] for row in lower["domain_intents"]] == [1, 2]
    loaded = catalog.load_catalog(proposal_zip(tmp_path, document=document))
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    assert [row["dx_km"] for row in preview["geometry"]["domain_intents"]] == [3, 1]


def test_catalog_domain_ids_may_use_any_vocabulary():
    from woof.case_catalog_import import convert_proposal_v1
    document = proposal_document()
    names = ["outer", "middle", "inner"]
    for preset in document["cases"][0]["presets"].values():
        for index, domain in enumerate(preset["domains"]):
            domain.update(id=names[index], parent_id=None if index == 0 else names[index - 1])
    lower = convert_proposal_v1(deepcopy(document))["cases"][0]["tiers"]["lower"]
    assert [row["dx_km"] for row in lower["domain_intents"]] == [12, 3, 1]
    assert [row["grid_id"] for row in lower["domain_intents"]] == [1, 2, 3]
    assert lower["nest_ratios"] == [4, 3]


def test_per_domain_import_issue_names_the_catalog_s_own_domain_id():
    from woof.case_catalog_import import convert_proposal_v1
    document = proposal_document()
    names = ["outer", "middle", "inner"]
    for preset in document["cases"][0]["presets"].values():
        for index, domain in enumerate(preset["domains"]):
            domain.update(id=names[index], parent_id=None if index == 0 else names[index - 1])
        preset["domains"][1]["selectors"]["command"] = "do not run"
    converted = convert_proposal_v1(deepcopy(document))["cases"][0]
    issues = converted["metadata"]["conversion"]["issues"]
    assert any(issue.startswith("minimum/middle: per-domain command") for issue in issues)
    assert not any("/d02" in issue for issue in issues)


def test_catalog_domains_may_be_listed_child_first():
    from woof.case_catalog_import import convert_proposal_v1
    forward = convert_proposal_v1(proposal_document())["cases"][0]["tiers"]
    document = proposal_document()
    for preset in document["cases"][0]["presets"].values():
        preset["domains"].reverse()
    assert convert_proposal_v1(document)["cases"][0]["tiers"] == forward


def test_v2_source_domain_recipe_imports_under_its_original_ids(tmp_path):
    document = worldwide_proposal_document()
    recipe = document["cases"][0]["presets"]["minimum"]["source_domain_recipes"]["regional_3_1"]["domains"]
    assert [(row["id"], row["parent_id"]) for row in recipe] == [("d02", None), ("d03", "d02")]
    loaded = catalog.load_catalog(proposal_zip(tmp_path, document=document))
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", source_option="hrrr", now=NOW)
    assert [row["dx_km"] for row in preview["geometry"]["domain_intents"]] == [3, 1]


def _cycle_on_the_root(rows):
    rows[0]["parent_id"] = rows[1]["id"]


def _cycle_beside_the_root(rows):
    rows[1]["parent_id"], rows[2]["parent_id"] = rows[2]["id"], rows[1]["id"]


def _parent_this_preset_never_declares(rows):
    rows[2]["parent_id"] = "d09"


def _a_second_root(rows):
    rows[2]["parent_id"] = None


def _two_nests_on_one_parent(rows):
    rows[2]["parent_id"] = rows[0]["id"]


@pytest.mark.parametrize("mutate,pattern", [
    (_cycle_on_the_root, r"'d01', 'd02' name each other as nest parents in a closed parent_id chain, so no domain is the outermost one"),
    (_cycle_beside_the_root, r"'d02', 'd03' name each other as nest parents in a closed parent_id chain, so they never reach the outermost domain 'd01'"),
    (_parent_this_preset_never_declares, r"domain 'd03' names nest parent 'd09', which this preset does not declare"),
    (_a_second_root, r"domains 'd01', 'd03' each declare no nest parent, and one tier carries one outermost domain"),
    (_two_nests_on_one_parent, r"domains 'd02', 'd03' all nest inside 'd01', and a tier's geometry is one nest ladder"),
])
def test_only_a_tree_a_native_tier_cannot_carry_is_refused(mutate, pattern):
    from woof.case_catalog_import import convert_proposal_v1
    document = proposal_document()
    for preset in document["cases"][0]["presets"].values():
        mutate(preset["domains"])
    with pytest.raises(catalog.CatalogError, match=pattern):
        convert_proposal_v1(document)


def test_sibling_nests_are_refused_for_the_shape_the_tier_emits_not_the_id_sequence():
    """The deferred half of the tree work: the emitted tier is one ladder."""
    from woof.case_catalog_import import convert_proposal_v1
    document = proposal_document()
    for preset in document["cases"][0]["presets"].values():
        preset["domains"][2]["parent_id"] = preset["domains"][0]["id"]
        preset["domains"][2]["dx_km"] = 3
    with pytest.raises(catalog.CatalogError) as refusal:
        convert_proposal_v1(document)
    message = str(refusal.value)
    assert "root_dx_km plus nest_ratios" in message
    assert "source_domain_recipes" in message
    assert "sequential" not in message


def test_archive_budget_applies_to_selected_bytes_not_unread_duplicates(tmp_path, monkeypatch):
    from woof.case_catalog_import import read_archive
    path = proposal_zip(tmp_path)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("proposal/duplicate-browser.html", "x" * 100_000)
    original_open = zipfile.ZipFile.open
    opened = []
    def selected_only(self, member, *args, **kwargs):
        name = getattr(member, "filename", member)
        opened.append(name)
        assert name == "proposal/catalog.json"
        return original_open(self, member, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", selected_only)
    document, report = read_archive(path.read_bytes(), limit=20_000, object_pairs_hook=catalog._json_object)
    assert document["schema"] == "arwen.case-catalog/v1"
    assert report["expanded_bytes"] > 20_000 and report["selected_bytes_read"] < 20_000
    assert opened == ["proposal/catalog.json"]
    with pytest.raises(catalog.CatalogError, match="expanded catalog.json"):
        read_archive(path.read_bytes(), limit=100, object_pairs_hook=catalog._json_object)
