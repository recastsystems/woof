"""Fast shared job-form/CLI contract proof; no weights, network or GPU."""
import json

import jsonschema
import pytest

from woof.earth2_door import SCHEMA_FILE, build_spec, canonical_json, load_spec


def test_canonical_spec_and_unknown_names(tmp_path, monkeypatch):
    spec = build_spec(model="fourcastnet3", cycle="2024-09-24T12", hours=24,
                      storm="auto", nest="2.5km", mix="gefs:18,fourcastnet3:18")
    assert spec["model"] == "fcn3"
    assert spec["cycle"] == "2024-09-24T12:00:00Z"
    assert spec["nest_km"] == 2.5
    assert spec["ensemble"] == {"members": 36, "mix": [
        {"source": "fcn3", "members": 18}, {"source": "gefs", "members": 18}]}
    document = canonical_json(spec)
    assert document == json.dumps(spec, sort_keys=True, indent=2) + "\n"
    jsonschema.Draft202012Validator(json.loads(SCHEMA_FILE.read_text())).validate(spec)
    saved = tmp_path / "spec.json"
    saved.write_text(document)
    assert canonical_json(load_spec(saved)) == document
    assert "card" not in spec
    card_spec = build_spec(model="fcn3", cycle="2024-09-24T12", card="16gb")
    assert card_spec["card"] == "16gb"
    saved.write_text(canonical_json(card_spec))
    assert load_spec(saved) == card_spec
    for options, match in ((dict(model="unknown-parent"), "valid names"),
                           (dict(mix="unknown-source:2"), "valid names"),
                           (dict(diagnose=["unknown-diagnostic"]), "valid names"),
                           (dict(ic_source="unknown-ic"), "valid names"),
                           (dict(card="unknown-card"), "card"),
                           (dict(hours=0), "positive integer")):
        with pytest.raises(ValueError, match=match):
            build_spec(**(dict(model="fcn3", cycle="2024-09-24T12") | options))
    spec["table"]["sha256"] = "0" * 64
    saved.write_text(canonical_json(spec))
    with pytest.raises(ValueError, match="SHA256 differs"):
        load_spec(saved)

    # A new table row, alias and cadence require no new builder code.
    from woof import earth2_door, earth2_run
    table = earth2_run.model_table()
    table["models"]["table-added-model"] = dict(table["models"]["fcn3"], aliases=["added-alias"], step_hours=3)
    authority = tmp_path / "earth2-models.v1.json"
    authority.write_text(json.dumps(table))
    monkeypatch.setattr(earth2_run, "TABLE", authority)
    monkeypatch.setattr(earth2_door, "TABLE", authority)
    added = build_spec(model="added-alias", cycle="2024-09-24T12", hours=9)
    assert added["model"] == "table-added-model"
    assert added["hours"] == 9
    assert build_spec(model="fcn3", cycle="2024-09-24T12", hours=1)["hours"] == 1
    assert build_spec(model="fcn3", cycle="2024-09-24T12", diagnose="windgust")["diagnose"] == ["windgust"]


def test_dry_run_applies_the_real_runs_refusals(tmp_path, capsys):
    # Defect: --dry-run returned 0 for jobs the real run refuses (a restricted --diagnose row without
    # --accept-licence, and a regional job with neither --area nor --nest).
    from woof import cli
    licence = "NVIDIA Community Model License"
    base = ["earth2", "--model", "fcn3", "--ic-source", "gfs", "--cycle", "2024-10-08T00", "--hours", "24",
            "--dry-run", "--spec-out", str(tmp_path / "s.json"), "--out", str(tmp_path / "o")]
    assert cli.main([*base, "--diagnose", "windgust"]) == 2
    assert "--accept-licence" in capsys.readouterr().err
    assert not (tmp_path / "s.json").exists()
    assert cli.main([*base, "--diagnose", "windgust", "--accept-licence", licence]) == 2
    assert "needs --area" in capsys.readouterr().err
    assert cli.main([*base, "--diagnose", "windgust", "--accept-licence", licence, "--area", "30,-100,40,-85"]) == 0
    assert json.loads((tmp_path / "s.json").read_text())["diagnose"] == ["windgust"]


def test_dry_run_honours_geog_root(tmp_path, capsys, monkeypatch):
    # Defect (2.9.4 landing, test host 2026-10-20): `woof earth2` had no --geog-root, so an Earth-2
    # nest's preparation could only read the default tree. The door now takes the shared option
    # (woof.geog_assets.add_geog_root_argument, the one `woof go` uses) and a dry run checks it.
    from woof import cli, go_cli
    base = ["earth2", "--model", "fcn3", "--ic-source", "gfs", "--cycle", "2024-10-08T00", "--hours", "24",
            "--area", "30,-100,40,-85", "--dry-run", "--out", str(tmp_path / "o")]
    empty = tmp_path / "no-geography"
    empty.mkdir()
    assert cli.main([*base, "--geog-root", str(empty)]) == 2
    assert str(empty) in capsys.readouterr().err
    seen = []
    monkeypatch.setattr(go_cli, "geography_refusal", lambda root, **_: seen.append(root))
    staged = tmp_path / "WPS_GEOG"
    staged.mkdir()
    assert cli.main([*base, "--geog-root", str(staged)]) == 0
    assert seen == [staged.resolve()]
    assert f"earth2: geography {staged.resolve()}" in capsys.readouterr().err


def test_earth2_and_go_share_the_geog_root_option():
    import argparse

    from woof import earth2_door, go_cli
    parsers = {}
    for name, module in (("earth2", earth2_door), ("go", go_cli)):
        sub = argparse.ArgumentParser().add_subparsers()
        module.register_cli(sub)
        parsers[name] = sub.choices[name]
    for parser in parsers.values():
        action = next(a for a in parser._actions if "--geog-root" in a.option_strings)
        assert action.metavar == "DIR" and action.default is None
        assert action.type is not None and action.type("x").name == "x"
