"""Stage all three pin shapes and verify the real local mirror URL layout."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

import pytest

SPEC = importlib.util.spec_from_file_location("stage_release_assets", Path(__file__).resolve().parents[1]
                                            / "tools/stage_release_assets.py")
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def fake_pins(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pins = []
    for index, schema in enumerate(stage.SCHEMAS):
        name = f"asset-{index}.json"
        data = f"asset bytes {index}".encode()
        (source / name).write_bytes(data)
        row = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        document = dict(schema=schema, release_assets=dict(repository="FahrenheitResearch/gpuwm-hex",
                                                         tag=f"assets-{index}"))
        if index == 0:
            document["bridges"] = {name: row}
        elif index == 1:
            document["assets"] = [dict(row, name=name)]
        else:
            document.update(breakage="size cap", assets={name: dict(row, kind="kernel-pack",
                            select=dict(nvrtc_major=13), compiler=dict(nvrtc=[13, 4]))})
        pin = tmp_path / f"pins-{index}.json"
        pin.write_text(json.dumps(document))
        pins.extend(["--pin", str(pin)])
    return source, pins


@pytest.mark.parametrize("repository", ["FahrenheitResearch/gpuwm-hex", "recastsystems/gpuwm-hex"])
def test_stages_all_pin_shapes_with_hashes_commands_and_http(tmp_path, repository):
    source, pins = fake_pins(tmp_path)
    for pin in pins[1::2]:
        path = Path(pin)
        document = json.loads(path.read_text())
        document["release_assets"]["repository"] = repository
        path.write_text(json.dumps(document))
    out = tmp_path / "staged with spaces"
    assert stage.main([*pins, "--source", str(source), "--out", str(out), "--serve-check"]) == 0
    manifest = json.loads((out / "assets-manifest.json").read_text())
    assert len(manifest["assets"]) == 3
    assert manifest["serve_check"]["passed"] == 3
    assert manifest["serve_check"]["failed"] == manifest["serve_check"]["skipped"] == 0
    commands = [shlex.split(line) for line in (out / "PUBLISH-ASSETS.txt").read_text().splitlines()
                if not line.startswith("#")]
    assert [command[3] for command in commands] == [f"assets-{i}" for i in range(3)]
    for row, command, check in zip(manifest["assets"], commands, manifest["serve_check"]["checks"]):
        assert row["found"] is True
        assert Path(row["staged_path"]).read_bytes() == (source / row["name"]).read_bytes()
        assert command[:3] == ["gh", "release", "create"]
        assert command[4:6] == ["--repo", repository]
        assert command[-2] == row["staged_path"]
        assert Path(command[-1]).read_text() == f"{row['sha256']}  {row['name']}\n"
        assert check["url"].endswith(f"/{row['tag']}/{row['name']}")


def test_missing_and_mismatched_are_named_and_not_staged(tmp_path, capsys):
    source, pins = fake_pins(tmp_path)
    (source / "asset-0.json").unlink()
    (source / "asset-1.json").write_bytes(b"wrong bytes")
    out = tmp_path / "out"
    assert stage.main([*pins, "--source", str(source), "--out", str(out), "--serve-check"]) == 1
    output = capsys.readouterr().out
    assert "missing: assets-0/asset-0.json" in output
    assert "mismatched: assets-1/asset-1.json" in output
    manifest = json.loads((out / "assets-manifest.json").read_text())
    assert [row["found"] for row in manifest["assets"]] == [False, False, True]
    assert not (out / "assets-1/asset-1.json").exists()
    assert (out / "assets-0/SHA256SUMS").read_text() == ""
    assert manifest["serve_check"]["passed"] == 1
    assert manifest["serve_check"]["skipped"] == 2
    assert not list(out.rglob(".stage-*"))


def test_can_find_valid_copy_in_second_source(tmp_path):
    source, pins = fake_pins(tmp_path)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "asset-0.json").write_bytes(b"corrupted")
    assert stage.main([*pins, "--source", str(bad), "--source", str(source),
                       "--out", str(tmp_path / "out")]) == 0


@pytest.mark.parametrize("field,value", [("name", "../outside"), ("tag", "../outside"),
                                        ("sha256", "wrong"), ("bytes", -1)])
def test_invalid_pins_refuse_path_escape_and_bad_identity(tmp_path, field, value):
    source, pins = fake_pins(tmp_path)
    path = Path(pins[1])
    document = json.loads(path.read_text())
    if field == "tag":
        document["release_assets"]["tag"] = value
    elif field == "name":
        document["bridges"][value] = document["bridges"].pop("asset-0.json")
    else:
        document["bridges"]["asset-0.json"][field] = value
    path.write_text(json.dumps(document))
    assert stage.main([*pins, "--source", str(source), "--out", str(tmp_path / "out")]) == 1
