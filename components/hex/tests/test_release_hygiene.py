"""tools/release_hygiene.py refuses an artifact that carries private material.

The breakage it prevents (release review, 2026-10-13): the 0.4.0 candidate's
wheel and sdist carried the owner's name, home-directory paths, internal
cloud host names and the live address of a rented box.  The fixtures below
are built in the test, so this file carries no private literal itself.
"""

from __future__ import annotations

import importlib.util
import io
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "release_hygiene.py"


def _engine_source():
    root = Path(os.environ.get('WOOF_HEX_TEST_ENGINE_SOURCE', SCRIPT.parents[3] / 'engine'))
    if not (root / 'tools/release/cut/hygiene.py').is_file():
        pytest.skip('engine source required for the cut integration; set WOOF_HEX_TEST_ENGINE_SOURCE')
    return root


@pytest.fixture(scope="module")
def hygiene():
    spec = importlib.util.spec_from_file_location("release_hygiene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def _sdist(path: Path, files: dict[str, str]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def _rules(hygiene, owner: str):
    return list(hygiene.RULES) + hygiene.identity_rules([owner])


def test_a_clean_artifact_passes(hygiene, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: ["Zebulon"])
    wheel = _wheel(tmp_path / "x-1.0-py3-none-any.whl", {
        "pkg/a.py": "# owner ruling 2026-10-10; loopback 127.0.0.1, private 10.1.2.3\n",
        "pkg/b.txt": "version 2.9.0 and 8.4.2; " + "/wo" + "rk/box/run and /home/runner/work\n",
    })
    assert hygiene.main([str(wheel)]) == 0


# Each sample is assembled at run time: this file ships in the sdist, and the
# gate scans it there, so no sample may appear here as one literal.
_SAMPLES = [
    ("see " + "C:" + "\\Users\\someone\\x", "home-path"),
    ("cd " + "/ho" + "me/someone/x", "home-path"),
    ("cat " + "Down" + "loads/LANE/REPORT.md", "downloads-path"),
    ("host " + "ip-" + "172-31-47-178" + " ran it", "cloud-host-name"),
    ("built on " + "weather-" + "no" + "de-2", "lab-host-name"),
    ("clone in " + "co" + "dex-" + "1007", "workstation-folder"),
    ("ssh root@" + ".".join(["3", "14", "159", "26"]), "public-ipv4"),
    ("ruled by Zebulon on 2026-10-10", "owner-name"),
    ("nested " + "u" + "ae", "private-case"),
    ("CU" + "DA " + "JE" + "DI", "private-da-name"),
    ("syn" + "radar", "private-da-name"),
    ("synthetic-radar " + "DA cycle", "private-da-name"),
    (".clau" + "de/worktrees", "private-worktree"),
    ("no" + "de-2", "private-host"),
    ("pool-" + "v3", "private-pool"),
    ("host V" + "8X" + "6000", "internal box"),
    ("host V" + "50" + "90", "internal box"),
    ("host V" + "8X", "internal box"),
    ("d" + "rew-ruling.json", "private-ruling-file"),
]


@pytest.mark.parametrize("text,rule", _SAMPLES)
def test_each_class_of_private_material_is_refused_by_name(hygiene, tmp_path, monkeypatch, capsys, text, rule):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: ["zebulon"])
    sdist = _sdist(tmp_path / "x-1.0.tar.gz", {"x-1.0/src/a.py": text + "\n"})
    assert hygiene.main([str(sdist)]) == 1
    printed = capsys.readouterr().out
    assert f":: x-1.0/src/a.py :: {rule} ::" in printed


def test_the_lower_case_login_is_not_an_english_verb_hit(hygiene):
    # a lower-case login that is also an English verb must not trip the gate;
    # the capitalised name does
    rules = _rules(hygiene, "d" + "rew")
    assert hygiene.scan([("a.txt", b"the door drew 92 maps")], rules) == []
    named = ("ruled by " + "D" + "rew").encode()
    assert [h[1] for h in hygiene.scan([("a.txt", named)], rules)] == ["owner-name"]


def test_array_index_arithmetic_is_not_a_host_name(hygiene):
    assert hygiene.scan([("third_party/a.c", b"return no" b"de-1;")], list(hygiene.RULES)) == []


def test_short_case_token_in_binary_bytes_is_not_a_text_hit(hygiene):
    case = ("u" + "ae").encode()
    assert hygiene.scan([("image.png", b"\x00" + case + b"\x00")], list(hygiene.RULES)) == []


def test_an_allowed_public_address_is_not_a_hit(hygiene, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: [])
    address = ".".join(["8"] * 4)
    wheel = _wheel(tmp_path / "x-1.0-py3-none-any.whl", {"pkg/a.py": f"server {address}\n"})
    assert hygiene.main([str(wheel)]) == 1
    assert hygiene.main([str(wheel), "--allow-address", address]) == 0


def test_an_unreadable_artifact_is_its_own_exit_code(hygiene, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: [])
    bogus = tmp_path / "x-1.0-py3-none-any.whl"
    bogus.write_bytes(b"not a zip")
    assert hygiene.main([str(bogus)]) == 2


def test_export_content_rules_apply_to_unchanged_artifact_members(hygiene, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, 'identity_tokens', lambda extra, where: [])
    wheel = _wheel(tmp_path / 'x.whl', {'pkg/a.py': '# CUDA JE' + 'DI\n',
                                     'pkg/b.py': '# U' + 'AE 9 km\n'})
    assert hygiene.main([str(wheel)]) == 1
    assert hygiene.content_scan('catalog.md', b'April 2024 U' + b'AE-Oman flood') == []


def test_hex_cut_repairs_allowed_lines_and_continues(hygiene, tmp_path):
    engine = _engine_source()
    export = tmp_path / 'export'
    export.mkdir()
    (export / 'a.py').write_text('# U' + 'AE 9 km\nx = 1\n')
    report = hygiene.fix_export(export, engine)
    assert len(report['hits_before']) == 1
    assert report['remaining'] == report['unrewritable'] == []
    assert (export / 'a.py').read_text() == 'x = 1\n'


def test_licence_export_runs_content_hygiene_inside_the_same_cut(tmp_path, capsys):
    import json
    spec = importlib.util.spec_from_file_location("hex_licence_export", SCRIPT.with_name("licence_export.py"))
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("# CUDA JE" + "DI\nx = 1\n")
    destination = tmp_path / "export"
    engine = _engine_source()
    assert export.main(["apply", str(source), str(destination), "--engine-source", str(engine)]) == 0
    assert (destination / "a.py").read_text() == "x = 1\n"
    assert json.loads(capsys.readouterr().out)["content_hygiene"]["hits_before"]


def test_sealed_comment_exception_never_allows_private_content(hygiene):
    path = "src/hexcore/data/compiler-recipes.json"
    assert hygiene.scan([(path, b"// no" + b"de-2; CLA" + b"UDE.md")], list(hygiene.RULES)) == []
    assert hygiene.scan([(path, b"// CUDA JE" + b"DI")], list(hygiene.RULES))
    assert hygiene.scan([("docs/build.md", b"no" + b"de-2; CLA" + b"UDE.md")], list(hygiene.RULES))


@pytest.mark.parametrize("text", [
    "config_je" + "di_da", '{"id":"je' + 'di-background"}', "https://github.com/NOAA-GSL/rrfs-mpas-" + "jedi",
    "hexcore.je" + "di_contract --je" + "di-background --da-analysis mpasout",
    '{"format":"je' + 'di-ensemble-stream","jedi_open_checked":true}',
])
def test_public_contracts_pass_in_source_and_artifacts(hygiene, text):
    for path in ("src/driver.py", "src/data/compiler-recipes.json", "README.md"):
        assert hygiene.scan([(path, text.encode())], list(hygiene.RULES)) == []


@pytest.mark.parametrize("private", [
    "CUDA" + "JEDI", "JE" + "DI-CUDA", "woof-" + "da", "obs-" + "pipeline",
    "da-" + "rows", "da-" + "cases", "synthetic-radar " + "DA",
    '"je' + 'di-private"',
])
def test_public_contract_allowlist_cannot_hide_private_tooling(hygiene, private):
    public = "--je" + "di-background "
    for path in ("src/driver.py", "src/data/compiler-recipes.json"):
        assert hygiene.scan([(path, (public + private).encode())], list(hygiene.RULES))
        if "private" not in private:
            assert hygiene.scan([("libexec/helper", b"\0" + private.encode())], list(hygiene.RULES))


def test_github_container_home_is_not_an_owner_name(hygiene, tmp_path, monkeypatch):
    """THE BREAKAGE (hex 0.4.12 publish.yml dry run 42033510796, 2026-11-05): GitHub runs a container job
    with HOME=/github/home, so "home" became an identity token and every "HOME" ($HOME, os.environ["HOME"],
    Rust's home_dir) was refused as the owner's name: 20 false hits in the wheel and sdist."""
    monkeypatch.setattr(hygiene.Path, "home", classmethod(lambda cls: Path("/github/home")))
    for key in ("USER", "USERNAME", "LOGNAME", "LNAME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(hygiene.getpass, "getuser", lambda: "root")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "none"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    tokens = hygiene.identity_tokens([], tmp_path)
    assert "home" not in [t.lower() for t in tokens]
    rules = list(hygiene.RULES) + hygiene.identity_rules(tokens)
    data = b'path = os.environ["HOME"]' + bytes([10]) + b'export PATH="$HOME/.cargo/bin:$PATH"'
    assert [h for h in hygiene.scan([("woof/hex/x.py", data)], rules) if h[1] == "owner-name"] == []


def test_a_per_user_home_folder_still_names_the_owner(hygiene, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene.Path, "home", classmethod(lambda cls: Path("/home") / "zebulon"))
    for key in ("USER", "USERNAME", "LOGNAME", "LNAME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(hygiene.getpass, "getuser", lambda: "root")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "none"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    assert "zebulon" in hygiene.identity_tokens([], tmp_path)
