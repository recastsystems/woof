"""tools/release_hygiene.py refuses an artifact that carries private material.

The breakage it prevents (release review, 2026-10-13): the 0.4.0 candidate's
wheel and sdist carried the owner's name, home-directory paths, internal
cloud host names and the live address of a rented box.  The project's own
private names are not in this file or in the tools it tests: they are read
from a pattern file that is never packaged (export_content.py).  The pattern
file used here is made up in the test, with made-up names, so these tests
check the mechanism: that every kind of rule the real file can carry is
loaded and applied, and that the scan refuses to pretend when no file is
loaded.
"""

from __future__ import annotations

import importlib.util
import io
import json
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


# A made-up pattern file with one rule of every option the real file uses.
MADE_UP = {
    "schema": "hex.private-patterns.v1",
    "content_rules": [
        {"kind": "made-up tooling", "pattern": r"\bzorbtool\b", "artifact": True},
        {"kind": "made-up case", "pattern": r"\bqqx\b", "historical": r"1999.{0,40}qqx"},
        {"kind": "made-up host", "pattern": r"\bhostling-\d+\b", "sealed_ok": True,
         "skip_line": r"\breturn\s+hostling-\d+;"},
        {"kind": "made-up box", "pattern": r"\bboxv7\b", "skip_line": r"boxv7\s*=>", "skip_line_suffix": ".f90"},
        {"kind": "made-up recipe", "pattern": r"(?:^|[/\"'])zorb-[a-z]", "public_contracts": True},
    ],
    "public_contracts": [{"group": "made-up public contract", "pattern": r"--zorb-background\b", "reason": "test"}],
    "artifact_rules": [
        {"kind": "made-up-case", "pattern": r"\bqqx\b", "skip_binary": True},
        {"kind": "made-up-host", "pattern": r"\bhostling-2\b", "sealed_ok": True},
        {"kind": "made-up-pool", "pattern": r"\bpoolish-v9\b", "case_sensitive": True},
    ],
}


@pytest.fixture(scope="module")
def hygiene():
    spec = importlib.util.spec_from_file_location("release_hygiene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patterns(hygiene, tmp_path):
    path = tmp_path / "made-up-patterns.json"
    path.write_text(json.dumps(MADE_UP))
    hygiene.configure(path)
    yield path
    hygiene.configure(os.environ.get(hygiene._content.PRIVATE_ENV))


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


def test_a_clean_artifact_passes(hygiene, patterns, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: ["Zebulon"])
    wheel = _wheel(tmp_path / "x-1.0-py3-none-any.whl", {
        "pkg/a.py": "# owner ruling 2026-10-10; loopback 127.0.0.1, private 10.1.2.3\n",
        "pkg/b.txt": "version 2.9.0 and 8.4.2; " + "/wo" + "rk/box/run and /home/runner/work\n",
    })
    assert hygiene.main([str(wheel)]) == 0


_SAMPLES = [
    ("see " + "C:" + "\\Users\\someone\\x", "home-path"),
    ("cd " + "/ho" + "me/someone/x", "home-path"),
    ("cat " + "Down" + "loads/LANE/REPORT.md", "downloads-path"),
    ("host " + "ip-" + "172-31-47-178" + " ran it", "cloud-host-name"),
    ("ssh root@" + ".".join(["3", "14", "159", "26"]), "public-ipv4"),
    ("ruled by Zebulon on 2026-10-10", "owner-name"),
    ("nested qqx", "made-up-case"),
    ("a zorbtool run", "made-up tooling"),
    ("built on hostling-2", "made-up-host"),
    ("queued in poolish-v9", "made-up-pool"),
]


@pytest.mark.parametrize("text,rule", _SAMPLES)
def test_each_class_of_private_material_is_refused_by_name(hygiene, patterns, tmp_path, monkeypatch, capsys, text, rule):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: ["zebulon"])
    sdist = _sdist(tmp_path / "x-1.0.tar.gz", {"x-1.0/src/a.py": text + "\n"})
    assert hygiene.main([str(sdist)]) == 1
    printed = capsys.readouterr().out
    assert f":: x-1.0/src/a.py :: {rule} ::" in printed


def test_the_shipped_tools_carry_no_private_rule(hygiene, tmp_path):
    """THE BREAKAGE (source packages of three candidates, 2026-11-15): the private rules were written in these
    tools, split across string literals, so every source package carried the names its scans refuse. With no
    pattern file the shipped tools hold the generic rules and nothing else."""
    hygiene.configure(None)
    try:
        assert hygiene.RULES == hygiene.GENERIC_RULES
        assert hygiene._content.RULES == hygiene._content.GENERIC_RULES
        assert hygiene._content.PUBLIC_CONTRACT_ALLOWLIST == ()
        assert not hygiene._content.PRIVATE_LOADED
    finally:
        hygiene.configure(os.environ.get(hygiene._content.PRIVATE_ENV))


def test_the_shipped_tools_tree_has_no_private_line(hygiene):
    """THE BREAKAGE (hex 0.6.2, 2026-11-20): a merge brought tools/torch-bridge/, training run scripts (with
    scratch folders, a home path and a card pool of the machines they ran on), into tools/, which the sdist grafts
    whole. The release's content hygiene could not rewrite their code lines (licence_export apply rc 1, 36 findings) and the release build
    refused the sdist ("tools/torch-bridge/baseline.sh :: home-path"). No file under tools/ may carry a line the
    rules find: the generic rules always, the private file's rules where the suite names it."""
    tools = SCRIPT.parent

    def scan():
        rows = [(p.relative_to(tools).as_posix(), p.read_bytes(), tools) for p in sorted(tools.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and "target" not in p.parts]
        return [hit[:3] for row in rows for hit in hygiene._export_member(row)]

    hygiene.configure(None)
    try:
        assert scan() == []
    finally:
        hygiene.configure(os.environ.get(hygiene._content.PRIVATE_ENV))
    if hygiene._content.PRIVATE_LOADED:
        assert scan() == []


def test_require_private_refuses_a_scan_without_the_pattern_file(hygiene, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: [])
    wheel = _wheel(tmp_path / "x-1.0-py3-none-any.whl", {"pkg/a.py": "x = 1\n"})
    hygiene.configure(None)
    try:
        assert hygiene.main([str(wheel), "--require-private"]) == 2
        assert "no private pattern file" in capsys.readouterr().err
        assert hygiene.main([str(wheel)]) == 0
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert hygiene.main([str(wheel), "--private-patterns", str(bad)]) == 2
        good = tmp_path / "good.json"
        good.write_text(json.dumps(MADE_UP))
        assert hygiene.main([str(wheel), "--require-private", "--private-patterns", str(good)]) == 0
    finally:
        hygiene.configure(os.environ.get(hygiene._content.PRIVATE_ENV))


def test_the_lower_case_login_is_not_an_english_verb_hit(hygiene):
    # a lower-case login that is also an English word must not trip the gate; the capitalised name does
    rules = _rules(hygiene, "will")
    assert hygiene.scan([("a.txt", b"the door will open")], rules) == []
    assert [h[1] for h in hygiene.scan([("a.txt", b"ruled by Will")], rules)] == ["owner-name"]


def test_array_index_arithmetic_is_not_a_host_name(hygiene, patterns):
    assert hygiene.content_scan("third_party/a.c", b"return hostling-1;") == []
    assert [h[2] for h in hygiene.content_scan("src/a.c", b"// ran on hostling-1")] == ["made-up host"]


def test_a_skip_line_applies_only_to_its_file_suffix(hygiene, patterns):
    assert hygiene.content_scan("tests/probe.f90", b"use m, only: boxv7 => boltzmann") == []
    assert hygiene.content_scan("tests/probe.py", b"boxv7 => boltzmann")


def test_a_skip_line_takes_a_list_of_file_suffixes(hygiene, tmp_path):
    """THE BREAKAGE THIS PREVENTS (dist48): generated C++ blocks of the joined boundary-layer backward number their
    variables (v0, v1, ...), and one number is spelled like a private box; the rule's line skip held for generated
    Fortran only, so the export cut stopped on 401 generated lines it could not rewrite. A list of suffixes names
    each generated file the skip holds for, and nothing else."""
    rules = dict(MADE_UP, content_rules=[{"kind": "made-up box", "pattern": r"\bboxv7\b", "skip_line": r"boxv7\s*=",
                                          "skip_line_suffix": [".f90", "/gen/blocks/one.hpp"]}])
    path = tmp_path / "list-suffix.json"
    path.write_text(json.dumps(rules))
    hygiene.configure(path)
    try:
        assert hygiene.content_scan("tests/probe.f90", b"boxv7 = 1") == []
        assert hygiene.content_scan("ad/gen/blocks/one.hpp", b"const P boxv7 = r6_add(a, b);") == []
        assert hygiene.content_scan("ad/gen/blocks/two.hpp", b"const P boxv7 = r6_add(a, b);")
        assert hygiene.content_scan("ad/gen/blocks/one.hpp", b"// ran on boxv7")
    finally:
        hygiene.configure(os.environ.get(hygiene._content.PRIVATE_ENV))


def test_short_case_token_in_binary_bytes_is_not_a_text_hit(hygiene, patterns):
    assert [h for h in hygiene.scan([("image.png", b"\x00" + b"qqx" + b"\x00")], list(hygiene.RULES))
            if h[1] == "made-up-case"] == []


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


def test_export_content_rules_apply_to_unchanged_artifact_members(hygiene, patterns, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, 'identity_tokens', lambda extra, where: [])
    wheel = _wheel(tmp_path / 'x.whl', {'pkg/a.py': '# zorbtool\n', 'pkg/b.py': '# qqx 9 km\n'})
    assert hygiene.main([str(wheel)]) == 1
    assert hygiene.content_scan('catalog.md', b'April 1999 qqx flood') == []


def test_hex_cut_repairs_allowed_lines_and_continues(hygiene, patterns, tmp_path):
    engine = _engine_source()
    export = tmp_path / 'export'
    export.mkdir()
    (export / 'a.py').write_text('# qqx 9 km\nx = 1\n')
    report = hygiene.fix_export(export, engine)
    assert len(report['hits_before']) == 1
    assert report['remaining'] == report['unrewritable'] == []
    assert (export / 'a.py').read_text() == 'x = 1\n'


def _licence_export():
    spec = importlib.util.spec_from_file_location("hex_licence_export", SCRIPT.with_name("licence_export.py"))
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    return export


def test_licence_export_runs_content_hygiene_inside_the_same_cut(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("# zorbtool\nx = 1\n")
    destination = tmp_path / "export"
    engine = _engine_source()
    made_up = tmp_path / "made-up-patterns.json"
    made_up.write_text(json.dumps(MADE_UP))
    assert _licence_export().main(["apply", str(source), str(destination), "--engine-source", str(engine),
                                   "--private-patterns", str(made_up)]) == 0
    assert (destination / "a.py").read_text() == "x = 1\n"
    assert json.loads(capsys.readouterr().out)["content_hygiene"]["hits_before"]


def test_licence_export_refuses_a_cut_without_the_pattern_file(tmp_path, monkeypatch, capsys):
    """THE BREAKAGE: the cut rewrites the lines its content rules find; with the generic rule alone every private
    line would pass into the export."""
    monkeypatch.delenv("HEX_PRIVATE_PATTERNS", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n")
    destination = tmp_path / "export"
    assert _licence_export().main(["apply", str(source), str(destination), "--engine-source", str(tmp_path)]) == 2
    assert "no private pattern file" in capsys.readouterr().err
    assert not destination.exists()


def test_sealed_comment_exception_never_allows_private_content(hygiene, patterns):
    path = "src/hexcore/data/compiler-recipes.json"
    assert hygiene.scan([(path, b"// hostling-2")], list(hygiene.RULES)) == []
    assert hygiene.scan([(path, b"// zorbtool")], list(hygiene.RULES))
    assert hygiene.scan([("docs/build.md", b"hostling-2")], list(hygiene.RULES))


@pytest.mark.parametrize("text", ["--zorb-background", "run --zorb-background file.nc"])
def test_public_contracts_pass_in_source_and_artifacts(hygiene, patterns, text):
    for path in ("src/driver.py", "src/data/compiler-recipes.json", "README.md"):
        assert hygiene.scan([(path, text.encode())], list(hygiene.RULES)) == []


@pytest.mark.parametrize("private", ['"zorb-private"', "zorbtool"])
def test_public_contract_allowlist_cannot_hide_a_private_rule(hygiene, patterns, private):
    public = "--zorb-background "
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
