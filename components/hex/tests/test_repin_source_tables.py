"""A repin cannot label dirty or different engine bytes as a named commit."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import repin_source_tables as repin


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def engine(tmp_path):
    root = tmp_path / "engine"
    root.mkdir()
    (root / "woof").mkdir()
    (root / "woof/seam.py").write_bytes(b"SEAM = 1\n")
    (root / "pyproject.toml").write_text('[project]\nversion = "2.8.7"\n')
    git(root, "init")
    git(root, "config", "user.name", "fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-m", "fixture")
    return root, git(root, "rev-parse", "HEAD")


def test_exact_commit_and_bytes_are_measured(engine):
    root, commit = engine
    assert repin.verify_engine_tree(root, commit, "2.8.7", ["woof/seam.py"]) == {
        "woof/seam.py": hashlib.sha256(b"SEAM = 1\n").hexdigest()}


@pytest.mark.parametrize("commit", ["HEAD", "main", "1234567"])
def test_moving_or_short_commit_is_refused(engine, commit):
    with pytest.raises(SystemExit, match="full immutable commit"):
        repin.verify_engine_tree(engine[0], commit, "2.8.7", ["woof/seam.py"])


def test_different_head_is_refused(engine):
    with pytest.raises(SystemExit, match="refusing false provenance"):
        repin.verify_engine_tree(engine[0], "0" * 40, "2.8.7", ["woof/seam.py"])


@pytest.mark.parametrize("relative", ["woof/seam.py", "pyproject.toml"])
def test_changed_working_bytes_are_refused(engine, relative):
    root, commit = engine
    (root / relative).write_bytes(b"changed\n")
    with pytest.raises(SystemExit, match="differs from"):
        repin.verify_engine_tree(root, commit, "2.8.7", ["woof/seam.py"])


def test_version_must_be_the_commits_declaration(engine):
    with pytest.raises(SystemExit, match="version 2.8.7 != requested 2.8.8"):
        repin.verify_engine_tree(*engine, "2.8.8", ["woof/seam.py"])


def test_manifest_cannot_read_outside_the_engine(engine):
    with pytest.raises(SystemExit, match="unsafe engine manifest path"):
        repin.verify_engine_tree(*engine, "2.8.7", ["gpuwm/../../outside"])


def test_requirement_stops_at_first_unqualified_engine():
    def row(version, moved=(), on_pypi=True, offline_build_road=True):
        return dict(version=version, moved=list(moved), on_pypi=on_pypi,
                    offline_build_road=offline_build_road)
    rows = [row("2.8.8", ["woof/seam.py"]), row("2.8.7"),
            row("2.8.0", on_pypi=False), row("2.8.9")]
    # plan call 14: the floor's whole minor, so a moved 2.8.8 no longer caps it
    assert repin.measured_requirement(rows) == "woof>=2.8.7,<2.9"
    rows[1]["offline_build_road"] = False
    assert repin.measured_requirement(rows) == "woof>=2.8.9,<2.9"
    assert repin.measured_requirement([row("2.9.0"), row("2.9.1", ["woof/seam.py"])]) == "woof>=2.9.0,<2.10"


def test_no_unqualified_dependency_is_invented():
    with pytest.raises(SystemExit, match="no measured published engine"):
        repin.measured_requirement([dict(version="2.8.7", moved=["woof/seam.py"],
                                         on_pypi=True, offline_build_road=True)])


def test_engine_options_are_all_required_and_offline_is_not_silently_ignored():
    with pytest.raises(SystemExit) as error:
        repin.main(["--engine-tree", ".", "--write"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        repin.main(["--offline"])
    assert error.value.code == 2


def test_engine_repin_rederives_the_contract_surface_from_the_pinned_bytes(tmp_path):
    """THE BREAKAGE: the repin left MPAS_SEAM_CONTRACT_SURFACE_SHA256 naming
    the old column batch, so an engine whose mpas_column_batch.py moved
    passed the manifest and refused every forecast at the surface check."""

    root = tmp_path / "engine"
    payloads = {"woof/core/mpas_column_batch.py": b"def run_mpas_column_batch(): ...\n",
                "docs/mpas-seam.md": b"# the seam\n"}
    for relative, payload in payloads.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(payload)
    manifest = {relative: hashlib.sha256(payload).hexdigest()
                for relative, payload in payloads.items()}
    expected = hashlib.sha256(b"".join(payloads[r] for r in repin.CONTRACT_SURFACE_PATHS))
    assert repin.contract_surface_digest(root, manifest) == expected.hexdigest()
    (root / "docs/mpas-seam.md").write_bytes(b"# another seam\n")
    with pytest.raises(SystemExit, match="not the manifest's bytes"):
        repin.contract_surface_digest(root, manifest)

    adapter = tmp_path / "adapter.py"
    adapter.write_text('MPAS_SEAM_CONTRACT_SURFACE_SHA256 = (\n    "' + "a" * 64 + '"\n)\n',
                       encoding="utf-8", newline="")
    repin._rewrite_contract_surface(adapter, "b" * 64)
    assert ('"' + "b" * 64 + '"') in adapter.read_text(encoding="utf-8")


def test_the_repin_measures_and_writes_the_helper_set_row(tmp_path, monkeypatch):
    """A release review of 0.4.0 found that the repin moved the manifest and not the helper rows,
    so every default forecast on the frozen 2.9.0 was refused at start."""
    import json

    root = tmp_path / "engine"
    names = sorted(json.loads((repin.TREE / "../../woof/hex/radiation_source_sets.json").read_text())["baseline"])
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(name.encode() + b" v2.9.1\n")
    (root / "pyproject.toml").write_text("[project]\nversion = \"2.9.1\"\n")
    git(root, "init")
    git(root, "config", "user.name", "fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-m", "fixture")
    commit = git(root, "rev-parse", "HEAD")
    row = repin.helper_set_row(root, commit, "2.9.1")
    assert row["id"] == "engine-2.9.1-source" and row["engine_commit"] == commit
    assert sorted(row["sources"]) == names
    table = tmp_path / "engine-helper-sets.v1.json"
    table.write_text(repin.HELPER_SETS.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(repin, "HELPER_SETS", table)
    assert not repin.helper_set_admitted(row)
    repin.write_helper_set(row)
    assert repin.helper_set_admitted(row)
    before = table.read_bytes()
    repin.write_helper_set(row)  # the same bytes again add nothing
    assert table.read_bytes() == before
    with pytest.raises(SystemExit, match="never rewritten in place"):
        repin.write_helper_set(dict(row, sources={name: "0" * 64 for name in names}))
