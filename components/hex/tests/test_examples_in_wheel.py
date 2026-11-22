"""The runnable examples travel in the wheel, and a command finds them.

THE BREAKAGE (0.6.0, a fresh install from the published files): the README, FIRST-HOUR and the differentiable pages
pointed at `examples/differentiable/`, and `pip install "recast-woof[ad]"` placed none of it. A user had to download
the source archive to find `run-32gb.sh`. The wheel build now copies `examples/differentiable/` to
`woof/hex/examples/differentiable/`, and `woof hex examples` prints that folder or copies it.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import examples_door  # noqa: E402

NEEDED = ("README.md", "expected.json", "own_measure.py", "run-32gb.sh", "run-96gb.sh")


def _setup_source():
    return (ROOT / "setup.py").read_text(encoding="utf-8")


def test_the_wheel_build_copies_the_same_sets_the_command_lists():
    assert examples_door.SETS == ("differentiable",)
    for name in NEEDED:
        assert (ROOT / "examples/differentiable" / name).is_file(), name
    if not (ROOT / "setup.py").is_file() or "EXAMPLE_SETS" not in _setup_source():
        import pytest
        pytest.skip("this tree is built by another project's packaging, not woof hex's setup.py")
    tree = ast.parse(_setup_source())
    sets = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EXAMPLE_SETS")
    assert tuple(sets) == examples_door.SETS == ("differentiable",)
    source = _setup_source()
    assert '"woof/hex/examples"' in source and 'root / "examples" / name' in source, (
        "setup.py no longer copies examples/<set> into the wheel's woof/hex/examples/")
    for name in NEEDED:
        assert (ROOT / "examples/differentiable" / name).is_file(), name
    # The source archive keeps carrying the same folder.
    assert "graft examples/differentiable" in (ROOT / "MANIFEST.in").read_text(encoding="utf-8")


def test_an_installed_package_finds_its_own_copy_first(tmp_path, monkeypatch):
    package = tmp_path / "site-packages/hexcore"
    (package / "examples/differentiable").mkdir(parents=True)
    for name in NEEDED:
        (package / "examples/differentiable" / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(examples_door, "__file__", str(package / "examples_door.py"))
    assert examples_door.examples_root() == package / "examples"
    assert [path.name for path in examples_door.example_files(package / "examples")] == sorted(NEEDED)


def test_a_checkout_finds_the_tree_copy():
    root = examples_door.examples_root()
    assert root is not None and (root / "differentiable/run-32gb.sh").is_file()


def test_the_command_prints_the_folder_and_copies_it(tmp_path, capsys):
    assert examples_door.run_examples(argparse.Namespace(copy=None, force=False)) == 0
    said = capsys.readouterr().out
    root = examples_door.examples_root()
    assert f"examples: {root}" in said and "differentiable/run-32gb.sh" in said and "--copy DIR" in said
    target = tmp_path / "mine"
    assert examples_door.run_examples(argparse.Namespace(copy=str(target), force=False)) == 0
    for name in NEEDED:
        assert (target / "differentiable" / name).read_bytes() == (root / "differentiable" / name).read_bytes()
    assert (target / "differentiable/run-32gb.sh").stat().st_mode & 0o111
    # A copy over a file the user changed stops and changes nothing; --force replaces it.
    (target / "differentiable/own_measure.py").write_text("mine\n", encoding="utf-8")
    assert examples_door.run_examples(argparse.Namespace(copy=str(target), force=False)) == 2
    assert (target / "differentiable/own_measure.py").read_text(encoding="utf-8") == "mine\n"
    assert "nothing was copied" in capsys.readouterr().err
    assert examples_door.run_examples(argparse.Namespace(copy=str(target), force=True)) == 0
    assert (target / "differentiable/own_measure.py").read_bytes() == (root / "differentiable/own_measure.py").read_bytes()


def test_an_install_without_them_says_where_they_are(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(examples_door, "__file__", str(tmp_path / "a/b/hexcore/examples_door.py"))
    assert examples_door.run_examples(argparse.Namespace(copy=None, force=False)) == 1
    assert "source archive" in capsys.readouterr().err


def test_the_front_door_has_the_command():
    from woof.hex import cli
    assert any("examples" in names for names, _ in cli._DOORS)


def test_the_pages_name_the_command_where_they_point_at_the_examples():
    for page in ("README.md", "docs/differentiable/FIRST-HOUR.md", "docs/differentiable/README.md",
                 "examples/differentiable/README.md"):
        text = (ROOT / page).read_text(encoding="utf-8")
        assert "woof hex examples" in text, f"{page} points at the examples without the command that finds them"
