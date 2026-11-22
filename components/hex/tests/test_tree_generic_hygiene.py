"""Every file git tracks in the distribution tree passes the release's generic hygiene rules.

THE BREAKAGE (0.6.2, 20 Nov 2026): the learned-correction merge brought its training driver scripts
(tools/torch-bridge/*.sh: work folders, home folders, a card pool, machine addresses) into the tree. The
unit suite was green, and the publish workflow's own step "refuse private material in both release artifacts"
(tools/release_hygiene.py) refused the source package for one home path in tools/torch-bridge/baseline.sh, so no
release could build. This scan is that step's generic rules over the tracked tree, run before any build.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[1]
SCRIPT = TREE / "tools" / "release_hygiene.py"


def _hygiene():
    spec = importlib.util.spec_from_file_location("release_hygiene_tree_scan", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the scan pickles its worker by module name for its process pool
    spec.loader.exec_module(module)
    return module


def test_the_tracked_tree_has_no_generic_hygiene_hit():
    probe = subprocess.run(["git", "-C", str(TREE), "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip("not a git checkout (an unpacked source package): the publish step scans the package itself")
    hygiene = _hygiene()
    hygiene.configure(None)
    try:
        hits = hygiene.scan(hygiene.tree_members(TREE), list(hygiene.GENERIC_RULES))
    finally:
        hygiene.configure(os.environ.get(hygiene._content.PRIVATE_ENV))
    assert hits == [], "files the release build would refuse:\n" + "\n".join(
        f"  {path} :: {rule} :: {text}" for path, rule, text in hits)


def test_the_lab_run_scripts_are_not_in_the_tree():
    # The training scripts read machine folders and a card pool that a user does not have.
    assert not (TREE / "tools" / "torch-bridge").exists()
    assert not (TREE / "tools" / "ad-box").exists()
