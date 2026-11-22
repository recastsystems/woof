"""Skip a case in the public export when its input is a file the export drops.

THE BREAKAGE THIS PREVENTS
--------------------------
Stage 1 (``tools/battery/stage1_files.txt``) runs on the public candidate:
the commit the export builds from the private tree minus
``RELEASE-EXCLUDE.txt``.  A shipped test that reads a file the export drops
passes in the development tree and fails at the exported commit.  Dry cut 4
of 2.8.0 measured seven such cases: five evidence-pack cases that bind the
nesting ledger under ``docs/internal/**``, the speed-anchor case that
opens its run report under ``evidence/**`` and the rescued-tools control
that needs ``tilestream/rescued-tools/**`` on disk.  All seven are correct
where their input exists and cannot run where it does not.  The speed-anchor
case has since stopped needing this: its run report ships with the public
receipts.  A sixth evidence-pack case joined, because the pack now refuses
before reading a rung on a tree without the ledger.

WHEN IT SKIPS
-------------
Only in an exported tree, and only when every named input that is absent is
one ``RELEASE-EXCLUDE.txt`` drops.  The tree counts as an export when the
snapshot builder, ``work/build_release_snapshot.py``, is absent: it lives
under ``work/**``, which the export drops, and every development checkout
carries it.  So:

* in the development tree nothing is skipped; a missing input fails the
  case where it reads the file, naming the path;
* in an export, an input that is absent but NOT excluded fails the same
  way, because its absence is not the export working;
* if ``RELEASE-EXCLUDE.txt`` stops dropping the builder, the two trees can
  no longer be told apart and nothing is skipped anywhere.

The skip reason names the input and the exclusion rule that dropped it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.release_exclusions import matches, read_exclusions

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Present in every development checkout, dropped by ``work/**`` from every
#: export.  Its absence is how a test knows it runs on the public tree.
SNAPSHOT_BUILDER = "work/build_release_snapshot.py"


def export_skip_reason(*inputs: str, root: Path = REPOSITORY_ROOT) -> str | None:
    """Why a case reading ``inputs`` cannot run in this tree, or ``None``.

    ``inputs`` are repository-relative, forward-slash paths to files or
    directories.  Returns a reason only when ``root`` is an export and every
    absent input is dropped by the tree's own ``RELEASE-EXCLUDE.txt``.
    """

    if not inputs:
        raise ValueError("name the release-excluded input the case reads")
    root = Path(root)
    exclusions = root / "RELEASE-EXCLUDE.txt"
    if not exclusions.is_file():
        return None
    rules = read_exclusions(root)
    if (root / SNAPSHOT_BUILDER).is_file() or matches(SNAPSHOT_BUILDER, rules) is None:
        return None
    absent = [name for name in inputs if not (root / name).exists()]
    dropped = [(name, matches(name, rules)) for name in absent]
    if not absent or any(rule is None for _, rule in dropped):
        return None
    named = "; ".join(f"{name} (RELEASE-EXCLUDE.txt: {rule})" for name, rule in dropped)
    return (f"reads {named}, which the public export drops; this case runs in "
            "the development tree")


def private_input(*inputs: str) -> pytest.MarkDecorator:
    """A ``skipif`` mark for a case whose input the public export drops."""

    reason = export_skip_reason(*inputs)
    return pytest.mark.skipif(reason is not None, reason=reason or "")


def skip_if_export_drops(*inputs: str) -> None:
    """The runtime form, for an input path the case reads out of data."""

    reason = export_skip_reason(*inputs)
    if reason is not None:
        pytest.skip(reason)
