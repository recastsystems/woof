"""The disclaimer receipts stay bound to the tree they describe.

A committed receipt that nothing re-checks is a snapshot of a claim, not
evidence for it: edit the docs, leave the JSON alone, and the receipt now
describes a tree that no longer exists.  These tests re-measure the
essential fields against the live tree, so drift fails the build rather
than ageing quietly.

The internal half is checked here too.  The performance-gate sentence is
only allowed to be written in a release-excluded file, so the exclusion
line and the sentence's own vocabulary are both gates, not notes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.check_negation_invariant import check
from tools.report_resolved_scale_disclaimer import (
    GATE_BLOCK_FORBIDDEN,
    GATE_BLOCK_SUBSTRINGS,
    INTERNAL_RECEIPT,
    INTERNAL_SPEC,
    PUBLIC_RECEIPT,
    SCALE_PAGE,
    README_ITEM_SUBSTRINGS,
    README_PINNED_FIGURE_SUBSTRINGS,
    VERIFICATION_ITEM_SUBSTRINGS,
    VERIFICATION_PINNED_FIGURE_SUBSTRINGS,
    find_item,
    gate_block,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The internal half's fixtures are the very files ``RELEASE-EXCLUDE.txt``
#: removes from the published snapshot, so in a public clone they are
#: absent *because the gate worked*.  Failing there would make the
#: exclusion working look like a broken test suite; these tests skip
#: instead, and stay hard failures in the development tree where the
#: files exist.  Everything above this line is checked in both.
_INTERNAL_HALF_PRESENT = (
    INTERNAL_RECEIPT.is_file() and (REPO_ROOT / INTERNAL_SPEC).is_file()
)
internal_half = pytest.mark.skipif(
    not _INTERNAL_HALF_PRESENT,
    reason=(
        "the internal half of this receipt lives under a release-excluded "
        "path; a published clone does not have it, which is the property "
        "test_the_gate_sentence_lives_where_the_release_cut_drops_it "
        "asserts in the development tree"
    ),
)


def _public() -> dict:
    return json.loads(PUBLIC_RECEIPT.read_text(encoding="utf-8"))


def _internal() -> dict:
    return json.loads(INTERNAL_RECEIPT.read_text(encoding="utf-8"))


def _outside_readme(occurrences):
    """Recast WOOF ships its own README.md (the receipt measured the engine's);
    every docs/public occurrence is still compared."""
    return [row for row in occurrences if row["path"] != "README.md"]


def test_public_receipt_negation_verdict_still_describes_the_tree():
    recorded = _public()["negation_invariant"]["post_edit"]
    live = check(REPO_ROOT)
    assert recorded["verdict"] == live["verdict"]
    assert _outside_readme(recorded["occurrences"]) == _outside_readme(live["occurrences"])


def test_public_receipt_records_the_three_mutants_as_discriminating():
    control = _public()["negation_invariant"]["mutation_control"]
    assert set(control["mutants"]) == {"M1", "M2", "M3"}
    assert all(
        entry["differs_from_unmutated"] for entry in control["mutants"].values()
    )
    assert control["adopted"] is True
    assert control["mutant_files_committed"] is False


def test_published_items_still_carry_the_authorized_prose():
    receipt = _public()
    assert receipt["figure_variant_authorized"] == "pinned"

    # The README links the scale note at this release's own tag (bump_version moves it every release).
    # Recast WOOF's own README carries no scale note; where it links one, the link is this release's tag.
    version = (REPO_ROOT / "recast-woof-data/woof_data/VERSION").read_text().strip()
    readme_text = (REPO_ROOT / "README.md").read_text()
    if "RESOLVED-SCALE.md" in readme_text:
        assert (f"[scale note](https://github.com/recastsystems/woof/blob/v{version}/docs/public/RESOLVED-SCALE.md)"
                in readme_text)
    readme = find_item(REPO_ROOT / SCALE_PAGE, "- **Resolved scale.**")
    assert readme["item_count"] == 1
    for token in README_ITEM_SUBSTRINGS + README_PINNED_FIGURE_SUBSTRINGS:
        assert token in readme["normalized"], token

    verification = find_item(
        REPO_ROOT / "docs" / "public" / "VERIFICATION.md",
        "- **No resolved tornado dynamics.**",
    )
    assert verification["item_count"] == 1
    for token in (
        VERIFICATION_ITEM_SUBSTRINGS + VERIFICATION_PINNED_FIGURE_SUBSTRINGS
    ):
        assert token in verification["normalized"], token


def test_no_case_identity_reached_the_published_disclaimer():
    scan = _public()["case_identity_scan"]
    assert scan["mode"] == "added-lines-only"
    assert scan["added_line_count"] > 0
    assert scan["hit_count"] == 0, scan["hits"]


def _declared_revisions(scope: dict) -> list[str]:
    """The endpoints ``edit_range`` says the diff fields were measured over."""
    return [scope["base"]] if scope["tip"] is None else [
        scope["base"], scope["tip"]
    ]


def _resolves_to(ref: str, sha: str) -> bool:
    """A recorded ref names a recorded commit (abbreviated or in full)."""
    return sha.startswith(ref) or ref == sha


def test_every_diff_field_measured_the_change_the_receipt_declares():
    """A diff is only evidence once you know what it was diffed against.

    Once this branch is merged into a tree that other work also landed in,
    ``git diff <base>`` no longer names this change -- it names everything
    since.  Regenerating with that reach silently re-scopes the receipt:
    the structure counts start measuring other lanes' prose, the case scan
    starts reporting other lanes' config names, and the untouched-
    announcement claim inverts on an edit this package never made.

    So the range is recorded, and every diff-measured field has to name the
    same one.  This does not pin a particular range -- a branch still being
    worked on records one endpoint and a merged one records two -- it pins
    the receipt to being internally accurate about its own scope.
    """
    receipt = _public()
    scope = receipt["edit_range"]
    declared = _declared_revisions(scope)
    assert declared, scope

    measured = [receipt["case_identity_scan"]["revisions"]]
    measured += [entry["revisions"] for entry in receipt["markdown_structure"]]
    for revisions in measured:
        assert len(revisions) == len(declared), (revisions, declared)
        for ref, sha in zip(revisions, declared):
            assert _resolves_to(ref, sha), (ref, sha)

    # The commands that carry no structured field say it in their text.
    command = receipt["announcement_draft_untouched"]["command"]
    for ref in measured[0]:
        assert ref in command, (ref, command)


def test_the_untouched_announcement_claim_is_scoped_and_attributed():
    """The claim is about this change, and the tree is reported either way.

    ``announcement_draft_untouched`` says this package did not edit the
    draft.  On an assembled tree that is not the same statement as "nobody
    edited the draft", and a receipt that recorded only the first could be
    read as the second.  The tree's own state is therefore recorded beside
    it, and whenever the tree did move the file, the commits that moved it
    have to be named.
    """
    receipt = _public()
    assert receipt["announcement_draft_untouched"]["stdout"] == ""
    assert receipt["announcement_draft_untouched"]["exit_code"] == 0

    in_tree = receipt["announcement_draft_in_the_tree"]
    moved = bool(in_tree["diffstat"]["stdout"].strip())
    attributed = bool(in_tree["commits_since_base"]["stdout"].strip())
    assert moved == attributed, in_tree


@internal_half
def test_gate_sentence_names_the_instrument_and_disowns_the_wrong_one():
    block = gate_block(REPO_ROOT / INTERNAL_SPEC)
    assert block["block_count"] == 1
    for token in GATE_BLOCK_SUBSTRINGS:
        assert token in block["normalized"], token
    assert block["disowns_dual_run_screen"]
    for token in GATE_BLOCK_FORBIDDEN:
        assert token not in block["normalized"], token


@internal_half
def test_the_instrument_qualification_rule_is_declared_before_it_runs():
    """A rule chosen after the arms are seen is not a control."""
    control = _internal()["instrument_qualification"]
    assert control["status"] == "deferred"
    assert control["ratified_as"] == "D-37"
    assert set(control["arms"]) == {"a", "b"}
    assert "skip_or_run" in control["records_per_arm"]
    assert "hard failure" in control["skip_policy"]
    assert (REPO_ROOT / control["suite"]).is_file()


@internal_half
def test_gate_sentence_cites_only_paths_that_resolve():
    cited = _internal()["cited_paths_resolve"]
    assert cited
    for path, record in cited.items():
        assert record["resolves"], path
        assert (REPO_ROOT / path).is_file(), path


@internal_half
def test_the_gate_sentence_lives_where_the_release_cut_drops_it():
    excluded = (REPO_ROOT / "RELEASE-EXCLUDE.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert "docs/internal/**" in excluded
    assert INTERNAL_SPEC.startswith("docs/internal/")
    proof = _internal()["public_clone_proof"]
    assert proof["status"] == "run"
    assert proof["subtree_absent_from_public_clone"] is True
    assert proof["tracked_count"] == 0


def _exclusion_patterns() -> list[str]:
    return [
        line.strip()
        for line in (REPO_ROOT / "RELEASE-EXCLUDE.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _excluded(path: Path, patterns: list[str]) -> bool:
    """Match a repo-relative path against the exclusion globs.

    ``PurePath.match`` does not give ``**`` its recursive meaning before
    3.13, so the pattern is translated here: ``**`` spans separators,
    ``*`` does not.
    """
    posix = path.relative_to(REPO_ROOT).as_posix()
    for pattern in patterns:
        expr = "".join(
            ".*" if part == "**" else re.escape(part).replace(r"\*", "[^/]*")
            for part in re.split(r"(\*\*)", pattern)
        )
        if re.fullmatch(expr, posix):
            return True
    return False


def test_public_receipt_survives_the_release_exclusion_list():
    """The evidence for a public claim has to ship with the public cut."""
    patterns = _exclusion_patterns()
    assert not _excluded(PUBLIC_RECEIPT, patterns), patterns
    # Negative control: the internal receipt IS excluded, so the check
    # above discriminates rather than passing vacuously.
    assert _excluded(INTERNAL_RECEIPT, patterns)
    assert _excluded(REPO_ROOT / INTERNAL_SPEC, patterns)
