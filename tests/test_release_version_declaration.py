"""The version in ``pyproject.toml`` must be the one being cut.

This file exists because the 2.5.0 integration line carried
``version = "2.4.1"`` -- the number of the newest *published* release --
for its whole life, while ``## Unreleased`` in ``CHANGELOG.md`` filled up
with the work that line exists to ship.

Nothing in the tree noticed, and nothing could have: ``woof.__version__``
is read from installed distribution metadata, so it reports whatever pip
last wrote; the publish workflow's ``cut`` job reads the version out of
``pyproject.toml`` and only compares it against the *tag it was handed*.
Tag ``v2.4.1`` against pyproject ``2.4.1`` agrees perfectly and republishes
an already-published number; tag ``v2.5.0`` against pyproject ``2.4.1``
fails at job 3 of 10 -- after the tag is pushed and the draft exists, which
is to say after the number is spent.

So the oracle here is the changelog, which is the only place in the tree
that records what has actually shipped.  Two invariants come out of it and
both are mechanical:

* the declared version may not be a version that already shipped, and
* it must lead every entry the changelog records.

Both are conditioned the way the release cycle actually runs.  On the day
of a cut the version and the changelog head SHOULD be equal -- that is the
state the workflow's tag comparison wants -- so the equality gate keys off
``## Unreleased`` carrying pending work, which is the one arrangement in
which a spent number is unambiguously wrong.  Nobody has to remember to
bump anything: the first commit that lands work under a published number
fails here by name.
"""

from __future__ import annotations

from pathlib import Path
import re
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

#: ``## 2.4.1 (2026-08-14)``, and the tagged-but-unpublished spellings
#: beside it (``## 1.8.1 (2026-08-07, tagged but not published)``).
_RELEASE_HEADING = re.compile(r"^##\s+(\d+\.\d+\.\d+)\s*\(")

_UNRELEASED_HEADING = re.compile(r"^##\s+Unreleased\s*$", re.IGNORECASE)


def _require_source_tree() -> None:
    if not PYPROJECT.is_file() or not CHANGELOG.is_file():
        pytest.skip("the release-version gate needs the source tree")


def declared_version() -> str:
    with PYPROJECT.open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def _release_sections() -> list[str]:
    """Every version the changelog records as cut, newest first."""

    found: list[str] = []
    for line in CHANGELOG.read_text(encoding="utf-8").splitlines():
        match = _RELEASE_HEADING.match(line)
        if match:
            found.append(match.group(1))
    return found


def _unreleased_body() -> str | None:
    """The text under ``## Unreleased``, or ``None`` if there is no such head."""

    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not _UNRELEASED_HEADING.match(line):
            continue
        body: list[str] = []
        for candidate in lines[index + 1:]:
            if candidate.startswith("## "):
                break
            body.append(candidate)
        return "\n".join(body).strip()
    return None


def _as_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def test_the_changelog_can_be_read_at_all() -> None:
    """The oracle has to exist, or the two gates below are vacuous.

    A changelog whose heading style drifts would turn both assertions into
    silent passes -- the same class of failure as a skipped packaging gate.
    """

    _require_source_tree()
    sections = _release_sections()
    assert len(sections) >= 5, (
        "found "
        f"{len(sections)} release heading(s) in CHANGELOG.md; the "
        f"'## X.Y.Z (date)' style this file parses has changed, so the "
        f"version gates below are no longer measuring anything")
    assert re.fullmatch(r"\d+\.\d+\.\d+", declared_version()), (
        f"pyproject version {declared_version()!r} is not a stable X.Y.Z; the "
        "publish workflow's cut job only refuses that when "
        "stable_release_expected is ticked, so decide deliberately")


def test_the_declared_version_is_never_behind_the_changelog() -> None:
    """The floor, and it holds at every point in the cycle.

    On the day of a cut the two are equal -- ``## 2.5.0 (date)`` sits at the
    head of the changelog and ``pyproject.toml`` says ``2.5.0`` -- which is
    the state the workflow's tag comparison wants.  What can never be true
    is the version trailing the changelog, because then the tag the cut
    demands names a release that has already gone out.
    """

    _require_source_tree()
    version = declared_version()
    newest = max(_release_sections(), key=_as_tuple)
    assert _as_tuple(version) >= _as_tuple(newest), (
        f"pyproject.toml declares {version} while CHANGELOG.md already "
        f"records {newest} as released. The publish workflow's `cut` job "
        f"requires the tag to equal v<pyproject version>, so this tree can "
        f"only cut v{version} -- a number that has already shipped.")


def test_pending_work_is_not_sitting_under_a_published_number() -> None:
    """THE defect, and the one state in which it is unambiguous.

    ``pyproject.toml`` said 2.4.1 on the 2.5.0 integration line while
    ``## 2.4.1 (2026-08-14)`` sat in the changelog under 27 lines of
    ``## Unreleased`` work.  An ``## Unreleased`` section with content says
    a cut is pending; a version field naming an already-published release
    says the cut has nowhere to go.  Both cannot be true.

    Conditioning on the pending section is what keeps this accurate rather
    than merely strict: once the changelog lane renames ``## Unreleased``
    to ``## X.Y.Z (date)``, version and changelog head SHOULD agree, and
    this gate steps aside instead of failing every release cut.
    """

    _require_source_tree()
    body = _unreleased_body()
    if not body:
        pytest.skip("no pending Unreleased entries to attribute to a version")
    version = declared_version()
    shipped = _release_sections()
    newest = max(shipped, key=_as_tuple)
    assert version not in shipped and _as_tuple(version) > _as_tuple(newest), (
        f"CHANGELOG.md's `## Unreleased` section carries "
        f"{len(body.splitlines())} line(s) of pending work while "
        f"pyproject.toml still declares {version}, which shipped as "
        f"`## {newest}`. Bump the version to the one this line will cut: the "
        f"`cut` job compares the tag against this field, so tagging "
        f"v{version} republishes a spent number and tagging anything else "
        f"fails at job 3 of 10 -- after the tag is pushed, which is to say "
        f"after the number is spent. That is how 2.4.0 was burned.")


def test_the_cut_job_still_reads_the_version_from_pyproject() -> None:
    """One source of truth, and the workflow must still be reading it.

    Everything above pins ``pyproject.toml``. That is only worth pinning
    while the release cut is the thing that consumes it -- if the workflow
    ever grows a second version input, the tag/version agreement it proves
    stops being about this field.
    """

    workflow = REPO_ROOT / ".github" / "workflows" / "publish.yml"
    if not workflow.is_file():
        pytest.skip("no publish workflow in this tree")
    text = workflow.read_text(encoding="utf-8")
    if "promote_prepared_release.py" not in text:
        # WOOF's publish workflow uploads a published release's own files; the
        # engine's cut job, which this test pins, is not in this tree.
        pytest.skip("no engine cut job in this tree")
    controller = (REPO_ROOT / "tools/promote_prepared_release.py").read_text(encoding="utf-8")
    assert "promote_prepared_release.py verify" in text
    assert 'config = pyproject["project"]' in controller
    assert 'declared = config.get("version")' in controller
    assert 'require(declared == version, "pyproject version differs from release tag")' in controller
    assert 'metadata.get("Version") == version' in controller
