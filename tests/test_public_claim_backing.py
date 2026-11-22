"""Every public t=0 parity claim is backed by a receipt, or it is pending.

The external review's finding was not that a number was wrong.  It was
that six published sentences state a t=0 result whose breadth nothing in
the tree measures: the matched-run comparator scores eight carriers, and
the sentences generalize from them to the initial state.  A doc fix alone
would just move the unbacked sentence somewhere else, so the durable
artifact is this guard -- it reads the claim, resolves the receipt the
claim links, and compares the breadth the sentence asserts against the
``covered_groups`` the receipt measured.

The guard is ENFORCED: there is no carried list any more.  Every audited
site was rewritten against the committed receipt, so any finding at all in
the published set fails the build.  The published set is ``docs/public``
AND the repository root, because the prediction above came true: rooted at
``docs/public`` alone, this gate was green through a release in which
``README.md``'s verification table carried the retired sentence over a
receipt whose verdict is FAIL.  See :func:`published_markdown`.

Four rules bite.  Three read the claim -- is it linked to a receipt, does
it infer the whole state from one carrier group, is it broader than the
receipt's ``covered_groups``.  The fourth reads the receipt back: a block
that cites a receipt must state that receipt's verdict.  Without it a
sentence could assert parity while linking a receipt that says FAIL, and
every other rule would pass it, because breadth and backing say nothing
about which way the measurement came out.

The essential tests in this file are the negative controls: the guard
must REJECT a fixture containing the retired surface-to-full-state
sentence, must REJECT that same sentence when it is quoted rather than
written plainly -- and without letting a neighbouring quoted paragraph's
receipt back it -- and must REJECT a fixture that cites a failing receipt
without saying so.  A guard that only ever passes is not evidence of
anything.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from woof.verify.t0_state_digest import (
    BOUNDARY_GROUP,
    CARRIER_GROUPS,
    SCHEMA_ID,
)
from tools.release_exclusions import matches, read_exclusions

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = REPO_ROOT / "docs" / "public"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "public_claim_backing"

#: A block is a t=0 claim when it names the initial state AND asserts
#: agreement about it.  Either alone is prose -- "the initial state is
#: interpolated by real.exe" states no result, and "the forecast matches"
#: is not about t=0.
_T0_MARKER = re.compile(
    r"\bt\s*=\s*0\b|initiali[sz]ation parity|initial(?:-|\s+)condition"
    r"|initial\s+(?:\w+\s+){0,2}state", re.IGNORECASE)
_PARITY_MARKER = re.compile(
    r"parit|reproduc|bit-identical|bit identical|floor\b|agree", re.IGNORECASE)

#: Phrases that make a claim a claim about the state as a whole.
_WHOLE_STATE = (
    "initial state", "initial-condition error", "initial condition error",
    "prognostic state", "full state", "whole state", "full 3-d state",
    "initialization parity", "initialisation parity",
)

#: Prose terms that scope a claim to one registered carrier group.  The key
#: set is asserted equal to the registered group names below, so a group
#: added to the digest cannot be silently unaddressable in prose.
GROUP_CLAIM_TERMS: dict[str, tuple[str, ...]] = {
    "dry_dynamics": ("dry dynamics", "wind field", "geopotential",
                     "mass field", "dry-air mass"),
    "moisture": ("moisture", "water vapor", "water vapour", "hydrometeor",
                 "mixing ratio"),
    "surface": ("surface", "near-surface", "skin temperature",
                "two-metre", "ten-metre"),
    "soil": ("soil", "snowpack", "land state", "canopy"),
    "accumulation": ("accumulator", "accumulated precipitation"),
    "diagnostic": ("diagnostic field", "reflectivity"),
    BOUNDARY_GROUP: ("lateral boundary", "boundary table", "boundary group"),
}

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
#: Link *text* is prose; a link *target* is a path.  Scanning the target
#: would let a receipt named ``surface_only.json`` scope the sentence that
#: cites it, which is backwards.
_LINK_PROSE_RE = re.compile(r"\[([^\]]*)\]\([^)\s]+\)")
_LIST_START_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")
#: Sentence boundary: terminal punctuation, whitespace, then something that
#: starts a sentence.  Version numbers and abbreviations inside a sentence
#: are followed by lower case, so they do not split.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*_`#0-9-])")
#: A blockquote marker is presentation, and leaving it in the text made the
#: guard read quoted prose as something it is not.  Both splitters miss it:
#: ``> * `` is not a list start, so a quoted run of bullets lumps into one
#: block and one bullet's receipt link backs its neighbour's claim -- the
#: exact leak the block rule exists to prevent -- and ``>`` is not a
#: sentence opener, so quoted prose never splits into sentences at all and
#: markers bleed across bullets.  Both were live: a runbook bullet stating
#: the t=0 history frame carries no ``REFL_10CM`` (a fact about the frame,
#: asserting no result) fused with the next bullet's "coverage floor" and
#: was reported as one unbacked parity claim.  Stripping the marker first
#: makes a quoted claim read exactly as the same claim unquoted, which is
#: strictly more of the tree scanned, not less.
_BLOCKQUOTE_RE = re.compile(r"^(\s*)(?:>\s?)+")


@dataclass(frozen=True)
class Block:
    path: str
    line: int
    text: str

    @property
    def normalized(self) -> str:
        return " ".join(self.text.split())


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    detail: str
    quote: str

    @property
    def key(self) -> tuple[str, str]:
        """Line-independent: an edit above a claim must not move its key."""
        return (self.path, self.quote)

    def __str__(self) -> str:
        return (f"{self.path}:{self.line}: [{self.rule}] {self.detail}\n"
                f"    {self.quote}")


def split_blocks(text: str, path: str) -> list[Block]:
    """Paragraphs and list items, each with the line it starts on.

    A list item is its own block: the six audited sites include bullets in
    a run of bullets, and lumping the run together would let one bullet's
    receipt link back a neighbour that has none.

    Blockquote markers come off first, so a quoted paragraph, bullet or
    fence is the same structure as an unquoted one.  Line numbers are the
    file's own either way -- the marker is removed from the text, not from
    the count.
    """
    blocks: list[Block] = []
    current: list[str] = []
    start = 0
    fenced = False

    def flush() -> None:
        if current and any(line.strip() for line in current):
            blocks.append(Block(path, start, "\n".join(current)))
        current.clear()

    for number, raw in enumerate(text.splitlines(), start=1):
        line = _BLOCKQUOTE_RE.sub(r"\1", raw)
        if line.lstrip().startswith("```"):
            fenced = not fenced
            flush()
            continue
        if fenced:
            continue
        if not line.strip():
            flush()
            continue
        if _LIST_START_RE.match(line) and current:
            flush()
        if not current:
            start = number
        current.append(line)
    flush()
    return blocks


def prose(text: str) -> str:
    """Block text with link targets replaced by their link text."""
    return " ".join(_LINK_PROSE_RE.sub(r"\1", text).split())


def claim_sentences(block: Block) -> list[str]:
    """The sentences in a block that state a t=0 result.

    Findings are per sentence, not per block: a paragraph that mentions
    reflectivity in one sentence and the initial state in another has not
    inferred one from the other, and a guard that could not tell the two
    apart would be reporting the paragraph's shape rather than its claims.
    """
    return [sentence for sentence in _SENTENCE_SPLIT_RE.split(prose(block.text))
            if _T0_MARKER.search(sentence) and _PARITY_MARKER.search(sentence)]


def claimed_groups(text: str) -> set[str]:
    lowered = text.lower()
    return {name for name, terms in GROUP_CLAIM_TERMS.items()
            if any(term in lowered for term in terms)}


def claims_whole_state(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _WHOLE_STATE)


def _linked_receipts(block: Block, root: Path) -> list[tuple[str, dict | None]]:
    """Resolve every link in the block that points at a digest receipt."""
    found: list[tuple[str, dict | None]] = []
    base = (root / block.path).parent
    for target in _LINK_RE.findall(block.text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        clean = target.split("#", 1)[0]
        if not clean.endswith(".json"):
            continue
        candidates = [base / clean, root / clean.lstrip("/")]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    document = json.loads(
                        candidate.read_text(encoding="utf-8"))
                except ValueError:
                    document = None
                if isinstance(document, dict) and document.get(
                        "schema") == SCHEMA_ID:
                    found.append((clean, document))
                break
        else:
            found.append((clean, None))
    return found


def check_block(block: Block, root: Path) -> list[Finding]:
    sentences = claim_sentences(block)
    if not sentences:
        return []

    # Receipt links are resolved over the whole block: a bullet may carry
    # its citation at the end, after the sentence that makes the claim.
    receipts = _linked_receipts(block, root)
    resolved = [document for _target, document in receipts
                if document is not None]
    unresolved = [target for target, document in receipts
                  if document is None]
    covered: set[str] = set()
    for document in resolved:
        covered.update(document.get("covered_groups") or [])

    findings: list[Finding] = []
    # A receipt that PASSed needs no disclosure: the sentence asserting
    # agreement and the receipt agreeing are the same statement.  Anything
    # else has to be said out loud in the block that cites it, or backing
    # and breadth -- neither of which knows which way the measurement came
    # out -- would carry a claim its own evidence contradicts.
    stated = prose(block.text)
    for verdict in sorted({str(document.get("verdict"))
                           for document in resolved} - {"PASS"}):
        if verdict not in stated:
            findings.append(Finding(
                block.path, block.line, "verdict-not-stated",
                f"cites a receipt whose verdict is {verdict} without stating "
                f"it; backing and breadth do not say which way a measurement "
                f"came out", sentences[0]))
    for quote in sentences:
        named = claimed_groups(quote)
        narrow = named - {BOUNDARY_GROUP}
        if narrow and claims_whole_state(quote):
            findings.append(Finding(
                block.path, block.line, "surface-to-full-state",
                f"scopes the measurement to {sorted(narrow)} and concludes "
                "about the state as a whole", quote))
        if not resolved:
            detail = ("states a t=0 result with no link to a t=0 digest "
                      "receipt" if not unresolved else
                      f"links {unresolved} which do not resolve to a "
                      f"{SCHEMA_ID} receipt")
            findings.append(Finding(block.path, block.line, "unbacked-claim",
                                    detail, quote))
            continue
        if not named:
            missing = {group.name for group in CARRIER_GROUPS
                       if group.required} - covered
            if missing:
                findings.append(Finding(
                    block.path, block.line, "breadth-exceeds-receipt",
                    f"states an unscoped t=0 result while the receipt covers "
                    f"{sorted(covered)} and omits {sorted(missing)}", quote))
        elif named - covered:
            findings.append(Finding(
                block.path, block.line, "breadth-exceeds-receipt",
                f"names {sorted(named - covered)}, which the linked receipt's "
                f"covered_groups {sorted(covered)} does not include", quote))
    return findings


def check_markdown(path: Path, root: Path) -> list[Finding]:
    rel = path.relative_to(root).as_posix()
    findings: list[Finding] = []
    for block in split_blocks(path.read_text(encoding="utf-8"), rel):
        findings.extend(check_block(block, root))
    return findings


def check_tree(docs_root: Path, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(docs_root.rglob("*.md")):
        findings.extend(check_markdown(path, root))
    return findings


def published_markdown(root: Path) -> list[Path]:
    """Every Markdown document the release publishes, in a stable order.

    ``docs/public`` was the whole scan root, and it is not the whole
    published set.  ``README.md``, ``SPEEDRUN.md``, ``PROVENANCE.md``,
    ``PUBLIC_RELEASE_ACCEPTANCE.md`` and ``CHANGELOG.md`` sit at the
    repository root, ship with the release, and are read far more often
    than the deep pages -- and the sentence retired from
    ``VERIFICATION.md`` reappeared in the README's verification table,
    unlinked, over a receipt whose verdict is FAIL, while the build
    stayed green.  That is the failure this module's own docstring
    predicted: a doc fix alone just moves the unbacked sentence
    somewhere else, and "somewhere else" turned out to be one directory
    up.

    Nothing else in the tree is walked.  ``tools/`` and ``docs/manual``
    are working material rather than published claims, and
    ``tests/fixtures/public_claim_backing`` holds the negative controls,
    which are written to fail this guard on purpose.
    """
    return sorted({*root.glob("*.md"),
                   *(root / "docs" / "public").rglob("*.md")})


def check_published(root: Path) -> list[Finding]:
    """The guard over the published set rather than over one directory."""
    findings: list[Finding] = []
    for path in published_markdown(root):
        findings.extend(check_markdown(path, root))
    return findings


# ---------------------------------------------------------------------------
# the audited sites, retired
# ---------------------------------------------------------------------------

#: The sentence the external review named, kept here so its retirement is
#: provable rather than asserted.  It is the negative-control fixture's
#: text, and a test below requires it to be ABSENT from the live document.
RETIRED_OVERREACH = (
    "WOOF-ArWen's ingest reproduces the WRF initial surface state at the\n"
    "FP32/operator floor on every domain, so everything in the tables below\n"
    "is forecast divergence, not initial-condition error.")


# ---------------------------------------------------------------------------
# the guard bites: fixtures
# ---------------------------------------------------------------------------


def _fixture_findings(name: str) -> list[Finding]:
    return check_markdown(FIXTURES / name, FIXTURES)


def test_the_negative_control_fixture_is_rejected():
    """THE control.  The fixture holds the published sentence verbatim.

    It must be rejected for the reason the review gave -- a measurement of
    the surface used to conclude about the initial state -- not merely for
    lacking a link.
    """
    findings = _fixture_findings("negative_control_surface_inference.md")
    rules = {finding.rule for finding in findings}
    assert "surface-to-full-state" in rules, findings
    assert "unbacked-claim" in rules, findings
    quoted = " ".join(finding.quote for finding in findings)
    assert "not initial-condition error" in quoted


def test_the_negative_control_still_holds_the_reviewed_sentence():
    """A control that drifted off the real sentence controls nothing."""
    fixture = (FIXTURES / "negative_control_surface_inference.md").read_text(
        encoding="utf-8")
    assert RETIRED_OVERREACH in fixture


def test_the_reviewed_sentence_is_gone_from_the_published_document():
    """The retirement itself, pinned.

    The guard rejecting a fixture proves the detector works; only this
    proves the document was actually fixed.
    """
    published = (PUBLIC_DOCS / "VERIFICATION.md").read_text(encoding="utf-8")
    assert RETIRED_OVERREACH not in published
    assert "initialization parity" not in published.lower()


def test_a_backed_claim_of_matching_breadth_passes():
    """The other half: the guard is not a constant 'reject'."""
    assert _fixture_findings("backed_claim.md") == []


def test_a_claim_broader_than_its_receipt_is_rejected():
    findings = _fixture_findings("overbroad_claim.md")
    assert [finding.rule for finding in findings] == [
        "breadth-exceeds-receipt"], findings
    assert "soil" in findings[0].detail


def test_an_unscoped_claim_over_a_partial_receipt_is_rejected():
    findings = _fixture_findings("unscoped_over_partial_receipt.md")
    assert [finding.rule for finding in findings] == [
        "breadth-exceeds-receipt"], findings
    assert "unscoped" in findings[0].detail


def test_a_link_to_a_receipt_that_does_not_resolve_is_rejected():
    findings = _fixture_findings("dangling_receipt_link.md")
    assert [finding.rule for finding in findings] == ["unbacked-claim"]
    assert "do not resolve" in findings[0].detail


def test_a_quoted_claim_is_rejected_and_borrows_no_neighbours_receipt():
    """The blockquote control, and the leak it closes.

    The reviewed sentence is the same one the fixture above holds; here it
    sits inside a blockquote beside a quoted paragraph that cites a real
    receipt.  Reading the quote as one block -- which is what leaving the
    ``>`` in place did -- hands that receipt to the claim and the
    ``unbacked-claim`` rule goes quiet on the exact sentence the external
    review named.  Both rules must fire.
    """
    findings = _fixture_findings("negative_control_quoted_overreach.md")
    rules = {finding.rule for finding in findings}
    assert "surface-to-full-state" in rules, findings
    assert "unbacked-claim" in rules, findings


def test_the_quoted_control_still_holds_the_reviewed_sentence():
    """Quoted, but not paraphrased: the marker is all that was added."""
    fixture = (FIXTURES / "negative_control_quoted_overreach.md").read_text(
        encoding="utf-8")
    unquoted = "\n".join(_BLOCKQUOTE_RE.sub(r"\1", line)
                         for line in fixture.splitlines())
    assert RETIRED_OVERREACH in unquoted
    assert RETIRED_OVERREACH not in fixture


def test_quoted_bullets_are_read_one_at_a_time():
    """Markers do not bleed between quoted bullets, and sight is kept.

    Two of this fixture's bullets state nothing between them: one names
    the zero-hour frame without asserting a result, the next says "floor"
    about frame coverage.  Fused they carried both detector markers and no
    receipt, which is how a shipped runbook was reported as making an
    unbacked t=0 claim it never made.  The third bullet is a genuine
    unbacked claim, quoted identically -- so this asserts the count, not
    merely silence, and a splitter that blinded the guard fails here.
    """
    findings = _fixture_findings("quoted_bullets_do_not_bleed.md")
    assert [finding.rule for finding in findings] == ["unbacked-claim"], (
        findings)
    assert "WOOF-ArWen reproduces the WRF initial state" in findings[0].quote
    assert "REFL_10CM" not in findings[0].quote
    assert "coverage floor" not in findings[0].quote


def test_prose_about_the_initial_state_that_claims_nothing_is_quiet():
    """Discrimination control: the detector is not 'any mention of t=0'."""
    assert _fixture_findings("no_claim.md") == []


def test_the_group_vocabulary_covers_every_registered_group():
    """A group the digest scores but prose cannot name is unaddressable."""
    registered = {group.name for group in CARRIER_GROUPS} | {BOUNDARY_GROUP}
    assert set(GROUP_CLAIM_TERMS) == registered


# ---------------------------------------------------------------------------
# the guard runs on the real tree
# ---------------------------------------------------------------------------


def test_the_public_docs_carry_no_unbacked_t0_claim():
    """THE gate, enforced: any finding at all fails the build.

    Over the whole published set -- ``docs/public`` AND the repository
    root -- because rooted at ``docs/public`` alone this gate reported
    green while ``README.md`` carried the retired sentence.
    """
    findings = check_published(REPO_ROOT)
    assert findings == [], "\n".join(str(finding) for finding in findings)


def test_the_rewritten_sites_are_still_scanned_as_claims():
    """Enforced-and-silent must not mean the scanner stopped looking.

    The rewrite could have passed the gate by removing every recognizable
    claim instead of backing it.  These are the sites the audit named; at
    least one sentence at each must still register as a t=0 claim, which
    is what makes the receipt link and the breadth rule essential.
    """
    for name in ("VERIFICATION.md", "ANNOUNCEMENT-DRAFT.md"):
        path = PUBLIC_DOCS / name
        rel = path.relative_to(REPO_ROOT).as_posix()
        claims = [block for block in split_blocks(
            path.read_text(encoding="utf-8"), rel) if claim_sentences(block)]
        assert claims, name
        for block in claims:
            assert _linked_receipts(block, REPO_ROOT), (name, block.line)


def test_an_injected_claim_fails_the_enforced_gate(tmp_path: Path):
    """The gate is not silent because the detector went away."""
    scratch = tmp_path / "docs" / "public"
    scratch.mkdir(parents=True)
    (scratch / "NEW.md").write_text(
        "WOOF-ArWen reproduces the WRF initial state at the FP32 floor on every\n"
        "domain of every case.\n", encoding="utf-8")
    findings = check_tree(scratch, tmp_path)
    assert [finding.rule for finding in findings] == ["unbacked-claim"], (
        findings)


def test_a_claim_planted_in_a_root_document_fails_the_enforced_gate(
        tmp_path: Path):
    """THE scope control, and the one that was missing.

    The same sentence, in the same tree, in the file every reader opens
    first.  Rooted at ``docs/public`` the guard returns nothing for it --
    asserted here beside the finding, because a scope hole is invisible
    from inside the scope -- which is how the retired claim lived in
    ``README.md`` through a release with the enforced gate green.
    """
    (tmp_path / "docs" / "public").mkdir(parents=True)
    (tmp_path / "README.md").write_text(
        "WOOF-ArWen reproduces the WRF initial state at the FP32 floor on every\n"
        "domain of every case.\n", encoding="utf-8")
    findings = check_published(tmp_path)
    assert [finding.rule for finding in findings] == ["unbacked-claim"], (
        findings)
    assert findings[0].path == "README.md"
    # The old scan root is why this was invisible, not the detector.
    assert check_tree(tmp_path / "docs" / "public", tmp_path) == []


def test_every_root_release_document_is_inside_the_scan():
    """The widening is a set, and the set is named.

    A guard that scans "the root" by accident scans whatever happens to
    be there.  These five are the root documents the release publishes,
    and the README is the one the finding was found in.
    """
    scanned = {path.relative_to(REPO_ROOT).as_posix()
               for path in published_markdown(REPO_ROOT)}
    assert {"README.md", "CHANGELOG.md", "PROVENANCE.md", "SPEEDRUN.md",
            "PUBLIC_RELEASE_ACCEPTANCE.md"} <= scanned
    assert "docs/public/VERIFICATION.md" in scanned
    # Working material and the negative-control fixtures stay out: the
    # fixtures are written to fail this guard, so scanning them would make
    # the enforced gate permanently red for the wrong reason.
    assert not any(name.startswith(("tools/", "tests/", "docs/manual/"))
                   for name in scanned), sorted(scanned)


def test_the_readme_verification_table_is_still_scanned_as_a_claim():
    """Every t=0 claim the README makes links its receipt.

    The README stopped stating a t=0 result on 2026-09-09, when the
    release documentation was refreshed; the claim of record, with its
    FAIL verdict and receipt, lives in ``docs/public/VERIFICATION.md``
    and is held there by the rewritten-sites test above.  The README
    stays inside the enforced scan (the root-document control proves a
    planted claim is caught), so a claim can only come back with a
    receipt attached.
    """
    rel = "README.md"
    blocks = [block for block in split_blocks(
        (REPO_ROOT / rel).read_text(encoding="utf-8"), rel)
        if claim_sentences(block)]
    for block in blocks:
        assert _linked_receipts(block, REPO_ROOT), block.line


# ---------------------------------------------------------------------------
# a cited receipt has to survive the release, and say how it came out
# ---------------------------------------------------------------------------


def _release_exclusion_rules() -> tuple[list[str], Callable[[str, list[str]], str | None]]:
    """The same rules and matcher used by the private snapshot builder.

    The matcher ships publicly because receipt-survival checks still run in
    the sanitized clone, where work/build_release_snapshot.py is absent.
    """
    return read_exclusions(REPO_ROOT), matches


def test_release_rules_work_in_a_public_copy_without_publisher_scaffolding(tmp_path):
    """The sanitized clone must still reject a receipt excluded from release."""
    public = tmp_path / "public"
    (public / "tools").mkdir(parents=True)
    helper = public / "tools" / "release_exclusions.py"
    helper.write_bytes((REPO_ROOT / "tools" / helper.name).read_bytes())
    (public / "RELEASE-EXCLUDE.txt").write_text(
        "# public rules\n\nevidence/**\ndocs/private.md\n", encoding="utf-8")
    assert not (public / "work").exists()
    spec = importlib.util.spec_from_file_location("_public_release_rules", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rules = module.read_exclusions(public)
    assert rules == ["evidence/**", "docs/private.md"]
    assert module.matches("evidence/receipt.json", rules) == "evidence/**"
    assert module.matches("docs/private.md", rules) == "docs/private.md"
    assert module.matches("evidence-public/receipt.json", rules) is None
    assert module.matches("docs/private.md/child", rules) is None


def _linked_receipt_targets() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in sorted(PUBLIC_DOCS.rglob("*.md")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        for block in split_blocks(path.read_text(encoding="utf-8"), rel):
            for target in _LINK_RE.findall(block.text):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                clean = target.split("#", 1)[0]
                if clean.endswith((".json", ".md")) and "certification/" in clean:
                    found.append((rel, clean))
    return found


def test_every_linked_receipt_resolves_and_survives_the_release():
    rules, matches = _release_exclusion_rules()
    targets = _linked_receipt_targets()
    assert targets, "no public document links a certification receipt"
    for source, target in targets:
        resolved = ((REPO_ROOT / source).parent / target).resolve()
        assert resolved.is_file(), (source, target)
        rel = resolved.relative_to(REPO_ROOT).as_posix()
        assert matches(rel, rules) is None, (source, rel)


def test_the_release_exclusion_check_can_fail():
    """Control: the same assertion over a path the manifest does exclude."""
    rules, matches = _release_exclusion_rules()
    assert matches("evidence/anything.json", rules) is not None


def test_a_claim_citing_a_failing_receipt_must_state_the_verdict():
    """The verdict control.

    Backing and breadth are both satisfied in this fixture; only the
    verdict is unsaid, which is exactly how a failing measurement could
    be cited as if it supported the sentence.
    """
    findings = _fixture_findings("verdict_unstated.md")
    assert [finding.rule for finding in findings] == [
        "verdict-not-stated"], findings
    assert "FAIL" in findings[0].detail


def test_the_same_claim_stating_the_verdict_is_accepted():
    assert _fixture_findings("verdict_stated.md") == []


@pytest.mark.parametrize("name", [
    "negative_control_surface_inference.md", "backed_claim.md",
    "overbroad_claim.md", "unscoped_over_partial_receipt.md",
    "dangling_receipt_link.md", "no_claim.md",
    "verdict_unstated.md", "verdict_stated.md",
])
def test_every_fixture_exists(name):
    assert (FIXTURES / name).is_file()
