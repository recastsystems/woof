"""Regenerate the resolved-scale disclaimer change receipt.

Two receipts come out of this tool, and the split is deliberate:

``docs/public/receipts/resolved-scale-disclaimer.json``
    everything a public-repo holder can reproduce -- the baseline and
    post-edit greps, the normalized-item substring measurements, the
    negation-invariant verdict with its three mutant verdicts, the
    added-lines-only case-identity scan, the Markdown structure and width
    measurements, and the cross-reference check.

``docs/internal/receipts/perf-gate-sentence.json``
    the internal half: the performance-gate sentence added to the private
    design spec, its substring measurements, and the two-sided proof that
    the subtree it lives in is release-excluded.  It is written under a
    release-excluded path because that is what it describes.

Every field is measured here and none is authored: verdicts are computed
from the same production paths a verifier would run by hand, and the tool
records what it finds.  Mutants are built in a temporary directory and
deleted; only their verdicts are recorded.

Two kinds of field, and they answer to different things
-------------------------------------------------------
*Live-tree* fields -- the negation verdict and its occurrences, the item
finds, the gate block, the post-edit greps -- describe the tree as it
stands.  They must be re-measured whenever the tree moves, which is what
keeps the receipt from ageing into a snapshot of a claim.

*Diff* fields -- the case-identity scan, the Markdown structure counts,
the CHANGELOG entry, the untouched-announcement proof -- measure one
change.  They are scoped by ``--base``, and, once this branch has been
merged into a tree that other work also landed in, ``git diff <base>``
no longer names that change: it names everything since.  ``--edit-tip``
pins the other end, so the diff fields keep measuring the edit the
receipt is about while the live-tree fields re-anchor to the assembled
tree.  Without it the tool behaves exactly as before -- base against the
working tree.

Usage::

    python tools/report_resolved_scale_disclaimer.py --base <sha>
    python tools/report_resolved_scale_disclaimer.py \\
        --base <sha> --edit-tip <sha> --public-clone <path>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_negation_invariant import (  # noqa: E402
    check as negation_check,
    iter_blocks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_RECEIPT = REPO_ROOT / "docs" / "public" / "receipts" / (
    "resolved-scale-disclaimer.json"
)
INTERNAL_RECEIPT = REPO_ROOT / "docs" / "internal" / "receipts" / (
    "perf-gate-sentence.json"
)

SCALE_PAGE = Path("docs/public/RESOLVED-SCALE.md")

PUBLIC_DOC_PATHSPEC = ("README.md", "docs/public/*.md")

BASELINE_GREPS = {
    "resolved_scale_claims": (
        "tornado-resolv|resolv[a-z]* tornado|suction vort|tornadogenesis|"
        "tornado-scale"
    ),
    "unshipped_performance_claims": "cuda graph|kernel fusion|operator fusion",
    "capacity_relief_claims": "lift[a-z]* the ceiling|single-GPU ceiling",
}

#: Substrings the README item must carry (acceptance criterion AC2).
README_ITEM_SUBSTRINGS = (
    "does not resolve tornado dynamics",
    "near-surface corner flow",
    "suction vortices",
    "tornado-scale wind intensity",
    "tornadic-supercell",
    "mesocyclone-proxy",
    "convection-permitting to sub-kilometer case studies",
    "not tornado-resolving simulations",
    "PBL gray zone",
    "docs/public/FIRST-LIGHT.md",
)
#: Added by the pinned-figure variant (AC2b).
README_PINNED_FIGURE_SUBSTRINGS = ("roughly 25 m", "10 m vertical spacing")
#: Present only in the qualitative variant, which was not authorized.
README_QUALITATIVE_SUBSTRINGS = ("one to two orders of magnitude finer",)

#: Substrings the VERIFICATION item must carry (AC3).
VERIFICATION_ITEM_SUBSTRINGS = (
    "tornadic-supercell environment",
    "mesocyclone-scale morphology",
    "do not resolve",
    "near-surface corner flow",
    "suction vortices",
    "tornado-scale wind intensity",
    "convection-permitting to sub-kilometer",
    "not a tornado-resolving claim",
)
#: Added by the pinned-figure variant (AC3b).
VERIFICATION_PINNED_FIGURE_SUBSTRINGS = (
    "roughly 25 m horizontal and 10 m vertical",
)

#: Substrings the internal performance-gate block must carry (AC8).
GATE_BLOCK_SUBSTRINGS = (
    "certified-profile",
    "trajectory-hash",
    "tests/test_surface_certified_identity.py",
    "certified_surface_identity_v112.json",
    "before merge",
)
#: Vocabulary the block must NOT invent (AC8).
GATE_BLOCK_FORBIDDEN = ("conformance gate",)
#: Every path the block names must resolve on the branch head (AC8).
GATE_BLOCK_CITED_PATHS = (
    "tests/test_surface_certified_identity.py",
    "tools/certified_surface_identity.py",
    "tests/fixtures/certified_surface_identity_v112.json",
    "tests/fixtures/certified_surface_identity_v112.README.md",
    "docs/public/DETERMINISM.md",
)
#: Quotations the block attributes to DETERMINISM.md sections 1 and 3.
DETERMINISM_QUOTATIONS = (
    "WOOF-ArWen version and git commit",
    "comparing a run against itself or against a cached artifact",
)

GATE_BLOCK_MARKER = "GATE (Phase-6 performance pass):"
INTERNAL_SPEC = "docs/internal/specs/2026-07-13-gpuwm-design.md"
ANNOUNCEMENT_DRAFT = "docs/public/ANNOUNCEMENT-DRAFT.md"

#: The two-arm control that turns the gate sentence into a gate.  It is not
#: run here: arm (b) perturbs a kernel a certified profile executes, which is
#: a build change outside a documentation slice, and the control is due
#: before the Phase-6 pass merges rather than before the sentence lands.  The
#: rule is declared now, so it cannot be chosen after the arms are seen.
INSTRUMENT_QUALIFICATION = {
    "status": "deferred",
    "due": "before any Phase-6 performance change merges",
    "reason": (
        "arm (b) requires perturbing a kernel a certified profile executes; "
        "that build change is outside this documentation slice"
    ),
    "suite": "tests/test_surface_certified_identity.py",
    "arms": {
        "a": "unmodified branch head, on a CUDA GPU, through the production path",
        "b": (
            "same head with one FMA or reduction reordered, or one constant "
            "moved by one ULP, in a kernel a certified profile executes"
        ),
    },
    "records_per_arm": ["verdict", "run_profiles() digest", "skip_or_run"],
    "decision_rule": (
        "verdicts differ -> the suite is adopted as the perf-pass gate and "
        "this receipt is cited from the gate sentence; verdicts agree -> the "
        "suite's coverage does not reach the perturbed kernel, the gate is "
        "not yet real, and the fixture is extended before any perf change "
        "merges"
    ),
    "skip_policy": (
        "a GPU-less SKIP is recorded explicitly and counts as a hard failure "
        "on the gate runner, because a silent skip is a false-negative gate"
    ),
    "ratified_as": "D-37",
}

#: (id, exact source line, exact replacement) for the AC4 mutants.
MUTANTS = (
    (
        "M1",
        "  not tornado-resolving simulations.",
        "  tornado-resolving simulations.",
    ),
    (
        "M2",
        "  and mesocyclone-proxy diagnostic, not as a resolved-tornado claim:",
        "  and mesocyclone-proxy diagnostic, as a resolved-tornado claim:",
    ),
    (
        "M3",
        "  It does not resolve tornado dynamics:",
        "  It resolves tornado dynamics:",
    ),
)


def run(argv: Sequence[str], cwd: Path = REPO_ROOT) -> dict:
    """Run a command and record its stdout and exit code verbatim."""
    done = subprocess.run(
        list(argv), cwd=str(cwd), capture_output=True, text=True
    )
    return {
        "command": " ".join(argv),
        "exit_code": done.returncode,
        "stdout": done.stdout,
        "stderr": done.stderr,
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_item(path: Path, prefix: str) -> dict:
    """Locate the one list item whose first line begins with ``prefix``."""
    items = [
        block
        for block in iter_blocks(path.read_text(encoding="utf-8"))
        if block.lines[0].startswith(prefix)
    ]
    return {
        "prefix": prefix,
        "item_count": len(items),
        "first_line": items[0].first_line if items else None,
        "line_count": len(items[0].lines) if items else 0,
        "normalized": items[0].normalized() if items else "",
    }


def substring_report(normalized: str, required: Sequence[str]) -> dict:
    present = {token: token in normalized for token in required}
    return {
        "required": list(required),
        "present": present,
        "all_present": all(present.values()),
    }


def blob_section_max_width(
    base: str, path: str, start_marker: str, stop: re.Pattern
) -> dict:
    """Widest line of one section, read from a committed blob."""
    text = run(["git", "show", f"{base}:{path}"])["stdout"]
    return _section_max_width(text.splitlines(), path, start_marker, stop)


def _section_max_width(
    lines: Sequence[str], path: str, start_marker: str, stop: re.Pattern
) -> dict:
    start = next(i for i, line in enumerate(lines) if line == start_marker)
    end = next(
        i for i in range(start + 1, len(lines)) if stop.match(lines[i])
    )
    return {
        "path": path,
        "marker": start_marker,
        "first_line": start + 1,
        "last_line": end,
        "max_width": max(len(line) for line in lines[start:end]),
    }


def diff_endpoints(base: str, tip: str | None) -> list[str]:
    """The revision arguments naming the change under measurement.

    ``tip`` is None for a branch still being worked on: the change is
    everything from ``base`` to the working tree.  Once the branch has been
    merged, ``base`` alone reaches past it, so the tip is named explicitly
    and the diff fields go on measuring the same edit.
    """
    return [base] if tip is None else [base, tip]


def diff_structure(base: str, path: str, tip: str | None = None) -> dict:
    """AC10: one bullet line, two-space continuations, measured width."""
    diff = run(["git", "diff", *diff_endpoints(base, tip), "--", path])[
        "stdout"
    ]
    added = [
        line
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    bullet = [line for line in added if re.match(r"^\+- \*\*", line)]
    others = [line for line in added if line not in bullet]
    nonconforming = [line for line in others if not re.match(r"^\+  \S", line)]
    return {
        "path": path,
        "revisions": diff_endpoints(base, tip),
        "added_line_count": len(added),
        "bullet_line_count": len(bullet),
        "nonconforming_continuation_count": len(nonconforming),
        "nonconforming_continuations": nonconforming,
        "max_added_width": max((len(line) - 1 for line in added), default=0),
    }


def case_identity_scan(
    base: str, paths: Sequence[str], tip: str | None = None
) -> dict:
    """AC5: case tokens in ADDED lines only, never file-scoped."""
    diff = run(["git", "diff", *diff_endpoints(base, tip), "--", *paths])[
        "stdout"
    ]
    added = [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    pattern = re.compile(r"real74|1974|ohio|hrrr", re.IGNORECASE)
    hits = [line for line in added if pattern.search(line)]
    return {
        "scope": list(paths),
        "revisions": diff_endpoints(base, tip),
        "mode": "added-lines-only",
        "pattern": pattern.pattern,
        "added_line_count": len(added),
        "hits": hits,
        "hit_count": len(hits),
    }


def mutant_verdicts() -> dict:
    """AC4 adoption rule: each mutant's verdict must differ from the tree's."""
    unmutated = negation_check(REPO_ROOT)["verdict"]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, before, after in MUTANTS:
            root = Path(tmp) / name
            (root / "docs" / "public").mkdir(parents=True)
            shutil.copy2(REPO_ROOT / "README.md", root / "README.md")
            for page in (REPO_ROOT / "docs" / "public").glob("*.md"):
                shutil.copy2(page, root / "docs" / "public" / page.name)
            readme = root / SCALE_PAGE
            text = readme.read_text(encoding="utf-8")
            occurrences = text.count(before)
            readme.write_text(text.replace(before, after), encoding="utf-8")
            report = negation_check(root)
            results[name] = {
                "source_line": before.strip(),
                "mutated_line": after.strip(),
                "anchor_occurrences_in_tree": occurrences,
                "verdict": report["verdict"],
                "failure_count": report["failure_count"],
                "differs_from_unmutated": report["verdict"] != unmutated,
            }
            shutil.rmtree(root)
    return {
        "unmutated_verdict": unmutated,
        "mutants": results,
        "mutant_files_committed": False,
        "adoption_rule": (
            "adopted only if every mutant verdict differs from the "
            "unmutated verdict"
        ),
        "adopted": all(
            entry["differs_from_unmutated"] for entry in results.values()
        ),
    }


def gate_block(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    blocks = [
        block
        for block in iter_blocks(text)
        if block.lines[0].startswith(GATE_BLOCK_MARKER)
    ]
    normalized = blocks[0].normalized() if blocks else ""
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "block_count": len(blocks),
        "first_line": blocks[0].first_line if blocks else None,
        "normalized": normalized,
        "substrings": substring_report(normalized, GATE_BLOCK_SUBSTRINGS),
        "disowns_dual_run_screen": (
            "dual-run byte-identity screen is explicitly NOT the instrument"
            in normalized
        ),
        "forbidden_vocabulary": {
            token: token in normalized for token in GATE_BLOCK_FORBIDDEN
        },
    }


def edit_range(base: str, tip: str | None) -> dict:
    """What the diff-measured fields are scoped to, resolved to SHAs."""
    return {
        "base": run(["git", "rev-parse", base])["stdout"].strip(),
        "tip": (
            run(["git", "rev-parse", tip])["stdout"].strip()
            if tip is not None
            else None
        ),
        "tip_ref": tip,
        "rule": (
            "the diff-measured fields -- case identity, Markdown structure, "
            "the CHANGELOG entry, the untouched-announcement proof -- are "
            "scoped to this range, so they keep measuring this change after "
            "the branch is merged into a tree other work also landed in.  "
            "Every other field is measured against the tree as it stands."
        ),
    }


def build_public_receipt(base: str, tip: str | None = None) -> dict:
    baseline = {
        name: run(
            ["git", "grep", "-inE", pattern, base, "--", *PUBLIC_DOC_PATHSPEC]
        )
        for name, pattern in BASELINE_GREPS.items()
    }
    post = {
        name: run(["git", "grep", "-inE", pattern, "--", *PUBLIC_DOC_PATHSPEC])
        for name, pattern in BASELINE_GREPS.items()
    }
    readme_item = find_item(REPO_ROOT / SCALE_PAGE, "- **Resolved scale.**")
    verification = REPO_ROOT / "docs" / "public" / "VERIFICATION.md"
    verification_item = find_item(
        verification, "- **No resolved tornado dynamics.**"
    )
    verification_lines = verification.read_text(
        encoding="utf-8"
    ).splitlines()
    # Anchored on the bullet's opening tokens only: quoting the rest of that
    # sentence would enrol this tool in the flush-to-zero claim census.
    fp32_line = 1 + next(
        i
        for i, line in enumerate(verification_lines)
        if line.startswith("- **FP32 ")
    )
    # The bullet the item follows.  7268b9482 replaced the page's "No data
    # assimilation" bullet, the anchor until then, with the
    # observation-validation one now in that place.
    preceding_line = 1 + next(
        i
        for i, line in enumerate(verification_lines)
        if line.startswith("- **No observation-validation claim")
    )
    changelog_diff = run(
        ["git", "diff", *diff_endpoints(base, tip), "--", "CHANGELOG.md"]
    )["stdout"]
    changelog_added = [
        line
        for line in changelog_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return {
        "receipt": "resolved-scale-disclaimer",
        "base_commit": run(["git", "rev-parse", base])["stdout"].strip(),
        "edit_range": edit_range(base, tip),
        "scope_rule": (
            "all three baseline greps empty scopes the change to inserting "
            "the disclaimer; any hit expands it to removing an affirmative "
            "claim first"
        ),
        "baseline_greps": baseline,
        "baseline_all_empty": all(
            entry["exit_code"] == 1 and not entry["stdout"]
            for entry in baseline.values()
        ),
        "post_edit_greps": post,
        "figure_variant_authorized": "pinned",
        "readme_item": {
            **readme_item,
            "substrings": substring_report(
                readme_item["normalized"], README_ITEM_SUBSTRINGS
            ),
            "pinned_figure_substrings": substring_report(
                readme_item["normalized"], README_PINNED_FIGURE_SUBSTRINGS
            ),
            "qualitative_substrings_absent": {
                token: token not in readme_item["normalized"]
                for token in README_QUALITATIVE_SUBSTRINGS
            },
        },
        "verification_item": {
            **verification_item,
            "substrings": substring_report(
                verification_item["normalized"], VERIFICATION_ITEM_SUBSTRINGS
            ),
            "pinned_figure_substrings": substring_report(
                verification_item["normalized"],
                VERIFICATION_PINNED_FIGURE_SUBSTRINGS,
            ),
            "precedes_fp32_bullet_line": fp32_line,
            "follows_no_observation_validation_bullet_line": preceding_line,
            "ordering_holds": (
                preceding_line
                < (verification_item["first_line"] or 0)
                < fp32_line
            ),
        },
        "negation_invariant": {
            "post_edit": negation_check(REPO_ROOT),
            "mutation_control": mutant_verdicts(),
        },
        "case_identity_scan": case_identity_scan(
            base,
            ["README.md", "docs/public/VERIFICATION.md", "CHANGELOG.md"],
            tip,
        ),
        "unshipped_claim_scan": run(
            [
                "git",
                "grep",
                "-inE",
                "cuda graph|kernel fusion|operator fusion|lift[a-z]* the "
                "ceiling|single-GPU ceiling",
                "--",
                *PUBLIC_DOC_PATHSPEC,
            ]
        ),
        "markdown_structure": [
            diff_structure(base, "README.md", tip),
            diff_structure(base, "docs/public/VERIFICATION.md", tip),
        ],
        "section_widths_before_edit": {
            "note": (
                "measured on the base blob, so the width rule is scored "
                "against the section as it stood before the insertion"
            ),
            "README_limits": blob_section_max_width(
                base, "README.md", "## Limits", re.compile(r"^## ")
            ),
            "VERIFICATION_not_claimed": blob_section_max_width(
                base,
                "docs/public/VERIFICATION.md",
                "**Not claimed:**",
                re.compile(r"^## "),
            ),
        },
        "changelog": {
            "added_version_headings": [
                line for line in changelog_added if line.startswith("+## ")
            ],
            "added_entry_lines": [
                line for line in changelog_added if line.startswith("+- ")
            ],
            "added_line_count": len(changelog_added),
            "newest_heading_before_edit": run(
                ["git", "grep", "-m1", "-n", "^## ", base, "--", "CHANGELOG.md"]
            )["stdout"].strip(),
            "heading_authorized": "new heading, ratified under D-11",
        },
        "cross_reference": {
            "first_light_exists": (
                REPO_ROOT / "docs" / "public" / "FIRST-LIGHT.md"
            ).is_file(),
            "grep": run(
                [
                    "git",
                    "grep",
                    "-niE",
                    "gray.zone|SASE",
                    "--",
                    "docs/public/FIRST-LIGHT.md",
                ]
            ),
        },
        "announcement_draft_untouched": run(
            [
                "git",
                "diff",
                "--stat",
                *diff_endpoints(base, tip),
                "--",
                ANNOUNCEMENT_DRAFT,
            ]
        ),
        # The claim above is about THIS change.  Saying only that would let
        # the receipt be read as a statement about the whole tree, so what
        # the tree actually did to that file is recorded beside it -- and
        # attributed, so it is clear whose edit it was.
        "announcement_draft_in_the_tree": {
            "note": (
                "the field above scopes the untouched claim to this change; "
                "this one records the file's state in the tree the receipt "
                "was regenerated on, whoever moved it"
            ),
            "diffstat": run(
                ["git", "diff", "--stat", base, "HEAD", "--",
                 ANNOUNCEMENT_DRAFT]
            ),
            "commits_since_base": run(
                ["git", "log", "--oneline", f"{base}..HEAD", "--",
                 ANNOUNCEMENT_DRAFT]
            ),
        },
    }


def _last_commit_files(path: str) -> dict:
    """The commit that last touched ``path``, and every file it touched."""
    sha = run(["git", "log", "-1", "--format=%H", "--", path])["stdout"].strip()
    if not sha:
        return {"sha": None, "files": [], "touches_only_the_target": False}
    files = run(
        ["git", "show", "--format=", "--name-only", sha]
    )["stdout"].split()
    return {
        "sha": sha,
        "files": files,
        "touches_only_the_target": files == [path],
    }


def public_clone_proof(clone: Path | None) -> dict:
    """The half of the internal-only proof a public-repo holder can run.

    ``git ls-files docs/internal`` executed in the published clone.  Any
    tracked path there would mean the performance-gate sentence ships, and
    the edit would have to be relocated or dropped.
    """
    if clone is None:
        return {
            "status": "not-run",
            "reason": "no public clone supplied on this invocation",
        }
    listing = run(["git", "ls-files", "docs/internal"], cwd=clone)
    head = run(["git", "log", "-1", "--format=%H %s"], cwd=clone)
    tracked = listing["stdout"].split()
    return {
        "status": "run",
        "clone_head": head["stdout"].strip(),
        "remote": run(["git", "remote", "get-url", "origin"], cwd=clone)[
            "stdout"
        ].strip(),
        "listing": listing,
        "tracked_count": len(tracked),
        "tracked_paths": tracked,
        "subtree_absent_from_public_clone": not tracked,
    }


def build_internal_receipt(
    base: str, clone: Path | None = None, tip: str | None = None
) -> dict:
    spec = REPO_ROOT / INTERNAL_SPEC
    changed = run(
        ["git", "diff", "--name-only", *diff_endpoints(base, tip)]
    )["stdout"].split()
    public_touched = [
        name
        for name in changed
        if name.startswith("docs/public/") and name.endswith(".md")
    ]
    cited = {}
    for rel in GATE_BLOCK_CITED_PATHS:
        target = REPO_ROOT / rel
        cited[rel] = {
            "resolves": target.is_file(),
            "bytes": target.stat().st_size if target.is_file() else None,
        }
    determinism = (
        REPO_ROOT / "docs" / "public" / "DETERMINISM.md"
    ).read_text(encoding="utf-8")
    return {
        "receipt": "perf-gate-sentence",
        "base_commit": run(["git", "rev-parse", base])["stdout"].strip(),
        "edit_range": edit_range(base, tip),
        "scope": "internal-only; the target subtree is release-excluded",
        "gate_block": gate_block(spec),
        "edit_c_commit": _last_commit_files(INTERNAL_SPEC),
        "public_markdown_touched_by_the_branch": public_touched,
        "release_exclusion": run(
            ["git", "grep", "-n", "docs/internal", "--", "RELEASE-EXCLUDE.txt"]
        ),
        "development_tree_tracked_files": run(
            ["git", "ls-files", "docs/internal"]
        ),
        "public_clone_proof": public_clone_proof(clone),
        "private_path_rule": (
            "no claim here rests on a release-excluded snapshot builder, "
            "which a public-repo holder can neither read nor run; the public "
            "half of the proof is `git ls-files docs/internal` executed in "
            "the published clone, recorded above"
        ),
        "instrument_qualification": INSTRUMENT_QUALIFICATION,
        "cited_paths_resolve": cited,
        "determinism_quotations_present": {
            quote: quote in determinism for quote in DETERMINISM_QUOTATIONS
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        default="HEAD",
        help="commit the change is measured against",
    )
    parser.add_argument(
        "--edit-tip",
        default=None,
        help=(
            "other end of the change the diff-measured fields describe; "
            "omit while the branch is still being worked on, name it once "
            "the branch has been merged so those fields keep measuring this "
            "edit rather than everything since --base"
        ),
    )
    parser.add_argument(
        "--public-clone",
        type=Path,
        default=None,
        help=(
            "checkout of the published repository; the internal-only proof "
            "runs `git ls-files docs/internal` there"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    public = build_public_receipt(args.base, args.edit_tip)
    internal = build_internal_receipt(
        args.base,
        args.public_clone.resolve() if args.public_clone else None,
        args.edit_tip,
    )
    for path, payload in ((PUBLIC_RECEIPT, public), (INTERNAL_RECEIPT, internal)):
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": .gitattributes sets `* -text`, so whatever this
        # writes is what the repository stores.
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
