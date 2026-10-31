"""Register every FTZ / subnormal claim in the public tree.

Scope is machine-defined: ``git ls-files`` minus the RELEASE-EXCLUDE globs
minus the vendored trees, tests INCLUDED.  A hand-kept list would drift, and
a sentence that drifts out of the census is a public claim nobody is
checking.

``--check`` refuses, each for the breakage named:

* a claim line nobody registered: an FTZ sentence ships that no curator has
  held against the receipt;
* a registered line that is gone: the register ships a claim the tree no
  longer makes;
* a registered line whose words changed: an ``asserted_token`` a curator
  wrote for one sentence goes on vouching for a different one;
* an ``anchor_sha256`` that is not the hash of its ``anchor``: a record
  edited by hand says one thing and hashes another;
* a ``site_count`` or ``site_count_by_kind`` that disagrees with the
  records: two branches that each register one claim both write the same
  new total, git merges that without a conflict, and the register ships a
  count one short;
* a token the receipt contradicts: the register says a route flushes where
  the measurement says it does not, or the reverse.

A site is anchored by its file and the exact text of its line (``anchor``,
hashed in ``anchor_sha256``).  Identical lines in one file are counted, so a
second copy of a registered sentence is still an unregistered claim.
``line`` is where this tool last found the site: a hint for readers that
``--check`` does not fail on.  While ``line`` was part of the key, a row
added to CHANGELOG.md above its claims moved about 20 records down one line
without changing a word of any of them.  Every branch that added a row had
to regenerate the census, every merge of two such branches conflicted in it,
and two branches that each added one row could merge cleanly and leave all
of those records a line short, red on the merged line.  A line that only
moved changes no claim, so it no longer touches the census.  A context
window was left out of the key on purpose: it would tie each claim to its
neighbours, and a row landing next to a claim would fail the gate again
with no claim changed.

``asserted_token`` binds a record to the receipt's verdict for its (route,
mechanism) cell.  Attribution across the five routes is judgment, not
mechanics: this tool never guesses one.  A record it cannot attribute is
written with ``asserted_token: null`` and ``attribution: "unattributed"``,
and the gate requires only that a token, once present, agrees with the
receipt.

Usage::

    python -m tools.ftz_receipt.claim_census
    python -m tools.ftz_receipt.claim_census --check
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

if not __package__:
    # Run by file name: make `tools` resolve to this tree, not to another
    # installed checkout (D1; see route_inventory.require_same_tree).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ftz_receipt import route_inventory as ri  # noqa: E402

SCHEMA_ID = "gpuwm.ftz-claim-census/v2"

CLAIM_PATTERN = re.compile(r"ftz|subnormal", re.IGNORECASE)

#: Vendored third-party trees.  Their arithmetic is not ours to describe.
#:
#: ``tools/region_global_dealias/`` is the whole crate rather than a
#: ``vendor/`` subdirectory of one, because that is how it is vendored: a
#: verbatim copy at a recorded upstream commit that this tree never edits.
#: A census record pins a sentence by its exact text, so registering one
#: inside a tree we do not author would fail the gate on the next
#: re-vendor -- and the fix would be to edit somebody else's source.
#:
#: ``tools/rw_wps/vendor/`` arrived with the 2.5.0 mapped-engine port, after
#: this tuple was written, and carries 46 matching lines: the crates-io
#: mirror (num-traits describes IEEE subnormals in its own doc comments) plus
#: the read-only donor snapshot of grib-core and netcrust.  Only the
#: ``vendor/`` subtree is listed, never ``tools/rw_wps/`` itself, because
#: ``crates/mapped-engine`` beside it IS gpuwm-authored and its claims stay in
#: scope.  Measured at this tip: all 46 sit under ``vendor/``, none in
#: authored rw_wps code.
#:
#: ``tools/uwpbl_wrf471_oracle/libm64/upstream/`` holds the verbatim MIT
#: sources (Arm optimized-routines exp/log/pow, CORE-MATH cos/acos) that
#: woof/core/kernels/glibc_flt64.cuh transcribes, hash-pinned in
#: ``libm64/upstream-sha256.json``; their subnormal comments describe the
#: upstream C, and editing them would break the pin.
VENDOR_PREFIXES = ("tools/grib1_bridge/vendor/", "tools/rustwx/vendor/",
                   "tools/region_global_dealias/", "tools/rw_wps/vendor/",
                   "tools/uwpbl_wrf471_oracle/libm64/upstream/")

#: The receipt itself.  Registering a measurement against itself is circular:
#: the bit table and the receipt are the authority these records point AT, so
#: they are not among the claims the census checks.  Everything else stays in
#: scope, the probe and this tool's own tests included.
SELF_PREFIXES = ("tools/ftz_receipt/receipt/", "tools/ftz_receipt/fixtures/")
#: Compiled-image records, not prose.  The shipped cold-start kernel packs
#: (fix/singlecore-stalls, rebuilt for 2.9.11) are one JSON line each: the
#: NVRTC options every image was compiled with ("-ftz=true" is a fact the
#: loader re-checks, not a sentence about arithmetic) and the xz/base64 cubin
#: payload, whose letters spell "FTZ" by chance.  Registering a 5.8 MB line as
#: an anchor would pin the image bytes, which the pack's own identity already
#: does.
COMPILED_IMAGE_FILES = ("woof/data/kernel-pack-cuda12.json",
                        "woof/data/kernel-pack-cuda13.json")

#: Files whose FTZ mention motivates shipped behaviour rather than describing
#: it.  They are REGISTERED here and left untouched: revisiting any of them
#: changes numbers and breaks bitwise gates, which is [D-36], not this work.
BEHAVIOURAL_FILES = (
    "woof/core/kernels/rrtmg_sw.cu",
    "woof/core/rrtmg_legacy_prep.py",
    "tools/grib1_bridge/src/lib.rs",
    "woof/ingest/hrrr.py",
)

ATTRIBUTION_PENDING = "unattributed"
ATTRIBUTION_SET = "attributed"

CENSUS_REL = "woof/verify/ftz_claim_sites.json"

ANCHORING_NOTE = (
    "a site is its file plus the exact text of its line (anchor, "
    "anchor_sha256), and identical lines in one file are counted; line is "
    "where the generator last found the site, a hint the check does not "
    "fail on, so lines added or removed above a claim leave the census "
    "valid")

TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".txt", ".rst", ".toml", ".cfg", ".ini", ".json",
    ".jsonl", ".csv", ".cu", ".cuh", ".c", ".h", ".cpp", ".rs", ".F", ".F90",
    ".f90", ".sh", ".bat", ".ps1", ".yml", ".yaml", ".cff",
})


def release_exclusions(root: Path) -> list[str]:
    path = root / "RELEASE-EXCLUDE.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def is_excluded(relpath: str, globs: list[str]) -> bool:
    if relpath.startswith(VENDOR_PREFIXES) or relpath.startswith(
            SELF_PREFIXES) or relpath in COMPILED_IMAGE_FILES:
        return True
    for pattern in globs:
        if fnmatch.fnmatch(relpath, pattern):
            return True
        if pattern.endswith("/**") and relpath.startswith(pattern[:-2]):
            return True
    return False


def public_files(root: Path) -> list[str]:
    # `set`, and it is essential rather than tidiness.  During an
    # unresolved merge `git ls-files` prints a conflicted path ONCE PER
    # STAGE -- three rows for one file -- and without the dedupe every
    # claim in that file is registered three times, with identical line
    # numbers.  A release train regenerates the census while merges are
    # open, which is exactly when this fires: it inflated CHANGELOG.md's
    # six claims to eighteen and the census total by twelve, and
    # `--check` stayed green throughout because every duplicate agreed
    # with the file it was read from.  Measured on this repo: 3 rows
    # conflicted, 1 clean.
    out = subprocess.run(["git", "ls-files"], cwd=str(root),
                         capture_output=True, text=True, check=True)
    globs = release_exclusions(root)
    # Recast WOOF: the hex model folded in (woof/hex, components/hex) carries
    # its own FTZ authority and claims; this census is the regional engine's.
    return sorted({path for path in out.stdout.splitlines()
                   if path.strip() and not is_excluded(path, globs)
                   and not path.startswith(("woof/hex/", "components/hex/"))})


def _long_path(path: Path) -> Path:
    """``path`` in a form Windows opens past MAX_PATH (260 characters).

    The census was regenerated on the desktop with the tree under a deep
    scratch directory, where tests/data/wrf471_diffusion/producer-snapshots/
    <sha256>-rounding_attribution.py.txt is longer than MAX_PATH.  Reading
    it raised OSError, the scan skipped it as if it were binary, and its
    three claim lines dropped out of the census (2.9.4 fold records
    a2814d3403, a7e2b8dca8, 14f69ad418), so the Linux check reported them
    unregistered.  Only a file git tracks reaches here, so a read failure
    other than absence or non-UTF-8 content now raises instead of
    silently shrinking the census.
    """
    if sys.platform != "win32":
        return path
    text = str(path.resolve())
    if text.startswith("\\\\?\\"):
        return path
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def anchor_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_sites(root: Path, files: list[str],
                  census_path: str | None = None) -> list[dict]:
    """One record per matching line, in path then line order."""
    records: list[dict] = []
    for relpath in files:
        if relpath == census_path:
            continue
        path = root / relpath
        if path.suffix and path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = _long_path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            # Tracked but absent from this working tree (a deletion not
            # yet committed).  _long_path makes this a real absence on
            # Windows too, not a path longer than MAX_PATH.
            continue
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if not CLAIM_PATTERN.search(line):
                continue
            anchor = line.strip()
            records.append({
                "file": relpath,
                "line": number,
                "kind": ("behavioural" if relpath in BEHAVIOURAL_FILES
                         else "prose"),
                "anchor": anchor,
                "anchor_sha256": anchor_sha256(anchor),
                "route": None,
                "mechanism": None,
                "asserted_token": None,
                "attribution": ATTRIBUTION_PENDING,
            })
    return records


def merge_attribution(records: list[dict],
                      previous: dict | None) -> list[dict]:
    """Carry a curator's attribution forward; never invent one."""
    if not previous:
        return records
    index = {(item["file"], item["anchor_sha256"]): item
             for item in previous.get("sites", [])}
    for record in records:
        prior = index.get((record["file"], record["anchor_sha256"]))
        if prior is None:
            continue
        for key in ("route", "mechanism", "asserted_token", "attribution"):
            if prior.get(key) is not None:
                record[key] = prior[key]
    return records


def build_census(root: Path, previous: dict | None = None) -> dict:
    records = merge_attribution(
        collect_sites(root, public_files(root), CENSUS_REL), previous)
    by_kind: dict[str, int] = {}
    for record in records:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
    return {
        "schema": SCHEMA_ID,
        "generator": "tools/ftz_receipt/claim_census.py",
        "scope": "git ls-files minus RELEASE-EXCLUDE.txt globs minus "
                 "vendored trees; tests included",
        "pattern": CLAIM_PATTERN.pattern,
        "behavioural_files": list(BEHAVIOURAL_FILES),
        "attribution_note":
            "route/mechanism attribution is judgment across five measured "
            "routes and is not machine-derivable; unattributed records carry "
            "asserted_token: null and the gate requires only that a token, "
            "once written, agrees with the receipt",
        "anchoring_note": ANCHORING_NOTE,
        "site_count": len(records),
        "site_count_by_kind": dict(sorted(by_kind.items())),
        "sites": records,
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def default_output(root: Path) -> Path:
    return root / "woof" / "verify" / "ftz_claim_sites.json"


def _line_hint(record: dict) -> int:
    hint = record.get("line")
    return hint if isinstance(hint, int) else 0


def _pair_nearest(registered: list[dict], found: list[dict]
                  ) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Pair registered with found records, closest line hint first.

    Returns the pairs, the registered records left over and the found
    records left over.  The pairing only chooses which line a problem is
    reported against; whether the census fails never depends on it.
    """
    if not registered or not found:
        return [], list(registered), list(found)
    candidates = sorted(
        (abs(_line_hint(reg) - _line_hint(hit)), i, j)
        for i, reg in enumerate(registered)
        for j, hit in enumerate(found))
    used_registered: set[int] = set()
    used_found: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for _, i, j in candidates:
        if i in used_registered or j in used_found:
            continue
        used_registered.add(i)
        used_found.add(j)
        pairs.append((registered[i], found[j]))
    return (pairs,
            [reg for i, reg in enumerate(registered)
             if i not in used_registered],
            [hit for j, hit in enumerate(found) if j not in used_found])


def check_census(root: Path, census: dict,
                 receipt: dict | None = None,
                 notes: list[str] | None = None) -> list[str]:
    """Return every reason the census does not describe the tree.

    Sites are matched on (file, exact line text) as a multiset: every copy
    of a line in the tree needs its own record and every record needs its
    own copy in the tree.  A match whose line number moved is not a
    problem; when ``notes`` is given, the count of such moves is appended
    there for the caller to show.  What is left unmatched in one file is
    paired by nearest line and reported as changed text; the rest are
    unregistered or no longer present.
    """
    problems: list[str] = []
    found = collect_sites(root, public_files(root), CENSUS_REL)
    registered = list(census.get("sites", []))

    for record in registered:
        if anchor_sha256(record["anchor"]) != record["anchor_sha256"]:
            problems.append(
                f"anchor hash stale at {record['file']}:{_line_hint(record)}")

    found_by_site: dict[tuple[str, str], list[dict]] = {}
    for record in found:
        found_by_site.setdefault((record["file"], record["anchor"]),
                                 []).append(record)
    registered_by_site: dict[tuple[str, str], list[dict]] = {}
    for record in registered:
        registered_by_site.setdefault((record["file"], record["anchor"]),
                                      []).append(record)

    moved = 0
    loose_registered: dict[str, list[dict]] = {}
    loose_found: dict[str, list[dict]] = {}
    for site in found_by_site.keys() | registered_by_site.keys():
        pairs, spare_registered, spare_found = _pair_nearest(
            registered_by_site.get(site, []), found_by_site.get(site, []))
        moved += sum(1 for reg, hit in pairs
                     if _line_hint(reg) != hit["line"])
        loose_registered.setdefault(site[0], []).extend(spare_registered)
        loose_found.setdefault(site[0], []).extend(spare_found)

    for relpath in sorted(loose_registered.keys() | loose_found.keys()):
        pairs, gone, new = _pair_nearest(loose_registered.get(relpath, []),
                                         loose_found.get(relpath, []))
        for reg, hit in sorted(pairs, key=lambda pair: pair[1]["line"]):
            problems.append(
                f"anchor text changed at {relpath}:{hit['line']} "
                f"(registered at line {_line_hint(reg)}): "
                f"{reg['anchor'][:80]!r} is now {hit['anchor'][:80]!r}")
        for hit in sorted(new, key=lambda record: record["line"]):
            problems.append(
                f"unregistered claim {relpath}:{hit['line']}: "
                f"{hit['anchor'][:80]}")
        for reg in sorted(gone, key=_line_hint):
            problems.append(
                f"registered claim no longer present "
                f"{relpath}:{_line_hint(reg)}: {reg['anchor'][:80]}")

    by_kind: dict[str, int] = {}
    for record in registered:
        by_kind[record.get("kind")] = by_kind.get(record.get("kind"), 0) + 1
    if "site_count" in census and census["site_count"] != len(registered):
        problems.append(
            f"site_count says {census['site_count']} but the census holds "
            f"{len(registered)} records")
    if ("site_count_by_kind" in census
            and census["site_count_by_kind"] != by_kind):
        problems.append(
            f"site_count_by_kind says {census['site_count_by_kind']} but "
            f"the records count {by_kind}")

    if receipt is not None:
        cells = {(cell["route"], cell["mechanism"]): cell["verdict"]
                 for cell in receipt["cells"]}
        for record in registered:
            where = f"{record['file']}:{_line_hint(record)}"
            token = record.get("asserted_token")
            if token is None:
                if record.get("attribution") != ATTRIBUTION_PENDING:
                    problems.append(
                        f"{where} claims attribution with no token")
                continue
            cell = cells.get((record.get("route"), record.get("mechanism")))
            if cell is None:
                problems.append(
                    f"{where} asserts a token for a cell the "
                    f"receipt does not carry: "
                    f"({record.get('route')}, {record.get('mechanism')})")
            elif cell != token:
                problems.append(
                    f"{where} asserts {token!r} but the receipt's "
                    f"cell says {cell!r}")

    if notes is not None and moved:
        notes.append(
            f"{moved} of {len(registered)} sites sit on a different line "
            f"than their line hint; the check does not fail on that, and "
            f"regenerating refreshes the hints")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    ri.require_same_tree(__file__)
    root = Path(args.root).resolve() if args.root else ri.repo_root()
    out = default_output(root)
    previous = (json.loads(out.read_text(encoding="utf-8"))
                if out.exists() else None)
    if args.check:
        if previous is None:
            print(f"census missing: {out}", file=sys.stderr)
            return 1
        notes: list[str] = []
        problems = check_census(root, previous, notes=notes)
        for problem in problems:
            print(problem, file=sys.stderr)
        for note in notes:
            print(note)
        return 1 if problems else 0
    document = build_census(root, previous)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(document), encoding="utf-8", newline="\n")
    print(f"wrote {out} ({document['site_count']} claim lines in "
          f"{len({r['file'] for r in document['sites']})} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
