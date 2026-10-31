"""Read the exact path exclusions shared by release assembly and public checks.

Only literal paths and a trailing ``/**`` subtree rule are supported. Keep the
first matching rule so a refusal names the same exclusion the builder applied.
This module needs only the supplied repository's RELEASE-EXCLUDE.txt; publisher
scaffolding under work/ is deliberately absent from a public source snapshot.

It also carries the private-machine rule both release scans apply to every
file a wheel carries, code and data: the snapshot and export scan over the
public tree (``work/build_release_snapshot.py``, which the cut's battery and export run)
and the artifact verifier over the built wheel
(``tools/verify_release_artifacts.py``).
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re


def read_exclusions(repo_root: str | Path) -> list[str]:
    with (Path(repo_root) / "RELEASE-EXCLUDE.txt").open(encoding="utf-8") as stream:
        return [line for raw in stream if (line := raw.strip())
                and not line.startswith("#")]


def matches(rel: str, rules: list[str]) -> str | None:
    """Return the first rule matching a forward-slash relative path."""
    for rule in rules:
        if rule.endswith("/**"):
            root = rule[:-3]
            if rel == root or rel.startswith(root + "/"):
                return rule
        elif rel == rule:
            return rule
    return None


# ---------------------------------------------------------------------------
# Private machine names and internal folder names in shipped files
# ---------------------------------------------------------------------------
# The breakage: a private machine's name, or the name of an internal working
# folder, published in a wheel.  A wheel ships every byte it holds, and 2.8.1's
# fetch route table carried "posting watch" plus the lab host that watched in
# 20 measured rows (A154); the repository configs and the MPAS sizing table
# carried the same kind of name, and one NetCDF terrain file a LAN address.
# Measurement records name the measurement, never the machine.
#
# Code is in scope.  Through 2.9.0 it was left out on the reasoning that a
# provenance comment is read only by a developer, and the 2.9.0 wheel shipped
# 158 lines in 61 members, nearly all code, naming lab hosts and internal
# report folders (comments, docstrings and recording ``box`` fields): an
# installed package is published text, and an agency's supply-chain review
# reads it.  Comments name the card and the date of a measurement ("test host
# A, RTX 5070 Ti"), never the machine's own name.
#
# The patterns are assembled so this file carries no host name itself.

#: ``(pattern, kind)``: the lab hosts' names (``node`` plus a hyphen and a
#: number, and the same with a ``weather-`` prefix), a private IPv4 address,
#: and an internal working folder (a capitalised ``Downloads`` followed by a
#: path separator).  "node-wide", ``node_4`` and the engine's own lower-case
#: ``downloads/`` cache folder are not these and do not match.
PRIVATE_HOST_MARKERS = (
    (re.compile(r"(?i)(?<![A-Za-z0-9_.-])(?:weather" + "-)?node" + r"-[0-9]+(?![A-Za-z0-9])"),
     "private machine name"),
    (re.compile(r"(?<![0-9.])192" + r"\.168\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9])"),
     "private network address"),
    (re.compile(r"(?<![A-Za-z0-9_])Down" + r"loads(?:/|\\{1,2})"),
     "internal folder name"),
)

#: Suffixes of code.  The private-machine rule reads code a wheel carries
#: (see above); these decide which wheel members are code for the tree scope.
CODE_SUFFIXES = frozenset({
    ".bat", ".c", ".cc", ".cmd", ".cpp", ".cu", ".cuh", ".f", ".f90", ".h",
    ".hpp", ".ps1", ".py", ".pyi", ".rs", ".sh",
})

#: Tree subtrees every non-code file of which a wheel carries: the woof
#: package's package-data (tests/test_package_data_coverage.py holds every
#: data file under gpuwm/ declared), the data companion's ``data/**/*`` and
#: the repository configs (``configs = ["**/*"]``).
WHEEL_DATA_SUBTREES = ("woof/", "recast-woof-data/woof_data/", "configs/")

#: The other package-data declarations, as tree patterns: the tools package's
#: ``release/*.json``, the default-on hydro door's three source tables
#: (declared by e81d769e735, 2.9.3 improve-1), tilestream's one table and the
#: seam document the ``docs`` namespace package carries.
WHEEL_DATA_PATTERNS = ("tools/release/*.json",
                       "tools/hydro_world/data/sources.v1.json",
                       "tools/hydro_world/data/recipes.v1.json",
                       "tools/hydro_world_params/data/params.v1.json",
                       "tilestream/output-scaling.json", "docs/mpas-seam.md")

#: Tree subtrees whose code a wheel carries: the packages the root
#: pyproject's ``packages.find`` includes recursively (``woof.*``,
#: ``tilestream*``, ``configs*``, ``mpas_cycle_bridge*``) and the data
#: companion's package.  Read whole: a development module the build leaves
#: out (tilestream's ``test_*``) is still in the public tree.
WHEEL_CODE_SUBTREES = ("woof/", "recast-woof-data/woof_data/", "configs/",
                       "tilestream/", "mpas_cycle_bridge/")

#: The top-level-only packages the root pyproject includes (``tools`` and
#: ``docs``, not their subpackages), as tree patterns of their modules.
WHEEL_CODE_PATTERNS = ("tools/*.py", "docs/*.py")


def _is_code(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in CODE_SUFFIXES


def ships_as_wheel_data(rel: str) -> bool:
    """Is this tree path a data file one of the wheels carries?"""
    if _is_code(rel):
        return False
    if rel.startswith(WHEEL_DATA_SUBTREES):
        return True
    path = PurePosixPath(rel)
    return any(len(path.parts) == len(PurePosixPath(pattern).parts)
               and path.match(pattern) for pattern in WHEEL_DATA_PATTERNS)


def ships_in_wheel(rel: str) -> bool:
    """Is this tree path a file one of the wheels carries, code or data?"""
    if ships_as_wheel_data(rel):
        return True
    if not _is_code(rel):
        return False
    if rel.startswith(WHEEL_CODE_SUBTREES):
        return True
    path = PurePosixPath(rel)
    return any(len(path.parts) == len(PurePosixPath(pattern).parts)
               and path.match(pattern) for pattern in WHEEL_CODE_PATTERNS)


def is_wheel_member(name: str) -> bool:
    """Is this wheel member a file the package installs, code or data?

    Everything but a directory entry and the wheel's own metadata.
    """
    top = name.split("/", 1)[0]
    return not (top.endswith(".dist-info") or name.endswith("/"))


#: The rule's one allowance.  A shipped file whose exact bytes a committed
#: record pins is a record, and rewording the machine out of it would falsify
#: the record, not clean it: the experiment config a published WRF reference
#: was built for, whose sha256 is that reference's ``config_sha256`` and names
#: its files.  The allowance is by digest, so any edit to the file ends it.
PINNED_RECORD_MANIFESTS = "docs/public/wrf-reference/*.manifest.json"


def pinned_record_digests(repo_root: str | Path) -> frozenset[str]:
    """The sha256 of every config a committed WRF reference manifest pins.

    A manifest that cannot be read pins nothing, so the rule stays strict.
    """
    import json

    digests = set()
    for manifest in sorted(Path(repo_root).glob(PINNED_RECORD_MANIFESTS)):
        try:
            digest = json.loads(manifest.read_text(encoding="utf-8")).get(
                "config_sha256")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            digests.add(digest)
    return frozenset(digests)


#: The rule's second allowance, also by digest: CUDA kernel sources whose
#: exact bytes the kernel source freeze gate pins
#: (tests/test_kernel_source_freeze_per_module.py, tests/test_mp8_frozen.py)
#: and whose module identity keys the measured frame recordings
#: (woof/core/kernel_frame_recordings.py) and the rounding baseline.  Each
#: still carries one or two comment lines naming a lab host.  Rewording a
#: comment changes the module's bytes, so it would re-pin a frozen physics
#: kernel and stale the measured frames the launch budget reads, which is a
#: physics-tree change, not a notes fix.  The allowance ends the moment a file
#: is edited for any reason: its digest no longer matches, the scan refuses the
#: host name, and the edit that re-pins the kernel drops it in the same commit.
#: tests/test_release_snapshot_machine_paths.py fails on an entry whose file no
#: longer has these bytes, so a spent entry cannot linger.
PINNED_KERNEL_SOURCES = {
}


def pinned_kernel_source_digests() -> frozenset[str]:
    """The sha256 of every frozen kernel source the rule allows as it is."""
    return frozenset(PINNED_KERNEL_SOURCES.values())


def private_host_kinds() -> frozenset[str]:
    return frozenset(kind for _pattern, kind in PRIVATE_HOST_MARKERS)


# ---------------------------------------------------------------------------
# The private data-assimilation system in public files
# ---------------------------------------------------------------------------
# The breakage: the private data-assimilation system published in an ArWen
# release.  Owner rulings 2026-10-14 and 2026-10-16: the JCSDA-framework DA
# (our CUDA port of it, its observation-data writers, its LETKF orchestration),
# the DA rows and cases, the observation pipelines that feed it and the box
# tooling stay in the private DA repository; plain observation readers and
# decoders, and the engine's own local DA (woof.da, ``woof local-da``), are
# public.  2.9.2's compose merged observation lanes whose GLM door carried the
# DA side's zero-lightning observations and observation errors and whose
# comments named the private DA repository, its converter and its radar
# crates, and the 2.9.0 export already carried a stock JCSDA build recipe and
# three rented-box DA kits; no scan read for any of it.
#
# The markers read every in-scope file of the export, code included (a
# comment is published text).  They name the private system, not the
# engine's public DA vocabulary: LETKF, EnKF, superob, RTPS and the like are
# the engine's own local DA and are never matched, which is the allowance for
# the public local-DA names.  Every pattern is assembled from fragments so
# this file carries none of them.
_J = "je" + "di"
DA_TOOLING_MARKERS = (
    (re.compile(r"(?i)(?<![a-z])" + _J + r"(?![a-z])"),
     "JCSDA DA framework (the private DA system)"),
    (re.compile(r"(?i)(?<![a-z])" + "io" + r"da(?![a-z])"),
     "JCSDA observation-data format (the private DA system's writers)"),
    (re.compile(r"(?i)woof[-_]" + "da(?![a-z])|woof" + "da"),
     "private DA repository"),
    (re.compile(r"(?i)hex[-_]radar|rw[-_]radar[-_]qc|radar_" + "qc|lane" + "kit|gpu-" + "slot"
                r"|arwen_" + "cycle|obs_set_" + "passes|rapid5_" + "live|resident_" + "bench"),
     "private DA tooling"),
    (re.compile(r"(?i)column[-_ ]?letkf|letkf[-_]column"),
     "private LETKF orchestration"),
    (re.compile(r"(?i)enkf[-_]case|(?<![a-z-])da-(?:enkf|us-)"),
     "private DA case row"),
)

#: Upstream names that contain a marker and name nobody's private code: the
#: RRFS v2 workflow's application (it pins the physics hex and the engine
#: port) and MPAS-Atmosphere's own namelist option.  A line is read with
#: these masked, so anything else on it still counts.
DA_TOOLING_ALLOWED_PHRASES = (
    re.compile(r"(?i)rrfs-mpas-" + _J),
    re.compile(r"(?i)(?<![a-z])(?:config_)?" + _J + r"_da(?![a-z])"),
)

#: Files that name the upstream JCSDA system as a fact, with the exact number
#: of lines that do (``(lines, reason)``).  The allowance is the count, so a
#: new line in one of these files is read like any other file's;
#: tests/test_release_snapshot_machine_paths.py fails on an entry whose count
#: no longer matches, so a spent entry cannot linger.
DA_TOOLING_ALLOWED_FILES = {
    "RELEASE-EXCLUDE.txt": (1, "the exclusion line that keeps the stock JCSDA build recipe out of the export"),
    "docs/da-vs-wofs.md": (1, "states NSSL's published plan for WoFS v2"),
    "docs/nwpsuite-plan-json.md": (1, "says the suite planner never fabricates a DA launch command"),
    "woof/data/nwpsuite/rates.v1.json": (2, "the suite planner's stock upstream CPU DA solve rate (public since 2.9.0)"),
    "woof/data/nwpsuite/receipts.v1.json": (6, "the stock upstream DA runs the planner's rates cite (public since 2.9.0)"),
    "woof/data/nwpsuite/references.v1.json": (3, "the citations of those receipts (public since 2.9.0)"),
    "woof/data/nwpsuite/shapes.v1.json": (10, "the stock upstream DA methods an agency suite shape may name (public since 2.9.0)"),
    "woof/data/nwpsuite/suite.v1.schema.json": (4, "the schema enum of those methods (public since 2.9.0)"),
    "woof/data/nwpsuite/templates/agency-2box.toml": (1, "the agency template's shadow DA row, stock upstream method (public since 2.9.0)"),
    "woof/data/kernel-pack-cuda13.json": (1, "no name at all: the base64 of a compiled cubin (cubin_xz_base64) in the cold-start kernel pack spells the marker by chance; the file is one JSON line"),
}


def da_tooling_kind(line: str) -> str | None:
    """The DA-tooling kind one line (or printable run) matches, if any."""
    for pattern in DA_TOOLING_ALLOWED_PHRASES:
        line = pattern.sub(" ", line)
    for pattern, kind in DA_TOOLING_MARKERS:
        if pattern.search(line):
            return kind
    return None


def da_tooling_violations(rel: str, text: str) -> list[tuple[int, str, str]]:
    """``[(lineno, kind, line)]`` for one public file's text, allowance applied."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        kind = da_tooling_kind(line)
        if kind is not None:
            found.append((number, kind, line.strip()[:120]))
    allowed = DA_TOOLING_ALLOWED_FILES.get(rel)
    if found and allowed is not None and len(found) == allowed[0]:
        return []
    return found


def da_tooling_kinds() -> frozenset[str]:
    return frozenset(kind for _pattern, kind in DA_TOOLING_MARKERS)
