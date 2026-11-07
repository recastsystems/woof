"""No shipped file may carry one machine's absolute paths.

RELEASE-EXCLUDE has always *said* "no developer-specific absolute
paths".  Nothing enforced it, and the v1.0.1 exit audit found 51 such
lines across 25 tracked files that every exclusion rule had let
through -- a WSL home in an executable default, a Windows profile in a
committed process capture, a developer's tree in an oracle receipt.

An exclusion manifest cannot close that class: it cannot name the file
that grows a machine path next week.  So the snapshot builder scans
every staged text file and refuses to build.  This module holds the
tests for that scan, plus the regression that matters most -- the
current tree, minus its exclusions, is clean.

Every offending string below is assembled from fragments rather than
written out, because this file is one of the files the scan reads.  A
literal here would flag the test that tests the scan, which is both
confusing and, at the moment it happened, indistinguishable from a real
finding.  ``tools/build_native_wrf_distribution.py`` does the same for
the same reason.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "work" / "build_release_snapshot.py"

#: The snapshot builder lives under ``work/``, which RELEASE-EXCLUDE
#: keeps out of the public release -- so in a published tree there is
#: nothing here to test.  A development checkout always has it.
requires_builder = pytest.mark.skipif(
    not BUILDER.is_file(),
    reason="work/build_release_snapshot.py is not in this tree "
           "(published snapshot: the builder is publisher scaffolding)",
)

requires_git = pytest.mark.skipif(
    not (REPO / ".git").exists(),
    reason="not a git checkout (installed package or exported tree)",
)


def _release_tree_files() -> list[str]:
    """Every path the release would carry, git checkout or not.

    The tree scan used to be gated on ``git ls-files``, and the 2.5.0
    Linux shakeout measured what that costs: run against the EXPORTED
    tree -- exactly the tree that gets packaged -- this module reported
    "9 passed, 3 skipped", and the three skipped were the only ones that
    read the tree at all.  A gate that reports green in the one context
    it exists for is not a gate, so off-checkout the walk stands in for
    the index.  In a checkout the index still wins: it is the only
    answer that excludes a developer's untracked scratch, which is not
    shipped and whose paths would be noise.
    """

    if (REPO / ".git").exists():
        return subprocess.run(
            ["git", "-C", str(REPO), "ls-files"],
            check=True, capture_output=True, text=True).stdout.splitlines()
    found = []
    for path in REPO.rglob("*"):
        if path.is_file():
            found.append(path.relative_to(REPO).as_posix())
    return found


def _snap():
    sys.path.insert(0, str(REPO / "work"))
    try:
        import build_release_snapshot as module
    finally:
        sys.path.pop(0)
    return module


# The four shapes a per-user absolute path takes, none of them naming a
# real person -- built at runtime so this file stays clean.
_BS = chr(92)
WINDOWS_PROFILE = "C:" + _BS + "Users" + _BS + "somebody" + _BS + "notes.txt"
WSL_PROFILE = "/mnt/" + "c/Users/" + "somebody/Downloads/bundle"
POSIX_HOME = "/ho" + "me/somebody/wrf-build"
MAC_HOME = "/Us" + "ers/somebody/Library/data"
ROOT_HOME = "/ro" + "ot/artifacts"
#: The same Windows path as it appears inside JSON -- one separator per
#: level becomes two, because JSON escapes the backslash.  Every receipt
#: and proof document in the tree writes it this way.
WINDOWS_PROFILE_JSON = WINDOWS_PROFILE.replace(_BS, _BS + _BS)


@requires_builder
@pytest.mark.parametrize(
    ("payload", "kind"),
    (
        (WINDOWS_PROFILE, "Windows user profile"),
        ("C:/" + "Users/somebody/notes.txt", "Windows user profile"),
        # Hole B, measured on the 2.5.0 tip: the marker demanded exactly
        # one separator after the drive colon, so every JSON-escaped
        # spelling walked past it.  Five such lines shipped in both
        # wheels and both sdists while the gate reported 12 passed.
        (WINDOWS_PROFILE_JSON, "Windows user profile"),
        (WSL_PROFILE, "WSL-mounted user profile"),
        (POSIX_HOME, "POSIX home directory"),
        (MAC_HOME, "macOS home directory"),
        (ROOT_HOME, "root home directory"),
    ),
)
def test_every_machine_path_shape_is_caught(payload, kind):
    snap = _snap()
    found = snap.machine_path_violations(f"prefix\nSRC={payload}\nsuffix\n")
    assert [(number, label) for number, label, _ in found] == [(2, kind)]


@requires_builder
def test_ordinary_absolute_paths_are_not_flagged():
    """The bar is a *per-user* path, not any absolute path."""

    snap = _snap()
    benign = "\n".join([
        "prefix = /usr/local/share/wps-geog",
        "OUT=/tmp/noahmp-oracle",
        "cache = /var/cache/arwen",
        'root = "C:' + _BS + 'Program Files' + _BS + 'NVIDIA"',
        "home = $HOME/WRF_BUILD/LIBRARIES",
        # A TEMPLATE is the remedy, not the defect: the user segment is
        # substituted at run time and names nobody.  Widening the
        # Windows marker to see through JSON escaping must not start
        # flagging the shapes that exist to avoid the problem --
        # tests/test_report_bundle.py builds exactly the first one to
        # prove the report bundler redacts a home directory.
        'out = f"C:' + _BS * 2 + 'Users' + _BS * 2 + '{username}"',
        "out = %USERPROFILE%" + _BS + "woof",
        "out = $env:USERPROFILE/gpuwm",
    ])
    assert snap.machine_path_violations(benign) == []


@requires_builder
def test_the_scan_walks_a_staged_tree_and_skips_vendored_and_binary(tmp_path):
    snap = _snap()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(
        f"run it from {POSIX_HOME}\n", encoding="utf-8")
    # Third-party source: upstream authors' paths are theirs, the trees
    # are checksum-locked, and editing them breaks the build.
    (tmp_path / "crate" / "vendor" / "foo").mkdir(parents=True)
    (tmp_path / "crate" / "vendor" / "foo" / "lib.rs").write_text(
        f"// built at {MAC_HOME}\n", encoding="utf-8")
    shared = tmp_path / "tools" / "arwen-ui-vendor"
    (shared / "crates-io" / "foo").mkdir(parents=True)
    (shared / "crates-io" / "foo" / "lib.rs").write_text(
        f"// upstream capture: {MAC_HOME}\n", encoding="utf-8")
    # Similar names and first-party siblings must remain in scope.
    (shared / "build.py").write_text(
        f"path = {POSIX_HOME}\n", encoding="utf-8")
    (shared / "crates-io-local").mkdir()
    (shared / "crates-io-local" / "lib.rs").write_text(
        f"// path = {MAC_HOME}\n", encoding="utf-8")
    # Real binary: a NUL in the first block is what "not text" means.
    (tmp_path / "capture.bin").write_bytes(
        b"\x00\x01\x02" + WINDOWS_PROFILE.encode("utf-8") + b"\n")

    hits = snap.machine_path_hits(str(tmp_path))
    assert sorted((rel, number) for rel, number, _, _ in hits) == [
        ("docs/guide.md", 1),
        ("tools/arwen-ui-vendor/build.py", 1),
        ("tools/arwen-ui-vendor/crates-io-local/lib.rs", 1)]


@requires_builder
@pytest.mark.parametrize(
    "name",
    (
        # Hole A, measured on the 2.5.0 tip: SCANNED_SUFFIXES listed 22
        # suffixes and these were not among them, so eleven developer
        # paths -- including three shipped WPS namelists pointing at a
        # geography root that exists on one box -- were never read at
        # all.  A suffix ALLOWLIST is the defect, not the entries it is
        # missing: the file that grows a machine path next has a suffix
        # nobody thought of, which is how a .cu comment, a .wps config
        # and an extensionless script all walked past a green gate.
        "kernels/thompson.cu",
        "kernels/thompson.cuh",
        "configs/probe.namelist.wps",
        "tools/run-the-case",
        "notes.rst",
        "Makefile",
    ),
)
def test_the_scan_reads_text_by_content_not_by_suffix(tmp_path, name):
    snap = _snap()
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"path = {POSIX_HOME}\n", encoding="utf-8")

    hits = snap.machine_path_hits(str(tmp_path))
    assert [(rel, number) for rel, number, _, _ in hits] == [
        (name, 1)], f"{name} was not read, so its machine path ships unseen"


def _netcdf_classic(attributes: dict) -> bytes:
    """A minimal NetCDF classic file (CDF-1): no dimensions, no variables, these global text attributes."""

    import struct

    def padded(data: bytes) -> bytes:
        return data + b"\x00" * (-len(data) % 4)

    body = b"CDF\x01" + struct.pack(">i", 0) + b"\x00" * 8
    body += struct.pack(">ii", 12, len(attributes))  # NC_ATTRIBUTE
    for name, value in attributes.items():
        encoded = value.encode("ascii")
        body += struct.pack(">i", len(name)) + padded(name.encode("ascii"))
        body += struct.pack(">ii", 2, len(encoded)) + padded(encoded)  # NC_CHAR
    return body + b"\x00" * 8


@requires_builder
def test_the_scan_reads_netcdf_attributes(tmp_path):
    """2.8.0 shipped a home folder inside three renderer fixtures.

    ncks writes its command line, output path included, into the global
    ``history`` attribute, and ``.nc`` was on the never-text suffix skip,
    so the scan never read those files.  NetCDF-4 keeps the same text in
    HDF5 object headers, which is why a container is read by its printable
    runs rather than by one layout's parser.
    """

    snap = _snap()
    history = "Tue Sep 29 00:11:21 2026: ncks -O -4 -d Time,1 in.nc " + POSIX_HOME + "/cut.nc"
    fixtures = tmp_path / "crate" / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "classic.nc").write_bytes(
        _netcdf_classic({"START_DATE": "2025-03-15_12:00:00", "history": history}))
    hdf5_like = (b"\x89HDF\r\n\x1a\n" + b"\x00" * 64 + b"history\x00\x00"
                 + history.encode("ascii") + b"\x00" * 64)
    (fixtures / "netcdf4.nc").write_bytes(hdf5_like)
    # The suffix skip still holds for binaries that are not containers.
    (fixtures / "capture.bin").write_bytes(hdf5_like)
    (fixtures / "clean.nc").write_bytes(
        _netcdf_classic({"START_DATE": "2025-03-15_12:00:00", "title": "time axis fixture"}))

    hits = snap.machine_path_hits(str(tmp_path))
    assert sorted((rel, kind) for rel, _where, kind, _ in hits) == [
        ("crate/tests/fixtures/classic.nc", "POSIX home directory"),
        ("crate/tests/fixtures/netcdf4.nc", "POSIX home directory")]
    assert all(where.startswith("byte ") and POSIX_HOME in text for _rel, where, _k, text in hits)


@requires_builder
def test_the_scan_is_wired_into_the_snapshot_verdict():
    """A finding must FAIL the build, not merely print."""

    source = BUILDER.read_text(encoding="utf-8")
    assert "machine_paths = machine_path_hits(SNAP)" in source
    verdict = source.split("ok = ", 1)[1].split("\n\n", 1)[0]
    assert "not machine_paths" in verdict


@requires_builder
def test_cpu_preparation_calibration_files_carry_no_machine_paths():
    snap = _snap()
    paths = sorted((REPO / "tests").glob("*cpu_preparation*"))
    assert paths, "the CPU preparation calibration is missing"
    offenders = []
    for path in paths:
        if path.is_file():
            offenders.extend(
                (path.name, number, kind)
                for number, kind, _ in snap.machine_path_violations(
                    path.read_text(encoding="utf-8")))
    assert offenders == []


@requires_builder
def test_the_staged_release_tree_carries_no_machine_paths():
    """The regression the audit finding actually asks for.

    Applies RELEASE-EXCLUDE to the release file set exactly as the
    builder does, then reads every in-scope file.  A new machine path
    anywhere in WOOF fails here, days before anyone runs the release
    build -- and now in an exported tree too, where this used to skip.
    """

    snap = _snap()
    rules = snap.read_exclusions()

    def kept(rel: str) -> bool:
        parts = rel.split("/")
        return not any(
            snap.matches("/".join(parts[:index]), rules)
            for index in range(1, len(parts) + 1))

    # In a private checkout the suite tables still hold the rows the cut
    # removes; an exported tree already carries the rewritten tables and not
    # the cut tooling, so the rewrite is read by path only where it exists.
    hygiene_path = REPO / "tools" / "release" / "cut" / "hygiene.py"
    suite_files, public_suite = (), None
    if hygiene_path.is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location("cut_hygiene_by_path", hygiene_path)
        hygiene = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = hygiene
        spec.loader.exec_module(hygiene)
        suite_files, public_suite = hygiene.SUITE_FILES, hygiene.public_suite

    offenders = []
    scanned = containers = 0
    for rel in _release_tree_files():
        if not rel or not kept(rel) or not snap.in_scan_scope(rel):
            continue
        path = REPO / rel
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if rel in suite_files:
            # The cut publishes these tables only after removing the
            # private suite rows (tools/release/cut/hygiene.py); read the
            # bytes it ships, as the export's own scan does.
            raw = public_suite(rel, raw.decode("utf-8")).encode("utf-8")
        container = Path(rel).suffix.lower() in snap.METADATA_CONTAINER_SUFFIXES
        if not container and not snap.reads_as_text(raw):
            continue
        scanned += 1
        containers += container
        for where, kind, line in snap.file_violations(rel, raw):
            offenders.append(f"{rel}:{where} ({kind}) {line}")

    assert scanned > 500, "the scan found almost nothing to read"
    assert containers > 0, "the scan read no NetCDF file, so their attributes ship unread"
    assert offenders == [], (
        f"{len(offenders)} developer-absolute path(s) or private machine "
        f"name(s) would ship:\n  "
        + "\n  ".join(offenders)
        + "\nParameterize/relativize the path if the file is useful to "
          "the public, else add it to RELEASE-EXCLUDE.txt; a data file a "
          "wheel carries names the measurement, not the machine it ran on "
          "(tools/release_exclusions.py holds that rule).")


@requires_builder
def test_every_pinned_record_the_scan_forgives_is_still_the_pinned_bytes():
    """The one allowance, and the proof it is still what it claims.

    ``ALLOWED_MACHINE_PATH_FILES`` forgives files whose exact bytes are
    pinned by a committed manifest: a WPS namelist recorded as the
    provenance of the WRF reference wrfouts is a RECORD of one build on
    one box, and editing the developer path out of it would break the
    ``namelist_sha256`` the manifest beside it pins -- falsifying the
    record rather than cleaning it.

    An allowance nobody re-checks is just a hole with a comment, so
    this reads the pin: if the file's bytes ever stop matching the
    manifest that justifies forgiving them, it is no longer a pinned
    record and the allowance dies here.
    """

    import hashlib
    import json

    snap = _snap()
    assert snap.ALLOWED_MACHINE_PATH_FILES, (
        "the allowance list is empty; delete it rather than leaving an "
        "unused escape hatch in a gate")
    for rel, pin in snap.ALLOWED_MACHINE_PATH_FILES.items():
        path = REPO / rel
        if not path.is_file():
            pytest.skip(f"{rel} is not in this tree")
        manifest = json.loads(
            (REPO / pin.manifest).read_text(encoding="utf-8"))
        expected = manifest
        for key in pin.keys:
            expected = expected[key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, (
            f"{rel} is forgiven by the machine-path scan because "
            f"{pin.manifest} pins its bytes, but it now hashes "
            f"{digest} against the pinned {expected}. It is no longer "
            "the record the allowance describes: scrub the machine "
            "path and drop the allowance, or restore the pinned bytes.")


@requires_builder
@requires_git
def test_the_campaign_harness_exclusions_still_match_something():
    """A rule that matches nothing is manifest drift, not hygiene.

    Checkout-only on purpose, unlike the scan above: an EXPORTED tree
    has already had the exclusions applied, so every rule matching
    nothing there is the correct answer, not drift.
    """

    snap = _snap()
    rules = snap.read_exclusions()
    tracked = _release_tree_files()
    for rule in ("tools/n5s/**", "tests/test_n5s_toolchain.py",
                 "tools/rrtmg_wrf461_oracle/sw_fixtures/"
                 "sw-oracle-sha256sums.txt"):
        assert rule in rules, f"{rule} is no longer in RELEASE-EXCLUDE.txt"
        root = rule[:-3] if rule.endswith("/**") else rule
        assert any(rel == root or rel.startswith(root + "/")
                   for rel in tracked), f"{rule} matches no tracked path"


@requires_builder
def test_nothing_shipped_imports_the_excluded_campaign_harness():
    """Excluding a package must not leave a dangling import behind."""

    snap = _snap()
    rules = snap.read_exclusions()
    tracked = [rel for rel in _release_tree_files() if rel.endswith(".py")]
    needle = "tools." + "n5s"
    importers = []
    for rel in tracked:
        parts = rel.split("/")
        if any(snap.matches("/".join(parts[:i]), rules)
               for i in range(1, len(parts) + 1)):
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8",
                                          errors="replace")
        except OSError:
            # The same skip the machine-path scan above already takes,
            # and for a reason this module can name: off-checkout the
            # listing is a WALK, and under `pytest -n` a sibling test may
            # delete a path between the walk and this read.
            # tests/test_fastfix_selector.py's deletion probe does
            # exactly that, deliberately -- it writes
            # tests/test_fastfix_deletion_probe_importer.py and
            # woof/_fastfix_deletion_probe.py into the real tree,
            # because the defect it covers was a disagreement between an
            # on-disk walk and an index lookup and a fake tree cannot
            # reproduce that -- and unlinks them in its own finally. A
            # path that is gone by the time this test reads it ships
            # nothing and imports nothing.
            continue
        if needle in text:
            importers.append(rel)
    assert importers == [], (
        f"these ship but import the excluded harness: {importers}")


# ---------------------------------------------------------------------
# Private machine names in the files a wheel carries (A154)
# ---------------------------------------------------------------------
# 2.8.1's fetch route table named the lab host that watched the posting
# times in 20 measured rows: a private machine's name published in a
# wheel.  2.9.0's wheel carried lab host and internal folder names in 61
# members, nearly all code, because code was out of scope.  Built from
# fragments, like the paths above.
HOST = "no" + "de-4"
WEATHER_HOST = "weather-" + "no" + "de-1"
LAN = "192." + "168.68.50"
FOLDER = "Down" + "loads/"


# Replaces test_private_machine_names_are_refused_in_shipped_data_files
# (122ce03da): the rule reads every shipped file, code included, because the
# 2.9.0 wheel carried lab host names in code the data-only rule never read.
@requires_builder
def test_private_machine_names_are_refused_in_shipped_files(tmp_path):
    snap = _snap()
    files = {
        # Shipped data: refused.
        "woof/authorities/routes.json":
            '{"measured": "2026-09-30 posting watch, ' + HOST + ' (x)"}\n',
        "configs/demo.toml": "# Linux (" + WEATHER_HOST + ")\n",
        "woof/data/oracle/PROVENANCE.md": "recorded on " + HOST + "\n",
        "tools/release/identities.json": '{"box": "' + LAN + '"}\n',
        # Shipped code: refused too.
        "woof/core/timing.py": "# measured on " + WEATHER_HOST + "\n",
        "tools/probe.py": "# see " + FOLDER + "RUN-REPORT/REPORT.md\n",
        # A document outside every wheel, a tools subpackage the wheel does
        # not carry, the engine's own lower-case cache folder and words that
        # are not a host: read for machine paths only.
        "woof/go_cli.py": "# downloads/<key>/<object>.part\n",
        "tools/release/cut/drivers/probe.py": "# on " + HOST + "\n",
        "docs/public/receipts/run.json": '{"host": "' + HOST + '"}\n',
        "tools/battery/list.txt": HOST + "\n",
        "woof/data/governor.json": '{"note": "the node-wide governor, node_4"}\n',
    }
    for rel, text in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    hdf5_like = (b"\x89HDF\r\n\x1a\n" + b"\x00" * 64 + b"source\x00\x00"
                 + ("met_em on " + WEATHER_HOST + " (" + LAN + ")").encode("ascii")
                 + b"\x00" * 64)
    (tmp_path / "configs" / "real").mkdir(parents=True)
    (tmp_path / "configs" / "real" / "terrain.nc").write_bytes(hdf5_like)

    hits = snap.machine_path_hits(str(tmp_path))
    assert sorted((rel, kind) for rel, _where, kind, _ in hits) == [
        ("configs/demo.toml", "private machine name"),
        ("configs/real/terrain.nc", "private machine name"),
        ("woof/authorities/routes.json", "private machine name"),
        ("woof/core/timing.py", "private machine name"),
        ("woof/data/oracle/PROVENANCE.md", "private machine name"),
        ("tools/probe.py", "internal folder name"),
        ("tools/release/identities.json", "private network address")]


def test_every_frozen_kernel_allowance_still_matches_its_file():
    """A kernel-source allowance whose file has moved on is spent.

    The allowance is by digest, so an edited kernel is already refused if it
    still names a host; this keeps a spent entry from lingering in the table.
    """

    import hashlib

    from tools.release_exclusions import PINNED_KERNEL_SOURCES

    spent = {rel: digest for rel, digest in PINNED_KERNEL_SOURCES.items()
             if not (REPO / rel).is_file()
             or hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != digest}
    assert spent == {}, (
        "these kernel sources no longer carry the bytes the host-name "
        f"allowance names; drop the entry (and any host name left): {spent}")


@requires_builder
def test_the_host_rule_covers_every_package_data_declaration():
    """The rule's scope is the wheels' data, so it has to follow them.

    Each package-data pattern of both pyprojects, instantiated as a tree
    path, is either code or in the rule's scope: a new declaration that
    ships data from somewhere else fails here instead of shipping unread.
    """

    import tomllib

    from tools.release_exclusions import CODE_SUFFIXES, ships_as_wheel_data

    declared = []
    for project, root in ((REPO, ""), (REPO / "recast-woof-data", "recast-woof-data/"), (REPO / "gpuwm-kernels", "gpuwm-kernels/"), (REPO / "gpuwm-kernels-startup", "gpuwm-kernels-startup/")):
        pyproject = project / "pyproject.toml"
        if not pyproject.is_file():
            continue
        with pyproject.open("rb") as stream:
            package_data = tomllib.load(stream)["tool"]["setuptools"][
                "package-data"]
        for package, patterns in package_data.items():
            for pattern in patterns:
                example = pattern.replace("**/", "sub/").replace("*", "x")
                declared.append(root + package.replace(".", "/") + "/" + example)
    assert len(declared) > 15, declared
    outside = [rel for rel in declared
               if Path(rel).suffix.lower() not in CODE_SUFFIXES
               and not ships_as_wheel_data(rel)]
    assert outside == [], (
        "package-data ships these data files and the private-machine rule "
        f"does not read them: {outside}")


@requires_builder
def test_the_host_rule_forgives_only_the_exact_bytes_a_reference_pins(
        tmp_path, monkeypatch):
    """The rule's one allowance is by digest, so an edit ends it."""

    import hashlib

    snap = _snap()
    record = ("# the reference's own met_em, retained on " + WEATHER_HOST
              + "\n").encode("utf-8")
    monkeypatch.setattr(snap, "_PINNED_RECORD_DIGESTS",
                        [frozenset({hashlib.sha256(record).hexdigest()})])
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "reference.toml").write_bytes(record)
    (tmp_path / "configs" / "edited.toml").write_bytes(record + b"# edited\n")

    hits = snap.machine_path_hits(str(tmp_path))
    assert [(rel, kind) for rel, _where, kind, _ in hits] == [
        ("configs/edited.toml", "private machine name")]


@requires_builder
def test_the_host_rule_allowance_is_live_and_names_one_shipped_config():
    """An allowance that forgives nothing is an unused escape hatch.

    The committed WRF reference manifests pin the config each reference was
    built for; each pinned digest is the digest of exactly one shipped
    config, which is the file the allowance exists for.
    """

    import hashlib

    from tools.release_exclusions import pinned_record_digests

    digests = pinned_record_digests(REPO)
    if not (REPO / "docs" / "public" / "wrf-reference").is_dir():
        pytest.skip("the WRF reference records are not in this tree")
    assert digests, "no committed WRF reference manifest pins a config"
    shipped = {}
    for path in sorted((REPO / "configs").rglob("*.toml")):
        shipped.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(),
                           []).append(path.relative_to(REPO).as_posix())
    for digest in digests:
        assert len(shipped.get(digest, [])) == 1, (
            f"the WRF reference pinned at {digest} names no single shipped "
            f"config: {shipped.get(digest)}")


#: The owner's identity, assembled so this file carries none of it.
OWNER_NAME = "Dr" + "ew"
OWNER_MAIL = "fahrenheit" + "agi" + "@example.invalid"


@requires_builder
def test_the_owners_mail_is_refused_and_the_first_name_is_a_word(tmp_path):
    """Code is read too: a comment is published text (owner ruling 2026-09-10).

    The e-mail is refused wherever it sits.  The owner's first name is not
    read in any spelling (owner ruling 2026-10-17: "for future remove the
    drew rule thats so dumb"); it is an English verb, and the scan that read
    it blocked cuts over renderer sentences.  Replaces
    test_the_owners_name_and_mail_are_refused_in_every_shipped_file.
    """

    snap = _snap()
    files = {
        "woof/namelist_chem.py": "# " + OWNER_NAME + "'s ruling (2026-10-12)\n",
        "tests/test_door.py": "# ---- &chem (" + OWNER_NAME + ", 2026-10-12) ----\n",
        "docs/public/receipts/run.json": '{"author": "' + OWNER_MAIL + '"}\n',
        "tools/rustwx/crates/x/src/gwd.rs": ('attr("PROPOSED FOR ' + OWNER_NAME.upper()
                                             + ': case-specific sign-off required");\n'),
        "woof/render.py": ("# the renderer " + OWNER_NAME.lower()
                            + " 43 pictures; with" + OWNER_NAME.lower() + "; An" + OWNER_NAME.lower()
                            + "sville\n"),
    }
    for rel, text in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    hits = snap.machine_path_hits(str(tmp_path))
    assert sorted((rel, kind) for rel, _where, kind, _ in hits) == [
        ("docs/public/receipts/run.json", "owner's e-mail")]


# The private DA system's names, assembled so this file carries none of them
# (tools/release_exclusions.py, owner rulings 2026-10-14 and 2026-10-16).
_J = "JE" + "DI"
_DA_SAMPLES = {
    # The 2.9.2 compose's GLM door described its table by the private DA
    # repository's converter; a comment is published text.
    "woof/obs/cli.py": "# the cycle's GLM converter in woof" + "-da\n",
    "tools/rustwx/crates/rw-goes/src/glm.rs": "//! MPAS-" + _J + " observations of the cycle\n",
    "tools/rustwx/crates/rw-radar-feeds/Cargo.toml": "# (`rw_odim`, `hex-" + "radar-superob --odim`) consume.\n",
    "tools/box/run.sh": "source ~/lane" + "kit/env.sh\n",
    "docs/da-cycle.md": "the CUDA Column " + "LETKF solves each window\n",
    "configs/rows/case.json": '{"schema": "hex.enkf' + '-case.v1"}\n',
}
_DA_KINDS = {
    "woof/obs/cli.py": "private DA repository",
    "tools/rustwx/crates/rw-goes/src/glm.rs": "JCSDA DA framework (the private DA system)",
    "tools/rustwx/crates/rw-radar-feeds/Cargo.toml": "private DA tooling",
    "tools/box/run.sh": "private DA tooling",
    "docs/da-cycle.md": "private LETKF orchestration",
    "configs/rows/case.json": "private DA case row",
}


@requires_builder
def test_the_private_da_system_is_refused_in_every_public_file(tmp_path):
    """The planted sample: every kind of private DA content fails the scan.

    The breakage: 2.9.2's compose merged a GLM door that wrote the DA side's
    zero-lightning observations and errors and named the private DA
    repository, and the 2.9.0 export carried a stock JCSDA build recipe and
    rented-box DA kits, while no scan read for any of it.
    """

    snap = _snap()
    for rel, text in _DA_SAMPLES.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    hits = snap.machine_path_hits(str(tmp_path))
    assert {rel: kind for rel, _where, kind, _ in hits} == _DA_KINDS
    # One sample per marker, so a marker nobody exercises cannot rot.
    from tools.release_exclusions import da_tooling_kinds
    assert set(_DA_KINDS.values()) | {"JCSDA observation-data format (the private DA system's writers)"} \
        == set(da_tooling_kinds())
    assert snap.da_violations("x.md", ("an " + "IO" + "DA file\n").encode())[0][1].startswith("JCSDA observation")


@requires_builder
def test_the_public_local_da_and_upstream_names_are_not_the_private_system(tmp_path):
    """The engine's own local DA vocabulary and upstream names pass.

    woof.da is public (LETKF, EnKF, superobs, RTPS, ``woof local-da``), a
    "WOOF DA" panel label names a product, and the RRFS v2 application and
    MPAS's namelist option contain the framework's name without being it.
    """

    snap = _snap()
    files = {
        "woof/da/letkf.py": "# local LETKF with RTPS; EnKF moments; radar superob\n",
        "docs/local-da.md": "woof local-da run --da-root $DA_ROOT (WOOF DA member 0 panel)\n",
        "tools/build_registry.py": '"pinned_by": "RRFS v2 (rrfs-workflow rrfs-mpas-' + _J.lower() + ')"\n',
        "tests/test_namelist.py": '{"' + _J.lower() + '_da": "false", "config_' + _J.lower() + '_da": false}\n',
    }
    for rel, text in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    assert snap.machine_path_hits(str(tmp_path)) == []
    # Masking an upstream name leaves the rest of the line read.
    line = "rrfs-mpas-" + _J.lower() + " and woof" + "da"
    assert snap.da_violations("tools/x.py", line.encode())[0][1] == "private DA repository"


def test_every_da_file_allowance_still_counts_its_lines():
    """A file allowance is an exact line count; a spent or grown one fails.

    A new line naming the framework in an allowed file is read like any
    other file's, because the count no longer matches.
    """

    from tools.release_exclusions import DA_TOOLING_ALLOWED_FILES, da_tooling_kind

    wrong = {}
    for rel, (lines, _reason) in DA_TOOLING_ALLOWED_FILES.items():
        path = REPO / rel
        if not path.is_file():
            wrong[rel] = "missing"
            continue
        counted = sum(da_tooling_kind(line) is not None
                      for line in path.read_text(encoding="utf-8").splitlines())
        if counted != lines:
            wrong[rel] = counted
    assert wrong == {}, f"DA allowances whose line count moved: {wrong}"
