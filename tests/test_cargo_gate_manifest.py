"""The Rust workspaces' battery list, kept in sync with the workspaces.

``tools/battery/cargo_gates.txt`` is how the release battery finds the
Cargo test suites.  The 2026-08-12 project ruling ("probably we would want
tests right") made them a leg; this suite is the gate on the list, the
same kind of gate ``tests/test_stage1_manifest.py`` is on the stage-1
list and ``tests/test_tiles_gate_manifest.py`` is on the ``[tiles]``
gates.

It carries one property those two do not, and it is the point of the
file: the list is checked against the workspace manifests in BOTH
directions.  Every entry has to name a real member package, and every
member package has to have an entry.  The second direction is the
accepted-implies-builds pattern (``tests/test_mp_accepted_builds.py``)
applied to a battery list rather than to a physics menu: a crate that
joins ``tools/rustwx/Cargo.toml``'s ``members`` turns THIS suite red
until somebody writes its line in the list, so a new crate cannot land
already ungated.  That is how a crate arriving from another lane -- the
Rust MPAS work is the live example -- joins the leg, and it needs no
edit here to happen.

Cheap and hermetic on purpose: it reads three text files and two
``Cargo.toml``s, needs no Rust toolchain and no card, so it runs on
every commit rather than at a cut.  What it CANNOT check is that the
suites pass -- that is the leg itself,
``tools/battery/run_cargo_gates.py``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "cargo_gates.txt"
RUNNER = REPOSITORY_ROOT / "tools" / "battery" / "run_cargo_gates.py"


_RUNNER_MODULE = "battery_run_cargo_gates"


def _runner():
    """Import the runner by path -- ``tools`` is not an importable package.

    Importing it rather than restating its parser is deliberate: a file
    this suite accepts has to be a file the runner can read, and the
    only way to keep those two statements from drifting is to have one
    of them.

    It is registered in ``sys.modules`` before execution because
    ``@dataclass`` resolves annotations through
    ``sys.modules[cls.__module__]``, and a module that executes without
    being registered has none.
    """

    if _RUNNER_MODULE in sys.modules:
        return sys.modules[_RUNNER_MODULE]
    spec = importlib.util.spec_from_file_location(_RUNNER_MODULE, RUNNER)
    assert spec and spec.loader, f"{RUNNER} is not importable"
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUNNER_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _entries():
    return _runner().read_manifest(MANIFEST)


def _lines() -> list[str]:
    return MANIFEST.read_text(encoding="utf-8").splitlines()


def _reasons() -> dict[str, str]:
    """Each entry line mapped to the comment block above its group.

    A blank line ends a block, so the file header does not attach itself
    to the first entry; a block covers the whole contiguous run beneath
    it, so a package split across two entries by target can share one
    rationale.
    """

    reasons: dict[str, str] = {}
    block: list[str] = []
    for raw in _lines():
        line = raw.strip()
        if not line:
            block = []
        elif line.startswith("#"):
            block.append(line.lstrip("#").strip())
        else:
            reasons[" ".join(line.split())] = " ".join(p for p in block if p)
    return reasons


def _path_dependency_members(root: Path, manifest: dict) -> set[str]:
    """Path dependencies that live inside the workspace directory.

    Cargo makes these workspace members automatically -- "all path
    dependencies residing in the workspace directory become members" --
    which is why ``cargo test -p grib-core`` works from
    ``tools/grib1_bridge`` even though nothing lists it.  The gate has to
    know that, or a crate the release ships and the whole tree decodes
    GRIB through is invisible to the both-directions check.
    """

    names = set()
    for table in ("dependencies", "dev-dependencies", "build-dependencies"):
        for spec in manifest.get(table, {}).values():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            candidate = (root / spec["path"]).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue  # outside the workspace directory: not a member
            member_toml = candidate / "Cargo.toml"
            if member_toml.is_file():
                names.add(tomllib.loads(
                    member_toml.read_text(encoding="utf-8"))["package"]["name"])
    return names


def _members(workspace: str) -> set[str]:
    """The package names a workspace directory declares.

    A ``[workspace]`` table means the members' own manifests carry the
    names; a bare ``[package]`` (``tools/grib1_bridge``) is a one-package
    workspace, is its own member, and additionally owns every path
    dependency vendored beneath it.
    """

    root = REPOSITORY_ROOT / workspace
    manifest = tomllib.loads(
        (root / "Cargo.toml").read_text(encoding="utf-8"))
    if "workspace" not in manifest:
        return {manifest["package"]["name"]} | _path_dependency_members(
            root, manifest)
    names = set()
    for member in manifest["workspace"].get("members", []):
        member_toml = root / member / "Cargo.toml"
        # A member cargo cannot open is a broken workspace, and it would
        # otherwise surface here as a bare FileNotFoundError during
        # collection -- an error that reads like this suite is broken
        # rather than like the tree is.
        assert member_toml.is_file(), (
            f"{workspace}/Cargo.toml lists member {member!r} but "
            f"{member}/Cargo.toml does not exist; cargo refuses the whole "
            "workspace in that state, so the cargo leg would fail on every "
            "entry")
        names.add(tomllib.loads(
            member_toml.read_text(encoding="utf-8"))["package"]["name"])
    return names


def _workspaces() -> list[str]:
    seen: list[str] = []
    for entry in _entries():
        if entry.workspace not in seen:
            seen.append(entry.workspace)
    return seen


def test_the_manifest_exists_and_lists_something() -> None:
    assert MANIFEST.is_file(), f"{MANIFEST} is missing"
    assert RUNNER.is_file(), f"{RUNNER} is missing"
    assert _entries(), (
        "tools/battery/cargo_gates.txt parsed to zero entries; the battery "
        "would compile no Rust and still report a clean leg")


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e.label)
def test_every_entry_names_a_real_workspace_and_package(entry) -> None:
    """Listed implies exists: the first direction of the sync.

    An entry naming a package cargo does not know fails the WHOLE leg at
    the first invocation, because ``cargo test -p <name>`` refuses
    rather than skipping.  Catching it here costs no toolchain.
    """

    root = REPOSITORY_ROOT / entry.workspace
    assert (root / "Cargo.toml").is_file(), (
        f"{entry.workspace}/Cargo.toml does not exist")
    assert (root / ".cargo" / "config.toml").is_file(), (
        f"{entry.workspace}/.cargo/config.toml does not exist, so "
        "`cargo test --offline` has no vendored source to resolve against; "
        "the leg would fail on a box with no crates.io cache and pass on "
        "one that had it")
    members = _members(entry.workspace)
    assert entry.package in members, (
        f"{entry.package} is listed in tools/battery/cargo_gates.txt but is "
        f"not a member of {entry.workspace}: {sorted(members)}")


@pytest.mark.parametrize(
    ("workspace", "package"),
    [(w, p) for w in _workspaces() for p in sorted(_members(w))],
    ids=lambda v: v.replace("/", "-"))
def test_every_workspace_member_is_listed(workspace: str,
                                          package: str) -> None:
    """Exists implies listed: the direction that makes this a gate.

    Without it the list is a snapshot -- correct on the day it was
    written and quietly incomplete afterwards.  A crate added to
    ``members`` is a crate the release ships and nothing compiles, which
    is exactly the state every crate in this tree was in before this
    leg existed.

    Adding the entry is the whole remedy: one line naming its shard, its
    floor and what it covers.  ``tools/battery/run_cargo_gates.py``
    prints the count to use as the floor.
    """

    listed = {(e.workspace, e.package) for e in _entries()}
    assert (workspace, package) in listed, (
        f"{package} is a member of {workspace} but has no entry in "
        "tools/battery/cargo_gates.txt, so the release battery never "
        "compiles or tests it.  Add a line -- SHARD WORKSPACE PACKAGE "
        "MIN_TESTS -- with its reason on the lines above, per the file's "
        "amendment discipline")


def test_no_invocation_is_listed_twice() -> None:
    """Identity is package PLUS cargo arguments.

    One package listed twice under different arguments is two
    invocations covering different targets, which is what the argument
    tail is for; the same package under the same arguments is a doubled
    row whose counts would overstate the coverage.
    """

    invocations = [(e.workspace, e.package, e.args) for e in _entries()]
    duplicates = sorted({i for i in invocations if invocations.count(i) > 1})
    assert not duplicates, (
        f"tools/battery/cargo_gates.txt lists {duplicates} more than once")


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e.label)
def test_every_entry_carries_its_reason_inline(entry) -> None:
    line = " ".join((entry.shard, entry.workspace, entry.package,
                     str(entry.min_tests)) + entry.env + entry.args)
    assert _reasons().get(line, ""), (
        f"{entry.label} has no comment block above it in "
        "tools/battery/cargo_gates.txt; the amendment discipline the file "
        "states is that an addition names what the package covers")


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e.label)
def test_every_floor_is_a_real_floor(entry) -> None:
    """``MIN_TESTS`` of zero would make the contract vacuous.

    ``cargo test`` on a package whose test modules stopped compiling
    under some ``#[cfg]`` exits 0 and reports ``0 passed``.  The floor is
    the only part of the contract that can tell those apart, so a floor
    that any run satisfies is the same as having none.  The parser
    refuses it; this is the statement of why.
    """

    assert entry.min_tests >= 1


def test_the_battery_shard_covers_both_workspaces() -> None:
    """A leg that compiled only the renderer would prove half of it.

    The two workspaces fail in opposite directions: tools/rustwx is
    everything model-output-to-pixels, tools/grib1_bridge is everything
    data-to-model-input.  A release ships binaries from both.
    """

    runner = _runner()
    shard = {e.workspace for e in _entries()
             if e.shard == runner.BATTERY_SHARD}
    for required in ("tools/rustwx", "tools/grib1_bridge"):
        assert required in shard, (
            f"no {runner.BATTERY_SHARD}-shard entry for {required} in "
            "tools/battery/cargo_gates.txt; that workspace ships in the "
            "release and the battery would not compile it")


def test_the_decoder_and_the_batch_renderer_are_both_on_the_battery_shard(
) -> None:
    """The two packages whose defects reach a user as wrong output.

    ``grib1_bridge`` decodes every GFS and HRRR field a run initialises
    from -- a quantization defect there is a wrong initial condition, not
    a crash.  ``rw-wrfbatch`` draws every product PNG a run publishes and
    is the one crate in tools/rustwx whose upstream is this project.
    Pinned by name so a shard reshuffle has to argue with this docstring
    rather than move a line.
    """

    runner = _runner()
    battery = {e.package for e in _entries()
               if e.shard == runner.BATTERY_SHARD}
    for required in ("grib1_bridge", "rw-wrfbatch"):
        assert required in battery, (
            f"{required} is not on the {runner.BATTERY_SHARD} shard of "
            "tools/battery/cargo_gates.txt")


def test_the_fixture_dependent_satellite_suite_is_still_listed() -> None:
    """Deferred coverage stays visible, or it is not deferred.

    ``rw-sat``'s ``tests/cloud_products.rs`` decodes real GOES-19
    granules the repository does not carry and PANICS by name when they
    are absent, so it cannot ride the offline leg.  The accurate form of
    that is two entries -- ``--lib`` on the battery shard and the
    integration target on ``fixtures`` -- and the inaccurate form is one
    ``--lib`` entry with nothing saying what was left out.  This test
    pins the accurate form: dropping the fixtures row is allowed, but it
    has to be an argument in a commit message, not a line that
    disappears while every gate stays green.
    """

    rows = {(e.shard, e.package, e.args) for e in _entries()}
    assert ("fixtures", "rw-sat", ("--test", "cloud_products")) in rows, (
        "the fixtures-shard row for rw-sat's cloud_products suite is gone "
        "from tools/battery/cargo_gates.txt.  The battery shard runs rw-sat "
        "with --lib, so without that row nothing in the tree records that "
        "the GOES L2 cloud-product decode has coverage at all")
    battery = {(e.package, e.args) for e in _entries() if e.shard == "cpu"}
    assert ("rw-sat", ("--lib",)) in battery, (
        "rw-sat's --lib row is gone from the battery shard, so the offline "
        "half of the satellite crate is no longer compiled by the leg")


def test_the_file_is_lf_only() -> None:
    """``.gitattributes`` says ``* -text``: disk bytes are committed bytes."""

    assert b"\r" not in MANIFEST.read_bytes(), (
        "tools/battery/cargo_gates.txt contains CR bytes; the repository "
        "commits LF and .gitattributes does no conversion")


def test_the_runner_refuses_an_in_tree_target_directory(monkeypatch) -> None:
    """The mandate, exercised rather than asserted in a comment.

    A cut hashes the worktree-local renderer at
    ``tools/rustwx/target/release`` before it believes any leg that
    renders, and refuses to assemble from a dirty source tree.  A test
    leg sharing that directory can replace the binary the cut is
    adjudicating.  Both refusals are checked here -- unset, and set
    inside the tree -- because a default would have hidden the mistake
    instead of stopping it.
    """

    runner = _runner()
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    with pytest.raises(RuntimeError, match="CARGO_TARGET_DIR is not set"):
        runner.resolve_target_dir(None)
    inside = REPOSITORY_ROOT / "tools" / "rustwx" / "target"
    with pytest.raises(RuntimeError, match="inside the repository"):
        runner.resolve_target_dir(str(inside))
    with pytest.raises(RuntimeError, match="inside the repository"):
        runner.resolve_target_dir(str(REPOSITORY_ROOT))


def test_the_runner_builds_offline_locked_invocations(monkeypatch) -> None:
    """``--locked --offline`` is the leg, not a preference.

    Both workspaces vendor their dependency closure with a
    ``.cargo/config.toml`` source replacement.  A leg that resolved from
    the network would pass on a developer box with a warm registry and
    fail on the release box, and would stop proving the vendored closure
    is complete -- which is the only reason a clean clone of this tree
    can build at all.
    """

    for entry in _entries():
        argv = entry.invocation()
        assert argv[:4] == ["cargo", "test", "--locked", "--offline"], argv
        assert argv[4:6] == ["-p", entry.package], argv
        assert argv[6:] == list(entry.args), argv


def test_an_external_target_directory_is_accepted(tmp_path) -> None:
    """The other direction of the refusal, so it is not refuse-everything."""

    runner = _runner()
    resolved = runner.resolve_target_dir(str(tmp_path / "cargo-target"))
    assert resolved == (tmp_path / "cargo-target").resolve()


def _directory_alias(link: Path, target: Path):
    """Owned fixture directories, using a junction where symlinks need privilege."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, capture_output=True, text=True)


@pytest.mark.parametrize("name", ["in_tree_link", "external_real_path", "external_alias", "external_child"])
def test_test_targets_cannot_share_release_artifacts_through_directory_aliases(tmp_path, monkeypatch, name):
    runner = _runner()
    source = tmp_path / "source"
    release_target = tmp_path / "release-target"
    release_target.mkdir()
    in_tree = source / "tools/rustwx/target"
    in_tree.parent.mkdir(parents=True)
    _directory_alias(in_tree, release_target)
    alias = tmp_path / "alias"
    _directory_alias(alias, release_target)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", source)
    monkeypatch.setattr(runner, "read_manifest", lambda: [
        runner.Entry("cpu", "tools/rustwx", "rw-wrfbatch", 1, (), ())])
    chosen = {"in_tree_link": in_tree, "external_real_path": release_target,
              "external_alias": alias, "external_child": alias / "test-child"}[name]
    expected = "inside the repository" if name == "in_tree_link" else "release artifact target"
    with pytest.raises(RuntimeError, match=expected):
        runner.resolve_target_dir(str(chosen))
    # A separate owned target is still admitted; this guard isolates artifacts,
    # it does not ban external caches merely because symlinks exist elsewhere.
    separate = tmp_path / "test-target"
    assert runner.resolve_target_dir(str(separate)) == separate.resolve()


def test_worker_gate_uses_the_battery_python_without_shell_expansion(monkeypatch, tmp_path):
    from types import SimpleNamespace
    if not (REPOSITORY_ROOT / "tools/arwen-tui").is_dir():
        pytest.skip("the terminal UI workspace (tools/arwen-tui), the only gate row whose environment names "
                    "@python, is not in this tree; its row leaves the gate list with it")
    runner = _runner()
    entry = next(e for e in _entries() if e.package == "arwen-tui")
    seen = {}
    def invoke(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0, stdout=(
            "test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;\n"))
    monkeypatch.setattr(runner.subprocess, "run", invoke)
    python = str(tmp_path / "engine environment" / "python.exe")
    monkeypatch.setattr(runner.sys, "executable", python)
    result = runner.run_entry(entry, target_dir=tmp_path / "target", cargo="cargo")
    assert seen["env"]["WOOF_TUI_TEST_PYTHON"] == python
    assert not seen.get("shell", False)
    assert seen["cwd"] == REPOSITORY_ROOT / entry.workspace
    assert result.verdict == "PASSED"


def test_a_failed_entry_names_its_failing_tests_and_keeps_the_full_output(
        monkeypatch, tmp_path):
    """A failed gate that cannot name its failing tests is a defect.

    THE BREAKAGE: the 2.9.3 R8 Windows gate reported rw-wrfbatch as
    "2 targets failed: --test chem_products, --test snowfall" and nothing
    else.  Cargo prints each test binary's output in turn and the runner
    kept only the last 40 lines, so the failing tests' names and panics,
    printed under earlier binaries, were gone; the same targets then
    passed alone on the same tree and nobody could say what had failed.
    So every entry's whole output is written to its own log, and the
    failing tests are named from the whole output, with their panic
    site, however many binaries printed after them.
    """

    from types import SimpleNamespace
    runner = _runner()
    entry = runner.Entry("cpu", "tools/rustwx", "rw-wrfbatch", 1, (), ())
    output = (
        "     Running tests/chem_products.rs (target/debug/deps/chem_products-1)\n"
        "running 2 tests\n"
        "test chem_presence_is_checked_per_source ... ok\n"
        "test synthetic_chem_wrfout_renders_all_six_products ... FAILED\n"
        "\nfailures:\n\n---- synthetic_chem_wrfout_renders_all_six_products stdout ----\n"
        "thread 'synthetic_chem_wrfout_renders_all_six_products' (4242) panicked at "
        "crates/rw-wrfbatch/tests/chem_products.rs:136:5:\n"
        "RENDERED aq_pm25 missing\n"
        "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
        "\ntest result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.23s\n"
        + "\n".join(f"     Running tests/other_{i}.rs\nrunning 1 test\ntest other_{i} ... ok\n"
                     "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
                     for i in range(20))
        + "error: 1 target failed:\n    `-p rw-wrfbatch --test chem_products`\n")
    assert len(output.splitlines()) > 40
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(
        returncode=101, stdout=output))
    log_dir = tmp_path / "gate-logs"
    outcome = runner.run_entry(entry, target_dir=tmp_path / "target", cargo="cargo",
                               log_dir=log_dir)
    assert outcome.verdict == "FAILED"
    assert (outcome.passed, outcome.failed, outcome.binaries) == (21, 1, 21)
    assert outcome.log_path == log_dir / "rw-wrfbatch.log"
    kept = outcome.log_path.read_text(encoding="utf-8")
    assert kept.startswith("$ cargo test --locked --offline -p rw-wrfbatch\n")
    assert output in kept
    named = runner.failing_tests(outcome.output)
    assert named == [
        "synthetic_chem_wrfout_renders_all_six_products FAILED  "
        "(crates/rw-wrfbatch/tests/chem_products.rs:136:5: RENDERED aq_pm25 missing)"]
    # The failing test is not in the 40-line tail the report also prints;
    # only cargo's closing "1 target failed" line is, and it names targets.
    tail = output.strip().splitlines()[-40:]
    assert not any("synthetic_chem_wrfout_renders_all_six_products" in line for line in tail)
    assert tail[-1] == "    `-p rw-wrfbatch --test chem_products`"


def test_the_runner_names_unit_test_failures_by_their_module_path() -> None:
    """A unit test's panic thread is its full path; the name line is too."""

    runner = _runner()
    output = (
        "test tests::coordinate_variables_are_recovered ... FAILED\n"
        "thread 'tests::coordinate_variables_are_recovered' (1618375) panicked at "
        "crates/rw-netcdf/src/main.rs:2502:9:\nassertion `left == right` failed\n"
        "  left: [\"time\"]\n right: [\"t2\"]\n"
        "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n")
    assert runner.failing_tests(output) == [
        "tests::coordinate_variables_are_recovered FAILED  "
        "(crates/rw-netcdf/src/main.rs:2502:9: assertion `left == right` failed "
        "left: [\"time\"] right: [\"t2\"])"]
    assert runner.failing_tests("test result: ok. 3 passed; 0 failed; 0 ignored\n") == []


# ---------------------------------------------------------------------------
# No exported cargo-gate target links the system netCDF C library
# ---------------------------------------------------------------------------
# THE BREAKAGE: ci.yml's native job runs these entries on GitHub's
# ubuntu-24.04 and windows-2025 runners, which carry no libnetcdf, and
# `cargo test -p <package>` builds every example and test target of the
# package. 2.9.0's first public ci (run 38684507300) failed both native jobs
# at link time ("unable to find library -lnetcdf"; LNK1181 netcdf.lib) on
# three rw-mpas proof examples declaring #[link(name = "netcdf")]. The release
# hosts have libnetcdf, so the battery and the nine-route rehearsal passed.
# A target the export carries loads netCDF at run time (as rw_mpas_static's
# verify does through libloading) or stays out of the export
# (RELEASE-EXCLUDE.txt).

import re as _re

_NETCDF_LINK = _re.compile(r'#\s*\[\s*link\s*\(\s*name\s*=\s*"netcdf"')
_NETCDF_LINK_LIB = _re.compile(r'rustc-link-lib\s*=\s*(?:dylib=|static=)?netcdf\b')


def _package_directories() -> dict[str, Path]:
    """Package name -> directory, for every package a cargo-gate workspace declares."""

    found: dict[str, Path] = {}
    for workspace in _workspaces():
        root = REPOSITORY_ROOT / workspace
        manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
        candidates = [root / m for m in manifest.get("workspace", {}).get("members", [])]
        if "workspace" not in manifest:
            candidates = [root]
            for table in ("dependencies", "dev-dependencies", "build-dependencies"):
                for spec in manifest.get(table, {}).values():
                    if isinstance(spec, dict) and "path" in spec:
                        # Lexical, never resolved: the CI routes link tools/<workspace> to a key-stable native source
                        # cache outside the checkout, and a resolved path then left REPOSITORY_ROOT (2.9.4 pass
                        # 4173d549 R7b route B: ValueError 'not in the subpath of /work/src').
                        candidates.append(Path(os.path.normpath(root / spec["path"])))
        for directory in candidates:
            toml = directory / "Cargo.toml"
            if toml.is_file():
                name = tomllib.loads(toml.read_text(encoding="utf-8"))["package"]["name"]
                found.setdefault(name, directory)
    return found


def _exported_rust_files(directory: Path, root: Path = REPOSITORY_ROOT) -> list[str]:
    from tools.release_exclusions import matches, read_exclusions

    rules = read_exclusions(root)
    listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--",
                              directory.relative_to(root).as_posix()],
                             capture_output=True)
    if listing.returncode == 0 and listing.stdout:
        names = [n for n in listing.stdout.decode("utf-8").split("\0") if n]
    else:
        names = [p.relative_to(root).as_posix() for p in directory.rglob("*")]
    return [n for n in names if n.endswith(".rs") and "/target/" not in n
            and "/vendor/" not in n and matches(n, rules) is None]


def _netcdf_link_sites(root: Path, names: list[str]) -> list[str]:
    sites = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            text = line.strip()
            if text.startswith("//"):
                continue
            if _NETCDF_LINK.search(text) or (path.name == "build.rs" and _NETCDF_LINK_LIB.search(text)):
                sites.append(f"{name}:{number}")
    return sites


def test_no_exported_cargo_gate_target_links_the_system_netcdf_library() -> None:
    packages = _package_directories()
    gated = {entry.package for entry in _entries()}
    offenders = []
    for package in sorted(gated):
        directory = packages.get(package)
        if directory is None:
            continue  # test_every_entry_names_a_real_workspace_and_package owns that refusal
        offenders += _netcdf_link_sites(REPOSITORY_ROOT, _exported_rust_files(directory))
    assert not offenders, (
        "these exported targets link the system netCDF C library at build time, which the "
        "public ci runners do not have (`cargo test -p` builds every example and test): "
        + ", ".join(offenders) + ". Load it at run time, or add the file to RELEASE-EXCLUDE.txt.")


def test_the_netcdf_link_scan_sees_a_directive_and_not_a_comment(tmp_path) -> None:
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "linked.rs").write_text('#[link(name="netcdf")]\nunsafe extern "C" {}\n')
    (tmp_path / "examples" / "doc.rs").write_text('//! Linked at build time (`#[link(name = "netcdf")]`)\n')
    (tmp_path / "build.rs").write_text('println!("cargo:rustc-link-lib=netcdf");\n')
    found = _netcdf_link_sites(tmp_path, ["examples/linked.rs", "examples/doc.rs", "build.rs"])
    assert found == ["examples/linked.rs:1", "build.rs:1"]


# ---------------------------------------------------------------------------
# Unix-only and Windows-only std APIs sit under a matching cfg
# ---------------------------------------------------------------------------
# THE BREAKAGE: ci.yml's native job builds every test target of each cargo-gate
# package on windows-2025 as well as ubuntu-24.04. 2.9.0's public ci on P8 (run
# 38718383587) failed the Windows rw-mpas gate before any test ran, because
# tests/verify_mpas_cli.rs called std::os::unix::fs::symlink with no cfg; the
# release hosts and the rehearsal build on Linux only, so nothing local saw it.
# A use of std::os::unix (or std::os::windows) must sit in a file, item or block
# whose cfg names that platform family.

_OS_API = _re.compile(r"\bstd::os::(unix|windows)\b|\bos::(unix|windows)::")
_CFG_ATTR = _re.compile(r"#\s*(!?)\s*\[\s*cfg\s*\((.*)\)\s*\]")


def _cfg_family(condition: str) -> str | None:
    """The platform family a cfg condition restricts to, or None."""

    text = condition.replace(" ", "")
    if "not(windows)" in text or "not(target_os=\"windows\")" in text or "unix" in text \
            or "target_os=\"linux\"" in text or "target_os=\"macos\"" in text:
        return "unix"
    if "windows" in text:
        return "windows"
    return None


_MOD_DECL = _re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")


def _module_families(root: Path, names: list[str]) -> dict[str, str]:
    """Files compiled only under a platform cfg because their `mod` declaration carries it."""

    families: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        pending = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            code = line.split("//", 1)[0]
            attribute = _CFG_ATTR.search(code)
            if attribute and attribute.group(1) != "!":
                pending = _cfg_family(attribute.group(2))
                continue
            declared = _MOD_DECL.match(code)
            if declared and pending:
                # A crate root (lib.rs, main.rs, or any [[bin]]/[[test]] path) keeps its
                # modules beside it; any other file keeps them in a folder of its name.
                for base in (path.parent, path.parent / path.stem):
                    for candidate in (base / f"{declared.group(1)}.rs", base / declared.group(1) / "mod.rs"):
                        families[candidate.relative_to(root).as_posix()] = pending
            if code.strip():
                pending = None
    return families


def _unguarded_os_api_sites(root: Path, names: list[str]) -> list[str]:
    sites = []
    gated_files = _module_families(root, names)
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        file_family = gated_files.get(name)
        stack: list[str | None] = []
        pending: str | None = None
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            code = line.split("//", 1)[0]
            attribute = _CFG_ATTR.search(code)
            if attribute:
                family = _cfg_family(attribute.group(2))
                if attribute.group(1) == "!":
                    file_family = family or file_family
                else:
                    pending = family
                code = code[:attribute.start()] + code[attribute.end():]
            use = _OS_API.search(code)
            if use:
                wanted = use.group(1) or use.group(2)
                guarded = wanted in (file_family, pending) or wanted in stack
                if not guarded:
                    sites.append(f"{name}:{number}")
            for character in code:
                if character == "{":
                    stack.append(pending)
                    pending = None
                elif character == "}" and stack:
                    stack.pop()
            if code.strip().endswith(";") and pending is not None and "{" not in code:
                pending = None
    return sites


def test_no_exported_cargo_gate_target_uses_a_platform_api_outside_its_cfg() -> None:
    packages = _package_directories()
    gated = {entry.package for entry in _entries()}
    offenders = []
    for package in sorted(gated):
        directory = packages.get(package)
        if directory is None:
            continue
        offenders += _unguarded_os_api_sites(REPOSITORY_ROOT, _exported_rust_files(directory))
    assert not offenders, (
        "these exported targets use std::os::unix or std::os::windows outside a cfg for that "
        "platform, so the other platform's public ci cannot build them (`cargo test -p` builds "
        "every test target): " + ", ".join(offenders))


def test_the_platform_api_scan_reads_files_items_blocks_and_statements(tmp_path) -> None:
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "bare.rs").write_text(
        "fn f() {\n    std::os::unix::fs::symlink(a, b).unwrap();\n}\n")
    (tmp_path / "t" / "file.rs").write_text(
        "//! doc\n#![cfg(unix)]\nfn f() { std::os::unix::fs::symlink(a, b).unwrap(); }\n")
    (tmp_path / "t" / "statement.rs").write_text(
        "fn f() {\n    #[cfg(unix)]\n    std::os::unix::fs::symlink(a, b).unwrap();\n"
        "    #[cfg(not(unix))]\n    std::fs::copy(a, b).unwrap();\n}\n")
    (tmp_path / "t" / "block.rs").write_text(
        "fn f() {\n    #[cfg(unix)]\n    {\n        use std::os::unix::fs::PermissionsExt;\n"
        "        let x = 1;\n    }\n}\n")
    (tmp_path / "t" / "item.rs").write_text(
        "#[cfg(windows)]\nfn f() {\n    if x {\n        use std::os::windows::process::CommandExt;\n    }\n}\n")
    (tmp_path / "t" / "wrong.rs").write_text(
        "#[cfg(windows)]\nfn f() {\n    use std::os::unix::fs::MetadataExt;\n}\n")
    (tmp_path / "t" / "after.rs").write_text(
        "#[cfg(unix)]\nuse std::os::unix::fs::MetadataExt;\nfn g() { std::os::unix::fs::symlink(a, b); }\n")
    (tmp_path / "t" / "main.rs").write_text("#[cfg(windows)]\nmod launcher;\nmod plain;\nfn main() {}\n")
    (tmp_path / "t" / "launcher.rs").write_text("use std::os::windows::ffi::OsStrExt;\n")
    (tmp_path / "t" / "plain.rs").write_text("use std::os::windows::ffi::OsStrExt;\n")
    names = [f"t/{n}.rs" for n in ("bare", "file", "statement", "block", "item", "wrong", "after",
                                   "main", "launcher", "plain")]
    assert _unguarded_os_api_sites(tmp_path, names) == [
        "t/bare.rs:2", "t/wrong.rs:3", "t/after.rs:3", "t/plain.rs:1"]


@pytest.mark.skipif(os.name != 'posix', reason='packages of one workspace overlap on POSIX only (per-row '
                    'clones); off POSIX they run one at a time')
def test_package_gates_overlap_without_losing_contracts_or_manifest_order(tmp_path, monkeypatch):
    """The old package loop cannot reach this barrier with idle host cores."""
    import threading
    runner = _runner()
    entries = [runner.Entry("cpu", "workspace", name, 1, (), ())
               for name in ("first", "red", "under-ran")]
    from types import SimpleNamespace
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="host: test-host\n"))
    barrier = threading.Barrier(3, timeout=5)
    def run(entry, **kwargs):
        barrier.wait()
        return runner.Outcome(entry, int(entry.package != "under-ran"),
                              int(entry.package == "red"), 0, 1,
                              int(entry.package == "red"), 0, entry.package)
    monkeypatch.setattr(runner, "run_entry", run)
    outcomes = list(runner.run_entries(entries, target_dir=tmp_path, cargo="cargo",
                                       log_dir=tmp_path, workers=3))
    assert [o.entry for o in outcomes] == entries
    assert [o.verdict for o in outcomes] == ["PASSED", "FAILED", "UNDER-RAN"]


@pytest.mark.skipif(os.name != "posix", reason="executable fake Cargo uses a POSIX shebang")
def test_cargo_selected_integration_binaries_overlap_and_keep_failure(tmp_path, monkeypatch):
    """Cargo normally waits for each single-core integration binary in turn."""
    runner = _runner()
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    (workspace / '.cargo').mkdir()
    (workspace / '.cargo/config.toml').write_text('')
    (workspace / 'Cargo.toml').write_text('')
    program = tmp_path / 'binary.py'
    program.write_text("""import pathlib, sys, time
root=pathlib.Path(sys.argv[1]); index=int(sys.argv[2])
(root / str(index)).write_text('ready')
end=time.monotonic()+5
while not (root / str(1-index)).exists():
    assert time.monotonic()<end, 'integration binaries serialized'
    time.sleep(.01)
print('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;')
raise SystemExit(7 if index else 0)
""")
    cargo = tmp_path / 'cargo'
    cargo.write_text(f"""#!{sys.executable}
import pathlib, subprocess, sys, tomllib
if sys.argv[1:] == ['-vV']:
    print('host: fake-host'); raise SystemExit(0)
assert sys.argv[1] == 'test', 'release cargo wrappers dispatch on the test subcommand'
command=tomllib.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())['target']['fake-host']['runner']
for index in range(2):
    subprocess.run(command+[{sys.executable!r}, {str(program)!r}, {str(tmp_path)!r}, str(index)], check=True)
""")
    cargo.chmod(0o755)
    monkeypatch.setattr(runner, 'REPOSITORY_ROOT', tmp_path)
    entry = runner.Entry('cpu', 'workspace', 'test-package', 2, (), ())
    outcome, = runner.run_entries([entry], target_dir=tmp_path/'target',
        cargo=str(cargo), log_dir=tmp_path/'logs', workers=2)
    assert outcome.passed == 2
    assert outcome.returncode == 7
    assert outcome.verdict == 'FAILED'


def test_test_runner_connection_burst_does_not_lose_clients(tmp_path):
    """A backlog of one dropped a runner handshake and hung the release gate."""
    code = r"""
import importlib.util, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client
spec=importlib.util.spec_from_file_location('gate_burst',sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class BurstPool(m.BinaryPool):
    def serve(self):
        time.sleep(.2)
        super().serve()
pool=BurstPool('test-host',24)
barrier=threading.Barrier(24)
def submit(index):
    barrier.wait()
    with Client(pool.listener.address,authkey=pool.key) as client:
        client.send((str(index),[sys.executable,'-c','pass'],os.getcwd(),dict(os.environ)))
        assert client.recv() is True
with ThreadPoolExecutor(max_workers=24) as clients:
    list(clients.map(submit, range(24)))
for index in range(24):
    assert pool.collect(str(index))[0][1].returncode == 0
pool.close()
"""
    completed = subprocess.run([sys.executable,'-c',code,str(RUNNER)],cwd=tmp_path,
                               capture_output=True,text=True,timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_gpu_shard_does_not_overlap_a_single_admitted_card(tmp_path, monkeypatch):
    """The CPU optimization must not submit two GPU entries to one card."""
    import threading, time
    runner = _runner()
    lease = threading.Lock()
    def run(entry, **kwargs):
        assert lease.acquire(blocking=False), 'concurrent GPU entry exceeded its card lease'
        try:
            time.sleep(.01)
            return runner.Outcome(entry, 1, 0, 0, 1, 0, 0, '')
        finally:
            lease.release()
    monkeypatch.setattr(runner, 'run_entry', run)
    entries=[runner.Entry('gpu','workspace',name,1,(),()) for name in ('first','second')]
    outcomes=list(runner.run_entries(entries,target_dir=tmp_path,cargo='unused',log_dir=tmp_path,workers=2))
    assert all(o.verdict=='PASSED' for o in outcomes)


def test_windows_runner_toml_roundtrips_paths_spaces_quotes_and_long_label(tmp_path):
    runner = _runner()
    command = [r'C:\Program Files\Python "test"\python.exe',
               r'C:\rar295\checkout with spaces\tools\battery\run_cargo_gates.py',
               '--gate-test-runner', '127.0.0.1', '65433',
               'static-fields --lib --test geotiff_envelope --test geotiff_malformed --test highres_parity --test seam_probe "label"']
    document = runner.runner_toml('x86_64-pc-windows-msvc', command)
    assert tomllib.loads(document)['target']['x86_64-pc-windows-msvc']['runner'] == command
    pool = runner.BinaryPool('x86_64-pc-windows-msvc', 1)
    try:
        flag, path = pool.configuration(command[-1])
        assert flag == '--config'
        parsed = tomllib.loads(Path(path).read_text())
        assert parsed['target']['x86_64-pc-windows-msvc']['runner'][-1] == command[-1]
    finally:
        pool.close()


@pytest.mark.skipif(os.name != 'posix', reason='packages of one workspace overlap on POSIX only (per-row '
                    'clones); off POSIX they run one at a time')
def test_two_rows_of_one_package_never_overlap_but_other_packages_do(tmp_path, monkeypatch):
    import threading, time
    from types import SimpleNamespace
    runner = _runner()
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='host: test-host\n'))
    entries = [runner.Entry('cpu', 'workspace', 'rw-mpas', 1, (), ()),
               runner.Entry('cpu', 'workspace', 'rw-mpas', 1, (), ('--test','radar_history_contract')),
               runner.Entry('cpu', 'workspace', 'other-package', 1, (), ())]
    barrier = threading.Barrier(2, timeout=5)
    active = set();guard = threading.Lock();order=[]
    def run(entry, **kwargs):
        with guard:
            assert entry.package not in active, 'overlapping package rows race on test fixture files'
            active.add(entry.package);order.append(entry)
        try:
            if not entry.args:barrier.wait()
            time.sleep(.02)  # includes completion of the row's queued test binaries
            return runner.Outcome(entry, 1, 0, 0, 1, 0, 0, '')
        finally:
            with guard:active.remove(entry.package)
    monkeypatch.setattr(runner, 'run_entry', run)
    results=list(runner.run_entries(entries,target_dir=tmp_path,cargo='cargo',log_dir=tmp_path,workers=3))
    assert [o.entry for o in results] == entries
    assert order.index(entries[0]) < order.index(entries[1])


def test_overlapping_rows_of_different_workspaces_never_share_a_target_folder(tmp_path, monkeypatch):
    """2.9.4 pass 4b R8 (2026-10-21 02:38Z): one CARGO_TARGET_DIR for every workspace let rw-zarr and
    mapped-engine rebuild deps/librayon-<hash>.rlib while grib1_bridge's doctests compiled against it
    ("can't find crate for rayon"). Overlapping rows get one folder per workspace; serial shards keep one."""
    from types import SimpleNamespace
    runner = _runner()
    seen = []
    def invoke(argv, **kwargs):
        seen.append((kwargs['cwd'], kwargs['env']['CARGO_TARGET_DIR']))
        return SimpleNamespace(returncode=0, stdout='test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;\n')
    monkeypatch.setattr(runner.subprocess, 'run', invoke)
    entries = [e for e in _entries() if e.shard == runner.BATTERY_SHARD]
    pool = SimpleNamespace(key=b'k', configuration=lambda label: [], collect=lambda label: [])
    for entry in entries:
        runner.run_entry(entry, target_dir=tmp_path / 'target', cargo='cargo', binary_pool=pool)
    folders = {}
    for cwd, folder in seen:
        folders.setdefault(folder, set()).add(cwd)
        assert Path(folder).parent == tmp_path / 'target' / 'ws'
    assert all(len(cwds) == 1 for cwds in folders.values()), 'two workspaces share one overlapping target folder'
    assert len(folders) == len({cwd for cwd, _ in seen}) > 1
    seen.clear()
    runner.run_entry(entries[0], target_dir=tmp_path / 'target', cargo='cargo')
    assert seen == [(REPOSITORY_ROOT / entries[0].workspace, str(tmp_path / 'target'))]


@pytest.mark.skipif(os.name != 'posix', reason='per-row clones are POSIX only (cp -al); Windows runs a '
                    "workspace's rows one at a time")
def test_parallel_rows_of_one_workspace_each_run_on_a_private_link_clone(tmp_path, monkeypatch):
    """Many cores (test host, 64; development branch 2026-10-21): rows of ONE workspace with different package selections
    rebuilt each other's units in the shared workspace folder under a running rustdoc (rw-wrfbatch: "can't find
    crate for rw_nexrad"). Every row compiles first, then runs on its own hard-link clone (POSIX and Windows), so
    a row that replaces a unit never changes the file another row reads."""
    runner = _runner()
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    (workspace / '.cargo').mkdir()
    (workspace / '.cargo/config.toml').write_text('')
    (workspace / 'Cargo.toml').write_text('')
    seen = {}
    lock = __import__('threading').Lock()
    barrier = __import__('threading').Barrier(2, timeout=10)

    def cargo_run(argv, cwd=None, env=None, **kwargs):
        from types import SimpleNamespace
        target = Path(env['CARGO_TARGET_DIR'])
        unit = target / 'debug' / 'deps' / 'libshared.rlib'
        if '--no-run' in argv:
            unit.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                if not unit.exists():
                    unit.write_text('built once')
            return SimpleNamespace(returncode=0, stdout='')
        package = argv[argv.index('-p') + 1]
        barrier.wait()  # both rows run at the same time
        before = unit.read_text()
        if package == 'b':  # this row's selection rebuilds the shared unit: rustc replaces the file
            unit.unlink()
            unit.write_text('rebuilt by b')
        barrier.wait()
        with lock:
            seen[package] = (str(target), before, unit.read_text())
        return SimpleNamespace(returncode=0, stdout='test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;\n')

    monkeypatch.setattr(runner, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(runner, 'BinaryPool', lambda host, workers: __import__('types').SimpleNamespace(
        key=b'k', configuration=lambda label: [], collect=lambda label: [], close=lambda: None))
    real_run = runner.subprocess.run

    def dispatch(argv, **kw):
        # Only cargo is faked: the POSIX clone runs the real `cp -al` (the first version of this test handed it
        # to the cargo fake, which read env=None: TypeError on Linux, 2.9.4 folded pass R5 on a Linux host).
        if argv[0] != 'cargo':
            return real_run(argv, **kw)
        if argv[1:] == ['-vV']:
            return __import__('types').SimpleNamespace(returncode=0, stdout='host: fake-host\n')
        return cargo_run(argv, **kw)

    monkeypatch.setattr(runner.subprocess, 'run', dispatch)
    entries = [runner.Entry('cpu', 'workspace', package, 1, (), ()) for package in ('a', 'b')]
    outcomes = list(runner.run_entries(entries, target_dir=tmp_path / 'target', cargo='cargo',
                                       log_dir=tmp_path / 'logs', workers=2))
    assert [o.verdict for o in outcomes] == ['PASSED', 'PASSED']
    shared = runner.workspace_target_dir(tmp_path / 'target', entries[0])
    assert seen['a'][0] != seen['b'][0] and str(shared) not in (seen['a'][0], seen['b'][0])
    assert seen['a'][1:] == ('built once', 'built once'), "a's unit changed under it"
    assert seen['b'][1:] == ('built once', 'rebuilt by b')
    assert (shared / 'debug' / 'deps' / 'libshared.rlib').read_text() == 'built once'
    assert not list(shared.parent.glob('*--row--*')), 'row clones are removed after their run'


def test_each_workspace_is_built_once_before_its_rows_fingerprint_and_run(tmp_path, monkeypatch):
    """2.9.4 R8 on the desktop (2026-10-21): 59 rows each ran their own `cargo test --no-run` at once on one
    workspace folder with CARGO_BUILD_JOBS=4; cargo's build-directory lock serialised them into one 4-job build
    ("Blocking waiting for file lock on build directory" in every row) and the leg took 2,016-2,130 s with a warm
    target.  The CPU shard now builds each workspace ONCE at the prebuild jobs (a short burst, below normal priority
    on Windows), then fingerprints every row's exact selection, and only then runs the rows."""
    from types import SimpleNamespace
    runner = _runner()
    for workspace in ('ws-a', 'ws-b'):
        (tmp_path / workspace / '.cargo').mkdir(parents=True)
        (tmp_path / workspace / '.cargo/config.toml').write_text('')
        (tmp_path / workspace / 'Cargo.toml').write_text('')
    calls = []

    def cargo_run(argv, cwd=None, env=None, **kwargs):
        if argv[1:] == ['-vV']:
            return SimpleNamespace(returncode=0, stdout='host: fake-host\n')
        calls.append((list(argv), Path(cwd).name, env['CARGO_BUILD_JOBS'], env['CARGO_TARGET_DIR'],
                      kwargs.get('creationflags', 0)))
        return SimpleNamespace(returncode=0, stdout='   Compiling x v0.1.0\ntest result: ok. 1 passed; 0 failed; '
                                                   '0 ignored; 0 measured;\n')

    monkeypatch.setenv('CARGO_BUILD_JOBS', '4')
    monkeypatch.delenv('ARWEN_CARGO_PREBUILD_JOBS', raising=False)
    monkeypatch.setattr(runner, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(runner, 'BinaryPool', lambda host, workers: SimpleNamespace(
        key=b'k', configuration=lambda label: [], collect=lambda label: [], close=lambda: None))
    monkeypatch.setattr(runner.subprocess, 'run', cargo_run)
    entries = [runner.Entry('cpu', 'ws-a', 'a', 1, (), ()),
               runner.Entry('cpu', 'ws-a', 'b', 1, (), ('--lib',)),
               runner.Entry('cpu', 'ws-a', 'b', 1, (), ('--test', 'goldens', '--', 'one_case', '--exact')),
               runner.Entry('cpu', 'ws-b', 'c', 1, ('KEY=value',), ())]
    outcomes = list(runner.run_entries(entries, target_dir=tmp_path / 'target', cargo='cargo',
                                       log_dir=tmp_path / 'logs', workers=4, prebuild_jobs=7))
    assert [o.verdict for o in outcomes] == ['PASSED'] * 4
    builds = [c for c in calls if '--no-run' in c[0]]
    runs = [c for c in calls if '--no-run' not in c[0]]
    assert calls[:len(builds)] == builds, 'a row ran before every workspace was built and every row fingerprinted'
    assert [(c[0][1:], c[1]) for c in builds] == [
        (['test', '--no-run', '--locked', '--offline', '-p', 'a'], 'ws-a'),               # ws-a built once...
        (['test', '--no-run', '--locked', '--offline', '-p', 'b', '--lib', '--test', 'goldens'], 'ws-a'),
        (['test', '--no-run', '--locked', '--offline', '-p', 'a'], 'ws-a'),               # ...then each row's own
        (['test', '--no-run', '--locked', '--offline', '-p', 'b', '--lib'], 'ws-a'),      # selection fingerprints
        (['test', '--no-run', '--locked', '--offline', '-p', 'b', '--test', 'goldens'], 'ws-a'),
        (['test', '--no-run', '--locked', '--offline', '-p', 'c'], 'ws-b'),
        (['test', '--no-run', '--locked', '--offline', '-p', 'c'], 'ws-b')]
    assert {c[2] for c in builds} == {'7'}, 'the one-time build runs at the prebuild jobs, not the rows\' 4'
    assert {c[2] for c in runs} == {'4'} and len(runs) == 4
    assert all(c[3] == str(runner.workspace_target_dir(tmp_path / 'target', entries[0])) for c in builds[:5])
    if sys.platform == 'win32':
        assert {c[4] for c in builds} == {subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    assert sorted(p.name for p in (tmp_path / 'logs' / 'prebuild').iterdir()) == ['ws-a.log', 'ws-b.log']
    assert (tmp_path / 'logs' / 'prebuild' / 'ws-a.log').read_text().count('$ cargo test --no-run') == 5


def test_the_manifest_prebuilds_as_one_whole_call_per_workspace_plus_target_unions() -> None:
    runner = _runner()
    calls = runner.prebuild_calls([e for e in _entries() if e.shard == runner.BATTERY_SHARD])
    by_workspace: dict[str, list] = {}
    for workspace, arguments, env in calls:
        by_workspace.setdefault(workspace, []).append((arguments, env))
    assert list(by_workspace) == _workspaces()[:len(by_workspace)]
    rustwx = by_workspace['tools/rustwx']
    whole = rustwx[0][0]
    assert whole[:4] == ['test', '--no-run', '--locked', '--offline'] and '--lib' not in whole and '--test' not in whole
    assert whole.count('-p') >= 30 and 'rw-mpas' in whole and 'rw-sat' not in whole and 'static-fields' not in whole
    assert (['test', '--no-run', '--locked', '--offline', '-p', 'rw-sat', '--lib'], ()) in rustwx
    union = next(a for a, _ in rustwx if a[5] == 'static-fields')
    assert union[6:8] == ['--lib', '--test'] and union.count('--test') == 6 and '--' not in union
    assert len(rustwx) == 3
    if (REPOSITORY_ROOT / 'tools/arwen-tui').is_dir():
        assert by_workspace['tools/arwen-tui'] == [(['test', '--no-run', '--locked', '--offline', '-p', 'arwen-tui'],
                                                    ('WOOF_TUI_TEST_PYTHON=@python',))]
    else:
        # a tree without the terminal UI workspace carries no gate row for it, so nothing prebuilds it
        assert 'tools/arwen-tui' not in by_workspace


def test_prebuild_jobs_come_from_the_driver_before_the_rows_jobs(monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setenv('CARGO_BUILD_JOBS', '4')
    monkeypatch.setenv('ARWEN_CARGO_PREBUILD_JOBS', '28')
    assert runner.prebuild_jobs_default() == 28
    monkeypatch.delenv('ARWEN_CARGO_PREBUILD_JOBS')
    assert runner.prebuild_jobs_default() == 4
    monkeypatch.delenv('CARGO_BUILD_JOBS')
    assert runner.prebuild_jobs_default() == runner.available_cores()
def test_a_doctest_runs_in_place_and_a_binary_that_cannot_start_fails_only_its_row(tmp_path, monkeypatch):
    """2.9.4 fold pass R7b route A (2026-10-21): the runner hook queued rustdoc's doctest binary to the pool and
    returned; rustdoc deleted its temp folder, the pool met FileNotFoundError and the whole gate run ended."""
    import subprocess as sp
    runner = _runner()
    doctest = tmp_path / 'rustdoctest7SDHLp' / 'rust_out'
    doctest.parent.mkdir()
    assert runner.is_doctest([str(doctest)]) and not runner.is_doctest([str(tmp_path / 'deps' / 'x-123')])
    seen = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda argv, **kw: seen.append(argv) or sp.CompletedProcess(argv, 3))
    monkeypatch.delenv('ARWEN_CARGO_GATE_KEY', raising=False)  # no pool connection is made for a doctest
    assert runner.test_runner(['127.0.0.1', '1', 'row', str(doctest), '--test-threads=1']) == 3
    assert seen == [[str(doctest), '--test-threads=1']]
    from concurrent.futures import Future
    gone = Future()
    gone.set_exception(FileNotFoundError(2, 'No such file or directory', str(doctest)))
    pool = runner.BinaryPool.__new__(runner.BinaryPool)
    pool.pending = {'row': [([str(doctest)], gone)]}
    (argv, result), = pool.collect('row')
    assert result.returncode == 127 and 'could not run' in result.stdout


def test_off_posix_rows_of_one_workspace_run_one_at_a_time_without_clones(tmp_path, monkeypatch):
    """2.9.4 pass ec5dbbdb R8 (desktop, 2026-10-21): on NTFS hard-link clones rw-wrfbatch's lib failed E0308
    'multiple different versions of crate serde_json' beside the fire-indices row (float_roundtrip)."""
    import threading
    runner = _runner()
    monkeypatch.setattr(runner, 'POSIX', False)  # not os.name: on Linux that breaks every pathlib.Path()
    workspace = tmp_path / 'workspace'
    (workspace / '.cargo').mkdir(parents=True)
    (workspace / '.cargo/config.toml').write_text('')
    (workspace / 'Cargo.toml').write_text('')
    (tmp_path / 'target' / 'ws' / 'workspace').mkdir(parents=True)
    live, peak, targets, lock = [0], [0], [], threading.Lock()

    def fake_run(argv, **kw):
        if argv[1:] == ['-vV']:
            return __import__('types').SimpleNamespace(returncode=0, stdout='host: fake-host\n')
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            targets.append((kw.get('env') or {}).get('CARGO_TARGET_DIR'))
        __import__('time').sleep(0.05)
        with lock:
            live[0] -= 1
        return __import__('types').SimpleNamespace(returncode=0, stdout='test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;\n')

    monkeypatch.setattr(runner, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    monkeypatch.setattr(runner, 'prebuild_workspaces', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'BinaryPool', lambda host, workers: __import__('types').SimpleNamespace(
        key=b'k', configuration=lambda label: [], collect=lambda label: [], close=lambda: None))
    entries = [runner.Entry('cpu', 'workspace', package, 1, (), ()) for package in ('rw-wrfbatch', 'rw-fire-indices')]
    outcomes = list(runner.run_entries(entries, target_dir=tmp_path / 'target', cargo='cargo',
                                       log_dir=tmp_path / 'logs', workers=4))
    assert [o.verdict for o in outcomes] == ['PASSED', 'PASSED']
    assert peak[0] == 1, 'rows of one workspace overlapped'
    assert not any('--row--' in str(t) for t in targets) and not list((tmp_path / 'target' / 'ws').glob('*--row--*'))
