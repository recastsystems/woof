#!/usr/bin/env python3
"""Run the Rust workspaces' test suites as one release-battery leg.

``tools/battery/cargo_gates.txt`` enumerates them, one package per line,
each with the expected-green contract it has to satisfy.  This is the
runner the battery invokes:

    python tools/battery/run_cargo_gates.py

from the repository root, with ``CARGO_TARGET_DIR`` pointing OUTSIDE the
tree.  It prints a matrix -- one row per entry, with the counts cargo
reported -- and ends in a single verdict line, the same shape as the
``[tiles]`` gates, because a leg that reports only an exit status throws
away the numbers that make a regression legible.

EVERY ENTRY'S FULL CARGO OUTPUT IS KEPT, in ``--log-dir`` (default
``<CARGO_TARGET_DIR>/gate-logs``), one file per entry, and a failed entry's
report names its failing tests and their panics before the tail of its
output.  Origin: the 2.9.3 R8 Windows gate reported "2 targets failed:
--test chem_products, --test snowfall" for rw-wrfbatch and nothing else,
because only the last 40 lines of the crate's output survived and the
failures sat in earlier binaries' output; the same targets passed alone
on the same tree, so the gate had proven a failure it could not name.

THREE REFUSALS, ALL LOUD
    * no ``CARGO_TARGET_DIR``, one named inside the repository, or an alias
      of a worktree's release-artifact target directory. A symlink outside
      the source tree does not make sharing that target safe.
      A cut hashes the worktree-local renderer at
      ``tools/rustwx/target/release`` before it believes any leg that
      renders; a test leg sharing that directory can replace the binary
      the cut is adjudicating.  Defaulting to a scratch path would hide
      the mistake instead of stopping it, so there is no default.
    * no ``cargo`` on PATH.  This leg is not optional and does not skip:
      a battery that reports a clean leg on a box with no Rust toolchain
      is exactly the green-on-nothing failure the 2026-08-09 audit was
      called to close.
    * an entry whose workspace or package the manifest does not
      describe.  ``tests/test_cargo_gate_manifest.py`` catches that
      without a toolchain; this catches it again at the point of use.

EXIT STATUS IS THE VERDICT: 0 when every entry on the selected shard
passed its contract, 1 when any failed, 3 when the leg could not run at
all.  Those are different answers and the battery reads them
differently -- a leg that could not run has proven nothing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import json
import secrets
import threading
from multiprocessing.connection import Client, Listener
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tempfile

# Off POSIX (the desktop NTFS runner) rows of one workspace run one at a time and get no per-row link clones.
# A module flag, so a test can take the off-POSIX branch on Linux without setting the global os.name, which
# makes every pathlib.Path() raise NotImplementedError (2.9.4 R7b route B INTERNALERROR, 2026-10-21).
POSIX = os.name == "posix"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "cargo_gates.txt"

#: The shards the battery knows how to run.  ``cpu`` and ``gpu`` are the
#: shared vocabulary with tools/battery/tiles_gates.txt; ``fixtures`` is
#: this list's own, for a suite that needs a data set the repository does
#: not carry.  A fixtures entry is LISTED and not run by default, which
#: is the difference between coverage that is deferred and coverage that
#: silently vanished.
SHARDS = ("cpu", "gpu", "fixtures", "qualification")

#: The shard a release cut runs.  Named rather than spelled inline so the
#: manifest gate and the runner cannot disagree about it.
BATTERY_SHARD = "cpu"

#: Cargo's per-binary summary, e.g.
#: ``test result: ok. 26 passed; 0 failed; 0 ignored; 0 measured; ...``
_RESULT = re.compile(
    r"^test result: (?P<verdict>ok|FAILED)\. "
    r"(?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored",
    re.MULTILINE)

#: One failing test of a binary, ``test tests::name ... FAILED``.
_FAILED_TEST = re.compile(r"^test (?P<name>\S.*) \.\.\. FAILED$", re.MULTILINE)

#: A test thread's panic, ``thread 'name' panicked at file:line:col:``,
#: followed by its message on the next line(s).
_PANIC = re.compile(
    r"^thread '(?P<thread>[^']*)' .*panicked at (?P<site>[^\n]*):\n"
    r"(?P<message>(?:[^\n]*\n)*?)(?=note: |thread '|$)", re.MULTILINE)

#: Exit status when the leg could not be run at all, as distinct from a
#: leg that ran and failed (1).  A missing toolchain disproves nothing.
CANNOT_RUN = 3


def failing_tests(output: str) -> list[str]:
    """The failing tests of one cargo run, each with its panic.

    Cargo prints every binary's result in turn, so with several test
    targets a failure early in the output is out of the tail; this reads
    the whole output and names each ``test ... FAILED`` line, with the
    panic site and message that test's thread printed, so a failed entry
    is reported by test name, file and line rather than by target.
    """

    panics: dict[str, str] = {}
    for match in _PANIC.finditer(output):
        message = " ".join(line.strip() for line in match["message"].splitlines()
                           if line.strip())
        panics.setdefault(match["thread"], f"{match['site']}: {message}".rstrip(": "))
    lines: list[str] = []
    for match in _FAILED_TEST.finditer(output):
        name = match["name"]
        panic = panics.get(name) or panics.get(name.rsplit("::", 1)[-1])
        lines.append(f"{name} FAILED" + (f"  ({panic})" if panic else ""))
    return lines


def log_file_name(entry: "Entry") -> str:
    """One file per entry, named after its label, safe on every filesystem."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", entry.label).strip("_") + ".log"


@dataclass(frozen=True)
class Entry:
    """One line of the manifest, parsed."""

    shard: str
    workspace: str
    package: str
    min_tests: int
    env: tuple[str, ...]
    args: tuple[str, ...]

    @property
    def label(self) -> str:
        suffix = (" " + " ".join(self.args)) if self.args else ""
        return f"{self.package}{suffix}"

    def invocation(self) -> list[str]:
        """The exact argv, so a failing row can be re-run by hand.

        ``--locked`` and ``--offline`` are not tunable: the workspaces
        vendor their dependencies and a green leg is supposed to prove
        that vendored closure is complete.
        """

        return (["cargo", "test", "--locked", "--offline", "-p",
                 self.package] + list(self.args))


@dataclass
class Outcome:
    entry: Entry
    passed: int
    failed: int
    ignored: int
    binaries: int
    returncode: int
    seconds: float
    output: str
    #: Where the full cargo output was written, or None when the caller
    #: did not ask for a log directory.
    log_path: Path | None = None

    @property
    def verdict(self) -> str:
        if self.returncode != 0 or self.failed:
            return "FAILED"
        if self.passed < self.entry.min_tests:
            return "UNDER-RAN"
        return "PASSED"


def parse_manifest(text: str) -> list[Entry]:
    """Parse the manifest exactly the way this runner consumes it.

    ``tests/test_cargo_gate_manifest.py`` imports this function rather
    than restating the format, so a file the suite accepts is a file the
    runner can read.
    """

    entries: list[Entry] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"tools/battery/cargo_gates.txt line {number}: expected "
                f"SHARD WORKSPACE PACKAGE MIN_TESTS [KEY=VALUE...], got "
                f"{line!r}")
        shard, workspace, package, min_tests = parts[:4]
        if shard not in SHARDS:
            raise ValueError(
                f"tools/battery/cargo_gates.txt line {number}: shard "
                f"{shard!r} is not one of {SHARDS}")
        if not min_tests.isdigit() or int(min_tests) < 1:
            raise ValueError(
                f"tools/battery/cargo_gates.txt line {number}: MIN_TESTS "
                f"{min_tests!r} must be a positive integer -- a floor of 0 "
                "is satisfied by a suite that compiled nothing")
        env: list[str] = []
        args: list[str] = []
        for token in parts[4:]:
            # KEY=VALUE is environment; anything else goes to cargo.  A
            # bare word that is neither is refused rather than guessed
            # at: `--test cloud_products` is two tokens and the second
            # one is a bare word, so it is accepted only AFTER a cargo
            # argument has opened the tail.
            if "=" in token and not token.startswith("-") and not args:
                env.append(token)
            elif token.startswith("-") or args:
                args.append(token)
            else:
                raise ValueError(
                    f"tools/battery/cargo_gates.txt line {number}: "
                    f"{token!r} is neither KEY=VALUE nor a cargo argument; "
                    "environment assignments come first, then cargo "
                    "arguments beginning with a dash")
        entries.append(Entry(shard=shard, workspace=workspace,
                             package=package, min_tests=int(min_tests),
                             env=tuple(env), args=tuple(args)))
    return entries


def read_manifest(path: Path = MANIFEST) -> list[Entry]:
    return parse_manifest(path.read_text(encoding="utf-8"))


def resolve_target_dir(explicit: str | None) -> Path:
    """The external target directory, or a refusal that says why."""

    raw = explicit or os.environ.get("CARGO_TARGET_DIR")
    if not raw:
        raise RuntimeError(
            "CARGO_TARGET_DIR is not set.  This leg builds two Cargo "
            "workspaces and it is not allowed to build them inside the "
            "tree: a release cut hashes the worktree-local renderer at "
            "tools/rustwx/target/release and refuses to assemble from a "
            "dirty source tree, and a shared target directory would let "
            "this leg replace the binary the cut is adjudicating.  Set "
            "CARGO_TARGET_DIR to a path outside the repository, or pass "
            "--target-dir.")
    # Check the named path before following symlinks: a staged in-tree target
    # may point to an external build cache that still holds release artifacts.
    named = Path(os.path.abspath(raw))
    target = named.resolve()
    if (named == REPOSITORY_ROOT or REPOSITORY_ROOT in named.parents
            or target == REPOSITORY_ROOT or REPOSITORY_ROOT in target.parents):
        raise RuntimeError(
            f"CARGO_TARGET_DIR={named} is inside the repository at "
            f"{REPOSITORY_ROOT}.  Point it outside the tree: the cut's "
            "renderer check and its dirty-tree refusal both read this "
            "worktree, and cargo artefacts written into it make both "
            "answers depend on which leg ran last.")
    # The manifest names the workspaces whose ordinary Cargo target folders
    # may be linked to a shared build cache. Passing that cache's real path
    # would bypass a lexical-only check and still share the cut's artifacts.
    protected = {(REPOSITORY_ROOT / entry.workspace / "target").resolve()
                 for entry in read_manifest()}
    for artifact_target in sorted(protected):
        if target == artifact_target or artifact_target in target.parents:
            raise RuntimeError(
                f"CARGO_TARGET_DIR={named} resolves to a release artifact target at "
                f"{artifact_target}. A test leg sharing it can replace the renderer "
                "or other native bytes the cut is adjudicating. Use a separate "
                "external test target directory.")
    return target


def available_cores() -> int:
    """Respect the host affinity, including CI containers and Windows."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def runner_toml(host: str, argv: list[str]) -> str:
    """Serialize the runner table as TOML basic strings, without shell quoting.

    JSON string escapes with ensure_ascii=False are also TOML basic-string
    escapes. Encoding strings separately avoids non-TOML JSON types and
    surrogate-pair escapes, including paths or labels with quotes/backslashes.
    """
    string = lambda value: json.dumps(value, ensure_ascii=False)
    return f'[target.{string(host)}]\nrunner = [' + ', '.join(map(string, argv)) + ']\n'


class BinaryPool:
    """Run Cargo-selected test binaries on a shared host-sized worker pool.

    Cargo's runner hook preserves its exact argv, package cwd and runtime
    environment. The hook acknowledges submission so Cargo can submit the next
    integration binary instead of serializing the whole package. This parent
    retains every result and makes any failed binary fail its manifest row.
    Doctests still run through Cargo and keep their original accounting.
    """
    def __init__(self, host, workers):
        self.host = host
        self.key = secrets.token_bytes(32)
        self.listener = Listener(('127.0.0.1', 0), backlog=max(128, workers * 2), authkey=self.key)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = {}
        self.configs = tempfile.TemporaryDirectory(prefix='cargo-gate-runners-')
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while True:
            with self.listener.accept() as connection:
                job = connection.recv()
                if job is None:
                    return
                entry, argv, cwd, environment = job
                future = self.pool.submit(subprocess.run, argv, cwd=cwd,
                    env=environment, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
                self.pending.setdefault(entry, []).append((argv, future))
                connection.send(True)

    def configuration(self, entry):
        address, port = self.listener.address
        runner = [sys.executable, str(Path(__file__).resolve()),
                  '--gate-test-runner', address, str(port), entry]
        # Keep the TOML out of Windows batch argument parsing. Each row gets
        # its own file so concurrent Cargo processes cannot replace a runner.
        path = Path(self.configs.name) / (secrets.token_hex(16) + '.toml')
        path.write_text(runner_toml(self.host, runner), encoding='utf-8')
        return ['--config', str(path)]

    def collect(self, entry):
        rows = []
        for argv, future in self.pending.get(entry, []):
            try:
                rows.append((argv, future.result()))
            except Exception as error:  # a binary that could not start fails its row, never the whole run
                rows.append((argv, subprocess.CompletedProcess(argv, 127, f'could not run: {error!r}\n')))
        return rows

    def close(self):
        with Client(self.listener.address, authkey=self.key) as connection:
            connection.send(None)
        self.thread.join()
        self.listener.close()
        self.pool.shutdown(wait=True)
        self.configs.cleanup()


def is_doctest(command) -> bool:
    """A doctest binary rustdoc built in its own temporary folder (``.../rustdoctestXXXX/rust_out``)."""
    return bool(command) and any(part.startswith('rustdoctest') for part in Path(command[0]).parts)


def test_runner(argv):
    address, port, entry, *command = argv
    if is_doctest(command):
        # rustdoc deletes its temporary folder as soon as the runner returns, so a doctest runs here, in place.
        # Breakage it prevents (2.9.4 fold pass R7b route A, 2026-10-21): the pool ran it later and found
        # '/work/tmp/rustdoctest7SDHLp/rust_out' gone (FileNotFoundError), and the whole gate run ended.
        return subprocess.run(command).returncode
    environment = dict(os.environ)
    key = environment.pop("ARWEN_CARGO_GATE_KEY")
    # Runtime context is transferred only over authenticated loopback IPC.
    # It is never written to logs or to a file.
    with Client((address, int(port)), authkey=bytes.fromhex(key)) as connection:
        connection.send((entry, command, os.getcwd(), environment))
        assert connection.recv() is True
    return 0


def run_entries(entries, *, target_dir, cargo, log_dir, workers=None, prebuild_jobs=None):
    """Overlap packages and their integration binaries without changing gates.

    The CPU shard builds every workspace ONCE (``prebuild_workspaces``) before any row runs."""
    # The CPU gate owns this parallel budget. GPU shards share an admitted
    # card and overlapping their binaries can exceed that card's VRAM lease.
    # Other shards retain their existing invocation order.
    if any(entry.shard != BATTERY_SHARD for entry in entries):
        for entry in entries:
            yield run_entry(entry, target_dir=target_dir, cargo=cargo, log_dir=log_dir)
        return
    workers = workers or available_cores()
    version = subprocess.run([cargo, '-vV'], capture_output=True, text=True)
    host = re.search(r'^host: (\S+)$', version.stdout, re.M)
    if not host:
        raise RuntimeError('cargo -vV did not report its host target; cannot configure the '
                           'runner that prevents serial integration binaries on the release path')
    binaries = BinaryPool(host[1], workers)
    def run(entry):
        kwargs = {'binary_pool': binaries}
        clone = row_target_clone(workspace_target_dir(target_dir, entry), entry)
        try:
            return run_entry(entry, target_dir=target_dir, cargo=cargo,
                             log_dir=log_dir, cargo_target=clone, **kwargs)
        finally:
            if clone is not None:
                shutil.rmtree(clone, ignore_errors=True)
    # Rows may select the same test binary, whose fixed fixture filenames
    # cannot tolerate two instances. Complete every row, including its queued
    # binaries, before starting another row of the same workspace/package.
    groups = {}
    for index, entry in enumerate(entries):
        key = ((REPOSITORY_ROOT / entry.workspace).resolve(), entry.package)
        if not POSIX:
            # Without per-row clones (row_target_clone), rows of one workspace share its folder: one at a time,
            # so no row rebuilds a unit another row's rustdoc or linker is reading. Workspaces still overlap.
            key = (REPOSITORY_ROOT / entry.workspace).resolve()
        groups.setdefault(key, []).append((index, entry))
    def run_group(group):
        return [(index, run(entry)) for index, entry in group]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Rows of ONE workspace also rebuild each other's units: package
            # selections differ, so a row's run phase rebuilt static_fields /
            # rw_nexrad under rw-wrfbatch's running rustdoc ("can't find crate",
            # "possibly newer version", V58 lane/cap-base 2026-10-21, 720 tests
            # passed and the row FAILED).  Every workspace is built once and every
            # row fingerprinted BEFORE any row runs; then each row runs on its own
            # hard-link clone of its workspace folder, so no row can replace a file
            # another row reads (rustc replaces files, never writes through a link;
            # no extra disk unless a row rebuilds).
            prebuild_workspaces(entries, target_dir=target_dir, cargo=cargo, log_dir=log_dir,
                                jobs=prebuild_jobs)
            results = {index: outcome for group in pool.map(run_group, groups.values())
                       for index, outcome in group}
            yield from (results[index] for index in range(len(entries)))
    finally:
        binaries.close()


def workspace_target_dir(target_dir: Path, entry: Entry) -> Path:
    """The overlapping runner's target folder for ``entry``'s workspace, below the external target folder."""
    name = entry.workspace.strip("/").replace("/", "__") or "root"
    return target_dir / "ws" / ("root" if name == "." else name)


def row_target_clone(folder: Path, entry: Entry) -> Path | None:
    """A private hard-link copy of a built workspace folder for one row (`cp -al` where POSIX cp exists, else one
    os.link per file, which NTFS supports); None only when the folder does not exist yet or links are refused, and
    the row then runs on the shared workspace folder.

    Breakage it prevents on Windows (2.9.4 R8 runs on the desktop): the first version cloned on POSIX only, so the
    desktop's rows of one workspace still rebuilt each other's units under a running rustdoc."""
    if not Path(folder).is_dir():
        return None
    clone = Path(folder).parent / (Path(folder).name + "--row--"
                                   + re.sub(r"[^A-Za-z0-9_.-]", "_", entry.label))
    shutil.rmtree(clone, ignore_errors=True)
    if POSIX and shutil.which("cp") is not None:
        done = subprocess.run(["cp", "-al", str(folder), str(clone)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if done.returncode == 0:
            return clone
        shutil.rmtree(clone, ignore_errors=True)
        return None
    # No clone off POSIX: rows of one workspace run one after another instead (run_entries groups them).
    # Breakage it prevents (2.9.4 pass ec5dbbdb R8, desktop, 2026-10-21): on the NTFS hard-link clone
    # rw-wrfbatch's lib failed E0308 x5 'multiple different versions of crate serde_json' against the one vendored
    # serde_json, mixing rlibs of two feature sets (fire-indices turns on float_roundtrip); the same rows on
    # V58's cp -al clones pass.
    return None


def run_entry(entry: Entry, *, target_dir: Path, cargo: str,
              log_dir: Path | None = None, binary_pool=None,
              cargo_target: Path | None = None, compile_only: bool = False) -> Outcome | None:
    workspace = REPOSITORY_ROOT / entry.workspace
    if not (workspace / "Cargo.toml").is_file():
        raise RuntimeError(
            f"{entry.label}: {entry.workspace}/Cargo.toml does not exist")
    if not (workspace / ".cargo" / "config.toml").is_file():
        raise RuntimeError(
            f"{entry.label}: {entry.workspace}/.cargo/config.toml does not "
            "exist, so --offline has no vendored source to resolve against")

    environment = row_environment(entry, target_dir=target_dir, binary_pool=binary_pool, cargo_target=cargo_target)
    if compile_only:
        # Same selection and environment as the run, `--no-run`: the run then
        # compiles nothing of its own unless its selection differs.  Not a
        # verdict: a compile error recurs, and is reported, in the row's run.
        selection = entry.invocation()[2:]
        if "--" in selection:
            selection = selection[:selection.index("--")]
        subprocess.run([cargo, "test", "--no-run"] + selection, cwd=workspace, env=environment,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    argv = [cargo, "test"] + (binary_pool.configuration(entry.label) if binary_pool else []) + entry.invocation()[2:]
    # cwd is the WORKSPACE, never the repository root: cargo reads
    # .cargo/config.toml from the working directory upward, and that file
    # is what replaces crates.io with the vendored directory.  Invoked
    # from the root with --manifest-path the source replacement would not
    # apply and --offline would fail on a clean box for the wrong reason.
    started = time.monotonic()
    completed = subprocess.run(
        argv, cwd=workspace, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if binary_pool:
        for command, result in binary_pool.collect(entry.label):
            completed.stdout += "\n$ " + " ".join(command) + "\n" + result.stdout
            if result.returncode and not completed.returncode:
                completed.returncode = result.returncode
    seconds = time.monotonic() - started

    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file_name(entry)
        log_path.write_text(
            f"$ {' '.join(argv)}\n(cwd {workspace}; cargo exit "
            f"{completed.returncode}; {seconds:.1f}s)\n\n{completed.stdout}",
            encoding="utf-8", errors="replace")

    passed = failed = ignored = binaries = 0
    for match in _RESULT.finditer(completed.stdout):
        binaries += 1
        passed += int(match["passed"])
        failed += int(match["failed"])
        ignored += int(match["ignored"])
    return Outcome(entry=entry, passed=passed, failed=failed,
                   ignored=ignored, binaries=binaries,
                   returncode=completed.returncode, seconds=seconds,
                   output=completed.stdout, log_path=log_path)



def prebuild_jobs_default() -> int:
    """Parallel rustc jobs for the one-time build: ARWEN_CARGO_PREBUILD_JOBS, else CARGO_BUILD_JOBS, else the host's cores."""
    for key in ("ARWEN_CARGO_PREBUILD_JOBS", "CARGO_BUILD_JOBS"):
        value = os.environ.get(key, "")
        if value.isdigit() and int(value) > 0:
            return int(value)
    return available_cores()


def target_flags(entry: Entry) -> tuple[tuple[str, ...], ...]:
    """The target selection of a row's cargo arguments (``--lib``, ``--test NAME`` ...) as flag groups, without the
    test-name filter after ``--``: the part that decides WHICH binaries cargo builds."""
    args = list(entry.args)
    if "--" in args:
        args = args[:args.index("--")]
    groups: list[tuple[str, ...]] = []
    for token in args:
        if token.startswith("-") or not groups:
            groups.append((token,))
        else:
            groups[-1] = groups[-1] + (token,)
    return tuple(groups)


def prebuild_calls(entries) -> list[tuple[str, list[str], tuple[str, ...]]]:
    """The ``cargo test --no-run`` selections that build every row's binaries once: (workspace, cargo arguments, env).

    Rows of one workspace are merged into as few calls as their shapes allow: every package with an unrestricted row
    (``-p pkg`` and no target flag) is built whole, in ONE call with the other whole packages; a package whose rows all
    name targets is built with the union of those targets (target flags apply to every ``-p`` of a call, so packages
    with different unions get their own call); rows whose environment differs keep their own call, since a build
    script may read it.  Workspaces keep manifest order; inside one, the whole-package call comes first.
    """
    shapes: dict[tuple, dict] = {}
    for entry in entries:
        key = (entry.workspace, entry.package, tuple(sorted(entry.env)))
        record = shapes.setdefault(key, {"whole": False, "flags": []})
        flags = target_flags(entry)
        if not flags:
            record["whole"] = True
        for group in flags:
            if group not in record["flags"]:
                record["flags"].append(group)
    calls: dict[tuple, list[str]] = {}
    for (workspace, package, env), record in shapes.items():
        flags = () if record["whole"] else tuple(record["flags"])
        calls.setdefault((workspace, env, flags), []).append(package)
    ordered = []
    for workspace in dict.fromkeys(entry.workspace for entry in entries):
        mine = [(key, packages) for key, packages in calls.items() if key[0] == workspace]
        mine.sort(key=lambda item: (item[0][2] != (), item[0][2], item[0][1]))
        for (_, env, flags), packages in mine:
            arguments = ["test", "--no-run", "--locked", "--offline"]
            for package in packages:
                arguments += ["-p", package]
            for group in flags:
                arguments += list(group)
            ordered.append((workspace, arguments, env))
    return ordered


def prebuild_workspaces(entries, *, target_dir: Path, cargo: str, log_dir: Path | None, jobs: int | None = None,
                        report=print) -> dict[str, dict]:
    """Build every row's test binaries ONCE per workspace, then fingerprint each row's exact selection, before any
    row runs.  Not a verdict: a compile error recurs, and is reported, in the row's own run.

    The build runs at ``jobs`` parallel rustc jobs (``prebuild_jobs_default``; the driver sets it) and, on Windows,
    below normal priority, so it is a short burst on a shared desktop rather than a long trickle.  Each row's own
    ``cargo test --no-run`` then takes 1-3 s: it compiles only a feature variant the merged call did not produce,
    at the same ``jobs``, into the shared workspace folder, so the next run is warm for it too.

    Breakage it prevents (2.9.4 R8 on the desktop, 2026-10-21): 59 rows each ran their own ``cargo test --no-run``
    at once on one workspace folder with CARGO_BUILD_JOBS=4; cargo's build-directory lock serialised them into one
    4-job build ("Blocking waiting for file lock on build directory" in every row) and the leg took 2,016-2,130 s
    with a warm target.
    """
    jobs = jobs or prebuild_jobs_default()
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if sys.platform == "win32" else 0
    summary: dict[str, dict] = {}

    def invoke(workspace_name: str, arguments: list[str], env: tuple[str, ...]):
        probe = Entry("cpu", workspace_name, "", 1, env, ())
        environment = row_environment(probe, target_dir=target_dir, binary_pool=None,
                                      cargo_target=workspace_target_dir(target_dir, probe))
        environment["CARGO_BUILD_JOBS"] = str(jobs)
        started = time.monotonic()
        done = subprocess.run([cargo] + arguments, cwd=REPOSITORY_ROOT / workspace_name, env=environment, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
        output = getattr(done, "stdout", "") or ""
        if not isinstance(output, str):
            output = output.decode("utf-8", "replace")
        return (time.monotonic() - started, getattr(done, "returncode", 0),
                len(re.findall(r"^\s*Compiling ", output, re.M)), output,
                len(re.findall(r"Blocking waiting for file lock", output)), [cargo] + arguments)

    for workspace_name in dict.fromkeys(entry.workspace for entry in entries):
        workspace = REPOSITORY_ROOT / workspace_name
        if not (workspace / "Cargo.toml").is_file() or not (workspace / ".cargo" / "config.toml").is_file():
            continue  # the row's run reports it
        record = {"jobs": jobs, "builds": 0, "build_seconds": 0.0, "build_compiled": 0, "build_failed": 0,
                  "rows": 0, "row_seconds": 0.0, "row_compiled": 0, "row_lock_waits": 0}
        lines: list[str] = []
        for call_workspace, arguments, env in prebuild_calls(entries):
            if call_workspace != workspace_name:
                continue
            seconds, code, compiled, output, _, argv = invoke(workspace_name, arguments, env)
            record["builds"] += 1
            record["build_seconds"] += seconds
            record["build_compiled"] += compiled
            record["build_failed"] += int(code != 0)
            lines.append(f"$ {' '.join(argv)}\n(cargo exit {code}; {seconds:.1f}s; {compiled} compiled)\n{output}\n")
        for entry in entries:
            if entry.workspace != workspace_name:
                continue
            selection = entry.invocation()[1:]
            if "--" in selection:
                selection = selection[:selection.index("--")]
            seconds, code, compiled, output, waits, argv = invoke(
                workspace_name, [selection[0], "--no-run"] + selection[1:], entry.env)
            record["rows"] += 1
            record["row_seconds"] += seconds
            record["row_compiled"] += compiled
            record["row_lock_waits"] += waits
            lines.append(f"$ {' '.join(argv)}\n(cargo exit {code}; {seconds:.1f}s; {compiled} compiled)\n{output}\n")
        if log_dir is not None:
            folder = Path(log_dir) / "prebuild"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / (re.sub(r"[^A-Za-z0-9._-]+", "_", workspace_name).strip("_") + ".log")).write_text(
                "".join(lines), encoding="utf-8", errors="replace")
        summary[workspace_name] = record
        report(f"  prebuild {workspace_name:<24} {record['builds']} build(s) at {jobs} jobs "
               f"{record['build_seconds']:.1f}s, {record['build_compiled']} compiled"
               + (f", {record['build_failed']} failed (the row reports it)" if record["build_failed"] else "")
               + f"; {record['rows']} rows fingerprinted {record['row_seconds']:.1f}s, "
               f"{record['row_compiled']} compiled, {record['row_lock_waits']} lock waits", flush=True)
    return summary


def row_environment(entry: Entry, *, target_dir: Path, binary_pool=None, cargo_target: Path | None = None) -> dict:
    """The environment one row's cargo runs with (target folder, build jobs, the row's own KEY=VALUE assignments)."""
    environment = os.environ.copy()
    # Overlapping rows build into one target folder per WORKSPACE. Breakage it prevents (2.9.4 pass 4b R8,
    # 2026-10-21 02:38Z): with one folder for every workspace, grib1_bridge's doctests compiled against
    # deps/librayon-<hash>.rlib while rw-zarr and mapped-engine, other workspaces with the same crate name and
    # hash but their own vendored copy and settings, rebuilt that same file ("can't find crate for rayon/sha2").
    # Rows of one workspace still share their folder and its incremental reuse; serial shards keep the one folder.
    environment["CARGO_TARGET_DIR"] = str(cargo_target or (workspace_target_dir(target_dir, entry) if binary_pool else target_dir))
    environment.setdefault("CARGO_BUILD_JOBS", str(available_cores()))
    # A library that is also a cdylib (static-fields, netcdf-writer, obs-score, ...) is written UNHASHED,
    # deps/libstatic_fields.rlib, so every dependency variant of it shares one file in the shared workspace
    # folder.  Breakage it prevents (3.0.0 seal pass 1, 2026-11-02): the static-fields rows rebuilt that file
    # against serde_json without float_roundtrip, and the rw-wrfbatch row then compiled against it with its own
    # serde_json (float_roundtrip on): E0308 "expected serde_json::value::Value, found serde_json::Value" in
    # src/fire_observations.rs, a red no source change caused.  Cargo's own switch (the one the Rust
    # bootstrap uses for std) gives those artifacts a hashed name too, so each variant has its own file; the
    # uplifted target/<profile>/libX.so keeps its plain name.  The same value on every row keeps reuse.
    environment["__CARGO_DEFAULT_LIB_METADATA"] = "gpuwm-cargo-gate"
    for assignment in entry.env:
        key, _, value = assignment.partition("=")
        # Worker integration tests need the same installed Python as this
        # battery, including when its path contains spaces. No shell expansion.
        environment[key] = sys.executable if value == "@python" else value
    if binary_pool:
        environment["ARWEN_CARGO_GATE_KEY"] = binary_pool.key.hex()
    return environment


_compile_entry = run_entry

def _print_matrix(outcomes: list[Outcome]) -> None:
    header = (f"{'package':<34} {'passed':>7} {'failed':>7} {'ign':>5} "
              f"{'bins':>5} {'floor':>6} {'secs':>7}  verdict")
    print(header)
    print("-" * len(header))
    for outcome in outcomes:
        print(f"{outcome.entry.label:<34} {outcome.passed:>7} "
              f"{outcome.failed:>7} {outcome.ignored:>5} "
              f"{outcome.binaries:>5} {outcome.entry.min_tests:>6} "
              f"{outcome.seconds:>7.1f}  {outcome.verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard", default=BATTERY_SHARD, choices=SHARDS,
                        help=f"which shard's entries to run (default "
                             f"{BATTERY_SHARD}, the shard a release cut "
                             f"runs)")
    parser.add_argument("--target-dir", default=None,
                        help="external CARGO_TARGET_DIR; defaults to the "
                             "environment variable, and there is no "
                             "in-tree fallback")
    parser.add_argument("--cargo", default="cargo",
                        help="cargo executable (default: cargo on PATH)")
    parser.add_argument("--log-dir", default=None,
                        help="directory for each entry's full cargo output "
                             "(default: <CARGO_TARGET_DIR>/gate-logs)")
    parser.add_argument("--list", action="store_true",
                        help="print the entries and exit without running")
    parser.add_argument("--prebuild-jobs", type=int, default=None,
                        help="parallel rustc jobs for the one-time build of each workspace before the rows "
                             "run (default: ARWEN_CARGO_PREBUILD_JOBS, else CARGO_BUILD_JOBS, else the cores)")
    args = parser.parse_args(argv)

    try:
        entries = read_manifest()
    except (OSError, ValueError) as exc:
        print(f"CARGO GATE COULD NOT RUN: {exc}", file=sys.stderr)
        return CANNOT_RUN

    selected = [e for e in entries if e.shard == args.shard]
    if args.list:
        for entry in selected:
            tail = " ".join(entry.env + entry.args)
            print(f"{entry.shard}  {entry.workspace}  {entry.package}  "
                  f"{entry.min_tests}" + (f"  {tail}" if tail else ""))
        return 0
    if not selected:
        print(f"CARGO GATE COULD NOT RUN: no {args.shard} entries in "
              f"{MANIFEST}", file=sys.stderr)
        return CANNOT_RUN

    if shutil.which(args.cargo) is None:
        print(f"CARGO GATE COULD NOT RUN: {args.cargo!r} is not on PATH.  "
              "This leg does not skip -- a clean report from a box with no "
              "Rust toolchain would say the crates are green when nothing "
              "compiled them.", file=sys.stderr)
        return CANNOT_RUN

    try:
        target_dir = resolve_target_dir(args.target_dir)
    except RuntimeError as exc:
        print(f"CARGO GATE COULD NOT RUN: {exc}", file=sys.stderr)
        return CANNOT_RUN

    log_dir = Path(args.log_dir).resolve() if args.log_dir else target_dir / "gate-logs"

    print(f"tree            {REPOSITORY_ROOT}")
    print(f"CARGO_TARGET_DIR {target_dir}")
    print(f"full output     {log_dir}")
    print(f"shard           {args.shard}, {len(selected)} entries, "
          f"cargo test --locked --offline")
    print(f"prebuild jobs   {args.prebuild_jobs or prebuild_jobs_default()} "
          f"(each workspace built once before its rows run)")
    print()

    outcomes: list[Outcome] = []
    try:
        for outcome in run_entries(selected, target_dir=target_dir,
                                   cargo=args.cargo, log_dir=log_dir, prebuild_jobs=args.prebuild_jobs):
            outcomes.append(outcome)
            print(f"  {outcome.entry.label:<34} {outcome.verdict:<10} "
                  f"{outcome.passed} passed, {outcome.failed} failed, "
                  f"{outcome.seconds:.1f}s", flush=True)
    except RuntimeError as exc:
        print(f"CARGO GATE COULD NOT RUN: {exc}", file=sys.stderr)
        return CANNOT_RUN

    print()
    _print_matrix(outcomes)
    print()

    bad = [o for o in outcomes if o.verdict != "PASSED"]
    for outcome in bad:
        print(f"--- {outcome.entry.label} ({outcome.verdict}, "
              f"cargo exit {outcome.returncode}) "
              f"{' '.join(outcome.entry.invocation())}")
        failures = failing_tests(outcome.output)
        if failures:
            print(f"    failing tests ({len(failures)}):")
            for line in failures:
                print(f"      {line}")
        tail = outcome.output.strip().splitlines()[-40:]
        print(f"    last {len(tail)} lines of output:")
        for line in tail:
            print(f"    {line}")
        if outcome.log_path is not None:
            print(f"    full output: {outcome.log_path}")

    total = sum(o.passed for o in outcomes)
    if bad:
        print(f"CARGO GATE FAILED  {len(bad)} of {len(outcomes)} entries: "
              + ", ".join(f"{o.entry.label} ({o.verdict})" for o in bad))
        return 1
    print(f"CARGO GATE PASSED  {len(outcomes)} entries, {total} tests "
          f"passed, 0 failed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--gate-test-runner':
        raise SystemExit(test_runner(sys.argv[2:]))
    raise SystemExit(main())
