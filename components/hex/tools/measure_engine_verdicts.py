#!/usr/bin/env python3
"""Measure every published woof against this port's seam manifest.

THE INSTRUMENT OF RECORD for ``woof.hex.engine_pin.PUBLISHED_ENGINES``.  The
table in that module is the OUTPUT of this program, never a transcription:
every hand-written statement of these figures in this tree has been wrong
(the module's docstring counts three), and the earlier instrument lived in a
receipt directory that is held out of every published surface, so a reader
of the public tree could cite it and never run it.  This one lives beside
the table it writes.

WHAT IT MEASURES, per engine version, from 2.5.0 up:

* ``on_pypi`` -- whether PyPI serves at least one un-yanked file for the
  version.  A git tag is not a release; ``gpuwm>=2.5.5`` once named a
  version pip could not install.
* ``moved`` -- the manifest paths whose bytes in the PUBLISHED WHEEL differ
  from the manifest.  Every wheel of a version is downloaded (the 2.7 line
  publishes one per platform) and they must agree on all sixteen paths, or
  the version is refused rather than measured off whichever wheel came
  first.  A manifest path the wheel does not carry at all is recorded under
  ``absent`` and counted as moved, because an install that lacks a pinned
  file cannot satisfy the pin.  A version with no release is measured off
  its git tag instead, and says so.
* ``offline_build_road`` -- whether the tag's tree carries
  ``tools/rustwx/Cargo.lock`` and
  ``tools/rustwx/vendor/crates-io/cc/src/target/generated.rs``, the lockfile
  the manual's ``cargo build --locked --offline`` needs and the vendored file
  whose absence stopped that build at 2.5.5 while the same command finished
  at 2.5.6.  This is a presence check against the measured breakage, not a
  build.
* ``tag_disagrees_with_wheel`` -- manifest paths on which the tag's blobs
  and the wheel's bytes differ, for a version that has both.  The forecast
  lane clones the tag and pip installs the wheel; a path that differs
  between them is a finding in its own right.

WHAT IT WRITES.  ``--json`` is the measurement.  ``--splice`` renders it into
the generated block of ``engine_pin.py`` so no hand touches a row.
``--pin-to VERSION`` measures every engine against THAT version's published
bytes instead of the shipped manifest, which is how a re-pin is previewed;
with ``--write-manifest`` it then rewrites ``ARWEN_SOURCE_MANIFEST``,
``ARWEN_BUILD_COMMIT`` and ``ARWEN_GLACIER_COMPOSED_TU_SHA256`` in the
manifest module from the same bytes -- refusing if the tag's tree and the
wheel disagree on any pinned path, because the checkout guard would then
refuse a clone of the tag the remedy tells a user to make.  The glacier
digest is computed by the engine's own ``translation_unit_source`` in a
subprocess whose ``PYTHONPATH`` is the extracted wheel, so it is the value
the running adapter will compare against.  Validated in both directions on
2026-09-13: run against the 2.6.5 wheel it reproduces the digest the 0.2.3
tree shipped, and against the 2.7.3 wheel it produces a different one.

``--admit`` MEASURES AN ENGINE THAT IS NOT ON PYPI -- a private build
carrying a PEP 440 local version label on a published base -- from its own
artifacts and registers it as a row of ``ADMITTED_ENGINES`` in the same
module.  Every artifact named (wheels, a source tree) must declare the same
version and carry the same sixteen bytes, none absent; ``--admit-git`` names
a git working tree whose HEAD becomes the row's build commit and whose blobs
at HEAD must equal those bytes with the paths clean; the base version must
sit inside the derived specifier, because the specifier is what pip
enforces and the row is what the seam enforces; the glacier digest is
computed by the engine's own hand from the extracted wheel, as above.  The
row is written into ``verification/engine-verdicts/admitted-engines.json``
(one document for every admitted engine, keyed by version) and
``--splice-admitted`` renders that document into the module's second
generated block.  The row records what moved from the public pin and the
digest of every artifact it was hashed from, and nothing about what the
engine is for.

NETWORK.  Reads ``pypi.org`` (the project JSON and the wheel files) and
nothing else; every download is verified against PyPI's own SHA-256 before
it is opened, and the cache directory is reused on the next run.  A git
clone of the engine repository with its tags is required for the build-road
clause and the tag-only rows; ``git clone --filter=blob:none --no-checkout``
is enough, the tool fetches the blobs it needs by hash.

Usage::

    python tools/measure_engine_verdicts.py --engine-git <arwen clone> \\
        --wheel-cache <dir> --json verification/engine-verdicts/<name>.json \\
        [--pin-to 2.7.4 [--write-manifest]] [--splice src/hexcore/engine_pin.py] \\
        [--table]

    python tools/measure_engine_verdicts.py --admit <wheel>... <tree> \\
        --admit-git <tree> [--admit-json verification/engine-verdicts/admitted-engines.json] \\
        [--splice-admitted src/hexcore/engine_pin.py]
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

TREE = Path(__file__).resolve().parents[1]
MANIFEST_MODULE = TREE / "src" / "hexcore" / "cuda_arwen_physics_v841.py"
ENGINE_PIN_MODULE = TREE / "src" / "hexcore" / "engine_pin.py"

DISTRIBUTION = "woof"
PYPI_JSON = f"https://pypi.org/pypi/{DISTRIBUTION}/json"
ENGINE_REPOSITORY = "https://github.com/FahrenheitResearch/arwen"

#: The lowest engine this port has ever pinned a seam against.  Earlier
#: engines predate the column-batch seam and have nothing to measure.
DEFAULT_FLOOR = "2.5.0"

#: The vendored file whose absence stopped the 2.5.5 build, and the lockfile
#: the ``--locked`` build reads.  Both must be in the tag's tree.
BUILD_ROAD_FILES = (
    "tools/rustwx/Cargo.lock",
    "tools/rustwx/vendor/crates-io/cc/src/target/generated.rs",
)

BEGIN_MARK = "# --- BEGIN GENERATED: PUBLISHED_ENGINES ---"
END_MARK = "# --- END GENERATED: PUBLISHED_ENGINES ---"

ADMITTED_BEGIN_MARK = "# --- BEGIN GENERATED: ADMITTED_ENGINES ---"
ADMITTED_END_MARK = "# --- END GENERATED: ADMITTED_ENGINES ---"
ADMITTED_JSON = TREE / "verification" / "engine-verdicts" / "admitted-engines.json"
ADMITTED_SCHEMA = "gpuwm-hex.admitted-engines/v1"

#: A PEP 440 public release followed by a local version label.  An engine
#: without the label is a published one and goes through the table above.
LOCAL_VERSION = re.compile(r"\d+\.\d+\.\d+\+[A-Za-z0-9]+(\.[A-Za-z0-9]+)*")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *arguments: str, binary: bool = False):
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True, capture_output=True, text=not binary)
    return completed.stdout if binary else completed.stdout.strip()


# ---------------------------------------------------------------------------
# the manifest this run measures against
# ---------------------------------------------------------------------------
def _module_constant(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                assert node.value is not None
                return node.value
    raise LookupError(f"{name} is not assigned at module level")


def shipped_manifest(module: Path = MANIFEST_MODULE) -> dict[str, str]:
    """``ARWEN_SOURCE_MANIFEST`` read structurally, without importing hexcore.

    Importing the module pulls numpy and the CUDA backend; an instrument that
    only needs sixteen strings should not need either.
    """

    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    node = _module_constant(tree, "ARWEN_SOURCE_MANIFEST")
    if isinstance(node, ast.Call):  # MappingProxyType({...})
        node = node.args[0]
    if not isinstance(node, ast.Dict):
        raise ValueError("ARWEN_SOURCE_MANIFEST is not a dict literal")
    manifest: dict[str, str] = {}
    for key, value in zip(node.keys, node.values):
        assert isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
        manifest[str(key.value)] = str(value.value)
    return manifest


# ---------------------------------------------------------------------------
# PyPI: what is published, and the bytes of every wheel
# ---------------------------------------------------------------------------
class WheelStore:
    """Downloads verified against PyPI's digests, cached by filename."""

    def __init__(self, cache: Path, offline: bool) -> None:
        self.cache = cache
        self.offline = offline
        self.cache.mkdir(parents=True, exist_ok=True)

    def project(self) -> dict:
        cached = self.cache / "pypi-project.json"
        if self.offline:
            if not cached.is_file():
                raise SystemExit("--offline and no cached PyPI project JSON")
            return json.loads(cached.read_text(encoding="utf-8"))
        with urllib.request.urlopen(PYPI_JSON, timeout=60) as response:
            payload = response.read()
        cached.write_bytes(payload)
        return json.loads(payload.decode("utf-8"))

    def fetch(self, entry: dict) -> Path:
        target = self.cache / entry["filename"]
        expected = entry["digests"]["sha256"]
        if target.is_file() and sha256_bytes(target.read_bytes()) == expected:
            return target
        if self.offline:
            raise SystemExit(f"--offline and {entry['filename']} is not cached")
        print(f"  downloading {entry['filename']} ({entry['size'] / 1e6:.1f} MB)",
              file=sys.stderr)
        with urllib.request.urlopen(entry["url"], timeout=600) as response:
            payload = response.read()
        if sha256_bytes(payload) != expected:
            raise SystemExit(
                f"{entry['filename']}: downloaded bytes do not hash to the "
                f"digest PyPI publishes for them")
        target.write_bytes(payload)
        return target


def wheel_digests(wheel: Path, paths: list[str]) -> tuple[dict[str, str], list[str]]:
    """Digest of every manifest path inside one wheel, and the absent ones."""

    present: dict[str, str] = {}
    absent: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for relative in paths:
            if relative in names:
                present[relative] = sha256_bytes(archive.read(relative))
            else:
                absent.append(relative)
    return present, sorted(absent)


# ---------------------------------------------------------------------------
# git: the tag's tree
# ---------------------------------------------------------------------------
def tag_commit(repo: Path, version: str) -> str | None:
    try:
        return _git(repo, "rev-parse", f"v{version}^{{commit}}")
    except subprocess.CalledProcessError:
        return None


def tag_digests(repo: Path, commit: str, paths: list[str]) -> tuple[dict[str, str], list[str]]:
    present: dict[str, str] = {}
    absent: list[str] = []
    for relative in paths:
        try:
            blob = _git(repo, "cat-file", "blob", f"{commit}:{relative}", binary=True)
        except subprocess.CalledProcessError:
            absent.append(relative)
            continue
        present[relative] = sha256_bytes(blob)
    return present, sorted(absent)


def build_road(repo: Path, commit: str) -> bool:
    listed = _git(repo, "ls-tree", "-r", "--name-only", commit, *BUILD_ROAD_FILES)
    return set(listed.split()) == set(BUILD_ROAD_FILES)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
def measure(
    *,
    manifest: dict[str, str],
    store: WheelStore,
    engine_git: Path,
    floor: str,
) -> list[dict]:
    project = store.project()
    releases: dict[str, list[dict]] = project["releases"]
    paths = sorted(manifest)

    tags = {
        line[1:] for line in _git(engine_git, "tag", "--list", "v*").split()
        if re.fullmatch(r"v\d+\.\d+\.\d+", line)
    }
    versions = sorted(
        {v for v in releases if re.fullmatch(r"\d+\.\d+\.\d+", v)} | tags,
        key=version_key,
    )
    versions = [v for v in versions if version_key(v) >= version_key(floor)]

    rows: list[dict] = []
    for version in versions:
        files = [f for f in releases.get(version, []) if not f.get("yanked")]
        wheels = [f for f in files if f["filename"].endswith(".whl")]
        on_pypi = bool(files)
        commit = tag_commit(engine_git, version)
        row: dict = {
            "tag": f"v{version}",
            "version": version,
            "on_pypi": on_pypi,
            "pypi_files": sorted(f["filename"] for f in files),
            "tag_commit": commit,
        }

        wheel_present: dict[str, str] | None = None
        wheel_absent: list[str] = []
        if wheels:
            per_wheel: dict[str, tuple[dict[str, str], list[str]]] = {}
            for entry in wheels:
                path = store.fetch(entry)
                per_wheel[entry["filename"]] = wheel_digests(path, paths)
            first_name, (wheel_present, wheel_absent) = next(iter(per_wheel.items()))
            for name, (present, absent) in per_wheel.items():
                if present != wheel_present or absent != wheel_absent:
                    raise SystemExit(
                        f"woof {version}: {name} and {first_name} disagree on "
                        "the manifest paths; refusing to measure the version "
                        "off one platform's wheel")
            row["wheels"] = {
                entry["filename"]: entry["digests"]["sha256"] for entry in wheels}
            row["wheel_digests"] = wheel_present
            row["wheel_absent"] = wheel_absent
        elif on_pypi:
            raise SystemExit(
                f"woof {version} is on PyPI with no wheel (sdist only); this "
                "instrument measures what pip installs and has no sdist arm")

        tag_present: dict[str, str] | None = None
        tag_absent: list[str] = []
        if commit is not None:
            tag_present, tag_absent = tag_digests(engine_git, commit, paths)
            row["tag_digests"] = tag_present
            row["tag_absent"] = tag_absent
            row["offline_build_road"] = build_road(engine_git, commit)
        else:
            row["offline_build_road"] = None

        if wheel_present is not None and tag_present is not None:
            row["tag_disagrees_with_wheel"] = sorted(
                relative for relative in paths
                if wheel_present.get(relative) != tag_present.get(relative))

        # The verdict: what pip installs when there is a release, the tag
        # when there is not.
        if wheel_present is not None:
            measured, absent, source = wheel_present, wheel_absent, "wheel"
        elif tag_present is not None:
            measured, absent, source = tag_present, tag_absent, "tag"
        else:
            raise SystemExit(f"woof {version}: neither a wheel nor a tag to measure")
        moved = sorted(
            {relative for relative in paths
             if measured.get(relative) != manifest[relative]} | set(absent))
        row["measured_from"] = source
        row["moved"] = moved
        row["absent"] = absent
        if row["offline_build_road"] is None:
            raise SystemExit(
                f"woof {version}: no tag v{version} in {engine_git}, so the "
                "build road cannot be measured; fetch the engine's tags")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# --pin-to: the manifest becomes another version's published bytes
# ---------------------------------------------------------------------------
def pinned_manifest(version: str, paths: list[str], store: WheelStore,
                    engine_git: Path) -> tuple[dict[str, str], str]:
    project = store.project()
    wheels = [f for f in project["releases"].get(version, [])
              if f["filename"].endswith(".whl") and not f.get("yanked")]
    if not wheels:
        raise SystemExit(f"--pin-to {version}: no published wheel to pin to")
    digests = None
    for entry in wheels:
        present, absent = wheel_digests(store.fetch(entry), paths)
        if absent:
            raise SystemExit(
                f"--pin-to {version}: {entry['filename']} lacks {absent}; a "
                "manifest cannot pin a path the install does not carry")
        if digests is None:
            digests = present
        elif digests != present:
            raise SystemExit(f"--pin-to {version}: the platform wheels disagree")
    commit = tag_commit(engine_git, version)
    if commit is None:
        raise SystemExit(f"--pin-to {version}: no tag v{version} in {engine_git}")
    tag_present, tag_absent = tag_digests(engine_git, commit, paths)
    disagree = sorted(p for p in paths if tag_present.get(p) != digests[p])
    if disagree or tag_absent:
        raise SystemExit(
            f"--pin-to {version}: the tag's tree and the wheel disagree on "
            f"{disagree + tag_absent}; the checkout guard would refuse a "
            "clone of the tag the remedy names")
    assert digests is not None
    return dict(digests), commit


def _glacier_digest_from_root(root: Path) -> str:
    """``sha256(translation_unit_source('noahmp_glacier'))`` by the engine's
    own hand, with ``root`` (an extracted wheel or a source tree) as the
    only ``PYTHONPATH`` entry, in a subprocess."""

    program = (
        "from hashlib import sha256\n"
        "from woof.core import noahmp_kernel_sources as n\n"
        "s = n.translation_unit_source('noahmp_glacier')\n"
        "print(sha256(s.encode('ascii')).hexdigest())\n"
    )
    env = dict(os.environ, PYTHONPATH=str(root))
    completed = subprocess.run(
        [sys.executable, "-c", program], env=env, capture_output=True,
        text=True, check=True)
    digest = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit(f"glacier digest subprocess printed {digest!r}")
    return digest


def glacier_digest_from_wheel(wheel: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="gpuwm-wheel-") as scratch:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(scratch)
        return _glacier_digest_from_root(Path(scratch))


def glacier_composed_digest(version: str, store: WheelStore) -> str:
    """The glacier digest of a PUBLISHED version, from its first wheel."""

    project = store.project()
    wheels = [f for f in project["releases"][version]
              if f["filename"].endswith(".whl") and not f.get("yanked")]
    return glacier_digest_from_wheel(store.fetch(wheels[0]))


def _rewrite_hex_on_line(lines: list[str], line: int, found: str, expected: str) -> None:
    index = line - 1
    if lines[index].count(found) != 1:
        raise SystemExit(
            f"{MANIFEST_MODULE}:{line} does not hold exactly one copy of "
            f"{found[:16]}...; refusing to guess which occurrence is the pin")
    lines[index] = lines[index].replace(found, expected)


def write_manifest(module: Path, manifest: dict[str, str], commit: str,
                   glacier: str) -> int:
    """Rewrite the three pinned constants in place, by AST position."""

    text = module.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(module))
    lines = text.split("\n")
    moved = 0

    node = _module_constant(tree, "ARWEN_SOURCE_MANIFEST")
    if isinstance(node, ast.Call):
        node = node.args[0]
    assert isinstance(node, ast.Dict)
    for key, value in zip(node.keys, node.values):
        assert isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
        relative = str(key.value)
        if relative not in manifest:
            raise SystemExit(f"{relative} is in the module and not in the measurement")
        if value.value != manifest[relative]:
            _rewrite_hex_on_line(lines, value.lineno, value.value, manifest[relative])
            moved += 1
    for name, expected in (("ARWEN_BUILD_COMMIT", commit),
                           ("ARWEN_GLACIER_COMPOSED_TU_SHA256", glacier)):
        value = _module_constant(tree, name)
        assert isinstance(value, ast.Constant) and isinstance(value.value, str)
        if value.value != expected:
            _rewrite_hex_on_line(lines, value.lineno, value.value, expected)
            moved += 1
    module.write_text("\n".join(lines), encoding="utf-8", newline="")
    return moved


# ---------------------------------------------------------------------------
# --admit: an engine that is not on PyPI, measured from its own artifacts
# ---------------------------------------------------------------------------
def wheel_version(wheel: Path) -> str:
    """The version the wheel's own METADATA declares."""

    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist()
                 if n.endswith(".dist-info/METADATA") and n.count("/") == 1]
        if len(names) != 1:
            raise SystemExit(f"{wheel.name}: expected one dist-info METADATA, found {names}")
        text = archive.read(names[0]).decode("utf-8")
    match = re.search(r"(?m)^Version:\s*(\S+)\s*$", text)
    if match is None:
        raise SystemExit(f"{wheel.name}: METADATA declares no Version")
    return match.group(1)


def tree_version(root: Path) -> str:
    """The version a source tree's ``pyproject.toml`` declares."""

    import tomllib

    declaration = root / "pyproject.toml"
    if not declaration.is_file():
        raise SystemExit(f"{root}: no pyproject.toml, so no version to key a row by")
    parsed = tomllib.loads(declaration.read_text(encoding="utf-8"))
    version = parsed.get("project", {}).get("version")
    if not isinstance(version, str):
        raise SystemExit(f"{root}: pyproject.toml states no static project.version")
    return version


def tree_digests(root: Path, paths: list[str]) -> tuple[dict[str, str], list[str]]:
    present: dict[str, str] = {}
    absent: list[str] = []
    for relative in paths:
        path = root / relative
        if path.is_file():
            present[relative] = sha256_bytes(path.read_bytes())
        else:
            absent.append(relative)
    return present, sorted(absent)


def _engine_pin_module():
    """``woof.hex.engine_pin`` from this tree, for the specifier the row must
    sit inside.  The module imports nothing heavy; the package init reads
    installed metadata only."""

    import importlib

    source = str(TREE / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("woof.hex.engine_pin")


def measure_admitted(
    *,
    manifest: dict[str, str],
    artifacts: list[Path],
    git_tree: Path,
) -> dict:
    """One admitted-engine row, or a refusal naming what disagreed."""

    paths = sorted(manifest)
    if not artifacts:
        raise SystemExit("--admit needs at least one wheel or source tree")
    measured: list[dict] = []
    for artifact in artifacts:
        artifact = artifact.resolve()
        if artifact.is_file() and artifact.suffix == ".whl":
            present, absent = wheel_digests(artifact, paths)
            measured.append({
                "name": artifact.name, "kind": "wheel",
                "sha256": sha256_bytes(artifact.read_bytes()),
                "version": wheel_version(artifact),
                "digests": present, "absent": absent, "path": artifact,
            })
        elif artifact.is_dir():
            present, absent = tree_digests(artifact, paths)
            measured.append({
                "name": f"{artifact.name} (source tree)", "kind": "tree", "sha256": None,
                "version": tree_version(artifact),
                "digests": present, "absent": absent, "path": artifact,
            })
        else:
            raise SystemExit(f"--admit {artifact}: neither a .whl nor a directory")

    for entry in measured:
        if entry["absent"]:
            raise SystemExit(
                f"--admit {entry['name']} lacks {entry['absent']}; an engine "
                "that does not carry a pinned path cannot be admitted")
    versions = sorted({entry["version"] for entry in measured})
    if len(versions) != 1:
        raise SystemExit(
            f"--admit: the artifacts declare different versions {versions}; "
            "refusing to key one row by several")
    version = versions[0]
    if not LOCAL_VERSION.fullmatch(version):
        raise SystemExit(
            f"--admit: {version} carries no PEP 440 local version label; a "
            "published engine is measured by the table above, not admitted")
    first = measured[0]
    for entry in measured[1:]:
        if entry["digests"] != first["digests"]:
            differing = sorted(p for p in paths if entry["digests"][p] != first["digests"][p])
            raise SystemExit(
                f"--admit: {entry['name']} and {first['name']} disagree on "
                f"{differing}; refusing to admit an engine off one artifact's bytes")
    digests = dict(first["digests"])

    base = version.split("+", 1)[0]
    engine_pin = _engine_pin_module()
    floor, ceiling = engine_pin.gpuwm_floor(), engine_pin.gpuwm_ceiling()
    if not (version_key(floor) <= version_key(base) < version_key(ceiling)):
        raise SystemExit(
            f"--admit: {version} rides base {base}, outside the derived "
            f"specifier {engine_pin.gpuwm_requirement()}; pip would never "
            "resolve it, so a row for it would admit nothing")

    git_tree = git_tree.resolve()
    commit = _git(git_tree, "rev-parse", "HEAD")
    dirty = _git(git_tree, "status", "--porcelain", "--", *paths)
    if dirty:
        raise SystemExit(
            f"--admit-git {git_tree}: pinned paths are dirty at HEAD:\n{dirty}\n"
            "a row must name the commit its bytes came from")
    head_present, head_absent = tag_digests(git_tree, commit, paths)
    disagree = sorted(p for p in paths if head_present.get(p) != digests[p]) + head_absent
    if disagree:
        raise SystemExit(
            f"--admit-git {git_tree}: HEAD {commit[:12]} disagrees with the "
            f"artifacts on {disagree}; the row would name a commit that does "
            "not carry its bytes")

    wheels = [entry for entry in measured if entry["kind"] == "wheel"]
    if wheels:
        glacier = glacier_digest_from_wheel(wheels[0]["path"])
    else:
        glacier = _glacier_digest_from_root(first["path"])

    return {
        "version": version,
        "base": base,
        "build_commit": commit,
        "measured_at": _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0).isoformat(),
        "instrument_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "pip_requirement": engine_pin.gpuwm_requirement(),
        "digests": digests,
        "moved_from_base": sorted(p for p in paths if digests[p] != manifest[p]),
        "glacier_composed_tu_sha256": glacier,
        "artifacts": [
            {"name": entry["name"], "kind": entry["kind"], "sha256": entry["sha256"]}
            for entry in measured
        ],
    }


def load_admitted(document: Path) -> dict:
    if document.is_file():
        loaded = json.loads(document.read_text(encoding="utf-8"))
        if loaded.get("schema") != ADMITTED_SCHEMA:
            raise SystemExit(f"{document}: schema {loaded.get('schema')!r} is not {ADMITTED_SCHEMA}")
        return loaded
    return {
        "schema": ADMITTED_SCHEMA,
        "instrument": Path(__file__).relative_to(TREE).as_posix(),
        "distribution": DISTRIBUTION,
        "engines": [],
    }


def write_admitted(document: Path, loaded: dict, row: dict) -> None:
    engines = [entry for entry in loaded["engines"] if entry["version"] != row["version"]]
    engines.append(row)
    engines.sort(key=lambda entry: (version_key(entry["version"]), entry["version"]))
    loaded["engines"] = engines
    loaded["instrument"] = Path(__file__).relative_to(TREE).as_posix()
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(json.dumps(loaded, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")


def render_admitted_rows(rows: list[dict]) -> str:
    out = [ADMITTED_BEGIN_MARK]
    if not rows:
        out.append("ADMITTED_ENGINES: tuple[AdmittedEngine, ...] = ()")
    else:
        out.append("ADMITTED_ENGINES: tuple[AdmittedEngine, ...] = (")
        for row in rows:
            out.append("    AdmittedEngine(")
            out.append(f'        version="{row["version"]}",')
            out.append(f'        base="{row["base"]}",')
            out.append(f'        build_commit="{row["build_commit"]}",')
            out.append("        digests=(")
            for relative in sorted(row["digests"]):
                out.append(f'            ("{relative}", "{row["digests"][relative]}"),')
            out.append("        ),")
            out.append(
                f'        glacier_composed_tu_sha256="{row["glacier_composed_tu_sha256"]}",')
            if row["moved_from_base"]:
                out.append("        moved_from_base=(")
                for relative in row["moved_from_base"]:
                    out.append(f'            "{relative}",')
                out.append("        ),")
            else:
                out.append("        moved_from_base=(),")
            out.append("        artifacts=(")
            for artifact in row["artifacts"]:
                digest = "None" if artifact["sha256"] is None else f'"{artifact["sha256"]}"'
                out.append(f'            ("{artifact["name"]}", {digest}),')
            out.append("        ),")
            out.append("    ),")
        out.append(")")
    out.append(ADMITTED_END_MARK)
    return "\n".join(out)


def splice_admitted(module: Path, rows: list[dict]) -> None:
    text = module.read_text(encoding="utf-8")
    start = text.index(ADMITTED_BEGIN_MARK)
    end = text.index(ADMITTED_END_MARK) + len(ADMITTED_END_MARK)
    module.write_text(text[:start] + render_admitted_rows(rows) + text[end:],
                      encoding="utf-8", newline="")


# ---------------------------------------------------------------------------
# rendering: the generated block, and the README table
# ---------------------------------------------------------------------------
def render_rows(rows: list[dict]) -> str:
    out = [BEGIN_MARK, "PUBLISHED_ENGINES: tuple[PublishedEngine, ...] = ("]
    for row in rows:
        out.append("    PublishedEngine(")
        out.append(f'        version="{row["version"]}",')
        out.append(f"        on_pypi={bool(row['on_pypi'])},")
        if row["moved"]:
            out.append("        moved=(")
            for relative in row["moved"]:
                out.append(f'            "{relative}",')
            out.append("        ),")
        else:
            out.append("        moved=(),")
        out.append(f"        offline_build_road={bool(row['offline_build_road'])},")
        out.append("    ),")
    out.append(")")
    out.append(END_MARK)
    return "\n".join(out)


def splice(module: Path, rows: list[dict]) -> None:
    text = module.read_text(encoding="utf-8")
    start = text.index(BEGIN_MARK)
    end = text.index(END_MARK) + len(END_MARK)
    module.write_text(text[:start] + render_rows(rows) + text[end:],
                      encoding="utf-8", newline="")


def render_table(rows: list[dict], manifest_size: int) -> str:
    lines = ["| woof | on PyPI | seam pin | `--offline` build road |",
             "| --- | --- | --- | --- |"]
    for row in rows:
        usable = row["on_pypi"] and not row["moved"] and row["offline_build_road"]
        version = f"**{row['version']}**" if usable else row["version"]
        pypi = "yes" if row["on_pypi"] else "**no, a git tag with no release**"
        pin = ("**matches**" if not row["moved"]
               else f"{len(row['moved'])} of {manifest_size} moved")
        road = "complete" if row["offline_build_road"] else "**broken**"
        lines.append(f"| {version} | {pypi} | {pin} | {road} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine-git", type=Path,
                        help="a clone of the engine repository carrying its tags")
    parser.add_argument("--wheel-cache", type=Path,
                        help="where downloaded wheels and the PyPI JSON are kept")
    parser.add_argument("--admit", type=Path, nargs="+", metavar="ARTIFACT",
                        help="admit an engine that is not on PyPI: its wheels "
                             "and/or source tree, which must agree")
    parser.add_argument("--admit-git", type=Path, metavar="TREE",
                        help="with --admit: the git working tree whose HEAD is "
                             "the row's build commit")
    parser.add_argument("--admit-json", type=Path, default=ADMITTED_JSON,
                        help="the admitted-engines document to read and write")
    parser.add_argument("--splice-admitted", type=Path, nargs="?",
                        const=ENGINE_PIN_MODULE,
                        help="render the admitted-engines document into "
                             "engine_pin.py's ADMITTED_ENGINES block")
    parser.add_argument("--floor", default=DEFAULT_FLOOR,
                        help=f"lowest version measured (default {DEFAULT_FLOOR})")
    parser.add_argument("--manifest-module", type=Path, default=MANIFEST_MODULE)
    parser.add_argument("--pin-to", metavar="VERSION",
                        help="measure against this version's published bytes "
                             "instead of the shipped manifest")
    parser.add_argument("--write-manifest", action="store_true",
                        help="with --pin-to: rewrite the manifest module's three "
                             "pinned constants from the pinned version")
    parser.add_argument("--json", type=Path, help="write the measurement here")
    parser.add_argument("--splice", type=Path, nargs="?", const=ENGINE_PIN_MODULE,
                        help="render the rows into engine_pin.py's generated block")
    parser.add_argument("--table", action="store_true",
                        help="print the README table")
    parser.add_argument("--offline", action="store_true",
                        help="use only the cache; never touch the network")
    args = parser.parse_args(argv)
    if args.write_manifest and not args.pin_to:
        parser.error("--write-manifest needs --pin-to")
    if args.admit:
        if args.admit_git is None:
            parser.error("--admit needs --admit-git")
        if args.pin_to or args.json or args.splice:
            parser.error("--admit measures against the shipped manifest and "
                         "writes only the admitted-engines document")
        row = measure_admitted(
            manifest=shipped_manifest(args.manifest_module),
            artifacts=list(args.admit), git_tree=args.admit_git)
        loaded = load_admitted(args.admit_json)
        write_admitted(args.admit_json, loaded, row)
        print(f"admitted woof {row['version']} (base {row['base']}, commit "
              f"{row['build_commit'][:12]}): {len(row['moved_from_base'])} of "
              f"{len(row['digests'])} paths moved from the pin; wrote "
              f"{args.admit_json}", file=sys.stderr)
        if args.splice_admitted:
            splice_admitted(args.splice_admitted, loaded["engines"])
            print(f"spliced {len(loaded['engines'])} admitted row(s) into "
                  f"{args.splice_admitted}", file=sys.stderr)
        return 0
    if args.splice_admitted and not (args.engine_git or args.wheel_cache):
        loaded = load_admitted(args.admit_json)
        splice_admitted(args.splice_admitted, loaded["engines"])
        print(f"spliced {len(loaded['engines'])} admitted row(s) into "
              f"{args.splice_admitted}", file=sys.stderr)
        return 0
    if args.engine_git is None or args.wheel_cache is None:
        parser.error("--engine-git and --wheel-cache are required to measure "
                     "the published engines")

    store = WheelStore(args.wheel_cache, args.offline)
    shipped = shipped_manifest(args.manifest_module)
    paths = sorted(shipped)
    pin: dict = {"version": None}
    if args.pin_to:
        manifest, commit = pinned_manifest(args.pin_to, paths, store, args.engine_git)
        pin = {"version": args.pin_to, "tag_commit": commit}
        if args.write_manifest:
            glacier = glacier_composed_digest(args.pin_to, store)
            pin["glacier_composed_tu_sha256"] = glacier
            moved = write_manifest(args.manifest_module, manifest, commit, glacier)
            print(f"rewrote {moved} pinned constant(s) in "
                  f"{args.manifest_module.relative_to(TREE).as_posix()}",
                  file=sys.stderr)
    else:
        manifest = shipped

    rows = measure(manifest=manifest, store=store, engine_git=args.engine_git,
                   floor=args.floor)
    document = {
        "instrument": Path(__file__).relative_to(TREE).as_posix(),
        "instrument_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "measured_at": _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0).isoformat(),
        "distribution": DISTRIBUTION,
        "engine_repository": ENGINE_REPOSITORY,
        "manifest_source": (
            f"published woof {args.pin_to} wheel bytes" if args.pin_to
            else "the shipped ARWEN_SOURCE_MANIFEST"),
        "pin": pin,
        "manifest": manifest,
        "floor": args.floor,
        "build_road_files": list(BUILD_ROAD_FILES),
        "tags": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    if args.splice:
        splice(args.splice, rows)
        print(f"spliced {len(rows)} rows into {args.splice}", file=sys.stderr)
    if args.splice_admitted:
        loaded = load_admitted(args.admit_json)
        splice_admitted(args.splice_admitted, loaded["engines"])
        print(f"spliced {len(loaded['engines'])} admitted row(s) into "
              f"{args.splice_admitted}", file=sys.stderr)
    if args.table or not (args.json or args.splice):
        print(render_table(rows, len(manifest)))
    usable = [row["version"] for row in rows
              if row["on_pypi"] and not row["moved"] and row["offline_build_road"]]
    print(f"usable: {usable or 'NONE'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
