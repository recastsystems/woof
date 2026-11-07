"""Two engine versions on one machine never touch each other's bridges.

The defect these bind, observed on test host 2 on 2026-10-05: a gpuwm 2.8.0
venv from PyPI found ``~/.woof/bridges/rw_netcdf`` built from source
revision ``fc5b34e26...`` (a local build another install was running),
printed "fetching v2.8.0's bridge bundle before this run continues" and
overwrote all 31 files in that shared directory.  Any box with two
engine versions ping-ponged, and each install silently broke the
other's runs.

What is bound here:

* a pinned install stages into ``~/.woof/bridges/<release>-<digest>``
  and the automatic refresh writes there and nowhere else;
* two releases side by side each resolve their own bytes, and neither
  one's resolution changes a byte or an mtime of the other's directory
  or of the flat legacy directory;
* the flat legacy directory is read only when its file is this
  release's exact pin, and a mismatch there is fetched into the
  versioned directory, never over the legacy file;
* an install with no pins keeps the flat layout, unjudged, as before;
* an explicit environment override still wins and is never judged;
* a bundle that does not verify whole installs nothing, and says why;
* two processes fetching at once (same release, or two releases) end
  with complete, verified directories and no scratch left behind;
* ``woof doctor`` names the directory this engine actually uses.

Every pin is computed from bytes these tests write, and every fetch is
a real verified download of a zip built here, served over ``file://``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import zipfile

import pytest

from woof import bridge_assets, bridges

_ARTIFACTS = ("rw_mpas_mesh", "rw_netcdf")

#: The local build the test host 2 flat directory held.
_LOCAL_REV = "fc5b34e26c0ffee0c0ffee0c0ffee0c0ffee0c0f"
_REV_A = "a" * 40
_REV_B = "b" * 40


def _payload(artifact: str, rev: str, filler: bytes = b"") -> bytes:
    return (b"MZ" + filler + bridge_assets.SOURCE_REV_MARKER
            + rev.encode("ascii") + b"\x00"
            + bridges.BRIDGE_ABI_MARKERS[artifact])


class _Release:
    """One release: its payloads, its bundle on a mirror, its pins."""

    def __init__(self, root: Path, name: str, rev: str, *, platform: str,
                 filler: bytes = b"", corrupt: str | None = None):
        self.name = name
        self.payloads = {
            artifact: _payload(artifact, rev, filler)
            for artifact in _ARTIFACTS}
        self.filenames = {artifact: bridges.executable_name(artifact)
                          for artifact in _ARTIFACTS}
        pins = []
        blob = io.BytesIO()
        with zipfile.ZipFile(blob, "w") as archive:
            for artifact in _ARTIFACTS:
                data = self.payloads[artifact]
                filename = self.filenames[artifact]
                pins.append(bridge_assets.BinaryPin(
                    artifact=artifact, filename=filename, bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest()))
                shipped = data
                if artifact == corrupt:
                    # Same length, different bytes: passes the size
                    # check, fails the SHA-256 one.
                    shipped = data[:-1] + bytes([data[-1] ^ 0xFF])
                archive.writestr(filename, shipped)
        body = blob.getvalue()
        self.mirror = root / f"mirror-{name}"
        self.mirror.mkdir(parents=True)
        bundle_name = f"gpuwm-bridges-{name}-{platform}.zip"
        (self.mirror / bundle_name).write_bytes(body)
        self.bundle = bridge_assets.BundlePin(
            platform=platform, filename=bundle_name, bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(), binaries=tuple(pins))
        self.pins = bridge_assets.BridgePins(
            release=name, platforms={platform: self.bundle})

    def tag(self) -> str:
        return f"{self.name}-{self.bundle.sha256[:12]}"

    def pin(self, artifact: str) -> bridge_assets.BinaryPin:
        return next(p for p in self.bundle.binaries
                    if p.artifact == artifact)

    def as_json(self) -> dict:
        return {
            "release": self.name,
            "platform": self.bundle.platform,
            "filename": self.bundle.filename,
            "bytes": self.bundle.bytes,
            "sha256": self.bundle.sha256,
            "binaries": [{"artifact": p.artifact, "filename": p.filename,
                          "bytes": p.bytes, "sha256": p.sha256}
                         for p in self.bundle.binaries],
            "mirror": self.mirror.resolve().as_uri(),
        }


def _snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under ``directory``: bytes and mtime, by relative path."""

    if not directory.exists():
        return {}
    return {str(path.relative_to(directory)): (path.read_bytes(),
                                               path.stat().st_mtime_ns)
            for path in sorted(directory.rglob("*")) if path.is_file()}


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A scratch home with no checkout or wheel rungs, and two releases."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    platform = "win-x86_64" if os.name == "nt" else "linux-x86_64"
    monkeypatch.setattr(bridge_assets, "host_platform", lambda: platform)
    empty = tmp_path / "no-checkout-build"
    empty.mkdir()
    monkeypatch.setattr(bridges, "crate_dir", lambda: empty)
    monkeypatch.setattr(bridges, "_package_parent", lambda: empty)
    monkeypatch.setattr(bridges, "packaged_bridge_dir", lambda: empty)
    monkeypatch.delenv(bridge_assets.STALE_POLICY_ENV, raising=False)
    for entry in bridge_assets.BUNDLED_ARTIFACTS:
        monkeypatch.delenv(entry.env_var, raising=False)

    releases = {
        "A": _Release(tmp_path, "v9.8.0-test", _REV_A, platform=platform),
        "B": _Release(tmp_path, "v9.9.0-test", _REV_B, platform=platform,
                      filler=b"\x90" * 32),
    }
    fetches: list[str] = []
    real_refresh = bridge_assets.refresh_staged_bundle

    def counted(**kwargs):
        fetches.append(state["running"])
        return real_refresh(**kwargs)

    monkeypatch.setattr(bridge_assets, "refresh_staged_bundle", counted)
    state = {"running": None}

    def run_as(key: str | None) -> _Release | None:
        """Become the engine of release ``key`` (None: no pins), freshly.

        Each engine is its own process in real life, so the one-attempt
        refresh memo and the pin cache start empty every time.
        """

        state["running"] = key
        if key is None:
            pins = bridge_assets.BridgePins(release=None, platforms={})
            release = None
        else:
            release = releases[key]
            pins = release.pins
            monkeypatch.setenv(bridge_assets.ASSET_URL_BASE_ENV,
                               release.mirror.resolve().as_uri())
        monkeypatch.setattr(bridge_assets, "load_pins",
                            lambda path=None: pins)
        monkeypatch.setattr(bridges, "_REFRESH_ATTEMPTED", False)
        monkeypatch.setattr(bridges, "_REFRESH_FAILURE", None)
        monkeypatch.setattr(bridges, "_STALE_ALLOWED", set())
        bridge_assets.forget_pin_memo()
        return release

    flat = home / ".woof" / "bridges"
    return {"home": home, "flat": flat, "releases": releases,
            "run_as": run_as, "fetches": fetches, "tmp": tmp_path,
            "platform": platform}


def _resolve(artifact: str) -> Path | None:
    entry = next(item for item in bridge_assets.BUNDLED_ARTIFACTS
                 if item.name == artifact)
    return bridges.find_artifact(entry.env_var,
                                 bridges.executable_name(artifact))


def _stage_local_build(flat: Path) -> dict[str, bytes]:
    """The test host 2 flat directory: a local build nobody published."""

    flat.mkdir(parents=True, exist_ok=True)
    staged = {}
    for artifact in _ARTIFACTS:
        data = _payload(artifact, _LOCAL_REV, b"\xcc" * 8)
        (flat / bridges.executable_name(artifact)).write_bytes(data)
        staged[artifact] = data
    return staged


# ---------------------------------------------------------------------------
# Where each install stages
# ---------------------------------------------------------------------------

def test_a_pinned_install_owns_a_versioned_directory(machine):
    release = machine["run_as"]("A")

    own = bridges.default_bridge_dir()

    assert own == machine["flat"] / release.tag()
    assert own.parent == bridges.legacy_bridge_dir() == machine["flat"]
    assert release.name in own.name
    assert release.bundle.sha256[:12] in own.name


def test_two_releases_never_share_a_directory(machine):
    a = machine["run_as"]("A")
    own_a = bridges.default_bridge_dir()
    b = machine["run_as"]("B")
    own_b = bridges.default_bridge_dir()

    assert own_a != own_b
    assert own_a.name == a.tag() and own_b.name == b.tag()


def test_an_install_without_pins_keeps_the_flat_layout_unjudged(machine):
    """A checkout, or a platform with no bundle: exactly as before."""

    machine["run_as"](None)
    local = _stage_local_build(machine["flat"])

    assert bridges.default_bridge_dir() == machine["flat"]
    assert bridges.legacy_bridge_candidates("rw_netcdf") == ()
    resolved = _resolve("rw_netcdf")
    assert resolved == (machine["flat"]
                        / bridges.executable_name("rw_netcdf")).resolve()
    assert resolved.read_bytes() == local["rw_netcdf"]
    assert machine["fetches"] == []


# ---------------------------------------------------------------------------
# The incident, and two versions side by side
# ---------------------------------------------------------------------------

def test_the_node2_incident_leaves_the_local_build_untouched(machine):
    """A stale flat file is fetched AROUND, never over."""

    local = _stage_local_build(machine["flat"])
    before = _snapshot(machine["flat"])
    release = machine["run_as"]("A")

    resolved = _resolve("rw_netcdf")

    own = machine["flat"] / release.tag()
    assert resolved == own / bridges.executable_name("rw_netcdf")
    assert resolved.read_bytes() == release.payloads["rw_netcdf"]
    assert machine["fetches"] == ["A"]
    # Every flat file is byte- and mtime-identical; the only new entries
    # under the shared root are inside this release's own directory.
    after = _snapshot(machine["flat"])
    for name, (data, mtime) in before.items():
        assert after[name] == (data, mtime), name
    new = set(after) - set(before)
    assert new and all(Path(name).parts[0] == release.tag() for name in new)
    for artifact, data in local.items():
        assert (machine["flat"]
                / bridges.executable_name(artifact)).read_bytes() == data


def test_two_versions_side_by_side_never_touch_each_others_files(machine):
    _stage_local_build(machine["flat"])
    a = machine["run_as"]("A")
    path_a = _resolve("rw_mpas_mesh")
    own_a = machine["flat"] / a.tag()
    snap_a = _snapshot(own_a)
    flat_files = {name: value for name, value
                  in _snapshot(machine["flat"]).items()
                  if len(Path(name).parts) == 1}

    b = machine["run_as"]("B")
    path_b = _resolve("rw_mpas_mesh")
    own_b = machine["flat"] / b.tag()
    snap_b = _snapshot(own_b)

    assert path_a.parent == own_a and path_b.parent == own_b
    assert path_a.read_bytes() == a.payloads["rw_mpas_mesh"]
    assert path_b.read_bytes() == b.payloads["rw_mpas_mesh"]
    assert _snapshot(own_a) == snap_a        # B never touched A

    # Ping-pong: each one resolves its own bytes again, with no fetch,
    # and neither directory moves.
    for _ in range(2):
        for key, own, release in (("A", own_a, a), ("B", own_b, b)):
            machine["run_as"](key)
            for artifact in _ARTIFACTS:
                resolved = _resolve(artifact)
                assert resolved.parent == own
                assert resolved.read_bytes() == release.payloads[artifact]
    assert machine["fetches"] == ["A", "B"]
    assert _snapshot(own_a) == snap_a
    assert _snapshot(own_b) == snap_b
    assert {name: value for name, value
            in _snapshot(machine["flat"]).items()
            if len(Path(name).parts) == 1} == flat_files


# ---------------------------------------------------------------------------
# The legacy flat layout
# ---------------------------------------------------------------------------

def test_a_legacy_flat_file_is_read_when_it_is_this_releases_pin(
        machine, monkeypatch):
    """An estate staged by 2.8.5 keeps working on 2.8.5: no download."""

    release = machine["releases"]["B"]
    machine["flat"].mkdir(parents=True)
    legacy = machine["flat"] / bridges.executable_name("rw_netcdf")
    legacy.write_bytes(release.payloads["rw_netcdf"])
    before = _snapshot(machine["flat"])
    machine["run_as"]("B")

    def forbidden(**kwargs):
        raise AssertionError("a matching legacy file must not be refetched")

    monkeypatch.setattr(bridge_assets, "refresh_staged_bundle", forbidden)

    assert _resolve("rw_netcdf") == legacy.resolve()
    assert _snapshot(machine["flat"]) == before
    assert not (machine["flat"] / release.tag()).exists()


def test_a_legacy_file_from_another_release_is_never_handed_to_a_door(
        machine, monkeypatch):
    """Offline (refuse), a mismatched flat file refuses and changes nothing."""

    other = machine["releases"]["B"]
    machine["flat"].mkdir(parents=True)
    legacy = machine["flat"] / bridges.executable_name("rw_netcdf")
    legacy.write_bytes(other.payloads["rw_netcdf"])
    before = _snapshot(machine["flat"])
    release = machine["run_as"]("A")
    monkeypatch.setenv(bridge_assets.STALE_POLICY_ENV, "refuse")

    with pytest.raises(bridges.StaleBridgeError) as raised:
        _resolve("rw_netcdf")

    assert str(legacy.resolve()) in str(raised.value)
    assert _REV_B in str(raised.value)
    assert _snapshot(machine["flat"]) == before
    assert not (machine["flat"] / release.tag()).exists()


def test_inspection_reads_the_legacy_file_without_fetching(machine):
    _stage_local_build(machine["flat"])
    before = _snapshot(machine["flat"])
    machine["run_as"]("A")

    with bridges.inspection_only():
        resolved = _resolve("rw_netcdf")

    assert resolved.parent == machine["flat"].resolve()
    assert machine["fetches"] == []
    assert _snapshot(machine["flat"]) == before


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------

def test_an_environment_override_wins_and_is_never_judged(
        machine, tmp_path, monkeypatch):
    _stage_local_build(machine["flat"])
    release = machine["run_as"]("A")
    mine = tmp_path / "mine" / bridges.executable_name("rw_netcdf")
    mine.parent.mkdir()
    mine.write_bytes(_payload("rw_netcdf", _LOCAL_REV))
    entry = next(item for item in bridge_assets.BUNDLED_ARTIFACTS
                 if item.name == "rw_netcdf")
    monkeypatch.setenv(entry.env_var, str(mine))

    assert _resolve("rw_netcdf") == mine.resolve()
    assert machine["fetches"] == []
    assert not (machine["flat"] / release.tag()).exists()


def test_an_override_naming_a_missing_file_still_fails_loudly(
        machine, tmp_path, monkeypatch):
    machine["run_as"]("A")
    entry = next(item for item in bridge_assets.BUNDLED_ARTIFACTS
                 if item.name == "rw_netcdf")
    monkeypatch.setenv(entry.env_var, str(tmp_path / "nowhere"))

    with pytest.raises(FileNotFoundError):
        _resolve("rw_netcdf")


# ---------------------------------------------------------------------------
# A partial fetch
# ---------------------------------------------------------------------------

def test_a_bundle_that_does_not_verify_whole_installs_nothing(machine):
    broken = _Release(machine["tmp"], "v9.9.9-broken", _REV_B,
                      platform=machine["platform"], corrupt="rw_netcdf")
    dest = machine["flat"] / broken.tag()

    with pytest.raises(bridge_assets.BridgeAssetError) as raised:
        bridge_assets.fetch_bundle(
            broken.pins, broken.bundle, dest, progress=lambda m: None,
            urlopen_fn=lambda request: open(
                broken.mirror / broken.bundle.filename, "rb"))

    message = str(raised.value)
    assert "installed none of it" in message
    assert bridges.executable_name("rw_netcdf") in message
    assert "another release's" in message           # the breakage, named
    # rw_mpas_mesh verified, and is still not installed: half a set is
    # exactly the mixed estate the refusal exists to prevent.
    for artifact in _ARTIFACTS:
        assert not (dest / bridges.executable_name(artifact)).exists()
    assert not list(dest.glob(f"{bridge_assets.ARCHIVE_SUBDIR}-stage-*"))


# ---------------------------------------------------------------------------
# Two processes at once
# ---------------------------------------------------------------------------

_FETCHER = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    from woof import bridge_assets

    spec = json.loads(Path(sys.argv[1]).read_text())
    dest = Path(sys.argv[2])
    pins = tuple(bridge_assets.BinaryPin(**entry)
                 for entry in spec["binaries"])
    bundle = bridge_assets.BundlePin(
        platform=spec["platform"], filename=spec["filename"],
        bytes=spec["bytes"], sha256=spec["sha256"], binaries=pins)
    release = bridge_assets.BridgePins(
        release=spec["release"], platforms={spec["platform"]: bundle})
    bridge_assets.fetch_bundle(release, bundle, dest,
                               progress=lambda message: None)
    print("done")
""")


def _spawn(tmp_path: Path, release: _Release, dest: Path, tag: str):
    spec = tmp_path / f"spec-{tag}.json"
    spec.write_text(json.dumps(release.as_json()))
    script = tmp_path / "fetcher.py"
    script.write_text(_FETCHER)
    env = dict(os.environ)
    env[bridge_assets.ASSET_URL_BASE_ENV] = release.mirror.resolve().as_uri()
    root = Path(bridges.__file__).resolve().parent.parent
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, str(script), str(spec), str(dest)], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _assert_complete(dest: Path, release: _Release) -> None:
    for artifact in _ARTIFACTS:
        assert bridge_assets.matches_pin(
            dest / bridges.executable_name(artifact),
            release.pin(artifact)), artifact
    assert not list(dest.glob(f"{bridge_assets.ARCHIVE_SUBDIR}-stage-*"))
    assert not (dest / bridge_assets.BRIDGE_OWNER_NAME).exists()


def test_two_processes_fetching_one_release_at_once_are_safe(machine):
    release = _Release(machine["tmp"], "v9.9.1-big", _REV_B,
                       platform=machine["platform"],
                       filler=os.urandom(4 * 1024 * 1024))
    dest = machine["flat"] / release.tag()

    workers = [_spawn(machine["tmp"], release, dest, f"same-{i}")
               for i in range(2)]
    results = [worker.communicate(timeout=300) for worker in workers]

    for worker, (out, err) in zip(workers, results):
        assert worker.returncode == 0, err
        assert "done" in out
    _assert_complete(dest, release)


def test_two_releases_fetching_at_once_each_land_in_their_own(machine):
    _stage_local_build(machine["flat"])
    flat_before = _snapshot(machine["flat"])
    a = _Release(machine["tmp"], "v9.8.1-big", _REV_A,
                 platform=machine["platform"],
                 filler=os.urandom(2 * 1024 * 1024))
    b = _Release(machine["tmp"], "v9.9.2-big", _REV_B,
                 platform=machine["platform"],
                 filler=os.urandom(2 * 1024 * 1024))
    dest_a = machine["flat"] / a.tag()
    dest_b = machine["flat"] / b.tag()

    workers = [_spawn(machine["tmp"], a, dest_a, "a"),
               _spawn(machine["tmp"], b, dest_b, "b")]
    for worker in workers:
        _out, err = worker.communicate(timeout=300)
        assert worker.returncode == 0, err

    _assert_complete(dest_a, a)
    _assert_complete(dest_b, b)
    flat_after = _snapshot(machine["flat"])
    for name, value in flat_before.items():
        assert flat_after[name] == value, name


# ---------------------------------------------------------------------------
# What doctor says
# ---------------------------------------------------------------------------

def test_doctor_names_the_directory_this_engine_resolves(machine):
    from woof import doctor

    _stage_local_build(machine["flat"])
    release = machine["run_as"]("A")
    own = machine["flat"] / release.tag()

    note = bridges.staging_location_note()
    check = doctor._staged_estate_check()

    assert str(own) in note
    assert str(machine["flat"]) in note
    assert str(own) in check.detail
    # The flat local build is reported as not this release's bytes, by
    # path, and reporting it fetched nothing and changed nothing.
    assert check.status == "missing"
    assert "left in place" in check.detail
    assert machine["fetches"] == []
    assert not own.exists()


def test_doctor_names_the_flat_directory_for_an_unpinned_install(machine):
    from woof import doctor

    machine["run_as"](None)

    check = doctor._staged_estate_check()

    assert str(machine["flat"]) in check.detail
    assert "flat layout" in check.detail
