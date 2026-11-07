"""``woof fetch-bridges``: staging, verification, and refusal contract.

The wheel ships no compiled Rust, so for a pip install this fetch path
IS the install path for nine artifacts: every byte must be verified
against the packaged pins before anything lands in ``~/.woof/bridges``,
a mismatch must be refused rather than installed, and the other
platform's bundle must be recognised as such instead of half-staged.

Nothing here fabricates a hash.  The synthetic-payload tests pin bytes
this file just wrote, by hashing them; the integration tests pack a real
bundle out of the Rust artifacts this machine has actually built and pin
*those* bytes with the release tool the release workflow runs.  Every
test but one runs entirely offline, and that one is gated twice.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from woof import bridge_assets, bridges

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = REPO_ROOT / "tools" / "build_bridge_bundle.py"


def test_intermediate_producer_is_reachable_from_the_release_bridge_roster(tmp_path):
    artifact = next(row for row in bridge_assets.BUNDLED_ARTIFACTS
                    if row.name == "met_intermediate")
    assert artifact.crate == bridges.CRATE_RELATIVE
    assert artifact.env_var == bridges.BRIDGE_ENV[artifact.name]
    binary = tmp_path / artifact.name
    binary.write_bytes(bridges.BRIDGE_ABI_MARKERS[artifact.name])
    bridge_assets.verify_contract_marker(artifact.name, binary)
    binary.write_bytes(b'old producer without the declared request contract')
    with pytest.raises(bridge_assets.BridgeAssetError):
        bridge_assets.verify_contract_marker(artifact.name, binary)


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**{"from_dir": None, "dest": None,
                                 "keep_bundle": False, "list": False,
                                 **kwargs})


def _pin(payload: bytes, artifact: str, filename: str
         ) -> bridge_assets.BinaryPin:
    """A pin computed from bytes that exist, never a literal."""

    return bridge_assets.BinaryPin(
        artifact=artifact, filename=filename, bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest())


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buffer.getvalue()


def _synthetic_bundle(directory: Path, *, platform: str = "win-x86_64",
                      names: tuple[str, ...] = ("alpha.bin", "beta.bin"),
                      corrupt: str | None = None
                      ) -> tuple[Path, bridge_assets.BundlePin]:
    """A bundle whose pins are computed from the bytes it carries.

    ``corrupt`` names a member whose archived bytes differ from the ones
    its pin was computed over -- the only way to build a bundle that
    must be refused without writing a hash by hand.  The member names
    are deliberately not real artifact names: these tests are about the
    staging contract, and a name in
    :data:`woof.bridges.BRIDGE_ABI_MARKERS` would additionally demand a
    contract marker no synthetic payload carries (the real-build tests
    below cover that check).
    """

    directory.mkdir(parents=True, exist_ok=True)
    payloads = {name: os.urandom(2048) + name.encode() for name in names}
    pins = tuple(_pin(payloads[name], name.split(".")[0], name)
                 for name in names)
    archived = dict(payloads)
    if corrupt is not None:
        # Same length, different bytes: the size gate must not be what
        # catches this, or the SHA-256 gate is never exercised.
        flipped = bytearray(payloads[corrupt])
        flipped[0] ^= 0xFF
        archived[corrupt] = bytes(flipped)
    blob = _zip_bytes(archived)
    archive = directory / f"gpuwm-bridges-v0-test-{platform}.zip"
    archive.write_bytes(blob)
    bundle = bridge_assets.BundlePin(
        platform=platform, filename=archive.name, bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(), binaries=pins)
    return archive, bundle


def _install_pins(monkeypatch, bundle: bridge_assets.BundlePin,
                  release: str = "v0-test") -> bridge_assets.BridgePins:
    """Point the command at pins built from real bytes, on this platform.

    Patching the loader rather than writing into the installed package
    keeps the shipped pins document untouched, and pinning the host
    platform to the bundle's makes these tests say the same thing on a
    Windows box and a Linux one.
    """

    pins = bridge_assets.BridgePins(release=release,
                                    platforms={bundle.platform: bundle})
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: bundle.platform)
    return pins


# ---------------------------------------------------------------------------
# Staging semantics, with synthetic artifacts
# ---------------------------------------------------------------------------

def test_staging_returns_the_assets_it_staged_as_well_as_the_binaries(
        tmp_path):
    """`stage_from_bundle` returns binaries AND map assets, and says so.

    #163: the live smoke asserted `len(installed) == len(bundle.binaries)`
    and stayed green because nothing in this file had ever packed a
    bundle carrying assets -- every synthetic bundle above is binaries
    only, so the half of the contract that grew later was tested by the
    one test that needs GitHub to run. On the published v2.0.0 bundle
    the old expectation reads 9 against a true 47.

    Offline, and about the contract rather than about one release's
    counts: whatever the manifest declares, the returned list is exactly
    the staged binaries followed by the staged assets, each at its own
    pinned path.
    """

    binary_names = ("alpha.bin", "beta.bin")
    asset_paths = (f"{bridge_assets.ASSET_ROOT}/ne_50m_admin_0.shp",
                   f"{bridge_assets.ASSET_ROOT}/ne_50m_admin_0.dbf",
                   f"{bridge_assets.ASSET_ROOT}/lakes/ne_50m_lakes.shp")
    payloads = {name: os.urandom(1024) + name.encode()
                for name in binary_names}
    payloads.update({path: os.urandom(512) + path.encode()
                     for path in asset_paths})
    blob = _zip_bytes(payloads)
    archive = tmp_path / "gpuwm-bridges-v0-test-win-x86_64.zip"
    archive.write_bytes(blob)
    bundle = bridge_assets.BundlePin(
        platform="win-x86_64", filename=archive.name, bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(),
        binaries=tuple(_pin(payloads[name], name.split(".")[0], name)
                       for name in binary_names),
        assets=tuple(bridge_assets.AssetPin(
            path=path, bytes=len(payloads[path]),
            sha256=hashlib.sha256(payloads[path]).hexdigest())
            for path in asset_paths))

    dest = tmp_path / "dest"
    installed = bridge_assets.stage_from_bundle(
        archive, bundle, dest, progress=lambda _line: None)

    assert len(installed) == len(bundle.binaries) + len(bundle.assets)
    # And NOT the binaries alone, which is the assertion #163 fixed.
    assert len(installed) != len(bundle.binaries)
    assert installed == ([dest / pin.filename for pin in bundle.binaries]
                         + [dest / pin.path for pin in bundle.assets])
    for pin in bundle.binaries:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)
    for pin in bundle.assets:
        staged = dest / pin.path
        # The pinned path is the staged path, subdirectories and all.
        assert staged.is_file()
        assert staged.stat().st_size == pin.bytes
        assert bridge_assets.sha256_file(staged) == pin.sha256


def test_a_corrupt_member_is_refused_and_never_installed(tmp_path):
    archive, bundle = _synthetic_bundle(tmp_path, corrupt="beta.bin")
    dest = tmp_path / "dest"

    with pytest.raises(bridge_assets.BridgeAssetError, match="SHA-256"):
        bridge_assets.stage_from_bundle(archive, bundle, dest,
                                        progress=lambda _line: None)

    # The refusal is for the WHOLE bundle (2.8.6): the good member ahead
    # of the bad one is not installed either, because half a release's
    # set beside older copies of the rest is a mixed estate the routes
    # would run without saying so.  Nothing half-written is left behind
    # for the resolver to find.
    assert not (dest / "alpha.bin").exists()
    assert not (dest / "beta.bin").exists()
    assert not list(dest.glob(f"{bridge_assets.ARCHIVE_SUBDIR}-stage*"))


def test_a_member_of_the_wrong_size_is_refused_before_the_hash(tmp_path):
    """Size and hash are both gates, and the size one names itself."""

    archive, bundle = _synthetic_bundle(tmp_path)
    wrong = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=bundle.bytes, sha256=bundle.sha256,
        binaries=(bridge_assets.BinaryPin(
            bundle.binaries[0].artifact, bundle.binaries[0].filename,
            bundle.binaries[0].bytes + 1, bundle.binaries[0].sha256),))

    with pytest.raises(bridge_assets.BridgeAssetError, match="bytes"):
        bridge_assets.stage_from_bundle(archive, wrong, tmp_path / "dest",
                                        progress=lambda _line: None)


def test_the_other_platforms_bundle_is_refused_by_name(tmp_path):
    """A linux bundle staged against windows pins names both shapes.

    The failure mode this replaces is silent: a bundle whose members are
    all called something else stages nothing, and the user learns about
    it from a decoder that is still missing.
    """

    linux_archive, linux = _synthetic_bundle(
        tmp_path / "linux", platform="linux-x86_64",
        names=("grib1_bridge", "rw_fetch"))
    windows = bridge_assets.BundlePin(
        platform="win-x86_64", filename=linux.filename, bytes=linux.bytes,
        sha256=linux.sha256,
        binaries=tuple(
            bridge_assets.BinaryPin(pin.artifact, f"{pin.filename}.exe",
                                    pin.bytes, pin.sha256)
            for pin in linux.binaries))

    with pytest.raises(bridge_assets.BridgeAssetError) as failure:
        bridge_assets.stage_from_bundle(
            linux_archive, windows, tmp_path / "dest",
            progress=lambda _line: None)
    message = str(failure.value)
    assert "grib1_bridge.exe" in message      # what was pinned
    assert "grib1_bridge" in message          # what the archive holds
    assert "not the win-x86_64 bundle" in message
    assert not (tmp_path / "dest" / "grib1_bridge").exists()


def test_an_existing_pin_valid_file_is_left_alone(tmp_path, monkeypatch):
    archive, bundle = _synthetic_bundle(tmp_path)
    _install_pins(monkeypatch, bundle)
    dest = tmp_path / "dest"
    bridge_assets.stage_from_bundle(archive, bundle, dest,
                                    progress=lambda _line: None)
    before = {pin.filename: (dest / pin.filename).stat().st_mtime_ns
              for pin in bundle.binaries}

    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(tmp_path))) == 0
    after = {pin.filename: (dest / pin.filename).stat().st_mtime_ns
             for pin in bundle.binaries}
    assert after == before


def test_a_stale_artifact_is_replaced_not_refused(tmp_path, monkeypatch,
                                                  capsys):
    """The deliberate divergence from ``fetch-tables``, pinned.

    A physics table that differs is the operator's file and is never
    overwritten.  A bridge executable that differs is yesterday's build,
    and leaving it in place is exactly the skew that made 1.1.0
    preparations fail while `doctor` reported the binary `ok`.
    """

    archive, bundle = _synthetic_bundle(tmp_path)
    _install_pins(monkeypatch, bundle)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "alpha.bin").write_bytes(b"yesterday's build")

    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "do not match this release's pins and will be replaced" in out
    assert bridge_assets.matches_pin(dest / "alpha.bin", bundle.binaries[0])


def test_from_dir_stages_loose_artifacts_under_the_same_pins(tmp_path):
    """What an air-gapped operator has: files, not an archive."""

    archive, bundle = _synthetic_bundle(tmp_path)
    loose = tmp_path / "loose"
    loose.mkdir()
    with zipfile.ZipFile(archive) as zf:
        for pin in bundle.binaries:
            (loose / pin.filename).write_bytes(zf.read(pin.filename))
    dest = tmp_path / "dest"

    installed = bridge_assets.stage_from_dir(loose, bundle, dest,
                                             progress=lambda _line: None)
    assert len(installed) == len(bundle.binaries)
    for pin in bundle.binaries:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)


def test_from_dir_stages_the_verified_subset_with_a_warning(
        tmp_path, capsys):
    """Warn-not-block: the artifacts are independent; what is present
    and pin-valid stages, what is absent is named for doctor.  An
    EMPTY directory still refuses -- there is nothing to stage."""
    archive, bundle = _synthetic_bundle(tmp_path)
    loose = tmp_path / "loose"
    loose.mkdir()
    with zipfile.ZipFile(archive) as zf:
        (loose / "alpha.bin").write_bytes(zf.read("alpha.bin"))

    installed = bridge_assets.stage_from_dir(
        loose, bundle, tmp_path / "dest", progress=lambda _line: None)
    err = capsys.readouterr().err
    assert [path.name for path in installed] == ["alpha.bin"]
    assert "warning:" in err and "beta.bin" in err

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(bridge_assets.BridgeAssetError, match="neither"):
        bridge_assets.stage_from_dir(empty, bundle, tmp_path / "dest2",
                                     progress=lambda _line: None)


def test_from_dir_refuses_a_bundle_whose_own_bytes_drifted(tmp_path):
    """``--from`` verifies the archive itself, not only its members."""

    archive, bundle = _synthetic_bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"appended")

    with pytest.raises(bridge_assets.BridgeAssetError, match="bytes"):
        bridge_assets.stage_from_dir(tmp_path, bundle, tmp_path / "dest",
                                     progress=lambda _line: None)


# ---------------------------------------------------------------------------
# Download: resume, restart, and the size gate
# ---------------------------------------------------------------------------

class _Response(io.BytesIO):
    def __init__(self, payload: bytes, status: int):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class _RangeServer:
    """A urlopen stand-in that either honours Range or ignores it.

    Two values of one dimension, because the resume path has to be
    correct under both: a server that continues the transfer, and one
    (or a ``file:`` URL) that has never heard of a range and answers
    with the whole object.  Appending to a partial in the second case is
    how a naive resume produces a corrupt archive.
    """

    def __init__(self, payload: bytes, *, honour_range: bool):
        self.payload = payload
        self.honour_range = honour_range
        self.requests: list[str | None] = []

    def __call__(self, request):
        header = request.headers.get("Range")
        self.requests.append(header)
        if header and self.honour_range:
            start = int(header.split("=")[1].split("-")[0])
            return _Response(self.payload[start:], 206)
        return _Response(self.payload, 200)


@pytest.mark.parametrize("honour_range", (False, True))
def test_a_partial_download_finishes_correctly_either_way(tmp_path,
                                                          honour_range):
    payload = os.urandom(64 * 1024)
    dest = tmp_path / "bundle.zip"
    dest.write_bytes(payload[:1024])            # an interrupted run
    server = _RangeServer(payload, honour_range=honour_range)
    notes: list[str] = []

    bridge_assets.download_bundle(
        "https://example.invalid/bundle.zip", dest,
        expected_bytes=len(payload), progress=notes.append,
        urlopen_fn=server)

    assert dest.read_bytes() == payload
    assert server.requests == ["bytes=1024-"]
    if honour_range:
        assert any("resuming at 1,024 B" in note for note in notes)
    else:
        assert any("ignored the resume range" in note for note in notes)


def test_a_partial_longer_than_the_pin_is_discarded(tmp_path):
    payload = os.urandom(8192)
    dest = tmp_path / "bundle.zip"
    dest.write_bytes(payload + b"extra bytes from somewhere else")
    server = _RangeServer(payload, honour_range=True)
    notes: list[str] = []

    bridge_assets.download_bundle(
        "https://example.invalid/bundle.zip", dest,
        expected_bytes=len(payload), progress=notes.append,
        urlopen_fn=server)

    assert dest.read_bytes() == payload
    assert server.requests == [None], "a discarded partial restarts at zero"
    assert any("discarded it and restarted" in note for note in notes)


def test_a_complete_partial_is_verified_rather_than_re_downloaded(tmp_path):
    payload = os.urandom(4096)
    dest = tmp_path / "bundle.zip"
    dest.write_bytes(payload)
    server = _RangeServer(payload, honour_range=True)

    bridge_assets.download_bundle(
        "https://example.invalid/bundle.zip", dest,
        expected_bytes=len(payload), progress=lambda _l: None,
        urlopen_fn=server)
    assert server.requests == [], "nothing left to download, nothing fetched"


def test_a_short_download_is_removed_and_refused(tmp_path):
    payload = os.urandom(4096)
    dest = tmp_path / "bundle.zip"
    server = _RangeServer(payload[:100], honour_range=False)

    with pytest.raises(bridge_assets.BridgeAssetError, match="expected"):
        bridge_assets.download_bundle(
            "https://example.invalid/bundle.zip", dest,
            expected_bytes=len(payload), progress=lambda _l: None,
            urlopen_fn=server)
    assert not dest.exists()


def test_a_bundle_file_this_computer_cannot_write_is_not_a_download_failure(
        tmp_path, monkeypatch):
    """The geography downloader's defect, in the bridge downloader too.

    An ``OSError`` from opening the local file reached the network
    handler and read "download failed from <url>: ...; the partial file
    is kept and a re-run resumes" with nothing written and the server
    fine.
    """

    payload = os.urandom(4096)
    dest = tmp_path / "bundle.zip"
    refused = "C:\\" + "b" * 269                 # 272 characters
    real_open = Path.open

    def windows_refuses(self, mode="r", *args, **kwargs):
        if self == dest and "b" in mode and ("w" in mode or "a" in mode):
            raise FileNotFoundError(2, "No such file or directory", refused)
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", windows_refuses)
    monkeypatch.setattr(bridge_assets.fetch_guard, "windows_path_limit",
                        lambda: 259)
    with pytest.raises(bridge_assets.BridgeAssetError) as caught:
        bridge_assets.download_bundle(
            "https://example.invalid/bundle.zip", dest,
            expected_bytes=len(payload), progress=lambda _l: None,
            urlopen_fn=_RangeServer(payload, honour_range=True))
    monkeypatch.setattr(Path, "open", real_open)

    message = str(caught.value)
    assert "download failed" not in message
    assert "resumes" not in message
    assert "272 characters" in message
    assert "13 characters shorter" in message
    assert "nothing was downloaded" in message


class _DroppedResponse(_Response):
    def read(self, size=-1):
        if self.tell():
            raise ConnectionResetError(104, "Connection reset by peer")
        return super().read(100)


@pytest.mark.parametrize("sent", (0, 100))
def test_a_failed_transfer_promises_a_resume_only_with_bytes_to_resume(
        tmp_path, sent):
    payload = os.urandom(4096)
    dest = tmp_path / "bundle.zip"

    def server(request):
        if not sent:
            raise bridge_assets.URLError("connection refused")
        return _DroppedResponse(payload, 200)

    with pytest.raises(bridge_assets.BridgeAssetError) as caught:
        bridge_assets.download_bundle(
            "https://example.invalid/bundle.zip", dest,
            expected_bytes=len(payload), progress=lambda _l: None,
            urlopen_fn=server)

    message = str(caught.value)
    assert "download failed from https://example.invalid/" in message
    if sent:
        assert "the partial file (100 B) is kept and a re-run resumes" in (
            message)
    else:
        assert "resumes" not in message
        assert "nothing was downloaded" in message


def test_a_bundle_whose_bytes_differ_never_reaches_staging(tmp_path):
    """The whole download path, against a served archive that lies."""

    archive, bundle = _synthetic_bundle(tmp_path)
    evil = bytearray(archive.read_bytes())
    evil[-1] ^= 0xFF
    server = _RangeServer(bytes(evil), honour_range=False)
    dest = tmp_path / "dest"
    pins = bridge_assets.BridgePins(release="v0-test",
                                    platforms={bundle.platform: bundle})

    with pytest.raises(bridge_assets.BridgeAssetError, match="SHA-256"):
        bridge_assets.fetch_bundle(pins, bundle, dest,
                                   progress=lambda _l: None,
                                   urlopen_fn=server)
    assert not (dest / "alpha.bin").exists()
    assert not (dest / bridge_assets.ARCHIVE_SUBDIR
                / bundle.filename).exists()


def test_a_verified_bundle_already_on_disk_is_not_downloaded_again(tmp_path):
    archive, bundle = _synthetic_bundle(tmp_path)
    dest = tmp_path / "dest"
    cache = dest / bridge_assets.ARCHIVE_SUBDIR
    cache.mkdir(parents=True)
    (cache / bundle.filename).write_bytes(archive.read_bytes())
    server = _RangeServer(b"", honour_range=False)
    notes: list[str] = []

    installed = bridge_assets.fetch_bundle(
        pins := bridge_assets.BridgePins(
            release="v0-test", platforms={bundle.platform: bundle}),
        bundle, dest, keep_bundle=True, progress=notes.append,
        urlopen_fn=server)
    assert pins.release == "v0-test"
    assert len(installed) == len(bundle.binaries)
    assert server.requests == []
    assert any("already downloaded and pin-verified" in note
               for note in notes)


def test_the_url_base_environment_override_wins(monkeypatch, tmp_path):
    _archive, bundle = _synthetic_bundle(tmp_path)
    pins = bridge_assets.BridgePins(release="v0-test",
                                    platforms={bundle.platform: bundle})
    assert bridge_assets.bundle_url(pins, bundle).startswith(
        f"{bridges.REPOSITORY_URL}/releases/download/v0-test/")
    monkeypatch.setenv(bridge_assets.ASSET_URL_BASE_ENV,
                       "https://mirror.invalid/woof/")
    assert bridge_assets.bundle_url(pins, bundle) == (
        f"https://mirror.invalid/woof/{bundle.filename}")


# ---------------------------------------------------------------------------
# The pins document
# ---------------------------------------------------------------------------

def test_the_packaged_pins_document_parses():
    """Whatever the release wrote, the runtime must be able to read.

    It is legitimate for a tree that has not been through a release cut
    to declare no platform -- that is what an unpinned document says,
    rather than carrying a hash nobody computed -- but the schema, and
    every field it does declare, are checked either way.
    """

    pins = bridge_assets.load_pins()
    for platform, bundle in pins.platforms.items():
        assert platform in bridge_assets.SUPPORTED_PLATFORMS
        assert pins.release, "a pinned platform needs a release to fetch from"
        expected = {
            bridge_assets.artifact_filename(artifact, platform)
            for artifact in bridge_assets.BUNDLED_ARTIFACTS}
        assert {pin.filename for pin in bundle.binaries} == expected


@pytest.mark.parametrize("mutation,pattern", (
    ({"schema": "something-else"}, "schema"),
    ({"platforms": []}, "platforms must be an object"),
    ({"release": ""}, "release must be"),
))
def test_a_malformed_pins_document_is_refused(mutation, pattern):
    payload = {"schema": bridge_assets.PINS_SCHEMA, "release": None,
               "platforms": {}}
    payload.update(mutation)
    with pytest.raises(bridge_assets.BridgeAssetError, match=pattern):
        bridge_assets.parse_pins(payload)


def test_an_unknown_platform_entry_is_skipped_with_a_warning(capsys):
    """Warn-not-block: a pins document from a newer release may pin a
    platform this build does not know; the entry is irrelevant to this
    host and is skipped, not fatal."""
    payload = {"schema": bridge_assets.PINS_SCHEMA, "release": None,
               "platforms": {"solaris-sparc": {}}}
    pins = bridge_assets.parse_pins(payload)
    err = capsys.readouterr().err
    assert "warning:" in err and "solaris-sparc" in err
    assert pins.platforms == {}


def test_a_pinned_platform_without_a_release_is_refused():
    """Pins with nowhere to download from are not pins."""

    payload = {
        "schema": bridge_assets.PINS_SCHEMA, "release": None,
        "platforms": {"win-x86_64": {
            "bundle": {"filename": "b.zip", "bytes": 1, "sha256": "0" * 64},
            "binaries": [{"artifact": "grib1_bridge",
                          "filename": "grib1_bridge.exe",
                          "bytes": 1, "sha256": "0" * 64}]}}}
    with pytest.raises(bridge_assets.BridgeAssetError, match="release"):
        bridge_assets.parse_pins(payload)


def test_a_hash_that_is_not_a_hash_is_refused():
    payload = {
        "schema": bridge_assets.PINS_SCHEMA, "release": "v0",
        "platforms": {"win-x86_64": {
            "bundle": {"filename": "b.zip", "bytes": 1,
                       "sha256": "not-a-digest"},
            "binaries": [{"artifact": "grib1_bridge",
                          "filename": "grib1_bridge.exe",
                          "bytes": 1, "sha256": "0" * 64}]}}}
    with pytest.raises(bridge_assets.BridgeAssetError, match="sha256"):
        bridge_assets.parse_pins(payload)


# ---------------------------------------------------------------------------
# Platform detection is a capability check
# ---------------------------------------------------------------------------

def test_the_platform_key_names_an_os_and_an_architecture(monkeypatch):
    """Both halves matter, and neither is about who is asking.

    An aarch64 Linux box and an x86-64 Linux box run different bytes,
    and so do the same architecture on Windows and on Linux.  A key that
    dropped either half would hand someone a bundle their machine
    cannot execute.
    """

    cases = {
        ("win32", "AMD64"): "win-x86_64",
        ("win32", "ARM64"): None,
        ("linux", "x86_64"): "linux-x86_64",
        ("linux", "aarch64"): None,
        ("darwin", "arm64"): None,
    }
    for (platform_name, machine), expected in cases.items():
        monkeypatch.setattr(bridge_assets.sys, "platform", platform_name)
        monkeypatch.setattr(bridge_assets.platform_module, "machine",
                            lambda m=machine: m)
        assert bridge_assets.host_platform() == expected, (
            f"{platform_name}/{machine}")


def test_artifact_filenames_agree_with_the_resolvers_on_this_host():
    """The bundle's names and the ladder's names cannot disagree.

    ``artifact_filename`` is parametric in the platform so the release
    tool can inspect both bundles from one machine; on the host it must
    produce exactly what :mod:`woof.bridges` and the CPU backend look
    for, or a perfectly staged bundle resolves to nothing.
    """

    from woof.ingest.cpu_backend import cpu_bridge_candidates

    host = bridge_assets.host_platform()
    if host is None:
        pytest.skip("no bundle platform for this host")
    from woof.io.nc_writer_bridge import library_names as ncwrite_names
    from woof.obs.dealias_region import library_name as region_library_name
    from woof.static.rust_bridge import library_names as static_names
    from woof.obs_regrid_bridge import (
        library_names as obsregrid_names)
    from woof.obs_score_bridge import library_names as obsscore_names
    from woof.isobaric_bridge import library_names as isobaric_names
    from woof.obs.superob_bridge import library_name as superob_name
    from woof.core.tempo_rust import library_name as tempo_name
    from woof.core.p3_rust import library_name as p3_name
    from woof.core.cam3_tables import library_name as cam_name

    from woof.core.cam3_tables import library_name as cam_tables_name
    from woof.core.dfi_runtime import library_name as dfi_name
    from woof.tslist_bridge import library_name as tslist_name
    from woof.core.cu99_launch import library_name as cu99_launch_name
    from woof.tc.mapped_vortex import library_name as tc_frames_name
    from woof.core.dfi_runtime import library_name as dfi_name
    from woof.tslist_bridge import library_name as tslist_name
    from woof.hydrofabric import library_names as hydrofabric_names
    from woof.hydro.forcing_inputs import library_names as forcing_names

    # One expected filename per library, from the resolver that actually
    # searches for it: libraries with one shared expectation would
    # pass while a bundle staged a file nothing looks for.
    libraries = {"arwen_dfi": dfi_name(), "gpuwm_tslist": tslist_name(),
                 "nextgen_hydrofabric": hydrofabric_names()[0],
                 "arwen_nextgen_forcing": forcing_names()[0],
                 "gpuwm_preprocess_cpu": cpu_bridge_candidates()[-1].name,
                 "region_global_dealias": region_library_name(),
                 "netcdf_writer": ncwrite_names()[0],
                 "static_fields": static_names()[0],
                 "obs_regrid": obsregrid_names()[0],
                 "obs_score": obsscore_names()[0],
                 "rw_isobaric": isobaric_names()[0],
                 "rw_superob": superob_name(),
                 "rw_tempo": tempo_name(),
                 "rw_p3": p3_name(),
                 "rw_cam_tables": cam_name(),

                 "rw_cam_tables": cam_tables_name(),
                 "arwen_dfi": dfi_name(),
                 "gpuwm_tslist": tslist_name(),
                 "rw_cu99_launch": cu99_launch_name(),
                 "tc_frames_io": tc_frames_name(),
                 # Resolved by woof hex (woof.hex.hrrr_intermediate), which
                 # spells it this way on each platform.
                 "rw_projected_intermediate": (
                     "rw_projected_intermediate.dll" if os.name == "nt" else
                     "librw_projected_intermediate.dylib" if sys.platform == "darwin" else
                     "librw_projected_intermediate.so"),
                 # woof.hydro.output._library_path spells it this way.
                 "rw_nextgen_output": (
                     "rw_nextgen_output.dll" if os.name == "nt" else
                     "librw_nextgen_output.dylib" if sys.platform == "darwin" else
                     "librw_nextgen_output.so")}
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        produced = bridge_assets.artifact_filename(artifact, host)
        if artifact.kind == "library":
            assert produced == libraries[artifact.name]
        else:
            assert produced == bridges.executable_name(artifact.name)


def test_both_platforms_spell_the_library_the_way_that_platform_does():
    """One dimension, two values: the extension is not a suffix guess."""

    library = next(a for a in bridge_assets.BUNDLED_ARTIFACTS
                   if a.kind == "library")
    executable = next(a for a in bridge_assets.BUNDLED_ARTIFACTS
                      if a.kind == "executable")
    assert bridge_assets.artifact_filename(library, "win-x86_64") == (
        f"{library.name}.dll")
    assert bridge_assets.artifact_filename(library, "linux-x86_64") == (
        f"lib{library.name}.so")
    assert bridge_assets.artifact_filename(executable, "win-x86_64") == (
        f"{executable.name}.exe")
    assert bridge_assets.artifact_filename(executable, "linux-x86_64") == (
        executable.name)


def test_every_bundled_artifact_is_one_the_resolver_searches_for():
    """A bundle must not carry a file nothing looks for, or miss one.

    The five decoders come from doctor's own consumer table, and the
    other four are the artifacts with their own resolution modules.
    """

    from woof import doctor, rustwx, rustwx_fetch, zarr_bridge
    from woof.ingest.cpu_backend import CPU_BRIDGE_ENV
    from woof.obs import dealias_region

    names = [artifact.name for artifact in bridge_assets.BUNDLED_ARTIFACTS]
    assert set(bridges.BRIDGE_ENV) <= set(names)
    assert set(doctor._BRIDGE_CONSUMERS) <= set(names)
    assert rustwx.RENDERER_NAME in names
    assert rustwx_fetch.FETCH_NAME in names
    envs = {artifact.name: artifact.env_var
            for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert envs[rustwx.RENDERER_NAME] == rustwx.RENDERER_ENV
    assert envs[rustwx_fetch.FETCH_NAME] == rustwx_fetch.FETCH_ENV
    assert envs["gpuwm_preprocess_cpu"] == CPU_BRIDGE_ENV
    # The dealiasing library's two strings are spelled literally in
    # bridge_assets so `woof fetch-bridges` does not import the
    # observation stack; this is the binding that keeps them true.
    assert envs["region_global_dealias"] == \
        dealias_region.REGION_DEALIAS_ENV
    crates = {artifact.name: artifact.crate
              for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert crates["region_global_dealias"] == dealias_region.CRATE_RELATIVE
    assert envs["rw_zarr"] == zarr_bridge.BRIDGE_ENV
    assert crates["rw_zarr"] == zarr_bridge.CRATE_RELATIVE
    assert bridges.BRIDGE_ABI_MARKERS["rw_zarr"] == zarr_bridge.ABI_MARKER
    # And the filename the bundle stages is the one the ladder opens.
    staged = bridge_assets.artifact_filename(
        next(a for a in bridge_assets.BUNDLED_ARTIFACTS
             if a.name == "region_global_dealias"),
        "win-x86_64" if os.name == "nt" else "linux-x86_64")
    assert staged == dealias_region.library_name()
    assert bridges.default_bridge_dir() / staged in set(
        dealias_region.region_bridge_candidates())
    for name, env in bridges.BRIDGE_ENV.items():
        assert envs[name] == env


def test_ensemble_bundle_contract_matches_the_default_cpu_reducer():
    from woof import rustwx

    artifact = next(row for row in bridge_assets.BUNDLED_ARTIFACTS
                    if row.name == "rw_ensbatch")
    assert artifact.kind == "executable"
    assert artifact.crate == bridges.RUSTWX_CRATE_RELATIVE
    assert artifact.env_var == "WOOF_ENSEMBLE_RENDERER"
    assert bridges.BRIDGE_ABI_MARKERS[artifact.name] == (
        rustwx.CPU_ENSEMBLE_REDUCTION_ABI.encode("ascii"))
    # The thin launcher must reject the same old reducer as staging does.
    contracts = Path("tools/rustwx/crates/rw-wrfbatch/src/cli_contracts.rs").read_text()
    marker = rustwx.CPU_ENSEMBLE_REDUCTION_ABI.replace("\t", r"\t")
    assert f'pub const ENSBATCH_STATIC: &str = "{marker}";' in contracts
    assert bridge_assets.artifact_filename(artifact, "linux-x86_64") == "rw_ensbatch"
    assert bridge_assets.artifact_filename(artifact, "win-x86_64") == "rw_ensbatch.exe"


def test_bundle_with_every_other_artifact_refuses_missing_ensemble_reducer(tmp_path):
    archive = tmp_path / "gpuwm-bridges-v0.0.0-test-linux-x86_64.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            if artifact.name != "rw_ensbatch":
                bundle.writestr(bridge_assets.artifact_filename(artifact, "linux-x86_64"),
                                b"member-list fixture, not executable proof")
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", "v0.0.0-test",
         "--source-rev", "ab12" * 10, "--bundle", str(archive),
         "--out", str(tmp_path / "pins.json")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "missing rw_ensbatch; refusing to pin a partial bundle" in (
        result.stderr + result.stdout)
    assert not (tmp_path / "pins.json").exists()


def test_missing_zarr_reader_names_the_actual_install_or_source_build_remedy(tmp_path, monkeypatch):
    from woof import zarr_bridge
    monkeypatch.delenv(zarr_bridge.BRIDGE_ENV, raising=False)
    monkeypatch.setattr(zarr_bridge, "__file__", str(tmp_path / "woof/zarr_bridge.py"))
    monkeypatch.setattr(zarr_bridge, "packaged_bridge_dir", lambda: tmp_path / "package")
    monkeypatch.setattr(zarr_bridge, "default_bridge_dir", lambda: tmp_path / "home")
    with pytest.raises(FileNotFoundError) as error:
        zarr_bridge.resolve_zarr_bin()
    message = str(error.value)
    assert "rw_zarr" in message and "zarr_bridge" in message
    assert "cargo build --release --locked --offline" in message


# ---------------------------------------------------------------------------
# The command: refusals a user actually meets
# ---------------------------------------------------------------------------

def test_an_unsupported_platform_is_told_so_and_sent_to_the_build(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(bridge_assets.sys, "platform", "darwin")
    monkeypatch.setattr(bridge_assets.platform_module, "machine",
                        lambda: "arm64")
    assert bridge_assets.fetch_bridges_main(_args(dest=str(tmp_path))) == 2
    out = capsys.readouterr().out
    assert "darwin/arm64" in out
    assert "build the artifacts from a clone" in out
    assert not bridge_assets.staging_available()


def test_a_tree_with_no_pins_refuses_instead_of_inventing_one(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(bridge_assets, "load_pins",
                        lambda path=None: bridge_assets.BridgePins(
                            release=None, platforms={}))
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: "linux-x86_64")
    assert bridge_assets.fetch_bridges_main(_args(dest=str(tmp_path))) == 2
    assert "carries no bundle pins for linux-x86_64" in capsys.readouterr().out
    assert not bridge_assets.staging_available()


#: The pointer the no-pins refusal used to end on.  ``woof doctor``'s
#: own next command for a missing bridge is ``woof fetch-bridges``
#: (and so is ``woof setup``, which runs it), so a reader who did as
#: they were told arrived back at this refusal having learned nothing.
#: Pinned as a literal so the loop cannot be reintroduced by wording.
_CIRCULAR_POINTER = "prints the exact steps for this install"


def _no_pins(monkeypatch, *, platform: str, sources: bool) -> None:
    """A pins document declaring no platforms, on a chosen install shape."""

    monkeypatch.setattr(bridge_assets, "load_pins",
                        lambda path=None: bridge_assets.BridgePins(
                            release=None, platforms={}))
    monkeypatch.setattr(bridge_assets, "host_platform", lambda: platform)
    monkeypatch.setattr(bridge_assets.bridges, "sources_present",
                        lambda *args, **kwargs: sources)


def test_the_no_pins_refusal_names_the_wheel_as_what_carries_them(
        monkeypatch, capsys, tmp_path):
    """A wheel install with no pins is told where pins come from.

    The pins are computed from the published bytes at release time, so a
    wheel installed from PyPI is the only artifact that carries them.
    Without that sentence the reader has a command that refuses for a
    reason nothing on their disk shows, and no way to tell "my install
    is broken" from "this install was never going to have them".
    """

    _no_pins(monkeypatch, platform="linux-x86_64", sources=False)
    assert bridge_assets.fetch_bridges_main(_args(dest=str(tmp_path))) == 2
    out = capsys.readouterr().out
    assert "carries no bundle pins for linux-x86_64" in out
    assert "PyPI" in out
    assert "pip install --upgrade recast-woof" in out


def test_the_no_pins_refusal_names_the_breakage_and_stops_pointing_at_gpuwm(
        monkeypatch, capsys, tmp_path):
    _no_pins(monkeypatch, platform="linux-x86_64", sources=False)
    assert bridge_assets.fetch_bridges_main(_args(dest=str(tmp_path))) == 2
    out = capsys.readouterr().out
    assert _CIRCULAR_POINTER not in out
    # The breakage, named: what stays broken while the artifacts are
    # absent, rather than "run this other command".
    assert "woof prep" in out and "woof render" in out
    # A remedy is commands the reader can run, not a command that
    # prints commands.
    assert "git clone" in out


def test_the_no_pins_refusal_in_a_checkout_leads_with_the_build(
        monkeypatch, capsys, tmp_path):
    """A checkout must not be told to pip-install over itself.

    Its sources are right there, so the build is the shorter true
    answer; the wheel is named as the reason the pins are absent, not
    offered as the step to take.
    """

    _no_pins(monkeypatch, platform="linux-x86_64", sources=True)
    assert bridge_assets.fetch_bridges_main(_args(dest=str(tmp_path))) == 2
    out = capsys.readouterr().out
    assert bridges.cargo_build_one_liner() in out
    assert "pip install --upgrade recast-woof" not in out
    assert _CIRCULAR_POINTER not in out


def test_the_unsupported_platform_refusal_prints_the_steps_it_used_to_defer(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(bridge_assets.sys, "platform", "darwin")
    monkeypatch.setattr(bridge_assets.platform_module, "machine",
                        lambda: "arm64")
    monkeypatch.setattr(bridge_assets.bridges, "sources_present",
                        lambda *args, **kwargs: False)
    assert bridge_assets.fetch_bridges_main(_args(dest=str(tmp_path))) == 2
    out = capsys.readouterr().out
    assert _CIRCULAR_POINTER not in out
    assert "cargo build --release --locked --offline" in out
    # No published wheel helps here, and the text must not imply one
    # would: the bundles are per OS/architecture.
    assert "pip install --upgrade recast-woof" not in out


def test_a_corrupt_bundle_makes_the_command_exit_two(tmp_path, capsys,
                                                     monkeypatch):
    archive, bundle = _synthetic_bundle(tmp_path, corrupt="beta.bin")
    _install_pins(monkeypatch, bundle)

    dest = tmp_path / "dest"
    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(tmp_path))) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert not (dest / "beta.bin").exists()


def test_the_listing_reports_state_without_touching_the_network(
        tmp_path, capsys, monkeypatch):
    archive, bundle = _synthetic_bundle(tmp_path)
    _install_pins(monkeypatch, bundle)
    dest = tmp_path / "dest"
    dest.mkdir()
    with zipfile.ZipFile(archive) as zf:
        (dest / "alpha.bin").write_bytes(zf.read("alpha.bin"))

    def _no_network(*_args, **_kwargs):
        raise AssertionError("--list must not open a connection")

    # `urlopen`, not `_default_urlopen`: the latter is bound as a default
    # argument at definition time, so patching it is a guard that cannot
    # fire.  This one is looked up when the call happens.
    monkeypatch.setattr(bridge_assets, "urlopen", _no_network)
    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), list=True)) == 0
    out = capsys.readouterr().out
    assert "staged   alpha.bin" in out
    assert "needed   beta.bin" in out
    assert "1 of 2 artifacts already staged" in out


def test_a_set_environment_override_is_reported_not_silently_shadowed(
        tmp_path, capsys, monkeypatch):
    _archive, raw = _synthetic_bundle(
        tmp_path, names=("grib1_bridge.exe", "rw_fetch.exe"))
    bundle = bridge_assets.BundlePin(
        platform=raw.platform, filename=raw.filename, bytes=raw.bytes,
        sha256=raw.sha256,
        binaries=(
            bridge_assets.BinaryPin("grib1_bridge", "grib1_bridge.exe",
                                    raw.binaries[0].bytes,
                                    raw.binaries[0].sha256),
            bridge_assets.BinaryPin("rw_fetch", "rw_fetch.exe",
                                    raw.binaries[1].bytes,
                                    raw.binaries[1].sha256)))
    _install_pins(monkeypatch, bundle)
    monkeypatch.setenv(bridges.BRIDGE_ENV["grib1_bridge"],
                       str(tmp_path / "elsewhere.exe"))
    # This synthetic payload carries no contract marker; that check has
    # its own test below, against a real build.
    monkeypatch.setattr(bridge_assets, "verify_contract_marker",
                        lambda artifact, path: None)

    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(tmp_path / "dest"), from_dir=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert bridges.BRIDGE_ENV["grib1_bridge"] in out
    assert "wins over the staged copy" in out


def test_the_cli_dispatches_fetch_bridges_for_real(monkeypatch, capsys,
                                                   tmp_path):
    """Through woof.cli.main, not just --help.

    fetch-tables once passed a --help-only probe while the real dispatch
    raised AttributeError; the same dispatch table gained an entry here.
    """

    from woof.cli import main

    monkeypatch.setattr(bridge_assets.sys, "platform", "darwin")
    monkeypatch.setattr(bridge_assets.platform_module, "machine",
                        lambda: "arm64")
    assert main(["fetch-bridges", "--dest", str(tmp_path)]) == 2
    assert "no bundle is published for" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The release tool
# ---------------------------------------------------------------------------

def test_the_bundle_tool_refuses_to_pin_a_partial_bundle(tmp_path):
    """A bundle missing an artifact must not become a wheel's pins."""

    archive = tmp_path / "gpuwm-bridges-v0.0.0-test-linux-x86_64.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("grib1_bridge", b"only one of nine")
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", "v0.0.0-test",
         "--source-rev", "ab12" * 10,
         "--bundle", str(archive), "--out", str(tmp_path / "pins.json")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "refusing to pin a partial bundle" in (
        result.stderr + result.stdout)
    assert not (tmp_path / "pins.json").exists()


def test_the_bundle_tool_refuses_a_bundle_named_for_another_release(tmp_path):
    archive = tmp_path / "gpuwm-bridges-v9.9.9-linux-x86_64.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("grib1_bridge", b"payload")
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", "v0.0.0-test",
         "--source-rev", "ab12" * 10,
         "--bundle", str(archive), "--out", str(tmp_path / "pins.json")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "is not a bundle name for release" in (
        result.stderr + result.stdout)


@pytest.mark.parametrize("platform", bridge_assets.SUPPORTED_PLATFORMS)
def test_the_bundle_tool_writes_a_deterministic_archive(tmp_path, platform):
    """Two packs of the same bytes produce the same archive, per platform.

    A release asset whose hash depends on when it was zipped cannot be
    re-derived by anyone auditing the pins later.
    """

    source = tmp_path / "src"
    source.mkdir()
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        name = bridge_assets.artifact_filename(artifact, platform)
        (source / name).write_bytes(name.encode() * 64)
    digests = []
    for run in ("a", "b"):
        out = tmp_path / run
        result = subprocess.run(
            [sys.executable, str(BUNDLE_TOOL), "pack", "--release",
             "v0.0.0-test", "--platform", platform, "--search", str(source),
             "--out", str(out)],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        archive = out / f"gpuwm-bridges-v0.0.0-test-{platform}.zip"
        ensemble = next(row for row in bridge_assets.BUNDLED_ARTIFACTS
                        if row.name == "rw_ensbatch")
        name = bridge_assets.artifact_filename(ensemble, platform)
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.read(name) == (source / name).read_bytes()
        digests.append(bridge_assets.sha256_file(archive))
    assert digests[0] == digests[1]


def test_the_bundle_tool_names_the_artifact_it_cannot_find(tmp_path, monkeypatch):
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pack", "--release", "v0.0.0-test",
         "--platform", "linux-x86_64", "--search", str(tmp_path),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert result.returncode != 0
    first = bridge_assets.BUNDLED_ARTIFACTS[0]
    assert bridge_assets.artifact_filename(first, "linux-x86_64") in (
        result.stderr + result.stdout)

    # Fixture discovery must find every crate the registry now bundles,
    # and a missing file must still make that set incomplete. Keep the
    # user-level fallback empty so no installed binary can mask the hole.
    root = tmp_path / "checkout"
    monkeypatch.setitem(globals(), "REPO_ROOT", root)
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: tmp_path / "empty")
    monkeypatch.setattr(bridge_assets, "host_platform", lambda: "linux-x86_64")
    built = []
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        path = (root / artifact.crate / "target" / "release"
                / bridge_assets.artifact_filename(artifact, "linux-x86_64"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"discovery fixture, not executable proof")
        built.append(path)
    assert set(_built_artifact_dirs()) == {path.parent for path in built}
    for path in built:
        contents = path.read_bytes()
        path.unlink()
        assert _built_artifact_dirs() is None, path.name
        path.write_bytes(contents)


# ---------------------------------------------------------------------------
# Integration: a real bundle, real pins, the real release tool
# ---------------------------------------------------------------------------

def _built_artifact_dirs() -> list[Path] | None:
    """Directories holding this machine's own built Rust artifacts.

    A checkout that has built every crate the bundle registry names,
    or the user-level directory a previous ``fetch-bridges``
    filled.  None means the machine simply has nothing to pack, which is
    a skip rather than a failure.
    """

    host = bridge_assets.host_platform()
    if host is None:
        return None
    wanted = [bridge_assets.artifact_filename(artifact, host)
              for artifact in bridge_assets.BUNDLED_ARTIFACTS]
    groups = [
        list(dict.fromkeys(REPO_ROOT / artifact.crate / "target" / "release"
                           for artifact in bridge_assets.BUNDLED_ARTIFACTS)),
        [bridges.default_bridge_dir()],
    ]
    for group in groups:
        if all(any((directory / name).is_file() for directory in group)
               for name in wanted):
            return group
    return None


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory):
    """Pack and pin a bundle out of this machine's own built artifacts.

    The release tool, run exactly as the release workflow runs it, over
    bytes a Rust compiler on this box produced.  Every pin is computed
    here; nothing is transcribed.
    """

    directories = _built_artifact_dirs()
    if directories is None:
        pytest.skip("no complete set of built Rust artifacts on this machine")
    host = bridge_assets.host_platform()

    # ``pin`` refuses any binary whose embedded GPUWM_BRIDGE_SOURCE_REV
    # stamp does not name the revision being released, so the revision
    # these artifacts really were built from is read out of their own
    # bytes.  Artifacts from before the stamped builds prove nothing and
    # skip; artifacts from two different revisions are exactly the
    # mixed-revision bundle the release check exists to refuse, and no
    # accurate --source-rev exists for them.
    revisions: set[str] = set()
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        if artifact.vendored:
            # The unchanged pin command below verifies its declared ABI and
            # bytes. A verbatim upstream crate carries no checkout HEAD stamp.
            continue
        name = bridge_assets.artifact_filename(artifact, host)
        path = next(directory / name for directory in directories
                    if (directory / name).is_file())
        found = bridge_assets.embedded_source_revisions(path.read_bytes())
        if not found:
            pytest.skip(f"{path} predates the stamped bridge builds; "
                        "rebuild with cargo build --release to run the "
                        "real-bundle integration tests")
        revisions.update(found)
    if len(revisions) != 1:
        pytest.skip("this machine's built artifacts are from "
                    f"{len(revisions)} different source revisions "
                    f"({', '.join(sorted(revisions))}); rebuild all "
                    "WOOF-authored workspaces from one checkout")
    source_rev = revisions.pop()
    # A user-level directory an older checkout filled holds binaries that
    # predate a contract this tree has since changed; staging refuses them
    # by their missing marker, so they are not a bundle for this release
    # and prove nothing here.  (A 2.8 build host's ~/.woof/bridges held
    # an rw_zarr from before its units contract, and this fixture packed it.)
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        marker = bridges.BRIDGE_ABI_MARKERS.get(artifact.name)
        if marker is None:
            continue
        name = bridge_assets.artifact_filename(artifact, host)
        path = next(directory / name for directory in directories
                    if (directory / name).is_file())
        if marker not in path.read_bytes():
            pytest.skip(f"{path} predates this tree's {artifact.name} contract; "
                        "rebuild the bridges from this checkout to run the "
                        "real-bundle integration tests")

    work = tmp_path_factory.mktemp("bridge-bundle")
    release = "v0.0.0-test"
    pack = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pack", "--release", release,
         "--platform", host, "--out", str(work)]
        + [arg for directory in directories
           for arg in ("--search", str(directory))],
        capture_output=True, text=True)
    assert pack.returncode == 0, pack.stderr
    archive = work / f"gpuwm-bridges-{release}-{host}.zip"
    assert archive.is_file()
    pins_path = work / "bridge-pins.json"
    manifest_path = work / "bridge-bundle-manifest.json"
    pin = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", release,
         "--source-rev", source_rev,
         "--bundle", str(archive), "--out", str(pins_path),
         "--manifest", str(manifest_path)],
        capture_output=True, text=True)
    assert pin.returncode == 0, pin.stderr
    return {"work": work, "archive": archive, "pins": pins_path,
            "manifest": manifest_path, "platform": host, "release": release}


def test_the_release_tool_produces_pins_the_runtime_accepts(real_bundle):
    pins = bridge_assets.load_pins(real_bundle["pins"])
    assert pins.release == real_bundle["release"]
    bundle = pins.bundle_for(real_bundle["platform"])
    assert bundle is not None
    assert len(bundle.binaries) == len(bridge_assets.BUNDLED_ARTIFACTS)
    # The pins describe the archive that exists, byte for byte.
    bridge_assets.verify_pinned_file(
        real_bundle["archive"], expected_bytes=bundle.bytes,
        expected_sha256=bundle.sha256, label=bundle.filename)
    manifest = json.loads(real_bundle["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema"] == bridge_assets.BUNDLE_MANIFEST_SCHEMA
    assert manifest["release"] == real_bundle["release"]
    assert real_bundle["platform"] in manifest["platforms"]


def test_a_real_bundle_stages_and_the_binaries_execute(real_bundle, tmp_path,
                                                       capsys, monkeypatch):
    """The artifact, not the mock: stage it, then run what was staged.

    ``--from`` over a bundle the release tool packed, staged into a
    scratch directory, and then probed with doctor's own execution check
    -- the only evidence that what landed on disk is a program rather
    than the right number of bytes.
    """

    from woof import doctor

    pins = bridge_assets.load_pins(real_bundle["pins"])
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    dest = tmp_path / "dest"
    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(real_bundle["work"]))) == 0
    out = capsys.readouterr().out
    assert "matches the packaged bundle pin" in out

    bundle = pins.bundle_for(real_bundle["platform"])
    for pin in bundle.binaries:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)
    # The first EXECUTABLE: the hydro libraries now lead BUNDLED_ARTIFACTS, and a
    # shared library run as a program dies with SIGSEGV instead of a usage line.
    probe_name = bridge_assets.artifact_filename(
        next(a for a in bridge_assets.BUNDLED_ARTIFACTS if a.kind == "executable"),
        real_bundle["platform"])
    ok, evidence = doctor._exec_probe(dest / probe_name)
    assert ok, evidence

    # Idempotent: a second run touches nothing and says so.
    assert bridge_assets.fetch_bridges_main(
        _args(dest=str(dest), from_dir=str(real_bundle["work"]))) == 0
    assert "nothing to fetch" in capsys.readouterr().out


def test_a_real_bundle_with_a_flipped_byte_is_refused(real_bundle, tmp_path):
    """Corruption at file level, in a bundle that is otherwise real."""

    pins = bridge_assets.load_pins(real_bundle["pins"])
    bundle = pins.bundle_for(real_bundle["platform"])
    with zipfile.ZipFile(real_bundle["archive"]) as source:
        members = {name: source.read(name) for name in source.namelist()}
    victim = bundle.binaries[0].filename
    members[victim] = members[victim][:-1] + bytes(
        [members[victim][-1] ^ 0xFF])
    tampered = tmp_path / bundle.filename
    with zipfile.ZipFile(tampered, "w") as sink:
        for name, payload in members.items():
            sink.writestr(name, payload)

    with pytest.raises(bridge_assets.BridgeAssetError, match="SHA-256"):
        bridge_assets.stage_from_bundle(tampered, bundle, tmp_path / "dest",
                                        progress=lambda _l: None)
    assert not (tmp_path / "dest" / victim).exists()


def test_an_artifact_that_predates_the_contract_is_refused_at_staging(
        real_bundle, tmp_path):
    """The static contract marker, checked before install rather than after.

    A decoder assembled from a tree that predates a contract change
    still executes and still prints its usage line; the marker is what
    catches it.  Staging runs that check on the temporary file, so those
    bytes never become the installed copy.
    """

    pins = bridge_assets.load_pins(real_bundle["pins"])
    bundle = pins.bundle_for(real_bundle["platform"])
    marked = [pin for pin in bundle.binaries
              if pin.artifact in bridges.BRIDGE_ABI_MARKERS]
    assert marked, "no bundled artifact declares a contract marker"
    pin = marked[0]
    with zipfile.ZipFile(real_bundle["archive"]) as source:
        payload = source.read(pin.filename)
    marker = bridges.BRIDGE_ABI_MARKERS[pin.artifact]
    stripped = payload.replace(marker, b"\x00" * len(marker))
    assert stripped != payload, "the marker was not in the built binary"
    # Pin the bytes actually being staged, so the only check that can
    # fail is the contract one.
    archive = tmp_path / bundle.filename
    with zipfile.ZipFile(archive, "w") as sink:
        sink.writestr(pin.filename, stripped)
    one = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=archive.stat().st_size,
        sha256=bridge_assets.sha256_file(archive),
        binaries=(bridge_assets.BinaryPin(
            pin.artifact, pin.filename, len(stripped),
            hashlib.sha256(stripped).hexdigest()),))

    with pytest.raises(bridge_assets.BridgeAssetError, match="predates"):
        bridge_assets.stage_from_bundle(archive, one, tmp_path / "dest",
                                        progress=lambda _l: None)
    assert not (tmp_path / "dest" / pin.filename).exists()


# ---------------------------------------------------------------------------
# One opt-in live smoke
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_the_published_bundle_downloads_and_verifies(tmp_path):
    """The only test that touches GitHub, and only when asked to.

    Gated twice -- the ``network`` marker and WOOF_NETWORK_TESTS=1 --
    because it proves the one thing no offline test can: that the URL
    the packaged pins resolve to serves the bytes those pins describe.
    """

    if os.environ.get("WOOF_NETWORK_TESTS") != "1":
        pytest.skip("live download smoke needs WOOF_NETWORK_TESTS=1")
    pins = bridge_assets.load_pins()
    host = bridge_assets.host_platform()
    # Two very different states used to leave through the same skip.
    #
    # A tree that has not been through a release cut declares `release:
    # null` and no platforms at all -- that is what an unpinned document
    # is supposed to say, and there is genuinely nothing published to
    # fetch, so skipping is the accurate answer.
    #
    # A tree that HAS been cut and still has no bundle for a platform the
    # project claims to support is a release that failed to publish, and
    # skipping there reports a hole as a pass. That case fails now.
    if pins.release is None:
        pytest.skip(
            "this tree has not been through a release cut: the packaged "
            "pins declare no release and no platforms, so no published "
            "bundle exists to download")
    bundle = pins.bundle_for(host)
    if bundle is None:
        assert host not in bridge_assets.SUPPORTED_PLATFORMS, (
            f"the packaged pins are stamped for release {pins.release} but "
            f"carry no bundle for {host}, which is a supported platform "
            f"(pinned: {sorted(pins.platforms) or 'nothing'}). That is a "
            "release that did not publish what it claims to support, not a "
            "reason to skip")
        pytest.skip(f"no bundle is published for {host}")
    dest = tmp_path / "dest"
    installed = bridge_assets.fetch_bundle(pins, bundle, dest, progress=print)
    # What `fetch_bundle` stages is every pinned BINARY *and* every pinned
    # MAP ASSET -- `stage_from_bundle` appends both to the list it returns.
    # This used to assert against `bundle.binaries` alone, which was true
    # only while bundles carried no assets; on the published v2.0.0 bundle
    # it reads 9 where the accurate answer is 47.  The expectation is derived
    # from the manifest rather than written down, so a bundle that gains or
    # loses either kind moves this test with it instead of falsifying it.
    expected = [dest / pin.filename for pin in bundle.binaries]
    expected += [dest / pin.path for pin in bundle.assets]
    assert len(installed) == len(bundle.binaries) + len(bundle.assets)
    assert sorted(installed) == sorted(expected)
    for pin in bundle.binaries:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)
    for pin in bundle.assets:
        staged = dest / pin.path
        assert staged.is_file(), f"pinned map asset not staged: {pin.path}"
        assert staged.stat().st_size == pin.bytes
        assert bridge_assets.sha256_file(staged) == pin.sha256


def test_the_superob_bundle_entry_and_handshake_match_the_module():
    """The bundle stages the file this seam opens, under its own variable,
    and the release probe asks it the question this module asks: without
    the entry `woof fetch-bridges` stages a bundle the superob stage
    refuses on, and every radar DA cycle stops at observation prep."""
    from woof import bridge_assets, bridges
    from woof.obs import superob_bridge as sb

    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == "rw_superob"]
    assert artifact.env_var == sb.SUPEROB_BRIDGE_ENV
    assert artifact.kind == "library"
    assert artifact.crate == bridges.RUSTWX_CRATE_RELATIVE
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        windows = platform.startswith("win-")
        assert bridge_assets.artifact_filename(artifact, platform) == (
            "rw_superob.dll" if windows else "librw_superob.so")
    assert bridge_assets.library_abi_for("rw_superob") == (
        "gpuwm_superob_abi_version", sb.SUPEROB_ABI)
    assert bridges.BRIDGE_ABI_MARKERS["rw_superob"] == sb.ABI_MARKER


def test_the_nowcast_frames_bundle_entry_matches_its_resolver_and_source(
        monkeypatch, tmp_path):
    """The bundle stages the converter under the variable its resolver
    reads, and the static marker, the ``--abi`` literal the wrapper
    expects and the Rust constant are one string.  Without the row the
    cut built rw_nowcast_frames and threw it away, so a wheel install had
    no way to turn a NetCDF nowcast into a frames root; with a marker that
    drifted from the binary, `woof fetch-bridges` would refuse the bundle
    the release just built."""
    import re

    from woof import rustwx_lanes as lanes

    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == lanes.NOWCAST_FRAMES_NAME]
    assert artifact.env_var == lanes.NOWCAST_FRAMES_ENV
    assert artifact.kind == "executable"
    assert artifact.crate == bridges.RUSTWX_CRATE_RELATIVE
    assert (bridges.BRIDGE_ABI_MARKERS[lanes.NOWCAST_FRAMES_NAME]
            == lanes.NOWCAST_FRAMES_ABI_MARKER.encode())

    source = (REPO_ROOT / bridges.RUSTWX_CRATE_RELATIVE / "crates"
              / "rw-wrfbatch" / "src" / "cli_contracts.rs")
    if not source.is_file():
        pytest.skip("the Rust source is not in this install")
    match = re.search(r'const NOWCAST_FRAMES: &str = "((?:[^"\\]|\\.)*)";',
                      source.read_text(encoding="utf-8"), re.S)
    assert match, "rw_nowcast_frames ABI_MARKER not found"
    # Rust's string continuation: a backslash before a newline drops the
    # newline and the next line's leading whitespace.
    literal = re.sub(r"\\\n\s*", "", match.group(1)).replace("\\t", "\t")
    assert literal == lanes.NOWCAST_FRAMES_ABI_MARKER

    # The resolver honours its own variable: a file named there is found,
    # a missing one is a loud refusal rather than a fall-through.
    binary = tmp_path / bridges.executable_name(lanes.NOWCAST_FRAMES_NAME)
    binary.write_bytes(b"build " + lanes.NOWCAST_FRAMES_ABI_MARKER.encode())
    monkeypatch.setenv(lanes.NOWCAST_FRAMES_ENV, str(binary))
    assert lanes.find_nowcast_frames_bin() == binary.resolve()
    monkeypatch.setenv(lanes.NOWCAST_FRAMES_ENV, str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match=lanes.NOWCAST_FRAMES_ENV):
        lanes.find_nowcast_frames_bin()
