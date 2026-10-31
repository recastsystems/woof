"""Prepared publication keeps identities exact and retries within a real deadline.

All manifests and index responses are in-memory fixtures. No test contacts a
package index, uploads an artifact, or reads a local release packet.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest


TAG = "v9.8.7"
VERSION = "9.8.7"
COMMIT = "a" * 40
REPOSITORY = "example-org/example-repo"
PIN = "b" * 64
OTHER_PIN = "c" * 64


@pytest.fixture(scope="module")
def publication():
    path = Path(__file__).resolve().parents[1] / "tools/promote_prepared_release.py"
    spec = importlib.util.spec_from_file_location("prepared_publication_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(filename: str) -> dict:
    return {
        "filename": filename,
        "bytes": 1000 + len(filename),
        "sha256": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
    }


@pytest.fixture
def manifest():
    return {
        "schema": "arwen.publication-assets.v1",
        "status": "PREPARED_NOT_PUBLISHED",
        "engine_source_revision": COMMIT,
        "desktop_source_revision": "d" * 40,
        "github": {
            "repository": REPOSITORY,
            "target_version": TAG,
            "assets": [
                row(f"gpuwm-bridges-{TAG}-linux-x86_64.zip"),
                row(f"gpuwm-bridges-{TAG}-win-x86_64.zip"),
                row("bridge-bundle-manifest.json"),
                row("START-HERE.md"),
                row("gpuwm-carried-physics-v9.8.6-v9.8.7.json"),
                row("gpuwm-carried-physics-v9.8.6-v9.8.7.verification.json"),
            ],
            "also_attach": ["PUBLICATION-ASSETS.json", "DOWNLOAD-SHA256SUMS.txt"],
        },
        "carried_physics": {
            "schema": "arwen.carried-release-asset.v1", "release_id": "e" * 64,
            "release": row("gpuwm-carried-physics-v9.8.6-v9.8.7.json"),
            "verification": row("gpuwm-carried-physics-v9.8.6-v9.8.7.verification.json"),
        },
        "pypi": {
            "artifacts": [
                row(f"gpuwm-{VERSION}-py3-none-any.whl"),
                row(f"gpuwm-{VERSION}-py3-none-manylinux_2_28_x86_64.whl"),
                row(f"gpuwm-{VERSION}-py3-none-win_amd64.whl"),
                row(f"gpuwm-{VERSION}.tar.gz"),
                row(f"woof_data-{VERSION}-py3-none-any.whl"),
                row(f"woof_data-{VERSION}.tar.gz"),
            ]
        },
    }


@pytest.fixture
def engine_rows(manifest):
    return [item for item in manifest["pypi"]["artifacts"] if item["filename"].startswith("gpuwm-")]


def event_with_pin(pin: str = PIN) -> dict:
    return {"release": {"body": f"Release notes.\n<!-- arwen-publication-sha256: {pin} -->\n"}}


def index_payload(rows, *, project="woof", version=VERSION):
    return {
        "info": {"name": project, "version": version},
        "urls": [
            {"filename": item["filename"], "size": item["bytes"],
             "digests": {"sha256": item["sha256"]}, "yanked": False}
            for item in rows
        ],
    }


def load(publication, document):
    return publication.load_manifest(document, TAG, COMMIT, REPOSITORY)


def test_manifest_pin_accepts_explicit_or_matching_release_marker(publication):
    assert publication.manifest_pin(PIN, {}) == PIN
    assert publication.manifest_pin("", event_with_pin()) == PIN
    assert publication.manifest_pin(PIN, event_with_pin()) == PIN


@pytest.mark.parametrize("explicit,event", [
    ("", {}),
    ("", {"release": {"body": "Unpinned release notes"}}),
    (PIN[:-1], {}),
    ("z" * 64, {}),
    (PIN, event_with_pin(OTHER_PIN)),
])
def test_manifest_pin_refuses_missing_invalid_or_conflicting_identity(publication, explicit, event):
    with pytest.raises(publication.PublicationError):
        publication.manifest_pin(explicit, event)


@pytest.mark.parametrize("second_pin", [PIN, OTHER_PIN, "not-a-sha"])
def test_duplicate_release_markers_are_ambiguous_even_with_explicit_pin(publication, second_pin):
    event = event_with_pin()
    event["release"]["body"] += f"<!-- arwen-publication-sha256: {second_pin} -->"
    with pytest.raises(publication.PublicationError):
        publication.manifest_pin(PIN, event)


def test_manifest_accepts_exact_six_distribution_release(publication, manifest):
    assert isinstance(load(publication, manifest), dict)


@pytest.mark.parametrize("path,value", [
    (("schema",), "arwen.publication-assets.v0"),
    (("engine_source_revision",), "e" * 40),
    (("github", "repository"), "OtherOwner/arwen"),
    (("github", "target_version"), "v9.8.8"),
])
def test_manifest_refuses_another_schema_commit_repository_or_tag(publication, manifest, path, value):
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("section,key,missing", [
    ("pypi", "artifacts", f"gpuwm-{VERSION}-py3-none-manylinux_2_28_x86_64.whl"),
    ("pypi", "artifacts", f"gpuwm-{VERSION}-py3-none-win_amd64.whl"),
    ("pypi", "artifacts", f"woof_data-{VERSION}.tar.gz"),
    ("github", "assets", f"gpuwm-bridges-{TAG}-linux-x86_64.zip"),
    ("github", "assets", f"gpuwm-bridges-{TAG}-win-x86_64.zip"),
    ("github", "assets", "bridge-bundle-manifest.json"),
])
def test_manifest_refuses_incomplete_asset_set(publication, manifest, section, key, missing):
    manifest[section][key] = [item for item in manifest[section][key] if item["filename"] != missing]
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


def test_manifest_refuses_same_count_but_wrong_distribution_version(publication, manifest):
    manifest["pypi"]["artifacts"][0] = row("gpuwm-9.8.8-py3-none-any.whl")
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("filename", ["../escape.zip", "sub/file.zip", "sub\\file.zip", "C:\\escape.zip", "/absolute.zip", "", "bad\nname.zip"])
def test_manifest_refuses_filenames_that_are_not_safe_basenames(publication, manifest, filename):
    manifest["github"]["assets"].append(row(filename))
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("section,key", [("github", "assets"), ("pypi", "artifacts")])
def test_manifest_refuses_duplicate_rows(publication, manifest, section, key):
    manifest[section][key].append(deepcopy(manifest[section][key][0]))
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("field,value", [("bytes", 0), ("bytes", -1), ("bytes", True), ("bytes", "12"), ("sha256", "g" * 64), ("sha256", "a" * 63)])
def test_manifest_refuses_unusable_artifact_size_or_digest(publication, manifest, field, value):
    manifest["github"]["assets"][0][field] = value
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


def test_manifest_cannot_add_an_unhashed_attachment(publication, manifest):
    manifest["github"]["also_attach"].append("unproven-binary.exe")
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


def test_index_reconciliation_resumes_only_missing_exact_files(publication, engine_rows):
    assert publication.reconcile_index(engine_rows, 404, None, "woof", VERSION) == engine_rows
    partial = index_payload(engine_rows[:2])
    assert publication.reconcile_index(engine_rows, 200, partial, "woof", VERSION) == engine_rows[2:]
    assert publication.reconcile_index(engine_rows, 200, index_payload(engine_rows), "woof", VERSION) == []


@pytest.mark.parametrize("change", ["foreign", "duplicate", "size", "digest", "yanked", "project", "version"])
def test_index_reconciliation_refuses_a_conflicting_publication(publication, engine_rows, change):
    payload = index_payload(engine_rows[:1])
    if change == "foreign":
        payload["urls"].append(index_payload([row("foreign-1.0.whl")])["urls"][0])
    elif change == "duplicate":
        payload["urls"].append(deepcopy(payload["urls"][0]))
    elif change == "size":
        payload["urls"][0]["size"] += 1
    elif change == "digest":
        payload["urls"][0]["digests"]["sha256"] = OTHER_PIN
    elif change == "yanked":
        payload["urls"][0]["yanked"] = True
    else:
        payload["info"]["name" if change == "project" else "version"] = "another-release"
    with pytest.raises(publication.PublicationError):
        publication.reconcile_index(engine_rows, 200, payload, "woof", VERSION)


@pytest.mark.parametrize("status,payload", [(200, None), (200, {}), (401, None), (429, None), (500, None)])
def test_index_reconciliation_does_not_treat_errors_as_absence(publication, engine_rows, status, payload):
    with pytest.raises(publication.PublicationError):
        publication.reconcile_index(engine_rows, status, payload, "woof", VERSION)


class VirtualTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0, "retry must make forward progress"
        self.sleeps.append(seconds)
        self.now += seconds


def test_index_wait_retries_404_and_partial_200_until_exact_set(publication, engine_rows):
    timeline = VirtualTime()
    full = index_payload(engine_rows)
    responses = [(404, None), (200, index_payload(engine_rows[:1])), (200, full)]
    calls = []

    def fetch(project, version, request_timeout):
        assert (project, version) == ("woof", VERSION)
        assert 0 < request_timeout <= 20 - timeline.now
        calls.append(request_timeout)
        assert responses, "unexpected retry after a complete index"
        return responses.pop(0)

    result = publication.wait_for_index("woof", VERSION, engine_rows, fetch, timeline.clock, timeline.sleep, timeout=20)
    assert isinstance(result, dict)
    assert len(calls) == 3
    assert len(timeline.sleeps) == 2


def test_index_wait_fails_changed_bytes_without_sleeping_or_retrying(publication, engine_rows):
    timeline = VirtualTime()
    payload = index_payload(engine_rows)
    payload["urls"][0]["digests"]["sha256"] = OTHER_PIN
    calls = []

    def fetch(project, version, request_timeout):
        calls.append(request_timeout)
        assert len(calls) == 1, "a byte mismatch must not be retried"
        return 200, payload

    with pytest.raises(publication.PublicationError):
        publication.wait_for_index("woof", VERSION, engine_rows, fetch, timeline.clock, timeline.sleep, timeout=20)
    assert len(calls) == 1
    assert timeline.sleeps == []


def test_index_wait_deadline_counts_time_spent_fetching(publication, engine_rows):
    timeline = VirtualTime()
    deadline = 5.0
    calls = []

    def fetch(project, version, request_timeout):
        assert 0 < request_timeout <= deadline - timeline.now
        calls.append(request_timeout)
        assert len(calls) <= 3, "network time must count toward the deadline"
        timeline.now += min(3.0, request_timeout)
        return 404, None

    def sleep(seconds):
        assert seconds <= deadline - timeline.now, "backoff must respect remaining time"
        timeline.sleep(seconds)

    with pytest.raises(publication.PublicationError):
        publication.wait_for_index("woof", VERSION, engine_rows, fetch, timeline.clock, sleep, timeout=deadline)
    assert calls
    assert timeline.now <= deadline


def test_index_wait_bounds_a_permanently_partial_200(publication, engine_rows):
    timeline = VirtualTime()
    calls = []

    def fetch(project, version, request_timeout):
        assert 0 < request_timeout <= 4 - timeline.now
        calls.append(request_timeout)
        assert len(calls) <= 8, "partial index must not spin indefinitely"
        return 200, index_payload(engine_rows[:1])

    with pytest.raises(publication.PublicationError):
        publication.wait_for_index("woof", VERSION, engine_rows, fetch, timeline.clock, timeline.sleep, timeout=4)
    assert calls
    assert timeline.now <= 4


def test_index_wait_rejects_a_complete_response_after_its_deadline(publication, engine_rows):
    timeline = VirtualTime()

    def fetch(project, version, request_timeout):
        assert 0 < request_timeout <= 1
        timeline.now = 1.01
        return 200, index_payload(engine_rows)

    with pytest.raises(publication.PublicationError, match="deadline"):
        publication.wait_for_index("woof", VERSION, engine_rows, fetch, timeline.clock, timeline.sleep, timeout=1)
    assert timeline.sleeps == []


def metadata_fixture(tmp_path, *, project="woof", readme="Prepared release documentation.\n", archive_kind="wheel"):
    repo = tmp_path / "repo"
    project_root = repo / "recast-woof-data" if project == "recast-woof-data" else repo
    package = project.replace("-", "_")
    source_dir = project_root / package
    source_dir.mkdir(parents=True)
    source = b'"""Fixture package only."""\n'
    (source_dir / "__init__.py").write_bytes(source)
    (project_root / "README.md").write_text(readme, encoding="utf-8")
    config = f'[project]\nname = "{project}"\nreadme = "README.md"\n'
    if project == "recast-woof-data":
        config += 'dynamic = ["version"]\n[tool.setuptools.dynamic]\nversion = {file = ["woof_data/VERSION"]}\n'
        # the data distribution's import package is woof_data, where the dynamic version file lives
        (project_root / "woof_data").mkdir(exist_ok=True)
        (project_root / "woof_data" / "VERSION").write_text(VERSION + "\n", encoding="utf-8")
    else:
        config += f'version = "{VERSION}"\n'
    (project_root / "pyproject.toml").write_text(config, encoding="utf-8")
    metadata = (f"Metadata-Version: 2.4\nName: {project}\nVersion: {VERSION}\n"
                f"Description-Content-Type: text/markdown\n\n{readme}")
    if archive_kind == "wheel":
        artifact = tmp_path / f"{package}-{VERSION}-py3-none-any.whl"
        with zipfile.ZipFile(artifact, "w") as archive:
            archive.writestr(f"{package}-{VERSION}.dist-info/METADATA", metadata.encode("utf-8"))
            archive.writestr(f"{package}/__init__.py", source)
    else:
        artifact = tmp_path / f"{package}-{VERSION}.tar.gz"
        with tarfile.open(artifact, "w:gz") as archive:
            for name, payload in [("PKG-INFO", metadata.encode("utf-8")), (f"{package}/__init__.py", source)]:
                member = tarfile.TarInfo(f"{package}-{VERSION}/{name}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    return repo, artifact


def test_metadata_accepts_companion_dynamic_version_file(publication, tmp_path):
    repo, wheel = metadata_fixture(tmp_path, project="recast-woof-data")
    result = publication.check_metadata(wheel, "recast-woof-data", VERSION, repo)
    assert result["version"] == VERSION


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_metadata_compares_utf8_readme_without_replacement_characters(publication, tmp_path, archive_kind):
    repo, artifact = metadata_fixture(tmp_path, readme="# WOOF\n\nWeather in °F, native maps.\n", archive_kind=archive_kind)
    result = publication.check_metadata(artifact, "woof", VERSION, repo)
    assert result["project"] == "woof"


def capture_fixture(tmp_path, manifest):
    """Self-consistent local capture, with no real archive or remote services."""
    packet = tmp_path / "packet"
    assets = packet / "assets"
    assets.mkdir(parents=True)
    document = deepcopy(manifest)
    for item in document["github"]["assets"] + document["pypi"]["artifacts"]:
        payload = ("fixture: " + item["filename"]).encode("utf-8")
        item.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        (assets / item["filename"]).write_bytes(payload)
    for key in ("release", "verification"):
        name = document["carried_physics"][key]["filename"]
        document["carried_physics"][key] = next(dict(item) for item in document["github"]["assets"] if item["filename"] == name)
    publication_manifest = assets / "PUBLICATION-ASSETS.json"
    publication_manifest.write_text(json.dumps(document), encoding="utf-8")
    checksums = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                        for path in sorted(assets.iterdir()))
    (assets / "DOWNLOAD-SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    captured = {
        "schema": "gpuwm.prepared-release-capture.v1",
        "tag": TAG, "version": VERSION, "commit": COMMIT, "repository": REPOSITORY,
        "release_id": 42, "draft": True, "prerelease": False, "immutable_required": False,
        "manifest_sha256": hashlib.sha256(publication_manifest.read_bytes()).hexdigest(),
        "assets": [
            {"filename": path.name, "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "asset_id": index}
            for index, path in enumerate(sorted(assets.iterdir()), start=1)
        ],
    }
    (packet / "CAPTURE.json").write_text(json.dumps(captured), encoding="utf-8")
    return packet, captured


def test_capture_plan_accepts_the_known_manifest_identity(publication, tmp_path, manifest, monkeypatch):
    packet, captured = capture_fixture(tmp_path, manifest)
    monkeypatch.setenv("PUBLICATION_COMMIT", COMMIT)
    monkeypatch.setenv("PUBLICATION_MANIFEST_SHA256", captured["manifest_sha256"])
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    actual, plan = publication._capture_plan(packet)
    assert actual == captured
    assert plan["commit"] == COMMIT


@pytest.mark.parametrize("variable,value", [
    ("PUBLICATION_COMMIT", "e" * 40),
    ("PUBLICATION_MANIFEST_SHA256", PIN),
])
def test_capture_cannot_rebind_known_workflow_authority(publication, tmp_path, manifest, monkeypatch, variable, value):
    packet, captured = capture_fixture(tmp_path, manifest)
    monkeypatch.setenv("PUBLICATION_COMMIT", COMMIT)
    monkeypatch.setenv("PUBLICATION_MANIFEST_SHA256", captured["manifest_sha256"])
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv(variable, value)
    with pytest.raises(publication.PublicationError):
        publication._capture_plan(packet)


@pytest.mark.parametrize("operation", ["verify", "smoke"])
def test_existing_proof_or_smoke_output_is_preserved(publication, tmp_path, manifest, monkeypatch, operation):
    packet, captured = capture_fixture(tmp_path, manifest)
    plan = load(publication, manifest)
    repo = tmp_path / "verification-source"
    repo.mkdir()
    output = tmp_path / "existing-proof"
    output.mkdir()
    sentinel = output / "previous-receipt.json"
    original = b'{"status":"existing evidence"}\n'
    sentinel.write_bytes(original)
    monkeypatch.setattr(publication, "_capture_plan", lambda path: (captured, plan))
    monkeypatch.setattr(publication, "proof", lambda path: {"captured": captured, "plan": plan})

    def git_read(command, **kwargs):
        if "rev-parse" in command:
            return COMMIT + "\n"
        if "status" in command:
            return ""
        raise AssertionError(f"unexpected external command: {command}")

    def no_environment_creation(*args, **kwargs):
        raise AssertionError("existing output must be refused before creating an environment")

    monkeypatch.setattr(publication.subprocess, "check_output", git_read)
    monkeypatch.setattr(publication, "checked_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(publication.venv, "EnvBuilder", no_environment_creation)
    args = SimpleNamespace(packet=packet, repo=repo, out=output, proof=tmp_path / "proof", platform="linux-x86_64")
    with pytest.raises(publication.PublicationError):
        getattr(publication, operation)(args)
    assert sentinel.read_bytes() == original


def native_wheel_fixture(tmp_path, *, platform="linux-x86_64", mutation=None):
    suffix = "-manylinux_2_28_x86_64.whl" if platform == "linux-x86_64" else "-win_amd64.whl"
    names = ("arwen-tui", "libnetcdf_writer.so") if platform == "linux-x86_64" else ("arwen-tui.exe", "netcdf_writer.dll")
    rows = []
    members = {}
    prefix = "woof/libexec/bridges/"
    for name, artifact, kind in zip(names, ("arwen-tui", "netcdf_writer"), ("executable", "library")):
        payload = ("fixture native bytes: " + name).encode("utf-8")
        rows.append({"artifact": artifact, "filename": name, "kind": kind, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()})
        members[prefix + name] = payload
    pins = {"schema": "gpuwm-bridge-pins-v1", "release": TAG, "platforms": {platform: {"binaries": deepcopy(rows)}}}
    bundle = {"schema": "gpuwm-wheel-bridge-bundle-v1", "platform": platform, "artifacts": deepcopy(rows)}
    if mutation == "missing-native":
        del members[prefix + names[1]]
    elif mutation == "changed-native":
        members[prefix + names[1]] += b"changed"
    elif mutation == "extra-native":
        members[prefix + "stray.dll"] = b"unlisted bytes"
    elif mutation == "wrong-platform":
        bundle["platform"] = "win-x86_64"
    elif mutation == "wrong-artifact":
        bundle["artifacts"][1]["artifact"] = "a-different-library"
    elif mutation == "wrong-kind":
        bundle["artifacts"][1]["kind"] = "script"
    elif mutation == "changed-manifest-hash":
        bundle["artifacts"][1]["sha256"] = OTHER_PIN
    elif mutation == "duplicate-manifest-row":
        bundle["artifacts"].append(deepcopy(bundle["artifacts"][0]))
    if mutation != "missing-manifest":
        members[prefix + "BUNDLE.json"] = json.dumps(bundle).encode("utf-8")
    wheel = tmp_path / (f"gpuwm-{VERSION}-py3-none" + suffix)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return wheel, pins


@pytest.mark.parametrize("platform", ["linux-x86_64", "win-x86_64"])
def test_embedded_native_files_match_the_platform_pins(publication, tmp_path, platform):
    wheel, pins = native_wheel_fixture(tmp_path, platform=platform)
    assert isinstance(publication.check_embedded_natives(wheel, pins, platform, set()), dict)


def _pin_bundle_only(pins, platform="linux-x86_64"):
    """Pin one more binary that the wheel fixture does not embed."""
    payload = b"fixture native bytes: fetched on demand"
    pins["platforms"][platform]["binaries"].append({
        "artifact": "rw_simradar", "filename": "rw_simradar", "kind": "executable",
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return pins


def test_an_on_demand_artifact_is_pinned_in_the_bundle_and_absent_from_the_wheel(publication, tmp_path):
    """2.9.1: the 2.9.0 Linux wheel was 96,049,884 bytes of PyPI's 100,000,000
    per file, so the observation-stream doors stay in the bundle and its pins
    and leave the wheel; the first door that needs one stages it."""
    wheel, pins = native_wheel_fixture(tmp_path)
    result = publication.check_embedded_natives(wheel, pins, "linux-x86_64", set())
    pins = _pin_bundle_only(pins)
    excused = publication.check_embedded_natives(wheel, pins, "linux-x86_64", {"rw_simradar"})
    assert result == excused == {"platform": "linux-x86_64", "artifacts": 2}


def test_a_wheel_missing_a_pinned_artifact_is_not_excused_unless_the_source_declares_it(publication, tmp_path):
    wheel, pins = native_wheel_fixture(tmp_path)
    pins = _pin_bundle_only(pins)
    with pytest.raises(publication.PublicationError, match="wheel-carried part of the proven bundle"):
        publication.check_embedded_natives(wheel, pins, "linux-x86_64", set())


def test_an_on_demand_artifact_must_be_pinned_or_fetch_cannot_install_it(publication, tmp_path):
    wheel, pins = native_wheel_fixture(tmp_path)
    with pytest.raises(publication.PublicationError, match="not pinned in the bundle"):
        publication.check_embedded_natives(wheel, pins, "linux-x86_64", {"rw_simradar"})


# Replaces test_a_wheel_exclusion_is_refused (45256c630): the six observation-
# stream doors are on-demand bundle artifacts again, so a wheel exclusion is no
# longer refused outright; what stays refused is a declaration that disagrees
# with the wheel, here a wheel-carried artifact declared on demand.
def test_a_wheel_carrying_an_on_demand_artifact_is_refused(publication, tmp_path):
    """The declaration and the wheel must agree both ways."""
    wheel, pins = native_wheel_fixture(tmp_path)
    carried = pins["platforms"]["linux-x86_64"]["binaries"][0]["artifact"]
    with pytest.raises(publication.PublicationError, match="wheel-carried part of the proven bundle"):
        publication.check_embedded_natives(wheel, pins, "linux-x86_64", {carried})


def test_the_on_demand_set_is_read_from_the_tagged_declaration(publication):
    from woof import bridge_assets
    repo = Path(__file__).resolve().parents[1]
    # The last five are the default-on hydro door's provider tools, declared on
    # demand by e81d769e735 (2.9.3 improve-1, default-on hydro never attached).
    assert publication.on_demand_artifacts(repo) == {"rw_igra2", "rw_amv", "rw_ndbc", "rw_gnssro", "rw_wis2", "rw_prepbufr", "rw_madis",
                                                    "rw_radar_feeds", "rw_mrms", "rw_stage4", "rw_asos", "rw_goes",
                                                    "rw_opera", "rw_soundings", "rw_nextgen_verify",
                                                    "rw_hydro_world", "rw_hydro_params", "rw_hydrofabric",
                                                    "nextgen-reservoir-prepare", "rw_hydro_init",
                                                    # 2.9.4 capability lanes, declared on demand: ocean
                                                    # preparation (cap-waves-couple), ocean SST verification
                                                    # (cap-slab-ocean), ice observation binding (cap-fzra),
                                                    # storm-surge bathymetry (cap-surge).
                                                    "rw_ocean", "rw_ocean_verify", "rw_ice_verify",
                                                    "rw_bathy",
                                                    # 2.9.4 (PyPI per-file limit): verification and
                                                    # post-run doors no forecast reaches.
                                                    "rw_hail", "rw_airnow", "rw_scoreboard",
                                                    "rw_tc_fields", "rw_fire_indices",
                                                    # Post-2.9.4 native Rust hydro output.
                                                    "rw_nextgen_output",
                                                    # 2.9.9 (PyPI per-file limit): the probability renderer.
                                                    "rw_probgrid",
                                                    # 2.9.10 (PyPI per-file limit): the Icechunk writer and
                                                    # the woof hex RRFS init binaries.
                                                    "rw_icechunk", "rw_mpas_init_v831",
                                                    "rw_mpas_gsl_oro",
                                                    # 2.9.11 (PyPI per-file limit): opt-in simulated
                                                    # radar, ML export, the wind-farm table decoder
                                                    # and the hydro door's payload library.
                                                    "rw_simradar", "rw_mlexport", "rw-wind-table",
                                                    "nextgen_hydrofabric",
                                                    # WOOF 1.1.0 r16 (PyPI per-file limit).
                                                    "rw_odim", "rw_mpas_geometry", "rw_mpas_convert",
                                                    "rw_mpas_lbc"} == {
        a.name for a in bridge_assets.on_demand_artifacts()}
    # The global worker's packer stays in the wheel: a global run without it
    # builds its statics natively (about 2,000 s at T255).
    assert {"rw_globe_pack"} <= {a.name for a in bridge_assets.wheel_artifacts()}


@pytest.mark.parametrize("mutation", ["missing-native", "changed-native", "extra-native", "missing-manifest", "wrong-platform", "wrong-artifact", "wrong-kind", "changed-manifest-hash", "duplicate-manifest-row"])
def test_embedded_native_gate_rejects_wheel_bundle_disagreement(publication, tmp_path, mutation):
    wheel, pins = native_wheel_fixture(tmp_path, mutation=mutation)
    with pytest.raises(publication.PublicationError):
        publication.check_embedded_natives(wheel, pins, "linux-x86_64", set())


def test_pure_wheel_must_not_hide_a_native_payload(publication, tmp_path):
    wheel, pins = native_wheel_fixture(tmp_path)
    with pytest.raises(publication.PublicationError):
        publication.check_embedded_natives(wheel, pins, None, set())


def test_pure_wheel_without_staged_native_files_passes(publication, tmp_path):
    repo, wheel = metadata_fixture(tmp_path)
    assert isinstance(publication.check_embedded_natives(wheel, {"platforms": {}}, None, set()), dict)


def python_inventory_fixture(tmp_path, mutation=None):
    repo, wheel = metadata_fixture(tmp_path)
    with (repo / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[tool.setuptools.packages.find]\ninclude = ["woof", "woof.*"]\nnamespaces = true\n')
    (repo / "setup.py").write_text(
        'DEVELOPMENT_MODULE_GLOBS: dict[str, tuple[str, ...]] = {"woof": ("dev_*",)}\n'
        'raise AssertionError("inventory must inspect setup.py without executing it")\n',
        encoding="utf-8",
    )
    source = b'"""Required runtime module."""\n'
    development = b'"""Excluded development probe."""\n'
    (repo / "woof/core.py").write_bytes(source)
    (repo / "woof/dev_probe.py").write_bytes(development)
    with zipfile.ZipFile(wheel, "a") as archive:
        if mutation != "omitted-runtime-module":
            archive.writestr("woof/core.py", source)
        if mutation == "included-development-module":
            archive.writestr("woof/dev_probe.py", development)
    return repo, wheel


def test_python_inventory_respects_declared_module_exclusions_without_running_setup(publication, tmp_path):
    repo, wheel = python_inventory_fixture(tmp_path)
    publication.check_python_inventory(wheel, repo, "woof")


@pytest.mark.parametrize("mutation", ["omitted-runtime-module", "included-development-module"])
def test_python_inventory_refuses_an_omission_or_excluded_probe(publication, tmp_path, mutation):
    repo, wheel = python_inventory_fixture(tmp_path, mutation)
    with pytest.raises(publication.PublicationError):
        publication.check_python_inventory(wheel, repo, "woof")


def declaration_only_fixture(tmp_path, declared=True):
    """A `docs` namespace package whose wheel payload is its package-data, beside a manifest that ships its examples."""
    repo, wheel = metadata_fixture(tmp_path)
    with (repo / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[tool.setuptools.packages.find]\ninclude = ["woof", "woof.*", "docs"]\nnamespaces = true\n'
                     '[tool.setuptools.package-data]\ndocs = ["seam.md"]\n')
    (repo / "setup.py").write_text(
        'DEVELOPMENT_MODULE_GLOBS: dict[str, tuple[str, ...]] = {}\n'
        + ('DECLARATION_ONLY_DATA_PACKAGES: tuple[str, ...] = ("docs",)\n' if declared else '')
        + 'raise AssertionError("inventory must inspect setup.py without executing it")\n',
        encoding="utf-8",
    )
    (repo / "MANIFEST.in").write_text("recursive-include docs *.md *.py\n", encoding="utf-8")
    (repo / "docs" / "examples").mkdir(parents=True)
    (repo / "docs" / "seam.md").write_text("contract\n", encoding="utf-8")
    (repo / "docs" / "examples" / "client.py").write_bytes(b'"""An example, shipped by the sdist."""\n')
    source = b'"""Required runtime module."""\n'
    (repo / "woof/core.py").write_bytes(source)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("woof/core.py", source)
        archive.writestr("docs/seam.md", "contract\n")
    return repo, wheel


def test_python_inventory_follows_a_declaration_only_data_package(publication, tmp_path):
    """2.9.2 cut: the verifier expected docs/integration-kit/examples/*.py from MANIFEST.in
    while setup.py's build_py ships the docs package's declaration only."""
    repo, wheel = declaration_only_fixture(tmp_path)
    publication.check_python_inventory(wheel, repo, "woof")


def test_python_inventory_still_expects_manifest_examples_without_the_declaration(publication, tmp_path):
    repo, wheel = declaration_only_fixture(tmp_path, declared=False)
    with pytest.raises(publication.PublicationError):
        publication.check_python_inventory(wheel, repo, "woof")
