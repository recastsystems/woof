"""Drive the release-artifact verifier outside a live cut.

``tools/verify_release_artifacts.py`` is the last thing between a built
wheel and PyPI, and until now the only way to run it was to be in the
middle of a release.  That is how a defect in it -- a bundle membership
test that compared the archive against the binary pins alone, and so
refused every bundle the current packer produces -- reached a live 1.4.1
round instead of a CI run.  ``--dry-run`` runs the same assertions, and
this module drives them.

The fixtures are deliberately not this module's own idea of a bundle:
the bundles, pins, and manifest come from ``tools/build_bridge_bundle.py``
-- the real writer the cut runs -- over fixture binaries carrying real
``GPUWM_BRIDGE_SOURCE_REV`` stamps, and, for a vendored artifact, the
real declared contract marker the packer asks it for instead.  A verifier
tested against a fixture its own author shaped proves only that two wrong
ideas agree.
"""

from __future__ import annotations

from base64 import urlsafe_b64encode
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest

from woof import bridge_assets, bridges
from tools import build_bridge_bundle, verify_release_artifacts
from tools import build_linux_release_bridges as linux_build

RELEASE = "v9.8.7"
VERSION = "9.8.7"
SOURCE_REV = "b" * 40
PINS_MEMBER = "woof/data/bridges/bridge-pins.json"


@pytest.fixture
def manylinux_policy():
    return {"name": "manylinux_2_28", "symbol_versions": {"x86_64": {
        "GLIBC": ["2.2.5", "2.28"], "GLIBCXX": ["3.4.25"],
        "GCC": ["7.0.0"], "CXXABI": ["1.3.11"]}},
        "lib_whitelist": ["libc.so.6", "libm.so.6", "libgcc_s.so.1", "libstdc++.so.6"],
        "blacklist": {"libc.so.6": ["private_function"]}}


def _elf_outputs(version="GLIBC_2.28", library="libc.so.6"):
    return {"header": "Class: ELF64\nData: 2's complement, little endian\n"
            "Type: DYN (Shared object file)\nMachine: Advanced Micro Devices X86-64\n",
            "dynamic": f"0x00000001 (NEEDED) Shared library: [{library}]\n",
            "versions": "Version needs section '.gnu.version_r' contains 1 entry:\n"
            f"  0x0000: Version: 1 File: {library} Cnt: 1\n"
            f"  0x0010: Name: {version} Flags: none Version: 2\n",
            "symbols": f"  1: 0000000000000000 0 FUNC GLOBAL DEFAULT UND function@{version} (2)\n"}


def test_manylinux_accepts_required_baseline_versions_not_unrelated_strings(manylinux_policy):
    outputs = _elf_outputs()
    outputs["versions"] = ("Version definition section '.gnu.version_d' contains 1 entry:\n"
                            "  Name: GLIBC_2.39\n" + outputs["versions"])
    result = linux_build.inspect_elf(**outputs, policy=manylinux_policy)
    assert result["required_symbol_versions"] == {"libc.so.6": ["GLIBC_2.28"]}


@pytest.mark.parametrize("version,library", [
    ("GLIBC_2.29", "libc.so.6"), ("GLIBC_2.39", "libc.so.6"),
    ("GLIBCXX_3.4.26", "libstdc++.so.6"), ("GCC_8.0.0", "libgcc_s.so.1"),
    ("CXXABI_1.3.12", "libstdc++.so.6"), ("GLIBC_ABI_DT_RELR", "libc.so.6")])
def test_manylinux_refuses_required_versions_outside_the_policy(manylinux_policy, version, library):
    with pytest.raises(linux_build.QualificationError, match="unsupported symbol version"):
        linux_build.inspect_elf(**_elf_outputs(version, library), policy=manylinux_policy)


@pytest.mark.parametrize("mutation,message", [
    ({"dynamic": "0x1 (NEEDED) Shared library: [libprivate.so]\n"}, "DT_NEEDED"),
    ({"dynamic": "0x1 (RUNPATH) Library runpath: [/opt/build/lib]\n"}, "RPATH/RUNPATH"),
    ({"header": "Class: ELF32\n"}, "x86-64 ELF"),
    ({"symbols": "1: 0 0 FUNC GLOBAL DEFAULT UND private_function@GLIBC_2.28 (2)\n"}, "blacklisted")])
def test_manylinux_refuses_wrong_elf_or_dependencies(manylinux_policy, mutation, message):
    outputs = _elf_outputs()
    outputs.update(mutation)
    with pytest.raises(linux_build.QualificationError, match=message):
        linux_build.inspect_elf(**outputs, policy=manylinux_policy)


def test_manylinux_loader_checks_transitive_dependencies(manylinux_policy):
    output = (" linux-vdso.so.1 (0x0001)\n"
              " libc.so.6 => /lib64/libc.so.6 (0x0002)\n"
              " /lib64/ld-linux-x86-64.so.2 (0x0003)\n")
    assert [p["soname"] for p in linux_build.loader_closure(output, manylinux_policy)] == [
        "libc.so.6", "ld-linux-x86-64.so.2"]
    for extra, message in ((" libprivate.so => /lib64/libprivate.so (0x0004)\n", "transitive"),
                           (" libm.so.6 => not found\n", "unresolved"),
                           (" libm.so.6 => /work/libm.so.6 (0x0004)\n", "outside baseline")):
        with pytest.raises(linux_build.QualificationError, match=message):
            linux_build.loader_closure(output + extra, manylinux_policy)


def test_manylinux_target_reuse_requires_matching_baseline_provenance(tmp_path):
    target = tmp_path / "target"
    compiler = {"rustc_vv": "rustc 1.94.0", "cargo_version": "cargo 1.94.0", "files": []}
    marker, first = linux_build.target_context(target, image=linux_build.IMAGE,
        toolchain=compiler, revision=SOURCE_REV, reuse=False)
    (target / "compiled-dependency").write_bytes(b"baseline-built dependency")
    with pytest.raises(linux_build.QualificationError, match="explicit --reuse-target"):
        linux_build.target_context(target, image=linux_build.IMAGE, toolchain=compiler,
                                   revision="c" * 40, reuse=False)
    with pytest.raises(linux_build.QualificationError, match="provenance mismatch: toolchain"):
        linux_build.target_context(target, image=linux_build.IMAGE, toolchain={"different": True},
                                   revision="c" * 40, reuse=True)
    _, reused = linux_build.target_context(target, image=linux_build.IMAGE,
        toolchain=compiler, revision="c" * 40, reuse=True)
    assert reused["initial_source_rev"] == SOURCE_REV
    assert reused["current_source_rev"] == "c" * 40
    assert reused["reuse_source_revisions"] == [SOURCE_REV, "c" * 40]
    assert marker.is_file() and first["target_was_empty"]
    foreign = tmp_path / "foreign-target"
    foreign.mkdir()
    (foreign / "ubuntu-object").write_bytes(b"not baseline-built")
    with pytest.raises(linux_build.QualificationError, match="unmarked, nonempty"):
        linux_build.target_context(foreign, image=linux_build.IMAGE, toolchain=compiler,
                                   revision=SOURCE_REV, reuse=True)


def test_manylinux_does_not_emit_partial_or_overwrite_existing_payload(tmp_path):
    origin = tmp_path / "native"
    origin.write_bytes(b"qualified bytes")
    artifact = {"artifact": "native", "file": linux_build.file_identity(origin)}
    artifact["file"]["sha256"] = "0" * 64
    output = tmp_path / "artifacts"
    with pytest.raises(linux_build.QualificationError, match="copy differs"):
        linux_build.emit_artifacts(output, [artifact])
    assert not output.exists()
    output.mkdir()
    kept = output / "existing"
    kept.write_bytes(b"keep me")
    with pytest.raises(linux_build.QualificationError, match="nonempty"):
        linux_build.emit_artifacts(output, [artifact])
    assert kept.read_bytes() == b"keep me"


def test_manylinux_missing_auditwheel_is_a_refusal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(linux_build.shutil, "which", lambda *args, **kwargs: None)
    with pytest.raises(linux_build.QualificationError, match="auditwheel is required"):
        linux_build.load_policy(SimpleNamespace(environment={}), tmp_path)


@pytest.mark.parametrize("failure", [None, "missing", "probe", "duplicate"])
def test_manylinux_probes_every_declared_artifact_before_output(tmp_path, monkeypatch, manylinux_policy, failure):
    from woof import doctor
    release = tmp_path / "release"
    release.mkdir()
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        filename = bridge_assets.artifact_filename(artifact, "linux-x86_64")
        (release / filename).write_bytes(b"\x7fELF" + _artifact_bytes(artifact, filename, SOURCE_REV))
    # The engine plants these failures on its terminal UI binary; a tree that
    # leaves that workspace out declares no such artifact, so they land on
    # the first declared executable instead.
    from _woof_snapshot import without_left_out_artifacts
    declared = {a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
    target = "arwen-tui" if "arwen-tui" in declared else next(
        a.name for a in bridge_assets.BUNDLED_ARTIFACTS if a.kind == "executable")
    if failure == "missing":
        (release / target).unlink()
    if failure == "duplicate":
        monkeypatch.setattr(bridge_assets, "BUNDLED_ARTIFACTS",
                            (*bridge_assets.BUNDLED_ARTIFACTS, bridge_assets.BUNDLED_ARTIFACTS[0]))
    called = []
    monkeypatch.setattr(linux_build.shutil, "which", lambda *args, **kwargs: "readelf")
    def probe(path):
        called.append(path.name)
        return (not (failure == "probe" and path.name == target), "fixture execution result")
    monkeypatch.setattr(doctor, "_exec_probe", probe)
    class FixtureCommands:
        environment = {}
        abi_calls = []
        def run(self, label, command, *, cwd):
            if label.endswith("-abi"):
                self.abi_calls.append(command[-2])
                return command[-1]
            # No DT_NEEDED in this fixture: real loader closure is tested above.
            fields = _elf_outputs()
            fields.update(dynamic="There is no dynamic section in this file.\n", versions="")
            return fields[label.rsplit("-", 1)[1]]
    commands = FixtureCommands()
    if failure:
        with pytest.raises(linux_build.QualificationError,
                           match="missing declared|executable probe failed|repeats a native artifact name"):
            linux_build.qualify_artifacts(release, Path(__file__).resolve().parents[1], SOURCE_REV,
                                          manylinux_policy, commands)
    else:
        result = linux_build.qualify_artifacts(release, Path(__file__).resolve().parents[1], SOURCE_REV,
                                              manylinux_policy, commands)
        assert {r["artifact"] for r in result} == {a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
        assert len(result) == len(bridge_assets.BUNDLED_ARTIFACTS)
        required = {
            "grib1_bridge", "gdt101_remap", "gfs_grib2_bridge", "hrrr_grib2_bridge",
            "grib2_inventory", "grib2_dump", "gpuwm_preprocess_cpu", "rw_fetch",
            "rw_wrfbatch", "rw_simradar", "rw_nexrad", "rw_odim", "rw_mrms",
            "rw_stage4", "rw_asos", "rw_goes", "rw_opera", "rw_netcdf",
            "region_global_dealias", "netcdf_writer", "static_fields",
            "gpuwm_mapped_engine", "obs_regrid", "obs_score", "rw_mpas_mesh", "rw_mpas_static",
            "rw_mpas_init", "rw_mpas_geometry", "rw_mpas_hostprep", "rw_mpas_convert", "rw_mpas_lbc", "rw_mlexport",
            "arwen-tui", "rw_zarr", "rw_grib2export", "rw_isobaric", "rw_superob",
            "rw_nowcast_frames",
        }
        assert without_left_out_artifacts(required) <= {r["artifact"] for r in result}
        assert set(called) == {a.name for a in bridge_assets.BUNDLED_ARTIFACTS if a.kind == "executable"}
        assert set(commands.abi_calls) == {bridge_assets.library_abi_for(a.name)[0]
                                          for a in bridge_assets.BUNDLED_ARTIFACTS if a.kind == "library"}


def _stamped(name: str, source_rev: str) -> bytes:
    """Fixture bytes carrying exactly one well-formed revision stamp."""

    return b"".join(
        (
            b"fixture native artifact ",
            name.encode("utf-8"),
            b"\n",
            bridge_assets.SOURCE_REV_MARKER,
            source_rev.encode("ascii"),
            b"\n",
        )
    )


def _artifact_bytes(
    artifact: bridge_assets.BundledArtifact, name: str, source_rev: str
) -> bytes:
    """Fixture bytes carrying whatever the packer asks this artifact for.

    A vendored artifact is a verbatim upstream crate frozen at a recorded
    commit, so it does not move with the release checkout and the packer
    never asks it for a revision stamp -- it asks for the contract marker
    declared in ``BRIDGE_ABI_MARKERS``, a property of those exact bytes.
    Stamping such a fixture with a revision would prove nothing the packer
    reads, so it carries the real marker instead.
    """

    if artifact.vendored:
        return b"".join(
            (
                b"fixture vendored artifact ",
                name.encode("utf-8"),
                b"\n",
                bridges.BRIDGE_ABI_MARKERS[artifact.name],
                b"\n",
            )
        )
    return _stamped(name, source_rev)


def _pack_and_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_rev: str = SOURCE_REV,
) -> tuple[Path, Path, Path]:
    """A cut's bundle payload, produced by the packer the workflow runs."""

    asset_source = tmp_path / "asset-source"
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        directory = asset_source / subdir
        directory.mkdir(parents=True)
        # Two files, so a test can drop one pin and still leave a bundle
        # that carries map assets at all -- the refusal for carrying none
        # is a different check, and it would mask this one.
        for index in (1, 2):
            (directory / f"fixture-geometry-{index}.bin").write_bytes(
                f"fixture {subdir} geometry {index}\n".encode("utf-8")
            )
    # The one seam the packer names for the asset tree.  Pointing it at a
    # fixture keeps the real walk, the real member naming, and the real
    # refusals, without carrying 36 MiB of basemaps through a unit test.
    monkeypatch.setattr(
        build_bridge_bundle, "source_asset_dir", lambda: asset_source
    )

    bundles = tmp_path / "dist-bridges"
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        search = tmp_path / f"target-{platform}"
        search.mkdir()
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            filename = bridge_assets.artifact_filename(artifact, platform)
            (search / filename).write_bytes(
                _artifact_bytes(artifact, filename, source_rev)
            )
        assert (
            build_bridge_bundle.main(
                [
                    "pack",
                    "--release",
                    RELEASE,
                    "--platform",
                    platform,
                    "--search",
                    str(search),
                    "--out",
                    str(bundles),
                ]
            )
            == 0
        )

    pins = tmp_path / "bridge-pins.json"
    manifest = bundles / build_bridge_bundle.MANIFEST_ASSET
    bundle_args: list[str] = []
    for archive in sorted(bundles.glob("*.zip")):
        bundle_args += ["--bundle", str(archive)]
    assert (
        build_bridge_bundle.main(
            [
                "pin",
                "--release",
                RELEASE,
                "--source-rev",
                source_rev,
                *bundle_args,
                "--out",
                str(pins),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    return bundles, pins, manifest


def _build_dists(tmp_path: Path, pins: Path) -> tuple[Path, Path]:
    """A wheel and sdist carrying the generated pins the way a cut's do."""

    payload = pins.read_bytes()
    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)

    wheel = dist / f"gpuwm-{VERSION}-py3-none-any.whl"
    digest = (
        urlsafe_b64encode(hashlib.sha256(payload).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    record = (
        f"{PINS_MEMBER},sha256={digest},{len(payload)}\n"
        f"gpuwm-{VERSION}.dist-info/RECORD,,\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(PINS_MEMBER, payload)
        archive.writestr(f"gpuwm-{VERSION}.dist-info/RECORD", record)

    sdist = dist / f"gpuwm-{VERSION}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo(f"gpuwm-{VERSION}/{PINS_MEMBER}")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return wheel, sdist


def _dry_run(
    tmp_path: Path,
    bundles: Path,
    pins: Path,
    manifest: Path,
    wheel: Path,
    sdist: Path,
    *,
    release: str = RELEASE,
    source_rev: str = SOURCE_REV,
) -> Path:
    receipt = tmp_path / "release-artifact-proof.json"
    assert (
        verify_release_artifacts.main(
            [
                "--dry-run",
                "--wheel",
                str(wheel),
                "--sdist",
                str(sdist),
                "--pins",
                str(pins),
                "--manifest",
                str(manifest),
                "--bundles",
                str(bundles),
                "--release",
                release,
                "--source-rev",
                source_rev,
                "--receipt",
                str(receipt),
            ]
        )
        == 0
    )
    return receipt


def test_dry_run_passes_the_packers_own_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    receipt_path = _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "gpuwm-release-artifact-proof-v2"
    assert receipt["status"] == "PASS"
    assert receipt["release"] == RELEASE
    assert receipt["source_rev"] == SOURCE_REV
    assert set(receipt["bundles"]) == set(bridge_assets.SUPPORTED_PLATFORMS)
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        assert receipt["bundles"][platform]["binaries"] == len(
            bridge_assets.BUNDLED_ARTIFACTS)
        assert receipt["bundles"][platform]["assets"] >= 1


def test_the_receipt_names_which_proof_each_binary_got(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2's invariant, stated as a partition rather than a boolean.

    v1 claimed one blanket thing about every binary, and a vendored
    artifact -- correct, pinned, deliberately unstamped -- made that claim
    false.  v2 says instead that every binary was proved by exactly one of
    two means, and names them, so bytes nobody proved would show up as a
    filename missing from both lists rather than as a still-true-looking
    ``True``.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    receipt = json.loads(
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist).read_text(
            encoding="utf-8"
        )
    )

    assert receipt["schema"] == "gpuwm-release-artifact-proof-v2"
    # The retired v1 key is gone, not aliased: a reader written against
    # the old blanket claim must fail loudly rather than silently pass.
    assert "source_rev_stamp_verified_in_every_binary" not in receipt

    stamped = receipt["binaries_proved_by_source_rev_stamp"]
    markered = receipt["binaries_proved_by_contract_marker"]
    assert not set(stamped) & set(markered)

    expected: set[str] = set()
    vendored_names: set[str] = set()
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            label = (
                f"{platform}: "
                f"{bridge_assets.artifact_filename(artifact, platform)}"
            )
            expected.add(label)
            if artifact.vendored:
                vendored_names.add(label)
    # Every binary in both bundles is accounted for by one proof or the
    # other, and the vendored one is accounted for by the marker.
    assert set(stamped) | set(markered) == expected
    assert set(markered) == vendored_names
    assert vendored_names, "fixture no longer covers a vendored artifact"


def test_dry_run_refuses_a_vendored_artifact_missing_its_contract_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stale vendored build the marker check exists to catch.

    A vendored artifact carries no revision stamp, so exempting it from
    the stamp check must not mean exempting it from staleness: the marker
    is the only question those bytes can answer, and it has to bite.

    The bundle here is packed and pinned accurately, and then the release's
    declared contract moves on -- which is exactly what re-vendoring is, a
    commit that advances ``BRIDGE_ABI_MARKERS`` together with the crate.
    Bundles built before that advance carry the older bytes.  The packer
    refuses such a bundle at pin time, but this verifier is the
    independent second reading of already-published artifacts, so it has
    to refuse on its own rather than inherit the packer's verdict.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    vendored = [
        artifact
        for artifact in bridge_assets.BUNDLED_ARTIFACTS
        if artifact.vendored
    ]
    assert vendored, "fixture no longer covers a vendored artifact"
    monkeypatch.setitem(
        bridges.BRIDGE_ABI_MARKERS,
        vendored[0].name,
        b"bw_dealias_rift_v2_not_in_these_bytes",
    )

    with pytest.raises(SystemExit) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    assert "contract marker" in str(refusal.value)


def test_dry_run_refuses_a_vendored_artifact_with_no_declared_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unstamped binary with nothing else to prove it is not proved.

    Skipping the stamp check for a vendored artifact is only sound while
    something else answers the staleness question.  If the marker table
    ever loses that artifact's entry, the exemption must become a refusal
    rather than a silent pass.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    vendored = [
        artifact
        for artifact in bridge_assets.BUNDLED_ARTIFACTS
        if artifact.vendored
    ]
    assert vendored, "fixture no longer covers a vendored artifact"
    monkeypatch.delitem(bridges.BRIDGE_ABI_MARKERS, vendored[0].name)

    with pytest.raises(SystemExit) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    assert "no declared" in str(refusal.value)


def test_a_dry_run_receipt_never_claims_what_it_did_not_prove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode is on the receipt, and so is everything it skipped."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    receipt = json.loads(
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist).read_text(
            encoding="utf-8"
        )
    )

    assert receipt["mode"] == "dry-run"
    assert receipt["not_proven"] == list(verify_release_artifacts.DRY_RUN_SKIPS)
    assert receipt["installed_version"] is None
    assert receipt["host_probes"] == []
    assert receipt["prebuilt_offer_live_lines"] == []


def test_a_live_cut_still_requires_the_installation_it_proves() -> None:
    """Without ``--dry-run`` the cut-only inputs stay mandatory, so the
    mode can never be entered by forgetting an argument."""

    with pytest.raises(SystemExit) as refusal:
        verify_release_artifacts._parse_args(
            [
                "--wheel", "w", "--sdist", "s", "--pins", "p",
                "--manifest", "m", "--bundles", "b", "--release", RELEASE,
                "--source-rev", SOURCE_REV, "--receipt", "r",
            ]
        )
    assert refusal.value.code == 2


def _rewrite_platforms(pins: Path, manifest: Path, mutate) -> None:
    """Apply the same edit to both documents the cut publishes.

    They are cross-checked for equality before membership is looked at,
    so a one-sided edit would only ever prove that check.
    """

    for path, key in ((pins, "platforms"), (manifest, "platforms")):
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document[key])
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


@pytest.mark.parametrize("direction", ("under-declared", "over-declared"))
def test_dry_run_refuses_pins_that_do_not_name_the_bundle_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    """The 1.4.1 defect class, now reachable without a live cut.

    A bundle must contain exactly what the release pinned, so the check
    is an equality and both directions have to bite: a pinned member the
    archive does not carry is a download that 404s member-by-member, and
    an unpinned member the archive does carry is bytes nobody verified.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)

    def mutate(platforms: dict) -> None:
        record = platforms[bridge_assets.SUPPORTED_PLATFORMS[0]]
        if direction == "under-declared":
            record["assets"].pop()
        else:
            record["assets"].append(
                {
                    "path": "assets/basemap/never-packed.bin",
                    "bytes": 3,
                    "sha256": "0" * 64,
                }
            )

    _rewrite_platforms(pins, manifest, mutate)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(AssertionError) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    expected = (
        "unexpected: ['assets/basemap/fixture-geometry-2.bin']"
        if direction == "under-declared"
        else "missing: ['assets/basemap/never-packed.bin']"
    )
    assert expected in str(refusal.value)


def test_dry_run_refuses_binaries_stamped_with_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hash proves the bytes; only the stamp proves their source."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(bridge_assets.BridgeAssetError, match="was built from"):
        _dry_run(
            tmp_path, bundles, pins, manifest, wheel, sdist,
            source_rev="c" * 40,
        )


def test_dry_run_refuses_a_wheel_whose_record_misstates_the_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RECORD binds the pins' hash and size, not merely their name."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    payload = pins.read_bytes()
    rebuilt = wheel.with_name("rebuilt.whl")
    with zipfile.ZipFile(rebuilt, "w") as archive:
        archive.writestr(PINS_MEMBER, payload)
        archive.writestr(
            f"gpuwm-{VERSION}.dist-info/RECORD",
            f"{PINS_MEMBER},sha256={'A' * 43},{len(payload)}\n",
        )

    with pytest.raises(AssertionError):
        _dry_run(tmp_path, bundles, pins, manifest, rebuilt, sdist)


def test_dry_run_refuses_pins_generated_for_another_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(AssertionError):
        _dry_run(
            tmp_path, bundles, pins, manifest, wheel, sdist, release="v0.0.1"
        )


# ---------------------------------------------------------------------------
# The host-staging leg's library probe
# ---------------------------------------------------------------------------
#
# The leg that loads each `kind == "library"` artifact is the one thing
# `--dry-run` cannot run, and it is where 2.1.0 lost a number: it asked
# EVERY library for `gpuwm_preprocess_cpu_abi_version` while the vendored
# dealiasing cdylib exports `bw_abi_version`, so a correct bundle was
# refused in the prepare job with the tag already public.  These cases
# drive `probe_library_abi` through its loader seam, so per-artifact
# dispatch and the refusal for an undeclared library are proved without a
# target-native binary -- on both platforms' filenames, in CI, on every
# commit.


class _FakeAbi:
    """One exported symbol, callable the way ctypes' function objects are."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.argtypes: list[object] | None = None
        self.restype: object | None = None

    def __call__(self) -> int:
        return self.value


class _FakeLibrary:
    """A loaded library that exports exactly the symbols it was given."""

    def __init__(self, path: str, exports: dict[str, int]) -> None:
        self.path = path
        self._exports = {name: _FakeAbi(value)
                         for name, value in exports.items()}

    def __getattr__(self, name: str) -> _FakeAbi:
        try:
            return self.__dict__["_exports"][name]
        except KeyError:
            raise AttributeError(
                f"{self.path}: undefined symbol: {name}") from None


def _loader_for(exports_by_filename: dict[str, dict[str, int]]):
    """A ctypes.CDLL stand-in serving each fixture library its own exports."""

    def load(path: str) -> _FakeLibrary:
        return _FakeLibrary(path, exports_by_filename[Path(path).name])

    return load


def _library_artifacts() -> tuple[bridge_assets.BundledArtifact, ...]:
    return tuple(artifact for artifact in bridge_assets.BUNDLED_ARTIFACTS
                 if artifact.kind == "library")


def test_the_release_ships_six_libraries_with_different_abi_symbols() -> None:
    """The premise of the dispatch: each library has its OWN handshake.

    If this ever collapses to one library the dispatch cases below stop
    proving anything, and this one says so out loud rather than letting
    them pass vacuously.  The third library is the NetCDF writer behind
    the default wrfout engine, which joined the bundle when the product
    tape flipped onto it; the fourth is the static-field builder, which
    joined when `build_static` flipped onto the Rust crate by default;
    the fifth is the observation remap, which joined when
    `woof.verify.obs.regrid` flipped off scipy by default. The sixth is
    the shared isobaric reader used by pressure-level products and exports.
    The coupled hydro output library (rw_nextgen_output) joined at 2.9.4,
    when its NetCDF moved onto the project's Rust writer.
    """

    artifacts = _library_artifacts()
    assert {a.name for a in artifacts} == {
        "gpuwm_preprocess_cpu", "region_global_dealias", "netcdf_writer",
        "static_fields", "obs_regrid", "obs_score", "rw_isobaric",
        "rw_superob", "rw_tempo", "rw_projected_intermediate",
        "nextgen_hydrofabric", "arwen_nextgen_forcing", "rw_nextgen_output", "rw_p3", "rw_cam_tables", "rw_cu99_launch", "arwen_dfi", "gpuwm_tslist",
        # 3.0: the TC frames library (canonical TC initialization and storm products).
        "tc_frames_io"}
    symbols = {bridge_assets.library_abi_for(a.name)[0] for a in artifacts}
    assert symbols == {"gpuwm_preprocess_cpu_abi_version", "bw_abi_version",
                       "gpuwm_ncwrite_abi_version",
                       "gpuwm_static_abi_version",
                       "gpuwm_obsregrid_abi_version", "gpuwm_obsscore_abi_version",
                       "gpuwm_isobaric_abi_version",
                       "gpuwm_superob_abi_version",
                       "gpuwm_tempo_abi_version",
                       "gpuwm_projected_intermediate_abi_version",
                       "arwen_hydrofabric_abi_version", "arwen_forcing_abi_version",
                       "rw_nextgen_output_abi_version", "gpuwm_p3_abi_version",
                       "gpuwm_cam_tables_abi_version", "gpuwm_cu99_launch_abi_version",
                       "aw_dfi_runtime_abi", "gpuwm_tslist_abi_version",
                       "tc_frames_abi_version"}


@pytest.mark.parametrize("platform", bridge_assets.SUPPORTED_PLATFORMS)
def test_each_library_is_probed_with_its_own_declared_symbol(
    tmp_path: Path, platform: str
) -> None:
    """Two library fixtures, each exporting ONLY its own ABI symbol.

    Neither fixture can answer the other's handshake, so a probe that
    reused one symbol for both -- the 2.1.0 defect -- fails here.
    """

    artifacts = _library_artifacts()
    exports: dict[str, dict[str, int]] = {}
    paths: dict[str, Path] = {}
    for artifact in artifacts:
        filename = bridge_assets.artifact_filename(artifact, platform)
        path = tmp_path / platform / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture library " + artifact.name.encode())
        symbol, version = bridge_assets.library_abi_for(artifact.name)
        exports[filename] = {symbol: version}
        paths[artifact.name] = path

    loader = _loader_for(exports)
    for artifact in artifacts:
        probe = verify_release_artifacts.probe_library_abi(
            paths[artifact.name], artifact, loader=loader
        )
        symbol, version = bridge_assets.library_abi_for(artifact.name)
        assert probe["artifact"] == artifact.name
        assert probe["symbol"] == symbol
        assert probe["abi"] == version
        assert probe["status"] == "PASS"
        assert probe["filename"] == bridge_assets.artifact_filename(
            artifact, platform
        )


def test_the_vendored_library_is_not_probed_with_the_stamped_ones_symbol(
    tmp_path: Path,
) -> None:
    """The exact 2.1.0 refusal, as a test.

    The vendored cdylib exports `bw_abi_version` and nothing else.  A
    probe that reaches for `gpuwm_preprocess_cpu_abi_version` gets an
    AttributeError from the loader; the shipped dispatch gets an answer.
    """

    vendored = next(artifact for artifact in _library_artifacts()
                    if artifact.vendored)
    stamped = next(artifact for artifact in _library_artifacts()
                   if not artifact.vendored)
    filename = bridge_assets.artifact_filename(vendored, "linux-x86_64")
    path = tmp_path / filename
    path.write_bytes(b"fixture vendored library\n")
    loader = _loader_for({filename: {"bw_abi_version": 1}})

    probe = verify_release_artifacts.probe_library_abi(
        path, vendored, loader=loader
    )
    assert probe["symbol"] == "bw_abi_version"

    # The stamped library's artifact record against the vendored bytes:
    # the wrong handshake, which is what the old code applied to both.
    with pytest.raises(AttributeError, match="undefined symbol"):
        verify_release_artifacts.probe_library_abi(
            path, stamped, loader=loader
        )


def test_a_library_with_no_declared_handshake_is_refused(
    tmp_path: Path,
) -> None:
    """Fail closed, the same shape as the vendored contract-marker refusal.

    A library added to BUNDLED_ARTIFACTS without a LIBRARY_ABI entry is
    refused by name; it is never probed with some other library's symbol.
    """

    undeclared = bridge_assets.BundledArtifact(
        "future_thing", "library", "tools/future_thing",
        "GPUWM_FUTURE_THING_BRIDGE", "a library nobody declared")
    assert undeclared.name not in bridge_assets.LIBRARY_ABI
    path = tmp_path / "libfuture_thing.so"
    path.write_bytes(b"fixture undeclared library\n")
    loader = _loader_for(
        {"libfuture_thing.so": {"gpuwm_preprocess_cpu_abi_version": 1}}
    )

    with pytest.raises(SystemExit) as refusal:
        verify_release_artifacts.probe_library_abi(
            path, undeclared, loader=loader
        )
    message = str(refusal.value)
    assert "future_thing" in message
    assert "LIBRARY_ABI" in message


def test_a_library_answering_another_version_is_refused(
    tmp_path: Path,
) -> None:
    """The right symbol answering the wrong number is still a stale build."""

    stamped = next(artifact for artifact in _library_artifacts()
                   if not artifact.vendored)
    filename = bridge_assets.artifact_filename(stamped, "linux-x86_64")
    path = tmp_path / filename
    path.write_bytes(b"fixture library\n")
    symbol, version = bridge_assets.library_abi_for(stamped.name)
    loader = _loader_for({filename: {symbol: version + 7}})

    with pytest.raises(SystemExit, match=symbol):
        verify_release_artifacts.probe_library_abi(
            path, stamped, loader=loader
        )


def test_the_workflow_probe_reads_the_shared_table_not_its_own_copy() -> None:
    """One table, cross-checked: the workflow must not re-declare it.

    `.github/workflows/publish.yml` runs the same per-artifact dispatch on
    each target runner.  It carried its own literal copy of the table
    while the verifier carried the older blanket rule, which is precisely
    how the two came to disagree.  The copy is gone; this case fails if
    one is reintroduced, because a second copy is a second thing to
    forget.
    """

    from _woof_snapshot import engine_workflow_reason
    if (reason := engine_workflow_reason(".github/workflows/publish.yml")) is not None:
        pytest.skip(reason)
    workflow = (
        Path(verify_release_artifacts.__file__).resolve().parent.parent
        / ".github" / "workflows" / "publish.yml"
    )
    text = workflow.read_text(encoding="utf-8")
    controller = (workflow.parents[2] / "tools/promote_prepared_release.py").read_text(encoding="utf-8")
    assert "promote_prepared_release.py smoke" in text
    assert "tools/verify_release_artifacts.py" in controller
    assert "bridge_assets.library_abi_for(" in Path(verify_release_artifacts.__file__).read_text(encoding="utf-8")
    for symbol, _version in bridge_assets.LIBRARY_ABI.values():
        assert f'"{symbol}"' not in text, symbol
        assert f"'{symbol}'" not in text, symbol
        assert f'"{symbol}"' not in controller, symbol


# ---------------------------------------------------------------------
# PyPI's per-file cap
# ---------------------------------------------------------------------
# The upload is the LAST step of a cut: the tag is pushed and the
# release assets are written before twine hands PyPI anything, so a file
# over the cap burns a public tag that non-reproducible Rust builds
# cannot be rebuilt behind.  The published pair's headroom is 8.07 MB
# (wheel) and 4.75 MB (sdist) as measured on 2026-08-17, and a platform
# wheel built in the same tree is 108.70 MB (win) / 111.70 MB (linux) --
# over the cap outright.  So the verifier measures every distribution
# file it can see, refuses the ones the cut is about to publish, and
# records the rest.


def _pad_archive(path: Path, extra_bytes: int) -> None:
    """Grow a zip past a size threshold with incompressible bytes."""

    with zipfile.ZipFile(path, "a", zipfile.ZIP_STORED) as archive:
        archive.writestr("woof/_size_probe.bin", os.urandom(extra_bytes))


def test_a_published_wheel_over_the_cap_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    monkeypatch.setattr(verify_release_artifacts, "PYPI_FILE_CAP_BYTES", 4096)

    with pytest.raises(SystemExit) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    message = str(refusal.value)
    assert "per-file cap" in message, message
    assert wheel.name in message, message
    assert "after the tag is public" in message, message


def test_a_platform_wheel_beside_the_pair_is_measured_not_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape the cut is NOT publishing still gets a number.

    A per-platform wheel built in the same tree lands in the same dist
    directory, does not fit under the cap at this tip, and used to be
    invisible to the receipt -- which named the universal wheel and
    nothing else.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    platform_wheel = wheel.parent / f"gpuwm-{VERSION}-py3-none-win_amd64.whl"
    shutil.copyfile(wheel, platform_wheel)
    _pad_archive(platform_wheel, 200_000)
    # A cap the published pair clears with room and the padded platform
    # wheel does not -- the shape this tip really has, in miniature.
    monkeypatch.setattr(verify_release_artifacts, "PYPI_FILE_CAP_BYTES",
                        100_000)

    receipt = json.loads(
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist).read_text(
            encoding="utf-8"))

    by_name = {row["filename"]: row for row in receipt["distributions"]}
    assert set(by_name) == {wheel.name, sdist.name, platform_wheel.name}
    assert by_name[wheel.name]["published"] is True
    assert by_name[wheel.name]["cap_verdict"] == "PASS"
    measured = by_name[platform_wheel.name]
    assert measured["published"] is False
    assert measured["cap_verdict"] == "OVER"
    assert measured["bytes"] == platform_wheel.stat().st_size
    assert measured["headroom_bytes_vs_100e6"] < 0
    assert receipt["pypi_file_cap_bytes"] == 100_000


# ---------------------------------------------------------------------
# Private machine names in a wheel's data members (A154)
# ---------------------------------------------------------------------
# 2.8.1's fetch route table named the lab host that watched the posting
# times in 20 measured rows, and nothing read a wheel's data files for
# one.  The names are assembled so this file carries none.
_HOST = "weather-" + "node-" + "2"
_LAN = "192." + "168.68.1"


def test_a_wheel_data_member_naming_a_private_machine_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    receipt = json.loads(
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist).read_text(
            encoding="utf-8"))
    assert receipt["wheel"]["private_machine_names_in_members"] == 0

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("woof/authorities/routes.json",
                         '{"measured": "posting watch, ' + _HOST + '"}\n')

    with pytest.raises(SystemExit) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)
    message = str(refusal.value)
    assert "publishes a private machine's name" in message, message
    assert "woof/authorities/routes.json" in message, message


# Replaces test_the_wheel_scan_reads_data_members_and_netcdf_not_code
# (122ce03da): the wheel scan now reads code members too (see the docstring).
def test_the_wheel_scan_reads_code_data_and_netcdf_not_metadata(
    tmp_path: Path,
) -> None:
    """Every installed member is read, code included.

    2.9.0's wheel shipped lab host names and internal folder names in 61
    members, nearly all code, because the scan read data members only.
    """
    wheel = tmp_path / f"gpuwm-{VERSION}-py3-none-any.whl"
    hdf5_like = (b"\x89HDF\r\n\x1a\n" + b"\x00" * 32 + b"history\x00"
                 + ("met_em on " + _HOST + " (" + _LAN + ")").encode("ascii")
                 + b"\x00" * 32)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("configs/demo.toml", "# Linux (" + _HOST + ")\n")
        archive.writestr("configs/real/terrain.nc", hdf5_like)
        archive.writestr("woof/data/governor.json",
                         '{"note": "the node-wide governor"}\n')
        archive.writestr("woof/core/timing.py", "# measured on " + _HOST + "\n")
        archive.writestr("woof/core/report.py",
                         "# see " + "Down" + "loads/RUN-REPORT/REPORT.md\n")
        archive.writestr("woof/go_cli.py",
                         "# the engine's own downloads/<key>/<object>.part\n")
        archive.writestr("woof/libexec/bridges/rw_tool",
                         b"\x7fELF\x00\x00" + _HOST.encode("ascii"))
        archive.writestr(f"gpuwm-{VERSION}.dist-info/METADATA",
                         "Summary: built on " + _HOST + "\n")

    rows = verify_release_artifacts.private_host_members(wheel)
    assert sorted((row["member"], row["kind"]) for row in rows) == [
        ("configs/demo.toml", "private machine name"),
        ("configs/real/terrain.nc", "private machine name"),
        ("woof/core/report.py", "internal folder name"),
        ("woof/core/timing.py", "private machine name")]


def test_a_frozen_kernel_source_is_allowed_by_its_exact_bytes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule's second allowance: a byte-pinned kernel source as it is.

    An edited copy of the same kernel names its host and is refused.  The
    shipped allowance is empty (every entry was spent), so the frozen
    source here is synthetic and names a host built from fragments.
    """

    import hashlib

    from tools import release_exclusions

    repo = Path(verify_release_artifacts.__file__).resolve().parents[1]
    rel = "woof/core/kernels/frozen_example.cu"
    payload = ("// measured on " + "no" + "de-9\n"
               "__global__ void frozen_example() {}\n").encode()
    monkeypatch.setattr(release_exclusions, "PINNED_KERNEL_SOURCES",
                        {rel: hashlib.sha256(payload).hexdigest()})

    exact = tmp_path / "exact.whl"
    with zipfile.ZipFile(exact, "w") as archive:
        archive.writestr(rel, payload)
    assert verify_release_artifacts._refuse_private_hosts(exact, repo) == [rel]

    edited = tmp_path / "edited.whl"
    with zipfile.ZipFile(edited, "w") as archive:
        archive.writestr(rel, payload + b"// edited\n")
    with pytest.raises(SystemExit):
        verify_release_artifacts._refuse_private_hosts(edited, repo)


def test_a_pinned_reference_config_is_allowed_by_its_exact_bytes_only(
    tmp_path: Path,
) -> None:
    """The rule's one allowance: the config a committed WRF reference
    manifest pins by sha256.  An edited copy of it is refused."""

    from tools.release_exclusions import pinned_record_digests

    repo = Path(verify_release_artifacts.__file__).resolve().parents[1]
    digests = pinned_record_digests(repo)
    records = [path for path in sorted((repo / "configs").rglob("*.toml"))
               if hashlib.sha256(path.read_bytes()).hexdigest() in digests]
    if not records:
        pytest.skip("no committed WRF reference pins a config in this tree")
    payload = records[0].read_bytes()
    if b"node-" not in payload:
        pytest.skip("the pinned config names no machine; nothing to allow")

    exact = tmp_path / "exact.whl"
    with zipfile.ZipFile(exact, "w") as archive:
        archive.writestr("configs/reference.toml", payload)
    assert verify_release_artifacts._refuse_private_hosts(exact, repo) == [
        "configs/reference.toml"]

    edited = tmp_path / "edited.whl"
    with zipfile.ZipFile(edited, "w") as archive:
        archive.writestr("configs/reference.toml", payload)
        archive.writestr("configs/edited.toml", payload + b"# edited\n")
    with pytest.raises(SystemExit) as refusal:
        verify_release_artifacts._refuse_private_hosts(edited, repo)
    message = str(refusal.value)
    assert "configs/edited.toml" in message, message
    assert "'configs/reference.toml'" not in message, message
