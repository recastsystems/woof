"""Bridge binaries staged for release carry the revision they were built from.

The 1.4.1 cut nearly shipped prepared bridge bundles that predated the
source tip -- the two platform zips were not even the same source
revision.  Every existing release check hashed the bytes it was handed,
so a stale binary pinned and verified perfectly; nothing anywhere asked
"were these bytes built from the commit being released?".

The closure has two halves, and these tests bind both:

* every bundled artifact's build embeds a plain-text stamp,
  ``GPUWM_BRIDGE_SOURCE_REV=<40-hex git commit>``, in the binary
  (the same convention as the GPUWM_* attribute literals the renderer
  already carries, which is how the staleness was noticed at all); and
* the release tool's ``pin`` step -- the cut-path choke point every
  bundle passes through before a wheel is built -- extracts that stamp
  from every binary member and refuses a bundle whose stamp is absent,
  unparseable, ambiguous, or names any commit other than the one being
  released.

The check reads bytes, never executes the artifact: it must hold on a
GPU-less runner and against a binary for the other platform.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from woof import bridge_assets, bridges, mapped_engine_bridge

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = REPO_ROOT / "tools" / "build_bridge_bundle.py"

MARKER = bridge_assets.SOURCE_REV_MARKER

#: The revision "being released" in these tests, and a stale one.  Both
#: are well-formed 40-hex commits; neither is a hash of anything.
RELEASED = "ab12" * 10
STALE = "cd34" * 10

PLATFORM = "linux-x86_64"
RELEASE = "v0.0.0-stamp"


def _stamp(revision: str) -> bytes:
    return MARKER + revision.encode("ascii")


def _bundle(tmp_path: Path, payloads: dict[str, bytes]) -> Path:
    """A synthetic bundle carrying every artifact and one map asset.

    Only the stamp contents vary per test; everything else satisfies the
    pin step's existing membership and asset requirements so a refusal
    can only come from the staleness check under test.
    """

    archive = tmp_path / f"gpuwm-bridges-{RELEASE}-{PLATFORM}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            name = bridge_assets.artifact_filename(artifact, PLATFORM)
            zf.writestr(name, payloads[artifact.name])
        zf.writestr("assets/basemap/coast.shp", b"not really a shapefile")
    return archive


def _payload_for(artifact, revision: str) -> bytes:
    """What a good build of ``artifact`` looks like to the pin step.

    Two shapes, because two different things are proved.  A
    gpuwm-authored artifact carries the release commit; a VENDORED one
    carries its declared contract marker and no stamp at all, which is
    what a build of an unmodified upstream crate actually looks like.
    """

    if artifact.vendored:
        return (b"\x7fELF junk " + bridges.BRIDGE_ABI_MARKERS[artifact.name]
                + b" tail")
    return b"\x7fELF junk " + _stamp(revision) + b" tail"


def _stamped_payloads(revision: str) -> dict[str, bytes]:
    return {artifact.name: _payload_for(artifact, revision)
            for artifact in bridge_assets.BUNDLED_ARTIFACTS}


def _one_vendored():
    vendored = [a for a in bridge_assets.BUNDLED_ARTIFACTS if a.vendored]
    if not vendored:                             # pragma: no cover - none
        pytest.skip("no vendored artifact is in the bundle")
    return vendored[0]


def _pin(archive: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", RELEASE,
         "--bundle", str(archive), "--out", str(out), *extra],
        capture_output=True, text=True, cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# The extractor and the verifier, over bytes
# ---------------------------------------------------------------------------

def test_extraction_finds_the_stamp_between_arbitrary_bytes():
    payload = b"\x00\x01" + _stamp(RELEASED) + b"\xff" + _stamp(RELEASED)
    assert bridge_assets.embedded_source_revisions(payload) == (RELEASED,)


def test_extraction_reports_distinct_revisions_separately():
    payload = _stamp(RELEASED) + b"::" + _stamp(STALE)
    assert bridge_assets.embedded_source_revisions(payload) == (
        RELEASED, STALE)


def test_extraction_ignores_a_stamp_that_is_not_forty_hex():
    # "unknown" is what a build outside a git checkout embeds.
    payload = MARKER + b"unknown" + b" " + MARKER + b"ABC123"
    assert bridge_assets.embedded_source_revisions(payload) == ()


def test_verification_accepts_the_released_revision():
    bridge_assets.verify_source_revision(
        b"prefix" + _stamp(RELEASED), expected=RELEASED, label="x")


def test_verification_refuses_an_absent_stamp():
    """The exact shape of the near-miss: a binary with no stamp at all."""

    with pytest.raises(bridge_assets.BridgeAssetError) as refusal:
        bridge_assets.verify_source_revision(
            b"no stamp anywhere", expected=RELEASED, label="rw_wrfbatch.exe")
    message = str(refusal.value)
    assert "rw_wrfbatch.exe" in message
    assert "carries no GPUWM_BRIDGE_SOURCE_REV" in message


def test_verification_refuses_an_unparseable_stamp():
    with pytest.raises(bridge_assets.BridgeAssetError) as refusal:
        bridge_assets.verify_source_revision(
            MARKER + b"unknown", expected=RELEASED, label="lib.so")
    assert "outside a git checkout" in str(refusal.value)


def test_verification_refuses_a_stale_revision_naming_both():
    with pytest.raises(bridge_assets.BridgeAssetError) as refusal:
        bridge_assets.verify_source_revision(
            _stamp(STALE), expected=RELEASED, label="grib1_bridge")
    message = str(refusal.value)
    assert STALE in message and RELEASED in message


def test_verification_refuses_two_distinct_revisions():
    with pytest.raises(bridge_assets.BridgeAssetError) as refusal:
        bridge_assets.verify_source_revision(
            _stamp(RELEASED) + _stamp(STALE), expected=RELEASED, label="x")
    assert "distinct" in str(refusal.value)


def test_verification_refuses_a_malformed_expected_revision():
    """The gate cannot be aimed at a non-commit and quietly pass."""

    with pytest.raises(bridge_assets.BridgeAssetError):
        bridge_assets.verify_source_revision(
            _stamp(RELEASED), expected="HEAD", label="x")


def test_grib2_export_abi_marker_matches_binary_bytes(tmp_path):
    marker = bridges.BRIDGE_ABI_MARKERS["rw_grib2export"]
    assert isinstance(marker, bytes)
    binary = tmp_path / "rw_grib2export"
    binary.write_bytes(b"\x7fELF\x00" + marker + b"\x00")
    matched, reason = bridges.bridge_abi_matches("rw_grib2export", binary)
    assert matched, reason
    binary.write_bytes(b"\x7fELF\x00older contract\x00")
    matched, reason = bridges.bridge_abi_matches("rw_grib2export", binary)
    assert not matched
    assert "predates this release" in reason


# ---------------------------------------------------------------------------
# The cut path: pin refuses what it cannot prove
# ---------------------------------------------------------------------------

def test_pin_refuses_a_bundle_with_an_unstamped_binary(tmp_path):
    """Point the check at a binary with no stamp and watch it fire.

    This is the 1.4.1 near-miss re-staged: eight current binaries and
    one stale one that predates the stamped builds.
    """

    payloads = _stamped_payloads(RELEASED)
    payloads["rw_wrfbatch"] = b"stale build, no stamp"
    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, payloads), out,
                  "--source-rev", RELEASED)
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "rw_wrfbatch" in message
    assert "carries no GPUWM_BRIDGE_SOURCE_REV" in message
    assert not out.exists()


def test_pin_refuses_a_bundle_stamped_with_another_revision(tmp_path):
    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, _stamped_payloads(STALE)), out,
                  "--source-rev", RELEASED)
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert STALE in message and RELEASED in message
    assert not out.exists()


def test_pin_takes_a_vendored_artifact_on_its_contract_marker(tmp_path):
    """No stamp is asked of it, and the bundle still pins.

    The vendored payload here carries the marker and no
    GPUWM_BRIDGE_SOURCE_REV at all.  If the cut ever started demanding a
    stamp from it, every release would fail on the one artifact that
    cannot accurately carry one -- its source is upstream's, frozen at a
    recorded commit, and stamping it would mean editing a tree whose
    whole claim is that it is unmodified.
    """

    vendored = _one_vendored()
    payloads = _stamped_payloads(RELEASED)
    assert MARKER not in payloads[vendored.name]
    assert bridges.BRIDGE_ABI_MARKERS[vendored.name] in payloads[vendored.name]
    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, payloads), out, "--source-rev", RELEASED)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pin_refuses_a_vendored_artifact_without_its_contract_marker(
        tmp_path):
    """The staleness that CAN reach it: a build of an older crate.

    A library predating this release's ABI loads cleanly, answers the
    legacy version probe, and then fails inside the first refined solve.
    The marker is what catches it in the cut, statically, without
    executing anything and on the other platform's bytes.
    """

    vendored = _one_vendored()
    payloads = _stamped_payloads(RELEASED)
    payloads[vendored.name] = b"\x7fELF an older vendored build"
    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, payloads), out, "--source-rev", RELEASED)
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert vendored.name in message
    assert "contract marker" in message
    assert vendored.crate in message
    assert not out.exists()


def test_every_vendored_artifact_declares_a_contract_marker():
    """The exemption is not a hole: something still proves the bytes."""

    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        if artifact.vendored:
            assert artifact.name in bridges.BRIDGE_ABI_MARKERS, (
                f"{artifact.name} is exempt from the source-revision stamp "
                "and declares no contract marker either, so nothing at all "
                "proves which build of it a release shipped")


def test_pin_requires_the_source_revision(tmp_path):
    """The gate is not optional: a cut cannot skip it by omission."""

    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, _stamped_payloads(RELEASED)), out)
    assert result.returncode != 0
    assert "--source-rev" in (result.stdout + result.stderr)
    assert not out.exists()


def test_pin_accepts_a_bundle_stamped_with_the_released_revision(tmp_path):
    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, _stamped_payloads(RELEASED)), out,
                  "--source-rev", RELEASED)
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(out.read_text(encoding="utf-8"))
    pins = bridge_assets.parse_pins(document)
    assert pins.release == RELEASE
    assert PLATFORM in pins.platforms


# ---------------------------------------------------------------------------
# The build half: every bundled artifact's source embeds the stamp
# ---------------------------------------------------------------------------

#: Where each artifact's entry point lives, relative to its workspace.
#: An artifact added to BUNDLED_ARTIFACTS without an entry here -- or an
#: entry whose file stops referencing the stamp -- fails below, so a new
#: bridge cannot ship un-provable.
_STAMP_SOURCES = {
    # The wind-turbine table decoder (labs/windfarm, 2.9.11): its build.rs
    # injects GPUWM_BRIDGE_SOURCE_REV and main.rs prints it with the ABI.
    "rw-wind-table": ("src/main.rs", 'env!("GPUWM_BRIDGE_SOURCE_REV")'),
    "rw_icechunk": ("src/main.rs", 'env!("GPUWM_BRIDGE_SOURCE_REV")'),
    "nextgen_hydrofabric": ("src/lib.rs", 'env!("GPUWM_HYDROFABRIC_REV")'),
    "arwen_nextgen_forcing": "src/lib.rs",
    "rw_nextgen_verify": ("src/main.rs", 'env!("GPUWM_BRIDGE_SOURCE_REV")'),
    "rw_zarr": "src/main.rs",
    "arwen-tui": "src/main.rs",
    "grib1_bridge": "src/main.rs",
    "gfs_grib2_bridge": "src/bin/gfs_grib2_bridge.rs",
    "hrrr_grib2_bridge": "src/bin/hrrr_grib2_bridge.rs",
    "grib2_inventory": "src/bin/grib2_inventory.rs",
    "grib2_dump": "src/bin/grib2_dump.rs",
    "gpuwm_preprocess_cpu": "src/lib.rs",
    "rw_fetch": "crates/rw-fetch/src/main.rs",
    "rw_soundings": "crates/rw-soundings/src/main.rs",
    "rw_globe_pack": "crates/rw-globe-pack/src/main.rs",
    "rw_wrfbatch": "crates/rw-wrfbatch/src/main.rs",
    "rw_compare": "crates/rw-wrfbatch/src/bin/compare.rs",
    # The ensemble reducer the AQ line bundled: same crate, same build.rs
    # stamp; its entry point black_boxes SOURCE_REV_STAMP in main.
    "rw_ensbatch": "crates/rw-wrfbatch/src/bin/ensbatch.rs",
    # Fixed-ladder probability maps (WOOF severe ML); same crate, same build.rs stamp.
    "rw_probgrid": "crates/rw-wrfbatch/src/bin/probgrid.rs",
    "rw_verify": "crates/rw-wrfbatch/src/bin/verify.rs",
    "rw_mpas_mesh": "crates/rw-mpas/src/bin/rw_mpas_mesh.rs",
    # One executable with several tools since 2.9.0; the stamp sits in the
    # dispatcher every tool runs through.
    "rw_mpas_static": "crates/rw-mpas/src/bin/rw_mpas_static/main.rs",
    "rw_mpas_init": "crates/rw-mpas/src/bin/rw_mpas_init.rs",
    "rw_mpas_convert": "crates/rw-mpas/src/bin/rw_mpas_convert.rs",
    # The fifth rw-mpas binary, added to BUNDLED_ARTIFACTS by commit
    # 9223560b4.  Its entry point already declares
    # GPUWM_BRIDGE_SOURCE_REV_STAMP and black_boxes it in `main` exactly
    # as its four siblings do, and one `crates/rw-mpas/build.rs` stamps
    # every bin in the crate, so the row below makes this test READ that
    # file rather than exempting it: the artifact was provable and simply
    # unmapped here.
    "rw_mpas_lbc": "crates/rw-mpas/src/bin/rw_mpas_lbc.rs",
    "rw_mpas_init_v831": "crates/rw-mpas/src/bin/rw_mpas_init_v831.rs",
    "rw_mpas_gsl_oro": "crates/rw-mpas/src/bin/rw_mpas_gsl_oro.rs",
    # 2.9.0: the MPAS point renderer (a bin of rw-wrfbatch, stamped by that
    # crate's build.rs) and the TEMPO init library (its own crate and
    # build.rs; the stamp is exported from the ctypes seam).
    "rw_hex_points": "crates/rw-wrfbatch/src/bin/hex_points.rs",
    "rw_tempo": "crates/rw-tempo/src/capi.rs",
    # The ML dataset exporter: its own crate, its own build.rs.
    "rw_mlexport": "crates/rw-mlexport/src/bin/rw_mlexport.rs",
    # The simulated radar: its own crate, whose build.rs injects the
    # revision beside the implementation hash it already emitted.  It
    # joined BUNDLED_ARTIFACTS without a stamp, so the cut would have
    # refused the bundle that carried it.
    "rw_simradar": "crates/rw-simradar/src/main.rs",
    "rw_grib2export": "crates/rw-grib2export/src/bin/rw_grib2export.rs",
    # The nowcast frames converter: a bin of rw-wrfbatch, stamped by that
    # crate's build.rs with rw_compare and rw_verify.
    "rw_nowcast_frames": "crates/rw-wrfbatch/src/bin/nowcast_frames.rs",
    # The observation scorecard and the scoreboard scorer, bundled by the
    # scoreboard lane: rw_scorecard is a bin of rw-wrfbatch (that crate's
    # build.rs stamps it); rw_scoreboard is its own crate with its own build.rs.
    "rw_scorecard": "crates/rw-wrfbatch/src/bin/scorecard.rs",
    "rw_scoreboard": "crates/rw-scoreboard/src/main.rs",
    # 2.9.3: the cyclone field reader behind woof tc-tracks, a bin of
    # rw-wrfbatch stamped by that crate's build.rs.
    "rw_tc_fields": "crates/rw-wrfbatch/src/bin/tc_fields.rs",
    # The GDT-101 remapper, added to BUNDLED_ARTIFACTS by the icon-global
    # lane.  It lives in the bridge crate, so `tools/grib1_bridge/build.rs`
    # stamps it with its five siblings, and its `main` black_boxes the
    # crate's own SOURCE_REV_STAMP; the row below makes this test READ that
    # file rather than exempting it.
    "gdt101_remap": "src/bin/gdt101_remap.rs",
    # The regular-grid intermediate producer, bundled in 2.9.0 (box/
    # rrfs-shape) for public initial-condition preparation: a bin of the
    # bridge crate, stamped by its build.rs, black_boxing SOURCE_REV_STAMP.
    "met_intermediate": "src/bin/met_intermediate.rs",
    "rw_nexrad": "crates/rw-nexrad/src/main.rs",
    "rw_odim": "crates/rw-odim/src/bin/rw_odim.rs",
    "rw_radar_feeds": "crates/rw-radar-feeds/src/bin/rw_radar_feeds.rs",
    "rw_mrms": "crates/rw-obs/src/bin/mrms.rs",
    "rw_stage4": "crates/rw-obs/src/bin/stage4.rs",
    "rw_asos": "crates/rw-obs/src/bin/asos.rs",
    "rw_airnow": "crates/rw-obs/src/bin/airnow.rs",
    "rw_igra2": "crates/rw-obs/src/bin/igra2.rs",
    "rw_amv": "crates/rw-obs/src/bin/amv.rs",
    "rw_ndbc": "crates/rw-obs/src/bin/ndbc.rs",
    "rw_gnssro": "crates/rw-obs/src/bin/gnssro.rs",
    "rw_wis2": "crates/rw-obs/src/bin/wis2.rs",
    "rw_prepbufr": "crates/rw-obs/src/bin/prepbufr.rs",
    "rw_madis": "crates/rw-obs/src/bin/madis.rs",
    "rw_opera": "crates/rw-obs/src/bin/opera.rs",
    "rw_goes": "crates/rw-goes/src/main.rs",
    "rw_netcdf": "crates/rw-netcdf/src/main.rs",
    "netcdf_writer": "crates/netcdf-writer/src/capi.rs",
    "rw_nextgen_output": ("src/lib.rs", 'env!("GPUWM_BRIDGE_SOURCE_REV")'),
    "static_fields": "crates/static-fields/src/capi/mod.rs",
    "gpuwm_mapped_engine": "crates/mapped-engine/src/main.rs",
    # 2.9.1: the projected MET library, the second gpuwm-authored crate
    # in the engine workspace; it builds with mapped-engine's build.rs.
    "rw_projected_intermediate": "crates/projected-intermediate/src/lib.rs",
    "obs_regrid": "crates/obs-regrid/src/capi.rs",
    "obs_score": "crates/obs-score/src/lib.rs",
    "rw_mpas_geometry": "crates/rw-mpas/src/bin/rw_mpas_geometry.rs",
    "rw_mpas_hostprep": "crates/rw-mpas/src/bin/rw_mpas_hostprep.rs",
    "rw_isobaric": "crates/rw-isobaric/src/capi.rs",
    "rw_superob": "crates/rw-superob/src/capi.rs",
    # 2.9.3 improve-1: the default-on hydro door's provider tools.
    "rw_hydro_world": "src/main.rs",
    "rw_hydro_params": "src/main.rs",
    "rw_hydrofabric": ("src/main.rs", "SOURCE_REVISION"),
    "nextgen-reservoir-prepare": "src/main.rs",
    "rw_hydro_init": "src/main.rs",
    # 2.9.4 capability lanes: ocean preparation (cap-waves-couple, its own
    # workspace and build.rs) and the freezing-rain, hail and ocean SST
    # verifiers (cap-fzra, cap-hail, cap-slab-ocean: bins of rw-obs, stamped
    # by that crate's build.rs).
    "rw_ocean": "src/main.rs",
    # cap-surge: the bathymetry tool, its own tools/rw-bathy workspace and build.rs.
    "rw_bathy": "src/main.rs",
    "rw_ice_verify": "crates/rw-obs/src/bin/ice_verify.rs",
    "rw_hail": "crates/rw-obs/src/bin/hail.rs",
    "rw_ocean_verify": "crates/rw-obs/src/bin/ocean_verify.rs",
    "rw_fire_indices": ("crates/fire-indices/src/main.rs", 'env!("GPUWM_BRIDGE_SOURCE_REV")'),
}

#: The build script that injects the revision for each artifact.
_STAMP_BUILDS = {
    "tools/rw_icechunk": ("build.rs",),
    "tools/nextgen_hydrofabric": (("build.rs", "GPUWM_HYDROFABRIC_REV"),),
    "tools/nextgen_forcing": ("build.rs",),
    "tools/nextgen_output": ("build.rs",),
    "tools/nextgen_verify": ("build.rs",),
    "tools/hydro_world": ("build.rs",),
    "tools/hydro_world_params": ("build.rs",),
    "tools/nextgen_reservoir_prepare": ("build.rs",),
    "tools/hydro_world_init": ("build.rs",),
    "tools/ocean_prepare": ("build.rs",),
    "tools/rw-bathy": ("build.rs",),
    "tools/wind-table": ("build.rs",),
    "tools/zarr_bridge": ("build.rs",),
    "tools/arwen-tui": ("build.rs",),
    bridges.CRATE_RELATIVE: ("build.rs",),
    bridges.RUSTWX_CRATE_RELATIVE: ("crates/rw-fetch/build.rs",
                                    "crates/rw-soundings/build.rs",
                                    "crates/rw-wrfbatch/build.rs",
                                    "crates/rw-mpas/build.rs",
                                    "crates/rw-mlexport/build.rs",
                                    "crates/rw-simradar/build.rs",
                                    "crates/rw-nexrad/build.rs",
                                    "crates/rw-odim/build.rs",
                                    # rw-obs builds four bins from one
                                    # crate, so one script stamps all
                                    # four.
                                    "crates/rw-obs/build.rs",
                                    "crates/rw-goes/build.rs",
                                    "crates/rw-netcdf/build.rs",
                                    "crates/netcdf-writer/build.rs",
                                    "crates/static-fields/build.rs",
                                    "crates/obs-regrid/build.rs",
                                    "crates/obs-score/build.rs",
                                    "crates/rw-isobaric/build.rs",
                                    "crates/rw-superob/build.rs",
                                    "crates/rw-scoreboard/build.rs",
                                    "crates/fire-indices/build.rs"),
    # The engine workspace carries one gpuwm-authored crate beside the
    # donor snapshot (tools/rw_wps/VENDOR.md); the snapshot's own crate
    # ships nothing, so it is stamped by nothing.
    mapped_engine_bridge.ENGINE_CRATE_RELATIVE: (
        "crates/mapped-engine/build.rs",),
}


def _require_bridge_sources() -> None:
    if not (REPO_ROOT / bridges.CRATE_RELATIVE / "src").is_dir():
        pytest.skip("the stamp contract needs the source tree, not an "
                    "install")


def test_every_bundled_artifact_entry_point_references_the_stamp():
    _require_bridge_sources()
    unmapped = [artifact.name
                for artifact in bridge_assets.BUNDLED_ARTIFACTS
                if artifact.name not in _STAMP_SOURCES
                and not artifact.vendored]
    assert not unmapped, (
        f"BUNDLED_ARTIFACTS gained {unmapped} with no stamp source mapped "
        "here; a bridge the cut cannot prove must not join a bundle")
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        if artifact.vendored:
            # Its entry point is upstream's and is not edited here, so it
            # is asked for its contract marker instead -- above.
            continue
        entry = _STAMP_SOURCES[artifact.name]
        relative, reference = (entry if isinstance(entry, tuple) else
                               (entry, "SOURCE_REV_STAMP"))
        source = REPO_ROOT / artifact.crate / relative
        text = source.read_text(encoding="utf-8")
        assert reference in text, (
            f"{source} does not reference the source-revision stamp, so "
            f"{artifact.name} would build without one and the cut would "
            "refuse it")


def test_every_bridge_workspace_build_script_injects_the_revision():
    _require_bridge_sources()
    from _woof_snapshot import left_out
    for workspace, scripts in _STAMP_BUILDS.items():
        if left_out(workspace):
            continue
        for entry in scripts:
            relative, variable = (entry if isinstance(entry, tuple) else
                                  (entry, "GPUWM_BRIDGE_SOURCE_REV"))
            script = REPO_ROOT / workspace / relative
            assert script.is_file(), f"{script} is missing"
            text = script.read_text(encoding="utf-8")
            assert f"rustc-env={variable}" in text, (
                f"{script} no longer injects {variable}")
