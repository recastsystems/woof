from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib

import pytest

from conftest import complete_runtime_manifest
from woof import __version__ as gpuwm_version
from woof.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from woof.physics_compat import (
    MYNN_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    RUC_PROFILE_ID,
    WSM6_PROFILE_ID,
)
from woof.source_adapters import (
    AdapterStatus,
    SourceKind,
    get_source_adapter,
    source_adapters,
    source_capability_manifest,
)
from woof.mapped_engine_bridge import ENGINE_RUST as MAPPED_ENGINE_RUST
from woof.source_authorities import packaged_profile, packaged_profile_ids
import woof.source_cli
from woof.source_cli import EXIT_CONFIG, EXIT_USAGE, _parser, main
from woof.source_frame import (
    FieldDescriptor,
    GridDescriptor,
    POLICY_CONTROLLED_FIELDS,
    REQUIRED_3D_FIELDS,
    REQUIRED_SURFACE_FIELDS,
    SourceFrameHeader,
    TimeDescriptor,
    VerticalDescriptor,
    validate_source_frame,
)


EXPECTED_SOURCE_IDS = (
    "aorc",
    "era5-land",
    "ngen-csv",
    "nwm-retrospective-forcing",
    "hrrr",
    "hrrr-prs",
    "gem-gdps",
    "gem-hrdps",
    "gem-rdps",
    "icon-global",
    "icon-eu",
    "icon-d2",
    "hrrr-ak",
    "gfs",
    "gfs-pressure",
    "hafs-a",
    "hafs-b",
    "gdas",
    "gefs-v11-flat",
    "gefs-reforecast",
    "gefs-reforecast-weekly",
    "gefs-reforecast-land",
    "ncep-r1-seaice",
    "gefs-v11",
    "gefs",
    "aigfs",
    "aigefs",
    "hgefs",
    "ecmwf-open-data",
    "ecmwf-open-data-0p1",
    "ecmwf-ens",
    "aifs",
    "earth2",
    "rap-native",
    "hrrr-native",
    "rap",
    "nam",
    "hiresw",
    "href",
    "sref",
    "rtma",
    "urma",
    "nbm",
    "rrfs",
    "rrfs-ens",
    "rrfs-a",
    "rrfs-public",
    "refs",
    "rrfs-firewx",
    "wrf",
    "era5",
    "era5-l137",
    "20crv3",
    "20crv3-cf",
    "hex-global",
    "mapped",
)
ROOT = Path(__file__).parents[1]
EARTH2_MODELS = json.loads((ROOT / "woof/authorities/earth2-models.v1.json").read_text())["models"]


def test_source_cli_import_is_cpu_only():
    script = """
from importlib.abc import MetaPathFinder
import sys

class RejectCupy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cupy" or fullname.startswith("cupy."):
            raise AssertionError(f"unexpected GPU import: {fullname}")
        return None

sys.meta_path.insert(0, RejectCupy())
import woof.source_cli
assert "cupy" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_forecast_start_hour_is_absent_globally_and_defaults_only_in_hrrr():
    parsed = _parser().parse_args([])
    assert parsed.forecast_start_hour is None
    assert parsed.history_interval_seconds is None


def test_registry_covers_bound_inventory_and_external_source_routes_including_earth2():
    adapters = source_adapters()
    from woof.source_adapters import _model_table_rows
    model_adapters = _model_table_rows(adapters)
    base_adapters = tuple(adapter for adapter in adapters if adapter.model_key is None)
    assert tuple(adapter.source_id for adapter in base_adapters) == EXPECTED_SOURCE_IDS
    assert tuple(adapter.source_id for adapter in model_adapters) == tuple(
        "earth2-" + key for key in EARTH2_MODELS)
    assert len({adapter.source_id for adapter in adapters}) == len(adapters)
    assert [adapter.source_id for adapter in base_adapters if adapter.runnable] == [
        "hrrr",
        "hrrr-prs",
        "gem-gdps",
    "gem-hrdps",
    "gem-rdps",
        "icon-global",
        "icon-eu",
        "icon-d2",
        "gfs",
        "gfs-pressure",
        "hafs-a",
        "hafs-b",
        "gdas",
        "gefs-v11-flat",
        "gefs-reforecast",
        "gefs-reforecast-weekly",
        "gefs-v11",
        "gefs",
        "aigfs",
        "aigefs",
        "ecmwf-open-data",
        "ecmwf-open-data-0p1",
        "ecmwf-ens",
        "aifs",
        "earth2",
        "rap-native",
        "hrrr-native",
        "rap",
        "rrfs",
        "rrfs-ens",
        "era5",
        "era5-l137",
        "20crv3",
        "20crv3-cf",
        "hex-global",
        "mapped",
    ]
    assert [adapter.source_id for adapter in model_adapters if adapter.runnable] == [
        "earth2-" + key for key, row in EARTH2_MODELS.items() if row.get("profile")]
    # The arbitrary-acceptance seam, asserted as a property rather than as
    # a list: a source whose mapping ships is decoded through the generic
    # mapped runner, so it costs table rows and JSON, never a runner.
    # The one narrowing: an atmosphere-only profile whose composition role
    # is a PENDING declaration decodes but is not runnable, and its row
    # must name what has to be composed.
    from woof.source_authorities import packaged_profile
    for adapter in adapters:
        if adapter.packaged_profile is None:
            continue
        assert adapter.packaged_profile in packaged_profile_ids()
        state = packaged_profile(
            adapter.packaged_profile)["composition_state"]
        if state == "pending_cross_source":
            assert adapter.runnable is False
            assert adapter.composition_requirement
        else:
            assert adapter.runnable is True
    hrrr = get_source_adapter("HRRR")
    assert hrrr.status is AdapterStatus.CERTIFIED
    assert hrrr.stock_wrf_gate.startswith("wrf-v4.6.1-pass")
    assert get_source_adapter("rrfs_a").source_id == "rrfs-a"
    assert get_source_adapter("ifs").source_id == "ecmwf-open-data"
    era5 = get_source_adapter("era5")
    assert era5.upstream_model_id is None
    assert era5.status is AdapterStatus.CERTIFIED
    assert era5.runner == "era5_combined_grib1_v1"
    assert era5.stock_wrf_gate.startswith("wrf-v4.6.1-pass")
    gfs = get_source_adapter("gfs-0.25")
    assert gfs.status is AdapterStatus.CERTIFIED
    assert gfs.runner == "gfs_pgrb2_0p25_v1"
    assert gfs.stock_wrf_gate.startswith("wrf-v4.6.1-pass")
    twentycr = get_source_adapter("20cr")
    assert twentycr.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert twentycr.runner == "twentycrv3_member_grib2_v1"
    assert "max_dom=4" in twentycr.notes
    assert twentycr.packaged_profile == "20crv3-member-grib2-v1"
    twentycr_cf = get_source_adapter("20cr-netcdf")
    assert twentycr_cf.source_id == "20crv3-cf"
    assert twentycr_cf.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert twentycr_cf.runnable is True
    # No runner of its own: the packaged profile IS the difference.
    assert twentycr_cf.runner == "mapped_composition_v1"
    assert twentycr_cf.packaged_profile == "20crv3-netcdf-v1"
    assert twentycr_cf.source_kind is SourceKind.ENSEMBLE_STATISTIC
    assert "ENSEMBLE MEAN" in twentycr_cf.notes
    mapped = get_source_adapter("mapping-v1")
    assert mapped.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert mapped.runnable is True
    assert mapped.runner == "mapped_composition_v1"
    assert mapped.stock_wrf_gate == "exact-mapping-composition-hash-evidence-required"


def test_manifest_binds_inventory_and_does_not_confuse_decode_with_readiness():
    manifest = source_capability_manifest()
    assert manifest["schema"] == "gpuwm-native-source-adapters-v1"
    assert manifest["runtime_forbidden"] == ["WPS", "real.exe"]
    assert manifest["rusty_weather_inventory"]["model_id_count"] == 23
    assert len(manifest["rusty_weather_inventory"]["head"]) == 40
    assert manifest["source_count"] == len(source_adapters())
    assert manifest["runnable_source_count"] == sum(row.runnable for row in source_adapters())
    assert set(manifest["packaged_source_authorities"]) == set(
        packaged_profile_ids())
    for profile_id, pins in manifest["packaged_source_authorities"].items():
        # The fourth pin appears only for a profile that declares an input
        # normalizer: the document that makes an unreadable native grid
        # reachable is sealed in the manifest like the other three.
        extra = ({"normalization"}
                 if packaged_profile(profile_id).get("input_normalizer")
                 else set())
        assert set(pins) == {"mapping", "composition", "provenance"} | extra
        assert all(len(digest) == 64 for digest in pins.values())
    assert all("stock_wrf_gate" in value for value in manifest["sources"])
    assert len(manifest["mapped_stock_wrf_evidence"]) == 2
    assert all(
        len(value["mapping_sha256"]) == len(value["composition_sha256"]) == 64
        for value in manifest["mapped_stock_wrf_evidence"]
    )
    era5_grib1 = manifest["mapped_stock_wrf_evidence"][0]
    assert era5_grib1["gate"] == "wrf-v4.6.1-pass-era5-grib1-single-domain"
    # Replacing the named soil packings changes the composition authority.
    # Exact historical stock-WRF evidence must not silently certify the new
    # declarative contract even though unit gates prove identical soil
    # arrays for ERA5/GFS.
    assert era5_grib1["composition_sha256"] != hashlib.sha256(
        (ROOT / "configs" / era5_grib1["composition_config"]).read_bytes()
    ).hexdigest()

    current_gfs = manifest["mapped_stock_wrf_evidence"][1]
    assert current_gfs["mapping_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json").read_bytes()
    ).hexdigest()
    assert current_gfs["composition_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-gfs-terrain.composition.json").read_bytes()
    ).hexdigest()
    assert "clean-d66e442" in current_gfs["gate"]
    assert current_gfs["wrf_commit"] == \
        "d66e442fccc04111067e29274c9f9eaccc3cef28"
    assert current_gfs["wrf_exe_sha256"] == \
        "cfac96554c8f9796c7522aaf023131ea7681ddf12110a327e51a548958874089"
    assert "d01 120x100x49 at 12 km" in current_gfs["domain_envelope"]
    assert "d02-d04" in current_gfs["domain_envelope"]
    assert "60x60x49 at 4 km" in current_gfs["domain_envelope"]
    assert current_gfs["stock_wrf_result"] == \
        "exit_0_success_complete_wrf_d02_d03_d04_six_steps_each"
    assert current_gfs["stock_wrf_input_receipt_sha256"] == \
        "25a643fe34ff1ddd39464129bdcfff074e938ced216608f8cab8d17196ea524c"

    invalidated = manifest["invalidated_mapped_stock_wrf_evidence"]
    assert len(invalidated) == 2
    gfs = invalidated[0]
    assert gfs["gate"] == "wrf-v4.6.1-pass-gfs-grib2-d01-d02"
    assert gfs["mapping_sha256"] == \
        "726677d8c2365e6f533cc6dd5d7c795e198164326660c3630d885c83f406a11e"
    assert gfs["composition_sha256"] == \
        "266c98099b24f03a3bc986f275b44bbd6bf20dce1006ad05f6da39bc4a373bfb"
    assert gfs["replacement_mapping_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json").read_bytes()
    ).hexdigest()
    assert gfs["replacement_composition_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-gfs-terrain.composition.json").read_bytes()
    ).hexdigest()
    assert gfs["replacement_status"] == "stock_wrf_certified_d01_d04_z49"
    assert gfs["replacement_gate"] == current_gfs["gate"]
    assert "evidence does not transfer" in gfs["reason"]

    era5_netcdf = invalidated[1]
    assert era5_netcdf["gate"] == "wrf-v4.6.1-pass-era5-netcdf-single-domain"
    assert era5_netcdf["mapping_sha256"] == \
        "f278705331a81767d4d3532ff4dd4f739242a79b224747f51e722d462142daa8"
    assert era5_netcdf["replacement_mapping_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-era5-netcdf.mapping.json").read_bytes()
    ).hexdigest()
    assert "evidence does not transfer" in era5_netcdf["reason"]
    # No stock wrf.exe has been run against the replacement bytes, so the
    # entry must not name a passing gate for them.  This is the field that
    # keeps "invalidated" from becoming a quiet re-certification.
    assert era5_netcdf["replacement_status"] == "stock_wrf_regate_required"
    assert era5_netcdf["replacement_gate"] is None


def test_every_retained_stock_wrf_evidence_still_names_the_bytes_we_ship():
    """Retained evidence binds the CURRENT config, or it is not evidence.

    The failure this closes, found on the 2.5.0 line: a commit changed
    ``configs/rw-wps-era5-netcdf.mapping.json`` (adding the accepted-spelling
    list for ECMWF's time/valid_time rename) and left the stock-WRF evidence
    pinned to the pre-change bytes.  ``rw-wps --list-sources`` then printed a
    ``mapping_sha256`` matching no file the product ships, next to a gate
    string claiming stock wrf.exe had accepted it.

    The old test caught it only because it happened to name that pair by
    hand.  This one names nothing: every retained entry declares the config
    it binds, and its hash must be that file's hash.  A future authority
    that moves without its evidence being re-earned fails here, and an entry
    that forgets to declare its config fails here too.
    """

    retained = source_capability_manifest()["mapped_stock_wrf_evidence"]
    assert retained, "there is no retained stock-WRF evidence to check"
    checked = 0
    drifted: list[str] = []
    for evidence in retained:
        name = evidence.get("mapping_config")
        assert isinstance(name, str) and name, (
            f"retained evidence for gate {evidence.get('gate')!r} does not "
            f"name the mapping config it binds, so nothing can check it")
        path = ROOT / "configs" / name
        assert path.is_file(), f"{name} is named by evidence but not shipped"
        checked += 1
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if evidence["mapping_sha256"] != current:
            drifted.append(
                f"{name}: evidence names {evidence['mapping_sha256']} for gate "
                f"{evidence.get('gate')!r} but the shipped file is {current}")
    assert checked == len(retained)
    assert not drifted, (
        "retained stock-WRF evidence names bytes this product no longer "
        "ships.  Re-run the gate on the new authority, or move the entry to "
        "invalidated_mapped_stock_wrf_evidence with a reason -- do NOT bump "
        "the hash, which silently certifies changed bytes with a gate that "
        "was never re-run:\n  " + "\n  ".join(drifted))


def _time() -> TimeDescriptor:
    return TimeDescriptor(
        reference_time="2026-07-18T00:00:00+00:00",
        valid_time="2026-07-18T01:00:00+00:00",
        lead_seconds=3600,
    )


def _field(name: str) -> FieldDescriptor:
    is_3d = name in REQUIRED_3D_FIELDS or name == "air_pressure"
    if name in {"soil_temperature", "volumetric_soil_moisture"}:
        dimensions = ("soil_depth", "y", "x")
        shape = (4, 3, 4)
        vertical = "soil"
    elif is_3d:
        dimensions = ("level", "y", "x")
        shape = (2, 3, 4)
        vertical = "pressure"
    else:
        dimensions = ("y", "x")
        shape = (3, 4)
        vertical = None
    return FieldDescriptor(
        canonical_name=name,
        units="1" if "fraction" in name else "SI",
        dimensions=dimensions,
        grid_location="mass",
        vertical_coordinate=vertical,
        time=_time(),
        data_reference=f"sha256:{name}",
        dtype="<f4",
        shape=shape,
        source_field=name.upper(),
    )


def _valid_header() -> SourceFrameHeader:
    names = sorted(REQUIRED_3D_FIELDS | REQUIRED_SURFACE_FIELDS | {"air_pressure"})
    return SourceFrameHeader(
        source_id="test",
        source_cycle="2026-07-18T00:00:00+00:00",
        grid=GridDescriptor(
            projection="lambert_conformal",
            nx=4,
            ny=3,
            earth_shape="sphere:6371229m",
            scan_order="+x,+y",
            wind_basis="earth_relative",
            parameters={"lov": -97.5},
        ),
        vertical_coordinates={
            "pressure": VerticalDescriptor(
                coordinate="pressure",
                level_count=2,
                level_values=(100000.0, 90000.0),
            ),
            "soil": VerticalDescriptor(
                coordinate="soil_depth",
                level_count=4,
                level_values=(0.05, 0.25, 0.70, 1.50),
                units="m",
                positive="down",
            ),
        },
        fields=tuple(_field(name) for name in names),
        initialization_policies={
            name: "explicit_zero_with_adapter_validation"
            for name in POLICY_CONTROLLED_FIELDS
        },
    )


def test_canonical_source_frame_accepts_complete_explicit_state():
    validate_source_frame(_valid_header())


def test_canonical_source_frame_rejects_missing_science_state():
    header = _valid_header()
    fields = tuple(
        value for value in header.fields if value.canonical_name != "specific_humidity"
    )
    with pytest.raises(ValueError, match="specific_humidity"):
        validate_source_frame(
            SourceFrameHeader(**{**header.__dict__, "fields": fields})
        )


def test_canonical_source_frame_rejects_ambiguous_missing_policy():
    header = _valid_header()
    policies = dict(header.initialization_policies)
    policies.pop("sea_ice_fraction")
    with pytest.raises(ValueError, match="sea_ice_fraction"):
        validate_source_frame(
            SourceFrameHeader(
                **{**header.__dict__, "initialization_policies": policies}
            )
        )


def test_interval_time_semantics_are_explicit():
    bad_time = TimeDescriptor(
        reference_time="2026-07-18T00:00:00Z",
        valid_time="2026-07-18T01:00:00Z",
        lead_seconds=3600,
        statistic="accumulation",
        interval_start="2026-07-18T00:00:00Z",
        interval_end="2026-07-18T01:00:00Z",
    )
    header = _valid_header()
    first = header.fields[0]
    fields = (
        FieldDescriptor(**{**first.__dict__, "time": bad_time}),
        *header.fields[1:],
    )
    with pytest.raises(ValueError, match="accumulation_reset"):
        validate_source_frame(
            SourceFrameHeader(**{**header.__dict__, "fields": fields})
        )


def test_cli_lists_machine_readable_inventory(capsys):
    assert main(["--list-sources"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_count"] == len(source_adapters())
    assert (
        payload["canonical_source_frame"]["schema"] == "gpuwm-canonical-source-frame-v1"
    )


def test_cli_reports_installed_version_without_run_arguments(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == f"RW-WPS {gpuwm_version}"


def test_cli_exposes_fail_closed_public_support_matrix(capsys):
    assert main(["--show-support-matrix"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "gpuwm-native-wrf-support-matrix-v1"
    assert payload["public_release_ready"] is False
    assert payload["release_state"] \
        == (
            "dedicated_rw_wps_cross_checkout_reproducible_clean_linux_cpu_"
            "package_gate"
        )
    assert (
        payload["final_products"]["wrfinput_d02_through_d06"]
        == "stock_wrf_proof_passed_for_certified_hrrr_slice"
    )
    assert payload["final_products"]["wrfinput_d02"] == (
        "stock_wrf_proof_passed_for_certified_hrrr_slice_and_canonical_lf_"
        "current_contract_gfs"
    )
    assert "stock-WRF proof passed through declared max_dom 4" in (
        payload["meteorological_sources"]["gfs_pgrb2_0p25"][
            "nested_hierarchy"
        ]
    )
    assert (
        payload["final_products"]["wrfinput_d07_through_d21"]
        == "parser_and_atomic_export_covered_not_stock_wrf_gated"
    )
    assert (
        payload["final_products"]["wrfinput_d03_through_dNN"]
        == "deprecated_aggregate_key_see_split_depth_keys"
    )
    assert payload["final_products"]["wrfinput_d02_through_d04_gfs"] == (
        "stock_wrf_proof_passed_for_canonical_lf_current_contract"
    )
    assert (
        payload["execution_backends"]["parallel_per_domain_initialization"]
        == "implemented_for_hrrr_d01_through_d21_and_live_gated_through_d06"
    )
    twentycr = payload["meteorological_sources"]["20crv3_member_grib2"]
    assert "adapter_and_native_preparation_implemented" in twentycr["single_domain"]
    assert "max_dom_4" in twentycr["nested_hierarchy"]
    assert twentycr["public_cli"] == (
        "rw-wps_source_20crv3_with_create_only_member_manifest_authoring"
    )
    assert "packaged_in_wheel" in twentycr["distribution"]
    assert "without_cupy_import" in (
        payload["execution_backends"]["cpu_only_setup_export"]
    )
    assert (
        payload["release_artifacts"]["model_independent_python_wheel"]
        == "dedicated_rw_wps_distribution_clean_linux_cpu_gate_pass_"
           "forecast_executors_absent"
    )
    assert "license_owner_decision" in payload["known_release_blockers"]


def test_cli_refuses_unimplemented_source_before_touching_files(capsys):
    assert main(["--source", "nam"]) == EXIT_CONFIG
    error = capsys.readouterr().err
    assert "REFUSED source=nam" in error
    # The refusal and the adapter's own reason print by default; the
    # certification-gate paragraph is the mechanism half.
    assert "stock-wrf evidence is a separate certification gate" not in error
    assert "--explain" in error

    assert main(["--source", "nam", "--explain"]) == EXIT_CONFIG
    explained = capsys.readouterr().err
    assert "REFUSED source=nam" in explained
    assert "stock-wrf evidence is a separate certification gate" in explained


def test_a_refusal_with_nothing_to_add_does_not_echo_its_own_status(capsys, monkeypatch):
    """`status=adapter_mapping_required: adapter_mapping_required`.

    The reason fell back to the status value, which is already printed
    two tokens earlier, so an adapter carrying neither a composition
    requirement nor notes said the same thing twice -- and a doubled
    token reads as a truncated message, which sent a test host 8 pilot
    looking for the rest of a sentence that was never there.  Fixed
    source-agnostically: bare rows (and every adapter added later
    with nothing yet to say) were still echoing.  `rap` and then
    `gdas` left this test when they became runnable packaged
    profiles; `nam` and `rrfs-public` are the remaining bare
    MAPPING_REQUIRED rows.
    """

    # The once-per-process provenance banner precedes the refusal in
    # whichever test of this worker first runs a front door; this test is
    # about the refusal line, so the banner is off here.
    from woof.provenance_gate import BANNER_ENV
    monkeypatch.setenv(BANNER_ENV, "0")
    for source in ("nam", "rrfs-public"):
        assert main(["--source", source]) == EXIT_CONFIG
        error = capsys.readouterr().err
        first = error.splitlines()[0]
        status = get_source_adapter(source).status.value
        assert first == f"REFUSED source={source} status={status}"
        assert first.count(status) == 1, first
        # The paragraph that DOES explain the refusal is untouched --
        # one flag away, and word for word what it always said.
        assert "--explain" in error
        assert main(["--source", source, "--explain"]) == EXIT_CONFIG
        explained = capsys.readouterr().err
        assert "stock-wrf evidence is a separate certification gate"             in explained

    # An adapter that has something to say still says it, on the same
    # line and in the same shape.
    assert main(["--source", "hrrr-ak"]) == EXIT_CONFIG
    error = capsys.readouterr().err
    assert error.splitlines()[0].startswith(
        "REFUSED source=hrrr-ak status="
        f"{get_source_adapter('hrrr-ak').status.value}: ")
    assert "Alaska grid/projection and field contract" in error


def test_cli_reports_the_runnable_20crv3_netcdf_route(capsys):
    assert main(["--show-source", "20crv3-netcdf"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_id"] == "20crv3-cf"
    assert payload["runnable"] is True
    assert payload["status"] == "runnable_mapping_not_stock_wrf_certified"
    assert payload["packaged_profile"] == "20crv3-netcdf-v1"
    assert payload["runner"] == "mapped_composition_v1"
    # The two limits that must never be readable only in prose somewhere
    # else: it is the ensemble MEAN, and its invariants are recovered.
    assert "ENSEMBLE MEAN" in payload["notes"]
    assert "published time-invariant surface height and land fraction" in payload["notes"]


def _twentycr_args():
    return [
        "--source",
        "20crv3",
        "--source-manifest",
        "/case/member072.manifest.json",
        "--source-manifest-sha256",
        "abc123",
        "--grib2-inventory",
        "/bin/grib2_inventory",
        "--grib2-dump",
        "/bin/grib2_dump",
        "--wps-namelist",
        "/case/namelist.wps",
        "--geog-root",
        "/static/WPS_GEOG",
        "--experiment-config",
        "/case/experiment.toml",
        "--output-root",
        "/output",
    ]


def test_cli_routes_20crv3_through_immutable_packaged_authorities(capsys):
    result = main(
        _twentycr_args()
        + [
            "--preprocess-backend",
            "cpu",
            "--preprocess-workers",
            "8",
            "--cpu-preprocess-bridge",
            "/bin/libgpuwm_preprocess_cpu.so",
            "--hierarchy-workers",
            "8",
            "--dry-run",
        ]
    )

    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m woof.twentycrv3_wrf" in command
    assert "woof/authorities/rw-wps-20crv3-member-grib2.mapping.json" in command
    assert "woof/authorities/rw-wps-20crv3-member-grib2.composition.json" in command
    assert "woof/authorities/rw-wps-20crv3-member-grib2.provenance.json" in command
    assert "--manifest /case/member072.manifest.json" in command
    assert "--hierarchy-workers 8" in command


def test_cli_20crv3_rejects_authority_override(capsys):
    result = main(
        _twentycr_args()
        + ["--mapping", "/case/replacement.json", "--dry-run"]
    )

    assert result == EXIT_USAGE
    assert "--mapping is not used by --source 20crv3" in capsys.readouterr().err


@pytest.mark.parametrize("engine", ("rust", "python"))
def test_cli_20crv3_forwards_the_mapped_engine_flag(engine, capsys):
    """The member door honours the engine selector it forwards.

    Named breakage this pins against: before the port, this route built
    its bundle in `woof.twentycrv3_wrf`'s host Python and never called
    the composition, so `--mapped-engine rust` either exited 0 while
    Python decoded (the measured defect) or was refused outright (the
    member lane's stopgap).  The port routed the door through
    `decode_composed_source`, so the route follows the engine table like
    every other composed source and the flag means what it says: it
    rides the child command, where `woof.twentycrv3_wrf` selects the
    engine that actually composes.  Silently DROPPING the flag here
    would resurrect the original defect -- an asked-for engine that
    never arrives.

    Both spellings, because both now select a real engine on a route
    that has one.
    """

    result = main(
        _twentycr_args() + ["--mapped-engine", engine, "--dry-run"])

    assert result == 0
    command = capsys.readouterr().out
    assert f"--mapped-engine {engine}" in command
    assert "-m woof.twentycrv3_wrf" in command


@pytest.mark.parametrize("source", ("gfs", "era5", "20crv3"))
@pytest.mark.parametrize(
    "flag",
    (
        "--forecast-start-hour",
        "--forecast-end-hour",
        "--history-interval-seconds",
    ),
)
def test_non_hrrr_sources_reject_explicit_zero_hrrr_only_flag(
        source, flag, capsys):
    result = main([
        "--source", source,
        flag, "0",
        "--dry-run",
    ])

    assert result == EXIT_USAGE
    assert f"{flag} is not used by --source {source}" \
        in capsys.readouterr().err


def _twentycr_source(tmp_path):
    """One two-time member inventory for the authoring door checks."""
    source = tmp_path / "member072"
    source.mkdir()
    for stamp in ("1932032100", "1932032103"):
        for role in ("pl", "sfc"):
            (source / f"mem072_{stamp}_{role}.grb2").write_bytes(
                f"fixture:{stamp}:{role}".encode()
            )
    manifest = tmp_path / "member072.manifest.json"
    return source, manifest


def test_cli_20crv3_authors_create_only_member_manifest(
    tmp_path: Path, capsys
):
    source, manifest = _twentycr_source(tmp_path)

    result = main([
        "--source",
        "20crv3",
        "--source-root",
        str(source),
        "--author-input-manifest",
        str(manifest),
        "--author-only",
    ])

    assert result == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["manifest"]["member"] == "072"
    assert receipt["manifest"]["file_count"] == 4
    assert receipt["manifest"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()


def test_cli_20crv3_authoring_says_what_to_do_with_the_manifest(
    tmp_path: Path, capsys
):
    """Parity with the GFS route's authoring step.

    GFS ends by printing the whole front-door command with its digest
    filled in; every mapped authoring step prints an ``AUTHORED`` line.
    20CRv3 printed nothing, so a user who had just watched a manifest be
    written still had to locate it and compute its SHA-256 by hand.
    """

    source, manifest = _twentycr_source(tmp_path)

    assert main([
        "--source", "20crv3",
        "--source-root", str(source),
        "--author-input-manifest", str(manifest),
        "--author-only",
    ]) == 0

    captured = capsys.readouterr()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    # stdout stays the machine-readable receipt.
    assert json.loads(captured.out)["status"] == "PASS"

    lines = captured.err.splitlines()
    assert f"AUTHORED input_manifest={manifest.resolve()} sha256={digest}" \
        in lines
    assert any("next:" in line for line in lines)
    # The half it knows is the door's own command, bound and exact -- no
    # placeholder, no digest for the user to compute.
    bound = [line.strip() for line in lines
             if line.strip().startswith("woof prep ")]
    assert len(bound) == 1
    assert shlex.split(bound[0], posix=os.name != "nt") == [
        "woof", "prep", "--source", "20crv3",
        "--source-manifest", str(manifest.resolve()),
        "--source-manifest-sha256", digest]
    assert "<" not in bound[0] and ">" not in bound[0]

    # And the half it cannot know is named, as comments, because
    # authoring refuses those flags outright.  The GRIB2 tool pair is
    # refused too, but it is not the reader's to supply: the door
    # resolves it, and naming it pins the Python decoder.
    comments = "\n".join(line for line in lines if line.strip().startswith("#"))
    for flag in ("--wps-namelist", "--geog-root", "--experiment-config",
                 "--output-root"):
        assert flag in comments, flag
    assert "--grib2" not in comments
    for flag in ("--grib2-inventory", "--grib2-dump", "--wps-namelist",
                 "--geog-root", "--experiment-config", "--output-root"):
        # Refused at authoring: this is a real limit, not an oversight.
        assert main([
            "--source", "20crv3",
            "--source-root", str(source),
            "--author-input-manifest", str(tmp_path / "again.json"),
            "--author-only",
            flag, str(tmp_path),
        ]) != 0


def test_the_20crv3_handoff_runs_at_the_door_with_exactly_the_flags_it_names(
    tmp_path: Path, monkeypatch, capsys
):
    """The printed line, completed with the flags its comment names, runs.

    The authoring step printed the manifest pair with no command and no
    ``--source`` in front of it, so the line did not run as printed, and
    it named ``--grib2-inventory`` and ``--grib2-dump`` among the
    reader's flags: completed with those, the door forwarded both tool
    pins and the member route decoded on the Python engine instead of
    the default one.  The check is the door itself, through ``woof prep
    --dry-run``, which validates the whole argument contract and prints
    the child command without opening an input.
    """

    from woof import cli
    from woof.mapped_engine_bridge import ENGINE_ENV

    monkeypatch.delenv(ENGINE_ENV, raising=False)
    source, manifest = _twentycr_source(tmp_path)
    assert main([
        "--source", "20crv3",
        "--source-root", str(source),
        "--author-input-manifest", str(manifest),
        "--author-only",
    ]) == 0
    lines = capsys.readouterr().err.splitlines()
    bound = [line.strip() for line in lines
             if line.strip().startswith("woof prep ")]
    assert len(bound) == 1
    argv = shlex.split(bound[0], posix=os.name != "nt")[2:]
    comments = " ".join(line.strip().lstrip("#").strip() for line in lines
                        if line.strip().startswith("#"))
    named = re.search(r"cannot bind them: (.*?) are yours", comments)
    assert named is not None, comments
    flags = re.findall(r"--[a-z][a-z0-9-]*", named.group(1))
    assert flags == ["--wps-namelist", "--geog-root", "--experiment-config",
                     "--output-root"]
    values = {
        "--wps-namelist": tmp_path / "namelist.wps",
        "--geog-root": tmp_path / "geog",
        "--experiment-config": tmp_path / "experiment.toml",
        "--output-root": tmp_path / "prepared",
    }
    completed = argv + [token for flag in flags
                        for token in (flag, str(values[flag]))]
    assert cli.main(["prep", *completed, "--dry-run"]) == 0
    said = capsys.readouterr()
    assert "-m woof.twentycrv3_wrf" in said.out
    assert f"--manifest {manifest.resolve().as_posix()}" in said.out.replace("'", "")
    # The default engine: no decoder tool rides the child command.
    assert "--grib2-inventory" not in said.out
    assert "--grib2-dump" not in said.out
    # One flag fewer is refused by name: every named flag is one the
    # door needs, not decoration.
    for needed in flags:
        at = completed.index(needed)
        cut = completed[:at] + completed[at + 2:]
        assert cli.main(["prep", *cut, "--dry-run"]) != 0
        assert needed in capsys.readouterr().err


def _mapped_args(source_format="grib2"):
    values = [
        "--source",
        "mapped",
        "--source-format",
        source_format,
        "--mapping",
        "/case/mapping.json",
        "--composition",
        "/case/composition.json",
        "--input",
        "/data/source-f000.grb2",
        "--input",
        "/data/source-f003.grb2",
        "--supplement",
        "terrain=/data/terrain.grb2",
        "--provenance",
        "terrain=/data/terrain-provenance.md",
        "--wps-namelist",
        "/case/namelist.wps",
        "--geog-root",
        "/static/WPS_GEOG",
        "--experiment-config",
        "/case/experiment.toml",
        "--source-sha256s",
        "/case/input-manifest.json",
        "--source-sha256s-sha256",
        "abc123",
        "--output-root",
        "/output",
    ]
    if source_format == "grib1":
        values.extend(("--bridge", "/bin/grib1_bridge"))
    elif source_format == "grib2":
        values.extend(
            (
                "--grib2-inventory",
                "/bin/grib2_inventory",
                "--grib2-dump",
                "/bin/grib2_dump",
            )
        )
    return values


def test_cli_routes_mapped_hierarchy_to_native_composition_engine(capsys):
    assert (
        main(
            _mapped_args()
            + [
                "--preprocess-backend",
                "cpu",
                "--preprocess-workers",
                "8",
                "--cpu-preprocess-bridge",
                "/bundle/libgpuwm_preprocess_cpu.so",
                "--hierarchy-workers",
                "8",
                "--dry-run",
            ]
        )
        == 0
    )
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m woof.mapped_direct" in command
    assert command.count("--input ") == 2
    assert "--supplement terrain=/data/terrain.grb2" in command
    assert "--provenance terrain=/data/terrain-provenance.md" in command
    assert "--grib2-inventory /bin/grib2_inventory" in command
    assert "--grib2-dump /bin/grib2_dump" in command
    assert "--preprocess-backend cpu" in command
    assert "--hierarchy-workers 8" in command


def test_cli_mapped_rejects_hrrr_history_interval(capsys):
    result = main(
        _mapped_args()
        + ["--history-interval-seconds", "3600", "--dry-run"]
    )

    assert result == EXIT_USAGE
    assert "--history-interval-seconds is not used by --source mapped" in (
        capsys.readouterr().err
    )


def test_cli_mapped_fails_closed_on_mixed_args(capsys):
    """An omitted GRIB2 tool flag is no longer a usage error.

    The dispatch resolves the two tools through the shared bridge
    ladder when they are omitted (tests/test_prep_tool_ladder.py owns
    that contract), so the only usage error left in this command line
    is the flag that belongs to another route -- and it must still be
    reported as one, before the estate is consulted.
    """

    args = _mapped_args()
    dump_index = args.index("--grib2-dump")
    del args[dump_index : dump_index + 2]
    result = main(args + ["--gfs-series", "/wrong/route.tsv", "--dry-run"])
    assert result == EXIT_USAGE
    error = capsys.readouterr().err
    assert "is required for mapped grib2" not in error
    assert "--gfs-series is not used by --source mapped" in error


def test_cli_mapped_dry_run_rejects_malformed_or_duplicate_role_bindings(capsys):
    args = _mapped_args("netcdf")
    supplement_index = args.index("--supplement")
    args[supplement_index + 1] = "bad role=/data/terrain.nc"
    args.extend(
        (
            "--provenance",
            "terrain=/data/duplicate-provenance.md",
        )
    )
    assert main(args + ["--dry-run"]) == EXIT_USAGE
    error = capsys.readouterr().err
    assert "--supplement must use" in error
    assert "--provenance repeats singleton role 'terrain'" in error


# ---------------------------------------------------------------------------
# The --input-list spelling, and the argv-limit relaunch net behind it.
#
# A field-per-file source publishes one file per field per level per lead,
# so one initial state is hundreds of --input flags -- and Windows caps a
# whole command line at 32 KB (CreateProcess reports the excess as
# WinError 206).  Two answers, one grammar: --input-list carries the same
# ordered paths as a file, and when the per-file spelling's RELAUNCH is
# what the platform refuses, the dispatch rewrites its own inner command
# through a temporary list file and retries once.  A command line the
# platform accepts is never rewritten -- pinned below.
# ---------------------------------------------------------------------------

def _mapped_args_with_input_list(tmp_path, source_format="grib2"):
    args = _mapped_args(source_format)
    while "--input" in args:
        index = args.index("--input")
        del args[index : index + 2]
    input_list = tmp_path / "inputs.list"
    input_list.write_bytes(
        b"/data/source-f000.grb2\r\n\r\n/data/source-f003.grb2\n"
    )
    args.extend(("--input-list", str(input_list)))
    return args, input_list


def test_cli_mapped_input_list_composes_the_list_not_the_expansion(
    tmp_path, capsys
):
    args, input_list = _mapped_args_with_input_list(tmp_path)
    assert main(args + ["--dry-run"]) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m woof.mapped_direct" in command
    assert "--input-list " + str(input_list).replace("\\", "/") in command
    assert "--input /" not in command


def test_cli_mapped_refuses_both_input_spellings(tmp_path, capsys):
    args, _ = _mapped_args_with_input_list(tmp_path)
    args.extend(("--input", "/data/source-f006.grb2"))
    assert main(args + ["--dry-run"]) == EXIT_USAGE
    assert "choose exactly one of --input or --input-list" in (
        capsys.readouterr().err
    )


def test_cli_mapped_requires_exactly_one_input_spelling(capsys):
    args = _mapped_args()
    while "--input" in args:
        index = args.index("--input")
        del args[index : index + 2]
    assert main(args + ["--dry-run"]) == EXIT_USAGE
    assert "choose exactly one of --input or --input-list" in (
        capsys.readouterr().err
    )


def test_cli_mapped_input_list_missing_or_empty_refuses_before_any_work(
    tmp_path, capsys
):
    args, input_list = _mapped_args_with_input_list(tmp_path)
    input_list.write_bytes(b"\r\n\n")
    assert main(args + ["--dry-run"]) == EXIT_USAGE
    assert "names no input files" in capsys.readouterr().err

    input_list.unlink()
    assert main(args + ["--dry-run"]) == EXIT_USAGE
    assert "--input-list" in capsys.readouterr().err


def test_cli_gfs_route_refuses_the_input_list_flag(capsys):
    result = main(
        ["--source", "gfs", "--input-list", "/case/inputs.list", "--dry-run"]
    )
    assert result == EXIT_USAGE
    assert "--input-list is not used by --source gfs" in (
        capsys.readouterr().err
    )


def test_cli_twentycr_route_refuses_the_input_list_flag(capsys):
    result = main(
        ["--source", "20crv3", "--input-list", "/case/inputs.list",
         "--dry-run"]
    )
    assert result == EXIT_USAGE
    assert "--input-list is not used by --source 20crv3" in (
        capsys.readouterr().err
    )


def test_cli_mapped_authoring_hashes_the_files_the_input_list_names(
    tmp_path, monkeypatch, capsys
):
    args, _ = _mapped_args_with_input_list(tmp_path)
    for flag in ("--source-sha256s", "--source-sha256s-sha256"):
        index = args.index(flag)
        del args[index : index + 2]
    args.extend(
        (
            "--author-input-manifest",
            str(tmp_path / "authored.inputs.json"),
            "--author-only",
        )
    )
    observed = {}

    def manifest(output, **kwargs):
        observed["manifest"] = (output, kwargs)
        return {"source_format": "grib2",
                "manifest": {"path": str(output), "sha256": "2" * 64}}

    monkeypatch.setattr("woof.source_cli.author_input_manifest", manifest)

    assert main(args) == 0
    assert observed["manifest"][1]["primary_files"] == [
        Path("/data/source-f000.grb2"),
        Path("/data/source-f003.grb2"),
    ]
    assert "AUTHORED input_manifest=" in capsys.readouterr().err


def test_cli_mapped_relaunch_retries_the_argv_limit_through_a_list_file(
    monkeypatch,
):
    calls: list[list[str]] = []
    recorded = {}

    def fake_run(command, check=False):
        calls.append(list(command))
        if len(calls) == 1:
            raise OSError(errno.E2BIG, "Argument list too long")
        list_file = Path(command[command.index("--input-list") + 1])
        recorded["path"] = list_file
        recorded["content"] = list_file.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr("woof.source_cli._run_adapter_command", fake_run)

    assert woof.source_cli._run_native_adapter(
        woof.source_cli._mapped_command(_parser().parse_args(_mapped_args()))) == 7
    assert len(calls) == 2
    first, second = calls
    assert first.count("--input") == 2
    assert "--input-list" not in first
    # The retry is the SAME command with the per-file pairs carried by
    # the list file instead -- nothing else moves, and the list carries
    # the first command's own path tokens verbatim.
    base = []
    moved = []
    index = 0
    while index < len(first):
        if first[index] == "--input":
            moved.append(first[index + 1])
            index += 2
            continue
        base.append(first[index])
        index += 1
    assert second == base + ["--input-list", str(recorded["path"])]
    assert recorded["content"] == "\n".join(moved) + "\n"
    assert len(moved) == 2
    # The temporary list file does not outlive the stage.
    assert not recorded["path"].exists()


@pytest.mark.skipif(sys.platform != "win32",
                    reason="winerror is a Windows-only OSError slot")
def test_cli_mapped_relaunch_retries_createprocess_error_206(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, check=False):
        calls.append(list(command))
        if len(calls) == 1:
            raise OSError(22, "The filename or extension is too long",
                          None, 206)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("woof.source_cli._run_adapter_command", fake_run)

    assert woof.source_cli._run_native_adapter(
        woof.source_cli._mapped_command(_parser().parse_args(_mapped_args()))) == 0
    assert len(calls) == 2
    assert "--input-list" in calls[1]


def test_cli_mapped_relaunch_success_never_rewrites_the_command(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, check=False):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("woof.source_cli._run_adapter_command", fake_run)

    assert woof.source_cli._run_native_adapter(
        woof.source_cli._mapped_command(_parser().parse_args(_mapped_args()))) == 0
    assert len(calls) == 1
    assert calls[0].count("--input") == 2
    assert "--input-list" not in calls[0]


def test_cli_relaunch_other_launch_failures_still_report_and_exit_70(
    monkeypatch, capsys
):
    calls: list[list[str]] = []

    def fake_run(command, check=False):
        calls.append(list(command))
        raise OSError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr("woof.source_cli._run_adapter_command", fake_run)

    assert woof.source_cli._run_native_adapter(
        woof.source_cli._mapped_command(_parser().parse_args(_mapped_args()))) == 70
    assert len(calls) == 1
    assert "failed to launch native adapter" in capsys.readouterr().err


def test_cli_relaunch_argv_limit_on_the_compact_spelling_reports(
    tmp_path, monkeypatch, capsys
):
    calls: list[list[str]] = []

    def fake_run(command, check=False):
        calls.append(list(command))
        raise OSError(errno.E2BIG, "Argument list too long")

    monkeypatch.setattr("woof.source_cli._run_adapter_command", fake_run)

    args, _ = _mapped_args_with_input_list(tmp_path)
    assert woof.source_cli._run_native_adapter(
        woof.source_cli._mapped_command(_parser().parse_args(args))) == 70
    assert len(calls) == 1
    assert "failed to launch native adapter" in capsys.readouterr().err


def test_cli_mapped_author_only_authors_descriptor_and_exact_manifest(
    monkeypatch, capsys
):
    args = _mapped_args()
    for flag in (
        "--mapping",
        "--source-sha256s",
        "--source-sha256s-sha256",
    ):
        index = args.index(flag)
        del args[index : index + 2]
    args.extend(
        (
            "--descriptor",
            "/case/source.descriptor.json",
            "--vtable",
            "/case/Vtable.GENERIC",
            "--author-mapping",
            "/case/generated.mapping.json",
            "--author-input-manifest",
            "/case/generated.inputs.json",
        )
    )
    observed = {}

    def mapping(descriptor, output, *, vtable_path, expected_format):
        observed["mapping"] = (
            descriptor,
            output,
            vtable_path,
            expected_format,
        )
        return {
            "format": "grib2",
            "mapping": {"sha256": "1" * 64},
        }

    def manifest(output, **kwargs):
        observed["manifest"] = (output, kwargs)
        return {
            "source_format": "grib2",
            "manifest": {"path": str(output), "sha256": "2" * 64},
        }

    monkeypatch.setattr("woof.source_cli.author_mapping", mapping)
    monkeypatch.setattr("woof.source_cli.author_input_manifest", manifest)

    assert main(args + ["--author-only"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["mapping"]["mapping"]["sha256"] == "1" * 64
    assert result["input_manifest"]["manifest"]["sha256"] == "2" * 64
    assert observed["mapping"] == (
        Path("/case/source.descriptor.json"),
        Path("/case/generated.mapping.json"),
        Path("/case/Vtable.GENERIC"),
        "grib2",
    )
    assert observed["manifest"][1]["primary_files"] == [
        Path("/data/source-f000.grb2"),
        Path("/data/source-f003.grb2"),
    ]
    assert "AUTHORED mapping=" in captured.err
    assert "AUTHORED input_manifest=" in captured.err


def test_cli_dry_run_is_side_effect_free_for_authoring(monkeypatch, capsys):
    called = False

    def author(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("woof.source_cli.author_mapping", author)
    args = _mapped_args()
    args.extend(
        (
            "--descriptor",
            "/case/source.descriptor.json",
            "--author-mapping",
            "/case/generated.mapping.json",
            "--author-input-manifest",
            "/case/generated.inputs.json",
        )
    )
    assert main(args + ["--dry-run"]) == EXIT_USAGE
    assert called is False
    assert "--dry-run is side-effect free" in capsys.readouterr().err


def test_cli_mapped_authoring_modes_are_exclusive_and_fail_before_write(
    monkeypatch, capsys
):
    called = False

    def author(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("woof.source_cli.author_mapping", author)
    result = main(
        _mapped_args()
        + [
            "--descriptor",
            "/case/also-descriptor.json",
            "--author-mapping",
            "/case/generated.mapping.json",
            "--vtable",
            "/case/Vtable",
        ]
    )
    assert result == EXIT_USAGE
    assert not called
    assert "choose exactly one of --mapping" in capsys.readouterr().err


def test_cli_mapped_author_only_does_not_require_run_geometry(monkeypatch, capsys):
    args = _mapped_args()
    for flag in (
        "--wps-namelist",
        "--geog-root",
        "--experiment-config",
        "--output-root",
        "--source-sha256s",
        "--source-sha256s-sha256",
    ):
        index = args.index(flag)
        del args[index : index + 2]
    args.extend(
        (
            "--author-input-manifest",
            "/case/generated.inputs.json",
            "--author-only",
        )
    )
    monkeypatch.setattr(
        "woof.source_cli.author_input_manifest",
        lambda *_args, **_kwargs: {
            "source_format": "grib2",
            "manifest": {"path": "/case/generated.inputs.json", "sha256": "3" * 64},
            "status": "PASS",
        },
    )

    assert main(args) == 0
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert receipt["schema"] == "rw-wps.contract-authoring.v1"
    assert receipt["input_manifest"]["status"] == "PASS"
    assert "AUTHORED input_manifest=" in captured.err


def test_cli_combined_authoring_rolls_back_owned_mapping_on_manifest_failure(
    tmp_path, monkeypatch, capsys
):
    args = _mapped_args()
    for flag in ("--mapping", "--source-sha256s", "--source-sha256s-sha256"):
        index = args.index(flag)
        del args[index : index + 2]
    mapping_path = tmp_path / "generated.mapping.json"
    manifest_path = tmp_path / "generated.inputs.json"
    args.extend(
        (
            "--descriptor",
            str(tmp_path / "descriptor.json"),
            "--vtable",
            str(tmp_path / "Vtable"),
            "--author-mapping",
            str(mapping_path),
            "--author-input-manifest",
            str(manifest_path),
            "--author-only",
        )
    )

    def mapping(_descriptor, output, **_kwargs):
        output = Path(output)
        output.write_bytes(b"owned mapping\n")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        receipt = {"mapping": {"sha256": digest}}
        output.with_name(f"{output.stem}.authoring.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        return receipt

    monkeypatch.setattr("woof.source_cli.author_mapping", mapping)
    monkeypatch.setattr(
        "woof.source_cli.author_input_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("manifest failed")),
    )

    assert main(args) == EXIT_CONFIG
    assert not mapping_path.exists()
    assert not mapping_path.with_name(f"{mapping_path.stem}.authoring.json").exists()
    assert not manifest_path.exists()
    assert "manifest failed" in capsys.readouterr().err


def test_cli_validates_preprocess_options_before_authoring(
    tmp_path, monkeypatch, capsys
):
    args = _mapped_args()
    for flag in ("--source-sha256s", "--source-sha256s-sha256"):
        index = args.index(flag)
        del args[index : index + 2]
    output = tmp_path / "inputs.json"
    args.extend(
        (
            "--author-input-manifest",
            str(output),
            "--preprocess-backend",
            "cuda",
            "--cpu-preprocess-bridge",
            str(tmp_path / "libgpuwm_preprocess_cpu.so"),
        )
    )
    called = False

    def author(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("woof.source_cli.author_input_manifest", author)
    assert main(args) == EXIT_USAGE
    assert called is False
    assert not output.exists()
    assert "cpu-preprocess-bridge requires" in capsys.readouterr().err


def _hrrr_args(*, cycle="2026-07-18_00:00:00",
               static_cache="/case/static.npz", static_receipt="/case/static.json"):
    """Fresh arguments for the cached-static HRRR preparation door."""
    return [
        "--source", "hrrr", "--source-root", "/source",
        "--source-sha256s", "/source/SHA256SUMS",
        "--source-sha256s-sha256", "abc123",
        "--static-cache", static_cache, "--static-receipt", static_receipt,
        "--namelist-input", "/case/namelist.input", "--valid-time", cycle,
        "--output-root", "/output",
    ]


def _gfs_inputs():
    """The common GFS series and decoder fixture."""
    return [
        "--gfs-series", "/source/gfs-series.tsv", "--cycle", "2026-07-20_00:00:00",
        "--bridge", "/bin/gfs_grib2_bridge",
    ]


def _regular_args(*, experiment="/case/experiment.toml"):
    """Fresh static and authority arguments shared by regular-grid source doors."""
    return [
        "--wps-namelist", "/case/namelist.wps",
        "--static-input", "/case/static.npz", "--static-receipt", "/case/static-receipt.json",
        "--experiment-config", experiment,
        "--source-sha256s", "/source/input-manifest.json",
        "--source-sha256s-sha256", "abc123", "--output-root", "/output",
    ]


def test_cli_hrrr_dry_run_routes_to_certified_internal_adapter(capsys):
    result = main(
        [
            *_hrrr_args(static_cache='/static/cache.npz', static_receipt='/static/receipt.json'),
            '--pipeline-workers', '8',
            '--history-interval-seconds', '3600',
            '--prepare-workers', '4',
            '--dry-run',
        ]
    )
    assert result == 0
    command = capsys.readouterr().out
    assert "prepare_hrrr_wrf.py" in command
    assert "/source/SHA256SUMS" in command.replace("\\", "/")
    assert "--run-seconds" not in command
    assert "--forecast-start-hour 0" in command
    # The relay substitutes the ROUTE's own default when the caller names
    # no profile, and reads it from the route rather than keeping a copy:
    # this relay and the wizard door disagreed for exactly as long as they
    # each held their own literal.
    assert "--physics-profile" not in command
    assert "--pipeline-workers 8" in command
    assert "--history-interval-seconds 3600.0" in command
    assert command.rstrip().endswith("--prepare-workers 4")


def test_cli_hrrr_relays_cpu_workers_under_the_preparers_auto_default(capsys):
    """The door relays a bare backend and the CPU worker count unchanged.

    The preparer it relays to resolves a bare backend with ``auto``,
    which lands on the CPU where no card is usable and takes the worker
    count there, so the door must pass both through as given.
    """

    result = main([
        *_hrrr_args(static_cache='/static/cache.npz', static_receipt='/static/receipt.json'),
        '--preprocess-workers', '4',
        '--dry-run',
    ])
    assert result == 0, capsys.readouterr().err
    command = capsys.readouterr().out
    assert "--preprocess-workers 4" in command
    assert "--preprocess-backend" not in command


@pytest.mark.parametrize("cadence", ("0", "-1", "nan", "inf"))
def test_cli_hrrr_rejects_invalid_history_interval(cadence, capsys):
    result = main(
        [
            *_hrrr_args(static_cache='/static/cache.npz', static_receipt='/static/receipt.json'),
            '--history-interval-seconds', cadence,
            '--dry-run',
        ]
    )
    assert result == EXIT_USAGE
    assert "history-interval-seconds must be positive and finite" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("cycle", "start", "end"),
    (("2026-07-18_05:00:00", "12", "18"),
     ("2026-07-18_18:00:00", "40", "46")),
)
def test_cli_hrrr_routes_absolute_source_windows_and_thompson_profile(
        cycle, start, end, capsys):
    result = main([
        *_hrrr_args(static_cache='/static/cache.npz', static_receipt='/static/receipt.json', cycle=cycle),
        '--run-seconds', str(6 * 3600),
        '--forecast-start-hour', start,
        '--forecast-end-hour', end,
        '--physics-profile', 'thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1',
        '--dry-run',
    ])
    assert result == 0
    command = capsys.readouterr().out
    assert f"--forecast-start-hour {start}" in command
    assert f"--forecast-end-hour {end}" in command
    assert "--physics-profile thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1" \
        in command


def test_cli_hrrr_rejects_source_window_beyond_standard_cycle(capsys):
    result = main([
        *_hrrr_args(static_cache='/static/cache.npz', static_receipt='/static/receipt.json', cycle='2026-07-18_05:00:00'),
        '--run-seconds', '3600',
        '--forecast-start-hour', '18',
        '--forecast-end-hour', '19',
        '--dry-run',
    ])
    assert result == EXIT_USAGE
    assert "horizon f18" in capsys.readouterr().err


def test_cli_hrrr_domain_dry_run_is_one_command_from_geog(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--geog-root",
            "/static/WPS_GEOG",
            "--domain-spec",
            "/case/domain.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_06:00:00",
            "--output-root",
            "/output",
            "--pipeline-workers",
            "8",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "prepare_hrrr_wrf.py" in command
    assert command.index("/static/WPS_GEOG") < command.index("/case/domain.json")
    assert "--run-seconds" not in command
    assert "--pipeline-workers 8" in command


def _hrrr_hierarchy_args():
    return [
        "--source",
        "hrrr",
        "--root-preparation",
        "/prepared/root",
        "--domain-spec",
        "/case/domain.json",
        "--wps-namelist",
        "/case/namelist.wps",
        "--namelist-input",
        "/case/namelist.native.input",
        "--stock-wrf-namelist-input",
        "/case/namelist.stock.input",
        "--geog-root",
        "/static/WPS_GEOG",
        "--source-sha256s",
        "/source/SHA256SUMS",
        "--source-sha256s-sha256",
        "abc123",
        "--valid-time",
        "2026-07-18_00:00:00",
        "--output-root",
        "/output",
    ]


def test_cli_hrrr_hierarchy_routes_to_public_parallel_orchestrator(capsys):
    result = main(
        _hrrr_hierarchy_args()
        + [
            "--child-workers",
            "8",
            "--cpu-preprocess-bridge",
            "/bundle/libgpuwm_preprocess_cpu.so",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m woof.hrrr_hierarchy_direct" in command
    assert "--root-preparation /prepared/root" in command
    assert "--stock-wrf-namelist-input /case/namelist.stock.input" in command
    assert "--workers 8" in command
    assert "--cpu-preprocess-bridge /bundle/libgpuwm_preprocess_cpu.so" in command


def test_cli_hrrr_hierarchy_forwards_the_cycle_and_the_lead_separately(
        capsys):
    """One door flag, two derived clocks.

    ``woof source --source hrrr --valid-time`` is the CYCLE -- it is
    validated as "an exact hourly HRRR cycle" and passed to
    ``hrrr_source_window`` as one.  Forwarding that string to the
    hierarchy under its own ``--valid-time``, which meant MODEL TIME
    ZERO, handed it a time K hours early at every nonzero lead.  Both
    values now go through, spelled for what they are.
    """
    result = main(
        _hrrr_hierarchy_args() + ["--forecast-start-hour", "6", "--dry-run"])
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--cycle 2026-07-18_00:00:00" in command
    assert "--forecast-start-hour 6" in command
    assert "--valid-time" not in command


def test_cli_hrrr_hierarchy_at_lead_zero_is_unchanged(capsys):
    """Backward compatibility: the same instant, by the same arithmetic."""
    result = main(_hrrr_hierarchy_args() + ["--dry-run"])
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--cycle 2026-07-18_00:00:00" in command
    assert "--forecast-start-hour 0" in command


def test_cli_hrrr_hierarchy_rejects_history_interval(capsys):
    result = main(
        _hrrr_hierarchy_args()
        + ["--history-interval-seconds", "3600", "--dry-run"]
    )

    assert result == EXIT_USAGE
    assert "--history-interval-seconds is not used by HRRR hierarchy export" in (
        capsys.readouterr().err
    )


def test_public_hierarchy_help_and_installed_entrypoint_cover_d01_through_d21():
    help_text = _parser().format_help()
    assert "d01..dNN hierarchy export" in help_text
    assert "max_dom 1..21" in help_text
    assert "d02..dNN" in help_text
    assert "initialization (1..32)" in help_text
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["scripts"]["woof-wrf-init"] == "woof.source_cli:main"
    assert project["project"]["scripts"]["rw-wps"] == "woof.source_cli:main"


def test_public_support_matrix_separates_live_depth_from_max_dom_coverage(capsys):
    assert main(["--show-support-matrix"]) == 0
    matrix = json.loads(capsys.readouterr().out)
    assert (
        matrix["domain_geometry"]["grid_ids"]
        == "contiguous_parent_before_child_d01_through_d21"
    )
    assert (
        matrix["final_products"]["wrfinput_d02_through_d06"]
        == "stock_wrf_proof_passed_for_certified_hrrr_slice"
    )
    assert (
        matrix["final_products"]["wrfinput_d07_through_d21"]
        == "parser_and_atomic_export_covered_not_stock_wrf_gated"
    )


@pytest.mark.parametrize("workers", (0, 33))
def test_cli_hrrr_hierarchy_rejects_out_of_bounds_workers(workers, capsys):
    result = main(
        _hrrr_hierarchy_args() + ["--child-workers", str(workers), "--dry-run"]
    )
    assert result == EXIT_USAGE
    assert "child-workers must be between 1 and 32" in capsys.readouterr().err


def test_cli_hrrr_hierarchy_refuses_single_domain_source_arguments(capsys):
    result = main(
        _hrrr_hierarchy_args() + ["--source-root", "/source/raw", "--dry-run"]
    )
    assert result == EXIT_USAGE
    assert (
        "--source-root is not used by HRRR hierarchy export" in capsys.readouterr().err
    )


def test_cli_hrrr_refuses_mixed_static_modes(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--geog-root",
            "/static/WPS_GEOG",
            "--domain-spec",
            "/case/domain.json",
            "--static-cache",
            "/case/static.npz",
            "--static-receipt",
            "/case/static.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_00:00:00",
            "--output-root",
            "/output",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert "cannot be mixed" in capsys.readouterr().err


def test_cli_hrrr_refuses_non_hourly_cycle(capsys):
    result = main(
        [
            *_hrrr_args(cycle='2026-07-18_00:30:00'),
            '--dry-run',
        ]
    )
    assert result == EXIT_USAGE
    assert "exact hourly HRRR cycle" in capsys.readouterr().err


def test_cli_hrrr_rejects_worker_count_above_decoder_limit(capsys):
    result = main(
        [
            *_hrrr_args(),
            '--pipeline-workers', '65',
            '--dry-run',
        ]
    )
    assert result == EXIT_USAGE
    assert "between 1 and 64" in capsys.readouterr().err


def test_cli_hrrr_accepts_64_decoder_hour_workers(capsys):
    result = main(
        [
            *_hrrr_args(),
            '--pipeline-workers', '64',
            '--dry-run',
        ]
    )

    assert result == 0
    assert "--pipeline-workers 64" in capsys.readouterr().out


def test_cli_hrrr_relays_no_stock_wrf_export_to_both_preparers(capsys):
    """U-E6: `woof prep --source hrrr --no-stock-wrf-export` reaches the
    preparer instead of being refused as an ERA5-only flag.

    The single-domain preparer's requested export is fail-closed, so a
    door that refused the opt-out left an HRRR-recipe preparation whose
    export cannot be written no way to finish (an ERA5 downscale lane,
    2026-10-26).  It is spelled
    --skip-stock-wrf-export there and --no-stock-wrf-export on the
    hierarchy stage, which turns it into the "off" export mode.
    """
    from woof import hrrr_hierarchy_direct

    result = main([*_hrrr_args(), "--no-stock-wrf-export", "--dry-run"])
    captured = capsys.readouterr()
    assert result == 0, captured.err
    assert "is not used by --source hrrr" not in captured.err
    assert "--skip-stock-wrf-export" in captured.out

    args = woof.source_cli._parser().parse_args([
        "--source", "hrrr", "--root-preparation", "/root",
        "--domain-spec", "/case/d01.json", "--wps-namelist", "/case/w",
        "--namelist-input", "/case/n", "--stock-wrf-namelist-input",
        "/case/s", "--geog-root", "/geog", "--source-sha256s", "/m",
        "--source-sha256s-sha256", "abc", "--valid-time",
        "2026-07-18_00:00:00", "--output-root", "/out",
        "--no-stock-wrf-export"])
    command = woof.source_cli._hrrr_command(args)
    assert command[2] == "woof.hrrr_hierarchy_direct"
    parsed = hrrr_hierarchy_direct._parser().parse_args(command[3:])
    assert parsed.stock_wrf_export == "off"
    args.no_stock_wrf_export = False
    parsed = hrrr_hierarchy_direct._parser().parse_args(
        woof.source_cli._hrrr_command(args)[3:])
    assert parsed.stock_wrf_export == "optional"


def test_cli_hrrr_refuses_era5_only_arguments(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--grib",
            "/source/era5.grb",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert "--grib is not used by --source hrrr" in capsys.readouterr().err


def test_cli_era5_dry_run_routes_to_certified_internal_adapter(capsys):
    result = main(
        [
            "--source",
            "era5",
            "--grib",
            "/source/era5.grb",
            "--vtable",
            "/source/Vtable.ERA5_CDO",
            "--bridge",
            "/bin/grib1_bridge",
            "--wps-namelist",
            "/case/namelist.wps",
            "--static-input",
            "/case/static.npz",
            "--static-receipt",
            "/case/static-receipt.json",
            "--source-orography",
            "/case/met_em.nc",
            "--experiment-config",
            "/case/experiment.toml",
            "--source-sha256s",
            "/source/input-manifest.json",
            "--source-sha256s-sha256",
            "abc123",
            "--output-root",
            "/output",
            "--preprocess-backend",
            "auto",
            "--preprocess-workers",
            "4",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m woof.era5_direct" in command
    assert "--bridge" in command and "/bin/grib1_bridge" in command
    assert "--input-manifest" in command
    assert "/source/input-manifest.json" in command
    assert "--input-manifest-sha256 abc123" in command
    assert "--source-orography-variable SOILHGT" in command
    assert "--preprocess-backend auto" in command
    assert "--preprocess-workers 4" in command


def test_cli_era5_d06_hierarchy_routes_all_domain_contracts(capsys):
    arguments = [
        "--source", "era5",
        "--grib", "/source/era5.grb",
        "--vtable", "/source/Vtable.ERA5_CDO",
        "--bridge", "/bin/grib1_bridge",
        "--wps-namelist", "/case/namelist.wps",
        "--experiment-config", "/case/experiment.toml",
        "--source-sha256s", "/source/input-manifest.json",
        "--source-sha256s-sha256", "abc123",
        "--output-root", "/output",
        "--geog-root", "/wps-geog",
        "--hierarchy-workers", "6",
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "8",
        "--cpu-preprocess-bridge", "/bundle/libgpuwm_preprocess_cpu.so",
    ]
    assert main(arguments + ["--dry-run"]) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--geog-root /wps-geog" in command
    assert "--hierarchy-workers 6" in command
    assert "--static-input" not in command
    assert "--source-orography" not in command


def test_cli_era5_explicit_orography_contract_is_atomic(capsys):
    result = main([
        "--source", "era5",
        "--geog-root", "/wps-geog",
        "--domain-source-orography", "d01=/case/d01.nc",
        "--dry-run",
    ])
    assert result == EXIT_USAGE
    assert "--source-orography is required with explicit" in capsys.readouterr().err


def test_cli_era5_refuses_hrrr_only_arguments(capsys):
    result = main(
        [
            "--source",
            "era5",
            "--source-root",
            "/hrrr-style-directory",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    error = capsys.readouterr().err
    assert "--source-root is not used by --source era5" in error
    assert "--grib" in error


def test_cli_gfs_dry_run_routes_to_certified_internal_adapter(capsys):
    result = main(
        [
            '--source', 'gfs',
            *_gfs_inputs(),
            *_regular_args(),
            '--preprocess-backend', 'cpu',
            '--preprocess-workers', '8',
            '--cpu-preprocess-bridge', '/bundle/libgpuwm_preprocess_cpu.so',
            '--dry-run',
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m woof.gfs_direct" in command
    assert "--series" in command
    assert "/source/gfs-series.tsv" in command
    assert "--cycle 2026-07-20_00:00:00" in command
    assert "--preprocess-backend cpu" in command
    assert "--preprocess-workers 8" in command
    assert "--cpu-preprocess-bridge /bundle/libgpuwm_preprocess_cpu.so" in command


@pytest.mark.parametrize(
    ("profile", "acknowledgement"),
    (
        (MYNN_PROFILE_ID, None),
        (RUC_PROFILE_ID, None),
        (NOAHMP_PROFILE_ID, "noahmp-host-column-throughput-v1"),
    ),
)
def test_cli_gfs_accepts_every_newly_reachable_profile(
        profile, acknowledgement, capsys):
    arguments = [
        '--source', 'gfs',
        '--preprocess-backend', 'cpu',
        *_gfs_inputs(),
        *_regular_args(),
        '--physics-profile', profile,
        '--dry-run',
    ]
    if acknowledgement is not None:
        arguments[-1:-1] = [
            "--ack", acknowledgement]

    assert main(arguments) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert f"--physics-profile {profile}" in command
    if acknowledgement is not None:
        assert f"--ack {acknowledgement}" in command


def test_cli_gfs_noahmp_warns_without_expert_acknowledgement(capsys):
    result = main([
        '--source', 'gfs',
        '--preprocess-backend', 'cpu',
        *_gfs_inputs(),
        *_regular_args(),
        '--physics-profile', NOAHMP_PROFILE_ID,
        '--dry-run',
    ])
    assert result == 0
    assert "noahmp-host-column-throughput-v1" in capsys.readouterr().err


def _declare_acknowledgement(config: Path, token: str) -> None:
    """Add TOKEN to ``[experiment].acknowledgements``, the way a user does.

    `woof domain` already emits declarations of its own for some
    profiles (a longwave-OFF suite carries constant-downward-longwave-v1),
    so a second ``acknowledgements =`` key spliced in beside that one is a
    DUPLICATE KEY and the file stops being decodable TOML.  Extending the
    array that is there is the edit a caller actually makes, and the
    duplicate-key shape has a refusal test of its own below.
    """
    lines = config.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("acknowledgements = ["):
            head, _, tail = line.rstrip("\n").rpartition("]")
            separator = "" if head.endswith("[") else ", "
            lines[index] = f'{head}{separator}"{token}"]{tail}\n'
            break
    else:
        lines = "".join(lines).replace(
            "[experiment]\n",
            f'[experiment]\nacknowledgements = ["{token}"]\n', 1).splitlines(
                keepends=True)
    config.write_text("".join(lines), encoding="utf-8")
    # Self-checking fixture: if the wizard's emission moves again, this
    # fails here, naming the edit, instead of surfacing as an unexplained
    # front-door exit code three asserts later.
    declared = tomllib.loads(
        config.read_text(encoding="utf-8"))["experiment"]["acknowledgements"]
    assert token in declared, declared


@pytest.mark.gpu
# ^ not for device work: `woof domain` requires the CuPy runtime for its
#   sizing estimator, so this CLI-level test needs the GPU estate the CI
#   runner does not have.  It evades the import-cupy auto-marker because
#   the requirement is inside the CLI it invokes.  The TOML-acknowledgement
#   acceptance logic itself is covered CPU-only by the registry/plan tests.
def test_cli_gfs_accepts_noahmp_acknowledgement_from_experiment_toml(
        tmp_path, capsys):
    from woof.cli import main as gpuwm_main

    experiment = tmp_path / "experiment.toml"
    assert gpuwm_main([
        "domain", "--point=39.7,-96.6", "--card", "24gb",
        "--ladder", "12", "--source", "gfs",
        "--physics-profile", WSM6_PROFILE_ID,
        "--cycle", "2026-07-29T18", "--out", str(experiment),
    ]) == 0
    _declare_acknowledgement(experiment, "noahmp-host-column-throughput-v1")
    capsys.readouterr()

    result = main([
        '--source', 'gfs',
        *_gfs_inputs(),
        *_regular_args(experiment=str(experiment)),
        '--physics-profile', NOAHMP_PROFILE_ID,
        '--dry-run',
    ])

    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert f"--physics-profile {NOAHMP_PROFILE_ID}" in command
    assert "--ack" not in command


def test_cli_gfs_names_an_undecodable_experiment_config_not_the_ack_it_holds(
        tmp_path, capsys):
    """The concrete breakage: a config that cannot be decoded silently
    dropped the whole TOML acknowledgement channel, and the refusal that
    followed told the caller to declare the acknowledgement their own
    config already carried.  A duplicate ``acknowledgements =`` key is
    exactly how a hand-edited wizard config gets there -- the wizard
    emits one of its own for a longwave-OFF suite -- so the caller reads
    "add acknowledgements = [...]", looks at the line they added, and has
    nothing to go on.  The decode fault is what this door must name.
    """

    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        "[experiment]\n"
        'acknowledgements = ["noahmp-host-column-throughput-v1"]\n'
        'name = "duplicate-key"\n'
        'acknowledgements = ["constant-downward-longwave-v1"]\n',
        encoding="utf-8",
    )

    result = main([
        '--source', 'gfs',
        *_gfs_inputs(),
        *_regular_args(experiment=str(experiment)),
        '--physics-profile', NOAHMP_PROFILE_ID,
        '--dry-run',
    ])

    assert result == EXIT_USAGE
    message = capsys.readouterr().err
    # It names the file, the fact that it did not decode, and the decoder's
    # own position -- which is what points at the duplicate key.
    assert "--experiment-config" in message
    assert "experiment.toml" in message
    assert "TOML" in message
    assert "Cannot overwrite a value" in message
    # And it does NOT send the caller back to write the declaration that
    # is sitting in the file it could not read.
    assert "--ack noahmp-host-column-throughput-v1" not in message


def test_cli_gfs_defers_a_decodable_but_invalid_experiment_config(
        tmp_path, capsys):
    """Decodable TOML that is not a valid experiment stays the direct
    front door's diagnostic to give, exactly as before: this door reads
    the acknowledgement channel and nothing else, so it must not grow a
    second, poorer copy of the experiment loader's refusals.
    """

    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        '[experiment]\nname = "no-tables-at-all"\n', encoding="utf-8")

    result = main([
        # A profile that needs no acknowledgement, so the only thing that
        # could refuse here is a diagnostic this door should not be giving.
        '--source', 'gfs',
        *_gfs_inputs(),
        *_regular_args(experiment=str(experiment)),
        '--physics-profile', WSM6_PROFILE_ID,
        '--dry-run',
    ])

    assert result == 0
    assert "is not decodable TOML" not in capsys.readouterr().err


def test_cli_gfs_d06_hierarchy_routes_source_neutral_controls(capsys):
    result = main([
        '--source', 'gfs',
        *_gfs_inputs(),
        '--wps-namelist', '/case/namelist.wps',
        '--experiment-config', '/case/experiment.toml',
        '--source-sha256s', '/source/input-manifest.json',
        '--source-sha256s-sha256', 'abc123',
        '--output-root', '/output',
        '--geog-root', '/wps-geog',
        '--hierarchy-workers', '6',
        '--preprocess-backend', 'cpu',
        '--preprocess-workers', '8',
        '--cpu-preprocess-bridge', '/bundle/libgpuwm_preprocess_cpu.so',
        '--dry-run',
    ])
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m woof.gfs_direct" in command
    assert "--geog-root /wps-geog" in command
    assert "--hierarchy-workers 6" in command
    assert "--static-input" not in command


@pytest.mark.parametrize("source", ("era5", "gfs"))
def test_cli_named_source_static_cache_pair_is_atomic(source, capsys):
    result = main([
        "--source", source,
        "--static-input", "/case/static.npz",
        "--dry-run",
    ])
    assert result == EXIT_USAGE
    assert "must be supplied together" in capsys.readouterr().err


@pytest.mark.parametrize("source", ("era5", "gfs"))
def test_cli_named_source_refuses_parallel_cuda_hierarchy(source, capsys):
    common = [
        "--source", source,
        "--preprocess-backend", "cuda",
        "--bridge", "/bin/source_bridge",
        "--wps-namelist", "/case/namelist.wps",
        "--experiment-config", "/case/experiment.toml",
        "--source-sha256s", "/source/input-manifest.json",
        "--source-sha256s-sha256", "abc123",
        "--output-root", "/output",
        "--geog-root", "/wps-geog",
        "--hierarchy-workers", "2",
        "--dry-run",
    ]
    if source == "era5":
        arguments = [*common,
            "--grib", "/source/era5.grb",
            "--vtable", "/source/Vtable.ERA5_CDO",
        ]
    else:
        arguments = [*common,
            "--gfs-series", "/source/gfs-series.tsv",
            "--cycle", "2026-07-20_00:00:00",
        ]
    result = main(arguments)
    assert result == EXIT_USAGE
    assert "explicit --preprocess-backend cpu" in capsys.readouterr().err


def test_cli_hrrr_routes_explicit_cpu_backend_options(capsys):
    result = main(
        [
            *_hrrr_args(),
            '--preprocess-backend', 'cpu',
            '--preprocess-workers', '8',
            '--cpu-preprocess-bridge', '/bundle/libgpuwm_preprocess_cpu.so',
            '--dry-run',
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--preprocess-backend cpu" in command
    assert "--preprocess-workers 8" in command
    assert (
        "--cpu-preprocess-bridge /bundle/libgpuwm_preprocess_cpu.so"
        in command
    )


def test_cli_hrrr_routes_auto_backend_workers(capsys):
    result = main(
        [
            *_hrrr_args(),
            '--preprocess-backend', 'auto',
            '--preprocess-workers', '4',
            '--dry-run',
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--preprocess-backend auto" in command
    assert "--preprocess-workers 4" in command


def test_cli_hrrr_forwards_cuda_host_workers(capsys):
    """--preprocess-workers with the CUDA backend reaches the HRRR tool.

    It sets the threads of the host steps that backend runs in the Rust
    library, and the tool no longer refuses it after this door has
    forwarded it.
    """
    result = main(
        [
            *_hrrr_args(),
            '--preprocess-backend', 'cuda',
            '--preprocess-workers', '8',
            '--dry-run',
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "prepare_hrrr_wrf.py" in command
    assert "--preprocess-backend cuda" in command
    assert "--preprocess-workers 8" in command
    assert "--cpu-preprocess-bridge" not in command


def test_cli_hrrr_rejects_cpu_bridge_with_auto(capsys):
    result = main(
        [
            *_hrrr_args(),
            '--preprocess-backend', 'auto',
            '--cpu-preprocess-bridge', '/bundle/libgpuwm_preprocess_cpu.so',
            '--dry-run',
        ]
    )
    assert result == EXIT_USAGE
    assert "requires --preprocess-backend cpu" in capsys.readouterr().err


def test_cli_gfs_uses_distribution_bridge_environment(monkeypatch, capsys):
    monkeypatch.setenv("WOOF_GFS_GRIB2_BRIDGE", "/bundle/gfs_grib2_bridge")
    result = main(
        [
            '--source', 'gfs',
            '--preprocess-backend', 'cpu',
            '--gfs-series', '/source/gfs-series.tsv',
            '--cycle', '2026-07-20_00:00:00',
            *_regular_args(),
            '--dry-run',
        ]
    )
    assert result == 0
    assert "/bundle/gfs_grib2_bridge" in capsys.readouterr().out.replace("\\", "/")


def test_cli_installed_distribution_rejects_explicit_decoder_substitution(
    tmp_path, monkeypatch, capsys
):
    _installed_grib2_distribution(tmp_path, monkeypatch, clear_engine=False)

    assert main(_mapped_args() + ["--dry-run"]) == EXIT_CONFIG
    error = capsys.readouterr().err
    assert "--grib2-inventory differs from the decoder bound" in error


def _installed_grib2_distribution(tmp_path, monkeypatch, suffix="", *, clear_engine=True):
    """A sealed installed runtime whose manifest binds both GRIB2 tools.

    This is the estate a wheel user has: a runtime manifest naming the
    two staged decoders by hash, and the two environment variables the
    installer exports.  It is the setup both decoder contracts below
    are argued on -- what changes between them is only the ENGINE.
    """

    bridge_root = tmp_path / "libexec" / "bridges"
    bridge_root.mkdir(parents=True)
    inventory = bridge_root / f"grib2_inventory{suffix}"
    dump = bridge_root / f"grib2_dump{suffix}"
    for path, payload in ((inventory, b"inventory"), (dump, b"dump")):
        path.write_bytes(payload)
        path.chmod(0o755)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            complete_runtime_manifest(
                {
                    f"libexec/bridges/{path.name}": {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "executable": True,
                    }
                    for path in (inventory, dump)
                }
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WOOF_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    monkeypatch.setenv("WOOF_GRIB2_INVENTORY", str(inventory))
    monkeypatch.setenv("WOOF_GRIB2_DUMP", str(dump))
    if clear_engine:
        monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    return inventory, dump


def _mapped_args_without_tool_flags():
    args = _mapped_args()
    for flag in ("--grib2-inventory", "--grib2-dump"):
        index = args.index(flag)
        del args[index : index + 2]
    return args


def test_cli_installed_distribution_binds_the_engine_not_subprocess_decoders(
    tmp_path, monkeypatch, capsys
):
    """The default mapped GRIB2 route decodes IN PROCESS on the Rust engine.

    The route this door composes runs no subprocess decoder at all, so
    the installed runtime's two staged tools are neither resolved nor
    forwarded.  Forwarding them would not merely be redundant: an
    explicit tool pin is the spelling that routes the decode BACK to
    the Python engine, so a door that helpfully passed the manifest's
    paths along would silently un-default the engine and report a Rust
    run that was a Python one.  The engine choice is asserted through
    the resolver the door actually consults, so this cannot pass by the
    door having forgotten about engines entirely.
    """

    inventory, dump = _installed_grib2_distribution(tmp_path, monkeypatch)
    answers = []
    real_resolver = woof.source_cli._resolve_mapped_engine

    def _recorded(explicit=None):
        answer = real_resolver(explicit)
        answers.append(answer)
        return answer

    monkeypatch.setattr(woof.source_cli, "_resolve_mapped_engine", _recorded)

    assert main(_mapped_args_without_tool_flags() + ["--dry-run"]) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert answers == [MAPPED_ENGINE_RUST], answers
    assert "-m woof.mapped_direct" in command
    assert "--grib2-inventory" not in command
    assert "--grib2-dump" not in command
    assert str(inventory).replace("\\", "/") not in command
    assert str(dump).replace("\\", "/") not in command


def test_cli_installed_distribution_uses_manifest_bound_decoders_on_python(
    tmp_path, monkeypatch, capsys
):
    """The Python engine still binds its decoders out of the manifest.

    ``--mapped-engine python`` is the documented workaround, and on it
    the decode IS a subprocess: the two tools then come from the sealed
    installed runtime rather than from whatever happens to be on PATH.
    Losing this would let an installed distribution launch an unbound
    decoder, which is the substitution the manifest exists to refuse.
    """

    inventory, dump = _installed_grib2_distribution(tmp_path, monkeypatch)
    args = _mapped_args_without_tool_flags()

    assert main(args + ["--mapped-engine", "python", "--dry-run"]) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert str(inventory).replace("\\", "/") in command
    assert str(dump).replace("\\", "/") in command


def test_cli_installed_windows_distribution_uses_exe_decoders_on_python(
    tmp_path, monkeypatch, capsys
):
    """Same contract, ``.exe`` platform spelling, environment workaround.

    The engine is selected here through ``GPUWM_MAPPED_ENGINE``, the
    other half of the documented workaround, so both spellings that
    reach the Python engine stay pinned against the sealed runtime.
    """

    inventory, dump = _installed_grib2_distribution(
        tmp_path, monkeypatch, suffix=".exe")
    monkeypatch.setenv("GPUWM_MAPPED_ENGINE", "python")
    args = _mapped_args_without_tool_flags()

    assert main(args + ["--dry-run"]) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert str(inventory).replace("\\", "/") in command
    assert str(dump).replace("\\", "/") in command


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--run-seconds", "3600"),
        ("--history-interval-seconds", "3600"),
        ("--pipeline-workers", "8"),
        ("--source-orography-variable", "SOILHGT"),
    ],
)
def test_cli_gfs_rejects_common_options_that_it_does_not_apply(flag, value, capsys):
    result = main(
        [
            '--source', 'gfs',
            *_gfs_inputs(),
            *_regular_args(),
            flag,
            value,
            '--dry-run',
        ]
    )
    assert result == EXIT_USAGE
    assert f"{flag} is not used by --source gfs" in capsys.readouterr().err


def test_cli_gfs_rejects_non_synoptic_cycle_and_era5_inputs(capsys):
    result = main(
        [
            "--source",
            "gfs",
            "--cycle",
            "2026-07-20_03:00:00",
            "--grib",
            "/source/era5.grb",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    error = capsys.readouterr().err
    assert "exact 00/06/12/18 UTC GFS cycle" in error
    assert "--grib is not used by --source gfs" in error


@pytest.mark.parametrize("source_format", ["grib2", "netcdf"])
@pytest.mark.parametrize("flags, mode", [
    (["--stock-wrf-export", "required"], "required"),
    (["--stock-wrf-export", "optional"], "optional"),
    (["--stock-wrf-export", "off"], "off"),
    (["--no-stock-wrf-export"], "off"),
])
def test_mapped_cli_forwards_explicit_export_intent(capsys, source_format, flags, mode):
    assert main(_mapped_args(source_format) + flags + ["--dry-run"]) == 0
    command = capsys.readouterr().out
    assert f"--stock-wrf-export {mode}" in command


def test_mapped_cli_refuses_conflicting_export_intent(capsys):
    with pytest.raises(SystemExit):
        main(_mapped_args() + ["--stock-wrf-export", "required",
                               "--no-stock-wrf-export", "--dry-run"])
    assert "not allowed with argument" in capsys.readouterr().err
