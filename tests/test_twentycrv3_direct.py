from __future__ import annotations

from datetime import datetime
import hashlib
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from woof.mapped_composition import load_composition
from woof.mapped_source import _DecodedCollection, _DirectValue, load_mapping
from woof.source_authorities import (
    twentycrv3_authorities,
    twentycrv3_authority_sha256,
)
from woof.twentycrv3_direct import (
    _bind_implicit_member,
    _expected_inventory,
    _manifest,
    discover_20crv3_grib2,
    write_20crv3_manifest,
)


ROOT = Path(__file__).parents[1]
MAPPING = (
    ROOT / "woof" / "authorities" /
    "rw-wps-20crv3-member-grib2.mapping.json"
)
COMPOSITION = (
    ROOT / "woof" / "authorities" /
    "rw-wps-20crv3-member-grib2.composition.json"
)


def test_exact_authorities_are_resolved_from_the_installable_package() -> None:
    paths = twentycrv3_authorities()
    digests = twentycrv3_authority_sha256()

    assert set(paths) == {"mapping", "composition", "provenance"}
    assert all(path.parent.name == "authorities" for path in paths.values())
    assert {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in paths.items()
    } == dict(digests)


def _write_series(
    root: Path,
    *,
    rows: tuple[tuple[str, str, str], ...] = (
        ("072", "1932032100", "pl"),
        ("072", "1932032100", "sfc"),
        ("072", "1932032103", "pl"),
        ("072", "1932032103", "sfc"),
    ),
) -> dict[tuple[str, str, str], Path]:
    paths = {}
    for member, stamp, role in rows:
        directory = root / ("PRESSURE" if role == "pl" else "SURFACE")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"mem{member}_{stamp}_{role}.grb2"
        path.write_bytes(f"fixture:{member}:{stamp}:{role}".encode())
        paths[(member, stamp, role)] = path
    return paths


def test_mapping_loads_with_exact_private_sample_contract() -> None:
    mapping = load_mapping(MAPPING)

    assert mapping["name"] == "noaa-20crv3-every-member-grib2-native"
    assert mapping["format"] == "grib2"
    assert mapping["coordinates"]["vertical"]["levels"] == [
        5000, 7000, 10000, 15000, 20000, 25000, 30000, 35000,
        40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000,
        80000, 85000, 90000, 92500, 95000, 97500, 100000,
    ]
    assert mapping["target"]["boundary_interval_seconds"] == 10800
    soil = mapping["fields"]["volumetric_soil_moisture"]["selectors"]
    assert len(soil) == 4
    assert all((row["discipline"], row["category"], row["parameter"])
               == (2, 0, 192) for row in soil)


def test_composition_binds_exact_noah_layers_and_in_band_terrain() -> None:
    contract = load_composition(COMPOSITION, MAPPING)

    assert contract["soil_layers"]["remap"]["kind"] \
        == "conservative_layer_means"
    assert [
        (row["top"], row["bottom"])
        for row in contract["soil_layers"]["source_layers"]
    ] == [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)]
    terrain = contract["supplements"]["terrain_height"]
    assert terrain["data_role"] == "twentycrv3_in_band_surface"
    assert terrain["require_invariant_across_time"] is True


def test_exact_archive_inventory_has_115_pressure_and_19_surface_fields() -> None:
    levels = load_mapping(MAPPING)["coordinates"]["vertical"]["levels"]

    assert sum(_expected_inventory("pl", levels).values()) == 115
    assert sum(_expected_inventory("sfc", levels).values()) == 19


def test_discovery_binds_member_pairs_cadence_and_hashes(tmp_path: Path) -> None:
    paths = _write_series(tmp_path)

    result = discover_20crv3_grib2(tmp_path)

    assert result["member"] == "072"
    assert result["file_count"] == 4
    assert result["cadence_seconds"] == 10800
    assert result["valid_times"] == [
        "1932-03-21T00:00:00", "1932-03-21T03:00:00",
    ]
    by_name = {row["filename"]: row for row in result["files"]}
    for path in paths.values():
        assert by_name[path.name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_discovery_rejects_mixed_filename_members(tmp_path: Path) -> None:
    _write_series(tmp_path, rows=(
        ("072", "1932032100", "pl"),
        ("072", "1932032100", "sfc"),
        ("073", "1932032103", "pl"),
        ("073", "1932032103", "sfc"),
    ))

    with pytest.raises(ValueError, match="mixes member IDs"):
        discover_20crv3_grib2(tmp_path)


def test_discovery_rejects_incomplete_time_pair(tmp_path: Path) -> None:
    _write_series(tmp_path, rows=(
        ("072", "1932032100", "pl"),
        ("072", "1932032100", "sfc"),
        ("072", "1932032103", "pl"),
    ))

    with pytest.raises(ValueError, match="pairing gaps"):
        discover_20crv3_grib2(tmp_path)


def test_manifest_rejects_input_changed_after_hash_binding(tmp_path: Path) -> None:
    paths = _write_series(tmp_path / "source")
    manifest = tmp_path / "manifest.json"
    write_20crv3_manifest(tmp_path / "source", manifest)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    paths[("072", "1932032103", "sfc")].write_bytes(b"changed")

    with pytest.raises(ValueError, match="input identity differs"):
        _manifest(manifest, manifest_sha)


def _collection(member: str | None) -> _DecodedCollection:
    valid = datetime(1932, 3, 21)
    value = _DirectValue(
        name="surface_pressure",
        valid_time=valid,
        member=member,
        source_cycle=valid,
        axes=("y", "x"),
        values=np.full((2, 2), 100000.0),
        missing_count=0,
        references=("fixture",),
    )
    return _DecodedCollection(
        latitude=np.asarray([10.0, 10.5]),
        longitude=np.asarray([20.0, 20.5]),
        vertical_values=np.asarray([100000.0]),
        direct=MappingProxyType({
            (valid, member, "surface_pressure"): value,
        }),
        source_cycles=MappingProxyType({(valid, member): valid}),
        grid_fingerprint="fixture-grid",
    )


def test_filename_member_retags_grib_records_without_pdt_member() -> None:
    bound = _bind_implicit_member(_collection(None), "072")
    key = (datetime(1932, 3, 21), "072", "surface_pressure")

    assert set(bound.direct) == {key}
    assert bound.direct[key].member == "072"
    assert set(bound.source_cycles) == {(datetime(1932, 3, 21), "072")}


def test_filename_member_rejects_unexpected_pdt_member() -> None:
    with pytest.raises(ValueError, match="unexpectedly encodes"):
        _bind_implicit_member(_collection("9"), "072")


# ---------------------------------------------------------------------------
# Front-door guidance (test host 8 quartet): a refusal is a sentence, and the
# route that ran end to end now says how to run the forecast.
# ---------------------------------------------------------------------------

def test_the_wizard_default_toml_is_refused_in_a_sentence(tmp_path, capsys):
    """A gate that is right, worded as an instruction rather than a crash.

    `woof domain --source era5` (the wizard default) emits a
    `[case_data]` table.  The native 20CRv3 front door does not read it,
    and `experiment.build_experiment` says so -- but a test host 8 pilot met
    that as a raw traceback with no route to a config that works.
    """
    from woof import twentycrv3_wrf

    config = tmp_path / "wizard-default.toml"
    config.write_text(
        "[experiment]\nname = \"x\"\n\n[case_data]\nroot = \"nope\"\n",
        encoding="utf-8")
    argv = []
    for flag in ("--mapping", "--composition", "--provenance", "--manifest",
                 "--grib2-inventory", "--grib2-dump", "--wps-namelist",
                 "--geog-root", "--output-root"):
        argv += [flag, str(tmp_path / "unused")]
    argv += ["--manifest-sha256", "0" * 64,
             "--experiment-config", str(config)]

    rc = twentycrv3_wrf.main(argv)
    err = capsys.readouterr().err
    assert rc == 2, "a config refusal fails closed"
    assert "Traceback" not in err
    # ONE line, and it is this door's own sentence.
    #
    # This assertion was briefly relaxed to "one warning line plus one
    # refusal line": when the experiment loader was changed to WARN about
    # an unknown table rather than refuse on it, `main`'s
    # `except ValueError` stopped catching the [case_data] case, so the
    # warning reached stderr and the refusal that followed was about
    # something else entirely (the missing [[domain]] table), through
    # `_refuse_incompatible_config`'s generic `else` branch.
    #
    # Unknown tables refuse again, so `_refuse_incompatible_config`'s
    # `if "case_data" in detail` branch fires as it was written to --
    # naming the incompatibility and the route that works, with no
    # preamble.  The pilot's complaint was a stack; this is one sentence.
    lines = [line for line in err.splitlines() if line.strip()]
    assert not [line for line in lines if line.startswith("warning:")], (
        f"the door's refusal is the whole message: {err!r}")
    assert len(lines) == 1, f"one sentence, not a stack: {err!r}"
    err = lines[0]
    # It names the incompatibility...
    assert "[case_data]" in err
    # ...and a supported route to a config this door accepts.
    assert "woof domain --source gfs" in err
    assert "--experiment-config" in err


def test_the_20crv3_front_door_prints_the_run_command_too(tmp_path):
    """Parity with GFS: a complete hash-bound command, not silence.

    The 1932 hindcast ran END TO END on shipped 1.0.1 and then printed
    no run instruction at all, while the GFS door printed a complete
    hash-bound command.  Same printer, same contract, `--source 20crv3`.
    """
    import hashlib as _hashlib
    import json

    from woof.gfs_direct import prepared_forecast_next_command

    root = tmp_path / "prepared"
    root.mkdir()
    from woof.physics_compat import WSM6_PROFILE_ID

    proof = {"schema": "gpuwm-mapped-direct-wrf-proof-v1",
             "input_manifest_sha256": "a" * 64,
             "physics": {"profile": WSM6_PROFILE_ID},
             "prepared_cache": {"content_sha256": "b" * 64}}
    (root / "proof.json").write_text(json.dumps(proof), encoding="utf-8")
    proof_digest = _hashlib.sha256(
        (root / "proof.json").read_bytes()).hexdigest()
    config = tmp_path / "c.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    namelist = tmp_path / "c.namelist.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")

    lines = prepared_forecast_next_command(
        proof, output_root=root, experiment_config=config,
        wps_namelist=namelist, source="20crv3")
    text = "\n".join(lines)
    assert "--source 20crv3" in text
    assert f"--physics-profile {WSM6_PROFILE_ID}" in text
    assert f"--proof-sha256 {proof_digest}" in text
    assert f"--source-manifest-sha256 {'a' * 64}" in text
    assert f"--prepared-content-sha256 {'b' * 64}" in text
    # No placeholders, and no holed command: every value resolved.
    assert "<" not in text and ">" not in text
    # The outdir is a SIBLING of the preparation, which is the only kind
    # the runner's protected-input guard accepts.  POSIX display form:
    # the printer renders forward slashes (and quotes on a space), the
    # same contract as `rw-wps --dry-run`.
    expected_outdir = str(
        root.parent / (root.name + "-forecast")).replace("\\", "/")
    assert f"--outdir {expected_outdir}" in text
    # And 20crv3 really is a source that runner takes.  MEMBERSHIP, not an
    # exact inventory: this assertion is about THIS door, and the source
    # list is open by design -- adding a model is table work, so every
    # packaged profile that lands widens it.  Pinned as an exact frozenset
    # it went stale four sources running (aigefs, aigfs, gefs, rrfs each
    # added their own membership assertion beside their profile, and none
    # could know to edit a list living in 20CRv3's file), which cost a red
    # suite that said nothing true about 20CRv3.
    from woof import prepared_single_domain_forecast as runner
    assert "20crv3" in runner.SUPPORTED_SOURCES
    assert "20crv3-cf" in runner.SUPPORTED_SOURCES


# ---------------------------------------------------------------------------
# The generic composed route carries this route's two gates
# ---------------------------------------------------------------------------
#
# These rows are what emptied `mapped_engine_bridge.ENGINE_GAPS`: the
# member door (`woof prep --source 20crv3`) now runs through
# `decode_composed_source` like every other composed source, so the two
# gates that used to hold it back are asserted HERE, on the real private
# member bytes, against both engines:
#
#   * ENSEMBLE IDENTITY.  The 20CRv3 PDT carries no member
#     (`member_identity` in the sealed manifest is
#     `filename_memNNN_not_grib2_pdt`).  The verified filename member is
#     therefore an EXPLICIT binding in the composition input manifest
#     (`member` + `member_identity`), and the composition -- Rust engine
#     and Python engine alike -- stamps it onto every canonical frame and
#     into the sealed alignment receipt.  A run that lost it would return
#     frames with `member=None` for an ENSEMBLE source, which is what the
#     pre-port measurement showed and what the row below refuses.
#
#   * PRODUCT IDENTITY.  `_verify_archive_inventory` binds the exact
#     every-member GRIB2 product identity (center/subcenter/table
#     versions, generating process 4/195, GDT 0, scan mode 0x40, shape 6,
#     0.5-degree, DRT 0, contiguous field indices).  Its measurement
#     instrument on the bare default is the engine's raw record-inventory
#     surface (`gpuwm_mapped_engine inventory`); on the documented
#     Python-engine workaround it is still `grib2_inventory`.  The gate's
#     own wording is identical either way, which the rows below assert
#     instrument against instrument.

_MEMBER_SAMPLE = Path(os.environ.get(
    "GPUWM_20CRV3_MEMBER_SAMPLE",
    str(Path.home() / "Downloads" / "1932-03-21 20CR Member 72 Files"),
))
_MEMBER_FILES = (
    _MEMBER_SAMPLE / "PRESSURE" / "mem072_1932032100_pl.grb2",
    _MEMBER_SAMPLE / "SURFACE" / "mem072_1932032100_sfc.grb2",
    _MEMBER_SAMPLE / "PRESSURE" / "mem072_1932032103_pl.grb2",
    _MEMBER_SAMPLE / "SURFACE" / "mem072_1932032103_sfc.grb2",
)
_SURFACE_FILES = tuple(
    path for path in _MEMBER_FILES if path.name.endswith("_sfc.grb2"))
_TERRAIN_ROLE = "twentycrv3_in_band_surface"
_MEMBER_IDENTITY = "filename_memNNN_not_grib2_pdt"


def _compose_generically(work, *, engine):
    """Run the GENERIC composed-source route over the 20CRv3 authorities.

    ``engine`` is ``"rust"`` for the bare default or ``"python"`` for the
    workaround; the manifest is authored under the same request, because
    it seals which binary read the bytes.  The verified filename member
    rides the manifest as the EXPLICIT binding the composition stamps.
    """

    from woof import bridges
    from woof.mapped_authoring import author_input_manifest
    from woof.mapped_composition import (decode_composed_source,
                                          mapped_composition_receipt)

    tools = {}
    if engine == "python":
        tools = {
            "grib2_inventory": bridges.find_bridge("grib2_inventory"),
            "grib2_dump": bridges.find_bridge("grib2_dump"),
        }
        if any(path is None for path in tools.values()):
            pytest.skip("the subprocess grib2 tools are not staged here")
    authorities = twentycrv3_authorities()
    supplements = {_TERRAIN_ROLE: _SURFACE_FILES}
    provenance = {
        f"{_TERRAIN_ROLE}_provenance": authorities["provenance"],
    }
    work.mkdir(parents=True, exist_ok=True)
    manifest = work / "input-manifest.json"
    author_input_manifest(
        manifest,
        mapping_path=authorities["mapping"],
        composition_path=authorities["composition"],
        primary_files=_MEMBER_FILES,
        supplement_files=supplements,
        provenance_files=provenance,
        member="072",
        member_identity=_MEMBER_IDENTITY,
        **tools,
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    bundle = decode_composed_source(
        authorities["composition"], authorities["mapping"], _MEMBER_FILES,
        supplements, provenance,
        input_manifest=manifest, input_manifest_sha256=digest,
        **tools,
    )
    return bundle, mapped_composition_receipt(bundle)


@pytest.mark.skipif(
    not all(path.is_file() for path in _MEMBER_FILES),
    reason="the private 20CRv3 member sample is not staged on this box",
)
def test_the_member_survives_the_generic_compose_route(
        tmp_path, monkeypatch):
    """Gate (a) of the port, held on the real member bytes, both engines.

    The pre-port measurement (this row's ancestor) proved the generic
    route read these bytes identically on both engines AND returned
    `member=None` everywhere -- the ensemble identity died at the seam.
    This is the same dual run asserting the identity now survives: the
    explicit manifest binding reaches every canonical frame and the
    sealed alignment receipt, byte-equal between the engines.
    """

    from woof.mapped_engine_bridge import ENGINE_ENV, ENGINE_NAME

    monkeypatch.delenv(ENGINE_ENV, raising=False)
    rust_bundle, rust_receipt = _compose_generically(
        tmp_path / "rust", engine="rust")
    # The bare default really did run the engine, so the comparison below
    # is between two engines and not between one engine and itself.
    assert set(rust_bundle.decoder_paths) == {ENGINE_NAME}

    monkeypatch.setenv(ENGINE_ENV, "python")
    python_bundle, python_receipt = _compose_generically(
        tmp_path / "python", engine="python")
    assert set(python_bundle.decoder_paths) == {
        "grib2_inventory", "grib2_dump"}

    # 1. Decode/compose parity: same frames, same terrain, same times.
    assert rust_receipt["frames"] == python_receipt["frames"]
    assert rust_receipt["valid_times"] == python_receipt["valid_times"]
    assert rust_receipt["alignment"] == python_receipt["alignment"]
    assert len(rust_bundle.frames) == 2

    # 2. Gate (a): the ensemble identity SURVIVES, on both engines.  A
    #    frame with `member=None` here is the pre-port defect: the member
    #    dropped out of the frame headers and out of the preparation's
    #    sealed alignment receipt, for an ENSEMBLE source.
    for bundle in (rust_bundle, python_bundle):
        assert {frame.member for frame in bundle.frames} == {"072"}
    assert rust_receipt["alignment"]["member"] == "072"
    assert rust_receipt["alignment"]["member_identity"] == _MEMBER_IDENTITY


@pytest.mark.skipif(
    not all(path.is_file() for path in _MEMBER_FILES),
    reason="the private 20CRv3 member sample is not staged on this box",
)
def test_the_engine_inventory_answers_the_product_identity_gate(tmp_path):
    """Gate (b): `_verify_archive_inventory`, fed by the ENGINE's surface.

    The product-identity gate keeps its exact per-file GRIB2 contract
    (authority 7/2 tables 2/1, process 4/195, GDT 0, scan 0x40, shape 6,
    0.5 degree, DRT 0, contiguous indices) and its own refusal wording;
    what moves is the measurement instrument -- the engine's raw
    record-inventory subcommand instead of the `grib2_inventory`
    subprocess.  Same gate, same sentences, new instrument.
    """

    from woof.mapped_engine_bridge import engine_record_inventory
    from woof.twentycrv3_direct import _verify_archive_inventory

    document = discover_20crv3_grib2(_MEMBER_SAMPLE)
    sources = tuple(
        path.resolve() for path in _MEMBER_FILES)
    levels = load_mapping(MAPPING)["coordinates"]["vertical"]["levels"]
    inventories = engine_record_inventory(sources)

    result = _verify_archive_inventory(
        document, sources, lambda source: inventories[source], levels,
    )
    assert result["status"] == "PASS_EXACT_20CRV3_EVERY_MEMBER_GRIB2_CONTRACT"
    assert result["authority"]["generating_process"] == 4
    assert result["grid"]["gdt"] == "0"
    assert result["grid"]["scan_mode"] == "0x40"
    assert {row["filename"] for row in result["files"]} \
        == {path.name for path in sources}

    # The refusal keeps its specificity on the new instrument: a contract
    # the bytes do not meet is refused by NAMING the divergent inventory,
    # exactly as the subprocess instrument worded it.
    with pytest.raises(ValueError, match="field inventory differs"):
        _verify_archive_inventory(
            document, sources, lambda source: inventories[source],
            levels[:-1],
        )


@pytest.mark.skipif(
    not all(path.is_file() for path in _MEMBER_FILES),
    reason="the private 20CRv3 member sample is not staged on this box",
)
def test_the_two_inventory_instruments_agree_on_the_real_bytes():
    """Validate the instrument: engine rows versus subprocess rows.

    The gate is only as strong as what feeds it, so the two instruments
    are compared column for column on a real member file -- the record
    keys the gate counts, and every authority/grid/packing octet it
    pins.  A silent disagreement here would let one engine's route pass
    a contract the other refuses.
    """

    from woof import bridges
    from woof.mapped_engine_bridge import engine_record_inventory
    from woof.mapped_source import _grib2_inventory
    from woof.twentycrv3_direct import _record_key

    tool = bridges.find_bridge("grib2_inventory")
    if tool is None:
        pytest.skip("the subprocess grib2_inventory is not staged here")
    source = _SURFACE_FILES[0].resolve()
    engine_rows = engine_record_inventory((source,))[source]
    tool_rows = _grib2_inventory(source, tool)

    assert len(engine_rows) == len(tool_rows)
    compared = (
        "index", "discipline", "category", "parameter", "center",
        "subcenter", "master_table_version", "local_table_version",
        "reference_time", "forecast_unit", "forecast_time", "pdt",
        "level_type", "level_value", "second_level_type",
        "second_level_value", "member", "generating_process",
        "forecast_generating_process_id", "gdt", "nx", "ny", "lat1",
        "lon1", "dx", "dy", "latin1", "latin2", "lov", "scan_mode",
        "shape_of_earth", "resolution_flags", "drt", "bitmap",
    )
    for engine_row, tool_row in zip(engine_rows, tool_rows):
        assert _record_key(engine_row) == _record_key(tool_row)
        for column in compared:
            assert engine_row[column] == tool_row[column], column
