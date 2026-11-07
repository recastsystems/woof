"""Acceptance gates for the arbitrary-but-verified GRIB front door."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from woof.source_authorities import packaged_gfs_vtable

from woof import adapt
from woof.adapt import author_adapter, build_composition, verify_grib2_inputs
from woof.cli import main
from woof.mapped_authoring import compile_mapping_descriptor


ROOT = Path(__file__).resolve().parents[1]
VTABLE = packaged_gfs_vtable()
DESCRIPTOR = ROOT / "configs" / "rw-wps-gfs-pressure-grib2.descriptor.json"
ACTUAL_GRIB = (
    ROOT
    / "tests"
    / "fixtures"
    / "gfs-scan-order"
    / "nomads-crop-20260729t18z-f000.grib2"
)


def _mapping():
    mapping, _receipt = compile_mapping_descriptor(
        DESCRIPTOR,
        vtable_path=VTABLE,
    )
    return mapping


def _row(index: int, selector: dict, *, level_value=None) -> dict[str, str]:
    values = {
        "index": str(index),
        "discipline": str(selector["discipline"]),
        "category": str(selector["category"]),
        "parameter": str(selector["parameter"]),
        "center": str(selector.get("center", 7)),
        "subcenter": str(selector.get("subcenter", 0)),
        "master_table_version": str(
            selector.get("master_table_version", 2)
        ),
        "local_table_version": str(selector.get("local_table_version", 1)),
        "reference_time": "2026-07-29T18:00:00",
        "forecast_unit": "1",
        "forecast_time": "0",
        "pdt": "0",
        "level_type": str(selector.get("level_type", 255)),
        "level_value": str(
            selector.get("level_value", 0)
            if level_value is None
            else level_value
        ),
        "second_level_type": str(selector.get("second_level_type", 255)),
        "second_level_value": str(selector.get("second_level_value", 0)),
        "member": str(selector.get("member", "-")),
        "generating_process": "2",
        "forecast_generating_process_id": "81",
        "gdt": "0",
        "nx": "41",
        "ny": "41",
        "lat1": "30",
        "lon1": "260",
        "dx": "0.25",
        "dy": "0.25",
        "latin1": "0",
        "latin2": "0",
        "lov": "0",
        "scan_mode": "0x40",
        "shape_of_earth": "6",
        "resolution_flags": "0x30",
        "drt": "0",
        "bitmap": "false",
    }
    return values


def _complete_gfs_inventory(mapping=None):
    mapping = mapping or _mapping()
    rows = []
    index = 0
    levels = mapping["coordinates"]["vertical"]["levels"]
    for field in mapping["fields"].values():
        if field.get("derivation") is not None:
            continue
        selectors = field["selectors"]
        if "vertical" in field["source_axes"]:
            for level in levels:
                rows.append(_row(index, selectors[0], level_value=level))
                index += 1
        else:
            for selector in selectors:
                rows.append(_row(index, selector))
                index += 1
    return rows


def _fake_bridge_identity(path, role):
    path = Path(path)
    data = path.read_bytes()
    return {
        "role": role,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_gfs_vtable_fixture_authors_runnable_uncertified_bundle(
    tmp_path,
    monkeypatch,
):
    """Full public authoring path: compile, battery, triple, and manifest."""

    rows = _complete_gfs_inventory()
    monkeypatch.setattr(adapt, "_grib2_inventory", lambda *_args: rows)
    monkeypatch.setattr(adapt, "_decode_grib", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "woof.mapped_authoring.bridge_identity",
        _fake_bridge_identity,
    )
    inventory = tmp_path / "grib2_inventory"
    dump = tmp_path / "grib2_dump"
    inventory.write_bytes(b"fixture inventory decoder")
    dump.write_bytes(b"fixture dump decoder")

    result = author_adapter(
        vtable_path=VTABLE,
        descriptor_path=DESCRIPTOR,
        input_files=(ACTUAL_GRIB,),
        output_dir=tmp_path / "adapter",
        grib2_inventory=inventory,
        grib2_dump=dump,
    )

    assert result["status"] == \
        "runnable_mapping_not_stock_wrf_certified"
    assert result["runnable"] is True
    assert result["stock_wrf_certified"] is False
    assert result["battery"]["status"] == "PASS"
    assert result["battery"]["record_inventory"]["selected_record_count"] \
        == len(rows)
    assert result["runtime_bindings"]["source"] == "mapped"
    assert result["runtime_bindings"]["supplements"] == {
        "adapt_in_band_terrain": [str(ACTUAL_GRIB.resolve())],
    }
    assert result["runtime_bindings"]["provenance"] == {
        "adapt_authority_provenance": str(
            (tmp_path / "adapter" / "adapter.provenance.json").resolve()
        ),
    }
    outputs = {
        path.name for path in (tmp_path / "adapter").iterdir()
    }
    assert outputs == {
        "adapter.mapping.json",
        "adapter.composition.json",
        "adapter.provenance.json",
        "adapter.inputs.json",
    }
    provenance = json.loads(
        (tmp_path / "adapter" / "adapter.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["stock_wrf_certified"] is False
    assert provenance["mapping"]["sha256"] == result["mapping"]["sha256"]
    assert provenance["composition"]["sha256"] \
        == result["composition"]["sha256"]
    assert provenance["inputs"][0]["sha256"] == hashlib.sha256(
        ACTUAL_GRIB.read_bytes()
    ).hexdigest()


def test_adapt_refuses_one_exact_missing_field(tmp_path, monkeypatch):
    mapping = _mapping()
    rows = _complete_gfs_inventory(mapping)
    selector = mapping["fields"]["surface_pressure"]["selectors"][0]
    rows = [
        row for row in rows if not adapt._selector_matches(selector, row)
    ]
    monkeypatch.setattr(adapt, "_grib2_inventory", lambda *_args: rows)
    decoder = tmp_path / "inventory"
    decoder.write_bytes(b"fixture")

    with pytest.raises(
        ValueError,
        match=(
            r"record-inventory check failed.*field 'surface_pressure' "
            r"missing selector"
        ),
    ):
        verify_grib2_inputs(
            mapping,
            (ACTUAL_GRIB,),
            inventory_executable=decoder,
        )


def test_adapt_refuses_wrong_grid_template_with_named_adapter_remedy(
    tmp_path,
    monkeypatch,
):
    mapping = _mapping()
    rows = _complete_gfs_inventory(mapping)
    rows[0]["gdt"] = "30"
    monkeypatch.setattr(adapt, "_grib2_inventory", lambda *_args: rows)
    decoder = tmp_path / "inventory"
    decoder.write_bytes(b"fixture")

    with pytest.raises(
        ValueError,
        match=(
            r"grid-family check failed:.*uses GDT 30; generic adapt "
            r"supports regular latitude/longitude GDT 0 only.*"
            r"named-adapter path"
        ),
    ):
        verify_grib2_inputs(
            mapping,
            (ACTUAL_GRIB,),
            inventory_executable=decoder,
        )


def test_adapt_refuses_declared_soil_gap():
    mapping = _mapping()
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        mapping["fields"][name]["selectors"][1]["level_value"] = 0.2

    with pytest.raises(
        ValueError,
        match=(
            r"soil-layer check failed: gap between source layers 0 and 1: "
            r"0\.1 m to 0\.2 m"
        ),
    ):
        build_composition(
            mapping,
            {"soil_policy": {"kind": "identity_complete_layers"}},
        )


def test_adapt_refuses_source_top_that_does_not_cover_model_top(
    tmp_path,
):
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    descriptor["adapt"]["model_top_pa"] = 5000
    path = tmp_path / "top-failure.descriptor.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            r"vertical-coverage check failed: source top 10000 Pa does not "
            r"cover model top 5000 Pa"
        ),
    ):
        author_adapter(
            vtable_path=VTABLE,
            descriptor_path=path,
            input_files=(ACTUAL_GRIB,),
            output_dir=tmp_path / "adapter",
        )
    assert not (tmp_path / "adapter").exists()


def test_adapt_decoder_failure_publishes_no_authority(
    tmp_path,
    monkeypatch,
):
    rows = _complete_gfs_inventory()
    monkeypatch.setattr(adapt, "_grib2_inventory", lambda *_args: rows)

    def refuse_decode(*_args, **_kwargs):
        raise RuntimeError("selected GRIB2 record uses unsupported packing")

    monkeypatch.setattr(adapt, "_decode_grib", refuse_decode)
    inventory = tmp_path / "grib2_inventory"
    dump = tmp_path / "grib2_dump"
    inventory.write_bytes(b"fixture inventory decoder")
    dump.write_bytes(b"fixture dump decoder")
    output = tmp_path / "adapter"

    with pytest.raises(RuntimeError, match="unsupported packing"):
        author_adapter(
            vtable_path=VTABLE,
            descriptor_path=DESCRIPTOR,
            input_files=(ACTUAL_GRIB,),
            output_dir=output,
            grib2_inventory=inventory,
            grib2_dump=dump,
        )
    assert not output.exists()


def test_adapt_skeleton_generator_is_create_only_and_review_required(
    tmp_path,
    capsys,
):
    output = tmp_path / "source.descriptor.json"
    assert main(
        [
            "adapt",
            "--vtable",
            str(VTABLE),
            "--skeleton",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "rw-wps.descriptor.v1"
    assert payload["adapt"]["model_top_pa"] \
        == "REPLACE_WITH_MODEL_TOP_PA"
    assert payload["coordinates"]["vertical"]["levels"] == []
    printed = capsys.readouterr().out
    # The scaffold announces its own gaps, by name, in the order the
    # authoring gate will hold the reader to -- so "what is left to do"
    # is answered once here rather than one type error per run.
    assert "4 value(s) need you" in printed
    for gap in ("adapt.model_top_pa", "coordinates.vertical.levels",
                "name", "target.name"):
        assert f"\n  {gap}\n" in printed

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "adapt",
                "--vtable",
                str(VTABLE),
                "--skeleton",
                str(output),
            ]
        )


def _skeleton_payload(tmp_path) -> tuple[Path, dict]:
    output = tmp_path / "source.descriptor.json"
    assert main(["adapt", "--vtable", str(VTABLE),
                 "--skeleton", str(output)]) == 0
    return output, json.loads(output.read_text(encoding="utf-8"))


def _fill_scaffold(payload: dict) -> dict:
    """Replace exactly what :func:`scaffold_gaps` reports, nothing else."""

    filled = json.loads(json.dumps(payload))
    filled["name"] = "example-adapter"
    filled["target"]["name"] = "example-target"
    filled["adapt"]["model_top_pa"] = 5000.0
    filled["coordinates"]["vertical"]["levels"] = [
        100000.0, 92500.0, 85000.0, 70000.0, 50000.0, 25000.0, 10000.0,
        5000.0,
    ]
    return filled


def test_the_skeleton_round_trips_through_its_own_descriptor_validator(
    tmp_path,
    capsys,
):
    """`woof adapt: descriptor.adapt.model_top_pa must be a positive
    finite number` -- what a user zero got from the two commands the
    documentation prints, in order, with nothing else on the machine.

    The scaffold's placeholders are deliberate; the defect was that its
    own validator met them as a type error about one field and said
    nothing about the other three, so filling them in was a sequence of
    guesses one run apart.  The round-trip contract asserted here is:
    the ONLY things the authoring gate objects to in a fresh skeleton
    are the gaps the skeleton itself named, and replacing exactly those
    carries the descriptor through the same validator clean.
    """

    from woof.adapt import _load_adapt_policy, scaffold_gaps

    path, payload = _skeleton_payload(tmp_path)
    capsys.readouterr()

    gaps = scaffold_gaps(payload)
    assert gaps == ["adapt.model_top_pa", "coordinates.vertical.levels",
                    "name", "target.name"]

    # 1. Authoring refuses the untouched scaffold as a scaffold, once,
    #    naming the size of the job rather than one of its fields.
    assert main(["adapt", "--vtable", str(VTABLE), "--descriptor", str(path),
                 "--input", str(ACTUAL_GRIB),
                 "--output-dir", str(tmp_path / "out")]) == 2
    refusal = capsys.readouterr().err
    assert "is still the `--skeleton` scaffold" in refusal
    assert "4 value(s) are unfilled" in refusal
    assert "adapt.model_top_pa" in refusal
    assert "must be a positive finite number" not in refusal

    # 2. Replacing exactly the named gaps round-trips: no gaps left, and
    #    the validator that produced the original refusal now accepts it.
    filled = tmp_path / "filled.descriptor.json"
    filled.write_text(json.dumps(_fill_scaffold(payload), indent=2),
                      encoding="utf-8")
    assert scaffold_gaps(json.loads(filled.read_text(encoding="utf-8"))) == []
    policy, _snapshot = _load_adapt_policy(filled)
    assert policy["model_top_pa"] == 5000.0

    # What the mapping compiler says next is a question about the
    # Vtable's own content (this one gives two soil layers overlapping
    # selectors), which is precisely the "review every selector" work
    # docs/arbitrary-verified-adapters.md sends the author to do.  The
    # gate asserted here is that it is no longer a SCAFFOLD complaint:
    # the template half of the job is finished and stays finished.
    with pytest.raises(ValueError) as raised:
        compile_mapping_descriptor(filled, vtable_path=VTABLE)
    assert "scaffold" not in str(raised.value)
    assert "REPLACE_WITH" not in str(raised.value)


def test_filling_only_the_reported_field_still_names_the_rest(tmp_path):
    """The one-at-a-time failure mode, pinned closed.

    Setting ``model_top_pa`` and re-running used to advance to the next
    placeholder's own error message.  The gate reports what is left.
    """

    from woof.adapt import _load_adapt_policy

    _path, payload = _skeleton_payload(tmp_path)
    payload["adapt"]["model_top_pa"] = 5000.0
    partial = tmp_path / "partial.descriptor.json"
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="3 value\\(s\\) are unfilled"):
        _load_adapt_policy(partial)


def test_without_the_scaffold_gate_the_generic_type_error_returns(
    tmp_path,
    monkeypatch,
):
    """Negative control: the gate is what changed the message.

    Neutralize ``_refuse_scaffold`` and the untouched scaffold is once
    again refused by ``descriptor.adapt.model_top_pa must be a positive
    finite number`` -- the exact string the wheel user was handed.
    """

    from woof.adapt import _load_adapt_policy

    path, _payload = _skeleton_payload(tmp_path)
    monkeypatch.setattr(adapt, "_refuse_scaffold", lambda raw, label: None)
    with pytest.raises(
            ValueError,
            match="descriptor.adapt.model_top_pa must be a positive finite"):
        _load_adapt_policy(path)


# ---------------------------------------------------------------------------
# V-9: the skeleton generator emitted a descriptor its own battery rejects
# ---------------------------------------------------------------------------


def test_the_skeleton_never_assigns_one_vtable_line_to_two_fields():
    """`woof adapt: Vtable line 11 is assigned more than once`.

    A test host 7 validation run filled in every REPLACE_WITH_* placeholder
    the skeleton emitted, declared the 21 isobaric levels actually in
    the GRIB, and had the battery refuse the result -- because the
    generator had given four 3-D fields their SURFACE counterpart's
    selector as well as their own.  `air_temperature` claimed the 2 m
    row that `air_temperature_2m` also claimed, and the same for
    relative humidity and both wind components.  The data is not
    ambiguous; the prefix rule that collects `soil_temperature_0_0.1m`
    under `soil_temperature` was swallowing `air_temperature_2m` under
    `air_temperature`.

    A generator whose output its own validator rejects is worse than no
    generator: the user cannot tell which of the two is wrong.
    """
    from collections import defaultdict

    from woof.adapt import descriptor_skeleton

    vtable = packaged_gfs_vtable()
    skeleton = descriptor_skeleton(vtable)

    claimed = defaultdict(list)
    for field_name, spec in skeleton["fields"].items():
        for reference in spec.get("vtable_selectors", ()):
            claimed[reference.get("metgrid_name")].append(field_name)
    doubled = {
        name: sorted(fields)
        for name, fields in claimed.items() if len(fields) > 1
    }
    assert doubled == {}, doubled

    # And the separation is right, not merely non-overlapping: the 3-D
    # field takes the isobaric row and the 2 m/10 m field takes its own.
    for three_d, surface in (
        ("air_temperature", "air_temperature_2m"),
        ("relative_humidity", "relative_humidity_2m"),
        ("eastward_wind", "eastward_wind_10m"),
        ("northward_wind", "northward_wind_10m"),
    ):
        names = {
            field: {
                reference["metgrid_name"]
                for reference in skeleton["fields"][field]["vtable_selectors"]
            }
            for field in (three_d, surface)
        }
        assert names[three_d] and names[surface]
        assert not (names[three_d] & names[surface]), names
        assert all(
            not name.endswith(("_2m_0", "_10m_0"))
            for name in names[three_d]), names[three_d]

    # The multi-layer prefix rule the exclusion must not break: four soil
    # layers still collect under one canonical field.
    soil = skeleton["fields"]["soil_temperature"]["vtable_selectors"]
    assert len(soil) == 4, soil


def test_a_descriptor_fault_is_not_reported_as_a_vtable_fault(tmp_path):
    """The message pointed the reader at the wrong file.

    `Vtable line 11 is assigned more than once` names the Vtable, and
    line 11 of the Vtable was fine -- the double assignment was in the
    descriptor, which in the test host 7 case the tool itself had written.
    """
    import pytest as _pytest

    from woof.adapt import descriptor_skeleton
    from woof.mapped_authoring import compile_mapping_descriptor

    vtable = packaged_gfs_vtable()
    skeleton = dict(descriptor_skeleton(vtable))
    fields = dict(skeleton["fields"])
    # Re-introduce exactly the defect that shipped.
    fields["air_temperature"] = {
        **fields["air_temperature"],
        "vtable_selectors": list(
            fields["air_temperature"]["vtable_selectors"])
        + list(fields["air_temperature_2m"]["vtable_selectors"]),
    }
    skeleton["fields"] = fields
    written = tmp_path / "descriptor.json"
    written.write_text(json.dumps(skeleton), encoding="utf-8")

    with _pytest.raises(ValueError) as caught:
        compile_mapping_descriptor(written, vtable_path=vtable)
    message = str(caught.value)
    assert "descriptor" in message
    assert "air_temperature" in message and "air_temperature_2m" in message


def test_the_gfs_vtable_ships_in_the_wheel():
    """A pip user must have the file the documented flow tells them to pass.

    It lived in `configs/`, which is not a package, so setuptools never
    carried it: `woof adapt --vtable configs/Vtable.GFS.rw-wps` named a
    path that exists only in a checkout, while `woof adapt` is exactly
    the surface a wheel-only user reaches for.  It ships beside the
    20CRv3 authorities now, under the same recursive package-data glob
    and the same byte contract.
    """
    import hashlib
    import tomllib

    from woof.source_authorities import (
        packaged_gfs_vtable, packaged_gfs_vtable_sha256,
    )

    path = packaged_gfs_vtable()
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() \
        == packaged_gfs_vtable_sha256()

    # Inside the package, and matched by a declared package-data glob --
    # resolving on this machine is a checkout property, not a wheel one.
    package_root = Path(adapt.__file__).resolve().parent
    relative = path.relative_to(package_root)
    assert relative.parts[0] == "authorities"

    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"))
    globs = pyproject["tool"]["setuptools"]["package-data"]["woof"]
    assert any(
        relative.match(pattern) or relative.as_posix().startswith(
            pattern.split("**")[0])
        for pattern in globs), (relative.as_posix(), globs)

    # And it is the real thing, not a stub: the parser reads it and the
    # skeleton generator produces a descriptor from it.
    assert len(adapt.parse_wps_vtable(path)) >= 20


# --- the trust boundary is published at the point of use ----------------
#
# The battery proves the emitted files implement the descriptor.  It does
# not prove the descriptor reads the files correctly, and a scientist has
# to learn that while writing the descriptor rather than after a
# plausible-looking run.  These gates bind the CLI's own words to the
# page that carries the two-column contract, so renaming or dropping the
# page fails here instead of leaving a dangling instruction in --help.

_TRUST_DOC = "docs/adapt-validation-contract.md"


def _adapt_parser_texts() -> list[str]:
    import argparse

    parser = argparse.ArgumentParser()
    adapt.register_cli(parser.add_subparsers(dest="command"))
    adapt_parser = parser._subparsers._group_actions[0].choices["adapt"]
    return [adapt_parser.description or "", adapt_parser.epilog or ""]


@pytest.mark.parametrize(
    "referenced",
    [_TRUST_DOC, "docs/arbitrary-verified-adapters.md"])
def test_every_doc_the_adapt_cli_names_exists(referenced):
    assert (ROOT / referenced).is_file()


def test_adapt_help_names_the_trust_boundary_and_the_page(capsys):
    description, epilog = _adapt_parser_texts()
    assert "trusted from your declaration" in description
    assert _TRUST_DOC.rsplit("/", 1)[-1] in epilog

    with pytest.raises(SystemExit) as exit_info:
        main(["adapt", "--help"])
    assert exit_info.value.code == 0
    rendered = capsys.readouterr().out
    assert "adapt-validation-contract.md" in rendered


def test_the_trust_page_carries_both_literal_columns():
    text = (ROOT / _TRUST_DOC).read_text(encoding="utf-8")
    assert "Validated for you" in text
    assert "Trusted from your declaration" in text
    # Every dimension the audit found trusted has to be named on the page.
    lowered = text.lower()
    for dimension in ("units", "registration", "sufficiency",
                      "land mask", "source cycle", "depth labels"):
        assert dimension in lowered, dimension
