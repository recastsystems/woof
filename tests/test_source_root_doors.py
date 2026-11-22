"""A hand-staged folder runs through the doors the documentation names.

Named breakage, measured on a real model-level run: the documented line
``woof prep --source era5-l137 --source-root DIR --experiment-config X
--wps-namelist Y`` exited 64 with "--source-root is not used by --source
mapped" and asked for --supplement, --output-root, an input list and a
manifest; ``woof domain``'s printed ``woof go CONFIG --data-dir DIR``
exited 2 because the folder had no ``prep-arguments.json``, a file only a
fetch route writes; and the fetch door's own remedy printed a
``--source-manifest DIR/SHA256SUMS`` pair the mapped preparation refuses.
Only a hand-written argument list got through.

The fix is table work: a source this WOOF-ArWen cannot download states in its
refusal row how its folder binds (``source_root``), and both doors read
that row.  Every test here is parametrized over the rows that declare a
layout, so a source added as a row inherits the coverage.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from woof import fetch_routes, local_preparation, source_cli
from woof.source_adapters import get_source_adapter


LAYOUT_SOURCES = tuple(sorted(
    source for source in set(fetch_routes.refusal_ids()) | set(fetch_routes.all_fetchable_sources())
    if fetch_routes.source_root_layout(source) is not None))
CYCLE = datetime(2026, 5, 5, 12)


def _write(path: Path, file_format: str, body: bytes = b"payload") -> Path:
    """A file whose leading bytes carry ``file_format``, nothing more."""

    head = {"grib1": b"GRIB\x00\x00\x00\x01", "grib2": b"GRIB\x00\x00\x00\x02",
            "netcdf": b"CDF\x01\x00\x00\x00\x00"}[file_format]
    path.write_bytes(head + body + b"7777")
    return path


def _literal(patterns) -> str:
    """A file name the row's patterns match."""

    pattern = patterns[0]
    return pattern.replace("*", "staged") if "*" in pattern else pattern


def _stage(source: str, root: Path) -> dict:
    """Fill ``root`` the way the row says a folder is laid out.

    Also a stray file of another format and a JSON receipt, the two
    things a real folder holds beside its inputs, so the binding is shown
    to take the files it should and nothing else.
    """

    layout = fetch_routes.source_root_layout(source)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    supplements = []
    for row in layout["supplements"]:
        supplements.append((row["role"], _write(root / _literal(row["match"]), row["format"])))
    wanted = layout["inputs"]
    inputs = [_write(root / _literal(wanted["match"]).replace("staged", name), wanted["format"])
              for name in ("b-second", "a-first")]
    for exclude in wanted["exclude"]:
        _write(root / exclude, wanted["format"])
    (root / "receipt.json").write_text("{}", encoding="utf-8")
    other = "grib1" if wanted["format"] != "grib1" else "grib2"
    _write(root / ("stray." + other), other)
    inputs = sorted(inputs, key=lambda path: path.name)
    for role, path in supplements:
        row = next(item for item in layout["supplements"] if item["role"] == role)
        if row["input"]:
            inputs = sorted(inputs + [path], key=lambda item: item.name)
    return {"inputs": inputs, "supplements": supplements}


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------

def test_the_documented_hand_staged_sources_declare_their_folder():
    """The two mapped sources the documentation sends to --source-root."""

    assert {"era5-l137", "20crv3-cf"} <= set(LAYOUT_SOURCES)


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_a_layout_binds_exactly_the_roles_its_composition_declares(source):
    """The row and the composition cannot disagree about a role.

    The local review (woof/local_preparation.py) refuses a handoff whose
    roles differ from the composition's, and the packaged profile refuses
    a --supplement role that is not its own, so a row that bound any other
    role would build a binding one of them rejects.
    """

    from woof.source_authorities import packaged_composition, packaged_profile

    adapter = get_source_adapter(source)
    composition = packaged_composition(adapter.packaged_profile)
    declared = {str(row["data_role"])
                for group in ("supplements", "field_sources")
                for row in composition.get(group, {}).values()
                if "data_role" in row}
    layout = fetch_routes.source_root_layout(source)
    assert {row["role"] for row in layout["supplements"]} == declared
    profile = packaged_profile(adapter.packaged_profile)
    assert layout["inputs"]["format"] == profile["source_format"]


@pytest.mark.parametrize("source", sorted(fetch_routes.refusal_ids()))
def test_every_fetch_refusal_remedy_is_a_line_that_runs(source):
    """The refusal used to hand every source a manifest pair the mapped
    preparation refuses and the member route takes only while authoring."""

    with pytest.raises(ValueError) as refusal:
        fetch_routes.route_for(source)
    message = str(refusal.value)
    assert "SHA256SUMS" not in message
    assert "remedy:" in message
    if fetch_routes.source_root_layout(source) is not None:
        assert fetch_routes.local_prep_line(source) in message


@pytest.mark.parametrize("path_name,head,expected", [
    ("a.grib", b"GRIB\x00\x00\x00\x01", "grib1"),
    ("a.grib2", b"GRIB\x00\x00\x00\x02", "grib2"),
    ("a.nc", b"CDF\x02\x00\x00\x00\x00", "netcdf"),
    ("a.nc4", b"\x89HDF\r\n\x1a\n", "netcdf"),
    ("a.json", b"{}", None),
])
def test_a_file_format_is_read_from_its_leading_bytes(tmp_path, path_name, head, expected):
    path = tmp_path / path_name
    path.write_bytes(head)
    assert fetch_routes.sniff_format(path) == expected


# ---------------------------------------------------------------------------
# The binding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_a_staged_folder_binds_its_inputs_in_name_order_and_its_supplements(
        source, tmp_path):
    staged = _stage(source, tmp_path / "data")
    binding = local_preparation.bind_source_root(source, tmp_path / "data")
    assert binding["inputs"] == staged["inputs"]
    assert binding["supplements"] == staged["supplements"]


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_a_missing_supplement_is_named_with_the_fetch_that_writes_it(source, tmp_path):
    layout = fetch_routes.source_root_layout(source)
    staged = _stage(source, tmp_path / "data")
    role, path = staged["supplements"][0]
    path.unlink()
    with pytest.raises(ValueError) as refusal:
        local_preparation.bind_source_root(source, tmp_path / "data")
    message = str(refusal.value)
    assert role in message
    row = layout["supplements"][0]
    if row["fetch"] is not None:
        assert f"woof fetch --source {row['fetch']['source']}" in message


def test_a_supplement_named_right_but_of_the_wrong_format_is_refused(tmp_path):
    source = next(source for source in LAYOUT_SOURCES
                  if any(row["format"] != fetch_routes.source_root_layout(source)
                         ["inputs"]["format"]
                         for row in fetch_routes.source_root_layout(source)["supplements"]))
    layout = fetch_routes.source_root_layout(source)
    _stage(source, tmp_path / "data")
    row = layout["supplements"][0]
    _write(tmp_path / "data" / _literal(row["match"]), layout["inputs"]["format"])
    with pytest.raises(ValueError, match=row["format"]):
        local_preparation.bind_source_root(source, tmp_path / "data")


# ---------------------------------------------------------------------------
# woof prep --source-root: the documented short line
# ---------------------------------------------------------------------------

def _captured_prep(monkeypatch, argv):
    """Run the prep door to the point it launches the preparer.

    The manifest authoring decodes real bytes, so it is recorded instead;
    everything before it -- the argument checks that used to refuse, the
    profile, the engine and tool choice -- runs as shipped.
    """

    authored: list[Path] = []

    def author(args):
        authored.append(Path(args.author_input_manifest))
        args.source_sha256s = args.author_input_manifest
        args.source_sha256s_sha256 = "0" * 64
        return {}

    launched: list[list[str]] = []

    def launch(command):
        launched.append(list(command))
        return 0

    monkeypatch.setattr(source_cli, "_author_mapped_contract", author)
    monkeypatch.setattr(source_cli, "_run_native_adapter", launch)
    code = source_cli.main(argv)
    return code, authored, launched


def _config(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "case.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    namelist = tmp_path / "case.namelist.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")
    return config, namelist


def _pairs(command: list[str], flag: str) -> list[str]:
    return [command[index + 1] for index, token in enumerate(command)
            if token == flag]


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_the_documented_source_root_prep_line_runs(source, tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    staged = _stage(source, tmp_path / "data")
    config, namelist = _config(tmp_path)
    code, authored, launched = _captured_prep(monkeypatch, [
        "--source", source, "--source-root", str(tmp_path / "data"),
        "--experiment-config", str(config), "--wps-namelist", str(namelist)])
    assert code == 0
    assert authored == [(tmp_path / "data" / "inputs.json").resolve()]
    [command] = launched
    assert _pairs(command, "--input") == [str(path) for path in staged["inputs"]]
    assert _pairs(command, "--supplement") == [
        f"{role}={path}" for role, path in staged["supplements"]]
    assert _pairs(command, "--output-root") == [str(tmp_path / "case-prepared")]
    assert _pairs(command, "--geog-root") == [str(tmp_path / "case-data" / "WPS_GEOG")]
    assert _pairs(command, "--prepared-forecast-source") == [source]


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_explicit_output_and_manifest_keep_their_meaning(source, tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    _stage(source, tmp_path / "data")
    config, namelist = _config(tmp_path)
    manifest = tmp_path / "pinned.json"
    manifest.write_text("{}", encoding="utf-8")
    code, authored, launched = _captured_prep(monkeypatch, [
        "--source", source, "--source-root", str(tmp_path / "data"),
        "--experiment-config", str(config), "--wps-namelist", str(namelist),
        "--source-manifest", str(manifest), "--source-manifest-sha256", "ab" * 32,
        "--output-root", str(tmp_path / "mine"), "--geog-root", str(tmp_path / "geog")])
    assert code == 0 and authored == []
    [command] = launched
    assert _pairs(command, "--input-manifest") == [str(manifest)]
    assert _pairs(command, "--output-root") == [str(tmp_path / "mine")]
    assert _pairs(command, "--geog-root") == [str(tmp_path / "geog")]


def _recorded_authoring(monkeypatch):
    """Record each manifest authoring and write a manifest as it would."""

    calls: list[dict] = []

    def author(path, **kwargs):
        calls.append({"path": Path(path), **kwargs})
        body = json.dumps({"inputs": [str(p) for p in kwargs["primary_files"]]})
        path = Path(path)
        if path.is_file() and path.read_text(encoding="utf-8") != body \
                and not kwargs.get("replace_different"):
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(body, encoding="utf-8")
        import hashlib
        return {"manifest": {"path": str(path),
                             "sha256": hashlib.sha256(body.encode()).hexdigest()},
                "reauthored": True}

    monkeypatch.setattr(source_cli, "author_input_manifest", author)
    monkeypatch.setattr(source_cli, "_run_native_adapter", lambda command: 0)
    return calls


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_the_documented_line_runs_again_after_the_folder_changes(
        source, tmp_path, monkeypatch, capsys):
    """Named breakage: the second run of the documented line, after a file
    in the folder was renamed, exited 78 ("refusing to overwrite
    DIR/inputs.json"), and so would every run after an upgrade changed the
    decoders the manifest seals.  The door chose that path, and each
    preparation keeps its own copy of the manifest it was made from, so
    the door replaces it and says which one it replaced."""

    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    staged = _stage(source, tmp_path / "data")
    config, namelist = _config(tmp_path)
    argv = ["--source", source, "--source-root", str(tmp_path / "data"),
            "--experiment-config", str(config), "--wps-namelist", str(namelist)]
    calls = _recorded_authoring(monkeypatch)
    assert source_cli.main(argv) == 0
    manifest = (tmp_path / "data" / "inputs.json").resolve()
    first = manifest.read_text(encoding="utf-8")
    renamed = staged["inputs"][0]
    renamed.rename(renamed.with_name("z-" + renamed.name))
    capsys.readouterr()
    assert source_cli.main(argv) == 0
    assert manifest.read_text(encoding="utf-8") != first
    assert [call["replace_different"] for call in calls] == [True, True]
    assert f"REPLACED input_manifest={manifest}" in capsys.readouterr().err


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_a_manifest_path_the_user_names_is_still_never_replaced(
        source, tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    _stage(source, tmp_path / "data")
    config, namelist = _config(tmp_path)
    named = tmp_path / "named.json"
    named.write_text("an earlier manifest", encoding="utf-8")
    calls = _recorded_authoring(monkeypatch)
    code = source_cli.main([
        "--source", source, "--source-root", str(tmp_path / "data"),
        "--experiment-config", str(config), "--wps-namelist", str(namelist),
        "--author-input-manifest", str(named)])
    assert code != 0
    assert [call["replace_different"] for call in calls] == [False]
    assert named.read_text(encoding="utf-8") == "an earlier manifest"


class _Published(Exception):
    """The preparer got past every check and would start building."""


def _preparer_to_its_first_build_step(monkeypatch, tmp_path):
    """Launch the real mapped preparer in process, up to its first build step.

    Everything the preparer checks before it builds runs as shipped,
    its own output-root check included; the engine and decoder lookups
    answer as a staged install would, and the first build step publishes
    the output folder as a finished preparation leaves it.  Returns the
    list of folders it published.
    """

    from woof import mapped_direct, mapped_engine_bridge

    def first_build_step(path):
        raise _Published

    monkeypatch.setattr(mapped_direct, "_mapped_engine_choice",
                        lambda **kwargs: mapped_direct._ENGINE_RUST)
    monkeypatch.setattr(mapped_engine_bridge, "require_engine",
                        lambda: tmp_path / "engine")
    monkeypatch.setattr(mapped_direct, "_decoder_inventory",
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(mapped_direct, "load_experiment", first_build_step)
    published: list[Path] = []

    def launch(command):
        assert command[1:3] == ["-m", "woof.mapped_direct"]
        root = Path(command[command.index("--output-root") + 1])
        try:
            return mapped_direct.main(command[3:])
        except _Published:
            root.mkdir(parents=True)
            published.append(root)
            return 0

    monkeypatch.setattr(source_cli, "_run_native_adapter", launch)
    return published


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_the_documented_line_passes_the_preparer_s_own_output_check_every_run(
        source, tmp_path, monkeypatch, capsys):
    """Named breakage, measured on the real command: the second run of the
    documented line exited 78 ("refusing to overwrite mapped output
    CONFIG-prepared"), whether or not the folder had changed, because the
    door defaulted the same output folder every time and the preparer
    never writes over one.  Each run now prepares into a folder no earlier
    run holds, and the manifest is replaced only when the folder changed."""

    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    (tmp_path / "case-data" / "WPS_GEOG").mkdir(parents=True)
    staged = _stage(source, tmp_path / "data")
    config, namelist = _config(tmp_path)
    argv = ["--source", source, "--source-root", str(tmp_path / "data"),
            "--experiment-config", str(config), "--wps-namelist", str(namelist)]
    calls = _recorded_authoring(monkeypatch)
    published = _preparer_to_its_first_build_step(monkeypatch, tmp_path)
    manifest = (tmp_path / "data" / "inputs.json").resolve()

    assert source_cli.main(argv) == 0
    first = manifest.read_bytes()
    capsys.readouterr()
    assert source_cli.main(argv) == 0
    assert manifest.read_bytes() == first
    assert "REPLACED" not in capsys.readouterr().err
    renamed = staged["inputs"][0]
    renamed.rename(renamed.with_name("z-" + renamed.name))
    assert source_cli.main(argv) == 0
    assert manifest.read_bytes() != first
    assert f"REPLACED input_manifest={manifest}" in capsys.readouterr().err
    assert published == [tmp_path / "case-prepared", tmp_path / "case-prepared-2",
                         tmp_path / "case-prepared-3"]
    assert len(calls) == 3


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_a_named_output_folder_that_exists_is_refused_before_the_folder_changes(
        source, tmp_path, monkeypatch, capsys):
    """The preparer refuses an --output-root that exists; the door used to
    reach that refusal only after it had replaced DIR/inputs.json, so a
    run that prepared nothing still changed the folder's manifest."""

    from woof.ingest.source_coverage import PREPARATION_REFUSAL_EXIT_CODE

    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    _stage(source, tmp_path / "data")
    config, namelist = _config(tmp_path)
    (tmp_path / "mine").mkdir()
    calls = _recorded_authoring(monkeypatch)
    code = source_cli.main([
        "--source", source, "--source-root", str(tmp_path / "data"),
        "--experiment-config", str(config), "--wps-namelist", str(namelist),
        "--output-root", str(tmp_path / "mine")])
    assert code == PREPARATION_REFUSAL_EXIT_CODE
    assert calls == [] and not (tmp_path / "data" / "inputs.json").exists()
    err = capsys.readouterr().err
    assert f"refusing to overwrite mapped output {tmp_path / 'mine'}" in err
    assert "pass a fresh --output-root" in err


@pytest.mark.parametrize("flag", ["--input", "--supplement"])
def test_a_second_input_list_beside_the_folder_is_refused_by_name(
        flag, tmp_path, monkeypatch, capsys):
    source = LAYOUT_SOURCES[0]
    staged = _stage(source, tmp_path / "data")
    config, namelist = _config(tmp_path)
    code, _, launched = _captured_prep(monkeypatch, [
        "--source", source, "--source-root", str(tmp_path / "data"),
        flag, str(staged["inputs"][0]),
        "--experiment-config", str(config), "--wps-namelist", str(namelist)])
    assert code == source_cli.EXIT_USAGE and launched == []
    assert f"{flag} is not used with --source-root" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# woof go --data-dir: the local review binds the same folder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_local_review_binds_a_folder_no_fetch_wrote(source, tmp_path):
    staged = _stage(source, tmp_path / "data")
    cadence = int(get_source_adapter(source).forcing_interval_seconds // 3600)
    snapshot = local_preparation.inspect_local_inputs(
        source, tmp_path / "data", cycle=CYCLE, hours=cadence, cadence=cadence)
    assert snapshot["kind"] == "prep_handoff"
    assert snapshot["ordered_inputs"] == [str(path) for path in staged["inputs"]]
    published = local_preparation.publish_local_handoff(snapshot, tmp_path / "run")
    handoff = json.loads(published.read_text(encoding="utf-8"))
    assert handoff["source"] == source and handoff["unbound_supplement_roles"] == []
    argv = handoff["argv"]
    listing = Path(argv[argv.index("--input-list") + 1])
    assert listing.read_text(encoding="utf-8").splitlines() == snapshot["ordered_inputs"]
    assert _pairs(argv, "--supplement") == [
        f"{role}={path}" for role, path in staged["supplements"]]
    # Nothing was written into the folder by the review itself.
    assert not (tmp_path / "data" / fetch_routes.PREP_ARGUMENTS_NAME).exists()


@pytest.mark.parametrize("source", LAYOUT_SOURCES)
def test_the_published_handoff_prepares_through_the_same_door(source, tmp_path, monkeypatch):
    """The go chain's own preparation line, from the handoff it publishes."""

    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    _stage(source, tmp_path / "data")
    cadence = int(get_source_adapter(source).forcing_interval_seconds // 3600)
    snapshot = local_preparation.inspect_local_inputs(
        source, tmp_path / "data", cycle=CYCLE, hours=cadence, cadence=cadence)
    published = local_preparation.publish_local_handoff(snapshot, tmp_path / "run")
    from woof.prep_handoff import preparation_arguments

    argv = preparation_arguments(json.loads(published.read_text(encoding="utf-8")))
    config, namelist = _config(tmp_path)
    code, authored, launched = _captured_prep(monkeypatch, [
        *argv, "--wps-namelist", str(namelist), "--experiment-config", str(config),
        "--geog-root", str(tmp_path / "geog"), "--output-root", str(tmp_path / "prep")])
    assert code == 0 and len(launched) == 1
    assert authored == [(tmp_path / "run" / "inputs.json").resolve()]


def test_an_empty_folder_still_names_the_handoff_and_the_layout(tmp_path):
    source = LAYOUT_SOURCES[0]
    with pytest.raises(ValueError) as refusal:
        local_preparation.inspect_local_inputs(
            source, tmp_path, cycle=CYCLE, hours=1, cadence=1)
    message = str(refusal.value)
    assert fetch_routes.PREP_ARGUMENTS_NAME in message
    assert "folder layout" in message


# ---------------------------------------------------------------------------
# woof domain: step 1 fills the folder step 3 reads
# ---------------------------------------------------------------------------

def test_the_staging_note_spells_the_request_and_the_supplement_fetch():
    from woof.domain_wizard import local_staging_lines

    source = next(source for source in LAYOUT_SOURCES
                  if fetch_routes.source_root_layout(source)["request"] is not None)
    layout = fetch_routes.source_root_layout(source)
    lines = local_staging_lines(
        source, cycle="2024-05-06T12", hours=3, cadence=1, start_hour=0,
        area="30.11,-104.76,40.79,-90.24", out="data/area")
    request = [line for line in lines if layout["request"]["keywords"] in line]
    assert request == [
        f"#     {layout['request']['keywords']} date=2024-05-06 "
        "time=12:00:00/13:00:00/14:00:00/15:00:00 area=40.75/-104.75/30.25/-90.25"]
    donors = [row["fetch"]["source"] for row in layout["supplements"] if row["fetch"]]
    fetches = [line for line in lines if line.startswith("woof fetch ")]
    assert [line.split()[3] for line in fetches] == donors
    for line in fetches:
        assert "--cycle 2024-05-06T12 --hours 3" in line
        assert line.endswith("--out data/area")


def test_the_staging_note_asks_for_the_lattice_points_the_surface_fetch_delivers():
    """A request whose corners sit off the provider's lattice may be
    interpolated onto points the surface analysis does not carry; the
    composition binds the two only on an exact coordinate subset.  The
    corners are taken inward onto the lattice the surface fetch delivers
    for the same area (measured: --area 30.11,-104.76,40.79,-90.24 came
    back as lat 30.25..40.75, lon -104.75..-90.25)."""

    from woof.domain_wizard import local_staging_lines

    source = next(source for source in LAYOUT_SOURCES
                  if fetch_routes.source_root_layout(source)["request"] is not None)
    for area, expected in (("30.11,-104.76,40.79,-90.24", "40.75/-104.75/30.25/-90.25"),
                           ("31.25,-112,52,-78", "52/-112/31.25/-78")):
        [line] = [line for line in local_staging_lines(
            source, cycle="2026-05-05T12", hours=3, cadence=1, start_hour=0,
            area=area, out="data/area") if " area=" in line]
        assert line.endswith(f"area={expected}")


def test_the_staging_note_asks_one_request_per_date():
    from woof.domain_wizard import local_staging_lines

    source = next(source for source in LAYOUT_SOURCES
                  if fetch_routes.source_root_layout(source)["request"] is not None)
    lines = local_staging_lines(
        source, cycle="2024-05-06T22", hours=3, cadence=1, start_hour=0,
        area="30,-100,40,-90", out="data/area")
    dated = [line for line in lines if " date=" in line]
    assert len(dated) == 2
    assert "date=2024-05-06 time=22:00:00/23:00:00 " in dated[0]
    assert "date=2024-05-07 time=00:00:00/01:00:00 " in dated[1]


FETCHED_SUPPLEMENTS = tuple(
    (source, row) for source in LAYOUT_SOURCES
    for row in fetch_routes.source_root_layout(source)["supplements"]
    if row["fetch"] is not None)


@pytest.mark.parametrize("source,row", FETCHED_SUPPLEMENTS,
                         ids=[f"{source}-{row['role']}"
                              for source, row in FETCHED_SUPPLEMENTS])
def test_a_supplement_row_cannot_restate_whether_its_fetch_retrieves(
        source, row):
    """Named breakage: the row and the donor's own adapter both able to
    say whether its fetch needs --retrieve, and saying different things,
    prints a line that either writes a request template instead of the
    file or passes a flag the fetch door refuses for that source.  The
    adapter's fetch_requires_retrieve is the one statement, the one the
    fetch door gates the flag on, so a row restating it, either way, is
    refused at load."""

    raw = json.loads((Path(fetch_routes.__file__).parent / "authorities"
                      / fetch_routes.ROUTE_TABLE_NAME).read_text(
                          encoding="utf-8"))
    layout = raw["refusals"][source]["source_root"]
    [index] = [index for index, supplement in enumerate(layout["supplements"])
               if supplement["role"] == row["role"]]
    assert dict(layout["supplements"][index]["fetch"]) == dict(row["fetch"])
    assert set(row["fetch"]) == {"source"}
    stated = get_source_adapter(str(row["fetch"]["source"])).fetch_requires_retrieve
    assert fetch_routes.supplement_fetch_retrieves(row["fetch"]) is stated
    for restated in (stated, not stated):
        layout["supplements"][index]["fetch"] = {**row["fetch"],
                                                 "retrieve": restated}
        with pytest.raises(ValueError, match="needs --retrieve"):
            fetch_routes._source_root_row(source, layout)


@pytest.mark.parametrize("source,row", FETCHED_SUPPLEMENTS,
                         ids=[f"{source}-{row['role']}"
                              for source, row in FETCHED_SUPPLEMENTS])
def test_every_printed_supplement_fetch_writes_the_file_it_names(
        source, row, tmp_path, monkeypatch):
    """The surface analysis fetch without --retrieve writes a request and a
    retrieval script, not the file the folder binds, so each printed line
    that names the file (the staging note, the missing-file refusal and
    the fetch refusal's remedy) carries the flag its source's fetch
    needs.  The staging note's line is run through the command line up to
    the remote transfer."""

    import shlex
    from woof import era5_acquisition
    from woof.cli import main as cli_main
    from woof.domain_wizard import local_staging_lines

    donor = str(row["fetch"]["source"])
    retrieves = get_source_adapter(donor).fetch_requires_retrieve
    out = (tmp_path / "area").resolve()
    [note] = [line for line in local_staging_lines(
        source, cycle="2024-05-06T12", hours=3, cadence=1, start_hour=0,
        area="30.11,-104.76,40.79,-90.24", out=str(out))
        if line.startswith(f"woof fetch --source {donor} ")]
    _stage(source, tmp_path / "data")
    missing = tmp_path / "data" / _literal(row["match"])
    missing.unlink()
    with pytest.raises(ValueError) as refusal:
        local_preparation.bind_source_root(source, tmp_path / "data")
    printed = {
        "staging note": note,
        "missing-file refusal": str(refusal.value),
        "fetch refusal remedy": fetch_routes.local_input_remedy(source),
    }
    for where, text in printed.items():
        assert ("--retrieve" in text) == retrieves, (
            f"{where} prints `woof fetch --source {donor}` "
            f"{'without' if retrieves else 'with'} --retrieve: {text}")
    if not retrieves:
        return

    calls = []
    monkeypatch.setattr(era5_acquisition, "retrieve_era5",
                        lambda **kwargs: calls.append(kwargs))
    assert cli_main(shlex.split(note)[1:]) == 0
    assert len(calls) == 1, (
        "the staging note's fetch wrote a retrieval template instead of the "
        f"{row['role']} file ({row['match'][0]})")
