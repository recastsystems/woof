"""The prep door forwards the tools the subcommand it actually runs needs.

Measured breakage, at integration tip a6867e712 on 2026-08-18::

    python -m woof.source_cli --source hrrr-prs ... --dry-run
    # -> ... -m woof.mapped_direct --source-format grib2 ...
    #    (no --grib2-inventory, no --grib2-dump)

    python -m woof.mapped_direct <that exact argv>
    # ValueError: grib2 decoder inventory differs from the contract;
    #             missing=['grib2_dump', 'grib2_inventory'], extra=[]

Bare default preparation was broken for every one of the twelve
registered ``mapped_composition_v1`` sources and for the generic
``mapped`` route, and it was broken as an unhandled traceback out of
``woof/mapped_composition.py``.

The cause was one question asked about the wrong thing.  The Rust
engine's capability table declares its answers PER SUBCOMMAND -- it
decodes GRIB2 in process and it composes nothing at all -- while the
front door asked only ``which engine is the default`` and concluded
"Rust, so no subprocess tools are wanted".  The mapped route ALWAYS
composes (``mapped_direct.prepare_mapped_wrf`` says so in its own
comment), so the composed child was launched with the Python engine's
work to do and none of the Python engine's tools.

The contract these tests hold:

* the door consults ``ENGINE_CAPABILITIES`` for the subcommand the
  route will run, not for the engine default, and forwards exactly what
  that answer needs;
* the invocation the door prints RUNS -- these tests execute it rather
  than reading it, because an argv assertion is exactly what stayed
  green while the route was dead;
* tools that genuinely cannot resolve produce a named refusal with the
  install-aware remedy, at the door and in the child, never a
  traceback.

**Both directions, since the compose port landed.**  The engine now
composes GRIB1, GRIB2 and NetCDF, so a bare prep forwards NO tool pins
and the composed child resolves the engine row instead.  That is not a
reason to stop measuring the other direction: the outage was a
DISPATCH fault, not a fact about one table state, and it comes back the
moment a site answers about the wrong subcommand.  So every forwarding
row below runs twice -- once against the real table, where nothing is
pinned, and once against a table monkeypatched to un-declare ``compose``
for the format in hand, where the ladder-resolved tools must appear.
A single-direction test would go green again if the door started
answering about ``decode``, because ``decode`` and ``compose`` now agree.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from woof import bridges, mapped_direct, mapped_engine_bridge, source_cli
from woof.ingest.source_coverage import PREPARATION_REFUSAL_EXIT_CODE
from woof.source_adapters import source_adapters
from woof import earth2_licensing
from _reserved_drives import gauntlet_staging_root


#: Every registered source whose runner is the mapped composition route.
#: Read from the registry rather than listed, so a model added as table
#: data inherits this coverage instead of needing a new row here.
COMPOSITION_SOURCES = tuple(sorted(
    adapter.source_id
    for adapter in source_adapters()
    if adapter.runner == "mapped_composition_v1" and adapter.runnable
    and adapter.packaged_profile is not None
))

#: The staged real-byte tree the parity battery reads, same environment
#: contract.
#: The staging root, never one on or linked onto a reserved drive
#: (tests/_reserved_drives.py names the breakage).
STAGING = gauntlet_staging_root()

#: Staged real-byte trees, as ``source -> (directory, cycle, hours)``.
#:
#: The INPUT/SUPPLEMENT split is deliberately NOT written here.  It is
#: read from ``woof.fetch_routes`` -- the same table ``woof fetch``
#: writes ``inputs.txt`` and ``prep-command.txt`` from -- so a model
#: added to the registry as table data becomes one row here rather than
#: a recipe of its own, and a row can never drift from the acquisition
#: door that produces the bytes it names.
#:
#: Three sources rather than one because the outage was registry-wide:
#: ``hrrr-prs`` and ``icon-eu`` are the two the boundary audit measured
#: dying, and ``icon-eu`` additionally carries every object bz2-wrapped
#: and 252 inputs, while ``rap`` is JPEG2000-packed (DRT 5.40).  One
#: source's bytes cannot show that the door is fixed for the others.
REAL_BYTE_TREES = {
    "hrrr-prs": (STAGING / "hrrr", datetime(2026, 8, 16, 0), 1),
    "icon-eu": (STAGING / "icon" / "eu-regular-latlon",
                datetime(2026, 8, 17, 0), 1),
    "rap": (STAGING / "rap", datetime(2026, 8, 16, 0), 1),
}


def _staged_recipe(source):
    """This source's bare-prep inputs and supplements, from the table.

    Skips rather than fails when the bytes are absent: they are real
    agency products measured in hundreds of megabytes and no checkout
    is required to carry them, so every other row in this file still
    runs on a machine that has not staged them.
    """

    from woof import fetch_routes

    tree, cycle, hours = REAL_BYTE_TREES[source]
    if not tree.is_dir():
        pytest.skip(f"staged tree absent: {tree}")
    plan = fetch_routes.resolve_request(source, cycle=cycle, hours=hours)
    inputs = tuple(tree / name for name in plan.primary_files)
    supplements = tuple(tree / name for name in plan.supplement_files)
    absent = [path for path in inputs + supplements if not path.is_file()]
    if absent:
        pytest.skip(f"staged bytes absent ({len(absent)}): {absent[0]}")
    return inputs, supplements


@pytest.fixture(autouse=True)
def _bare_posture(monkeypatch):
    """No engine request and no per-tool override unless a test sets one.

    Every assertion below is about the BARE default door, so an
    inherited environment variable would be the test grading a posture
    the user never typed.
    """

    for variable in ("GPUWM_MAPPED_ENGINE", "WOOF_GRIB2_INVENTORY",
                     "WOOF_GRIB2_DUMP", "WOOF_GRIB1_BRIDGE",
                     "WOOF_NATIVE_DISTRIBUTION_MANIFEST"):
        monkeypatch.delenv(variable, raising=False)


def _compose_undeclared(monkeypatch, *formats):
    """The capability table with ``compose`` un-declared for ``formats``.

    Not a hypothetical: it is the state this file was written against,
    and it is the state any future format arrives in before its port is
    measured.  Kept as a lever so every forwarding assertion can be made
    in BOTH directions off one table, rather than being deleted the day
    the real entry flipped -- deleting it would leave the door's question
    ("which subcommand does this route run?") untested, because ``decode``
    and ``compose`` now answer the same for every format.
    """

    kept = frozenset(
        mapped_engine_bridge.ENGINE_CAPABILITIES["compose"] or ()
    ) - frozenset(formats)
    monkeypatch.setattr(
        mapped_engine_bridge, "ENGINE_CAPABILITIES",
        dict(mapped_engine_bridge.ENGINE_CAPABILITIES, compose=kept))


def _staged_tools(tmp_path, monkeypatch):
    """A box where ``woof fetch-bridges`` has run: the ladder answers.

    The fakes carry the REAL contract markers, so the resolver's static
    ABI gate passes on genuine evidence rather than being stubbed out.
    """

    staged = {}
    for name in ("grib2_inventory", "grib2_dump"):
        path = tmp_path / bridges.executable_name(name)
        path.write_bytes(b"MZ\0\0" + bridges.BRIDGE_ABI_MARKERS[name])
        staged[name] = path
    monkeypatch.setattr(bridges, "find_bridge", lambda name: staged.get(name))
    return staged


def _bare_estate(tmp_path, monkeypatch):
    """A wheel install with nothing staged: ladder empty, no crate."""

    from woof import mapped_source

    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    monkeypatch.setattr(mapped_source, "_grib2_tools_crate",
                        lambda: tmp_path / "no-such-crate")

    class _NoCargo:
        returncode = 1
        stderr = "cargo must not run in this test"
        stdout = ""

    monkeypatch.setattr(mapped_source.subprocess, "run",
                        lambda *args, **kwargs: _NoCargo())


def _prep_argv(source, *, inputs, output_root, work, supplements=None,
               extra=()):
    """A bare packaged-profile preparation: no tool flags anywhere.

    ``supplements`` defaults to ``inputs`` because the in-band-terrain
    profiles bind the same files twice; a source whose terrain arrives
    as a separate invariant object passes its own list.

    A model row whose weights are not permissive refuses any door until
    its exact licence is named (woof.earth2_licensing.require_licence,
    exit 78).  That gate is tested in test_earth2_licensing; here such a
    row carries its own opt-in, read from the table, so the forwarding
    claim is still measured on every registered row.
    """

    argv = ["--source", source]
    found = earth2_licensing.model_row(source) if source.startswith("earth2-") else None
    if found and found[1].get("licence_class", "unresolved") != "permissive":
        licence = (found[1].get("weights") or {}).get("licence", "unresolved weight licence")
        argv += ["--accept-licence", licence]
    for path in inputs:
        argv += ["--input", str(path)]
    for path in (inputs if supplements is None else supplements):
        argv += ["--supplement", str(path)]
    # A composition that declares a companion is refused at the door
    # without the companion's files, so a bare prep names them under the
    # companion binding's own role, as the fetch handoff does.
    from woof.source_authorities import packaged_composition

    from woof.source_adapters import get_source_adapter

    profile = get_source_adapter(source).packaged_profile
    if profile is not None:
        for binding in packaged_composition(profile).get(
                "field_sources", {}).values():
            if "companion" in binding:
                argv += ["--supplement",
                         f"{binding['data_role']}=/data/companion.grib2"]
    return argv + [
        "--wps-namelist", str(work / "namelist.wps"),
        "--geog-root", str(work / "WPS_GEOG"),
        "--experiment-config", str(work / "experiment.toml"),
        "--source-manifest", str(work / "input-manifest.json"),
        "--source-manifest-sha256", "abc123",
        "--output-root", str(output_root),
        *extra,
    ]


def _run_control(work):
    """The files the composed child stats before it decodes anything."""

    work.mkdir(parents=True, exist_ok=True)
    (work / "WPS_GEOG").mkdir(exist_ok=True)
    for name in ("namelist.wps", "experiment.toml", "input-manifest.json"):
        (work / name).write_text("{}\n", encoding="utf-8")
    return work


def _child_argv(monkeypatch, argv):
    """The argv the door WOULD launch, captured as the door built it.

    Taken off ``_run_native_adapter`` rather than parsed back out of the
    printed ``--dry-run`` line: the printed command and the executed one
    are the same list, and a quoting round-trip would put a second
    suspect between the door and the crash.
    """

    captured: list[list[str]] = []

    def capture(command):
        captured.append(list(command))
        return 0

    monkeypatch.setattr(source_cli, "_run_native_adapter", capture)
    assert source_cli.main(argv) == 0
    assert len(captured) == 1, captured
    command = captured[0]
    assert command[1:3] == ["-m", "woof.mapped_direct"], command[:4]
    return command[3:]


# ---------------------------------------------------------------------------
# 1. The composed invocation RUNS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", sorted(REAL_BYTE_TREES))
def test_the_composed_child_invocation_runs_past_the_decoder_contract(
        source, tmp_path, monkeypatch):
    """THE deliverable: what the door prints is what the child can run.

    Named breakage: a bare ``woof prep --source hrrr-prs`` composed an
    invocation that died inside ``mapped_composition._decoder_inventory``
    with ``grib2 decoder inventory differs from the contract``, because
    the door decided the Rust engine wanted no tools and the route then
    asked the Python engine to compose.  Twelve registered sources plus
    the generic mapped route reached that traceback on a bare default
    run.

    Executed, not asserted about: the argv-only ladder tests were green
    through the whole outage.  The child is stopped at
    ``load_experiment`` -- the first step past the decoder gate that
    needs a real case -- so the reproduction stays hermetic and fast
    while the decoder contract itself runs for real.

    Run for each staged source rather than one, because the two shapes
    that could hide a residue are exactly the ones a single row misses:
    ``icon-eu`` reaches the door as 252 bz2-wrapped objects with a
    separate invariant terrain supplement, and ``rap`` is JPEG2000
    packed.  Every row was measured red at the pre-fix parent.

    Since the compose port the resolved inventory is the ENGINE row
    rather than the subprocess pair, and the assertion follows it: what
    is being held is that the door and the route agree about who decodes,
    not that any particular executable is named.
    """

    inputs, supplements = _staged_recipe(source)
    try:
        mapped_engine_bridge.require_engine()
    except mapped_engine_bridge.EngineUnavailable as absent:
        pytest.skip(f"mapped engine unusable: {absent}")

    work = _run_control(tmp_path / "case")
    argv = _prep_argv(source, inputs=inputs, supplements=supplements,
                      output_root=tmp_path / "out", work=work)
    child = _child_argv(monkeypatch, argv)

    inventories: list[dict[str, Path]] = []
    real_inventory = mapped_direct._decoder_inventory

    def spy(*args, **kwargs):
        resolved = real_inventory(*args, **kwargs)
        inventories.append(dict(resolved))
        return resolved

    monkeypatch.setattr(mapped_direct, "_decoder_inventory", spy)

    class _ReachedTheCase(Exception):
        """The child got past every decoder gate."""

    def stop(_path):
        raise _ReachedTheCase

    monkeypatch.setattr(mapped_direct, "load_experiment", stop)

    with pytest.raises(_ReachedTheCase):
        mapped_direct.main(child)

    assert inventories, "the child never reached the decoder inventory"
    resolved = inventories[0]
    assert sorted(resolved) == [mapped_engine_bridge.ENGINE_NAME], resolved
    for path in resolved.values():
        assert path.is_file(), path


# ---------------------------------------------------------------------------
# 2. The door asks the capability table for the subcommand it will run
# ---------------------------------------------------------------------------

def test_the_route_the_door_composes_is_the_compose_subcommand():
    """The premise: this route composes, so `compose` is the question.

    ``prepare_mapped_wrf`` runs ``mapped_composition``'s byte work on
    every call, contributing mappings or not.  If that ever stops being
    true this pin fails and the door's question has to be re-derived
    rather than silently answering about the wrong subcommand again.
    """

    assert mapped_engine_bridge.MAPPED_ROUTE_SUBCOMMAND == "compose"


def test_the_shipped_table_declares_compose_for_every_decode_format():
    """Where the two answers stand today, stated once.

    This was ``the capability table answers decode and compose
    differently`` -- true while compose was unported, and the reason one
    question could not serve both.  The port made every format agree, so
    the sentence is now the opposite one, and it is recorded here because
    every forwarding row below depends on knowing which state it runs in.
    """

    for source_format in ("grib1", "grib2", "netcdf"):
        assert mapped_engine_bridge.engine_supports("decode", source_format)
        assert mapped_engine_bridge.engine_supports("compose", source_format)


def test_the_lever_the_outage_rows_run_on_really_separates_the_answers(
        monkeypatch):
    """The two answers CAN still differ, which is what keeps them tested.

    Named breakage: with the shipped table agreeing about every format,
    a door that started asking about ``decode`` -- or about the engine
    default, which is the original fault -- would pass every row in this
    file.  The rows below therefore run against a table where the two
    answers differ, and this is the pin that the lever producing that
    table does what it says.
    """

    _compose_undeclared(monkeypatch, "grib2")
    assert mapped_engine_bridge.engine_supports("decode", "grib2") is True
    assert mapped_engine_bridge.engine_supports("compose", "grib2") is False
    # The lever is per FORMAT, not a switch that empties the entry: an
    # empty frozenset reads as "no format" and would make every row below
    # forward tools for the wrong reason.
    assert mapped_engine_bridge.engine_supports("compose", "netcdf") is True


@pytest.mark.parametrize("source", COMPOSITION_SOURCES)
def test_a_bare_prep_pins_no_tools_when_the_engine_composes(
        source, tmp_path, monkeypatch, capsys):
    """The shipped default: every registered source, bare, nothing pinned.

    Named breakage, and it is not merely cosmetic: an explicit tool path
    is precisely the spelling that routes a call back to the Python
    engine, so a door that forwarded the ladder-resolved tools while the
    engine composes would silently un-default itself -- every bare prep
    would run in host numpy while the receipts said Rust was the default.

    The registry is the coverage rather than a sample of it, for the same
    reason the outage row below uses it: the fault was registry-wide.
    """

    _staged_tools(tmp_path, monkeypatch)
    work = _run_control(tmp_path / "case")
    argv = _prep_argv(source, inputs=("/data/f00.grib2", "/data/f01.grib2"),
                      output_root=tmp_path / "out", work=work,
                      extra=("--dry-run",))

    assert source_cli.main(argv) == 0
    command = capsys.readouterr().out
    assert "-m woof.mapped_direct" in command
    for flag in ("--grib1-bridge", "--grib2-inventory", "--grib2-dump"):
        assert flag not in command, f"{source}: {flag} pins the Python engine"


@pytest.mark.parametrize("source", COMPOSITION_SOURCES)
def test_an_undeclared_compose_still_forwards_the_resolved_tools(
        source, tmp_path, monkeypatch, capsys):
    """The outage row, kept alive by un-declaring compose for the format.

    Named breakage: every row of the registry that runs
    ``mapped_composition_v1`` composed an unrunnable command, because the
    door asked which engine was the DEFAULT instead of what the route's
    subcommand needs.  The shipped table no longer reaches that state,
    but the dispatch that caused it is unchanged code, so the row is
    measured against a table that does.

    What must be forwarded is what the route's own FORMAT reads with:
    the GRIB2 profiles need both tools, and a NetCDF one needs neither
    and must not be handed a pin it would never launch.  The expectation
    is read off the composed command's ``--source-format`` rather than
    listed here, so a profile that changes format changes this test's
    demand with it.
    """

    staged = _staged_tools(tmp_path, monkeypatch)
    _compose_undeclared(monkeypatch, "grib1", "grib2", "netcdf")
    work = _run_control(tmp_path / "case")
    argv = _prep_argv(source, inputs=("/data/f00.grib2", "/data/f01.grib2"),
                      output_root=tmp_path / "out", work=work,
                      extra=("--dry-run",))

    assert source_cli.main(argv) == 0
    command = capsys.readouterr().out
    assert "-m woof.mapped_direct" in command
    tokens = command.split()
    source_format = tokens[tokens.index("--source-format") + 1]
    wanted = mapped_direct._FORMAT_TOOL_ROLES[source_format]
    for flag, name in (("--grib2-inventory", "grib2_inventory"),
                       ("--grib2-dump", "grib2_dump")):
        if name not in wanted:
            assert flag not in command, \
                f"{source}: {source_format} launches no {name}"
            continue
        assert flag in command, f"{source}: {flag} was not forwarded"
        assert str(staged[name]).replace("\\", "/") in \
            command.replace("\\", "/"), f"{source}: {flag} pins the ladder"


def _generic_mapped_argv(work, tmp_path):
    """The un-profiled ``--source mapped`` door: the thirteenth case."""

    return [
        "--source", "mapped", "--source-format", "grib2",
        "--mapping", str(work / "mapping.json"),
        "--composition", str(work / "composition.json"),
        "--input", "/data/source-f000.grb2",
        "--supplement", f"terrain={work / 'terrain.grb2'}",
        "--provenance", f"terrain={work / 'terrain-provenance.md'}",
        "--wps-namelist", str(work / "namelist.wps"),
        "--geog-root", str(work / "WPS_GEOG"),
        "--experiment-config", str(work / "experiment.toml"),
        "--source-sha256s", str(work / "input-manifest.json"),
        "--source-sha256s-sha256", "abc123",
        "--output-root", str(tmp_path / "out"), "--dry-run",
    ]


def test_the_generic_mapped_route_pins_no_tools_when_the_engine_composes(
        tmp_path, monkeypatch, capsys):
    """The generic door follows the table too, with no profile to read."""

    _staged_tools(tmp_path, monkeypatch)
    work = _run_control(tmp_path / "case")
    assert source_cli.main(_generic_mapped_argv(work, tmp_path)) == 0
    command = capsys.readouterr().out
    assert "--grib2-inventory" not in command
    assert "--grib2-dump" not in command


def test_the_generic_mapped_route_forwards_the_resolved_tools(
        tmp_path, monkeypatch, capsys):
    """The generic door's outage direction, on an undeclared compose."""

    staged = _staged_tools(tmp_path, monkeypatch)
    _compose_undeclared(monkeypatch, "grib2")
    work = _run_control(tmp_path / "case")
    assert source_cli.main(_generic_mapped_argv(work, tmp_path)) == 0
    command = capsys.readouterr().out
    assert str(staged["grib2_inventory"]).replace("\\", "/") in \
        command.replace("\\", "/")
    assert str(staged["grib2_dump"]).replace("\\", "/") in \
        command.replace("\\", "/")


def test_an_explicit_tool_flag_still_pins_the_python_engine(
        tmp_path, monkeypatch, capsys):
    """A named tool is a statement, and it survives the new question."""

    staged = _staged_tools(tmp_path, monkeypatch)
    pinned = tmp_path / bridges.executable_name("grib2_dump")
    work = _run_control(tmp_path / "case")

    assert source_cli.main(_prep_argv(
        "hrrr-prs", inputs=("/data/f00.grib2", "/data/f01.grib2"),
        output_root=tmp_path / "out", work=work,
        extra=("--grib2-dump", str(pinned), "--dry-run"))) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert str(pinned).replace("\\", "/") in command
    assert str(staged["grib2_inventory"]).replace("\\", "/") in command


# ---------------------------------------------------------------------------
# 3. Unresolvable tools refuse by name, at both doors
# ---------------------------------------------------------------------------

def test_the_bare_door_refuses_by_name_when_the_tools_cannot_resolve(
        tmp_path, monkeypatch, capsys):
    """Nothing staged: the install-aware remedy, not a demand for a flag.

    Run with ``compose`` un-declared, because that is the only state in
    which this door needs a subprocess tool at all -- and the remedy it
    delivers is the one a reader of an unported format must still get.
    """

    _bare_estate(tmp_path, monkeypatch)
    _compose_undeclared(monkeypatch, "grib1", "grib2", "netcdf")
    work = _run_control(tmp_path / "case")

    assert source_cli.main(_prep_argv(
        "hrrr-prs", inputs=("/data/f00.grib2", "/data/f01.grib2"),
        output_root=tmp_path / "out", work=work,
        extra=("--dry-run",))) == source_cli.EXIT_CONFIG
    message = capsys.readouterr().err
    assert "not installed here" in message
    assert "grib2_inventory" in message and "grib2_dump" in message
    assert "woof doctor" in message
    assert "Traceback" not in message


def test_the_child_names_its_refusal_instead_of_a_traceback(
        tmp_path, monkeypatch, capsys):
    """``python -m woof.mapped_direct`` with no tools is a sentence.

    Named breakage: this exact call raised
    ``ValueError: grib2 decoder inventory differs from the contract``
    through ``main`` with a nineteen-frame stack, so the reader met
    ``woof/mapped_composition.py`` line numbers before meeting a
    remedy.  The refusal is not weakened -- the contract still refuses,
    with the same breakage named -- it is delivered.
    """

    real_bytes, _ = _staged_recipe("hrrr-prs")
    _bare_estate(tmp_path, monkeypatch)
    _compose_undeclared(monkeypatch, "grib1", "grib2", "netcdf")
    work = _run_control(tmp_path / "case")
    authorities = Path(source_cli.__file__).parent / "authorities"

    status = mapped_direct.main([
        "--source-format", "grib2",
        "--composition",
        str(authorities / "rw-wps-hrrr-prs-grib2.composition.json"),
        "--mapping",
        str(authorities / "rw-wps-hrrr-prs-grib2.mapping.json"),
        "--input-manifest", str(work / "input-manifest.json"),
        "--input-manifest-sha256", "abc123",
        "--wps-namelist", str(work / "namelist.wps"),
        "--geog-root", str(work / "WPS_GEOG"),
        "--experiment-config", str(work / "experiment.toml"),
        "--output-root", str(tmp_path / "out"),
        *(argument for path in real_bytes
          for argument in ("--input", str(path))),
        *(argument for path in real_bytes
          for argument in ("--supplement",
                           f"hrrr_prs_in_band_surface={path}")),
        "--provenance", "hrrr_prs_in_band_surface_provenance="
        + str(authorities / "rw-wps-hrrr-prs-grib2.provenance.json"),
    ])

    assert status == PREPARATION_REFUSAL_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "decoder inventory differs from the contract" in captured.err
    assert "grib2_dump" in captured.err and "grib2_inventory" in captured.err
    assert "remedy:" in captured.err
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# 4. EVERY registered door, not only the mapped ones
# ---------------------------------------------------------------------------
#
# Every row above filters the registry to `runner ==
# "mapped_composition_v1"`, which is the route the compose port changed.
# That filter is also a blind spot, and it hid a live one: `woof prep
# --source 20crv3` is a registered, runnable, packaged-profile door whose
# runner (`woof.twentycrv3_wrf`) decodes through the subprocess GRIB2
# tools, builds its `MappedSourceBundle` in host Python and hands it to
# `prepare_mapped_wrf` as `_predecoded_bundle` -- the one branch that
# skips `decode_composed_source`.  A bare prep of it therefore resolves
# `grib2_inventory`/`grib2_dump` from the staged ladder while the
# capability table, `woof doctor` and the CLI reference all reported a
# clean estate.
#
# So the invariant is stated over the WHOLE registry, and it is the one
# a reader of `ENGINE_GAPS` is entitled to: a door either forwards no
# decoder tool, or it is named in `ENGINE_GAPS`.  Both directions, so the
# entry cannot outlive the gap it describes.

#: Every runnable registered source, from the registry rather than a
#: list: a model added as table data inherits this coverage.
RUNNABLE_SOURCES = tuple(sorted(
    adapter.source_id for adapter in source_adapters() if adapter.runnable))

#: The flags that name a subprocess decoder in a composed child command.
DECODER_FLAGS = ("--grib1-bridge", "--grib2-inventory", "--grib2-dump",
                 "--bridge")


def _bare_door_argv(source, adapter, tmp_path, work):
    """The smallest bare invocation this runner's door will consider.

    Keyed by RUNNER, never by source id, for the same reason the rows
    above read the registry: a new model on an existing runner inherits
    its row instead of needing one written for it.
    """

    if adapter.runner == "mapped_composition_v1":
        return _prep_argv(source, inputs=("/data/f00.grib2",
                                          "/data/f01.grib2"),
                          output_root=tmp_path / "out", work=work,
                          extra=("--dry-run",))
    return [
        "--source", source,
        "--source-manifest", str(work / "input-manifest.json"),
        "--source-manifest-sha256", "abc123",
        "--wps-namelist", str(work / "namelist.wps"),
        "--geog-root", str(work / "WPS_GEOG"),
        "--experiment-config", str(work / "experiment.toml"),
        "--output-root", str(tmp_path / "out"),
        "--dry-run",
    ]


@pytest.mark.parametrize("source", RUNNABLE_SOURCES)
def test_a_door_that_forwards_decoder_tools_is_a_named_gap(
        source, tmp_path, monkeypatch, capsys):
    """A front door that still needs a subprocess decoder says so.

    Named breakage, measured at the tip that declared ``compose``:
    ``woof prep --source 20crv3 ... --dry-run`` printed a child command
    carrying ``--grib2-inventory`` and ``--grib2-dump`` while
    ``ENGINE_GAPS`` was empty, ``woof doctor`` printed no exception and
    ``docs/cli-reference.md`` said "a bare prep resolves no decoder tools
    at all".  A user reading any of the three would have been told their
    bytes were decoded somewhere they were not.

    A door that REFUSES a bare invocation is not the same fault and is
    not counted as one: it resolved nothing, and its refusal has to name
    what it wants, which is asserted here rather than assumed.
    """

    adapter = next(row for row in source_adapters()
                   if row.source_id == source)
    _staged_tools(tmp_path, monkeypatch)
    work = _run_control(tmp_path / "case")
    argv = _bare_door_argv(source, adapter, tmp_path, work)

    status = source_cli.main(argv)
    captured = capsys.readouterr()
    written_down = f"--source {source}`" in " ".join(
        mapped_engine_bridge.ENGINE_GAPS)
    if status != 0:
        # Nothing was resolved, and the door has to say what it wants:
        # a silent refusal would leave "did it pick a tool?" unanswered.
        assert "Traceback" not in captured.err, captured.err
        assert "--" in captured.err, (
            f"{source}: the door refused without naming a flag, so a "
            "reader cannot tell whether it resolved a decoder first")
        # A gap entry has to be backed by a live measurement, or it is
        # prose: if this row cannot drive the named door to a command,
        # nothing here can tell whether the gap it describes still
        # exists, and `woof doctor` would go on printing it forever.
        assert not written_down, (
            f"ENGINE_GAPS names `--source {source}`, but this row cannot "
            "drive that door far enough to see a command, so the entry is "
            "unmeasured; give the row the arguments the door requires")
        return

    command = captured.out
    forwarded = tuple(flag for flag in DECODER_FLAGS if flag in command)
    assert not forwarded or written_down, (
        f"a bare `woof prep --source {source}` forwards {list(forwarded)} "
        "and no ENGINE_GAPS entry names that door; `woof doctor` and the "
        "CLI reference both read ENGINE_GAPS, so the estate would report "
        "a route that decodes in the engine while this one does not")
    assert not written_down or forwarded, (
        f"ENGINE_GAPS names `--source {source}` as a door that still "
        "forwards decoder tools, and a bare prep of it forwards none; the "
        "gap closed and its entry has to go with it, or doctor keeps "
        "printing a gap that no longer exists")
